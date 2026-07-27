import importlib.util
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import onnx
import pytest
import torch

from dcc.model import AxialRoPE, DetectorNet

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def train_detector():
    """tools.train_detector, loaded by absolute path -- an unrelated
    detectron2 'tools' package on sys.path shadows any `import
    tools.train_detector` (see tests/test_guards.py's identical pattern for
    tools/preflight.py)."""
    spec = importlib.util.spec_from_file_location("_train_detector_under_test", ROOT / "tools" / "train_detector.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- (a) attend_div=16 regression

def test_attend_div_16_matches_default_construction():
    torch.manual_seed(0)
    default = DetectorNet(240, 320)
    torch.manual_seed(0)
    explicit = DetectorNet(240, 320, attend_div=16)

    sd_default, sd_explicit = default.state_dict(), explicit.state_dict()
    assert list(sd_default.keys()) == list(sd_explicit.keys())
    for k in sd_default:
        assert torch.equal(sd_default[k], sd_explicit[k]), k

    for name in ("e5", "gate4", "d4"):
        assert hasattr(default, name) and hasattr(explicit, name)


# --------------------------------------------------------------------------- (b) attend_div=8 shapes

def test_attend_div_8_shapes():
    torch.manual_seed(0)
    m = DetectorNet(240, 320, attend_div=8).eval()

    for name in ("e5", "d4", "gate4"):
        assert not hasattr(m, name), f"attend_div=8 must not build {name}"

    hm, cls = m(torch.randn(2, 1, 240, 320))
    assert hm.shape == (2, 1, 240, 320)
    assert cls.shape == (2, 16, 60, 80)

    grid_h, grid_w = 240 // 8, 320 // 8   # 30, 40
    assert m.rope.cos.shape[2] == grid_h * grid_w

    total = sum(p.numel() for p in m.parameters())
    print(f"attend_div=8 param count: {total}")
    assert abs(total - 4_698_034) / 4_698_034 <= 0.02, total


# --------------------------------------------------------------------------- (c) RoPE anchor adapts to the H/8 grid

def test_rope_anchor_adapts_attend_div_8():
    grid_h, grid_w = 30, 40                 # DetectorNet(240, 320, attend_div=8)'s H/8 grid
    head_dim = 256 // 8                     # d=256, heads=8 defaults
    n = head_dim // 4
    rope = AxialRoPE(head_dim, grid_h, grid_w, lambda_min=2.5)
    cos, sin = rope.cos[0, 0], rope.sin[0, 0]

    # Mirrors tests/test_model.py::test_rope_no_global_alias's analytic check:
    # the phase increment between adjacent columns (row=0) is exactly each
    # axis's own per-cell omega for the col-axis half of the vector.
    ph0 = torch.atan2(sin[0], cos[0])
    ph1 = torch.atan2(sin[1], cos[1])
    dphi = torch.atan2(torch.sin(ph1 - ph0), torch.cos(ph1 - ph0))
    assert dphi[:n].abs().max() < 1e-5
    col_omegas = dphi[n:2 * n].abs()

    lambda_max_cells = (2 * math.pi / col_omegas.min()).item()
    assert lambda_max_cells >= 2 * max(grid_h, grid_w) - 1e-3


# --------------------------------------------------------------------------- (d) ONNX export smoke, attend_div=8

def test_onnx_export_attend_div_8(tmp_path):
    banned = {"Complex", "Loop", "If"}

    m = DetectorNet(240, 320, attend_div=8).eval()
    det_path = str(tmp_path / "detector8.onnx")
    torch.onnx.export(m, torch.randn(1, 1, 240, 320), det_path, opset_version=17,
                      dynamo=False, input_names=["input"], output_names=["hm", "cls"])
    det_onnx = onnx.load(det_path)
    onnx.checker.check_model(det_onnx)
    det_ops = {n.op_type for n in det_onnx.graph.node}
    assert not (det_ops & banned), det_ops & banned


# --------------------------------------------------------------------------- (e) early-stop decision function

def _entry(step, mean, tail, m02_val, acc, octave_names):
    return {"step": step, "m01": {"mean": mean, "tail_frac_gt4px": tail},
            "m02": {k: m02_val for k in octave_names}, "m04": {"accuracy": acc}}


def _flat_window(train_detector, steps):
    """3 full-vals, all metrics unchanged (mean/tail/m02/m04 identical
    across the window) and tail_frac_gt4px well under the default gate --
    plateaued and safely inside the refiner's capture range."""
    names = [n for n, _lo, _hi in train_detector.OCTAVE_BINS]
    return [_entry(s, mean=1.5, tail=0.005, m02_val=0.9, acc=0.95, octave_names=names) for s in steps]


def test_early_stop_triggers_on_flat_passing_window(train_detector):
    es_cfg = {"enabled": True, "patience": 3, "min_steps": 100000, "tail_gate": 0.01}
    history = _flat_window(train_detector, [100000, 110000, 120000])
    assert train_detector.early_stop_should_trigger(history, 120000, es_cfg) is True


def test_early_stop_no_trigger_when_tail_gate_fails(train_detector):
    es_cfg = {"enabled": True, "patience": 3, "min_steps": 100000, "tail_gate": 0.01}
    history = _flat_window(train_detector, [100000, 110000, 120000])
    history[-1]["m01"]["tail_frac_gt4px"] = 0.02       # exceeds tail_gate at the last full-val
    assert train_detector.early_stop_should_trigger(history, 120000, es_cfg) is False


def test_early_stop_no_trigger_when_metric_still_improves(train_detector):
    es_cfg = {"enabled": True, "patience": 3, "min_steps": 100000, "tail_gate": 0.01}
    names = [n for n, _lo, _hi in train_detector.OCTAVE_BINS]
    history = [
        _entry(100000, mean=2.0, tail=0.005, m02_val=0.9, acc=0.95, octave_names=names),
        _entry(110000, mean=1.9, tail=0.005, m02_val=0.9, acc=0.95, octave_names=names),
        _entry(120000, mean=1.7, tail=0.005, m02_val=0.9, acc=0.95, octave_names=names),
    ]   # m01 mean improves (2.0 -> 1.7) by 15% relative, well over the 1% bar
    assert train_detector.early_stop_should_trigger(history, 120000, es_cfg) is False


def test_early_stop_no_trigger_before_min_steps(train_detector):
    es_cfg = {"enabled": True, "patience": 3, "min_steps": 100000, "tail_gate": 0.01}
    history = _flat_window(train_detector, [30000, 40000, 50000])
    assert train_detector.early_stop_should_trigger(history, 50000, es_cfg) is False


def test_early_stop_no_trigger_when_disabled(train_detector):
    es_cfg = {"enabled": False, "patience": 3, "min_steps": 100000, "tail_gate": 0.01}
    history = _flat_window(train_detector, [100000, 110000, 120000])
    assert train_detector.early_stop_should_trigger(history, 120000, es_cfg) is False
