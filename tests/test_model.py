import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import onnx
import torch
import pytest

from dcc.model import DetectorNet, Refiner, AxialRoPE
from dcc.losses import focal, detector_loss, refiner_loss


def test_shapes_small():
    torch.manual_seed(0)
    m = DetectorNet(240, 320).eval()
    hm, cls = m(torch.randn(2, 1, 240, 320))
    assert hm.shape == (2, 1, 240, 320)
    assert cls.shape == (2, 16, 60, 80)

    r = Refiner().eval()
    out = r(torch.randn(5, 1, 24, 24))
    assert out.shape == (5, 1, 64, 64)


def test_param_count_full():
    total = sum(p.numel() for p in DetectorNet(1200, 1600).parameters())
    assert abs(total - 7_124_700) / 7_124_700 <= 0.02, total

    total_r = sum(p.numel() for p in Refiner().parameters())
    assert abs(total_r - 96_600) / 96_600 <= 0.05, total_r


def test_bias_inits():
    m = DetectorNet(64, 64)
    assert m.hm[-1].bias.detach().eq(-2.19).all()
    assert m.cls[-1].bias.detach().eq(-2.19).all()
    assert Refiner().out.bias.detach().eq(-2.19).all()

    expected_alpha = torch.sigmoid(torch.tensor(3.0))
    for gate in (m.gate3, m.gate4):
        assert torch.all(gate.psi.weight == 0)
        assert torch.all(gate.psi.bias == 3.0)
        skip = torch.randn(2, gate.wx.in_channels, 16, 16)
        g = torch.randn(2, gate.wg.in_channels, 8, 8)
        a = gate.alpha(skip, g)
        assert torch.allclose(a, expected_alpha.expand_as(a), atol=1e-6, rtol=0)


def test_stable_names():
    sd = list(DetectorNet(240, 320).state_dict().keys())
    prefixes = ("e1.", "e2.", "e3.", "e4.", "e5.", "blocks.", "norm.",
                "gate3.", "gate4.", "d1.", "d2.", "d3.", "d4.", "hm.", "cls.")
    assert all(k.startswith(prefixes) for k in sd)

    cls_keys = [k for k in sd if k.startswith("cls.")]
    other_keys = [k for k in sd if not k.startswith("cls.")]
    assert cls_keys
    assert set(cls_keys).isdisjoint(other_keys)


def test_rope_no_global_alias():
    grid_h, grid_w = 75, 100          # DetectorNet(1200,1600)'s H/16 bottleneck grid
    head_dim = 256 // 8               # d=256, heads=8 defaults
    n = head_dim // 4
    rope = AxialRoPE(head_dim, grid_h, grid_w, lambda_min=2.5)
    cos, sin = rope.cos[0, 0], rope.sin[0, 0]        # (T, 2n)

    # Analytic: the phase increment between adjacent columns (row=0) is
    # exactly each axis's own per-cell omega, for the col-axis half of the
    # vector (indices n..2n-1); the row-axis half (0..n-1) is unaffected by
    # column and serves as an internal sanity check.
    ph0 = torch.atan2(sin[0], cos[0])
    ph1 = torch.atan2(sin[1], cos[1])
    dphi = torch.atan2(torch.sin(ph1 - ph0), torch.cos(ph1 - ph0))
    assert dphi[:n].abs().max() < 1e-5                    # row-axis: unaffected by column
    col_omegas = dphi[n:2 * n].abs()

    lambda_max_cells = (2 * math.pi / col_omegas.min()).item()
    lambda_min_cells = (2 * math.pi / col_omegas.max()).item()
    assert lambda_max_cells >= 2 * max(grid_h, grid_w) - 1e-3
    assert lambda_min_cells >= 2.4

    # Sampled check: 2000 distinct in-grid position pairs, full 2n-dim phase
    # vector differs by L-inf > 1e-3 rad (empirical no-alias guard).
    torch.manual_seed(0)
    T = grid_h * grid_w
    gi = torch.randint(0, T, (4000,))
    gj = torch.randint(0, T, (4000,))
    keep = gi != gj
    gi, gj = gi[keep][:2000], gj[keep][:2000]
    assert gi.numel() >= 2000
    phi = torch.atan2(sin[gi], cos[gi])
    phj = torch.atan2(sin[gj], cos[gj])
    dphi_full = torch.atan2(torch.sin(phi - phj), torch.cos(phi - phj))
    assert torch.all(dphi_full.abs().amax(dim=-1) > 1e-3)


def test_loss_finite_n0():
    torch.manual_seed(0)
    m = DetectorNet(32, 32)
    m.train()
    x = torch.randn(2, 1, 32, 32)

    hm, cls = m(x)
    loss = detector_loss(hm, cls, torch.zeros_like(hm), torch.zeros_like(cls), n_vis_batch=0)
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name

    # Repeat with bf16-precision logits (what an autocast forward would hand
    # the loss), cast AFTER a plain-fp32 forward rather than running the model
    # itself under torch.autocast: this machine's oneDNN build has no bf16
    # Conv2d-backward kernel for its (non-AVX-512) CPU -- confirmed by direct
    # probe, an environment/library limitation orthogonal to what's under test
    # (Linear/SDPA/BatchNorm2d/LayerNorm all backward fine in bf16 on this
    # CPU; only oneDNN's conv path lacks the kernel). Casting logits to bf16
    # post-hoc exercises exactly the property the guarantee is about -- the
    # loss's robustness to bf16-precision logit values -- while every
    # parameter's own backward still runs through Conv2d in its native fp32
    # (the model's forward never entered autocast), portable to any CPU.
    m.zero_grad(set_to_none=True)
    hm2, cls2 = m(x)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss2 = detector_loss(hm2.to(torch.bfloat16), cls2.to(torch.bfloat16),
                              torch.zeros_like(hm2), torch.zeros_like(cls2), n_vis_batch=0)
    assert torch.isfinite(loss2)
    loss2.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def _focal_scalar(z, y, alpha=2, beta=4):
    """Independent, unvectorised reference: same formula, plain Python math."""
    p = 1.0 / (1.0 + math.exp(-z))
    if y == 1.0:
        return -((1 - p) ** alpha) * math.log(p)
    return -((1 - y) ** beta) * (p ** alpha) * math.log(1 - p)


def test_loss_y1_branch():
    logits = torch.tensor([2.0, -1.0, 0.5, -3.0]).reshape(1, 1, 2, 2)
    y1 = torch.tensor([1.0, 0.0, 0.0, 0.0]).reshape(1, 1, 2, 2)
    y_near = torch.tensor([0.9999, 0.0, 0.0, 0.0]).reshape(1, 1, 2, 2)

    loss1 = focal(logits, y1)
    loss_near = focal(logits, y_near)
    assert abs(loss1.item() - loss_near.item()) > 1e-3      # exact-1.0 branch fires distinctly

    reference = sum(_focal_scalar(z, yv) for z, yv in
                    zip(logits.flatten().tolist(), y1.flatten().tolist()))
    assert loss1.item() == pytest.approx(reference, abs=1e-5)


def test_forward_deterministic():
    torch.manual_seed(0)
    m = DetectorNet(64, 64).eval()
    x = torch.randn(2, 1, 64, 64)
    hm1, cls1 = m(x)
    hm2, cls2 = m(x)
    assert torch.equal(hm1, hm2)
    assert torch.equal(cls1, cls2)

    r = Refiner().eval()
    xr = torch.randn(3, 1, 24, 24)
    assert torch.equal(r(xr), r(xr))


def test_onnx_export(tmp_path):
    banned = {"Complex", "Loop", "If"}

    m = DetectorNet(240, 320).eval()
    det_path = str(tmp_path / "detector.onnx")
    torch.onnx.export(m, torch.randn(1, 1, 240, 320), det_path, opset_version=17,
                      dynamo=False, input_names=["input"], output_names=["hm", "cls"])
    det_onnx = onnx.load(det_path)
    onnx.checker.check_model(det_onnx)
    det_ops = {n.op_type for n in det_onnx.graph.node}
    assert not (det_ops & banned), det_ops & banned

    r = Refiner().eval()
    ref_path = str(tmp_path / "refiner.onnx")
    torch.onnx.export(r, torch.randn(1, 1, 24, 24), ref_path, opset_version=17,
                      dynamo=False, input_names=["input"], output_names=["logits"])
    ref_onnx = onnx.load(ref_path)
    onnx.checker.check_model(ref_onnx)
    ref_ops = {n.op_type for n in ref_onnx.graph.node}
    assert not (ref_ops & banned), ref_ops & banned


def test_gradflow_gates_and_attention():
    torch.manual_seed(0)
    m = DetectorNet(32, 32)
    m.train()
    hm, cls = m(torch.randn(2, 1, 32, 32))
    loss = detector_loss(hm, cls, torch.zeros_like(hm), torch.zeros_like(cls), n_vis_batch=0)
    loss.backward()

    assert m.gate3.psi.weight.grad is not None
    assert torch.any(m.gate3.psi.weight.grad != 0)
    assert m.blocks[0].qkv.weight.grad is not None
    assert torch.any(m.blocks[0].qkv.weight.grad != 0)


def test_refiner_loss_shape_and_value():
    """Regression guard for the (B,1,64,64) logits vs (B,64,64) targets shape
    mismatch: an unhandled broadcast pairs every logit-crop against every
    target-crop instead of matching them one-to-one. Reference computed via
    per-item focal() calls (B=1 slices), which sidestep the ambiguity."""
    torch.manual_seed(0)
    B = 3
    logits = torch.randn(B, 1, 64, 64)
    targets = torch.zeros(B, 64, 64)
    for i in range(B):
        targets[i, 20 + i, 30] = 1.0

    loss = refiner_loss(logits, targets)
    assert loss.dim() == 0
    assert torch.isfinite(loss)

    reference = sum(focal(logits[i:i + 1], targets[i:i + 1]) for i in range(B)) / B
    assert torch.allclose(loss, reference, atol=1e-4)
