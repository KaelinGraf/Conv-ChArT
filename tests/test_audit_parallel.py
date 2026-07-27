import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import importlib.util
import json
import os
import subprocess

import cv2
import numpy as np
import pytest
import yaml

from dcc.board import get_board, n_corners, render_board
from dcc.synth import generate_sample

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def bg_files(tmp_path_factory):
    """4 random-noise 160x160 backgrounds -- no COCO/network dependency (same
    style as tests/test_generator.py's own bg_files fixture)."""
    d = tmp_path_factory.mktemp("audit_par_bgs")
    rng = np.random.default_rng(0xC0FFEE)
    paths = []
    for i in range(4):
        img = rng.integers(0, 256, size=(160, 160, 3), dtype=np.uint8)
        p = d / f"bg{i}.png"
        cv2.imwrite(str(p), img)
        paths.append(str(p))
    return sorted(paths)


@pytest.fixture(scope="module")
def tiny_config_path(tmp_path_factory, bg_files):
    """configs/default.yaml shrunk for speed: small canvas/render resolution
    and a tiny refiner_val_composites. The refiner content-check's own
    range(500) loop isn't config-driven (tools/audit.py's own fixed scale),
    so it still runs in full -- at this resolution that stays fast."""
    with open(ROOT / "configs" / "default.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["input_size"] = [96, 64]
    cfg["scale_range_px"] = [8, 24]
    cfg["synth"] = dict(cfg["synth"])
    cfg["synth"]["render_res"] = 60  # multiple of the default board's nx=5
    cfg["synth"]["backgrounds"] = str(Path(bg_files[0]).parent)
    cfg["synth"]["cutouts"] = dict(cfg["synth"]["cutouts"])
    cfg["synth"]["cutouts"]["path"] = None
    cfg["synth"]["refiner_val_composites"] = 6
    path = tmp_path_factory.mktemp("audit_par_cfg") / "tiny.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


@pytest.fixture(scope="module")
def audit_mod():
    """tools/audit.py, loaded by absolute path -- see tests/test_guards.py's
    identical pattern; the machine-wide detectron2 `tools` package shadow
    makes `import tools.audit` unreliable. In-process calls into this module
    (this file's first three tests) are unaffected by how it was loaded --
    only a REAL cross-process multiprocessing.Pool dispatch would need the
    module resolvable by name from a fresh child, which a dynamically-loaded
    module here is not (see _run_audit_cli below for how the parallel path
    is actually exercised)."""
    spec = importlib.util.spec_from_file_location("_audit_under_test", ROOT / "tools" / "audit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_pool_serial_matches_direct_calls(audit_mod, tiny_config_path, bg_files):
    """_run_pool's workers<=1 branch is the in-process fallback every
    --workers 1 audit run takes; it must return exactly what calling the
    worker directly, in order, would."""
    with open(tiny_config_path) as f:
        cfg = yaml.safe_load(f)
    val_seed = cfg["synth"]["val_seed"]
    worker = audit_mod.functools.partial(audit_mod._dist_worker, generate_sample, val_seed, 12, False)
    pooled = audit_mod._run_pool(worker, range(12), cfg, bg_files, 1)
    direct = [audit_mod._dist_worker(generate_sample, val_seed, 12, False, i) for i in range(12)]
    assert pooled == direct
    # workers=0 (e.g. a maxed-out os.cpu_count()-4) must degrade to the same
    # plain loop, not attempt Pool(0) (which raises)
    assert audit_mod._run_pool(worker, range(3), cfg, bg_files, 0) == direct[:3]


def test_recompose_corners_nx_threading(audit_mod, bg_files):
    """_recompose_corners must derive SQ from the SAME per-side square count
    dcc.synth used to build H (nx from cfg["board"]), not a hardcoded 5 --
    proven on a non-default board where the two disagree: the old formula
    visibly diverges, the nx-threaded one doesn't."""
    with open(ROOT / "configs" / "default.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["board"] = {"squares": [4, 4], "dictionary": "DICT_4X4_50", "marker_ratio": 0.7}
    cfg["synth"] = dict(cfg["synth"])
    cfg["synth"]["render_res"] = 60
    cfg["synth"]["backgrounds"] = str(Path(bg_files[0]).parent)
    cfg["synth"]["cutouts"] = dict(cfg["synth"]["cutouts"])
    cfg["synth"]["cutouts"]["path"] = None

    nx = get_board(cfg["board"])[1]
    assert nx == 4
    render_res = cfg["synth"]["render_res"]
    W, H = cfg["input_size"]
    _, corner_px = render_board(render_res, cfg["board"])
    n_cls = n_corners(cfg["board"])
    assert n_cls == 9

    rng = np.random.default_rng([cfg["synth"]["val_seed"], 0])
    record, meta = generate_sample(cfg, rng, bg_files, photometric=False, occlude=False, force_negative=False)
    record_xy = np.zeros((n_cls, 2))
    for c in record["corners"]:
        record_xy[c["index"]] = (c["x"], c["y"])

    correct = audit_mod._recompose_corners(meta["components"], corner_px, render_res, nx, W, H)
    err_correct = float(np.linalg.norm(correct - record_xy, axis=1).max())
    assert err_correct < 0.01, f"nx-threaded recomposition should match generate_sample, got err={err_correct}"

    wrong = audit_mod._recompose_corners(meta["components"], corner_px, render_res, 5, W, H)
    err_wrong = float(np.linalg.norm(wrong - record_xy, axis=1).max())
    assert err_wrong > 0.01, "hardcoded nx=5 should visibly diverge from generate_sample on a 4x4 board"


def test_gate_roundtrip_empty_input(audit_mod, tiny_config_path, bg_files):
    """n_roundtrip=0 must not crash (max() on an empty error list) -- matches
    the pre-parallelisation code's worst=0.0 default for a loop that never
    runs."""
    from dcc import board as board_mod
    with open(tiny_config_path) as f:
        cfg = yaml.safe_load(f)
    ok, worst = audit_mod._gate_roundtrip(generate_sample, board_mod, cfg, bg_files, cfg["synth"]["val_seed"],
                                           0, 1)
    assert ok is True and worst == 0.0


def _run_audit_cli(config_path, out_dir, workers):
    """Real subprocess invocation of tools/audit.py -- the only reliable way
    to exercise its actual spawn-Pool parallel path: a module loaded via
    spec_from_file_location (the audit_mod fixture above) has no filesystem-
    discoverable name a spawned child could re-import it under, so real
    cross-process dispatch has to go through the script entry point
    (main()'s own __name__ == "__main__" guard), exactly as a user would
    invoke it -- mirroring how _gate_repeatability itself, inside audit.py,
    verifies across a real process boundary by running a fresh subprocess
    rather than reaching into another process's objects."""
    env = {**os.environ, "PYTHONPATH": ""}
    args = [sys.executable, str(ROOT / "tools" / "audit.py"), "--config", str(config_path),
            "--out", str(out_dir), "--n-dist", "12", "--n-overlay", "4", "--n-roundtrip", "6",
            "--workers", str(workers)]
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, env=env, timeout=120)
    assert result.returncode in (0, 1), f"audit.py crashed (rc={result.returncode}):\n{result.stderr[-4000:]}"
    return json.loads((out_dir / "report.json").read_text())


def test_serial_vs_parallel_report_equivalence(tiny_config_path, tmp_path):
    """The decisive check: --workers 1 (plain loop) and --workers 2 (real
    spawn Pool) must produce identical numeric gates -- every sample is
    independently seeded off (val_seed, index), so pooling changes wall
    time, never a value."""
    serial = _run_audit_cli(tiny_config_path, tmp_path / "serial", 1)
    parallel = _run_audit_cli(tiny_config_path, tmp_path / "parallel", 2)

    for key in ("roundtrip_max_px", "repeatability_ok", "gates", "backgrounds_sha1",
                "generator_fingerprint", "distributions", "refiner"):
        assert serial[key] == parallel[key], f"{key} differs:\nserial={serial[key]!r}\nparallel={parallel[key]!r}"
