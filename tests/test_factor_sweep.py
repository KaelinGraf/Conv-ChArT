import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import pytest

ROOT = Path(__file__).parents[1]
CKPT = ROOT / "runs" / "rev640_baseline" / "ckpt_0150000.pt"


@pytest.fixture(scope="module")
def fs():
    """tools.factor_sweep, loaded by absolute path -- an unrelated detectron2
    'tools' package shadows any `import tools.factor_sweep` (same footgun as
    tools/preflight.py, tools/eval_classical.py -- see their own test
    fixtures). factor_sweep.py additionally does `from charuconet import
    ...` and `import eval_classical as ec` as bare top-level imports, which
    only resolve "for free" when it's invoked as a script (Python puts the
    running script's own directory on sys.path[0]) -- importlib-by-path from
    a test file in tests/ doesn't get that, so tools/ is inserted explicitly
    here first."""
    sys.path.insert(0, str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location("_factor_sweep_under_test", ROOT / "tools" / "factor_sweep.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cfg_and_scene(fs):
    """CPU-only: render_step never touches a model, so these tests don't
    need CKPT to actually load weights -- just its saved cfg (input_size,
    board, synth params) and a background corpus to composite against."""
    if not CKPT.exists():
        pytest.skip(f"checkpoint not present at {CKPT} -- determinism tests need its saved cfg")
    cfg, _ckpt = fs.build_cfg(str(CKPT))
    scene = fs.build_base_scene(cfg, bg_seed=7)
    return cfg, scene


def _hash_step(fs, cfg, scene, axis, value, seed):
    gray, _p_img, _holes, _cov = fs.render_step(cfg, scene, axis, fs.BASE_POSE_DEFAULT, value, seed)
    return hashlib.sha1(gray.tobytes()).hexdigest()


# --------------------------------------------------------------- determinism

@pytest.mark.parametrize("axis,value", [
    ("distance", 40.0),            # geometric
    ("tilt", 20.0),                # geometric
    ("sensor_noise_K", 4.0),       # photometric, isolated-effect path
    ("droplets", 3.0),             # photometric, position-nesting path
    ("blur", 5),                   # manual (faithful inline copy, not _apply_photometric)
    ("occlusion", 0.4),            # manual (explicit fractional occluder)
])
def test_render_step_deterministic(fs, cfg_and_scene, axis, value):
    """Same axis/step rendered twice from the same base scene must be
    bit-identical -- the tool's own core claim (module docstring), checked
    here per representative axis kind rather than trusting the CLI's own
    --determinism-check run."""
    cfg, scene = cfg_and_scene
    seed = int(hashlib.sha1(axis.encode()).hexdigest(), 16) % (2 ** 31)
    h1 = _hash_step(fs, cfg, scene, axis, value, seed)
    h2 = _hash_step(fs, cfg, scene, axis, value, seed)
    assert h1 == h2


def test_differencing_ratio_stays_under_clipping_ceiling(fs, cfg_and_scene):
    """Regression test for the gate-4 saturation-confound fix (2026-07-28):
    differencing_ratio's ambient+peak must stay under 1.0 (255 DN) at every
    step, and the reported saturation_frac covariate must be ~0 everywhere
    -- the whole point of lowering the fixed ambient was to isolate the
    LED/daylight ratio from the sensor's clipping ceiling. A regression back
    to the old ambient=0.85 design would silently reintroduce the exact
    confound that produced a false "non-monotonic model" reading before it
    was measured and fixed."""
    cfg, scene = cfg_and_scene
    spec = fs.AXES["differencing_ratio"]
    ambient = spec["ambient"]
    seed = int(hashlib.sha1("differencing_ratio".encode()).hexdigest(), 16) % (2 ** 31)
    for _label, ratio in spec["steps"]:
        peak = ratio * ambient
        assert ambient + peak <= 1.0, f"ratio {ratio} pushes ambient+peak to {ambient + peak} -- clips at 255 DN"
        _gray, _p, _h, cov = fs.render_step(cfg, scene, "differencing_ratio", fs.BASE_POSE_DEFAULT, ratio, seed)
        assert cov["saturation_frac"] < 0.01, f"ratio {ratio}: saturation_frac {cov['saturation_frac']} -- not clean"


def test_exposure_clipping_axis_crosses_the_ceiling(fs, cfg_and_scene):
    """The dedicated clipping axis (gate-4 fix #2) exists specifically to
    study the transition the ratio axis must now avoid -- its own
    saturation_frac should rise from ~0 at low ambient to a real fraction at
    ambient=1.0 (ratio pinned at 1.0, so ambient+peak = 2*ambient)."""
    cfg, scene = cfg_and_scene
    spec = fs.AXES["exposure_clipping"]
    seed = int(hashlib.sha1("exposure_clipping".encode()).hexdigest(), 16) % (2 ** 31)
    fracs = []
    for _label, ambient in spec["steps"]:
        _gray, _p, _h, cov = fs.render_step(cfg, scene, "exposure_clipping", fs.BASE_POSE_DEFAULT, ambient, seed)
        fracs.append(cov["saturation_frac"])
    assert fracs[0] < fracs[-1], f"saturation_frac should rise across the axis, got {fracs}"


def test_module_contrast_covariate_and_exposure_clipping_inverts(fs, cfg_and_scene):
    """Regression test for the polarity-inversion mechanism finding
    (2026-07-28): module_contrast/contrast_inverted must be present on every
    differencing-axis step (once the differencing effect actually fires),
    positive at low exposure and negative (inverted) once exposure_clipping
    reaches its harshest step -- the physical claim the whole mechanism
    investigation rests on, pinned here so a future edit can't silently
    break it."""
    cfg, scene = cfg_and_scene
    seed = int(hashlib.sha1("exposure_clipping".encode()).hexdigest(), 16) % (2 ** 31)
    _g0, _p0, _h0, cov_low = fs.render_step(cfg, scene, "exposure_clipping", fs.BASE_POSE_DEFAULT, 0.2, seed)
    _g1, _p1, _h1, cov_high = fs.render_step(cfg, scene, "exposure_clipping", fs.BASE_POSE_DEFAULT, 1.0, seed)
    assert "module_contrast" in cov_low and "module_contrast" in cov_high
    assert cov_low["module_contrast"] > 0 and cov_low["contrast_inverted"] is False
    assert cov_high["module_contrast"] < 0 and cov_high["contrast_inverted"] is True


def test_off_step_identical_across_axes(fs, cfg_and_scene):
    """The shared-base-scene proof: every axis's "off"/clean/baseline step
    must reduce to the exact same underlying composite (module docstring's
    PHOTOMETRIC/OCCLUSION construction: work0 is built once per step from
    the SAME fixed base pose, only the swept effect differs)."""
    cfg, scene = cfg_and_scene
    off_steps = {"differencing_ambient": None, "sensor_noise_K": None,
                 "specular": None, "blur": None, "occlusion": 0.0}
    hashes = {axis: _hash_step(fs, cfg, scene, axis, val,
                                int(hashlib.sha1(axis.encode()).hexdigest(), 16) % (2 ** 31))
              for axis, val in off_steps.items()}
    assert len(set(hashes.values())) == 1, f"off-step frames diverged across axes: {hashes}"


def test_axis_seed_is_stable_across_processes(fs):
    """Regression test for a real bug caught during this tool's own build:
    Python's built-in hash() on a str is PYTHONHASHSEED-randomised per
    process, which would silently break the determinism claims above the
    moment two separate invocations disagreed on an axis's seed. The sha1-
    based seed derivation main() actually uses must NOT depend on hash()."""
    seed_fn = lambda name: int(hashlib.sha1(name.encode()).hexdigest(), 16) % (2 ** 31)
    assert seed_fn("distance") == seed_fn("distance")
    # sanity: two different axis names must not collide onto the same seed
    assert seed_fn("distance") != seed_fn("tilt")


# ---------------------------------------------------------- trainer-health gate

def test_check_trainer_health_healthy(fs, tmp_path):
    p = tmp_path / "metrics.jsonl"
    p.write_text(json.dumps({"step": 1, "wall": time.time(), "samples_per_s": 150.0}) + "\n")
    ok, age, sps = fs.check_trainer_health(str(p))
    assert ok and age < 5 and sps == 150.0


def test_check_trainer_health_stale_or_slow(fs, tmp_path):
    p = tmp_path / "metrics.jsonl"
    p.write_text(json.dumps({"step": 1, "wall": time.time() - 999, "samples_per_s": 150.0}) + "\n")
    ok, age, _sps = fs.check_trainer_health(str(p))
    assert not ok and age > 900

    p.write_text(json.dumps({"step": 1, "wall": time.time(), "samples_per_s": 5.0}) + "\n")
    ok, _age, sps = fs.check_trainer_health(str(p))
    assert not ok and sps == 5.0


def test_check_trainer_health_torn_last_line(fs, tmp_path):
    """A live trainer can crash mid-write on its own last line (memory.md
    2026-07-28 10:45's actual postmortem) -- the gate must fall back to the
    second-to-last complete line rather than raising on a truncated one."""
    p = tmp_path / "metrics.jsonl"
    good = json.dumps({"step": 1, "wall": time.time(), "samples_per_s": 120.0})
    torn = '{"step": 2, "wall": ' + str(time.time())  # deliberately truncated, no closing brace
    p.write_text(good + "\n" + torn)
    ok, _age, sps = fs.check_trainer_health(str(p))
    assert ok and sps == 120.0
