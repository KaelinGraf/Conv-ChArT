import importlib.util
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pytest

from dcc.board import render_board

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def ec():
    """tools.eval_classical, loaded by absolute path -- an unrelated
    detectron2 'tools' package on sys.path shadows any
    `import tools.eval_classical` (same footgun as tools/preflight.py, see
    tests/test_guards.py)."""
    spec = importlib.util.spec_from_file_location("_eval_classical_under_test", ROOT / "tools" / "eval_classical.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------- board convention --

def test_board_convention(ec):
    """The decisive test: classical charucoIds recovered from a clean
    render must land at dcc.board's own analytic corner_px(i) -- see
    ec.verify_board_convention's docstring. A mismatch here invalidates
    every ID comparison the eval tool ever makes."""
    max_err, n_matched = ec.verify_board_convention()
    assert n_matched == 16  # default 5x5 board, (5-1)^2 inner corners
    assert max_err < 1.0  # verify_board_convention's own gate; measured ~0.7px (subpixel refinement noise)


def test_board_convention_non_vacuous(ec):
    """Proves the check above isn't trivially passing: feeding it a
    deliberately shuffled corner-index mapping (simulating a real convention
    mismatch, e.g. a legacy-pattern or row/col-order slip) must fail loudly,
    not silently pass."""
    board, dictionary, nx = ec._get_board_and_dictionary(None)
    img, analytic_corners = render_board(480, None)
    ch_xy, ch_ids = ec.classical_charuco_detect(img, board, dictionary, ec._detector_params())
    assert ch_xy is not None and len(ch_ids) == 16

    shuffled = analytic_corners[np.random.default_rng(0).permutation(16)]
    err = np.linalg.norm(ch_xy - shuffled[ch_ids], axis=1)
    assert float(err.max()) > 1.0  # a broken convention must NOT sneak under the 1.0px gate


# ---------------------------------------------------------------- matching --

def test_match_ided_correct_and_wrong(ec):
    gt = {0: (10.0, 10.0), 1: (50.0, 10.0), 2: (90.0, 10.0)}
    det_xy = np.array([[10.2, 9.9],   # correct, near index 0
                        [51.0, 20.0],  # id 1 but 11px off -> wrong (loc mismatch)
                        [200.0, 200.0]])  # id 7, not in gt at all -> wrong (off-board)
    det_ids = np.array([0, 1, 7])
    n_correct, n_wrong, errs = ec._match_ided(gt, det_xy, det_ids, tol=5.0)
    assert n_correct == 1
    assert n_wrong == 2
    assert len(errs) == 1 and errs[0] < 1.0


def test_match_ided_no_detections(ec):
    assert ec._match_ided({0: (0.0, 0.0)}, None, None) == (0, 0, [])


def test_match_anonymous_claims_each_gt_once(ec):
    gt = {0: (10.0, 10.0), 1: (10.5, 10.5)}  # two GT corners very close together
    anon_xy = np.array([[10.1, 10.1]])  # a single detection near both
    n_matched, errs = ec._match_anonymous(gt, anon_xy, tol=3.0)
    assert n_matched == 1  # only one GT corner can claim the one detection
    assert len(errs) == 1


def test_match_anonymous_respects_tolerance(ec):
    gt = {0: (10.0, 10.0)}
    anon_xy = np.array([[50.0, 50.0]])
    n_matched, errs = ec._match_anonymous(gt, anon_xy, tol=3.0)
    assert n_matched == 0 and errs == []


# -------------------------------------------------------------- SNR tiers --

def test_tier_cfgs_isolate_effects(ec):
    """clean skips photometrics entirely; mid is an isolated single-effect
    probe (every other *_p key zeroed); deployment-dark/overcast are each
    "post rev-2 generator" (full stack at normal probabilities, only
    differencing forced/biased) -- the constructions build_tier_cfgs's
    docstring claims."""
    import yaml
    cfg = yaml.safe_load(open(ROOT / "configs" / "default.yaml"))
    tiers = ec.build_tier_cfgs(cfg)
    base_ph = cfg["synth"]["photometric"]

    clean_cfg, clean_photometric = tiers["clean"]
    assert clean_photometric is False

    mid_ph = tiers["mid"][0]["synth"]["photometric"]
    assert mid_ph["gauss_noise_p"] == 1.0 and mid_ph["gauss_blur_p"] == 1.0
    assert all(v == 0.0 for k, v in mid_ph.items()
               if k.endswith("_p") and k not in ("gauss_noise_p", "gauss_blur_p"))

    # deployment-dark := MIDDAY (team-lead 2026-07-28 gate-4 ruling: the
    # actual daytime rig physics, LED barely above a 90% daylight pedestal).
    dark_ph = tiers["deployment-dark"][0]["synth"]["photometric"]
    assert dark_ph["differencing_p"] == 1.0
    assert dark_ph["differencing_ambient"] == [0.8, 0.95]
    assert dark_ph["differencing_illum_peak"] == [0.15, 0.3]
    assert all(dark_ph[k] == base_ph[k] for k in base_ph
               if k.endswith("_p") and k not in ("differencing_p",))

    # overcast := kept as its own reported tier (measured non-collapse, not
    # folded into deployment-dark -- see build_tier_cfgs's docstring).
    overcast_ph = tiers["overcast"][0]["synth"]["photometric"]
    assert overcast_ph["differencing_p"] == 1.0
    assert overcast_ph["differencing_ambient"] == [0.4, 0.55]
    assert overcast_ph["differencing_illum_peak"] == [0.3, 0.6]
    assert all(overcast_ph[k] == base_ph[k] for k in base_ph
               if k.endswith("_p") and k not in ("differencing_p",))

    # original cfg's own photometric dict must be untouched (deepcopy, not aliasing)
    assert cfg["synth"]["photometric"]["gauss_noise_p"] == 0.5
