import copy
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import cv2
import numpy as np
import pytest
import yaml
from skimage.exposure import match_histograms

from dcc.board import get_board, render_board
from dcc.synth import _composite_board, _prep_background, cut_refiner_crops, generate_sample, visible
from dcc.refiner_data import _MARGIN, _PATCH, _render_fast_window, fast_refiner_crops, mixed_refiner_crops

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "default.yaml"


@pytest.fixture(scope="module")
def bg_files(tmp_path_factory):
    """4 random-noise 640x640 backgrounds -- no COCO/network dependency
    (copied from tests/test_generator.py's own fixture)."""
    d = tmp_path_factory.mktemp("bgs")
    rng = np.random.default_rng(0xFA57)
    paths = []
    for i in range(4):
        img = rng.integers(0, 256, size=(640, 640, 3), dtype=np.uint8)
        p = d / f"bg{i}.png"
        cv2.imwrite(str(p), img)
        paths.append(str(p))
    return sorted(paths)


@pytest.fixture(scope="module")
def cfg(bg_files):
    with open(CONFIG_PATH) as f:
        c = yaml.safe_load(f)
    c["synth"] = dict(c["synth"])
    c["synth"]["backgrounds"] = str(Path(bg_files[0]).parent)
    return c


def test_window_equivalence(cfg, bg_files):
    """The decisive check (module docstring point 2 in dcc/refiner_data.py):
    the fast arm's local-window render must be pixel-for-pixel what the full
    warp would produce at those coordinates. Photometrics and occlusion off,
    fixed components, the SAME H both paths: _composite_board renders the
    full canvas; cut_refiner_crops harvests its 24x24 exactly as the real
    stream does; _render_fast_window renders the identical window (same
    Hmat, matched board, and background content -- sliced from the very
    same bg_crop, not the fast arm's own cached-raw-decoded approximation,
    which is a separate, already-documented approximation this test isn't
    about) for the same (cx, cy)."""
    W, H = cfg["input_size"]
    size_mult = cfg["synth"]["refiner_res_mult"]
    w2, h2 = W * size_mult, H * size_mult
    render_res = cfg["synth"]["render_res"]
    bcfg = cfg.get("board")
    nx = get_board(bcfg)[1]

    fixed = {"s": 64.0, "theta": 0.2, "shear_x": 0.03, "shear_y": -0.02,
              "tx": 15.0, "ty": -10.0, "tilt": np.radians(15.0), "psi": 1.1, "fov_scale": 1.0}

    bg = cv2.imread(bg_files[0], cv2.IMREAD_COLOR)
    bg_crop = _prep_background(bg, np.random.default_rng(1), cfg["synth"], w2, h2)
    work_full, p_img, Hmat, comps = _composite_board(bg_crop, np.random.default_rng(2), cfg, w2, h2,
                                                       None, size_mult, fixed)

    # a visible corner comfortably interior, so the (24+2*_MARGIN) window
    # sits fully inside bg_crop for the fast-path bg_tile slice below
    interior = [p for p in p_img if visible(tuple(p), [], (w2, h2))
                and 40 <= p[0] <= w2 - 40 and 40 <= p[1] <= h2 - 40]
    assert interior, "fixed geometry produced no comfortably-interior visible corner"
    p = interior[0]

    image_full = cv2.cvtColor(np.clip(work_full, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    crops = cut_refiner_crops(cfg, np.random.default_rng(3), image_full, [p])
    assert crops, "jitter rejected 3x at this seed -- pick a different jitter seed"
    crop_full, d = crops[0]["crop"], crops[0]["d"]
    c = p - d
    cx, cy = int(round(c[0])), int(round(c[1]))

    board_img, _ = render_board(render_res, bcfg)
    board_3ch = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)
    matched = match_histograms(board_3ch, bg_crop, channel_axis=-1).astype(np.float32)
    pf = cfg["synth"]["prefilter"]
    s, SQ = comps["s"], render_res // nx
    if pf["enabled"] and s < SQ:
        sigma_r = pf["k"] * (SQ / s - 1.0)
        if sigma_r > 0.1:
            matched = cv2.GaussianBlur(matched, (0, 0), sigmaX=sigma_r)

    mask_src = np.ones((render_res, render_res), dtype=np.float32)
    wx0, wy0 = cx - 12 - _MARGIN, cy - 12 - _MARGIN
    bg_tile = bg_crop[wy0:wy0 + _PATCH, wx0:wx0 + _PATCH]
    work_win, origin = _render_fast_window(Hmat, matched, mask_src, bg_tile, cx, cy)
    assert origin == (wx0, wy0)

    image_win = cv2.cvtColor(np.clip(work_win, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    crop_win = image_win[_MARGIN:_MARGIN + 24, _MARGIN:_MARGIN + 24]

    diff = np.abs(crop_full.astype(np.int16) - crop_win.astype(np.int16))
    print("test_window_equivalence max abs diff:", diff.max())
    # matched/prefilter are recomputed here but from IDENTICAL inputs (both
    # deterministic, no rng) to what _composite_board used internally, so
    # they're bit-identical; the only remaining divergence source is
    # cv2.warpPerspective inverting two algebraically-equivalent but
    # differently-conditioned matrices (H vs H_win = T @ H) for the two
    # calls -- floating-point op-order, not a math difference.
    assert diff.max() <= 1
    assert np.array_equal(d, p - c)


def test_mixed_frac_boundaries(cfg, bg_files):
    """refiner_full_frac's coin is always drawn first, then delegates: at
    0.0 the coin is always False (fast arm runs on the CONTINUING rng
    stream); at 1.0 it's always True (full arm). Proven by replaying that
    exact sequence independently -- one coin draw, then the delegate call on
    the same rng -- and comparing byte-for-byte, not just checking the
    result is *a* schema-valid crop list (fast and full share that schema,
    so schema alone wouldn't distinguish which arm actually ran)."""
    size_mult = cfg["synth"]["refiner_res_mult"]
    c0 = copy.deepcopy(cfg)
    c0["synth"]["refiner_full_frac"] = 0.0
    c1 = copy.deepcopy(cfg)
    c1["synth"]["refiner_full_frac"] = 1.0

    total0 = total1 = 0
    for seed in range(10):
        mixed0 = mixed_refiner_crops(c0, np.random.default_rng(seed), bg_files)
        rng0 = np.random.default_rng(seed)
        rng0.random()  # the coin, drawn and discarded (False at frac=0.0)
        direct0 = fast_refiner_crops(c0, rng0, bg_files)
        assert len(mixed0) == len(direct0)
        for a, b in zip(mixed0, direct0):
            assert np.array_equal(a["crop"], b["crop"]) and np.array_equal(a["d"], b["d"])
        total0 += len(mixed0)

        mixed1 = mixed_refiner_crops(c1, np.random.default_rng(seed), bg_files)
        rng1 = np.random.default_rng(seed)
        rng1.random()  # the coin, drawn and discarded (True at frac=1.0)
        record, _ = generate_sample(c1, rng1, bg_files, size_mult=size_mult,
                                     occlude=False, force_negative=False)
        pts = [(cc["x"], cc["y"]) for cc in record["corners"] if cc["visible"]]
        direct1 = cut_refiner_crops(c1, rng1, record["image"], pts)
        assert len(mixed1) == len(direct1)
        for a, b in zip(mixed1, direct1):
            assert np.array_equal(a["crop"], b["crop"]) and np.array_equal(a["d"], b["d"])
        total1 += len(mixed1)

        for crops in (mixed0, mixed1):
            for crop in crops:
                assert set(crop.keys()) == {"crop", "d"}
                assert crop["crop"].shape == (24, 24) and crop["crop"].dtype == np.uint8
                assert crop["d"].shape == (2,) and crop["d"].dtype == np.float64

    print("test_mixed_frac_boundaries total crops over 10 seeds: frac0=%d frac1=%d" % (total0, total1))
    assert total0 > 0 and total1 > 0


def test_determinism(cfg, bg_files):
    """Same seed -> byte-identical crop sequence, run twice, for both
    fast_refiner_crops and mixed_refiner_crops (at the real, non-boundary
    default refiner_full_frac)."""
    def run(fn, seed, n_iters):
        rng = np.random.default_rng(seed)
        seq = []
        for _ in range(n_iters):
            seq.extend(fn(cfg, rng, bg_files))
        return seq

    for fn in (fast_refiner_crops, mixed_refiner_crops):
        seq_a = run(fn, 2026, 5)
        seq_b = run(fn, 2026, 5)
        assert len(seq_a) == len(seq_b) > 0
        for a, b in zip(seq_a, seq_b):
            assert np.array_equal(a["crop"], b["crop"])
            assert np.array_equal(a["d"], b["d"])


def test_fast_d_flatness(cfg, bg_files):
    """d over ~2000 fast-arm crops stays flat across the +-3.9375px support
    (linear equal-width bins, each within ~20% of an equal share) -- the
    same U(-refiner_jitter_px, refiner_jitter_px) draw cut_refiner_crops
    itself uses, mirrored exactly (see dcc/refiner_data.py)."""
    rng = np.random.default_rng(4242)
    all_d = []
    while len(all_d) < 2000:
        for crop in fast_refiner_crops(cfg, rng, bg_files):
            all_d.append(crop["d"])
    all_d = np.array(all_d[:2000])
    assert np.abs(all_d).max() <= 3.9375

    pooled = np.concatenate([all_d[:, 0], all_d[:, 1]])
    n_bins = 8
    hist, _ = np.histogram(pooled, bins=np.linspace(-3.9375, 3.9375, n_bins + 1))
    frac = hist / hist.sum()
    print("test_fast_d_flatness bin fractions:", frac.tolist())
    expected = 1.0 / n_bins
    assert np.all(np.abs(frac - expected) <= 0.20 * expected)


def test_throughput_smoke(cfg, bg_files):
    """Not asserted (single process, machine-dependent) -- printed so the
    measured fast-arm speedup can be quoted directly."""
    size_mult = cfg["synth"]["refiner_res_mult"]
    n_target = 200

    rng_fast = np.random.default_rng(7001)
    t0 = time.perf_counter()
    n_fast = 0
    while n_fast < n_target:
        n_fast += len(fast_refiner_crops(cfg, rng_fast, bg_files))
    t_fast = time.perf_counter() - t0

    rng_full = np.random.default_rng(7002)
    t0 = time.perf_counter()
    n_full = 0
    while n_full < n_target:
        record, _ = generate_sample(cfg, rng_full, bg_files, size_mult=size_mult,
                                     occlude=False, force_negative=False)
        pts = [(c["x"], c["y"]) for c in record["corners"] if c["visible"]]
        n_full += len(cut_refiner_crops(cfg, rng_full, record["image"], pts))
    t_full = time.perf_counter() - t0

    rate_fast, rate_full = n_fast / t_fast, n_full / t_full
    print(f"test_throughput_smoke: fast {rate_fast:.1f} crops/s ({n_fast} crops, {t_fast:.2f}s), "
          f"full {rate_full:.1f} crops/s ({n_full} crops, {t_full:.2f}s), "
          f"speedup {rate_fast / rate_full:.2f}x")
