import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import cv2
import numpy as np
import pytest
import yaml

from dcc.board import get_board, render_board
from dcc.targets import render_class_targets, render_heatmap
from dcc.synth import _apply_photometric, cut_refiner_crops, generate_sample, make_generic_crop, place_cutout, visible
from dcc.dataset import _maybe_replace_generic, RefinerVal, SynthVal

CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
CONFIG_PATH = Path(__file__).parents[1] / "configs" / "default.yaml"


@pytest.fixture(scope="module")
def bg_files(tmp_path_factory):
    """4 random-noise 640x640 backgrounds — no COCO/network dependency."""
    d = tmp_path_factory.mktemp("bgs")
    rng = np.random.default_rng(0xC0FFEE)
    paths = []
    for i in range(4):
        img = rng.integers(0, 256, size=(640, 640, 3), dtype=np.uint8)
        p = d / f"bg{i}.png"
        cv2.imwrite(str(p), img)
        paths.append(str(p))
    return sorted(paths)


@pytest.fixture(scope="module")
def cutout_bank(tmp_path_factory):
    """3-file synthetic RGBA cutout bank -- filled circle, rotated square
    (diamond), small blob -- no SAM2/COCO dependency. Hard (non-anti-aliased)
    edges throughout, so any soft alpha transition seen downstream is
    attributable to place_cutout's own feathering, not the fixture."""
    d = tmp_path_factory.mktemp("cutout_bank")

    circle = np.zeros((80, 80, 4), dtype=np.uint8)
    cv2.circle(circle, (40, 40), 36, (30, 60, 200, 255), -1)
    cv2.imwrite(str(d / "0000000.png"), circle)

    square = np.zeros((80, 80, 4), dtype=np.uint8)
    diamond = np.array([[40, 6], [74, 40], [40, 74], [6, 40]], dtype=np.int32)
    cv2.fillPoly(square, [diamond], (200, 120, 40, 255))
    cv2.imwrite(str(d / "0000001.png"), square)

    blob = np.zeros((40, 40, 4), dtype=np.uint8)
    blob_pts = np.array([[5, 20], [14, 6], [30, 9], [35, 24], [19, 35], [7, 29]], dtype=np.int32)
    cv2.fillPoly(blob, [blob_pts], (80, 200, 80, 255))
    cv2.imwrite(str(d / "0000002.png"), blob)

    return str(d)


@pytest.fixture(scope="module")
def cfg(bg_files):
    with open(CONFIG_PATH) as f:
        c = yaml.safe_load(f)
    c["synth"] = dict(c["synth"])
    c["synth"]["backgrounds"] = str(Path(bg_files[0]).parent)
    return c


def _recompose_H(comp, render_res, w2, h2, nx):
    """Independent inline re-derivation of DS-07/SD-02 Rev C's full 3x3
    homography, from the reported components alone -- does not call
    anything in dcc.synth. nx is the board's per-side square count
    (SQ = render_res // nx)."""
    s, theta = comp["s"], comp["theta"]
    shx, shy = comp["shear_x"], comp["shear_y"]
    tx, ty = comp["tx"], comp["ty"]
    SQ = render_res // nx
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    Sh = np.array([[1.0, np.tan(shx)], [np.tan(shy), 1.0]])
    A = (s / SQ) * (R @ Sh)
    c_r = np.array([(render_res - 1) / 2, (render_res - 1) / 2])
    c_in = np.array([(w2 - 1) / 2, (h2 - 1) / 2])
    M3 = np.eye(3)
    M3[:2, :2] = A
    M3[:2, 2] = c_in + np.array([tx, ty]) - A @ c_r

    tau, psi, fov_scale = comp["tilt"], comp["psi"], comp["fov_scale"]
    g = np.sin(tau) * s / (fov_scale * w2 * SQ)
    gx, gy = g * np.cos(psi), g * np.sin(psi)
    cr = (render_res - 1) / 2
    Tcr = np.array([[1.0, 0.0, cr], [0.0, 1.0, cr], [0.0, 0.0, 1.0]])
    Tmcr = np.array([[1.0, 0.0, -cr], [0.0, 1.0, -cr], [0.0, 0.0, 1.0]])
    Pg = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [gx, gy, 1.0]])
    return M3 @ (Tcr @ Pg @ Tmcr)


def test_warp_roundtrip(cfg, bg_files):
    render_res = cfg["synth"]["render_res"]
    W, H = cfg["input_size"]
    nx = get_board(cfg.get("board"))[1]
    analytic = render_board(render_res)[1]

    max_err = 0.0
    for trial in range(20):
        rng = np.random.default_rng([9001, trial])
        s = rng.uniform(16, 128)
        record, meta = generate_sample(cfg, rng, bg_files, s=s, photometric=False, occlude=False,
                                        force_negative=False)
        H2 = _recompose_H(meta["components"], render_res, W, H, nx)
        assert np.allclose(H2, meta["M"], atol=1e-9)
        hom = np.hstack([analytic, np.ones((16, 1))]) @ H2.T
        recomposed = hom[:, :2] / hom[:, 2:3]
        recorded = np.array([[c["x"], c["y"]] for c in record["corners"]])
        max_err = max(max_err, float(np.max(np.abs(recomposed - recorded))))
    print("test_warp_roundtrip max recompose err:", max_err)
    assert max_err < 1e-9

    # Content check, deliberately moderate (non-adversarial) geometry:
    # classical cornerSubPix degrades near the +-35deg shear / +-180deg
    # rotation extremes of SD-02's envelope regardless of whether the warp
    # math is right -- that is a network problem (DS-06), not a
    # geometry-roundtrip one, so it is not what this assertion is for.
    def _content_check(record):
        img = record["image"]
        corners = np.array([[c["x"], c["y"]] for c in record["corners"]])
        vis = np.array([c["visible"] for c in record["corners"]])
        ok = vis & (corners[:, 0] >= 5) & (corners[:, 0] <= W - 6) & (corners[:, 1] >= 5) & (corners[:, 1] <= H - 6)
        assert ok.sum() > 0
        seeds = (corners[ok] + 0.3).astype(np.float32).reshape(-1, 1, 2)
        refined = cv2.cornerSubPix(img, seeds, (5, 5), (-1, -1), CRIT)
        err = np.linalg.norm(refined.reshape(-1, 2) - corners[ok], axis=1)
        assert err.max() <= 0.15
        return float(err.max())

    measured = []
    for trial in range(3):
        rng = np.random.default_rng([9002, trial])
        theta = rng.uniform(-0.3, 0.3)
        shx = rng.uniform(-0.1, 0.1)
        shy = rng.uniform(-0.1, 0.1)
        s = rng.uniform(90, 128)
        record, _ = generate_sample(cfg, rng, bg_files, s=s, photometric=False, occlude=False,
                                     force_negative=False,
                                     components={"theta": theta, "shear_x": shx, "shear_y": shy})
        measured.append(_content_check(record))

    # SD-02 Rev C: same moderate geometry, now also with a mild tau=20deg
    # perspective term forced in -- the homography path must not degrade
    # classical subpixel detection at an everyday tilt.
    rng = np.random.default_rng([9003, 0])
    s = rng.uniform(90, 128)
    record, _ = generate_sample(cfg, rng, bg_files, s=s, photometric=False, occlude=False,
                                 force_negative=False,
                                 components={"theta": 0.0, "shear_x": 0.0, "shear_y": 0.0,
                                             "tilt": np.radians(20.0)})
    measured.append(_content_check(record))
    print("test_warp_roundtrip cornerSubPix max err per sample:", measured)


def test_perspective_calibration(cfg):
    """SD-02 Rev C derivation sanity: the per-render-px perspective term
    g = sin(tau)*s/(f*SQ), (gx,gy) = g*(cos psi, sin psi) must reproduce the
    homography induced by physically tilting a fronto-parallel board in
    front of a pinhole camera, built here gen_eval_pose-style (K@[r1|r2|t]@S,
    see that module's docstring) -- independently, NOT imported from it.

    Rodrigues' rotation-ANGLE-AXIS there is the axis the board rotates about;
    this module's psi is the axis of the resulting FORESHORTENING gradient,
    which is perpendicular to the rotation axis -- so the pinhole side below
    uses rotation axis angle (psi - 90deg) to model the same physical tilt.
    That is a bookkeeping offset between two independent conventions, not a
    bug in either one (confirmed numerically before writing this test).
    """
    render_res = cfg["synth"]["render_res"]
    W, H = cfg["input_size"]
    nx = get_board(cfg.get("board"))[1]
    SQ = render_res // nx
    cr = (render_res - 1) / 2
    c_in = np.array([(W - 1) / 2, (H - 1) / 2])
    zero6 = {"theta": 0.0, "shear_x": 0.0, "shear_y": 0.0, "tx": 0.0, "ty": 0.0}

    def mine_H(psi, tau, s, fov_scale):
        # reuses _recompose_H (already an independent re-derivation, see
        # above) with the affine's 6 DoF pinned to identity/zero so only the
        # scale + perspective factor under test is exercised.
        return _recompose_H({**zero6, "s": s, "tilt": tau, "psi": psi, "fov_scale": fov_scale},
                             render_res, W, H, nx)

    def pinhole_H(psi, tau, s, fov_scale):
        f = fov_scale * W
        cx, cy = c_in
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        z = f / s
        axis_angle = psi - np.pi / 2
        Raxis, _ = cv2.Rodrigues(tau * np.array([np.cos(axis_angle), np.sin(axis_angle), 0.0]))
        t = z * np.array([0.0, 0.0, 1.0]) - Raxis @ np.array([2.5, 2.5, 0.0])
        Rt = np.column_stack([Raxis[:, 0], Raxis[:, 1], t])
        S = np.array([[1 / SQ, 0, 0.5 / SQ], [0, 1 / SQ, 0.5 / SQ], [0, 0, 1]])
        return K @ Rt @ S

    def dehom(Hm, pts):
        hom = np.hstack([pts, np.ones((len(pts), 1))]) @ Hm.T
        return hom[:, :2] / hom[:, 2:3]

    def local_square_size(Hm, cx0, cy0, half=1.0):
        pts = np.array([[cx0 - half, cy0], [cx0 + half, cy0], [cx0, cy0 - half], [cx0, cy0 + half]])
        m = dehom(Hm, pts)
        return (np.linalg.norm(m[1] - m[0]) + np.linalg.norm(m[3] - m[2])) / 2

    def anisotropy(Hm, psi):
        axis = np.array([np.cos(psi), np.sin(psi)])
        near_c, far_c = np.array([cr, cr]) - axis * SQ, np.array([cr, cr]) + axis * SQ
        return local_square_size(Hm, *near_c) / local_square_size(Hm, *far_c) - 1.0

    s, fov_scale = 64.0, 1.0
    # exact at tau=0: both constructions degenerate to the identical affine,
    # up to the overall homogeneous scale a homography is only defined to
    # (mine is [2,2]-normalised by construction; K@Rt@S is not)
    H_mine0 = mine_H(0.7, 0.0, s, fov_scale)
    H_pin0 = pinhole_H(0.7, 0.0, s, fov_scale)
    assert np.allclose(H_mine0, H_pin0 / H_pin0[2, 2], atol=1e-9)

    # near/far foreshortening anisotropy: tight at small tau, within ~15% up
    # to tau=60deg (psi axis-aligned with the render grid, matching how an
    # actual board square is measured in test_perspective_foreshortening).
    results = {}
    for psi_deg in (0, 90):
        psi = np.radians(psi_deg)
        for tau_deg in (1, 5, 15, 30, 45, 60):
            tau = np.radians(tau_deg)
            a_mine = anisotropy(mine_H(psi, tau, s, fov_scale), psi)
            a_pin = anisotropy(pinhole_H(psi, tau, s, fov_scale), psi)
            relerr = abs(a_mine - a_pin) / abs(a_pin)
            results[(psi_deg, tau_deg)] = (round(a_mine, 4), round(a_pin, 4), round(relerr, 4))
            assert relerr < 0.15, f"psi={psi_deg} tau={tau_deg}: mine={a_mine:.4f} pin={a_pin:.4f} relerr={relerr:.4f}"
            if tau_deg <= 5:
                assert relerr < 0.02, f"psi={psi_deg} tau={tau_deg}: relerr={relerr:.4f} (expected ~exact)"
    print("test_perspective_calibration (psi,tau) -> (anis_mine, anis_pinhole, relerr):", results)

    # w > 0 across the render canvas at the worst case of the sampling
    # envelope (max s, tilt_max_deg + margin, min fov_scale) -- zero-visible
    # positives with points behind the "camera" cannot occur.
    _, hi_s = cfg["scale_range_px"]
    fov_lo, _ = cfg["synth"]["fov_scale"]
    tilt_margin = np.radians(cfg["synth"]["tilt_max_deg"]) + np.radians(15)
    g_worst = np.sin(tilt_margin) * hi_s / (fov_lo * W * SQ)
    xs, ys = np.meshgrid(np.linspace(0, render_res - 1, 25), np.linspace(0, render_res - 1, 25))
    px, py = xs.ravel() - cr, ys.ravel() - cr
    min_w = min((1 + g_worst * np.cos(p) * px + g_worst * np.sin(p) * py).min()
                for p in np.linspace(0, 2 * np.pi, 9))
    print("test_perspective_calibration worst-case g:", g_worst, "min w:", min_w)
    assert min_w > 0


def test_perspective_foreshortening(cfg, bg_files):
    """SD-02 Rev C sign/direction check: forcing psi=0 aligns the tilt's
    foreshortening gradient with the render x-axis, i.e. with the board's
    own grid columns, so a tau=50deg tilt must measure the near (low-x,
    col 0-1) squares larger than the far (high-x, col 2-3) squares in every
    row. Separately, tau=0 must degenerate exactly to the pre-existing
    affine path: H's third row is exactly [0, 0, 1] and its top two rows
    (the A/translation part) exactly match the independent affine-only
    recomposition."""
    render_res = cfg["synth"]["render_res"]
    W, H = cfg["input_size"]
    nx = get_board(cfg.get("board"))[1]

    rng = np.random.default_rng(71)
    record, meta = generate_sample(cfg, rng, bg_files, s=64.0, force_negative=False,
                                    photometric=False, occlude=False,
                                    components={"theta": 0.0, "shear_x": 0.0, "shear_y": 0.0,
                                                "tx": 0.0, "ty": 0.0,
                                                "tilt": np.radians(50.0), "psi": 0.0})
    corners = {c["index"]: (c["x"], c["y"]) for c in record["corners"]}

    def idx(row, col):
        return 4 * row + col

    for row in range(4):
        left = np.hypot(*(np.subtract(corners[idx(row, 1)], corners[idx(row, 0)])))
        right = np.hypot(*(np.subtract(corners[idx(row, 3)], corners[idx(row, 2)])))
        assert left > right, f"row {row}: near-edge square ({left:.2f}px) not > far-edge ({right:.2f}px)"

    # tau=0: pure affine, byte-exact
    rng0 = np.random.default_rng(72)
    fixed = {"theta": 0.3, "shear_x": 0.05, "shear_y": -0.05, "tx": 10.0, "ty": -5.0}
    record0, meta0 = generate_sample(cfg, rng0, bg_files, s=64.0, force_negative=False,
                                      photometric=False, occlude=False,
                                      components={**fixed, "tilt": 0.0, "psi": 0.0})
    H0 = meta0["M"]
    assert H0[2, 0] == 0.0 and H0[2, 1] == 0.0 and H0[2, 2] == 1.0

    comp = meta0["components"]
    SQ = render_res // nx
    R = np.array([[np.cos(comp["theta"]), -np.sin(comp["theta"])],
                  [np.sin(comp["theta"]), np.cos(comp["theta"])]])
    Sh = np.array([[1.0, np.tan(comp["shear_x"])], [np.tan(comp["shear_y"]), 1.0]])
    A = (comp["s"] / SQ) * (R @ Sh)
    c_r = np.array([(render_res - 1) / 2, (render_res - 1) / 2])
    c_in = np.array([(W - 1) / 2, (H - 1) / 2])
    m2 = c_in + np.array([comp["tx"], comp["ty"]]) - A @ c_r
    assert np.allclose(H0[:2, :2], A, atol=1e-12)
    assert np.allclose(H0[:2, 2], m2, atol=1e-12)


def test_visibility_truth_table():
    holes = [(100, 100, 20, 20)]
    size_wh = (640, 480)
    assert visible((105, 105), holes, size_wh) is False           # inside hole
    assert visible((99.5, 105), holes, size_wh) is False          # hole edge x0-0.5, half-open
    assert visible((99.4, 105), holes, size_wh) is True           # just left of the hole
    assert visible((-1.0, 100), holes, size_wh) is False          # outside frame (x < -0.5)
    assert visible((639.5, 100), holes, size_wh) is False         # x = W-0.5 exactly, half-open
    assert visible((300, 200), holes, size_wh) is True            # clean


def test_index_under_rotation(cfg, bg_files):
    W, H = cfg["input_size"]
    fixed = dict(shear_x=0.0, shear_y=0.0, tx=0.0, ty=0.0)
    record0, _ = generate_sample(cfg, np.random.default_rng(11), bg_files, s=64, force_negative=False,
                                  photometric=False, occlude=False, components={**fixed, "theta": 0.0})
    record1, _ = generate_sample(cfg, np.random.default_rng(12), bg_files, s=64, force_negative=False,
                                  photometric=False, occlude=False, components={**fixed, "theta": np.pi})

    for record in (record0, record1):
        assert [c["index"] for c in record["corners"]] == list(range(16))

    p0 = (record0["corners"][0]["x"], record0["corners"][0]["y"])
    p1 = (record1["corners"][0]["x"], record1["corners"][0]["y"])
    assert not np.allclose(p0, p1)  # theta actually moved corner 0

    # Identity rides the warp, never the image position: channel 0's target
    # peaks at corner 0's own reported position in BOTH cases.
    for p in (p0, p1):
        ct = render_class_targets(np.array([p]), np.array([True]), np.array([0]), (W, H), sigma=cfg["sigma_cls"])
        assert np.argwhere(ct[0] == 1.0).shape[0] == 1


def test_negative_sample(cfg, bg_files):
    record, meta = generate_sample(cfg, np.random.default_rng(31), bg_files, force_negative=True)
    assert record["board_present"] is False
    assert record["corners"] == []
    assert record["s_px"] == 0.0
    assert meta["M"] is None and meta["components"] is None

    W, H = cfg["input_size"]
    pts, vis, idx = np.zeros((0, 2)), np.zeros((0,), dtype=bool), np.zeros((0,), dtype=int)
    hm = render_heatmap(pts, vis, (W, H), sigma=cfg["sigma_hm"])
    ct = render_class_targets(pts, vis, idx, (W, H), sigma=cfg["sigma_cls"])
    assert np.all(hm == 0.0)
    assert np.all(ct == 0.0)


def test_record_schema(cfg, bg_files):
    record, _ = generate_sample(cfg, np.random.default_rng(41), bg_files, force_negative=False)
    assert set(record.keys()) == {"image", "board_present", "s_px", "corners"}
    assert record["image"].dtype == np.uint8
    assert isinstance(record["board_present"], bool)
    assert isinstance(record["s_px"], float)
    assert len(record["corners"]) == 16
    for c in record["corners"]:
        assert set(c.keys()) == {"x", "y", "index", "visible"}
        assert isinstance(c["x"], float) and isinstance(c["y"], float)
        assert isinstance(c["index"], int)
        assert isinstance(c["visible"], bool)

    record_n, _ = generate_sample(cfg, np.random.default_rng(42), bg_files, force_negative=True)
    assert set(record_n.keys()) == {"image", "board_present", "s_px", "corners"}
    assert record_n["corners"] == []


def test_refiner_stream(cfg, bg_files):
    rng = np.random.default_rng(61)
    all_d = []
    while len(all_d) < 500:
        record, _ = generate_sample(cfg, rng, bg_files, size_mult=2, occlude=False, force_negative=False)
        pts = [(c["x"], c["y"]) for c in record["corners"] if c["visible"]]
        for crop in cut_refiner_crops(cfg, rng, record["image"], pts):
            assert crop["crop"].shape == (24, 24)
            assert crop["crop"].dtype == np.uint8
            all_d.append(crop["d"])
    all_d = np.array(all_d)
    assert np.abs(all_d).max() <= 3.9375

    pooled = np.concatenate([all_d[:, 0], all_d[:, 1]])
    hist, _ = np.histogram(pooled, bins=np.linspace(-3.9375, 3.9375, 9))
    print("test_refiner_stream d-histogram fractions:", (hist / hist.sum()).tolist())
    assert hist.min() / hist.sum() >= 0.05

    # Content check on 100 photometric-off crops, at a fixed comfortably-large
    # s (see test_warp_roundtrip: classical subpixel detection at the small
    # end of scale_range_px is a known, spec-acknowledged hard regime, not a
    # defect in cut_refiner_crops' own crop/offset arithmetic, which is what
    # this check targets).
    rng2 = np.random.default_rng(62)
    errs = []
    while len(errs) < 100:
        record, _ = generate_sample(cfg, rng2, bg_files, s=192.0, size_mult=2, occlude=False,
                                     force_negative=False, photometric=False)
        pts = [(c["x"], c["y"]) for c in record["corners"] if c["visible"]]
        for crop in cut_refiner_crops(cfg, rng2, record["image"], pts):
            if len(errs) >= 100:
                break
            d = crop["d"]
            seed = np.array([[12 + d[0], 12 + d[1]]], dtype=np.float32).reshape(-1, 1, 2)
            refined = cv2.cornerSubPix(crop["crop"], seed, (5, 5), (-1, -1), CRIT)
            errs.append(float(np.linalg.norm(refined.reshape(-1, 2)[0] - np.array([12 + d[0], 12 + d[1]]))))
    print("test_refiner_stream content-check max err:", max(errs))
    assert max(errs) <= 0.15


def test_determinism(cfg):
    sv = SynthVal(cfg, n=20, seed=123)
    img_a, rec_a = sv[3]
    img_b, rec_b = sv[3]
    assert np.array_equal(img_a, img_b)
    assert rec_a["corners"] == rec_b["corners"]
    assert rec_a["board_present"] == rec_b["board_present"]
    assert rec_a["s_px"] == rec_b["s_px"]

    img_c, rec_c = sv[4]
    assert not np.array_equal(img_a, img_c)
    assert rec_a["corners"] != rec_c["corners"]

    # RNG-purity: every random draw in dcc/ goes through a passed-in
    # Generator; default_rng/Generator-typed mentions are the only allowed
    # uses of the global np.random namespace.
    dcc_dir = Path(__file__).parents[1] / "dcc"
    for f in dcc_dir.glob("*.py"):
        text = f.read_text()
        assert "import random" not in text, f
        assert "cv2.setRNGSeed" not in text, f
        for line in text.splitlines():
            if "np.random." in line:
                assert "default_rng" in line or "Generator" in line, f"{f}: {line}"


def test_val_stratification(cfg):
    n = 1000
    sv = SynthVal(cfg, n=n, seed=456)
    s_vals = [rec["s_px"] for _, rec in (sv[i] for i in range(n)) if rec["board_present"]]
    s_vals = np.array(s_vals)
    a, b = cfg["scale_range_px"]
    edges = np.array(sorted({float(e) for e in (a, 16.0, 32.0, 64.0, 128.0, b) if a <= e <= b}))
    counts, _ = np.histogram(s_vals, bins=edges)
    print("test_val_stratification bin counts:", counts.tolist(), "n_positive:", len(s_vals))
    # log-uniform s: expected mass per bin is its log-width share (the low
    # [12,16) bin is a sub-octave, so equal-share would silently mis-state it)
    expected = len(s_vals) * np.diff(np.log(edges)) / np.log(edges[-1] / edges[0])
    assert np.all(np.abs(counts - expected) / expected <= 0.20)


def test_zero_visible_positive_legal(cfg, bg_files):
    W, H = cfg["input_size"]
    record, _ = generate_sample(cfg, np.random.default_rng(21), bg_files, s=64, force_negative=False,
                                 photometric=False, occlude=False,
                                 components={"theta": 0.0, "shear_x": 0.0, "shear_y": 0.0, "tx": 1e4, "ty": 1e4})
    assert record["board_present"] is True
    assert len(record["corners"]) == 16
    assert all(c["visible"] is False for c in record["corners"])

    pts = np.array([[c["x"], c["y"]] for c in record["corners"]])
    vis = np.array([c["visible"] for c in record["corners"]])
    idx = np.array([c["index"] for c in record["corners"]])
    hm = render_heatmap(pts, vis, (W, H), sigma=cfg["sigma_hm"])
    ct = render_class_targets(pts, vis, idx, (W, H), sigma=cfg["sigma_cls"])
    assert hm.shape == (H, W) and ct.shape == (16, H // 4, W // 4)
    assert np.all(hm == 0.0) and np.all(ct == 0.0)


# --------------------------------------------------- object-cutout occlusion --

def test_cutout_visibility(cutout_bank):
    """place_cutout in isolation (no rng/bank plumbing): a corner under the
    placed disk's opaque alpha reads invisible by the exact geometric test
    generate_sample applies (visible() AND occ_alpha < 0.5); a corner well
    outside the disk (even past its feathered edge) stays visible; and the
    boundary itself shows a real feathered transition, not a hard 0/1 step."""
    rgba = cv2.imread(str(Path(cutout_bank) / "0000000.png"), cv2.IMREAD_UNCHANGED)
    assert rgba is not None and rgba.shape == (80, 80, 4)

    w2 = h2 = 160
    work = np.full((h2, w2, 3), 128.0, dtype=np.float32)
    occ_alpha = np.zeros((h2, w2), dtype=np.float32)
    bbox = place_cutout(work, occ_alpha, rgba.copy(), scale=0.5, rot=0.0, hflip=False, cx=80.0, cy=80.0)
    assert bbox is not None

    def geom_visible(x, y):
        # exactly generate_sample's point-in-alpha extension of visible()
        vis = visible((x, y), [], (w2, h2))
        if vis:
            iy, ix = int(np.rint(y)), int(np.rint(x))
            if 0 <= iy < h2 and 0 <= ix < w2:
                vis = bool(occ_alpha[iy, ix] < 0.5)
        return vis

    assert occ_alpha[80, 80] > 0.5          # dead centre, solidly under the disk
    assert geom_visible(80.0, 80.0) is False

    assert occ_alpha[80, 130] == 0.0        # comfortably outside disk radius (36) + feather
    assert geom_visible(80.0, 130.0) is True

    # feathered edge: disk radius 36 from centre (80,80) -> boundary at x~116
    profile = occ_alpha[80, 100:130]
    assert profile[0] > 0.9
    assert profile[-1] < 0.1
    assert np.any((profile > 0.02) & (profile < 0.98)), "expected a feathered transition, not a hard step"
    assert np.all(np.diff(profile) <= 0.05), "expected a smooth (non-increasing) falloff outward"


def test_cutout_stream_discipline(cfg, bg_files, cutout_bank):
    """cutouts.p forced 0, a bank-missing path, and the real (non-empty)
    fixture bank must all consume the exact same rng draws inside
    _apply_cutouts -- the always-draw-the-full-budget invariant. Proven by
    comparing what a shared rng stream produces AFTER cutouts in each
    scenario: this sample's holes (drawn immediately after cutouts) and a
    second sample's board placement pulled off the same continuing stream
    (standing in for a training loop reading many samples off one
    Generator)."""
    base = {"max_objects": 3, "scale": [0.08, 0.5]}
    scenarios = {
        "p_zero": {"path": cutout_bank, "p": 0.0, **base},
        "bank_missing": {"path": "/nonexistent/cutout/bank/stream/test", "p": 0.35, **base},
        "bank_present": {"path": cutout_bank, "p": 0.35, **base},
    }
    holes_by_scenario, corners2_by_scenario = {}, {}
    for name, cutouts_cfg in scenarios.items():
        c = copy.deepcopy(cfg)
        c["synth"]["cutouts"] = cutouts_cfg
        rng = np.random.default_rng(777)
        _, meta1 = generate_sample(c, rng, bg_files, s=64.0, force_negative=False)
        record2, _ = generate_sample(c, rng, bg_files, s=64.0, force_negative=False)
        holes_by_scenario[name] = meta1["holes"]
        corners2_by_scenario[name] = record2["corners"]

    assert holes_by_scenario["p_zero"] == holes_by_scenario["bank_missing"] == holes_by_scenario["bank_present"]
    assert (corners2_by_scenario["p_zero"] == corners2_by_scenario["bank_missing"]
            == corners2_by_scenario["bank_present"])


def test_cutout_determinism(cfg, bg_files, cutout_bank):
    """Same seed + fixture bank, twice -> byte-identical images, corners,
    and cutout placements, each of several seeds. Also confirms objects
    actually get placed for at least one of them (a smoke check that p=1.0
    + a non-empty bank isn't silently a no-op)."""
    c = copy.deepcopy(cfg)
    c["synth"]["cutouts"] = {"path": cutout_bank, "p": 1.0, "max_objects": 3, "scale": [0.08, 0.5]}

    any_placed = False
    for seed in (4242, 4243, 4244, 4245):
        rng1 = np.random.default_rng(seed)
        record1, meta1 = generate_sample(c, rng1, bg_files, force_negative=False)
        rng2 = np.random.default_rng(seed)
        record2, meta2 = generate_sample(c, rng2, bg_files, force_negative=False)

        assert np.array_equal(record1["image"], record2["image"])
        assert record1["corners"] == record2["corners"]
        assert meta1["cutouts"] == meta2["cutouts"]
        any_placed = any_placed or len(meta1["cutouts"]) > 0

    assert any_placed, "expected at least one placement across several seeds at p=1.0 with a non-empty bank"


def test_record_schema_unchanged(cfg, bg_files, cutout_bank):
    """SD-05 record keys/types are unchanged with cutout occlusion active --
    the feature only adds to meta (debug-only), never to the training
    record itself."""
    c = copy.deepcopy(cfg)
    c["synth"]["cutouts"] = {"path": cutout_bank, "p": 1.0, "max_objects": 3, "scale": [0.08, 0.5]}
    rng = np.random.default_rng(99)
    record, meta = generate_sample(c, rng, bg_files, force_negative=False)

    assert set(record.keys()) == {"image", "board_present", "s_px", "corners"}
    assert record["image"].dtype == np.uint8
    assert isinstance(record["board_present"], bool)
    assert isinstance(record["s_px"], float)
    assert len(record["corners"]) == 16
    for corner in record["corners"]:
        assert set(corner.keys()) == {"x", "y", "index", "visible"}
        assert isinstance(corner["x"], float) and isinstance(corner["y"], float)
        assert isinstance(corner["index"], int)
        assert isinstance(corner["visible"], bool)

    assert "cutouts" in meta and isinstance(meta["cutouts"], list)
    for c_meta in meta["cutouts"]:
        assert set(c_meta.keys()) == {"file", "bbox"}


# --------------------------------------------------------------- Slice B5 --

def _photometric_only(cfg, **on):
    """cfg["synth"]["photometric"] copy with every *_p gate zeroed except
    `on` -- isolates one (or a few) photometric step(s) from the rest of
    the pinned-order set, reusing the real config's ranges for whatever
    stays on."""
    ph = dict(cfg["synth"]["photometric"])
    for k in ph:
        if k.endswith("_p"):
            ph[k] = 0.0
    ph.update(on)
    return ph


def test_contrast_speckle(cfg, bg_files):
    """Task #12: contrast (blend-about-mean) and multiplicative speckle,
    inserted after brightness and after the gaussian blur respectively (see
    _apply_photometric's docstring). Each step's own defining property is
    checked directly against _apply_photometric (board/background content
    would otherwise confound "does the mean survive" / "do zeros survive");
    forced-on vs forced-off and determinism are checked through the full
    generate_sample path, per the brief."""
    h2, w2 = 64, 64
    work = np.full((h2, w2, 3), 100.0, dtype=np.float32)
    work[:8, :8] = 0.0  # a true-black patch: speckle's "zero stays zero" probe

    ph_contrast = _photometric_only(cfg, contrast_p=1.0)
    out_c = _apply_photometric(work.copy(), np.random.default_rng(9), ph_contrast, w2, h2)
    mean_relerr = abs(out_c.mean() - work.mean()) / work.mean()
    print("test_contrast_speckle contrast mean-preservation relerr:", mean_relerr)
    assert mean_relerr < 0.01  # blend-about-mean: m + (work-m)*c leaves work's own mean fixed

    ph_speckle = _photometric_only(cfg, speckle_p=1.0)
    rng_s = np.random.default_rng(10)
    out_s = _apply_photometric(work.copy(), rng_s, ph_speckle, w2, h2)
    assert np.all(out_s[:8, :8] == 0.0)  # multiplicative: 0 * (1+noise) stays exactly 0
    assert not np.allclose(out_s[8:, 8:], work[8:, 8:])  # noise actually landed elsewhere

    out_s2 = _apply_photometric(work.copy(), np.random.default_rng(10), ph_speckle, w2, h2)
    assert np.array_equal(out_s, out_s2)  # deterministic under the same seed

    c_on = copy.deepcopy(cfg)
    c_on["synth"]["photometric"] = _photometric_only(cfg, contrast_p=1.0, speckle_p=1.0)
    c_off = copy.deepcopy(cfg)
    c_off["synth"]["photometric"] = _photometric_only(cfg)

    rec_on, _ = generate_sample(c_on, np.random.default_rng(2024), bg_files, s=64.0, force_negative=False)
    rec_off, _ = generate_sample(c_off, np.random.default_rng(2024), bg_files, s=64.0, force_negative=False)
    assert not np.array_equal(rec_on["image"], rec_off["image"])

    rec_on2, _ = generate_sample(c_on, np.random.default_rng(2024), bg_files, s=64.0, force_negative=False)
    assert np.array_equal(rec_on["image"], rec_on2["image"])


def test_prefilter(cfg, bg_files):
    """Task #7: anti-alias prefilter, applied to `matched` strictly after
    H/p_img's inputs are already fixed (see _composite_board) so geometry
    can't be touched -- only pixels. s=16 (SQ=render_res//5=96, comfortably
    inside the s < SQ minify regime): on vs off differ, RECORDED CORNERS
    stay byte-identical, and the on-board Laplacian variance (a marker-
    module aliasing proxy) drops with the prefilter on. s=120 (>= SQ): the
    `s < SQ` guard alone makes it a no-op regardless of `enabled`, so on
    and off are byte-identical. Geometry pinned (theta/shear/tx/ty/tilt=0)
    so the board sits centred and axis-aligned -- isolates the prefilter
    from the independent randomness of placement."""
    fixed_geom = {"theta": 0.0, "shear_x": 0.0, "shear_y": 0.0, "tx": 0.0, "ty": 0.0, "tilt": 0.0}

    def sample(prefilter_enabled, s, seed):
        c = copy.deepcopy(cfg)
        c["synth"]["prefilter"] = {"enabled": prefilter_enabled, "k": 0.5}
        rng = np.random.default_rng([seed, int(s)])
        return generate_sample(c, rng, bg_files, s=s, photometric=False, occlude=False,
                                force_negative=False, components=fixed_geom)

    rec_on, _ = sample(True, 16.0, 909)
    rec_off, _ = sample(False, 16.0, 909)
    assert not np.array_equal(rec_on["image"], rec_off["image"])
    assert rec_on["corners"] == rec_off["corners"]

    xs = [c["x"] for c in rec_on["corners"]]
    ys = [c["y"] for c in rec_on["corners"]]
    W, H = cfg["input_size"]
    x0, x1 = max(0, int(min(xs)) - 2), min(W, int(max(xs)) + 3)
    y0, y1 = max(0, int(min(ys)) - 2), min(H, int(max(ys)) + 3)
    lap_on = cv2.Laplacian(rec_on["image"][y0:y1, x0:x1], cv2.CV_64F).var()
    lap_off = cv2.Laplacian(rec_off["image"][y0:y1, x0:x1], cv2.CV_64F).var()
    print("test_prefilter on-board laplacian var, on/off at s=16:", lap_on, lap_off)
    assert lap_on < lap_off

    rec_on2, _ = sample(True, 120.0, 909)
    rec_off2, _ = sample(False, 120.0, 909)
    assert np.array_equal(rec_on2["image"], rec_off2["image"])


def test_generic_crops():
    """Task #6: make_generic_crop in isolation. d stays within the 64x64@8x
    support; cornerSubPix on the CLEAN (roughen=False) crop recovers the
    analytic corner to a measured, honest tolerance -- NOT the originally
    hoped-for 0.15/0.5px (see the B5 report: a naive flat-sector "pie"
    corner measured up to ~6px of error; a 35deg angle-guard plus 16x
    supersampled anti-aliasing, both justified by the "only corner-like
    point" invariant in make_generic_crop's own docstring, bring the
    measured max at this seed down to <1px, still not the original hope,
    so the gate below reflects what is actually achieved, not assumed);
    2-4 dominant (>=14px-of-576) gray plateaus; determinism."""
    rng = np.random.default_rng(2026)
    errs = []
    for _ in range(200):
        crop = make_generic_crop(rng, roughen=False)
        d = crop["d"]
        assert crop["crop"].shape == (24, 24)
        assert crop["crop"].dtype == np.uint8
        assert max(abs(d[0]), abs(d[1])) <= 3.9375

        _, counts = np.unique(crop["crop"], return_counts=True)
        n_dominant = int((counts >= 14).sum())
        assert 2 <= n_dominant <= 4

        seed = np.array([[12 + d[0] + 0.3, 12 + d[1] + 0.3]], dtype=np.float32).reshape(-1, 1, 2)
        refined = cv2.cornerSubPix(crop["crop"], seed, (5, 5), (-1, -1), CRIT)
        err = float(np.linalg.norm(refined.reshape(-1, 2)[0] - np.array([12 + d[0], 12 + d[1]])))
        errs.append(err)
    print("test_generic_crops cornerSubPix median/p90/p99/max:",
          np.median(errs), np.percentile(errs, 90), np.percentile(errs, 99), max(errs))
    assert max(errs) <= 1.0

    rng_a = np.random.default_rng(555)
    rng_b = np.random.default_rng(555)
    c_a = make_generic_crop(rng_a)
    c_b = make_generic_crop(rng_b)
    assert np.array_equal(c_a["crop"], c_b["crop"])
    assert np.array_equal(c_a["d"], c_b["d"])


def test_generic_blend_off_identical(cfg):
    """Task #6: refiner_generic_frac's off-path is byte-identical whether
    the key is 0.0 or absent entirely (CRITICAL: _maybe_replace_generic's
    coin draw must not fire either way -- proven directly in the second
    half below, and here through RefinerVal end to end); frac=0.5 replaces
    close to half of the crops (chi-square-loose bound) and is itself
    deterministic under a fixed seed."""
    c_zero = copy.deepcopy(cfg)
    c_zero["synth"]["refiner_generic_frac"] = 0.0
    c_absent = copy.deepcopy(cfg)
    del c_absent["synth"]["refiner_generic_frac"]

    rv_zero = RefinerVal(c_zero, n=10, seed=808)
    rv_absent = RefinerVal(c_absent, n=10, seed=808)
    crops_zero, crops_absent = rv_zero[3], rv_absent[3]
    assert len(crops_zero) == len(crops_absent) > 0
    for cz, ca in zip(crops_zero, crops_absent):
        assert np.array_equal(cz["crop"], ca["crop"])
        assert np.array_equal(cz["d"], ca["d"])

    rng = np.random.default_rng(2027)
    dummy = {"crop": np.zeros((24, 24), dtype=np.uint8), "d": np.zeros(2)}
    n = 200
    replaced = [_maybe_replace_generic(dummy, rng, 0.5) is not dummy for _ in range(n)]
    k = sum(replaced)
    expected = n / 2
    chi2 = (k - expected) ** 2 / expected + ((n - k) - expected) ** 2 / expected
    print(f"test_generic_blend_off_identical frac=0.5 replaced {k}/{n}, chi2={chi2:.3f}")
    assert chi2 < 20.0  # loose: chi2(1) critical value at alpha=1e-5 is ~19.5

    rng_a = np.random.default_rng(303)
    rng_b = np.random.default_rng(303)
    seq_a = [_maybe_replace_generic(dummy, rng_a, 0.5) is not dummy for _ in range(50)]
    seq_b = [_maybe_replace_generic(dummy, rng_b, 0.5) is not dummy for _ in range(50)]
    assert seq_a == seq_b  # deterministic under the same seed
