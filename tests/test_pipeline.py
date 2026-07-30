import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from dcc.pipeline import (peaks, merge_close, cut_crops, soft_argmax, read_ids,
                           undistort, lattice_gate, recover, pnp, detect)
from dcc.targets import render_class_targets, render_refiner_target

CANON = np.array([[(i % 4) + 1.0, (i // 4) + 1.0] for i in range(16)], dtype=np.float64)


def _build_pose(rng, tilt_deg, s=60.0, W=1600, H=1200, phi=None, psi=None):
    """gen_eval_pose-style K@[r1|r2|t] synthetic pose (independent of
    tools/gen_eval_pose.py, not imported): in-plane angle phi and tilt axis
    psi (random unless pinned by the caller), tilt about that axis, apparent
    scale s at the board centre. Returns (K, R, t, img_pts) for the 16
    canonical corners."""
    cx, cy = (W - 1) / 2, (H - 1) / 2
    f = 1.0 * W
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    phi = rng.uniform(-np.pi, np.pi) if phi is None else phi
    psi = rng.uniform(0, 2 * np.pi) if psi is None else psi
    Raxis, _ = cv2.Rodrigues(np.radians(tilt_deg) * np.array([np.cos(psi), np.sin(psi), 0.0]))
    cp, sp = np.cos(phi), np.sin(phi)
    R = np.array([[cp, -sp, 0], [sp, cp, 0], [0, 0, 1.0]]) @ Raxis
    z = f / s
    t = z * (np.linalg.inv(K) @ np.array([cx, cy, 1.0])) - R @ np.array([2.5, 2.5, 0.0])
    rvec, _ = cv2.Rodrigues(R)
    lattice3 = np.hstack([CANON, np.zeros((16, 1))])
    img_pts = cv2.projectPoints(lattice3, rvec, t, K, None)[0].reshape(-1, 2)
    return K, R, t, img_pts


# ---------------------------------------------------------------- readout --

def test_readout_convention():
    """The decisive test: read_ids at the exact rendered corner xy must
    recover the rendered index always, with confidence >= the analytic
    worst case (bilinear split 4 ways at a shared cell corner, sigma=1
    cell -- measures ~0.83, well above the 0.24 floor asserted here) and
    ~1.0 at cell-centre phase. align_corners=True is independently checked
    wrong on a crafted case to lock the convention."""
    W, H = 256, 256  # 64x64 cells at H/4
    rng = np.random.default_rng(0)
    worst_conf = 1.0
    for _ in range(200):
        k = int(rng.integers(0, 16))
        x, y = rng.uniform(20, W - 20), rng.uniform(20, H - 20)
        ct = render_class_targets(np.array([[x, y]]), np.array([True]), np.array([k]), (W, H), sigma=1.0)
        idx_out, conf_out = read_ids(ct, np.array([[x, y]]))
        assert idx_out[0] == k, f"wrong index at phase ({x % 4}, {y % 4})"
        worst_conf = min(worst_conf, float(conf_out[0]))
    assert worst_conf >= 0.24, f"worst confidence {worst_conf} below analytic floor"
    print("test_readout_convention random-phase worst conf:", worst_conf)

    # deterministic worst-case phase: corner at a shared 4-cell boundary
    j = 20
    xw = yw = 4 * j - 0.5
    ctw = render_class_targets(np.array([[xw, yw]]), np.array([True]), np.array([9]), (W, H), sigma=1.0)
    idx_w, conf_w = read_ids(ctw, np.array([[xw, yw]]))
    assert idx_w[0] == 9 and conf_w[0] >= 0.24
    print("test_readout_convention analytic worst-phase conf:", float(conf_w[0]))

    # cell-centre phase: confidence must read back essentially exactly 1.0
    xc = yc = 4 * j + 1.5
    ctc = render_class_targets(np.array([[xc, yc]]), np.array([True]), np.array([4]), (W, H), sigma=1.0)
    idx_c, conf_c = read_ids(ctc, np.array([[xc, yc]]))
    assert idx_c[0] == 4
    assert conf_c[0] >= 0.95
    assert abs(conf_c[0] - 1.0) < 1e-5

    # align_corners=True locks WRONG: two corners (channels 3, 7) near the
    # map edge where the align_corners conventions diverge most; queried at
    # a point strictly closer to corner 3, the correct (align_corners=False)
    # convention picks channel 3, the wrong one picks channel 7.
    pts = np.array([[1.5, 1.5], [5.5, 1.5]])
    ct2 = render_class_targets(pts, np.array([True, True]), np.array([3, 7]), (64, 64), sigma=1.0)
    query = np.array([[2.25, 1.5]])
    idx_correct, _ = read_ids(ct2, query)
    assert idx_correct[0] == 3

    cls_t = torch.as_tensor(ct2, dtype=torch.float32)
    _, H4, W4 = cls_t.shape
    xy_t = torch.as_tensor(query, dtype=torch.float32)
    grid = torch.stack([2 * xy_t[:, 0] / (4 * W4 - 1) - 1, 2 * xy_t[:, 1] / (4 * H4 - 1) - 1],
                        dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(cls_t[None], grid, mode="bilinear", padding_mode="border",
                             align_corners=True)[0, :, :, 0]
    idx_wrong = int(sampled.argmax(dim=0)[0])
    assert idx_wrong != 3, "align_corners=True should diverge from the pinned convention here"


# ------------------------------------------------------------ peaks/merge --

def test_peaks_and_merge():
    """Two isolated Gaussians (well-separated) + one exact 2-px plateau tie.
    peaks() must find all 4 raw local maxima (both plateau pixels pass the
    3x3 equality test); merge_close collapses the tie to one; top_k caps in
    descending-score order."""
    hm = np.zeros((40, 40), dtype=np.float32)
    hm[10, 10] = 0.9   # bump1, (x,y)=(10,10)
    hm[30, 30] = 0.6   # bump2, (x,y)=(30,30)
    hm[15, 20] = 0.8   # plateau tie, (x,y)=(20,15)
    hm[15, 21] = 0.8   # plateau tie, (x,y)=(21,15)

    xy, sc = peaks(hm, tau_hm=0.3, top_k=64)
    assert len(xy) == 4
    found = {tuple(p): s for p, s in zip(xy.tolist(), sc.tolist())}
    assert found[(10, 10)] == pytest.approx(0.9)
    assert found[(30, 30)] == pytest.approx(0.6)
    assert found[(20, 15)] == pytest.approx(0.8)
    assert found[(21, 15)] == pytest.approx(0.8)
    assert list(sc) == sorted(sc, reverse=True)

    xy_m, sc_m = merge_close(xy, sc, radius=2.0)
    assert len(xy_m) == 3, "the 1-px-apart plateau tie must collapse to one"
    kept = {tuple(p) for p in xy_m.tolist()}
    assert (10, 10) in kept and (30, 30) in kept
    assert ((20, 15) in kept) ^ ((21, 15) in kept), "exactly one plateau survivor"
    assert (20, 15) in kept, "deterministic tiebreak keeps the lexicographically-first tie"

    xy_k, sc_k = peaks(hm, tau_hm=0.3, top_k=3)
    assert len(xy_k) == 3
    assert (30, 30) not in {tuple(p) for p in xy_k.tolist()}, "top_k must drop the lowest score"


# ------------------------------------------------------------- cut_crops --

def test_border_bypass():
    """A peak whose 24x24 sensor crop would cross the frame border is
    excluded (never reflect-padded); one that exactly touches (12 px
    margin) is kept. Checked at r=1 and r=2.5 against an independently
    recomputed half-pixel map, and crop content verified byte-for-byte."""
    def input_for_centre(c, r):
        return (c + 0.5) * r - 0.5

    rng = np.random.default_rng(3)
    frame = rng.integers(0, 256, size=(200, 200), dtype=np.uint8)
    for r in (1.0, 2.5):
        x5, x12 = input_for_centre(5, r), input_for_centre(12, r)
        mid = input_for_centre(100, r)  # input coord that maps to sensor centre 100, any r
        peaks_input = np.array([[x5, mid], [x12, mid], [mid, x5], [mid, x12]])
        crops, centres, kept_mask, extents = cut_crops(frame, peaks_input, r)
        assert list(extents) == [24, 24], "default extent must stay the fixed 24 px crop"
        assert list(kept_mask) == [False, True, False, True], f"r={r}"
        assert centres.tolist() == [[12, 100], [100, 12]], f"r={r} centre mismatch"
        assert crops.shape == (2, 1, 24, 24)
        for (cx, cy), crop in zip(centres, crops):
            expected = frame[cy - 12:cy + 12, cx - 12:cx + 12].astype(np.float32) / 255.0
            assert np.array_equal(crop[0], expected)
            assert cx - 12 >= 0 and cx + 12 <= 200 and cy - 12 >= 0 and cy + 12 <= 200


# ------------------------------------------------------------ soft_argmax --

def test_soft_argmax():
    """render_refiner_target(d) (real sigma=1.5, forced-peak pixel) recovers
    u* within 0.35; a manually-built, narrower Gaussian (sigma=0.5, so the
    5x5 readout window captures effectively all its mass) recovers u* within
    0.05 and locks the u=x/col, v=y/row orientation. d is sampled from
    [-3.5, 3.5], staying inside render_refiner_target's +/-3.9375 support --
    the outer ~0.4 px sliver is excluded because there the analytic Gaussian
    itself is truncated by the 64x64 canvas edge, a distinct and
    already-understood degeneracy from the window-vs-sigma truncation this
    test targets."""
    rng = np.random.default_rng(1)
    worst_forced = 0.0
    for _ in range(100):
        dx, dy = rng.uniform(-3.5, 3.5), rng.uniform(-3.5, 3.5)
        rt = render_refiner_target(np.array([dx, dy]), sigma=1.5)
        u = soft_argmax(rt.reshape(1, 1, 64, 64))[0]
        worst_forced = max(worst_forced, float(np.abs(u - [31.5 + 8 * dx, 31.5 + 8 * dy]).max()))
    assert worst_forced <= 0.35
    print("test_soft_argmax forced-peak worst err:", worst_forced)

    ys, xs = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    worst_manual = 0.0
    for _ in range(100):
        dx, dy = rng.uniform(-3.5, 3.5), rng.uniform(-3.5, 3.5)
        centre = (100.0, 200.0)  # arbitrary integer sensor centre
        u_star, v_star = 31.5 + 8 * dx, 31.5 + 8 * dy
        g = np.exp(-((xs - u_star) ** 2 + (ys - v_star) ** 2) / (2 * 0.5 ** 2)).astype(np.float32)
        u = soft_argmax(g.reshape(1, 1, 64, 64))[0]
        worst_manual = max(worst_manual, float(np.abs(u - [u_star, v_star]).max()))
        xy_refined = np.array(centre) + (u - 31.5) / 8.0
        assert np.abs(xy_refined - (np.array(centre) + [dx, dy])).max() < 0.05
    assert worst_manual <= 0.05
    print("test_soft_argmax clean-gaussian worst err:", worst_manual)


# -------------------------------------------------------------- undistort --

def test_undistort_identity():
    xy = np.array([[100.0, 150.0], [320.0, 240.0], [500.0, 10.0]])
    K = np.array([[550.0, 0, 320.0], [0, 550.0, 240.0], [0, 0, 1.0]])
    out = undistort(xy, K, None)
    assert np.abs(out - xy).max() < 1e-6
    assert undistort(np.zeros((0, 2)), K, None).shape == (0, 2)


# ---------------------------------------------------------- lattice_gate --

def test_lattice_gate():
    rng = np.random.default_rng(7)
    _, _, _, img_pts = _build_pose(rng, tilt_deg=30.0)

    # 12 ID'd, 2 deliberately wrong -> demotes exactly those 2
    idx = np.arange(12)
    idx_wrong = idx.copy()
    idx_wrong[3], idx_wrong[7] = 14, 15
    H, inlier_mask, demoted_mask, degenerate = lattice_gate(img_pts[:12], idx_wrong, np.ones(12), tol=3.0)
    assert degenerate is None and H is not None
    assert np.nonzero(demoted_mask)[0].tolist() == [3, 7]
    assert inlier_mask.sum() == 10

    # exactly 4, non-collinear -> vacuous, H still a valid exact fit
    idx4 = np.array([0, 1, 4, 5])  # canonical (1,1)(2,1)(1,2)(2,2): a 2x2 block
    H4, inlier4, demoted4, deg4 = lattice_gate(img_pts[idx4], idx4, np.ones(4), tol=3.0)
    assert deg4 == "vacuous" and H4 is not None and inlier4.all() and not demoted4.any()
    reproj = np.hstack([CANON[idx4], np.ones((4, 1))]) @ H4.T
    reproj = reproj[:, :2] / reproj[:, 2:3]
    assert np.abs(reproj - img_pts[idx4]).max() < 1e-3

    # 4 collinear (one lattice row: indices 0..3 all have canonical row=1)
    idx_row = np.array([0, 1, 2, 3])
    Hc, _, _, degc = lattice_gate(img_pts[idx_row], idx_row, np.ones(4), tol=3.0)
    assert degc == "collinear" and Hc is None

    # fewer than 4 ID'd
    Ht, _, _, degt = lattice_gate(img_pts[:3], np.array([0, 1, 2]), np.ones(3), tol=3.0)
    assert degt == "too_few" and Ht is None

    # recovery: all 16 correct, drop 5 IDs, recover() reassigns all 5
    idx16 = np.arange(16)
    drop = [2, 5, 9, 11, 14]
    idx_dropped = idx16.copy()
    idx_dropped[drop] = -1
    Hg, _, demg, degg = lattice_gate(img_pts, idx_dropped, np.ones(16), tol=3.0)
    assert degg is None and not demg.any()
    idx_rec, recovered_mask, corroborated = recover(Hg, img_pts, idx_dropped, np.ones(16), tol=3.0)
    assert np.nonzero(recovered_mask)[0].tolist() == drop
    assert idx_rec[drop].tolist() == drop
    assert corroborated is False  # 11 ID'd feeding H, not the vacuous 4-point case

    # corroboration: a vacuous (4-ID) fit that DOES recover extra points is
    # corroborated; one with no other detections to recover is not.
    idx_full_from_4 = np.full(16, -1)
    idx_full_from_4[idx4] = idx4
    _, _, corrob_yes = recover(H4, img_pts, idx_full_from_4, np.ones(16), tol=3.0)
    assert corrob_yes is True
    _, _, corrob_no = recover(H4, img_pts[idx4], idx4, np.ones(4), tol=3.0)
    assert corrob_no is False

    # regression: two different ID-less detections both within tol of the
    # SAME projected canonical corner must not both claim it
    Hs = np.diag([100.0, 100.0, 1.0])
    c5 = np.array([2.0, 2.0]) * 100  # canonical index 5's projection under Hs
    xy_dup = np.array([c5 + [1, 1], c5 + [-1, -1]])
    idx_dup, recovered_dup, _ = recover(Hs, xy_dup, np.array([-1, -1]), np.ones(2), tol=3.0)
    assert not (idx_dup[0] == 5 and idx_dup[1] == 5), "duplicate recovered index"
    assert recovered_dup.sum() == 1


# ------------------------------------------------------------------- pnp --

def test_pnp_ippe():
    rng = np.random.default_rng(11)
    K, R, t, img_pts = _build_pose(rng, tilt_deg=30.0, s=60.0)
    idx = np.arange(16)
    rvec, tvec, rms, ambiguous, n_used, reason, _cov = pnp(img_pts, idx, K, square_length_m=1.0)
    assert reason is None and n_used == 16 and not ambiguous
    R_est, _ = cv2.Rodrigues(rvec)
    ang = np.degrees(np.arccos(np.clip((np.trace(R_est.T @ R) - 1) / 2, -1, 1)))
    assert ang < 0.5, f"rotation error {ang} deg"
    assert np.linalg.norm(tvec.ravel() - t) < 0.01 * np.linalg.norm(t)

    # Near-fronto-parallel -> genuinely ambiguous. At exact noiseless
    # fronto-parallel (tilt=0) IPPE's two solutions coincide to numerical
    # precision (verified: rvecs agree to ~4e-8) -- the textbook two-fold
    # degeneracy; realistic detector noise (0.15 px, comparable to the
    # refiner's own sub-pixel error) is what makes both reprojection errors
    # nonzero and comparable, so ambiguous depends on the noise draw, not on
    # phi/psi.
    # tilt=2 deg does NOT reproduce this robustly (checked empirically: the
    # deterministic ~0.18 px separation from tilt alone competes with the
    # noise, ambiguous only ~50% of random draws) -- tilt=0 with a fixed
    # noise seed is the correct, reproducible way to hit this branch.
    K2, R2, t2, img_pts2 = _build_pose(rng, tilt_deg=0.0, phi=0.3, psi=0.7, s=60.0)
    noisy = img_pts2 + np.random.default_rng(2).normal(0, 0.15, img_pts2.shape)
    _, _, _, ambiguous2, _, reason2, _ = pnp(noisy, idx, K2, square_length_m=1.0)
    assert reason2 is None and ambiguous2 is True

    # too few correspondences
    rvec3, tvec3, rms3, amb3, n3, reason3, _ = pnp(img_pts[:3], np.array([0, 1, 2]), K, 1.0)
    assert rvec3 is None and tvec3 is None and reason3 is not None and n3 == 3


def test_pnp_empty_solver_result(monkeypatch):
    """OpenCV's solvepnp.cpp wraps IPPE in a bare catch(...): internal solver
    failures are silently swallowed and ZERO solutions returned. That
    reachable outcome must be a defined no-pose, never an IndexError."""
    rng = np.random.default_rng(11)
    K, R, t, img_pts = _build_pose(rng, tilt_deg=30.0, s=60.0)
    monkeypatch.setattr(cv2, "solvePnPGeneric", lambda *a, **k: (0, [], [], np.zeros((0, 1))))
    rvec, tvec, rms, ambiguous, n_used, reason, _ = pnp(img_pts, np.arange(16), K, 1.0)
    assert rvec is None and tvec is None and rms is None
    assert ambiguous is False and n_used == 16 and reason == "pnp_solver_failed"


# --------------------------------------------------------------- detect() --

def test_detect_contract_untrained():
    pytest.importorskip("dcc.model")
    from dcc.model import DetectorNet, Refiner

    torch.manual_seed(0)
    model = DetectorNet(240, 320)
    refiner = Refiner()
    rng = np.random.default_rng(42)
    frame = rng.integers(0, 256, size=(240, 320), dtype=np.uint8)
    cfg = {"tau_hm": 0.3, "tau_id": 0.5, "lattice_tol_px": 3.0, "input_size": [320, 240],
           "board": {"square_length_m": 0.02}}

    out = detect(frame, model, refiner, K=None, dist=None, cfg=cfg)
    for key in ("rvec", "tvec", "rms", "reason", "corners", "ambiguous", "demoted", "recovered"):
        assert key in out
    assert out["reason"] is not None or out["rvec"] is not None  # a legal outcome either way
    assert isinstance(out["corners"], list)
    for c in out["corners"]:
        assert set(c) == {"x", "y", "index", "source", "p_hm", "p_id"}
        assert c["source"] in (None, "head", "recovered")
