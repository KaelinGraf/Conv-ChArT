import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import cv2
import numpy as np
import pytest

from dcc.board import render_board, BOARD
from dcc.targets import render_heatmap, render_class_targets, render_refiner_target

CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)


def test_corner_formula():
    measured = {}
    for res in (480, 960):
        img, corners = render_board(res)
        seeds = (corners + 0.8).astype(np.float32).reshape(-1, 1, 2)
        refined = cv2.cornerSubPix(img, seeds, (5, 5), (-1, -1), CRIT)
        err = np.linalg.norm(refined.reshape(-1, 2) - corners, axis=1).max()
        measured[res] = float(err)
        assert err <= 0.05

    lattice = np.array([[(i % 4) + 1, (i // 4) + 1, 0] for i in range(16)], dtype=np.float32)
    assert np.array_equal(BOARD.getChessboardCorners(), lattice)
    print("test_corner_formula max px error:", measured)


def test_marker_identity():
    img, _ = render_board(480)
    sq = 480 // 5
    detector = cv2.aruco.ArucoDetector(BOARD.getDictionary(), cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(img)
    assert sorted(ids.ravel().tolist()) == list(range(12))

    objp = BOARD.getObjPoints()
    errs = [np.abs(c[0] - (objp[mid][:, :2] * sq - 0.5)).max() for c, mid in zip(corners, ids.ravel())]
    max_err = max(errs)
    print("test_marker_identity max px error:", max_err)
    assert max_err <= 1.5


def test_render_res_assert():
    with pytest.raises(ValueError):
        render_board(481)


def test_heatmap_target():
    W, H, sigma = 640, 480, 2.0
    hm = render_heatmap(np.array([[100.3, 50.7]]), np.array([True]), (W, H), sigma=sigma)
    assert hm.shape == (H, W)
    peaks = np.argwhere(hm == 1.0)
    assert peaks.shape[0] == 1
    assert tuple(peaks[0]) == (51, 100)

    for qx, qy in [(103, 52), (98, 49)]:
        r2 = (qx - 100.3) ** 2 + (qy - 50.7) ** 2
        assert hm[qy, qx] == pytest.approx(np.exp(-r2 / (2 * sigma ** 2)), abs=1e-6)

    hm2 = render_heatmap(np.array([[299.5, 200.0], [302.5, 200.0]]), np.array([True, True]),
                          (W, H), sigma=sigma)
    expected_mid = np.exp(-(301 - 299.5) ** 2 / (2 * sigma ** 2))
    assert hm2[200, 301] == pytest.approx(expected_mid, abs=1e-6)

    hm3 = render_heatmap(np.array([[50.0, 50.0]]), np.array([False]), (W, H), sigma=sigma)
    assert np.all(hm3 == 0.0)

    hm4 = render_heatmap(np.array([[0.3, 0.2]]), np.array([True]), (W, H), sigma=sigma)
    assert hm4[0, 0] == 1.0


def test_class_target():
    W, H = 640, 480
    pts, vis, idx = np.array([[101.5, 62.3]]), np.array([True]), np.array([5])
    ct = render_class_targets(pts, vis, idx, (W, H))
    assert ct.shape == (16, 120, 160)
    assert ct[5, 15, 25] == 1.0
    others = [k for k in range(16) if k != 5]
    assert np.all(ct[others, 15, 25] == 0.0)
    expected = np.exp(-((26 - 25.0) ** 2 + (15 - 15.2) ** 2) / 2)
    assert ct[5, 15, 26] == pytest.approx(expected, abs=1e-6)

    ct2 = render_class_targets(pts, np.array([False]), idx, (W, H))
    assert np.all(ct2 == 0.0)

    with pytest.raises(ValueError):
        render_class_targets(pts, vis, idx, (641, 480))


def test_refiner_target_encoding():
    rng = np.random.default_rng(42)
    for _ in range(200):
        d = rng.uniform(-3.9375, 3.9375, size=2)
        rt = render_refiner_target(d)
        peak = np.argwhere(rt == 1.0)
        assert peak.shape[0] == 1
        v_star, u_star = int(np.rint(31.5 + 8 * d[1])), int(np.rint(31.5 + 8 * d[0]))
        assert tuple(peak[0]) == (v_star, u_star)

    d = np.array([1.0, 0.5])
    rt = render_refiner_target(d)
    u_star, v_star = 31.5 + 8 * d[0], 31.5 + 8 * d[1]
    probe_u, probe_v = int(np.rint(u_star)) + 1, int(np.rint(v_star))
    r2 = (probe_u - u_star) ** 2 + (probe_v - v_star) ** 2
    assert rt[probe_v, probe_u] == pytest.approx(np.exp(-r2 / 4.5), abs=1e-6)

    with pytest.raises(ValueError):
        render_refiner_target(np.array([4.0, 0.0]))


def test_heatmap_edge_window():
    W, H, sigma = 640, 480, 2.0
    near = render_heatmap(np.array([[-2.0, 100.0]]), np.array([True]), (W, H), sigma=sigma)
    assert near.shape == (H, W)
    assert near[100, 0] == pytest.approx(np.exp(-((0 + 2.0) ** 2) / (2 * sigma ** 2)), abs=1e-6)
    assert np.all(near[:, 5:] == 0.0)
    assert near.max() < 1.0

    far = render_heatmap(np.array([[-1000.0, 100.0]]), np.array([True]), (W, H), sigma=sigma)
    assert np.all(far == 0.0)
