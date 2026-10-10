"""Video self-supervision: harvest geometry, visibility, masks, masked loss, mixed stream, ONNX import.
    PYTHONPATH= python -m pytest tests/test_video.py -q
"""
import json

import cv2
import numpy as np
import pytest
import torch
import yaml

from dcc import video as V
from dcc.board import render_board
from dcc.losses import detector_loss
from dcc.pipeline import canon_lattice
from dcc.realdata import MixedStream, render_real_targets

LAT = canon_lattice(4)


def _H(s=40.0, theta=0.3, tx=320.0, ty=240.0, persp=(2e-4, -1e-4)):
    c, sn = np.cos(theta), np.sin(theta)
    A = np.array([[s * c, -s * sn, tx - s * 2.5 * (c - sn)], [s * sn, s * c, ty - s * 2.5 * (sn + c)], [0, 0, 1.0]])
    P = np.array([[1, 0, 0], [0, 1, 0], [persp[0], persp[1], 1.0]])
    return A @ P


def _frame(H, bg=100.0, gain=1.0, noise=0.0, rng=None):
    img, _ = render_board(480)
    board = 30 + 180 * img.astype(np.float32) / 255.0
    Ar = np.array([[96.0, 0, -0.5], [0, 96.0, -0.5], [0, 0, 1.0]])
    Hr = H @ np.linalg.inv(Ar)
    warped = cv2.warpPerspective(cv2.GaussianBlur(board, (0, 0), 1.0), Hr, (640, 480), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.ones_like(board), Hr, (640, 480), flags=cv2.INTER_LINEAR)
    f = (bg * (1 - mask) + warped * mask) * gain
    if noise:
        f = f + rng.normal(0, noise, f.shape)
    return np.clip(np.rint(f), 0, 255).astype(np.uint8)


def test_degenerate_sets():
    assert V.degenerate(LAT[[0, 1, 2, 5]])
    assert V.degenerate(LAT[[0, 1, 2, 3, 6]])
    assert V.degenerate(LAT[[0, 1, 2]])
    assert not V.degenerate(LAT[[0, 3, 12, 15]])
    assert not V.degenerate(LAT[[0, 1, 2, 3, 4, 8]])


def test_fit_rejects_wrong_id_and_reports_spread():
    rng = np.random.default_rng(0)
    H = _H()
    idx = np.arange(16)
    xy = V.project(H, LAT) + rng.normal(0, 0.05, (16, 2))
    idx_bad = idx.copy()
    idx_bad[5] = 10
    keep = idx_bad != 10
    keep[5] = True
    fit = V.fit_lattice(idx_bad[keep], xy[keep], 4)
    assert fit is not None and 10 not in fit.inliers
    assert np.abs(V.project(fit.H, LAT) - V.project(H, LAT)).max() < 0.3
    spread = V.fit_lattice(np.array([0, 3, 12, 15, 5, 6, 9, 10]), xy[[0, 3, 12, 15, 5, 6, 9, 10]], 4)
    clustered = V.fit_lattice(np.array([0, 1, 2, 4, 5, 6, 8, 9]), xy[[0, 1, 2, 4, 5, 6, 8, 9]], 4)
    assert clustered.sigma.max() > spread.sigma.max()


def test_ecc_tracking_follows_without_drift():
    rng = np.random.default_rng(1)
    Hs = [_H(s=40 + 0.6 * t, theta=0.3 + 0.01 * t, tx=300 + 2.0 * t, ty=240 - 1.0 * t) for t in range(20)]
    frames = [_frame(H, noise=2.0, rng=rng) for H in Hs]
    tmpl = V.BoardTemplate(None, 320)
    out = V.track(frames, 0, Hs[0], tmpl, +1)
    assert sorted(out) == list(range(1, 20))
    errs = [V.corner_disagreement(out[t][0], Hs[t], 4, frames[t].shape) for t in out]
    assert max(errs) < 0.3, errs
    assert errs[-1] < 0.3


def test_visibility_flags_occlusion_not_darkness():
    rng = np.random.default_rng(2)
    H = _H()
    tmpl = V.BoardTemplate(None, 320)
    bright = _frame(H, noise=1.0, rng=rng)
    pts = V.project(H, LAT)
    occluded = bright.copy()
    for k in (5, 10):
        x, y = np.rint(pts[k]).astype(int)
        occluded[y - 10:y + 11, x - 10:x + 11] = rng.integers(0, 255, (21, 21))
    _, obs = V.corner_visibility(occluded, H, tmpl, 4)
    assert not obs[5] and not obs[10] and obs[[0, 3, 12, 15]].all()
    dark = _frame(H, gain=0.02, noise=1.2, rng=rng)
    _, obs = V.corner_visibility(dark, H, tmpl, 4)
    assert obs.mean() > 0.8


def test_real_targets_and_masks():
    cfg = {"sigma_hm": 0.5, "sigma_cls": 1.0, "board": None}
    image = np.zeros((480, 640), np.uint8)
    corners = [{"x": 100.2, "y": 100.4, "index": 0, "visible": True, "sigma": 0.05},
               {"x": 200.0, "y": 150.0, "index": 1, "visible": True, "sigma": 0.4},
               {"x": 300.0, "y": 200.0, "index": 2, "visible": None, "sigma": 0.1},
               {"x": -20.0, "y": 50.0, "index": 3, "visible": False, "sigma": 0.1}]
    t = render_real_targets(cfg, image, corners)
    hm, m, cm = t["heatmap"].numpy(), t["hm_mask"].numpy(), t["cls_mask"].numpy()
    assert t["n_vis"] == 2 and hm[100, 100] == 1.0 and hm[150, 200] == 1.0
    assert m[100, 99] == 1 and m[100, 101] == 1
    assert m[150, 200] == 1 and m[149, 199] == 0 and m[151, 201] == 0
    assert m[200, 300] == 0 and m[200, 303] == 0 and m[200, 310] == 1
    cx, cy = int((300 + 0.5) / 4), int((200 + 0.5) / 4)
    assert cm[2, cy, cx] == 0 and cm[0, cy, cx] == 1 and cm[1, cy, cx] == 1


def test_noise_model_and_darkening_reproduce_a_dark_frame():
    from dcc.realdata import darken
    rng = np.random.default_rng(4)
    shot, read_var = 0.05, 0.3
    clean = [cv2.GaussianBlur(rng.uniform(lv * 0.5, lv * 1.5, (240, 320)), (0, 0), 6) for lv in (4, 10, 40, 80, 150)]
    frames = [np.clip(np.rint(c + rng.normal(0, 1, c.shape) * np.sqrt(shot * c + read_var)), 0, 255).astype(np.uint8)
              for c in clean]
    nm = V.noise_model(frames)
    assert abs(nm["shot"] - shot) < 0.02 and abs(nm["read_var"] - read_var) < 0.15
    g = 0.05
    out = darken(frames[3], rng, nm, gain_min=g * 0.999)
    sigma, level = V.frame_noise(out)
    expect = np.sqrt(nm["shot"] * level + nm["read_var"])
    assert abs(sigma - expect) / expect < 0.25, (sigma, expect)


def test_masked_loss_is_backward_compatible():
    torch.manual_seed(0)
    hm, cls = torch.randn(2, 1, 32, 40), torch.randn(2, 16, 8, 10)
    ht, ct = torch.rand(2, 1, 32, 40) * 0.9, torch.rand(2, 16, 8, 10) * 0.9
    ht[0, 0, 5, 5] = 1.0
    ct[1, 3, 2, 2] = 1.0
    for kw in ({"loss_form": "focal"}, {"loss_form": "ce", "loss_form_cls": "focal"}):
        a = detector_loss(hm, cls, ht, ct, 2, 2.0, **kw)
        b = detector_loss(hm, cls, ht, ct, 2, 2.0, hm_mask=torch.ones_like(hm), cls_mask=torch.ones_like(cls), **kw)
        z = detector_loss(hm, cls, ht, ct, 2, 2.0, hm_mask=torch.zeros_like(hm), cls_mask=torch.zeros_like(cls), **kw)
        assert torch.equal(a, b) and float(z) == 0.0


@pytest.fixture()
def tiny_world(tmp_path):
    rng = np.random.default_rng(3)
    bgd = tmp_path / "bg"
    bgd.mkdir()
    for i in range(3):
        cv2.imwrite(str(bgd / f"{i}.png"), rng.integers(0, 255, (150, 200, 3), dtype=np.uint8))
    cfg = yaml.safe_load(open("configs/default.yaml"))
    cfg["input_size"] = [160, 120]
    cfg["scale_range_px"] = [12, 20]
    cfg["synth"]["backgrounds"] = str(bgd)
    cfg["synth"]["cutouts"]["path"] = None
    H = _H(s=14.0, tx=80.0, ty=60.0, persp=(0.0, 0.0))
    pts = V.project(H, LAT)
    outline = V.project(H, np.array([[0, 0], [5, 0], [5, 5], [0, 5]], float))
    rec = {"frame": 0, "source": "anchor", "rho": 0.9, "check": 0.1, "board_present": True, "s_px": 14.0,
           "board_level": 100.0, "anisotropy": 0.0, "outline": outline.tolist(),
           "corners": [{"x": float(x), "y": float(y), "index": k, "visible": True, "sigma": 0.1, "ncc": 0.9}
                       for k, (x, y) in enumerate(pts)]}
    images = [cv2.resize(_frame(_H(s=14.0 * 4, tx=320, ty=240, persp=(0, 0))), (160, 120))] * 2
    V.save_harvest(tmp_path / "harvest", "clip", images, [rec, {**rec, "frame": 1}], {})
    cfg["real"] = {"records": [str(tmp_path / "harvest" / "clip")], "frac": 0.5}
    return cfg


def test_mixed_stream_is_deterministic_and_masked(tiny_world):
    def take(n):
        it = iter(MixedStream(tiny_world, seed=7))
        return [next(it) for _ in range(n)]
    a, b = take(6), take(6)
    for x, y in zip(a, b):
        assert torch.equal(x["image"], y["image"]) and torch.equal(x["hm_mask"], y["hm_mask"])
    for x in a:
        assert set(x) >= {"image", "heatmap", "classes", "n_vis", "hm_mask", "cls_mask"}
        assert x["hm_mask"].shape == (120, 160) and x["cls_mask"].shape == (16, 30, 40)


def test_onnx_import_round_trip(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from dcc.model import DetectorNet, Refiner
    from dcc.onnx_import import detector_from_onnx, refiner_from_onnx
    cfg = {"input_size": [320, 240], "attend_div": 8, "width_mult": 0.25, "e4_dilated": False, "xsa": True,
           "board": None}
    torch.manual_seed(0)
    model = DetectorNet(240, 320, attend_div=8, width_mult=0.25, e4_dilated=False, xsa=True).eval()
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.2, 0.2)
            m.running_var.uniform_(0.5, 2.0)
    path = tmp_path / "det.onnx"
    torch.onnx.export(model, torch.randn(1, 1, 240, 320), str(path), opset_version=17, dynamo=False,
                      input_names=["input"], output_names=["hm", "cls"])
    rebuilt, diff = detector_from_onnx(path, cfg)
    assert diff < 1e-3
    ref = Refiner().eval()
    rpath = tmp_path / "ref.onnx"
    torch.onnx.export(ref, torch.randn(1, 1, 24, 24), str(rpath), opset_version=17, dynamo=False,
                      input_names=["input"], output_names=["logits"])
    _, rdiff = refiner_from_onnx(rpath)
    assert rdiff < 1e-3
    bad = {**cfg, "xsa": False}
    with pytest.raises(ValueError):
        detector_from_onnx(path, bad)
