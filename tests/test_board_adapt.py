import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn

from dcc.board import get_board, n_corners, render_board
from dcc.dataset import load_config
from dcc.losses import detector_loss
from dcc.model import DetectorNet
from dcc.pipeline import canon_lattice, lattice_gate
from dcc.synth import generate_sample
from dcc.targets import render_class_targets
from dcc.trainutil import load_retarget_ckpt, param_groups

CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
CONFIG_PATH = Path(__file__).parents[1] / "configs" / "default.yaml"
BOARD_4X4 = {"squares": [4, 4], "dictionary": "DICT_4X4_50", "marker_ids": [0, 7], "marker_ratio": 0.7,
             "square_length_m": None}


@pytest.fixture(scope="module")
def bg_files(tmp_path_factory):
    """4 random-noise 640x640 backgrounds -- no COCO/network dependency."""
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
def cfg4x4(bg_files):
    """Locally-built 4x4 DICT_4X4_50 board config, input_size [320, 240] --
    default.yaml is never touched."""
    c = load_config(CONFIG_PATH)
    c["input_size"] = [320, 240]
    c["board"] = dict(BOARD_4X4)
    c["synth"] = dict(c["synth"])
    c["synth"]["backgrounds"] = str(Path(bg_files[0]).parent)
    return c


def test_corner_formula_4x4():
    assert n_corners(BOARD_4X4) == 9

    img, corners = render_board(480, BOARD_4X4)
    assert corners.shape == (9, 2)
    nx = 4
    expected = np.array([[((i % (nx - 1)) + 1) * (480 // nx) - 0.5,
                           (i // (nx - 1) + 1) * (480 // nx) - 0.5] for i in range(9)])
    assert np.allclose(corners, expected)

    seeds = (corners + 0.8).astype(np.float32).reshape(-1, 1, 2)
    refined = cv2.cornerSubPix(img, seeds, (5, 5), (-1, -1), CRIT)
    err = np.linalg.norm(refined.reshape(-1, 2) - corners, axis=1).max()
    assert err <= 0.05

    with pytest.raises(ValueError):
        render_board(481, BOARD_4X4)


def test_rectangular_board_rejected():
    with pytest.raises(AssertionError):
        get_board({"squares": [4, 5], "dictionary": "DICT_4X4_50", "marker_ratio": 0.7})


def test_class_target_n9():
    W, H = 320, 240
    pts, vis, idx = np.array([[101.5, 62.3]]), np.array([True]), np.array([5])
    ct = render_class_targets(pts, vis, idx, (W, H), n_cls=9)
    assert ct.shape == (9, 60, 80)
    assert ct[5, 15, 25] == 1.0
    others = [k for k in range(9) if k != 5]
    assert np.all(ct[others, 15, 25] == 0.0)


def test_detector_net_n9():
    torch.manual_seed(0)
    m = DetectorNet(240, 320, n_cls=9).eval()
    hm, cls = m(torch.randn(2, 1, 240, 320))
    assert hm.shape == (2, 1, 240, 320)
    assert cls.shape == (2, 9, 60, 80)
    assert m.cls[-1].out_channels == 9
    assert m.cls[-1].bias.detach().eq(-2.19).all()


def test_generate_sample_4x4(cfg4x4, bg_files):
    rng = np.random.default_rng(123)
    record, _ = generate_sample(cfg4x4, rng, bg_files, occlude=False, force_negative=False)
    assert len(record["corners"]) == 9
    assert [c["index"] for c in record["corners"]] == list(range(9))


def test_retarget_load(tmp_path):
    """Mirrors the CLI's --retarget-from path (dcc/trainutil.py:
    load_retarget_ckpt): a base checkpoint's non-cls.* tensors load into a
    DIFFERENT-n_cls model byte-for-byte; cls.* is untouched by the load and
    only cls.* ends up trainable once the freeze recipe runs."""
    torch.manual_seed(0)
    base_model = DetectorNet(240, 320, n_cls=16)
    ckpt_path = tmp_path / "base_ckpt.pt"
    torch.save({"model": base_model.state_dict(), "cfg": {"board": {"squares": [5, 5]}}}, ckpt_path)

    torch.manual_seed(1)
    model = DetectorNet(240, 320, n_cls=9)
    cls_before = {k: v.clone() for k, v in model.state_dict().items() if k.startswith("cls.")}

    base_ckpt = load_retarget_ckpt(str(ckpt_path), model)

    for k, v in base_model.state_dict().items():
        if not k.startswith("cls."):
            assert torch.equal(v, model.state_dict()[k]), k
    for k, v in cls_before.items():
        assert torch.equal(v, model.state_dict()[k]), f"{k} should be untouched by the retarget load"
    assert base_ckpt["cfg"]["board"]["squares"] == [5, 5]

    # the freeze recipe train_detector.py applies under --freeze-trunk
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith("cls."))
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    for name, m in model.named_modules():
        if not name.startswith("cls") and isinstance(m, bn_types):
            m.eval()

    for name, p in model.named_parameters():
        assert p.requires_grad == name.startswith("cls."), name


def test_retarget_optim_steps_cls_only():
    torch.manual_seed(2)
    model = DetectorNet(240, 320, n_cls=9)
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith("cls."))
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    for name, m in model.named_modules():
        if not name.startswith("cls") and isinstance(m, bn_types):
            m.eval()

    optim = torch.optim.AdamW(param_groups(model, wd=1e-4), lr=1e-3)
    x = torch.randn(2, 1, 240, 320)
    hm_t = torch.rand(2, 1, 240, 320)
    ct_t = torch.rand(2, 9, 60, 80)

    model.train()
    for _ in range(3):
        optim.zero_grad(set_to_none=True)
        hm_logit, cls_logit = model(x)
        loss = detector_loss(hm_logit, cls_logit, hm_t, ct_t, n_vis_batch=2)
        assert torch.isfinite(loss)
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None and torch.any(p.grad != 0), f"cls param {name} has zero/no grad"
            else:
                assert p.grad is None, f"frozen param {name} unexpectedly has a grad"
        optim.step()


def test_canon_lattice_and_gate_n3():
    lattice3 = canon_lattice(3)
    assert lattice3.shape == (9, 2)
    expected = np.array([[(i % 3) + 1.0, (i // 3) + 1.0] for i in range(9)])
    assert np.array_equal(lattice3, expected)

    # A trivial, clearly non-degenerate homography (uniform scale + translate)
    # applied to the n=3 canonical lattice -- lattice_gate must fit it cleanly
    # with all 9 correspondences as inliers.
    scale, tx, ty = 50.0, 10.0, 20.0
    xy = lattice3 * scale + [tx, ty]
    idx = np.arange(9)
    H, inlier_mask, demoted_mask, degenerate = lattice_gate(xy, idx, np.ones(9), tol=3.0, n=3)
    assert degenerate is None and H is not None
    assert inlier_mask.all() and not demoted_mask.any()

    reproj = np.hstack([lattice3, np.ones((9, 1))]) @ H.T
    reproj = reproj[:, :2] / reproj[:, 2:3]
    assert np.abs(reproj - xy).max() < 1e-3
