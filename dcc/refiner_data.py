"""Refiner training crops: mixed_refiner_crops(cfg, rng, bg_files) returns [{crop (24x24 uint8), d
(sub-pixel offset)}], mostly from the fast local-window renderer (fast_refiner_crops) and a
synth.refiner_full_frac share from full composites.
"""
import cv2
import numpy as np
from skimage.exposure import match_histograms

from dcc.board import get_board, render_board
from dcc.synth import (_apply_photometric, _perspective_factor, _sample_affine,
                        _sample_perspective, cut_refiner_crops, generate_sample, visible)

_BG_REF_LUMA = 128.0
_MARGIN = 8
_PATCH = 24 + 2 * _MARGIN

_BG_MATCH_CACHE = {}
_BG_CACHE_MAXSIZE = 32


def _cached_bg_match(bg_path, board_3ch):
    if bg_path in _BG_MATCH_CACHE:
        entry = _BG_MATCH_CACHE.pop(bg_path)
        _BG_MATCH_CACHE[bg_path] = entry
        return entry
    decoded = cv2.imread(bg_path, cv2.IMREAD_COLOR)
    if decoded is None:
        return None
    h, w = decoded.shape[:2]
    if h < _PATCH or w < _PATCH:
        scale = _PATCH / min(h, w)
        decoded = cv2.resize(decoded, (max(_PATCH, int(np.ceil(w * scale))),
                                        max(_PATCH, int(np.ceil(h * scale)))),
                              interpolation=cv2.INTER_LINEAR)
    matched = match_histograms(board_3ch, decoded, channel_axis=-1).astype(np.float32)
    _BG_MATCH_CACHE[bg_path] = (decoded, matched)
    if len(_BG_MATCH_CACHE) > _BG_CACHE_MAXSIZE:
        _BG_MATCH_CACHE.pop(next(iter(_BG_MATCH_CACHE)))
    return decoded, matched


def _window_transform(Hmat, wx0, wy0):
    T = np.array([[1.0, 0.0, -wx0], [0.0, 1.0, -wy0], [0.0, 0.0, 1.0]])
    return T @ Hmat


def _render_fast_window(Hmat, matched, mask_src, bg_tile, cx, cy, cfg=None):
    wx0, wy0 = cx - 12 - _MARGIN, cy - 12 - _MARGIN
    H_win = _window_transform(Hmat, wx0, wy0)
    warped_board = cv2.warpPerspective(matched, H_win, (_PATCH, _PATCH), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped_mask = cv2.warpPerspective(mask_src, H_win, (_PATCH, _PATCH), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    m = warped_mask[..., None]
    r6 = cfg["synth"].get("integration", {}) if cfg is not None else {}
    if r6.get("relight_enabled", False):
        lum = cv2.cvtColor(bg_tile.astype(np.float32), cv2.COLOR_BGR2GRAY)
        lo, hi = r6.get("relight_clip", [0.45, 1.7])
        warped_board = warped_board * float(np.clip(lum.mean() / max(_BG_REF_LUMA, 1e-6), lo, hi))
    f = r6.get("feather_px", 0.0)
    if f > 0:
        warped_mask = cv2.GaussianBlur(warped_mask, (0, 0), sigmaX=float(f))
        m = warped_mask[..., None]
    work = bg_tile.astype(np.float32) * (1 - m) + warped_board * m
    return work, (wx0, wy0)


def fast_refiner_crops(cfg, rng, bg_files):
    syn = cfg["synth"]
    W, H = cfg["input_size"]
    size_mult = syn["refiner_res_mult"]
    w2, h2 = W * size_mult, H * size_mult
    assert w2 == int(w2) and h2 == int(h2), \
        f"input_size {cfg['input_size']} x size_mult {size_mult} is not a whole-pixel canvas"
    w2, h2 = int(w2), int(h2)
    render_res = syn["render_res"]
    bcfg = cfg.get("board")
    nx = get_board(bcfg)[1]
    jitter = cfg["refiner_jitter_px"]
    max_n = syn["refiner_max_corners"]
    ph = syn["photometric"]

    board_img, p_render = render_board(render_res, bcfg)
    board_3ch = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)
    while True:
        idx = int(rng.integers(len(bg_files)))
        entry = _cached_bg_match(bg_files[idx], board_3ch)
        if entry is not None:
            decoded, matched = entry
            break

    M, comps = _sample_affine(cfg, rng, w2, h2, None, size_mult, None, nx)
    tau, psi, fov_scale = _sample_perspective(cfg, rng, None)
    comps.update(tilt=tau, psi=psi, fov_scale=fov_scale)
    M3 = np.eye(3)
    M3[:2, :] = M
    Hmat = M3 @ _perspective_factor(tau, psi, fov_scale, comps["s"], w2, render_res, nx)

    pf = syn["prefilter"]
    s, SQ = comps["s"], render_res // nx
    if pf["enabled"] and s < SQ:
        sigma_r = pf["k"] * (SQ / s - 1.0)
        if sigma_r > 0.1:
            matched = cv2.GaussianBlur(matched, (0, 0), sigmaX=sigma_r)

    p_hom = np.hstack([p_render, np.ones((len(p_render), 1))]) @ Hmat.T
    p_img = p_hom[:, :2] / p_hom[:, 2:3]
    pts = np.array([p for p in p_img if visible(tuple(p), [], (w2, h2))],
                    dtype=np.float64).reshape(-1, 2)
    order = rng.permutation(len(pts))[:max_n] if len(pts) > max_n else np.arange(len(pts))

    mask_src = np.ones((render_res, render_res), dtype=np.float32)
    dh, dw = decoded.shape[:2]

    out = []
    for i in order:
        p = pts[i]
        for _try in range(3):
            j = rng.uniform(-jitter, jitter, size=2)
            c = np.rint(p + j)
            d = p - c
            cx, cy = int(c[0]), int(c[1])
            if max(abs(d[0]), abs(d[1])) <= 3.9375 and 12 <= cx <= w2 - 12 and 12 <= cy <= h2 - 12:
                y0 = int(rng.integers(0, dh - _PATCH + 1))
                x0 = int(rng.integers(0, dw - _PATCH + 1))
                bg_tile = decoded[y0:y0 + _PATCH, x0:x0 + _PATCH]
                work, origin = _render_fast_window(Hmat, matched, mask_src, bg_tile, cx, cy, cfg)
                H_win = _window_transform(Hmat, *origin)
                board_mask = cv2.warpPerspective(mask_src, H_win, (_PATCH, _PATCH), flags=cv2.INTER_LINEAR,
                                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                work = _apply_photometric(work, rng, ph, w2, h2, window_origin=origin,
                                           board_mask=board_mask, board_centroid=tuple(p))
                gray = cv2.cvtColor(np.clip(work, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
                out.append({"crop": gray[_MARGIN:_MARGIN + 24, _MARGIN:_MARGIN + 24].copy(), "d": d})
                break
    return out


def mixed_refiner_crops(cfg, rng, bg_files):
    if rng.random() < cfg["synth"]["refiner_full_frac"]:
        size_mult = cfg["synth"]["refiner_res_mult"]
        record, _ = generate_sample(cfg, rng, bg_files, size_mult=size_mult,
                                     occlude=False, force_negative=False)
        pts = [(c["x"], c["y"]) for c in record["corners"] if c["visible"]]
        return cut_refiner_crops(cfg, rng, record["image"], pts)
    return fast_refiner_crops(cfg, rng, bg_files)
