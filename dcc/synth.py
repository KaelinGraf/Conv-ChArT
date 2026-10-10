"""Synthetic sample generator. generate_sample(cfg, rng, bg_files) returns (record, meta): a composited
grayscale image at cfg["input_size"] with corners [{x, y, index, visible}]. list_backgrounds(path) lists a
background corpus (an image directory or a COCO .json); cut_refiner_crops and make_generic_crop produce
refiner crops. Every random draw comes from the numpy Generator passed in.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from skimage.exposure import match_histograms

from dcc.board import get_board, render_board


def list_backgrounds(path):
    p = Path(path)
    if p.suffix == ".json":
        data = json.loads(p.read_text())
        base = p.parent / p.stem
        return sorted(str(base / im["file_name"]) for im in data["images"])
    return sorted(str(f) for pat in ("*.jpg", "*.jpeg", "*.png") for f in p.rglob(pat))


def load_cutouts(path):
    if not path:
        return []
    p = Path(path)
    return sorted(str(f) for f in p.glob("*.png")) if p.is_dir() else []


_CUTOUT_CACHE = {}


def _cached_cutouts(path):
    if path not in _CUTOUT_CACHE:
        _CUTOUT_CACHE[path] = load_cutouts(path)
    return _CUTOUT_CACHE[path]


def visible(p, holes, size_wh):
    x, y = p
    w, h = size_wh
    if not (-0.5 <= x < w - 0.5 and -0.5 <= y < h - 0.5):
        return False
    for x0, y0, hw, hh in holes:
        if x0 - 0.5 <= x < x0 + hw - 0.5 and y0 - 0.5 <= y < y0 + hh - 0.5:
            return False
    return True


def _prep_background(bg, rng, syn, w2, h2):
    if rng.random() < syn["bg_hflip_p"]:
        bg = cv2.flip(bg, 1)
    h, w = bg.shape[:2]
    cover = max(w2 / w, h2 / h) * 1.15
    if cover > 1.0:
        bg = cv2.resize(bg, (int(np.ceil(w * cover)), int(np.ceil(h * cover))),
                        interpolation=cv2.INTER_LINEAR)
        h, w = bg.shape[:2]
    angle = rng.uniform(-syn["rot_deg"], syn["rot_deg"])
    Mrot = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, 1.0)
    bg = cv2.warpAffine(bg, Mrot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    pad_h, pad_w = max(0, h2 - h), max(0, w2 - w)
    if pad_h or pad_w:
        top, left = pad_h // 2, pad_w // 2
        bg = cv2.copyMakeBorder(bg, top, pad_h - top, left, pad_w - left, cv2.BORDER_REFLECT)
    h, w = bg.shape[:2]
    y0 = int(rng.integers(0, h - h2 + 1))
    x0 = int(rng.integers(0, w - w2 + 1))
    return bg[y0:y0 + h2, x0:x0 + w2]


def _sample_affine(cfg, rng, w2, h2, s_arg, size_mult, components, nx=5):
    comp = components or {}
    if "s" in comp:
        s = float(comp["s"])
    elif s_arg is not None:
        s = float(s_arg)
    else:
        a, b = cfg["scale_range_px"]
        s = float(np.exp(rng.uniform(np.log(a * size_mult), np.log(b * size_mult))))
    shear_deg = cfg["synth"]["shear_deg"]
    tf = cfg["synth"]["translate_frac"]
    theta = float(comp["theta"]) if "theta" in comp else float(rng.uniform(-np.pi, np.pi))
    shear_x = float(comp["shear_x"]) if "shear_x" in comp else float(np.radians(rng.uniform(-shear_deg, shear_deg)))
    shear_y = float(comp["shear_y"]) if "shear_y" in comp else float(np.radians(rng.uniform(-shear_deg, shear_deg)))
    tx = float(comp["tx"]) if "tx" in comp else float(rng.uniform(-tf, tf) * w2)
    ty = float(comp["ty"]) if "ty" in comp else float(rng.uniform(-tf, tf) * h2)

    render_res = cfg["synth"]["render_res"]
    SQ = render_res // nx
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    Sh = np.array([[1.0, np.tan(shear_x)], [np.tan(shear_y), 1.0]])
    A = (s / SQ) * (R @ Sh)
    c_r = np.array([(render_res - 1) / 2, (render_res - 1) / 2])
    c_in = np.array([(w2 - 1) / 2, (h2 - 1) / 2])
    M = np.zeros((2, 3), dtype=np.float64)
    M[:, :2] = A
    M[:, 2] = c_in + np.array([tx, ty]) - A @ c_r
    comps_out = {"s": s, "theta": theta, "shear_x": shear_x, "shear_y": shear_y, "tx": tx, "ty": ty}
    return M, comps_out


def _sample_perspective(cfg, rng, components):
    comp = components or {}
    syn = cfg["synth"]
    coin = rng.random() < syn["perspective_p"]
    tau_drawn = float(rng.uniform(0.0, np.radians(syn["tilt_max_deg"])))
    psi_drawn = float(rng.uniform(0.0, 2 * np.pi))
    fov_drawn = float(rng.uniform(*syn["fov_scale"]))
    tau = float(comp["tilt"]) if "tilt" in comp else (tau_drawn if coin else 0.0)
    psi = float(comp["psi"]) if "psi" in comp else (psi_drawn if coin else 0.0)
    fov_scale = float(comp["fov_scale"]) if "fov_scale" in comp else fov_drawn
    return tau, psi, fov_scale


def _perspective_factor(tau, psi, fov_scale, s, w2, render_res, nx=5):
    SQ = render_res // nx
    g = np.sin(tau) * s / (fov_scale * w2 * SQ)
    gx, gy = g * np.cos(psi), g * np.sin(psi)
    cr = (render_res - 1) / 2
    Tcr = np.array([[1.0, 0.0, cr], [0.0, 1.0, cr], [0.0, 0.0, 1.0]])
    Tmcr = np.array([[1.0, 0.0, -cr], [0.0, 1.0, -cr], [0.0, 0.0, 1.0]])
    Pg = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [gx, gy, 1.0]])
    return Tcr @ Pg @ Tmcr


def _warp_mask(H, render_res, w2, h2):
    mask_src = np.ones((render_res, render_res), dtype=np.float32)
    return cv2.warpPerspective(mask_src, H, (w2, h2), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _integrate_board(cfg, rng, bg, board, mask, w2, h2):
    r6 = cfg["synth"].get("integration", {})
    if r6.get("relight_enabled", False):
        lum = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        sigma = max(r6.get("relight_sigma_frac", 0.125) * min(w2, h2), 1.0)
        field = cv2.GaussianBlur(lum, (0, 0), sigmaX=sigma)
        field = field / max(float(field.mean()), 1e-6)
        lo, hi = r6.get("relight_clip", [0.45, 1.7])
        board = board * np.clip(field, lo, hi)[..., None]
    if r6.get("shadow_enabled", False):
        ang = float(rng.uniform(0, 2 * np.pi))
        off = float(rng.uniform(*r6.get("shadow_offset_frac", [0.005, 0.03]))) * min(w2, h2)
        Mt = np.float32([[1, 0, off * np.cos(ang)], [0, 1, off * np.sin(ang)]])
        sh = cv2.warpAffine(mask, Mt, (w2, h2), flags=cv2.INTER_LINEAR, borderValue=0)
        sh = cv2.GaussianBlur(sh, (0, 0), sigmaX=max(off * 0.6, 1.0))
        k = float(rng.uniform(*r6.get("shadow_strength", [0.15, 0.5])))
        board_free = np.clip(sh * (1.0 - mask), 0.0, 1.0)
        bg = bg * (1.0 - k * board_free)[..., None]
    f = r6.get("feather_px", 0.0)
    if f > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=float(f))
    return board, mask, bg

def _composite_board(bg_crop, rng, cfg, w2, h2, s_arg, size_mult, components):
    render_res = cfg["synth"]["render_res"]
    bcfg = cfg.get("board")
    pool = (bcfg or {}).get("pool")
    if pool:
        bcfg = {**bcfg, **pool[int(rng.integers(len(pool)))]}
        assert bcfg.get("squares", cfg["board"].get("squares")) == cfg["board"].get("squares"), \
            f"board.pool entries must keep board.squares {cfg['board'].get('squares')}; got {bcfg.get('squares')}"
    board_img, p_render = render_board(render_res, bcfg)
    nx = get_board(bcfg)[1]
    M, comps = _sample_affine(cfg, rng, w2, h2, s_arg, size_mult, components, nx)
    if pool:
        comps["board"] = {"dictionary": bcfg["dictionary"], "marker_id_offset": int(bcfg.get("marker_id_offset", 0) or 0)}
    tau, psi, fov_scale = _sample_perspective(cfg, rng, components)
    comps.update(tilt=tau, psi=psi, fov_scale=fov_scale)
    M3 = np.eye(3)
    M3[:2, :] = M
    H = M3 @ _perspective_factor(tau, psi, fov_scale, comps["s"], w2, render_res, nx)

    board_3ch = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)
    matched = match_histograms(board_3ch, bg_crop, channel_axis=-1).astype(np.float32)

    pf = cfg["synth"]["prefilter"]
    s, SQ = comps["s"], render_res // nx
    if pf["enabled"] and s < SQ:
        sigma_r = pf["k"] * (SQ / s - 1.0)
        if sigma_r > 0.1:
            matched = cv2.GaussianBlur(matched, (0, 0), sigmaX=sigma_r)

    warped_board = cv2.warpPerspective(matched, H, (w2, h2), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped_mask = _warp_mask(H, render_res, w2, h2)
    warped_board, warped_mask, bg_lit = _integrate_board(cfg, rng, bg_crop.astype(np.float32),
                                                          warped_board, warped_mask, w2, h2)
    m = warped_mask[..., None]
    work = bg_lit * (1 - m) + warped_board * m

    p_hom = np.hstack([p_render, np.ones((len(p_render), 1))]) @ H.T
    p_img = p_hom[:, :2] / p_hom[:, 2:3]
    return work, p_img, H, comps


def place_cutout(work, occ_alpha, rgba, scale, rot, hflip, cx, cy):
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
        return None
    if hflip:
        rgba = cv2.flip(rgba, 1)

    h2, w2 = work.shape[:2]
    ch, cw = rgba.shape[:2]
    long_side = max(ch, cw)
    if long_side <= 0:
        return None
    f = (scale * min(w2, h2)) / long_side
    new_w, new_h = max(1, int(round(cw * f))), max(1, int(round(ch * f)))
    rgba = cv2.resize(rgba, (new_w, new_h), interpolation=cv2.INTER_AREA if f < 1.0 else cv2.INTER_LINEAR)

    centre = (new_w / 2, new_h / 2)
    Mrot = cv2.getRotationMatrix2D(centre, rot, 1.0)
    cos_a, sin_a = abs(Mrot[0, 0]), abs(Mrot[0, 1])
    exp_w = int(new_h * sin_a + new_w * cos_a)
    exp_h = int(new_h * cos_a + new_w * sin_a)
    Mrot[0, 2] += exp_w / 2 - centre[0]
    Mrot[1, 2] += exp_h / 2 - centre[1]
    rgba = cv2.warpAffine(rgba, Mrot, (exp_w, exp_h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    rgba[..., 3] = cv2.GaussianBlur(rgba[..., 3], (3, 3), 0)

    ox, oy = int(round(cx - exp_w / 2)), int(round(cy - exp_h / 2))
    x0, y0 = max(ox, 0), max(oy, 0)
    x1, y1 = min(ox + exp_w, w2), min(oy + exp_h, h2)
    if x1 <= x0 or y1 <= y0:
        return None
    lx0, ly0 = x0 - ox, y0 - oy
    patch = rgba[ly0:ly0 + (y1 - y0), lx0:lx0 + (x1 - x0)]
    alpha = patch[..., 3:4].astype(np.float32) / 255.0
    work[y0:y1, x0:x1] = work[y0:y1, x0:x1] * (1 - alpha) + patch[..., :3].astype(np.float32) * alpha
    occ_alpha[y0:y1, x0:x1] = np.maximum(occ_alpha[y0:y1, x0:x1], alpha[..., 0])
    return (x0, y0, x1 - x0, y1 - y0)


def _apply_cutouts(work, rng, cfg, cutout_files, w2, h2):
    cutouts = cfg["synth"]["cutouts"]
    coin = rng.random() < cutouts["p"]
    occ_alpha = np.zeros((h2, w2), dtype=np.float32)
    placed = []
    for _ in range(cutouts["max_objects"]):
        idx = int(rng.integers(2 ** 31))
        scale = float(rng.uniform(*cutouts["scale"]))
        rot = float(rng.uniform(-180.0, 180.0))
        hflip = rng.random() < 0.5
        cx, cy = rng.uniform(-0.1, 1.1, size=2) * (w2, h2)
        use_slot = rng.random() < 0.7
        if not (coin and cutout_files and use_slot):
            continue
        path = cutout_files[idx % len(cutout_files)]
        rgba = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        bbox = place_cutout(work, occ_alpha, rgba, scale, rot, hflip, cx, cy)
        if bbox is not None:
            placed.append({"file": path, "bbox": bbox})

    return work, occ_alpha, placed


def _apply_occlusion(work, rng, occ, w2, h2):
    holes = []
    if rng.random() < occ["p"]:
        lo_n, hi_n = occ["holes"]
        lo_s, hi_s = occ["size"]
        n = int(rng.integers(lo_n, hi_n + 1))
        for _ in range(n):
            hw = int(rng.integers(lo_s, hi_s + 1))
            hh = int(rng.integers(lo_s, hi_s + 1))
            x0 = int(rng.integers(0, w2 - hw + 1))
            y0 = int(rng.integers(0, h2 - hh + 1))
            choice = int(rng.integers(0, 4))
            if choice == 3:
                fill = rng.integers(0, 256, size=(hh, hw, 3)).astype(np.float32)
            else:
                fill = (0.0, 128.0, 255.0)[choice]
            work[y0:y0 + hh, x0:x0 + hw] = fill
            holes.append((x0, y0, hw, hh))
    return holes


def _sample_on_mask(rng, mask, thresh=0.5, tries=5):
    ys, xs = np.where(mask > thresh)
    h, w = mask.shape[:2]
    if len(xs) == 0:
        return w / 2.0, h / 2.0
    x0, x1, y0, y1 = float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())
    for _ in range(tries):
        x, y = rng.uniform((x0, y0), (x1 + 1.0, y1 + 1.0))
        iy, ix = min(int(y), h - 1), min(int(x), w - 1)
        if mask[iy, ix] > thresh:
            return float(x), float(y)
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def _apply_refractive_droplet(work, cx, cy, radius, minify, alpha, blur_sigma):
    h, w = work.shape[:2]
    diam = max(2, int(round(2 * radius)))
    side = min(max(diam, int(round(diam / minify))), w, h)
    x0 = int(np.clip(round(cx - side / 2), 0, max(0, w - side)))
    y0 = int(np.clip(round(cy - side / 2), 0, max(0, h - side)))
    patch = cv2.resize(work[y0:y0 + side, x0:x0 + side], (diam, diam), interpolation=cv2.INTER_AREA)
    patch = cv2.flip(patch, 0)
    patch = cv2.GaussianBlur(patch, (0, 0), sigmaX=blur_sigma)

    yy, xx = np.mgrid[0:diam, 0:diam].astype(np.float32) - (diam - 1) / 2.0
    disc = ((xx ** 2 + yy ** 2) <= (diam / 2.0) ** 2).astype(np.float32) * alpha

    px0, py0 = int(round(cx - diam / 2.0)), int(round(cy - diam / 2.0))
    tx0, ty0 = max(px0, 0), max(py0, 0)
    tx1, ty1 = min(px0 + diam, w), min(py0 + diam, h)
    if tx1 <= tx0 or ty1 <= ty0:
        return work
    lx0, ly0 = tx0 - px0, ty0 - py0
    lx1, ly1 = lx0 + (tx1 - tx0), ly0 + (ty1 - ty0)
    a = disc[ly0:ly1, lx0:lx1, None]
    out = work.copy()
    out[ty0:ty1, tx0:tx1] = work[ty0:ty1, tx0:tx1] * (1 - a) + patch[ly0:ly1, lx0:lx1] * a
    return out


def _apply_photometric(work, rng, ph, w2, h2, window_origin=None, board_mask=None,
                        board_centroid=None, holes_out=None):
    ph_h, ph_w = work.shape[:2]
    wx0, wy0 = (0, 0) if window_origin is None else window_origin

    _grid = [None]

    def abs_grid():
        if _grid[0] is None:
            ys, xs = np.mgrid[wy0:wy0 + ph_h, wx0:wx0 + ph_w]
            _grid[0] = (ys.astype(np.float32), xs.astype(np.float32))
        return _grid[0]

    have_board = board_mask is not None and np.any(board_mask > 0.5)
    cxb, cyb = board_centroid if board_centroid is not None else ((w2 - 1) / 2.0, (h2 - 1) / 2.0)

    if rng.random() < ph["specular_p"]:
        if have_board:
            lx, ly = _sample_on_mask(rng, board_mask)
            cx, cy = wx0 + lx, wy0 + ly
            strength = rng.uniform(*ph["specular_strength"])
            n = rng.uniform(*ph["specular_exponent"])
            spread = rng.uniform(*ph["specular_spread_frac"]) * min(w2, h2)
            ys, xs = abs_grid()
            cos_nh = np.clip(np.exp(-0.5 * ((xs - cx) ** 2 + (ys - cy) ** 2) / spread ** 2), 0.0, 1.0)
            work = work + (strength * cos_nh ** n)[..., None].astype(np.float32)

    if rng.random() < ph["droplet_p"]:
        lo, hi = ph["droplet_n"]
        for _ in range(int(rng.integers(lo, hi + 1))):
            cx, cy = rng.uniform((0.0, 0.0), (float(w2), float(h2)))
            mode_b = window_origin is None and rng.random() < ph["droplet_mode_b_p"]
            if mode_b:
                radius = float(rng.uniform(*ph["droplet_refractive_radius"]))
                minify = rng.uniform(*ph["droplet_refractive_minify"])
                alpha = rng.uniform(*ph["droplet_refractive_alpha"])
                blur_sigma = rng.uniform(*ph["droplet_refractive_blur_sigma"])
                work = _apply_refractive_droplet(work, cx, cy, radius, minify, alpha, blur_sigma)
                if (holes_out is not None and alpha > ph["droplet_hole_alpha_thresh"]
                        and radius > ph["droplet_hole_radius_thresh"]):
                    holes_out.append((cx - radius, cy - radius, 2 * radius, 2 * radius))
            else:
                radius = rng.uniform(*ph["droplet_bokeh_radius"])
                brightness = rng.uniform(*ph["droplet_bokeh_brightness"])
                ys, xs = abs_grid()
                blob = brightness * np.exp(-0.5 * ((xs - cx) ** 2 + (ys - cy) ** 2) / radius ** 2)
                work = work + blob[..., None].astype(np.float32)

    if rng.random() < ph["vignette_p"]:
        v = rng.uniform(*ph["vignette_strength"])
        ys, xs = abs_grid()
        ccx, ccy = (w2 - 1) / 2.0, (h2 - 1) / 2.0
        r_max = 0.5 * np.sqrt(w2 ** 2 + h2 ** 2)
        r2 = ((xs - ccx) ** 2 + (ys - ccy) ** 2) / r_max ** 2
        work = work * ((1 - v * r2) ** 2)[..., None].astype(np.float32)

    if rng.random() < ph["ink_contrast_p"]:
        if have_board:
            mask_bool = board_mask > 0.5
            luma = work.mean(axis=-1)
            vals = luma[mask_bool]
            median, white = np.median(vals), np.percentile(vals, 95)
            scale = rng.uniform(*ph["ink_contrast_scale"])
            dark = (mask_bool & (luma < median))[..., None]
            work = np.where(dark, np.minimum(work * scale, white), work).astype(np.float32)

    if rng.random() < ph["motion_blur_p"]:
        ks = np.arange(3, ph["motion_blur_kmax"] + 1, 2)
        k = int(rng.choice(ks))
        angle = rng.uniform(0, np.pi)
        c = k // 2
        dx, dy = np.cos(angle) * c, np.sin(angle) * c
        kernel = np.zeros((k, k), dtype=np.float32)
        cv2.line(kernel, (int(round(c - dx)), int(round(c - dy))),
                  (int(round(c + dx)), int(round(c + dy))), 1.0, 1)
        kernel /= kernel.sum()
        work = cv2.filter2D(work, -1, kernel)
    if rng.random() < ph["gauss_blur_p"]:
        k = int(rng.choice([3, 5, 7]))
        work = cv2.GaussianBlur(work, (k, k), 0)
    dark = False
    diff_noised = False
    if rng.random() < ph["differencing_p"]:
        ambient = rng.uniform(*ph["differencing_ambient"])
        peak = rng.uniform(*ph["differencing_illum_peak"])
        floor = rng.uniform(*ph["differencing_illum_floor"])
        sigma = rng.uniform(*ph["differencing_illum_sigma_frac"]) * min(w2, h2)
        mag = rng.uniform(*ph["differencing_shift_px"])
        ang = rng.uniform(0, 2 * np.pi)
        ys, xs = abs_grid()
        illum_lobe = floor + (peak - floor) * np.exp(-0.5 * ((xs - cxb) ** 2 + (ys - cyb) ** 2) / sigma ** 2)
        lit_raw = work * (ambient + illum_lobe)[..., None].astype(np.float32)
        if ph.get("differencing_autoexposure", False):
            hi = float(np.percentile(lit_raw, 99.9))
            gain = min(1.0, ph.get("differencing_ae_target", 250.0) / max(hi, 1e-6))
            lit_raw = lit_raw * gain
            ambient_eff = ambient * gain
        else:
            ambient_eff = ambient
        i_lit = np.clip(lit_raw, 0, 255)
        i_unlit = np.clip(work * ambient_eff, 0, 255)

        K = rng.uniform(*ph["sensor_noise_electrons_per_dn"])
        sigma_read = rng.uniform(*ph["sensor_noise_read_std"])
        i_lit = (rng.poisson(np.maximum(i_lit, 0.0) * K).astype(np.float32) / K
                 + rng.normal(0, sigma_read, i_lit.shape).astype(np.float32))
        i_unlit = (rng.poisson(np.maximum(i_unlit, 0.0) * K).astype(np.float32) / K
                   + rng.normal(0, sigma_read, i_unlit.shape).astype(np.float32))
        diff_noised = True

        dxs, dys = mag * np.cos(ang), mag * np.sin(ang)
        Mshift = np.array([[1, 0, dxs], [0, 1, dys]], dtype=np.float64)
        i_unlit_shifted = cv2.warpAffine(i_unlit, Mshift, (ph_w, ph_h), flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_REFLECT)
        work = np.clip(i_lit - i_unlit_shifted, 0, 255)
        dark = np.percentile(work, 99) < ph["dark_greyout_white_thresh"]
    elif rng.random() < ph["brightness_p"]:
        b = rng.uniform(*ph["brightness_range"])
        if rng.random() < 0.5:
            work = work * (1 + b)
        else:
            work = work + max(b, ph["brightness_add_floor"]) * 255
        dark = b < ph["dark_greyout_brightness_thresh"]

    if dark:
        blend = rng.uniform(*ph["dark_greyout_blend"])
        m = work.mean()
        work = (1 - blend) * work + blend * m

    if rng.random() < ph["gauss_noise_p"]:
        if diff_noised:
            pass
        elif ph["sensor_noise_enabled"]:
            K = rng.uniform(*ph["sensor_noise_electrons_per_dn"])
            sigma_read = rng.uniform(*ph["sensor_noise_read_std"])
            rate = np.maximum(work, 0.0) * K
            shot = rng.poisson(rate).astype(np.float32) / K
            read = rng.normal(0, sigma_read, work.shape).astype(np.float32)
            work = shot + read
        else:
            std = rng.uniform(*ph["noise_std"])
            work = work + rng.normal(0, std, work.shape).astype(np.float32)

    if window_origin is None and rng.random() < ph["fpn_p"]:
        sigma_c = rng.uniform(*ph["fpn_col_std"])
        col = rng.normal(0, sigma_c, size=(1, ph_w, 1)).astype(np.float32)
        row = rng.normal(0, sigma_c * 0.5, size=(ph_h, 1, 1)).astype(np.float32)
        work = work + col + row
        if rng.random() < ph["fpn_prnu_p"]:
            prnu_std = rng.uniform(*ph["fpn_prnu_std"])
            work = work * rng.normal(1.0, prnu_std, size=(ph_h, ph_w, 1)).astype(np.float32)

    if rng.random() < ph["speckle_p"]:
        std = rng.uniform(*ph["speckle_std"])
        work = work * (1 + rng.normal(0, std, (ph_h, ph_w, 1)).astype(np.float32))
    if rng.random() < ph["mult_noise_p"]:
        field = rng.uniform(*ph["mult_noise_range"], size=(ph_h, ph_w, 1)).astype(np.float32)
        work = work * field
    if rng.random() < ph["contrast_p"]:
        c = rng.uniform(*ph["contrast_range"])
        m = work.mean()
        work = m + (work - m) * c
    if rng.random() < ph["rgb_shift_p"]:
        shift = rng.uniform(-ph["rgb_shift_limit"], ph["rgb_shift_limit"], size=3).astype(np.float32)
        work = work + shift
    if rng.random() < ph["glare_p"]:
        cx, cy = rng.uniform(0, w2), rng.uniform(0, h2)
        ax, ay = rng.uniform(20, 120), rng.uniform(20, 120)
        ang = rng.uniform(0, np.pi)
        peak = rng.uniform(*ph.get("glare_peak", (30, 130)))
        ys, xs = abs_grid()
        xr = (xs - cx) * np.cos(ang) + (ys - cy) * np.sin(ang)
        yr = -(xs - cx) * np.sin(ang) + (ys - cy) * np.cos(ang)
        glare = peak * np.exp(-0.5 * ((xr / ax) ** 2 + (yr / ay) ** 2))
        work = work + glare[..., None].astype(np.float32)
    if rng.random() < ph["ghost_p"]:
        alpha = rng.uniform(*ph["ghost_alpha"])
        mag = rng.uniform(*ph["ghost_shift_px"])
        ang = rng.uniform(0, 2 * np.pi)
        dx, dy = mag * np.cos(ang), mag * np.sin(ang)
        blurred = cv2.GaussianBlur(work, (3, 3), 0)
        Mshift = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float64)
        shifted = cv2.warpAffine(blurred, Mshift, (ph_w, ph_h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT)
        work = (1 + alpha) * work - alpha * shifted
    return work


def generate_sample(cfg, rng, bg_files, s=None, size_mult=1, force_negative=None,
                     photometric=True, occlude=True, components=None, cutout_files=None):
    W, H = cfg["input_size"]
    w2, h2 = W * size_mult, H * size_mult
    assert w2 == int(w2) and h2 == int(h2), \
        f"input_size {cfg['input_size']} x size_mult {size_mult} is not a whole-pixel canvas"
    w2, h2 = int(w2), int(h2)
    negative = force_negative if force_negative is not None else rng.random() < cfg["negative_p"]
    if cutout_files is None:
        cutout_files = _cached_cutouts(cfg["synth"].get("cutouts", {}).get("path"))

    while True:
        idx = int(rng.integers(len(bg_files)))
        bg = cv2.imread(bg_files[idx], cv2.IMREAD_COLOR)
        if bg is not None:
            break
    bg_crop = _prep_background(bg, rng, cfg["synth"], w2, h2)

    if negative:
        work = bg_crop.astype(np.float32)
        p_img, M, comps, s_px, board_mask = None, None, None, 0.0, None
    else:
        work, p_img, M, comps = _composite_board(bg_crop, rng, cfg, w2, h2, s, size_mult, components)
        s_px = comps["s"]
        board_mask = _warp_mask(M, cfg["synth"]["render_res"], w2, h2)
    board_centroid = tuple(p_img.mean(axis=0)) if p_img is not None else None

    holes, cutouts_meta, occ_alpha = [], [], np.zeros((h2, w2), dtype=np.float32)
    if occlude:
        work, occ_alpha, cutouts_meta = _apply_cutouts(work, rng, cfg, cutout_files, w2, h2)
        holes = _apply_occlusion(work, rng, cfg["synth"]["occlusion"], w2, h2)

    if photometric:
        work = _apply_photometric(work, rng, cfg["synth"]["photometric"], w2, h2,
                                   board_mask=board_mask, board_centroid=board_centroid, holes_out=holes)

    corners_out = []
    if not negative:
        for k, (x, y) in enumerate(p_img):
            vis = visible((x, y), holes, (w2, h2))
            if vis:
                iy, ix = int(np.rint(y)), int(np.rint(x))
                if 0 <= iy < h2 and 0 <= ix < w2:
                    vis = bool(occ_alpha[iy, ix] < 0.5)
            corners_out.append({"x": float(x), "y": float(y), "index": k, "visible": vis})

    image = cv2.cvtColor(np.clip(work, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

    record = {"image": image, "board_present": not negative, "s_px": float(s_px), "corners": corners_out}
    if comps is not None and "board" in comps:
        record["board"] = comps["board"]
    meta = {"M": M, "components": comps, "holes": holes, "cutouts": cutouts_meta, "bg_file": bg_files[idx]}
    return record, meta


def cut_refiner_crops(cfg, rng, image2x, corners_visible_xy):
    jitter = cfg["refiner_jitter_px"]
    max_n = cfg["synth"]["refiner_max_corners"]
    h2, w2 = image2x.shape
    pts = np.asarray(corners_visible_xy, dtype=np.float64).reshape(-1, 2)
    order = rng.permutation(len(pts))[:max_n] if len(pts) > max_n else np.arange(len(pts))

    out = []
    for i in order:
        p = pts[i]
        for _try in range(3):
            j = rng.uniform(-jitter, jitter, size=2)
            c = np.rint(p + j)
            d = p - c
            cx, cy = int(c[0]), int(c[1])
            if max(abs(d[0]), abs(d[1])) <= 3.9375 and 12 <= cx <= w2 - 12 and 12 <= cy <= h2 - 12:
                out.append({"crop": image2x[cy - 12:cy + 12, cx - 12:cx + 12].copy(), "d": d})
                break
    return out


def make_generic_crop(rng, jitter_px=4.0, roughen=True):
    for _try in range(3):
        d = rng.uniform(-jitter_px, jitter_px, size=2)
        if max(abs(d[0]), abs(d[1])) <= 3.9375:
            break
    d = np.clip(d, -3.9375, 3.9375)
    px, py = 12 + d[0], 12 + d[1]

    k = int(rng.integers(2, 5))
    min_gap = np.radians(35.0)
    for _try in range(30):
        angles = np.sort(rng.uniform(0, 2 * np.pi, size=k))
        gaps = np.diff(np.concatenate([angles, angles[:1] + 2 * np.pi]))
        if gaps.min() >= min_gap and not (k == 2 and abs(gaps[0] - np.pi) < min_gap):
            break

    levels = [int(rng.integers(0, 256))]
    for i in range(1, k):
        neighbours = [levels[i - 1]] + ([levels[0]] if i == k - 1 else [])
        for _try in range(20):
            lvl = int(rng.integers(0, 256))
            if all(abs(lvl - n) >= 40 for n in neighbours):
                break
        levels.append(lvl)

    ss = 16
    n = 24 * ss
    coords = (np.arange(n) - (ss - 1) / 2) / ss
    ys, xs = np.meshgrid(coords, coords, indexing="ij")
    ang = np.mod(np.arctan2(ys - py, xs - px), 2 * np.pi)
    sector = (np.searchsorted(angles, ang, side="right") - 1) % k
    hi = np.zeros((n, n), dtype=np.float32)
    for i in range(k):
        hi[sector == i] = levels[i]
    crop = cv2.resize(hi, (24, 24), interpolation=cv2.INTER_AREA).astype(np.uint8)

    if roughen:
        std = rng.uniform(2.0, 8.0)
        noisy = crop.astype(np.float32) + rng.normal(0, std, crop.shape).astype(np.float32)
        bk = int(rng.choice([0, 3]))
        if bk:
            noisy = cv2.GaussianBlur(noisy, (bk, bk), 0)
        crop = np.clip(noisy, 0, 255).astype(np.uint8)

    return {"crop": crop, "d": d}
