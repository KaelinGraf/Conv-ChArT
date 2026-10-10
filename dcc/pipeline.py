"""Stage-3 inference. detect(frame, model, refiner, K=None, dist=None, cfg=cfg) runs peaks -> refinement ->
IDs -> undistort -> lattice gate -> recovery -> PnP on a 4:3 grayscale uint8 frame and returns
{rvec, tvec, rms, reason, corners, pose_cov, ...}; without K the pose is refused. The stages (peaks,
cut_crops, soft_argmax, read_ids, lattice_gate, recover, pnp) can be called on their own.
"""
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dcc.board import n_corners


def canon_lattice(n):
    return np.array([[(i % n) + 1.0, (i // n) + 1.0] for i in range(n * n)], dtype=np.float64)


_CANON = canon_lattice(4)


def peaks(hm_sigmoid, tau_hm, top_k=64):
    hm = torch.as_tensor(hm_sigmoid, dtype=torch.float32)
    pooled = F.max_pool2d(hm[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
    keep = (hm == pooled) & (hm >= tau_hm)
    ys, xs = torch.nonzero(keep, as_tuple=True)
    scores = hm[ys, xs].detach().cpu().numpy().astype(np.float64)
    xy = np.stack([xs.detach().cpu().numpy(), ys.detach().cpu().numpy()], axis=1).astype(int)
    order = np.lexsort((xy[:, 0], xy[:, 1], -scores))[:top_k]
    return xy[order], scores[order]


def merge_close(xy, scores, radius=2.0):
    xy = np.asarray(xy)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.lexsort((xy[:, 0], xy[:, 1], -scores))
    kept = []
    for i in order:
        if all(np.hypot(*(xy[i] - xy[j])) > radius for j in kept):
            kept.append(i)
    kept = np.array(kept, dtype=int)
    return xy[kept], scores[kept]


def spacing_estimate(xy):
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if len(xy) < 2:
        return None
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))


def crop_extent(s_sensor, alpha=0.5, e_min=16):
    e = 2.0 * np.rint(alpha * np.asarray(s_sensor, dtype=np.float64) / 2.0)
    return np.clip(e, e_min, 24).astype(int)


CROP_UPSAMPLE = cv2.INTER_LANCZOS4


def refined_offset(u_star, E):
    k = np.asarray(E, dtype=np.float64) / 24.0
    if k.ndim:
        k = k.reshape(-1, 1)
    return ((np.asarray(u_star, dtype=np.float64) + 0.5) / 8.0 + 8.0 - 11.5) * k - 0.5


def refiner_support(E):
    return float(refined_offset(0.0, E)), float(refined_offset(63.0, E))


def cut_crops(frame_sensor, peaks_input, r, extent=24):
    Hs, Ws = frame_sensor.shape
    peaks_input = np.asarray(peaks_input, dtype=np.float64).reshape(-1, 2)
    centre = np.rint((peaks_input + 0.5) / r - 0.5).astype(int)
    E = np.broadcast_to(np.asarray(extent, dtype=int).reshape(-1), (len(centre),))
    h, cx, cy = E // 2, centre[:, 0], centre[:, 1]
    kept_mask = (cx - h >= 0) & (cx + h <= Ws) & (cy - h >= 0) & (cy + h <= Hs)
    idxs = np.nonzero(kept_mask)[0]
    crops = np.zeros((len(idxs), 24, 24), dtype=np.float32)
    for k, i in enumerate(idxs):
        c = frame_sensor[cy[i] - h[i]:cy[i] + h[i], cx[i] - h[i]:cx[i] + h[i]].astype(np.float32)
        crops[k] = c if E[i] == 24 else cv2.resize(c, (24, 24), interpolation=CROP_UPSAMPLE)
    return crops[:, None] / 255.0, centre[idxs], kept_mask, E[idxs]


def soft_argmax(ref_sigmoid, return_spread=False):
    t = torch.as_tensor(ref_sigmoid, dtype=torch.float32).reshape(-1, 64, 64)
    ys, xs = torch.meshgrid(torch.arange(64.0, device=t.device), torch.arange(64.0, device=t.device),
                             indexing="ij")
    out = np.zeros((t.shape[0], 2), dtype=np.float64)
    spread = np.zeros(t.shape[0], dtype=np.float64)
    for i in range(t.shape[0]):
        ay, ax = divmod(int(torch.argmax(t[i])), 64)
        y0, x0 = min(max(ay - 2, 0), 59), min(max(ax - 2, 0), 59)
        w = t[i, y0:y0 + 5, x0:x0 + 5]
        wsum = w.sum().clamp_min(1e-12)
        gx, gy = xs[y0:y0 + 5, x0:x0 + 5], ys[y0:y0 + 5, x0:x0 + 5]
        u = (w * gx).sum() / wsum
        v = (w * gy).sum() / wsum
        out[i] = [u.item(), v.item()]
        if return_spread:
            var = ((w * (gx - u) ** 2).sum() + (w * (gy - v) ** 2).sum()) / wsum
            spread[i] = float(torch.sqrt(var.clamp_min(0)) / 8.0)
    return (out, spread) if return_spread else out


def read_ids(cls_sigmoid, xy_input):
    cls = torch.as_tensor(cls_sigmoid, dtype=torch.float32)
    _, H4, W4 = cls.shape
    xy = torch.as_tensor(xy_input, dtype=torch.float32, device=cls.device).reshape(-1, 2)
    gx = 2 * (xy[:, 0] + 0.5) / (4 * W4) - 1
    gy = 2 * (xy[:, 1] + 0.5) / (4 * H4) - 1
    grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(cls[None], grid, mode="bilinear", padding_mode="border",
                             align_corners=False)[0, :, :, 0]
    conf, idx = sampled.max(dim=0)
    return idx.detach().cpu().numpy().astype(int), conf.detach().cpu().numpy().astype(np.float64)


def undistort(xy_sensor, K, dist):
    xy_sensor = np.asarray(xy_sensor, dtype=np.float64).reshape(-1, 2)
    if len(xy_sensor) == 0:
        return xy_sensor.copy()
    out = cv2.undistortPoints(xy_sensor.astype(np.float32).reshape(-1, 1, 2), K, dist, P=K)
    return out.reshape(-1, 2).astype(np.float64)


def lattice_gate(xy, idx, conf, tol=3.0, n=4):
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    idx = np.asarray(idx).reshape(-1)
    lattice = _CANON if n == 4 else canon_lattice(n)
    cnt = len(idx)
    inlier_mask, demoted_mask = np.zeros(cnt, dtype=bool), np.zeros(cnt, dtype=bool)
    idd = np.nonzero(idx >= 0)[0]
    if len(idd) < 4:
        return None, inlier_mask, demoted_mask, "too_few"
    canon = lattice[idx[idd]]
    if np.linalg.svd(canon - canon.mean(axis=0), compute_uv=False)[1] < 1e-6:
        return None, inlier_mask, demoted_mask, "collinear"
    H, mask = cv2.findHomography(canon.astype(np.float32), xy[idd].astype(np.float32),
                                  method=cv2.RANSAC, ransacReprojThreshold=tol)
    if H is None:
        return None, inlier_mask, demoted_mask, "collinear"
    mask = mask.ravel().astype(bool)
    inlier_mask[idd], demoted_mask[idd] = mask, ~mask
    return H, inlier_mask, demoted_mask, ("vacuous" if len(idd) == 4 else None)


def recover(H, xy_all, idx, conf, tol, n=4):
    xy_all = np.asarray(xy_all, dtype=np.float64).reshape(-1, 2)
    idx_in = np.asarray(idx).reshape(-1)
    idx_out = idx_in.copy()
    recovered_mask = np.zeros(len(idx_in), dtype=bool)
    lattice = _CANON if n == 4 else canon_lattice(n)
    proj = np.hstack([lattice, np.ones((n * n, 1))]) @ H.T
    proj = proj[:, :2] / proj[:, 2:3]
    claimed = set(idx_in[idx_in >= 0].tolist())
    for i in np.nonzero(idx_in < 0)[0]:
        d = np.linalg.norm(proj - xy_all[i], axis=1)
        for j in np.argsort(d):
            j = int(j)
            if d[j] > tol:
                break
            if j not in claimed:
                idx_out[i], recovered_mask[i] = j, True
                claimed.add(j)
                break
    corroborated = bool(int(np.sum(idx_in >= 0)) == 4 and recovered_mask.any())
    return idx_out, recovered_mask, corroborated


def pnp(xy_pinhole, idx, K, square_length_m, n=4, sigma_px=None):
    xy_pinhole = np.asarray(xy_pinhole, dtype=np.float64).reshape(-1, 2)
    idx = np.asarray(idx).reshape(-1)
    ok = np.nonzero(idx >= 0)[0]
    n_used = len(ok)
    if n_used < 4:
        return None, None, None, False, n_used, "too_few_correspondences", None
    lattice = _CANON if n == 4 else canon_lattice(n)
    obj = (np.hstack([lattice[idx[ok]], np.zeros((n_used, 1))]) * square_length_m).reshape(-1, 1, 3)
    img = xy_pinhole[ok].reshape(-1, 1, 2)
    _, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
    if len(rvecs) == 0:
        return None, None, None, False, n_used, "pnp_solver_failed", None
    errs = np.asarray(errs).ravel()
    order = np.argsort(errs)
    rms = float(errs[order[0]])
    ambiguous = bool(len(order) > 1 and errs[order[1]] / max(rms, 1e-12) < 1.5)
    rvec, tvec = rvecs[order[0]], tvecs[order[0]]
    cov = pose_covariance(obj, rvec, tvec, K, sigma_px[ok] if sigma_px is not None else None)
    return rvec, tvec, rms, ambiguous, n_used, None, cov


def pose_covariance(obj, rvec, tvec, K, sigma_px=None):
    _, jac = cv2.projectPoints(obj, rvec, tvec, K, None)
    J = np.asarray(jac, dtype=np.float64)[:, :6]
    m = J.shape[0] // 2
    if sigma_px is None:
        JtRiJ = J.T @ J
    else:
        var = np.repeat(np.asarray(sigma_px, dtype=np.float64).reshape(m), 2) ** 2
        var = np.where(np.isfinite(var) & (var > 1e-12), var, np.nan)
        if not np.isfinite(var).all():
            return None
        JtRiJ = (J.T * (1.0 / var)) @ J
    try:
        return np.linalg.inv(JtRiJ)
    except np.linalg.LinAlgError:
        return None


def detect(frame_sensor, model, refiner, K=None, dist=None, cfg=None, id_readout="coarse"):
    cfg = cfg or {"tau_hm": 0.3, "tau_id": 0.5, "lattice_tol_px": 3.0, "input_size": [1600, 1200]}
    tau_hm, tau_id, tol = cfg["tau_hm"], cfg["tau_id"], cfg["lattice_tol_px"]
    W_in, H_in = cfg["input_size"]
    bcfg = cfg.get("board") or {}
    sqlen = bcfg.get("square_length_m") or 1.0
    n = math.isqrt(n_corners(bcfg))
    model.eval()
    if refiner is not None:
        refiner.eval()

    Hs, Ws = frame_sensor.shape
    r = W_in / Ws
    frame_input = frame_sensor if r == 1 else cv2.resize(frame_sensor, (W_in, H_in), interpolation=cv2.INTER_AREA)
    dev = next(model.parameters()).device
    with torch.no_grad():
        inp = (torch.from_numpy(frame_input).float()[None, None] / 255.0).to(dev)
        hm_logits, cls_logits = model(inp)
        hm_sigmoid, cls_sigmoid = hm_logits[0, 0].sigmoid().cpu(), cls_logits[0].sigmoid().cpu()

    xy_pk, p_hm = merge_close(*peaks(hm_sigmoid, tau_hm))
    alpha = cfg.get("refiner_crop_alpha", 0.0)
    s_est = spacing_estimate(xy_pk) if alpha > 0 else None
    E = 24 if s_est is None else crop_extent(s_est * r, alpha, cfg.get("refiner_crop_min_px", 12))
    crops, centres_sensor, kept_mask, E_kept = cut_crops(frame_sensor, xy_pk, r, E)
    xy_sensor = np.zeros((len(xy_pk), 2), dtype=np.float64)
    sigma_px = np.full(len(xy_pk), np.nan, dtype=np.float64)
    if refiner is None:
        kept_mask[:] = False
    if refiner is not None and len(crops):
        idxs = np.nonzero(kept_mask)[0]
        with torch.no_grad():
            rdev = next(refiner.parameters()).device
            rmap = refiner(torch.from_numpy(crops).to(rdev)).sigmoid().cpu()
        u_star, u_spread = soft_argmax(rmap, return_spread=True)
        peak = rmap.reshape(rmap.shape[0], -1).amax(dim=1).numpy()
        ok = peak >= cfg.get("refine_min_peak", 0.3)
        sel = idxs[ok]
        xy_sensor[sel] = centres_sensor[ok] + refined_offset(u_star[ok], E_kept[ok])
        sigma_px[sel] = u_spread[ok] * (E_kept[ok] / 24.0)
        kept_mask[idxs[~ok]] = False
    xy_coarse = (xy_pk + 0.5) / r - 0.5
    xy_sensor[~kept_mask] = xy_coarse[~kept_mask]
    xy_id = xy_coarse if id_readout == "coarse" else xy_sensor
    idx_raw, p_id = read_ids(cls_sigmoid, (xy_id + 0.5) * r - 0.5)
    idx_thr = np.where(p_id >= tau_id, idx_raw, -1)
    K_eff = K if K is not None else np.array([[max(Ws, Hs), 0, (Ws - 1) / 2],
                                               [0, max(Ws, Hs), (Hs - 1) / 2], [0, 0, 1]], dtype=np.float64)
    xy_pinhole = undistort(xy_sensor, K_eff, dist)
    xy_pin_id = xy_pinhole if (refiner is None or id_readout != "coarse") \
        else undistort(xy_coarse, K_eff, dist)

    H, inlier_mask, demoted_mask, degenerate = lattice_gate(xy_pin_id, idx_thr, p_id, tol, n)
    idx_final = idx_thr.copy()
    idx_final[demoted_mask] = -1
    recovered_mask = np.zeros(len(idx_final), dtype=bool)
    rvec = tvec = rms = pose_cov = None
    ambiguous, reason = False, degenerate

    if H is not None:
        idx_final, recovered_mask, corroborated = recover(H, xy_pin_id, idx_final, p_id, tol, n)
        if K is None:
            reason = "no_intrinsics"
        elif degenerate == "vacuous" and not corroborated:
            reason = "vacuous_uncorroborated"
        else:
            rvec, tvec, rms, ambiguous, _, reason, pose_cov = pnp(
                xy_pinhole, idx_final, K, sqlen, n, sigma_px=sigma_px)

    corners = []
    for i in range(len(idx_final)):
        idx_i = int(idx_final[i]) if idx_final[i] >= 0 else None
        source = None if idx_i is None else ("recovered" if recovered_mask[i] else "head")
        corners.append({"x": float(xy_sensor[i, 0]), "y": float(xy_sensor[i, 1]), "index": idx_i,
                         "x_coarse": float((xy_pk[i, 0] + 0.5) / r - 0.5),
                         "y_coarse": float((xy_pk[i, 1] + 0.5) / r - 0.5),
                         "source": source, "p_hm": float(p_hm[i]), "p_id": float(p_id[i]),
                         "sigma_px": float(sigma_px[i])})
    return {"rvec": rvec, "tvec": tvec, "rms": rms, "reason": reason, "corners": corners,
            "pose_cov": None if pose_cov is None else pose_cov.tolist(),
            "ambiguous": ambiguous, "demoted": int(demoted_mask.sum()), "recovered": int(recovered_mask.sum())}
