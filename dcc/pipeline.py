"""Stage-3 inference pipeline: peaks -> refine -> IDs -> undistort ->
lattice gate -> recovery -> PnP. Deployment code: pure functions over plain
numpy arrays/torch tensors, no training deps. grid_sample lives HERE only
(read_ids) -- never in the exported nn.Modules of dcc/model.py. `detect()`
never imports dcc.model: model/refiner arrive pre-built and are only ever
called, so every function here stays importable/unit-testable without it.

Coordinate spaces (pixel-centre convention, points are (x, y)):
  "input"   network H x W / class-head H4 x W4, per cfg["input_size"]=[W,H].
  "sensor"  native camera frame; refiner crops are always cut here, never
            upsampled. r = input/sensor resize ratio (r=1 here). input->sensor:
            x_s=(x_i+0.5)/r-0.5; inverse: x_i=(x_s+0.5)*r-0.5.
  "pinhole" undistorted sensor space; undistort/lattice_gate/recover/pnp all
            operate here -- a homography is only a valid board model once
            distortion is removed.

Canonical lattice points are UNITLESS (col+1, row+1) for corner index
i = row*n+col (row=i//n, col=i%n, n the board's per-side inner-corner
count) -- board.py's indexing, and tools/gen_eval_pose.py's K@[r1|r2|t]
lattice construction, both built from canon_lattice below.
"""
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dcc.board import n_corners


def canon_lattice(n):
    """Canonical lattice for an n x n grid of inner corners (unitless
    col+1, row+1; corner i: row = i // n, col = i % n) -- board.py's
    row-major indexing convention, generalised from the default 5x5 board's
    n=4 (16 corners)."""
    return np.array([[(i % n) + 1.0, (i // n) + 1.0] for i in range(n * n)], dtype=np.float64)


_CANON = canon_lattice(4)  # default 5x5 board's 16 inner corners; lattice_gate/recover/pnp's n=4 default


def peaks(hm_sigmoid, tau_hm, top_k=64):
    """3x3 max-pool equality peak decode (CenterNet-style). Plateau ties all
    pass the equality test and come back as separate peaks -- merge_close
    dedups them. (x, y) int coords, descending score, capped at top_k, with
    a deterministic (y, x) tiebreak for reproducible ties."""
    hm = torch.as_tensor(hm_sigmoid, dtype=torch.float32)
    pooled = F.max_pool2d(hm[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
    keep = (hm == pooled) & (hm >= tau_hm)
    ys, xs = torch.nonzero(keep, as_tuple=True)
    scores = hm[ys, xs].detach().cpu().numpy().astype(np.float64)
    xy = np.stack([xs.detach().cpu().numpy(), ys.detach().cpu().numpy()], axis=1).astype(int)
    order = np.lexsort((xy[:, 0], xy[:, 1], -scores))[:top_k]
    return xy[order], scores[order]


def merge_close(xy, scores, radius=2.0):
    """Greedy NMS in descending-score order (deterministic (y, x) tiebreak):
    a point within `radius` of an already-kept higher-score point is
    dropped; an exact plateau tie keeps whichever sorts first."""
    xy = np.asarray(xy)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.lexsort((xy[:, 0], xy[:, 1], -scores))
    kept = []
    for i in order:
        if all(np.hypot(*(xy[i] - xy[j])) > radius for j in kept):
            kept.append(i)
    kept = np.array(kept, dtype=int)
    return xy[kept], scores[kept]


def cut_crops(frame_sensor, peaks_input, r):
    """Sensor-frame 24x24 crops for the refiner. Centre c = rint((x+0.5)/r -
    0.5) per axis, the half-pixel input->sensor map. A peak whose crop would
    cross the sensor border is EXCLUDED (kept_mask False, never
    reflect-padded): it bypasses Stage 2 with its coarse coord."""
    Hs, Ws = frame_sensor.shape
    peaks_input = np.asarray(peaks_input, dtype=np.float64).reshape(-1, 2)
    centre = np.rint((peaks_input + 0.5) / r - 0.5).astype(int)
    cx, cy = centre[:, 0], centre[:, 1]
    kept_mask = (cx - 12 >= 0) & (cx + 12 <= Ws) & (cy - 12 >= 0) & (cy + 12 <= Hs)
    idxs = np.nonzero(kept_mask)[0]
    if len(idxs):
        crops = np.stack([frame_sensor[cy[i] - 12:cy[i] + 12, cx[i] - 12:cx[i] + 12] for i in idxs])
    else:
        crops = np.zeros((0, 24, 24), dtype=frame_sensor.dtype)
    crops = crops.astype(np.float32)[:, None] / 255.0
    return crops, centre[idxs], kept_mask


def soft_argmax(ref_sigmoid):
    """5x5 soft-argmax around the hard argmax: window top-left clamped to
    [0, 59] so the 5x5 stays fully in bounds, renormalised probability-
    weighted centroid. u <- x/col, v <- y/row (targets.render_refiner_target's
    orientation). Caller applies xy_refined = centre + (u* - 31.5)/8."""
    t = torch.as_tensor(ref_sigmoid, dtype=torch.float32).reshape(-1, 64, 64)
    ys, xs = torch.meshgrid(torch.arange(64.0, device=t.device), torch.arange(64.0, device=t.device),
                             indexing="ij")
    out = np.zeros((t.shape[0], 2), dtype=np.float64)
    for i in range(t.shape[0]):
        ay, ax = divmod(int(torch.argmax(t[i])), 64)
        y0, x0 = min(max(ay - 2, 0), 59), min(max(ax - 2, 0), 59)
        w = t[i, y0:y0 + 5, x0:x0 + 5]
        wsum = w.sum().clamp_min(1e-12)
        u = (w * xs[y0:y0 + 5, x0:x0 + 5]).sum() / wsum
        v = (w * ys[y0:y0 + 5, x0:x0 + 5]).sum() / wsum
        out[i] = [u.item(), v.item()]
    return out


def read_ids(cls_sigmoid, xy_input):
    """THE REQUIRED FORM: F.grid_sample(align_corners=False,
    padding_mode='border'), grid_x = 2*(x_in+0.5)/W_in - 1 (W_in = 4*W4 from
    cls_sigmoid's own shape) -- algebraically identical to first mapping into
    cell-space via (x+0.5)/4-0.5 (targets.render_class_targets' convention)
    and then grid_sample's own align_corners=False cell-map; no separate /4
    step needed. align_corners=True is WRONG (diverges near map edges).
    Channel-wise max -> (index, confidence) per corner; confidence is a
    per-channel value, not a posterior -- thresholding is the caller's job."""
    cls = torch.as_tensor(cls_sigmoid, dtype=torch.float32)
    _, H4, W4 = cls.shape
    xy = torch.as_tensor(xy_input, dtype=torch.float32, device=cls.device).reshape(-1, 2)
    gx = 2 * (xy[:, 0] + 0.5) / (4 * W4) - 1
    gy = 2 * (xy[:, 1] + 0.5) / (4 * H4) - 1
    grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(cls[None], grid, mode="bilinear", padding_mode="border",
                             align_corners=False)[0, :, :, 0]  # (n_cls, N)
    conf, idx = sampled.max(dim=0)
    return idx.detach().cpu().numpy().astype(int), conf.detach().cpu().numpy().astype(np.float64)


def undistort(xy_sensor, K, dist):
    """cv2.undistortPoints with P=K so output stays in pixel units (not
    normalised camera coords). Identity when dist is None: the K-normalise /
    K-reproject round trip cancels exactly regardless of K's value, since no
    real distortion correction happens in between."""
    xy_sensor = np.asarray(xy_sensor, dtype=np.float64).reshape(-1, 2)
    if len(xy_sensor) == 0:
        return xy_sensor.copy()
    out = cv2.undistortPoints(xy_sensor.astype(np.float32).reshape(-1, 1, 2), K, dist, P=K)
    return out.reshape(-1, 2).astype(np.float64)


def lattice_gate(xy, idx, conf, tol=3.0, n=4):
    """RANSAC-fit canonical-lattice -> image homography over the ID'd subset
    (canonical points UNITLESS, an n x n grid -- n=4 the default 5x5 board's
    16 corners, see canon_lattice; callers derive n from cfg["board"]).
    Guards in order: < 4 ID'd -> (None, all-False, all-False, 'too_few') --
    a homography's own minimum-correspondence count, independent of board
    size; ID'd canonical points collinear (min singular value of centred
    coords -- homography-invariant, so the always-exactly-known canonical
    side suffices) -> 'collinear'. Else cv2.findHomography(RANSAC, tol):
    exactly 4 ID'd is an exact fit with no redundancy to reject on
    (degenerate='vacuous', H still valid); >=5 demotes dissenters.
    inlier_mask/demoted_mask are full-length (N,), True only at ID'd
    positions."""
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
    if H is None:  # cv2's own degeneracy guard, belt-and-suspenders past the SVD check above
        return None, inlier_mask, demoted_mask, "collinear"
    mask = mask.ravel().astype(bool)
    inlier_mask[idd], demoted_mask[idd] = mask, ~mask
    return H, inlier_mask, demoted_mask, ("vacuous" if len(idd) == 4 else None)


def recover(H, xy_all, idx, conf, tol, n=4):
    """Project all n^2 canonical corners through H (n=4 the default 5x5
    board's 16 corners, see canon_lattice; callers derive n from
    cfg["board"]); an ID-less detection (idx < 0) within tol of a projected
    corner inherits its index (source 'recovered'). Each canonical index is
    claimed at most once -- greedy, nearest-available match in detection
    order -- so two different ID-less detections (or an ID-less one and an
    already-surviving corner) can never collide on the same recovered index.
    Vacuousness is read off the INPUT idx (pre-edit): a fit from exactly 4
    ID'd corners (a homography's own minimum-correspondence count,
    independent of board size) has no internal redundancy, so a recovery
    hit against independently-detected points is the only corroborating
    evidence available -- corroborated is True only then, and only if at
    least one recovery landed."""
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


def pnp(xy_pinhole, idx, K, square_length_m, n=4):
    """cv2.solvePnPGeneric SOLVEPNP_IPPE on the ID'd (idx >= 0) subset,
    object points = canonical lattice (n x n grid, n=4 the default 5x5
    board, see canon_lattice; callers derive n from cfg["board"]) *
    square_length_m (z=0), distCoeffs None (already undistorted). IPPE
    always returns 2 planar solutions; ambiguous = err2/err1 < 1.5 on their
    own reprojection RMS -- unrelated to board rotational symmetry. < 4
    correspondences -> no-pose (IPPE's own minimum, independent of board
    size). Returns (rvec, tvec, rms, ambiguous, n_used, reason); reason is
    None on success."""
    xy_pinhole = np.asarray(xy_pinhole, dtype=np.float64).reshape(-1, 2)
    idx = np.asarray(idx).reshape(-1)
    ok = np.nonzero(idx >= 0)[0]
    n_used = len(ok)
    if n_used < 4:
        return None, None, None, False, n_used, "too_few_correspondences"
    lattice = _CANON if n == 4 else canon_lattice(n)
    obj = (np.hstack([lattice[idx[ok]], np.zeros((n_used, 1))]) * square_length_m).reshape(-1, 1, 3)
    img = xy_pinhole[ok].reshape(-1, 1, 2)
    _, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
    # OpenCV wraps the IPPE solver in a bare catch(...) (solvepnp.cpp): internal
    # failures are swallowed and ZERO solutions returned -- a reachable, defined
    # no-pose outcome (see test_pnp_empty_solver_result), not an exception.
    if len(rvecs) == 0:
        return None, None, None, False, n_used, "pnp_solver_failed"
    errs = np.asarray(errs).ravel()
    order = np.argsort(errs)
    rms = float(errs[order[0]])
    ambiguous = bool(len(order) > 1 and errs[order[1]] / max(rms, 1e-12) < 1.5)
    return rvecs[order[0]], tvecs[order[0]], rms, ambiguous, n_used, None


def detect(frame_sensor, model, refiner, K=None, dist=None, cfg=None):
    """Orchestrates peaks->refine->IDs->undistort->gate->recovery->PnP:
    resizes sensor->input with cv2.INTER_AREA when r != 1 (r=1 skips).
    Without real intrinsics (K=None) the gate/recovery still run (the
    undistort round-trip is a no-op regardless of K when dist=None) but PnP
    is refused (reason 'no_intrinsics') rather than fabricating a metric
    pose from a guessed focal length. A vacuous (exactly-4-ID) fit with no
    corroborating recovery hit is likewise refused ('vacuous_uncorroborated')
    since it carries no internal validation at all."""
    cfg = cfg or {"tau_hm": 0.3, "tau_id": 0.5, "lattice_tol_px": 3.0, "input_size": [1600, 1200]}
    tau_hm, tau_id, tol = cfg["tau_hm"], cfg["tau_id"], cfg["lattice_tol_px"]
    W_in, H_in = cfg["input_size"]
    bcfg = cfg.get("board") or {}
    sqlen = bcfg.get("square_length_m") or 1.0
    n = math.isqrt(n_corners(bcfg))
    model.eval()
    refiner.eval()

    Hs, Ws = frame_sensor.shape
    r = Ws / W_in
    frame_input = frame_sensor if r == 1 else cv2.resize(frame_sensor, (W_in, H_in), interpolation=cv2.INTER_AREA)
    dev = next(model.parameters()).device
    with torch.no_grad():
        inp = (torch.from_numpy(frame_input).float()[None, None] / 255.0).to(dev)
        hm_logits, cls_logits = model(inp)
        hm_sigmoid, cls_sigmoid = hm_logits[0, 0].sigmoid().cpu(), cls_logits[0].sigmoid().cpu()

    xy_pk, p_hm = merge_close(*peaks(hm_sigmoid, tau_hm))
    crops, centres_sensor, kept_mask = cut_crops(frame_sensor, xy_pk, r)
    xy_sensor = np.zeros((len(xy_pk), 2), dtype=np.float64)
    if len(crops):
        with torch.no_grad():
            rdev = next(refiner.parameters()).device
            u_star = soft_argmax(refiner(torch.from_numpy(crops).to(rdev)).sigmoid().cpu())
        xy_sensor[kept_mask] = centres_sensor + (u_star - 31.5) / 8.0
    xy_sensor[~kept_mask] = (xy_pk[~kept_mask] + 0.5) / r - 0.5

    idx_raw, p_id = read_ids(cls_sigmoid, (xy_sensor + 0.5) * r - 0.5)
    idx_thr = np.where(p_id >= tau_id, idx_raw, -1)
    K_eff = K if K is not None else np.array([[max(Ws, Hs), 0, (Ws - 1) / 2],
                                               [0, max(Ws, Hs), (Hs - 1) / 2], [0, 0, 1]], dtype=np.float64)
    xy_pinhole = undistort(xy_sensor, K_eff, dist)

    H, inlier_mask, demoted_mask, degenerate = lattice_gate(xy_pinhole, idx_thr, p_id, tol, n)
    idx_final = idx_thr.copy()
    idx_final[demoted_mask] = -1
    recovered_mask = np.zeros(len(idx_final), dtype=bool)
    rvec = tvec = rms = None
    ambiguous, reason = False, degenerate

    if H is not None:
        idx_final, recovered_mask, corroborated = recover(H, xy_pinhole, idx_final, p_id, tol, n)
        if K is None:
            reason = "no_intrinsics"
        elif degenerate == "vacuous" and not corroborated:
            reason = "vacuous_uncorroborated"
        else:
            rvec, tvec, rms, ambiguous, _, reason = pnp(xy_pinhole, idx_final, K, sqlen, n)

    corners = []
    for i in range(len(idx_final)):
        idx_i = int(idx_final[i]) if idx_final[i] >= 0 else None
        source = None if idx_i is None else ("recovered" if recovered_mask[i] else "head")
        corners.append({"x": float(xy_sensor[i, 0]), "y": float(xy_sensor[i, 1]), "index": idx_i,
                         "source": source, "p_hm": float(p_hm[i]), "p_id": float(p_id[i])})
    return {"rvec": rvec, "tvec": tvec, "rms": rms, "reason": reason, "corners": corners,
            "ambiguous": ambiguous, "demoted": int(demoted_mask.sum()), "recovered": int(recovered_mask.sum())}
