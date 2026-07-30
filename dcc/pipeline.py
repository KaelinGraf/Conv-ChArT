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


def spacing_estimate(xy):
    """Corner spacing, in the coordinate frame of `xy`, from the DETECTED PEAKS
    ALONE -- no board pose, no PnP, nothing the refiner has touched. The inner
    corners form a lattice of unit spacing, so under any homography the local
    nearest-neighbour distance IS the local spacing; the median over all peaks is
    robust both to the perspective spread across the board and to stray non-board
    peaks. Measured 2026-07-29 over s = 12-128 input px: 0.3-5.2% bias, p90
    relative error 13-18%. None with fewer than two peaks."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if len(xy) < 2:
        return None
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))


def crop_extent(s_sensor, alpha=0.5, e_min=16):
    """Refiner crop extent E (even, sensor px) for a corner spacing of s_sensor;
    the crop is then resized to the 24x24 the network expects, so E < 24 is an
    UPSAMPLE and the corner's junction fills the same fraction of the input at
    every scale. That scale normalisation is the point (Kaelin, 2026-07-29):
    masking a narrow view instead would truncate the junction's edges at E px
    while the network's filters stay tuned to a 24 px field, so the apparent
    geometry would still change with scale.

    CAPPED AT 24 -- only ever narrowed, never widened, so the network never
    downsamples. Widening was measured and rejected: tools/refiner_crop_scale.py
    2026-07-29 (current weights) gives median error 0.101 px at E=24 vs 0.256 at
    32, 0.474 at 40, 0.724 at 48 -- extra context is not merely useless to a
    sub-pixel regressor, it is actively harmful. Adaptation therefore engages
    only below s_sensor = 2*24/alpha; above it E = 24 exactly and the path is
    bit-identical to the fixed-crop one."""
    e = 2.0 * np.rint(alpha * np.asarray(s_sensor, dtype=np.float64) / 2.0)
    return np.clip(e, e_min, 24).astype(int)


#: Upsample kernel for a sub-24 px crop. Lanczos4, not bilinear, and the reason is
#: exact rather than aesthetic (Kaelin, 2026-07-29: the upsample must be
#: artifact-free): UPSAMPLING adds samples, so the true signal provably has no
#: content above the source Nyquist and ideal sinc reconstruction is the EXACT
#: answer -- windowed sinc is its best practical approximation. Bilinear
#: attenuates precisely the high-frequency edge content a sub-pixel estimate is
#: read from, and carries a phase-dependent interpolation bias that is the
#: classic source of sub-pixel error. Ringing arises only where the source
#: violated band-limiting; the synth prefilter (synth.prefilter) and the sensor's
#: own optics keep that small, and it is deterministic, so the retrained network
#: sees it as a fixed property of its input rather than as noise. PixelShuffle is
#: NOT an option at this seam: it rearranges C*r^2 learned channels into space and
#: this is a 1-channel uint8 sensor crop with nothing to shuffle.
CROP_UPSAMPLE = cv2.INTER_LANCZOS4


def refined_offset(u_star, E):
    """64-grid soft-argmax coordinate -> sub-pixel offset from the crop's centre
    PIXEL, in sensor px. The crop is cut at extent E and resized to the 24x24 the
    network expects, so the answer arrives in the NETWORK's frame and has to be
    mapped back out of it: 64-grid -> 24-frame (the grid covers the central 8 px
    at 8x) -> E-frame (half-pixel resize convention) -> offset from centre.
    Reduces EXACTLY to the historical (u* - 31.5)/8 at E = 24, the no-resize
    case, so that convention is a special case here rather than a second path."""
    k = np.asarray(E, dtype=np.float64) / 24.0
    if k.ndim:
        k = k.reshape(-1, 1)
    return ((np.asarray(u_star, dtype=np.float64) + 0.5) / 8.0 + 8.0 - 11.5) * k - 0.5


def refiner_support(E):
    """(d_lo, d_hi): the offsets an extent-E crop can represent -- those whose
    target lands inside the 64-grid. +-3.9375 px at E=24 and SHRINKING WITH E,
    mildly asymmetric below it because the centre pixel sits half a pixel right
    of the crop's geometric centre. This is the real cost of a narrower view --
    capture radius, traded for a neighbour-free field -- so the training jitter
    must be drawn against THESE bounds, never the fixed +-3.9375, and E_min is
    floored by the coarse peak's p95 error rather than chosen for tidiness."""
    return float(refined_offset(0.0, E)), float(refined_offset(63.0, E))


def cut_crops(frame_sensor, peaks_input, r, extent=24):
    """Sensor-frame refiner crops, cut at extent E and RESIZED to the 24x24 the
    network expects. Centre c = rint((x+0.5)/r - 0.5) per axis, the half-pixel
    input->sensor map. `extent` is a scalar or one even E per peak (crop_extent);
    E < 24 is upsampled with CROP_UPSAMPLE, E == 24 is returned untouched so the
    fixed-crop path stays bit-identical. A peak whose crop would cross the sensor
    border is EXCLUDED (kept_mask False, never reflect-padded): it bypasses
    Stage 2 with its coarse coord. Also returns the per-kept-peak E, which the
    caller needs to decode the answer (refined_offset)."""
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
    """5x5 soft-argmax around the hard argmax: window top-left clamped to
    [0, 59] so the 5x5 stays fully in bounds, renormalised probability-
    weighted centroid. u <- x/col, v <- y/row (targets.render_refiner_target's
    orientation). Caller applies xy_refined = centre + (u* - 31.5)/8.

    return_spread=True additionally returns the probability-weighted standard
    deviation of that same 5x5 window, converted to SENSOR PIXELS (/8, the grid's
    own scale). This is the refiner's own statement of how sharp its peak is, and
    it is the network's only per-corner localisation-uncertainty signal -- the
    centroid alone cannot distinguish a confident sub-pixel estimate from a flat,
    ambiguous one. Kept opt-in so existing single-value callers are unaffected."""
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
            spread[i] = float(torch.sqrt(var.clamp_min(0)) / 8.0)   # grid units -> sensor px
    return (out, spread) if return_spread else out


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


def pnp(xy_pinhole, idx, K, square_length_m, n=4, sigma_px=None):
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
        return None, None, None, False, n_used, "too_few_correspondences", None
    lattice = _CANON if n == 4 else canon_lattice(n)
    obj = (np.hstack([lattice[idx[ok]], np.zeros((n_used, 1))]) * square_length_m).reshape(-1, 1, 3)
    img = xy_pinhole[ok].reshape(-1, 1, 2)
    _, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
    # OpenCV wraps the IPPE solver in a bare catch(...) (solvepnp.cpp): internal
    # failures are swallowed and ZERO solutions returned -- a reachable, defined
    # no-pose outcome (see test_pnp_empty_solver_result), not an exception.
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
    """First-order (Gauss-Newton / inverse-Fisher) pose covariance at the PnP
    optimum: SIGMA = (J^T R^-1 J)^-1, with J = d(reprojection)/d(rvec, tvec) taken
    straight from cv2.projectPoints' own analytic Jacobian -- so the pose
    uncertainty follows from the GEOMETRY of the correspondences rather than from
    anything fitted. Returns a 6x6 in (rvec, tvec) order, or None if singular.

    R is the per-corner measurement covariance, built isotropically per corner
    from `sigma_px` (in PINHOLE px, matching `obj`/`img`'s space). sigma_px=None
    falls back to the identity, which yields SIGMA in units of "per 1 px of
    measurement noise" -- scale it by your own sigma^2 to interpret.

    *** MEASURED OVER-CONFIDENT BY ~6x. NOT VALIDATED FOR FILTER USE. ***
    NEES on 225 unambiguous poses of eval_pose_rev5 = 36.4 against a chi-square_6
    expectation of 6.0, even with per-corner sigma calibrated so that the CORNER
    NEES was correct (2.08 vs 2.0). The residual is the independence assumption
    below failing, not a scale error. Reporting this as a filter covariance without
    an inflation factor would be wrong. See
    paper/results_rev5/10_uncertainty/NOTES_what_didnt_work.md.

    THREE FAILURE MODES THIS CANNOT SEE, all of which make it OVER-confident, and
    which is why it must be validated by a NEES/chi-square test rather than trusted:
      * it is a LOCAL linearisation, so it says nothing about the IPPE planar
        ambiguity -- when `ambiguous` is True the true posterior is BIMODAL and no
        single Gaussian describes it;
      * it assumes the residuals are zero-mean Gaussian with covariance R, so a
        mis-IDed or outlier corner is not modelled at all;
      * it is only as well-scaled as R is: a sigma miscalibrated by 4x makes SIGMA
        wrong by 16x, which is why per-corner sigma has to be RMS-calibrated first.
    """
    _, jac = cv2.projectPoints(obj, rvec, tvec, K, None)
    J = np.asarray(jac, dtype=np.float64)[:, :6]           # d(u,v) / d(rvec, tvec)
    m = J.shape[0] // 2
    if sigma_px is None:
        JtRiJ = J.T @ J
    else:
        var = np.repeat(np.asarray(sigma_px, dtype=np.float64).reshape(m), 2) ** 2
        var = np.where(np.isfinite(var) & (var > 1e-12), var, np.nan)
        if not np.isfinite(var).all():
            return None                                     # never silently treat unknown as certain
        JtRiJ = (J.T * (1.0 / var)) @ J
    try:
        return np.linalg.inv(JtRiJ)
    except np.linalg.LinAlgError:
        return None


def detect(frame_sensor, model, refiner, K=None, dist=None, cfg=None, id_readout="coarse"):
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
    if refiner is not None:
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
    # SCALE-ADAPTIVE APERTURE: the coarse peaks already carry the board's scale
    # (the inner corners are a unit lattice, so the median NN distance IS the
    # spacing), so the refiner's field of view can be matched to the board live,
    # from Stage 1's own output, with no extra forward pass and no second model.
    # It only ever NARROWS (see crop_extent): at ordinary and large scales E = 24
    # and this is a no-op; below s_sensor = 48 it stops the window swallowing the
    # neighbouring corner, the refiner's measured far-range failure.
    # OFF BY DEFAULT (alpha <= 0): an adaptive crop is only correct against a refiner
    # TRAINED on adaptive crops. Enabling it under fixed-crop weights feeds the
    # network an upsampled input it has never seen and silently degrades the very
    # tail it is meant to fix -- so the config key, not the code default, turns it on.
    alpha = cfg.get("refiner_crop_alpha", 0.0)
    s_est = spacing_estimate(xy_pk) if alpha > 0 else None
    E = 24 if s_est is None else crop_extent(s_est * r, alpha, cfg.get("refiner_crop_min_px", 12))
    crops, centres_sensor, kept_mask, E_kept = cut_crops(frame_sensor, xy_pk, r, E)
    xy_sensor = np.zeros((len(xy_pk), 2), dtype=np.float64)
    # Per-corner localisation uncertainty, in sensor px: the refiner's own peak
    # spread. NaN where Stage 2 did not run (refiner absent, or a border-bypassed
    # crop): "unknown" and "confident" must never be conflated -- a caller weighting
    # by 1/sigma would silently treat an unmeasured corner as a perfect one.
    #
    # *** THIS IS AN ORDERING SIGNAL, NOT A VARIANCE. DO NOT USE AS A FILTER'S R. ***
    # Measured 2026-07-28: it ranks corners by error well (rank corr ~0.63, 12x
    # median separation across deciles) but its SCALE is wrong -- it spans ~1.9x
    # while actual error spans ~12x, over-stating error on easy corners by ~4x and
    # under-stating the hard tail by ~4.5x. A calibration was fitted and then
    # ABANDONED; see paper/results_rev5/10_uncertainty/NOTES_what_didnt_work.md
    # before touching this again.
    sigma_px = np.full(len(xy_pk), np.nan, dtype=np.float64)
    if refiner is None:
        kept_mask[:] = False   # coarse-only arm (M-01 refiner ablation): every peak
                                # takes the same path border-bypassed crops already use
    if refiner is not None and len(crops):
        idxs = np.nonzero(kept_mask)[0]
        with torch.no_grad():
            rdev = next(refiner.parameters()).device
            rmap = refiner(torch.from_numpy(crops).to(rdev)).sigmoid().cpu()
        u_star, u_spread = soft_argmax(rmap, return_spread=True)
        # REFINEMENT GUARD (Kaelin, 2026-07-29: "add the guard to not refine on crops
        # with no valid target"). The refiner's own peak height is its statement that
        # it found a junction at all; a flat map means the crop carries no sub-pixel
        # information -- blown out, crushed, or featureless -- and the sub-pixel answer
        # it returns anyway is noise. Such a corner keeps its coarse peak.
        #
        # NOTE this REVERSES the "refiner guard is abandoned" pin in CLAUDE.md, on
        # Kaelin's explicit instruction and on new evidence. That pin retired a guard
        # for a tail caused by a TRAINING-DATA GAP, which retraining correctly fixed.
        # This tail is different in kind: measured 2026-07-29 (n=6421, guard_sweep.py)
        # the refiner is WORSE than the coarse peak on 16.3% of crops, and those crops
        # are 55% featureless (std < 15 vs 20% of the rest). No amount of training
        # recovers a sub-pixel position from a crop that does not contain the corner.
        #
        # Threshold swept, not guessed -- peak < 0.3 fires on 13.8% of crops, is right
        # 69.5% of the time, and takes p95 from 3.011 px to 0.761 px while the median
        # moves only 0.1014 -> 0.1026. That is better than coarse-only on BOTH axes
        # (coarse: median 0.4244, p95 0.8449). Image-statistic guards were tested and
        # all lost to it; sigma_px never fired at any threshold.
        peak = rmap.reshape(rmap.shape[0], -1).amax(dim=1).numpy()
        ok = peak >= cfg.get("refine_min_peak", 0.3)
        sel = idxs[ok]
        xy_sensor[sel] = centres_sensor[ok] + refined_offset(u_star[ok], E_kept[ok])
        # spread is grid units / 8; one grid unit is E/24 as many SENSOR px under a
        # resized crop, so the scale rides along with the offset it describes
        sigma_px[sel] = u_spread[ok] * (E_kept[ok] / 24.0)
        kept_mask[idxs[~ok]] = False       # guarded -> falls through to the coarse peak
    xy_coarse = (xy_pk + 0.5) / r - 0.5
    xy_sensor[~kept_mask] = xy_coarse[~kept_mask]

    # id_readout: WHERE the class map is sampled. Default "coarse" = the
    # pre-refinement peak; "refined" = the sub-pixel position (the original
    # behaviour). These were only ever coupled by accident -- the class map is at
    # H/4, so ONE CELL IS 4 INPUT PIXELS, and the coarse peak (0.41 px median
    # error) and the refined one (0.08 px) both land well inside the same cell.
    # Sub-pixel accuracy carries no information for identity; refinement exists
    # for geometry. Measured (2026-07-28, tools/refiner_id_effect.py, n=3520):
    # reading at the refined position cost -1.82 pp of ID accuracy, 73% of it
    # from corners that still MATCHED but whose p_id jittered under tau_id -- a
    # hard threshold converting symmetric noise into one-sided loss. Taking
    # max(p_refined, p_coarse) recovered only part of it (-1.39 pp) because a
    # more CONFIDENT read is not necessarily a CORRECT one.
    xy_id = xy_coarse if id_readout == "coarse" else xy_sensor
    idx_raw, p_id = read_ids(cls_sigmoid, (xy_id + 0.5) * r - 0.5)
    idx_thr = np.where(p_id >= tau_id, idx_raw, -1)
    K_eff = K if K is not None else np.array([[max(Ws, Hs), 0, (Ws - 1) / 2],
                                               [0, max(Ws, Hs), (Hs - 1) / 2], [0, 0, 1]], dtype=np.float64)
    xy_pinhole = undistort(xy_sensor, K_eff, dist)
    # THE REFINER IS A LOCALISATION LEVER, NOT AN IDENTITY ONE (Kaelin, 2026-07-29:
    # "the coarse ID is what SHOULD be taken as the final id ... the identification
    # should be EXACTLY the same"). Reading the class map coarsely was only half of
    # that: lattice_gate DEMOTES and recover ASSIGNS on the strength of POSITIONS, so
    # feeding them refined coords let the refiner rewrite identity through the back
    # door. Measured 2026-07-28 before this change: 1.34% of per-detection IDs differed
    # between the arms and ID accuracy on a COMMON matching was 98.209% refined vs
    # 99.424% coarse -- the refiner's scale-floor tail corrupting the homography fit,
    # which then demoted corners that were correctly identified. Running the identity
    # chain on xy_pin_id makes idx_final a function of {xy_pk, cls_sigmoid, K, dist}
    # alone, none of which the refiner touches, so the two arms are BIT-IDENTICAL in
    # identity by construction. The fit here is used ONLY for ID recovery; pnp below
    # re-solves from scratch on the REFINED points and keeps the full sub-pixel gain.
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
            # sigma_px is per-corner in SENSOR px; xy_pinhole is undistorted sensor
            # space, so the two share a scale and R needs no conversion here.
            rvec, tvec, rms, ambiguous, _, reason, pose_cov = pnp(
                xy_pinhole, idx_final, K, sqlen, n, sigma_px=sigma_px)

    corners = []
    for i in range(len(idx_final)):
        idx_i = int(idx_final[i]) if idx_final[i] >= 0 else None
        source = None if idx_i is None else ("recovered" if recovered_mask[i] else "head")
        # x_coarse/y_coarse: the pre-refinement peak, always reported. Costs nothing
        # (xy_pk is already in hand) and lets a caller measure the refiner's per-corner
        # gain -- refined vs coarse error for the SAME detection -- from one forward
        # pass, which is the only way to tell whether a refinement helped or hurt.
        corners.append({"x": float(xy_sensor[i, 0]), "y": float(xy_sensor[i, 1]), "index": idx_i,
                         "x_coarse": float((xy_pk[i, 0] + 0.5) / r - 0.5),
                         "y_coarse": float((xy_pk[i, 1] + 0.5) / r - 0.5),
                         "source": source, "p_hm": float(p_hm[i]), "p_id": float(p_id[i]),
                         "sigma_px": float(sigma_px[i])})
    return {"rvec": rvec, "tvec": tvec, "rms": rms, "reason": reason, "corners": corners,
            "pose_cov": None if pose_cov is None else pose_cov.tolist(),
            "ambiguous": ambiguous, "demoted": int(demoted_mask.sum()), "recovered": int(recovered_mask.sum())}
