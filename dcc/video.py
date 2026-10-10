"""Video pseudo-labels. For each frame, anchor_from_detection(dcc.pipeline.detect result, r, n, Lens(...))
fits the lattice; harvest_labels(frames, anchors, BoardTemplate(bcfg), n, params) labels the frames between
anchors by ECC tracking; build_record(...) writes corners with visible True / False / None (unknown);
save_harvest(out, name, images, records, meta) stores images.npy + records.json. Driven by
tools/video_bootstrap.py.
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from dcc.board import get_board, render_board
from dcc.pipeline import canon_lattice

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm"}

HARVEST_DEFAULTS = {
    "min_inliers": 8,
    "fit_tol": 1.0,
    "max_rms": 0.75,
    "min_pid": 0.5,
    "sigma_floor": 0.05,
    "anchor_ecc_tol": 1.0,
    "rho_min": 0.3,
    "max_track": 600,
    "ecc_levels": 2,
    "template_size": 320,
    "agree_tol": 1.0,
    "closure_tol": 1.0,
    "track_sigma_floor": 0.1,
    "max_sigma": 1.0,
    "vis_floor": 0.25,
    "vis_rel": 0.75,
}


def crop_43(frame):
    h, w = frame.shape[:2]
    k = min(w // 4, h // 3)
    x0, y0 = (w - 4 * k) // 2, (h - 3 * k) // 2
    return frame[y0:y0 + 3 * k, x0:x0 + 4 * k]


def _mono8(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    assert img.dtype == np.uint8, f"unsupported frame dtype {img.dtype}"
    return img


def iter_frames(src, step=1, max_frames=None):
    p = Path(src)
    n = 0
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        for i, f in enumerate(files):
            if i % step:
                continue
            img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            yield i, crop_43(_mono8(img))
            n += 1
            if max_frames and n >= max_frames:
                return
        return
    assert p.suffix.lower() in VIDEO_EXTS, f"{src}: not a video ({sorted(VIDEO_EXTS)}) or an image directory"
    cap = cv2.VideoCapture(str(p))
    assert cap.isOpened(), f"cannot open {src}"
    i = -1
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                return
            i += 1
            if i % step:
                continue
            yield i, crop_43(_mono8(img))
            n += 1
            if max_frames and n >= max_frames:
                return
    finally:
        cap.release()


def to_input(frame_sensor, W_in, H_in):
    if frame_sensor.shape[1] == W_in:
        return frame_sensor
    return cv2.resize(frame_sensor, (W_in, H_in), interpolation=cv2.INTER_AREA)


def sensor_to_input(xy, r):
    return (np.asarray(xy, dtype=np.float64) + 0.5) * r - 0.5


def project(H, pts):
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return p[:, :2] / p[:, 2:3]


class Lens:
    def __init__(self, K_sensor=None, dist=None, r=1.0, size=(640, 480)):
        self.K = self.dist = self.maps = None
        if K_sensor is not None:
            S = np.array([[r, 0, 0.5 * r - 0.5], [0, r, 0.5 * r - 0.5], [0, 0, 1.0]])
            self.K = S @ np.asarray(K_sensor, dtype=np.float64).reshape(3, 3)
        if self.K is not None and dist is not None and np.any(np.asarray(dist)):
            self.dist = np.asarray(dist, dtype=np.float64).ravel()
            self.maps = cv2.initUndistortRectifyMap(self.K, self.dist, None, self.K, size, cv2.CV_32FC1)

    def undistort_image(self, img):
        return img if self.maps is None else cv2.remap(img, *self.maps, cv2.INTER_LINEAR)

    def undistort_points(self, xy):
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        if self.maps is None or not len(xy):
            return xy
        return cv2.undistortPoints(xy.reshape(-1, 1, 2), self.K, self.dist, P=self.K).reshape(-1, 2)

    def distort_points(self, xy):
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        if self.maps is None or not len(xy):
            return xy
        nrm = (np.linalg.inv(self.K) @ np.hstack([xy, np.ones((len(xy), 1))]).T).T
        out, _ = cv2.projectPoints(nrm / nrm[:, 2:3], np.zeros(3), np.zeros(3), self.K, self.dist)
        return out.reshape(-1, 2)


def degenerate(canon):
    canon = np.asarray(canon, dtype=np.float64)
    m = len(canon)
    if m < 4:
        return True
    most = 2
    for i in range(m):
        for j in range(i + 1, m):
            d = canon[j] - canon[i]
            on = np.abs(d[0] * (canon[:, 1] - canon[i, 1]) - d[1] * (canon[:, 0] - canon[i, 0])) < 1e-9
            most = max(most, int(on.sum()))
    return most >= m - 1


def _hjac(H, P):
    X, Y = P[:, 0], P[:, 1]
    w = H[2, 0] * X + H[2, 1] * Y + 1.0
    u = (H[0, 0] * X + H[0, 1] * Y + H[0, 2]) / w
    v = (H[1, 0] * X + H[1, 1] * Y + H[1, 2]) / w
    J = np.zeros((len(P), 2, 8))
    J[:, 0, 0], J[:, 0, 1], J[:, 0, 2], J[:, 0, 6], J[:, 0, 7] = X / w, Y / w, 1 / w, -u * X / w, -u * Y / w
    J[:, 1, 3], J[:, 1, 4], J[:, 1, 5], J[:, 1, 6], J[:, 1, 7] = X / w, Y / w, 1 / w, -v * X / w, -v * Y / w
    return J.reshape(-1, 8)


@dataclass
class Fit:
    H: np.ndarray
    sigma: np.ndarray
    inliers: np.ndarray
    rms: float
    rho: float = 0.0


def fit_lattice(idx, xy, n, min_inliers=8, fit_tol=1.0, max_rms=0.75, sigma_floor=0.05, **_):
    idx = np.asarray(idx, dtype=int)
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    uniq, counts = np.unique(idx, return_counts=True)
    keep = np.isin(idx, uniq[counts == 1])
    idx, xy = idx[keep], xy[keep]
    lattice = canon_lattice(n)
    if len(idx) < min_inliers or degenerate(lattice[idx]):
        return None
    H, mask = cv2.findHomography(lattice[idx], xy, cv2.RANSAC, fit_tol)
    if H is None:
        return None
    inl = mask.ravel().astype(bool)
    if inl.sum() < min_inliers or degenerate(lattice[idx[inl]]):
        return None
    H, _ = cv2.findHomography(lattice[idx[inl]], xy[inl], 0)
    if H is None:
        return None
    H = H / H[2, 2]
    resid = project(H, lattice[idx[inl]]) - xy[inl]
    m = int(inl.sum())
    rms = float(np.sqrt((resid ** 2).sum() / (2 * m)))
    if rms > max_rms:
        return None
    s2 = max(float((resid ** 2).sum()) / max(2 * m - 8, 1), sigma_floor ** 2)
    J = _hjac(H, lattice[idx[inl]])
    cov = s2 * np.linalg.pinv(J.T @ J)
    Jk = _hjac(H, lattice).reshape(len(lattice), 2, 8)
    sigma = np.sqrt(np.maximum(np.einsum("kij,jl,kil->k", Jk, cov, Jk) / 2.0, 0.0))
    return Fit(H=H, sigma=np.maximum(sigma, sigma_floor), inliers=idx[inl], rms=rms)


def anchor_from_detection(result, r, n, lens, min_pid=0.5, **fit_kw):
    cs = [c for c in result["corners"] if c["source"] == "head" and c["index"] is not None
          and c["p_id"] >= min_pid and np.isfinite(c["sigma_px"])]
    if not cs:
        return None
    idx = np.array([c["index"] for c in cs])
    xy = lens.undistort_points(sensor_to_input([[c["x"], c["y"]] for c in cs], r))
    return fit_lattice(idx, xy, n, **fit_kw)


class BoardTemplate:
    def __init__(self, bcfg=None, size=320):
        nsq = get_board(bcfg)[1]
        img, _ = render_board(480, bcfg)
        self.nsq, self.size = nsq, size
        self.img = cv2.GaussianBlur(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA), (0, 0), 0.6)
        sq = size / nsq
        self.A = np.array([[sq, 0, -0.5], [0, sq, -0.5], [0, 0, 1.0]])
        self.Ainv = np.linalg.inv(self.A)
        self.hi = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.0)
        sq_hi = 480.0 / nsq
        self.A_hi = np.array([[sq_hi, 0, -0.5], [0, sq_hi, -0.5], [0, 0, 1.0]])
        self.outline = np.array([[0, 0], [nsq, 0], [nsq, nsq], [0, nsq]], dtype=np.float64)


def ecc_align(template, frame, H_init, levels=2, gauss=5):
    warp = H_init / H_init[2, 2]
    rho = 0.0
    for lvl in range(levels - 1, -1, -1):
        T, I = template, frame
        for _ in range(lvl):
            T, I = cv2.pyrDown(T), cv2.pyrDown(I)
        D = np.diag([0.5 ** lvl, 0.5 ** lvl, 1.0])
        W = D @ warp @ np.linalg.inv(D)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100 if lvl == 0 else 50, 1e-6 if lvl == 0 else 1e-5)
        try:
            rho, W = cv2.findTransformECC(T, I, W.astype(np.float32), cv2.MOTION_HOMOGRAPHY, crit, None, gauss)
        except cv2.error:
            return None, 0.0
        warp = np.linalg.inv(D) @ W.astype(np.float64) @ D
    if not np.all(np.isfinite(warp)) or abs(warp[2, 2]) < 1e-12:
        return None, 0.0
    return warp / warp[2, 2], float(rho)


def _quad_ok(H, outline, shape, ref_area=None):
    p = np.hstack([outline, np.ones((4, 1))]) @ H.T
    if np.any(p[:, 2] <= 0):
        return False
    q = p[:, :2] / p[:, 2:3]
    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]
    cross = [cross(q[(i + 1) % 4] - q[i], q[(i + 2) % 4] - q[(i + 1) % 4]) for i in range(4)]
    if not (all(c > 0 for c in cross) or all(c < 0 for c in cross)):
        return False
    h, w = shape[:2]
    if q[:, 0].max() < 0 or q[:, 1].max() < 0 or q[:, 0].min() > w or q[:, 1].min() > h:
        return False
    area = abs(cv2.contourArea(q.astype(np.float32)))
    return ref_area is None or (ref_area / 1.5 < area < ref_area * 1.5)


def _area(H, outline):
    return abs(cv2.contourArea(project(H, outline).astype(np.float32)))


def track(frames, start, H0, tmpl, direction, stop=None, rho_min=0.3, max_track=600, ecc_levels=2, **_):
    out, hist, t = {}, [H0], start
    while True:
        t += direction
        if t < 0 or t >= len(frames) or abs(t - start) > max_track:
            break
        pred = hist[-1] if len(hist) < 2 else hist[-1] @ np.linalg.inv(hist[-2]) @ hist[-1]
        pred = pred / pred[2, 2]
        if not _quad_ok(pred, tmpl.outline, frames[t].shape):
            pred = hist[-1]
        Ht, rho = ecc_align(tmpl.img, frames[t], pred @ tmpl.Ainv, ecc_levels)
        if Ht is None or rho < rho_min:
            break
        Hc = Ht @ tmpl.A
        Hc = Hc / Hc[2, 2]
        if not _quad_ok(Hc, tmpl.outline, frames[t].shape, _area(hist[-1], tmpl.outline)):
            break
        out[t] = (Hc, rho)
        hist.append(Hc)
        if stop is not None and t == stop:
            break
    return out


def corner_disagreement(H1, H2, n, shape, margin=4.0):
    lat = canon_lattice(n)
    a, b = project(H1, lat), project(H2, lat)
    h, w = shape[:2]
    inside = ((a[:, 0] > -margin) & (a[:, 0] < w + margin) & (a[:, 1] > -margin) & (a[:, 1] < h + margin))
    return float(np.linalg.norm(a - b, axis=1)[inside].max()) if inside.any() else float("inf")


def corner_visibility(frame, H, tmpl, n, vis_floor=0.25, vis_rel=0.75, **_):
    h, w = frame.shape[:2]
    lat = canon_lattice(n)
    pts = project(H, lat)
    Hr = H @ np.linalg.inv(tmpl.A_hi)
    rend = cv2.warpPerspective(tmpl.hi, Hr, (w, h), flags=cv2.INTER_LINEAR, borderValue=-1.0)
    img = cv2.GaussianBlur(frame.astype(np.float32), (0, 0), 1.0)
    P = pts.reshape(n, n, 2)
    ncc = np.full(n * n, np.nan)
    for k, (x, y) in enumerate(pts):
        rr, cc = divmod(k, n)
        d = np.mean([np.linalg.norm(P[rr + dr, cc + dc] - P[rr, cc]) for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0))
                     if 0 <= rr + dr < n and 0 <= cc + dc < n])
        vals = []
        for frac, lo in ((0.6, 4), (0.3, 3)):
            half = int(np.clip(round(frac * d), lo, 24))
            cx, cy = int(round(x)), int(round(y))
            x0, x1, y0, y1 = cx - half, cx + half + 1, cy - half, cy + half + 1
            if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
                break
            a = img[y0:y1, x0:x1].astype(np.float64).ravel()
            b = rend[y0:y1, x0:x1].astype(np.float64).ravel()
            if np.any(b < 0):
                break
            a, b = a - a.mean(), b - b.mean()
            den = np.sqrt((a * a).sum() * (b * b).sum())
            vals.append((a * b).sum() / den if den > 1e-9 else 0.0)
        if len(vals) == 2:
            ncc[k] = min(vals)
    valid = ncc[np.isfinite(ncc)]
    ref = float(np.percentile(valid, 75)) if len(valid) else 1.0
    thr = max(vis_floor, vis_rel * ref)
    return ncc, np.nan_to_num(ncc, nan=-1.0) >= thr


def _label(H, sigma, source, rho, check):
    return {"H": H, "sigma": np.asarray(sigma, dtype=np.float64), "source": source,
            "rho": float(rho), "check": float(check)}


def _fb_accept(frames, run, origin_t, origin_H, tmpl, direction, n, hp):
    frames_t = sorted(run, key=lambda t: abs(t - origin_t))
    for _ in range(4):
        if not frames_t:
            return [], float("inf")
        t_end = frames_t[-1]
        back = track(frames, t_end, run[t_end][0], tmpl, -direction, stop=origin_t, **hp)
        if origin_t in back:
            err = corner_disagreement(back[origin_t][0], origin_H, n, frames[origin_t].shape)
            if err <= hp["closure_tol"]:
                return frames_t, err
        frames_t = frames_t[: max(1, (2 * len(frames_t)) // 3)] if len(frames_t) > 1 else []
    return [], float("inf")


def harvest_labels(frames, anchors, tmpl, n, hp):
    hp = {**HARVEST_DEFAULTS, **(hp or {})}
    good = {}
    for t, fit in sorted(anchors.items()):
        He, rho = ecc_align(tmpl.img, frames[t], fit.H @ tmpl.Ainv, hp["ecc_levels"])
        if He is None:
            continue
        if corner_disagreement(He @ tmpl.A, fit.H, n, frames[t].shape) <= hp["anchor_ecc_tol"]:
            fit.rho = rho
            good[t] = fit
    labels = {t: _label(f.H, f.sigma, "anchor", f.rho, f.rms) for t, f in good.items()}
    ts = sorted(good)
    N = len(frames)
    stats = {"anchors_in": len(anchors), "anchors": len(good), "closures": [], "fb": []}
    bounds = [(None, ts[0])] + list(zip(ts[:-1], ts[1:])) + [(ts[-1], None)] if ts else []
    for a, b in bounds:
        if a is not None and b is not None and b - a <= 1:
            continue
        fwd = track(frames, a, good[a].H, tmpl, +1, stop=b, **hp) if a is not None else {}
        bwd = track(frames, b, good[b].H, tmpl, -1, stop=a, **hp) if b is not None else {}
        ok_f = ok_b = None
        if b is not None and b in fwd:
            e = corner_disagreement(fwd.pop(b)[0], good[b].H, n, frames[b].shape)
            stats["closures"].append(e)
            ok_f = e if e <= hp["closure_tol"] else None
            fwd = fwd if ok_f is not None else {}
        if a is not None and a in bwd:
            e = corner_disagreement(bwd.pop(a)[0], good[a].H, n, frames[a].shape)
            stats["closures"].append(e)
            ok_b = e if e <= hp["closure_tol"] else None
            bwd = bwd if ok_b is not None else {}
        if fwd and ok_f is None:
            keep, e = _fb_accept(frames, fwd, a, good[a].H, tmpl, +1, n, hp)
            stats["fb"].append(e)
            fwd, ok_f = {t: fwd[t] for t in keep}, (e if keep else None)
        if bwd and ok_b is None:
            keep, e = _fb_accept(frames, bwd, b, good[b].H, tmpl, -1, n, hp)
            stats["fb"].append(e)
            bwd, ok_b = {t: bwd[t] for t in keep}, (e if keep else None)
        lo = 0 if a is None else a + 1
        hi = N - 1 if b is None else b - 1
        for t in range(lo, hi + 1):
            f, g = fwd.get(t), bwd.get(t)
            if f is not None and g is not None:
                d = corner_disagreement(f[0], g[0], n, frames[t].shape)
                if d > hp["agree_tol"]:
                    continue
                lat = canon_lattice(n)
                Hm, _ = cv2.findHomography(lat, (project(f[0], lat) + project(g[0], lat)) / 2.0, 0)
                sig = max(d / 2.0, ok_f / 2.0, ok_b / 2.0, hp["track_sigma_floor"])
                labels[t] = _label(Hm / Hm[2, 2], sig, "tracked", min(f[1], g[1]), d)
            elif f is not None or g is not None:
                Ht, rho = f if f is not None else g
                err = ok_f if f is not None else ok_b
                labels[t] = _label(Ht, max(err / 2.0, hp["track_sigma_floor"]), "tracked", rho, err)
    stats["labelled"] = len(labels)
    return labels, stats


def build_record(t, frame_raw, frame_und, label, tmpl, n, lens, hp):
    hp = {**HARVEST_DEFAULTS, **(hp or {})}
    H = label["H"]
    sigma = np.broadcast_to(label["sigma"], (n * n,))
    lat = canon_lattice(n)
    ncc, observable = corner_visibility(frame_und, H, tmpl, n, **hp)
    pts = lens.distort_points(project(H, lat))
    hgt, wid = frame_raw.shape[:2]
    corners = []
    for k, (x, y) in enumerate(pts):
        jx, jy = int(np.rint(x)), int(np.rint(y))
        if not (0 <= jx < wid and 0 <= jy < hgt):
            vis = False
        elif observable[k] and sigma[k] <= hp["max_sigma"]:
            vis = True
        else:
            vis = None
        corners.append({"x": float(x), "y": float(y), "index": k, "visible": vis, "sigma": float(sigma[k]),
                        "ncc": None if np.isnan(ncc[k]) else float(ncc[k])})
    outline = lens.distort_points(project(H, tmpl.outline))
    P = project(H, lat).reshape(n, n, 2)
    s_px = float(np.mean(np.concatenate([np.linalg.norm(np.diff(P, axis=0), axis=2).ravel(),
                                         np.linalg.norm(np.diff(P, axis=1), axis=2).ravel()])))
    mask = np.zeros((hgt, wid), np.uint8)
    cv2.fillConvexPoly(mask, np.rint(outline).astype(np.int32), 1)
    board_level = float(np.median(frame_raw[mask > 0])) if mask.any() else 0.0
    board_mean = float(np.mean(frame_raw[mask > 0])) if mask.any() else 0.0
    Jc = H[:2, :2] - np.outer(project(H, lat.mean(axis=0, keepdims=True))[0], H[2, :2])
    sv = np.linalg.svd(Jc, compute_uv=False)
    return {"frame": int(t), "source": label["source"], "rho": label["rho"], "check": label["check"],
            "board_present": True, "s_px": s_px, "board_level": board_level, "board_mean": board_mean,
            "anisotropy": float(1.0 - sv[-1] / max(sv[0], 1e-12)),
            "outline": outline.tolist(), "corners": corners}


def frame_noise(img):
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    r = np.abs(cv2.filter2D(img.astype(np.float32), -1, k, borderType=cv2.BORDER_REFLECT))[1:-1, 1:-1].ravel()
    r = np.sort(r)[: max(1, int(0.9 * r.size))]
    return float(np.sqrt(np.pi / 2.0) * r.mean() / 6.0), float(np.median(img))


def noise_model(frames, records=None):
    idx = range(0, len(frames), max(1, len(frames) // 80))
    s, lv = zip(*(frame_noise(frames[i]) for i in idx))
    var, lev = np.square(s), np.asarray(lv)
    coef, *_ = np.linalg.lstsq(np.stack([lev, np.ones_like(lev)], axis=1), var, rcond=None)
    out = {"shot": float(max(coef[0], 0.0)), "read_var": float(max(coef[1], 0.05))}
    if records:
        levels = np.array([r.get("board_mean", r["board_level"]) for r in records])
        bright = levels[levels >= 20]
        out["board_level_p5"] = float(np.percentile(levels, 5))
        out["board_level_bright_median"] = float(np.median(bright)) if len(bright) else None
    return out


def bucket_of(rec):
    s = rec["s_px"]
    so = 0 if s < 16 else 1 if s < 32 else 2 if s < 64 else 3
    lv = rec["board_level"]
    bo = 0 if lv < 20 else 1 if lv < 60 else 2
    return so, bo, int(rec["anisotropy"] > 0.15)


def cap_records(records, cap_per_bucket=None, stride=1):
    recs = records[::max(1, stride)]
    if not cap_per_bucket:
        return recs
    by = {}
    for r in recs:
        by.setdefault(bucket_of(r), []).append(r)
    out = []
    for group in by.values():
        if len(group) > cap_per_bucket:
            pick = np.linspace(0, len(group) - 1, cap_per_bucket).round().astype(int)
            group = [group[i] for i in sorted(set(pick.tolist()))]
        out.extend(group)
    return sorted(out, key=lambda r: r["frame"])


def save_harvest(out_dir, name, images, records, meta):
    d = Path(out_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "images.npy", np.ascontiguousarray(np.stack(images) if images else
                                                   np.zeros((0, 1, 1), np.uint8)))
    (d / "records.json").write_text(json.dumps({"meta": meta, "records": records}, indent=1))
    sha = {f.name: hashlib.sha1(f.read_bytes()).hexdigest() for f in (d / "images.npy", d / "records.json")}
    return d, sha


def overlay(image, rec):
    from dcc.viz import draw_overlay

    col = {True: (0, 200, 0), None: (0, 140, 255)}
    shown = {"corners": [c for c in rec["corners"] if c["visible"] is not False]}
    out = draw_overlay(image, shown, color_fn=lambda c: col[c["visible"]], radius=3, filled=True)
    for c in shown["corners"]:
        if c["visible"] is None:
            cv2.circle(out, (int(round(c["x"])), int(round(c["y"]))), 6, col[None], 1, cv2.LINE_AA)
    cv2.polylines(out, [np.rint(np.asarray(rec["outline"])).astype(np.int32)], True, (200, 120, 0), 1, cv2.LINE_AA)
    tag = f"{rec['frame']} {rec['source'][0].upper()} rho={rec['rho']:.2f}"
    cv2.putText(out, tag, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, tag, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out
