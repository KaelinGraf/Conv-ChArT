"""Conv-ChArT training-target renderers — pure numpy, pure functions of the
label record, never persisted to disk (sigma stays tunable without regen).

Shared convention: continuous positions are (x, y) float64; targets combine
per-source Gaussians by MAX (never sum, per CornerNet), then force the exact
containing pixel/cell to 1.0 so the Y=1 branch of the focal loss is reachable.
"""
import numpy as np


def _splat_max(canvas: np.ndarray, cx: float, cy: float, sigma: float) -> None:
    """Max-combine an unnormalised Gaussian centred at (cx, cy) into canvas
    (row-major [y][x]), window +/- 3 sigma, floor/ceil, clipped to canvas."""
    h, w = canvas.shape
    x0, x1 = max(0, int(np.floor(cx - 3 * sigma))), min(w - 1, int(np.ceil(cx + 3 * sigma)))
    y0, y1 = max(0, int(np.floor(cy - 3 * sigma))), min(h - 1, int(np.ceil(cy + 3 * sigma)))
    if x0 > x1 or y0 > y1:
        return
    gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
    g = np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * sigma ** 2))
    canvas[y0:y1 + 1, x0:x1 + 1] = np.maximum(canvas[y0:y1 + 1, x0:x1 + 1], g)


def render_heatmap(pts: np.ndarray, vis: np.ndarray, size_wh: tuple[int, int],
                    sigma: float = 2.0) -> np.ndarray:
    """Detector heatmap target, shape (H, W)."""
    w, h = size_wh
    hm = np.zeros((h, w), dtype=np.float32)
    for (px, py), v in zip(pts, vis):
        if v:
            _splat_max(hm, px, py, sigma)
    for (px, py), v in zip(pts, vis):
        if v:
            jx, jy = int(np.rint(px)), int(np.rint(py))
            if 0 <= jx < w and 0 <= jy < h:
                hm[jy, jx] = 1.0
    return hm


def render_class_targets(pts: np.ndarray, vis: np.ndarray, idx: np.ndarray,
                          size_wh: tuple[int, int], sigma: float = 1.0) -> np.ndarray:
    """Per-corner-index class target, shape (16, H//4, W//4). Cell j aggregates input
    pixels 4j..4j+3; cell-space position xc = (x+0.5)/4 - 0.5 keeps the
    pixel-centre convention (pixel x=1.5, the centre of cell 0, maps to 0.0)."""
    w, h = size_wh
    if w % 4 or h % 4:
        raise ValueError(f"size_wh must be divisible by 4, got {size_wh}")
    ct = np.zeros((16, h // 4, w // 4), dtype=np.float32)
    for (px, py), v, k in zip(pts, vis, idx):
        if v:
            _splat_max(ct[k], (px + 0.5) / 4 - 0.5, (py + 0.5) / 4 - 0.5, sigma)
    for (px, py), v, k in zip(pts, vis, idx):
        if v:
            jy, jx = int(np.floor((py + 0.5) / 4)), int(np.floor((px + 0.5) / 4))
            if 0 <= jy < ct.shape[1] and 0 <= jx < ct.shape[2]:
                ct[k, jy, jx] = 1.0
    return ct


def render_refiner_target(d: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Refiner target, shape (64, 64) at 8x resolution over the central
    8x8 px. d = (dx, dy) sub-pixel offset; u (col) <- x, v (row) <- y."""
    dx, dy = d
    if max(abs(dx), abs(dy)) > 3.9375:
        raise ValueError(f"offset {tuple(d)} exceeds the 64x64 @ 8x support (max 3.9375 px)")
    rt = np.zeros((64, 64), dtype=np.float32)
    u_star, v_star = 31.5 + 8 * dx, 31.5 + 8 * dy
    _splat_max(rt, u_star, v_star, sigma)
    rt[int(np.rint(v_star)), int(np.rint(u_star))] = 1.0
    return rt
