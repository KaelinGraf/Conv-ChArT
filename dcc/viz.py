"""Acceptance visuals: one canonical overlay used by tools/view.py and
tools/audit.py. Pure cv2 (no matplotlib) so callers can tile/save sheets
headless without a second plotting dependency.
"""
import cv2
import numpy as np

_GREEN, _RED, _ORANGE = (0, 200, 0), (0, 0, 220), (0, 140, 255)


def draw_overlay(image_u8_gray: np.ndarray, record: dict, meta: dict | None = None,
                  draw_indices: bool = True, filled: bool = True, radius: int = 3,
                  color_fn=None) -> np.ndarray:
    """Corner dots (visible green / invisible red, or per color_fn) + index
    text + hole rects (orange, from meta) over the image. `image_u8_gray` may
    already be BGR (ndim==3, copied not mutated) as well as grayscale --
    composable across two calls onto the same canvas, e.g. GT then
    predictions in one cell (see tools/factor_sweep.py's filmstrip grid).
    filled/radius pick the marker style (hollow+larger reads as GT,
    filled+smaller as a prediction); color_fn(corner_dict) -> BGR triple
    overrides the default visible/invisible green/red when given, for
    records (like raw predictions) with no "visible" field of their own.
    Returns BGR uint8."""
    out = image_u8_gray.copy() if image_u8_gray.ndim == 3 else cv2.cvtColor(image_u8_gray, cv2.COLOR_GRAY2BGR)
    for x0, y0, w, h in (meta or {}).get("holes", []):
        cv2.rectangle(out, (int(round(x0)), int(round(y0))),
                       (int(round(x0 + w)), int(round(y0 + h))), _ORANGE, 1)
    for c in record["corners"]:
        pt = (int(round(c["x"])), int(round(c["y"])))
        color = color_fn(c) if color_fn else (_GREEN if c["visible"] else _RED)
        cv2.circle(out, pt, radius, color, -1 if filled else 1, lineType=cv2.LINE_AA)
        if draw_indices:
            cv2.putText(out, str(c["index"]), (pt[0] + 5, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return out


def tile(imgs: list, cols: int = 5) -> np.ndarray:
    """Grid-tile same-size BGR images, padding the last row with black."""
    h, w = imgs[0].shape[:2]
    rows = [imgs[i:i + cols] for i in range(0, len(imgs), cols)]
    rows[-1] = rows[-1] + [np.zeros((h, w, 3), np.uint8)] * (cols - len(rows[-1]))
    return cv2.vconcat([cv2.hconcat(r) for r in rows])


def heatmap_overlay(image: np.ndarray, hm: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """JET colormap of hm (any float range, own shape) alpha-blended over
    image (grayscale or already-BGR). Returns BGR uint8."""
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
    span = hm.max() - hm.min()
    norm = (hm - hm.min()) / span if span > 0 else np.zeros_like(hm, dtype=np.float32)
    cmap = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    if cmap.shape[:2] != base.shape[:2]:
        cmap = cv2.resize(cmap, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(base, 1 - alpha, cmap, alpha, 0)


def surface3d(ax, hm: np.ndarray, stride: int = 2, zlim: tuple = (0, 1.05),
              elev: float = 45, azim: float = -60, cmap: str = "viridis"):
    """3D surface of heatmap `hm` (H, W) onto a provided mpl 3D axis (built
    with projection='3d'). Y-axis inverted so row 0 (image top) matches
    image orientation. `stride` subsamples both axes -- 2 is safe for the
    sigma_hm=2px Gaussians used project-wide at full-frame scale; pass
    stride=1 for small zoom windows where individual splats matter. One
    canonical 3D-surface path for both ground-truth heatmaps (this module's
    callers) and predicted heatmaps (tools/introspect.py).
    """
    h, w = hm.shape
    xs, ys = np.arange(0, w, stride), np.arange(0, h, stride)
    Xs, Ys = np.meshgrid(xs, ys)
    ax.plot_surface(Xs, Ys, hm[::stride, ::stride], cmap=cmap, linewidth=0,
                     antialiased=(stride == 1), rcount=len(ys), ccount=len(xs))
    ax.set_zlim(*zlim)
    ax.view_init(elev=elev, azim=azim)
    ax.invert_yaxis()
    return ax


def overlay_alpha(image_gray: np.ndarray, heat: np.ndarray, cmap: str = "magma", alpha: float = 0.55,
                   vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    """mpl-figure counterpart to heatmap_overlay: normalises `heat` to [0,1]
    (via vmin/vmax if given, else its own min/max -- pass a shared vmin/vmax
    across two calls to keep them on the same colour scale, e.g. an A/B ERF
    comparison), maps it through any matplotlib colormap name, alpha-blends
    over `image_gray` (uint8 or float, any range -> scaled to [0,1] by /255
    so a dim frame isn't stretched to false contrast), and returns an RGB
    float array in [0,1] ready for ax.imshow. heatmap_overlay stays the
    BGR/JET uint8 path for the cv2-only audit/view tools; this is the
    matplotlib path (default magma -- colourblind-safe, never jet) for
    tools/introspect.py and future GT views. Imports matplotlib lazily so
    the four cv2-only functions above stay dependency-free.
    """
    import matplotlib
    img = np.clip(image_gray.astype(np.float32) / (255.0 if image_gray.max() > 1.5 else 1.0), 0.0, 1.0)
    if heat.shape != image_gray.shape:
        heat = cv2.resize(heat.astype(np.float32), (image_gray.shape[1], image_gray.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
    lo = heat.min() if vmin is None else vmin
    hi = heat.max() if vmax is None else vmax
    heat_n = np.clip((heat - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.zeros_like(heat, dtype=np.float32)
    rgb = matplotlib.colormaps[cmap](heat_n)[..., :3]
    base_rgb = np.stack([img] * 3, axis=-1)
    return (1 - alpha) * base_rgb + alpha * rgb
