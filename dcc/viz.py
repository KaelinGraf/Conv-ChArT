"""Drawing helpers: draw_overlay(image, record), tile(images, cols), heatmap_overlay, overlay_alpha,
surface3d (matplotlib 3D heatmap) and save_panels (each axes of a figure saved as its own image).
"""
import cv2
import numpy as np

_GREEN, _RED, _ORANGE = (0, 200, 0), (0, 0, 220), (0, 140, 255)


def draw_overlay(image_u8_gray: np.ndarray, record: dict, meta: dict | None = None,
                  draw_indices: bool = True, filled: bool = True, radius: int = 3,
                  color_fn=None) -> np.ndarray:
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
    h, w = imgs[0].shape[:2]
    rows = [imgs[i:i + cols] for i in range(0, len(imgs), cols)]
    rows[-1] = rows[-1] + [np.zeros((h, w, 3), np.uint8)] * (cols - len(rows[-1]))
    return cv2.vconcat([cv2.hconcat(r) for r in rows])


def heatmap_overlay(image: np.ndarray, hm: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
    span = hm.max() - hm.min()
    norm = (hm - hm.min()) / span if span > 0 else np.zeros_like(hm, dtype=np.float32)
    cmap = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    if cmap.shape[:2] != base.shape[:2]:
        cmap = cv2.resize(cmap, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(base, 1 - alpha, cmap, alpha, 0)


def surface3d(ax, hm: np.ndarray, stride: int = 2, zlim: tuple = (0, 1.05),
              elev: float = 45, azim: float = -60, cmap: str = "viridis"):
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


def save_panels(fig, out, dpi: int = 150, pad: float = 0.1) -> list:
    import re
    from pathlib import Path
    out = Path(out)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    panels = [a for a in fig.axes if a.get_visible() and a.has_data()]
    artists = [(o, o.get_visible()) for o in [*fig.axes, *fig.texts]]
    def _bbox(a):
        bb = a.get_tightbbox(r)
        return bb.union([bb, a.xaxis.label.get_window_extent(r), a.yaxis.label.get_window_extent(r)])
    bbs = [_bbox(a).transformed(fig.dpi_scale_trans.inverted()).padded(pad) for a in panels]
    paths = []
    for i, (ax, bb) in enumerate(zip(panels, bbs), 1):
        slug = re.sub(r"[^a-z0-9]+", "_", ax.get_title().lower()).strip("_")[:40].rstrip("_")
        for o, _ in artists:
            o.set_visible(o is ax)
        paths.append(out.with_name("_".join(filter(None, [out.stem, f"p{i}", slug])) + out.suffix))
        fig.savefig(paths[-1], dpi=dpi, bbox_inches=bb)
    for o, v in artists:
        o.set_visible(v)
    return paths
