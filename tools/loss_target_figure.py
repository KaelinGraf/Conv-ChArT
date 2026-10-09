"""tools/loss_target_figure.py -- the loss-formulation slide figure.

Two panels: the raw input with its ground-truth corners overlaid, and the TARGET
heatmap the loss is computed against, rendered as a 3D surface so the Gaussian is
a thing you can see rather than a term in an equation.

This is the GT target (dcc.targets.render_heatmap), NOT a model prediction --
introspect.py's heatmap3d panel shows the latter. The point of this figure is the
loss's own construction: per-corner Gaussians at sigma_hm, combined by MAX (never
sum, per CornerNet), with the rint(p) pixel forced to exactly 1.0 so focal's Y=1
branch is always reachable.

Defaults to a LARGE-board frame: at s ~ 87 px the sixteen Gaussians are well
separated and read as sixteen distinct peaks. On a small board they merge into a
ridge and the figure stops making its point.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/rev640.yaml")
    p.add_argument("--index", type=int, default=4300,
                   help="SynthVal index. Default board is ~26%% of frame width (s~33 px):\n"
                        "big enough to read in the left panel, small enough that zooming the\n"
                        "3D to the whole board still leaves each sigma=2 Gaussian a visible\n"
                        "fraction of the crop. A 87 px board needs a 500 px crop to show all\n"
                        "16 corners, where a 4.7 px FWHM peak is 1%% of the width -- needles.")
    p.add_argument("--zoom-spacings", type=float, default=2.1,
                   help="3D crop half-width in units of CORNER SPACING. The whole board is the\n"
                        "wrong crop: sigma_hm is 2 px, so across a 500 px board the Gaussians\n"
                        "render as needles and the figure shows a spike train, not a Gaussian.\n"
                        "1.7 spacings puts ~4 corners in frame with each peak ~8% of the width.")
    p.add_argument("--elev", type=float, default=52.0)
    p.add_argument("--azim", type=float, default=-60.0)
    p.add_argument("--out", default="paper/results_rev6/07_introspection/loss_target.png")
    p.add_argument("--dpi", type=int, default=170)
    return p


def main():
    args = build_parser().parse_args()
    import numpy as np, cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cv2.setNumThreads(1)
    from dcc.dataset import load_config, SynthVal
    from dcc.targets import render_heatmap

    cfg = load_config(args.config)
    W, H = cfg["input_size"]
    sigma = cfg["sigma_hm"]
    ds = SynthVal(cfg, cfg["synth"]["val_size"], cfg["synth"]["val_seed"])
    img, rec = ds[args.index]
    pts = np.array([[c["x"], c["y"]] for c in rec["corners"]], dtype=np.float64)
    vis = np.array([bool(c["visible"]) for c in rec["corners"]])
    hm = render_heatmap(pts, vis, (W, H), sigma=sigma)
    print(f"val{args.index}: s_px={rec['s_px']:.1f}  visible={int(vis.sum())}/16  "
          f"sigma_hm={sigma}  heatmap max={hm.max():.4f}")

    # Crop to a few corners around the board centre, sized in units of the board's OWN
    # corner spacing so the zoom is scale-invariant across frames.
    vp = pts[vis]
    dd = np.linalg.norm(vp[:, None] - vp[None], axis=2)
    np.fill_diagonal(dd, np.inf)
    spacing = float(np.median(dd.min(axis=1)))
    c = vp.mean(0)
    half = args.zoom_spacings * spacing
    x0 = int(max(c[0] - half, 0)); x1 = int(min(c[0] + half, W))
    y0 = int(max(c[1] - half, 0)); y1 = int(min(c[1] + half, H))
    print(f"  corner spacing {spacing:.1f} px -> 3D crop {x1-x0}x{y1-y0} px, "
          f"Gaussian FWHM {2.355*sigma:.1f} px = {100*2.355*sigma/(x1-x0):.1f}% of crop width")
    sub = hm[y0:y1, x0:x1]
    gy, gx = np.mgrid[y0:y1, x0:x1]

    fig = plt.figure(figsize=(15.5, 6.4))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(img, cmap="gray")
    ax.plot(vp[:, 0], vp[:, 1], "+", color="#00ff66", ms=13, mew=2.0, label="ground-truth corner")
    if (~vis).any():
        ax.plot(pts[~vis, 0], pts[~vis, 1], "x", color="#ff3030", ms=9, mew=1.8, label="occluded / off-frame")
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#ffd000", lw=1.2, ls="--"))
    ax.set_title(f"Input  ({W}x{H}, s = {rec['s_px']:.0f} px)   —   dashed box = 3D region at right",
                 fontsize=11)
    ax.legend(fontsize=8, loc="lower right"); ax.axis("off")

    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    # rcount/ccount default to 50, which would decimate a sigma=2 peak out of existence;
    # sample at ~1 quad per 2 px so every Gaussian survives the render
    # ONE QUAD PER PIXEL. The default rcount/ccount is 50, and even crop//2 (one quad per
    # 2 px) leaves a 4.7 px FWHM peak spanning ~2 quads -- which is why earlier versions
    # rendered spikes rather than Gaussians. At 1 quad/px the profile is fully resolved.
    ax3.plot_surface(gx, gy, sub, cmap="viridis", linewidth=0, antialiased=True,
                     rcount=min(y1 - y0, 600), ccount=min(x1 - x0, 600))
    ax3.set_zlim(0, 1.05)
    ax3.view_init(elev=args.elev, azim=args.azim)
    ax3.set_xlabel("x (px)", fontsize=9); ax3.set_ylabel("y (px)", fontsize=9)
    ax3.set_zlabel("target $Y$", fontsize=9)
    ax3.set_title(f"Target heatmap $Y$ (zoomed)   —   per-corner Gaussians, $\\sigma$ = {sigma} px,\n"
                  f"combined by MAX, peak pixel forced to exactly 1.0", fontsize=11)

    fig.suptitle("The heatmap head's training target: a permissive Gaussian, not a one-hot label",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
