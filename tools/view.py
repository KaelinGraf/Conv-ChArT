"""Sample viewer — eyeball detector/refiner training samples and their
GT targets before or during training. Argparse runs before any heavy import
so --help never needs dcc/numpy/cv2/matplotlib.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--stream", choices=["detector", "refiner"], default="detector")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--index", type=int, default=None, help="view exactly SynthVal[index] (detector) "
                                                             "or the composite at this index (refiner)")
    p.add_argument("--out", default="sheet.png")
    p.add_argument("--show", action="store_true", help="interactive window, keys n/p/q")
    p.add_argument("--channels", action="store_true", help="also plot the 16 class-target channels")
    return p


def _detector_samples(SynthVal, cfg, args):
    # --index must reproduce the CANONICAL val set: SynthVal's stratified s
    # depends on n, so index mode pins n to cfg val_size, not index+1.
    n_needed = cfg["synth"]["val_size"] if args.index is not None else args.n
    ds = SynthVal(cfg, n_needed, args.seed)
    idxs = [args.index] if args.index is not None else list(range(args.n))
    return [(i, *ds[i]) for i in idxs]


def _refiner_samples(RefinerVal, cfg, args):
    if args.index is not None:
        crops = RefinerVal(cfg, args.index + 1, args.seed)[args.index]
        return [(args.index, c) for c in crops[:args.n]]
    cap = args.n * 4 + 8                        # composites needed to likely yield n crops
    ds = RefinerVal(cfg, cap, args.seed)
    flat = []
    for i in range(cap):
        flat += [(i, c) for c in ds[i]]
        if len(flat) >= args.n:
            break
    return flat[:args.n]


def _detector_panels(targets, viz, i, image, record, cfg, seed):
    import numpy as np
    corners = record["corners"]
    pts = np.array([[c["x"], c["y"]] for c in corners]).reshape(-1, 2)
    vis = np.array([c["visible"] for c in corners], dtype=bool)
    idx = np.array([c["index"] for c in corners], dtype=int)
    size_wh = (image.shape[1], image.shape[0])
    hm = targets.render_heatmap(pts, vis, size_wh, sigma=cfg["sigma_hm"])
    ct = targets.render_class_targets(pts, vis, idx, size_wh, sigma=cfg["sigma_cls"])
    cls_up = np.repeat(np.repeat(ct.max(axis=0), 4, axis=0), 4, axis=1)   # cell j -> px 4j..4j+3
    tag = (f"idx={i} seed={seed} NEGATIVE" if not record["board_present"] else
           f"idx={i} seed={seed} s={record['s_px']:.1f}px vis={int(vis.sum())}/{len(corners)}")
    panels = [(f"overlay | {tag}", viz.draw_overlay(image, record)),
              (f"heatmap | {tag}", viz.heatmap_overlay(image, hm)),
              (f"class-max x4 | {tag}", viz.heatmap_overlay(image, cls_up))]
    return panels, ct


def _refiner_panels(targets, i, rec):
    import cv2
    crop, d = rec["crop"], rec["d"]
    up = cv2.cvtColor(cv2.resize(crop, (192, 192), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    cx, cy = int(round((12 + d[0]) * 8)), int(round((12 + d[1]) * 8))
    cv2.drawMarker(up, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
    tag = f"composite={i} d=({d[0]:.2f},{d[1]:.2f})"
    rt = targets.render_refiner_target(d)
    return [(f"crop x8 | {tag}", up), (f"target 64x64 | {tag}", rt)], None


def _show_panel(ax, title, arr):
    import cv2
    if arr.ndim == 2:
        ax.imshow(arr, cmap="hot")
    else:
        ax.imshow(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def _draw_row(axes_row, panels):
    for ax, (title, arr) in zip(axes_row, panels):
        _show_panel(ax, title, arr)


def _save_sheet(rows, ncols, out_path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(rows), ncols, figsize=(4 * ncols, 3.2 * len(rows)), squeeze=False)
    for r, panels in enumerate(rows):
        _draw_row(axes[r], panels)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    return fig


def _save_channel_grid(ct, out_path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 4, figsize=(8, 8), squeeze=False)
    for k in range(16):
        ax = axes[k // 4][k % 4]
        ax.imshow(ct[k], cmap="hot")
        ax.set_title(str(k), fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def _interactive(rows, ncols, cts, show_channels):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 3.2), squeeze=False)
    fig_ch, axes_ch = plt.subplots(4, 4, figsize=(7, 7), squeeze=False) if show_channels else (None, None)
    pos = [0]

    def render():
        for ax in axes[0]:
            ax.cla()
        _draw_row(axes[0], rows[pos[0]])
        fig.suptitle(f"[{pos[0] + 1}/{len(rows)}]  n=next  p=prev  q=quit", fontsize=9)
        fig.canvas.draw_idle()
        if show_channels:
            for k in range(16):
                ax = axes_ch[k // 4][k % 4]
                ax.cla()
                ax.imshow(cts[pos[0]][k], cmap="hot")
                ax.set_title(str(k), fontsize=8)
                ax.axis("off")
            fig_ch.canvas.draw_idle()

    def on_key(event):
        if event.key == "q":
            plt.close(fig)
            if fig_ch is not None:
                plt.close(fig_ch)
        elif event.key in ("n", "p"):
            pos[0] = (pos[0] + (1 if event.key == "n" else -1)) % len(rows)
            render()

    fig.canvas.mpl_connect("key_press_event", on_key)
    if fig_ch is not None:
        fig_ch.canvas.mpl_connect("key_press_event", on_key)
    render()
    plt.show()


def main():
    args = build_parser().parse_args()
    args.n = max(args.n, 0)

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    from dcc.dataset import load_config, SynthVal, RefinerVal
    from dcc import targets, viz

    cfg = load_config(args.config)
    detector = args.stream == "detector"
    if detector:
        samples = _detector_samples(SynthVal, cfg, args)
        built = [_detector_panels(targets, viz, i, img, rec, cfg, args.seed) for i, img, rec in samples]
    else:
        samples = _refiner_samples(RefinerVal, cfg, args)
        built = [_refiner_panels(targets, i, rec) for i, rec in samples]

    if not built:
        print("no samples to display")
        return
    rows, cts = [b[0] for b in built], [b[1] for b in built]
    ncols = len(rows[0])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = _save_sheet(rows, ncols, str(out_path))
    if detector and args.channels:
        for (i, _, _), ct in zip(samples, cts):
            _save_channel_grid(ct, str(out_path.with_stem(f"{out_path.stem}_ch{i}")))

    import matplotlib.pyplot as plt
    plt.close(fig)
    if args.show:
        _interactive(rows, ncols, cts, detector and args.channels)


if __name__ == "__main__":
    main()
