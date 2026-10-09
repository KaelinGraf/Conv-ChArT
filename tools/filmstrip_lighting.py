"""tools/filmstrip_lighting.py -- 3 x N visual LOW-LIGHT sweep: three algorithms down the
rows, falling brightness across the columns, each cell the actual frame with that
algorithm's PREDICTIONS drawn on it.

Kaelin, 2026-07-29: SNR is the right axis for this because it has "the most ramp-ish
character over the 3 types" -- all three degrade monotonically and at visibly
different rates, so the failure order reads straight off the picture. Distance would
work too but classical falls off a cliff rather than a ramp.

ONE FRAME, N noise levels. The board pose, background and every other augmentation
are pinned by a fixed seed and an explicit `components` override, so the only thing
changing left-to-right is the noise. Anything a viewer sees change IS the noise.

Rows: ours (COARSE -- the refiner is a localisation stage and would not change what
is detected), fine-tuned Deep ChArUco, classical OpenCV. Same frame, same GT, same
matching rule (4 px) for all three.

Markers: green = detected and correctly identified; red = detected but wrong ID;
faint grey ring = ground-truth corner nothing matched. Per-cell caption gives the
correct-ID count out of the visible corners, which is the number the eye should
confirm.

GITIGNORED (imports charuconet, which vendors third-party dcModel) -- same reason as
robustness_sweep_dc.py and eval_classical.py.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 10 columns: the cells are TIGHT CROPS of the board, so they are small and square and
# ten fit across a 16:9 slide comfortably (cf. Deep ChArUco Fig. 7, which runs twelve).
N_COLS = 10          # 10 tight board crops fit a 16:9 slide (DC Fig. 10 runs eleven)
MATCH_PX = 4.0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/rev640.yaml")
    p.add_argument("--ckpt", default="runs/rev640_160k_rev6/ckpt_0160000.pt")
    p.add_argument("--dc-ckpt", default="runs/charuconet_rev6_ft/ckpt_latest.pt")
    p.add_argument("--cols", type=int, default=N_COLS, help="columns (brightness steps)")
    p.add_argument("--dark-start", type=float, default=0.60,
                   help="brightness of the FIRST column. 0.60 is the darkest step at which all "
                        "three systems still score 100%%, so every subsequent loss is caused by "
                        "the light and nothing else. Starting at 1.0 wastes a column.")
    p.add_argument("--dark-end", type=float, default=0.015,
                   help="brightness of the LAST column; geometric spacing in between. 0.60->0.015 "
                        "over ten columns is ratio 0.664 rather than Deep ChArUco's 0.6, which puts "
                        "FOUR steps inside the 0.05-0.02 band where the baselines actually "
                        "collapse instead of jumping straight over it.")
    p.add_argument("--s-px", type=float, default=52.0, help="pinned board square size")
    p.add_argument("--crop-margin", type=float, default=0.80,
                   help="board-bbox margin as a fraction of board size. The cell is CROPPED TO\n                        THE BOARD: at full frame the board is a fraction of the cell and the\n                        markers are unreadable at slide size.")
    p.add_argument("--seed", type=int, default=18)
    p.add_argument("--out", default="paper/results_rev6/05_comparison/figures/filmstrip_lighting.png")
    p.add_argument("--dpi", type=int, default=150)
    return p


def main():
    args = build_parser().parse_args()
    import cv2, numpy as np, torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cv2.setNumThreads(1); torch.set_num_threads(2)
    import eval_classical as ec
    from charuconet import IN_H, IN_W, dcModel, decode_cells, pre_bgr_normalize
    from dcc.board import n_corners
    from dcc.dataset import load_config
    from dcc.model import DetectorNet, detector_kwargs
    from dcc.pipeline import detect
    from dcc.synth import generate_sample, list_backgrounds

    cfg = load_config(args.config)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W, H = cfg["input_size"]
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ours = DetectorNet(H, W, **detector_kwargs(cfg)).to(dev).eval()
    ours.load_state_dict(ck.get("ema", ck.get("model")))
    dc = dcModel(n_ids=n_corners(cfg.get("board"))).to(dev).eval()
    obj = torch.load(args.dc_ckpt, map_location=dev, weights_only=False)
    dc.load_state_dict(obj.get("ema", obj.get("model")))
    cboard, cdict, _ = ec._get_board_and_dictionary(cfg.get("board"))
    cparams = ec._detector_params()
    bg = list_backgrounds(cfg["synth"]["backgrounds"])

    def score(xy, ids, gt_xy, gt_idx):
        """-> per-prediction (x, y, ok) and the count of correctly-identified GT corners."""
        out, used, n_ok = [], set(), 0
        for (x, y), i in zip(xy, ids):
            d = np.linalg.norm(gt_xy - np.array([x, y]), axis=1)
            j = int(np.argmin(d)) if len(d) else -1
            ok = j >= 0 and d[j] <= MATCH_PX and int(i) == gt_idx[j] and j not in used
            if ok:
                used.add(j); n_ok += 1
            out.append((x, y, ok))
        return out, n_ok

    rows = ["Conv-ChArT\n(ours)", "Deep ChArUco\n(fine-tuned)", "classical\nOpenCV"]
    ncol = args.cols
    fig, axes = plt.subplots(3, ncol, figsize=(1.62 * ncol, 1.78 * 3), squeeze=False)
    box = {}          # crop box, computed once from the first column and reused everywhere

    # ONE frame, generated once, then DARKENED by a deterministic factor per column
    # (Deep ChArUco Fig. 10's protocol: brightness rescale 0.6^k). Two reasons this beats
    # sweeping sensor gain here:
    #   1. It is EXACT. Re-generating per column with a pinned seed does not give an
    #      identical frame, because rng.poisson() consumes a variable amount of the RNG
    #      stream depending on its rate, so every augmentation drawn after the sensor-noise
    #      step diverges. That produced one visibly washed-out column in the K sweep -- the
    #      figure was comparing different images while claiming only the noise changed. A
    #      multiplicative rescale has no such coupling.
    #   2. Low light is the deployment story (barrier-robot docking at night), and it is
    #      directly comparable to the published figure.
    # The scale is applied to the FINISHED frame, sensor noise included, exactly as the
    # reference does to a captured image -- so as k grows the signal collapses toward the
    # quantisation floor, which is the real failure mechanism.
    rec = generate_sample(cfg, np.random.default_rng([args.seed, 0]), bg,
                           components={"s": args.s_px}, force_negative=False)[0]
    base = rec["image"].astype(np.float64)
    vis_all = [c for c in rec["corners"] if c["visible"]]

    ratio = (args.dark_end / args.dark_start) ** (1.0 / max(ncol - 1, 1))
    for ci in range(ncol):
        f = args.dark_start * ratio ** ci
        img = np.clip(np.rint(base * f), 0, 255).astype(np.uint8)
        vis = vis_all
        gt_xy = np.array([[c["x"], c["y"]] for c in vis], dtype=np.float64)
        gt_idx = [c["index"] for c in vis]
        if not box:
            # crop to the board so it FILLS the cell -- the whole point of the reference
            # layout. Identical box in every cell, so the sweep stays visually registered.
            m = args.crop_margin * float(np.ptp(gt_xy, axis=0).max())
            box["x0"] = int(max(gt_xy[:, 0].min() - m, 0)); box["x1"] = int(min(gt_xy[:, 0].max() + m, W))
            box["y0"] = int(max(gt_xy[:, 1].min() - m, 0)); box["y1"] = int(min(gt_xy[:, 1].max() + m, H))
        x0, y0, x1, y1 = box["x0"], box["y0"], box["x1"], box["y1"]

        o = detect(img, ours, None, cfg=cfg)
        o_pts, o_ok = score(np.array([[c["x"], c["y"]] for c in o["corners"]]).reshape(-1, 2),
                            [c["index"] if c["index"] is not None else -1 for c in o["corners"]],
                            gt_xy, gt_idx)
        i320 = cv2.resize(img, (IN_W, IN_H), interpolation=cv2.INTER_AREA).astype(np.float32)
        with torch.no_grad():
            ll, il = dc(torch.from_numpy(pre_bgr_normalize(i320)).unsqueeze(0).unsqueeze(0).to(dev))
        xy320, didc, _a, _b = decode_cells(ll[0], il[0])
        dxy = np.empty_like(xy320, dtype=np.float64)
        dxy[:, 0] = (xy320[:, 0] + 0.5) * (W / IN_W) - 0.5
        dxy[:, 1] = (xy320[:, 1] + 0.5) * (H / IN_H) - 0.5
        d_pts, d_ok = score(dxy, didc, gt_xy, gt_idx)
        cxy, cids = ec.classical_charuco_detect(img, cboard, cdict, cparams)
        cxy = np.zeros((0, 2)) if cxy is None else np.asarray(cxy, np.float64).reshape(-1, 2)
        cids = np.zeros((0,), int) if cids is None else np.asarray(cids).reshape(-1)
        c_pts, c_ok = score(cxy, cids, gt_xy, gt_idx)

        for ri, (pts, nk) in enumerate(((o_pts, o_ok), (d_pts, d_ok), (c_pts, c_ok))):
            ax = axes[ri][ci]
            ax.imshow(img[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255)
            for x, y, ok in pts:
                if x0 <= x < x1 and y0 <= y < y1:
                    ax.plot(x - x0, y - y0, "o", color="#00e63c" if ok else "#ff1a1a",
                            ms=4.2, mec="black", mew=0.45)
            ax.text(0.03, 0.03, f"{nk}/{len(vis)}", transform=ax.transAxes, fontsize=7.5,
                    color="w", va="bottom", ha="left",
                    bbox=dict(fc="#00902a" if nk >= 0.9 * len(vis) else
                                 ("#b36b00" if nk >= 0.5 * len(vis) else "#b00000"),
                              ec="none", pad=1.1, alpha=0.88))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
            if ci == 0:
                ax.set_ylabel(rows[ri], fontsize=8, rotation=0, ha="right", va="center", labelpad=6)
            if ri == 0:
                ax.set_title(f"{f:.3f}" if f < 0.1 else f"{f:.2f}", fontsize=8.5,
                             fontweight="bold", pad=3)

    fig.suptitle(f"Brightness rescale factor  {args.dark_start:g} \u2192 {args.dark_end:g}"
                 f"   (geometric, ratio {ratio:.3f})  \u2192  darker", fontsize=11, y=0.995)
    fig.text(0.5, 0.015, "identical frame throughout; only the brightness scale changes.   "
             "green = detected and correctly identified,   red = detected, wrong ID.   "
             "badge = correctly identified / visible corners", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.035, 1, 0.955))
    fig.subplots_adjust(wspace=0.04, hspace=0.04)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
