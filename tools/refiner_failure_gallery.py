"""tools/refiner_failure_gallery.py -- SHOW the crops the refiner fails on.

Kaelin, 2026-07-29: "maybe visualise for me (like literally show me visually) a
few of the crops that the refiner fails on? i might be able to find stuff."

The numbers say the refiner improves the MEDIAN at every scale (0.09-0.16 px vs
the coarse peak's 0.41-0.43) and wrecks the p95 below s = 32 (2.16-2.69 px vs a
flat coarse 0.82-0.91). Two candidate mechanisms were measured and REJECTED --
capture clipping (|d| p95 is 0.80 px against a +-3.9375 px support) and, in the
TRAINING canvas, neighbour contamination (min NN distance 24.26 px vs a 12 px
half-width). What is left is the crop's own CONTENT, which is a thing to look at
rather than to argue about.

Renders the worst-error crops at each scale, 8x nearest-neighbour so every sensor
pixel is a visible block, with:
  green cross  = ground-truth corner
  red cross    = where the refiner put it
  blue cross   = the coarse peak the crop was centred on (crop centre + its offset)
  yellow box   = the +-3.9375 px capture support
A control row of MEDIAN-error crops at the same scale is drawn alongside, because
"what does a failure look like" is only answerable against "what does a success
look like" -- otherwise every crop looks plausibly bad.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ZOOM = 8
SUP = 3.9375


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/rev640.yaml")
    p.add_argument("--ckpt", default="runs/rev640_baseline/ckpt_0150000.pt")
    p.add_argument("--refiner-ckpt", default="runs/refiner_rev6/ckpt_0013000.pt")
    p.add_argument("--scales", type=int, nargs="+", default=[12, 16, 24, 64])
    p.add_argument("--n", type=int, default=60, help="frames per scale")
    p.add_argument("--worst", type=int, default=6, help="worst-error crops shown per scale")
    p.add_argument("--seed", type=int, default=20260729)
    p.add_argument("--out", default="paper/results_rev6/02_refiner/failure_gallery.png")
    return p


def main():
    args = build_parser().parse_args()
    import cv2, numpy as np, torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cv2.setNumThreads(1); torch.set_num_threads(2)
    from dcc.dataset import load_config
    from dcc.model import DetectorNet, Refiner, detector_kwargs
    from dcc.pipeline import detect
    from dcc.synth import generate_sample, list_backgrounds

    cfg = load_config(args.config)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W, H = cfg["input_size"]
    mk = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = DetectorNet(H, W, **detector_kwargs(cfg)).to(dev).eval()
    model.load_state_dict(mk.get("ema", mk.get("model")))
    rk = torch.load(args.refiner_ckpt, map_location="cpu", weights_only=False)
    ref = Refiner().to(dev).eval(); ref.load_state_dict(rk.get("ema", rk.get("model")))
    bg = list_backgrounds(cfg["synth"]["backgrounds"])

    per_scale = {}
    for s in args.scales:
        found = []
        for i in range(args.n):
            rng = np.random.default_rng([args.seed, s, i])
            rec, _ = generate_sample(cfg, rng, bg, components={"s": float(s)}, force_negative=False)
            img = rec["image"]
            gt = np.array([[c["x"], c["y"]] for c in rec["corners"] if c["visible"]])
            if not len(gt):
                continue
            out = detect(img, model, ref, cfg=cfg)
            for c in out["corners"]:
                p = np.array([c["x"], c["y"]]); pc = np.array([c["x_coarse"], c["y_coarse"]])
                j = int(np.argmin(np.linalg.norm(gt - pc, axis=1)))
                if np.linalg.norm(gt[j] - pc) > 8:
                    continue
                centre = np.rint(pc).astype(int)     # r = 1 in this config
                if not (12 <= centre[0] <= img.shape[1] - 12 and 12 <= centre[1] <= img.shape[0] - 12):
                    continue
                crop = img[centre[1] - 12:centre[1] + 12, centre[0] - 12:centre[0] + 12].copy()
                found.append({"err": float(np.linalg.norm(gt[j] - p)),
                               "crop": crop, "centre": centre, "gt": gt[j], "pred": p, "coarse": pc,
                               "err_coarse": float(np.linalg.norm(gt[j] - pc))})
        found.sort(key=lambda d: -d["err"])
        per_scale[s] = found
        print(f"s={s:>4}: {len(found)} crops, worst {found[0]['err']:.2f} px, "
              f"median {np.median([f['err'] for f in found]):.3f} px", flush=True)

    ncol = args.worst
    nrow = 2 * len(args.scales)
    fig, ax = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, 2.25 * nrow))
    ax = np.atleast_2d(ax)
    for si, s in enumerate(args.scales):
        found = per_scale[s]
        if not found:
            continue
        mid = len(found) // 2
        picks = [("WORST", found[:ncol]),
                  ("typical", found[mid:mid + ncol])]
        for ri, (tag, sel) in enumerate(picks):
            for ci in range(ncol):
                a = ax[2 * si + ri, ci]
                a.set_xticks([]); a.set_yticks([])
                if ci >= len(sel):
                    a.axis("off"); continue
                f = sel[ci]
                big = cv2.resize(f["crop"], (24 * ZOOM, 24 * ZOOM), interpolation=cv2.INTER_NEAREST)
                a.imshow(big, cmap="gray", vmin=0, vmax=255)
                # crop-local coords -> zoomed display coords (half-pixel convention)
                def to_disp(pt):
                    u = pt - (f["centre"] - 12)
                    return (u + 0.5) * ZOOM - 0.5
                for pt, col, mk_ in ((f["gt"], "#00ff00", "+"), (f["pred"], "#ff2020", "x"),
                                      (f["coarse"], "#40a0ff", "1")):
                    d = to_disp(pt)
                    a.plot(d[0], d[1], mk_, color=col, ms=11, mew=2)
                c0 = to_disp(f["centre"].astype(float))
                a.add_patch(plt.Rectangle((c0[0] - SUP * ZOOM, c0[1] - SUP * ZOOM), 2 * SUP * ZOOM,
                                           2 * SUP * ZOOM, fill=False, ec="#ffd000", lw=1.0, ls="--"))
                a.set_title(f"{tag}  err {f['err']:.2f}px\n(coarse {f['err_coarse']:.2f}px)", fontsize=7)
            ax[2 * si + ri, 0].set_ylabel(f"s = {s} px\n{tag}", fontsize=9)
    fig.suptitle("Refiner failures vs typical cases, by board scale   |   "
                 "green + = ground truth, red x = refiner, blue 1 = coarse peak, "
                 "yellow dashed = ±3.9375 px capture support\n"
                 "crops are the real 24×24 sensor input, nearest-neighbour ×8 so every pixel is visible",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
