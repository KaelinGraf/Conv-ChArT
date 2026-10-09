"""tools/refiner_crop_scale.py -- does the EXISTING refiner survive a scale-adaptive crop?

Kaelin's proposal (2026-07-29): s_px is knowable at inference from the coarse
peaks (median nearest-neighbour spacing estimates it to within 0.3-5% bias across
12-128 px, measured), so the refiner crop could be sized proportionally instead of
fixed at 24x24. That would fix the scale-floor failure, where a fixed 24x24 crop
spans more than one corner spacing at s=12 and the refiner can lock onto a
neighbour.

His follow-up question is the one that decides whether a retrain is needed at all:
maybe the refiner already generalises to a resized crop. This measures that
directly -- NO retraining, the current weights, crops cut at extent E and resized
to the 24x24 the network expects, with the predicted offset scaled back by E/24.

  E = 24  is the native case (no resize) and is the control.
  E < 24  UPSAMPLES a smaller window: removes neighbouring corners from view,
          adds no information, costs interpolation blur.
  E > 24  DOWNSAMPLES a larger window: adds context, destroys sub-pixel detail.

If M-03 holds roughly flat across E, the refiner is already scale-robust and the
scale-adaptive crop can be adopted at inference with NO retrain. If it degrades
sharply off E=24, retraining on scale-adaptive crops is required and the idea
costs a training run.

Errors are reported in SENSOR px throughout, so numbers are comparable to the
refiner's usual M-03 regardless of E.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/rev640.yaml")
    p.add_argument("--refiner-ckpt", default="runs/refiner_rev6/ckpt_0013000.pt")
    p.add_argument("--extents", type=int, nargs="+", default=[12, 16, 20, 24, 32, 40, 48])
    p.add_argument("--n", type=int, default=120, help="composites per extent")
    p.add_argument("--seed", type=int, default=20260729)
    p.add_argument("--out", default="paper/results_rev6/02_refiner/crop_scale.json")
    return p


def main():
    args = build_parser().parse_args()
    import cv2, numpy as np, torch
    cv2.setNumThreads(1); torch.set_num_threads(2)
    from dcc.dataset import load_config
    from dcc.model import Refiner
    from dcc.pipeline import soft_argmax
    from dcc.synth import generate_sample, list_backgrounds, visible

    cfg = load_config(args.config)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rk = torch.load(args.refiner_ckpt, map_location="cpu", weights_only=False)
    ref = Refiner().to(dev).eval()
    ref.load_state_dict(rk.get("ema", rk.get("model")))
    bg = list_backgrounds(cfg["synth"]["backgrounds"])
    mult = cfg["synth"]["refiner_res_mult"]
    W, H = cfg["input_size"]
    w2, h2 = int(W * mult), int(H * mult)
    jit = cfg["refiner_jitter_px"]

    rows = []
    for E in args.extents:
        half = E / 2.0
        errs, s_all = [], []
        for i in range(args.n):
            rng = np.random.default_rng([args.seed, i])
            rec, _ = generate_sample(cfg, rng, bg, size_mult=mult, force_negative=False)
            img = rec["image"]
            pts = [(c["x"], c["y"]) for c in rec["corners"] if c["visible"]]
            for (px, py) in pts[:8]:
                j = rng.uniform(-jit, jit, size=2)
                c = np.rint([px + j[0], py + j[1]])
                d = np.array([px - c[0], py - c[1]])          # true offset, sensor px
                if max(abs(d)) > 3.9375:
                    continue
                x0, y0 = int(c[0] - half), int(c[1] - half)
                if x0 < 0 or y0 < 0 or x0 + E > w2 or y0 + E > h2:
                    continue
                crop = img[y0:y0 + E, x0:x0 + E]
                if E != 24:
                    interp = cv2.INTER_AREA if E > 24 else cv2.INTER_LINEAR
                    crop = cv2.resize(crop, (24, 24), interpolation=interp)
                t = torch.from_numpy(crop.astype(np.float32)[None, None] / 255.0).to(dev)
                with torch.no_grad():
                    u = soft_argmax(ref(t).sigmoid().cpu())
                # the network answers in ITS OWN 24-px frame; scale back to sensor px
                pred = (np.asarray(u)[0] - 31.5) / 8.0 * (E / 24.0)
                errs.append(float(np.linalg.norm(pred - d)))
            s_all.append(rec["s_px"])
        e = np.array(errs)
        rows.append({"extent": E, "n": int(len(e)), "median": float(np.median(e)),
                     "p95": float(np.percentile(e, 95)), "frac_lt_0p25": float((e < 0.25).mean()),
                     "frac_lt_1": float((e < 1.0).mean())})
        tag = "  <- native, control" if E == 24 else ("  (upsampled)" if E < 24 else "  (downsampled)")
        print(f"  extent {E:>3} px  n={len(e):>5}  median {np.median(e):.4f}  p95 {np.percentile(e,95):.4f}  "
              f"<0.25px {100*(e<0.25).mean():5.1f}%  <1px {100*(e<1).mean():5.1f}%{tag}", flush=True)

    base = next(r for r in rows if r["extent"] == 24)
    print(f"\nrelative to the native 24 px control (median {base['median']:.4f} px):")
    for r in rows:
        print(f"  extent {r['extent']:>3}: {r['median']/base['median']:.2f}x median, "
              f"{r['p95']/max(base['p95'],1e-9):.2f}x p95")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "provenance": {
        "refiner_ckpt": args.refiner_ckpt, "refiner_step": rk.get("step"), "config": args.config,
        "n_composites_per_extent": args.n, "seed": args.seed,
        "note": "NO retraining -- current weights, crops resized to 24x24, offsets scaled by E/24"}}, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
