"""tools/refiner_gain.py -- does refinement HELP or HURT, as a function of how
wrong the coarse peak already was?

The robustness sweeps showed the refiner improving median localisation ~3x while
producing ~22x more gross (>3 px) errors, costing 2-3 pp of recall and ID
accuracy. Marginal distributions cannot say which corners it hurt, so they
cannot justify a fix. This pairs them: ONE detect() call yields both the
refined and the pre-refinement coarse position for the SAME detection
(dcc/pipeline.py reports x_coarse/y_coarse), so every matched corner gives an
exact (coarse_err, refined_err) pair.

Binning those pairs by coarse_err answers the design question directly:
- if refinement wins only where coarse_err is already small, the training jitter
  (U(-4,4), dcc/refiner_data.py:202) is too wide and should be re-shaped;
- if it wins across the range and loses only rarely, the problem is the absent
  inference guard at dcc/pipeline.py, not the training distribution;
- the crossover, if any, IS the guard threshold -- measured, not eyeballed.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MATCH_PX = 4.0
BINS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/rev640.yaml")
    p.add_argument("--ckpt", default="runs/rev640_baseline/ckpt_0150000.pt")
    p.add_argument("--refiner-ckpt", default="runs/refiner_rev4_ft/ckpt_0013000.pt")
    p.add_argument("--n", type=int, default=400, help="frames")
    p.add_argument("--seed", type=int, default=20260728)
    p.add_argument("--out", default="paper/results_rev4/refiner_gain.json")
    return p


def main():
    args = build_parser().parse_args()
    import cv2
    import numpy as np
    import torch
    cv2.setNumThreads(1); torch.set_num_threads(2)

    from dcc.dataset import load_config
    from dcc.model import DetectorNet, Refiner, detector_kwargs
    from dcc.pipeline import detect
    from dcc.synth import generate_sample, list_backgrounds
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_td", str(Path(__file__).resolve().parents[1] / "tools" / "train_detector.py"))
    td = importlib.util.module_from_spec(spec); spec.loader.exec_module(td)

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W, H = cfg["input_size"]
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = DetectorNet(H, W, **detector_kwargs(cfg)).to(device).eval()
    model.load_state_dict(ck.get("ema", ck.get("model")))
    rk = torch.load(args.refiner_ckpt, map_location="cpu", weights_only=False)
    refiner = Refiner().to(device).eval()
    refiner.load_state_dict(rk.get("ema", rk.get("model")))
    bg = list_backgrounds(cfg["synth"]["backgrounds"])

    pairs = []   # (coarse_err, refined_err, jump) per matched corner
    for i in range(args.n):
        rng = np.random.default_rng([args.seed, 991, i])
        rec, _ = generate_sample(cfg, rng, bg, force_negative=False)
        vis = [c for c in rec["corners"] if c["visible"]]
        if not vis:
            continue
        gt = np.array([[c["x"], c["y"]] for c in vis], dtype=np.float64)
        res = detect(rec["image"], model, refiner, K=None, dist=None, cfg=cfg)
        det = res["corners"]
        if not det:
            continue
        # Match on the COARSE positions so the pairing is not itself selected by
        # refinement quality -- matching on refined would silently drop exactly
        # the corners the refiner ruined, which is the effect being measured.
        cxy = np.array([[c["x_coarse"], c["y_coarse"]] for c in det], dtype=np.float64)
        rxy = np.array([[c["x"], c["y"]] for c in det], dtype=np.float64)
        for gi, di, dist_c in td._match_greedy(gt, cxy, MATCH_PX):
            dist_r = float(np.linalg.norm(gt[gi] - rxy[di]))
            pairs.append((float(dist_c), dist_r, float(np.linalg.norm(rxy[di] - cxy[di]))))

    a = np.array(pairs)
    coarse, refined, jump = a[:, 0], a[:, 1], a[:, 2]
    rows = []
    print(f"n={len(a)} matched corners\n")
    print(f"{'coarse_err bin':>16}{'n':>7}{'coarse':>9}{'refined':>9}{'gain':>9}{'help%':>8}{'worse>1px':>11}")
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (coarse >= lo) & (coarse < hi)
        if m.sum() < 5:
            continue
        c, r = coarse[m], refined[m]
        row = {"lo": lo, "hi": hi, "n": int(m.sum()), "coarse_med": float(np.median(c)),
               "refined_med": float(np.median(r)), "help_frac": float((r < c).mean()),
               "worse_gt1px_frac": float(((r - c) > 1.0).mean())}
        rows.append(row)
        print(f"{f'[{lo:.2f},{hi:.2f})':>16}{row['n']:>7}{row['coarse_med']:9.3f}{row['refined_med']:9.3f}"
              f"{row['coarse_med']-row['refined_med']:9.3f}{row['help_frac']*100:8.1f}{row['worse_gt1px_frac']*100:11.1f}")

    # Guard sweep: accept the refinement only when it moved the corner <= T px.
    print(f"\n{'guard T':>9}{'median':>9}{'p95':>9}{'>1px%':>8}{'>3px%':>8}{'accepted%':>11}")
    guards = []
    for T in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 99.0]:
        use = jump <= T
        err = np.where(use, refined, coarse)
        g = {"T": T, "median": float(np.median(err)), "p95": float(np.percentile(err, 95)),
             "gt1px": float((err > 1).mean()), "gt3px": float((err > 3).mean()),
             "accepted": float(use.mean())}
        guards.append(g)
        print(f"{T:9.2f}{g['median']:9.4f}{g['p95']:9.4f}{g['gt1px']*100:8.2f}{g['gt3px']*100:8.2f}{g['accepted']*100:11.1f}")
    base = {"median": float(np.median(coarse)), "p95": float(np.percentile(coarse, 95)),
            "gt1px": float((coarse > 1).mean()), "gt3px": float((coarse > 3).mean())}
    print(f"{'coarse':>9}{base['median']:9.4f}{base['p95']:9.4f}{base['gt1px']*100:8.2f}{base['gt3px']*100:8.2f}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"provenance": {"ckpt": args.ckpt, "ckpt_step": ck.get("step"),
                                   "refiner_ckpt": args.refiner_ckpt, "refiner_step": rk.get("step"),
                                   "config": args.config, "n_frames": args.n, "seed": args.seed,
                                   "match_px": MATCH_PX, "n_pairs": len(a)},
                   "bins": rows, "guard_sweep": guards, "coarse_baseline": base,
                   "pairs": a.tolist()}, f, indent=2)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
