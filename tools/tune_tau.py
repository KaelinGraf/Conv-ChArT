"""tools/tune_tau.py -- pick the heatmap read-out threshold tau_hm for a checkpoint.

WHY THIS EXISTS. tau_hm is an INFERENCE-ONLY constant: it appears nowhere in dcc/losses.py,
so it has no gradient path and cannot affect training. It is applied once, at read-out, in
dcc/pipeline.py:50 -- `keep = (hm == pooled) & (hm >= tau_hm)`. Weights are identical whatever
tau is, which means tau can be retuned on an already-trained checkpoint for free.

THE BUG IT EXISTS TO FIX. tau_hm = 0.3 was chosen for sigma_hm = 2.0. Every rev-6 tier arm
trains at sigma_hm = 0.5, whose targets are 4x tighter, so its learned peaks are sharper and
lower-amplitude. Scoring both families at 0.3 is not a like-for-like comparison, and it showed
up as a cliff: on the sensor_noise_K axis the sigma=0.5 composite fell to 20.8% recall at
K=0.02 where the sigma=2.0 production model held 91.6% -- while its ID accuracy among the
corners it DID find stayed at 82%. Recall collapsing with identity intact is the signature of
a threshold problem, not a representation problem.

WHY IT MEASURES PRECISION, WHICH run_validation DOES NOT. Lowering tau always raises recall;
the question is whether the extra detections are real corners or noise peaks that happen to
land within match_px of a ground-truth corner. train_detector.run_validation accumulates
n_gt/n_match/n_id but never the DETECTION COUNT, so precision is unobservable there and a tau
chosen on recall alone would be self-justifying. This tool counts n_det and reports
precision = n_match/n_det and F1 alongside, so the trade is explicit.

ONE FORWARD PASS, MANY THRESHOLDS. tau only filters an already-computed heatmap, so the
network runs once per frame and every tau is evaluated against the same cached sigmoid map.
Sweeping 8 thresholds therefore costs the same GPU time as one ordinary validation, which
matters while training is saturating the GPU.
"""
import os
# MUST precede any import that pulls in numpy/MKL, and must live in the MODULE (not the
# caller's environment): the val DataLoader uses multiprocessing_context="spawn", so every
# worker re-imports this file in a fresh interpreter and would otherwise re-inherit conda's
# default MKL_THREADING_LAYER=INTEL, which is incompatible with the libgomp that torch links.
# Setting it only in the launching shell fixes the parent and leaves the workers dying at
# import -- the DataLoader then blocks forever on workers that never arrive.
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_TAUS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--n", type=int, default=1000, help="val samples (fixed seed, same set every run)")
    p.add_argument("--taus", nargs="+", type=float, default=DEFAULT_TAUS)
    p.add_argument("--label", default=None, help="name in the output JSON; defaults to the ckpt's run dir")
    p.add_argument("--out", default=None)
    return p


def main():
    args = build_parser().parse_args()

    from functools import partial

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    import train_detector as td
    from dcc.board import n_corners
    from dcc.dataset import SynthVal, load_config
    from dcc.model import DetectorNet, detector_kwargs
    from dcc.pipeline import merge_close, peaks
    try:
        from dcc.pipeline import read_ids
    except ImportError:
        read_ids = None

    device = torch.device("cuda")
    cfg = load_config(args.config)
    W, H = cfg["input_size"]
    match_px = cfg["train"]["match_px"]

    model = DetectorNet(H, W, n_cls=n_corners(cfg.get("board")),
                        **detector_kwargs(cfg)).to(device, memory_format=torch.channels_last)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    # EMA weights are what validation scores and what ships -- scoring the raw weights here
    # would tune tau against a model nobody deploys.
    state = ck.get("ema") or ck["model"]
    model.load_state_dict(state)
    model.eval()
    step = ck.get("step")

    ds = SynthVal(cfg, n=args.n, seed=cfg["synth"]["val_seed"])
    loader = DataLoader(ds, batch_size=cfg["train"]["batch"], num_workers=6,
                        multiprocessing_context="spawn", collate_fn=partial(td._val_collate, cfg))

    acc = {t: {"n_gt": 0, "n_det": 0, "n_match": 0, "n_id": 0, "errs": []} for t in args.taus}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for images, _hms, _cts, _nvis, records in loader:
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            hm_logits, cls_logits = model(images)
            hm_prob = torch.sigmoid(hm_logits.float())
            cls_prob = torch.sigmoid(cls_logits.float())
            for b in range(images.shape[0]):
                rec = records[b]
                gt = [(c["x"], c["y"], c["index"]) for c in rec["corners"] if c["visible"]]
                gt_xy = np.array([(x, y) for x, y, _ in gt], dtype=np.float64).reshape(-1, 2)
                for tau in args.taus:                      # SAME heatmap, every threshold
                    det_xy, _p = merge_close(*peaks(hm_prob[b, 0], tau))
                    det_xy = det_xy.astype(np.float64)
                    m = td._match_greedy(gt_xy, det_xy, match_px)
                    a = acc[tau]
                    a["n_gt"] += len(gt_xy)
                    a["n_det"] += len(det_xy)
                    a["n_match"] += len(m)
                    a["errs"] += [d for _g, _d, d in m]
                    if read_ids is not None and m:
                        xy = torch.from_numpy(np.stack([det_xy[di] for _, di, _ in m])).to(device)
                        pred, _c = read_ids(cls_prob[b], xy)
                        pred = np.asarray(pred.detach().cpu() if torch.is_tensor(pred) else pred)
                        a["n_id"] += sum(int(int(p) == gt[gi][2]) for (gi, _di, _d), p in zip(m, pred))

    label = args.label or Path(args.ckpt).parent.name
    rows = []
    for tau in args.taus:
        a = acc[tau]
        rec = a["n_match"] / max(a["n_gt"], 1)
        prec = a["n_match"] / max(a["n_det"], 1)
        errs = np.sort(np.array(a["errs"])) if a["errs"] else np.array([0.0])
        rows.append({"tau": tau, "recall": rec, "precision": prec,
                     "f1": 0.0 if rec + prec == 0 else 2 * rec * prec / (rec + prec),
                     "id_acc": a["n_id"] / max(a["n_match"], 1),
                     "err_median": float(np.median(errs)), "err_p95": float(np.percentile(errs, 95)),
                     "n_gt": a["n_gt"], "n_det": a["n_det"], "n_match": a["n_match"]})

    print(f"\n{label}  (step {step}, sigma_hm={cfg.get('sigma_hm')}, n={args.n}, match_px={match_px})")
    print(f"{'tau':>6} {'recall':>8} {'precis':>8} {'F1':>8} {'ID':>8} {'err_med':>9} {'err_p95':>9} {'n_det':>8}")
    best = max(rows, key=lambda r: r["f1"])
    for r in rows:
        star = "  <-- best F1" if r is best else ""
        print(f"{r['tau']:>6.2f} {100*r['recall']:>7.2f}% {100*r['precision']:>7.2f}% {100*r['f1']:>7.2f}% "
              f"{100*r['id_acc']:>7.2f}% {r['err_median']:>9.4f} {r['err_p95']:>9.4f} {r['n_det']:>8}{star}")

    out = Path(args.out or f"paper/results_rev6/15_cost/tau_sweep_{label}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"label": label, "ckpt": args.ckpt, "config": args.config,
                               "step": step, "sigma_hm": cfg.get("sigma_hm"), "n": args.n,
                               "match_px": match_px, "rows": rows}, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
