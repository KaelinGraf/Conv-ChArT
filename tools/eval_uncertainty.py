"""tools/eval_uncertainty.py -- is the detector's uncertainty USEFUL?

Emitting a confidence is not a result; a confidence that predicts error is. The
pipeline reports three per-corner signals (`p_hm` detection, `p_id` identity,
`sigma_px` localisation spread from the refiner's own peak) and this asks the
only questions that make them first-class outputs rather than decoration:

  1. CALIBRATION -- when the network says 0.9, is it right 90% of the time?
     Reliability curve over p_id bins + Expected Calibration Error.
  2. DISCRIMINATION -- does sigma_px actually rank corners by how wrong they are?
     Localisation error per sigma decile. A flat curve means the signal is
     worthless however well-calibrated it looks.
  3. SELECTIVE PREDICTION -- the operationally decisive one. Reject the least
     confident fraction and plot accuracy against coverage. A docking controller
     does not need every corner; it needs to know which ones to trust, and
     "at 90% coverage ID accuracy is X%" is a number a system integrator can act on.

Uncertainty is only worth reporting if (2) and (3) show real separation, so both
are reported even when unflattering. `sigma_px` is NaN wherever Stage 2 did not
run (no refiner, or a border-bypassed crop); those corners are excluded from the
sigma analyses and the excluded count is reported rather than silently dropped.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MATCH_PX = 4.0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/rev640.yaml")
    p.add_argument("--ckpt", default="runs/rev640_baseline/ckpt_0150000.pt")
    p.add_argument("--refiner-ckpt", default="runs/refiner_rev5/ckpt_0010000.pt")
    p.add_argument("--n", type=int, default=400, help="frames")
    p.add_argument("--seed", type=int, default=20260728)
    p.add_argument("--out", default="paper/results_rev6/10_uncertainty/uncertainty.json")
    p.add_argument("--fig", default="paper/results_rev6/10_uncertainty/uncertainty.png")
    return p


def main():
    args = build_parser().parse_args()
    import cv2, numpy as np, torch, importlib.util
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cv2.setNumThreads(1); torch.set_num_threads(2)

    from dcc.dataset import load_config
    from dcc.model import DetectorNet, Refiner, detector_kwargs
    from dcc.pipeline import detect
    from dcc.synth import generate_sample, list_backgrounds
    spec = importlib.util.spec_from_file_location(
        "_td", str(Path(__file__).resolve().parents[1] / "tools" / "train_detector.py"))
    td = importlib.util.module_from_spec(spec); spec.loader.exec_module(td)

    cfg = load_config(args.config)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W, H = cfg["input_size"]
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = DetectorNet(H, W, **detector_kwargs(cfg)).to(dev).eval()
    model.load_state_dict(ck.get("ema", ck.get("model")))
    rk = torch.load(args.refiner_ckpt, map_location="cpu", weights_only=False)
    refiner = Refiner().to(dev).eval()
    refiner.load_state_dict(rk.get("ema", rk.get("model")))
    bg = list_backgrounds(cfg["synth"]["backgrounds"])

    err, sig, pid, phm, idok = [], [], [], [], []
    for i in range(args.n):
        rng = np.random.default_rng([args.seed, 4242, i])
        rec, _ = generate_sample(cfg, rng, bg, force_negative=False)
        vis = [c for c in rec["corners"] if c["visible"]]
        if not vis:
            continue
        gt = np.array([[c["x"], c["y"]] for c in vis], dtype=np.float64)
        gidx = [c["index"] for c in vis]
        res = detect(rec["image"], model, refiner, K=None, dist=None, cfg=cfg)
        d = res["corners"]
        if not d:
            continue
        xy = np.array([[c["x"], c["y"]] for c in d])
        for gi, di, dist in td._match_greedy(gt, xy, MATCH_PX):
            err.append(dist); sig.append(d[di]["sigma_px"])
            pid.append(d[di]["p_id"]); phm.append(d[di]["p_hm"])
            idok.append(1.0 if d[di]["index"] == gidx[gi] else 0.0)

    err = np.array(err); sig = np.array(sig); pid = np.array(pid)
    phm = np.array(phm); idok = np.array(idok)
    fin = np.isfinite(sig)
    out = {"n_corners": int(len(err)), "n_sigma_finite": int(fin.sum()),
           "n_sigma_nan_stage2_skipped": int((~fin).sum())}

    # ---- 1. calibration of p_id ------------------------------------------------
    bins = np.linspace(0, 1, 11)
    rel = []
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (pid >= lo) & (pid < hi)
        if m.sum() < 10:
            continue
        conf, acc = float(pid[m].mean()), float(idok[m].mean())
        rel.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()), "conf": conf, "acc": acc})
        ece += m.sum() / len(pid) * abs(acc - conf)
    out["reliability_p_id"] = rel
    out["ece_p_id"] = float(ece)

    # ---- 2. does sigma_px rank localisation error? -----------------------------
    dec = []
    if fin.sum() > 50:
        e, s = err[fin], sig[fin]
        q = np.quantile(s, np.linspace(0, 1, 11))
        for lo, hi in zip(q[:-1], q[1:]):
            m = (s >= lo) & (s <= hi)
            if m.sum() < 5:
                continue
            dec.append({"sigma_lo": float(lo), "sigma_hi": float(hi), "n": int(m.sum()),
                        "err_median": float(np.median(e[m])), "err_p95": float(np.percentile(e[m], 95))})
        out["spearman_sigma_vs_err"] = float(np.corrcoef(
            np.argsort(np.argsort(s)), np.argsort(np.argsort(e)))[0, 1])
    out["sigma_deciles"] = dec

    # ---- 3. selective prediction ----------------------------------------------
    sel = []
    order_id = np.argsort(-pid)                      # most confident first
    order_sig = np.argsort(np.where(fin, sig, np.inf))
    for cov in (1.0, 0.99, 0.95, 0.9, 0.8, 0.7, 0.5):
        k = max(int(cov * len(err)), 1)
        ki = order_id[:k]
        ks = order_sig[:max(int(cov * fin.sum()), 1)]
        sel.append({"coverage": cov,
                    "id_acc_at_coverage": float(idok[ki].mean()),
                    "err_median_at_coverage": float(np.median(err[ks])),
                    "err_p95_at_coverage": float(np.percentile(err[ks], 95))})
    out["selective"] = sel

    print(f"n={len(err)} corners  (sigma finite on {fin.sum()}, NaN on {(~fin).sum()})")
    print(f"ECE(p_id) = {ece:.4f}")
    if dec:
        print(f"spearman(sigma, err) = {out['spearman_sigma_vs_err']:+.4f}")
        print(f"  lowest-sigma decile: err median {dec[0]['err_median']:.4f} px")
        print(f"  highest-sigma decile: err median {dec[-1]['err_median']:.4f} px "
              f"({dec[-1]['err_median']/max(dec[0]['err_median'],1e-9):.1f}x)")
    print(f"\n{'coverage':>9}{'ID acc':>9}{'err med':>10}{'err p95':>10}")
    for r in sel:
        print(f"{r['coverage']*100:>8.0f}%{r['id_acc_at_coverage']*100:>8.2f}%"
              f"{r['err_median_at_coverage']:>10.4f}{r['err_p95_at_coverage']:>10.4f}")

    # ---- figure ---------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    if rel:
        ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        ax[0].plot([r["conf"] for r in rel], [r["acc"] for r in rel], "o-", color="#1f77b4")
    ax[0].set_xlabel("predicted confidence (p_id)"); ax[0].set_ylabel("empirical ID accuracy")
    ax[0].set_title(f"Calibration of identity confidence (ECE = {ece:.3f})"); ax[0].legend(fontsize=8)
    if dec:
        ax[1].plot([0.5 * (d["sigma_lo"] + d["sigma_hi"]) for d in dec],
                   [d["err_median"] for d in dec], "o-", color="#1f77b4", label="median")
        ax[1].plot([0.5 * (d["sigma_lo"] + d["sigma_hi"]) for d in dec],
                   [d["err_p95"] for d in dec], "^:", color="#1f77b4", alpha=0.6, label="p95")
        ax[1].set_yscale("log"); ax[1].legend(fontsize=8)
    ax[1].set_xlabel("predicted localisation uncertainty sigma (px)")
    ax[1].set_ylabel("actual localisation error (px)")
    ax[1].set_title("Does sigma predict error?")
    cs = [r["coverage"] * 100 for r in sel]
    ax[2].plot(cs, [r["id_acc_at_coverage"] * 100 for r in sel], "o-", color="#1f77b4", label="ID accuracy")
    ax[2].set_xlabel("coverage (% of corners kept, most confident first)")
    ax[2].set_ylabel("ID accuracy (%)"); ax[2].set_title("Selective prediction"); ax[2].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(f"Conv-ChArT uncertainty | detector step {ck.get('step')}, refiner step {rk.get('step')}, "
                 f"{args.config}, n={len(err)} corners", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=150)

    out["provenance"] = {"ckpt": args.ckpt, "ckpt_step": ck.get("step"),
                         "refiner_ckpt": args.refiner_ckpt, "refiner_step": rk.get("step"),
                         "config": args.config, "n_frames": args.n, "seed": args.seed,
                         "match_px": MATCH_PX}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {args.out}\n-> {args.fig}")


if __name__ == "__main__":
    main()
