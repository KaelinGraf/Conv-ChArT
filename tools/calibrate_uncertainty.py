"""tools/calibrate_uncertainty.py -- turn the refiner's raw peak-spread into a
VARIANCE a Kalman filter can consume, and prove it on held-out data.

Raw `sigma_px` is the spread of the predicted heatmap over the soft-argmax
window. Measured, it ORDERS corners by error well (rank corr ~0.63, 12x median
separation across deciles) but its SCALE is meaningless: it spans ~1.9x while
actual error spans ~12x, over-stating error at the low end and badly
under-stating the tail. Fed to a filter as R that is not conservative, it is
actively dangerous -- over-confidence is how a filter diverges.

So: keep the ordering, fix the scale. Isotonic (monotone) regression sigma_raw ->
sigma_cal, fitted so that within each bin **sigma_cal = RMS(actual error) / sqrt(2)**
-- the /sqrt(2) because the corner error is 2-D and we report an isotropic
per-axis sigma, so ||e||^2 / sigma^2 should be chi-square with 2 DOF (mean 2).

RMS, not median: variance is a second moment and the tail dominates it. The
earlier evaluation reported medians, which demonstrates DISCRIMINATION but says
nothing about CALIBRATION -- different claims, and only the second one licenses
using sigma as R.

FIT AND TEST SPLITS ARE DISJOINT (different seeds). A calibration validated on
its own fit data would be a tautology.
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
    p.add_argument("--n-fit", type=int, default=400)
    p.add_argument("--n-test", type=int, default=250)
    p.add_argument("--bins", type=int, default=12)
    p.add_argument("--out", default="paper/results_rev6/10_uncertainty/sigma_calibration.json")
    return p


def _collect(cfg, model, refiner, bg, seed, n, td, detect, np, generate_sample):
    sig, err = [], []
    for i in range(n):
        rng = np.random.default_rng([seed, i])
        rec, _ = generate_sample(cfg, rng, bg, force_negative=False)
        vis = [c for c in rec["corners"] if c["visible"]]
        if not vis:
            continue
        gt = np.array([[c["x"], c["y"]] for c in vis], dtype=np.float64)
        res = detect(rec["image"], model, refiner, K=None, dist=None, cfg=cfg)
        d = res["corners"]
        if not d:
            continue
        xy = np.array([[c["x"], c["y"]] for c in d])
        for gi, di, dist in td._match_greedy(gt, xy, MATCH_PX):
            s = d[di]["sigma_px"]
            if np.isfinite(s):
                sig.append(s); err.append(dist)
    return np.array(sig), np.array(err)


def main():
    args = build_parser().parse_args()
    import cv2, numpy as np, torch, importlib.util
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

    s_fit, e_fit = _collect(cfg, model, refiner, bg, 11111, args.n_fit, td, detect, np, generate_sample)
    s_tst, e_tst = _collect(cfg, model, refiner, bg, 99999, args.n_test, td, detect, np, generate_sample)
    print(f"fit n={len(s_fit)}   test n={len(s_tst)}  (disjoint seeds)")

    # --- fit: equal-count bins, sigma_cal = RMS(err)/sqrt(2) per bin -----------
    q = np.quantile(s_fit, np.linspace(0, 1, args.bins + 1))
    q[0], q[-1] = s_fit.min() - 1e-9, s_fit.max() + 1e-9
    xs, ys = [], []
    for lo, hi in zip(q[:-1], q[1:]):
        m = (s_fit >= lo) & (s_fit < hi)
        if m.sum() < 20:
            continue
        xs.append(float(s_fit[m].mean()))
        ys.append(float(np.sqrt((e_fit[m] ** 2).mean() / 2.0)))
    # enforce monotonicity (pool-adjacent-violators, the isotonic step): the raw
    # ordering is sound, so a non-monotone bin is sampling noise, not signal
    ys = np.array(ys, dtype=float)
    for _ in range(len(ys)):
        bad = np.nonzero(np.diff(ys) < 0)[0]
        if not len(bad):
            break
        i = bad[0]
        ys[i:i + 2] = ys[i:i + 2].mean()
    xs = np.array(xs, dtype=float)
    print("\ncalibration map (sigma_raw -> sigma_cal, px):")
    for a, b in zip(xs, ys):
        print(f"  {a:.4f} -> {b:.4f}   ({b/a:.2f}x)")

    def apply(s):
        return np.interp(s, xs, ys)

    # --- validate on the held-out split ---------------------------------------
    def chi2(sig, err):
        c = (err ** 2) / np.maximum(apply(sig) ** 2, 1e-12)     # ||e||^2 / sigma^2, expect chi2_2
        return c

    c_raw = (e_tst ** 2) / np.maximum(s_tst ** 2, 1e-12)
    c_cal = chi2(s_tst, e_tst)
    print(f"\nper-corner NEES on HELD-OUT data (expect mean 2.0 for chi2_2):")
    print(f"  raw sigma        : mean {c_raw.mean():8.3f}  median {np.median(c_raw):7.3f}")
    print(f"  calibrated sigma : mean {c_cal.mean():8.3f}  median {np.median(c_cal):7.3f}")
    print(f"  within chi2_2 95% bound (5.991): raw {100*(c_raw<=5.991).mean():.1f}%  "
          f"cal {100*(c_cal<=5.991).mean():.1f}%   (ideal 95%)")

    out = {"knots_sigma_raw": xs.tolist(), "knots_sigma_cal": ys.tolist(),
           "fit": {"n": int(len(s_fit)), "seed": 11111, "n_frames": args.n_fit},
           "test": {"n": int(len(s_tst)), "seed": 99999, "n_frames": args.n_test,
                     "nees_raw_mean": float(c_raw.mean()), "nees_cal_mean": float(c_cal.mean()),
                     "nees_raw_median": float(np.median(c_raw)), "nees_cal_median": float(np.median(c_cal)),
                     "frac_within_chi2_2_95_raw": float((c_raw <= 5.991).mean()),
                     "frac_within_chi2_2_95_cal": float((c_cal <= 5.991).mean()),
                     "expected_dof": 2},
           "provenance": {"ckpt": args.ckpt, "ckpt_step": ck.get("step"),
                           "refiner_ckpt": args.refiner_ckpt, "refiner_step": rk.get("step"),
                           "config": args.config, "match_px": MATCH_PX},
           "usage": "piecewise-linear (np.interp) on sigma_px; isotonic-corrected, RMS-matched, "
                    "so ||err||^2 / sigma_cal^2 is chi-square_2. Apply BEFORE building R for PnP."}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
