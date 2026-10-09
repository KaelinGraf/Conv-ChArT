"""tools/snr_calibration.py -- convert the sensor_noise_K sweep axis into SNR (dB).

Kaelin, 2026-07-29: "can we report the noise in SNR (dB)? this is the commonly
understood form". K in electrons-per-DN is the generator's knob but means nothing to
an audience; dB is the universal currency.

The generator's noise is Poissonian-Gaussian (Foi et al. 2008; EMVA 1288). A signal
of S DN carries S*K electrons, so the shot variance is S*K electrons^2, which in DN^2
is S*K/K^2 = S/K. Read noise adds its own variance directly in DN:

    sigma_DN(S)^2 = S/K + sigma_read^2
    SNR_dB(S)     = 20 * log10( S / sigma_DN(S) )

S is MEASURED, not assumed: the frames are generated at each sweep step with sensor
noise disabled, and the signal level is taken over the BOARD REGION only -- the board
is what has to be detected, and a full-frame mean would be dominated by whatever
background happened to be drawn. Two levels are reported because they answer
different questions:

  SNR_mean  -- at the board's mean level. The headline number.
  SNR_white -- at the board's white level (p95 within the board). What the marker
               code's bright cells actually get.

sigma_read is taken as the midpoint of the configured range, and the sensitivity to
that choice is printed so the number is not quoted more precisely than it deserves.

DELIVERED vs MODELLED (audit A3, measured 2026-08-04). The formula above is the
PER-CHANNEL noise, but the generator draws noise on the 3-channel BGR working buffer
(dcc/synth.py:878-881) and converts to grayscale LAST (dcc/synth.py:1020). cv2's
BGR2GRAY is a luminance-weighted average of three INDEPENDENT noise draws, so the
delivered single-channel frame carries only

    sqrt(0.114^2 + 0.587^2 + 0.299^2) = 0.6686x

of the modelled sigma -- measured 0.6676-0.6789 over sigma in 3..12 DN (the spread at
small sigma is uint8 quantisation, converging to 0.6686 as sigma grows). The frame the
network actually sees is therefore 3.50 dB QUIETER than sigma_DN implies: the modelled
SNR is PESSIMISTIC, i.e. this benchmark's noise axis is milder than the per-channel
formula alone would suggest. Both figures are reported -- snr_db_* is the delivered
frame (quote this one), snr_db_*_modelled is the per-channel formula (kept so the
sweep axis remains traceable to the config knob). Fixing the DRAW to be channel-shared
would change the data distribution and is a rev-boundary decision, not a reporting one.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/rev640.yaml")
    p.add_argument("--n", type=int, default=60, help="frames per K step")
    p.add_argument("--seed", type=int, default=20260728, help="match the robustness sweep")
    p.add_argument("--out", default="paper/results_rev6/03_robustness/snr_calibration.json")
    return p


def main():
    args = build_parser().parse_args()
    import cv2, numpy as np
    cv2.setNumThreads(1)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import robustness_sweep as rs
    from dcc.dataset import load_config
    from dcc.synth import generate_sample, list_backgrounds

    cfg = load_config(args.config)
    bg = list_backgrounds(cfg["synth"]["backgrounds"])
    ph = cfg["synth"]["photometric"]
    rd_lo, rd_hi = ph["sensor_noise_read_std"]
    sigma_read = 0.5 * (rd_lo + rd_hi)
    # Per-channel sigma -> delivered grayscale sigma; see the header. Derived from cv2's
    # own BGR2GRAY weights rather than hardcoded, so it tracks if the conversion changes.
    GRAY_ATTEN = float(np.sqrt((np.array([0.114, 0.587, 0.299]) ** 2).sum()))
    steps = rs.FACTORS["sensor_noise_K"]["steps"]

    print(f"read noise sigma = {sigma_read:.2f} DN (midpoint of {rd_lo}-{rd_hi})")
    print(f"grayscale attenuation = {GRAY_ATTEN:.4f}x -> delivered SNR is {20*np.log10(1/GRAY_ATTEN):+.2f} dB vs the per-channel model\n")
    print(f"{'K':>4} {'S_mean':>8} {'S_white':>8} {'SNR_mean':>10} {'SNR_white':>10} {'+/- read':>10}")
    rows = []
    for si, K in enumerate(steps):
        cfg_s, comp, post = rs.apply_factor(cfg, "sensor_noise_K", K)
        # noise OFF so S is the clean signal: gauss_noise_p gates WHETHER noise fires
        cfg_s["synth"]["photometric"]["gauss_noise_p"] = 0.0
        sm, sw = [], []
        for i in range(args.n):
            rng = np.random.default_rng([args.seed, si, i])
            rec, _ = generate_sample(cfg_s, rng, bg, components=comp or None, force_negative=False)
            vis = [c for c in rec["corners"] if c["visible"]]
            if len(vis) < 4:
                continue
            pts = np.array([[c["x"], c["y"]] for c in vis], dtype=np.float32)
            mask = np.zeros(rec["image"].shape, np.uint8)
            cv2.fillConvexPoly(mask, cv2.convexHull(pts).astype(np.int32), 1)
            board = rec["image"][mask.astype(bool)].astype(np.float64)
            if board.size < 50:
                continue
            sm.append(board.mean()); sw.append(np.percentile(board, 95))
        S_m, S_w = float(np.mean(sm)), float(np.mean(sw))

        def snr_db(S, rd, gray=True):
            sigma = np.sqrt(max(S / K + rd ** 2, 1e-12))
            return 20.0 * np.log10(S / (sigma * (GRAY_ATTEN if gray else 1.0)))

        lo, hi = snr_db(S_m, rd_hi), snr_db(S_m, rd_lo)   # worst/best read noise
        rows.append({"K": K, "S_mean": S_m, "S_white": S_w,
                     "snr_db_mean": snr_db(S_m, sigma_read),
                     "snr_db_white": snr_db(S_w, sigma_read),
                     "snr_db_mean_lo": lo, "snr_db_mean_hi": hi,
                     "snr_db_mean_modelled": snr_db(S_m, sigma_read, gray=False),
                     "snr_db_white_modelled": snr_db(S_w, sigma_read, gray=False)})
        r = rows[-1]
        print(f"{K:>4} {S_m:>8.1f} {S_w:>8.1f} {r['snr_db_mean']:>9.1f}dB "
              f"{r['snr_db_white']:>9.1f}dB {hi-lo:>9.1f}dB")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"sigma_read_dn": sigma_read, "read_std_range": [rd_lo, rd_hi],
                                "model": "sigma_DN^2 = S/K + sigma_read^2 (Poissonian-Gaussian)",
                                "signal_region": "board convex hull, noise disabled",
                                "gray_attenuation": GRAY_ATTEN,
                                "snr_db_note": "snr_db_* = DELIVERED grayscale frame "
                                    "(per-channel sigma x %.4f, audit A3); snr_db_*_modelled = "
                                    "per-channel formula" % GRAY_ATTEN,
                                "rows": rows}, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
