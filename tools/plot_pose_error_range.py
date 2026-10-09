"""tools/plot_pose_error_range.py -- the measurement-noise figure for the Kalman section: how the 882k model's
pose error scales with range, and why a constant R is the wrong choice.

Three panels from `26_pose_error_variance/pose_REL882_B1_per_image.jsonl` (accepted = unambiguous solves):
rotation and translation robust sigma per apparent-scale octave with the fitted power laws
(`kalman_R_REL882.json: range_scaled_R`), and the median normalised error |e|/sigma per octave under the pooled
constant R vs the range-scaled R (1.0 = matched). Reads only the banked files.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dcc.viz import save_panels

COL = ["#0072B2", "#E69F00", "#009E73"]     # x, y, z: fixed, CVD-safe
D = Path("paper/results_rev6/26_pose_error_variance")
EDGES = [12, 16, 32, 64, 128.0001]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(D / "pose_error_vs_range.png"))
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--separate", action="store_true", help="also save each panel as its own image beside the figure")
    a = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    rows = [json.loads(l) for l in (D / "pose_REL882_B1_per_image.jsonl").read_text().splitlines() if l.strip()]
    U = [r for r in rows if not r["ambiguous"]]
    E = np.array([r["drot_rad"] + r["dt_sq"] for r in U]); s = np.array([r["s_px"] for r in U]); z = np.array([r["f_px"] / r["s_px"] for r in U])
    fit = json.loads((D / "kalman_R_REL882.json").read_text())["range_scaled_R"]
    A = np.array(fit["a_rot_rad"] + fit["a_trans_sq"]); K = np.array(fit["k_rot"] + fit["k_trans"])
    mad = lambda X: 1.4826 * np.median(np.abs(X - np.median(X, 0)), 0)
    octs = [(lo, hi, (s >= lo) & (s < hi)) for lo, hi in zip(EDGES[:-1], EDGES[1:])]
    zc = np.array([np.median(z[m]) for _, _, m in octs]); sig = np.array([mad(E[m]) for _, _, m in octs])   # (4, 6)
    sig_const = mad(E); sig_z = A * z[:, None] ** K
    zz = np.geomspace(z.min(), z.max(), 50)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for i, name in enumerate("xyz"):
        ax[0].plot(zc, np.degrees(sig[:, i]), "o", color=COL[i], ms=6, label=f"rot {name}: measured per octave")
        ax[0].plot(zz, np.degrees(A[i] * zz ** K[i]), "-", color=COL[i], lw=1.5, label=f"fit  z^{K[i]:.2f}")
        ax[1].plot(zc, sig[:, 3 + i], "o", color=COL[i], ms=6, label=f"trans {name}: measured per octave")
        ax[1].plot(zz, A[3 + i] * zz ** K[3 + i], "-", color=COL[i], lw=1.5, label=f"fit  z^{K[3 + i]:.2f}")
    for a_, yl, t in [(ax[0], "robust sigma, rotation (deg)", "Rotation error vs range"), (ax[1], "robust sigma, translation (board squares)", "Translation error vs range")]:
        a_.set_xscale("log"); a_.set_yscale("log"); a_.set_xlabel("range z = depth / square length (board squares)"); a_.set_ylabel(yl); a_.set_title(t); a_.grid(alpha=0.3, which="both"); a_.legend(fontsize=7)
    x = np.arange(4); w = 0.38
    nc = [np.median(np.abs(E[m]) / sig_const, 0) / 0.6745 for _, _, m in octs]; nz = [np.median(np.abs(E[m]) / sig_z[m], 0) / 0.6745 for _, _, m in octs]
    ax[2].bar(x - w / 2, [np.median(v) for v in nc], w, color="#D55E00", label="constant R (pooled)")
    ax[2].bar(x + w / 2, [np.median(v) for v in nz], w, color="#0072B2", label="range-scaled R (fit)")
    for j, (v1, v2) in enumerate(zip(nc, nz)):
        ax[2].plot([x[j] - w / 2] * 6, v1, "k_", ms=10, alpha=0.6); ax[2].plot([x[j] + w / 2] * 6, v2, "k_", ms=10, alpha=0.6)
    ax[2].axhline(1.0, color="k", ls="--", lw=1); ax[2].set_yscale("log")
    ax[2].set_xticks(x, [f"s {int(lo)}-{int(hi)} px\n(far)" if lo == 12 else (f"s {int(lo)}-{int(hi)} px\n(near)" if hi > 100 else f"s {int(lo)}-{int(hi)} px") for lo, hi, _ in octs])
    ax[2].set_ylabel("median |error| / sigma  (1 = R matched)"); ax[2].set_title("Is R matched?  (bar: median of 6 components; ticks: each)", fontsize=11)
    ax[2].grid(axis="y", alpha=0.3, which="both"); ax[2].legend(fontsize=8)
    fig.suptitle(f"Conv-ChArT 882k pose measurement noise, {len(U)} accepted solves of the 1000-frame benchmark: a constant covariance is 2-7x off at the ends of the range", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); Path(a.out).parent.mkdir(parents=True, exist_ok=True); fig.savefig(a.out, dpi=a.dpi)
    if a.separate:
        save_panels(fig, a.out, a.dpi)
    plt.close(fig)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
