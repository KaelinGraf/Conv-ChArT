"""tools/plot_robustness.py -- figures from tools/robustness_sweep.py's JSON.

One figure per factor, three panels: localisation error (median + p95),
detection recall, and ID accuracy -- each with a WITH-REFINER and a
COARSE-ONLY curve so the refiner's contribution is visible on every axis
rather than asserted once. Rate panels carry +/-1 SE bars (normal approx on
the binomial); the error panel carries the p95 as a lighter dashed line,
since the tail is what the sign-off gate is defined on, not the median.

Reads only the JSON, never the model -- figures can be restyled without
re-running a sweep. The raw per-corner errors are in the JSON for the same
reason.
"""
import argparse
import json
from pathlib import Path

ARMS = [("refined", "with refiner", "#1f77b4", "-"),
        ("coarse", "coarse only (no refiner)", "#d62728", "--")]

# "s = 20 px" means nothing to an audience. The board is SQUARES_PER_SIDE squares
# across, so board width = SQUARES_PER_SIDE * s, and expressing that as a
# percentage of the frame width is immediately legible: 9% of frame = far away,
# 100% = filling it. s_px is kept on a secondary top axis so the underlying
# quantity is still readable and the figure stays checkable against the JSON.
SQUARES_PER_SIDE = 5
FRAME_W = 640


def _board_pct(s_px):
    return 100.0 * SQUARES_PER_SIDE * s_px / FRAME_W


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-dir", default="paper/results_rev4/robustness")
    p.add_argument("--out-dir", default=None, help="default: <in-dir>/figures")
    p.add_argument("--dpi", type=int, default=150)
    return p


def plot_one(data, out_path, dpi):
    import matplotlib.pyplot as plt
    import numpy as np

    steps = data["steps"]
    is_dist = data["factor"] == "distance"
    # sensor_noise_K -> SNR (dB). K (electrons/DN) is the generator knob; dB is what a
    # reader understands. Mapping MEASURED by tools/snr_calibration.py (signal taken over
    # the board region with noise disabled, then sigma_DN^2 = S/K + sigma_read^2). Falls
    # back to raw K if the calibration file is absent.
    snr = None
    if data["factor"] == "sensor_noise_K":
        cal = Path("paper/results_rev6/03_robustness/snr_calibration.json")
        if cal.exists():
            m = {r["K"]: r["snr_db_mean"] for r in json.loads(cal.read_text())["rows"]}
            if all(st["value"] in m for st in steps):
                snr = [m[st["value"]] for st in steps]
    xs = ([_board_pct(s["value"]) for s in steps] if is_dist
          else snr if snr is not None else [s["value"] for s in steps])
    xlabel = "board width (% of frame width)" if is_dist else data["unit"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    for key, label, colour, ls in ARMS:
        med = [s["arms"][key]["err_median"] for s in steps]
        p95 = [s["arms"][key]["err_p95"] for s in steps]
        axes[0].plot(xs, med, ls, color=colour, marker="o", ms=4, label=f"{label} (median)")
        axes[0].plot(xs, p95, ":", color=colour, alpha=0.55, marker="^", ms=3, label=f"{label} (p95)")

        rec = np.array([s["arms"][key]["recall"] for s in steps]) * 100
        rse = np.array([s["arms"][key]["recall_se"] for s in steps]) * 100
        axes[1].errorbar(xs, rec, yerr=rse, fmt="o" + ls, color=colour, ms=4, capsize=3, label=label)

        idc = np.array([s["arms"][key]["id_acc"] for s in steps]) * 100
        ise = np.array([s["arms"][key]["id_acc_se"] for s in steps]) * 100
        axes[2].errorbar(xs, idc, yerr=ise, fmt="o" + ls, color=colour, ms=4, capsize=3, label=label)

    axes[0].set_ylabel("localisation error (px)  \u2193 lower is better"); axes[0].set_yscale("log")
    axes[0].set_title("Corner localisation error")
    axes[1].set_ylabel("detection recall (%)  \u2191 higher is better"); axes[1].set_title("Corner detection rate")
    axes[2].set_ylabel("ID accuracy (%)  \u2191 higher is better"); axes[2].set_title("Corner identification accuracy")
    if snr is not None:
        # dB already runs cleaner-to-the-right, so the invert_x that made the raw K axis
        # read noisier-to-the-right must NOT also be applied
        xlabel = "SNR (dB) at board mean level  \u2191 cleaner"
    for ax in axes:
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        if data.get("invert_x") and snr is None:
            ax.invert_xaxis()
        if is_dist:
            # secondary top axis in the underlying unit, so the figure stays
            # checkable against the JSON without re-deriving the conversion
            sec = ax.secondary_xaxis("top", functions=(lambda v: v * FRAME_W / (100.0 * SQUARES_PER_SIDE),
                                                        lambda v: _board_pct(v)))
            sec.set_xlabel("board square s (px)", fontsize=8)
            sec.tick_params(labelsize=7)

    p = data["provenance"]
    n_gt = steps[0]["arms"]["refined"]["n_gt"] if steps else 0
    fig.suptitle(f"Conv-ChArT robustness: {data['factor']}   |   detector step {p['ckpt_step']}, "
                 f"refiner step {p['refiner_step']}, {p['config']}, "
                 f"n={p['n_per_step']} frames/step (~{n_gt} corners), match {p['match_px']} px",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main():
    args = build_parser().parse_args()
    import matplotlib
    matplotlib.use("Agg")

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # sweeps only: the dir also holds sidecars (snr_calibration.json), and reading one
    # as a sweep raises KeyError: 'factor'
    def _is_sweep(f):
        try:
            return "steps" in json.loads(f.read_text())
        except Exception:
            return False
    files = sorted(f for f in in_dir.glob("*.json") if _is_sweep(f))
    if not files:
        raise SystemExit(f"no factor JSON in {in_dir}")
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        out = out_dir / f"robustness_{data['factor']}.png"
        plot_one(data, out, args.dpi)
        print(f"{data['factor']:>16} -> {out}")


if __name__ == "__main__":
    main()
