"""tools/plot_four_way.py -- the algorithm-comparison figures.

Per factor: three panels (localisation, recall, identification), each carrying
FOUR curves -- ours coarse, ours + refiner, Deep ChArUco (fine-tuned), and
classical OpenCV. Plus a single multi-factor OVERLAY PANEL so the separation
between algorithms is visible at a glance rather than reconstructed from four
separate figures.

All four arms are scored on IDENTICAL FRAMES (same seed triple in every sweep
tool), so a difference between curves is a difference between algorithms and
nothing else.

TWO READING NOTES, both baked onto the figures because they are easy to get
wrong and both cut AGAINST over-claiming:

  1. CLASSICAL'S ID ACCURACY IS NOT COMPARABLE. OpenCV only ever reports corners
     it has already identified, so its id_acc is ~100% among matched corners by
     construction. Its failure mode is NOT DETECTING AT ALL -- read its RECALL,
     not its ID panel. The ID panel plots it dashed-faint for completeness.
  2. DEEP CHARUCO IS UNREFINED. Their published system includes a RefineNet we
     do not run, so their localisation curve should be read against OUR COARSE
     arm, not our refined one. Both of ours are drawn so the comparison the
     reader should make is the one they can see.

Reads only the sweep JSON -- no model, no GPU. Restyling never needs a re-run.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dcc.viz import save_panels

ARMS = [
    ("%OURS% (coarse)",       "#1f77b4", "-",  "ours",      "coarse",  3.6, "o"),
    ("%OURS% + refiner",      "#0b3d5c", "--", "ours",      "refined", 1.4, "x"),
    ("%FAST%",                "#2ca02c", "-",  "fast",      "coarse",  2.0, "s"),
    ("Deep ChArUco",          "#d62728", "--", "dc",        None,      1.8, "o"),
    ("classical OpenCV",      "#7f7f7f", ":",  "classical", None,      1.8, "o"),
]
SQUARES_PER_SIDE, FRAME_W = 5, 640
SNR_CAL = "paper/results_rev6/03_robustness/snr_calibration.json"


def _snr_map():
    f = Path(SNR_CAL)
    if not f.exists():
        return None
    return {r["K"]: r["snr_db_mean"] for r in json.loads(f.read_text())["rows"]}
PANEL_FACTORS = ["distance", "darkness",
                 "sensor_noise_K", "motion_blur"]

PANEL_METRIC = {"darkness": "joint", "board_occlusion": "joint"}

FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND = 15, 14, 12, 9

LOG_X_FACTORS = {"distance", "darkness"}
TRAIN_S_RANGE = (12, 128)


def _merged_distance(dirs):
    """`distance` [12,128] and `distance_extrap` [6,256] as ONE series per source, so a single
    panel shows in-envelope behaviour AND how it degrades outside (Kaelin: "add the extrapolated
    data to condense it a single value").
    """
    merged = {}
    for key, d in dirs.items():
        by_s = {}
        for factor in ("distance_extrap", "distance"):
            f = Path(d) / f"{factor}.json"
            if not f.exists():
                continue
            for st in json.loads(f.read_text())["steps"]:
                by_s[st["value"]] = st
        if by_s:
            merged[key] = ([by_s[v] for v in sorted(by_s)], "board square s (px)",
                            list(by_s[sorted(by_s)[0]]["arms"]))
    return merged


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ours", default="paper/results_rev6/03_robustness")
    p.add_argument("--dc", default="paper/results_rev6/05_comparison/robustness_dc")
    p.add_argument("--classical", default="paper/results_rev6/05_comparison/robustness_classical")
    p.add_argument("--fast", default="paper/results_rev6/03_robustness_FAST",
                   help="Conv-ChArT-FAST sweep dir; absent or empty -> the curve is silently "
                        "skipped by _load, so the panel still builds before FAST has swept")
    p.add_argument("--out-dir", default="paper/results_rev6/05_comparison/figures")
    p.add_argument("--panel-factors", nargs="+", default=PANEL_FACTORS,
                   help="factors in the overlay panel, in order")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--separate", action="store_true", help="also save each panel as its own image beside the figure")
    p.add_argument("--panel-metric", default="recall", choices=["recall", "id_acc", "err_median"])
    p.add_argument("--ours-label", default="Conv-ChArT")
    p.add_argument("--fast-label", default="Conv-ChArT-FAST")
    p.add_argument("--panel-tag", default="", help="suffix for the output filename, to keep variants apart")
    p.add_argument("--fast-refined", action="store_true", help="also draw the --fast dir's refined arm")
    return p


def _load(dirs, factor):
    """-> {source: (steps, arm_key)}; silently skips a source missing this factor."""
    out = {}
    for key, d in dirs.items():
        f = Path(d) / f"{factor}.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        arms = data["steps"][0]["arms"] if data["steps"] else {}
        out[key] = (data["steps"], data.get("unit", factor), list(arms))
    return out


def _series(steps, arm, field):
    if field == "joint":
        return [(s["arms"][arm]["recall"] or 0) * (s["arms"][arm].get("id_acc") or 0)
                for s in steps]
    return [s["arms"][arm][field] for s in steps]


def _xs(factor, steps):
    if factor == "distance":
        return [100.0 * SQUARES_PER_SIDE * s["value"] / FRAME_W for s in steps], \
               "board width (% of frame width)"
    if factor == "sensor_noise_K":
        m = _snr_map()
        if m and all(s["value"] in m for s in steps):
            return [m[s["value"]] for s in steps], "SNR (dB) at board mean level"
    return [s["value"] for s in steps], None


def draw(ax, srcs, factor, field, pct):
    for label, colour, ls, key, arm, lw, mk in ARMS:
        if key not in srcs:
            continue
        steps, unit, arm_keys = srcs[key]
        a = arm if arm else arm_keys[0]
        if a not in steps[0]["arms"]:
            continue
        xs, xl = _xs(factor, steps)
        ys = _series(steps, a, field)
        ys = [v * 100 for v in ys] if pct else ys
        faint = (key == "classical" and field == "id_acc")
        thick = lw > 2.0
        ax.plot(xs, ys, ls, color=colour, marker=mk, ms=3.5, lw=lw,
                alpha=0.35 if faint else (0.55 if thick else 1.0),
                zorder=2 if thick else 3,
                label=label + (" (not comparable)" if faint else ""))
    return _xs(factor, srcs[list(srcs)[0]][0])[1] or srcs[list(srcs)[0]][1]


def main():
    args = build_parser().parse_args()
    global ARMS
    if args.fast_refined:
        ARMS.insert(3, ("%FAST% + refiner", "#145214", "--", "fast", "refined", 1.4, "x"))
    ARMS = [(a[0].replace('%OURS%', args.ours_label).replace('%FAST%', args.fast_label),) + tuple(a[1:])
            for a in ARMS]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dirs = {"ours": args.ours, "fast": args.fast, "dc": args.dc, "classical": args.classical}
    def _is_sweep(f):
        try:
            return "steps" in json.loads(f.read_text())
        except Exception:
            return False
    factors = sorted({f.stem for d in dirs.values() for f in Path(d).glob("*.json") if _is_sweep(f)}
                     if all(Path(d).exists() for d in dirs.values()) else [])
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    for factor in factors:
        srcs = _load(dirs, factor)
        if len(srcs) < 2:
            continue
        fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))
        xl = draw(ax[0], srcs, factor, "err_median", False)
        draw(ax[1], srcs, factor, "recall", True)
        draw(ax[2], srcs, factor, "id_acc", True)
        ax[0].set_ylabel("localisation error, median (px)  \u2193 lower is better"); ax[0].set_yscale("log")
        ax[0].set_title("Corner localisation error")
        ax[1].set_ylabel("detection recall (%)  \u2191 higher is better"); ax[1].set_title("Corner detection rate")
        ax[2].set_ylabel("ID accuracy among matched (%)  \u2191 higher is better"); ax[2].set_title("Corner identification accuracy")
        for a in ax:
            a.set_xlabel(xl); a.grid(alpha=0.3)
        ax[2].legend(fontsize=7)
        fig.suptitle(f"Algorithm comparison: {factor}   |   identical frames, all arms   |   "
                     f"classical ID not comparable (it only reports already-identified corners); "
                     f"Deep ChArUco is unrefined -- read against our COARSE curve", fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(out / f"fourway_{factor}.png", dpi=args.dpi)
        if args.separate:
            for a in ax[:2]:
                a.legend(fontsize=7)
            save_panels(fig, out / f"fourway_{factor}.png", args.dpi)
        plt.close(fig)
        print(f"{factor:>18} -> {out / ('fourway_' + factor + '.png')}")

    panel = [f for f in args.panel_factors
             if f in factors or (f == "distance" and "distance_extrap" in factors)]
    PM_FIELD, PM_PCT, PM_YLABEL, PM_YLIM, PM_YSCALE = {
        "recall": ("recall", True, "detection recall (%)  ↑ higher is better", (0, 102), None),
        "id_acc": ("id_acc", True, "ID accuracy among matched (%)  ↑ higher is better", (0, 102), None),
        "err_median": ("err_median", False, "localisation error, median (px)  ↓ lower is better",
                       None, "log"),
    }[args.panel_metric]
    if panel:
        ncol = 2 if len(panel) <= 4 else 3
        nrow = -(-len(panel) // ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 4.1 * nrow), squeeze=False)
        flat = [a for row in axes for a in row]
        for a, factor in zip(flat, panel):
            srcs = _merged_distance(dirs) if factor == "distance" else _load(dirs, factor)
            if not srcs:
                a.axis("off"); continue
            fld = PANEL_METRIC.get(factor, PM_FIELD) if args.panel_metric == "recall" else PM_FIELD
            xl = draw(a, srcs, factor, fld, True if fld != "err_median" else PM_PCT)
            if factor == "distance":
                for s_px in TRAIN_S_RANGE:
                    a.axvline(100.0 * SQUARES_PER_SIDE * s_px / FRAME_W,
                              color="k", ls="--", lw=1.0, alpha=0.55, zorder=1)
                a.axvspan(100.0 * SQUARES_PER_SIDE * TRAIN_S_RANGE[0] / FRAME_W,
                          100.0 * SQUARES_PER_SIDE * TRAIN_S_RANGE[1] / FRAME_W,
                          color="k", alpha=0.05, zorder=0)
                a.set_title("distance  (dashed = trained envelope)", fontsize=FS_TITLE)
            else:
                suffix = "" if fld == PM_FIELD else {"id_acc": "  (id)", "joint": "  (found + ID)"}[fld]
                a.set_title(factor + suffix, fontsize=FS_TITLE)
            if factor in LOG_X_FACTORS:
                a.set_xscale("log")
            a.set_xlabel(xl, fontsize=FS_LABEL); a.grid(alpha=0.3)
            a.tick_params(labelsize=FS_TICK)
            if PM_YLIM:
                a.set_ylim(*PM_YLIM)
            if PM_YSCALE:
                a.set_yscale(PM_YSCALE)
        for a in flat[len(panel):]:
            a.axis("off")
        ylab = ("recall / found + ID'd (%)  \u2191"
                if (args.panel_metric == "recall" and any(f in PANEL_METRIC for f in panel)) else PM_YLABEL)
        fig.supylabel(ylab, fontsize=FS_LABEL)
        leg_i = next((panel.index(f) for f in ("tilt", "rotation", "defocus_blur", "motion_blur")
                      if f in panel), min(1, len(panel) - 1))
        leg = dict(fontsize=FS_LEGEND, loc="best", framealpha=0.9, labelspacing=0.3, handlelength=1.6,
                   handletextpad=0.5, borderpad=0.4, borderaxespad=0.4)
        flat[leg_i].legend(**leg)
        fig.tight_layout()
        _name = f"fourway_PANEL_{args.panel_metric}{args.panel_tag}.png"
        fig.savefig(out / _name, dpi=args.dpi)
        if args.separate:
            for a in flat[:len(panel)]:
                a.set_ylabel(ylab, fontsize=FS_LABEL)
                a.legend(**dict(leg, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2))
            save_panels(fig, out / _name, args.dpi)
        plt.close(fig)
        print(f"{'PANEL':>18} -> {out / _name} ({len(panel)} factors, metric={args.panel_metric})")


if __name__ == "__main__":
    main()
