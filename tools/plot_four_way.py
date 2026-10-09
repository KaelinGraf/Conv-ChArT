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

# (label, colour, linestyle, source-dir key, arm key or None = auto, linewidth, marker)
#
# THE TWO OURS ARMS COINCIDE ON RECALL AND ID AND MUST STILL BOTH BE VISIBLE.
# Identity is now BIT-IDENTICAL between them by construction (the whole ID chain runs
# on coarse coords -- see dcc/pipeline.py), so the two ID curves differ by ~0.0005 pp
# and recall by ~0.07 pp. Drawn with equal weight, whichever is plotted second hides
# the other completely and the figure looks like it is missing an arm (Kaelin reported
# exactly this on fourway_contrast). So the coarse arm is drawn FIRST, thick and
# semi-transparent, and the refined arm rides on top thin and dashed: where they agree
# you see the thick band with the dashed line inside it, where they diverge they
# separate visibly. Not cosmetic -- coincidence IS the result for identity.
ARMS = [
    ("%OURS% (coarse)",       "#1f77b4", "-",  "ours",      "coarse",  3.6, "o"),
    ("%OURS% + refiner",      "#0b3d5c", "--", "ours",      "refined", 1.4, "x"),
    # FAST is drawn from its COARSE arm, matching the Deep ChArUco convention: their
    # RefineNet is not run, so every cross-system curve on this panel is unrefined. Its
    # refined arm exists in the sweep JSON and differs on localisation only -- identity is
    # bit-identical between the two by construction, and recall by ~0.07 pp -- so a second
    # FAST curve would add a line that overlaps itself on eight of the nine panels.
    ("%FAST%",                "#2ca02c", "-",  "fast",      "coarse",  2.0, "s"),
    ("Deep ChArUco",          "#d62728", "--", "dc",        None,      1.8, "o"),
    ("classical OpenCV",      "#7f7f7f", ":",  "classical", None,      1.8, "o"),
]
SQUARES_PER_SIDE, FRAME_W = 5, 640
#: sensor_noise_K is the generator's knob (electrons per DN) and means nothing to a
#: reader; SNR in dB is the universal currency (Kaelin, 2026-07-29). The mapping is
#: MEASURED, not assumed -- tools/snr_calibration.py generates frames with noise
#: disabled, takes the signal level over the board region, and applies the
#: Poissonian-Gaussian model sigma_DN^2 = S/K + sigma_read^2. Absent file -> fall back
#: to plotting K, so the figures still build on a fresh checkout.
SNR_CAL = "paper/results_rev6/03_robustness/snr_calibration.json"


def _snr_map():
    f = Path(SNR_CAL)
    if not f.exists():
        return None
    return {r["K"]: r["snr_db_mean"] for r in json.loads(f.read_text())["rows"]}
# The overlay panel's factors. Default is the four regimes a deployed rig actually
# meets -- scale, sensor noise, motion blur, occlusion -- rather than two occlusion
# variants plus tilt (tilt is the weakest separation of the seventeen: 97.5 vs 92.8).
# 3x3 overlay panel (Kaelin, 2026-07-29). Chosen to span the three families a deployed
# rig meets rather than to flatter: GEOMETRY (scale, twist, tilt), SENSOR (SNR, motion
# blur, defocus) and SCENE (real-object occlusion, brightness, ink contrast).
# "distance" here is the MERGED axis -- see _merged_distance.
# object_occlusion (SAM2 cutouts) replaces the rectangular-hole `occlusion` factor:
# real objects are the honest test, synthetic rectangles the easy one.
# 3x3 overlay layout, specified by Kaelin 2026-07-30. Reading order matters: distance first
# (the headline axis, with the trained envelope marked), then the two regimes where the
# baselines actually break -- darkness and motion blur -- then sensor/optics, then the
# rig-specific row (NIR ink, differencing ratio) closing on tilt as the "geometry is simply
# solved" reassurance.
#
# DROPPED from the previous layout, on measurement: `rotation` (L recall moves 2.8% across
# its whole axis, 7.7 pp separation from Deep ChArUco -- the weakest of all 19 factors) and
# `object_occlusion` (3.7% / 13.1 pp, AND its unit was a COUNT of SAM2 objects, so a step
# from 1->2 could mean 2% or 40% of the board). Replaced by `diff_ratio` (21.9 / 29.3 pp,
# and it describes the actual sensing modality) and an occlusion axis measured as PERCENT
# OF BOARD OCCLUDED.
# COLLAPSED 3x3 -> 2x2 (Kaelin 2026-07-31): "for the conference frame, we don't really have
# room for 3x3. Let's collapse it to 2x2, keeping distance, darkness, sensor noise and motion
# blur." The dropped five (board_occlusion, defocus_blur, ink_contrast, diff_ratio, tilt) are
# still swept and still plottable -- pass them to --panel-factors to get the full grid back;
# the column count follows the factor count automatically. Nothing was re-measured to make
# this change, so the four surviving panels are the identical curves from the 3x3.
PANEL_FACTORS = ["distance", "darkness",
                 "sensor_noise_K", "motion_blur"]

# PER-PANEL METRIC OVERRIDE. Everything defaults to recall ("did it find the corners"), but
# darkness is plotted as ID accuracy: under a deep brightness ramp the interesting failure is
# not whether a corner is FOUND but whether it can still be NAMED, which is where Deep ChArUco
# collapses (33.3% ID at its worst darkness step, vs ours at 85.8%).
#
# Mixing recall and id_acc in one panel grid is safe where mixing in localisation was not: both are
# percentages on 0-100 and both are HIGHER-IS-BETTER, so no panel silently inverts the
# reader's sense of direction. The panel title carries "(id)" so the swap is never implicit.
# board_occlusion joins darkness on ID for the same reason, and it is measured rather than
# assumed: at 50% of the board occluded Conv-ChArT still RECALLS 89.4% of the corners that
# remain visible, but only IDENTIFIES 56.0% of them. Recall is near-flat because occlusion
# removes corners from the ground truth (visibility is gated, as in the generator) rather
# than making the survivors harder to find -- so a recall curve here would say nothing,
# which is exactly the flat-line failure of the old count-based occlusion cell.
PANEL_METRIC = {"darkness": "joint", "board_occlusion": "joint"}

# PANEL TYPOGRAPHY, one place (Kaelin 2026-07-31: "make the text bigger on all 4"). These
# sizes are for the multi-factor PANEL only -- the single-factor fourway_<f>.png figures keep
# their own smaller sizes, since those are read on a screen at full size rather than projected.
FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND = 15, 14, 12, 9

# Factors whose x axis is LOG. Both are swept geometrically, so a linear axis misrepresents
# the sampling: darkness steps by a constant 0.6 ratio (1.0, 0.6, 0.36, ... 0.01), which puts
# six of its ten measured points inside the leftmost 13% of a linear axis -- exactly the region
# where every arm is separating. distance is the merged in+out-of-envelope decade sweep.
# Neither reaches 0, so log is safe (darkness bottoms out at 0.01).
LOG_X_FACTORS = {"distance", "darkness"}
TRAIN_S_RANGE = (12, 128)      # configs/rev640.yaml scale_range_px


def _merged_distance(dirs):
    """`distance` [12,128] and `distance_extrap` [6,256] as ONE series per source, so a
    single panel shows in-envelope behaviour AND how it degrades outside (Kaelin: "add
    the extrapolated data to condense it a single value"). Where both factors carry the
    same s the IN-ENVELOPE value wins -- it is the canonical sweep, and the two were run
    on different seeds, so mixing them at a shared s would put two different frame sets
    on one curve."""
    merged = {}
    for key, d in dirs.items():
        by_s = {}
        for factor in ("distance_extrap", "distance"):     # distance second == wins ties
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
    # Which quantity the 3x3 overlay plots. Recall answers "did it find the corners", id_acc
    # "did it name them", err_median "how precisely did it place them". They rank the arms
    # very differently: measured 2026-07-30, diff_ratio separates us from Deep ChArUco by
    # 21.9 pp on recall but 29.3 pp on ID, and localisation is our largest margin of all
    # (0.076-0.139 px refined vs their 1.55-1.84) yet is invisible in a recall-only panel.
    p.add_argument("--panel-metric", default="recall", choices=["recall", "id_acc", "err_median"])
    # The two ours-family slots carry whichever TIER is being plotted. Hardcoding
    # "Conv-ChArT" / "Conv-ChArT-FAST" was fine with one dev model per slot, but the release
    # tiers are named by parameter count, and a figure that mislabels which model produced a
    # curve is worse than no figure at all.
    p.add_argument("--ours-label", default="Conv-ChArT")
    p.add_argument("--fast-label", default="Conv-ChArT-FAST")
    p.add_argument("--panel-tag", default="", help="suffix for the output filename, to keep variants apart")
    # The FAST slot is coarse-only by the Deep ChArUco convention above. For a SAME-SYSTEM comparison
    # (one model on two boards, or two tiers) the refined arm is the one deployed, so it must be
    # drawable: this adds "%FAST% + refiner" in the same thin-dashed style as ours.
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
    # "joint" is DERIVED, not stored: recall = n_match/n_gt and id_acc = n_id/n_match, so
    # their product is exactly n_id/n_gt -- the fraction of visible corners both FOUND and
    # correctly NAMED. Computable from any banked sweep without re-running one.
    #
    # WHY IT EXISTS (Kaelin, 2026-07-30): id_acc alone is conditioned on the detector's OWN
    # detections, which rewards conservatism -- a detector that only reports the corners it
    # is sure of scores a high ID on an easier subset. That is not a subtle effect here. At
    # 50% board occlusion Conv-ChArT attempts 89.4% of corners and IDs 56.0% of them, while
    # Deep ChArUco attempts 73.1% and IDs 63.8%: on conditioned ID we LOSE, on joint we win
    # 50.1% to 46.6%. Same at the noise floor (16.5% vs 10.1% where conditioned ID said
    # 60.2% vs 73.8%). Both apparent losses were the selection effect, not the models.
    #
    # It is also the quantity the downstream pose actually consumes -- a corner found but
    # misnamed and a corner never found are equally useless to the PnP -- and it collapses
    # classical OpenCV's meaningless 100% ID to its true 9.6-37.9%, so all four arms sit on
    # one comparable axis with no "not comparable" asterisk.
    if field == "joint":
        return [(s["arms"][arm]["recall"] or 0) * (s["arms"][arm].get("id_acc") or 0)
                for s in steps]
    return [s["arms"][arm][field] for s in steps]


def _xs(factor, steps):
    if factor == "distance":       # % of frame width reads far better than "s = 20 px"
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
        faint = (key == "classical" and field == "id_acc")   # see reading note 1
        thick = lw > 2.0                                     # the underlaid coarse arm
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
    # Only files that are actually sweeps. The sweep dirs also hold sidecars --
    # snr_calibration.json among them -- and globbing *.json blindly made this crash
    # with KeyError: 'steps' the moment one landed there.
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
        # ONE legend for the whole figure, on the ID panel: all three panels draw the
        # same four arms, so repeating it three times is pure clutter, and the ID panel
        # is where it sits clear of the curves (Kaelin, 2026-07-29).
        ax[2].legend(fontsize=7)
        fig.suptitle(f"Algorithm comparison: {factor}   |   identical frames, all arms   |   "
                     f"classical ID not comparable (it only reports already-identified corners); "
                     f"Deep ChArUco is unrefined -- read against our COARSE curve", fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(out / f"fourway_{factor}.png", dpi=args.dpi)
        if args.separate:   # alone, every panel carries the shared legend itself
            for a in ax[:2]:
                a.legend(fontsize=7)
            save_panels(fig, out / f"fourway_{factor}.png", args.dpi)
        plt.close(fig)
        print(f"{factor:>18} -> {out / ('fourway_' + factor + '.png')}")

    # ---- overlay panel: recall across several factors, 3x3, shared legend ----
    panel = [f for f in args.panel_factors
             if f in factors or (f == "distance" and "distance_extrap" in factors)]
    # (field, is_percentage, y-label, y-limits, y-scale) per metric. err_median is a px
    # error so it is log-scaled and NOT clamped to 0-102 -- clamping it, as the recall path
    # does, would silently flatten the 12-20x gap this panel exists to show.
    PM_FIELD, PM_PCT, PM_YLABEL, PM_YLIM, PM_YSCALE = {
        "recall": ("recall", True, "detection recall (%)  ↑ higher is better", (0, 102), None),
        "id_acc": ("id_acc", True, "ID accuracy among matched (%)  ↑ higher is better", (0, 102), None),
        "err_median": ("err_median", False, "localisation error, median (px)  ↓ lower is better",
                       None, "log"),
    }[args.panel_metric]
    if panel:
        # Columns follow the factor count rather than being pinned at 3: four factors lay out
        # 2x2 (the conference frame), nine still lay out 3x3. Derived rather than a flag so
        # --panel-factors is the single place the panel's shape is decided.
        ncol = 2 if len(panel) <= 4 else 3
        nrow = -(-len(panel) // ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 4.1 * nrow), squeeze=False)
        flat = [a for row in axes for a in row]
        for a, factor in zip(flat, panel):
            srcs = _merged_distance(dirs) if factor == "distance" else _load(dirs, factor)
            if not srcs:
                a.axis("off"); continue
            # the override belongs to the recall grid only: an ID or localisation grid plots its own metric
            # everywhere (before 2026-10-09 darkness showed found+ID on the px axis of the err_median grid)
            fld = PANEL_METRIC.get(factor, PM_FIELD) if args.panel_metric == "recall" else PM_FIELD
            xl = draw(a, srcs, factor, fld, True if fld != "err_median" else PM_PCT)
            if factor == "distance":
                # mark the trained envelope so out-of-distribution degradation is visibly
                # OUT of distribution rather than looking like ordinary failure
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
        # ONE figure-level y-label, not one per row. The combined recall/found+ID string is
        # longer than a 4.1in panel is tall, so per-row labels overflowed into the row above
        # and overprinted each other once the grid narrowed to 2 columns (caught rendering the
        # 2x2). A supylabel is also the honest presentation: every panel shares this axis.
        #
        # SHORT FORM (Kaelin 2026-07-31, "reduce the text on the y axis"). The long version
        # spelled out "detection recall / corners found + ID'd (%) \u2191 higher is better"; at
        # slide size that is unreadable-by-length rather than by point size. The per-panel
        # titles already carry "(found + ID)" wherever the metric is the joint one, so the
        # distinction is not lost by shortening here.
        ylab = ("recall / found + ID'd (%)  \u2191"
                if (args.panel_metric == "recall" and any(f in PANEL_METRIC for f in panel)) else PM_YLABEL)
        fig.supylabel(ylab, fontsize=FS_LABEL)
        # Legend on the ROTATION panel, not the first (Kaelin, 2026-07-29): the distance
        # panel is the merged in+out-of-envelope axis and its curves sweep the full height,
        # so a legend there overlaps the data. Rotation is flat and high for every arm --
        # ours ~98%, DC ~93%, classical ~52% -- leaving the lower-left corner empty.
        # Legend needs a panel whose curves do NOT sweep the full height. tilt is flat and high
        # for every arm; darkness and distance both span 0-100 and would be overlapped.
        #
        # motion_blur added to the preference list 2026-07-31: the 2x2 conference panel drops
        # tilt/rotation/defocus entirely, so the fallback put the legend on index 1 = darkness,
        # where it sat squarely over that panel's rise from 0 to 90% (Kaelin: "the legend cuts
        # off a lot of the darkness curve"). motion_blur's arms all sit above ~75% -- classical
        # OpenCV, the lowest, tops out at 49% and falls -- leaving the lower half clear. Ordered
        # AFTER the original three so the 3x3 panel's legend placement is unchanged.
        leg_i = next((panel.index(f) for f in ("tilt", "rotation", "defocus_blur", "motion_blur")
                      if f in panel), min(1, len(panel) - 1))
        # loc="best" rather than a hand-picked corner: on the 2x2, "lower left" in motion_blur
        # sat on top of the descending classical-OpenCV curve. Matplotlib's "best" scores every
        # candidate location against the actual plotted data and takes the emptiest, so the
        # placement stays clear whichever factors the panel is built from.
        # loc="best" rather than a hand-picked corner, and a COMPACT box: at full size the
        # legend still clipped the descending classical-OpenCV curve in motion_blur (Kaelin:
        # "make it a tiny bit smaller ... but good placement"). Tightened padding and shorter
        # handles shrink the footprint without touching the placement "best" chose.
        leg = dict(fontsize=FS_LEGEND, loc="best", framealpha=0.9, labelspacing=0.3, handlelength=1.6,
                   handletextpad=0.5, borderpad=0.4, borderaxespad=0.4)
        flat[leg_i].legend(**leg)
        # NO SUPTITLE (Kaelin 2026-07-31). On a slide the surrounding deck already says what
        # the figure is, and the banner cost a full text row of vertical space that the four
        # panels can use instead. tight_layout therefore reclaims the whole canvas.
        fig.tight_layout()
        _name = f"fourway_PANEL_{args.panel_metric}{args.panel_tag}.png"
        fig.savefig(out / _name, dpi=args.dpi)
        if args.separate:   # alone, every panel carries the shared y-label and legend itself; the legend goes
            # BELOW the axes: 'best' has no clear corner on every panel (it covered the refiner curve in darkness)
            for a in flat[:len(panel)]:
                a.set_ylabel(ylab, fontsize=FS_LABEL)
                a.legend(**dict(leg, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2))
            save_panels(fig, out / _name, args.dpi)
        plt.close(fig)
        print(f"{'PANEL':>18} -> {out / _name} ({len(panel)} factors, metric={args.panel_metric})")


if __name__ == "__main__":
    main()
