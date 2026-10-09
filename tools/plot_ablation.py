"""tools/plot_ablation.py -- per-regime ablation curves AND tables, for any set of
labelled arms.

The ablation protocol says A1 is judged PER-REGIME, not on aggregate M-04
(Kaelin): "the attention hypothesis is better at FAR, VERY CLOSE, and OCCLUDED"
-- and an aggregate averages exactly that structure away. So this reports each
factor's curve, and tables the regimes the hypothesis actually names.

Takes arms as `label=dir` pairs, so it serves every ablation rather than being
A1-specific: A-CE, XSA and A-SIGMA1 all reduce to the same "two or more arms,
identical frames, one factor at a time" shape.

Reads only the sweep JSON -- no model, no GPU -- so restyling never needs a
re-run, and the tables cannot disagree with the curves because both derive from
the same files.

OUTPUT CONVENTION (Kaelin, 2026-07-29 -- the comparison outputs were landing loose
in 08_ablations/ and that does not scale to four arms): --out-dir is the ABLATION's
OWN folder, and the arms' sweep JSON lives in <ablation>/sweeps/ with the shared
reference arm in 08_ablations/_reference_sweeps/. See
paper/results_rev6/08_ablations/00_README_LAYOUT.md.

Emits, into --out-dir:
  ablation_<factor>.png     one panel per metric, one curve per arm
  ablation_regimes.md       THE table: the named regimes (far / very close /
                            occluded), per arm, with deltas vs the FIRST arm
                            (treated as the reference).
  ablation_<factor>.md      full per-step table for each factor
  ablation_all_steps.csv    every arm at every step, for the appendix

Localisation is plotted from the COARSE arm. The refiner is shared across
ablation arms and identical in each, so a refined-arm difference would only
re-measure the refiner, not the architecture under test.
"""
import argparse
import csv
import json
from pathlib import Path

COLOURS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
SQUARES_PER_SIDE, FRAME_W = 5, 640
# the regimes the A1 hypothesis actually names, as (label, factor, predicate)
REGIMES = [
    ("FAR  (s=12)",          "distance",         lambda v: v == 12),
    ("FAR  (s=16)",          "distance",         lambda v: v == 16),
    ("VERY CLOSE  (s=96)",   "distance",         lambda v: v == 96),
    ("VERY CLOSE  (s=128)",  "distance",         lambda v: v == 128),
    ("OCCLUDED  (max holes)", "occlusion",       lambda v: True),   # worst step
    ("OCCLUDED  (max objects)", "object_occlusion", lambda v: True),
]
METRICS = [("recall", "detection recall (%)  ↑ higher is better", True),
           ("id_acc", "ID accuracy among matched (%)  ↑ higher is better", True),
           ("err_median", "localisation error, median (px)  ↓ lower is better", False)]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("arms", nargs="+", metavar="LABEL=DIR",
                   help="e.g. reference=paper/.../a1_reference convonly=paper/.../a1_convonly "
                        "-- the FIRST arm is the reference that deltas are taken against")
    p.add_argument("--arm-key", default="coarse",
                   help="which sweep arm to read (default coarse: the refiner is shared "
                        "across ablation arms, so a refined comparison re-measures the refiner)")
    p.add_argument("--out-dir", default="paper/results_rev6/08_ablations")
    p.add_argument("--title", default="Ablation")
    p.add_argument("--dpi", type=int, default=150)
    return p


def load(arms, key):
    """-> ({label: {factor: [steps]}}, {factor: unit})"""
    data, units = {}, {}
    for label, d in arms:
        for f in sorted(Path(d).glob("*.json")):
            j = json.loads(f.read_text())
            if not j.get("steps"):
                continue
            units[f.stem] = j.get("unit", f.stem)
            a = key if key in j["steps"][0]["arms"] else list(j["steps"][0]["arms"])[0]
            data.setdefault(label, {})[f.stem] = [dict(s["arms"][a], value=s["value"])
                                                   for s in j["steps"]]
    return data, units


def _md(headers, rows):
    w = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    out = ["| " + " | ".join(str(h).ljust(w[i]) for i, h in enumerate(headers)) + " |",
           "|" + "|".join("-" * (x + 2) for x in w) + "|"]
    out += ["| " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) + " |" for r in rows]
    return "\n".join(out)


def main():
    args = build_parser().parse_args()
    arms = [tuple(a.split("=", 1)) for a in args.arms]
    if any(len(a) != 2 for a in arms):
        raise SystemExit("arms must be LABEL=DIR")
    data, units = load(arms, args.arm_key)
    if not data:
        raise SystemExit("no sweep JSON found -- has the sweep finished?")
    labels = [l for l, _ in arms]
    ref = labels[0]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    factors = sorted({f for v in data.values() for f in v})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- curves -----------------------------------------------------------
    for factor in factors:
        fig, ax = plt.subplots(1, len(METRICS), figsize=(5.2 * len(METRICS), 4.4))
        for j, (field, ylab, pct) in enumerate(METRICS):
            for i, lab in enumerate(labels):
                steps = data.get(lab, {}).get(factor)
                if not steps:
                    continue
                xs = ([100.0 * SQUARES_PER_SIDE * s["value"] / FRAME_W for s in steps]
                      if factor == "distance" else [s["value"] for s in steps])
                ys = [s[field] * (100 if pct else 1) for s in steps]
                ax[j].plot(xs, ys, "-o", ms=3.5, color=COLOURS[i % len(COLOURS)], label=lab)
            ax[j].set_ylabel(ylab)
            ax[j].set_xlabel("board width (% of frame width)" if factor == "distance"
                             else units.get(factor, factor))
            ax[j].grid(alpha=0.3); ax[j].legend(fontsize=8)
            if field == "err_median":
                ax[j].set_yscale("log")
        fig.suptitle(f"{args.title}: {factor}   |   identical frames, all arms   |   "
                     f"coarse arm (shared refiner excluded)", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(out / f"ablation_{factor}.png", dpi=args.dpi)
        plt.close(fig)
        print(f"{factor:>18} -> {out / ('ablation_' + factor + '.png')}")

        rows = []
        for s_i in range(len(next(iter(data.values()))[factor])):
            row = [data[ref][factor][s_i]["value"]]
            for lab in labels:
                st = data.get(lab, {}).get(factor)
                row += ["--"] if not st else [
                    f"{100*st[s_i]['recall']:.1f} / {100*st[s_i]['id_acc']:.1f} / {st[s_i]['err_median']:.3f}"]
            rows.append(row)
        (out / f"ablation_{factor}.md").write_text(
            f"# {args.title} — {factor} ({units.get(factor, factor)})\n\n"
            f"Cells are `recall% / ID% / median err px`. Identical frames across arms; "
            f"coarse arm only (the refiner is shared, so a refined comparison would "
            f"re-measure the refiner rather than the architecture).\n\n"
            + _md([units.get(factor, factor)] + labels, rows) + "\n")

    # ---- the regime table: what the hypothesis actually claims -------------
    rows = []
    for name, factor, pred in REGIMES:
        if factor not in factors:
            continue
        row = [name]
        base = None
        for lab in labels:
            steps = data.get(lab, {}).get(factor)
            if not steps:
                row.append("--"); continue
            sel = [s for s in steps if pred(s["value"])]
            # occlusion regimes have no single named value: take the arm's WORST step
            s = min(sel or steps, key=lambda s: s["id_acc"]) if factor.endswith("occlusion") \
                else (sel[0] if sel else None)
            if s is None:
                row.append("--"); continue
            v = 100 * s["id_acc"]
            if base is None:
                base, cell = v, f"{v:.2f}"
            else:
                cell = f"{v:.2f}  ({v-base:+.2f})"
            row.append(cell)
        rows.append(row)
    (out / "ablation_regimes.md").write_text(
        f"# {args.title} — per-regime ID accuracy (%), deltas vs `{ref}`\n\n"
        "**Judged per regime, not on aggregate.** The aggregate averages away exactly the "
        "structure the hypothesis names. Distance regimes are read at their named `s`; the "
        "occlusion regimes are read at each arm's WORST step, since that is where an "
        "occlusion claim lives.\n\nIdentical frames across arms; coarse arm only.\n\n"
        + _md(["Regime"] + labels, rows) + "\n")
    print(f"{'REGIMES':>18} -> {out / 'ablation_regimes.md'}")

    with open(out / "ablation_all_steps.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "factor", "unit", "value", "n_gt", "n_match",
                    "recall", "id_acc", "err_median", "err_p95"])
        for lab in labels:
            for factor, steps in data.get(lab, {}).items():
                for s in steps:
                    w.writerow([lab, factor, units.get(factor, factor), s["value"], s.get("n_gt"),
                                s.get("n_match"), s.get("recall"), s.get("id_acc"),
                                s.get("err_median"), s.get("err_p95")])
    print(f"{'CSV':>18} -> {out / 'ablation_all_steps.csv'}")


if __name__ == "__main__":
    main()
