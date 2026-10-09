"""tools/make_tables.py -- the robustness and inter-method comparison TABLES.

Companion to plot_robustness.py / plot_four_way.py, which produce the same data
as curves. Figures show shape; tables give the numbers a reader can quote, and a
conference audience asking "what is the actual recall under occlusion" wants the
number, not a pixel position on an axis.

Reads only the sweep JSON -- no model, no GPU -- so restyling never needs a
re-run, and the tables are guaranteed to agree with the figures because both
derive from the same files.

FOUR OUTPUTS:
  fourway_summary.{md,tex}   headline: one row per factor, one column per arm.
                             Recall, because it is the metric all four arms
                             report comparably (see the ID caveat below).
  robustness_<metric>.md     per-metric, factors x arms, each cell "mean (worst)"
                             across the swept range -- mean says typical, worst
                             says how bad it gets, and a robustness claim needs
                             both.
  degradation.md             benign -> worst drop per arm per factor: the direct
                             measure of ROBUSTNESS as opposed to raw accuracy.
                             A method can lead on average and still fall apart.
  all_steps.csv              every arm at every step of every factor, for the
                             appendix and for anyone re-deriving the figures.

TWO CAVEATS ARE PRINTED INTO THE TABLES THEMSELVES, both cutting AGAINST
over-claiming and both easy to get wrong when reading numbers out of context:

  1. CLASSICAL'S ID ACCURACY IS NOT COMPARABLE. OpenCV only ever reports corners
     it has already identified, so its id_acc is ~100% among matched corners BY
     CONSTRUCTION. Its failure mode is not detecting at all -- read its RECALL.
     Its ID cells are emitted as "n/c" rather than a flattering number.
  2. DEEP CHARUCO IS UNREFINED. Their published system includes a RefineNet we
     do not run, so their localisation belongs against our COARSE column, never
     our refined one. Both of ours are shown so either comparison is available.
"""
import argparse
import csv
import json
from pathlib import Path

# (column label, source-dir key, arm key or None = first arm found)
ARMS = [
    ("Ours (coarse)",  "ours",      "coarse"),
    ("Ours (refined)", "ours",      "refined"),
    ("Deep ChArUco",   "dc",        None),
    ("Classical",      "classical", None),
]
METRICS = [
    ("recall",     "recall",     True,  "higher is better"),
    ("id_acc",     "id_acc",     True,  "higher is better"),
    ("err_median", "err_median", False, "lower is better"),
    ("err_p95",    "err_p95",    False, "lower is better"),
]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ours", default="paper/results_rev6/03_robustness")
    p.add_argument("--dc", default="paper/results_rev6/05_comparison/robustness_dc")
    p.add_argument("--classical", default="paper/results_rev6/05_comparison/robustness_classical")
    p.add_argument("--out-dir", default="paper/results_rev6/06_tables")
    return p


def load(dirs):
    """-> {factor: {arm_label: [per-step dicts]}}, plus the factor's unit."""
    out, units = {}, {}
    for key, d in dirs.items():
        for f in sorted(Path(d).glob("*.json")):
            data = json.loads(f.read_text())
            if not data.get("steps"):
                continue
            factor = f.stem
            units[factor] = data.get("unit", factor)
            available = list(data["steps"][0]["arms"])
            for label, src, arm in ARMS:
                if src != key:
                    continue
                a = arm if arm else available[0]
                if a not in data["steps"][0]["arms"]:
                    continue
                out.setdefault(factor, {})[label] = [
                    dict(s["arms"][a], value=s["value"]) for s in data["steps"]]
    return out, units


def _cell(rows, field, pct, label):
    """"mean (worst)" for one arm/metric, or n/c where the number would mislead."""
    if label == "Classical" and field == "id_acc":
        return "n/c"                                    # see caveat 1
    vals = [r[field] for r in rows if r.get(field) is not None]
    if not vals:
        return "--"
    worst = min(vals) if pct else max(vals)              # worst = low recall / high error
    m = sum(vals) / len(vals)
    return f"{100*m:.1f} ({100*worst:.1f})" if pct else f"{m:.3f} ({worst:.3f})"


def _md(headers, rows):
    w = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    out = ["| " + " | ".join(str(h).ljust(w[i]) for i, h in enumerate(headers)) + " |",
           "|" + "|".join("-" * (x + 2) for x in w) + "|"]
    out += ["| " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) + " |" for r in rows]
    return "\n".join(out)


CAVEATS = (
    "\n**Reading notes.** Classical OpenCV only reports corners it has already "
    "identified, so its ID accuracy is ~100% among matched corners by construction "
    "and is not comparable -- read its **recall** instead (its ID cells show `n/c`). "
    "Deep ChArUco is **unrefined** (their RefineNet is not run), so its localisation "
    "belongs against our **coarse** column, not our refined one. All arms are scored "
    "on identical frames.\n")


def main():
    args = build_parser().parse_args()
    data, units = load({"ours": args.ours, "dc": args.dc, "classical": args.classical})
    if not data:
        raise SystemExit("no sweep JSON found -- has the sweep finished?")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    labels = [l for l, _, _ in ARMS]
    factors = sorted(data)

    # ---- headline: recall, one row per factor, one column per arm ----------
    rows = [[f] + [_cell(data[f][l], "recall", True, l) if l in data[f] else "--"
                    for l in labels] for f in factors]
    head = "# Inter-method comparison -- detection recall, % (worst case in parentheses)\n"
    (out / "fourway_summary.md").write_text(
        head + "\nHigher is better. Cells are `mean (worst)` across the factor's swept range.\n"
        + CAVEATS + "\n" + _md(["Factor"] + labels, rows) + "\n")
    tex = ["\\begin{tabular}{l" + "r" * len(labels) + "}", "\\hline",
           "Factor & " + " & ".join(labels) + " \\\\", "\\hline"]
    tex += [" & ".join([r[0].replace("_", "\\_")] + r[1:]) + " \\\\" for r in rows]
    tex += ["\\hline", "\\end{tabular}"]
    (out / "fourway_summary.tex").write_text("\n".join(tex) + "\n")

    # ---- one table per metric ---------------------------------------------
    for name, field, pct, dirn in METRICS:
        rows = [[f] + [_cell(data[f][l], field, pct, l) if l in data[f] else "--"
                        for l in labels] for f in factors]
        (out / f"robustness_{name}.md").write_text(
            f"# {name} by factor and method ({dirn})\n\nCells are `mean (worst)` across "
            f"the factor's swept range.\n" + CAVEATS + "\n" + _md(["Factor"] + labels, rows) + "\n")

    # ---- degradation: benign -> worst, the robustness number proper --------
    rows = []
    for f in factors:
        row = [f]
        for l in labels:
            if l not in data[f]:
                row.append("--"); continue
            v = [r["recall"] for r in data[f][l] if r.get("recall") is not None]
            # "benign" = the arm's own best step: the swept axis is not always
            # ordered benign-to-harsh (rotation, brightness sweep both ways), so
            # best-to-worst is the only reading that is correct for every factor
            row.append(f"{100*max(v):.1f} -> {100*min(v):.1f}  ({100*(min(v)-max(v)):+.1f})" if v else "--")
        rows.append(row)
    (out / "degradation.md").write_text(
        "# Robustness: recall from each method's BEST step to its WORST, per factor\n\n"
        "`best -> worst (drop, pp)`. This is the robustness number proper -- a method can "
        "lead on mean accuracy and still collapse at one end of a factor. Best-to-worst "
        "rather than first-to-last step because several axes (rotation, brightness) sweep "
        "outward in both directions from benign.\n" + CAVEATS + "\n"
        + _md(["Factor"] + labels, rows) + "\n")

    # ---- full per-step dump ------------------------------------------------
    with open(out / "all_steps.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["factor", "unit", "value", "arm", "n_gt", "n_match", "recall",
                    "id_acc", "err_median", "err_mean", "err_p95"])
        for f in factors:
            for l in labels:
                for r in data[f].get(l, []):
                    w.writerow([f, units[f], r["value"], l, r.get("n_gt"), r.get("n_match"),
                                r.get("recall"), r.get("id_acc"), r.get("err_median"),
                                r.get("err_mean"), r.get("err_p95")])
    n_arms = sum(len(v) for v in data.values())
    print(f"{len(factors)} factors, {n_arms} factor-arm series -> {out}")
    for p in sorted(out.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
