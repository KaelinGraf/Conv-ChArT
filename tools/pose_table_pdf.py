"""tools/pose_table_pdf.py -- the four-way pose table as a one-page, screenshot-able PDF.

Every number is read from the eval JSON at render time, so the document cannot drift
from the measurement. Nothing is transcribed by hand.
"""
import argparse
import json
import subprocess
from pathlib import Path

# REFINED ARMS ONLY (Kaelin 2026-07-31: "only include our with refiner benchmarks (i.e
# exclude coarse) just so it's more readable"). The coarse arms are still evaluated and still
# in the JSON -- the refiner-gain line below is computed from `ours_coarse` at render time --
# they are simply not given table rows. Deep ChArUco's own RefineNet is NOT run, so its row is
# unrefined; that asymmetry now favours us and is stated in the notes rather than hidden.
ROWS = [("ours_refined", "**Conv-ChArT + refiner**"),
        ("deep_charuco", "Deep ChArUco (fine-tuned)"),
        ("classical", "classical OpenCV")]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", default="paper/results_rev6/13_pose/pose_fourway_hard.json")
    p.add_argument("--json-fast", default=None,
                   help="second eval JSON (Conv-ChArT-FAST). Its arms are a SEPARATE run over "
                        "the same 1500-image set with the same seed, so the rows are directly "
                        "comparable; they are appended rather than merged into the first file.")
    p.add_argument("--out", default="paper/results_rev6/13_pose/pose_tables.pdf")
    a = p.parse_args()

    d = json.loads(Path(a.json).read_text())
    arms = dict(d["arms"])
    if a.json_fast and Path(a.json_fast).is_file():
        fa = json.loads(Path(a.json_fast).read_text())["arms"]
        # Namespaced so the FAST rows cannot collide with the primary model's, and so the
        # lead-multiplier rows below keep comparing the PRIMARY refined arm to the baselines.
        arms["fast_coarse"], arms["fast_refined"] = fa["ours_coarse"], fa["ours_refined"]
        ROWS.insert(1, ("fast_refined", "**Conv-ChArT-FAST + refiner**"))
    body = "\n".join(
        "| {} | {:.1f} | {:.3f} | {:.3f} | {:.3f} | {:.4f} | {:.4f} | {:.4f} |".format(
            lab, 100 * v["solve_rate"], v["rot_deg"]["median"], v["rot_deg"]["mean"],
            v["rot_deg"]["p95"], v["trans_squares"]["median"], v["trans_squares"]["mean"],
            v["trans_squares"]["p95"])
        for k, lab in ROWS for v in [arms[k]])

    # ---- lead multipliers -------------------------------------------------------------
    # DIRECTION IS PER COLUMN and getting it wrong inverts the claim: solve % is
    # higher-is-better (lead = ours/theirs) while every error column is lower-is-better
    # (lead = theirs/ours). One shared ratio would report our biggest win as a loss.
    def col(v):
        return [(100 * v["solve_rate"], +1), (v["rot_deg"]["median"], -1), (v["rot_deg"]["mean"], -1),
                (v["rot_deg"]["p95"], -1), (v["trans_squares"]["median"], -1),
                (v["trans_squares"]["mean"], -1), (v["trans_squares"]["p95"], -1)]

    def lead_row(label, other):
        """Bolded, because it is the one comparison the table exists to make."""
        ours, theirs = col(arms["ours_refined"]), col(arms[other])
        cells = []
        for (o, sign), (t, _) in zip(ours, theirs):
            if o <= 0 or t <= 0:
                cells.append("--"); continue
            r = (o / t) if sign > 0 else (t / o)
            cells.append("**" + (f"{r:.1f}x" if r >= 1 else f"{r:.2f}x") + "**")
        return "| " + label + " | " + " | ".join(cells) + " |"

    # ONLY the Deep ChArUco lead row (Kaelin 2026-07-31: "exclude the ours vs classical
    # multiplier line, and bold the ours vs deep charuco"). The classical multipliers were
    # the least informative row in the table: classical refuses 45% of frames, so its error
    # columns are conditioned on a much easier subset and a ratio against them compares
    # different populations. The coverage claim against classical is made in prose instead.
    leads = lead_row("**lead: ours vs Deep ChArUco**", "deep_charuco")
    # Refiner gain, computed rather than transcribed -- these moved when the headline
    # model changed and a hardcoded "2.7x" silently became wrong.
    c, r = arms["ours_coarse"], arms["ours_refined"]
    g_rot = c["rot_deg"]["median"] / r["rot_deg"]["median"]
    g_tr = c["trans_squares"]["median"] / r["trans_squares"]["median"]

    md = f"""---
title: "Conv-ChArT --- pose accuracy, four systems"
geometry: margin=1.6cm
fontsize: 10pt
---

Rotation in **degrees** (deg) --- the geodesic angle between true and estimated board
rotation. Translation in **board squares** (sq): the evaluation set fixes
square\\_length\\_m = 1.0, so the unit is board squares, not metres. {d['n_images']} images.

**All four algorithms share the identical downstream pipeline** --- lattice gate, recovery
pass, then PnP (IPPE) --- so the only thing that differs between rows is the corners
each detector produced. Classical OpenCV and Deep ChArUco are therefore *not* using
their own pose solvers; these numbers will not match their native output, by design.
A **refusal** (a pose the pipeline declines to corroborate) is counted in the solve
rate and excluded from the error statistics --- it is a correct outcome, not an error.

| algorithm | solve % | rot median (deg) | rot mean (deg) | rot p95 (deg) | trans median (sq) | trans mean (sq) | trans p95 (sq) |
|:-------------------------|-------:|--------:|-------:|-------:|---------:|--------:|-------:|
{body}
{leads}

# Reading the table

- **The `lead` row reads as "how many times better our refined arm is"**, computed
  per column in that column's own direction: solve % is higher-is-better, every error
  column is lower-is-better. A value below 1.0 means we are behind on that column.
  It is quoted against Deep ChArUco only --- see the classical note below for why a
  multiplier against classical would compare two different populations.
- **Every row here is refined except the baselines.** Our coarse (pre-refiner) arm is
  measured but not shown; the refiner is worth {g_rot:.1f}x on rotation median and
  {g_tr:.1f}x on translation median, so quoting the refined arm is quoting the system a
  docking controller actually consumes. **Deep ChArUco's own RefineNet is not run**, so its
  row is unrefined --- an asymmetry that favours us, stated here rather than buried.
- **Classical OpenCV is accurate when it succeeds** --- median rotation
  {arms['classical']['rot_deg']['median']:.3f} deg against our {arms['ours_refined']['rot_deg']['median']:.3f} deg --- but it solves only
  {100 * arms['classical']['solve_rate']:.0f}% of frames against our {100 * arms['ours_refined']['solve_rate']:.0f}%. The claim against it is coverage, not precision,
  and that is exactly why it gets no multiplier row: its error columns are conditioned on
  the {100 * arms['classical']['solve_rate']:.0f}% of frames it chose to attempt, which is an easier set than the one we are
  scored on.
- **Deep ChArUco has a heavy tail.** Its median is respectable, but p95 rotation is
  {arms['deep_charuco']['rot_deg']['p95']:.0f} degrees: confidently wrong poses on several percent of frames. For a
  docking robot a wrong pose is more dangerous than no pose, which makes the p95
  column more operationally important than the median.
- **Solve rate and p95 must be read together.** Solve rate alone flatters Deep
  ChArUco; p95 alone hides that classical refuses nearly half the time.
"""
    out = Path(a.out)
    md_path = out.with_suffix(".md")
    md_path.write_text(md)
    subprocess.run(["pandoc", str(md_path), "-o", str(out), "--pdf-engine=pdflatex"],
                   check=True)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
