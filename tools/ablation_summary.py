"""tools/ablation_summary.py -- per-arm training metrics into each arm's OWN folder,
plus one head-to-head table across arms.

Kaelin 2026-07-29: "good admin in terms of making sure checkpoints and MODEL'S OWN
PERFORMANCE are in their own ablation folders ... we should be able to do a direct
metric head to head (mean, median, p95 etc), but none of the full sweep is required
yet." This is the training-metric half only -- no robustness sweeps, no figures.

Every arm gets paper/results_rev6/08_ablations/<ARM>/train_metrics.{json,md}: the full
validation curve, the best and final validation, and a pointer to its checkpoints
(pointer, not a copy -- a checkpoint is ~100 MB and already lives in runs/<arm>/).

CONTROL is abl_reference_50k_rev6 and nothing else. abl_reference_50k is the
rev-2-trained ladder baseline and comparing to it confounds architecture with
generator revision -- the exact error a full day was spent unwinding.
"""
import argparse
import json
import shutil
from pathlib import Path

CONTROL = "abl_reference_50k_rev6"
# run dir -> ablation folder name. Anything not listed is reported but not filed.
FOLDER = {
    "abl_reference_50k_rev6": "_reference_50k",
    "abl_convonly_50k_rev6": "A1_conv_only",
    "abl_ce_50k_rev6": "A_CE_one_hot",
    "abl_xsa_50k_rev6": "A_XSA",
    "abl_sigma1_50k_rev6": "A_SIGMA1",
    "abl_sigma05_50k_rev6": "A_SIGMA05",
    "abl_nodilate_50k_rev6": "A_NODILATE",
    "abl_xsa_nodilate_50k_rev6": "A_XSA_NODILATE",
    "abl_beta0_50k_rev6": "A_BETA0",
    "abl_gates_off_50k_rev6": "A_GATES_OFF",
}
# Exact names, not substrings: "abl_reference_50k" is a PREFIX of the rev-6 control,
# so a substring test silently drops the control itself.
SKIP_EXACT = {"abl_reference_50k"}                  # pre-rev6 ladder baseline, confounded
SKIP_SUBSTR = ("_stalled",)                         # abandoned restarts


def load(run: Path):
    """(steps_trained, [{step, val...}]) from a run's metrics.jsonl.

    A metrics.jsonl is an append-only log written by a live process, not a contract, and it
    CAN be damaged out from under this tool: on 2026-08-01 a line of conference-talk text was
    pasted into the middle of abl_nodilate_width_half_s05_50k_rev6's file (a terminal had it
    open), splitting one train record across two unparseable lines. Two lines of 37,102, all
    16 val records intact -- yet a bare json.loads list comprehension took the whole summary
    down. So: skip a damaged line and keep going, but WARN, because a silent skip would let
    real corruption -- a truncated val record, a half-flushed tail -- pass as a clean run."""
    recs, damaged = [], 0
    for lineno, l in enumerate((run / "metrics.jsonl").read_text().splitlines(), 1):
        if not l.strip():
            continue
        try:
            recs.append(json.loads(l))
        except json.JSONDecodeError as e:
            damaged += 1
            print(f"  !! {run.name}/metrics.jsonl line {lineno} unparseable, skipped ({e})")
    if damaged:
        print(f"  !! {run.name}: {damaged} damaged line(s) -- inspect before trusting its curve")
    train = [d for d in recs if "loss" in d]
    vals = []
    for d in recs:
        if "val" not in d:
            continue
        v = d["val"]
        vals.append({"step": d["step"], "val_loss": v.get("val_loss"),
                     "m01": v.get("m01"), "m02": v.get("m02"),
                     "m04": (v.get("m04") or {}).get("accuracy"),
                     "m04_by_octave": (v.get("m04") or {}).get("by_octave"),
                     "diag": v.get("diag")})
    return (train[-1]["step"] if train else 0), vals


def num(x, spec=".4f", scale=1.0, suffix=""):
    """Format a metric that CAN legitimately be undefined. An early validation with zero
    matched corners has no mean/median/p95 and no M-04 -- the trainer writes null, correctly.
    Formatting those with a bare f-spec raised TypeError and killed the whole summary; a run
    that has only reached its first validation is exactly when this tool is most wanted."""
    return "--" if x is None else format(scale * x, spec) + suffix


def row(v):
    m = v["m01"] or {}
    return {"step": v["step"], "val_loss": v["val_loss"], "m04": v["m04"],
            "m01_mean": m.get("mean"), "m01_median": m.get("median"),
            "m01_p95": m.get("p95"), "m01_tail_gt4px": m.get("tail_frac_gt4px")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="paper/results_rev6/08_ablations")
    a = p.parse_args()

    out = Path(a.out)
    arms = {}
    for run in sorted(Path(a.runs).glob("abl_*")):
        if not run.is_dir() or not (run / "metrics.jsonl").exists():
            continue
        if run.name in SKIP_EXACT or any(s in run.name for s in SKIP_SUBSTR):
            continue
        step, vals = load(run)
        if not vals:
            print(f"  {run.name}: no validations yet, skipped")
            continue
        best = max(vals, key=lambda v: (v["m04"] if v["m04"] is not None else -1))
        ck = sorted(run.glob("ckpt_*.pt"))
        rec = {"run": run.name, "config": str(run / "config.yaml"),
               "steps_trained": step, "n_validations": len(vals),
               "checkpoints": [{"file": str(c), "mb": round(c.stat().st_size / 1e6, 1)}
                               for c in ck],
               "best_val": row(best), "final_val": row(vals[-1]),
               "curve": [row(v) for v in vals],
               "m04_by_octave_final": vals[-1]["m04_by_octave"],
               "m02_final": vals[-1]["m02"], "diag_final": vals[-1]["diag"]}
        arms[run.name] = rec

        d = out / FOLDER.get(run.name, run.name)
        d.mkdir(parents=True, exist_ok=True)
        (d / "train_metrics.json").write_text(json.dumps(rec, indent=2))
        # the config the arm actually ran, beside its numbers -- the one-key claim
        # is unverifiable later without it
        if (run / "config.yaml").exists():
            shutil.copy2(run / "config.yaml", d / "config_used.yaml")
        b, f = rec["best_val"], rec["final_val"]
        (d / "train_metrics.md").write_text(
            f"# {run.name}\n\n"
            f"Trained {step:,} steps, {len(vals)} validations. "
            f"Checkpoints: {len(ck)} in `{run}/`.\n\n"
            f"| | step | m01 mean | m01 median | m01 p95 | tail>4px | M-04 | val loss |\n"
            f"|---|---:|---:|---:|---:|---:|---:|---:|\n"
            + "".join(
                f"| {lab} | {r['step']:,} | {num(r['m01_mean'])} | {num(r['m01_median'])} | "
                f"{num(r['m01_p95'])} | {num(r['m01_tail_gt4px'], '.3f', 100, '%')} | "
                f"{num(r['m04'], '.2f', 100, '%')} | {num(r['val_loss'])} |\n"
                for lab, r in (("best M-04", b), ("final", f)))
            + f"\nFull curve and per-octave breakdown in `train_metrics.json`.\n")
        print(f"  {run.name:30s} -> {d.relative_to(out.parent.parent)}  "
              f"({step:,} steps, {len(ck)} ckpt)")

    # head-to-head, deltas against the rev-6 control only
    ctl = arms.get(CONTROL)
    lines = ["# Ablation head-to-head — training metrics only",
             "",
             f"Control: **{CONTROL}**. Deltas are vs that control **at its own final step**;",
             "arms still training are marked, and their deltas are provisional.",
             "M-01 is coarse localisation error in input px (lower better); M-04 is ID accuracy.",
             "No robustness sweeps here — this is each model's own validation performance.",
             "",
             "| arm | steps | m01 mean | m01 median | m01 p95 | tail>4px | M-04 | ΔM-04 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in sorted(arms.items(), key=lambda kv: -(kv[1]["best_val"]["m04"] or 0)):
        b = r["best_val"]
        d = "" if not ctl or name == CONTROL or b["m04"] is None \
            or ctl["best_val"]["m04"] is None else \
            f"{100*(b['m04'] - ctl['best_val']['m04']):+.2f} pp"
        tag = "" if r["steps_trained"] >= 50000 else " *(running/stopped early)*"
        lines.append(
            f"| {name}{tag} | {r['steps_trained']:,} | {num(b['m01_mean'])} | "
            f"{num(b['m01_median'])} | {num(b['m01_p95'])} | "
            f"{num(b['m01_tail_gt4px'], '.3f', 100, '%')} | "
            f"{num(b['m04'], '.2f', 100, '%')} | {d} |")
    lines += ["", "Rows are each arm's BEST validation by M-04, not necessarily its last.",
              "Arms below 50,000 steps were stopped early or are still training — a",
              "best-so-far number from a short run is not comparable to a converged one."]
    (out / "HEAD_TO_HEAD.md").write_text("\n".join(lines) + "\n")
    print(f"\n-> {out}/HEAD_TO_HEAD.md")


if __name__ == "__main__":
    main()
