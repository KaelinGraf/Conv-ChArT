"""tools/plot_finetune.py -- the minimal-fine-tuning ladder, side by side: accuracy vs steps, steps and
wall time to target, peak training VRAM, and the final new-board accuracy, one bar/curve per arm.

Inputs per arm (config stem = run name): runs/<arm>/metrics.jsonl (in-loop val curve, peak_mem, done
record), paper/results_rev6/28_finetune_minimal/{finetune_cost.json, fullval_<arm>.json, pose_<arm>.json
+ _per_image.jsonl, timing.jsonl}. Reads only JSON -- restyling never needs a re-run.

"Steps to target" is the first in-loop validation (1,000 samples, every 250 steps, EMA weights) whose
M-04 reaches the target; wall time to target is that step's wall-clock offset from the run's first step
(so it includes validation time spent up to then, which an on-device run would also pay, but not the
loader warm-up before step 1). Arms that never reach the target get no bar and are listed in the title.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dcc.viz import save_panels

# Okabe-Ito first, then three validated extras; fixed order, never cycled. Markers double-encode identity.
# Okabe-Ito (minus black and yellow, which fail the chroma/lightness checks on a white surface) plus four
# validated extras; validated as a 10-slot categorical set (all checks pass, CVD worst pair dE 7.6 with
# markers as the required secondary encoding).
COLOURS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#9467BD", "#8C510A", "#C7B33A", "#17BECF"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
D = Path("paper/results_rev6/28_finetune_minimal")
# GPU cost depends on WHICH parameters train and the batch, not on the LR or the step budget, so round-2 arms
# and the multi-board fine-tunes read the benchmark row of the arm with the same trainable set.
COST_ALIAS = {"ft11_head_lr3x": "ft01_head", "ft12_head_20k": "ft01_head", "mb_ft01_head": "ft01_head",
              "ft13_head_bneck_hm_lr0p1_20k": "ft02_head_bneck_hm_lr0p1", "ft14_head_bneck_hm_lr1": "ft02_head_bneck_hm_lr0p1",
              "ft15_head_bneck_lr0p3_hm_lr0p1": "ft02_head_bneck_hm_lr0p1", "ft16_head_bneck_lr1_hm_lr0p1": "ft02_head_bneck_hm_lr0p1",
              "mb_ft02_head_bneck_hm_lr0p1": "ft02_head_bneck_hm_lr0p1", "mb_ft14_head_bneck_hm_lr1": "ft02_head_bneck_hm_lr0p1",
              "ft18_head_bneck_lr3x_hm_lr0p1": "ft02_head_bneck_hm_lr0p1", "mb15_ft01_head": "ft01_head",
              "mb3_ft01_head": "ft01_head", "mb3_ft02_head_bneck_hm_lr0p1": "ft02_head_bneck_hm_lr0p1",
              "mb3_ft14_head_bneck_hm_lr1": "ft02_head_bneck_hm_lr0p1",
              "mb15_ft17_post_bneck": "ft17_post_bneck", "mb3_ft17_post_bneck": "ft17_post_bneck", "ft20_full_lr3x": "ft10_full_lr1", "ft21_head_bneck_lr3x_hm_lr0p1_20k": "ft02_head_bneck_hm_lr0p1", "ft19_head_bneck_e4_hm_lr1": "ft19_head_bneck_e4_hm_lr1"}
REF_M04 = 99.53   # the trained board, 882k, 10k full val (06_tables / 27_transfer_zeroshot control)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", nargs="+", required=True, help="run names, in display order")
    p.add_argument("--labels", nargs="+", default=None, help="display labels (default: run names)")
    p.add_argument("--targets", type=float, nargs="+", default=[80.0, 90.0, 95.0, 99.0],
                   help="M-04 %% thresholds for steps/wall-to-target (grouped bars; the last is the headline target)")
    p.add_argument("--batch", type=int, default=16, help="batch row of finetune_cost.json to plot")
    p.add_argument("--out", default=str(D / "finetune_ladder.png"))
    p.add_argument("--table", default=str(D / "finetune_ladder.md"))
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--separate", action="store_true", help="also save each panel as its own image beside the figure")
    p.add_argument("--title", default="Minimal fine-tuning of Conv-ChArT 882k to a new board (DICT_6X6_250): what has to move, and what it costs")
    return p


def load_arm(arm, targets, batch):
    recs = [json.loads(l) for l in (Path("runs") / arm / "metrics.jsonl").read_text().splitlines() if l.strip()]
    t_first = next(r["wall"] for r in recs if "loss" in r)
    vals = [(r["step"], r["wall"] - t_first, 100 * (r["val"]["m04"]["accuracy"] or 0), r["val"]["m01"]["p95"])
            for r in recs if "val" in r]
    done = next((r["done"] for r in recs if "done" in r), {})
    hits = {tg: next(((s, w) for s, w, m04, _ in vals if m04 >= tg), None) for tg in targets}
    hit = hits[targets[-1]]
    cost = json.loads((D / "finetune_cost.json").read_text())["rows"].get(f"{COST_ALIAS.get(arm, arm)}@B{batch}", {})
    # the trainer's own 10k full val at the final step (same SynthVal/EMA/scorer as tools/eval_checkpoint.py)
    fv = next((r["full_val"] for r in reversed(recs) if "full_val" in r), None)
    pose = json.loads((D / f"pose_{arm}.json").read_text()) if (D / f"pose_{arm}.json").exists() else None
    correct = None
    if pose:
        per = [json.loads(l) for l in (D / f"pose_{arm}_per_image.jsonl").read_text().splitlines() if l.strip()]
        correct = 100 * sum(r["rot_deg"] < 2.0 and not r["ambiguous"] for r in per) / pose["n_images"]
        wrong = 100 * sum(r["rot_deg"] >= 2.0 and not r["ambiguous"] for r in per) / pose["n_images"]
    timing = {}
    if (D / "timing.jsonl").exists():
        timing = next((json.loads(l) for l in (D / "timing.jsonl").read_text().splitlines() if l.strip() and json.loads(l)["arm"] == arm), {})
    return {"arm": arm, "curve": vals, "steps_to_target": hit[0] if hit else None, "wall_to_target_s": hit[1] if hit else None,
            "steps_to": {tg: (h[0] if h else None) for tg, h in hits.items()}, "wall_to": {tg: (h[1] if h else None) for tg, h in hits.items()},
            "n_trainable": done.get("n_trainable"), "peak_mem_mb_run": done.get("peak_mem_mb"), "train_elapsed_s": done.get("train_elapsed_s"),
            "wall_elapsed_s": done.get("wall_elapsed_s"), "samples_per_s": done.get("samples_per_s"),
            "peak_mem_mb_bench": cost.get("peak_mem_mb"), "ms_per_step_bench": cost.get("ms_per_step"), "optim_state_mb": cost.get("optim_state_mb"),
            "final_m04": 100 * fv["m04"]["accuracy"] if fv else None, "final_p95": fv["m01"]["p95"] if fv else None,
            "final_median": fv["m01"]["median"] if fv else None, "pose_correct": correct, "pose_wrong": wrong if pose else None,
            "train_wall_s_driver": timing.get("train_s")}


def main():
    args = build_parser().parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    labels = args.labels or args.arms
    assert len(labels) == len(args.arms) <= len(COLOURS)
    arms = [load_arm(a, args.targets, args.batch) for a in args.arms]
    target = args.targets[-1]
    x = np.arange(len(arms))

    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    a = ax[0, 0]
    for i, (r, lab) in enumerate(zip(arms, labels)):
        s = [c[0] for c in r["curve"]]; m = [c[2] for c in r["curve"]]
        a.plot(s, m, color=COLOURS[i], marker=MARKERS[i], ms=4, lw=1.6, label=lab)
    for tg in args.targets:
        a.axhline(tg, color="k", ls="--", lw=0.8, alpha=0.45)
    a.text(0.01, target - 2.2, f"targets {', '.join(f'{tg:.0f}' for tg in args.targets)}%", fontsize=8, transform=a.get_yaxis_transform())
    a.axhline(REF_M04, color="k", ls=":", lw=1, alpha=0.6)
    a.text(0.99, REF_M04 + 0.5, f"trained board {REF_M04}%", fontsize=8, ha="right", transform=a.get_yaxis_transform())
    a.set_ylim(max(0, min(min(c[2] for c in r["curve"]) for r in arms) - 5), 101); a.set_xlabel("fine-tune step"); a.set_ylabel("ID accuracy, 1k val (%)")
    a.set_title("Learning curve: ID accuracy on the new board"); a.grid(alpha=0.3); a.legend(fontsize=7, loc="lower right")

    a = ax[0, 1]
    for i, (r, lab) in enumerate(zip(arms, labels)):
        s = [c[0] for c in r["curve"]]; p = [c[3] for c in r["curve"]]
        a.plot(s, p, color=COLOURS[i], marker=MARKERS[i], ms=4, lw=1.6, label=lab)
    a.set_xlabel("fine-tune step"); a.set_ylabel("corner error p95, 1k val (px)"); a.set_title("Localisation during fine-tuning (lower is better)"); a.grid(alpha=0.3)

    def bars(a, key, ylabel, title, fmt, scale=1.0):
        vals = [(r[key] or 0) * scale for r in arms]
        b = a.bar(x, vals, 0.7, color=[COLOURS[i] for i in range(len(arms))])
        a.bar_label(b, labels=[fmt % v if r[key] is not None else "n/a" for v, r in zip(vals, arms)], fontsize=7, padding=2)
        a.set_xticks(x, labels, rotation=35, ha="right", fontsize=7); a.set_ylabel(ylabel); a.set_title(title); a.grid(axis="y", alpha=0.3); a.set_axisbelow(True)

    def threshold_bars(a, key, ylabel, title, fmt, scale=1.0):
        # one group per arm, one bar per threshold (light -> dark); "n/a" where the arm never got there
        n_t = len(args.targets); w = 0.8 / n_t
        for j, tg in enumerate(args.targets):
            vals = [((r[key][tg] or 0) * scale) for r in arms]
            b = a.bar(x + (j - (n_t - 1) / 2) * w, vals, w * 0.9, color=[COLOURS[i] for i in range(len(arms))],
                      alpha=0.35 + 0.65 * j / max(n_t - 1, 1), edgecolor="white", linewidth=0.8)
            # one "n/a" per arm that never reached ANY threshold (on its first bar), not one per threshold
            a.bar_label(b, labels=[(fmt % v) if r[key][tg] is not None else ("n/a" if j == 0 else "")
                                   for v, r in zip(vals, arms)], fontsize=6, padding=2, rotation=90)
        a.set_xticks(x, labels, rotation=35, ha="right", fontsize=7); a.set_ylabel(ylabel); a.set_ylim(bottom=0)
        a.set_title(title + "  (bars: " + " / ".join(f"{tg:.0f}%" for tg in args.targets) + ", light to dark)", fontsize=10)
        a.grid(axis="y", alpha=0.3); a.set_axisbelow(True)
    threshold_bars(ax[0, 2], "steps_to", "steps", "Steps to reach ID threshold", "%.0f")
    threshold_bars(ax[1, 0], "wall_to", "minutes (this machine, incl. val)", "Wall time to reach ID threshold", "%.0f", 1 / 60)
    bars(ax[1, 1], "peak_mem_mb_bench", "peak allocated (MB)", f"Peak training VRAM, batch {args.batch} (isolated benchmark)", "%.0f")
    a = ax[1, 2]; w = 0.38
    m04 = [r["final_m04"] or 0 for r in arms]; pc = [r["pose_correct"] or 0 for r in arms]
    b1 = a.bar(x - w / 2, m04, w, color=[COLOURS[i] for i in range(len(arms))], label="ID accuracy, 10k val")
    b2 = a.bar(x + w / 2, pc, w, color=[COLOURS[i] for i in range(len(arms))], alpha=0.45, hatch="//", label="correct pose, 1k frames")
    a.bar_label(b1, labels=[f"{v:.1f}" for v in m04], fontsize=6, padding=1); a.bar_label(b2, labels=[f"{v:.0f}" for v in pc], fontsize=6, padding=1)
    a.axhline(REF_M04, color="k", ls=":", lw=1, alpha=0.6); a.set_ylim(0, 108)
    a.set_xticks(x, labels, rotation=35, ha="right", fontsize=7); a.set_ylabel("%"); a.set_title("Final accuracy on the new board (each arm at its own budget)"); a.legend(fontsize=7, loc="lower right"); a.grid(axis="y", alpha=0.3); a.set_axisbelow(True)
    fig.suptitle(args.title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); Path(args.out).parent.mkdir(parents=True, exist_ok=True); fig.savefig(args.out, dpi=args.dpi)
    if args.separate:   # alone, the p95 panel needs the arm legend it shares with the learning curve
        ax[0, 1].legend(fontsize=7); save_panels(fig, args.out, args.dpi)
    plt.close(fig)

    f = lambda v, fmt: (fmt % v) if v is not None else "n/a"
    rows = [f"| arm | trainable params | steps to {target:.0f}% | wall to {target:.0f}% | peak VRAM B16 (MB) | ms/step B16 | final ID 10k | p95 px | correct pose | wrong accepted | train wall |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r, lab in zip(arms, labels):
        rows.append(f"| {lab} | {f(r['n_trainable'], '%d')} | {f(r['steps_to_target'], '%d')} | {f(r['wall_to_target_s'] and r['wall_to_target_s']/60, '%.1f min')} | {f(r['peak_mem_mb_bench'], '%.0f')} | {f(r['ms_per_step_bench'], '%.1f')} | {f(r['final_m04'], '%.2f%%')} | {f(r['final_p95'], '%.4f')} | {f(r['pose_correct'], '%.1f%%')} | {f(r['pose_wrong'], '%.1f%%')} | {f(r['wall_elapsed_s'] and r['wall_elapsed_s']/60, '%.0f min')} |")
    rows.append("\nSteps to each threshold (1k in-loop val): " + "; ".join(f"{lab}: " + ", ".join(f"{tg:.0f}%->{r['steps_to'][tg] if r['steps_to'][tg] is not None else 'never'}" for tg in args.targets) for r, lab in zip(arms, labels)))
    Path(args.table).write_text(f"# Minimal fine-tuning ladder (headline target {target}% ID on the 1k in-loop val; trained-board reference {REF_M04}%)\n\n" + "\n".join(rows) + "\n")
    print("\n".join(rows)); print(f"-> {args.out}\n-> {args.table}")


if __name__ == "__main__":
    main()
