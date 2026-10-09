"""tools/plot_transfer.py -- the board-transfer summary figure: one model, several boards/arms.

Three panels, one bar group per arm, arms in the order given:
  1. corner localisation (M-01 median and p95, px, 10k full val)        -- lower is better
  2. identity and recall (M-04 ID accuracy; M-02 recall, octave mean)   -- higher is better
  3. pose outcome per frame (1000-frame B1 set), a 100% stacked bar:
     correct (<2 deg, accepted) / flagged ambiguous / WRONG but accepted / refused.
     The third slice is the one a docking controller cannot survive: a confidently wrong pose
     that passes the lattice gate (a self-consistent but permuted labelling -- the lattice is
     invariant under its own 90-degree symmetries) and is not flagged ambiguous.

Arms are `LABEL=fullval.json:pose.json` (tools/eval_checkpoint.py and tools/eval_pose_ours.py
outputs); the pose per-image JSONL is found beside pose.json as `<stem>_per_image.jsonl` and is
required for the outcome split. Reads only JSON -- restyling never needs a re-run. Built for the
fine-tuning story: add each fine-tuned arm as another --arm and the figure grows a bar.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dcc.viz import save_panels

# Okabe-Ito order, fixed and never cycled: validated CVD-safe for every adjacent pair (the tab10
# orange/green pair is indistinguishable to protanopes, dE 0.7, and was rejected for that reason).
ARM_COLOURS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9"]
OUTCOMES = [("correct", "correct (<2 deg)", "#009E73"), ("ambiguous", "flagged ambiguous", "#E69F00"),
            ("wrong", "WRONG, accepted", "#D55E00"), ("refused", "refused", "#56B4E9")]
WRONG_DEG = 2.0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", action="append", required=True, metavar="LABEL=fullval.json:pose.json")
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="Conv-ChArT 882k: board transfer")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--separate", action="store_true", help="also save each panel as its own image beside the figure")
    return p


def load_arm(spec):
    label, paths = spec.split("=", 1)
    fv_path, pose_path = paths.split(":")
    fv = json.loads(Path(fv_path).read_text())["detector"]
    pose = json.loads(Path(pose_path).read_text())
    per = [json.loads(l) for l in Path(pose_path).with_name(Path(pose_path).stem + "_per_image.jsonl").read_text().splitlines() if l.strip()]
    n = pose["n_images"]
    amb = sum(r["ambiguous"] for r in per)
    correct = sum(r["rot_deg"] < WRONG_DEG and not r["ambiguous"] for r in per)
    wrong = sum(r["rot_deg"] >= WRONG_DEG and not r["ambiguous"] for r in per)
    assert correct + wrong + amb == pose["n_solved"], (label, correct, wrong, amb, pose["n_solved"])
    m02 = [v for v in fv["m02"].values() if v is not None]
    return {"label": label, "median": fv["m01"]["median"], "p95": fv["m01"]["p95"],
            "id_acc": 100 * fv["m04"]["accuracy"], "recall": 100 * sum(m02) / len(m02),
            "outcome": {"correct": 100 * correct / n, "ambiguous": 100 * amb / n,
                        "wrong": 100 * wrong / n, "refused": 100 * pose["n_refused"] / n}}


def _bars(ax, arms, keys, names, ylabel, fmt):
    import numpy as np
    x, w = np.arange(len(keys)), 0.8 / len(arms)
    for i, a in enumerate(arms):
        vals = [a[k] for k in keys]
        b = ax.bar(x + (i - (len(arms) - 1) / 2) * w, vals, w * 0.9, color=ARM_COLOURS[i], label=a["label"])
        ax.bar_label(b, labels=[fmt % v for v in vals], fontsize=7, padding=2, rotation=90 if len(arms) > 3 else 0)
    ax.set_xticks(x, names); ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    ax.margins(y=0.18 if len(arms) > 3 else 0.05)   # headroom for rotated value labels


def main():
    args = build_parser().parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = [load_arm(s) for s in args.arm]
    assert len(arms) <= len(ARM_COLOURS), "add a colour to ARM_COLOURS before adding a 6th arm"

    fig, ax = plt.subplots(1, 3, figsize=(17.5, 4.8), gridspec_kw={"width_ratios": [1, 1, 1.35]})
    _bars(ax[0], arms, ["median", "p95"], ["median", "p95"], "corner error (px)  ↓ lower is better", "%.3f")
    ax[0].set_title("Corner localisation (M-01, 10k val)")
    _bars(ax[1], arms, ["id_acc", "recall"], ["ID accuracy\n(M-04)", "recall\n(M-02, octave mean)"],
          "%  ↑ higher is better", "%.1f")
    ax[1].set_ylim(0, 118 if len(arms) > 3 else 108); ax[1].set_title("Identification and recall (10k val)")
    # legend below the axes: inside, it hides whichever arm has collapsed (the zero-shot ID bar)
    leg = dict(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=min(len(arms), 3), frameon=False)
    ax[1].legend(**leg)

    bottom = [0.0] * len(arms)
    for key, name, colour in OUTCOMES:
        vals = [a["outcome"][key] for a in arms]
        b = ax[2].bar(range(len(arms)), vals, 0.6, bottom=bottom, color=colour, label=name, edgecolor="white", linewidth=1.5)
        ax[2].bar_label(b, labels=[f"{v:.1f}%" if v >= 4 else "" for v in vals], label_type="center", fontsize=8, color="white", weight="bold")
        bottom = [bt + v for bt, v in zip(bottom, vals)]
    ax[2].set_xticks(range(len(arms)), [a["label"] for a in arms], rotation=25 if len(arms) > 2 else 0,
                     ha="right" if len(arms) > 2 else "center", fontsize=7)
    ax[2].set_ylim(0, 100)
    ax[2].set_ylabel("frames (%)"); ax[2].set_title("Pose outcome per frame (1000 frames)")
    ax[2].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)   # outside: the arm names sit below
    fig.suptitle(args.title, fontsize=12)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95), w_pad=2.0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    if args.separate:   # alone, the localisation panel needs the arm legend too
        ax[0].legend(**leg); save_panels(fig, args.out, args.dpi)
    plt.close(fig)
    print(f"-> {args.out}")
    for a in arms:
        print(f"  {a['label']:28s} median {a['median']:.4f} p95 {a['p95']:.4f} id {a['id_acc']:.2f}% recall {a['recall']:.2f}% "
              f"pose correct {a['outcome']['correct']:.1f}% wrong {a['outcome']['wrong']:.1f}% amb {a['outcome']['ambiguous']:.1f}% refused {a['outcome']['refused']:.1f}%")


if __name__ == "__main__":
    main()
