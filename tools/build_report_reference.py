"""tools/build_report_reference.py -- assemble the single Conv-ChArT reference document for the thesis.

Writes paper/results_rev6/00_REPORT_REFERENCE/CONV_CHART_REFERENCE.md and copies every figure it embeds into
figures/ beside it (section-prefixed names). Part A is the facts-of-record document verbatim; Part B is every
figure, table and record of the results tree, by study, with historical (pre-release) material labelled as such;
Part C indexes every file. Tables are read from the JSON records, never retyped, so the document cannot drift
from the files it cites. Re-runnable and deterministic.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R6 = ROOT / "paper" / "results_rev6"
OUT = R6 / "00_REPORT_REFERENCE"
FIG = OUT / "figures"
COLOURS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#9467BD", "#8C510A", "#C7B33A", "#17BECF"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
HISTORICAL = ("**Historical, not a release model.** The figures/tables in this subsection were produced with the pre-ablation "
              "4,698,034-parameter `rev640` model or ablation-era checkpoints; they document the campaign and must not be "
              "quoted as results of the released tiers (see Part A, Section 19).")
doc, figlog = [], []


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT))


def h(level, text):
    doc.append("\n" + "#" * level + " " + text + "\n")


def fig(src, prefix, caption):
    src = Path(src)
    if not src.exists():
        doc.append(f"_missing figure: {rel(src)}_\n"); return
    dst = FIG / f"{prefix}__{src.stem}.png" if src.suffix.lower() == ".pdf" else FIG / f"{prefix}__{src.name}"
    if src.suffix.lower() == ".pdf":
        subprocess.run(["pdftoppm", "-r", "150", "-png", "-f", "1", "-l", "1", "-singlefile", str(src), str(dst.with_suffix(""))], check=True)
    else:
        shutil.copy2(src, dst)
    figlog.append((dst.name, rel(src)))
    doc.append(f"![{caption}](figures/{dst.name})\n<sub>{caption} -- source `{rel(src)}`</sub>\n")


def figs(paths, prefix, caption_fn=lambda p: p.stem):
    for p in sorted(paths):
        fig(p, prefix, caption_fn(Path(p)))


def include_md(path, shift=3, label=None, strip_title=True):
    p = Path(path)
    if not p.exists():
        doc.append(f"_missing file: {rel(p)}_\n"); return
    t = p.read_text(errors="replace")
    if t.startswith("---"):                                    # pandoc front matter
        t = t.split("---", 2)[2] if t.count("---") >= 2 else t
    lines = []
    for line in t.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            if strip_title and len(m.group(1)) == 1:
                continue
            line = "#" * min(6, len(m.group(1)) + shift - 1) + " " + m.group(2)
        lines.append(line)
    if label:
        doc.append(f"> {label}\n")
    doc.append(f"<sub>included verbatim from `{rel(p)}`</sub>\n")
    doc.append("\n".join(lines) + "\n")


def table(cols, rows, fmt="{}"):
    doc.append("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for r in rows:
        doc.append("| " + " | ".join(fmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    doc.append("")


def load(p):
    return json.load(open(p))


def pct(x):
    return "n/a" if x is None else f"{100 * x:.2f}%"


def f4(x):
    return "n/a" if x is None else f"{x:.4f}"


def mmp(block):
    return "n/a" if not block else " / ".join(f4(block.get(k)) for k in ("median", "mean", "p95"))


# ----------------------------------------------------------------------------------------------- figures from logs
def learning_curves(out_png, series, title):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    for i, (label, steps, m04, p95) in enumerate(series):
        c, m, ls = COLOURS[i % 10], MARKERS[i % 10], ["-", "--", ":"][i // 10 % 3]
        ax[0].plot(steps, [100 * v if v is not None else None for v in m04], ls, color=c, marker=m, ms=3.5, lw=1.3, label=label)
        ax[1].plot(steps, p95, ls, color=c, marker=m, ms=3.5, lw=1.3, label=label)
    ax[0].set_ylabel("ID accuracy M-04, in-loop val (%)"); ax[1].set_ylabel("corner error p95, in-loop val (px)")
    for a in ax:
        a.set_xlabel("step"); a.grid(alpha=0.3)
    ax[0].legend(fontsize=6.5, ncol=2, loc="lower right"); fig_.suptitle(title, fontsize=11); fig_.tight_layout()
    fig_.savefig(out_png, dpi=150); plt.close(fig_)


def release_curves():
    series = []
    for name, tier in [("rel_w25_lam2_100k_rev6", "222k"), ("rel_w375_lam15_100k_rev6", "502k"), ("rel_w882_c2_100k_rev6", "882k")]:
        p = ROOT / "runs" / name / "metrics.jsonl"
        if not p.exists():
            continue
        steps, m04, p95 = [], [], []
        for line in open(p):
            r = json.loads(line)
            if "val" in r:
                steps.append(r["step"]); m04.append(r["val"]["m04"]["accuracy"] if r["val"]["m04"] else None); p95.append(r["val"]["m01"]["p95"])
        series.append((tier, steps, m04, p95))
    if series:
        out = FIG / "B2__release_learning_curves.png"
        learning_curves(out, series, "Release tiers, 100,000 steps: in-loop 2,000-sample validation every 2,500 steps (EMA weights)")
        figlog.append((out.name, "runs/rel_*/metrics.jsonl"))
        doc.append(f"![release learning curves](figures/{out.name})\n<sub>generated by this builder from `runs/rel_*/metrics.jsonl`; full 10k validation values are in Part A, Section 7</sub>\n")
    else:
        doc.append("_release run logs not present on this machine; curves not generated_\n")


def ablation_curves(group, arms, title):
    series = []
    for arm in arms:
        p = R6 / "08_ablations" / arm / "train_metrics.json"
        if not p.exists():
            continue
        c = load(p)["curve"]
        series.append((arm.replace("_35k_rev6", "").replace("_50k_rev6", " (50k)").replace("_30k_rev6", " (30k)"),
                       [r["step"] for r in c], [r["m04"] for r in c], [r["m01_p95"] for r in c]))
    if not series:
        return
    out = FIG / f"B6__curves_{group}.png"
    learning_curves(out, series, title)
    figlog.append((out.name, "paper/results_rev6/08_ablations/*/train_metrics.json"))
    doc.append(f"![{title}](figures/{out.name})\n<sub>generated by this builder from each arm's `train_metrics.json` (in-loop 2,000-sample validation)</sub>\n")


# ----------------------------------------------------------------------------------------------- build
def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    FIG.mkdir(parents=True)
    doc.append("# Conv-ChArT -- complete reference for the thesis (report branch)\n")
    doc.append("Assembled by `tools/build_report_reference.py` on 2026-10-10 from `paper/results_rev6/`. **Part A** is the facts-of-record "
               "document verbatim (every number traceable to a file; Section 19 lists where older documents disagree). **Part B** holds "
               "every figure, table and record of the results tree, by study, with pre-release material labelled *historical*. "
               "**Part C** indexes every file. Every embedded figure is also a file in `figures/` (name = `<section>__<original>`), "
               "and the table under Part C maps each back to its source. Release models of record: `rel_w25_lam2_100k_rev6` (222k), "
               "`rel_w375_lam15_100k_rev6` (502k), `rel_w882_c2_100k_rev6` (882k), refiner `ref_s15_10k_rev6`.\n")

    # ---------------- Part A
    h(2, "Part A -- Facts of record (verbatim, `00_FACTS_FOR_REPORT.md`, 2026-10-09)")
    include_md(R6 / "00_FACTS_FOR_REPORT.md", shift=3)

    # ---------------- Part B
    h(2, "Part B -- Figures, tables and records by study")

    h(3, "B1. Paper figures and conference deliverables")
    doc.append("Paper source `paper/conv_chart.tex` (PDF `paper/conv_chart.pdf`); conference deck `paper/conference/`.\n")
    figs((ROOT / "paper" / "figs").glob("*"), "B1_paper")
    figs((ROOT / "figures").glob("*.png"), "B1_deliverable")
    figs((ROOT / "figures").glob("*.pdf"), "B1_deliverable")
    for s in (ROOT / "figures").glob("*.svg"):
        shutil.copy2(s, FIG / f"B1_deliverable__{s.name}"); figlog.append((f"B1_deliverable__{s.name}", rel(s)))
        doc.append(f"![{s.stem}](figures/B1_deliverable__{s.name})\n<sub>{s.stem} -- source `{rel(s)}`</sub>\n")

    h(3, "B2. Release-run learning curves (generated from the training logs)")
    release_curves()

    h(3, "B3. Release detector validation records")
    rows = []
    for tier, name in [("882k", "rel_w882_c2_100k_rev6"), ("502k", "rel_w375_lam15_100k_rev6"), ("222k", "rel_w25_lam2_100k_rev6")]:
        p = ROOT / "runs" / name / "metrics.jsonl"
        if not p.exists():
            continue
        for line in open(p):
            r = json.loads(line)
            if "full_val" in r:
                v = r["full_val"]
                rows.append([tier, r["step"], f4(v["m01"]["median"]), f4(v["m01"]["p95"]), pct(v["m01"].get("tail_frac_gt4px")), pct(v["m04"]["accuracy"]),
                             " / ".join(pct(v["m02"][k]) for k in ("12-16", "16-32", "32-64", "64-128")), v["m01"].get("n_matched")])
    doc.append("Full 10,000-sample validations at every 25,000 steps (`runs/rel_*/metrics.jsonl`, EMA weights):\n")
    table(["tier", "step", "M-01 median px", "p95 px", "tail > 4 px", "M-04", "M-02 recall 12-16 / 16-32 / 32-64 / 64-128", "matched corners"], rows)
    doc.append("Control and zero-shot full validations of the 882k model (`27_transfer_zeroshot/fullval_*.json`, `tools/eval_checkpoint.py`):\n")
    rows = []
    for lab, f in [("882k, trained board DICT_5X5_50", "27_transfer_zeroshot/fullval_control_dict5x5.json"), ("882k, zero-shot DICT_6X6_250", "27_transfer_zeroshot/fullval_transfer_dict6x6.json"),
                   ("fine-tuned 5k, head+bneck 3x (ft18)", "30_REPORT_transfer_finetune/fullval_ft18_head_bneck_lr3x_hm_lr0p1.json"), ("fine-tuned 20k, head+bneck 3x (ft21)", "30_REPORT_transfer_finetune/fullval_ft21_head_bneck_lr3x_hm_lr0p1_20k.json"),
                   ("fine-tuned 5k, whole model 3x (ft20)", "30_REPORT_transfer_finetune/fullval_ft20_full_lr3x.json"), ("3-family base, original board", "29_multiboard_base/mb3/fullval_dict5x5_10k.json"), ("3-family base, zero-shot DICT_6X6_250", "29_multiboard_base/mb3/zeroshot_fullval_dict6x6_10k.json")]:
        p = R6 / f
        if p.exists():
            v = load(p)["detector"]
            rows.append([lab, f4(v["m01"]["median"]), f4(v["m01"]["p95"]), pct(v["m01"].get("tail_frac_gt4px")), pct(v["m04"]["accuracy"]),
                         " / ".join(pct(v["m04"]["by_octave"][k]) for k in ("12-16", "16-32", "32-64", "64-128")), " / ".join(pct(v["m02"][k]) for k in ("12-16", "16-32", "32-64", "64-128"))])
    table(["model / board", "median px", "p95 px", "tail > 4 px", "M-04", "M-04 by octave", "M-02 by octave"], rows)

    h(3, "B4. Robustness: release tiers vs Deep ChArUco vs classical, identical frames")
    doc.append("Per-factor figures (`tools/plot_four_way.py`): 882k vs 502k, and 882k vs 222k, each against the fine-tuned Deep ChArUco and classical OpenCV arms. "
               "Our coarse and refined arms are both drawn; Deep ChArUco is unrefined (read against our coarse arm); classical ID is not comparable (reports only identified corners).\n")
    figs((R6 / "22_fourway_RELEASE").glob("*.png"), "B4_882_vs_502")
    figs((R6 / "22_fourway_RELEASE_882_vs_222").glob("*.png"), "B4_882_vs_222")
    doc.append("\n**Per-step tables** (refined arm for ours, n = 60 frames per step; Deep ChArUco and classical n = 100, same seed 20260728, match 4.0 px). "
               "Columns: 882k | 502k | 222k | Deep ChArUco | classical. Localisation is the median error of matched corners in px (refined for ours, unrefined for the baselines).\n")
    arms = {"882k": R6 / "20_robustness_REL882", "502k": R6 / "20_robustness_REL502", "222k": R6 / "20_robustness_REL222", "DC": R6 / "05_comparison" / "robustness_dc", "classical": R6 / "05_comparison" / "robustness_classical"}
    for f in sorted((R6 / "20_robustness_REL882").glob("*.json")):
        fac = f.stem
        data = {k: load(d / f"{fac}.json") for k, d in arms.items() if (d / f"{fac}.json").exists()}
        ref = data["882k"]
        doc.append(f"\n**{fac}** -- unit: {ref.get('unit')}; steps: {[s['value'] for s in ref['steps']]}\n")
        rows = []
        for i, st in enumerate(ref["steps"]):
            cells = [st["value"]]
            for metric in ("recall", "id_acc", "err_median"):
                vals = []
                for k, d in data.items():
                    a = d["steps"][i]["arms"]; arm = a.get("refined") or a[list(a)[0]]; v = arm.get(metric)
                    if metric == "id_acc" and k == "classical":
                        vals.append("n/c")
                    elif v is None:
                        vals.append("n/a")
                    else:
                        vals.append(f"{100 * v:.1f}" if metric != "err_median" else f"{v:.3f}")
                cells.append(" / ".join(vals))
            rows.append(cells)
        table(["step", "recall % (882k / 502k / 222k / DC / classical)", "ID % among matched", "localisation median px"], rows)

    h(3, "B5. Pose benchmark B1 (1000 frames, corrected set)")
    rows = []
    for tier in ("882", "502", "222"):
        d = load(R6 / f"21_pose_REL{tier}_B1.json")
        rows.append([tier + "k", d["n_solved"], pct(d["solve_rate"]), d["n_ambiguous"], json.dumps(d["refusal_reasons"]),
                     mmp(d.get("m05_rot_deg")), mmp(d.get("m06_trans_squares")), json.dumps(d.get("nees"))[:120]])
    table(["tier", "solved", "solve rate", "ambiguous", "refusal reasons", "rot deg median / mean / p95", "trans sq median / mean / p95", "NEES (pose_cov; not a filter input)"], rows)
    d = load(R6 / "13_pose" / "pose_release_vs_baselines_B1.json")
    rows = [[k, v["n_solved"], pct(v["solve_rate"]), " / ".join(f4(v["rot_deg"][m]) for m in ("median", "mean", "p95")), " / ".join(f4(v["trans_squares"][m]) for m in ("median", "mean", "p95"))] for k, v in d["arms"].items()]
    doc.append("All eight arms through the same PnP (`13_pose/pose_release_vs_baselines_B1.json`):\n")
    table(["arm", "solved", "solve rate", "rot deg median / mean / p95", "trans squares median / mean / p95"], rows)
    d = load(R6 / "13_pose" / "pose_fourway_B1.json")
    doc.append("Baseline deltas file (`13_pose/pose_fourway_B1.json`; its `ours_*` rows are the 4.7M model and must not be quoted):\n")
    table(["arm", "solved", "solve rate", "rot deg median / mean / p95", "trans squares median / mean / p95"],
          [[k, v["n_solved"], pct(v["solve_rate"]), " / ".join(f4(v["rot_deg"][m]) for m in ("median", "mean", "p95")), " / ".join(f4(v["trans_squares"][m]) for m in ("median", "mean", "p95"))] for k, v in d["arms"].items()])
    doc.append(HISTORICAL + " The pose-table PDFs below are the conference-era tables.\n")
    figs((R6 / "13_pose").glob("*.pdf"), "B5_pose_tables_historical")
    for m in sorted((R6 / "13_pose").glob("*.md")):
        include_md(m, shift=5, label="historical pose table (pre-release models)")

    h(3, "B6. Ablations: every arm's own record")
    doc.append("All arms at a matched budget (35k unless the name says otherwise), one key changed per arm, verified before launch. Values are the arm's final in-loop "
               "and full validations as banked by `tools/ablation_summary.py`. Deltas are valid at matched budget; absolute values are budget-limited (Part A, Section 11).\n")
    rows = []
    for p in sorted((R6 / "08_ablations").glob("*/train_metrics.json")):
        d = load(p); f = d["final_val"]; oc = d.get("m04_by_octave_final") or {}
        full = "no run log here"
        log_ = ROOT / "runs" / d["run"] / "metrics.jsonl"
        if log_.exists():
            fv = [json.loads(l) for l in open(log_) if '"full_val"' in l]
            if fv:
                r = fv[-1]; v = r["full_val"]
                full = f"{f4(v['m01']['median'])} / {f4(v['m01']['p95'])} / {pct(v['m01'].get('tail_frac_gt4px'))} / {pct(v['m04']['accuracy'])} @ {r['step']}"
            else:
                full = "none logged"
        rows.append([p.parent.name, d["steps_trained"], f4(f.get("m01_median")), f4(f.get("m01_p95")), pct(f.get("m01_tail_gt4px")), pct(f.get("m04")),
                     " / ".join(pct(oc.get(k)) for k in ("12-16", "16-32", "32-64", "64-128")), full])
    doc.append("**Summary of all 61 banked arms.** The first block of columns is the arm's final **in-loop 2,000-sample** validation "
               "(`08_ablations/<arm>/train_metrics.json`, which is what the per-arm records below hold); the last column is the "
               "**10,000-sample full validation** at the final step from the run log where this machine has it. Part A and the paper "
               "quote the full validation; the in-loop value reads about 0.1-0.2 pp optimistic (campaign log, Part 1 header), so the two "
               "differ slightly for the same arm by construction, e.g. `abl_wh_ce_35k_rev6` p95 0.6983 in-loop vs 0.6970 full.\n")
    table(["arm", "steps", "in-loop median px", "in-loop p95 px", "in-loop tail > 4 px", "in-loop M-04", "in-loop M-04 by octave", "10k full val: median / p95 / tail / M-04 @ step"], rows)
    doc.append("**Learning curves by width group** (in-loop validation of every banked arm):\n")
    ablation_curves("882k_supervision", ["abl_wh_ctrl35k_rev6", "abl_wh_ce_35k_rev6", "abl_lx_wh_hmce_clsfocal_35k_rev6", "abl_lx_wh_hmfocal_clsce_35k_rev6", "abl_c2_wh_clsfocal_lam2_35k_rev6", "abl_c3_wh_lam2_35k_rev6", "abl_wh_s1_35k_rev6", "abl_wh_s025_35k_rev6"], "882k: supervision arms (control = focal/focal sigma 0.5)")
    ablation_curves("882k_structure", ["abl_wh_ctrl35k_rev6", "abl_wh_heads4_35k_rev6", "abl_wh_attend16_35k_rev6", "abl_wh_attn1_35k_rev6", "abl_wh_rope5_35k_rev6", "abl_g_w50_a2_ce_nogate_35k_rev6", "abl_t_w50_a1_ce_35k_rev6"], "882k: structural arms")
    ablation_curves("502k", sorted(p.name for p in (R6 / "08_ablations").glob("*w375*")) + ["abl_r502_g32_35k_rev6", "abl_r502_lam2_35k_rev6", "abl_c5_w375_g32_lam2_scls05_35k_rev6"], "502k arms (incl. the four noise-floor seeds)")
    ablation_curves("222k", sorted(p.name for p in (R6 / "08_ablations").glob("*fast*")) + sorted(p.name for p in (R6 / "08_ablations").glob("*w25*")) + ["abl_c1_fast_g32_lam2_35k_rev6", "abl_c4_fast_g32_lam2_scls05_35k_rev6"], "222k arms (incl. the four noise-floor seeds)")
    ablation_curves("4p7M_and_width", ["abl_composite_s05_50k_rev6", "abl_sigma025_50k_rev6", "abl_width_half_50k_rev6", "abl_nodilate_width_half_s05_50k_rev6", "abl_nodilate_width_half_xsa_s05_50k_rev6", "abl_nodilate_width_quarter_s05_30k_rev6", "abl_nodilate_width_quarter_xsa_s05_30k_rev6"], "4.7M-base and width arms (50k / 30k budgets)")
    doc.append("\n**Per-arm records** (each arm's `train_metrics.md`, verbatim; in-loop 2,000-sample validation, see the note above):\n")
    for p in sorted((R6 / "08_ablations").glob("*/train_metrics.md")):
        include_md(p, shift=5, strip_title=False)
    doc.append("\n**Per-regime ablations on the 4.7M base** (A1 conv-only, A-CE one-hot, A-XSA; sweeps on identical frames, n = 100/step, coarse arm; the reference is `abl_reference_50k_rev6`):\n")
    for a in ("A1_conv_only", "A_CE_one_hot", "A_XSA"):
        d = R6 / "08_ablations" / a
        h(4, f"{a}")
        if (d / "README.md").exists():
            include_md(d / "README.md", shift=5)
        for m in sorted(d.glob("ablation_*.md")):
            include_md(m, shift=5, strip_title=False)
        figs(d.glob("ablation_*.png"), f"B6_{a}")
    doc.append("\n**Peak sharpness** (`08_ablations_peak_sharpness.json`; the mechanism behind the one-hot heatmap result, 4.7M base):\n")
    d = load(R6 / "08_ablations_peak_sharpness.json")
    table(["arm", "step", "n", "spread median px", "spread mean px", "peak median", "concentration median"], [[k, v["step"], v["n"], f4(v["spread_med"]), f4(v["spread_mean"]), f4(v["peak_med"]), f4(v["conc_med"])] for k, v in d.items()])
    doc.append("Notes of record (links): `08_ablations_NOTE_sigma_justification.md`, `08_ablations/00_QUEUE_AND_DECISION_RULE.md`, `08_ablations/00_BACK_POCKET_deferred_arms.md`, `08_ablations/00_README_LAYOUT.md`, `08_ablations/HEAD_TO_HEAD.md`.\n")
    doc.append("(`08_ablations/HEAD_TO_HEAD.md` is a mid-campaign admin snapshot with stale running/stopped markers and is deliberately not reproduced; the summary table above supersedes it.)\n")

    h(3, "B7. Refiner (`02_refiner`)")
    doc.append("Guard-sweep numbers of record are in Part A, Section 5 (n = 6,421 crops). Raw per-crop records: `guard_sweep.json`, `tail_composition.json`.\n")
    d = load(R6 / "02_refiner" / "capture_clipping.json")
    doc.append("Capture clipping per apparent scale (`capture_clipping.json`):\n")
    table(list(d[0].keys()), [[r[k] for k in d[0].keys()] for r in d], fmt="{:.4f}")
    d = load(R6 / "02_refiner" / "crop_scale.json")
    doc.append(f"Crop-extent sweep (`crop_scale.json`; {d['provenance']['note'][:300]}):\n")
    table(list(d["rows"][0].keys()), [[r[k] for k in d["rows"][0].keys()] for r in d["rows"]], fmt="{:.4f}")
    d = load(R6 / "02_refiner" / "saturation_attribution.json")
    doc.append("Saturation attribution (`saturation_attribution.json`; fractions of crops blown out / crushed / blank per generator variant):\n")
    table(list(d[0].keys()), [[r[k] for k in d[0].keys()] for r in d], fmt="{:.4f}")
    fig(R6 / "02_refiner" / "failure_gallery.png", "B7_refiner", "refiner failure gallery: crops where the refiner is worse than the coarse peak")

    h(3, "B8. Loss and supervision records (`12_loss`)")
    include_md(R6 / "12_loss" / "cls_form_gradient_share.md", shift=4)
    include_md(R6 / "12_loss" / "loss_formulation.md", shift=4, label="HISTORICAL formulation note: written for the 4.7M model with focal on both heads, sigma_hm 2.0 and lambda_cls 1.0; the released supervision is in Part A, Section 6. Kept for the target construction, which is unchanged.")
    figs((R6 / "12_loss").glob("*.pdf"), "B8_loss")
    fig(R6 / "07_introspection" / "loss_target.png", "B8_loss", "rendered heatmap and class targets for one frame")
    doc.append("Long-form explanation: `12_loss/loss_explained.md` (440 lines) and the interactive `12_loss/focal_target_explorer.html`.\n")

    h(3, "B9. Receptive field and attention gate (`10_receptive_field`, `11_attention_gate`)")
    include_md(R6 / "10_receptive_field" / "receptive_field.md", shift=4, label="Constants are the 4.7M model's (dilated pair present); the release model is the 'without the pair' column: 68 px theoretical, ~21 px effective.")
    figs((R6 / "10_receptive_field").glob("*.png"), "B9_rf")
    include_md(R6 / "11_attention_gate" / "attention_gate.md", shift=4, label="Gate dimensions and the 24,705-parameter count are the 4.7M model's; at width 0.5 the gate is 64->32 / 128->32 / 32->1 with 6,209 parameters (Part A, Section 4). Formulation, initialisation and the measured alpha statistics are as stated.")
    figs((R6 / "11_attention_gate").glob("*.pdf"), "B9_gate")

    h(3, "B10. Introspection (`07_introspection`, `07_introspection_L`) -- " + "historical models")
    doc.append(HISTORICAL + " Attention, gate and feature panels were rendered on `rev640_160k_rev6` (4.7M) and the composite `L` checkpoint; `width_half_s05_step7000/` is an early 882k-class checkpoint at step 7,000.\n")
    d = load(R6 / "07_introspection" / "attention_frames.json")
    import statistics as st
    rows = d["rows"]
    doc.append(f"Attention statistics over {len(rows)} validation frames (`attention_frames.json`, {d['ckpt']} step {d['ckpt_step']}; uniform entropy ln T = {d['uniform_entropy']:.3f} nats):\n")
    table(["block", "median entropy (nats)", "min", "max", "median board mass", "min", "max"],
          [[b, f4(st.median(r["entropy"][b] for r in rows)), f4(min(r["entropy"][b] for r in rows)), f4(max(r["entropy"][b] for r in rows)),
            f4(st.median(r["board_mass"][b] for r in rows)), f4(min(r["board_mass"][b] for r in rows)), f4(max(r["board_mass"][b] for r in rows))] for b in (0, 1)])
    d = load(R6 / "07_introspection" / "attention_head_roles.json")
    doc.append(f"Attention head roles (`attention_head_roles.json`, {d['ckpt']}, {d['n_frames']} frames; median attention mass on corners / markers / elsewhere per head, and lift over uniform):\n")
    for b in d["blocks"]:
        table(["block " + str(b["block"]) + " head", "mass corner", "mass marker", "mass off", "lift corner", "lift marker", "lift off", "marker reader"],
              [[i, f4(b["median_mass_corner"][i]), f4(b["median_mass_marker"][i]), f4(b["median_mass_off"][i]), f4(b["median_lift_corner"][i]), f4(b["median_lift_marker"][i]), f4(b["median_lift_off"][i]), i in b["marker_readers"]] for i in range(b["n_heads"])])
    figs([p for p in (R6 / "07_introspection").glob("*.png") if p.name != "loss_target.png"], "B10_introspection")
    figs((R6 / "07_introspection" / "width_half_s05_step7000").glob("*.png"), "B10_w05_step7000")
    figs((R6 / "07_introspection_L").glob("*.png"), "B10_L")

    h(3, "B11. Training data (`14_data`)")
    fig(R6 / "14_data" / "training_samples_3x3.png", "B11_data", "nine deliberately contrasted training samples (tools/sample_board.py)")
    doc.append("```\n" + (R6 / "14_data" / "augmentations.txt").read_text() + "```\n")
    fig(R6 / "29_multiboard_base" / "pool_sheet.png", "B11_data", "16 generator samples from the 15-board pool (multi-board pretraining)")

    h(3, "B12. Cost, memory, latency and export (`15_cost`, `24_export_parity*`, `25_runtime_latency`)")
    d = load(R6 / "15_cost" / "vram_report_tiers.json")
    doc.append(f"VRAM at batch 1 (`vram_report_tiers.json`; {d['note'][:200]}...):\n")
    table(["variant", "params", "weights fp32 MB", "weights fp16 MB", "activations MB", "torch reserved MB", "onnx estimate MB"],
          [[k, v["params"], f"{v['weights_fp32_mb']:.2f}", f"{v['weights_fp16_mb']:.2f}", f"{v['activations_mb']:.1f}", v["torch_reserved_mb"], f"{v['onnx_estimate_mb'][0]:.0f}-{v['onnx_estimate_mb'][1]:.0f}"] for k, v in d["rows"].items()])
    d = load(R6 / "15_cost" / "model_cost.json")
    doc.append(f"`model_cost.json` (4.7M model, historical; MACs {d['macs']['total_G']:.1f} G, {d['macs']['total_GFLOPs']:.1f} GFLOPs/frame; measured 5090 fp16 {d['measured_5090_fp16_ms']['total']:.2f} ms, contended; Orin figures are datasheet rooflines).\n")
    include_md(R6 / "15_cost" / "cost_table.md", shift=4, label="HISTORICAL cost table: the ID columns are 35k / 45k ablation checkpoints, the Orin columns are rooflines, not measurements; GFLOPs (52.0 / 19.8) are architecture-level and valid.")
    for name in ("tau_sweep_L_composite_s05.json", "tau_sweep_smoke.json"):
        d = load(R6 / "15_cost" / name)
        doc.append(f"`{name}` ({d.get('label', '')}, {d.get('ckpt', '')} step {d.get('step')}, n {d.get('n')}): heatmap threshold sweep\n")
        table(list(d["rows"][0].keys()), [[r[k] for k in d["rows"][0].keys()] for r in d["rows"]], fmt="{:.4f}")
    for name, lab in (("24_export_parity/export_parity.json", "both networks fp16"), ("24_export_parity_detfp16/export_parity.json", "detector fp16, refiner fp32")):
        d = load(R6 / name)
        doc.append(f"P15 export parity, {lab} (`{name}`, n = {d.get('n_val')}, thresholds {d.get('thresholds')}):\n")
        tiers = d.get("tiers", {})
        rows = []
        for t, v in tiers.items():
            onnx = v.get("onnx", {}); par = v.get("parity") or v.get("fp16_parity") or {k: v[k] for k in v if k not in ("onnx", "ckpt", "config_input_size")}
            rows.append([t, v.get("n_params"), onnx.get("n_nodes"), onnx.get("size_mb"), ", ".join(onnx.get("ops", []))])
            doc.append(f"parity record, {t}: `{json.dumps(par)}`\n")
        table(["tier", "params", "ONNX nodes", "MB", "ops"], rows)
        if "refiner_onnx" in d:
            doc.append(f"refiner ONNX: {json.dumps(d['refiner_onnx'])}\n")
    for name in ("runtime_latency.json", "trt_vs_cuda_ep.json", "python_detect_stages.json"):
        doc.append(f"`25_runtime_latency/{name}`:\n```json\n" + json.dumps(load(R6 / "25_runtime_latency" / name), indent=1) + "\n```\n")

    h(3, "B13. Working range (`17_range/working_range.json`)")
    d = load(R6 / "17_range" / "working_range.json")
    doc.append(f"Board {d['board_mm']} mm ({d['square_mm']} mm squares), OV2311 pitch {d['pixel_pitch_um']} um, rho {d['rho']} (1600x1200 -> 640x480), bands from measured ID/recall: {json.dumps(d['bands_s'])}\n")
    table(["lens", "f_px at input", "HFOV deg", "core m", "usable m", "degraded m"],
          [[k, f"{v['f_px_input']:.0f}", f"{v['hfov_deg']:.1f}", f"{v['core']['near_m']:.2f}-{v['core']['far_m']:.2f}", f"{v['usable']['near_m']:.2f}-{v['usable']['far_m']:.2f}", f"{v['degraded']['near_m']:.2f}-{v['degraded']['far_m']:.2f}"] for k, v in d["ranges_m"].items()])
    table(["s px", "ID", "recall"], [[k, pct(v["id_acc"]), pct(v["recall"])] for k, v in d["measured"].items()])

    h(3, "B14. SNR calibration of the sensor-noise axis (`03_robustness/snr_calibration.json`)")
    d = load(R6 / "03_robustness" / "snr_calibration.json")
    doc.append(f"{d['model']}; gray attenuation {d['gray_attenuation']:.4f}; {d['snr_db_note'][:300]}\n")
    table(list(d["rows"][0].keys()), [[r[k] for k in d["rows"][0].keys()] for r in d["rows"]], fmt="{:.3f}")

    h(3, "B15. Measurement noise for the Kalman filter (`26_pose_error_variance`)")
    fig(R6 / "26_pose_error_variance" / "pose_error_vs_range.png", "B15_noise", "pose error vs range: robust sigma per octave with power-law fits; constant vs range-scaled R")
    include_md(R6 / "26_pose_error_variance" / "README.md", shift=4)
    include_md(R6 / "26_pose_error_variance" / "KALMAN_COVARIANCE_HOWTO.md", shift=4)

    h(3, "B16. Board transfer, zero-shot (`27_transfer_zeroshot`)")
    include_md(R6 / "27_transfer_zeroshot" / "README.md", shift=4)
    include_md(R6 / "27_transfer_zeroshot" / "robustness_hardest_step.md", shift=4, strip_title=False)
    fig(R6 / "27_transfer_zeroshot" / "boards_dict5x5_vs_dict6x6.png", "B16_transfer", "trained board (DICT_5X5_50) and new board (DICT_6X6_250)")
    fig(R6 / "27_transfer_zeroshot" / "transfer_summary.png", "B16_transfer", "zero-shot transfer summary")
    figs((R6 / "27_transfer_zeroshot" / "figures_vs_control").glob("*.png"), "B16_vs_control")
    figs((R6 / "27_transfer_zeroshot" / "figures_zeroshot_only").glob("*.png"), "B16_zeroshot_only")

    h(3, "B17. Minimal fine-tuning (`28_finetune_minimal`)")
    include_md(R6 / "28_finetune_minimal" / "README.md", shift=4)
    for m in ("finetune_ladder_round1.md", "finetune_ladder_rate.md", "finetune_ladder_bases.md"):
        include_md(R6 / "28_finetune_minimal" / m, shift=4, strip_title=False)
    figs((R6 / "28_finetune_minimal").glob("*.png"), "B17_finetune")
    d = load(R6 / "28_finetune_minimal" / "finetune_cost.json")
    doc.append(f"Isolated GPU cost per trainable set (`finetune_cost.json`; {d['gpu']}, {d['note']}):\n")
    table(["arm @ batch", "params", "trainable", "peak alloc MB", "reserved MB", "ms/step", "ms/step p90", "optimiser MB", "lr_mult"],
          [[k, v["n_params"], v["n_trainable"], f"{v['peak_mem_mb']:.0f}", f"{v['reserved_mb']:.0f}", f"{v['ms_per_step']:.1f}", f"{v['ms_per_step_p90']:.1f}", f"{v['optim_state_mb']:.1f}", json.dumps(v.get("lr_mult"))[:120]] for k, v in d["rows"].items()])
    rows = [json.loads(l) for l in open(R6 / "28_finetune_minimal" / "timing.jsonl") if l.strip()]
    doc.append("Wall time per arm as run (two arms shared the GPU; `timing.jsonl`):\n")
    f0 = lambda x: "n/a" if x is None else f"{x:.0f}"
    table(["arm", "train s", "full val s", "pose s", "return codes"], [[r["arm"], f0(r.get("train_s")), f0(r.get("fullval_s")), f0(r.get("pose_s")), json.dumps(r.get("rc"))] for r in rows])
    rows = []
    for p in sorted((R6 / "28_finetune_minimal").glob("pose_*.json")):
        if p.name.endswith("_per_image.jsonl"):
            continue
        d = load(p)
        rows.append([p.stem.replace("pose_", ""), d["n_solved"], pct(d["solve_rate"]), d["n_ambiguous"], json.dumps(d["refusal_reasons"])[:120], mmp(d.get("m05_rot_deg")), mmp(d.get("m06_trans_squares"))])
    doc.append("Pose benchmark per fine-tuned arm (`pose_<arm>.json`):\n")
    table(["arm", "solved", "solve rate", "ambiguous", "refusal reasons", "rot deg median / mean / p95", "trans sq median / mean / p95"], rows)

    h(3, "B18. Multi-board pretraining (`29_multiboard_base`)")
    include_md(R6 / "29_multiboard_base" / "README.md", shift=4)
    include_md(R6 / "29_multiboard_base" / "diag_step14k" / "NOTE.md", shift=4, strip_title=False)
    rows = []
    for p in sorted((R6 / "29_multiboard_base" / "mb3").glob("perboard_*.json")):
        v = load(p)["detector"]; rows.append([p.stem.replace("perboard_", ""), f4(v["m01"]["median"]), f4(v["m01"]["p95"]), pct(v["m04"]["accuracy"]), " / ".join(pct(v["m02"][k]) for k in ("12-16", "16-32", "32-64", "64-128"))])
    doc.append("3-family base, per training board (2,000 samples each; `mb3/perboard_*.json`):\n")
    table(["board", "median px", "p95 px", "M-04", "M-02 by octave"], rows)
    rows = []
    for p in sorted((R6 / "29_multiboard_base" / "mb3").glob("*pose*_B1.json")):
        d = load(p); rows.append([p.stem, d["n_solved"], pct(d["solve_rate"]), d["n_ambiguous"], json.dumps(d["refusal_reasons"])[:120], mmp(d.get("m05_rot_deg"))])
    table(["pose set", "solved", "solve rate", "ambiguous", "refusal reasons", "rot deg median / mean / p95"], rows)

    h(3, "B19. Report figures for transfer and fine-tuning (`30_REPORT_transfer_finetune`)")
    include_md(R6 / "30_REPORT_transfer_finetune" / "REPORT.md", shift=4)
    include_md(R6 / "30_REPORT_transfer_finetune" / "robustness_hardest_step_3way.md", shift=4, strip_title=False)
    figs((R6 / "30_REPORT_transfer_finetune" / "figures").glob("*.png"), "B19_report")
    doc.append("Every panel of the composites above is also a separate file: `30_REPORT_transfer_finetune/figures_separate/` (49 files, listed at the end of REPORT.md above).\n")
    figs((R6 / "30_REPORT_transfer_finetune" / "robustness_figs_finetuned_vs_trained").glob("*.png"), "B19_ft_vs_trained")

    h(3, "B20. Print-ready board (`31_print_board`)")
    include_md(R6 / "31_print_board" / "README.md", shift=4)
    fig(R6 / "31_print_board" / "DICT_5X5_50_5x5_24mm.png", "B20_board", "the release board at 24 mm squares (120 mm edge), 600 dpi raster")

    h(3, "B21. Historical sweeps and tables (pre-release models)")
    doc.append(HISTORICAL + "\n")
    for d, lab in (("03_robustness", "rev640_160k_rev6 (4.7M), n = 100/step"), ("03_robustness_PREGUARD", "rev640_160k_rev6 before the refiner guard")):
        h(4, f"`{d}` -- {lab}")
        figs((R6 / d / "figures").glob("*.png"), f"B21_{d}")
    h(4, "`05_comparison/figures` -- 4.7M vs Deep ChArUco vs classical, and the lighting filmstrip")
    figs((R6 / "05_comparison" / "figures").glob("*.png"), "B21_05_comparison")
    h(4, "`06_tables` -- inter-method tables, 'Ours' = 4.7M")
    for m in sorted((R6 / "06_tables").glob("*.md")):
        include_md(m, shift=5, strip_title=False)

    h(3, "B22. Campaign log, audit and knowledge documents")
    doc.append("- `PLAN_autonomous_campaign.md` -- the dated campaign log (1,084 lines): every decision with its evidence; headings:\n")
    doc.append("```\n" + "\n".join(l for l in (R6 / "PLAN_autonomous_campaign.md").read_text().splitlines() if l.startswith("## ")) + "\n```\n")
    doc.append("- `23_code_audit_FINDINGS.md` -- audit, P15 fp16 gate, B1 correction (tables reproduced in Part A).\n- `docs/PROJECT_KNOWLEDGE.md` -- consolidated knowledge (2026-08-06).\n- `16_future_work/ssl_from_video.md` -- self-supervised fine-tuning from robot video (design, not built).\n- `paper/results_rev5/` -- superseded generator revision; do not quote.\n")

    # ---------------- Part C
    h(2, "Part C -- Index")
    h(3, "C1. Embedded figures -> source files")
    table(["figure file (in `figures/`)", "source"], [[f"`{n}`", f"`{s}`"] for n, s in figlog])
    h(3, "C2. Every file under `paper/results_rev6/`")
    rows = []
    for p in sorted(R6.rglob("*")):
        if p.is_file() and OUT not in p.parents:
            rows.append([f"`{p.relative_to(R6)}`", f"{p.stat().st_size / 1024:.0f} KB"])
    table(["path", "size"], rows)

    (OUT / "CONV_CHART_REFERENCE.md").write_text("\n".join(doc) + "\n")
    n_fig = len(list(FIG.iterdir()))
    print(f"-> {OUT / 'CONV_CHART_REFERENCE.md'}: {sum(len(x.splitlines()) for x in doc)} lines, {n_fig} figures in figures/ ({sum(p.stat().st_size for p in FIG.iterdir()) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
