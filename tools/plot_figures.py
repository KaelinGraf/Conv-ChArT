"""tools/plot_figures.py -- paper figure pipeline: consumes the results.json
files tools/eval_classical.py and tools/factor_sweep.py already write and
emits the paper's figures directly, so Friday morning is one command once
final-weight results land, not a debugging session.

Missing-data-tolerant by design (team-lead 2026-07-28): each learned-arm
grid (--ours-grid, --dc-grid) is optional and simply omitted from every
figure/table when its file doesn't exist yet -- rerun this script once a
file appears and that system's series/column populates automatically, no
code change needed. The classical grid and the factor-sweep results are
already fully populated today, so this can be built and tested now.

Conventions follow this repo's existing plotting code (tools/audit.py,
tools/view.py, tools/introspect.py): plain matplotlib, Agg backend for
headless save, argparse defined before any heavy import so --help stays
cheap, one fig/ax per figure, fig.savefig + plt.close.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TIER_ORDER = ("clean", "mid", "deployment-dark", "overcast")
SYSTEM_LABELS = {"classical": "classical", "ours": "ours", "deepcharuco_zeroshot": "Deep ChArUco (zero-shot)"}
SYSTEM_COLORS = {"classical": "tab:orange", "ours": "tab:blue", "deepcharuco_zeroshot": "tab:green"}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--classical-grid", default="eval_classical/results_full_grid.json")
    p.add_argument("--ours-grid", default=None, help="same {'grid': [...]} schema as eval_classical's own output")
    p.add_argument("--dc-grid", default=None, help="same schema, for the (retrained, not zero-shot) Deep ChArUco arm")
    p.add_argument("--factor-sweep", default="factor_sweep_out/results.json")
    p.add_argument("--metrics", default=None,
                    help="a trainer's metrics.jsonl (e.g. runs/rev640_baseline/metrics.jsonl) -- "
                         "optional, plots the training/val curves; see load_metrics_series for why "
                         "this needs dedupe (a crashed-and-relaunched run's file can contain a "
                         "replayed segment)")
    p.add_argument("--out", default="paper/figures")
    return p


# --------------------------------------------------------------- data loading

def load_grid(path):
    """{tier: {s: cell_dict}} or None if `path` is falsy/missing -- the
    missing-data tolerance every figure function below relies on."""
    if not path or not Path(path).exists():
        return None
    with open(path) as f:
        data = json.load(f)
    by_tier = {}
    for cell in data["grid"]:
        by_tier.setdefault(cell["tier"], {})[cell["s"]] = cell
    return by_tier


def load_factor_sweep(path):
    if not path or not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)


def _get_path(d, dotted_key):
    """d["a"]["b"] for dotted_key "a.b"; None if any segment is missing or
    d stops being a dict along the way."""
    for part in dotted_key.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


def load_metrics_series(path, has_key):
    """[(step, value), ...] sorted by step, extracted from a trainer's
    metrics.jsonl for the dotted key path `has_key` (e.g. "loss" for the
    per-step training record, "val.m04.accuracy" for a periodic validation
    record) -- ANY curve/metrics consumer must go through this rather than
    reading metrics.jsonl directly (team-lead 2026-07-28, data-integrity
    finding): a trainer that crashes mid-run and resumes from its last
    checkpoint re-executes and re-logs every step from that checkpoint
    onward into the SAME file, so a frozen-then-relaunched run's
    metrics.jsonl contains a REPLAYED segment -- the same step numbers
    appearing twice, with a backwards jump in file order at the resume
    point. Plotted in file order without deduping, this produces a sawtooth
    that looks exactly like training instability but is actually two runs
    concatenated. Deduping is done PER KEY PATH (not globally over whole
    records) because a single step can legitimately carry multiple record
    KINDS in one pass -- e.g. a per-step train_fields record and a periodic
    val record both logged at the same step -- so a blanket
    dedupe-the-whole-line-by-step would silently drop one kind in favour of
    the other; scoping to one key path at a time keeps every legitimate
    record kind while still fixing the replay.

    Last occurrence (in file order = write order = wall-clock order) wins
    for a duplicated step: the replay is deterministic (same checkpoint,
    same resume_count, same stream_seed) and executes strictly AFTER the
    dead run's own write of that step, so it is the authoritative record.
    Never silent: logs (prints) how many records were deduped and, if a
    backwards step jump was found, the step it occurred at."""
    with open(path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    by_step = {}
    n_seen = 0
    prev_step = None
    backtrack_at = None
    for rec in records:
        step = rec.get("step")
        value = _get_path(rec, has_key)
        if step is None or value is None:
            continue
        n_seen += 1
        if prev_step is not None and step < prev_step and backtrack_at is None:
            backtrack_at = step
        prev_step = step
        by_step[step] = value  # last occurrence in file order overwrites -- the dedupe rule

    n_unique = len(by_step)
    n_dropped = n_seen - n_unique
    if n_dropped:
        msg = f"[metrics] {path}: deduped {n_dropped} record(s) across {n_unique} step(s) for key {has_key!r}"
        if backtrack_at is not None:
            msg += f" -- backtrack detected at step {backtrack_at} (replayed segment)"
        print(msg)

    return [(s, by_step[s]) for s in sorted(by_step)]


# ------------------------------------------------------------- grid figures

def plot_envelope_heatmap(grids, out_dir):
    """One heatmap PNG per available system: s (log-x) x tier, coloured by
    id_rate. `grids` is {system_name: load_grid(...) result or None}."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    for system, by_tier in grids.items():
        if by_tier is None:
            print(f"  [envelope] skipping {system}: no grid file")
            continue
        tiers = [t for t in TIER_ORDER if t in by_tier]
        s_values = sorted({s for cells in by_tier.values() for s in cells})
        mat = np.full((len(tiers), len(s_values)), np.nan)
        for i, tier in enumerate(tiers):
            for j, s in enumerate(s_values):
                cell = by_tier[tier].get(s)
                if cell is not None:
                    mat[i, j] = cell["id_rate"]

        fig, ax = plt.subplots(figsize=(1.1 * len(s_values) + 2, 0.6 * len(tiers) + 1.5))
        im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(s_values))); ax.set_xticklabels(s_values, rotation=45)
        ax.set_yticks(range(len(tiers))); ax.set_yticklabels(tiers)
        ax.set_xlabel("s (px)"); ax.set_title(f"{SYSTEM_LABELS.get(system, system)}: ID rate by s x SNR tier")
        for i in range(len(tiers)):
            for j in range(len(s_values)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                            color="white" if mat[i, j] < 0.6 else "black", fontsize=7)
        fig.colorbar(im, ax=ax, label="ID rate")
        fig.tight_layout()
        out_path = out_dir / f"envelope_heatmap_{system}.png"
        fig.savefig(out_path, dpi=150); plt.close(fig)
        written.append(out_path)
    return written


def plot_envelope_curves(grids, out_dir):
    """One curve-family PNG per available system (id_rate vs s, log-x, one
    line per tier) -- the companion view to the heatmap, better for reading
    off the exact collapse point per tier."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    for system, by_tier in grids.items():
        if by_tier is None:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for tier in TIER_ORDER:
            if tier not in by_tier:
                continue
            cells = by_tier[tier]
            s_values = sorted(cells)
            ax.plot(s_values, [cells[s]["id_rate"] for s in s_values], marker="o", label=tier)
        ax.set_xscale("log"); ax.set_xlabel("s (px)"); ax.set_ylabel("ID rate")
        ax.set_title(f"{SYSTEM_LABELS.get(system, system)}: ID rate vs s by SNR tier")
        ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=8)
        fig.tight_layout()
        out_path = out_dir / f"envelope_curves_{system}.png"
        fig.savefig(out_path, dpi=150); plt.close(fig)
        written.append(out_path)
    return written


def plot_envelope_overlay(grids, out_dir, tier="clean"):
    """One figure overlaying every AVAILABLE system's curve at a single
    tier -- only meaningful once 2+ systems have grid data; silently
    skipped (not an error) while only one does."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    present = {s: g for s, g in grids.items() if g is not None and tier in g}
    if len(present) < 2:
        print(f"  [envelope overlay] skipping tier={tier}: only {list(present)} available, need 2+")
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    for system, by_tier in present.items():
        cells = by_tier[tier]
        s_values = sorted(cells)
        ax.plot(s_values, [cells[s]["id_rate"] for s in s_values], marker="o",
                label=SYSTEM_LABELS.get(system, system), color=SYSTEM_COLORS.get(system))
    ax.set_xscale("log"); ax.set_xlabel("s (px)"); ax.set_ylabel("ID rate")
    ax.set_title(f"System comparison, {tier} tier"); ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = out_dir / f"envelope_overlay_{tier}.png"
    fig.savefig(out_path, dpi=150); plt.close(fig)
    return out_path


# --------------------------------------------------------- factor-sweep figures

def plot_factor_sweep_curves(fs_results, out_dir):
    """One PNG per axis in the factor-sweep results: detected/16 (solid) and
    id_correct/16 (dashed) vs step, one colour per system -- all three
    systems are already populated in factor_sweep_out/results.json today, so
    this needs no missing-data handling the way the grid figures do."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if fs_results is None:
        print("  [factor-sweep] skipping: no results.json found")
        return []
    written = []
    for axis_name, axis_data in fs_results.items():
        steps = axis_data["steps"]
        scores = axis_data["scores"]
        x = range(len(steps))
        fig, ax = plt.subplots(figsize=(1.1 * len(steps) + 2, 4))
        for system, per_step in scores.items():
            n_gt = [r["n_gt"] for r in per_step]
            detected_frac = [r["detected"] / g if g else 0.0 for r, g in zip(per_step, n_gt)]
            id_frac = [r["id_correct"] / g if g else 0.0 for r, g in zip(per_step, n_gt)]
            color = SYSTEM_COLORS.get(system)
            ax.plot(x, detected_frac, marker="o", color=color, label=f"{SYSTEM_LABELS.get(system, system)} detected")
            ax.plot(x, id_frac, marker="s", linestyle="--", color=color, label=f"{SYSTEM_LABELS.get(system, system)} ID-correct")
        ax.set_xticks(list(x)); ax.set_xticklabels(steps, rotation=45, ha="right")
        ax.set_ylim(-0.02, 1.02); ax.set_ylabel("fraction of 16 corners")
        ax.set_title(f"factor sweep: {axis_name}")
        ax.legend(fontsize=7, ncol=2)

        covariates = axis_data.get("covariates")
        if covariates and any(c.get("saturation_frac") is not None for c in covariates):
            ax2 = ax.twinx()
            sat = [c.get("saturation_frac") for c in covariates]
            sat_x = [i for i, v in enumerate(sat) if v is not None]
            sat_y = [v for v in sat if v is not None]
            ax2.plot(sat_x, sat_y, color="red", linestyle=":", marker="x", label="saturation_frac")
            ax2.set_ylabel("board saturation fraction", color="red")
            ax2.set_ylim(-0.02, 1.02)
            ax2.tick_params(axis="y", colors="red")

        fig.tight_layout()
        out_path = out_dir / f"factor_{axis_name}.png"
        fig.savefig(out_path, dpi=150); plt.close(fig)
        written.append(out_path)
    return written


# ---------------------------------------------------------- training curves

def plot_training_curves(metrics_path, out_dir):
    """(loss vs step, val M-04 accuracy vs step) twin-panel PNG from a
    trainer's metrics.jsonl -- built entirely on load_metrics_series, so a
    replayed segment (see that function's docstring) never reaches the
    figure as a sawtooth. Missing-data-tolerant like every other figure
    function here: returns None and prints why if `metrics_path` is falsy,
    missing, or has neither series."""
    if not metrics_path or not Path(metrics_path).exists():
        print(f"  [training curves] skipping: no metrics file at {metrics_path!r}")
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loss = load_metrics_series(metrics_path, "loss")
    val_m04 = load_metrics_series(metrics_path, "val.m04.accuracy")
    if not loss and not val_m04:
        print(f"  [training curves] skipping: {metrics_path} has neither a 'loss' nor a "
              f"'val.m04.accuracy' series")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if loss:
        steps, vals = zip(*loss)
        axes[0].plot(steps, vals, linewidth=0.5)
        axes[0].set_xlabel("step"); axes[0].set_ylabel("loss"); axes[0].set_title("training loss")
    else:
        axes[0].set_visible(False)
    if val_m04:
        steps, vals = zip(*val_m04)
        axes[1].plot(steps, vals, marker="o")
        axes[1].set_xlabel("step"); axes[1].set_ylabel("M-04 accuracy"); axes[1].set_ylim(0, 1.02)
        axes[1].set_title("val ID accuracy (M-04)")
    else:
        axes[1].set_visible(False)
    fig.tight_layout()
    out_path = Path(out_dir) / "training_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# -------------------------------------------------------------------- table

def write_three_way_table(grids, fs_results, out_path):
    """Markdown three-way comparison table -- grid mean id_rate per tier per
    AVAILABLE system, plus a factor-sweep summary (mean detected/id_correct
    fraction per axis per system). Missing systems just don't get a column."""
    lines = ["# Three-way comparison\n"]

    present_systems = [s for s, g in grids.items() if g is not None]
    if present_systems:
        lines.append("## s x SNR-tier grid: mean ID rate per tier\n")
        header = "| tier | " + " | ".join(SYSTEM_LABELS.get(s, s) for s in present_systems) + " |"
        sep = "|---|" + "---|" * len(present_systems)
        lines += [header, sep]
        for tier in TIER_ORDER:
            row = [tier]
            for system in present_systems:
                by_tier = grids[system]
                if tier in by_tier:
                    rates = [c["id_rate"] for c in by_tier[tier].values()]
                    row.append(f"{sum(rates) / len(rates):.3f}")
                else:
                    row.append("-")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    else:
        lines.append("_(no grid files available yet)_\n")

    if fs_results:
        lines.append("## Factor sweep: mean detected / ID-correct fraction per axis\n")
        all_systems = sorted({s for axis in fs_results.values() for s in axis["scores"]})
        header = "| axis | " + " | ".join(f"{SYSTEM_LABELS.get(s, s)} det/id" for s in all_systems) + " |"
        sep = "|---|" + "---|" * len(all_systems)
        lines += [header, sep]
        for axis_name, axis_data in fs_results.items():
            row = [axis_name]
            for system in all_systems:
                per_step = axis_data["scores"].get(system)
                if not per_step:
                    row.append("-")
                    continue
                det = sum(r["detected"] / r["n_gt"] for r in per_step if r["n_gt"]) / len(per_step)
                idc = sum(r["id_correct"] / r["n_gt"] for r in per_step if r["n_gt"]) / len(per_step)
                row.append(f"{det:.2f} / {idc:.2f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    else:
        lines.append("_(no factor-sweep results available yet)_\n")

    out_path.write_text("\n".join(lines))
    return out_path


# --------------------------------------------------------------------- main

def main():
    args = build_parser().parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    grids = {"classical": load_grid(args.classical_grid),
             "ours": load_grid(args.ours_grid),
             "deepcharuco_zeroshot": load_grid(args.dc_grid)}
    fs_results = load_factor_sweep(args.factor_sweep)

    print("=== envelope heatmaps ===")
    for p in plot_envelope_heatmap(grids, out_dir):
        print(f"  wrote {p}")
    print("=== envelope curves ===")
    for p in plot_envelope_curves(grids, out_dir):
        print(f"  wrote {p}")
    print("=== envelope overlay (clean tier) ===")
    p = plot_envelope_overlay(grids, out_dir, tier="clean")
    if p:
        print(f"  wrote {p}")
    print("=== factor-sweep curves ===")
    for p in plot_factor_sweep_curves(fs_results, out_dir):
        print(f"  wrote {p}")
    print("=== three-way table ===")
    p = write_three_way_table(grids, fs_results, out_dir / "three_way_table.md")
    print(f"  wrote {p}")
    print("=== training curves ===")
    p = plot_training_curves(args.metrics, out_dir)
    if p:
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
