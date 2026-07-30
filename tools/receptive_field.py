"""tools/receptive_field.py -- what does one bottleneck token actually see?

Two deliverables, both derived from the LIVE MODULE LIST rather than a hand-written
layer table, so they cannot drift from dcc/model.py:

  1. receptive_field.md (-> PDF via pandoc): the recursion, applied layer by layer
     from the input to the bottleneck, WITH and WITHOUT the dilated (2, 4) pair.
     That difference is the quantitative form of the A-NODILATE question.
  2. receptive_field_overlay.png: a real training frame with one pixel marked and
     its receptive field drawn around it -- theoretical extent, and the effective
     (Luo) extent, which is the one that matters.

THE RECURSION (Dumoulin & Visin 2016). Walking layers in forward order, with
r = receptive field in input px and j = jump (input px per output step):

    k_eff = d * (k - 1) + 1          dilation d inflates the kernel
    r     = r + (k_eff - 1) * j
    j     = j * s

Start r = 1, j = 1.

THEORETICAL vs EFFECTIVE. r is an upper bound: it is the set of input pixels that
CAN influence an output, not the set that meaningfully does. Luo et al. (2016) show
the influence is approximately Gaussian and that for a stack of n layers the
effective radius grows as O(sqrt(n)) rather than linearly, so the usable extent is a
shrinking fraction of r as depth grows. Both are reported; the effective figure is
the honest one to quote against a board size.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/rev640.yaml")
    p.add_argument("--index", type=int, default=4300, help="SynthVal frame for the overlay")
    p.add_argument("--out-dir", default="paper/results_rev6/10_receptive_field")
    p.add_argument("--dpi", type=int, default=170)
    p.add_argument("--nodilate-only", action="store_true",
                   help="overlay draws ONLY the undilated boxes. The with/without comparison "
                        "existed to justify dropping the dilated pair; once dropped (A-NODILATE "
                        "is the adopted baseline) showing it invites the question 'why are there "
                        "two?' about an architecture that no longer has the choice. The .md report "
                        "still carries both numbers -- the comparison is evidence, not a figure.")
    return p


def walk(model, attend_div):
    """Layers from input to the bottleneck, in forward order, as (name, k, s, d)."""
    import torch.nn as nn
    stages = ["e1", "e2", "e3", "e4"] + (["e5"] if attend_div == 16 else [])
    seq = []
    for i, sname in enumerate(stages):
        if i:                                    # forward() pools BEFORE every stage but e1
            seq.append((f"pool -> {sname}", 2, 2, 1))
        stage = getattr(model, sname)
        for mname, m in stage.named_modules():
            if isinstance(m, nn.Conv2d):
                k = m.kernel_size[0]
                seq.append((f"{sname}.{mname} k{k}" + (f" d{m.dilation[0]}" if m.dilation[0] > 1 else ""),
                            k, m.stride[0], m.dilation[0]))
    return seq


def rf(seq):
    """-> (rows, r, j). Standard recursion; see module docstring."""
    r, j, rows = 1, 1, []
    for name, k, s, d in seq:
        k_eff = d * (k - 1) + 1
        r = r + (k_eff - 1) * j
        j = j * s
        rows.append((name, k, s, d, k_eff, r, j))
    return rows, r, j


def main():
    args = build_parser().parse_args()
    import numpy as np, cv2, torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cv2.setNumThreads(1)
    from dcc.dataset import load_config, SynthVal
    from dcc.model import DetectorNet, detector_kwargs

    cfg = load_config(args.config)
    W, H = cfg["input_size"]
    div = cfg.get("attend_div", 16)
    kw = detector_kwargs(cfg)
    with_d = DetectorNet(H, W, **kw)
    kw_nd = dict(kw); kw_nd["e4_dilated"] = False
    without_d = DetectorNet(H, W, **kw_nd)

    rows_w, r_w, j_w = rf(walk(with_d, div))
    rows_n, r_n, j_n = rf(walk(without_d, div))
    n_layers_w = len(rows_w)
    # Luo: effective radius ~ theoretical / sqrt(n_layers); reported as a radius
    eff_w, eff_n = r_w / np.sqrt(n_layers_w), r_n / np.sqrt(len(rows_n))
    s_lo, s_hi = cfg["scale_range_px"]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    def table(rows):
        t = ["| layer | k | s | d | k_eff | r (px) | j |", "|---|---|---|---|---|---|---|"]
        t += [f"| `{n}` | {k} | {s} | {d} | {ke} | **{r}** | {jj} |" for n, k, s, d, ke, r, jj in rows]
        return "\n".join(t)

    md = f"""---
title: "Conv-ChArT --- receptive field of the bottleneck"
subtitle: "Computed from the live module list, not a hand-written layer table"
geometry: margin=2.2cm
fontsize: 10pt
---

# The receptive field, analytically

For a stack of $L$ layers, layer $l$ having kernel $k_l$, stride $s_l$ and dilation
$d_l$, the receptive field $r$ of one output unit measured in **input pixels**, and
the jump $j$ (input pixels per output step), are

$$\\boxed{{\\;r \\;=\\; 1 \\;+\\; \\sum_{{l=1}}^{{L}} \\Big( d_l\\,(k_l - 1) \\prod_{{i=1}}^{{l-1}} s_i \\Big), \\qquad j \\;=\\; \\prod_{{l=1}}^{{L}} s_l \\;}}$$

Read it as: each layer contributes its own **dilated** kernel extent $d_l(k_l-1)$,
magnified by the total downsampling **accumulated before it**, $\\prod_{{i<l}} s_i$.
A layer late in the stack is worth far more reach than an early one, which is why
dilating at the deepest stage is the cheap way to buy receptive field --- and why a
stride-2 pool doubles the value of every layer that follows it.

Equivalently, as the recursion actually evaluated below (Dumoulin \\& Visin 2016),
starting from $r = 1$, $j = 1$:

$$k_{{\\text{{eff}}}} = d\\,(k-1) + 1, \\qquad r \\leftarrow r + (k_{{\\text{{eff}}}}-1)\\,j,
\\qquad j \\leftarrow j \\cdot s$$

Dilation inflates the kernel **without adding parameters** --- that is the whole
appeal, and the reason the pair was cheap enough to leave in.

## Worked, for the deepest stage

Every conv here is $k=3$, $s=1$; the three pools are $k=2$, $s=2$. By the time the
signal reaches `e4` the accumulated stride is $\\prod_{{i<l}} s_i = 8$, so each `e4`
layer contributes $8\\,d_l(k_l-1) = 16\\,d_l$ input pixels:

$$\\underbrace{{16 \\times 1}}_{{\\text{{e4 conv 1}}}} + \\underbrace{{16 \\times 1}}_{{\\text{{e4 conv 2}}}}
+ \\underbrace{{16 \\times 2}}_{{\\text{{dilated }} d=2}} + \\underbrace{{16 \\times 4}}_{{\\text{{dilated }} d=4}}
\\;=\\; 32 + 96 \\;\\text{{px}}$$

The undilated pair of `e4` buys 32 px; **the dilated pair buys the other 96 px** ---
which is exactly the difference in the table below, and what 1,180,672 parameters
are being spent on.

# Result at the bottleneck ({div}x downsample, {H // div} x {W // div} tokens)

| | theoretical $r$ | jump $j$ | Luo effective radius |
|---|---|---|---|
| **with** dilated (2, 4) pair | **{r_w} px** | {j_w} | ~{eff_w:.0f} px |
| **without** the pair | **{r_n} px** | {j_n} | ~{eff_n:.0f} px |
| difference | {r_w - r_n} px | -- | ~{eff_w - eff_n:.0f} px |

**Theoretical $r$ is an upper bound**, not what the network uses. Luo et al. (2016)
show the influence of input pixels on an output is approximately Gaussian, and that
for a stack of $n$ layers the effective radius grows as $O(\\sqrt{{n}})$ rather than
linearly with depth --- so the usable extent is a shrinking fraction of $r$. With
$n = {n_layers_w}$ layers to the bottleneck, the effective radius is roughly
$r/\\sqrt{{n}} \\approx {eff_w:.0f}$ px.

# Why this is the design question

The identity read needs to see an **adjacent marker**: the marker centre nearest a
given inner corner sits about $0.85\\,s$ away, where $s$ is the board square size.
Over the trained envelope $s \\in [{s_lo}, {s_hi}]$ px that distance is
**{0.85 * s_lo:.0f} to {0.85 * s_hi:.0f} px**.

- At $s = {s_lo}$ px the neighbour is {0.85 * s_lo:.0f} px away --- comfortably inside
  even the effective radius. Convolutions alone suffice, which is exactly what the A1
  ablation measured: conv-only is *equal or better* at far range.
- At $s = {s_hi}$ px the neighbour is {0.85 * s_hi:.0f} px away --- outside the
  effective radius ({eff_w:.0f} px) and, without the pair, outside it by a wide margin.
  Local evidence cannot answer the question, and the bottleneck attention (global by
  construction) is the only mechanism that can. A1 measures the cost of removing it:
  **99.20% vs 91.82% ID accuracy at $s=128$**.

# The dilation question

The dilated pair buys **{r_w - r_n} px of theoretical reach for 1,180,672 parameters**
--- 25.1% of the detector, 0.75x the cost of the attention that superseded it. It is
inherited from the pre-transformer design (Rev B / DS-02), where the dilated cascade
*was* the board-scale context mechanism; the Rev G review marked it moot once the MHSA
bottleneck landed, and it was never removed.

Two readings, and the ablation `configs/abl_nodilate.yaml` decides between them:

1. **Redundant** --- attention already provides unbounded reach, so the extra
   {r_w - r_n} px is paid for and unused.
2. **Actively harmful** --- RoPE attention discriminates tokens by content *and*
   relative position; a wide dilated aggregation makes neighbouring tokens more alike
   and erodes the local distinctiveness the attention depends on.

Note that even *with* the pair the effective radius ({eff_w:.0f} px) does not reach an
adjacent marker at large $s$ --- so the dilation does not solve the close-range problem
either. That is the argument for its removal being free.

# Layer-by-layer, with the dilated pair

{table(rows_w)}

# Layer-by-layer, without it

{table(rows_n)}
"""
    (out / "receptive_field.md").write_text(md)
    print(f"with dilation:    r = {r_w} px, j = {j_w}, effective ~ {eff_w:.0f} px")
    print(f"without dilation: r = {r_n} px, j = {j_n}, effective ~ {eff_n:.0f} px")

    # ---- overlay ----------------------------------------------------------
    ds = SynthVal(cfg, cfg["synth"]["val_size"], cfg["synth"]["val_seed"])
    img, rec = ds[args.index]
    pts = np.array([[c["x"], c["y"]] for c in rec["corners"] if c["visible"]])
    cx, cy = pts[len(pts) // 2]                       # a real corner, not an arbitrary pixel

    fig, ax = plt.subplots(figsize=(11.5, 8.6))
    ax.imshow(img, cmap="gray")
    # Four boxes: theoretical AND effective, each with and without the dilated pair, so
    # the comparison is symmetric. The two effective boxes are the ones to read against
    # the marker distance -- theoretical r is an upper bound nothing actually uses.
    boxes = [(r_w / 2,   f"theoretical, WITH dilation      {r_w} px",       "#ffd000", "-",  2.2),
             (r_n / 2,   f"theoretical, without            {r_n} px",       "#ff8c1a", "--", 2.0),
             (eff_w / 2, f"EFFECTIVE (Luo), WITH dilation  ~{eff_w:.0f} px", "#00e63c", "-",  2.6),
             (eff_n / 2, f"EFFECTIVE (Luo), without        ~{eff_n:.0f} px", "#12b886", "--", 2.2)]
    if args.nodilate_only:
        # Solid lines, not the dashed "without" styling: dashed only ever meant "the other arm
        # of a comparison", and with the comparison gone a dashed box reads as uncertainty.
        boxes = [(r_n / 2,   f"theoretical      {r_n} px",        "#ff8c1a", "-", 2.2),
                 (eff_n / 2, f"EFFECTIVE (Luo)  ~{eff_n:.0f} px", "#12b886", "-", 2.6)]
    for r, lab, col, ls, lw in boxes:
        ax.add_patch(plt.Rectangle((cx - r, cy - r), 2 * r, 2 * r, fill=False, ec=col, lw=lw,
                                    ls=ls, label=lab))
    ax.plot(cx, cy, "+", color="cyan", ms=16, mew=2.5)
    ax.plot(pts[:, 0], pts[:, 1], ".", color="#66ccff", ms=4, alpha=0.8)
    sub = (f"val{args.index}, s = {rec['s_px']:.0f} px, input {W}x{H}, bottleneck H/{div}"
           if args.nodilate_only else
           f"val{args.index}, s = {rec['s_px']:.0f} px, input {W}x{H}, bottleneck H/{div}   |   "
           f"the dilated pair adds {r_w - r_n} px theoretical / ~{eff_w - eff_n:.0f} px effective, "
           f"for 1,180,672 parameters")
    ax.set_title("Receptive field of ONE bottleneck token, centred on a corner\n" + sub, fontsize=11)
    ax.legend(fontsize=9, loc="lower right"); ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / ("receptive_field_overlay_nodilate.png" if args.nodilate_only
                       else "receptive_field_overlay.png"), dpi=args.dpi)
    print(f"-> {out}/receptive_field.md, receptive_field_overlay.png")


if __name__ == "__main__":
    main()
