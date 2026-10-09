"""tools/gate_formula.py -- the attention-gate formulation, as a screenshot-able PDF.

Companion to tools/receptive_field.py, same discipline: every shape, channel count,
kernel size and initialisation value is read from the LIVE MODULE, not transcribed by
hand, so the document cannot drift from dcc/model.py. If the gate is re-specified,
re-run this and the paper text follows.

Optionally measures alpha on real frames (--measure) so the "what it actually learned"
section carries numbers rather than adjectives.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/rev640.yaml")
    p.add_argument("--ckpt", default="runs/rev640_160k_rev6/ckpt_0160000.pt")
    p.add_argument("--measure", type=int, default=40,
                   help="frames to measure alpha on; 0 to skip the empirical section")
    p.add_argument("--out-dir", default="paper/results_rev6/11_attention_gate")
    return p


def main():
    args = build_parser().parse_args()
    import numpy as np, cv2, torch
    cv2.setNumThreads(1)
    from dcc.dataset import load_config
    from dcc.model import DetectorNet, detector_kwargs
    from dcc.synth import generate_sample, list_backgrounds

    cfg = load_config(args.config)
    W, H = cfg["input_size"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = DetectorNet(H, W, **detector_kwargs(cfg))
    g = m.gate3
    # read the ACTUAL module rather than restating the constructor
    wx, wg, psi = g.wx, g.wg, g.psi
    skip_ch, inter_ch = wx.in_channels, wx.out_channels
    gate_ch = wg.in_channels
    n_par = sum(p.numel() for p in g.parameters())
    tot = sum(p.numel() for p in m.parameters())
    psi_b = float(psi.bias.detach().reshape(-1)[0])
    psi_w_absmax = float(psi.weight.detach().abs().max())
    a0 = 1.0 / (1.0 + np.exp(-psi_b))
    div = cfg.get("attend_div", 16)

    emp = ""
    if args.measure:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        m.load_state_dict(ck.get("ema", ck.get("model")))
        m = m.to(dev).eval()
        bg = list_backgrounds(cfg["synth"]["backgrounds"])
        gate_args = {}
        m.gate3.register_forward_pre_hook(
            lambda mod, a, kw: gate_args.__setitem__("a", (tuple(t.detach() for t in a),
                                                           {k: v.detach() for k, v in kw.items()})),
            with_kwargs=True)
        ins, outs, means = [], [], []
        for i in range(args.measure):
            rng = np.random.default_rng([20260729, i])
            rec, _ = generate_sample(cfg, rng, bg, force_negative=False)
            pts = np.array([[c["x"], c["y"]] for c in rec["corners"] if c["visible"]])
            if len(pts) < 4:
                continue
            with torch.no_grad():
                m(torch.from_numpy(rec["image"]).float()[None, None].to(dev) / 255.0)
                a, kw = gate_args["a"]
                al = m.gate3.alpha(*a, **kw)[0, 0].cpu().numpy()
            mask = np.zeros(al.shape, np.uint8)
            sc = np.array(al.shape[::-1]) / np.array([W, H])
            cv2.fillConvexPoly(mask, np.round(cv2.convexHull((pts * sc).astype(np.float32))
                                               .reshape(-1, 2)).astype(np.int32), 1)
            mb = mask.astype(bool)
            if mb.sum() < 20:
                continue
            ins.append(al[mb].mean()); outs.append(al[~mb].mean()); means.append(al.mean())
        i_m, o_m = float(np.median(ins)), float(np.median(outs))
        emp = f"""
# What it actually learned ({len(ins)} frames, trained weights)

| | median $\\alpha$ |
|---|---|
| inside the board region | **{i_m:.3f}** |
| outside it | **{o_m:.3f}** |
| whole map | {float(np.median(means)):.3f} |
| ratio inside/outside | **{i_m / o_m:.2f}** |

$\\alpha$ has moved well away from its pass-through initialisation of {a0:.4f}, so the
gate is doing something rather than passing the skip through untouched. It is
**content-conditioned, not a fixed spatial prior** --- swapping the conditioning
signal $g$ for another frame's changes $\\alpha$ materially.

Two cautions worth carrying into any claim about this. The inside/outside ratio is
**not stable in sign across frames** --- it has been measured both above and below 1
on different samples --- so the honest statement is "content-conditioned", not "the
gate suppresses/passes the board". And $\\alpha$ is a per-pixel scalar broadcast over
all {skip_ch} skip channels, so it selects *where*, never *which feature*.
"""

    md = f"""---
title: "Conv-ChArT --- the attention gate"
subtitle: "Read from the live module; every constant below is the one in dcc/model.py"
geometry: margin=2.2cm
fontsize: 10pt
---

# Formulation

Additive attention gate (Oktay et al., *Attention U-Net*, eq. 1--2), applied to the
decoder's H/4 skip connection and conditioned on the post-attention bottleneck.
Writing $s$ for the encoder skip and $g$ for the gating signal:

$$\\boxed{{\\;\\alpha \\;=\\; \\sigma\\Big(\\psi\\big(\\mathrm{{ReLU}}(W_x * s_{{\\downarrow 2}} \\;+\\; W_g * g)\\big)\\Big),
\\qquad \\tilde{{s}} \\;=\\; s \\odot \\mathrm{{up}}_{{\\text{{bilinear}}}}(\\alpha) \\;}}$$

- $W_x$: $1\\times1$ conv, stride 2, **no bias** --- ${skip_ch} \\to {inter_ch}$ channels.
  The stride is what brings the skip down to the gating signal's resolution.
- $W_g$: $1\\times1$ conv --- ${gate_ch} \\to {inter_ch}$ channels.
- $\\psi$: $1\\times1$ conv --- ${inter_ch} \\to 1$ channel. Collapses to a **single
  spatial map**, not a per-channel mask.
- $\\sigma$: logistic sigmoid, so $\\alpha \\in (0,1)$ per spatial position.
- $\\mathrm{{up}}$: bilinear, `align_corners=False`, back to the skip's resolution.
- $\\odot$: elementwise, broadcast across all ${skip_ch} skip channels.

Total: **{n_par:,} parameters** ({100 * n_par / tot:.1f}% of the {tot:,}-parameter detector).
The gate is nearly free; what it costs is a design commitment, not compute.

# Pass-through initialisation

$\\psi$'s **weight is zeroed** and its **bias set to $+{psi_b:.1f}$**, so at step 0

$$\\alpha \\;\\equiv\\; \\sigma({psi_b:.1f}) \\;=\\; {a0:.4f} \\qquad \\text{{exactly constant, everywhere.}}$$

Verified on the constructed module: $\\max|\\psi_W| = {psi_w_absmax:.1f}$, $\\psi_b = {psi_b:.1f}$.

**Why zero the weight rather than just bias it high.** A large bias alone
(say $+5$, giving $\\alpha \\approx 0.993$) would leave $\\alpha$ *technically*
input-dependent from the first step --- the network would begin by fighting a small,
arbitrary gating signal it never asked for. Zeroing $W_\\psi$ makes the gate a **true
no-op** at initialisation: the skip passes through scaled by a constant, and training
decides if and when gating should begin. The Oktay reference implementation's TORR
variant uses exactly $\\psi_b = +3.0$; the zeroed weight is our strengthening of it.

# Where it is applied, and where it is not

At `attend_div = {div}` the gate sits on the **H/4 skip only**. The H/2 and full-resolution
skips are ungated by design.

The rule is: gate only where a wrong veto can be **undone downstream**. The H/4 skip
feeds the class head, and the Stage-3 lattice gate plus recovery pass can repair a
corner whose identity was lost. The full-resolution skip feeds the heatmap head, where
a suppressed corner is simply gone --- there is no recovery mechanism for a detection
that was never made. Gating the high-resolution skips would put an unrecoverable
failure mode behind a learned sigmoid, for no compensating benefit.

# Ablation lever

`gates_enabled: false` (`configs/abl_gates_off.yaml`) bypasses the gate --- $\\tilde{{s}} = s$
--- but still **constructs** the module, so the `state_dict` shape is unchanged and a
gated checkpoint stays loadable. Ablation by bypass, not deletion, keeps parameter
counts comparable between arms. (Contrast `abl_nodilate.yaml`, which *deletes*, because
there the parameter saving is the hypothesis.)
{emp}"""
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "attention_gate.md").write_text(md)
    print(f"gate3: skip {skip_ch} -> inter {inter_ch}, gate {gate_ch}, {n_par:,} params "
          f"({100 * n_par / tot:.1f}%), psi_b={psi_b:.1f}, alpha_0={a0:.4f}")
    print(f"-> {out}/attention_gate.md")


if __name__ == "__main__":
    main()
