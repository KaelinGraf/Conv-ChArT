# Why we might want exclusive self-attention: the value-correlation measurement

**Status**: measured, not acted on. Motivates ablation **A14** in `ablations.md`.
**Date**: 2026-07-28, re-measured against the project baseline after rev-3 was retired.
**Weights**: `runs/rev640_baseline/ckpt_0150000.pt` @ **step 150,000** -- the baseline
checkpoint (Kaelin, 2026-07-28; see `memory.md`), 640 variant, `attend_div=8`, 2
bottleneck blocks, 8 heads, d=256. **The weights themselves predate rev-3 entirely**
(150k sits inside the rev-2 training phase; rev-3 ran 150k->190k and bought nothing
transferable, hence its retirement) -- what changed between this measurement and the
original one is the checkpoint step (177k -> 150k) AND the generator config the probe
frames are sampled from (`configs/rev640.yaml` is now rev-4: shared-gain auto-exposure,
`differencing_p` 0.5 -> 0.10). **Both moved at once, so this is not an apples-to-apples
re-run of a fixed measurement -- treat it as "does the finding hold on the baseline
under current conditions", not "did 27k more steps change anything".**
**Scripts** (kept beside this note so the numbers stay reproducible after the session
scratchpad is gone): `xsa_diagnostic.py`, `xsa_commonmode.py`.

`paper/` is gitignored — internal working note, not part of the public repo.

---

## 1. Why we looked

Zhai, *Exclusive Self Attention* (arXiv:2603.09078, 10 Mar 2026) observes an
**attention similarity bias**: in trained transformers the attention output `y_i`
has high cosine similarity with the token's own value vector `v_i`, so SA spends
capacity duplicating a point-wise transform the residual path and the MLP already
provide. XSA removes it, per their Eq. 2:

```
z_i = y_i − (y_iᵀ v_i) · v_i / ‖v_i‖²
```

Their Algorithm 1 is two lines, per-head, BEFORE the output projection `W_O`:

```python
Y  = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
Vn = F.normalize(V, dim=-1)
Z  = Y - (Y * Vn).sum(dim=-1, keepdim=True) * Vn
```

No parameters, negligible overhead. **Read the paper, not the abstract** — an
automated summarisation of the PDF got both the application point (claimed "after
`W_O`") and the per-head question wrong. The equations above are transcribed from
the paper's own text and pseudocode.

**Their evaluation is language modelling only**, causal, up to 2.7B parameters,
with gains that *grow* with model size (Table 2 average: +0.26 at 0.7B, +1.03 at
1.4B, +1.36 at 2.7B) and with sequence length. Ours is bidirectional 2-D vision,
4.7M parameters, 2 blocks, a fixed 4,800-token grid. Nothing in that paper says it
transfers. Hence: measure the premise before spending anything.

## 2. What we measured, and how

`cos(y_i, v_i)` **IS the magnitude XSA deletes**, so it is the whole question.
Reproducing the paper's Figure 1 diagnostics on our bottleneck:

- tap each `Block`'s input `x` with a `forward_pre_hook`;
- `q, k, v = blk.qkv_heads(blk.n1(x))` — the model's own exposed path, so RoPE is
  applied to q,k exactly as in `forward` (and NOT to v);
- `y = F.scaled_dot_product_attention(q, k, v)` — **no causal mask**, matching our
  bidirectional `forward`;
- (a) `cos(v_i, v_j)` over random distinct pairs; (b) diagonal attention `a_ii`
  from an explicit softmax, computed head-by-head to bound memory; (c) `cos(y_i, v_i)`.

Four validation frames spanning the scale range (indices 500 / 3600 / 7000 / 9600),
CPU-only so the live trainer was never touched. Frames now come from `SynthVal` under
`configs/rev640.yaml`, i.e. the rev-4 generator -- the same four indices as the
original measurement, but not the same underlying images (rev-4's autoexposure and
lower `differencing_p` change what index N renders).

## 3. Results

Both tables below: original (177k, rev-3-sampled) -> re-measured (150k,
rev-4-sampled). The finding is a **replication, not a surprise** -- every number
moves by a few hundredths at most and the ranking/structure is identical; see
section 4 for the one number worth flagging (the a_ii/uniform ratio narrows a bit).

**Diagnostic 1** — T = 4800 tokens (unchanged), so uniform attention would give
`a_ii` = 2.08e-4:

| block | cos(y_i, v_i) | a_ii | a_ii ÷ uniform | cos(v_i, v_j) |
|---|---|---|---|---|
| 0 | 0.2750 -> **0.2868** | 6.24e-4 -> 5.31e-4 | 3.00x -> 2.55x | 0.1056 -> 0.1079 |
| 1 | 0.5314 -> **0.5380** | 9.97e-4 -> 8.23e-4 | 4.79x -> 3.95x | 0.5434 -> **0.5675** |

**Diagnostic 2** — how much of the output is a constant vector, and what survives
centring (subtracting the token-mean from both `y` and `v`):

| block | ‖mean_i y_i‖ ÷ mean_i‖y_i‖ | same for v | cos(y,v) | cos after centring |
|---|---|---|---|---|
| 0 | 0.8228 -> **0.8321** | 0.3280 -> 0.3337 | 0.2750 -> 0.2868 | 0.0969 -> 0.0902 |
| 1 | 0.7160 -> **0.7144** | 0.7228 -> 0.7390 | 0.5314 -> 0.5380 | 0.1703 -> 0.1396 |

**Attention entropy** (referenced in section 4, not printed by either script above --
recomputed the same way, from the same per-head softmax `A` the `a_ii` loop already
builds, over the same 4 frames): uniform would be `ln(4800)` = 8.48 nats regardless of
checkpoint. Block 0: 7.24 -> **6.81** nats. Block 1: 5.63 -> **5.83** nats.

Reference: the paper's 24-layer 1.3B causal LM runs `cos(y_i,v_i)` ≈ 0.2 at its
first layers rising to ≈ 0.6 by layer 23.

## 4. What it means

**The premise transfers, and by more than expected, and it holds up on the
baseline.** Our 2-block bottleneck sits near the *deep* end of their curve (0.54 in
block 1) — the opposite of the prior expectation that a shallow model would sit at
the weak end of a depth-driven effect. That reversal is the reason A14 exists at
all, and it reproduces cleanly at 150k under rev-4 sampling (0.5314 -> 0.5380):
same magnitude, same block-to-block gap, no material change.

**The mechanism is different from theirs.** Diagonal attention is negligible in
absolute terms — even at ~2.5-4× its fair share it is only ~0.05-0.08% of the mass
over 4,800 tokens (a_ii = 5.31e-4 / 8.23e-4; narrower than the original 3.0-4.8x /
0.06-0.1% measurement, but nowhere near enough mass to explain a 29-54%
attention-output/value cosine on its own). The driver is
**value-vector correlation**: 0.108 → 0.567 between blocks (was 0.106 → 0.543).
For random 32-dim directions (d_head = 256/8 = 32) chance cosine has std
1/√32 ≈ 0.18, so 0.57 means the value vectors occupy a narrow cone rather than
spreading over the sphere.

**The decomposition that explains it.** Write `v_j = μ + r_j` (shared direction
plus token-specific part). Attention weights sum to 1, so

```
y_i = Σ_j a_ij v_j = μ + Σ_j a_ij r_j
```

and `v_i = μ + r_i`. The similarity is dominated by the shared `μ·μ` term — the two
agree because both contain the common mode, not because token *i* attended to
itself. Diagnostic 2 confirms it directly: **71–83% of the attention output is a
constant vector** (was 72–82%), and centring collapses the cosine from 0.287 → 0.090
and 0.538 → 0.140 (was 0.275 → 0.097 and 0.531 → 0.170) — roughly two-thirds to
three-quarters of the bias is common-mode, same conclusion as the original
measurement.

**Why the constant is not harmless.** LayerNorm normalises **per token across
channels**, not across tokens, so a direction shared by *all* tokens survives it and
propagates into the decoder. The consequence is a conditioning problem rather than
wasted FLOPs: the informative, spatially-varying part of the attention output is
only ~17–29% of its magnitude (was ~18–28%), and downstream weights must extract a
small difference from a large offset.

**Two different causes, same symptom.** Block 0's values are relatively diverse
(0.33 common-mode) but its attention is diffuse (entropy 6.81 nats vs 8.48 uniform,
was 7.24), so averaging 4,800 vectors lands near their mean — 83% common-mode
output (was 82%). Block 1 attends more selectively (5.83 nats, was 5.63) but its
values are already aligned (0.74, was 0.72), so the output is still 71% constant
(was 72%). Same two-cause structure at 150k as at 177k.

## 5. The risk that could invalidate this

A common-mode direction is the natural encoding for a **global property of the
frame** — exposure, noise level, or contrast polarity. The original version of this
section pointed at a specific number: ~10.8% of the rev-3 distribution was
contrast-inverted, so "this whole frame is inverted" looked like a concrete,
measured example of a shared direction XSA would strip. **That anchor is gone.**
rev-4's shared-gain autoexposure was built for exactly this reason and cut inversion
from ~19.4% of differencing frames to ~0.5% (compounded with `differencing_p`
falling 0.5 → 0.10, inversion is now on the order of 0.05% of the whole
distribution, not 10.8%). Citing contrast inversion as the risk's justification is
no longer honest and is dropped.

**The general argument survives; only its illustrating example changes.** The
photometric/differencing augmentation stack still varies several genuinely
frame-global scalars over a wide range on every sample that uses it: differencing
ambient pedestal (`differencing_ambient`, 0.05–0.95), illumination-lobe peak
(`differencing_illum_peak`, 0.15–0.9), vignette strength (0.05–0.25), NIR
ink-contrast scale (1.0–1.6), plus ordinary exposure/brightness and sensor-noise
level on every frame regardless of differencing. Any of these is a plausible thing
for a common-mode attention direction to encode — a single scalar (or a handful)
broadcast to every token is architecturally the cheapest way to carry "how bright /
how noisy / what pedestal is this frame" into the decoder. Contrast inversion was
never the whole risk, just the most dramatic member of that category and the one we
happened to have a clean measured prevalence for. **What changed is severity, not
kind**: the specific failure mode is now rare enough (~0.05% of frames) that it
alone would not justify caution, but the category it belongs to — global photometric
state riding a common-mode channel — is untouched by rev-4 and still the reason to
verify empirically (A14) rather than assume XSA is free.

So the two outcomes have concrete mechanisms rather than being a hedge: XSA helps if
the common mode is dead weight, and hurts if it is how global frame state reaches the
decoder. A negative result is publishable and should be reported as one.

Also unresolved: our 4,800 tokens cover a mostly-background frame, so high
`cos(v_i,v_j)` may reflect genuinely similar *content* rather than wasted capacity —
unlike an LM where every token is a different word.

## 6. Reproducing

```
CUDA_VISIBLE_DEVICES="" env PYTHONPATH= /home/kaelin/anaconda3/envs/MLWS/bin/python \
  paper/xsa_diagnostic.py  --ckpt runs/rev640_baseline/ckpt_0150000.pt
CUDA_VISIBLE_DEVICES="" env PYTHONPATH= /home/kaelin/anaconda3/envs/MLWS/bin/python \
  paper/xsa_commonmode.py --ckpt runs/rev640_baseline/ckpt_0150000.pt
```

`CUDA_VISIBLE_DEVICES=""` keeps it off the GPU so it can run beside other work;
`PYTHONPATH=` must be cleared or a ROS install on the default path shadows imports.
`--config` defaults to `configs/rev640.yaml` in both scripts (pass it explicitly if
that ever changes). `ckpt_0150000.pt` is a rolled, stable checkpoint, not the
trainer's rolling `ckpt_latest.pt` slot, so there is no copy-before-load race to
guard against here — note this file's own prior revision pointed at
`runs/rev640_baseline/ckpt_latest.pt`, which as of 2026-07-28 has been renamed to
`ckpt_latest_rev3_dont_use.pt` and no longer exists at that path; every regeneration
command in this note now targets the pinned baseline step instead of the rolling
slot.

**Caveats on these numbers**: four frames, one checkpoint, step 150,000 (the
project baseline, not a final/converged checkpoint). Indicative, not settled. The
block-to-block contrast is large and consistent across all four frames in both the
original 177k/rev-3-sampled measurement and this 150k/rev-4-sampled one, which is
what makes it worth acting on; the absolute values should still be re-measured on
final weights before anything is written into the paper.

## 7. What this motivates

A14 in `ablations.md`: paired from-scratch arms, Reference and XSA, identical seed
and data order, differing only in the two lines above behind an `xsa: true` config
flag. Compare **learning curves**, not endpoints — the paper's own Figure 3 shows
separation well before convergence, and a consistent gap across many validation
points is far more convincing than one endpoint difference that could sit inside the
measured ~0.05–0.1 pp M-04 noise floor.

Correctness check that costs nothing: with XSA enabled, `cos(y_i, v_i)` must collapse
to ≈ 0 by construction. If it does not, the implementation is wrong, not the idea.

**Do NOT compare against `ckpt_0025000.pt`.** That checkpoint is from the legacy
augmentation phase, so using it as the baseline would confound XSA with the
curriculum change. The baseline must be a matched from-scratch run on the same
generator revision as the XSA arm (rev-4, now that rev-3 is retired) — which the
ablation queue needs anyway, so XSA costs one extra arm, not two.
