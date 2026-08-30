# Conv-ChArT — consolidated project knowledge

A single document covering what this project built, what it measured, what it got wrong, and
what a future engineer should not have to rediscover. Written 2026-08-06 from the primary
records (`paper/results_rev6/PLAN_autonomous_campaign.md`, `23_code_audit_FINDINGS.md`, the
result JSONs) rather than from recollection; every number here is quotable against a file.

---

## 1. What the system is

A three-stage learned pipeline for ChArUco board corner detection and pose estimation, targeting
a barrier-docking robot with a Jetson AGX Orin, an OV2311 global-shutter monochrome sensor, and
940 nm active illumination. Input frames are **lit-minus-unlit differences**: sunlight cancels by
subtraction, so blur, glare, ghosting and heavy sensor noise are first-class conditions rather
than corner cases.

**Stage 1 — detector.** A convolutional U-Net encoder/decoder with a multi-head self-attention
bottleneck. Two heads: a full-resolution corner heatmap and a 16-channel class map at H/4 giving
per-corner identity.

**Stage 2 — refiner.** A separate 97,056-parameter network on 24x24 crops around each coarse
peak, producing a 64x64 logit map read out by a 5x5 soft-argmax centroid. Sub-pixel only.

**Stage 3 — geometry.** Bilinear class-map readout, RANSAC lattice-homography gate, an ID
recovery pass, then IPPE PnP. The gate is what lets the system *refuse* rather than emit a wrong
pose, which turns out to be the single most important property in the baseline comparison.

### The reference architecture is the 882k release model, NOT the pre-ablation 4.7M

The 4,698,034-parameter network the project started from — dilated `e4` cascade, wider
everything — is **superseded and must not be presented as the architecture**. The ablation
campaign showed its extra capacity and its dilation cascade to be null within noise. The
reference configuration is the **882,402-parameter C2 release model**:

| setting | value |
|---|---|
| input | 640 x 480 |
| `width_mult` | 0.5 |
| `e4_dilated` | **false** (dilation cascade removed; −25% params, null effect) |
| attention | 2 pre-norm blocks, 8 heads, at H/8 (`attend_div` 8) -> 80x60 = 4800 tokens |
| positional encoding | axial RoPE, wavelength-anchored, `lambda_min` 2.5 cells |
| gates | single additive attention gate on the H/4 skip |
| `sigma_hm` / `sigma_cls` | 0.5 / 1.0 |
| loss | **C2**: heatmap one-hot BCE, class focal, `lambda_cls` 2.0. NOTE the class head is the
ONLY place focal survives, and only at this width: 222k and 502k are BCE on BOTH heads
(`lambda_cls` 2.0 and 1.5 respectively). All three release models are BCE on the heatmap. |
| schedule | 100,000 steps, batch 16 x accum 2, AdamW lr 3e-4 -> 3e-6 cosine, 1k warmup, wd 1e-4, EMA 0.999, clip 5.0 |

Parameter distribution (882,402 total):

| module | params | share |
|---|---|---|
| transformer blocks | 396,544 | **44.9%** |
| `e4` encoder stage | 221,696 | 25.1% |
| `d3` decoder | 110,720 | 12.5% |
| `e3` | 55,552 | 6.3% |
| class head | 37,968 | 4.3% |
| `d2` | 27,712 | 3.1% |
| `e2` | 13,952 | 1.6% |
| `d1` | 6,944 | 0.8% |
| gate (H/4) | 6,209 | 0.7% |
| `e1` | 2,512 | 0.3% |
| heatmap head | 2,337 | 0.3% |
| final norm | 256 | 0.0% |
| refiner (separate net) | 97,056 | — |

**The attention bottleneck is 45% of the parameters** and the encoder's deepest stage another
25%; the two heads together are under 5%. The 502k and 222k tiers are the same TOPOLOGY at
`width_mult` 0.375 and 0.25, but not the same SUPERVISION -- see the loss row above.

### Metric definitions (used throughout)

| metric | definition |
|---|---|
| M-01 | localisation error of matched corners; greedy NN match within 8 px. Median and p95. |
| tail >4 px | fraction of matched corners outside the refiner's capture range. **The operational gate.** |
| M-02 | match ratio by scale octave |
| M-04 | corner-ID accuracy, **conditioned on detection** |
| M-03 | refiner error |
| M-05 / M-06 | pose rotation (deg, geodesic) / translation (board squares) |

**M-04 is conditioned on detection.** A smaller model that detects fewer, easier corners is
scored on an easier set, so part of any narrow identity gap is selection rather than parity.
Never quote M-04 without recall alongside it.

---

## 2. THE CENTRAL METHODOLOGICAL RESULT: measure your noise floor first

This is the most transferable thing the project produced, and it invalidated five previously
reported findings.

**Procedure.** Train N seeds of an *identical* config and take the range of the final full-val
metric. That range is the smallest effect the rung can resolve. Anything below it is not a
result.

| rung | seeds | M-04 range | p95 range | tail range |
|---|---|---|---|---|
| 222,138 | 4 | **6.18 pp** | 0.0231 px | 0.0100 pp |
| 502,322 | 4 | 0.09 pp | 0.0061 px | 0.0139 pp |
| 882,402 | 3 | **0.07 pp** | **0.0020 px** | 0.0049 pp |
| refiner (97,056) | 3 | median **0.0017 px** | p95 **0.0940 px** | — |

**Three findings follow, all non-obvious:**

1. **Noise floors are per-width AND per-metric.** The 882k p95 floor is **11.6x tighter** than
   the 222k one. Reproducibility improves monotonically with capacity.
2. **Within one model, metrics differ by ~55x in reproducibility.** The refiner resolves median
   to 0.0017 px but p95 only to 0.0940 px. The detector shows the same split (502k: identity to
   0.09 pp, p95 only to 0.0061 px). **Median is the reliable axis; p95 is not.** Report both, but
   never call a p95 result from a single seed.
3. **The 222k rung was retired as an ablation platform.** A 6.18 pp M-04 range makes every
   single-seed 222k comparison uninterpretable. Five previously reported findings did not survive.

**Corollary that cost the most time:** mid-training deltas are convergence *timing*, not quality.
The 502k 4-seed M-04 range was **35.39 pp at step 7,500** and **0.09 pp at step 35,000**. Any
ablation called from a mid-training validation is measuring which seed ignited first.

---

## 3. The ablation programme — what is real and what is not

All comparisons at a matched 35k-step budget, one shared reference run, one key changed per arm,
scored against the measured floor for that width.

### Real levers (all in the loss / target representation)

| lever | true effect | where measured |
|---|---|---|
| `loss_form=ce` (one-hot BCE) vs focal | p95 −0.038, M-04 **+0.43 pp**, tail **7.4x smaller** | 882k, far outside any range |
| `sigma_hm` 0.5 vs 1.0 | p95 **+0.0429** (~7x the p95 range) for **zero** identity change | 882k, replicates at full width |
| `lambda_cls` 2.0 | **+0.29 pp** M-04 (3.3x range), −0.007 px p95 | 502k |

Full-width `sigma_hm` ladder, monotone to the degenerate limit with no identity cost anywhere:
2.0 -> 0.8149, 1.0 -> 0.7430, 0.5 -> 0.7146, 0.25 -> 0.7016, one-hot BCE -> 0.6690.

### Null levers (every structural knob tested)

`attn_heads` 8->4 · `attend_div` 8->16 (**+607k params**) · `gate_skips` topology · `gates_enabled`
· `e4_dilated` (**−25% params**) · `xsa` · `sigma_cls` (both directions) · `sigma_ref` · width
beyond 502k.

> **THE CAMPAIGN'S CENTRAL FINDING, and the opposite of where the search started:** the
> architecture is at its optimum for this problem. Every real lever is in the loss and the target
> representation; none is in the network. A 607k-parameter increase in attention resolution
> bought nothing measurable.

### The sigma_hm / sigma_cls asymmetry — a mechanism worth knowing

`sigma_hm` is one of the two largest levers; `sigma_cls` is null in both directions. The class map
lives at H/4, so one cell spans 4 input px and sub-pixel structure in the class target carries no
information — the same reason ID readout is decoupled from refinement. The heatmap is at full
resolution, so its target width directly sets how precisely the peak can be located.

The refiner analogue (`sigma_ref`) is **null**, which is a real asymmetry. Hypothesis (recorded as
such, *not* established): the detector reads a hard argmax so target sharpness sets precision
directly, while the refiner reads a 5x5 soft-argmax centroid that integrates over the target width.

### Composites do not stack cleanly

Same-axis levers stop adding **identity** while still delivering **localisation**. C1
(gates [3,2] + lambda 2.0) was *worse than lambda 2.0 alone on all three metrics*. Do not assume
independent gains compose.

### `lambda_cls` is a trade, not a free lever, above 502k

At 882k the ladder is monotone and linear — 1.5 -> +0.09 pp/+0.0047, 2.0 -> +0.15/+0.0085,
4.0 -> +0.26/+0.0169 — an exchange rate of ~0.06 px p95 per pp of identity. The free-at-502k
result does not transfer.

---

## 4. Training lessons

**Budget was binding, not capacity.** The headline result: the **502,322**-param model at 100k
steps (0.6885 / 99.41%) **beats the 882,402**-param model at the 35k ablation budget
(0.6894 / 99.15%) on both axes — 1.76x fewer parameters, better accuracy, bought with training
time rather than capacity. Trained equally, 882k still wins (0.6733 vs 0.6885) and holds a 3x
better tail.

**Consequence for every ablation table:** deltas stand (matched budget), but absolute numbers are
budget-limited, not capacity-limited, and must be labelled so.

**Stop rules must be applied to the projection, not re-litigated.** At the 75k decision point the
stop threshold was <0.003 px p95; measured −0.0077 and −0.0068, and continued. The 75k->100k gain
was **0.95x and 0.4x the floor** — i.e. not measurable — and the 882k tail moved the *wrong* way.
~1.5 h of GPU for nothing.

**Small models can fail to ignite.** One 222k seed in four collapsed: detection and localisation
were fine (2nd-best p95, best recall), but **identity ignited ~5k steps late and never caught up
within the cosine schedule**. A schedule interaction, not saturation. Dead-ReLU was investigated
and **falsified**: zero hard-dead units in all four seeds, and the best model (882k) has the
*most* near-dead units at the *lowest* firing rate.

**Early exit on convergence is correct**, but a competing baseline must never be stopped on a
looser criterion than your own arms. Comparative claims against an under-trained baseline are
worthless and a reviewer will discount them.

---

## 5. Testing and verification methodology

**Verify, don't assume.** Status is what the tools output, not what your mental model predicts.
Quote numbers only from a file read in the same session. This project produced several
confident-and-wrong mechanistic stories; measurement is the only defence.

**Programmatic one-key invariants.** Ablation configs are full copies (the loader is a bare
`yaml.safe_load`, no inheritance), so a hand-cut arm silently differed in **ten** keys once — an
arm trained on easier data and "beat" the reference, costing an hour of GPU and producing a false
architectural finding. The cutter now diffs leaves and refuses anything that is not exactly the
requested keys.

**Generator fingerprint lock.** A content hash of the generator files plus the config subset that
shapes their output. Training refuses to start if either drifted since the last green audit. The
test asserts a **hardcoded** copy of the covered file list, deliberately not an import, so an
accidental *removal* fails loudly.

**Gates must verify what they claim.** Three separate gates in this project returned green while
checking something other than what they asserted:

- **Preflight loss check** hardcoded focal's constants and called the loss with no form kwargs, so
  every `loss_form=ce` arm predicted one loss and measured another. It passed because focal-vs-focal
  sits inside the tolerance band. The CE arm's real init loss is **4256** against the **47.87** the
  gate was comparing — an **89x** discrepancy.
- **fp16 parity gate, v1** compared `peaks()` output, which returns **integer** coordinates — against
  a 0.05 px threshold that can only read 0 or >=1. It reported p95 = 0.00000.
- **fp16 parity gate, v2** judged on p95 alone and passed 2 of 3 tiers, despite the **mean exceeding
  the p95** — arithmetically impossible without rare outliers. Explicit tail counts flipped all
  three to FAIL.

> **A gate that is green while certifying a code path the run never executes is worse than no
> gate.** Check that your check checks the thing.

**Byte-identical regeneration is proof, not argument.** Removing dead code and regenerating a
figure to a byte-identical result *proves* the code was dead. The same technique reconstructed
unrecorded tool invocations: regenerate to a scratch dir, byte-compare the outputs that should not
have changed.

**Everything of record goes to permanent storage.** The first code audit enumerated 52 findings
into conversation context and a task blob, and nowhere else; when that context was compacted the
list became unrecoverable and had to be re-derived from scratch.

---

## 6. Measured performance

### Detector — 10,000-sample full validation, all tiers at 100,000 steps

| tier | params | p95 | median | M-04 | tail >4 px | M-02 far octave |
|---|---|---|---|---|---|---|
| 222k | 222,138 | 0.7337 | 0.4128 | 99.09% | 0.0190% | 87.53% |
| 502k | 502,322 | 0.6885 | 0.4077 | 99.41% | 0.0146% | 90.49% |
| 882k | 882,402 | **0.6733** | **0.4063** | **99.53%** | **0.0048%** | **91.74%** |

Across a **4x parameter range**, in-envelope identity spans only **0.44 pp** and median
localisation **0.0065 px**. The tiers separate on p95 (+0.060 px), tail (**4.0x**) and far-octave
recall (−4.2 pp) — not on median accuracy.

### Pose — 1000 images, corrected benchmark, all arms through the same PnP

| arm | params | solve% | rot med | rot mean | rot p95 | tr med | tr mean | tr p95 |
|---|---|---|---|---|---|---|---|---|
| 222k refined | 222,138 | 90.0 | 0.1269 | 0.6855 | 1.2261 | 0.0099 | 0.0691 | 0.1523 |
| 502k refined | 502,322 | 92.3 | 0.1295 | 0.5759 | 1.2297 | 0.0103 | 0.0604 | 0.1509 |
| 882k refined | 882,402 | **93.6** | 0.1318 | **0.5368** | **1.1737** | 0.0104 | **0.0376** | **0.1282** |
| Deep ChArUco | — | 93.3 | 0.9131 | 10.6234 | **89.3603** | 0.0770 | 1.2636 | **7.2695** |
| classical OpenCV | — | **54.4** | 0.1602 | 0.8690 | 2.3487 | 0.0200 | 0.0654 | 0.2236 |

**Solve rate alone must never be quoted head-to-head.** Deep ChArUco's 93.3% sits between our
502k and 882k, but its rotation p95 is **89.36 deg against 1.17–1.23**, and its mean is **19x its
own median** — the distribution is dominated by catastrophic solves. It fails by emitting a
confidently wrong pose; ours refuses. Classical has the opposite failure mode: accurate on what
it solves, but it solves only 54.4%.

**Even the coarse (unrefined) arm beats Deep ChArUco on accuracy** — 0.374–0.382 deg median
against 0.913.

**Median accuracy is flat across the ladder; the TAIL is not.** Rotation mean 0.6855 -> 0.5759 ->
0.5368 and translation mean falls **46%** from 222k to 882k. "Pose accuracy is indistinguishable
across the ladder" is true of the median and false of the tail. Capacity buys tail robustness,
mirroring what it does for identity under degradation.

### Robustness — 20 factors, identical frames, recall% / ID% at the hardest step

| factor | 222k | 502k | 882k | Deep ChArUco | classical |
|---|---|---|---|---|---|
| distance s=128 | 91.7 / 88.4 | 92.6 / 91.8 | 92.8 / 97.4 | 89.6 / 50.2 | 18.6 / — |
| **darkness 0.01** | **5.6** / 40.4 | 37.3 / 27.7 | 45.8 / 69.1 | 0.2 / 33.3 | 0.0 / — |
| sensor noise K=0.02 | 16.5 / 61.8 | 15.4 / 63.0 | 16.6 / 83.2 | 13.8 / 73.8 | 2.6 / — |
| motion blur 9 px | 63.6 / 82.0 | 75.3 / 86.5 | 72.8 / 98.0 | 76.8 / 78.8 | 12.5 / — |
| ink contrast 1.6 | 54.9 / 74.1 | 63.6 / 75.4 | 70.3 / 80.8 | 58.5 / 68.3 | 11.3 / — |
| object occlusion x4 | 92.8 / 96.5 | 94.2 / 96.8 | 96.0 / 99.7 | 87.5 / 85.1 | 32.8 / — |
| board occlusion 50% | 82.3 / 40.2 | 87.1 / 56.8 | 85.9 / 71.2 | 73.1 / 63.8 | 9.6 / — |
| tilt 60 deg | 91.5 / 96.6 | 91.9 / 99.4 | 92.7 / 99.4 | 90.0 / 92.8 | 51.6 / — |

**Capacity buys IDENTITY UNDER DEGRADATION, not clean-set accuracy.** Board occlusion is the
cleanest case: 40.2 -> 56.8 -> 71.2% ID, a **31 pp** spread on frames where clean validation
separates the tiers by 0.44 pp. **The validation table badly understates what the parameters buy.**

**Deployment warning — the 222k tier and darkness.** At the darkest step it collapses to **5.6%**
recall against 37.3% and 45.8%: an 8x gap and a cliff, not a graceful decline. On a 940 nm
active-illumination rig, low ambient is not an edge case. Its clean-validation numbers give no
warning of this.

Classical's ID column is ~100% by construction (it only reports corners it has already
identified); read its **recall**, which is 18.6% at s=128 and 0.0% in the dark.

### Deployment — ONNX / fp16

All four artefacts export at opset 17 with zero banned ops: detectors 284/263/263 nodes
(2.05/3.70/5.74 MB) and the refiner 20 nodes (0.37 MB). Op-level agreement with PyTorch is ~1e-4.

**fp16 attribution, measured by halving one net at a time (additive, 41 + 3 = 44):**

| fp16 applied to | corners >0.05 px | max | ID diff |
|---|---|---|---|
| both nets | 44 (0.908%) | 3.589 px | 0.0206% |
| **detector only** | **3 (0.062%)** | 1.000 px | 0.0206% |
| **refiner only** | **41 (0.846%)** | 3.589 px | 0.0000% |

**The refiner owns the entire sub-pixel tail and none of the identity error; the detector owns
integer peak flips (each exactly one cell) and all of it.** Shipping recommendation: **detector
fp16 + refiner fp32**, which takes the tail from ~0.85% to 0.008–0.067% — inside budget on every
tier. The refiner is 97k params, so the cost is negligible. **The naive quantise-everything build
fails on all three tiers.** This is a precision-policy decision, not a retraining one.

---

## 7. Engineering practices and pitfalls

### Build models from the checkpoint's own config, never from a passed-in config

Three separate tools built a network from `--config` rather than the checkpoint's embedded `cfg`.
With models differing in width, gates and `lambda_cls`, that raises a shape error — or, far worse,
**silently scores the wrong architecture** when the shapes happen to match. Infer refiner width
from its state dict rather than instantiating a default.

### Tool defaults are where wrong results come from

A pose tool defaulted `--ckpt` to a superseded model, so *simply running it* produced a
non-release table. Six other tools defaulted their output into a superseded results tree. **Make
the default the thing you want; a wrong default is a wrong result produced by inaction.**

### Resume semantics

- The stream seed is `train_seed * 1000 + resume_count`. Saving a literal `0` for `resume_count`
  means a second resume **replays the first resume's exact sample sequence**.
- Resume must diff the checkpoint's own config against the live one. `cosine_lr` anneals to the
  *budget*, so resuming a 35k arm under a 250k default puts the LR back near peak where it should
  be at the floor — measured at 11–91x the correct value, with nothing in the logs saying so.
- `save_ckpt` must write-then-rename. A bare save truncates the previous good bytes the moment the
  writer is constructed, so a crash mid-serialisation leaves a 0-byte file and no resume point.

### Numerical traps

- **TF32 is on by default for cuDNN.** "fp32" convolutions on Ampere+ run at **fp16's 10-bit
  mantissa**. An ONNX check reported 6.9e-2 "disagreement" that was PyTorch's own precision, not an
  export fault; with TF32 disabled it is 1.5e-4. Any fp32 *reference* you compare against is
  probably TF32 unless you disabled it.
- Compute focal loss in logit space; the naive `log(1-p)` form NaNs under bf16 near forced peaks.
- Grayscale conversion of independently-drawn per-channel noise delivers only
  `sqrt(0.114^2+0.587^2+0.299^2) = 0.6686x` the modelled sigma — measured 0.6676–0.6789. The
  delivered frame is **3.50 dB quieter** than the per-channel formula implies.

### Process traps

- `pkill -f <pattern>` matches **its own shell's argv** and kills the agent. Use exact-argv
  matching, and `setsid` with detached stdin to launch anything that must outlive the shell.
- Never reuse a results directory for a cheap side-run: a tool that writes its report
  unconditionally will destroy a completed record. Add a `--skip-*` flag for artefact-only reruns.
- A DataLoader `fork` context costs **7x throughput** versus `spawn` here (55 vs 404 samples/s).

### Code discipline that paid off

- One canonical function per capability, findable in a single grep. Three unrelated functions
  named `load` is a navigability failure even when each is individually correct.
- Comment the *why* at the site, especially for anything that looks wrong: every non-obvious
  constant in this codebase carries the measurement that justifies it.
- Delete dead code, but **prove** it is dead first (byte-identical regeneration).

---

## 8. Errata — confident-and-wrong stories, recorded deliberately

Kept because each was plausible, was believed, and was false. The pattern is always the same:
a mechanism was proposed before it was measured.

1. **"The refiner guard causes the fp16 jumps."** Disabling the guard made the max **worse**
   (3.35 px vs 0.81). The guard *suppresses* fp16 divergence by substituting the deterministic
   coarse peak exactly where the refiner is least stable.
2. **"Dead ReLUs explain the 222k collapse."** Zero hard-dead units in any seed; the best model has
   the most near-dead units.
3. **"gate_skips [3,2] is the one unambiguous free win."** Called from a 25k validation at +0.16 pp
   (2.6x range); at 35k it decayed to +0.09 pp — exactly **1.0x** the range. Null.
4. **"conv-only tests the loss."** It removes the transformer and leaves the loss alone. Only the
   `loss_form` arm tests the loss.
5. **"rev-5 results are comparable."** Rev-5 was an *easier* dataset. The same model, same steps,
   scored 0.7470 / 99.64% on rev-5 and 0.8129 / 98.98% on rev-6 — a gap an order of magnitude
   larger than the seed floor. Generator revision confounds every cross-revision comparison.
6. **"The 4.7M model's marginal 15 Hz budget applies."** That figure was measured on a model 5.3x
   larger than what ships.

---

## 9. What remains open

- **Noise draw is per-channel** (documented and accepted, not fixed): the delivered SNR is 3.50 dB
  better than the per-channel formula reports. Every arm saw the same data, so all comparisons
  hold; the paper must state that reported SNR is the *delivered* grayscale frame.
- **222k exceeds the fp16 identity budget** by 0.0005 pp (0.1005% vs 0.1%), accepted and stated.
  Negligible beside its real constraint, the darkness cliff.
- **Pose sets still omit SAM2 cutout occlusion** and its alpha visibility test — deliberate, since
  object occlusion is measured on its own robustness axis, but it is a divergence from the
  training generator, not a match.
- **Uncertainty calibration is parked.** `sigma_px` is an ordering signal, never a filter's R;
  `pose_cov` is ~6x over-confident and must not be given a fitted inflation factor.
