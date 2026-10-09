# Tiered model family — one architecture, three deployment points

**Decision (Kaelin, 2026-07-30):** rather than pick a single headline model, present the
width ladder as a **tier menu** — *"half width if you have compute budget, quarter width for
strict / computationally starved / low-cost edge deployment"* — provided the quarter rung
lands close on accuracy.

This is a stronger four-minute story than one number, because it converts an ablation into a
product claim: **the same architecture, retrained at three widths, spans an 11× parameter
range with one code path.** No per-tier engineering, no distillation, no separate designs.

---

## Cost — ALL MEASURED 2026-07-30, not estimated

Every rung is the adopted baseline **`configs/base_nodilate_s05.yaml`** (= `rev640.yaml` +
`e4_dilated: false` + `sigma_hm: 0.5`, adopted by Kaelin 2026-07-30), differing ONLY in
`width_mult` and the parameter-free `xsa`. Re-expressed against that base the family is a
clean 3×2 grid: L `xsa`, M `width_mult=0.5` (± `xsa`), S `width_mult=0.25` (± `xsa`).
Source JSONs: `paper/results_rev6/15_cost/model_cost_abl_{composite,nodilate_width_half,
nodilate_width_quarter}_s05.json` (the composite JSON is the width-1.0 rung; it additionally
carries `xsa: true`, which is parameter-free and so does not affect any number in this table).

| tier | `width_mult` | detector | **+ refiner** | vs Deep ChArUco | GFLOPs/frame | channels e1–e4 | d_attn |
|---|---|---|---|---|---|---|---|
| **L** compute-rich | 1.0 | 3,517,362 | **3,614,418** | 1.61× | 157.3 | 32/64/128/256 | 256 |
| **M** balanced | 0.5 | 882,402 | **979,458** | 0.44× | 52.0 | 16/32/64/128 | 128 |
| **S** edge | 0.25 | 222,138 | **319,194** | **0.14× (7.0× smaller)** | 19.8 | 8/16/32/64 | 64 |

**Deep ChArUco reference:** 2,241,235 params total (ChArUcoNet 1,242,002 + RefineNet 999,233,
counted from their source), converged M-04 **85.44%**.

### Jetson AGX Orin, datasheet roofline

The application budget is **15 Hz pose = 30 fps of lit/dark pairs = 66 ms per frame.**

| tier | fp16 conservative | fp16 optimistic | int8 conservative | headroom vs 66 ms (fp16 cons.) |
|---|---|---|---|---|
| **L** | 15.3 ms · 65.5 fps | 7.7 ms · 131 fps | 7.7 ms · 131 fps | **4.3×** |
| **M** | 5.0 ms · 198.2 fps | 2.5 ms · 396.4 fps | 2.5 ms · 396.4 fps | **13.2×** |
| **S** | 1.9 ms · 521.6 fps | 1.0 ms · 1043.2 fps | 1.0 ms · 1043.2 fps | **34.7×** |

Conservative = 15% of the 68.8 TFLOPS fp16 datasheet peak, optimistic = 30%. Roofline from
datasheet FLOPs/s, NOT a measurement on Orin hardware — we have never run on the module.
5090 latencies in the JSONs are **contended** (two trainers resident) and are not a clean
benchmark; they are recorded with the competing PIDs for exactly that reason.

### VRAM — measured 2026-07-30, B=1 (the deployment case)

`paper/results_rev6/15_cost/vram_report_tiers.json`. Weights are fp32 as trained; the ONNX
column is fp16 weights + [0.35, 1.0] × the eager activation peak, the range spanning full
buffer reuse to none — a **model, not a measurement**, since no TensorRT engine has been built.

| tier | weights (fp32) | activations | torch reserved | ONNX/TRT estimate |
|---|---|---|---|---|
| L | 14.0 MB | 254.4 MB | 454.0 MB | 96–261 MB |
| M | 3.7 MB | 122.4 MB | 210.0 MB | 45–124 MB |
| S | **1.0 MB** | 61.5 MB | 112.0 MB | **22–62 MB** |

**VRAM is not a constraint at any tier** — even L fits in a fraction of any Orin module's
shared memory. Note weights are a rounding error against activations (1.0 MB vs 61.5 MB at S),
so parameter count is the *marketing* axis here, not the memory-limiting one; activations
dominate, and they scale with input resolution, not width. Quote params for the size claim and
activations for a memory claim — they are not interchangeable.

`torch reserved` is the caching-allocator high-water mark, i.e. what `nvidia-smi` would show;
it is 1.8× the true peak and is recorded only to stop anyone quoting `nvidia-smi` as the
model's requirement.

### Be honest about why anyone would tier down

**Tier L already clears the real-time budget with 4.3× headroom.** So the smaller tiers are
NOT required by our own latency target, and claiming otherwise would be easy to puncture.
What they actually buy:

- **Unit cost / BOM** — S at 19.8 GFLOPs opens Orin NX and Nano-class modules, which L cannot
  reach. That is a hardware-price argument, not a speed argument.
- **Power and thermals** — a barrier-docking robot is power-budgeted; 8× fewer FLOPs is 8×
  less energy per frame in the compute-bound limit.
- **Headroom for everything else** — the detector is one task on a robot that also plans,
  controls and localises. 1.9 ms leaves the SoC almost entirely free.

---

## Two structural facts worth a sentence each on the slide

**1. Attention is a scale-invariant 45% of parameters at every rung** — 44.9% / 45.0% / 45.1%
across L / M / S. That is not a coincidence: `width_mult` scales conv channels **and** the
attention dim `d` together (`d = c(d)`, `dcc/model.py:213`), so the conv:attention split is
preserved exactly as the model shrinks. The e4 block likewise holds 25.0–25.2% throughout.
The family is a genuine self-similar scaling, not three hand-tuned models.

**2. The refiner is a FIXED 97,056 params and becomes the dominant cost at the small end:**

| tier | refiner share of total |
|---|---|
| L | 2.7% |
| M | 9.9% |
| **S** | **30.4%** |

Two consequences. Honest caveat: **the shared refiner was trained against the FULL-width
detector's coarse peaks** and is reused unchanged at every tier, so its crop distribution is
slightly off-nominal for S — the refiner-gain measurement should be re-run per tier before
any of these numbers go in a paper. Opportunity: at tier S the single cheapest remaining
parameter saving is a narrower refiner, not a narrower detector.

**The ladder bottoms out at S.** `width_mult: 0.125` will not build — `d` falls to 32 and
head dim to 4, tripping `assert head_dim // 4 >= 2` in `AxialRoPE` (`dcc/model.py:76`), which
needs ≥2 dims per axis-pair for the geometric wavelength spectrum. Going lower requires
`heads` 8→4 as a **second** key change, which is no longer a clean rung of this ladder. Do
not add it silently.

---

## Accuracy — TO BE FILLED, do not quote until training lands

Cost is settled; accuracy is not. Fill from `runs/<arm>/metrics.jsonl` and **quote only from a
file read in the same turn**.

| tier | run | budget | m01 p95 (px) | M-04 (%) | status |
|---|---|---|---|---|---|
| L | `abl_composite_s05_50k_rev6` | 50k → **stopped ~31.6k** | **0.7039** @ 27.5k | **99.38** @ 27.5k | **CONVERGED, not annealed** |
| M | `abl_nodilate_width_half_s05_50k_rev6` | 50k → **stopped 30k** | **0.7396** @ 30k | **98.94** @ 30k | **CONVERGED, not annealed** |
| M+XSA | `abl_nodilate_width_half_xsa_s05_50k_rev6` | 50k | — | — | queued |
| S | `abl_nodilate_width_quarter_s05_30k_rev6` | 30k | — | — | queued |
| S+XSA | `abl_nodilate_width_quarter_xsa_s05_30k_rev6` | 30k | — | — | queued |

**Budgets differ by design and the reason must be stated wherever these are compared.**
`cosine_lr` anchors the LR anneal to `--steps` (`dcc/trainutil.py:53`), so a 50k-scheduled run
stopped at 30k sits at lr 1.09e-4 — 36% of peak, weights still bouncing — whereas a
30k-**scheduled** run has annealed to the 3.0e-6 floor. Scheduling short is therefore strictly
better than stopping early at the same wall-clock. Matching is enforced **within each pair**
(the only like-for-like claim: does parameter-free XSA help at this width?); across tiers the
comparison is against Deep ChArUco's converged number, not against our other rungs.

**L's number is converged but UNANNEALED, and must be labelled that way wherever it appears.**
It was early-exited at ~31,600 of a 50k schedule on the stated criterion (p95 deltas −0.0001
then −0.0009, i.e. two consecutive sub-0.002, followed by a +0.0025 reversal; M-04 flat inside
0.06 pp over five vals). Because the cosine anneal was anchored to 50k, the LR at the stop was
still ~9.5e-5 — 32% of peak — so these weights never annealed and **L's true ceiling is better
than 0.7039**. That biases the tier comparison *against* L and therefore *in favour of* the
small tiers, i.e. conservatively for the claim being made. Do not quote L as this
architecture's best achievable localisation.

### ANNEAL POLICY (Kaelin, 2026-07-30): the conference model must be fully annealed

*"the model that gives the best size to performance tradeoff (i.e. the one we will use for the
conference) we will resume to let fully anneal."*

**CORRECTED 2026-07-30 16:05 — early exit STAYS ON; the resume is the mechanism.** Kaelin:
*"for M width_half ... we still want to let it early-exit, so we can train one or two of the S
models too."* An earlier draft of this section made it a no-early-exit rule, which was wrong:
letting four arms each spend ~20k steps annealing, when only ONE ever ships, buys ~1% on three
models that get thrown away. **Coverage of the ladder is worth more than the anneal on an
also-ran.** The policy is therefore:

- **Early-exit any arm on the usual criterion** (two consecutive sub-0.002 p95 deltas, M-04
  flat inside ±0.15 pp) the moment it triggers, and hand the slot to the next arm.
- **Anneal EXACTLY ONE model — the size/performance winner — by resuming it** with the
  compressed schedule below, once the ladder is complete and the winner is known.
- **The S pair needs no resume**: their cosine is anchored to their own 30k budget, so they
  anneal for free if they run to completion. Only arms stopped short of their schedule
  (L at ~31.6k of 50k, M at 30k of 50k) need it.

**Resume recipe for the winner** (also the fallback after a crash or thermal stop): resume with
a SHORTER total than the original to compress the anneal, since `--steps` overrides the budget
before the checkpoint restores model/EMA/optimiser (`tools/train_detector.py:493`, `539-546`)
and optimiser momentum carries across. Measured for L's step-31,000 checkpoint:

| resume with | lr at resume instant | extra steps | instantaneous drop |
|---|---|---|---|
| `--steps 36000` | 1.77e-5 (5.9% of peak) | 5,000 | **5.7×** |
| `--steps 40000` | 4.03e-5 (13.4%) | 9,000 | 2.5× |
| `--steps 50000` | 1.00e-4 (33.4%) | 19,000 | 1.0× (smooth) |

Shortening buys speed at the price of a **step discontinuity** in LR — standard practice for
step-decay schedules, but not identical to the smooth cosine tail the measured gain came from.
Prefer the smooth full schedule when time allows; that is exactly why the no-early-exit rule
above is the primary mechanism.

**How much is at stake: ~1% relative on p95, and nothing on M-04.** Measured over the anneal
phase (30k→50k) of the two arms that completed a full cosine:

| run | p95 @30k | p95 @50k | gain | M-04 @30k → @50k |
|---|---|---|---|---|
| `abl_nodilate_50k_rev6` | 0.8086 | 0.8030 | −0.70% | 98.95% → 99.01% |
| `abl_sigma1_50k_rev6` | 0.7447 | 0.7366 | −1.09% | 99.23% → 99.20% |

For scale: σ=2.0→0.5 is worth 0.112 px and the L-vs-M tier gap is 0.037 px, against the
anneal's 0.006–0.008 px. **Annealing cannot flip a tier ranking** — it is a final-number
polish, worth having on the shipped model and not worth spending a slot to chase on an
also-ran. Caveat: n=2, both full-width; a capacity-limited model may gain more, which is
untested and is the second reason the S pair keeps its built-in anneal.

**The tier claim needs S to land close, not to win.** Against Deep ChArUco's 85.44%, tier M
was already at 98.25% by step 10,000. S has 7.0× fewer parameters than Deep ChArUco; anything
in the mid-90s makes the menu claim, and matching M makes it emphatic.

## Tier curves on every graph (Kaelin, 2026-07-30)

*"for the 9x9 grid, and the headline figures etc (or anywhere with a graph) we can add curves
for L, M, and S once they are converged, to show the different options in terms of
performance."*

**Cost: affordable.** `tools/robustness_sweep.py` at the default `--n 100` took ~1.5–2 min per
factor on the banked run (20 factors, mtimes 11:16→11:43), so **~30 min per arm uncontended**;
three tiers is ~90 min, roughly 2–3 h if run against live trainers. **Deep ChArUco and
classical do NOT need re-running** — their sweeps are banked at
`05_comparison/robustness_{dc,classical}` and are model-independent of our arms.

### TWO CONSTRAINTS THAT WOULD SILENTLY CORRUPT THE FIGURE

**1. The tier band must be drawn from the COARSE arm.** `CLAUDE.md` pins the Deep ChArUco
comparison as coarse-vs-coarse, because their RefineNet is not run. Plotting refined tier
curves on the same axes as their unrefined detector overstates our advantage. Our refined curve
stays as the existing single overlay showing what the full pipeline adds — it is NOT part of the
tier comparison against baselines.

**2. Curve count has to come DOWN, not up.** `plot_four_way.py:42` already draws 4 series. Three
tiers × (coarse + refined) + DC + classical = **8 per panel**, on a 3×3, projected to a room.
Unreadable. The fix is to make the tiers read as ONE family rather than three competitors:

- **L / M / S: three shades of the same hue** (light→dark = S→L), identical linestyle, so the
  eye reads a *band* spanned by our family, which IS the tier message.
- **DC red dashed, classical grey dotted** — unchanged, stay visually distinct as baselines.
- **Refined overlay only for the headline tier**, keeping the existing draw-order trick
  (`plot_four_way.py:33-41`: coarse thick + semi-transparent, refined thin dashed on top, so
  coincidence is visible rather than one curve hiding the other).

That gives **5 lines**: a 3-shade family band + 2 baselines, + 1 dashed refined overlay.

### Sequencing

Generate sweep data for **all three tiers regardless** — the data is the expensive part, it is
cheap, and it makes every plotting choice available afterwards without re-running anything. Then
decide density from the rendered figure: if the 5-line 3×3 is too dense at slide size, fall back
to a 3-line 3×3 (headline tier + 2 baselines) plus **one** dedicated tier figure on the most
discriminating factors (`distance`, `darkness`, `object_occlusion`). Do not decide this
in advance of seeing it rendered.

**Honest caveat to carry onto any tier graph:** the refiner is SHARED and unchanged across
tiers and was trained against the FULL-width detector's coarse peaks, so refined tier curves are
not per-tier-optimal. Another reason the comparison band should be the coarse arm.

## Regeneration impact

This **supersedes the single-winner assumption** in `00_PENDING_HEADLINE_SWAP.md`. Net effect
on workload: the qualitative artefacts (architecture SVG, introspection panels, darkness
filmstrip) are generated **once for the headline tier only** — the tier table above carries
the rest — but the architecture SVG must now be annotated as parameterised by width rather
than hard-coding one set of channel counts. The 3×3 robustness panel remains the long pole
and is still needed for the headline tier.

---

# What the 7x reduction actually costs (measured 2026-07-30, both sweeps complete)

Kaelin's reading of the panel: FAST's occlusion curve peels away from Conv-ChArT's rather
than sitting uniformly below it, and the explanation is representational capacity -- the
222k model cannot hold a partial-lattice hypothesis together once evidence is missing.

**The data supports it.** Joint rate (found AND correctly named), board_occlusion:

| occluded | Conv-ChArT | FAST | gap | gap as % of Conv-ChArT |
|---|---|---|---|---|
| 0%  | 92.3% | 79.9% | -12.3 pp | **-13.4%** |
| 10% | 93.1% | 78.8% | -14.3 pp | -15.3% |
| 20% | 89.5% | 58.5% | -31.0 pp | -34.7% |
| 30% | 83.6% | 44.8% | -38.8 pp | -46.5% |
| 40% | 64.4% | 23.8% | -40.6 pp | **-63.0%** |
| 50% | 50.1% | 18.2% | -31.9 pp | -63.6% |

The gap widens MONOTONICALLY from 13% to 63% of Conv-ChArT's value. FAST does not merely
start lower; it degrades faster, which is the signature the capacity explanation predicts.

**TWO THINGS TO OWN RATHER THAN DISCOVER IN A QUESTION.**

1. **FAST falls BELOW Deep ChArUco past ~20% occlusion** (44.8% vs 55.5% at 30%), despite
   beating them by +5.5 pp on aggregate corner ID. Their per-corner classifier does not
   depend on global lattice structure, so it degrades more gracefully when the lattice is
   destroyed. This crossing is visible on the panel and a reader will find it.

2. **It is NOT specific to occlusion.** The same gap at each factor's worst step:
   diff_ratio -41.3 pp, motion_blur -26.4 pp, ink_contrast -19.2 pp, defocus -11.5 pp,
   tilt -8.8 pp, distance -8.6 pp. FAST's deficit appears wherever evidence is REMOVED OR
   CORRUPTED, by any cause. Occlusion is one instance of a general pattern, so do not
   claim a mechanism specific to occluders.

**The defensible statement**: identity depends on global lattice structure (the attention
bottleneck is the sole board-scale mechanism -- the undilated conv trunk's effective
receptive radius is only ~21 px). Under favourable conditions the 222k model reproduces the
882k model closely (13% gap). As evidence is lost the smaller model cannot sustain the
partial-lattice inference, and the gap compounds. That is what 7x fewer parameters buys and
costs, stated as a trade rather than hidden.

**Corroborating**: the pose table shows the same shape -- FAST's solve rate is 84.7% vs
94.3%, and its rotation MEAN is worse (1.510 vs 0.838 deg) while its MEDIAN is better
(0.130 vs 0.137). Good median, bad mean, lower solve rate = a heavier failure tail, not a
uniformly worse model.
