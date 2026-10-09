# Autonomous ablation campaign — plan and running results

**Mandate (Kaelin, 2026-07-31):** *"continue autonomously for a few days and at the end have
literally exhaustive, irrefutable ablations and hyperparameter tuning, and then use these results
to make composite results to conduct more ablations to eventually land on the optimal architecture
for the PROBLEM (this does not have to be the same as our starting architecture)."* Ordering is
explicitly **not rigid** — new results may re-rank it. Shared-dataloader training is mandatory for
efficiency. Refiner changes are in scope.

This file is the campaign's index. It is updated as arms land; every number in it is read from
`runs/<arm>/metrics.jsonl` at write time.

---

## Part 1 — What has been measured

All rows are the run's **final 10,000-sample full validation** (the 2,000-sample in-loop val runs
~0.1–0.2 pp optimistic). `p95`/`med` are M-01 coarse localisation in input px; `M-04` is corner-ID
accuracy on matched corners; `tail` is the fraction beyond 4 px, i.e. outside the refiner's capture
range — the operational gate.

### 1.1 Loss (full width, 4,698,034 params, matched 50k)

| arm | p95 | med | M-04 | tail | verdict |
|:--|--:|--:|--:|--:|:--|
| reference (σ=2.0, focal β=4) | 0.8149 | 0.4221 | 98.80% | 0.1221% | baseline |
| σ=1.0 | 0.7430 | 0.4139 | 99.06% | 0.0687% | better |
| σ=0.5 (@25k) | 0.7146 | 0.4110 | 99.01% | 0.0637% | better |
| β=0 (Gaussian kept, discount removed) | 0.7032 | 0.4097 | 98.87% | 0.0463% | better |
| **one-hot BCE (A-CE)** | **0.6690** | **0.4058** | **99.62%** | **0.0056%** | **WINNER** |

**The single largest effect in the campaign: −0.146 px p95, +0.82 pp M-04, and a 22× smaller tail.**

**Mechanism, decomposed.** β=0 isolates the Gaussian's `(1-y)^β` penalty discount while keeping
focal's α term. It captures **77% of A-CE's p95 gain** and 65% of its tail reduction, but only
**8.5% of its identity gain** (+0.07 pp vs +0.82 pp). So two separable mechanisms:
the penalty discount governs **localisation**; the focal→BCE switch — which also drops α's
easy-negative modulation on the **class head** — governs **identity**.

**Corollary that retires a whole axis:** `dcc/losses.py:strict_bce` reads only `(y == 1.0)`, so a
CE arm is **invariant to `sigma_hm`**. The σ ladder cannot inform a CE model, and CE beat every σ.

### 1.2 Architecture (full width unless noted)

| arm | params | p95 | M-04 | tail | verdict |
|:--|--:|--:|--:|--:|:--|
| reference | 4,698,034 | 0.8149 | 98.80% | 0.1221% | baseline |
| conv-only (attn_blocks 0, @25k) | 3,118,514 | 0.8114 | 98.25% | 0.0981% | **attention buys IDENTITY, not localisation** (−0.55 pp M-04, p95 unchanged) |
| nodilate | 3,517,362 | 0.8072 | 98.77% | 0.1161% | **wash for −1,180,672 params — adopted** |
| XSA + nodilate (@25k) | 3,517,362 | 0.8148 | 98.82% | 0.1111% | null at this width |
| gates off | 4,698,034 | 0.8134 | 98.88% | 0.1291% | **neutral**; saves 0 params as coded (constructed + bypassed, `model.py:269`); deleting them would save 24,705 |
| width 0.5 **with** dilation | 1,177,826 | 0.8605 | 98.67% | 0.1409% | dilation does not rescue reduced capacity |
| composite (nodilate+xsa+σ0.5, @25k) | 3,517,362 | 0.7043 | 99.23% | 0.0544% | — |

### 1.3 The capacity × loss 2×2 (fresh, schedule-matched 35k controls, identical batches)

| width | params | focal p95 / M-04 / tail | **CE** p95 / M-04 / tail | Δ |
|:--|--:|:--|:--|:--|
| 1.0 | 4,698,034 | 0.8149 / 98.80 / 0.1221 | **0.6690 / 99.62 / 0.0056** | −0.146 px, +0.82 pp, 22× |
| 0.5 | 882,402 | 0.7352 / 98.86 / 0.0849 | **0.6970 / 99.29 / 0.0114** | −0.038 px, +0.43 pp, 7.4× |
| 0.25 | 222,138 | 0.8096 / 94.87 / 0.0952 | **0.7611 / 94.76 / 0.0218** | −0.049 px, −0.11 pp, 4.4× |

**Kaelin's hypothesis** (big models tolerate BCE because they don't need the stabilisation) is
**not supported at the endpoint** but identified something real: identity gain does erode with
capacity (+0.82 → +0.43 → −0.11 pp), and the clear capacity effect is on **onset** — identity took
~10k steps to catch up at 882k and ~22.5k at 222k. An onset cost, not a ceiling cost.

**Structural finding that shapes the rest of the campaign:** at 222k the failure is **identity**
(94.8% vs 98.9% at 882k) while localisation barely moves (0.761 vs 0.697). Combined with conv-only
showing attention buys identity, the central question for the *problem* becomes:
**can identity be bought more cheaply than by widening the whole network?**

### 1.4 In flight

`wh_attn1` (684,130, one attention block) and `wh_rope5` (λ_min 2.5→5.0), both 35k, vs the fresh
Conv-ChArT control. At 22,500: ATTN1 **−0.35 pp M-04, +0.017 px p95** (deficit shrinking monotonically
from −5.46); ROPE5 a **clean null** across six vals.

---

## Part 2 — The plan

Ordering is by expected information per GPU-hour, and is provisional.

### Phase A — finish the attribution set (queued, ~12 h)
Composite tier ladder first (below), then the remaining one-key arms for exhaustiveness:
σ=0.25 at both widths, `heads4` (parameter-free), `attend16` (H/16 attention), `wh_s1`.

### Phase B — composite tier ladder (queued, ~10 h)
`width_mult × attn_blocks` on the proven base (nodilate + CE; XSA at 0.25 only). Buildable widths
are **only 0.5 / 0.375 / 0.25** — 0.4375 and 0.3125 fail channel divisibility, 0.1875 trips
`model.py:76`'s `head_dim//4 >= 2`.

| width | attn | detector | + refiner | status |
|--:|--:|--:|--:|:--|
| 0.50 | 2 | 882,402 | 979,458 | banked (`wh_ce`) |
| 0.50 | 1 | 684,130 | 781,186 | queued |
| 0.375 | 2 | 502,322 | 599,378 | queued |
| 0.375 | 1 | 390,482 | 487,538 | queued |
| 0.25 | 2* | 222,138 | 319,194 | banked (`fast_ce`) |
| 0.25 | 1* | 172,154 | 269,210 | queued |

### Phase C — loss, exhaustively (the biggest lever so far, ~16 h)
1. **Per-head loss form.** `loss_form` currently switches BOTH heads. β=0 proved the heads are
   driven by different mechanisms, so: focal-hm + BCE-cls, and BCE-hm + focal-cls. Needs a small
   `dcc/losses.py` change (two form keys). **Highest-value single experiment left.**
2. **λ_cls sweep** {0.5, 1.0, 2.0, 4.0} — never ablated, and it directly trades the two axes that
   the 2×2 showed are separable. Run at 0.25 width where identity is the binding constraint.
3. **focal α** {0, 1, 2, 4} — now config-wired (was a dead key until 2026-07-31).
4. **σ_cls** — the class target's width, never touched.

### Phase D — buying identity cheaply (the architecture question, ~20 h)
Motivated by §1.3: at small widths identity fails first, and attention is the identity mechanism.
1. `attn_blocks` ∈ {0,1,2,3,4} **at width 0.25** — does more attention beat more width, per param?
2. `attn_heads` ∈ {4,8,16} — parameter-free.
3. `attend_div` 8 vs 16 at small width.
4. **Class-head resolution.** The class map is H/4; identity is the bottleneck. An H/2 class head
   is a genuine architecture change and a strong candidate for "optimal for the problem".
5. Asymmetric width: wide class head + narrow trunk.

### Phase E — refiner (now unblocked, ~8 h)
`refiner_width` was wired on 2026-07-31 (`Refiner(width_mult=...)`, default reproduces the 97,056
architecture bit-identically; `refiner_for()` infers width from a checkpoint). It matters because
the refiner is **fixed** — 11% of Conv-ChArT but **36% of the 172k tier**.
Sweep {1.0, 0.5, 0.375, 0.25} → {97,056 / 25,520 / 15,744 / 7,032}. Then crop size and `sigma`.

### Phase F — hyperparameters (~10 h)
lr {1e-4, 3e-4, 1e-3}, warmup, EMA decay {0.99, 0.999, 0.9999}, wd, batch×accum. Run on the
*winning* architecture, not the current one — tuning a config we are about to replace is waste.

### Phase G — free, no training
`tau_hm` / `tau_id` / `refine_min_peak` re-tuned per final model (`tune_tau.py`, one forward pass).

### Phase H — final composites and convergence
Best-of-everything at 2–3 tiers, trained to convergence (not 35k), then the full robustness sweep,
pose eval, and figures regenerated.

### Efficiency
Every batch runs through `tools/train_pair.py` (one loader, N models, identical batches — measured
147 → 275 model-samples/s). Supervisor updated 2026-07-31 so a group **larger** than
`MAX_MODELS` may take an otherwise-idle GPU: at 222k the pair ran at 42–53% GPU, i.e.
generation-bound, so 4 small arms on one loader cost barely more than 2. Two separate loader pools
are never allowed concurrently — that is the CPU-starvation configuration this design exists to
avoid.

## 2026-08-01 — composites C1-C3 cut; the "free lever" test

Three levers each bought ~+2 to +3 pp of identity at 222,138 params. Only two were free.
Final 35k full-val, comparator `abl_fast_ce_35k_rev6` (0.7611 / 94.76% / 0.0218%):

| lever | M-04 | p95 | tail >4px | params | free? |
|---|---|---|---|---|---|
| `gate_skips [3,2]` | **+1.99 pp** | 0.7429 (−0.0182) | 0.0217% | +401 | yes |
| `lambda_cls 2.0` | **+3.04 pp** | 0.7580 (−0.0031) | 0.0168% | +0 | yes |
| `hm=focal cls=BCE` | +1.95 pp | 0.8222 (**+0.0611**) | **0.1265% (5.8x)** | +0 | **no** |

`hm=focal` is EXCLUDED from every composite despite winning on identity. The >4px tail is the
operational gate — it is the fraction of corners outside the refiner's capture range — and the
arm also INVERTED at 882k (−0.29 pp M-04 for 8x the tail: 0.0114 -> 0.0931%). An identity gain
paid for in tail is not a frontier point. Recorded because the endpoint M-04 column alone would
have folded this lever in.

At 882k the OTHER per-head arm is the standout: `hm=BCE cls=focal` gives the best localisation
measured at that width (p95 **0.6874**, tail **0.0081%** vs all-BCE's 0.6970 / 0.0114%) for
−0.42 pp M-04. C2 tests whether `lambda_cls 2.0` buys that identity back at no localisation cost.

Queued (one shared loader, `tools/check_queue.py` clean):

| arm | params | change vs base |
|---|---|---|
| C1 `abl_c1_fast_g32_lam2` | 222,539 | `abl_fast_ce` + gate_skips [3,2] + lambda_cls 2.0 |
| C2 `abl_c2_wh_clsfocal_lam2` | 882,402 | `abl_wh_ce` + loss_form_cls focal + lambda_cls 2.0 |
| C3 `abl_c3_wh_lam2` | 882,402 | `abl_wh_ce` + lambda_cls 2.0 |

C3 is C2's attribution control AND the falsifiable half of the lambda claim: if 222k identity was
UNDER-TRAINED (a loss-balance problem) rather than capacity-limited, lambda's +3.04 pp must shrink
toward zero at 882k, which already reaches 99.29%. A null there CONFIRMS the mechanism; a repeat
gain means `lambda_cls: 1.0` is simply the wrong default everywhere and the shipping config changes.

## 2026-08-02 — sigma_cls lands, and a pattern worth naming

`sigma_cls` 0.5 (35k, final 10k full val):

| width | arm | p95 | M-04 | tail >4px | vs control |
|---|---|---|---|---|---|
| 222,138 | `abl_hp_w25_scls05_35k_rev6` | 0.7543 | 96.21% | 0.0184% | −0.0068 px, **+1.45 pp**, tail −0.0034 |
| 502,322 | `abl_hp_w375_scls05_35k_rev6` | 0.7110 | 98.87% | 0.0115% | −0.0010 px, +0.05 pp, tail −0.0024 |

Zero parameters, and the same capacity split every other identity lever has shown.

**THE PATTERN.** Four independent levers now each buy identity at 222,138 params and each go
null at >=502k:

| lever | @222k M-04 | @>=502k M-04 | params | free? |
|---|---|---|---|---|
| `lambda_cls` 2.0 | +3.04 pp | untested (C3) | +0 | yes |
| `gate_skips [3,2]` | +1.99 pp | +0.12 pp | +401 | yes |
| `sigma_cls` 0.5 | +1.45 pp | +0.05 pp | +0 | yes |
| `hm=focal cls=BCE` | +1.95 pp | −0.29 pp | +0 | NO — 5.8x tail |

Four unrelated knobs — a loss weight, a skip-gating topology, a target width, and a per-head
loss form — all helping only where capacity is scarce is too consistent to be coincidence. The
claim it points at: **at 222k identity is UNDER-TRAINED, not capacity-limited.** If so the free
levers are three views of one deficit and stacking them lands near the best single lever (~+3 pp),
not near the +6.5 pp sum. If they are independent, a 222,539-param model comes within reach of the
502k rung's 98.82% and the Pareto frontier moves.

C4 (stack at 222k) and C5 (identical stack at 502k, where each lever was null alone) are queued to
decide it. C5 is what makes C4 interpretable: null at 502k confirms under-training; a gain there
means these are improvements to ship at every width, not a low-capacity patch.

### Methodological note: mid-training M-04 deltas at 222k are NOT predictive

The `sigma_cls` 0.5 arm at 222,138 params, subset-val M-04 vs its control:

| step | 10,000 | 12,500 | 15,000 | 17,500 | 25,000 | 35,000 (full) |
|---|---|---|---|---|---|---|
| delta | **−14.49 pp** | −18.73 pp | −10.22 pp | −0.14 pp | +2.16 pp | **+1.45 pp** |

A 18.7 pp deficit at 12.5k became a 1.45 pp win by 35k. At this width the identity curve is in
its steep phase until roughly 20k (the control itself goes 41.86% -> 85.77% between 10k and
17.5k), so any delta read before ~25k measures CONVERGENCE SPEED, not final quality, and the
rank order can and does invert. Same trap in both directions: this arm was called "badly behind"
at 12.5k and it was not.

RULE: at 222k, no arm is called before its 25k validation; the 35k full val decides. The 502k and
882k rungs converge by ~12.5k and can be read earlier.

## 2026-08-02 — sigma_cls sweep complete, and it exposed the campaign's biggest gap

Final 35k, 10k-sample full val. Both directions away from 1.0 were run.

| width | sigma_cls | p95 | M-04 | tail >4px |
|---|---|---|---|---|
| 222,138 | 1.0 (control) | 0.7611 | 94.76% | 0.0218% |
| 222,138 | **0.5** | 0.7543 | **96.21%** | 0.0184% |
| 222,138 | **2.0** | **0.7490** | 95.65% | **0.0160%** |
| 502,322 | 1.0 (control) | 0.7120 | 98.82% | 0.0139% |
| 502,322 | 0.5 | 0.7110 | 98.87% | 0.0115% |
| 502,322 | 2.0 | 0.7104 | 98.80% | 0.0107% |

**At 222k, BOTH neighbours beat the middle value on ALL THREE metrics.** No monotone story about
class-target width explains that. It admits exactly two readings:

1. the effect is real and non-monotone (1.0 is a genuine local worst), or
2. **`sigma_cls: 1.0` drew a bad seed** — and 1.0 is the CONTROL for the entire 222k ladder.

Those imply opposite conclusions about half this campaign's results, and nothing measured so far
can separate them, because **the noise floor has never been measured**. Every 222k delta on
record (+1.45, +1.99, +3.04, +0.89 pp) is one seed minus one seed. So is every "null" at 502k.

QUEUED: three seeds per width (base 2000 + 2001 + 2002) at 222k and 502k,
`abl_seed200{1,2}_*`, one key off their bases. `train_seed` is inside `cfg['synth']` and IS a
loader key, so seeds cannot pair with each other — each seed pairs across the two widths instead.
Until that spread exists, treat sub-1 pp M-04 differences at 222k as unresolved, not as findings.

Tooling: `tools/cut_ablation.py` gained dotted-key support (`--set synth.train_seed=2001`). It
matches the leaf at any indent and REFUSES a non-unique match rather than guessing; the existing
re-parse-and-diff gate is what makes that sound, since a wrong-scope rewrite cannot pass it.

## 2026-08-02 — C1/C2/C3 final (35k, 10k-sample full val). Composites do NOT stack.

**222,138-param rung** (control `abl_fast_ce_35k_rev6`):

| arm | p95 | M-04 | tail >4px |
|---|---|---|---|
| control | 0.7611 | 94.76% | 0.0218% |
| gates [3,2] alone | **0.7429** | 96.75% | 0.0217% |
| **lambda 2.0 alone** | 0.7580 | **97.80%** | **0.0168%** |
| sigma_cls 0.5 alone | 0.7543 | 96.21% | 0.0184% |
| C1 = gates [3,2] + lambda 2.0 | 0.7624 | 97.25% | 0.0217% |

**C1 IS WORSE THAN LAMBDA 2.0 ALONE ON ALL THREE METRICS** (−0.55 pp M-04, +0.0044 p95,
+0.0049 pp tail). The composite of two winners is beaten by the better of its two parts. Not
merely non-additive -- actively negative. `lambda_cls: 2.0` alone is the best 222k configuration
found to date, and the campaign's second-best 222k arm is gates [3,2] alone on p95 (0.7429).

Mechanistically consistent with the under-training reading: both levers push capacity toward the
class head, and past the point where identity is no longer the binding constraint the extra push
costs localisation without buying anything. The gates lever's OWN p95 advantage (0.7429, the best
at this width) is destroyed by adding lambda on top.

**882,402-param rung** (control `abl_wh_ce_35k_rev6`):

| arm | p95 | M-04 | tail >4px |
|---|---|---|---|
| control | 0.6970 | 99.29% | 0.0114% |
| cls=focal alone | 0.6874 | 98.87% | 0.0081% |
| C3 = lambda 2.0 alone | 0.7059 | **99.40%** | 0.0098% |
| **C2 = cls=focal + lambda 2.0** | 0.6894 | 99.15% | **0.0065%** |

C2 is a REAL Pareto point and the only composite that worked: it recovered 0.28 pp of cls=focal's
0.42 pp identity deficit (98.87 -> 99.15%) while holding essentially all of its localisation win,
and it has **the best tail measured at any width, 0.0065% = 0.57x the control**. Since the tail
is the operational gate (fraction outside the refiner's capture range), that is the 882k arm to
ship if localisation leads. C3 confirms lambda 2.0 is capacity-dependent: +0.11 pp identity for
+0.0089 px of p95 -- free at 222k, a losing trade at 882k.

**CAMPAIGN CONCLUSION ON COMPOSITES.** Three of four "stack the winners" arms failed. Levers that
each fix the same under-training deficit (lambda, gates, sigma_cls -- all identity-side at low
capacity) do not add and can subtract. The one that worked (C2) combines levers acting on
DIFFERENT axes: cls=focal is a localisation lever, lambda is an identity lever. Future composites
should pair across axes, not stack within one.

## 2026-08-02 — C4/C5 final. The rule was too strong: same-axis levers stop adding IDENTITY,
## but they keep delivering LOCALISATION.

222,138-param rung, 35k 10k-sample full val:

| arm | p95 | M-04 | tail >4px |
|---|---|---|---|
| control | 0.7611 | 94.76% | 0.0218% |
| lambda 2.0 alone | 0.7580 | 97.80% | 0.0168% |
| C1 = gates [3,2] + lambda | 0.7624 | 97.25% | 0.0217% |
| **C4 = gates [3,2] + lambda + sigma_cls 0.5** | **0.7502** | 97.75% | **0.0101%** |

**C4 IS THE BEST 222k ARM OF THE CAMPAIGN**, and it is a Pareto improvement on lambda 2.0 alone:
identity matched (97.75 vs 97.80, −0.05 pp -- noise-floor territory), p95 better by 0.0078, and
**tail 0.60x** (0.0101 vs 0.0168%). Against the control: −0.0109 px, +2.99 pp, tail **0.46x**.

This CORRECTS the rule stated after C1. "Same-axis levers do not stack" is right about IDENTITY
-- three identity levers saturate at lambda's ~97.8% and no combination exceeded it -- but wrong
as a blanket claim. Once identity saturates, the same levers keep paying out on LOCALISATION:
gates [3,2] and sigma_cls 0.5 each improved p95 alone, and stacked on lambda they deliver those
p95/tail gains with no further identity. C1 looked like evidence of subtraction only because it
was missing sigma_cls, which is what repairs the p95 that gates alone gives away when combined.

502,322-param rung: C5 = 0.7146 / 99.12% / 0.0107% vs control 0.7120 / 98.82% / 0.0139%
(+0.30 pp M-04, +0.0026 p95, tail 0.77x). The stack is mildly positive at 502k where every
individual lever was null -- but +0.30 pp is exactly the size of claim that needs the noise floor.

**FRONTIER POSITION.** C4 at 222,539 params reaches 97.75% M-04; the 502,322-param control
reaches 98.82%. The stack closed roughly three quarters of the 222k->502k identity gap
(94.76 -> 97.75 of a possible 94.76 -> 98.82) for 2.26x fewer parameters.

## 2026-08-02 — THE NOISE FLOOR, FIRST READ. Several 222k claims do not survive it.

`abl_seed2001_fast_ce_35k_rev6` is `abl_fast_ce_35k_rev6` with ONE key changed:
`synth.train_seed` 2000 -> 2001. Same architecture, same loss, same schedule, same everything
else. 25,000-step 10k-sample full val:

| | p95 | M-04 | tail >4px |
|---|---|---|---|
| seed 2000 (THE CONTROL) | 0.7645 | 93.61% | 0.0218% |
| seed 2001 | 0.7428 | 95.15% | 0.0142% |
| **seed-to-seed difference** | **0.0217 px** | **1.54 pp** | **0.0076 pp** |

**At 502k the same comparison gives +0.0006 px / −0.01 pp** — that rung is reproducible to the
third decimal. The variance is a 222k phenomenon.

### What this does to the 222k results

Measured deltas at 222,138 params against the one-seed control, sorted against a 1.54 pp / 0.022 px
seed difference:

| claim | delta | verdict vs one replicate |
|---|---|---|
| cls=focal collapse | −36.2 pp | SURVIVES (23x) |
| lambda_cls 2.0 | +3.04 pp | SURVIVES (2.0x) |
| C4 three-lever stack | +2.99 pp | SURVIVES (1.9x) |
| C1 gates+lambda | +2.49 pp | marginal (1.6x) |
| gates [3,2] | +1.99 pp | **MARGINAL — comparable to noise** |
| hm=focal cls=BCE | +1.95 pp | **MARGINAL** |
| sigma_cls 0.5 | +1.45 pp | **WITHIN NOISE** |
| sigma_cls 2.0 | +0.89 pp | **WITHIN NOISE** |
| C4 vs lambda-alone | −0.05 pp | WITHIN NOISE (as already suspected) |
| every p95 claim at 222k | <=0.018 px | **ALL WITHIN the 0.022 px spread** |

The "four independent levers each buy identity at 222k and go null at 502k" pattern is NOT
overturned -- lambda's +3.04 pp is real and the 502k nulls are measured on a rung with a
0.01 pp noise floor -- but two of its four members (sigma_cls, and arguably gates) are no longer
separable from seed variance, and the sigma_cls non-monotonicity that triggered this whole
investigation (0.5 AND 2.0 both beating 1.0) is now most simply explained as **the control drew a
low seed**. That was reading 2 of the two offered on 2026-08-02, and it is now the favoured one.

CAVEAT, stated plainly: n=2 gives ONE difference, not a distribution, and this is the 25k full val
(35k pending). Seeds 2002 and 2003 are queued and will turn this into a spread. But a 1.54 pp gap
between byte-identical configs is already enough to say that sub-2 pp identity claims and all p95
claims at 222k must be quoted with a tolerance, not as findings.

ACTION: no 222k claim below ~2 pp M-04 goes in a paper or table without the seed spread beside it.
502k and 882k claims are unaffected.

## 2026-08-02 — seed 2001 FINAL (35k). Every 222k claim, recomputed against the 2-seed mean.

| | p95 | M-04 | tail >4px |
|---|---|---|---|
| 222k control, seed 2000 | 0.7611 | 94.76% | 0.0218% |
| 222k control, seed 2001 | 0.7400 | 96.01% | 0.0151% |
| **2-seed mean** | **0.7505** | **95.38%** | **0.0184%** |
| **seed range** | **0.0210 px** | **1.25 pp** | 0.0067 pp |
| 502k control, seed 2000 | 0.7120 | 98.82% | 0.0139% |
| 502k control, seed 2001 | 0.7125 | 98.84% | 0.0156% |
| **502k seed range** | **0.0005 px** | **0.02 pp** | 0.0017 pp |

Deltas against the 2-seed MEAN (not the single low-seed control every earlier table used):

| arm | dp95 | dM-04 | tail |
|---|---|---|---|
| gates [3,2] | −0.0076 | +1.37 pp | 1.18x |
| sigma_cls 0.5 | +0.0038 | +0.82 pp | 1.00x |
| **lambda 2.0** | **+0.0075 (WORSE)** | **+2.42 pp** | 0.91x |
| C1 gates+lambda | +0.0119 (worse) | +1.86 pp | 1.18x |
| **C4 gates+lambda+sigma_cls** | **−0.0003 (neutral)** | **+2.37 pp** | **0.55x** |

### What changes

1. **EVERY p95 claim at 222k is dead.** The seed range (0.0210 px) exceeds the largest p95 effect
   ever measured at this width. lambda 2.0's "p95 improvement" REVERSES SIGN against the mean
   (+0.0075, i.e. worse). C4's 0.0078 px advantage over lambda-alone becomes −0.0003, exactly
   nothing. Earlier reports called both of those wins; they are not.
2. **gates [3,2] does not survive.** +1.37 pp against a 1.25 pp seed range. Worse, the control's
   OWN seed-2001 run (0.7400 / 96.01%) beats gates [3,2] on p95 and nearly matches its identity
   with zero architectural change. The +1.99 pp originally reported was measured entirely against
   the low seed.
3. **sigma_cls 0.5 does not survive** (+0.82 pp, tail 1.00x). Its non-monotone companion result
   (2.0 also beating 1.0) is now fully explained: the control drew a low seed.
4. **lambda_cls 2.0 and C4 SURVIVE**, at +2.42 and +2.37 pp -- roughly 2x the seed range, and
   consistent with each other. Identity is the only axis on which anything at 222k is real.
5. **C4's TAIL advantage is the one localisation-side result that survives**: 0.0101% vs a 0.0184%
   two-seed mean = **0.55x**, well outside the 0.0067 pp seed range. Since the tail is the
   operational gate (fraction outside the refiner's capture range), C4 remains the arm to ship at
   this width -- but for its tail, NOT for its p95.
6. **502k and 882k are untouched.** 0.0005 px / 0.02 pp reproducibility means every delta reported
   at those widths stands as measured.

n=2 gives a range, not a standard deviation; seeds 2002 and 2003 are queued and will tighten this.
But the direction is unambiguous and the corrections above are not going to reverse.

## 2026-08-02 — NOISE FLOOR at n=3 (final 35k full vals). The 222k ladder is almost all noise.

Three seeds of the IDENTICAL config (`train_seed` 2000/2001/2002 only):

| width | metric | mean | range (n=3) |
|---|---|---|---|
| 222,138 | p95 | 0.7547 | **0.0231 px** |
| 222,138 | M-04 | 95.60% | **1.26 pp** |
| 222,138 | tail >4px | 0.0162% | **0.0100 pp** |
| 502,322 | p95 | 0.7142 | 0.0061 px |
| 502,322 | M-04 | 98.82% | **0.05 pp** |
| 502,322 | tail >4px | 0.0126% | 0.0074 pp |

The two rungs are qualitatively different. At 502k IDENTITY is reproducible to 0.05 pp while p95
is only good to 0.006; at 222k BOTH are loose. Any "noise floor" quoted as one number per width
is wrong -- it is per width AND per metric.

### Every 222k lever, scored against the n=3 range

| arm | dp95 | dM-04 | dtail | verdict |
|---|---|---|---|---|
| gates [3,2] | −0.0118 (0.51x) | +1.16 pp (0.92x) | +0.0055 (0.55x) | **NOISE** |
| sigma_cls 0.5 | −0.0004 (0.02x) | +0.61 pp (0.48x) | +0.0021 (0.22x) | **NOISE** |
| sigma_cls 2.0 | −0.0058 (0.25x) | +0.06 pp (0.05x) | −0.0002 (0.02x) | **NOISE** |
| C1 gates+lambda | +0.0077 (0.33x) | +1.65 pp (1.31x) | +0.0055 (0.56x) | marginal |
| **lambda_cls 2.0** | +0.0033 (0.14x) | **+2.20 pp (1.74x)** | +0.0006 (0.06x) | **REAL, identity only** |
| **C4 stack** | −0.0045 (0.19x) | **+2.15 pp (1.71x)** | −0.0061 (0.61x) | **REAL, identity only** |

**ONE lever survives at 222k: `lambda_cls: 2.0`, +2.20 pp M-04, and C4 reproduces it (+2.15 pp)
without adding anything.** Everything else -- gates [3,2], both sigma_cls directions, every p95
claim, and every tail claim -- is inside the seed range.

FURTHER CORRECTION to the n=2 write-up above: I reported C4's tail advantage (0.55x) as the one
localisation-side result that survived. At n=3 the tail range is 0.0100 pp and C4's advantage is
0.0061 pp -- **0.61x, inside the noise**. It does not survive. The n=2 range understated tail
variance because seeds 2000/2001 happened to bracket it narrowly.

### What this means for the campaign

- The 222k rung cannot resolve effects below ~1.3 pp M-04 or ~0.023 px p95 at n=1. Every
  single-seed 222k comparison in this campaign inherits that limit.
- **Design implication**: further 222k ablations are near-worthless at n=1. Either run 3 seeds per
  arm (3x cost) or move the ablation programme to 502k, where 0.05 pp identity resolution makes
  single-seed comparison sound. 502k costs 2.26x the params but ~25x the identity resolution.
- 502k/882k results already reported stand. C5's +0.30 pp is 6x its 0.05 pp range and is REAL;
  C5's +0.0026 px p95 is inside 0.0061 and is not.
- n=3 range is a crude estimator; seed 2003 is running and will extend it to n=4.

## 2026-08-02 — NOISE FLOOR, FINAL (n=4). The 222k rung is retired as an ablation platform.

Four seeds of the IDENTICAL config, `train_seed` 2000/2001/2002/2003, final 35k 10k-sample full val:

| width | M-04 by seed | range | p95 range | tail range |
|---|---|---|---|---|
| 222,138 | 94.76 / 96.01 / 96.02 / **89.84** % | **6.18 pp** | 0.0231 px | 0.0100 pp |
| 502,322 | 98.82 / 98.84 / 98.79 / 98.75 % | **0.09 pp** | 0.0061 px | 0.0139 pp |

**Seed 2003 did not converge into the pack.** It trailed from 12.5k and finished 4.9 pp below the
next-worst seed. This is not a wider Gaussian -- three seeds cluster in 1.26 pp and one collapses
-- it is a HEAVY TAIL: roughly one 222k run in four simply fails to train out at 35k.

### Consequence: no single-seed 222k result in this campaign is interpretable

That includes the two that survived the n=3 read. Against the 4-seed mean (94.16%):

| arm | M-04 | vs mean | vs 6.18 pp range |
|---|---|---|---|
| lambda_cls 2.0 | 97.80% | +3.64 pp | 0.59x |
| C4 stack | 97.75% | +3.59 pp | 0.58x |

The one thing that can still be said for them: **both exceed the MAXIMUM of all four control
seeds** (96.02%), by 1.78 and 1.73 pp. Under the null, a new draw beating four prior draws has
probability 1/5. Suggestive, not established -- and the two arms share `lambda_cls`, so they are
not independent evidence. Establishing lambda at 222k would need seed replicates OF THE ARM,
which is 3x cost per lever.

### Decision: the remaining ablation programme moves to 502k

502k resolves identity ~69x better (0.09 vs 6.18 pp) for 2.26x the parameters. At that width a
SINGLE seed decides a lever; at 222k three seeds per arm would be the minimum and the tail means
even that is shaky. Queued immediately: `abl_r502_lam2` and `abl_r502_g32`, one key each off
`abl_t_w375_a2_ce`, re-running the only two levers that ever looked real at the width where the
measurement is sound.

Standing corrections from this measurement, applied to everything above:
- gates [3,2], sigma_cls 0.5, sigma_cls 2.0, C1, and every 222k p95/tail claim: **NOISE**.
- lambda_cls 2.0 and C4 at 222k: **UNRESOLVED**, pending the 502k re-test.
- C5 (+0.32 pp vs the 4-seed 502k mean, 3.6x its 0.09 pp range): **REAL**.
- All 502k and 882k single-seed results stand.

## 2026-08-02 — 502k RE-TESTS FINAL. The campaign's real effect sizes.

Both levers re-run at 502,322 params, where four seeds of the control give a 35k full-val range of
**0.09 pp M-04 / 0.0061 px p95 / 0.0139 pp tail**. Control 4-seed mean: 0.7139 / 98.80% / 0.0150%.

| arm | p95 | vs range | M-04 | vs range | tail | vs range |
|---|---|---|---|---|---|---|
| **lambda_cls 2.0** | 0.7208 (+0.0069) | 1.1x | **99.09% (+0.29 pp)** | **3.3x** | 0.0140% | 0.1x |
| gate_skips [3,2] | 0.7063 (−0.0076) | 1.2x | 98.89% (+0.09 pp) | **1.0x** | 0.0090% | 0.4x |

**lambda_cls 2.0 is the ONE confirmed lever of the campaign**: +0.29 pp M-04 at 3.3x the noise
range, zero parameters. Its p95 cost (+0.0069, 1.1x) sits at the edge of the range, but the SIGN
is consistent at all three widths measured (222k, 502k, 882k C3 +0.0089), so a small real cost is
the better reading than noise.

**gate_skips [3,2] is NOT confirmed.** CORRECTION to the read taken at its 25k val, where it
showed +0.16 pp (2.6x) and was called "the one unambiguous free win": at 35k the identity gain
decays to +0.09 pp, **exactly 1.0x the range**. Only its p95 (−0.0076, 1.2x) clears the range at
all, and barely. With n=4 the RANGE systematically understates the spread of a new draw, so 1.2x
is not evidence. Verdict: gates [3,2] is neutral-to-marginally-positive on localisation, null on
identity, for +401 params.

### Effect sizes, corrected for noise, across the whole campaign

| lever | true effect | where measured |
|---|---|---|
| loss_form ce (one-hot BCE) vs focal | **p95 −0.038, M-04 +0.43 pp, tail 7.4x smaller** | 882k, far outside any range |
| lambda_cls 2.0 | +0.29 pp M-04, −0.007 px p95 | 502k, 3.3x range |
| gate_skips [3,2] | ~0, maybe −0.008 px p95 | 502k, 1.0-1.2x range |
| sigma_cls (either direction) | 0 | 502k |
| attn_heads 8->4 | 0 | 882k |
| attend_div 8->16 (+607k params) | 0 | 882k |
| e4_dilated false (−25% params) | 0 | full width |

**The architecture is at its optimum for this problem.** Every structural knob tested -- attention
head count, attention resolution, gate topology, dilation, width beyond 502k -- is null within
noise. The only levers with real effect are in the LOSS: the target representation (one-hot BCE
over Gaussian focal, a large effect) and the head balance (lambda_cls 2.0, a small one). That is
the campaign's central finding and it is the opposite of where the search started.

## 2026-08-03 — sigma_hm ladder complete. The heatmap target's width is a LARGE real lever.

`abl_wh_s1_35k_rev6` (sigma_hm 1.0, 882,402 params, focal) vs `abl_wh_ctrl35k_rev6` (sigma_hm 0.5,
same base, one key): final 35k 10k-sample full val.

| arm | p95 | M-04 | tail >4px |
|---|---|---|---|
| sigma_hm 0.5 (control) | 0.7352 | 98.86% | 0.0849% |
| sigma_hm 1.0 | **0.7781 (+0.0429)** | 98.90% (+0.04 pp) | 0.1006% (1.19x) |

Widening the heatmap target costs **0.0429 px of p95 -- about 7x the measured p95 noise range
(0.0061 at 502k) -- for exactly zero identity change.** The effect REPLICATES at two widths: the
full-width ladder gave 1.0 -> 0.7430 vs 0.5 -> 0.7146, a delta of 0.0284 in the same direction.

Full-width sigma_hm ladder (4,698,034 params, 50k), for the record:
2.0 -> 0.8149, 1.0 -> 0.7430, 0.5 -> 0.7146, 0.25 -> 0.7016, one-hot BCE -> 0.6690.
Monotone all the way to the degenerate limit, with no identity cost anywhere on the ladder.

### The sigma_hm / sigma_cls asymmetry

`sigma_hm` is one of the two largest levers in the campaign. `sigma_cls` is NULL at 502k in both
directions (0.5: +0.05 pp; 2.0: −0.03 pp; both p95 deltas inside the range). The heatmap target's
width matters enormously and the class target's not at all.

That asymmetry has a mechanical explanation already established independently: the class map is at
H/4, so one cell spans 4 input px and sub-pixel structure in the class target carries no
information -- the same reason ID readout was decoupled from refinement (`id_readout="coarse"`,
worth +0.20 pp measured, and the whole identity chain runs on coarse coords). A Gaussian narrower
or wider than a cell changes nothing that can be read out. The heatmap is at full resolution, so
its target width directly sets how precisely the peak can be located.

**Campaign position: every real lever found is in the loss/target representation, none in the
architecture.** loss_form (one-hot BCE over Gaussian focal), sigma_hm, and lambda_cls are the
three; attn_heads, attend_div, gate_skips, gates_enabled, e4_dilated, xsa, sigma_cls, and width
beyond 502k are all null within the measured noise.

## 2026-08-03 — sigma_ref (refiner target width): NULL. The sigma_hm analogue does not transfer.

First sweep of this axis ever (the key did not exist before 358e98e). Both arms 10,000 steps,
schedule-matched, n=9,862 crops per validation.

| sigma_ref | median px | p95 px | <0.25 px | <1 px |
|---|---|---|---|---|
| 1.5 (control) | 0.0969 | **0.9641** | **81.89%** | **95.11%** |
| 0.75 | **0.0937** | 1.0038 | 81.80% | 94.99% |

**No usable effect.** 0.75 takes the median by 0.0032 px (3.3%, and consistent over the last three
validations: 0.0958/0.0975, 0.0940/0.0970, 0.0937/0.0969), 1.5 takes p95 and both fractions. The
p95 column cannot arbitrate: the CONTROL's own p95 moved 1.0276 -> 1.0501 -> 0.9641 over its last
three validations, a ±0.04 wobble the same size as the between-arm gap. There is no refiner noise
floor measured, so a 3% median difference is not callable.

**This is a real asymmetry with the detector.** sigma_hm is one of the two largest levers in the
campaign (0.5 -> 1.0 costs 0.0429 px p95, ~7x the detector noise range, and the full-width ladder
is monotone from 2.0 to one-hot BCE). Its exact refiner analogue does essentially nothing.

A plausible mechanism -- NOT established: the detector reads a hard argmax, so target sharpness
sets peak precision directly, while the refiner reads a 5x5 soft-argmax centroid that integrates
over the target's width and is therefore insensitive to it. Consistent with the known ~0.19-grid-
unit truncation bias of that 5x5 window at sigma=1.5 being a readout-side floor. Recorded as a
hypothesis; this project has produced confident-and-wrong mechanistic stories before.

PRACTICAL CONSEQUENCE: leave sigma_ref at 1.5. The knob is now reachable from config if a future
change to the readout (e.g. a wider soft-argmax window) makes it matter.

## 2026-08-03 — REFINER NOISE FLOOR (3 seeds) and what it does to today's two refiner verdicts

Three seeds of the IDENTICAL refiner config (train_seed 2000/2001/2002), 10,000 steps,
n=9,862 crops per validation:

| | median px | p95 px |
|---|---|---|
| 3-seed mean | 0.0957 | 1.0036 |
| **3-seed RANGE** | **0.0017** | **0.0940** |

The two metrics differ by ~55x in reproducibility -- the same split the detector showed (502k
resolved identity to 0.05 pp but p95 only to 0.006 px). **Refiner median is the reliable axis;
refiner p95 is not.**

Scored against that floor:

| arm | median | vs mean | x range | p95 | vs mean | x range | verdict |
|---|---|---|---|---|---|---|---|
| sigma_ref 0.75 | 0.0937 | −0.0020 | **1.2x** | 1.0038 | +0.0002 | **0.0x** | **NULL, confirmed** |
| refiner width 0.5 | 0.1097 | +0.0140 | **8.2x** | 1.3918 | +0.3881 | **4.1x** | **REAL degradation** |
| refiner width 0.25 | 0.1367 | +0.0409 | **24.2x** | 1.9052 | +0.9016 | **9.6x** | **REAL, severe** |

**Width verdict CONFIRMED against a measured floor**: shrinking the refiner costs real accuracy at
8.2x (0.5) and 24.2x (0.25) the median noise range. Unlike every detector-side structural knob,
this one is NOT free. Ship the refiner at width 1.0 (97,056 params).

**sigma_ref remains NULL**: 1.2x the median range and 0.0x the p95 range. The earlier call stands.
But note the REASONING was partly wrong: it leaned on p95, which this measurement shows is the
noisy axis (range 0.0940, wider than any sigma_ref effect could be). The right axis was median,
where the effect is 1.2x -- still inside the noise, but for a reason that had to be measured
rather than assumed.

**Method note carried forward**: quote refiner results on MEDIAN with a 0.0017 px tolerance;
treat any p95 difference under ~0.09 px as unresolved. The detector's per-metric, per-width
tolerance table and this one now cover every rung the campaign reports on.

## 2026-08-03 — BCE ON THE REFINER: no. The detector's biggest lever does NOT transfer.

Kaelin asked whether the refiner uses BCE. It did not -- focal was hard-wired in refiner_loss with
no config path at all (fixed 7aea0d4, `refiner_loss_form`). Tested against the measured three-seed
floor (mean median 0.0957 px, range 0.0017; mean p95 1.0036, range 0.0940):

| step | BCE median | focal 3-seed mean | x range | BCE p95 | focal p95 | BCE <0.25px | focal |
|---|---|---|---|---|---|---|---|
| 7,000 | 0.1007 | 0.0981 | 1.1x | 1.2431 | 1.1097 | 80.55% | 81.56% |
| 8,000 | 0.0986 | 0.0967 | 1.2x | 1.1907 | 1.0558 | 80.66% | 81.78% |
| 9,000 | 0.0984 | 0.0960 | 1.4x | 1.1903 | 1.0401 | 80.63% | 82.05% |
| **10,000** | **0.0969** | **0.0957** | **0.7x** | **1.1281** | **1.0036** | **81.14%** | **82.11%** |

**BCE loses on every axis, mildly.** Median +0.0012 (0.7x the range -- inside noise). p95 +0.1245
(1.3x -- marginally outside). The clearest signal is the sub-0.25 px fraction: BCE trails by
~1 pp at EVERY validation from 3,000 on (81.14 vs 82.11 at the end), and that is precisely the
sub-pixel-precision measure the mechanism predicts.

**The prediction recorded before launching was directionally right and quantitatively too strong.**
It said a one-hot target would degenerate the 5x5 soft-argmax centroid toward hard-argmax
quantisation (0.125 px). The centroid does NOT fully degenerate -- median lands at 0.0969, nowhere
near 0.125 -- but the tightest precision band degrades exactly as the mechanism says it should.
Partial confirmation, and the overshoot is recorded rather than quietly dropped.

**The asymmetry is now the finding.** focal -> one-hot BCE is the LARGEST single effect measured on
the detector heatmap (p95 0.8149 -> 0.6690, tail 22x smaller) and is mildly NEGATIVE on the
refiner. Both heads regress a Gaussian splat with a forced 1.0 peak; they differ in READOUT --
hard argmax vs 5x5 soft-argmax centroid. Target sharpness helps a readout that takes the max and
hurts one that takes a weighted mean over a window.

This also reframes the sigma_ref null (1.2x the median range): sigma_ref 0.75 vs 1.5 barely moved
anything because BOTH are wide enough for the centroid, and even the degenerate limit only costs
~1 pp of sub-0.25 px. The refiner's target width is simply not the binding constraint.

DECISION: keep the refiner on focal, sigma_ref 1.5, width 1.0. Every refiner lever tested is
null-to-negative; the shipped configuration was already the right one.

NEXT (no training needed): the mechanism predicts the binding constraint is the READOUT WINDOW,
not the target. Sweep the soft-argmax window (5x5 -> 7x7/9x9) on the BANKED checkpoints -- the
one refiner knob this campaign has never touched, and the only one the evidence actually points at.

## 2026-08-03 — ALL FOUR NOISE FLOORS MEASURED. The campaign's tolerance table.

| rung | seeds | M-04 range | p95 range | tail range |
|---|---|---|---|---|
| 222,138 | 4 | **6.18 pp** | 0.0231 px | 0.0100 pp |
| 502,322 | 4 | 0.09 pp | 0.0061 px | 0.0139 pp |
| **882,402** | 3 | **0.07 pp** | **0.0020 px** | 0.0049 pp |
| refiner (97,056) | 3 | median 0.0017 px | p95 0.0940 px | — |

882k 3-seed mean @35k full: p95 0.6974, M-04 99.25%, tail ~0.0087%.

Reproducibility improves monotonically with capacity, and the p95 floor at 882k is **11.6x tighter
than at 222k**. Every number this campaign reports is now quotable with a per-rung, per-metric
tolerance -- which is what "irrefutable ablations" actually requires.

### Every 882k claim, re-scored against the 35k floor (0.0020 px / 0.07 pp)

| arm | dp95 | x floor | dM-04 | x floor | verdict |
|---|---|---|---|---|---|
| **C2** (hm=BCE, cls=focal, lam 2) | **−0.0080** | **4.0x** | −0.10 pp | 1.4x | **REAL localisation win** |
| C3 (lam 2 alone) | +0.0085 | 4.3x | +0.15 pp | 2.1x | REAL trade, both axes |
| lambda 1.5 | +0.0047 | 2.4x | +0.09 pp | 1.3x | net negative |
| lambda 4.0 | +0.0169 | 8.5x | +0.26 pp | 3.7x | REAL trade, both axes |

CORRECTION to the earlier read: C3 (lambda 2.0) was reported as "identity gain decayed to null" at
882k. Against the MEASURED floor its +0.15 pp is 2.1x -- a real, if small, gain. What is true is
that it costs 4.3x the floor in p95 to get it. lambda at 882k is a TRADE at every setting tested,
never free; the free-at-502k result does not transfer, which is the claim that survives.

At 882k the lambda ladder is monotone and linear: 1.5 -> +0.09 pp/+0.0047, 2.0 -> +0.15/+0.0085,
4.0 -> +0.26/+0.0169 -- an exchange rate of ~0.06 px of p95 per pp of identity.

**C2 IS THE 882k RELEASE CONFIG**: best p95 measured at this width (4.0x the floor) and the best
tail (0.0065% vs the 0.0087% seed mean), for an identity cost of 1.4x that is not resolvable.
Since the tail is the operational gate -- the fraction outside the refiner's capture range -- this
is the right trade for the deployed system.

### Release models launched (100,000 steps, 2.9x the ablation budget)

The 222k seed data showed BUDGET, not capacity, binding on identity at low width (seed 2003 was
still climbing at 35k), so release models get a full-length schedule rather than the 35k ablation
budget that every comparison above used.

| run | params | config |
|---|---|---|
| `rel_w882_c2_100k_rev6` | 882,402 | C2: hm=BCE, cls=focal, lambda_cls 2.0 |
| `rel_w375_lam15_100k_rev6` | 502,322 | BCE + lambda_cls 1.5 (the only strictly free lever at that width) |

cls=focal is deliberately NOT folded into the 502k arm: it is the 882k winner but has never been
tested at 502k, and an untested combination does not belong in a release model.

## 2026-08-04 — RELEASE MODELS FINAL (100,000 steps). Budget was binding, not capacity.

Both arms trained 2.9x the 35k ablation budget. Final 10k-sample full val:

| model | params | p95 | M-04 | tail >4px |
|---|---|---|---|---|
| **rel_w882_c2_100k** | 882,402 | **0.6733** | **99.53%** | **0.0048%** |
| **rel_w375_lam15_100k** | 502,322 | 0.6885 | 99.41% | 0.0146% |

Against their own 35k-budget twins, every delta is multiples of the measured floor:

| model | dp95 | x floor | dM-04 | x floor |
|---|---|---|---|---|
| 882k C2 (vs 0.6894/99.15%) | −0.0161 | **8.1x** | +0.38 pp | **5.4x** |
| 502k lam1.5 (vs 0.7122/98.98%) | −0.0237 | **3.9x** | +0.43 pp | **4.8x** |

**THE HEADLINE RESULT FOR THE PAPER.** The 502,322-param release model at 100k
(0.6885 / 99.41%) BEATS the 882,402-param model at the 35k ablation budget (0.6894 / 99.15%) on
BOTH axes, and beats its 3-seed mean (0.6974 / 99.25%) by more. **1.76x fewer parameters, better
accuracy, bought with training time rather than capacity.** Directly relevant to the AGX Orin
target. The 882k model still wins outright when trained equally (0.6733 vs 0.6885, 7.6x the floor)
and holds a 3x better tail, so it remains the accuracy-first choice.

Consequence for the ablation tables: every comparison in this campaign was run at a matched 35k
budget, so the DELTAS stand -- but the ABSOLUTE numbers are budget-limited, not capacity-limited,
and must be labelled as such.

### Calibration note: the last 25k steps were not worth it

At the 75k decision point I set a stop threshold of <0.003 px p95 improvement, measured
-0.0077 (882k) and -0.0068 (502k), and continued. From 75k to 100k the gain was:

| model | dp95 75k->100k | x floor | tail 75k -> 100k |
|---|---|---|---|
| 882k C2 | −0.0019 | **0.95x** | 0.0040% -> 0.0048% |
| 502k lam1.5 | −0.0024 | **0.4x** | 0.0163% -> 0.0146% |

Both inside their floors, i.e. NOT measurable. The 75k plateau was real and the hedge I used to
justify continuing -- "the tail may still be improving, and it is only visible at full vals" --
did not pay: the 882k tail moved the WRONG way (within its 0.0049 pp range). ~1.5 h of GPU spent
for nothing. The stop rule should have been applied to the 75k->100k projection, not re-litigated.
Recorded so the next long run stops on its own criterion.

## 2026-08-04 — THE THREE-TIER RELEASE LADDER, complete. All arms 100,000 steps, matched budget.

### Detector, 10,000-sample full val @ step 100,000

| tier | params | p95 | median | M-04 | tail >4px | M-02 far octave |
|---|---|---|---|---|---|---|
| `rel_w25_lam2_100k_rev6` | 222,138 | 0.7337 | 0.4128 | 99.09% | 0.0190% | 87.53% |
| `rel_w375_lam15_100k_rev6` | 502,322 | 0.6885 | 0.4077 | 99.41% | 0.0146% | 90.49% |
| `rel_w882_c2_100k_rev6` | 882,402 | **0.6733** | **0.4063** | **99.53%** | **0.0048%** | **91.74%** |

Across a 4x parameter range, IN-ENVELOPE identity spans only 0.44 pp and median localisation
0.0065 px. The tiers separate on p95 (+0.060 px), tail (4.0x) and far-octave recall (−4.2 pp).
NOTE M-04 is conditioned on detection, so the small tier's 99.09% is scored on a smaller,
easier matched set -- part of the narrow identity gap is selection, not parity.

### Pose, SD-10 set (1000 images, with refiner)

| tier | solve rate | M-05 rot median / p95 | M-06 trans median / p95 |
|---|---|---|---|
| 222k | 92.9% | 0.1167 deg / 1.3674 | 0.0081 sq / 0.1476 |
| 502k | 95.2% | 0.1206 / 1.3740 | 0.0084 / 0.1485 |
| 882k | 95.5% | 0.1213 / 1.3583 | 0.0085 / 0.1573 |

Pose ACCURACY is indistinguishable across the ladder -- 0.005 deg of median rotation separates
222k from 882k, which is noise. What separates them is SOLVE RATE (92.9 -> 95.5%), and the
refusal breakdown says why: `too_few` corners, 68 vs 43. The small tier fails by not finding
enough corners to fit a lattice, never by fitting a bad pose. That is the correct failure mode
for a docking controller, and it is a RECALL story, not a pose-precision story.
(The pose benchmark is photometrically easier than train/val -- audit finding B1 -- so these
absolute figures are optimistic for every tier. Cross-tier ranking is unaffected.)

### Robustness, 20 factors, identical frames -- recall% / ID% at the HARDEST step

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

**THE TIER STORY.** Recall is near-flat across 4x of parameters in benign conditions; capacity
buys IDENTITY UNDER DEGRADATION. Board occlusion is the cleanest case: 40.2 -> 56.8 -> 71.2% ID,
a 31 pp spread on frames where the clean val set separates the tiers by 0.44 pp. The validation
table badly understates what the extra parameters are for.

**DEPLOYMENT WARNING -- the 222k tier and darkness.** At the darkest step it collapses to 5.6%
recall, against 37.3% (502k) and 45.8% (882k) -- an 8x gap to the next rung, and a cliff rather
than a graceful decline. This system is a 940nm ACTIVE-ILLUMINATION docking rig, so low ambient
is not an edge case. Do not ship the 222k tier where ambient light can drop. Its clean-val
numbers give no warning of this.

Classical OpenCV's ID column is ~100% by construction (it only reports corners it has already
identified); read its RECALL, which is 18.6% at s=128 and 0.0% in the dark.

---

## 2026-08-04 -- audit correctness fixes A3 / A4 / A6 (verified, not asserted)

Three findings from the Fable over-complexity/correctness audit, each verified against source
and exercised before being called done.

### A6 (fixed) -- resume silently re-anchored the LR schedule

No trainer compared a checkpoint's OWN `cfg` against the config the resume was running under;
the only read of `ckpt["cfg"]` anywhere was `train_detector.py`'s retarget provenance. Because
`cosine_lr` anneals to the budget, resuming a 35k arm under the YAML's 250k default puts the LR
back near peak where it should be at the 3.0e-6 floor -- and nothing in the logs said so.

`dcc/trainutil.py: warn_cfg_drift(ckpt, live_cfg, steps, budget_key)` now diffs the two configs
leaf-by-leaf and warns per changed key, plus an explicit LR-RE-ANCHORED warning when the budget
moved. Wired into all four trainers (`train_detector.py:546`, `train_refiner.py:232`,
`train_charuconet.py:366`, `train_pair.py:164`). `budget_key` is a parameter, not a hardcoded
`"train.steps"`, because the refiner's budget lives under `refiner_train` and reading the wrong
section would have made it silently never fire.

It WARNS rather than raises: a deliberate config change on resume is legitimate (the compressed
30.6k -> 35k anneal was exactly that), so it must be loud, not blocking.

Exercised against `runs/rel_w25_lam2_100k_rev6/ckpt_latest.pt`: identical cfg -> silent, `{}`;
budget 100k -> 250k plus `sigma_hm` 0.5 -> 2.0 -> both keys reported plus the re-anchor warning;
`refiner_train.steps` -> fires only with the right `budget_key`; a cfg-less legacy ckpt -> no-op,
no crash.

*Caught while fixing*: my first patch referenced `total_steps`, which in BOTH `train_detector.py`
and `train_refiner.py` is defined AFTER the resume block (`:572` / `:249`) -- a guaranteed
NameError on every resume. `py_compile` passes such code happily. The working reference is
`tcfg["steps"]` / `rcfg["steps"]`, which is also correct because `--steps` is folded into `cfg`
at `:494` / `:206`, before those dicts are taken.

### A4 (fixed) -- resume_count was saved as a literal 0, so a second resume REPLAYED the first

`tools/train_charuconet.py:437,446` passed `0` to `save_ckpt` while the live `resume_count`
existed at `:362-367` and fed `stream_seed = train_seed*1000 + resume_count` at `:374`. Every
checkpoint therefore claimed to be a first run. Resume #1 correctly used seed+1 but recorded 0;
resume #2 would recompute 0+1 = 1 and replay resume #1's exact sample sequence.

This is the DEEP CHARUCO BASELINE, so it is a rule-4b fairness issue, not a curiosity.

**Measured blast radius -- the shipped baseline is CLEAN.** `charuconet_rev6_ft` is a resumed run
(it begins at step 50,000, continuing `charuconet_rev5_ft`) and every one of its checkpoints
records `resume_count=0`, confirming the bug was live. But `metrics.jsonl` has NO non-monotonic
step transitions across 14,463 records (48,001 -> 62,458): it was resumed exactly once, and that
single resume used the correct distinct stream. The bug was LATENT here -- one more resume would
have triggered it. No baseline number needs re-running.

### A3 (reporting side fixed; the DRAW is Kaelin's call)

Noise is drawn on the 3-channel BGR working buffer (`dcc/synth.py:878-881`) and converted to
grayscale LAST (`:1020`). `BGR2GRAY` is a luminance-weighted average of three INDEPENDENT draws,
so the delivered single-channel frame carries only `sqrt(0.114^2+0.587^2+0.299^2) = 0.6686x` of
the modelled sigma.

MEASURED, not assumed: 0.6789 / 0.6723 / 0.6680 / 0.6676 at sigma = 3 / 5 / 8 / 12 DN, converging
on 0.6686 as sigma grows (the spread at small sigma is uint8 quantisation).

**Direction matters and is easy to get backwards.** The delivered frame is QUIETER than modelled,
so the true SNR the network sees is 3.50 dB BETTER than `snr_calibration.py` reported. The noise
axis of this benchmark is MILDER than the per-channel formula implied -- it was never harsher.

`tools/snr_calibration.py` now reports the delivered-frame SNR as `snr_db_*` and preserves the
old per-channel figure as `snr_db_*_modelled`, with the factor derived from cv2's own BGR2GRAY
weights rather than hardcoded. Regenerated at the full default n=60: `S_mean` is bit-identical to
the previous record for all 11 rows and every row moved by exactly +3.497 dB, so the change is
the correction alone and not resampling drift. Headline row K=0.02: 4.63 -> 8.13 dB.

Consumers that read `snr_db_mean` had stale SNR axes and were regenerated:
`22_fourway_RELEASE/`, `22_fourway_RELEASE_882_vs_222/` (`plot_four_way.py`), and
`03_robustness/figures/`, `03_robustness_PREGUARD/figures/` (`plot_robustness.py`).

`filmstrip_lighting.py` was regenerated too and came back BYTE-IDENTICAL. The reason is a bug,
not a coincidence: `filmstrip_lighting.py:94` builds `snr_of` from the calibration file and then
NEVER USES IT -- the name appears exactly once in the file. So that figure never displayed a
calibrated SNR at all and was never affected.

(RESOLVED 2026-08-05, superseding the "left in place as evidence" decision recorded here
originally: the block could never have worked -- the calibration is keyed by sensor gain K while
the filmstrip sweeps darkness, so the lookup shared no key domain with the figure's axis. Removed;
the figure still regenerates byte-identical. Full triage in `23_code_audit_FINDINGS.md`.)

**STILL OPEN -- Kaelin's call.** Making the DRAW channel-shared (so the delivered sigma equals the
modelled one) changes the data distribution and is a rev-boundary change requiring a retrain. It
is not a reporting fix and was not made. The alternative is to state the delivered-vs-modelled
distinction in the paper, which the tool now supports directly.

### Reproducibility gap closed while doing this

The `plot_four_way.py` invocations behind the two release figure sets were recorded NOWHERE. They
were reconstructed and PROVEN correct: regenerating to a scratch dir reproduced every
SNR-independent factor BYTE-IDENTICALLY (`contrast`, `tilt`, `rotation`, `vignette`), which pins
the sweep dirs and the labels. Pinned here so they are never guessed at again:

    python tools/plot_four_way.py --ours paper/results_rev6/20_robustness_REL882 \
        --fast paper/results_rev6/20_robustness_REL502 \
        --out-dir paper/results_rev6/22_fourway_RELEASE \
        --ours-label "Conv-ChArT-882k" --fast-label "Conv-ChArT-502k" --panel-tag release

    python tools/plot_four_way.py --ours paper/results_rev6/20_robustness_REL882 \
        --fast paper/results_rev6/20_robustness_REL222 \
        --out-dir paper/results_rev6/22_fourway_RELEASE_882_vs_222 \
        --ours-label "Conv-ChArT-882k" --fast-label "Conv-ChArT-222k" --panel-tag tiers

Note `PANEL_FACTORS` (`plot_four_way.py:96`) includes `sensor_noise_K`, so the PANEL figure is
also affected by the SNR correction -- expected, not a discrepancy.

### Also: six tool defaults pointed into the SUPERSEDED rev-5 tree

`eval_uncertainty.py` (x2), `refiner_id_effect.py`, `calibrate_uncertainty.py`,
`robustness_sweep_dc.py`, `find_attention_frames.py` all defaulted their `--out` into
`paper/results_rev5/`, which `00_SUPERSEDED.md` forbids quoting. Retargeted to `results_rev6/`.
All six create their parents, so the two paths with no rev-6 counterpart yet
(`10_uncertainty`, `04_introspection`) are fine. This does NOT restart the PARKED uncertainty
work -- it only stops a future run writing into a tree nothing may be quoted from.

---

## 2026-10-08 — Kalman measurement noise for the 882k model (26) and the zero-shot board-transfer baseline (27)

Both directories carry their own README with every number file-referenced; this is the log entry.

**26_pose_error_variance.** Kaelin asked for "the variance of the error for a kalman filter" for the
882k model in deployment. `tools/eval_pose_ours.py` gained `--per-image` (per-solve camera-frame
6-vector error) and an `error_cov` block (ensemble stats on the accepted = unambiguous set, plus by
apparent-scale octave). The gitignored pose sets were gone; `eval_pose_rev6_b1` was regenerated and
reproduces `21_pose_REL882_B1.json` EXACTLY (936/1000, 64 refused, same medians), so the set is
seed-deterministic as designed. Findings: robust (MAD) std 0.09/0.09/0.04 deg rotation and
0.0029/0.0027/0.0116 squares translation; sample variance 9-14x the robust variance (heavy tail:
33.6% of accepted solves beyond 3 robust sigma); off-diagonals weak (|rho| <= 0.23). Kaelin's
follow-up "constant or per-frame?": a constant R is 2-7x over-confident at s=12-16 px and 2-3.6x
over-cautious at s=64-128 px; a RANGE-scaled R (sigma = a z^k, z = range in squares, k ~1 rotation,
~1.5 lateral, ~1.7 depth) is matched within 0.83-1.24 in every octave; the network's own `pose_cov`
stays out (NEES 11.2 mean / 2.75 median -- does not rank frames reliably; the PARKED pin stands);
`rms_px` > 0.15 gates half the >5-sigma tail at 15.8% rejection. The tail survives every R, so the
filter needs a chi-square innovation gate regardless.

**27_transfer_zeroshot.** Baseline for the fine-tuning/transfer figures Kaelin wants for the report.
Target `configs/transfer_dict6x6.yaml` = 882k config + exactly `board.dictionary: DICT_6X6_250`
(cut_ablation one-key gate; corner geometry verified identical; DICT_5X5_100/250/1000 rejected
because their first 50 markers ARE DICT_5X5_50's). Full suite on identical frames: localisation and
recall transfer within noise (median 0.4063 both, p95 0.6733 vs 0.6754, M-02 within 0.2 pp); M-04
99.53% -> 13.36% (chance 6.25%); post-gate ID in the 20-factor sweep 95.3% -> 3.9% mean.
**The headline is the pose set**: 22.0% of frames return a CONFIDENTLY WRONG pose (accepted, not
flagged ambiguous), 73.8% of all zero-shot solves are the true pose composed with a 90-degree lattice
symmetry (translation error exactly 5.000 squares), 14.6% with 180 degrees; their PnP residual
(0.098 px) is indistinguishable from correct solves (0.088). Mechanism, stated from the measurement
not from theory: a relabelling by a symmetry of the 4x4 lattice fits the canonical lattice exactly,
so the RANSAC gate passes it and IPPE's two-fold flip test never fires. Kaelin's prior "good
localisation, 0 ID" was right on the first half and too kind on the second: a foreign board does not
degrade the pipeline, it turns the gate's abstention guarantee off.

Tooling: `plot_four_way.py --fast-refined` (draws the second slot's refined arm; baselines suppressed by
pointing `--dc/--classical` at an empty dir), new `tools/plot_transfer.py` (arms as `--arm` so fine-tuned
models become extra bars). Surfaced, not fixed: `dcc/pipeline.py:291` `recover` has no guard on the
homogeneous divide; nan distances from an infinite projection compare False against `tol` and could
assign a bogus recovered ID -- unreachable with a sane H, reachable with the garbage H of a foreign board.

---

## 2026-10-08/09 — Minimal fine-tuning ladder (28) and multi-board pretraining (29)

Kaelin: "figure out what the absolute minimum amount of fine-tuning required is" to move the 882k release model to
a new board, with VRAM / steps / accuracy / wall time side by side for an on-device feasibility call; then "training
the main model from scratch on a larger set of boards (say 15)"; then "take the 25k model, freeze everything before
the bottleneck (inclusive) and train it to be good at one specific board".

**Infrastructure.** `finetune:` entry in train_detector (load EMA weights, fresh step/optimizer/schedule,
`lr_mult` {fnmatch: multiplier} decides trainability + per-group LR via `param_groups(lr_mult=)`, `reinit`
patterns, per-step `peak_mem_mb`, final `done` record); `cut_finetune_arms.py` (21 arms, flat-diff asserted);
`run_finetune_arms.sh` (train -> trainer's own 10k full val -> B1 pose set; checkpoint name derived from the
budget after a 20k arm's pose eval silently skipped); `finetune_cost.py` (isolated GPU cost, merges);
`plot_finetune.py` (curves, steps/wall to thresholds, VRAM, final; `--arms` subsets, cost aliases by trainable
set); board pool in `synth.py:_composite_board` + `marker_id_offset` in `board.py` (square count asserted
constant); `cut_multiboard.py`, `run_multiboard_chain.sh`. Two lanes side by side at 8 workers (RAM-bound, ~20 GB
per run); the RAM guard never fired.

**Ladder (21 arms, 5k steps unless stated, onto DICT_6X6_250; ID = 10k val, pose = 1000-frame correct rate):**
bottleneck frozen in any form 57-61% (head, +gate, +d3, re-init, head 3x, head 20k 61.8%); BitFit 73%;
post-bottleneck (gate+decoder+heads) 72%; bottleneck 0.01x 69-72% (incl. whole-model LLRD 0.3); 0.1x 90.5%
(96.0% at 20k); 0.3x 93.5%; 1x 97.1% (corner-head LR irrelevant); +e4 97.8%; **3x 98.2% / 86% poses, 95% by
step 1,000**; whole model 1x 98.4%, 3x 98.9% / 88.3% (99% in-loop at 3.5k; the only arm that moved p95,
0.673 -> 0.682); **bottleneck 3x for 20k: 99.1% / 87.6%, p95 0.672** (99% in-loop at 17.5k). Trained-board
reference 99.53% / 89.1%. Wrong-accepted poses 0.0-0.5% for every arm (zero-shot: 22%). Localisation never
moved except for the full-3x arm. VRAM at B16: head 2.1 GB, post-bottleneck 2.7, head+bottleneck 3.5, +e4 3.6,
BitFit 4.6 (gradient depth, not parameter count), whole 4.9; B4 divides by ~4. GPU: 48 ms/step for the minimal
recipe on the 5090 -> ~1 GPU-minute to 95%, ~14 to 99%; wall time is the CPU generator (76 samples/s/lane).

**Multi-board (29).** 15 families, 50k budget: ID flat at the 25% lattice-symmetry floor (uniform over families and
octaves, loss creeping) -> stopped at 25k by a pre-set rule; checkpoint kept. 3 families, 30k: same floor until
15k, then a transition to 82.6% (far octave 44%, mid/near 95/92); original board 80.6% ID / 62.9% poses (release
99.5 / 89.1); zero-shot on DICT_6X6_250 17.7% ID with 19% wrong-accepted poses (single-board: 13.4% / 22%).
Fine-tunes onto DICT_6X6_250 from the 3-family base are WORSE than from the release model at every trainable
set (head 51 vs 57, bottleneck 0.1x 72 vs 90, 1x 87 vs 97, post-bottleneck 67 vs 72); from the 15-family base
post-bottleneck 30%, head 28%. Kaelin's post-bottleneck theory fails on both bases. Caveat recorded: the 15-family
stop may have been premature given the 3-family transition at 15k; a 100k-step 15-family base is untested.

**Side findings.** Corner-count generality verified end to end on 9/25/36-corner boards (retarget path and the
finetune path via `reinit: [cls.*]`); robustness_sweep and finetune_cost hard-coded 16 (fixed); PyTorch's fused
LayerNorm backward raises on a bias-only-trainable LayerNorm (BitFit includes LN scales for that reason);
`dcc/pipeline.py:291` recover has no guard on its homogeneous divide (surfaced in 27, not fixed).

**Addendum 2026-10-09 (30_REPORT_transfer_finetune).** Report compilation: headline tables + all figures in one
directory, plus a measurement-noise-vs-range figure (`tools/plot_pose_error_range.py`), the end-to-end transfer
figure with the fine-tuned arms as bars, and the 20-factor robustness sweep of the 20k fine-tune on identical
frames: recall 89.8 vs 89.6%, localisation equal, ID 91.1 vs 93.9% mean -- equal in benign regimes, weaker at the
hardest darkness (24 vs 69%) and noise (52 vs 72%) steps. Clean-condition identity transfers; identity under heavy
degradation only partly.

**Addendum 2026-10-09, later (separate panels).** Kaelin: "make each separable figure separate so I can choose which
ones to include." Every panel of the report composites is now its own file (`30_REPORT_transfer_finetune/figures_separate/`,
49 files; `regen_figures.sh` there rebuilds them and cmp-checks the composites): `dcc/viz.py: save_panels` crops the
identical render per axes with neighbours and figure-level text hidden (axis-label extents unioned back in --
`Axes.get_tightbbox` collapses them to 1 px), and the four plot tools take `--separate`, each panel first getting the
legend / y-label it shared in the composite (grid panels: legend below the axes, since `best` covered data on one).
Found while splitting: `plot_four_way.py`'s darkness override (found+ID, designed for the RECALL grid) applied to every
`--panel-metric`, so the ID and localisation grids' darkness panel showed found+ID (on a px axis for localisation).
Fixed; `fig2a`, `fig6b`, `fig6c` and 27's `fourway_PANEL_{id_acc,err_median}transfer_*` regenerated, recall grids
byte-identical, no table number affected. `05_comparison/figures_L_demo/fourway_PANEL_{id_acc_C,err_median_D}_proposed.png`
are layout demos with the same flaw, left as they were.

**Addendum 2026-10-09, evening (12_loss).** Kaelin asked why focal on the class head helps when the head is at H/4.
Measured the candidate mechanism (`12_loss/cls_form_gradient_share.{json,md}`, `tools/cls_form_gradient_share.py`):
at fixed weights on 128 stratified SynthVal frames, the class head's share of the shared-trunk gradient is 50% in the
all-BCE 882k arm and 33-36% in the focal-class arms (attention blocks: 1.29 : 1 -> 0.70-0.74 : 1); a focal-trained
class head scored with BCE carries a class loss 14-20x the heatmap loss (easy background cells left un-crushed).
Consistent with "cls=focal is a localisation lever" acting through the trunk's gradient budget; a snapshot, not a
proof, and the identity cost of focal remains unexplained.
