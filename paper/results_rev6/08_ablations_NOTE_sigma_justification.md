# A-SIGMA1 — why it exists, and how to write it up

**Status 2026-07-29: queued, not yet run. Do not draw conclusions until 50k lands.**
Decision deferred by Kaelin: "wait till the sigma-1 results are in before discussing
further." This file records the justification only.

## What prompted it

A-CE (strict one-hot BCE, dense head) was expected to fail. It did not — it beat the
reference at matched steps:

| step | reference (Gaussian σ=2, focal β=4) | A-CE (one-hot, plain BCE) |
|---|---|---|
| 2,500 | m01 0.4420 / m04 33.37% | null / null |
| 5,000 | m01 0.4322 / m04 97.79% | m01 0.4039 / m04 97.42% |
| 7,500 | m01 0.4283 / m04 98.63% | **m01 0.4112 / m04 99.22%** |

Kaelin's hypothesis for why — "the gaussian encourages soft guessing/imprecision" —
was measured and **confirmed** (`08_ablations_peak_sharpness.json`, n=830 corners,
same frames both arms):

| arm | step | peak spread (RMS radius, px) | peak value | concentration |
|---|---|---|---|---|
| reference (Gaussian σ=2) | 50,000 | 1.0028 | 0.9836 | 0.2609 |
| A-CE (one-hot) | 9,000 | **0.2665** | 0.9215 | **0.9349** |

**73.4% narrower, 3.6× more concentrated** — and the one-hot arm achieves it at 9k
steps against a fully-trained 50k reference, so training time is not the explanation.
`concentration` = fraction of local probability mass on the peak pixel rather than
smeared over ±4 px: 93.5% vs 26.1%.

## The mechanism (this is the paper's explanation)

`focal()`'s positive set is `y == 1.0`, identical to `strict_bce`'s. The Gaussian's
graded values enter in exactly ONE place: `(1-y)^beta` on the NEGATIVE term. With
σ=2.0 and β=4, a background pixel one px off the corner has y≈0.8825, so its weight is
`(1-0.8825)^4 = 1.9e-4`. **The loss charges essentially nothing for smearing mass onto
the corner's neighbours.** It is not a training deficiency — the loss asks for a blurry
peak and gets one. A broader peak means a noisier argmax, which is what m01 measures.

Forgiveness profile `w(r) = (1 - exp(-r²/2σ²))^beta`, β=4:

| r (px) | σ=2.0 | σ=1.0 | σ=0.75 |
|---|---|---|---|
| 1 | 0.0002 | 0.024 | 0.107 |
| 2 | 0.024 | 0.559 | 0.895 |

## Why σ=1.0 specifically

The net already sharpens **2.6× beyond its target** (target intrinsic spread 2.6185 →
measured 1.0028). If that ratio scales, σ=1.0 (target spread 1.4113) predicts a
measured spread near **0.5 px** — half the current value, while KEEPING a graded target
(±3σ = ±3 px, 7×7 support). **The ratio holding is an extrapolation, not a
measurement — that is what the run tests.**

`configs/abl_sigma1.yaml`: ONE key vs `rev640.yaml` (`sigma_hm` 2.0 → 1.0), verified
programmatically. NOTE: the first attempt prepended the key and silently produced ZERO
effective difference, because the body already carried `sigma_hm: 2.0` later in the
file and the duplicate key won. Always edit the key in place, and always re-verify.

## The framing, if σ=1.0 wins (Kaelin's own words, 2026-07-29)

A **hyperparameter** finding, not an architectural deviation. Shape:

> We observed that BCE results in a meaningfully tighter spread of confidence in
> localisation (quote figures), which resulted in better localisation accuracy compared
> to baseline. To improve the confidence distribution of the baseline architecture, we
> tightened σ to 1, which resulted in … If this outperforms both the baseline and the
> BCE version, we defend the claim that the Gaussian loss is superior **subject to
> task-appropriate hyperparameter tuning**.

## Three caveats to check before writing that

1. **The claim needs σ=1.0 > A-CE, not merely > baseline.** If σ=1.0 lands *between*
   the two, the honest conclusion flips to "for this task the one-hot target is
   better", and that is what should be reported. Prior is genuinely uncertain: the
   extrapolation says ~0.5 px, which is halfway to A-CE's 0.27, not past it.
2. **Fairness gap.** `strict_bce` has neither the α focal modulation nor β, so the arms
   differ by more than the target. "You tuned yours and not theirs" is the objection.
   The fourth cell that closes it is **`configs/abl_beta0.yaml`** (α retained, β=0 —
   grading removed, imbalance fix kept), already built and one-key verified. Run it if
   a slot allows to make the table a complete 2×2.
3. **Watch far range.** A tighter Gaussian gives fewer high-`y` pixels per corner, so
   the positive signal is weaker and small-scale learning may slow. No overlap problem
   (at s=12 corners are ~12 px apart vs ±3 px support). Report `m04 far (12-16)`
   alongside the aggregate so a far-range regression cannot hide in a good average.

## Decision table to produce at 50k

reference / conv-only / A-CE / A-SIGMA1, matched 50k, one row each:
`m01 median · m04 · m04 far(12-16) · peak spread · concentration` — mechanism and
outcome side by side, so the write-up can quote straight from it.
Probe: `scratchpad/peak_sharpness.py` (re-point ARMS at the four 50k checkpoints).

## If it wins and becomes the new baseline

A full 160k retrain is not affordable before the deadline. Cheaper path: **fine-tune
`runs/rev640_160k_rev6/ckpt_0160000.pt` under σ=1.0 for ~10k steps** (~40 min at ~140
samples/s) — the representation is learned, what changes is the head's output
distribution. Verify with the sharpness probe: if the fine-tuned spread matches
A-SIGMA1's from-scratch value, the fine-tune captured it; if not, report the 50k
ablation and state that the production model was not retrained.
