# Uncertainty calibration — what was tried, what failed, and where to restart

**Status: PARKED, deliberately.** Kaelin, 2026-07-28: *"Stop patching the variance
stuff. Let's undo any damaging/misguiding work, make notes of what was tried and
didn't work, and pick it up with thorough analysis another day."*

That was the right call. The work below was iterating toward a target number under
deadline pressure, which is how a patch gets mistaken for an understanding. Nothing
here is wrong *mathematics*; the problem is that the last step was being tuned to
make a statistic come out, without a model of why it was off.

---

## What survives (measurements, keep)

- **`sigma_px`** — per-corner localisation uncertainty from the refiner's own peak
  spread, `soft_argmax(..., return_spread=True)`. Opt-in, back-compatible.
- **It is a genuinely good ORDERING signal.** Rank correlation with actual error
  **+0.633**; lowest-σ decile has median error **0.033 px**, highest **0.392 px**
  — **12× separation**, and 36× on p95. Selective prediction on σ cuts p95 error
  0.617 → 0.154 px at 50% coverage. This result stands and is reportable.
- **`pose_covariance()`** — Σ = (JᵀR⁻¹J)⁻¹ from `cv2.projectPoints`' analytic
  Jacobian. The structure is physically right: depth σ 3–5× worse than lateral,
  in-plane rotation 4× better constrained than out-of-plane tilt. Correct as
  geometry.
- **NEES/χ² machinery** in `tools/eval_pose_ours.py` — the right test, keep it.
- **The ambiguity flag is validated as a hard reject gate.** NEES on IPPE-ambiguous
  poses ≈ **8,185** (vs 6.0 expected). A single Gaussian cannot describe a bimodal
  posterior. Any filter must DROP `ambiguous=True` fixes, not down-weight them.
  This one is solid and actionable.

## What was tried and abandoned

**1. Isotonic RMS calibration of σ** (`tools/calibrate_uncertainty.py`, kept as a
record, NOT wired into the pipeline).

Fitted σ_raw → σ_cal on 300 frames so that within each bin σ_cal = RMS(err)/√2,
monotonicity enforced by pool-adjacent-violators. Validated on a disjoint seed:

| per-corner NEES (target 2.0, χ²₂) | mean | within 95% bound (ideal 95%) |
|---|---|---|
| raw σ | 8.055 | 91.9% |
| calibrated σ | **2.082** | **94.6%** |

**By its own metric this worked.** The map is strongly non-constant — 0.23× at the
easy end to 4.45× at the hard end, a 19× swing — which is why no single scale
factor could ever have fixed it.

**2. Feeding calibrated σ into R for the pose covariance. THIS IS WHERE IT BROKE.**

| pose NEES (target 6.0, χ²₆) | mean | median | within 95% |
|---|---|---|---|
| raw σ → R | 11.14 | 2.46 | 85.3% |
| **calibrated σ → R** | **36.44** | 9.78 | 54.7% |

Calibrating σ made the POSE covariance **worse** (11.1 → 36.4), while the corner
covariance became correct. Not a contradiction — it means:

- per-corner R is now right (corner NEES 2.08 ≈ 2.0), and
- **Σ = (JᵀR⁻¹J)⁻¹ is ~6× over-confident even with correct R.**

The Gauss-Newton covariance assumes the 16 corner errors are **independent**. They
are not — they share the board's appearance, the detector's systematic biases, and
the pose itself. Treating them as independent over-states the information, implying
an effective sample size of roughly 16/6 ≈ **2.7 independent corners**.

**The earlier, better-looking 11.14 was two errors partially cancelling**: raw σ
over-stated per-corner error on the easy majority, which accidentally compensated
for the correlation. Fixing σ removed one error and exposed the other. The
"improvement" from 36.4 back to 11.1 would be a step backwards in honesty.

**3. What I was about to do, and should not have.** Fit a scalar variance-inflation
factor on Σ to force NEES to 6. That is curve-fitting a symptom. It would have
produced a number that passes its own test while modelling nothing, and it would
have hidden the actual finding — that corner errors are correlated.

## Where to restart, with time to do it properly

The open question is **the correlation structure of corner errors**, not the scale.

1. Measure the empirical 2N×2N corner-error covariance directly (per-frame residual
   outer products over many frames). Is the correlation mostly a common
   translation? A scale/pose-like mode? Board-region dependent?
2. If it is low-rank (likely — a shared pose-like perturbation), model it as
   R = diag(σ²) + Bᵀ Λ B with a small B, rather than inflating a scalar.
3. Only then re-run pose NEES. The target is a covariance that is consistent
   *because the model is right*, not because a factor was fitted.
4. Separately: emit **per-axis σ_x, σ_y** rather than an isotropic scalar. The
   refiner's distribution already has both and a KF wants a 2×2 R; at high tilt the
   true corner uncertainty is genuinely anisotropic.
5. Also unresolved: `p_id` is badly calibrated (**ECE = 0.302**), systematically
   UNDER-confident — the [0.4, 0.5) bin is 97.9% correct. One-parameter temperature
   scaling would likely fix it, and it implies `tau_id = 0.5` is discarding correct
   identifications.

## State of the code as parked

- Calibration is **NOT** wired into `dcc/pipeline.py`. `detect()` returns **raw**
  `sigma_px`, documented in-file as an ordering signal that must not be used as R.
- `pose_cov` is still returned, with the measured 6× over-confidence stated in
  `pose_covariance`'s docstring. Not validated for filter use.
- `tools/calibrate_uncertainty.py` and its JSON are kept as a record of the fit.
  Do not wire them in without doing (1)–(3) above.
- `tests/test_pipeline.py` — 9 passed after the `pnp` arity change.
