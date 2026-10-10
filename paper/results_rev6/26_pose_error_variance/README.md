# 26 — Pose error variance of the 882k release model (Kalman measurement noise R)

Kaelin, 2026-10-08: "We are using the 882k model for inference - I need the variance of the error
for a kalman filter."

**What this is.** The ENSEMBLE error statistics of the poses `dcc.pipeline.detect()` returns, over
the 1000-frame B1 pose benchmark, for the 882k release detector + shared refiner. This is the
constant measurement-noise covariance a Kalman filter can use. It is NOT the parked per-frame
`pose_cov` / `sigma_px` (CLAUDE.md: "Uncertainty calibration is PARKED"; `pose_cov` is ~6x
over-confident, NEES 11.2 against the chi-square-6 expectation of 6.0 on this very run). Those
remain forbidden as R. An ensemble variance over frames is a different object and is what was asked.

## Files

| file | what |
|---|---|
| `pose_REL882_B1.json` | `tools/eval_pose_ours.py` output; `error_cov` block holds the statistics below |
| `pose_REL882_B1_per_image.jsonl` | per solved frame: `drot_rad[3]`, `dt_sq[3]` (camera frame), `s_px`, `f_px`, `n_used`, `rms_px`, `ambiguous` |
| `kalman_R_REL882.json` | the R values extracted for the filter, with notes |
| `KALMAN_COVARIANCE_HOWTO.md` | how the live ROS node publishes R and what the filter must do with it (2026-10-10) |
| `logs/gen_eval_pose_rev6_b1.log` | pose-set regeneration (the gitignored set had to be rebuilt; it is seed-deterministic) |

**Reproduction gate passed.** The regenerated `eval_pose_rev6_b1` reproduces the banked
`21_pose_REL882_B1.json` exactly: 936/1000 solved, 64 refused (too_few 60, collinear 2,
pnp_solver_failed 2), 43 ambiguous, rotation median 0.1318 / mean 0.5368 / p95 1.1737 deg,
translation median 0.0104 / mean 0.0376 / p95 0.1282 squares. So the statistics below are on the
same frames as every other release number.

## Convention

Error 6-vector `e = [drot_x, drot_y, drot_z, dt_x, dt_y, dt_z]`, **camera frame**:
`R_err = R_est @ R_gt.T`, `drot = Rodrigues(R_err)` (rad); `dt = t_est - t_gt`. `detect()` returns
the board-in-camera pose (`X_cam = R X_board + t`). Translation is in **board squares** because the
benchmark fixes `square_length_m = 1`: multiply translation variances by `square_length_m^2`.
Rotation statistics do not scale.

Computed on the **accepted** set only — unambiguous solves, n = 893 — because the filter contract
(`deploy/PIPELINE_SPEC.md` §9) skips the update on refusal and when `ambiguous` is true. Including
the 43 ambiguous solves inflates every std ~10x and introduces |rho| ~ 0.9 correlations (the
IPPE two-fold flip); the flag is doing its job and must stay wired to the filter.

## The numbers (n = 893)

| component | sample std | sample variance | robust std (1.4826·MAD) | robust variance |
|---|---|---|---|---|
| drot_x | 0.276 deg (4.81e-3 rad) | 2.31e-5 rad² | 0.091 deg (1.59e-3 rad) | 2.54e-6 rad² |
| drot_y | 0.290 deg (5.06e-3 rad) | 2.56e-5 rad² | 0.088 deg (1.54e-3 rad) | 2.38e-6 rad² |
| drot_z | 0.093 deg (1.63e-3 rad) | 2.65e-6 rad² | 0.037 deg (6.4e-4 rad) | 4.09e-7 rad² |
| dt_x | 0.01271 sq | 1.62e-4 sq² | 0.00291 sq | 8.45e-6 sq² |
| dt_y | 0.00857 sq | 7.34e-5 sq² | 0.00272 sq | 7.40e-6 sq² |
| dt_z | 0.04332 sq | 1.88e-3 sq² | 0.01158 sq | 1.34e-4 sq² |

Means are ≤ 3e-4 rad and ≤ 3e-3 sq: no bias worth modelling.

**Heavy tails, stated plainly.** The sample variance is 9–14x the robust variance on every axis.
33.6% of accepted solves have at least one component beyond 3 robust sigma and 20.9% beyond 5.
The error is not Gaussian; it is a tight core plus a tail. Two defensible choices:

- `R_robust` (the core) **plus an innovation gate** (Mahalanobis / chi-square test on the
  residual) that rejects tail measurements. This is the recommended pairing: the gate turns the
  tail into skipped updates, which the pipeline's own refusals already are.
- `R_sample` with no gate: ~10x less trusting of every fix, so the filter is slow, but never
  over-trusts a tail sample.

**Off-diagonals are weak** on the accepted set (|rho| ≤ 0.23; the largest are dt_x–dt_z 0.23 and
drot_y–dt_z −0.20). A diagonal R is defensible. The full 6x6 sample covariance is in the JSON.

## Range dependence — a constant R is a compromise

Pixel error is roughly constant, so pose error grows with range `z = f/s` (focal length over
apparent square size, i.e. range in board squares). Median |err| scales as z^0.9 (rotation),
z^1.3 (lateral translation), z^1.7 (depth) over z = 6–41 squares. Per apparent-scale octave
(robust std; `s` is the square size in input pixels, so 12–16 is far and 64–128 is near):

| s (px) | n | drot_x,y,z (deg) | dt_x,y,z (sq) |
|---|---|---|---|
| 12–16 | 65 | 0.267 / 0.172 / 0.106 | 0.0181 / 0.0138 / 0.0724 |
| 16–32 | 258 | 0.151 / 0.163 / 0.067 | 0.0071 / 0.0071 / 0.0371 |
| 32–64 | 276 | 0.094 / 0.081 / 0.033 | 0.0028 / 0.0028 / 0.0112 |
| 64–128 | 294 | 0.043 / 0.043 / 0.017 | 0.0011 / 0.0009 / 0.0032 |

If the filter knows the range (it does: `|tvec|`), scaling R by octave is ~40x better matched
at the near end than the pooled constant. Sample-variance versions of the same table are in
`pose_REL882_B1.json` → `error_cov.unambiguous_by_s_px_octave`.

## Caveats

- Synthetic benchmark (B1 photometrics; object-cutout occlusion absent, measured on its own
  robustness axis). Real-camera variance will be larger; this is the floor, not the ceiling.
- Depth (dt_z) is the weakest axis by 4–15x, as expected for a planar target.
- If the filter tracks the camera/robot in the BOARD frame, invert the pose first; the
  translation error then picks up a rotation-error coupling (`dt' ≈ -R^T dt + [R^T t]_x drot`)
  and is not the raw dt above.

## Constant or per-frame R? (Kaelin's follow-up, 2026-10-08) — per-frame, but driven by RANGE, not by the network

Measured on the same 893 accepted solves (`kalman_R_REL882.json` → `range_scaled_R`):

- **A constant R is mis-scaled by up to 7x depending on where the robot is.** Normalising the
  errors by the pooled robust sigma, median |e|/sigma is 2.0–7.4 at the far octave (s = 12–16 px:
  the filter over-trusts every fix) and 0.28–0.49 at the near octave (s = 64–128 px: it ignores
  fixes that are 3x better than it thinks). Both ends are wrong in the direction that hurts.
- **A range-scaled R fixes it with two numbers per axis.** Robust sigma follows a power law in the
  range z = f/s (= tvec_z / square_length, in board squares): sigma_i(z) = a_i z^k_i with
  k = 0.97 / 0.81 / 1.02 (rotation) and 1.50 / 1.49 / 1.73 (translation x, y, depth). Under that
  model median |e|/sigma is within 0.83–1.24 of 1.0 in every octave and component. Range is
  observable from the measurement itself (tvec_z) or the predicted state; the fitted span is
  z = 3.9–70.7 squares.
- **The network's own per-frame covariance is not the answer.** `pose_cov` has NEES mean 11.2
  (median 2.75) against the chi-square-6 expectation of 6: conservative on most frames and badly
  over-confident on a tail, i.e. it does not even rank frames reliably. The project tried
  calibrating it and parked the attempt (`paper/results_rev5/10_uncertainty/NOTES_what_didnt_work.md`;
  CLAUDE.md pin). Leave it out of the filter.
- **The PnP reprojection residual is the one useful per-frame signal beyond range.** `rms_px`
  rank-correlates 0.47–0.61 with the error (0.66–0.80 as rms·z), and it separates the tail: tail
  frames have median rms 0.152 px against 0.080 for the core. Gating at rms_px > 0.15 rejects
  15.8% of accepted frames and removes 51% of the >5-sigma tail. `n_used` adds little once range
  is known (rank corr 0.06–0.23).
- **The tail survives every model.** 13.5% of frames still exceed 5 sigma under the range-scaled R
  (20.9% under constant R; a Gaussian would give ~0.0001%). The filter needs a chi-square
  innovation gate regardless of which R it uses.

Recommendation: R_k = diag(a² z_k^2k) from the fit above, skip updates on refusal / ambiguous /
rms_px > 0.15, and gate the innovation at the 95% chi-square-6 bound (12.59). The paste file
`paste2.txt` in the session scratchpad holds this as a function.
