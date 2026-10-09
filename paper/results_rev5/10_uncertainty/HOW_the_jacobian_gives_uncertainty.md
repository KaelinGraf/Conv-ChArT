# How the PnP Jacobian yields a pose covariance

**Kaelin, 2026-07-29: NOT FOR THE CONFERENCE TALK.** *"its going over my head too much
to include in the conference talk in case i get asked about it in qna."* Recorded here
for the paper and for a later conversation. Do not put this on a slide, and do not put
`pose_cov` in the deck at all — the honest version needs the over-confidence caveat, and
that is not a 4-minute topic.

Theory, not data — this file is rev-agnostic. It sits beside
`NOTES_what_didnt_work.md`, which records the measurements and the parked status.

---

## The one-line idea

The Jacobian does not contain uncertainty. It contains **sensitivity**, and uncertainty
is sensitivity *inverted*.

## Read forwards, then backwards

`cv2.projectPoints` returns J = d(u,v)/d(rvec, tvec), a 2N x 6 matrix
(`dcc/pipeline.py:372`). Forwards it answers: *nudge the pose by delta, and the 16
corners move by J·delta pixels.*

Backwards: you measured the corners to a precision of sigma pixels, so **any pose change
that moves the corners by less than sigma is invisible to you** — it hides inside the
noise. The pose uncertainty along a direction is however far you must move before the
image notices.

A pose direction that barely moves a pixel cannot be pinned down. One that swings all 16
corners a long way is known precisely.

## From that to a covariance matrix

PnP minimises weighted reprojection error. Near the optimum, the residual after a small
pose perturbation is (r - J·delta), so the cost

    (r - J·delta)^T R^-1 (r - J·delta)

is quadratic in delta. Its Hessian — the curvature of the cost bowl — is

    F = J^T R^-1 J           (the Fisher information)

Curvature and variance are reciprocal: a sharp bowl is a well-determined minimum. Hence

    SIGMA = (J^T R^-1 J)^-1

which is literally `JtRiJ = (J.T * (1.0/var)) @ J` then `np.linalg.inv(JtRiJ)` at
`dcc/pipeline.py:378-383`. By Cramer-Rao this is a LOWER BOUND on the achievable
covariance: the best any unbiased estimator could do given this geometry.

R is the per-corner measurement covariance, isotropic per corner, from `sigma_px`.
`sigma_px=None` degenerates to (J^T J)^-1 = SIGMA in units of "per 1 px^2 of measurement
noise"; scale by your own sigma^2.

## Why the structure comes out physically right

The measured "depth 3-5x worse than lateral" is not fitted — it falls out of J's columns.

- **Lateral translation**: move the board 1 cm sideways at range z and every corner
  shifts by f/z pixels. Large column in J.
- **Depth translation**: move it 1 cm along the optical axis and the board only CHANGES
  APPARENT SIZE, by roughly f·s/z^2 for board extent s. Much smaller column, and it
  shrinks quadratically with range.

Bigger column -> more information -> smaller variance.

Rotation is the same argument: in-plane rotation swings every corner tangentially
through a large arc, while out-of-plane tilt near fronto-parallel is second-order
(cos(tau) ~ 1 - tau^2/2, so the first derivative vanishes at tau=0). Weakly observed,
hence the measured 4x.

## The three reasons it is over-confident

All three are in `pose_covariance`'s docstring; all three push the same way.

1. **Local linearisation.** It describes the shape of ONE bowl. The IPPE planar
   ambiguity is a second bowl elsewhere in pose space and no single Gaussian sees it.
   Measured NEES on ambiguous fixes ~8,185 against an expectation of 6.
2. **Independence.** Residuals are assumed independent zero-mean Gaussian. A mis-IDed
   corner is not modelled at all, and the independence assumption is the one we measured
   failing — 16 corners behaving like ~2.7 independent ones, the 6x over-confidence.
3. **Scale inherits from R, squared.** sigma wrong by 4x makes SIGMA wrong by 16x. This
   is why calibrating sigma made the POSE NEES worse: it removed the error that had been
   accidentally cancelling the correlation error.

See `NOTES_what_didnt_work.md` for the measurements and the restart plan.
