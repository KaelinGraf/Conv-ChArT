# How the Kalman filter gets its measurement covariance from Conv-ChArT

Written 2026-10-10 for the live ROS 2 pipeline (`Conv-ChArT-Wireless-Inference`, `convchart_ros` node) and
whoever writes the filter that consumes `inference_result`. Every number comes from
`kalman_R_REL882.json` in this directory (882k release model, 1000-frame B1 benchmark, 893 accepted solves).

## 1. The decision, in one paragraph

The inference node publishes, with every solved pose, a **6x6 measurement-noise covariance R** that depends
only on the board's range. It is the measured error of the released model as a power law in range, not the
network's own per-frame covariance (`pose_cov`), which is not usable as R: on this benchmark its NEES has
mean 11.2 and median 2.75 against 6 expected, i.e. conservative on most frames and badly over-confident on a tail and must never reach a filter. The filter uses the published R as its measurement noise,
skips the update on the three conditions in Section 4, and gates innovations with a chi-square test because
the error is heavy-tailed whatever R is used.

## 2. What is wrong in the node today (`convchart_ros/convchart.py`, main at fcacd45)

`_package_result` fills `pose.covariance` from `result["pose_cov"]` through `_pack_pose`, permuted into ROS
order, and sets `covariance_valid = True` whenever that analytic matrix exists. That matrix is
`pose_covariance()` from the pipeline port: `(J^T R^-1 J)^-1` with the refiner's `sigma_px` as R. Both the
pipeline spec (`deploy/PIPELINE_SPEC.md` section 9: "do not port `pose_covariance`; it is not a filter
input") and the measurement say it is not usable as R. So today the filter receives a covariance that is too
mis-scaled frame to frame: larger than the real error on most frames, far too small on the tail.

## 3. The measurement model to publish

Error 6-vector of the board-in-camera pose, camera frame: rotation is the small-rotation vector of
`R_est R_true^T` (a rotation about the camera axes, which is the convention `PoseWithCovariance` uses for its
rotation block), translation is `t_est - t_true`. Robust (MAD) sigma per component follows

    sigma_i(z) = a_i * z^k_i,    z = tvec_z / square_length    (range along the optical axis, in board squares)

| component | a_i (sigma at z = 1) | k_i | unit of a_i |
|---|---|---|---|
| rot_x | 1.1196e-4 | 0.9674 | rad |
| rot_y | 1.5808e-4 | 0.8074 | rad |
| rot_z | 3.8022e-5 | 1.0171 | rad |
| t_x | 5.2027e-5 | 1.5016 | board squares |
| t_y | 4.6866e-5 | 1.4944 | board squares |
| t_z | 1.0236e-4 | 1.7337 | board squares |

- `R = diag(sigma_i^2)`. Off-diagonals among accepted solves are weak (|rho| <= 0.23), so diagonal is right.
- Translation sigmas are in board squares: multiply by `square_length_m` to get metres. Rotation is already
  in radians and does not scale.
- Fitted for z in **[3.88, 70.67]** squares (apparent square size 128 px down to about 10 px at the 640x480
  input). Outside that, clamp z to the nearest bound and set `covariance_valid = False`: the matrix is still
  a sane number, the flag says it is an extrapolation.
- Fit quality: median |error| / sigma(z) is within 0.83 to 1.24 of 1.0 in every range octave and component.
  A single constant R is 2.0 to 7.4x over-confident at the far end and 2.0 to 3.6x over-cautious at the near
  end, which is why it is not used.

**ROS ordering.** `geometry_msgs/PoseWithCovariance.covariance` is a row-major 6x6 over
`(x, y, z, rot_x, rot_y, rot_z)`. The diagonal to publish is therefore

    [sigma_tx^2, sigma_ty^2, sigma_tz^2, sigma_rx^2, sigma_ry^2, sigma_rz^2]   (translation first)

Worked examples (from the constants above):

| board square | tvec_z | z (squares) | sigma rotation x / y / z (deg) | sigma translation x / y / z (mm) |
|---|---|---|---|---|
| 40 mm | 0.56 m | 14.0 | 0.082 / 0.076 / 0.032 | 0.11 / 0.10 / 0.40 |
| 24 mm (the printed 120 mm board) | 0.56 m | 23.3 | 0.135 / 0.115 / 0.054 | 0.14 / 0.12 / 0.58 |
| 24 mm | 1.20 m | 50.0 | 0.282 / 0.213 / 0.116 | 0.44 / 0.39 / 2.17 |

Depth is the weak axis by 4 to 15x, as expected for a planar target.

## 4. The filter's side: the contract on `inference_result`

Per message (`convchart_interfaces/RosInferenceResult`):

1. `reason` non-empty: the pipeline refused (too few identified corners, collinear, vacuous fit without
   corroboration, no intrinsics, PnP failure). **No update.** The pose fields are defaults.
2. `ambiguous` true: the two planar (IPPE) solutions have reprojection errors within a factor of 1.5, and the
   true posterior is bimodal. Either **skip the update**, or compute the innovation of both `pose` and
   `pose_alt` against the predicted state and accept the one that passes the gate below, never both. On the
   benchmark the flag fires on 4.6% of solved frames.
3. `rms > 0.15` px (PnP reprojection residual, native pixels): **skip**. It rejects 15.8% of accepted frames
   and removes 51% of the frames whose error exceeds 5 sigma; it is the one per-frame signal beyond range
   that correlates with the error (rank correlation 0.47 to 0.61).
4. `covariance_valid` false: the pose is outside the fitted range (or no solution); use the fallback R or skip.
5. Otherwise update with `pose.covariance` as R, then apply a **chi-square innovation gate** at the 95% bound
   for 6 degrees of freedom (12.59). Even under the range-scaled R, 13.5% of accepted frames exceed 5 sigma
   on some component; the gate is what turns that tail into skipped updates.

Frame conventions: the published pose is the board in the camera frame (`X_cam = R X_board + t`); the
covariance is for that pose. If the filter tracks the camera or robot in the board frame, invert the pose first
and propagate the covariance through the inversion (the translation error then picks up a rotation-error term
`[R^T t]_x drot`); the raw diagonal above is not the covariance of the inverted pose.

## 5. What to change in `Conv-ChArT-Wireless-Inference`

Pure numpy; nothing in the Docker images, the message definition or the Pi side changes.

**(a) New module `src/ros/src/convchart_ros/convchart_ros/measurement_noise.py`** (full text in `paste5.txt`
beside this session's other pastes, and reproduced here):

```python
"""Measurement-noise covariance R for the Kalman filter: the 882k release model's measured pose error as a
function of range. Source: Conv-ChArT paper/results_rev6/26_pose_error_variance/kalman_R_REL882.json
(1000-frame B1 benchmark, 893 accepted solves; robust sigma per apparent-scale octave fitted as a power law
in the range z = tvec_z / square_length, in board squares). The pipeline's per-frame analytic pose_cov is NOT
a filter input (NEES mean 11.2, median 2.75, against 6); this replaces it on the wire."""
import numpy as np

# robust sigma at z = 1 square, order (rot_x, rot_y, rot_z [rad], t_x, t_y, t_z [squares]), and the exponents
_A = np.array([1.1196e-04, 1.5808e-04, 3.8022e-05, 5.2027e-05, 4.6866e-05, 1.0236e-04])
_K = np.array([0.9674, 0.8074, 1.0171, 1.5016, 1.4944, 1.7337])
Z_FITTED = (3.88, 70.67)      # board squares; outside this the law is extrapolated and the flag goes False


def measurement_covariance(tvec_z: float, square_length_m: float) -> tuple[np.ndarray, bool]:
    """(6x6 diagonal covariance in PoseWithCovariance order (x, y, z, rot_x, rot_y, rot_z), in_fitted_range).
    Translation in the units of square_length_m (metres for a metric board), rotation in rad^2. z is clamped
    to the fitted range so the matrix is always usable; the flag says whether clamping happened."""
    z = float(tvec_z) / float(square_length_m)
    zc = min(max(z, Z_FITTED[0]), Z_FITTED[1])
    sigma = _A * zc ** _K
    sigma[3:] *= square_length_m                      # squares -> metres
    return np.diag(np.r_[sigma[3:], sigma[:3]] ** 2), zc == z
```

**(b) `convchart_ros/convchart.py`** (replacement for the pack path in `paste6.txt`):

- `from .measurement_noise import measurement_covariance`.
- `_image_callback`: pass the board's square length to the packer:
  `self._package_result(msg, result, self._inference_pipeline.board.square_length_m or 1.0)`.
- `_package_result(img_msg, result, square_length_m)`: call
  `_pack_pose(result["rvec"], result["tvec"], square_length_m)` and the same for the `_alt` fields. Stop
  passing `pose_cov` / `pose_cov_alt` anywhere.
- `_pack_pose(rvec, tvec, square_length_m)`: no solution keeps the default pose and `False`; otherwise fill the
  pose as now, then `cov, in_range = measurement_covariance(float(np.ravel(tvec)[2]), square_length_m)`,
  `pose.covariance = cov.ravel().tolist()`, return `in_range`.
- Delete `ROS_COV_ORDER` and the permutation; the module returns ROS order directly.
- Leave `pose_covariance()` in `inference.py` alone: `tests/test_parity.py` checks it against the reference
  pipeline, and `InferenceResult` keeps the field. It simply no longer reaches the wire.

**(c) `cfg/cfg.yaml`**: `BOARD.square_length_m` must be the printed square edge in metres (0.024 for the
120 mm print from `tools/print_board.py`). It is `null` today, which makes both the pose and the covariance
come out in board squares. The end-to-end test already sets it to 0.04 for its synthetic board.

**(d) Tests** (`src/ros/src/convchart_ros/test/`):

- `test_result_packaging.py`: the `pack` fixture must pass a square length, e.g.
  `lambda img, res: node._package_result(img, res, 0.04)` (its `TVEC` is already written as 40 mm squares at
  0.56 m). Replace `test_covariance_is_row_major_in_ros_order` with a test that recomputes
  `a_i * (tvec_z / 0.04)^k_i` from the constants written into the test, scales the translation terms by 0.04,
  squares them, and asserts the packed diagonal equals them in `(x, y, z, rot_x, rot_y, rot_z)` order with
  every off-diagonal zero. Replace `test_missing_covariance_is_flagged_not_faked` (there is no "missing"
  covariance any more) with an out-of-range case: `tvec_z = 0.05` m at 0.04 m squares is z = 1.25, so
  `covariance_valid` must be False and the matrix must equal the one at z = 3.88. Keep the refusal, header,
  quaternion and alternative-solution tests as they are.
- `test_node_pose_output.py` needs no change: at its z = 14 the depth variance (1.58e-7) exceeds the lateral
  ones (1.20e-8, 9.36e-9) and the rot_z variance (3.10e-7) is below rot_x / rot_y (2.07e-6, 1.77e-6), which is
  exactly what `test_covariance_is_ros_ordered` asserts, and z = 14 is inside the fitted range so both
  validity flags stay True.
- Run them with the system Python that has ROS 2 Jazzy, after `colcon build` of `convchart_interfaces`,
  `convchart_qos` and `convchart_ros`; the node also imports `ros2_numpy`, which the Dockerfile installs with
  two patches (see `docker/pi/Dockerfile` lines 62 to 85) and which is absent from the bare system Python.

**(e) `readme.md` / `docs/`**: copy this file into the repo's `docs/` so the filter author finds the contract
next to the node.

## 6. Caveats to carry into the thesis and the filter

- The constants are the **synthetic benchmark floor**: B1 frames at the 640x480 input with sensor resolution
  equal to the input and no object-cutout occluders. A real OV2311 at 1600x1200 refines on native crops, so the
  pixel error in sensor pixels is not the benchmark's; re-measure with `tools/eval_pose_ours.py` on a
  calibrated capture when one exists, and expect real variance to be larger, not smaller.
- Sample variance is 9 to 14x the robust variance on every axis: the published R is the core of the
  distribution, and the innovation gate is not optional.
- The range used is the measured `tvec_z`, which is accurate enough for this purpose (its own relative error is
  under 1% at every octave).
