# Conv-ChArT robustness sweeps — rev-4, ckpt_0150000

**Our model only.** No Deep ChArUco, no classical OpenCV — this directory answers
"is *our* detector robust", not "is it better than X". Every factor carries a
**with-refiner** and a **coarse-only** curve.

## Provenance

| | |
|---|---|
| Detector | `runs/rev640_baseline/ckpt_0150000.pt` — **step 150,000**, the baseline |
| Refiner | `runs/refiner_rev4_ft/ckpt_0013000.pt` — **step 13,000**, rev-4 fine-tuned |
| Config | `configs/rev640.yaml` (rev-4: `differencing_autoexposure: true`, `differencing_p: 0.10`) |
| Backgrounds | COCO train2017, 118,287 images |
| Match radius | 4.0 px (MT-03 capture radius) |
| Seed | 20260728; per-sample `default_rng([seed, step_index, i])` |
| Tools | `tools/robustness_sweep.py` → JSON, `tools/plot_robustness.py` → PNG |

`n` = 100 frames/step for `distance`/`tilt`/`rotation`/`sensor_noise_K`/
`brightness`/`contrast` (~1,400 corners per point), 60 for the rest (~850).

## Method — and why it is not `tools/factor_sweep.py`

`factor_sweep.py` holds ONE base scene fixed and varies one parameter, for a
visually-comparable filmstrip. That is **n=1 frame, ≤16 corners per point** —
binomial SE ≈ 2.5 pp at our accuracy, which cannot resolve 99% from 96%. It is a
qualitative tool and is still wanted for what it does.

These sweeps instead pin the swept factor and **randomise everything else**
(background, pose, all other augmentations), N frames per step. Each point is a
marginal accuracy estimate with a real confidence interval.

- Geometry (`distance`, `tilt`, `rotation`) is pinned exactly via
  `generate_sample`'s `components` override.
- Range-valued photometric knobs are pinned by collapsing `[lo,hi] → [v,v]` and
  forcing that augmentation's probability to 1.0.
- The two blur families are applied post-generation, because their kernel size is
  drawn from a set rather than a range and cannot be pinned through config alone.
- `tilt` and `rotation` hold `s = 40 px` so they measure foreshortening/orientation
  rather than distance in disguise.
- `rotation` is the **control**: it should be flat. It is.

Raw per-corner errors are stored in every JSON so figures can be restyled, or new
statistics computed, without re-running a sweep.

## The headline result, and a problem it exposed

**The refiner improves typical localisation ~3× and makes the tail ~3× worse.**
Pooled over the `distance` sweep (n = 10,667 matched corners):

| | median | p95 | >1 px | >2 px | >3 px |
|---|---|---|---|---|---|
| with refiner | **0.13–0.18 px** | 1.6–2.7 px | 8.62% | 5.47% | **3.35%** |
| coarse only | 0.43–0.45 px | 0.86–1.05 px | 3.95% | 0.59% | **0.15%** |

The gross-error rate is **22× higher with the refiner than without it**, and it
costs **2–3 pp of detection recall and ID accuracy** on every factor, because
refinements that overshoot push a corner outside the 4 px match radius and, when
they don't, can still read the wrong cell of the H/4 class map.

**Root cause — a train/deploy distribution mismatch, verified in source:**

- `dcc/refiner_data.py:202` — training jitter is `j ~ U(−4, +4)` px per axis, so the
  refiner is trained expecting the true corner **anywhere within ±3.9375 px** of the
  crop centre. That is deliberate: it is the MT-03 4 px capture radius.
- In deployment the crop is centred on the **detector's own peak**, whose error is
  0.44 px median / 0.9 px p95 (the coarse column above). The refiner therefore
  almost never sees the distribution it was trained on.
- `dcc/pipeline.py:284` — `xy_sensor[kept_mask] = centres_sensor + (u_star − 31.5)/8`
  is applied **unconditionally**. There is no guard: a wild refinement is accepted
  exactly as readily as a good one.

This also explains why fine-tuning the refiner on rev-4 moved p95 (2.742 → 2.386 px)
but not the median (0.1656 → 0.1662 px) — the mismatch is in the **jitter
distribution**, not the photometrics, so photometric adaptation cannot fix it.

**Not yet measured**: whether refinement helps or hurts *as a function of the coarse
peak's own error*. That is the decisive analysis and it needs paired per-corner
(coarse, refined) errors, which these runs do not save. Until it is run, treat the
guard threshold as unquantified — do not pick one by eye.

## Files

- `<factor>.json` — per-step metrics for both arms + raw per-corner errors + provenance
- `figures/robustness_<factor>.png` — 3 panels (M-01 localisation, M-02 recall, M-04 ID)

## Factors

Geometry: `distance` (s = 12→128 px), `tilt` (0→60°), `rotation` (0→180°, control).
Sensor/exposure: `sensor_noise_K` (30→1 e⁻/DN), `brightness` (−0.9→+0.35),
`contrast` (0.6→1.4), `ink_contrast` (1.0→1.6).
Optics: `defocus_blur` (0→9 px), `motion_blur` (0→9 px), `vignette` (0→0.25),
`specular` (0→220 DN), `droplets` (0→6).
Differencing regime: `diff_ambient` (0.05→0.95), `diff_ratio` (0.9→0.15),
`diff_ghosting` (0→3 px inter-frame shift).

`exposure_clipping` was deliberately **dropped**: rev-4's auto-exposure abolished the
phenomenon, so the axis no longer discriminates.
