# Conv-ChArT — facts of record for the thesis report

Compiled 2026-10-09 for Kaelin's honours report. **Everything here is what is true now**: every number was
read from the named file, config, checkpoint or code on the day of writing, and the release models are the
only source of headline numbers (`CLAUDE.md` pin of 2026-08-05). Where an older document, table or the
paper itself disagrees with the artefacts, that is listed in Section 20 rather than silently corrected.

Conventions: `px` are input pixels of the 640x480 frame; `s` is the board square's apparent size in px;
`pp` is percentage points; "full val" is the fixed 10,000-sample validation set; "B1" is the corrected
1000-frame pose benchmark `eval_pose_rev6_b1`. Paths are relative to `dense deep charuco/`; results live
under `paper/results_rev6/` (abbreviated `R6/`).

---

## 1. What the system is, and what it is for

- **Task**: detect, identify and sub-pixel-localise the 16 inner corners of a 5x5 ChArUco board in one
  640x480 grayscale frame, then solve the board's pose or **refuse**. Output signature (paper eq. 1):
  a set of (position, identity-or-none) pairs plus a pose or an explicit refusal.
- **Target platform** (`docs/PROJECT_KNOWLEDGE.md` §1): a barrier-docking robot with a Jetson AGX Orin,
  an OV2311 global-shutter monochrome sensor and 940 nm active illumination. Deployment frames are
  **lit-minus-unlit differences**, so blur, glare, ghosting and heavy sensor noise are first-class
  conditions. Latency budget 66 ms per frame (15 Hz pose from 30 fps lit/dark pairs; `R6/15_cost`).
- **Three stages** (`docs/ARCHITECTURE.md`, `dcc/pipeline.py:detect`):
  1. detector network: corner heatmap at full resolution + 16-channel class map at H/4;
  2. refiner network: 24x24 crop around each coarse peak -> 64x64 logit map -> 5x5 soft-argmax;
  3. geometry: class-map readout at the coarse peaks, undistort, RANSAC lattice-homography gate,
     ID recovery, IPPE PnP; typed refusals.
- **Training is 100% synthetic**, generated on the fly; no real image is labelled (`README.md`, paper §4.1).

## 2. The board

Source: `configs/abl_c2_wh_clsfocal_lam2.yaml` (`board:`), `dcc/board.py`, `R6/31_print_board/README.md`.

| fact | value |
|---|---|
| geometry | 5x5 squares, 16 inner corners (`(nx-1)^2`); square boards only (asserted) |
| markers | 12 ArUco markers from `DICT_5X5_50`, ids 0-11 in raster order, in the white squares |
| marker/square ratio | 0.7 |
| pattern | cv2 4.10 `CharucoBoard`, non-legacy pattern: top-left square black |
| physical size | **not fixed** (`square_length_m: null`); the detector works on apparent size `s` |
| apparent-size envelope | trained `s` in [12, 128] px, log-uniform; measured bands: core 32-128, usable 16-128, degraded 10-160 (`R6/17_range/working_range.json`) |
| range formula | `z = f_px * S / s` (S = square edge in metres, f_px at the 640x480 input) |
| print-ready file | `R6/31_print_board/DICT_5X5_50_5x5_24mm.pdf` (A4, 24 mm squares, 120 mm edge); `tools/print_board.py --square-mm` for other sizes; verified by detecting all 12 markers and 16 corners on the rasterised PDF, edge 120.02 mm |
| working range at 120 mm (core band, OV2311 at 640x480) | 3 mm lens 0.08-0.34 m; 4 mm 0.10-0.40; 6 mm 0.15-0.60; 8 mm 0.20-0.80; 12 mm 0.30-1.20 m; range scales linearly with the edge |
| other dictionaries | same-family `_100/_250/_1000` variants share their first 50 markers with `_50`, so they are the same board; a different family (e.g. `DICT_6X6_250`) is a new board (`R6/27`) |
| other corner counts | 4x4 (9), 6x6 (25), 7x7 (36) boards verified end to end: config change plus a fresh class head; render resolution must divide by the square count (`R6/30_REPORT…/REPORT.md` §5) |

## 3. The release models

Source: `docs/TOOLING.md` (release table), checkpoints' own `cfg`, parameter counts recomputed from
`dcc/model.py` on 2026-10-09.

| tier | checkpoint | config | params | width_mult | XSA | heatmap loss | class loss | lambda_cls |
|---|---|---|---:|---|---|---|---|---|
| **882k (reference)** | `runs/rel_w882_c2_100k_rev6/ckpt_0100000.pt` | `configs/abl_c2_wh_clsfocal_lam2.yaml` | 882,402 | 0.5 | off | one-hot BCE | focal, Gaussian sigma_cls 1.0 | 2.0 |
| 502k | `runs/rel_w375_lam15_100k_rev6/ckpt_0100000.pt` | `configs/rel_w375_lam15.yaml` | 502,322 | 0.375 | off | one-hot BCE | one-hot BCE | 1.5 |
| 222k | `runs/rel_w25_lam2_100k_rev6/ckpt_0100000.pt` | `configs/rel_w25_lam2.yaml` | 222,138 | 0.25 | **on** | one-hot BCE | one-hot BCE | 2.0 |
| refiner (shared) | `runs/ref_s15_10k_rev6/ckpt_0010000.pt` | `refiner_train:` block of the same configs | 97,056 | — | — | focal, Gaussian sigma_ref 1.5 | — | — |

- The three detector configs differ in exactly four leaves: `width_mult`, `xsa`, `lambda_cls`,
  `loss_form_cls` (verified by flattening all three; everything else, generator included, is identical).
- The class head falls back to the heatmap form when `loss_form_cls` is absent (`dcc/losses.py:71`,
  `cls_form = loss_form_cls or loss_form`), which is why 222k and 502k are BCE on both heads.
- `e4_dilated: false`, `attend_div 8`, `attn_blocks 2`, `attn_heads 8`, `rope_lambda_min_cells 2.5`,
  `gates: true`, `sigma_hm 0.5`, `sigma_cls 1.0`, `tau_hm 0.3`, `tau_id 0.5`, `lattice_tol_px 3.0` in all three.
- Full pipelines (detector + refiner): 979,458 / 599,378 / 319,194 parameters.
- **Names**: the paper names tiers by parameter count. "Conv-ChArT" / "Conv-ChArT-FAST" in the conference
  material (`R6/15_cost`, `R6/03_robustness_FAST`) are the 882k-class and 222k-class architectures at
  ablation-era checkpoints (35k / 45k steps), **not** the release checkpoints.
- **The 4,698,034-parameter `rev640` model** (`runs/rev640_160k_rev6`, dilated e4 cascade, width 1.0,
  attention d = 256) is the pre-ablation predecessor. It appears in provenance and in explicitly historical
  comparisons (A1 conv-only, the loss decomposition, `R6/03_robustness`, `R6/06_tables`,
  `R6/07_introspection`) and must not be presented as the architecture.

### 3.1 Parameter distribution of the 882k reference (recomputed; matches paper Table 1 exactly)

| module | params | share |
|---|---:|---:|
| `blocks` (2 transformer blocks) | 396,544 | 44.9% |
| `e4` | 221,696 | 25.1% |
| `d3` | 110,720 | 12.5% |
| `e3` | 55,552 | 6.3% |
| `cls` head | 37,968 | 4.3% |
| `d2` | 27,712 | 3.1% |
| `e2` | 13,952 | 1.6% |
| `d1` | 6,944 | 0.8% |
| `gate3` | 6,209 | 0.7% |
| `e1` | 2,512 | 0.3% |
| `hm` head | 2,337 | 0.3% |
| `norm` (final LayerNorm) | 256 | 0.0% |
| **total** | **882,402** | |

The bottleneck is 45% of the parameters; the two heads together are under 5%.

## 4. Architecture, exactly as built at width 0.5 (882k)

Source: `dcc/model.py`, instantiated from the 882k config on 2026-10-09; `docs/ARCHITECTURE.md`;
`R6/10_receptive_field/receptive_field.md`; `R6/11_attention_gate/attention_gate.md`.

- **Input**: 1x480x640 grayscale. **Outputs**: heatmap logits 1x480x640, class logits 16x120x160 (H/4).
- **Encoder** (each stage two 3x3 Conv-BN-ReLU, 2x2 stride-2 pooling between stages):
  `e1` 1->16->16, `e2` 16->32->32, `e3` 32->64->64, `e4` 64->128->128 (plain convolutions; the dilated
  r=2,4 pair of the 4.7M design is removed). Channel widths at width 0.5: 16/32/64/128.
- **Attention bottleneck** on the H/8 map: 80x60 = 4,800 tokens, **d = 128**, 8 heads, **head_dim 16**,
  MLP 128->512->128 (ratio 4), pre-norm LayerNorm eps 1e-6, qkv bias on, no dropout, no class token, no
  patch embedding, no absolute position embedding. Two blocks.
- **Axial RoPE** (`dcc/model.py:AxialRoPE`): rotates Q and K only; per axis `n = head_dim // 4 = 4`
  frequencies (cos/sin buffers are 4800 x 8: 4 row + 4 col phases), wavelengths geometric from
  `lambda_min` 2.5 cells to `lambda_max = 2 * max(h, w) = 160` cells; the slowest pair cannot complete a
  half-cycle inside the grid, so the phase code is injective over all in-frame offsets
  (`tests/test_model.py::test_rope_no_global_alias`). Buffers are fp32 and not serialised.
- **XSA** (222k tier only): exclusive self-attention, a token is excluded from its own attention
  distribution; parameter-free (paper §3.2).
- **Attention gate** `gate3` on the H/4 skip only: `W_x` 1x1 stride-2 64->32 (no bias), `W_g` 1x1
  128->32, `psi` 1x1 32->1, sigmoid, bilinear upsample, broadcast over the skip's channels; 6,209 params
  at width 0.5. Initialised pass-through: `psi.weight = 0`, `psi.bias = +3.0`, alpha = sigmoid(3) = 0.9526
  everywhere at step 0. H/2 and full-resolution skips are never gated (a wrong veto there is unrecoverable).
- **Decoder**: bilinear upsample + 3x3 conv with skip concatenation: `d3` 192->64, `d2` 96->32, `d1` 48->16.
- **Heads**: `hm` 16->16 (3x3) -> 1 (1x1) at full resolution; `cls` 64->64 (3x3) -> 16 (1x1) at H/4.
  Final logit convolutions bias-initialised to -2.19 (CornerNet/CenterNet value; not re-derived for this
  task's 5.2e-5 positive rate, paper §3.2).
- **Receptive field** of the conv path to the bottleneck (kernel/stride structure is width-independent):
  theoretical 68 px without the dilated pair (release), 164 px with it (4.7M); Luo effective radius
  ~21 px / ~45 px. The adjacent marker sits ~0.85 s from a corner (10-109 px over the envelope), so at
  large `s` only the attention can supply identity (`R6/10_receptive_field`).
- **Refiner** (`dcc/model.py:Refiner`, 97,056 params): 3x3 conv 1->32, 32->64, 64->64 on the 24x24 crop,
  centre crop to 8x8, 3x3 64->64, 1x1 64->64 logits, PixelShuffle(8) -> 64x64 map over the central 8x8 px
  at 8x; decoded by a probability-weighted centroid over the 5x5 window around the argmax; no resize
  operator in the graph. Jitter range +/-4 px (`refiner_jitter_px 4`).
- **Not trained anywhere**: the `attend_div=16` native-resolution variant (1600x1200, `e5`, `gate4`, `d4`)
  described in `docs/ARCHITECTURE.md` and `configs/default.yaml`.

## 5. Stage 3: from maps to a pose or a refusal

Source: `dcc/pipeline.py` (constants read 2026-10-09), `docs/ARCHITECTURE.md`, `CLAUDE.md` pins.

1. `peaks`: 3x3 max-pool equality and `hm >= tau_hm` (0.3), descending score, capped at `top_k` 64.
2. `merge_close`: drop any peak within 2.0 px of a kept higher-score peak.
3. `cut_crops`: 24x24 crops on the sensor frame; border peaks bypass the refiner.
4. Refiner + `soft_argmax` (5x5 window). **Confidence guard** `refine_min_peak` 0.3 on the refiner's own
   peak sigmoid: below it the coarse peak is kept. Measured (n = 6,421, `R6/02_refiner/guard_sweep.json`):
   the refiner is worse than the coarse peak on 16.3% of crops, 55% of those featureless (std < 15);
   the guard fires on 13.8%, is right 69.5% of the time, and takes refinement p95 from 3.011 to 0.761 px
   while the median moves 0.1014 -> 0.1026 px (coarse-only: 0.4244 / 0.8449). Image-statistic guards
   lost to it. Disabling the guard made fp16 divergence worse (3.35 vs 0.81 px max), so it also
   stabilises deployment.
5. `read_ids`: bilinear `grid_sample` on the class map at the **coarse** peak positions
   (`id_readout="coarse"`, +0.20 pp measured; one H/4 cell is 4 px so sub-pixel carries no identity);
   `tau_id` 0.5 for an identity to be accepted. The whole identity chain (gate and recovery) runs on the
   coarse coordinates, so refined and unrefined arms are bit-identical in identity (2054/2054 verified).
6. `undistort` (`cv2.undistortPoints`) with K and dist; with no intrinsics the pose is refused
   (`no_intrinsics`) rather than guessed.
7. `lattice_gate`: RANSAC homography from identified corners to the canonical lattice, tolerance
   `lattice_tol_px` 3.0. Guards: fewer than 4 identified -> `too_few`; collinear canonical points ->
   `collinear`; exactly 4 -> fit is `vacuous`; 5 or more -> dissenters demoted.
8. `recover`: project all 16 canonical corners through H; an ID-less detection within tolerance inherits
   the identity. A vacuous fit with no corroborating recovery is refused (`vacuous_uncorroborated`).
9. `pnp`: `cv2.solvePnPGeneric` with `SOLVEPNP_IPPE` on the refined, undistorted points; returns
   rvec, tvec, reprojection RMS, `ambiguous = err2 / err1 < 1.5` (the planar two-fold flip), `n_used`.
   Failure -> `pnp_solver_failed`; fewer than 4 correspondences -> `too_few_correspondences`.
- Refusal reasons seen on B1 (882k): `too_few` 60, `collinear` 2, `pnp_solver_failed` 2 of 1000; the
  ambiguity flag fires on 43 of 936 solved frames (4.6%); 502k: 38/923 (4.1%); 222k: 38/900 (4.2%).
- The pipeline is pure functions with no training dependencies; PnP stays classical (no differentiable PnP,
  by design: synthetic labels give exact intermediates, and pose supervision is ill-conditioned at the
  planar ambiguity).
- Known unguarded path: `recover` divides by the projected homogeneous coordinate without a guard; with the
  degenerate homographies a foreign board produces this emitted RuntimeWarnings (`R6/27` README). Not
  reachable with a sane H; surfaced, not fixed.

## 6. Supervision

Source: `dcc/losses.py`, release configs, paper §3.3, `R6/12_loss/loss_formulation.md`.

- **Targets** (`dcc/targets.py` via `R6/12_loss`): per-corner Gaussians combined by **max** (never sum),
  peak pixel forced to exactly 1.0 at `rint(p_i)`; heatmap at full resolution with `sigma_hm` 0.5 px;
  class map at H/4 with `sigma_cls` 1.0 cell (cell coordinate `(x+0.5)/4 - 0.5`). Only visible corners
  are splatted; an illegible but visible corner keeps its true index (labelling is geometric).
- **Heatmap loss (all tiers)**: `strict_bce` = `binary_cross_entropy_with_logits` against `y == 1.0` as
  the positive set, every other pixel an equally-wrong negative. A BCE arm is therefore invariant to
  `sigma_hm` (only the forced peak pixel matters).
- **Class loss**: 882k uses the CornerNet penalty-reduced focal loss in logit space, alpha = 2, beta = 4,
  against the Gaussian class target; 502k and 222k use `strict_bce` on the class map too.
- **Total**: `(L_hm + lambda_cls * L_cls) / max(sum of visible corners in the batch, 1)`; one shared
  normaliser for both heads; sum reduction inside each head; all in fp32 logit space (the naive
  `log(1 - p)` form NaNs under bf16 near forced peaks).
- **Refiner loss**: focal (alpha 2, beta 4) on a 64x64 Gaussian target, `sigma_ref` 1.5, peak forced at
  `rint(u*)`, `u* = 31.5 + 8 d`, `d` in [-3.9375, 3.9375] px.
- `loss_kwargs(cfg)` is the single resolution point for form/alpha/beta (precedence `focal_beta` ->
  `beta` -> 4); an invalid form string asserts at step 0 (a `"bce"` typo once trained a full arm on the
  wrong loss).

## 7. Training configuration

Source: the release checkpoints' embedded `cfg` (`train:` and `refiner_train:` blocks), `runs/*/metrics.jsonl`,
`tools/train_detector.py`, `docs/TOOLING.md`, paper §3.5 and Appendix A.

| setting | detector (all three tiers) | refiner |
|---|---|---|
| step budget | **100,000** (the checkpoints' own `train.steps` is 100000; the generic YAML default of 250000 was overridden at launch) | 10,000 |
| batch | 16 x accumulation 2 (effective 32) | 256 (checkpoint cfg; **the paper's Table 2 says 512**, see §20) |
| optimiser | AdamW, betas (0.9, 0.999), peak LR 3e-4, floor 3e-6, cosine anchored to the budget, 1000 warmup steps | AdamW, LR 1e-3 -> 1e-5, 200 warmup |
| weight decay | 1e-4, excluding biases and normalisation parameters (`dcc/trainutil.py:param_groups`) | 1e-4 |
| gradient clip | 5.0 | 5.0 |
| EMA | 0.999, the EMA weights are what is validated and shipped | 0.999 |
| precision | bf16 autocast, fp32 master weights, `channels_last` | same |
| loader | spawn context, 12 workers, prefetch factor 4, `cv2.setNumThreads(1)` per worker (fork measured 7x slower: 55 vs 404 samples/s) | 8 workers |
| validation | every 2,500 steps on a fixed 2,000-sample subset; every 25,000 steps on the full 10,000-sample set (`val_every`, `full_val_every`); checkpoints at the full-val cadence | every 1,000 steps |
| seeds | `train_seed` 2000, `val_seed` 1000, `pose_seed` 3000; stream seed = `train_seed*1000 + resume_count` | — |
| early stopping | config block present (`min_steps 162000`, patience 3, tail gate 0.01) but never reached: the runs ended at their 100k budget | — |
| wall time on the RTX 5090 (first to last `metrics.jsonl` stamp) | 882k 8.38 h, 502k 8.41 h, 222k 9.36 h (resumed once) | 1.75 h |
| git hash in the checkpoints | `7aea0d4` | — |
| software | PyTorch 2.8.0+cu129, OpenCV 4.10.0, onnxruntime 1.28.0 (MLWS env) | |
| hardware | RTX 5090 32 GB, Intel Core Ultra 9 285K (24 cores), 62 GB RAM | |

- Throughput is **generation-bound on the CPU**, not GPU-bound: `generate_sample` costs ~86.5 ms per
  sample per core (~11.6 samples/s per worker; `R6/08_ablations/00_QUEUE_AND_DECISION_RULE.md`); the
  reference run averaged 67 samples/s solo, two arms ~127 aggregate; 16 workers exhausted 62 GB RAM and
  hard-locked the machine twice, so 12 (or 8 per concurrent arm) is the standing limit.
- **The last 25k steps were not worth it**: 75k -> 100k moved p95 by -0.0019 px (882k, 0.95x its seed
  floor) and -0.0024 (502k, 0.4x); the tail moved the wrong way on 882k. Recorded as a stop-rule lesson
  (`PLAN_autonomous_campaign.md`, 2026-08-04).
- **Release-run full-val trajectory** (median / p95 px, M-04; `runs/*/metrics.jsonl`):

| step | 222k | 502k | 882k |
|---|---|---|---|
| 25,000 | 0.4162 / 0.7597, 97.66% | 0.4105 / 0.7144, 98.87% | 0.4085 / 0.6945, 99.29% |
| 50,000 | 0.4140 / 0.7431, 98.85% | 0.4088 / 0.6977, 99.25% | 0.4072 / 0.6829, 99.42% |
| 75,000 | 0.4131 / 0.7368, 99.01% | 0.4078 / 0.6909, 99.39% | 0.4066 / 0.6752, 99.54% |
| 100,000 | 0.4128 / 0.7337, 99.09% | 0.4077 / 0.6885, 99.41% | 0.4063 / 0.6733, 99.53% |

- **Resume semantics** (fixed 2026-08-04): resume diffs the checkpoint's own config against the live one
  and warns when the cosine budget moved (`warn_cfg_drift`); `resume_count` increments per resume;
  `save_ckpt` writes then renames. Deep ChArUco's trainer once saved `resume_count` as a literal 0 (latent,
  the shipped baseline resumed only once and is clean).
- **Generator fingerprint lock** (`tests/test_guards.py`): a content hash of the generator sources plus the
  config subset that shapes their output is recorded with every checkpoint; training refuses to start if
  either drifted since the last green `tools/audit.py` run. Preflight (`tools/preflight.py`) predicts the
  init loss analytically from the resolved loss form (fixed 2026-08-05: it used to hardcode focal and
  passed CE arms whose real init loss was 89x what it compared against).

## 8. The synthetic data generator (rev-6)

Source: the 882k config's `synth:` block (identical in all tiers), `dcc/synth.py`, `R6/14_data/augmentations.txt`,
`docs/TOOLING.md`, paper Appendix B, `memory.md` (rev-6 entry, 2026-07-29).

- **Pipeline per sample**: render the board (`render_res` 480) -> sample a homography (affine: scale so
  `s` is log-uniform in [12, 128] px, in-plane rotation up to 180 deg, shear up to 35 deg, translation up
  to 0.45 of the frame; perspective with probability 0.8: tilt up to 60 deg about a random axis,
  `fov_scale` [0.7, 1.4]) -> warp onto a COCO train2017 background (118,287 images, random crop,
  horizontal flip p 0.5, rotation, reflect-pad) with histogram matching -> **scene integration**
  (relight by the background's blurred luminance, sigma 0.125 of min(w,h), clipped [0.45, 1.7]; edge
  feather 0.9 px; contact shadow offset [0.005, 0.03], strength [0.15, 0.5]) -> occluders (rectangles
  p 0.55, 1-6 holes of 16-64 px; SAM2 real-object cutouts p 0.5, up to 3, scale [0.08, 0.5] of the frame,
  from a bank of 14,721 RGBA cutouts cut from 3,000 COCO images) -> photometrics in a pinned order ->
  grayscale.
- **Corner labels are computed through the same homography**, never from pixels; visibility is geometric
  only (in frame and outside every occluder); photometric degradation never changes a label.
- **Negatives**: 5% of frames contain no board (`negative_p 0.05`).
- **Photometric stages and probabilities** (config values): specular lobe p 0.15 (strength 30-150,
  exponent 20-200); adherent droplets p 0.1 (1-4, refractive or bokeh mode); vignette p 0.5 (0.05-0.25);
  NIR ink-contrast jitter p 0.3 (scale 1.0-1.3); brightness p 0.5 (range -0.6..0.3, floor -0.58);
  contrast p 0.5 (0.8-1.2); glare p 0.25 (peak 30-130); ghosting p 0.25 (alpha 0.05-0.2, shift 1-4 px);
  RGB shift p 0.5 (limit 10); multiplicative field p 0.5 (0.98-1.02); motion blur p 0.5 (kernel up to 5);
  Gaussian blur p 0.25; Gaussian noise p 0.5 (std 3.2-7.1 DN); Poissonian-Gaussian sensor noise on
  (3-30 e-/DN, read noise 1-3 DN); fixed-pattern noise p 0.3 (column std 0.3-1.2) with PRNU p 0.5
  (0.005-0.015); speckle implemented but disabled (p 0); dark grey-out for deep-exposure frames.
- **Differencing domain** (`differencing_p 0.1`): a lit/unlit pair with daylight pedestal (ambient
  0.05-0.95), LED illumination lobe (peak 0.25-0.9, floor 0.05-0.3, sigma 0.25-0.45 of the frame),
  auto-exposure to target 250, inter-frame shift 0.2-3.0 px; the lit-minus-unlit difference is the frame.
  It fires on 10% of frames: it is the deployment domain, not the default one. Saturated lit regions
  produce physically-correct contrast-inverted residue (`dcc/synth.py`).
- **Severity calibration**: centred on an OV2311-class industrial camera (global shutter, monochrome,
  940 nm bandpass, no consumer ISP); the 2026-07-28 realism audit reduced several ranges (e.g.
  multiplicative noise from +/-5% to +/-2%, ghost alpha from 0.5 to 0.2).
- **Delivered SNR**: noise is drawn per BGR channel before the grayscale conversion, so the delivered
  frame carries 0.6686x the modelled sigma (measured 0.6676-0.6789): reported SNR is of the delivered
  frame and is 3.50 dB better than the per-channel formula implies. Documented and accepted, not changed
  (a change would be a generator revision requiring a full retrain).
- **Validation sets**: `SynthVal` = 10,000 samples from `val_seed` 1000, stratified by `s` octave;
  refiner val from `val_seed + 1`; bit-identical across processes for a fixed `{numpy, cv2, skimage}`
  version set (`tools/audit.py` gate 5).
- **Refiner stream**: 24x24 crops around corners jittered by +/-4 px; 92% of crops from a fast
  per-window render path matched to the full compositor, 8% (`refiner_full_frac 0.08`) from full
  composites; up to 8 corners per composite; `refiner_res_mult 2.5`; 1,250 validation composites.
- **Generator history** (what the results of record do NOT include): rev-2 augmentation pack (2026-07-28);
  rev-3 daylight differencing (retired and purged, "features nowhere"); rev-4 standard; rev-5 realism
  audit; **rev-6** (2026-07-29): the board is lit into the scene (relight, feather, shadow) and occlusion
  raised (p 0.40 -> 0.55, cutouts 0.35 -> 0.50). Every rev-5 number is stale; rev-5 was an *easier*
  distribution (the same model scored 0.7470 / 99.64% on rev-5 vs 0.8129 / 98.98% on rev-6), so nothing
  may be compared across generator revisions. `paper/results_rev5/` is kept for the record only.
- **Acceptance gates** (`tools/audit.py`, 7 gates): s-octave flatness within 20%, negative fraction within
  +/-0.005 of 0.05, occlusion incidence, warp round-trip < 0.01 px by an independently re-derived
  homography, subprocess byte-identical repeatability over 50 samples, refiner jitter histogram flat
  within 25%, refiner crops re-refined by `cv2.cornerSubPix` within median 0.10 / p90 0.50 px.

## 9. Evaluation protocol

Source: paper §4.1, `tools/train_detector.py` (M-01..M-04), `tools/eval_pose_ours.py`, `tools/robustness_sweep.py`
provenance blocks, `tools/gen_eval_pose.py`, `R6/23_code_audit_FINDINGS.md`.

- **M-01** localisation error of matched corners, greedy nearest-neighbour matching within 8 px
  (`train.match_px`); median and p95. **Tail > 4 px**: fraction of matched corners outside the refiner's
  capture radius, the operational gate. **M-02** match ratio (recall) by `s` octave (12-16, 16-32, 32-64,
  64-128). **M-04** corner-ID accuracy **conditioned on detection**: always read it beside recall.
  **M-03** refiner error. **M-05 / M-06** pose rotation (geodesic, deg) and translation (board squares).
- The in-loop 2,000-sample validation reads ~0.1-0.2 pp optimistic against the full 10k set; all
  reported numbers are from the full set at step 100,000, EMA weights.
- **Pose benchmark B1** (`eval_pose_rev6_b1`): 1000 images from `tools/gen_eval_pose.py`, `pose_seed`
  3000, per image a pinhole with f ~ U(0.7, 1.4) x W, `s` log-uniform on [12, 128] (depth z = f/s),
  in-plane rotation, tilt up to 60 deg, accepted when at least 8 of 16 corners project in frame; the
  homography is asserted against `cv2.projectPoints` to < 1e-6 px on every image; `square_length_m`
  fixed at 1.0 (translation in board squares). **B1 is the corrected set**: the original pose generator
  omitted the board-anchored specular / ink-contrast / differencing steps and did not let droplets gate
  visibility (audit finding B1, 2026-08-05); the fix lowered every arm's solve rate (ours by 1.0 pp,
  Deep ChArUco 2.8, classical 6.9) and widened our margins. B1 still omits the SAM2 cutout occluders
  (object occlusion is measured on its own robustness axis). Use only `*_B1.json`.
- **Robustness sweep**: 20 factors, one at a time, on frames that are bit-identical across methods and
  steps (seed 20260728), match tolerance 4.0 px, refined and coarse arms both scored; release sweeps
  `R6/20_robustness_REL{222,502,882}` use n = 60 frames per step; the Deep ChArUco, classical and 4.7M
  sweeps (`R6/05_comparison`, `R6/03_robustness`) use n = 100. Ladders:

| factor | unit | steps |
|---|---|---|
| distance | board square s (px) | 12, 16, 24, 32, 48, 64, 96, 128 |
| distance_extrap | s (px), outside the trained 12-128 | 6, 8, 10, 12, 16, 64, 128, 160, 192, 256 |
| rotation | in-plane deg | 0, 30, 60, 90, 120, 150, 180 |
| tilt | out-of-plane deg | 0, 10, 20, 30, 40, 50, 60 |
| brightness | offset | -0.9, -0.7, -0.5, -0.25, 0.0, 0.35 |
| darkness | brightness rescale factor | 1.0, 0.6, 0.36, 0.216, 0.13, 0.078, 0.047, 0.028, 0.017, 0.01 |
| contrast | scale | 0.6, 0.8, 1.0, 1.2, 1.4 |
| ink_contrast | NIR ink contrast scale | 1.0, 1.15, 1.3, 1.45, 1.6 |
| vignette | strength | 0.0, 0.05, 0.12, 0.2, 0.25 |
| specular | lobe strength (DN) | 0, 60, 120, 180, 220 |
| droplets | adherent droplets (n) | 0, 1, 2, 4, 6 |
| sensor_noise_K | gain K (e-/DN; lower is noisier) | 30, 16, 8, 4, 2, 1, 0.5, 0.25, 0.1, 0.05, 0.02 |
| motion_blur | kernel (px) | 0, 3, 5, 7, 9 |
| defocus_blur | Gaussian kernel (px) | 0, 3, 5, 7, 9 |
| board_occlusion | board area occluded (%) | 0, 10, 20, 30, 40, 50 |
| object_occlusion | SAM2 object occluders (n) | 0, 1, 2, 3, 4 |
| occlusion | rectangular occluders (n) | 0, 2, 4, 6, 8, 10 |
| diff_ratio | LED peak vs ambient 0.45 | 0.9, 0.6, 0.4, 0.25, 0.15 |
| diff_ambient | daylight pedestal | 0.05, 0.2, 0.4, 0.6, 0.8, 0.95 |
| diff_ghosting | inter-frame shift (px) | 0.0, 0.5, 1.0, 2.0, 3.0 |

  The SNR axis for `sensor_noise_K` comes from `tools/snr_calibration.py` (delivered-frame SNR; the K = 0.02
  step is 8.13 dB, which the pre-correction figures reported as 4.63 dB).
- **Baselines**:
  - *Deep ChArUco* (Hu et al. 2019): the public PyTorch reference port (`tools/charuconet.py`,
    JunkyByte/deepcharuco, `dcModel`), identity head re-initialised for 16 ids + dustbin (17 outputs),
    initialised from the public checkpoint and fine-tuned on **our** rev-6 generator with the same
    early-stopping criterion as our arms, resumed once, terminated at 62,458 steps on convergence
    (`runs/charuconet_rev6_ft`, checkpoint step 62,000). Input 320x240 against our 640x480. Its RefineNet
    is **not run**, so every localisation comparison is coarse vs coarse. Its corners pass through our
    identical gate -> recovery -> PnP stage. Parameters: 1,242,259 trainable in the 17-output detector
    (plus 2,560 BatchNorm statistics); the paper quotes 1,244,572 + 1,001,420 (RefineNet) = 2,245,992 and
    the cost table 1,242,002 + 999,233 = 2,241,235 (a 16-output head, no buffers); the 2.3x / 7.0x ratios
    hold under either count.
  - *Classical*: the OpenCV ChArUco pipeline at default detector parameters (`tools/eval_classical.py`);
    it reports only corners it has already identified, so its ID accuracy is ~100% by construction and
    only its recall is informative.
- **Rule 4b**: a baseline is never stopped on a looser criterion than our own arms.

## 10. Headline results (release models, 100,000 steps)

### 10.1 Corner detection, full 10k validation (`runs/*/metrics.jsonl` step 100000; `PLAN…md` 2026-08-04)

| tier | params | p95 (px) | median (px) | M-04 | tail > 4 px | recall by octave 12-16 / 16-32 / 32-64 / 64-128 |
|---|---:|---:|---:|---:|---:|---|
| 222k | 222,138 | 0.7337 | 0.4128 | 99.09% | 0.0190% | 87.53 / 91.57 / 93.62 / 93.92 % |
| 502k | 502,322 | 0.6885 | 0.4077 | 99.41% | 0.0146% | 90.49 / 93.23 / 94.87 / 95.07 % |
| **882k** | 882,402 | **0.6733** | **0.4063** | **99.53%** | **0.0048%** | 91.74 / 94.02 / 95.31 / 95.61 % |

Across a 4x parameter range, in-envelope identity spans 0.44 pp and median localisation 0.0065 px; the
tiers separate on p95 (+0.060 px), tail (4.0x) and far-octave recall (-4.2 pp). The 222k tier also
differs architecturally (XSA), so the ladder is a comparison of released configurations, not a clean
capacity sweep.

### 10.2 Pose, B1 (`R6/21_pose_REL*_B1.json`, `R6/13_pose/pose_release_vs_baselines_B1.json`)

All arms' corners go through the same `dcc.pipeline.pnp`; translation in board squares.

| arm | pipeline params | solve % | rot median / mean / p95 (deg) | trans median / mean / p95 (sq) |
|---|---:|---:|---|---|
| classical OpenCV | — | 54.4 | 0.1602 / 0.8690 / 2.3487 | 0.0200 / 0.0654 / 0.2236 |
| Deep ChArUco (fine-tuned) | 2,245,992 | 93.3 | 0.9131 / 10.6234 / **89.3603** | 0.0770 / 1.2636 / **7.2695** |
| 222k refined | 319,194 | 90.0 | 0.1269 / 0.6855 / 1.2261 | 0.0099 / 0.0691 / 0.1523 |
| 502k refined | 599,378 | 92.3 | 0.1295 / 0.5759 / 1.2297 | 0.0103 / 0.0604 / 0.1509 |
| **882k refined** | 979,458 | **93.6** | 0.1318 / **0.5368** / **1.1737** | 0.0104 / **0.0376** / **0.1282** |
| 882k coarse (no refiner) | 882,402 | 93.6 | 0.3744 / 1.1176 / 2.3424 | 0.0289 / 0.0739 / 0.2698 |
| 502k coarse | 502,322 | 92.3 | 0.3817 / 0.9507 / 2.1863 | 0.0268 / 0.0866 / 0.2577 |
| 222k coarse | 222,138 | 90.0 | 0.3762 / 1.0216 / 2.4357 | 0.0288 / 0.1006 / 0.2787 |

- Refusals (882k / 502k / 222k): `too_few` 60 / 66 / 93, `collinear` 2 / 3 / 2, `pnp_solver_failed` 2 / 3 / 2,
  `vacuous_uncorroborated` 0 / 5 / 3; ambiguous flags 43 / 38 / 38 of the solved frames.
- **Solve rate is not comparable across methods**: Deep ChArUco's 93.3% sits between our 502k and 882k, but
  its rotation p95 is 89.36 deg against 1.17-1.23 and its mean is 11.6x its median; it fails by emitting a
  confidently wrong pose where our gate refuses. Classical is accurate on what it solves and abstains on
  45.6% of frames. Even our coarse output (0.374-0.382 deg median) beats Deep ChArUco's 0.913.
- **Capacity buys the tail, not the median**: rotation median is flat (0.1269 / 0.1295 / 0.1318), the mean
  falls 0.6855 -> 0.5759 -> 0.5368 and mean translation falls 46% from 222k to 882k.
- Margins over the baselines: 882k reduces median rotation error 6.9x and p95 76x against Deep ChArUco
  with 2.3x fewer pipeline parameters; the 222k pipeline is 7.0x smaller and still better on every accuracy
  metric (paper abstract; the numbers above reproduce it).

### 10.3 Robustness, 20 factors, identical frames (computed 2026-10-09 from the sweep JSONs; recall % / ID %)

Release tiers refined arm (n = 60/step); Deep ChArUco and classical (n = 100/step, same seed). Classical
ID is not comparable and is omitted. **A. At the final ladder step** (the most severe step for one-sided
factors; brightness and rotation sweep outward in both directions, so their final step is one end only):

| factor | final step | 882k | 502k | 222k | Deep ChArUco | classical |
|---|---|---|---|---|---|---|
| board_occlusion | 50% | 85.9 / 71.2 | 87.1 / 56.8 | 82.3 / 40.2 | 73.1 / 63.8 | 9.6 |
| brightness | +0.35 | 85.9 / 97.8 | 84.9 / 96.2 | 82.0 / 93.7 | 74.1 / 82.9 | 22.7 |
| contrast | 1.4 | 93.3 / 98.6 | 91.9 / 93.2 | 88.4 / 92.4 | 81.2 / 84.4 | 33.3 |
| darkness | 0.01 | 45.8 / 69.1 | 37.3 / 27.7 | **5.6** / 40.4 | 0.2 / 33.3 | 0.0 |
| defocus_blur | 9 px | 93.7 / 99.1 | 92.8 / 96.0 | 91.2 / 94.2 | 84.9 / 87.6 | 22.5 |
| diff_ambient | 0.95 | 92.0 / 96.9 | 88.8 / 88.7 | 82.8 / 85.1 | 83.1 / 74.2 | 21.2 |
| diff_ghosting | 3.0 px | 94.7 / 97.1 | 93.6 / 92.2 | 91.6 / 87.4 | 86.3 / 77.5 | 30.0 |
| diff_ratio | 0.15 | 90.1 / 91.7 | 82.6 / 83.7 | 80.0 / 80.9 | 75.7 / 66.5 | 9.8 |
| distance | s = 128 | 92.8 / 97.4 | 92.6 / 91.8 | 91.7 / 88.4 | 89.6 / 50.2 | 18.6 |
| distance_extrap | s = 256 | 67.5 / 2.7 | 54.9 / 7.2 | 76.5 / 1.4 | 74.4 / 6.6 | 2.2 |
| droplets | 6 | 95.4 / 98.6 | 95.3 / 96.6 | 93.6 / 96.3 | 85.5 / 85.6 | 36.6 |
| ink_contrast | 1.6 | 70.3 / 80.8 | 63.6 / 75.4 | 54.9 / 74.1 | 58.5 / 68.3 | 11.3 |
| motion_blur | 9 px | 72.8 / 98.0 | 75.3 / 86.5 | 63.6 / 82.0 | 76.8 / 78.8 | 12.5 |
| object_occlusion | 4 | 96.0 / 99.7 | 94.2 / 96.8 | 92.8 / 96.5 | 87.5 / 85.1 | 32.8 |
| occlusion | 10 | 95.4 / 99.4 | 95.4 / 97.8 | 94.8 / 97.5 | 84.7 / 82.5 | 25.1 |
| rotation | 180 deg | 95.8 / 98.9 | 94.5 / 96.4 | 92.6 / 96.4 | 89.5 / 91.9 | 48.8 |
| sensor_noise_K | 0.02 | 16.6 / 83.2 | 15.4 / 63.0 | 16.5 / 61.8 | 13.8 / 73.8 | 2.6 |
| specular | 220 | 90.6 / 99.2 | 90.2 / 97.8 | 87.7 / 94.7 | 84.0 / 83.5 | 34.1 |
| tilt | 60 deg | 92.7 / 99.4 | 91.9 / 99.4 | 91.5 / 96.6 | 90.0 / 92.8 | 51.6 |
| vignette | 0.25 | 94.2 / 97.7 | 93.9 / 96.6 | 91.8 / 94.0 | 86.1 / 86.2 | 37.1 |

**B. Mean over every ladder step:**

| factor | 882k | 502k | 222k | Deep ChArUco | classical |
|---|---|---|---|---|---|
| board_occlusion | 90.6 / 89.3 | 89.8 / 82.6 | 86.3 / 74.7 | 79.5 / 72.9 | 21.2 |
| brightness | 92.3 / 97.8 | 91.5 / 97.3 | 89.1 / 94.4 | 82.5 / 85.0 | 30.3 |
| contrast | 94.9 / 99.4 | 94.1 / 97.6 | 91.6 / 96.6 | 85.6 / 86.0 | 36.1 |
| darkness | 80.6 / 91.4 | 77.4 / 82.5 | 66.3 / 78.4 | 54.5 / 56.6 | 11.1 |
| defocus_blur | 94.0 / 99.0 | 93.5 / 97.3 | 91.8 / 96.3 | 86.8 / 87.0 | 32.8 |
| diff_ambient | 95.0 / 98.0 | 93.1 / 95.6 | 90.6 / 92.6 | 87.1 / 81.8 | 30.3 |
| diff_ghosting | 95.9 / 98.8 | 94.5 / 96.8 | 92.2 / 93.2 | 87.8 / 83.2 | 30.9 |
| diff_ratio | 94.4 / 97.1 | 91.3 / 93.6 | 89.2 / 92.0 | 85.1 / 78.9 | 27.7 |
| distance | 94.4 / 99.1 | 93.5 / 96.8 | 92.3 / 94.9 | 83.2 / 79.0 | 30.6 |
| distance_extrap | 73.1 / 58.5 | 72.5 / 56.0 | 70.9 / 49.4 | 55.3 / 32.9 | 9.3 |
| droplets | 94.7 / 98.8 | 94.2 / 98.0 | 92.9 / 97.6 | 85.4 / 86.7 | 36.8 |
| ink_contrast | 85.9 / 93.1 | 83.7 / 90.6 | 79.0 / 89.3 | 76.0 / 80.0 | 26.3 |
| motion_blur | 89.2 / 98.7 | 89.2 / 95.3 | 84.3 / 93.1 | 84.6 / 85.1 | 28.8 |
| object_occlusion | 93.8 / 98.9 | 93.2 / 97.8 | 91.5 / 96.4 | 86.3 / 87.3 | 35.5 |
| occlusion | 94.4 / 99.3 | 94.0 / 97.5 | 92.6 / 97.3 | 85.6 / 85.0 | 34.1 |
| rotation | 95.5 / 99.1 | 94.9 / 98.0 | 93.7 / 96.7 | 92.7 / 91.1 | 51.7 |
| sensor_noise_K | 78.3 / 92.9 | 76.3 / 89.7 | 74.5 / 87.4 | 67.7 / 81.7 | 25.9 |
| specular | 94.4 / 99.3 | 94.0 / 98.3 | 92.3 / 96.4 | 85.3 / 84.9 | 33.9 |
| tilt | 95.1 / 99.7 | 94.3 / 98.9 | 93.5 / 97.9 | 92.8 / 92.9 | 53.8 |
| vignette | 95.1 / 98.3 | 94.6 / 97.6 | 92.4 / 96.5 | 86.3 / 87.0 | 37.0 |

Readings of record (`PLAN…md` 2026-08-04, paper §4.4): recall is near-flat across 4x of parameters in
benign conditions; **capacity buys identity under degradation** (board occlusion ID 40.2 -> 56.8 ->
71.2%, a 31 pp spread where clean validation separates the tiers by 0.44 pp); **the 222k tier collapses
in the dark** (5.6% recall at darkness 0.01 against 37.3% and 45.8%), a cliff invisible in clean numbers
and a deployment warning for an active-illumination rig. Localisation per factor for the release tiers
is in the JSONs; the 4.7M-era per-factor localisation tables are `R6/06_tables/robustness_err_*.md`.

### 10.4 Reproducibility floor (paper Table 4; `PLAN…md` 2026-08-02/03)

Seed-to-seed range of an identical configuration at the 35k ablation budget, full 10k val:

| width | seeds | M-04 range | p95 range | tail range |
|---|---|---|---|---|
| 222,138 | 4 | **6.18 pp** | 0.0231 px | 0.0100 pp |
| 502,322 | 4 | 0.09 pp | 0.0061 px | 0.0139 pp |
| 882,402 | 3 | 0.07 pp | 0.0020 px | 0.0049 pp |
| refiner 97,056 | 3 | median 0.0017 px | p95 0.0940 px | — |

Consequences: floors are per-width and per-metric (the 882k p95 floor is 11.6x tighter than 222k's; the
refiner's median is 55x more reproducible than its p95); mid-training deltas are convergence timing, not
quality (502k 4-seed M-04 range 35.39 pp at step 7,500 vs 0.09 pp at 35,000); the 222k rung was retired
as an ablation platform and five single-seed 222k findings did not survive; one 222k seed in four failed to
ignite identity within the cosine schedule (dead-ReLU explanation tested and falsified).

## 11. Ablations (all at matched budget, one key per arm, verified programmatically before launch)

Source: paper §4.6 and Appendix C (tables), `R6/08_ablations/*/train_metrics.md`, `PLAN…md` Part 1 and
2026-08-01..03 entries, `docs/PROJECT_KNOWLEDGE.md` §3. Values are p95 px / M-04 % / tail %.
**Absolute ablation numbers are budget-limited (35k or 50k steps) and must not be compared with the
100k release numbers; the deltas stand.** The 502k release at 100k (0.6885 / 99.41) beats the 882k arm at
35k (0.6894 / 99.15) on both axes: budget was binding, not capacity.

### 11.1 Attention bottleneck
- **A1 conv-only** (`attn_blocks 0`, 4.7M base, 50k vs 41k early-exit): aggregate 98.25% vs 98.80% ID,
  p95 unchanged (0.8114 vs 0.8149). Per regime (sweep, n = 100/step): FAR s = 12 **better** without
  attention (97.52 vs 96.19), s = 16 level (94.42 vs 93.98); VERY CLOSE s = 128 **-7.39 pp** (91.82 vs
  99.20), s = 160 beyond the envelope -30.6 pp (52.3 vs 82.9); occluded no effect (-0.32, -0.96 pp).
  Attention's entire contribution is board-scale context for identity at close range, where the conv
  receptive field is exhausted. Localisation identical to the fourth decimal.
- Bottleneck internals at 882k, 35k (control 0.7352 / 98.86 / 0.0849): 4 heads 0.7359 / 98.77 / 0.0746;
  attention at H/16 (+607k params) 0.7398 / 98.96 / 0.0586; 1 block 0.7398 / 98.51 / 0.0803; RoPE anchor
  lambda_min 5 cells 0.7400 / 98.89 / 0.0794. All within the 0.0020 px / 0.07 pp floor: the bottleneck is
  necessary and already at its smallest useful configuration.

### 11.2 Supervision (the real levers)
- Heatmap target width ladder at 882k, 35k: Gaussian sigma 1.0 0.7781 / 98.90 / 0.1006; 0.5 0.7352 /
  98.86 / 0.0849; 0.25 0.7335 / 99.00 / 0.0803; **one-hot BCE 0.6970 / 99.29 / 0.0114**. Identity varies
  0.14 pp across the Gaussian rungs; the tail improves 8.8x at the one-hot limit.
- At 4.7M, 50k: sigma 2.0 focal 0.8149 / 98.80 / 0.1221; sigma 1.0 0.7430 / 99.06 / 0.0687; sigma 0.25
  0.7016 / 99.05 / 0.0392; **sigma 2.0 with beta = 0** 0.7032 / 98.87 / 0.0463; **one-hot BCE 0.6690 /
  99.62 / 0.0056** (the single largest effect in the campaign: -0.146 px, +0.82 pp, 22x smaller tail).
- **Mechanism, measured** (n = 830 corners): under Gaussian focal supervision the trained peak has RMS
  radius 1.0028 px with 26.1% of local mass on the peak pixel; under one-hot 0.2665 px and 93.5%. The
  graded target enters only through `(1-Y)^beta` on the negative branch; at sigma 2, beta 4 a pixel one px
  from the corner has Y = 0.8825 and weight 1.9e-4, so displaced mass is nearly free. beta = 0 recovers
  76.6% of the localisation gain but 8.5% of the identity gain: the proximity discount governs
  localisation, the focal easy-example modulation governs identity, hence per-head supervision.
- Per-head crossing, 35k (heatmap x class): at 882k focal/focal 0.7352 / 98.86 / 0.0849; focal/BCE 0.7572 /
  99.00 / 0.0931; **BCE/focal 0.6874 / 98.87 / 0.0081** (best p95 and tail, released); BCE/BCE 0.6970 /
  99.29 / 0.0114. At 222k: focal/focal 0.8096 / 94.87; focal/BCE 0.8222 / 96.71; **BCE/focal collapses
  identity to 58.54%**; BCE/BCE 0.7611 / 94.76, so the 882k combination is not carried to 222k.
- `lambda_cls`, 35k: 882k 1.5 0.7021 / 99.34; 2.0 0.7059 / 99.40; 4.0 0.7143 / 99.51 (a trade: ~0.06 px
  p95 per pp of identity); 502k 1.5 0.7122 / 98.98; 2.0 0.7208 / 99.09; 3.0 0.7234 / 99.12; 4.0 0.7262 /
  99.29 (1.5 is the only strictly free lever at 502k, hence the release value); 222k 0.5 0.7318 / 91.96;
  2.0 0.7580 / 97.80; 4.0 0.7703 / 97.78.
- `sigma_cls` is null in both directions at 502k (0.5 -> 0.7110 / 98.87; 2.0 -> 0.7104 / 98.80; control
  0.7139 / 98.80): one H/4 cell is four input px, so sub-pixel class-target structure is unrecoverable.
- `sigma_ref` (refiner target width) null: median 0.0937 vs 0.0969 px at 0.75 and 1.5, inside the median
  floor; the refiner's p95 floor (0.0940 px) cannot arbitrate its p95 column. BCE on the refiner: no, the
  detector's lever does not transfer (`PLAN…md` 2026-08-03).

### 11.3 Structure (all null, within the floor)
- Skip gate off: p95 -0.0011 at 882k (0.6959 vs 0.6970), +0.0018 at 502k; a second gate on the H/8 skip
  at 502k 0.7063 / 98.89 vs 0.7139 / 98.80 (1.2x / 1.0x the ranges). Gating retained on H/4 only.
  gate_skips [3,2] had been called "the one unambiguous free win" from a 25k validation (+0.16 pp); at 35k
  it decayed to exactly 1.0x the range: null.
- At 4.7M, 50k: removing the dilated e4 cascade 0.8072 / 98.77 vs control 0.8149 / 98.80 for -25%
  parameters (adopted); adding cross-scale attention 0.8148 / 98.82; gates off 0.8134 / 98.88; halving the
  width with dilation kept 0.8605 / 98.67 (the one structural change clearly outside the range, a
  degradation).
- XSA at 222k: -0.0040 px p95 and +0.21 pp identity, both inside the 6.18 pp floor; adopted because it is
  free, stated as unresolved.
- Learning rate at 502k, 35k: 6e-4 gives 0.7100 / 99.30 vs 0.7238 / 98.10 at 1.5e-4; 3e-4 retained for the
  longer release schedule.
- Composites do not stack: C1 (two gates + lambda 2) at 222k 0.7624 / 97.25 is worse than lambda 2 alone
  (0.7580 / 97.80); same-axis levers stop adding identity while still delivering localisation. C2
  (one-hot heatmap + focal class + lambda 2) at 882k 0.6894 / 99.15 / 0.0065 is the released recipe.
- Null levers list: `attn_heads` 8 -> 4, `attend_div` 8 -> 16, gate topology, `gates_enabled`,
  `e4_dilated`, `xsa`, `sigma_cls` both directions, `sigma_ref`, width beyond 502k. The campaign's central
  finding: every real lever is in the loss and target representation; none is in the network.
- Deferred arms (built or specified, deliberately not run; `R6/08_ablations/00_BACK_POCKET_deferred_arms.md`):
  XSA on top of nodilate + half width; decoupled conv-width vs attention-width attribution (needs a
  3-line model change); A-BETA0 at 882k width.
- Arm inventory (run names under `R6/08_ablations/`): `abl_wh_ctrl35k`, `abl_wh_ce`, `abl_wh_s1`,
  `abl_wh_s025`, `abl_wh_heads4`, `abl_wh_attend16`, `abl_wh_attn1`, `abl_wh_rope5`, `abl_lx_wh_*` and
  `abl_lx_fast_*` (per-head crossings), `abl_c1..c5_*` (composites), `abl_hp_w25_*`, `abl_hp_w375_*`
  (lambda / sigma_cls), `abl_lam15/30/40_w375`, `abl_r502_*`, `abl_gs_*` and `abl_g_*` (gates),
  `abl_t_*` (tier ladder), `abl_seed200[1-3]_*` (noise floor), `abl_nodilate_width_{half,quarter}[_xsa]_s05`,
  `abl_width_half`, `abl_sigma025`, `abl_composite_s05`, `A1_conv_only`, `A_CE_one_hot`, `A_XSA`,
  `A_SIGMA05`, `A_SIGMA1`, `A_BETA0`, `A_GATES_OFF`, `A_NODILATE`. Each folder's `train_metrics.md` is the
  arm's own validation record.

## 12. Efficiency and deployment

Source: `R6/24_export_parity/export_parity.json`, `R6/24_export_parity_detfp16`, `R6/23_code_audit_FINDINGS.md`
P15 section, `R6/25_runtime_latency/*.json`, `R6/15_cost/*`, `/home/kaelin/p4p/wireless_inference`.

- **ONNX export** (opset 17, `onnx.checker` clean, no Loop / If / Complex ops): detectors 284 / 263 / 263
  nodes at 2.05 / 3.70 / 5.74 MB (222k / 502k / 882k; the 222k's extra nodes are XSA); refiner 20 nodes,
  0.37 MB (ops Conv, Relu, Slice, DepthToSpace, Constant). Op-level agreement with PyTorch ~1e-4 in logit
  space once TF32 is disabled (TF32 is on by default for cuDNN and masquerades as a 6.9e-2 "disagreement").
- **fp16 parity gate** (P15, n = 1000, threshold: < 0.1% of corners displaced > 0.05 px, ID change < 0.1%):
  both nets in fp16 **fail on every tier** (107 / 101 / 104 corners displaced, 0.896 / 0.833 / 0.851%).
  Attribution by halving one net at a time (882k, n = 400): detector-only 3 corners (0.062%, each exactly
  one cell flip, all the ID change 0.0206%); refiner-only 41 (0.846%, max 3.589 px, zero ID change);
  additive (41 + 3 = 44). **Shippable policy: detector fp16 + refiner fp32**, certified at n = 1000:
  displaced corners 8 / 1 / 5 (0.067 / 0.008 / 0.041%), `d_max` exactly 1.000 px, ID diff 0.1005 / 0.0000 /
  0.0164%; 502k and 882k PASS; 222k fails the identity budget by 0.0005 pp (accepted and stated).
- **Measured latency on the RTX 5090** (`25_runtime_latency`, PyTorch 2.8.0+cu129, median of 30):
  detector forward 882k fp16 2.04 ms / fp32 4.49 ms; 222k 1.70 / 3.88 ms; refiner on 44 crops batched
  0.16 ms (CUDA fp32); onnxruntime CUDA EP fp32 (onnxruntime-gpu 1.30, incl. H2D/D2H) 5.54 / 5.25 ms;
  TensorRT fp32 4.81 / 4.49 ms, **TensorRT fp16 0.80 / 2.46 ms** with 0 threshold-pixel flips and 0 ID
  argmax flips on ~202 checked pixels; CPU onnxruntime 8 threads 151 / 130 ms. The Python `detect()` path
  totals 10.7 ms (882k), of which `peaks + merge` is 4.25 ms on the CPU (a threshold-first scan would
  remove most of it); every other post-processing stage is sub-millisecond.
- **Compute and memory** (`15_cost`): 52.0 GFLOPs per 640x480 frame for the 882k-class architecture,
  19.8 for the 222k-class (model-level counts, independent of checkpoint); at batch 1 fp16 the weights are
  1.68 / 0.42 MB and activations 122.4 / 61.5 MB (882k / 222k), so memory is activation-dominated and
  scales with resolution, not width. **The Orin figures in `cost_table.md` (5.0 / 1.9 ms, 198 / 522 fps)
  are datasheet rooflines at 15% of fp16 peak, not measurements; no Orin hardware has run anything.**
- **Deployment code** (`/home/kaelin/p4p/wireless_inference`): onnxruntime sessions with the CUDA
  execution provider, static shapes asserted, IO binding to pre-allocated GPU buffers; `cfg/cfg.yaml`
  points at `checkpoints/detector_882k.onnx` and `checkpoints/refiner.onnx` (a `detector_222k.onnx` is also
  present). The config carries no board size or camera intrinsics yet.

## 13. Measurement noise for a Kalman filter (`R6/26_pose_error_variance`)

Ensemble statistics of the 882k pipeline's accepted (unambiguous) poses on B1, n = 893; camera frame;
error 6-vector [drot_x, drot_y, drot_z (rad); dt_x, dt_y, dt_z (board squares)].

| component | robust std (1.4826 MAD) | sample std |
|---|---|---|
| rot x / y / z | 0.091 / 0.088 / 0.037 deg | 0.276 / 0.290 / 0.093 deg |
| trans x / y / z | 0.0029 / 0.0027 / 0.0116 sq | 0.0127 / 0.0086 / 0.0433 sq |

- Means are negligible (<= 3e-4 rad, <= 3e-3 sq). Heavy tails: sample variance 9-14x the robust variance;
  33.6% of accepted solves have a component beyond 3 robust sigma, 20.9% beyond 5. Off-diagonals weak
  (|rho| <= 0.23): a diagonal R is defensible. Including the 43 ambiguous solves inflates every std ~10x.
- **A constant R is wrong by up to 7x across the range**: median |e|/sigma is 2.0-7.4 at the far octave and
  0.28-0.49 at the near one. Robust sigma follows a power law in range z = f/s (board squares):
  exponents 0.97 / 0.81 / 1.02 (rotation), 1.50 / 1.49 / 1.73 (translation x, y, depth); under that model
  median |e|/sigma is within 0.83-1.24 in every octave; fitted span z = 3.9-70.7. Constants:
  `kalman_R_REL882.json` -> `range_scaled_R`.
- Recommendation of record: R_k = diag(a^2 z_k^(2k)); skip the update on refusal, on the ambiguous flag, and
  on PnP residual `rms_px > 0.15` (rejects 15.8% of accepted frames, removes 51% of the > 5 sigma tail);
  chi-square innovation gate at the 95% bound for 6 dof (12.59). 13.5% of frames still exceed 5 sigma
  under the range-scaled model, so the gate is required regardless of R.
- **Not usable as R**: the network's own `pose_cov` (NEES mean 11.2 against 6 expected; it does not rank
  frames) and `sigma_px` (an ordering signal: rank correlation +0.633 with error, 12x separation between
  the lowest and highest deciles, but not a variance). Calibration of either is parked by decision
  (`paper/results_rev5/10_uncertainty/NOTES_what_didnt_work.md`); no inflation factor may be fitted.
- Multiply translation variances by `square_length_m^2`; if the filter tracks in the board frame invert the
  pose first (the translation error then picks up a rotation-error coupling).

## 14. Board transfer and minimal fine-tuning (`R6/27`, `R6/28`, `R6/29`, `R6/30`)

- **Zero-shot to `DICT_6X6_250`** (same geometry, unseen marker family, 882k untouched): localisation and
  recall unchanged (median 0.4063 both; p95 0.6733 vs 0.6754; recall 94.2 vs 94.3%); ID 99.53% -> 13.36%
  (chance 6.25%); poses: correct 89.1% -> 1.8%, **wrong-but-accepted 0.2% -> 22.0%**, refused 6.4% -> 73.3%.
  Of 267 zero-shot solves, 197 are ~90 deg wrong with translation error exactly one board side and 39 are
  ~180 deg wrong: lattice-symmetric relabellings pass the lattice gate with a normal residual (median
  0.098 vs 0.088 px) and IPPE's ambiguity test does not cover them. Across the 20-factor sweep: recall
  91.1 vs 91.1%, ID 95.3 vs 3.9%.
- **Minimal fine-tuning** (21 arms, 5,000 steps = 1/20 of the release budget unless stated, batch 16 x 2,
  AdamW 3e-4 cosine, EMA 0.99, from the 882k EMA checkpoint; `finetune.lr_mult` per fnmatch group, unmatched
  parameters frozen with their BatchNorm in eval): the attention bottleneck's learning rate is the lever.
  ID head only 57.4% (re-initialised 58.9, 3x LR 59.7, 20k steps 61.8); head + gate 57.4; head + d3 61.3;
  BitFit 73.3; post-bottleneck (gate, decoder, both heads) 72.3; head + bottleneck + corner head at 0.01x
  69.3, 0.1x 90.5 (96.0 at 20k), 0.3x 93.5, 1x 97.1, **3x 98.2% ID / 86.2% poses, 95% ID by step 1,000**,
  **20k steps 99.1% / 87.6%**; whole model 1x 98.4 / 86.9, 3x 98.9 / 88.3 (p95 0.682, the only arm that
  moves localisation). Trainable set 437,105 params (50%) for the minimal recipe; peak VRAM 3.5 GB at
  batch 16, 0.9 GB at batch 4, ~48 ms/step on the 5090 (~1 GPU-minute to 95%, ~14 to 99%); memory follows
  gradient depth, not parameter count (BitFit 42k params, 4.6 GB). Wall time was the CPU generator
  (76 samples/s per lane). Wrong-accepted poses stay at 0.0-0.5% for every fine-tuned arm.
- **Robustness of the 20k fine-tune** (same 20 factors): recall 89.8 vs 89.6% and localisation equal to
  the trained board; ID 91.1 vs 93.9% mean, the gap concentrated at the hardest darkness (24 vs 69%) and
  noise (52 vs 72%) steps. Clean-condition identity transfers; identity under heavy degradation only partly.
- **Multi-board pretraining does not help at these budgets**: a 15-family base (two boards per family via
  `marker_id_offset`, 50k budget) stayed at the 25% lattice-symmetry floor for 25k steps and was stopped;
  a 3-family base transitioned after 15k steps to 82.58% at 30k but scores 80.60% ID / 62.9% poses on the
  original board and is a worse base for a fourth family at every trainable set (head 51.0 vs 57.4;
  bottleneck 0.1x 72.5 vs 90.5; 1x 86.7 vs 97.1; post-bottleneck 67.3 vs 72.3; from the 15-family base
  28.4 / 30.0). Caveat: the 15-family stop may have been premature; a 100k-step 15-family base is untested.

## 15. Introspection (measured on the 4.7M `rev640_160k_rev6` model unless stated; `R6/07_introspection`, `R6/11_attention_gate`)

- Attention entropy at initialisation equals ln(T) (8.92 nats at T = 7,500; the 882k grid has T = 4,800);
  trained bottleneck entropy fell to ~7.0 nats (reference) and 4.96 with XSA at step 5k, i.e. attention
  concentrates rather than collapses (`R6/08_ablations/00_BACK_POCKET…`).
- Head roles (40 frames, `attention_head_roles.json`): `tools/attention_head_roles.py` flags block-1 heads 3,
  5 and 7 as marker readers; heads 3 and 5 put 0.53 / 0.44 of their median attention mass on marker
  interiors (lift 22 / 20 over uniform). Block-0 heads concentrate on corners (median corner mass 0.32-0.58
  for six of the eight heads).
- Gate alpha on trained weights: median 0.444 inside the board region, 0.559 outside (ratio 0.79, sign not
  stable across frames): content-conditioned, not a fixed spatial prior; it selects where, never which
  feature.
- The effective-receptive-field probe (`tools/introspect.py --panels erf`, M-07) shows frame-spanning reach
  for the hybrid and a bounded local footprint for the conv-only ablation.

## 16. Testing and verification

- **Unit and conventions tests** (`tests/`, 16 files, 137 test functions, 142 tests collected with
  parametrisation; **all 142 pass on 2026-10-09**, 7 min 48 s on the 5090, 8 deprecation warnings from the
  legacy ONNX exporter; run with `PYTHONPATH=` cleared and pytest plugin autoload disabled because a ROS
  install and a `detectron2` checkout shadow imports):
  `test_synth.py` (corner formula, marker identity, targets), `test_generator.py` (warp round trip,
  perspective calibration, visibility truth table, index under rotation, negatives, record schema,
  determinism, val stratification, cutout visibility and determinism, prefilter), `test_augmentations.py`
  (determinism, label preservation, droplet hole registration, differencing and its noise scaling,
  Poisson-Gaussian noise, vignette, ink contrast, FPN, config keys), `test_model.py` (shapes, parameter
  count, bias inits, stable parameter names, RoPE no-global-alias, loss finite at N = 0, XSA properties,
  ONNX export, gradient flow through gates and attention, refiner loss), `test_variant640.py`
  (`attend_div` 8/16 construction, RoPE anchor adaptation, early-stop rule, preflight, grad groups, wandb
  wiring), `test_pipeline.py` (readout convention, peaks/merge, border bypass, soft-argmax, undistort,
  lattice gate, IPPE PnP, empty solver result, `detect` contract), `test_board_adapt.py` (4x4 / n9 boards,
  rectangular boards rejected, retarget load, class-head-only optimisation), `test_trainutil.py` (cosine
  LR, EMA, checkpoint round trip, parameter groups incl. frozen), `test_guards.py` (generator fingerprint
  lock), `test_refiner_fast.py` (fast crop path equivalence, jitter flatness), `test_audit_parallel.py`,
  `test_eval_classical.py`, `test_factor_sweep.py`, `test_cross_eval.py`, `test_plot_figures.py`,
  `test_viz.py` (per-panel figure export).
- **Data acceptance**: `tools/audit.py` seven gates (Section 8); the generator fingerprint lock blocks
  training after an un-audited generator change.
- **Pre-training**: `tools/preflight.py` (analytic init-loss prediction against the measured first batch,
  resolved through the same `loss_kwargs`), `tools/cut_ablation.py` (refuses any arm that differs from its
  base in anything but the named keys; the loader is a bare `yaml.safe_load`, so configs are full copies
  and a hand-cut arm once differed in ten keys and trained on the easier rev-5 stream).
- **Release gates**: P15 fp16 export parity (Section 12), the B1 reproduction gate (regenerated pose set
  reproduces the banked JSON exactly), byte-identical figure regeneration as proof of dead code.
- **Gates that were found vacuous and fixed** (`R6/23_code_audit_FINDINGS.md`): the preflight loss check
  (hardcoded focal), fp16 parity v1 (compared integer peaks against a 0.05 px threshold), fp16 parity v2
  (judged on p95 while the mean exceeded the p95), `--freeze-trunk` (once applied unconditionally, so a
  smoke run trained 2.1% of the network).
- Code audit of 2026-08-05 (`ast` sweep): 31 assigned-never-read hits, 0 defects; 14 duplicate top-level
  names, 0 defects (the two `score_frame` implementations both match at 4.0 px); 7 unused imports removed;
  one dead block removed with byte-identical regeneration.

## 17. Known issues and open items

- `dcc/pipeline.py:recover` has no guard on its homogeneous divide (Section 5).
- Noise is drawn per channel (delivered SNR 3.50 dB better than modelled); documented, not changed.
- The 222k tier exceeds the fp16 identity budget by 0.0005 pp; accepted and stated. Its real constraint is
  the darkness cliff.
- B1 omits SAM2 object occluders (deliberate; measured on its own axis).
- Uncertainty calibration is parked; `pose_cov` ~6x over-confident, `sigma_px` ordering-only.
- The plotting tool's per-factor darkness override leaked into the ID and localisation grids until
  2026-10-09 (fixed; `fig2a`, `fig6b`, `fig6c` and the two zero-shot panel figures regenerated; no table
  number was affected). Two layout demo grids in `R6/05_comparison/figures_L_demo/` still carry it.
- Self-supervised fine-tuning from robot video is designed, not built (`R6/16_future_work/ssl_from_video.md`).
- A 100k-step 15-family multi-board base is untested.
- `tools/eval.py` and `tools/curves.py` named in `docs/TOOLING.md` do not exist.

## 18. Errata: things once believed and now known false (keep out of the report)

From `docs/PROJECT_KNOWLEDGE.md` §8 and the campaign log:
1. "The refiner guard causes the fp16 jumps": disabling it made the max worse (3.35 vs 0.81 px).
2. "Dead ReLUs explain the 222k collapse": zero hard-dead units in any seed.
3. "gate_skips [3,2] is a free win": 1.0x the range at 35k.
4. "conv-only tests the loss": it removes the transformer and leaves the loss alone.
5. "rev-5 results are comparable": rev-5 was easier by an order of magnitude more than the seed floor.
6. "The 4.7M model's marginal 15 Hz budget applies": measured on a model 5.3x larger than what ships.
7. "A permissive Gaussian target improves stability and expressiveness" (the original main claim): the
   one-hot arm beat every Gaussian rung on localisation, identity and tail; the shipped models use one-hot
   BCE on the heatmap.
8. "The refiner guard is abandoned" (2026-07-28): reversed 2026-07-29 on new evidence; the guard is in.
9. "The 150k -> 160k rev-5 refinement helps": rejected, no accuracy gain, attention board-mass worse.
10. Pre-B1 pose numbers (solve 92.9 / 95.2 / 95.5%) are superseded by B1 (90.0 / 92.3 / 93.6%).

## 19. Where the paper and the older documents disagree with the artefacts

Checked 2026-10-09 against the checkpoints, configs and code:
1. **Paper Appendix A, architecture**: "embedding dimension 256 with 8 heads" and "8 frequencies per axis
   over 16 dimensions per axis" describe the width-1.0 model. The 882k release has **d = 128, head_dim 16,
   4 RoPE frequencies per axis over 8 dimensions per axis** (MLP ratio 4, LayerNorm eps 1e-6 and qkv bias
   are correct as stated; encoder widths 16/32/64/128 are correct).
2. **Paper Table 2**: refiner batch 512; the release refiner checkpoint's own config says **256**.
3. **Paper §4.1 and Table 3**: Deep ChArUco parameter counts 1,244,572 + 1,001,420; the local port with the
   17-output head counts 1,242,259 trainable (+2,560 BN statistics), and `R6/15_cost` quotes 1,242,002 +
   999,233. Ratios unaffected.
4. **`docs/ARCHITECTURE.md`** (loss paragraph): one focal loss on both heads with `sigma_hm` 2 px; the
   dilated e4 pair described as present. **`README.md`** lines 35 and 118: "penalty-reduced focal losses"
   and focal exponents "shared by heatmap, class and refiner". **`paper/conference/SLIDES.md`** line 133:
   "both heads share one penalty-reduced focal loss". **`R6/12_loss/loss_formulation.md`**: lambda_cls 1.0,
   sigma_hm 2.0, focal on both heads. All describe the pre-ablation 4.7M model.
5. **`R6/06_tables/*.md`** and **`R6/15_cost/cost_table.md`**: "Ours" is the 4.7M `rev640_160k_rev6`
   (sweep provenance) and the cost table's ID columns are 35k / 45k ablation checkpoints, not the release
   models; its GFLOPs are valid architecture-level numbers.
6. **`R6/11_attention_gate/attention_gate.md`**: gate dimensions and 24,705 params are the 4.7M model's; at
   width 0.5 the gate is 64->32 / 128->32 / 32->1 with 6,209 params.
7. **`R6/07_introspection`** head-role and gate-alpha measurements are on the 4.7M model.
8. **`docs/TOOLING.md`** says seven tools ship; the local `tools/` directory holds 59 scripts (listed on the
   `report` branch), and `tools/eval.py` / `tools/curves.py` do not exist.

## 20. Result directories (`paper/results_rev6/`) and what each is

| directory | status | holds |
|---|---|---|
| `02_refiner` | of record (refiner, 4.7M-era detector crops) | guard sweep, tail composition, saturation attribution, failure gallery |
| `03_robustness`, `_PREGUARD`, `_L`, `_FAST`, `_ConvChArT`, `_L_tau015` | **historical** (4.7M and ablation-era checkpoints) | 20-factor sweeps |
| `05_comparison` | of record for the baselines | Deep ChArUco and classical sweeps (n = 100), figures |
| `06_tables` | historical ("Ours" = 4.7M) | inter-method tables |
| `07_introspection`, `_L` | historical | attention, gates, ERF, feature panels |
| `08_ablations` | of record (matched-budget deltas) | per-arm folders, reference sweeps, queue and decision notes |
| `10_receptive_field`, `11_attention_gate`, `12_loss` | explanatory (4.7M constants) | formulations as PDFs |
| `13_pose` | of record: `pose_release_vs_baselines_B1.json`; the rest historical | pose tables |
| `14_data` | of record | augmentation list, training-sample sheet |
| `15_cost` | mixed (see §19) | cost table, MACs, VRAM |
| `16_future_work` | design note | SSL from video |
| `17_range` | of record | working range vs lens and board size |
| `20_robustness_REL{222,502,882}` | **of record** | release-tier sweeps (n = 60) |
| `21_pose_REL*_B1.json` | **of record** (the non-B1 files are superseded) | pose per tier |
| `22_fourway_RELEASE`, `_882_vs_222` | of record | per-factor comparison figures and the 2x2 panel |
| `23_code_audit_FINDINGS.md` | of record | audit, P15 gate, B1 correction |
| `24_export_parity`, `_detfp16` | of record | ONNX files (local only), parity JSON |
| `25_runtime_latency` | of record | 5090 latency: torch, onnxruntime, TensorRT, stage timings |
| `26_pose_error_variance` | of record | Kalman R, per-frame errors, range fits |
| `27_transfer_zeroshot`, `28_finetune_minimal`, `29_multiboard_base`, `30_REPORT_transfer_finetune` | of record | transfer and fine-tuning studies, report figures (every panel also as its own file) |
| `31_print_board` | of record | print-ready board |
| `PLAN_autonomous_campaign.md` | the dated campaign log | decisions and their evidence |
| `paper/results_rev5/` | superseded generator revision | do not quote |
