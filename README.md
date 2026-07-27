# Conv-ChArT

**Conv-ChArT** (Convolutional ChArUco Transformer) is a scale-robust, marker-specific ChArUco corner-and-pose detector: a hybrid conv–transformer network (conv encoder → bottleneck self-attention with 2D axial RoPE → gated-skip decoder → two factorised heads) detects and identifies all 16 inner corners of a 5×5 ChArUco board in one forward pass across an 8× apparent-scale range, a small board-agnostic refiner lifts each detection to sub-pixel accuracy on the native sensor frame, and a classical geometry stage (undistort → RANSAC lattice fit → recovery → `SOLVEPNP_IPPE`) turns the result into a camera pose. Training is **100% synthetic** — every sample is composited on the fly from a rendered board, a COCO background, a randomly sampled homography (affine + out-of-plane tilt), and a photometric pipeline extended for an active-illumination lit/dark-differencing deployment domain (heavy noise, glare, ghosting, motion blur). No real image is ever labelled; a small real-capture set exists only as a sim-to-real transfer check, never for training.

This repository is a from-scratch implementation: every architectural and numerical choice below is a deliberate design decision, documented inline with its reasoning, or a measurement quoted with its provenance. `paper/PAPER.md` is the full research-paper writeup of the same system — problem statement, method, derivations, data engine, and the ablation plan, all in one place with every measured number.

---

## Architecture

Three stages. Stage 1 detects corners and reads identities from a single network. Stage 2 refines each corner to sub-pixel on the native sensor crop. Stage 3 turns detections into a pose through a lattice-consistency gate and classical PnP.

```
STAGE 1 — DetectorNet (dcc/model.py) — input 1×H×W, native 1600×1200 (ρ = sensor/input = 1)
──────────────────────────────────────────────────────────────────────────────────────────
  e1 -> e2 -> e3 -> e4 -> e5        conv encoder, H -> H/16, widths 32-64-128-256-256
  (H)   (H/2) (H/4) (H/8) (H/16, final stage adds dilated 3x3 convs, rates 2 & 4)
                          |
                       blocks        2x pre-norm MHSA+MLP, d=256, h=8, 2D axial RoPE on Q,K
                          |          tokens = H/16 x W/16 = 75x100 = 7,500, row-major flatten
                          |          final LayerNorm  — sole board-scale context mechanism
                          |
  d4 <- gate4(e4) <-------+         H/16 -> H/8   (gate4: gates the H/8 skip)
  d3 <- gate3(e3) <- d4             H/8  -> H/4   (gate3: gates the H/4 skip) --tap--> cls head (H/4 x W/4 x 16, sigmoid)
  d2 <-       e2  <- d3             H/4  -> H/2   (skip UNGATED)
  d1 <-       e1  <- d2             H/2  -> H     (skip UNGATED)             --tap--> hm  head (H x W x 1,   sigmoid)

STAGE 2 — Refiner (dcc/model.py) — per detected peak, board-agnostic, own checkpoint
──────────────────────────────────────────────────────────────────────────────────────────
  24x24 sensor-frame crop -> 3x conv (1->32->64->64) -> centre-crop to central 8x8
    -> conv (64->64) -> 1x1 conv (64->64 = 8^2 channels) -> PixelShuffle(8)
    -> sigmoid -> soft-argmax (5x5 window around the hard argmax) -> u* = 31.5 + 8d -> xy_refined

STAGE 3 — Inference pipeline (dcc/pipeline.py:detect) — pure functions, no training deps
──────────────────────────────────────────────────────────────────────────────────────────
  peaks (3x3 max-pool equality, tau_hm) -> merge_close (NMS radius 2 px)
    -> cut_crops (sensor frame, border peaks bypass Stage 2) -> Refiner -> soft_argmax -> xy_sensor
    -> read_ids (grid_sample on the class map, align_corners=False) -> per-corner index + confidence
    -> undistort (cv2.undistortPoints) -> pinhole-space corners
    -> lattice_gate (RANSAC-fit canonical-lattice -> image homography) -> inliers / demoted IDs
    -> recover (project the 16 canonical corners through H; ID-less detections within tol inherit)
    -> pnp (cv2.solvePnPGeneric, SOLVEPNP_IPPE) -> rvec, tvec, reprojection RMS, ambiguity flag
```

The refiner's soft-argmax readout follows integral pose regression [Sun2018]: a probability-weighted centroid over a small window around the hard argmax, not a hard argmax itself — a few lines of code, no reference implementation needed, chosen so the sub-pixel offset stays differentiable in training even though nothing downstream currently backpropagates through it.

**Why the hybrid trunk.** Convolutions ([Ronneberger2015]-style paired 3×3 blocks, with a dilated-convolution pair [Yu2016] added at the final encoder stage to cheaply widen its receptive field before the bottleneck) supply high-resolution, translation-equivariant features for localisation; the class head's identity read needs board-scale context (adjacent-marker radius ≈ 0.85$s$, full context propagation 1.85–2.85$s$, i.e. up to ≈ 365 px at $s=128$) that a conv-only stack cannot reach — its theoretical receptive field is ≈ 164 px and its *effective* RF is a Gaussian fraction of that ($\sigma \approx 31$ px, usable radius ~31–62 px) [Luo2016], further shrunk by skip dominance and ReLU. The two pre-norm MHSA blocks [Vaswani2017] at the H/16 bottleneck are the closure: after them, only three convolutions and one bilinear upsample separate attention output from class logits, none of which can destroy global content [Chen2021]. The decoder upsamples with bilinear interpolation + 3×3 conv at every stage, never a transposed convolution — stride-periodic checkerboard artifacts would beat directly against this system's own checkerboard target [Odena2016].

**Why two factorised heads.** [Hu2019]'s localisation/identity split is the organising idea this whole detector extends to a single-shot, board-scale-context-aware network: localisation reads skip-rich full-resolution features; identity reads globally-mixed context at H/4. "Corner present, ID unknown" stays expressible (heatmap high, all 16 class channels low) — essential in the dark/blur regime the system targets. Both heads use CornerNet/CenterNet-style penalty-reduced focal loss on Gaussian-splatted targets (max-combined across corners, never summed) [Law2018] [Zhou2019]:

$$
L = \underbrace{-\tfrac{1}{N}\sum_{xy}\Big[(1-\hat p)^{\alpha}\log \hat p\Big]_{Y=1} + \Big[(1-Y)^{\beta}\hat p^{\alpha}\log(1-\hat p)\Big]_{Y\neq 1}}_{L_{hm},\ \sigma_{hm}=2\text{ px}} \;+\; \lambda_{cls}\underbrace{\sum_{k} L_k}_{L_{cls},\ \sigma_{cls}=1\text{ cell (H/4)}}, \qquad \alpha=2,\ \beta=4
$$

$N$ is the batch-total count of visible corners, clamped $\geq 1$ (handles all-negative batches with no special case); $\hat p$ are predicted probabilities, $Y$ the rendered targets. The implementation computes this in **logit space** (`torch.nn.functional.logsigmoid`) so it is finite and NaN-free under bf16 near forced-1.0 peaks — `dcc/losses.py:focal`.

**2D axial RoPE** [Su2021] [Heo2024] rotates Q/K per head so the attention dot product depends only on content and relative offset $(\Delta\text{row}, \Delta\text{col})$ — translation-equivariant by construction. Per axis, $n = d_h/4 = 8$ frequencies are spaced **geometrically in wavelength**, not in the standard RoPE base-exponent form: $\omega_i = 2\pi/\lambda_i$, $\lambda_i$ geometric on $[2.5,\ 2\!\cdot\!\max(H',W')]$ cells ($H'\times W' = 75\times 100$ at native res, 16-px cells). This is a deliberate departure from a textbook `rope-vit` lift: standard RoPE's fastest wavelength is $2\pi\approx 6.28$ cells for *any* base, so the spec's own "span from 2 cells" requirement is unsatisfiable by the textbook parameterisation; wavelength-anchoring hits it directly (2.5 rather than exactly 2, to clear the Nyquist-degenerate sign-alternation at $\lambda=2$). The long end guarantees no aliasing: $\lambda_{\max}\geq 2\!\cdot\!\max(H',W')$ cells makes the slowest pair's phase injective over every in-frame offset, so the full 8-frequency phase code never repeats for two distinct in-grid positions (`dcc/model.py:AxialRoPE`, locked by `tests/test_model.py::test_rope_no_global_alias`).

**Attention-gated skips** [Oktay2018] gate only the H/4-and-coarser skips (`gate3`, `gate4`); the H/2 and full-resolution skips feeding the heatmap head are **always ungated**. Rationale: a corner whose identity is lost to a closed gate falls back to ID-less and is recovered by the Stage-3 lattice projection — recoverable. A corner vetoed at full resolution by a learned "looks like board" gate is gone before any recovery mechanism can see it — unrecoverable, and exactly where the 3–7 px motion/Gaussian-blur augmentation range hides junction energy at H/2 scale. Gates initialise to pass-through: `psi.weight = 0`, `psi.bias = +3.0` ⟹ $\alpha_0 = \sigma(3)\approx 0.953$ exactly constant at step 0 (the zero weight kills all input-dependence outright), yet `psi.weight` still receives gradient from the very first step (no dead-gate trap) — `dcc/model.py:AttnGate`, verified live at native resolution (`gate3_alpha_mean = gate4_alpha_mean = 0.953125` in `runs/smoke/metrics.jsonl`).

**No differentiable PnP.** Synthesis provides exact intermediate labels (corner position, index, visibility); pose supervision would be sparse and ill-conditioned at the planar two-fold ambiguity. Training stays modular on intermediates (heatmap, class map, refiner offset); PnP stays classical (`cv2.solvePnPGeneric`, `SOLVEPNP_IPPE` [Collins2014]) with intrinsics supplied only at inference.

**Deployment target**: NVIDIA Jetson AGX Orin, INT8/fp16 TensorRT engines exported from the fp32/bf16-trained checkpoint (a parity gate — corner error Δ < 0.05 px, ID accuracy Δ < 0.1 % on 1k val images — must pass before an exported engine is trusted). All development, training and evaluation run at full precision on an RTX 5090; no quantisation happens during development. At 15 Hz pose (a 66 ms/frame budget for one lit+dark differencing pair), a measured 833 GFLOPs/frame puts the Orin at 42–83 ms in fp16 (marginal) and 21–42 ms in INT8 (comfortable) — see `paper/PAPER.md` §6 for the full budget arithmetic.

---

## Repository layout

```
dense deep charuco/
├── README.md                 this file
├── configs/
│   └── default.yaml          every tunable knob: input size, loss/threshold constants, synth: and train: blocks
├── dcc/                       library code — imported by tools/ and tests/, never run as a script
│   ├── __init__.py            empty; marks dcc as a package
│   ├── board.py                board convention: cv2 CharucoBoard + the analytic corner-index formula (23 ln)
│   ├── synth.py                  generator: compositing, perspective warp, occlusion, cutouts, photometrics (489 ln)
│   ├── targets.py                 target renderers: heatmap / class / refiner Gaussian splats (70 ln)
│   ├── dataset.py                  torch Dataset/IterableDataset shims over synth.py: SynthStream, SynthVal, RefinerVal (118 ln)
│   ├── viz.py                       visualisation primitives shared by view.py / audit.py / introspect.py (95 ln)
│   ├── model.py                      networks: DetectorNet and Refiner (247 ln)
│   ├── losses.py                      penalty-reduced focal losses, logit-space (34 ln)
│   ├── pipeline.py                     Stage-3 inference: peaks -> refine -> IDs -> gate -> PnP (285 ln)
│   └── trainutil.py                    EMA, cosine LR schedule, checkpoint I/O, JSONL logger, no-decay param groups (139 ln)
├── tools/                    CLI entry points — run as scripts ONLY, see the import footgun below (322+146+184+443+460+266+192 ln)
│   ├── audit.py                acceptance gate: overlays, distributions, round-trip, repeatability, refiner content check
│   ├── view.py                  interactive/sheet viewer for detector or refiner samples and their rendered GT targets
│   ├── gen_eval_pose.py           pose-consistent evaluation-set generator (the only place camera pose exists)
│   ├── gen_cutouts.py              offline SAM2 cutout-bank builder for realistic object occlusion (run under the `eomt` env)
│   ├── train_detector.py            Stage-1 (detector) training loop
│   ├── train_refiner.py              Stage-2 (refiner) training loop
│   └── introspect.py                  conference-grade introspection panels (attention, ERF, gates, features, pipeline, 3D heatmap)
├── tests/                    pytest, 49 tests total, ~231 s (see Testing below)
│   ├── test_synth.py          board + target-renderer conventions (7 tests)
│   ├── test_generator.py       full generator: warps, perspective calibration, visibility, cutouts, determinism (15 tests)
│   ├── test_model.py            DetectorNet/Refiner shapes, param counts, RoPE, losses, ONNX export (11 tests)
│   ├── test_pipeline.py          Stage-3 functions: readout convention, gate degeneracy table, PnP/IPPE (9 tests)
│   └── test_trainutil.py          EMA, checkpoint round-trip, param groups, JSONL logger (7 tests)
├── docs/
│   └── TOOLING.md            full CLI reference: every flag, every output artifact, gates and exit codes
├── paper/                    research-paper master document — full problem statement, method, derivations, data engine, ablation plan
├── introspect_out/           example introspection panels (generated by tools/introspect.py)
└── sheets/                   example audit/view sheets (generated by tools/audit.py, tools/view.py)
```

Training run outputs (`runs/<name>/{metrics.jsonl, ckpt_*.pt}`, plus a config snapshot inside each checkpoint) are generated artifacts, not part of the shipping repository, and belong in a local `.gitignore` entry rather than the tree above.

---

## Environment

- **Conda env `MLWS`** — Python 3.13.5, `torch` 2.8.0+cu129 (CUDA-verified), `numpy` 2.3.1, `cv2` 4.10.0, `skimage` 0.25.2, `onnx` 1.22.0, `matplotlib`, `pyyaml`. This is the environment every `dcc/`, `tools/`, and `tests/` invocation below runs under: `/home/kaelin/anaconda3/envs/MLWS/bin/python`.
- **`PYTHONPATH` footgun (machine-wide).** A global `PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages` leaks ROS's `launch_pytest` plugin into every Python invocation on this machine, which breaks pytest collection outright (`ModuleNotFoundError: lark`, before any test runs). **Always clear it**: prefix every command below with `PYTHONPATH=`.
- **`tools` package-shadow footgun (machine-wide).** An unrelated, editable-installed `detectron2` checkout puts its own `tools/` package on `sys.path` ahead of this project's — `import tools` resolves to `/home/kaelin/detectron2/tools/__init__.py`, not this repository's `tools/`. **Never `import tools.*` or run `python -m tools.foo`.** Every script in `tools/` inserts the project root onto `sys.path[0]` itself (`sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`) and is meant to be invoked as a plain script: `python tools/audit.py ...`. `dcc/` has no such conflict and is always imported normally (`from dcc.model import DetectorNet`).
- **Conda env `eomt`** — a *separate* environment (Python 3.13.2, its own torch build) used only by `tools/gen_cutouts.py` to run SAM2 automatic mask generation. The two environments carry incompatible torch builds and must never be imported in the same interpreter; `gen_cutouts.py` imports only `cv2`/`numpy`/`torch`/`sam2`/stdlib and never touches `dcc`.
- **pytest invocation**: `PYTHONPATH= /home/kaelin/anaconda3/envs/MLWS/bin/python -m pytest tests/ -q` (see Testing below for expected runtime).
- **Background corpus**: COCO `train2017` at `/home/kaelin/datasets/coco/train2017` (`configs/default.yaml`'s `synth.backgrounds`; 118,287 images present on this machine). Any directory of `.jpg`/`.jpeg`/`.png` works, or a COCO annotation `.json` used purely as a file-name manifest (`dcc/synth.py:list_backgrounds`) — corpus choice is a config path, not code.
- **Cutout bank**: `synth.cutouts.path` defaults to `/home/kaelin/datasets/cutouts`, built once by `tools/gen_cutouts.py` (under `eomt`, see `docs/TOOLING.md`). **Present and complete on this machine**: the full $3{,}000$-image production sweep has run to completion — $14{,}721$ RGBA cutout files on disk, with `manifest.json` present and fully self-consistent (every listed file verified present, zero missing). `dcc/synth.py:load_cutouts` still returns `[]` for a missing or partial directory (fail-soft by design, so every other tool and test runs correctly even against an incomplete bank), but that fallback is no longer the operative case here.

---

## Quickstart

Every command below assumes `cwd` = `dense deep charuco/` (quote the space) and `PYTHONPATH=` cleared. See `docs/TOOLING.md` for every flag.

**1. Generate and eyeball samples.**
```
PYTHONPATH= /home/kaelin/anaconda3/envs/MLWS/bin/python tools/view.py --stream detector --n 16 --out sheets/detector_sheet.png --channels
PYTHONPATH= /home/kaelin/anaconda3/envs/MLWS/bin/python tools/view.py --stream refiner  --n 16 --out sheets/refiner_sheet.png
```

**2. Run the acceptance audit** (gates: s_px octave flatness, negative fraction 0.05±0.005, occlusion incidence, round-trip <0.01 px, byte-identical repeatability, refiner offset histogram + content check). Exits 0 on all-pass, 1 on any gate failure, 2 if the background corpus is missing:
```
PYTHONPATH= /home/kaelin/anaconda3/envs/MLWS/bin/python tools/audit.py --config configs/default.yaml --out audit/
```

**3. Preflight sanity pass** (seven checks against a freshly constructed, untrained model/refiner pair — initial-loss prediction, translation equivariance, RoPE relativity/no-alias, gradient balance, refiner zero-offset closure, bf16 parity, and one-batch overfit; see `docs/TOOLING.md` for the full reference) ahead of any real multi-hour run:
```
PYTHONPATH= /home/kaelin/anaconda3/envs/MLWS/bin/python tools/preflight.py --config configs/default.yaml
```

**4. Train the detector** (Stage 1; `runs/<name>/{metrics.jsonl, ckpt_*.pt}`; needs a CUDA GPU with the flash/mem-efficient SDPA backend):
```
PYTHONPATH= /home/kaelin/anaconda3/envs/MLWS/bin/python tools/train_detector.py --config configs/default.yaml --name run1
```
Retarget an already-trained trunk onto a differently-labelled board's corpus with `--freeze-trunk --resume runs/run1/ckpt_XXXXXXX.pt` — see the `freeze_trunk` note in Conventions for the mechanism; this path is correctly implemented (see `docs/TOOLING.md`) but has not yet been exercised end-to-end against a second physical board.

**5. Train the refiner** (Stage 2, separate run and checkpoint, board-agnostic):
```
PYTHONPATH= /home/kaelin/anaconda3/envs/MLWS/bin/python tools/train_refiner.py --config configs/default.yaml --name run1
```

**6. Introspect a checkpoint** (six presentation-grade panels: pipeline, 3D heatmap, attention, gates, ERF, encoder features — omit `--ckpt`/`--refiner-ckpt` for an untrained-network baseline):
```
PYTHONPATH= /home/kaelin/anaconda3/envs/MLWS/bin/python tools/introspect.py --ckpt runs/run1/ckpt_0250000.pt --refiner-ckpt runs/run1_refiner/ckpt_0010000.pt --index 7 --out introspect_out/
```

**7. Evaluate.** *Pending — a dedicated evaluation tool (`tools/eval.py`, `tools/curves.py`, M-01..M-06 vs a checkpoint + the sign-off PASS/FAIL line) has not been started; neither file exists on disk yet.* Until it lands, per-checkpoint M-01/M-02/M-04 numbers are available from `train_detector.py`'s own in-loop validation (`runs/<name>/metrics.jsonl`, `val`/`full_val` records), and M-03/bias-vs-jitter from `train_refiner.py`'s.

---

## Configuration reference (`configs/default.yaml`)

**Top-level (Stage 1 loss/threshold constants):**

| Key | Meaning |
|---|---|
| `input_size: [1600, 1200]` | Network input `[W, H]`; native OV2311 4:3 sensor crop, so ρ = sensor/input = 1 (no resize in the deployment path). Must be divisible by 16 (`DetectorNet` asserts it). |
| `sigma_hm` | Heatmap Gaussian width, px at input resolution ($\sigma_{hm}=2$). |
| `sigma_cls` | Class-map Gaussian width, cells at H/4 ($\sigma_{cls}=1$, i.e. 4 input px). |
| `alpha`, `beta` | Focal-loss exponents ($\alpha=2,\ \beta=4$), shared by heatmap, class, and refiner losses. |
| `lambda_cls` | Class-loss weight in $L = L_{hm} + \lambda_{cls} L_{cls}$; 1.0 is calibrated at init (both heads' element counts are exactly equal: 1,920,000 px each at native res). |
| `tau_hm`, `tau_id` | Stage-3 operating thresholds: heatmap peak acceptance (0.3) and per-channel ID confidence (0.5). Also duplicated under `train:` for validation-time peak extraction — see below. |
| `lattice_tol_px` | RANSAC reprojection tolerance for the Stage-3 lattice-consistency gate, sensor-scale px. |
| `negative_p` | Fraction of generated samples with no board at all, keeps class channels and the heatmap honest on empty scenes. |
| `scale_range_px: [16, 128]` | Sampling range for square size $s$ at input resolution, log-uniform (the training operating envelope; see `paper/PAPER.md` Appendix C). |
| `refiner_jitter_px` | Refiner crop-centre jitter and hard capture range, px (±4). |
| `freeze_trunk` | Gates Stage-1 training to the class head alone for board retargeting — see the Conventions section and `docs/TOOLING.md`'s `train_detector.py` notes for the mechanism. |
| `attn_blocks`, `attn_heads` | Bottleneck transformer depth (2) and head count (8); $d=256 \Rightarrow d_h = 32$. |
| `rope_lambda_min_cells` | Fastest RoPE wavelength anchor, 2.5 cells (see Architecture; replaces the standard base-exponent RoPE parameterisation, which cannot satisfy this system's wavelength-span requirement at any base value). |

**`board:`** — board identity. `squares: [5,5]`, `dictionary: DICT_5X5_50`, `marker_ids: [0, 11]` (12 markers, raster order), `marker_ratio: 0.7` (marker/square length ratio); `square_length_m` is the *only* runtime-variable field — metric scale enters solely at PnP, never at training time (the 16-channel class head is locked to the inner-corner count).

**`synth:`** — generator parameters.

| Key | Meaning |
|---|---|
| `render_res` | Board render resolution in px (480 ⟹ render square $SQ = 480/5 = 96$ px). |
| `backgrounds` | Path to the background corpus (directory glob or COCO `.json` manifest). |
| `val_seed`, `train_seed`, `pose_seed` | Root seeds for `SynthVal`/`RefinerVal` (+1), `SynthStream`, and `tools/gen_eval_pose.py` respectively — every draw is `default_rng([root, index])`, never the numpy global RNG. |
| `val_size` | Fixed-seed detector validation-set size (10,000). |
| `refiner_res_mult` | Refiner-stream canvas multiplier over `input_size`; **1** at native res (ρ=1, the sensor frame *is* the input frame — a larger multiplier only matters at ρ≥2, inapplicable here). |
| `refiner_max_corners` | Crops cut per composite for the refiner stream (≤8). |
| `refiner_val_composites` | Composites in the fixed-seed refiner validation set (1,250 ⟹ ≥10k crops). |
| `rot_deg`, `shear_deg`, `translate_frac` | Affine sampling ranges: background/board rotation (±180°), shear (±35°), translation (±45% of canvas). |
| `perspective_p`, `tilt_max_deg`, `fov_scale` | Tilt-calibrated homography perturbation: probability a sample draws nonzero tilt, tilt range, and the virtual-pinhole focal-scale range. A fronto-parallel board tilted by $\tau_{\text{tilt}}\sim U(0,\text{tilt\_max\_deg})$ about an in-plane axis at foreshortening-axis angle $\psi\sim U(0,2\pi)$, viewed by a virtual pinhole of focal length $f = \text{fov\_scale}\cdot W$ ($\text{fov\_scale}\sim U(0.7,1.4)$), is reproduced (first-order, exact at $\tau_{\text{tilt}}=0$) by right-multiplying the 6-DoF affine by a perspective factor conjugated about the render centre, with third-row perturbation $g=\sin(\tau_{\text{tilt}})\cdot s/(f\cdot SQ)\cdot(\cos\psi,\sin\psi)$ — `dcc/synth.py:_perspective_factor`, calibrated against an independently-built pinhole model in `tests/test_generator.py::test_perspective_calibration` (relative error ≤ 0.142 at $\tau_{\text{tilt}}=60°$ vs. the first-order expectation). Note $\psi$ here is the *foreshortening-gradient* axis; the physical Rodrigues tilt axis is $\psi-90°$. |
| `bg_hflip_p` | Background horizontal-flip probability. |
| `occlusion.{p,holes,size}` | CoarseDropout-style rectangular holes: probability, hole-count range, side-length range (px). |
| `cutouts.{path,p,max_objects,scale}` | Object-cutout occlusion (realistic clutter, additional to the rect holes): bank directory, per-sample probability, max objects placed, size range as a fraction of the canvas's shorter side. |
| `photometric.*` | Photometric set: Gaussian noise, motion blur, Gaussian blur, multiplicative noise, brightness (mixed multiplicative/additive semantics), RGB shift, plus the differencing-domain extensions **glare** and **ghosting** (additive elliptical glare patch; `img′ = (1+α)·img − α·shift_blur(img,δ)`) — added for the lit/dark-differencing deployment domain, config-gated, on by default. |

**`train:`** (Stage-1 recipe) — `batch`/`accum` (micro-batch 4 × grad-accumulation 8 = effective batch 32), `steps` (250,000), `lr`/`lr_floor`/`warmup_steps` (3e-4 cosine to 3e-6 over 250k steps, 1000-step linear warmup), `wd` (1e-4, biases/norm params excluded — `dcc/trainutil.py:param_groups`), `clip_norm` (5.0, global-norm grad clip), `ema_decay` (0.999), `workers`/`prefetch_factor` (spawn-context DataLoader), `val_every`/`full_val_every`/`val_subset` (2k-subset validation every 2,500 steps, full 10k-set + checkpoint every 25,000), `match_px` (8 px greedy NN-match radius for M-01/M-02, distinct from the ±4 px refiner-tail threshold, which is a fixed constant in code, not a config key), `tau_hm` (validation-time peak-extraction threshold — a second copy of the top-level `tau_hm`, read independently by `run_validation`).

**`refiner_train:`** (Stage-2 recipe) — `batch` (256 crops, no grad accumulation), `steps` (10,000), `lr`/`lr_floor`/`warmup_steps` (1e-3 cosine to 1e-5 over 200-step warmup), `wd`/`clip_norm`/`ema_decay` (same conventions as `train:`), `workers`, `val_every` (1,000 steps; every validation also writes a checkpoint).

---

## Conventions

> **Pixel-centre coordinates.** An integer pixel coordinate is that pixel's *centre* (OpenCV convention) — a square boundary in a 480 px board render falls **between** pixels, hence board.py's corner formula lands on half-integers: `corner_px(i) = ((col+1)·SQ − 0.5, (row+1)·SQ − 0.5)`.
>
> **Half-pixel resolution maps, everywhere a resolution changes.** Going from a fine grid to a coarser grid whose cells span $k$ fine-resolution units (input→sensor with $k=\rho$; input→H/4 class grid with $k=4$):
> $$x_{\text{coarse}} = \frac{x_{\text{fine}} + 0.5}{k} - 0.5 \qquad\Longleftrightarrow\qquad x_{\text{fine}} = (x_{\text{coarse}} + 0.5)\cdot k - 0.5$$
> This single convention replaces the spec's literal "coords × 1/r" and "÷4" phrasing everywhere it appears; the binding form for the class-map readout is `F.grid_sample(mode=bilinear, padding_mode=border, align_corners=False)` at `grid_x = 2·(x_in+0.5)/W_in − 1` — algebraically identical to the cell-space formula above, `align_corners=True` is independently locked *wrong* by `tests/test_pipeline.py::test_readout_convention`.
>
> **Float64 keypoints end-to-end; no integer cast anywhere in the label/warp chain.** Corners are carried through every homography as float keypoints from `dcc/board.py`'s analytic geometry onward; the only integer casts in the whole pipeline are deliberate readout-time roundings (`np.rint`) at fixed, documented sites (heatmap peak forcing, refiner crop centring, ID-map cell lookup).
>
> **RNG-through-`Generator` purity.** Every random draw in `dcc/board.py`, `dcc/synth.py`, `dcc/targets.py`, and `dcc/dataset.py` takes an explicit `numpy.random.Generator` argument; none of them reaches into the numpy global RNG, the stdlib `random` module, or cv2's RNG — enforced by a static grep in `tests/test_generator.py::test_determinism`. `dcc/trainutil.py` is the one legitimate exception: `save_ckpt`/`load_ckpt` capture and restore **torch's** global RNG state (`torch.get_rng_state()`) for bit-exact resume of the model side, which is orthogonal to data-generation purity — every numpy draw for sample generation still flows through an explicit, config-seeded `Generator` (`stream_seed = train_seed·1000 + resume_count`, bumped on every resume so a resumed run never replays the identical sample sequence).
>
> **Checkpoint contract.** One `.pt` file, `torch.save`d as a dict with exactly the keys `{step, resume_count, model, ema, optim, cfg, git_hash, torch_rng, last_val}` (`dcc/trainutil.py:save_ckpt`); `model`/`ema` are `state_dict()`s keyed on the stable module-name contract `e1..e5, pool, rope, blocks, norm, gate4, gate3, d4, d3, d2, d1, hm, cls` (`tests/test_model.py::test_stable_names` locks this). The board definition travels inside `cfg`, so a checkpoint is self-describing for retargeting.
>
> **`freeze_trunk` boundary (model-level contract).** At the `nn.Module` level, `cls.*` is a disjoint, independently-addressable parameter subtree from everything else (`e1..e5`, `rope`, `blocks`, `norm`, `gate3`, `gate4`, `d1..d4`, `hm`) — a retarget onto a new board's corpus freezes every parameter except `cls.*` (trunk, both gates, the full decoder, and the heatmap head transfer unchanged; only the class head retrains) and holds frozen BatchNorm layers in `eval()` mode so their running statistics don't drift on the new corpus's images. The `--freeze-trunk` CLI flag correctly implements this contract: the parameter-trainability mask is gated by the same conditional that switches BatchNorm to `eval()` mode, so a default (non-retarget) run trains the full network and only an explicit retarget request restricts training to the class head. This is a corrected defect, not the original behaviour — **see `docs/TOOLING.md`'s `train_detector.py` section** for the mechanism and a short account of the earlier, unconditional version of this check that this project's own documentation-and-verification review caught.

---

## Testing

Five files, 49 tests, pure-pytest (no COCO or GPU download dependency for the suite itself — `test_generator.py`'s fixtures synthesise their own tiny noise backgrounds and a 3-file synthetic cutout bank; `test_model.py`'s ONNX/RoPE/loss tests and `test_pipeline.py` run on CPU).

| File | Tests | Locks | Measured runtime |
|---|---|---|---|
| `test_synth.py` | 7 | Board convention (analytic corner formula ≤0.05 px vs `cornerSubPix`, marker identity ≤1.5 px), target renderers (heatmap/class Y=1 forcing, max-not-sum combine, refiner offset encoding, edge-window clipping). | 0.16 s |
| `test_generator.py` | 15 | Full composite pipeline: warp round-trip <0.01 px, perspective calibration against an independently-built pinhole model, visibility truth table (hole/frame edges), corner-index invariance under 180° rotation, negative-sample emptiness, refiner stream + content check, byte-identical determinism, val-set stratification, and 4 object-cutout tests (visibility, RNG-budget discipline, determinism, record-schema stability). | **≈226 s** — dominated by `test_val_stratification`'s 1,000 sequential single-process samples at the current native 1600×1200 `input_size` (this loop pre-dates the 640×480→native migration and was not re-timed after it; ~15 s at the old resolution). |
| `test_model.py` | 11 | `DetectorNet`/`Refiner` shapes, native-res param count (7,124,700 ± 2%), bias initialisation (−2.19 heads, gate pass-through 0.953), stable `state_dict` key/prefix contract, RoPE no-global-alias (analytic + 2,000-pair sampled check), loss finiteness on $N{=}0$ batches under fp32 and bf16, forward determinism, ONNX opset-17 export (no `Complex`/`Loop`/`If` ops), gate/attention gradient flow. | 4.23 s |
| `test_pipeline.py` | 9 | Stage-3 functions: ID-readout convention (locks `align_corners=False` against a crafted counter-example), peak extraction + NMS merge + `top_k`, border-bypass crop cutting (byte-verified content, $\rho\in\{1,2.5\}$), soft-argmax orientation and accuracy, undistort identity, lattice-gate degeneracy table (too-few / collinear / vacuous / demotion / recovery / no-double-claim), PnP/IPPE (rotation/translation accuracy, ambiguity flag, the silently-empty-solver-result no-pose outcome), and an untrained end-to-end `detect()` contract check. | 1.31 s |
| `test_trainutil.py` | 7 | Cosine LR shape (warmup/decay/floor monotonicity), EMA convergence, checkpoint round-trip (bit-exact model/EMA/optimiser/RNG restore), `restore_optim=False` retarget path, param-group bias/norm exclusion (twice: default and pre-frozen params), JSONL logger. | 1.23 s |

**Full suite**: `PYTHONPATH= /home/kaelin/anaconda3/envs/MLWS/bin/python -m pytest tests/ -q` → **49 passed in ≈231 s** (verified live; almost entirely `test_val_stratification`'s wall-clock, see above).

---

## Further reading

- `docs/TOOLING.md` — every CLI tool: full flag reference, output artifacts file-by-file, gates/exit codes, operational notes (GPU exclusivity, spawn-DataLoader rationale, val-set bit-identity scope), and a guide to reading the `introspect.py` panels.
- `paper/PAPER.md` — the full research-paper writeup: problem statement, method with every derivation, the synthetic data engine, the ablation matrix (conv-only trunk, RoPE variants, skip-gate scope, GT labelling policy, differencing augmentations, affine-only warps, loss formulation, input resolution, refiner corpus blend) with claims and protocols, the deployment application, and every measured number with its provenance.

---

## References

[Chen2021] Chen et al., "TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation," 2021. arXiv:2102.04306
[Ronneberger2015] Ronneberger, Fischer, Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," 2015. arXiv:1505.04597
[Yu2016] Yu, Koltun, "Multi-Scale Context Aggregation by Dilated Convolutions," 2016. arXiv:1511.07122
[Odena2016] Odena, Dumoulin, Olah, "Deconvolution and Checkerboard Artifacts," Distill, 2016. distill.pub/2016/deconv-checkerboard
[Vaswani2017] Vaswani et al., "Attention Is All You Need," 2017. arXiv:1706.03762
[Su2021] Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding," 2021. arXiv:2104.09864
[Heo2024] Heo et al., "Rotary Position Embedding for Vision Transformer," ECCV 2024. arXiv:2403.13298
[Oktay2018] Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas," MIDL 2018. arXiv:1804.03999
[Law2018] Law, Deng, "CornerNet: Detecting Objects as Paired Keypoints," ECCV 2018. arXiv:1808.01244
[Zhou2019] Zhou, Wang, Krähenbühl, "Objects as Points," 2019. arXiv:1904.07850
[Hu2019] Hu, DeTone, Malisiewicz, "Deep ChArUco: Dark ChArUco Marker Pose Estimation," CVPR 2019. arXiv:1812.03247
[Sun2018] Sun et al., "Integral Human Pose Regression," 2018. arXiv:1711.08229
[Collins2014] Collins, Bartoli, "Infinitesimal Plane-Based Pose Estimation," IJCV 2014.
[Luo2016] Luo et al., "Understanding the Effective Receptive Field in Deep Convolutional Neural Networks," NeurIPS 2016. arXiv:1701.04128
