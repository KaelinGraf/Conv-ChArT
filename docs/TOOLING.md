# ConChArT — Tooling Reference

Full CLI reference for every script under `tools/`. All commands assume `cwd = dense deep charuco/` (quote the space), the `MLWS` conda environment, and `PYTHONPATH=` cleared (see the root `README.md`'s Environment section for why both matter). Every tool inserts the project root onto `sys.path[0]` itself and must be invoked as a script — `python tools/<name>.py ...` — never `import tools.*` or `python -m tools.<name>` (the machine-wide `tools`-package shadow from an unrelated `detectron2` checkout). Argparse in every tool runs before any heavy import, so `--help` never needs `dcc`/`numpy`/`cv2`/`torch`/`matplotlib` and is safe to run in any environment with a bare Python 3.

Eight tools exist on disk. `tools/eval.py` and `tools/curves.py` (M-01..M-06 sign-off evaluation and a metrics-to-PNG curve plotter) are **pending** — neither file exists yet.

---

## `tools/audit.py` — acceptance gate

**Purpose.** The formal acceptance gate for the synthetic-data generator: overlay sheets for human inspection, distribution checks, a warp round-trip regression tripwire, byte-identical repeatability, and a refiner-stream content check. Exits **0** iff every gate passes, **1** if any gate fails, **2** if the background corpus is missing or empty (so a CI/orchestrator can distinguish "the generator is broken" from "the corpus hasn't downloaded yet").

**Usage**
```
tools/audit.py [-h] [--config CONFIG] [--out OUT] [--n-dist N_DIST]
               [--n-overlay N_OVERLAY] [--n-roundtrip N_ROUNDTRIP] [--save]
```

| Flag | Default | Meaning |
|---|---|---|
| `--config` | `configs/default.yaml` | Config to audit. |
| `--out` | `audit/` | Output directory (created if absent). |
| `--n-dist` | 10000 | Sample count for the distribution gates (s_px octaves, negative fraction, occlusion incidence) — matches `synth.val_size` by convention; using a much smaller `n` can transiently fail the negative-fraction gate on sampling noise alone (the gate is calibrated for n≈10,000). |
| `--n-overlay` | 200 | Sample count for the eyeball overlay sheets. |
| `--n-roundtrip` | 1000 | Sample count for the warp round-trip check. |
| `--save` | off | Also materialise the distribution pass to `val_set.npz` (images + records) — off by default since it duplicates the fixed val set's images to disk. |

**Example**
```
PYTHONPATH= /home/kaelin/anaconda3/envs/MLWS/bin/python tools/audit.py --config configs/default.yaml --out audit/
```

**The seven gates** (each printed as `PASS: <name>` / `FAIL: <name>`, and recorded under `report.json`'s `"gates"` dict):

1. **s_px octave flatness** — the `SynthVal`-equivalent sample stream's square-size distribution across `[16,32) [32,64) [64,128]` must each hold within 20% (relative) of an equal 1/3 share.
2. **negative fraction** — measured fraction of `board_present=false` samples within ±0.005 of `cfg["negative_p"]` (0.05).
3. **occlusion incidence** — measured fraction of samples with at least one rectangular hole within ±0.05 (absolute) of `cfg["synth"]["occlusion"]["p"]` (0.4).
4. **round-trip max px** — for `--n-roundtrip` samples, an independently re-derived 3×3 homography (rebuilt from the reported affine + perspective *components* alone, not by calling `dcc.synth`'s own matrix-construction code) must reproduce the stored corner coordinates to **< 0.01 px** — this is a regression tripwire against a convention slip (column order, dehomogenisation, the perspective conjugation point), not a check of `dcc.synth` against itself.
5. **repeatability (subprocess byte-identical)** — the first 50 `SynthVal` samples (image bytes + a canonical `repr` of the label record) must hash identically between the current process and a **fresh subprocess** re-running the same seed — catches any accidental dependence on process-local state (e.g. a cache that isn't purely a function of the seed).
6. **refiner d histogram** — the jitter-offset histogram on the canonical `RefinerVal` set (default seed `val_seed+1`) must be flat within 25% (relative) per bin over `[-3.9375, 3.9375]`, both axes.
7. **refiner content check** — 100 refiner crops (photometrics off, an independent `val_seed+2` stream) re-refined with `cv2.cornerSubPix` must agree with the recorded ground-truth offset at **median ≤ 0.10 px and p90 ≤ 0.50 px** (gated on median + p90, deliberately **not** max: `cornerSubPix` — the check's *instrument*, not the data under test — is ill-conditioned on small-`s` crops and high-shear/rotation "knife-edge" wedges, so its own error tail is heavy even on perfectly correct crops; a systematic labelling bug would shift the *median*, which a bad-conditioning tail would not).

**Output artifacts** (all under `--out`):

| File | Contents |
|---|---|
| `overlay_00.png`, `overlay_01.png`, … | 25-per-tile overlay sheets (corner dots green=visible/red=invisible, index labels, hole rectangles), `⌈n_overlay/25⌉` files. |
| `dist_s_px.png` | Log-scale histogram of `s_px`, titled with the measured per-octave shares. |
| `dist_visible.png` | Histogram of visible-corner count per sample (0–16). |
| `val_set.npz` | Only with `--save`: stacked images + JSON-per-record label array. |
| `refiner_d_hist.png` | Two histograms, jitter dx and dy over the ±3.9375 px support. |
| `report.json` | Everything: the full config used, all distribution numbers, `roundtrip_max_px`, `repeatability_ok`, the refiner report (crop count, per-bin fractions, content median/p90/max), library versions (`numpy`/`cv2`/`skimage`), a SHA1 of the sorted background file list (`backgrounds_sha1` — the version+corpus-hash pair that scopes any bit-identity claim), and the `gates` pass/fail dict. |

**Operational notes.** The repeatability gate spawns `sys.executable -c <code>` rather than importing `tools.audit` in the subprocess, because the machine-wide `tools`-package shadow makes `from tools.audit import ...` unreliable in a fresh interpreter — the hashing logic is intentionally duplicated (`_hash_pair` / `_REPRO_CODE`) rather than imported, with a comment at the duplication site so the two copies are kept in sync deliberately, not by accident.

---

## `tools/view.py` — sample viewer

**Purpose.** Eyeball detector or refiner training samples and their rendered ground-truth targets, as a saved sheet or an interactive stepper — for looking at the data, not gating it (that's `audit.py`).

**Usage**
```
tools/view.py [-h] [--config CONFIG] [--stream {detector,refiner}] [--n N]
              [--seed SEED] [--index INDEX] [--out OUT] [--show] [--channels]
```

| Flag | Default | Meaning |
|---|---|---|
| `--stream` | `detector` | `detector` (whole composites) or `refiner` (jittered corner crops). |
| `--n` | 16 | Sample count (rows in the sheet), unless `--index` is given. |
| `--seed` | 1000 | Val-set seed. |
| `--index` | none | View exactly `SynthVal[index]` (detector) or the composite at this index (refiner) — pins `n` internally to `cfg.synth.val_size` in detector mode so the stratified-`s` assignment matches the *canonical* val set, not an arbitrary `n`. |
| `--out` | `sheet.png` | Output sheet path. |
| `--show` | off | Also open an interactive matplotlib window (`n`/`p` to step, `q` to quit). |
| `--channels` | off | Detector stream only: also save a 4×4 grid of the 16 individual class-target channels, per sample. |

**Examples**
```
PYTHONPATH= /home/kaelin/anaconda3/envs/MLWS/bin/python tools/view.py --stream detector --n 16 --out sheets/detector_sheet.png --channels
PYTHONPATH= /home/kaelin/anaconda3/envs/MLWS/bin/python tools/view.py --stream refiner  --n 16 --out sheets/refiner_sheet.png
PYTHONPATH= /home/kaelin/anaconda3/envs/MLWS/bin/python tools/view.py --stream detector --index 7 --out sheets/detector_idx7.png --channels --show
```

**Output artifacts.**
- `{out}` — the main sheet: one row per sample. Detector rows are 3 panels (composite + corner overlay | full-res GT heatmap, JET alpha-blend | class-target max-over-16, upsampled ×4, JET alpha-blend). Refiner rows are 2 panels (crop zoomed ×8 with a cross at the true sub-pixel offset | the 64×64 target).
- `{out_stem}_ch{i}.png` — only with `--channels` on the detector stream: **one file per requested sample**, a 4×4 grid of that sample's 16 individual class channels. The `{i}` suffix is the *sample's index* (e.g. `detector_idx7_ch7.png` is sample index 7's channel grid, not "channel 7") — verified against the shipped example in `sheets/detector_idx7_ch7.png`.
- With `--show`: a live matplotlib window over the sheet's rows (and, with `--channels`, a second window over the channel grids), `n`/`p` to step, `q` to close both.

---

## `tools/gen_eval_pose.py` — pose-consistent evaluation set

**Purpose.** The **only** place in this codebase where camera pose `(K, R, t)` bookkeeping exists. Generates images with an explicitly sampled pinhole intrinsic and board pose, for the M-06 pose-error metric (`paper/PAPER.md` §5.2).

**Usage**
```
tools/gen_eval_pose.py [-h] [--config CONFIG] [--out OUT] [--n N] [--sheet-n SHEET_N]
```

| Flag | Default | Meaning |
|---|---|---|
| `--out` | `eval_pose/` | Output directory. |
| `--n` | 1000 | Image count. |
| `--sheet-n` | 20 | How many of the first `n` images also go into the overlay sheet. |

**Example**
```
PYTHONPATH= /home/kaelin/anaconda3/envs/MLWS/bin/python tools/gen_eval_pose.py --config configs/default.yaml --out eval_pose/ --n 1000
```

**Method, briefly.** Per image $i$, `rng = default_rng([pose_seed, i])` draws a focal length $f\sim U(0.7,1.4){\cdot}W$, a target apparent scale $s$ (log-uniform on `scale_range_px`, giving depth $z=f/s$), an in-plane rotation, a tilt (≤60°) about a random axis, and a small centre-of-projection offset; the pose is accepted once ≥8 of the 16 lattice corners project in-frame and all 16 are in front of the camera (up to 100 resample tries — a config pathology raises `RuntimeError` rather than silently degrading). The induced homography $H = K\,[r_1\ r_2\ t]\,S$ (where $S$ maps a render-pixel homogeneous coordinate to board metres) is asserted against `cv2.projectPoints` of the same lattice to **< 1e-6 px** on *every* image — a hard `assert`, not a soft gate, since a convention slip here (column order, dehomogenisation, the metre-offset in $S$) would silently corrupt every recorded pose. Everything downstream of the pose draw (background prep, occlusion, photometrics, geometric visibility) reuses `dcc.synth`'s exact machinery.

**Output artifacts.**
- `images/{i:06d}.png` — `n` grayscale composites.
- `labels.jsonl` — one JSON object per line: `file`, `K`, `R`, `t` (as nested lists), `square_length_m` (fixed at 1.0 — the config's `board.square_length_m` is unused here), `s_px`, and the 16-entry `corners` list (`x`, `y`, `index`, `visible`).
- `overlay_sheet.png` — the first `--sheet-n` images tiled with corner overlays.
- `meta.json` — `pose_seed`, `n`, library versions, `backgrounds_sha1`, the full config, `resample_tries` (`max` and how many images needed more than one try), and `s_matrix_assert_max_px` (the worst-case homography-identity error actually observed, always ≪ 1e-6 by construction).

**Note.** At `n=1000` the output is roughly 0.3 GB.

---

## `tools/gen_cutouts.py` — offline SAM2 cutout-bank builder

**Purpose.** Builds the reusable RGBA cutout bank that `dcc.synth`'s realistic object-occlusion mode (`_apply_cutouts` / `load_cutouts`) reads back at generation time. Run **once**, ahead of time — the generator only ever reads the bank directory; nothing in the training loop invokes SAM2.

**Run under the `eomt` conda environment, never `MLWS`** — the two environments carry incompatible torch builds and must never be imported in the same interpreter. This script imports only `cv2`/`numpy`/`torch`/`sam2`/stdlib and never touches `dcc`.

**Usage**
```
tools/gen_cutouts.py [-h] [--coco COCO] [--out OUT] [--ckpt CKPT] [--sam-config SAM_CONFIG]
                     [--n-images N_IMAGES] [--max-per-image MAX_PER_IMAGE] [--seed SEED]
                     [--min-area-frac MIN_AREA_FRAC] [--max-area-frac MAX_AREA_FRAC] [--force]
```

| Flag | Default | Meaning |
|---|---|---|
| `--coco` | `/home/kaelin/datasets/coco/train2017` | Source image directory (`.jpg` only). |
| `--out` | `/home/kaelin/datasets/cutouts` | Cutout-bank output directory; refuses to write into a non-empty directory unless `--force`. |
| `--ckpt` | `.../sam2_ckpts/sam2.1_hiera_base_plus.pt` | SAM2 checkpoint. |
| `--sam-config` | `configs/sam2.1/sam2.1_hiera_b+.yaml` | SAM2 model config name. |
| `--n-images` | 3000 | How many source images to sweep (randomly chosen via `--seed`). |
| `--max-per-image` | 8 | Cap on kept masks per image (by descending `predicted_iou`). |
| `--min-area-frac`, `--max-area-frac` | 0.005, 0.25 | Accepted mask area as a fraction of the source image. |
| `--force` | off | Allow writing into a non-empty `--out`. |

**Example**
```
/home/kaelin/anaconda3/envs/eomt/bin/python tools/gen_cutouts.py \
    --coco /home/kaelin/datasets/coco/train2017 --out /home/kaelin/datasets/cutouts --n-images 3000
```

**Mask filter** (`_accept`): area fraction within range, `predicted_iou ≥ 0.85`, `stability_score ≥ 0.9`, bounding box clears every image border by ≥ 2 px (drops border-clipped objects with artificial straight edges), and solidity — mask pixel area over its convex-hull area — `≥ 0.4` (drops thin wire-frame/hollow masks).

**Output artifacts.**
- `{out}/{count:07d}.png` — one RGBA PNG per kept mask, save-order numbered; RGB channels are the source-image crop, alpha is the mask × 255.
- `{out}/manifest.json` — the run's `params`, `n_images_swept`, `n_cutouts`, the sorted `files` list, and a `files_sha1` of that list.

**Operational notes.** ~1–2 s/image on an RTX 5090 (SAM2 automatic mask generation), so a full 3,000-image sweep takes roughly 1–1.5 h; smoke-test with a small `--n-images` first. Exits 1 if `--out` is non-empty without `--force`, or if CUDA is unavailable (`SAM2AutomaticMaskGenerator` requires a GPU). **The default sweep has run to completion on this machine**: `/home/kaelin/datasets/cutouts` holds $14{,}721$ RGBA cutout files from the full $3{,}000$-image sweep (`n_images_swept: 3000` in `manifest.json`), with the manifest fully self-consistent against the directory contents (every listed file present, zero missing) — averaging $\approx4.9$ cutouts per swept image. `dcc.synth.load_cutouts` still returns `[]` for a missing or partial bank directory and `synth.cutouts.p` remains a no-op in that case (every other tool, test, and training run proceeds correctly without object-cutout occlusion), but that fallback is no longer the operative case here.

---

## `tools/preflight.py` — pre-training verification suite

**Purpose.** A mandatory sanity pass to run before every long training run (including every ablation): proves a freshly constructed, untrained `DetectorNet`/`Refiner` pair's numerical behaviour matches the design's own init-time invariants, and that the full data→targets→loss→optimiser→heatmap→peaks→readout→gate loop can express a known answer — in minutes, catching a wiring defect that would otherwise only surface hours into a wasted multi-day run. Argparse runs before any heavy import, so `--help` never needs `torch`/`dcc`/`cv2`.

**Usage**
```
tools/preflight.py [-h] [--config CONFIG] [--out OUT] [--quick] [--device {cuda,cpu}]
```

| Flag | Default | Meaning |
|---|---|---|
| `--config` | `configs/default.yaml` | Config to check against. |
| `--out` | — | Report output location. |
| `--quick` | off | Skip the one-batch-overfit check (the sole check that actually trains the pair in place; every other check is a pure init-time characterisation — no weights are ever updated, since backward populates `.grad` but no optimiser `.step()` is ever taken outside this one check). |
| `--device` | `cuda` if available | `cuda` or `cpu`. |

**The seven checks**, each targeting one specific failure mode named in this project's own pre-flight verification discipline (`paper/PAPER.md` Appendix E): predicted-vs-observed initial loss, translation equivariance of the attention module, RoPE relativity/no-global-alias, gradient balance across the heatmap/class/attention/gate paths, the refiner's zero-offset readout closure, bf16-vs-fp32 numerical parity, and — the one check that actually trains the pair, run last regardless of its position above, and the only one `--quick` skips — one-batch overfit capability. Off-CUDA, the one-batch-overfit check is unconditionally skipped (it is not meaningful without the training precision path this system targets) and the bf16-parity check is attempted and gracefully warns rather than fails on any exception, since bf16 semantics are itself a CUDA/Tensor-Core concern.

**Example**
```
PYTHONPATH= /home/kaelin/anaconda3/envs/MLWS/bin/python tools/preflight.py --config configs/default.yaml --out preflight_report.json
```

---

## `tools/train_detector.py` — Stage-1 (detector) training

**Purpose.** Stage-1 training loop: AdamW over micro-batch × grad-accumulation steps, bf16 autocast + `channels_last`, cosine LR, EMA, in-loop validation (M-01, M-02, M-04, plus gate-α and attention-entropy diagnostics), periodic checkpointing.

**Usage**
```
tools/train_detector.py [-h] [--config CONFIG] [--name NAME] [--resume RESUME]
                        [--steps STEPS] [--freeze-trunk]
```

| Flag | Default | Meaning |
|---|---|---|
| `--name` | `run` | Run directory: `runs/<name>/`. |
| `--resume` | none | Checkpoint `.pt` to resume from — restores model/EMA/optimiser/RNG and continues `step`/`resume_count`. |
| `--steps` | `cfg.train.steps` | Override the configured step budget (useful for smoke runs). |
| `--freeze-trunk` | off | Intended to train only the class head (board retarget) — **see the caveat below before relying on this.** |

**Examples**
```
PYTHONPATH= /home/kaelin/anaconda3/envs/MLWS/bin/python tools/train_detector.py --config configs/default.yaml --name run1
PYTHONPATH= /home/kaelin/anaconda3/envs/MLWS/bin/python tools/train_detector.py --config configs/default.yaml --name run1 --resume runs/run1/ckpt_0025000.pt
PYTHONPATH= /home/kaelin/anaconda3/envs/MLWS/bin/python tools/train_detector.py --config configs/default.yaml --name smoke --steps 30
```

**Output artifacts.**
- `runs/<name>/metrics.jsonl` — one JSON object per line, appended, never truncated: step-level `{step, loss, lr, grad_norm, samples_per_s}`, and validation-cadence records `{step, val: {val_loss, m01, m02, m04, diag}}` / `{step, full_val: {...}}` (same inner shape). `m01 = {mean, median, p95, tail_frac_gt4px, n_matched}` (greedy NN-match within `cfg.train.match_px`, tail defined against the fixed `TAIL_PX = 4.0` constant — the refiner's own capture range, not a config key). `m02` = match ratio per octave `{"16-32", "32-64", "64-128"}`. `m04` = ID accuracy overall and by octave, `None` if `dcc.pipeline.read_ids` wasn't importable at the time (one-time warning, not a crash). `diag` = `{gate3_alpha_mean, gate3_alpha_min, gate4_alpha_mean, gate4_alpha_min, block0_attn_entropy, block1_attn_entropy}`.
- `runs/<name>/ckpt_{step:07d}.pt` — written at `full_val_every` cadence (default every 25,000 steps) and at the final step; see the root README's checkpoint contract.

**Metrics, in more detail.**
- **M-01 / M-02** use `_extract_peaks` (3×3 max-pool-equality ≥ `tau_hm` — the CornerNet/CenterNet peak-decode convention [Law2018] [Zhou2019]) and greedy nearest-neighbour matching, independent of (but algorithmically identical in spirit to) `dcc.pipeline.peaks`/`merge_close` — the training loop's own copy has no `top_k` cap or plateau-tie de-duplication, which is fine at the peak counts observed in practice but is a documented latent difference from the Stage-3 pipeline's more defensive version.
- **Gate-α and attention-entropy diagnostics** are read via forward hooks that *recompute* the signal from `dcc.model`'s own public methods, never by reimplementing the gate/attention math: `AttnGate.alpha(skip, g)` for gate-α (cheap — safe across a whole validation batch), and `Block.n1`/`qkv_heads` for attention entropy. The entropy hook is deliberately **only ever attached around a single fp32 image**, never a batched sweep: materialising the full `(heads, T, T)` attention matrix at `T≈7,500` tokens (native resolution) across a real validation batch allocates tens of GB and will OOM the entire training run — this is wrapped in `try/except torch.cuda.OutOfMemoryError` with a one-time warning as a second line of defence, since a crashed diagnostic must never take down a real multi-hour run.

**Operational notes.**
- **GPU exclusivity.** Asserts `torch.backends.cuda.flash_sdp_enabled()` and `mem_efficient_sdp_enabled()` at startup — the math-backend SDPA fallback would materialise an attention matrix costing ~11.8 GB per block at `B=32`, `T≈4,800` (640×480 variant), an instant OOM. **ollama VRAM note**: at native resolution (`B=4`, the measured ceiling — `B=8` OOMs needing ~41.7 GiB), peak training usage is ~20.9–21.7 GiB, leaving only ~8.4 GiB of headroom against a resident ollama instance; drop to `B=2` (same ~25.6 samples/s, per the native-res measurement) if ollama's reservation exceeds that. Check `nvidia-smi` before a real run.
- **spawn-context DataLoader.** `multiprocessing_context="spawn"`, `persistent_workers=True`, and `cv2.setNumThreads(1)` in each worker's `worker_init_fn` are pinned in code (not config): the `fork` context measured **7× lower** throughput (55 vs 404 samples/s at 640×480) in this project's own benchmarking, almost certainly from OpenCV's internal thread pool surviving `fork()` in a broken state.
- **Worker oversubscription.** 12 train + 8 val + 8 full-val = 28 persistent worker processes on a 24-core machine — a direct, documented consequence of the pinned per-loader worker counts, not a bug; the GPU is the binding constraint either way (projected generation throughput exceeds GPU consumption by ≥3× at both resolutions this project has measured).
- **Resume semantics.** `stream_seed = train_seed·1000 + resume_count`, and `resume_count` increments on every `--resume` — a resumed run never replays the identical training-sample sequence it would have seen had it not stopped. Two independent resumes from the *same* checkpoint (not chained) intentionally reproduce each other bit-for-bit (same `resume_count ⇒` same seed) — this is the resume-determinism guarantee, not a bug.
- **`samples/s` excludes validation wall-time** from its denominator throughout (both training scripts) — otherwise every step's logged throughput after the first validation would be diluted by all past validation time, since the numerator counts only training samples.
- **Val-set bit-identity is scoped to a code + library version triple.** `SynthVal`/`RefinerVal` are deterministic functions of `(val_seed, index)` and the *current* `dcc/synth.py` — bit-identity across two runs holds only for a fixed `{numpy, cv2, skimage}` version set and an unchanged generator implementation; `tools/audit.py`'s `report.json` records the exact versions and a corpus SHA1 that scope any such comparison, and a resolution/config change (e.g. the 640×480→1600×1200 migration this project has already been through) invalidates prior "fixed" validation numbers even though the *code path* computing them is unchanged.
- **Corrected — `--freeze-trunk` / `cfg.freeze_trunk` now correctly gates which parameters train.** At `tools/train_detector.py:335-347`:
  ```python
  freeze = args.freeze_trunk or cfg.get("freeze_trunk", False)
  model.train()
  if freeze:
      for name, p in model.named_parameters():
          p.requires_grad_(name.startswith("cls."))
      ...    # frozen BatchNorm layers switched to eval() in the same block
  ```
  the `requires_grad_` loop now sits **inside** the `if freeze:` guard, alongside the BatchNorm eval-mode switch. A default (`freeze_trunk: false`) invocation trains the full network; only an explicit `--freeze-trunk`/`freeze_trunk: true` retarget request restricts training to `cls.*`, exactly as the model-level contract in the root README's Conventions section describes. This is a corrected defect, not the original behaviour — worth recording briefly as methodology: an earlier version of this entry point set the `requires_grad_` mask unconditionally, so *every* invocation — including the default full-network configuration used to produce the smoke-run evidence below — restricted gradient updates to the class head's own two convolutions (88 other parameter tensors, `e1..e5, blocks, norm, gate3, gate4, d1..d4, hm`, ≈97.9% of the network, sat at their random initialisation regardless of the flag). The shipped `runs/smoke/metrics.jsonl` was captured under that earlier, unconditional version: `m01.n_matched` reads `0` and `gate3_alpha_mean`/`gate4_alpha_mean` sit exactly at the pass-through-init value (0.953125) at every logged validation, across both the fresh run and its resume — those numbers describe a run made before the fix, not evidence against the corrected code. The defect was caught during this project's own documentation-and-verification review, after three earlier acceptance passes had missed it, rather than by a training-loop smoke test — a loss that decreases and a run that completes without crashing are both still true when only 2.1% of a network is actually learning, which is exactly why an independent review pass matters.

---

## `tools/train_refiner.py` — Stage-2 (refiner) training

**Purpose.** Stage-2 training loop — a separate model (`dcc.model.Refiner`), separate optimiser/schedule, separate `runs/<name>/` and checkpoint. Board-agnostic, so there is no `freeze_trunk`/octave concept here.

**Usage**
```
tools/train_refiner.py [-h] [--config CONFIG] [--name NAME] [--resume RESUME] [--steps STEPS]
```

Flags mirror `train_detector.py`'s `--config`/`--name`/`--resume`/`--steps` (overriding `cfg.refiner_train.steps`); there is no `--freeze-trunk` equivalent.

**Example**
```
PYTHONPATH= /home/kaelin/anaconda3/envs/MLWS/bin/python tools/train_refiner.py --config configs/default.yaml --name run1
```

**Output artifacts.** Same `runs/<name>/{metrics.jsonl, ckpt_{step:07d}.pt}` shape as the detector; checkpoints are written every `refiner_train.val_every` steps (default 1,000 — much more frequent than the detector's 25,000, since a refiner step is far cheaper). Validation records carry `{val_loss, m03: {mean, median, p95, n}, bias_vs_jitter: {x: {...}, y: {...}}}` — `m03` is the L2 readout error in 1/8-px units on the fixed refiner validation stream; `bias_vs_jitter` is the mean *signed* error per axis, binned by the true jitter magnitude (`[-4,-2) [-2,0) [0,2) [2,4)`) — the concrete form of the "verify no bias vs. the jitter distribution" acceptance check.

**Operational notes.** No grad accumulation (`refiner_train` has no `accum` key — batch 256 at 24×24 px is cheap enough to be the full per-step batch). Readout uses `dcc.pipeline.soft_argmax` when importable — a probability-weighted centroid in the spirit of integral pose regression [Sun2018] — a local fallback with the *identical* algorithm (5×5 window around the hard argmax, border-clamped) covers the case it isn't, since M-03 is this script's own headline metric and can't be silently skipped the way `train_detector.py` skips M-04.

---

## `tools/introspect.py` — introspection & visualisation

**Purpose.** Six presentation-grade panels from a single forward pass over one sample: end-to-end pipeline output, a 3D heatmap landscape, bottleneck-attention maps, decoder skip-gate maps, an M-07 effective-receptive-field probe, and encoder feature maps. With no `--ckpt`, every panel runs on a freshly-initialised (untrained) `DetectorNet`, and every figure title says so.

**Usage**
```
tools/introspect.py [-h] [--config CONFIG] [--ckpt CKPT] [--refiner-ckpt REFINER_CKPT]
                    [--ckpt-b CKPT_B] [--index INDEX] [--image IMAGE] [--out OUT]
                    [--panels PANELS] [--query-xy X,Y] [--gif] [--dpi DPI] [--show]
```

| Flag | Default | Meaning |
|---|---|---|
| `--ckpt` | none | Detector checkpoint (accepts a raw `state_dict`, or a trainer checkpoint dict — `ema` preferred over `model` if both are present). Absent ⟹ untrained `DetectorNet`. |
| `--refiner-ckpt` | none | Refiner checkpoint for the `pipeline` panel. Absent ⟹ untrained `Refiner`. |
| `--ckpt-b` | none | A second detector checkpoint for the `erf` panel's side-by-side ablation compare (e.g. full model vs. a conv-only `attn_blocks=0` ablation), sharing one colour scale. |
| `--index` | 0 if `--image` absent | `SynthVal[index]`. |
| `--image` | none | Use a raw grayscale/colour file instead of a `SynthVal` sample (resized to `cfg.input_size` with `INTER_AREA`). |
| `--panels` | `all` | Comma list from `pipeline,heatmap3d,attention,gates,erf,features`. |
| `--query-xy` | strongest peak | `X,Y` in input-pixel coordinates; steers the attention/erf/heatmap3d query point. Default is the heatmap's strongest peak (attention, heatmap3d) or the class head's global `(channel, cell)` argmax (erf). |
| `--gif` | off | `heatmap3d` only: also render a 72-frame rotating-azimuth GIF. |
| `--dpi` | 160 | Figure DPI. |
| `--show` | off | Also open interactive windows after saving. |

**Examples**
```
PYTHONPATH= /home/kaelin/anaconda3/envs/MLWS/bin/python tools/introspect.py --index 7 --out introspect_out/
PYTHONPATH= /home/kaelin/anaconda3/envs/MLWS/bin/python tools/introspect.py \
    --ckpt runs/run1/ckpt_0250000.pt --refiner-ckpt runs/run1_refiner/ckpt_0010000.pt \
    --index 7 --panels all --gif --out introspect_out/
PYTHONPATH= /home/kaelin/anaconda3/envs/MLWS/bin/python tools/introspect.py \
    --ckpt runs/A1_full/ckpt_0250000.pt --ckpt-b runs/A1_convonly/ckpt_0250000.pt \
    --panels erf --index 7 --out ablation_A1/
```

**Output artifacts** (all under `--out`; `{tag}` = `val{index}` for a `SynthVal` sample or the image filename stem for `--image`):

| File | Panel | Contents |
|---|---|---|
| `pipeline_{tag}.png` | `pipeline` | 5 panels: input · predicted-heatmap overlay · 3D heatmap surface · class-map-max overlay · corners+pose, via the **real** `dcc.pipeline.detect()` (falls back to heatmap-peaks-only, annotated, if `dcc.pipeline` is unavailable). |
| `heatmap3d_{tag}.png` (+ `.gif` with `--gif`) | `heatmap3d` | Full-frame 3D heatmap surface + a 96 px zoomed window at the query point; the GIF is a 72-frame azimuth rotation of a coarser-stride rebuild of the full-frame surface (the static PNG keeps full detail — a dense-stride 3D surface re-rasterised 72× hangs mplot3d's renderer). |
| `attention_{tag}.png` | `attention` | Per bottleneck block: a mean-over-heads attention map (reshaped to the H/16 token grid, overlaid on the image, entropy in nats) plus one panel per head, cyan crosshair at the query token. |
| `gates_{tag}.png` | `gates` | `gate3.alpha` and `gate4.alpha`, upsampled to full resolution, viridis, colourbar ticked "0 = vetoed" / "1 = passed". |
| `erf_{tag}.png` | `erf` | M-07 probe: `log10(|∂(class logit)/∂(input)| + ε)`, magma overlay; two panels side by side (shared colour scale) if `--ckpt-b` is given. |
| `features_{tag}.png` | `features` | `e1..e5` mean-`|`activation`|` per encoder stage, each independently normalised, viridis. |

Colourmaps are **viridis/magma only, never jet**, throughout — deliberately colourblind-safe for conference presentation. Panels degrade gracefully (`print("SKIP <panel>: ...")` and return) rather than crash if the checkpoint's architecture lacks an expected attribute (e.g. no `.blocks`, no `.gate3`/`.gate4`) — instrumentation around a model contract that could still change, not an acceptance gate. Exits 3 (after printing a message) if `dcc.model` itself isn't importable; `--help` is unaffected either way.

---

## Reading the introspection panels

**Attention plaid at init, and its entropy.** An untrained network's bottleneck attention is close to uniform — a flat, structureless "plaid" over the H/16 token grid — because there is no signal yet for the query token to prefer. Entropy $H=-\sum_i p_i\log p_i$ over the attention row quantifies this directly: uniform attention over $T$ tokens gives $H=\ln T$, and at native resolution ($T=7{,}500$) that is $\ln 7500\approx 8.92$ nats — exactly what both blocks report on a freshly-constructed network (verified: `attention_val7.png`'s untrained baseline reads $H=8.92$ for both blocks, matching the analytic maximum to four figures). As training proceeds, watch this number fall and the map concentrate onto legible markers relative to the query corner: the ground-truth labelling policy (photometrically-degraded corners keep their *true* index, §4.4 of `paper/PAPER.md`) makes the class loss on illegible corners reducible **only** through this attention pathway, so a healthy run's entropy should trend down over training and differ block-to-block (the reference prototype benchmark saw $8.48\to7.09$ nats on block 2 over 300 steps on noise images — the anti-collapse pressure visibly working). A model whose entropy never moves off $\ln T$ despite substantial training is the signature of **attention collapse** — and critically, the ERF panel alone cannot tell collapsed attention apart from healthy attention, since uniform attention *also* produces a frame-spanning gradient footprint. Track entropy per block, per checkpoint (`train_detector.py` logs `block0_attn_entropy`/`block1_attn_entropy` automatically), not just once.

**ERF: reach vs. mass.** The `erf` panel backpropagates a unit gradient from one class-head logit to the input and plots $\log_{10}|{\cdot}|$ — this is the M-07 probe [Luo2016]. Two properties matter separately: **reach** (how far from the query point does the nonzero-gradient region extend) and **mass** (how much of the total gradient concentrates near the query vs. spreads thinly across the whole frame). The conv-only path's theoretical receptive field is ≈164 px, but its *effective* RF — where most of the gradient mass actually sits — is a much smaller Gaussian fraction of that (σ≈31 px, usable radius ~31–62 px); a conv-only ablation (`attn_blocks=0`, ablation A1) is expected to reproduce exactly this bounded-reach, locally-concentrated pattern. The full hybrid model, once its attention is actually engaged (not collapsed — see above), should show reach spanning the *entire frame*: after the bottleneck blocks there are only three convolutions and one bilinear upsample between attention output and the class logits, none of which can destroy global content. `--ckpt-b` puts both side by side on one colour scale — this comparison is ablation A1's conference visual.

**Gate-α maps.** `gate3`/`gate4` are Oktay-style additive attention gates [Oktay2018] controlling how much of the H/4 (and, in the native-resolution variant, H/8) encoder skip reaches the decoder, conditioned on globally-informed context from beyond the bottleneck. At initialisation both read a uniform ≈0.953 (pass-through: `psi.weight=0`, `psi.bias=+3.0` ⟹ $\sigma(3)$, verified exactly in the shipped `gates_val7.png`) — solid colour, no structure, by design. During training, a gate closing (α → 0) over background is expected and healthy; the failure mode to watch for is a gate closing over genuine, still-legible board texture, since — unlike a heatmap veto — a class-path gate closing is in principle *recoverable* by the Stage-3 lattice-projection pass, but only if enough other corners survive to fit a homography. Cross-reference a suspicious gate map against the M-02 recall curve on the darkest brightness bins (ablation A3, variant (b) `gate_all_skips: true`, in `paper/PAPER.md` §5.4, is exactly this question made systematic).

**Feature taps (`e1..e5`).** Mean absolute activation per encoder stage, each independently normalised so depth is compared by *pattern* rather than absolute scale (deeper stages naturally carry different activation magnitudes). Expect a sharpening-to-blobbier progression: `e1` should track crisp local edges and marker texture, `e5` (post-pooling, post-dilated-convolution) should show smoother, larger-support blobs as the receptive field grows — a qualitative sanity check that the encoder is actually building the spatial hierarchy the architecture assumes, not a pass/fail gate.

---

## References

[Luo2016] Luo et al., "Understanding the Effective Receptive Field in Deep Convolutional Neural Networks," NeurIPS 2016. arXiv:1701.04128
[Law2018] Law, Deng, "CornerNet: Detecting Objects as Paired Keypoints," ECCV 2018. arXiv:1808.01244
[Zhou2019] Zhou, Wang, Krähenbühl, "Objects as Points," 2019. arXiv:1904.07850
[Sun2018] Sun et al., "Integral Human Pose Regression," 2018. arXiv:1711.08229
[Oktay2018] Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas," MIDL 2018. arXiv:1804.03999

See the root `README.md` for the full architecture reference list.
