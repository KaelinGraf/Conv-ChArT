# 27 — Zero-shot board transfer: the 882k release model on a board it was never trained on

Kaelin, 2026-10-08: "run a first test, where the target board is changed (without re-training the
model in any way) - and then run the full eval suite so we can see how it performs. I expect good
corner localisation but 0 id accuracy." This is the baseline for the fine-tuning / model-transfer
figures; fine-tuned arms are added as further bars to `transfer_summary.png` via `tools/plot_transfer.py`.

## Setup

- **Model**: `runs/rel_w882_c2_100k_rev6/ckpt_0100000.pt` (the 882k release detector) + the shared
  refiner `runs/ref_s15_10k_rev6/ckpt_0010000.pt`. Nothing retrained, nothing re-exported.
- **Target board**: `configs/transfer_dict6x6.yaml`, cut by `tools/cut_ablation.py` from the 882k
  config `configs/abl_c2_wh_clsfocal_lam2.yaml` with EXACTLY one key changed:
  `board.dictionary: DICT_5X5_50 -> DICT_6X6_250`. Same 5x5 squares, same 16 inner corners (verified
  identical corner coordinates), same `marker_ratio`; 15,217 of 230,400 render pixels differ (the
  marker interiors). Same geometry is REQUIRED for a zero-shot test: the class head has 16 channels,
  so a board with a different square count cannot be scored without retraining it (`--retarget-from`).
  `boards_dict5x5_vs_dict6x6.png` shows the two boards side by side.
- **Why this dictionary**: a different marker FAMILY (6x6 bits vs 5x5). `DICT_5X5_100/250/1000`
  would NOT have been a transfer -- their first 50 markers are identical to `DICT_5X5_50`
  (checked: `bytesList` equal for ids 0-11).
- **Eval suite**: the same three tools, seeds and sizes as the release numbers, so control and
  zero-shot are scored on identical frames up to the marker pixels: `tools/eval_checkpoint.py`
  (10k-sample full val), `tools/eval_pose_ours.py` (1000-frame B1 pose set, regenerated under each
  board config), `tools/robustness_sweep.py` (20 factors, n=60/step, seed 20260728).
- **Control reproduction gate passed**: the control full val reproduces the trainer's step-100k
  record exactly (median 0.4063, p95 0.6733, M-04 99.53%), and the regenerated control pose set
  reproduces `21_pose_REL882_B1.json` exactly (936/1000, 64 refused, same medians).

## Results

### 10k full validation (`fullval_control_dict5x5.json`, `fullval_transfer_dict6x6.json`)

| | M-01 median | M-01 p95 | tail >4 px | M-02 recall by octave (12-16 / 16-32 / 32-64 / 64-128) | M-04 ID accuracy |
|---|---|---|---|---|---|
| trained board | 0.4063 px | 0.6733 px | 0.0048% | 91.74 / 94.02 / 95.31 / 95.61 % | **99.53%** |
| zero-shot DICT_6X6_250 | 0.4063 px | 0.6754 px | 0.0056% | 91.94 / 93.97 / 95.47 / 95.71 % | **13.36%** (9.5 / 15.0 / 14.3 / 12.1 by octave) |

Localisation and recall transfer to within noise (123,856 vs 123,965 matched corners). Identity
does not: 13.4% raw class-head agreement against a 6.25% chance level for 16 classes. The head is
not blind on the new board -- it is above chance -- but it is wrong on 87% of corners.

### Pose, 1000-frame B1 set (`pose_zeroshot_dict6x6_B1.json`, per-image JSONL beside it)

| outcome per frame | trained board | zero-shot |
|---|---|---|
| correct pose (<2 deg, accepted) | 89.1% | 1.8% |
| flagged ambiguous (filter skips) | 4.3% | 2.9% |
| **wrong pose, accepted** | **0.2%** | **22.0%** |
| refused | 6.4% (too_few 60, collinear 2, pnp_solver_failed 2) | 73.3% (too_few 683, pnp_solver_failed 23, vacuous_uncorroborated 17, collinear 10) |

**The expectation "0 ID accuracy" undersells the failure.** Of the 267 zero-shot solves, 197
(73.8%) have a rotation error of ~90 deg with a translation error of 5.000 squares (one board
side) and 39 (14.6%) are ~180 deg; only 19 are correct. These are not noisy poses -- they are the
true pose composed with a symmetry of the lattice: a labelling that is a 90/180-degree
relabelling of the truth projects onto the canonical lattice EXACTLY, so the RANSAC lattice gate
accepts it, PnP solves it with a normal residual (accepted-wrong rms median 0.098 px vs 0.088 px
for correct control solves -- indistinguishable), and IPPE's ambiguity test does not fire because
it only covers the planar two-fold flip, not lattice relabellings. **On a foreign-dictionary
board the pipeline emits a confidently wrong pose on 22% of frames.** That is the Deep ChArUco
failure mode the gate exists to prevent, and it is the strongest argument for retraining the
class head before deployment on any new board.

### Robustness, 20 factors, identical frames (`robustness_dict6x6/`, `robustness_hardest_step.md`)

Mean over all factors and steps, refined arm: recall 91.1% vs 91.1%; ID accuracy 95.3% vs 3.9%.
The post-gate ID (what `detect()` reports) is BELOW the 13.4% raw head agreement because the
lattice gate demotes inconsistent labels -- correctly, but it cannot create identity it was never
given. Hardest-step table for every factor is in `robustness_hardest_step.md`; localisation
differs by <= 0.06 px on every factor, recall by <= 5 pp (distance_extrap at s=256, n small).

## Figures

- `transfer_summary.png` -- the report figure (`tools/plot_transfer.py`): localisation, ID/recall,
  and the per-frame pose outcome stack. Add fine-tuned arms with further `--arm` flags.
- `figures_vs_control/fourway_<factor>.png` (20) -- per-factor localisation / recall / ID curves,
  trained board vs zero-shot, both arms each (`tools/plot_four_way.py --fast-refined`, baselines
  suppressed via the empty `_no_baselines/`).
- `figures_vs_control/fourway_PANEL_*transfer_{recall,id,loc}.png` -- 2x2 overlay panels
  (distance, darkness, sensor noise, motion blur) for each metric. The `id` and `loc` grids were regenerated
  2026-10-09 after a `tools/plot_four_way.py` fix: the darkness override (found+ID, meant for the recall grid)
  had leaked into them, so their darkness panel showed found+ID instead of the grid's own metric.
- `figures_zeroshot_only/robustness_<factor>.png` (20) -- the zero-shot sweep's own coarse vs
  refined curves (`tools/plot_robustness.py`).

## Reproduce

```
P="env PYTHONPATH= DCC_TRAINER_METRICS= /home/kaelin/anaconda3/envs/MLWS/bin/python"
CK=runs/rel_w882_c2_100k_rev6/ckpt_0100000.pt; RF=runs/ref_s15_10k_rev6/ckpt_0010000.pt; D=paper/results_rev6/27_transfer_zeroshot
$P tools/cut_ablation.py --base configs/abl_c2_wh_clsfocal_lam2.yaml --out configs/transfer_dict6x6.yaml --set board.dictionary=DICT_6X6_250
$P tools/eval_checkpoint.py --ckpt $CK --config configs/transfer_dict6x6.yaml --out $D/fullval_transfer_dict6x6.json
$P tools/gen_eval_pose.py --config configs/transfer_dict6x6.yaml --out eval_pose_rev6_b1_dict6x6 --n 1000
$P tools/eval_pose_ours.py --pose-set eval_pose_rev6_b1_dict6x6 --ckpt $CK --refiner-ckpt $RF --allow-board-mismatch --out $D/pose_zeroshot_dict6x6_B1.json --per-image $D/pose_zeroshot_dict6x6_B1_per_image.jsonl
$P tools/robustness_sweep.py --config configs/transfer_dict6x6.yaml --ckpt $CK --refiner-ckpt $RF --n 60 --seed 20260728 --out $D/robustness_dict6x6
$P tools/plot_four_way.py --ours paper/results_rev6/20_robustness_REL882 --fast $D/robustness_dict6x6 --dc $D/_no_baselines --classical $D/_no_baselines --out-dir $D/figures_vs_control --ours-label "882k, trained board DICT_5X5_50" --fast-label "882k, zero-shot DICT_6X6_250" --fast-refined --panel-tag transfer_recall
$P tools/plot_transfer.py --arm "trained board DICT_5X5_50=$D/fullval_control_dict5x5.json:paper/results_rev6/26_pose_error_variance/pose_REL882_B1.json" --arm "zero-shot DICT_6X6_250=$D/fullval_transfer_dict6x6.json:$D/pose_zeroshot_dict6x6_B1.json" --out $D/transfer_summary.png
```

## Notes for the fine-tuning arms that follow

- The natural arms are (a) class-head-only: `train_detector.py --config configs/transfer_dict6x6.yaml
  --freeze-trunk --resume runs/rel_w882_c2_100k_rev6/ckpt_0100000.pt` (same `n_cls`, so `--resume`
  not `--retarget-from`), and (b) a full fine-tune from the same checkpoint; each then runs this same
  suite and becomes one more `--arm`.
- `dcc/pipeline.py:291` (`recover`) divides by the projected homogeneous coordinate without a guard.
  With the garbage homographies a foreign board produces it emitted RuntimeWarnings during this run;
  a lattice point mapped to infinity gives nan distances, and `nan > tol` is False, so such a point
  could be "recovered" onto an unidentified detection. Not reachable with a sane H; surfaced, not fixed
  (release pipeline, line numbers pinned by `deploy/PIPELINE_SPEC.md`).
