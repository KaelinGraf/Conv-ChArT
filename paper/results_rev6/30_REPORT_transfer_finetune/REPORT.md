# Conv-ChArT 882k: pose measurement noise, board transfer and minimal fine-tuning — headline results for the report

Compiled 2026-10-09 from `26_pose_error_variance`, `27_transfer_zeroshot`, `28_finetune_minimal` and
`29_multiboard_base` (each has a README with provenance for every number). All figures are in `figures/`; every
panel is also its own file in `figures_separate/` (list at the end) so any subset can go in the report.
Model throughout: the 882k release detector (`runs/rel_w882_c2_100k_rev6/ckpt_0100000.pt`) with the shared
refiner; new board = same 5x5 geometry (16 corners) with the DICT_6X6_250 marker dictionary, an unseen family.
Metrics: ID = corner-identity accuracy on the 10,000-sample validation set (conditioned on detection); pose =
correct-pose rate (<2 deg rotation error, not flagged ambiguous) on the 1000-frame B1 pose set; p95 = 95th
percentile corner error after refinement, px.

## 1. Measurement noise for the Kalman filter (`fig0_kalman_error_vs_range.png`)

Ensemble statistics of accepted poses (unambiguous, n = 893 of 1000 frames), camera frame, 6-vector
[drot_x, drot_y, drot_z (rad); dt_x, dt_y, dt_z (board squares; multiply by square_length^2 for m^2)].

| component | robust std (1.4826·MAD) | sample std | range law, sigma = a·z^k (z = depth / square length) |
|---|---|---|---|
| rot x / y / z | 0.091 / 0.088 / 0.037 deg | 0.276 / 0.290 / 0.093 deg | k = 0.97 / 0.81 / 1.02 |
| trans x / y / z | 0.0029 / 0.0027 / 0.0116 sq | 0.0127 / 0.0086 / 0.0433 sq | k = 1.50 / 1.49 / 1.73 |

- **A single averaged covariance is the wrong choice.** Pooled sigma is 2.0-7.4x over-confident at the far
  octave and 2.0-3.6x over-cautious at the near one; the range-scaled fit is matched within 0.83-1.24 in every
  octave and component (figure, right panel). Constants: `26_pose_error_variance/kalman_R_REL882.json`.
- The error is heavy-tailed (sample variance 9-14x the robust variance; 13.5% of frames beyond 5 robust sigma
  even under the range-scaled model): the filter needs a chi-square innovation gate regardless of R. Skip the
  update on refusal, on the ambiguous flag, and on PnP reprojection residual > 0.15 px (removes half the tail at
  16% rejection). The network's own per-frame `pose_cov` is NOT usable as R (NEES 11.2 vs 6; it does not even
  rank frames).
- Off-diagonal correlations among accepted solves are weak (|rho| <= 0.23): a diagonal R is defensible.

## 2. Zero-shot transfer to a new board (`fig1b`, `fig2a`, `fig2b`, `figA`)

| | trained board | new board, zero-shot |
|---|---|---|
| corner error median / p95 | 0.4063 / 0.6733 px | 0.4063 / 0.6754 px |
| recall (octave mean) | 94.2% | 94.3% |
| ID accuracy | 99.5% | 13.4% (chance 6.25%) |
| correct pose / **wrong pose accepted** / refused | 89.1% / 0.2% / 6.4% | 1.8% / **22.0%** / 73.3% |

Localisation and recall transfer unchanged; identity does not; and the failure is worse than "no ID": on 22% of
frames the lattice gate accepts a 90-degree-rotated (lattice-symmetric) labelling and PnP returns a confidently
wrong pose with a normal residual (rotation error median 90.0 deg, translation error exactly one board side).
Across the 20-factor robustness sweep on identical frames: recall 91.1% vs 91.1%, ID 95.3% vs 3.9% (mean over
factors and steps). This is the Deep ChArUco failure mode the gate exists to prevent, and the reason the class
head must be retrained before deployment on any new board.

## 3. Minimal fine-tuning (`fig1`, `fig3`, `fig4`) — the main result

21 arms, each 5,000 steps unless stated (1/20 of the release budget), from the release checkpoint onto the new
board. Only the attention bottleneck's learning rate matters:

| what trains (base LR 3e-4 unless noted) | trainable | ID | pose | steps to 95% | VRAM at batch 16 / 4 |
|---|---:|---:|---:|---:|---|
| ID head only (any variant; also 3x LR, also 20k steps) | 38k (4%) | 57-62% | 0-2% | never | 2.1 / 0.5 GB |
| head + gate, head + decoder rung, BitFit, post-bottleneck | 44-192k | 57-73% | 0-15% | never | 2.7-4.6 GB |
| head + bottleneck + corner head, bottleneck at 0.01x | 437k | 69% | 9% | never | 3.5 / 0.9 GB |
| same, bottleneck 0.1x (and for 20k steps) | 437k | 90.5% (96.0%) | 70% (82%) | never (10,500) | 3.5 / 0.9 GB |
| same, bottleneck 1x | 437k | 97.1% | 85% | 1,750 | 3.5 / 0.9 GB |
| **same, bottleneck 3x** | **437k (50%)** | **98.2%** | **86%** | **1,000** | **3.5 / 0.9 GB** |
| **same, bottleneck 3x, 20,000 steps** | 437k | **99.1%** | **87.6%** | 1,000 (99% at 17,500) | 3.5 / 0.9 GB |
| whole model, 1x | 882k | 98.4% | 87% | 1,000 | 4.9 / 1.2 GB |
| whole model, 3x | 882k | 98.9% | 88.3% | 750 (99% at 3,500) | 4.9 / 1.2 GB |
| reference: release model on its own board | | 99.5% | 89.1% | | |

- **Full accuracy is reachable**: the minimal recipe reaches 99.1% ID / 87.6% correct poses, within half a point
  of the trained board, with localisation unchanged (p95 0.672 vs 0.673) and wrong-accepted poses at 0.2%.
- **On-device cost.** The minimal recipe is ~48 ms/step at batch 16 on an RTX 5090: ~1 GPU-minute to 95% ID,
  ~14 GPU-minutes to 99%; 0.9 GB at batch 4. Wall time here (8-13 min per 1,000 steps) was the CPU synthetic-data
  generator (76 samples/s per lane), which is the real on-device bottleneck and the thing to pre-render.
- Memory follows gradient depth, not parameter count (BitFit: 42k parameters, 4.6 GB).
- Localisation never moved except for the whole model at 3x (p95 0.682 px): the minimal recipe does not touch
  the corner detector at all.

### 3b. Is the fine-tuned model as robust as the original? (`fig6a/b/c`, `robustness_hardest_step_3way.md`)

Same 20-factor sweep, identical frames, trained board vs zero-shot vs the 20k fine-tune (head + bottleneck 3x):

| mean over all factors and steps | trained board | zero-shot | fine-tuned 20k |
|---|---|---|---|
| recall | 89.6% | 89.7% | 89.8% |
| ID accuracy | 93.9% | 3.7% | 91.1% |
| localisation median (refined) | 0.145 px | 0.145 px | 0.144 px |

Recall and localisation are identical in every regime. Identity is recovered in benign and moderate conditions
(within 1-3 points on most factors) but NOT fully at the hardest steps: darkness 0.01 -> 24% vs 69%; sensor
noise at 12 dB SNR -> 52% vs 72%; differencing ambient 0.95 -> 88% vs 97%. The 20k fine-tune transfers
clean-condition identity; the identity-under-degradation that the full 100k pretraining bought (see
`PROJECT_KNOWLEDGE.md`, "capacity buys identity under degradation") is only partly transferred. For a rig that must
work at low light, budget the fine-tune longer or fine-tune under the deployment's photometric regime.

## 4. Multi-board pretraining as a starting point (`fig5`, `figB`)

Hypothesis tested: a trunk pretrained on many marker families gives general features so a new board needs less.

- 15 families, 50k budget: identity stuck at the 25% lattice-symmetry floor for 25k steps (uniform over families
  and octaves, every marker legible) -- stopped. 3 families, 30k: same floor until step 15k, then a transition to
  82.6% (44% at the far octave). Board count delays marker reading rather than preventing it; the 15-family stop
  may have been premature (untested: a 100k-step 15-family base).
- The 3-family base costs 19 points on the original board (80.6% vs 99.5% ID; 62.9% vs 89.1% poses) and is a
  WORSE base for the new board at every trainable set (head 51 vs 57; bottleneck 0.1x 72 vs 90; 1x 87 vs 97;
  post-bottleneck 67 vs 72). Freezing through the bottleneck and training the rest -- the "general features"
  theory -- gives 30% (15-family) and 67% (3-family) against 72% from the single-board release.
- Zero-shot on the unseen family still fails by accepting lattice-symmetric poses (19% wrong-accepted).

## 5. Other verified facts worth a sentence

- The pipeline handles different corner counts (verified end to end on 9-, 25- and 36-corner boards: config
  change plus a fresh class head via the retarget or fine-tune path; square boards only; render resolution must
  divide by the square count). Deep ChArUco's identity output is sized for one board.
- The multi-board pool varies markers only; the class head's width fixes the corner count (asserted).

## Figure list and captions

- `fig0_kalman_error_vs_range.png` -- Pose measurement noise of the 882k model vs range (893 accepted solves):
  robust sigma per octave with power-law fits (rotation, translation); right: median normalised error per octave
  under a constant vs a range-scaled R (1 = matched).
- `fig1_transfer_end_to_end.png` -- Trained board vs zero-shot vs three fine-tuned arms: localisation (median,
  p95), ID accuracy and recall, and the per-frame pose outcome (correct / ambiguous / wrong-accepted / refused).
- `fig1b_transfer_zeroshot_only.png` -- The same, trained board vs zero-shot only.
- `fig2a_zeroshot_robustness_id.png`, `fig2b_..._recall.png` -- 2x2 robustness panels (distance, darkness, sensor
  noise, motion blur), trained board vs zero-shot on identical frames: recall unchanged, ID collapses.
- `fig3_finetune_ladder_round1.png` -- The ten first-round arms: learning curves, steps/wall to 80/90/95/99% ID,
  peak VRAM, final ID and pose.
- `fig4_finetune_ladder_rate_budget.png` -- The bottleneck learning-rate ladder (0.01x-3x), +e4, whole model at
  1x/3x, and the two 20k-step runs, thresholds 80/90/95/98/99%.
- `fig5_finetune_from_multiboard_bases.png` -- The same recipes from the single-board, 3-family and 15-family
  bases.
- `fig6a/b/c_finetuned_vs_trained_robustness_{recall,id,loc}.png` -- 2x2 robustness panels, trained board vs the
  20k fine-tune on identical frames: recall and localisation identical, identity equal except at the hardest
  darkness and noise steps. Per-factor figures: `robustness_figs_finetuned_vs_trained/`.
- `figA_boards_trained_vs_new.png` -- The trained board (DICT_5X5_50) and the new board (DICT_6X6_250) side by
  side: identical corner geometry, different markers.
- `figB_multiboard_pool_samples.png` -- 16 generator samples from the 15-board pool.

## Separate panels (`figures_separate/`, one file per panel; added 2026-10-09)

Every panel of the composites above as its own image, cropped from the identical render (same data, same styling)
after giving each panel the legend / y-label it shared with its neighbours in the composite. Names are
`<composite stem>_p<n>_<panel title>.png`. 49 files; `figB` is a contact sheet and is left whole.

- `fig0_kalman_error_vs_range_p1_rotation_error_vs_range`, `_p2_translation_error_vs_range` -- robust sigma per
  octave with the power-law fits; `_p3_is_r_matched_...` -- median normalised error per octave, constant vs
  range-scaled R (1 = matched).
- `fig1_transfer_end_to_end_p1_corner_localisation...`, `_p2_identification_and_recall...`,
  `_p3_pose_outcome_per_frame...` -- the three panels of the end-to-end figure (five arms); `fig1b_..._p1/p2/p3` --
  the same three, trained board vs zero-shot only.
- `fig2a_zeroshot_robustness_id_p1..p4`, `fig2b_zeroshot_robustness_recall_p1..p4` -- one factor each (p1 distance
  with the trained envelope, p2 darkness, p3 sensor noise, p4 motion blur), zero-shot vs trained board; 2a = ID
  accuracy among matched, 2b = recall.
- `fig3_...`, `fig4_...`, `fig5_...` `_p1` learning curve (ID vs step), `_p2` p95 during fine-tuning, `_p3` steps to
  each ID threshold, `_p4` wall time to each threshold, `_p5` peak training VRAM (batch 16), `_p6` final ID and
  correct-pose rate -- for the first-round arms, the rate/budget ladder and the multi-board bases respectively.
- `fig6a_..._recall_p1..p4`, `fig6b_..._id_p1..p4`, `fig6c_..._loc_p1..p4` -- one factor each as above, the 20k
  fine-tune vs the trained board; a = recall, b = ID accuracy, c = localisation median (log px).
- `figA_p1_board_trained_DICT_5X5_50`, `figA_p2_board_new_DICT_6X6_250` -- the two boards.

Regenerate everything with `regen_figures.sh` (the four plot tools take `--separate`; the crop is
`dcc/viz.py: save_panels`). The script re-renders the composites into scratch and `cmp`s them against the copies
here before copying the panels out, so a drift in any tool shows up as `DIFFER`.

**Correction made while splitting (2026-10-09).** `tools/plot_four_way.py` applied its per-factor override
(darkness plotted as found+ID, designed for the RECALL grid) to every `--panel-metric`, so the darkness panel of
the ID grids showed found+ID instead of ID accuracy and, in the localisation grid, a percentage on the px axis.
Fixed (the override now applies to the recall grid only); `fig2a`, `fig6b`, `fig6c` and
`27_transfer_zeroshot/figures_vs_control/fourway_PANEL_{id_acc,err_median}transfer_*.png` are regenerated. The
recall grids (`fig2b`, `fig6a`) are byte-identical to before, and no number in the tables above came from a figure.
