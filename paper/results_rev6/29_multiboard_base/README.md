# 29 — Multi-board pretraining: does a trunk trained on many marker families transfer better?

Kaelin, 2026-10-08: "training the main model from scratch on a larger set of boards (say 15 different boards)";
then: "take the 25k model, freeze everything before the bottleneck (inclusive of the bottleneck) and try train it
to be good at one specific board? the theory is that the general pretrain will have taught it to extract general
features."

**Answer: no, not at these budgets.** A 15-family base never learned to read markers in 25k steps (identity stuck
at the 25% lattice-symmetry floor); a 3-family base did learn, after a 15k-step delay, but was a WORSE starting
point for a fourth family than the single-board release model under every fine-tune recipe tried, and the
"freeze through the bottleneck, train the rest" theory gave 30% (15-board) / 67% (3-board) ID against 72% from the
single-board model with the identical trainable set.

## Mechanism

- `dcc/synth.py:_composite_board` draws one board per positive sample from `cfg["board"]["pool"]` (uniform);
  entries override `dictionary` and `marker_id_offset` on top of `cfg["board"]`. `dcc/board.py:get_board` builds
  a `CharucoBoard` with ids `offset..offset+n-1`, so one dictionary family yields several distinct boards (the
  `_50/_100/_250/_1000` variants of a family share their first markers and are ONE board, asserted in
  `tools/cut_multiboard.py`). A pool entry may not change `squares` (the class head is built for one corner
  count; asserted). Single-board configs consume exactly the RNG stream they always did (verified bit-identical).
- `tools/cut_multiboard.py` cuts `configs/<name>_base.yaml` (882k release recipe + pool + budget) and one
  single-board `configs/<name>_eval_*.yaml` per entry; `tools/run_multiboard_chain.sh NAME STEPS` trains, scores
  per board (2k), scores the original board (10k + pose set), zero-shots the held-out DICT_6X6_250 (10k + pose
  set), then runs the fine-tune arms `configs/<name>_ft*.yaml`.

## 15 families, 50k budget, STOPPED at 25k (`mb15_base.yaml`, `runs/mb15_base_50k/ckpt_0025000.pt`)

Pool: 4x4, 5x5, 7x7, AprilTag 16h5, 25h9, 36h10, 36h11 (two boards each, ids 0-11 and 12-23) + ArUco MIP 36h12.
Held out: the whole OpenCV 6x6 family (the transfer target) and ArUco Original. The original board
(DICT_5X5_50) is in the pool as DICT_5X5_1000 offset 0, byte-identical.

In-loop ID (2k val) from 5k to 22.5k: 24.1, 25.3, 24.8, 25.2, 25.7, 25.9, 25.5, 25.9%; full 10k val at 25k:
**25.9%** (by octave 26.0 / 26.3 / 25.9 / 25.3), recall 92-95%, p95 0.69 px. Per training board at step 14k
(`diag_step14k/`): 24.9-26.4% on all fifteen. Train loss 2.59 at 15k vs 1.50 for the single-board run at the
same step (whose ID was 50.7% at 5k, 96.4% at 7.5k, 98.3% at 10k).

25% is one in four: the corner's lattice position learned from the board outline, which fixes the index only up
to the lattice's 90-degree symmetry -- uniform across families and octaves, even where every marker is legible.
Stopped by the rule set in advance (ID < 40% at the 25k validation). See the 3-family result below for why this
stop may have been premature.

## 3 families, 30k budget (`mb3_base.yaml`, `runs/mb3_base_30k/ckpt_0030000.pt`)

Pool: DICT_5X5_1000@0 (the original board), DICT_4X4_1000@0, DICT_APRILTAG_36h11@0.

In-loop ID: 24.5 (5k), 26.6, 27.0, 28.0, 29.3 (15k), **47.6 (17.5k), 70.0, 77.4, 81.1 (25k)** -- a phase transition
after ten thousand steps on the symmetry floor. Full 10k val: 80.47% at 25k, **82.58% at 30k** (by octave
44.4 / 79.8 / 94.9 / 91.6). So the board count DELAYS marker reading (one family: before 7.5k; three: after 15k;
fifteen: not within 25k) rather than preventing it; small apparent markers are what lag.

| evaluation (`mb3/`) | result | single-board release |
|---|---|---|
| per training board, 2k each | 4x4 84.9%, 5x5 81.0%, 36h11 82.6% ID | -- |
| ORIGINAL board, 10k val | 80.60% ID, median 0.4094 px, p95 0.6996 | 99.53%, 0.4063, 0.6733 |
| ORIGINAL board, 1000-frame pose set | 62.9% correct, 0.3% wrong-accepted | 89.1%, 0.2% |
| ZERO-SHOT on DICT_6X6_250 (unseen family), 10k | 17.73% ID (31.6 / 22.9 / 12.0 / 10.6 by octave) | 13.36% |
| ZERO-SHOT pose set | 0.9% correct, **19.0% wrong-accepted** | 1.8%, 22.0% |

Zero-shot to an unseen family fails the same way from a 3-family trunk as from a single-board one: the lattice
gate accepts 90-degree-symmetric relabellings. Where the unseen markers are legible (near) the model misreads
them with confidence; where they are not (far) it falls back to geometry.

## Fine-tuning onto DICT_6X6_250 from each base (5,000 steps, same recipes as `28_finetune_minimal`)

| trainable set | from 882k release | from 3-family base (30k) | from 15-family base (25k) |
|---|---|---|---|
| head only | 57.4% | 51.0% | 28.4% |
| head + bottleneck + corner head, 0.1x | 90.5% (70% pose) | 72.5% (34% pose) | -- |
| head + bottleneck + corner head, 1x | 97.1% (84.5% pose) | 86.7% (64% pose) | -- |
| **post-bottleneck** (gate, decoder, heads; encoder + bottleneck frozen) | 72.3% (15% pose) | 67.3% (9% pose) | **30.0%** (0% pose) |

ID accuracy on the 10k validation of the new board; pose = correct-pose rate on the 1000-frame set.
Localisation p95 after fine-tuning: 0.674-0.675 px from the 882k base, 0.699-0.700 from the 3-family base,
0.690 from the 15-family base (the multi-board trunks are also less converged on localisation).

**Reading.** The multi-board trunks are worse starting points at every trainable set, by 5-18 points from the
3-family base and by 29-42 points from the 15-family base. The post-bottleneck theory fails on both: nothing
downstream of a frozen bottleneck can supply the marker reading it did not learn (15-family), and even a trunk that
reads three families does not read a fourth without the bottleneck moving (3-family: 67% frozen vs 87% at 1x).
The single-board release model remains the best base for transfer, and the bottleneck's learning rate remains the
lever (`28_finetune_minimal`).

**Caveats, stated.** Budgets were short (25k-30k steps vs the release's 100k); the 15-family run was stopped on a
rule that the 3-family transition suggests was premature; a 100k-step 15-family base is the untested case. The
3-family base's own accuracy on its training boards (81-85%) is far from converged at 30k.

## Files

`diag_step14k/` (15-family per-board diagnostic + NOTE.md with the stop decision), `mb3/` (every evaluation of
the 3-family base and its chain's timing), `pool_sheet.png` (16 samples from the 15-board pool), run dirs
`runs/mb15_base_50k`, `runs/mb3_base_30k`, `runs/mb15_ft*`, `runs/mb3_ft*`; the fine-tune arms' metrics and
pose files sit beside the ladder's in `28_finetune_minimal/`.
