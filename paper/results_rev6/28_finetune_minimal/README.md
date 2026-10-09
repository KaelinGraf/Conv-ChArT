# 28 — Minimal fine-tuning: what has to move to put Conv-ChArT 882k on a new board, and what it costs

Kaelin, 2026-10-08: "figure out what the absolute minimum amount of fine-tuning required is ... graphically show
(side by side) the VRAM, training steps, and accuracy to transfer learning to a new board - we need to see if its
doable easily on consumer hardware, on-device. Also log the time taken for each run."

**Answer.** The attention bottleneck must train, and at a HIGH learning rate; nothing else matters much.
Head-only (and every other recipe that leaves the bottleneck frozen) plateaus at 57-61% ID on the new board.
Head + bottleneck + corner head at 3x the base LR reaches **98.2% ID and 86% correct poses in 5,000 steps,
95% ID by step 1,000**, matching the whole model at 1x (98.4%) with half the parameters trainable, **3.5 GB at
batch 16 (0.9 GB at batch 4)** and ~48 ms/step on an RTX 5090 -- about one GPU-minute to 95%. Localisation is
untouched by every arm (p95 within 0.004 px of the release model). The trained board's 99.5% was not reached in 5,000 steps by
any frozen-encoder arm; the whole model at 3x reaches 98.9% / 88.3% poses (99% on the in-loop val at step
3,500) for a small localisation cost, and the minimal recipe for 20,000 steps (arm 21) reaches **99.1% ID / 87.6% poses with localisation
unchanged** -- within half a point of the trained board, from 437k trainable parameters in 3.5 GB.
Wall time is dominated by the CPU synthetic-data generator (76 samples/s per lane here), not the GPU -- that is
the real on-device cost.

## Setup (identical for every arm)

Start from `runs/rel_w882_c2_100k_rev6/ckpt_0100000.pt` (EMA weights); target board = the zero-shot transfer board
(`configs/transfer_dict6x6.yaml`: same 5x5 geometry, DICT_6X6_250 markers); base `configs/ft_base_dict6x6.yaml`:
5,000 steps (1/20 of the release budget), batch 16 x accum 2, AdamW 3e-4 cosine to 3e-6, 100 warmup, EMA 0.99,
val every 250 steps on 1,000 samples, fresh generator seed. Arms differ ONLY in `finetune.lr_mult` ({fnmatch
pattern: multiplier}; unmatched = frozen, frozen BatchNorm held in eval) and optionally `finetune.reinit`; every
arm's diff against the base is asserted by `tools/cut_finetune_arms.py` and written into its header. Evaluation:
the trainer's own 10k full validation at the final step, then the 1000-frame B1 pose set
(`tools/run_finetune_arms.sh`). Two arms ran side by side (RAM-bound: ~20 GB per run), so logged wall times are
contended; isolated GPU cost per trainable set is `finetune_cost.json` (`tools/finetune_cost.py`: random
tensors of the real shapes, bf16, batch 4/8/16).

## Results (ID = 10k full val on the new board; pose = correct-pose rate on 1000 frames; VRAM/ms at batch 16)

| # | what trains (LR multiplier) | trainable | ID | pose | steps to 95% | VRAM | ms/step |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | ID head | 37,968 | 57.4% | 0.4% | never | 2.1 GB | 22 |
| 7 | ID head, re-initialised | 37,968 | 58.9% | 0.7% | never | 2.1 GB | 22 |
| 11 | ID head at 3x | 37,968 | 59.7% | 0.7% | never | 2.1 GB | 22 |
| 5 | ID head + H/4 gate | 44,177 | 57.4% | 0.3% | never | 2.7 GB | 33 |
| 6 | ID head + d3 (0.1x) | 148,688 | 61.3% | 1.3% | never | 2.7 GB | 32 |
| 8 | BitFit (biases + LN scales + head) | 42,194 | 73.3% | 15.1% | never | 4.6 GB | 55 |
| 17 | post-bottleneck (gate, d3, d2, d1, both heads) | 191,890 | 72.3% | 15.0% | never | 2.7 GB | 41 |
| 3 | head + bottleneck 0.01x + corner head 0.1x | 437,105 | 69.3% | 8.6% | never | 3.5 GB | 48 |
| 4 | whole model, layer-wise decay 0.3 (bottleneck 0.008x) | 882,402 | 71.9% | 14.7% | never | 4.9 GB | 67 |
| 2 | head + bottleneck 0.1x + corner head 0.1x | 437,105 | 90.5% | 69.6% | never | 3.5 GB | 48 |
| 13 | arm 2 for 20,000 steps | 437,105 | 96.0% | 82.0% | 10,500 | 3.5 GB | 48 |
| 15 | bottleneck 0.3x, corner head 0.1x | 437,105 | 93.5% | 78.5% | never | 3.5 GB | 48 |
| 9 | whole model 0.1x, head 1x | 882,402 | 94.3% | 79.2% | never | 4.9 GB | 66 |
| 16 | bottleneck 1x, corner head 0.1x | 437,105 | 97.1% | 84.8% | 1,750 | 3.5 GB | 48 |
| 14 | bottleneck 1x, corner head 1x | 437,105 | 97.1% | 84.5% | 1,750 | 3.5 GB | 48 |
| 19 | arm 14 + e4 at 1x | 658,801 | 97.8% | 85.4% | 1,250 | 3.6 GB | 49 |
| **18** | **bottleneck 3x, corner head 0.1x** | 437,105 | **98.2%** | **86.2%** | **1,000** | **3.5 GB** | **48** |
| 10 | whole model 1x | 882,402 | 98.4% | 86.9% | 1,000 | 4.9 GB | 66 |
| 12 | ID head for 20,000 steps | 37,968 | 61.8% | 2.2% | never | 2.1 GB | 22 |
| 20 | whole model 3x | 882,402 | 98.9% | 88.3% | 750 (99% at 3,500) | 4.9 GB | 66 |
| **21** | **arm 18 for 20,000 steps** | 437,105 | **99.1%** | **87.6%** | 1,000 (99% at 17,500) | 3.5 GB | 48 |

Reference: the release model on its own board is 99.53% ID / 89.1% correct poses; zero-shot on this board it is
13.4% / 1.8% with 22% confidently wrong poses (`27_transfer_zeroshot`). Wrong-accepted poses stay at 0.0-0.5% for
every fine-tuned arm. Full table with wall times: `finetune_ladder_round1.md` (arms 1-10) and the figures below.

## What the ladder says

1. **The bottleneck's learning rate is the lever.** Bottleneck frozen: 57-61% (head only, +gate, +d3, re-init,
   3x head LR). Biases only: 73%. Bottleneck at 0.01x: 69-72% (arms 3 and 4 -- arm 4 trains everything but its
   layer decay leaves the bottleneck at 0.008x). 0.1x: 90.5%. 0.3x: 93.5%. 1x: 97.1%. 3x: 98.2%. The corner head's
   rate is irrelevant (arms 14 vs 16); adding e4 (arm 19) buys 0.7 pp where tripling the bottleneck rate buys 1.1.
2. **Steps do not substitute for rate.** Arm 2 for 20,000 steps (arm 13) converges at 96.0%; arm 18 passes that
   by step 1,000 and ends at 98.2% in 5,000; the same recipe for 20,000 steps (arm 21) reaches 99.1% (99% on
   the in-loop val at step 17,500), i.e. budget DOES buy the last point once the rate is right. Head-only for
   20,000 steps (arm 12): 61.8%, four points for four times the budget -- the head-only ceiling is capacity.
3. **Memory follows gradient depth, not parameter count.** BitFit trains 42k parameters but needs 4.6 GB because
   its biases sit in every layer; head + bottleneck trains 437k in 3.5 GB; head only 2.1 GB. Batch 4 divides all
   of these by ~4 (`finetune_cost.json`).
4. **Localisation barely moves** (p95 0.673-0.677 px in every arm but one; release 0.673). The exception is the
   whole model at 3x (arm 20): 0.682 px, the price of running the encoder at 9e-4 -- still inside the release
   tiers' spread, but it is the one arm that touches the corner detector at all.
5. **On-device budget.** 1,000 steps of arm 18 at batch 16 is ~48 s of RTX 5090 compute (95% ID); 17,500 steps
   for 99% is ~14 GPU-minutes; the 8-13 min wall
   time observed per 1,000 steps here is the CPU data generator at 76 samples/s. On a Jetson-class device the
   GPU side scales by roughly 10-20x and the generator becomes the bottleneck by a wider margin; pre-rendering a
   sample bank is the obvious mitigation.

Robustness of the 20k fine-tune on the 20-factor sweep (`30_REPORT_transfer_finetune/`): recall and
localisation identical to the trained board; ID 91.1% vs 93.9% mean, with the gap concentrated at the hardest
darkness (24 vs 69%) and noise (52 vs 72%) steps -- clean-condition identity transfers, identity under heavy
degradation only partly.

Base-model comparison (multi-board pretraining as the starting point): `29_multiboard_base/README.md` and
`finetune_ladder_bases.png` -- the single-board release model is the best base at every trainable set.

## Figures

- `finetune_ladder_round1.png` -- arms 1-10: learning curves (ID, p95), steps and wall to 80/90/95/99%, peak
  VRAM, final ID and pose.
- `finetune_ladder_rate.png` -- the bottleneck-rate / budget ladder (arms 3, 2, 15, 16, 18, 19, 10, 20, 13, 21),
  thresholds 80/90/95/98/99%.
- `finetune_ladder_bases.png` -- the same recipes from the 882k, 3-family and 15-family bases.
- Tables alongside each figure (`*.md`); `tools/plot_finetune.py --arms ... --labels ...` regenerates any subset.

## Tooling added

`dcc/trainutil.py:param_groups(lr_mult=...)`, the `finetune:` entry in `tools/train_detector.py` (+ per-step
`peak_mem_mb`, a final `done` record with elapsed/trainable counts), `tools/cut_finetune_arms.py`,
`tools/run_finetune_arms.sh`, `tools/finetune_cost.py`, `tools/plot_finetune.py`. The PyTorch fused LayerNorm
backward raises on a bias-only-trainable LayerNorm (NativeLayerNormBackward0, "invalid gradient at index 2"),
which is why BitFit includes the LN scales.
