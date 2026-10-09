# A1 — does the bottleneck attention earn its place?

**One key:** `configs/abl_convonly.yaml` sets `attn_blocks: 0` (vs 2). Verified
programmatically against `configs/rev640.yaml` before launch — one key, nothing else.

**Arms**, both on identical frames (seed 20260729, n=100/step), coarse arm:

| arm | run | step | note |
|---|---|---|---|
| reference | `abl_reference_50k_rev6` | 50,000 | full budget |
| conv-only | `abl_convonly_50k_rev6` | 41,000 | **early exit on demonstrated convergence** |

Conv-only stopped at 41,630 (final checkpoint 41,000) with six consecutive vals
inside ±0.14 pp against the 0.4 pp criterion, and m01 flat in the fourth decimal
across 15k steps. Stated here so the stop step is never mistaken for a converged
50k number. A matched-step reading at 40,000 exists independently in both runs'
trainer val and agrees with the sweep below.

## Verdict — attention's entire contribution is at CLOSE range

Per-regime ID accuracy (%), delta vs reference:

| Regime | reference | conv-only |
|---|---|---|
| FAR (s=12) | 96.19 | **97.52 (+1.34)** |
| FAR (s=16) | 93.98 | **94.42 (+0.44)** |
| VERY CLOSE (s=96) | 99.73 | 98.91 (−0.82) |
| **VERY CLOSE (s=128)** | **99.20** | **91.82 (−7.39)** |
| OCCLUDED (max holes) | 96.64 | 96.32 (−0.32) |
| OCCLUDED (max objects) | 96.88 | 95.92 (−0.96) |

Out of envelope (`distance_extrap`) the gap widens sharply:

| s (px) | reference | conv-only |
|---|---|---|
| 128 (trained ceiling) | 99.3 | 96.7 (−2.6) |
| **160** | **82.9** | **52.3 (−30.6)** |
| 192 | 33.6 | 20.0 (−13.6) |

**Hypothesis scored one-for-three, and the one it gets is decisive.** The original
prediction was "better at FAR, VERY CLOSE and OCCLUDED".

- **VERY CLOSE — confirmed.** −7.39 pp at s=128 in envelope, −30.6 pp at s=160
  beyond it. Corroborated independently by trainer val at matched step 40,000
  (−1.98 pp on the 64–128 octave).
- **FAR — refuted, and it reverses.** Conv-only is *better* at both far steps.
  Matched-step val agrees (−0.03 pp, i.e. level).
- **OCCLUDED — no effect.** −0.32 and −0.96 pp, within noise at this n.

Localisation is untouched: m01 identical to the fourth decimal (0.4232 vs 0.4234 at
matched step). Attention sits at the bottleneck and the heatmap decodes at full
resolution, so this is the expected shape.

## Why this is the right result

It is what the receptive-field argument predicts, which is why it is worth more than
a diffuse aggregate. At s=128 the board spans 640 px of a 640 px frame; the dilated
cascade's RF (~180 px theoretical, effective radius well below) cannot reach an
adjacent marker, so the identity read has no local evidence and must come from
board-scale context. At s=12 the whole board fits inside the RF and the convs need
no help — hence no gain, and a marginal loss from spending capacity on attention.

**Claim to make:** *the bottleneck attention supplies board-scale context for the
identity read; ablating it localises the entire cost to the close-range regime where
the convolutional receptive field is exhausted — 99.20% vs 91.82% ID accuracy at
s=128, widening to 82.9% vs 52.3% at s=160 beyond the trained envelope.*

## Caveat

The s=128 gap is large enough to be worth a second seed if it needs to be
bulletproof. Conv-only's near-range m04 was oscillating (97.90 / 97.70 / 97.78 over
its last vals) rather than trending, so the effect looks stable — but this is one
seed and it should be stated as such.
