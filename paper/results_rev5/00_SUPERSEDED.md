# SUPERSEDED — every number in this tree was measured on an EASIER dataset

**Status: historical record. Do NOT quote any figure in this directory in the paper, a
table, or a comparison against `paper/results_rev6/`.**

Kaelin, 2026-08-04: *"the reason rev-5 was better was rev-5 was an incredibly easy
dataset."* This file exists so that a future reader — or a future agent picking a number
out of `03_robustness/` because it looked authoritative — does not quote an easy-dataset
figure as if it were current.

## The evidence, as a controlled comparison

The same 4,701,439-parameter model was fine-tuned from the same 150k checkpoint to the
same step 160,000, twice: once on rev-5 data, once on rev-6. **Architecture, step count,
initialisation and optimiser are identical; only the generator revision differs.** Each
was scored on its own revision's validation set (10,000 samples):

| run | M-01 p95 | M-01 median | tail >4 px | M-04 |
|---|---|---|---|---|
| `runs/rev640_160k_rev5` | 0.7470 | 0.4155 | 0.0364% | 99.64% |
| `runs/rev640_160k_rev6` | 0.8129 | 0.4223 | 0.1205% | 98.98% |
| **difference** | **+0.0659 px** | +0.0068 | **3.3x worse** | **−0.66 pp** |

That gap is the DATASET, not the model. It is an order of magnitude larger than the
measured seed floor at any width (882k: 0.0020 px / 0.07 pp), so it cannot be dismissed
as run-to-run variation.

The practical consequence: a rev-5 number will look ~0.066 px better on p95 and ~0.66 pp
better on identity than the same system measured honestly, and its >4 px tail — the
operational gate, i.e. corners outside the refiner's capture range — will look roughly
three times smaller than it really is.

## Why this tree is kept rather than deleted

The conference deck was presented from these results. Deleting them would destroy the
provenance of a talk that was actually given. They are a truthful record of what was
measured at the time, under the generator that existed at the time. They are simply not
comparable to anything measured since.

## Where the current numbers live

`paper/results_rev6/`. In particular:

| what | where |
|---|---|
| release models, robustness (20 factors) | `20_robustness_REL882/`, `20_robustness_REL502/` |
| release models, pose (SD-10) | `21_pose_REL882.json`, `21_pose_REL502.json` |
| 4-way vs Deep ChArUco + classical OpenCV | `22_fourway_RELEASE/` |
| ablation campaign, floors, decisions | `PLAN_autonomous_campaign.md`, `08_ablations/` |

## One caveat that applies to results_rev6 as well

Audit finding B1 (2026-08-03): the SD-10 **pose** evaluation sets never fire specular or
ink-contrast — the measured worst robustness factor — and compute corner visibility
BEFORE photometrics. So the pose benchmark is photometrically easier than train/val in
BOTH revisions. Cross-arm rankings on it are internally consistent and the comparisons
are sound, but the absolute pose figures are optimistic. That is an open decision, not a
resolved one: either regenerate the pose sets and re-run, or state the limitation in the
paper.

**RESOLVED 2026-08-05 (Kaelin: regenerate).** Root cause: `_apply_photometric` was called with
neither `board_mask` (which gates specular/ink-contrast) nor `holes_out` (which lets droplets
occlude), and visibility ran before photometrics instead of after. Fixed; `eval_pose_rev6_b1`
regenerated and all three release tiers re-evaluated. The figures WERE optimistic -- solve rate
falls 1.9-2.9 pp and rotation median worsens ~8% on every tier -- but the tier ordering and the
"accuracy is indistinguishable, solve rate separates them" conclusion both survive. Full result in
`paper/results_rev6/23_code_audit_FINDINGS.md`; quote `21_pose_REL*_B1.json`, not the originals.
