# Conv-ChArT — conference results (rev-5)

**Everything for the talk is in this one directory.** Numbered subdirectories,
newest generator revision (rev-5), one baseline checkpoint throughout.

## The three fixed things every number here shares

| | |
|---|---|
| **Detector** | `runs/rev640_baseline/ckpt_0150000.pt` — step 150,000 |
| **Refiner** | `runs/refiner_rev5/ckpt_0025000.pt` — from scratch on rev-5 |
| **Generator** | rev-5, `synth_hash = 97f1d01570f9`, snapshot in `config_rev5_snapshot.yaml` |

A number measured under a different `synth_hash` is **not comparable** to these
and must not be placed in the same table. Rows within a column are comparable;
columns are not.

---

## HEADLINE NUMBERS

**Detector**, n=2000, 26,261 matched corners (`01_detector/`):

| metric | value |
|---|---|
| M-04 identification | **99.62%** |
| M-01 localisation median | **0.4151 px** (coarse) |
| M-01 p95 | 0.7377 px |
| M-01 tail > 4 px | 0.042% |
| M-02 recall @ 12–16 px (hardest octave) | **99.38%** |
| M-02 @ 16–32 / 32–64 / 64–128 px | 99.53 / 99.45 / 99.36% |
| M-04 by octave | 98.53 / 99.54 / 99.89 / **100.00%** |

**Recall is above 99.3% at every scale octave** across the full 12–128 px envelope.

**Classical OpenCV baseline** on the same distribution (`05_comparison/`), for
contrast — at the *easiest* point in its whole grid (s=128, clean tier) it
manages **44.5% ID accuracy**, falling to **8.5%** in the deployment-dark tier.

**Pose**, coarse arm, 1000 images (`06_pose/`): 970 solved (97.0%),
M-05 rotation median **0.380°**, M-06 translation median **0.0271 board squares**.
The 30 refusals are all `too_few` — the pipeline *declines* a pose it cannot
corroborate rather than fabricating one, so a refusal is a correct outcome, not
an error. Never drop refusals from the denominator.

---

## Directory map

| dir | contents |
|---|---|
| `01_detector/` | M-01 / M-02 / M-04 for the baseline on rev-5 |
| `02_refiner/` | M-03, and the with/without-refiner contribution |
| `03_robustness/` | 15-factor sweeps + figures, each with a **with-refiner and a coarse-only curve** |
| `04_introspection/` | attention, gate flow, encoder features, pipeline; plus `attention_frame_scan.json` |
| `05_comparison/` | classical OpenCV grid; Deep ChArUco arms |
| `06_pose/` | M-05 / M-06 |
| `07_generator_audit/` | rev-5 generator audit — all 7 gates PASS |
| `08_training_logs/` | training logs for every run cited |
| `09_archive_rev4/` | superseded rev-4 results, kept only for the refiner-provenance finding |

---

## Two findings worth telling, not just tabulating

**1. The refiner had never seen the augmentations.** `runs/refiner_v1` — called
"the rev-2 refiner" for most of a day — predates the entire sim-to-real
augmentation pack. Its own saved config has no sensor noise, no vignette, no
specular, no droplets, no ink-contrast, no fixed-pattern noise and no differencing
at all. Its flattering p95 of 1.08 px was measured on near-clean renders; on
realistic capture it degrades to 2.74 px. Two other explanations (offset
distribution, differencing regime) were tested and **killed** before this one was
found by diffing the checkpoint's stored config. See `09_archive_rev4/`.

**2. The attention figure had to be chosen by measurement.** The obvious "clean,
easy scene" frame turned out to have block-0 entropy 8.34 nats against a uniform
ceiling of 8.48 — essentially no selectivity. Typical frames run 4.4–6.4 nats.
`tools/find_attention_frames.py` ranks candidates by the fraction of attention
mass landing inside the GT board quad. The published frame (val7691) puts **~60%
of attention mass on a board occupying 15% of the image**, and at small scale the
concentration reaches 60–100×.

---

## KNOWN LIMITATION — read before quoting the absolute numbers

**These results are synthetic-to-synthetic.** The board is alpha-composited onto
COCO photos with a *global* histogram match, **no local relighting, no contact
shadow, and effectively no alpha feathering** — so it reads as "cut into" the scene
and is unusually easy to find. A randomly-initialised model reaches 99.07% M-04 in
12,500 steps, which is itself evidence of this.

Present the absolute figures as an **upper bound**. The defensible claim is the
**relative** one: on the *same* data, Deep ChArUco fine-tuned reaches ~86.7% and
classical OpenCV 44.5%. If the data were trivially easy, both would be near-perfect
too.

Full diagnosis, source references and the prioritised fix list:
`REV6_generator_realism_TODO.md`. Expect every number here to drop once it lands —
that is the intended outcome, not a regression.

## Reproducing anything here

Every JSON carries its own `provenance` block: checkpoint path and step, config,
n, seed, match radius, and `synth_hash`. Figures are regenerable from the JSON
alone — the raw per-corner errors are stored, so a restyle never needs a re-run.
