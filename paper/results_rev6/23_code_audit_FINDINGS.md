# Code audit — findings of record (2026-08-05)

**Why this file exists.** The Fable over-complexity/correctness audit (2026-08-03) enumerated its
findings into conversation context and a 67 MB task-output blob, and nowhere else. When that
context was compacted the list became unrecoverable in practice — a direct breach of the standing
rule that everything of record goes to permanent storage. Findings are banked here from now on.

The three correctness-major items (A3 / A4 / A6) are written up in
`PLAN_autonomous_campaign.md` under "2026-08-04 — audit correctness fixes". This file covers the
mechanical sweep that replaced the lost list.

---

## Method

`ast`-based scan over `dcc/`, `tools/`, `tests/` for the three mechanical classes the audit
targeted, deliberately conservative (reports only what is provable from a single file's AST):

1. a name **assigned but never read** in the same scope — dead computation
2. the **same top-level function name in ≥2 files** — parallel implementations
3. **imports never used**

Class (1) is how `filmstrip_lighting.py`'s dead `snr_of` was originally found, so the scan is a
generalisation of a hit that had already proven real.

---

## Class 1 — assigned but never read: 31 hits, **0 defects**

Every hit was traced to source. All are benign; the details matter because three of them look
alarming and are not:

| site | verdict |
|---|---|
| `dcc/pipeline.py:519` `inlier_mask` | **NOT dead.** Tested contract — `test_pipeline.py:220` and `test_board_adapt.py:182` assert on it. The production caller discards it correctly: `lattice_gate` sets `inlier_mask[idd], demoted_mask[idd] = mask, ~mask`, so within the ID'd subset it is the exact complement of the `demoted_mask` that IS used. |
| `tools/snr_calibration.py:85` `post` | Correct to discard. `sensor_noise_K` is `kind == "range"`, and `apply_factor` only ever builds a `post_fn` for blur/gain factors — it returns `None` here unconditionally. |
| `tools/introspect.py:668` `hm_logits` | Tuple unpacking. `_erf_grad`'s docstring states it backprops from the **class** head; discarding the heatmap logits is the documented behaviour. |
| `tools/eval_pose_fourway.py:140` `reason` | Failure is already captured via `rvec is None` in `score(...)`. Its siblings are `_rms`, `_amb`, `_cov` — `reason` alone lacks the underscore, which is the only reason it tripped the scan. |
| remaining 27 (mostly `tests/`) | Unpacking for readability, or values asserted on indirectly. No defects. |

**Take-away:** the underscore-prefix convention is applied inconsistently. That is not a bug, but
it is what makes a mechanical scan produce false positives that cost time to clear — the four
sites above were each investigated from source before being cleared.

## Class 2 — duplicate top-level names: 14, **0 defects, 1 navigability wart**

- **`score_frame` in `robustness_sweep.py` and `factor_sweep.py` — CHECKED, not a problem.**
  They are genuinely different implementations (greedy single-pass vs `eval_classical`'s two-pass
  anonymous+ID matching, tuple vs dict return). The question that mattered was whether they score
  at different tolerances, which would make their numbers incomparable: **both are 4.0 px**
  (`robustness_sweep.py:34`, `factor_sweep.py:95`). Every reported release sweep stamps
  `"match_px": 4.0` in its own provenance block and comes from `robustness_sweep`. No headline
  number is affected.
- **Three unrelated functions named `load`** (`ablation_summary.py` parses a run's
  `metrics.jsonl`; `make_tables.py` and `plot_ablation.py` each read sweep JSONs into different
  shapes). This is the "findable in a single grep" rule losing: grepping `def load` returns three
  unrelated things. `make_tables.load` and `plot_ablation.load` do overlap by ~8 lines, but they
  differ in arm-selection strategy and return shape, so merging them means designing a new common
  contract for two working report tools. **Left alone deliberately** — churn without functional
  gain. Recorded so the next reader does not re-derive it.
- The rest (`_worker_init`, `_val_collate`, `run_validation`, `_load_train_detector` in
  `train_charuconet.py`) are the deliberate isolation of the Deep ChArUco baseline trainer from
  ours. Sharing code there would couple our trainer to a baseline, which is the opposite of what
  a fair comparison wants.

## Class 3 — unused imports: 7 found, **7 removed, now 0**

`dcc/trainutil.py` (`np`), `tests/test_trainutil.py` (`np`), `factor_sweep.py` (`importlib`),
`polarity_invariance.py` (`n_corners`), `receptive_field.py` (`torch` — note `:53`'s
`import torch.nn as nn` binds `nn`, not `torch`), `filmstrip_lighting.py` (`rs`, then `json`
once its only consumer went).

## The one real deletion: `filmstrip_lighting.py`'s abandoned SNR labelling

`:94` built `snr_of = {r["K"]: r["snr_db_mean"] ...}` from the calibration file and **never used
it** — the name appeared exactly once in the file. It could never have worked as intended either:
the calibration is keyed by sensor gain **K**, while the filmstrip sweeps **darkness**, so the
lookup had no key domain in common with the figure's axis. Removed, along with the `cal` read and
the `rs` import that had gone with it.

**Verified by construction:** the figure regenerated **byte-identical** to its pre-cleanup
version, which is proof the removed code was dead rather than an argument that it was.

This corrects the note written on 2026-08-04, which said the dead block was being left in place as
evidence. Having established it can never have functioned, leaving known-dead code in a figure
tool is worse than recording why it went.

---

## Open items — ALL FOUR DECIDED BY KAELIN, 2026-08-05

- ~~**A3 draw-side.**~~ **CLOSED — DOCUMENTED AND ACCEPTED** (Kaelin's call: "a"). The noise draw
  stays per-channel; the delivered frame remains 0.6686x the modelled sigma, i.e. 3.50 dB quieter
  than the per-channel formula. This is a LABELLING fact, not a validity problem: every arm saw
  the same data, so every comparison in this campaign holds. `snr_calibration.py` already reports
  both figures (`snr_db_*` delivered, `snr_db_*_modelled` per-channel) and the four affected figure
  sets were regenerated. **PAPER REQUIREMENT: state that the reported SNR is the DELIVERED
  grayscale frame, and that the per-channel Poissonian-Gaussian formula overstates the noise by
  3.50 dB.** No retrain; the alternative would have invalidated the whole comparison set.
- **B1.** **DECIDED: REGENERATE** (Kaelin's call: "b"). Root cause found and fixed 2026-08-05 --
  see "B1 root cause" below. Regeneration and re-evaluation in progress.
- ~~**222k fp16 identity.**~~ **CLOSED — ACCEPTED** (Kaelin's call: "b"): 0.1005% against a 0.1%
  budget is accepted and stated rather than bought back with an fp32 detector. It is entirely
  detector-side, so an fp32 refiner recovers none of it. **PAPER/RELEASE-NOTE REQUIREMENT: the
  222k tier exceeds the P15 identity budget by 0.0005 pp under fp16.** Context for anyone weighing
  it: that margin is negligible beside the tier's real constraint, 5.6% recall in darkness against
  37.3% (502k) and 45.8% (882k) on an active-illumination rig.
- ~~**onnxruntime.**~~ **CLOSED — INSTALLED** (Kaelin's call: "a"). onnxruntime 1.28.0 into MLWS,
  closing the op-level ONNX-vs-PyTorch leg the P15 gate previously declared out of scope.

## B1 root cause — `_apply_photometric` was called with neither `board_mask` nor `holes_out`

The claim was verified from source before regenerating anything, and BOTH halves are real, with
one precise mechanism each:

1. **Specular / NIR ink-contrast / differencing lobe never fired.** `dcc/synth.py:631-632` states
   these steps are BOARD-ANCHORED and gated by `board_mask`/`board_centroid`.
   `tools/gen_eval_pose.py` called `_apply_photometric(work, rng, ph, W, H)` with neither. So the
   pose benchmark omitted **ink-contrast -- the worst of the twenty robustness factors**.
2. **Droplet occlusions never gated visibility.** `_apply_photometric` takes `holes_out`, and a
   strong refractive droplet appends an occluding rect to it (`dcc/synth.py:697`).
   `generate_sample` passes `holes_out=holes` and deliberately runs photometrics ABOVE its
   visibility loop -- its own comment at `:995-1000` says this is exactly why. The pose generator
   passed no `holes_out` AND computed visibility first, so droplet-occluded corners stayed
   labelled visible.

**The docstring is why this survived**: it asserted that "background prep, occlusion, photometrics,
geometric corner visibility -- is the exact dcc.synth machinery generate_sample uses". It was not.
The docstring now records the discrepancy and the one remaining KNOWN divergence: the pose sets
still omit SAM2 `_apply_cutouts` and its alpha visibility test, deliberately (object occlusion is
measured on its own robustness axis) and out of B1's scope.

Fix reuses the canonical `_warp_mask`, which takes a 3x3 and `warpPerspective`s it, so the pose
path's own `Hmat` drops straight in with no reimplementation of the board alpha.
- ~~**B2.**~~ **FIXED 2026-08-05** (`695bf03`) — this turned out not to be a judgement call: a
  verification gate that certifies a code path the run never executes is simply wrong, and fixing
  it changes no trained result. `check_init_loss_prediction` hardcoded focal's alpha=2/beta=4 on
  the analytic side and called `detector_loss` with no loss kwargs on the measured side, so every
  `loss_form: ce` arm had one loss predicted and a different one measured — neither the loss it
  trains under. It reported PASS regardless, because focal-vs-focal sits well inside the 0.5–2.0
  band. Both sides now resolve through `loss_kwargs()`, and the analytic term mirrors
  `dcc/losses.py`'s per-head branch.

  | arm | predicted | measured | ratio | forms |
  |---|---|---|---|---|
  | `rev640` (focal) | 46.2455 | 47.8738 | 1.035 | focal / focal |
  | `abl_ce` (ce) | 4209.47 | 4256.16 | 1.011 | ce / ce |

  The ce arm's real init loss is **89× larger** than the 47.87 the old gate was comparing against.
  The resolved forms and alpha/beta now appear in the check's own reported numbers, so the gate
  states which loss it verified. Effect on the campaign: no trained number moves; past ce arms
  were simply never loss-verified.

---

# P15 fp16 export gate — first run on trained weights (2026-08-05)

**It had never been run.** `tests/test_model.py` exports an UNTRAINED DetectorNet at 320x240 with
default kwargs, which proves the architecture family is exportable and nothing about the three
shipping checkpoints (width_mult 0.25/0.375, per-arm gates, 640x480). **No `.onnx` file existed
anywhere in the repo.** Deployment is a Jetson AGX Orin running a TRT fp16 engine, so this is a
gap in the deliverable, not a nicety. Tool: `tools/export_parity.py`. Data:
`paper/results_rev6/24_export_parity/`.

## 1. Exportability — all three tiers PASS

opset 17, `onnx.checker` clean, **zero banned ops** ({Complex, Loop, If}, same contract as the
unit test so it cannot fork).

| tier | nodes | size |
|---|---|---|
| 222k | 284 | 2.05 MB |
| 502k | 263 | 3.70 MB |
| 882k | 263 | 5.74 MB |

## 2. fp16 numerical parity — all three tiers FAIL, on the TAIL

n=1000 val images, paired fp32-vs-fp16 through the full `detect()` pipeline.

| tier | d p95 | d max | **>0.05 px** | ID diff | verdict |
|---|---|---|---|---|---|
| 222k | 0.00156 px | 4.021 | **107 (0.896%)** | 0.1005% | FAIL |
| 502k | 0.00156 px | 3.878 | **101 (0.833%)** | 0.0000% | FAIL |
| 882k | 0.00156 px | 3.878 | **104 (0.851%)** | 0.0164% | FAIL |

**THE FIRST VERSION OF THIS GATE WAS VACUOUS, TWICE, AND BOTH ARE INSTRUCTIVE.**

1. It compared `peaks()` output, which returns INTEGER coordinates -- so against a 0.05 px budget
   the delta could only ever be 0 or >=1, and it duly reported p95 = 0.00000 on the 222k tier.
   P15's sub-pixel budget can only be a statement about the REFINED corners, so the comparison
   moved to the full `detect()` output.
2. It then judged on p95 alone and reported 2 of 3 tiers PASSING. But the **mean exceeded the
   p95** (0.00282 vs 0.00156 px) -- arithmetically impossible without rare large outliers. Adding
   an explicit count of corners past 0.05 px flipped all three tiers to FAIL. A p95 of 0.0016 px
   hid a 0.85% tail completely.

Both are the same failure this audit closed in preflight (B2): a gate that returns green while
certifying something other than what it claims to check.

## 3. Attribution — MEASURED, by halving one net at a time

The tail count was near-identical across three DIFFERENT detectors (107/101/104) while the
refiner is shared, which pointed at the refiner. `--half {both,detector,refiner}` settles it
(882k, n=400):

| fp16 applied to | >0.05 px | d max | ID diff |
|---|---|---|---|
| both nets | 44 (0.908%) | 3.589 px | 0.0206% |
| **detector only** | **3 (0.062%)** | **1.000 px** | **0.0206%** |
| **refiner only** | **41 (0.846%)** | **3.589 px** | **0.0000%** |

**The decomposition is additive: 41 + 3 = 44.** It splits perfectly by failure mode:

- **The REFINER owns the sub-pixel position tail** -- 41 of 44 corners, and the entire max
  (3.589 px, identical to the both-nets run). It costs ZERO identity error.
- **The DETECTOR owns the integer peak flips** -- 3 corners, each at exactly 1.000 px (one cell),
  and **all** of the ID delta (0.0206% in both the both-nets and detector-only arms; the
  refiner-only arm is 0.0000%).

**DEPLOYMENT CONSEQUENCE.** The detector is effectively fp16-safe. The refiner is where fp16
costs accuracy, and it is ~97k params -- so keeping it at higher precision (a separate fp32
engine, or fp16 with an fp32 fallback) buys back 93% of the tail for a small latency cost. This
is a precision-policy decision, NOT a retraining one.

## A hypothesis that was tested and was WRONG

The ~4 px jumps looked like the `refine_min_peak` guard firing in one precision and not the
other, swapping refined for coarse. **Disabling the guard made it worse** -- max 3.35 px vs
0.81 px at matched n=300. The guard SUPPRESSES fp16 divergence, because it substitutes the
deterministic coarse peak in both precisions exactly where the refiner is least stable. Recorded
because it is the kind of plausible mechanism that would otherwise get written into the paper.

## Not covered

The exported graph's OWN numerics are unverified: `onnxruntime` is absent from the MLWS env, so
this gate covers export validity plus PyTorch fp16 and says so in its own report rather than
skipping silently. Installing onnxruntime would close the last leg (ONNX/TRT op-level agreement).

## 4. THE SHIPPABLE CONFIGURATION — detector fp16 + refiner fp32, certified at n=1000

Section 3 attributed the tail to the refiner, which implies a mixed-precision policy. That
implication is now MEASURED on all three tiers at the full n=1000, not asserted from the n=400
attribution run. Data: `paper/results_rev6/24_export_parity_detfp16/`.

| tier | >0.05 px, BOTH fp16 | >0.05 px, DET fp16 only | reduction | d max | ID diff | verdict |
|---|---|---|---|---|---|---|
| 222k | 107 (0.896%) | **8 (0.067%)** | 13x | 1.000 px | 0.1005% | FAIL (ID only) |
| 502k | 101 (0.833%) | **1 (0.008%)** | 100x | 1.000 px | 0.0000% | **PASS** |
| 882k | 104 (0.851%) | **5 (0.041%)** | 21x | 1.000 px | 0.0164% | **PASS** |

**Keeping the ~97k-param refiner in fp32 takes the position tail from ~0.85% to 0.008-0.067% --
inside P15's budget on every tier, including the one that fails.** `d_corner_p95` is 0.00000 px
and `d_corner_max` is EXACTLY 1.00000 px on all three, which is the signature of the only
detector-side position effect there is: an integer peak flipping one cell. There is no
sub-pixel detector drift at all.

**222k is the one exception, and only on IDENTITY: 0.1005% vs the 0.1% budget.** That figure is
IDENTICAL to four decimal places in the both-fp16 run, which independently confirms section 3's
split -- identity error is entirely the detector's, and an fp32 refiner cannot buy any of it
back. Two honest options, both Kaelin's call: run the 222k DETECTOR at fp32 as well (it is the
smallest net, so the cost is least there), or accept 0.1005% against a 0.1% line and say so.
Note the 222k tier already carries a harder deployment warning than this one -- 5.6% recall in
darkness (`PLAN_autonomous_campaign.md`) -- so fp16 identity is not its binding constraint.

**RECOMMENDATION OF RECORD:** ship 502k and 882k as detector-fp16 + refiner-fp32 TRT engines.
This is a precision-policy decision and needs no retraining. It was NOT free to discover: the
naive "quantise everything" configuration fails P15 on all three tiers.

## B1 — RESULT: the pose figures WERE optimistic; the tier conclusion SURVIVES

`eval_pose_rev6_b1` (same config `rev640.yaml`, same `pose_seed` 3000, same n=1000; the ONLY
change is the fix above). Kept alongside the original rather than overwriting it -- the
comparison IS the finding. Re-evaluated all three release tiers, same checkpoints, same refiner.

| tier | solve rate | M-05 rot median (deg) | M-05 rot p95 | M-06 trans median (sq) |
|---|---|---|---|---|
| 222k | 92.9% -> **90.0%** | 0.1167 -> **0.1269** | 1.3674 -> 1.2261 | 0.0081 -> **0.0099** |
| 502k | 95.2% -> **92.3%** | 0.1206 -> **0.1295** | 1.3740 -> 1.2297 | 0.0084 -> **0.0103** |
| 882k | 95.5% -> **93.6%** | 0.1213 -> **0.1318** | 1.3583 -> 1.1737 | 0.0085 -> **0.0104** |

**The audit was right and the correction is uniform**: solve rate falls 1.9-2.9 pp, rotation
median worsens ~8%, translation median worsens ~22%, on every tier. Refusals rise exactly where
expected -- `too_few` 68->93, 44->66, 43->60 -- i.e. the added ink-contrast/specular and the
droplet-gated visibility remove corners, and the lattice gate declines to fit rather than fitting
badly.

**Rotation p95 IMPROVES on all three** (1.367->1.226, 1.374->1.230, 1.358->1.174), which is not a
contradiction: the frames that got harder are now REFUSED rather than solved poorly, so the
surviving solution set is cleaner at the tail. A benchmark that refuses more and is better on the
survivors is behaving correctly for a docking controller.

**THE TIER CONCLUSION IS UNCHANGED.** Ordering is preserved (90.0 < 92.3 < 93.6, as 92.9 < 95.2 <
95.5), and rotation medians stay within 0.005 deg of each other across the ladder in BOTH sets. So
the standing claim -- "pose ACCURACY is indistinguishable across the ladder; what separates the
tiers is SOLVE RATE, and the small tier fails by not finding enough corners rather than by fitting
a bad pose" -- holds on the corrected benchmark, and now holds on one that is not photometrically
easier than train/val.

**HONEST CAVEAT ON THE DELTAS.** The two sets are NOT paired. The new photometric steps consume
rng, so the stream shifted and image *i* differs between sets -- the deltas therefore conflate
"harder photometrics" with "a different pose draw". The uniformity of the shift across three
independent tiers is what makes the direction trustworthy; the exact magnitudes are not
per-image attributable. Pairing would have required threading a separate rng for the added steps,
which changes the generator for every future sample -- not worth it to make one table prettier.

**USE `*_B1.json` FOR THE PAPER.** The originals are retained for provenance only.

## B1 — baselines re-run on the corrected set (SUPERSEDED TABLE — kept for the baseline deltas only)

> **DO NOT QUOTE THE `ours_*` ROWS BELOW.** They are the pre-ablation 4.7M `rev640_160k_rev6`,
> which is NOT a release model (Kaelin, 2026-08-05: final metrics come from the release models
> only). This run exists because it was produced with the SAME settings as the original
> `pose_fourway.json`, which is what makes the BASELINE old-vs-new deltas cleanly attributable to
> the pose-set correction. For any headline number use
> `13_pose/pose_release_vs_baselines_B1.json` — the three release tiers against both baselines.

`tools/eval_pose_fourway.py --pose-set eval_pose_rev6_b1`, with the SAME `--ckpt`,
`--refiner-ckpt` and `--dc-ckpt` as the original `pose_fourway.json`, so every arm moves for
exactly one reason: the corrected pose set. All four arms share `dcc.pipeline.pnp`, which isolates
the CORNERS rather than the PnP implementation. Data: `13_pose/pose_fourway_B1.json`.

| arm | solve rate | rot median (deg) | rot p95 | trans median (sq) | trans p95 |
|---|---|---|---|---|---|
| ours_coarse | 97.0 -> **96.0%** | 0.3908 -> 0.3971 | 2.2537 -> 2.3629 | 0.0290 -> 0.0306 | 0.3129 -> 0.3209 |
| ours_refined | 97.0 -> **96.0%** | 0.1156 -> 0.1379 | 1.2246 -> 1.3455 | 0.0089 -> 0.0106 | 0.1612 -> 0.1841 |
| deep_charuco | 96.1 -> **93.3%** | 0.8070 -> 0.9131 | 78.95 -> **89.36** | 0.0704 -> 0.0770 | 8.4921 -> 7.2695 |
| classical | 61.3 -> **54.4%** | 0.1338 -> 0.1602 | 2.0954 -> 2.3487 | 0.0178 -> 0.0200 | 0.1932 -> 0.2236 |

**EVERY arm degrades** -- which is the fix validating itself; a corrected benchmark that changed
nothing would have meant the correction did nothing.

**THE DEGRADATION IS NOT UNIFORM, AND THAT IS THE RESULT.** Solve rate falls by 1.0 pp for ours,
2.8 pp for Deep ChArUco, and **6.9 pp for classical OpenCV**. The old, photometrically-easy pose
set was FLATTERING THE BASELINES relative to us, so every margin in the paper was UNDERSTATED:

| margin (ours_refined vs) | on the old set | on the corrected set |
|---|---|---|
| Deep ChArUco, solve rate | +0.9 pp | **+2.7 pp** |
| classical, solve rate | +35.7 pp | **+41.6 pp** |
| Deep ChArUco, rot p95 | 64x better | 66x better |

That direction is mechanically sensible and was predictable once the root cause was known: the two
restored steps are NIR ink-contrast and specular. Ink-contrast attacks exactly the black/white
polarity that OpenCV's ArUco thresholding and marker decode depend on, so the classical detector
has the most to lose -- and it lost the most.

**Deep ChArUco's tail gets WORSE still: rot p95 78.95 -> 89.36 deg.** Its 93.3% solve rate must not
be read as comparable to our 96.0%: it fails by emitting a confidently wrong pose rather than by
refusing, so its solve rate counts frames that a docking controller could not act on. Ours refuses
(the lattice gate declines to fit) and the surviving poses stay tight -- rot p95 1.35 deg against
89.36. Classical has the opposite failure mode: it refuses 45.6% of frames outright but is
accurate on what survives.

**Bottom line: correcting B1 cost us ~1 pp of solve rate and widened our margin over both
baselines.** The honest benchmark is the better one for this work.

## THE HEADLINE CHART — all three release tiers vs both baselines, corrected pose set

`13_pose/pose_release_vs_baselines_B1.json`. Eight arms, ONE pass over the same 1000 frames of
`eval_pose_rev6_b1`, so the baselines are computed once rather than once per tier. Every arm's
corners go through the SAME `dcc.pipeline.pnp` (gate -> recovery -> PnP), which isolates the
CORNERS rather than the PnP implementation.

| arm | params | solve% | rot med | rot mean | rot p95 | tr med | tr mean | tr p95 |
|---|---|---|---|---|---|---|---|---|
| rel222 coarse | 222,138 | 90.0 | 0.3762 | 1.0216 | 2.4357 | 0.0288 | 0.1006 | 0.2787 |
| **rel222 refined** | 222,138 | 90.0 | **0.1269** | 0.6855 | 1.2261 | **0.0099** | 0.0691 | 0.1523 |
| rel502 coarse | 502,322 | 92.3 | 0.3817 | 0.9507 | 2.1863 | 0.0268 | 0.0866 | 0.2577 |
| **rel502 refined** | 502,322 | 92.3 | **0.1295** | 0.5759 | 1.2297 | **0.0103** | 0.0604 | 0.1509 |
| rel882 coarse | 882,402 | 93.6 | 0.3744 | 1.1176 | 2.3424 | 0.0289 | 0.0739 | 0.2698 |
| **rel882 refined** | 882,402 | **93.6** | **0.1318** | **0.5368** | **1.1737** | **0.0104** | **0.0376** | **0.1282** |
| deep_charuco | -- | 93.3 | 0.9131 | 10.6234 | **89.3603** | 0.0770 | 1.2636 | **7.2695** |
| classical | -- | **54.4** | 0.1602 | 0.8690 | 2.3487 | 0.0200 | 0.0654 | 0.2236 |

CROSS-CHECK: the three refined rows reproduce `21_pose_REL*_B1.json` EXACTLY (solve 90.0/92.3/93.6,
rot median 0.1269/0.1295/0.1318) despite coming from a different tool. The two agree.

### Reading it

**Deep ChArUco's SOLVE RATE is competitive and its ACCURACY is not.** At 93.3% it lands between
our 502k (92.3) and 882k (93.6) -- but its rotation p95 is **89.36 deg against our 1.17-1.23**, and
its translation p95 is **7.27 board squares against our 0.128-0.152**. Its mean rotation error,
10.62 deg, is nineteen times its own median: the distribution is dominated by catastrophic
solves. It fails by emitting a confidently wrong pose, where ours refuses. For a docking
controller those are not the same event, and solve rate alone must never be quoted as a
head-to-head.

**Even our COARSE arm beats Deep ChArUco on accuracy** -- 0.374-0.382 deg median against 0.913 --
i.e. our UNREFINED peak output is ~2.4x better than their detector's, before the refiner runs.

**Classical OpenCV is the accuracy runner-up and the availability loser**: median 0.1602 deg, but
it solves only **54.4%** of frames and its p95 (2.35 deg) is worse than every refined arm of ours.
Opposite failure mode to Deep ChArUco -- it refuses rather than lying.

**NEW, AND IT REFINES THE STANDING TIER CLAIM.** Median accuracy is flat across the ladder
(0.1269 / 0.1295 / 0.1318 -- the 222k is nominally BEST, which is noise), but the MEAN and the
TAIL improve monotonically with capacity:

| tier | rot mean | rot p95 | trans mean | trans p95 |
|---|---|---|---|---|
| 222k | 0.6855 | 1.2261 | 0.0691 | 0.1523 |
| 502k | 0.5759 | 1.2297 | 0.0604 | 0.1509 |
| 882k | **0.5368** | **1.1737** | **0.0376** | **0.1282** |

Translation mean falls 46% from 222k to 882k. So "pose accuracy is indistinguishable across the
ladder" is true of the MEDIAN and false of the TAIL: capacity buys tail robustness, exactly as it
does for identity under degradation in the robustness sweep. State it that way.

### Tool fixes this required (same class as the eval_pose_ours fix)

`eval_pose_fourway.py` built its detector from `--config` rather than the checkpoint's own cfg,
and built a bare `Refiner()` instead of inferring width via `refiner_for`. With the release tiers
that raises a shape error; the dangerous version is silent, where a checkpoint whose architecture
happens to match the config's shape loads and scores the WRONG model. Both now build from the
checkpoint, `detect()` receives each arm's own cfg (tau_hm, board), and `--refiner-ckpt` defaults
to `ref_s15_10k_rev6` -- the refiner the release evals actually used.

CAVEAT: the release rows here use that release refiner, while `pose_fourway_B1.json`'s
`ours_refined` row uses the older `refiner_rev6/ckpt_0013000` with the 4.7M model. Those two rows
are NOT comparable to each other. The BASELINE rows are unaffected and are comparable across both
files.
