# Ablation queue and the composite decision rule

Live record of what is running, what is queued, and the rule that decides the
conditional arm. Kaelin handed over autonomous execution of this queue on 2026-07-29.

## Standing scope

- **Admin only, no sweeps.** Kaelin: *"Don't worry about running performance sweeps etc
  till my say-so later, just do good admin in terms of making sure checkpoints and
  MODEL'S OWN PERFORMANCE are in their own ablation folders."* `tools/ablation_summary.py`
  writes each arm's own validation metrics to its own folder plus `HEAD_TO_HEAD.md`.
- **Report headline values on each completion**, 30-minute health reports in between.
- Run to 50k unless a run genuinely converges; early exit is permitted but must state
  the stop step and the gain-at-stop.

## The arms

Two concurrent trainers, 6 loader workers each (this machine has hard-locked twice on
kswapd `shmem_writepage` reclaim at higher aggregate worker counts).

| order | arm | sigma_hm | beta | keys vs rev640 | control |
|---|---|---|---|---|---|
| live | A-SIGMA1 | 1.0 | 4 | `sigma_hm` | reference |
| live | A-NODILATE | 2.0 | 4 | `e4_dilated` | reference |
| 1 | A-SIGMA05 | 0.5 | 4 | `sigma_hm` | reference |
| 2 | A-XSA+NODILATE | 2.0 | 4 | `xsa`, `e4_dilated` | reference |
| **3a** | **COMPOSITE** (conditional) | winner | 4 | `xsa`, `e4_dilated`, `sigma_hm` | reference |
| 3b | A-BETA0 | 2.0 | **0** | `focal_beta` | reference |
| 4 | A-GATES-OFF | 2.0 | 4 | `gates_enabled` | reference |
| 5 | A-WIDTH-HALF | 2.0 | 4 | `width_mult` | reference |

Control is **`abl_reference_50k_rev6`** and nothing else. `abl_reference_50k` is the
rev-2-trained ladder baseline; comparing against it confounds architecture with
generator revision, the exact error a full day was spent unwinding.

Everything except the sigma ladder shares the sigma=2 base, giving a clean 2x2 on
attention/dilation (reference | +xsa | +nodilate | +both) plus A-CE on the same base.
Kaelin: *"it would be bad to switch 3 things at once."*

## SUPERSEDED 2026-07-29 22:15 — the composite is now UNCONDITIONAL

Kaelin, after seeing that M-04 had saturated at ~99.1% for every arm and the
earns-its-keep bar was about to kill the composite on a metric that could no longer
discriminate: *"No, please still do the composite run, but do it AFTER the sigma=0.5
run. You can do beta=0 at the same time as sigma =0.5."*

**The composite is no longer gated on anything.** It waits on A-SIGMA05 only to learn
WHICH sigma to carry, not for permission to exist. The rule below is retained as the
record of what was decided beforehand and why it was overridden — it should not be
applied.

**Revised order** (event-driven; the queue script launches the first two then stops):

Corrected 23:15 — Kaelin: *"why would we not do Xsa+nodilate in parallel with sigma=0.5
like originally planned. Do that instead of with beta, then composite after."* beta0 had
been slotted into the parallel position; it belongs later.

| | arm | sigma_hm | beta | trigger |
|---|---|---|---|---|
| running | A-SIGMA05 | 0.5 | 4 | launched 22:56 |
| next | A-XSA+NODILATE | 2.0 | 4 | next free slot (parallel with A-SIGMA05) |
| then | **COMPOSITE** | winner | 4 | when A-SIGMA05 ends |
| then | A-BETA0 | 2.0 | **0** | when A-XSA+NODILATE ends |
| then | A-GATES-OFF, A-WIDTH-HALF | 2.0 | 4 | as slots free |

A-XSA+NODILATE must not trail the composite: it is the 2-key cell without which the
composite's 3-key result cannot be attributed to anything.

**Sigma selection for the composite**, pre-committed here BEFORE A-SIGMA05 reports so
the rule cannot be fitted to the result: take the sigma arm with the **better final
m01 p95**, provided its M-04 is not materially worse (within the +/-0.15 pp
inter-validation noise band). p95 is chosen because it is the only metric still showing
signal — A-SIGMA1 has held -0.076 to -0.081 px against the reference across nine
consecutive validations while M-04 converged to a ~0.1 pp spread across all arms — and
because tail error, not median, is what a docking controller is exposed to.

## The composite decision rule (SUPERSEDED — record only)

Kaelin, 2026-07-29, in final form: *"run the composite if either of the sigma runs come
back superior, and use the superior of the two. If neither of the sigma runs are
superior, then theres nothing to run"* — and A-XSA+NODILATE must itself have earned its
keep.

**"Earns its keep"**, at 50k against `abl_reference_50k_rev6`:

1. **M-04 >= +0.2 pp**, and
2. **m01 p95 not worse**

The +0.2 pp floor is not arbitrary: the reference's own M-04 moves +/-0.10-0.15 pp
between adjacent validations, so a smaller margin is not separable from run noise. The
p95 condition exists because a tail regression bought with a mean improvement is not a
win for a docking controller — a confidently wrong pose is worse than no pose.

**Resolution:**

- both A-XSA+NODILATE and at least one sigma arm earn their keep
  -> run the composite at the **better** sigma
  (`configs/abl_composite_s1.yaml` or `configs/abl_composite_s05.yaml`, both pre-built
  and one-key-verified), **before** A-BETA0
- otherwise -> straight to A-BETA0

The composite is a **best-model candidate, not an ablation**: three keys cannot
attribute anything. It tests only whether independently-earned gains COMPOSE, and is
readable only alongside the single- and double-key cells.

## Throughput: measured, understood, DELIBERATELY NOT TUNED

Kaelin, 2026-07-29: *"Leave it, I'd rather training be stable."* Do not re-litigate this
or "optimise" the loader mid-campaign.

The pipeline is **generation-bound on CPU, not GPU-bound**. Measured directly:
`generate_sample` costs **86.5 ms/sample on one core = 11.6 samples/s per loader
worker**, so throughput is ~`workers x 11.6` minus contention. Two arms at 8 and 6
workers give 65 + 62 = 127 samples/s aggregate.

`nvidia-smi` showing 99-100% is MISLEADING here -- it reports "any kernel active", not
saturation. Two independent confirmations that the GPU is not the limit: most loader
children sit in `Sl` (sleeping), and the reference run **solo** averaged only 67
samples/s in its last 500 steps, so two concurrent arms nearly double aggregate
throughput. A GPU-bound pipeline could not do that.

The Rev G "140-142 samples/s" figure is a GPU-STEP microbenchmark on synthetic tensors.
The full rev-6 pipeline has never reached it and should not be expected to: rev-6's
realism upgrades (scene integration, SAM2 cutouts, relight, Poissonian-Gaussian noise,
FPN) cost roughly 30-40% throughput versus rev-5, which is visible in the reference's
own curve (first-500 79 samples/s -> last-500 67).

**Why not simply add workers:** 12 workers currently leaves ~16 GiB RAM free. The
16-worker config exhausted all 62 GiB and hard-locked this machine twice. With arms
still queued, a lock-up costs far more than the ~20% on offer.

Known, accepted, not a defect: **A-SIGMA1 runs 8 workers, A-NODILATE 6** -- SIGMA1
predates the 6-worker standard. It affects wall-clock only, never the metrics.

The real lever, for AFTER the conference: cache decoded backgrounds (the JPEG decode
and histogram match are repeated work). Not to be done mid-campaign -- it alters the
data stream and would break cross-arm comparability.

## Why the queue script stops at a decision point

`tools/queue_ablations.sh` launches A-SIGMA05 and A-XSA+NODILATE and then exits. If
beta0 were appended, it would launch the moment A-SIGMA05 finished — roughly 0.6 h
BEFORE A-XSA+NODILATE reports, hence before the composite decision can be made, silently
violating the "composite before beta0" ordering. One slot idles for that 0.6 h; that is
the accepted cost of an unambiguous sequence.

`tools/wait_trainer_exit.sh` runs in the background and returns whenever the set of
running trainers changes, so completions are noticed immediately rather than at the next
30-minute tick.
