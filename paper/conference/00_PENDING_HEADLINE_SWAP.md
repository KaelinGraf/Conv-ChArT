> **SUPERSEDED IN PART, 2026-07-30 — read `TIERED_MODELS.md` FIRST.** Kaelin's decision moved
> from "pick the better of two 882k arms" to presenting the width ladder as a **tier menu**
> (L 3.61M / M 979k / S 319k, one architecture, `width_mult` the only difference). The
> single-winner framing below is no longer the plan. What still stands: the regeneration
> checklist itself, which now runs **once for the headline tier**, with the architecture SVG
> annotated as parameterised by width rather than hard-coding one set of channel counts.
> All cost numbers in this file are superseded by the measured table in `TIERED_MODELS.md`.

# PENDING: swap the conference headline model to the small arm

**Decision taken 2026-07-30 by Kaelin, before the results landed:** *"once we have trained
both nodilate+half_width and the nodilate+half_width+xsa models, whichever one is better we
will change our headline results to the conference for ... Being able to beat their model at
a fraction of the parameters will be even more impressive."*

This is a **regeneration task, not a re-analysis task**. Every artefact below is produced by
a tool that takes `--ckpt` and `--config`; nothing needs rewriting, only re-running against
the new checkpoint. The risk is not difficulty, it is **missing one** and shipping a deck
where eight figures describe one model and one describes another.

---

## The decision

| candidate | run | params (det) | +refiner | vs Deep ChArUco |
|---|---|---|---|---|
| A | `abl_nodilate_width_half_s05_50k_rev6` | 882,402 | 979,458 | **0.44x** |
| B | `abl_nodilate_width_half_xsa_s05_50k_rev6` | 882,402 | 979,458 | **0.44x** |

Both carry `e4_dilated: false`, `width_mult: 0.5`, `sigma_hm: 0.5`. **They differ in exactly
one key, `xsa`, and have IDENTICAL parameter counts** -- XSA is parameter-free, so the
comparison is clean and the swap decision costs nothing in model size either way.

**Pick on the same rule used for the sigma ladder:** better final **m01 p95**, provided
M-04 is not worse by more than the +/-0.15 pp inter-validation noise band. p95 because it is
the metric that has kept discriminating after M-04 saturated near 99%, and because tail
error is what a docking controller is exposed to.

**Reference numbers to beat (Deep ChArUco, fine-tuned, converged):** M-04 **85.44%**,
2,241,235 params total (ChArUcoNet 1,242,002 + RefineNet 999,233, counted from their source).
At step 10,000 the half-width arm was already at **98.25%** M-04 -- 12.8 pp above their
converged number at 2.3x fewer parameters.

---

## What must be regenerated -- ALL of it, against the winning checkpoint

Tick these off in one pass. Every one currently depicts `runs/rev640_160k_rev6/ckpt_0160000.pt`.

- [ ] **3x3 headline robustness panel** -- `tools/robustness_sweep.py` then
      `tools/plot_four_way.py`. This is the expensive one (full sweep over distance,
      rotation, tilt, SNR, motion blur, defocus, object occlusion, darkness, ink contrast).
      Budget hours, not minutes. The 4-way comparison arms (Deep ChArUco, classical) do NOT
      need re-running -- only our two arms.
- [ ] **Distance / `distance_extrap`** -- the merged log-scale figure with the trained
      envelope shaded, s = 6..256.
- [ ] **Architecture SVG** -- channel widths change throughout (e1 16, e2 32, e3 64, e4 128,
      attention d=128, head dim 16), and the **dilated pair must be removed from the
      diagram** entirely. The current SVG shows the (2,4) dilation on e4; that block does
      not exist in the new model. Parameter annotations all change.
- [ ] **Introspection panels** -- `tools/introspect.py --panels pipeline,features,attention,
      heatmap3d,gates`. Note the feature slide picks the top-N channels by spatial sd, and
      the channel INDICES will differ; that is expected, not a bug.
- [ ] **Darkness filmstrip** -- `tools/filmstrip_lighting.py` (gitignored). 3xN over the
      brightness ramp, ours vs Deep ChArUco vs classical.
- [ ] **Pose table** -- `tools/eval_pose_fourway.py` then `tools/pose_table_pdf.py`.
      Regenerates from JSON, so this one is nearly free.
- [ ] **Model cost** -- `tools/model_cost.py` (params, MACs, Orin estimate) and
      `tools/vram_report.py`. The headline "5.3x smaller, 4.4x fewer MACs" comes from here.
- [ ] **Working range** -- `tools/working_range.py` only if the reliable-s band moves; it is
      read from the robustness sweep, so re-run AFTER the sweep lands.
- [ ] **`paper/conference/SLIDES.md` and `conference_notes.md`** -- every quoted number.

## Numbers that change in the prose, not just the figures

- Parameter count: 4,698,034 -> **882,402** (detector), 4,795,090 -> **979,458** (with refiner)
- MACs/frame: 84.33 G -> **13.66 G**; GFLOPs 168.7 -> **27.3**
- Orin fp16 estimate: 61-122 fps -> roughly **5x faster**, re-derive with `model_cost.py`
- "25.1% fewer parameters" (the nodilate line) becomes **"81.2% fewer"**
- The Deep ChArUco comparison gains a parameter axis it did not have: **2.3x fewer
  parameters AND better accuracy**, which is a strictly stronger claim than accuracy alone.

## Two things to hold honest in the swap

1. **The refiner is unchanged.** It was trained against the FULL-width detector's coarse
   peaks. Its 97,056 params and its guard (`refine_min_peak` 0.3) carry over untouched, but
   the crop distribution it sees will differ slightly if the small model's peak positions
   differ. Worth a re-run of the refiner gain measurement; not worth retraining before Friday.
2. **Budget.** A 50k arm finishes ~5 h after launch; the full robustness sweep is the long
   pole after that. If time runs short, the priority order is: **3x3 panel > pose table >
   architecture SVG > introspection > filmstrip**. The first two are the claims; the rest
   are illustration.
