# Conference folder — start here

Everything for the 4–5 minute talk, self-contained. Duplicated from
`paper/results_rev6/` on purpose so nothing here depends on that tree.

```
conference/
  README.md                 this file
  SLIDES.md                 running order, figure per slide, what to say, timings
  conference_notes.md       every technical detail, section by section
  conference_notes.pdf      ^ rendered
  figures/                  the 9 talk figures, numbered in talk order
    01_architecture.svg     ... 09_scale_generalisation.png
    reference/              all 53 figures, for questions
  tables/                   every table and raw data file worth quoting
```

## The talk figures

| # | file | slide |
|---|---|---|
| 01 | `architecture.svg` | architecture |
| 02 | `headline_panel_recall.png` | **headline result** -- 4 factors x 4 systems |
| 03 | `headline_distance.png` | headline, distance axis (cut first if short) |
| 04 | `close_range_distance.png` | design choice 1 -- receptive field |
| 05 | `gate_alpha.png` | design choice 2 -- attention gate |
| 06 | `attention_heads.png` | introspection -- attention |
| 07 | `layers_slide.png` | introspection -- encoder/decoder taps (16:9) |
| 08 | `scale_generalisation.png` | spare / backup |
| 09 | `filmstrip_lighting.png` | **low-light filmstrip** -- 3 systems x 10 brightness steps |
| 06b | `loss_target.png` | loss slide -- input + 3D Gaussian target |

**Deliberately NOT in the deck** (all in `figures/reference/`):
`ablation_*.png` -- the A1 attention ablation is a good paper result and a bad slide:
conv-only is equal or better at 6 of 8 in-envelope distance steps and only loses at
s=96 and s=128, so the figure invites the wrong question in a 30-second slot. Keep the
s=128 number (99.20% vs 91.82%) ready for questions.
`gateprobe_*` / `gateflow_*` -- the alpha map on slide 5 makes the point more clearly;
the probe's inside/outside ratio flips sign between frames and needs a caveat there is
no time for.

`01_architecture.svg` is SVG — PowerPoint, Keynote, Google Slides and LibreOffice
all import it directly. There is no SVG→PNG converter on this machine, so if a tool
refuses it, screenshot at high zoom.

## Where the numbers come from

| you want | look in |
|---|---|
| 4-system headline recall | `tables/fourway_summary.md` (`.tex` for LaTeX) |
| how far each system degrades | `tables/degradation.md` |
| per-metric robustness | `tables/robustness_{recall,id_acc,err_median,err_p95}.md` |
| A1 attention ablation | `tables/A1_ablation_README.md`, `A1_ablation_regimes.md` |
| attention head specialisation | `tables/attention_head_roles.json` |
| refiner guard | `tables/refiner_guard_sweep.json`, notes §6.1 |
| loss equations, gate equation, RoPE | `conference_notes.md` §3–§5 |
| params, FLOPs, deployment | `conference_notes.md` §2, §9 |
| every step of every sweep | `tables/all_steps.csv` |

## Still open at time of writing

- **Ablations slide is a placeholder.** A1 (attention) is done and already used on
  slide 4. One-hot-vs-Gaussian, exclusive self-attention, and a tighter-σ arm were
  still training; fill from `paper/results_rev6/08_ablations/` when they term.
- No real-frame evaluation — all results synthetic. See notes §10.

## Three things not to claim

Repeated from `SLIDES.md` because they are the ones that could slip out under time
pressure:

1. **Attention does not help at far range or under occlusion.** Measured: far is
   level-to-worse, occlusion within noise. Close range only.
2. **Do not point at a numbered attention head** on the figure — per-frame appearance
   varies and the per-panel normalisation misleads. Quote the aggregate.
3. **The ablation reference is not the production model** — ablations are 50 k-budget
   arms against a shared 50 k reference; production is 160 k.
