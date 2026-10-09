# Conv-ChArT — 4–5 minute conference talk

**Running order, the figure for each slide, and what to say.** Figures are in
`paper/conference/figures/`, numbered in talk order. Every number quoted here is
traceable to a file listed under "source"; the deep technical detail lives in
`conference_notes.md` (and its PDF).

Budget: **4–5 minutes**. That is roughly **9 slides at ~30 s each**. The timings
below add to 4:30 and leave slack. If you are running long, the two cuttable slides
are marked **[CUT FIRST]**.

---

## 1 · Title — 20 s · *no figure*

> **Conv-ChArT: a Convolution–Transformer Hybrid Network for Robust Fiducial Marker Detection**

One sentence of framing: a learned ChArUco corner-and-pose detector for a barrier
robot that has to dock outdoors, in low light, against a marker board — where the
classical detector simply stops working.

---

## 2 · Architecture — 40 s · `figures/01_architecture.svg`

Speak to the parts; the diagram carries it.

- Conv encoder e1–e4 (paired 3×3, dilated 2/4 at the final stage) → **H/8 bottleneck**
- **2 pre-norm transformer blocks**, 8 heads, d=256, over a **60×80 = 4800-token** grid, 2D axial RoPE
- **Gated skip** (Oktay additive gate) into the decoder
- **Two factorised heads**: heatmap at full resolution, class map at H/4
- Separate **97 k-parameter refiner** for sub-pixel, on 24×24 crops

**Numbers if asked:** 4.70 M detector + 97 k refiner = **4.80 M total**. Attention is
33.6% of parameters, the e4 encoder stage 44.0%.
*Source: `conference_notes.md` §2, measured this session.*

---

## 3 · Headline result — 40 s · `figures/02_headline_panel_recall.png`

The money figure: **detection recall across four factors, four systems, identical
frames**. Ours (coarse and refined) sit together at the top; Deep ChArUco is a clear
band below; classical OpenCV is on the floor.

> "Across every stress factor we tested, on identical frames, we hold 96–99% recall
> where the fine-tuned Deep ChArUco baseline sits at 76–93% and classical OpenCV
> between 26 and 54%."

**[CUT FIRST if short]** `figures/03_headline_distance.png` — the distance axis is
the most dramatic single panel: at the far end, **ours 93.5%, Deep ChArUco 42.7%,
classical 0.6%**.

*Source: `paper/results_rev6/06_tables/fourway_summary.md`, `degradation.md`.*

The panel is **3x3, nine factors**, spanning the three families a deployed rig meets:

- **geometry** -- distance (merged in- and out-of-envelope, trained range marked), twist
  (in-plane rotation), tilt (out-of-plane)
- **sensor** -- SNR in dB, motion blur, defocus blur
- **scene** -- real-object occlusion (SAM2 cutouts, not synthetic rectangles), brightness,
  NIR ink contrast

The distance panel now runs the **merged** axis, log-scaled, with the trained envelope
(s = 12--128 px) shaded and marked by dashed verticals -- so degradation outside the
training distribution is visibly *outside* it rather than looking like ordinary failure.

**Say the caveat once, briefly — it buys credibility:** Deep ChArUco is run
**unrefined** (we do not run their RefineNet), so their curve belongs against our
*coarse* line; and classical's ID accuracy is ~100% by construction because it only
reports corners it has already identified — read its **recall**, not its ID.

---

## 4 · Design choice 1: the receptive-field problem — 45 s · `figures/04_close_range_distance.png`

Motivate it geometrically, then show that we solved it. **No ablation on this slide**
(see "why" below).

> "Identity needs board-scale context — to read a corner's ID you have to see an
> adjacent marker. At close range the board fills the frame: the marker you need is
> 300 pixels away, and the convolutional receptive field is nowhere near that. So we
> put self-attention at the bottleneck to give the identity read a *selective* global
> receptive field — global reach, but only where it's needed."

Then the figure, which is our model across the full distance axis:

> "ID accuracy holds between 96.6% and 99.9% across the whole range — including at
> 100% of frame width, where the board completely fills the image. And you can see
> the refiner's contribution on the left: a 4x reduction in localisation error."

**Numbers on the figure:** ID 99.0% at 100% frame width, 99.9% at 75%; localisation
median 0.42 px coarse to **0.093 px refined**.
*Source: `tables/robustness_id_acc.md`, `robustness_err_median.md`.*

**Why the A1 ablation is NOT here.** It is a good result *for the paper* and a bad
slide. In-envelope, conv-only is equal or better at 6 of 8 distance steps and only
loses at s=96 (-0.82 pp) and s=128 (-7.39 pp). The honest reading — "attention costs
nothing across the range and buys 7.4 pp exactly where the receptive field runs out"
— needs more than 30 s, and the figure invites "so the simpler model is better most
of the time?". Full result and figures: `tables/A1_ablation_README.md` and
`figures/reference/ablation_*.png`. **Have the s=128 number ready in case you are
asked**: 99.20% vs 91.82%.

---

## 5 · Design choice 2: the attention gate — 35 s · `figures/05_gate_alpha.png`

Selective feature anchoring: the decoder's skip connection is multiplied by a
learned, content-conditioned mask, so the bottleneck decides *which* high-resolution
features to re-admit rather than taking all of them.

$$\alpha = \sigma\!\left(\psi\left(\mathrm{ReLU}\left(W_x \ast s_{\downarrow} + W_g \ast g\right)\right)\right), \qquad \tilde{s} = s \odot \mathrm{up}(\alpha)$$

The initialisation is worth one sentence — it is a genuine design decision:
$\psi$'s **weight is zeroed** and its **bias set to +3**, so $\alpha \equiv
\sigma(3) \approx 0.953$ *exactly constant* at step 0. The gate starts as a true
no-op and training decides when to begin gating.

The figure is the learned $\alpha$ map: it lights up in a **lattice exactly on the
corner grid** — yellow where the skip is passed, dark between. The gate has learned
to re-admit high-resolution detail *at the corners* and suppress it elsewhere.

*Full equation and rationale: `conference_notes.md` §4.*

---

## 6 · Loss formulation — 30 s · `figures/06b_loss_target.png` + math on the slide

Figure: the input with its 16 ground-truth corners, and the TARGET heatmap as a 3D
surface -- the Gaussian is a thing you can see rather than a term in an equation.

Both heads share one penalty-reduced focal loss in logit space. Copy the LaTeX from
**`conference_notes.md` §5**, which has both equations plus the target definitions.

The two points worth saying aloud:

- The target is a **Gaussian**, not one-hot — a near-miss is forgiven in proportion
  to how near it is, via the $(1-Y)^\beta$ term.
- **One shared normaliser** across both heads: the batch total of visible corners.
  Per-channel normalisation would inflate the class-head loss ~16×.

---

## 7 · Introspection: attention heads — 45 s · `figures/06_attention_heads.png`

**The slide you wanted to editorialise on.** Measured over 40 frames on the trained model (`tools/attention_head_roles.py`),
splitting each head's attention mass into corner / marker / off-board, and reporting
**lift** (mass divided by that region's share of tokens, so 1.0 = uniform):

| block 1 | off-board mass | lift: corner | lift: marker |
|---|---|---|---|
| **head 3** | **7.8%** | 14.1 | **22.0** |
| **head 5** | **4.3%** | 19.9 | 20.0 |
| head 0 | 38.4% | 22.8 | 7.9 |
| head 6 | 41.6% | 23.8 | 4.3 |

- **Block 0: 0 of 8 heads read markers.** Every head is junction-focused.
- **Block 1: only 2 of 8 heads stay on the board at all** -- heads 3 and 5 put 7.8%
  and 4.3% of their mass off-board, where every other head leaks 27--82%.
- **Head 3 is the marker reader proper:** 22.0x enrichment on code cells against
  14.1x on junctions.

> "The model didn't just learn to attend to the board — it learned to be lazy about
> it in exactly the right way. One head does the expensive work of actually reading a
> marker; the rest triangulate off the corner lattice. That is the same algorithm our
> symbolic recovery pass uses, discovered independently one level down: identify a
> few, infer the rest by elimination."

**Honesty guard-rail — do not point at a specific head index on this figure.** The
panel shows one frame and one query token; which head *looks* spread varies frame to
frame. Quote the aggregate, gesture at the panel as illustration.

*Source: `paper/results_rev6/07_introspection/attention_head_roles.json`.*

---

## 8 · Introspection: encoder/decoder taps — 30 s · `figures/07_layers_slide.png`

16:9, depth left to right: `e1 -> e2 -> e3 -> e4 -> POST-ATTN -> d3 -> d2 -> d1`, two
individual channels per stage, signed (yellow +, blue -, green 0), **no channel
averaging** — averaging would cancel the sign-flipped channel pairs this network
carries.

> "Everything through e4 is local evidence — edges, checker texture, nothing
> corner-selective. Then the bottleneck, and at d3 — the first stage that sees the
> attention output fused through the gated skip — you get wells at exactly the 16
> lattice sites. That is the division of labour working: convolutions supply local
> precision, attention supplies the board-scale context."

The gate probe figure is **not** in the deck — the alpha map on slide 5 already makes
the point, and the probe's inside/outside ratio flips sign between frames (1.25 vs
0.51), which needs a caveat you do not have time for. It is in
`figures/reference/gateprobe_gate3_val2650.png` if asked.

---

## 9 · Ablations — 30 s · **[PLACEHOLDER — arms still running]**

Reserved. A1 (attention) is complete and is already used on slide 4. Still in flight:
one-hot-vs-Gaussian target, exclusive self-attention, and a tighter-$\sigma$ arm.
Fill from `paper/results_rev6/08_ablations/` when they term.

---

## 9b · Low-light filmstrip — 30 s · `figures/09_filmstrip_lighting.png`

Optional but strong, and directly comparable to Deep ChArUco's own Fig. 10 (same
0.6^k brightness rescale). Three systems down the rows, ten brightness steps across.
One frame throughout -- only the brightness scale changes.

| brightness | ours | Deep ChArUco | classical |
|---|---|---|---|
| 1.00 | 16/16 | 16/16 | 16/16 |
| 0.22 | **16/16** | 15/16 | 10/16 |
| 0.13 | **16/16** | 13/16 | **0/16** |
| 0.028 | **16/16** | **0/16** | 0/16 |
| **0.010** | **16/16** | 0/16 | 0/16 |

> "At the last column the whole image spans three grey levels and the board's
> black-to-white contrast is ONE. That is the quantisation floor -- one bit of signal --
> and we still recover all sixteen corners with the right IDs. The fine-tuned learned
> baseline recovers none three stops earlier; the classical detector failed four stops
> before that."

Verified rather than assumed: at k=9 the image max is 3 DN, std 0.688, board contrast
1 DN, and all 16 predictions land within the 4 px match radius.

---

## 10 · Deployment — 30 s · *no figure — all text*

Pull the bullet list from **`conference_notes.md` §7**. Headlines:

- **NVIDIA Jetson AGX Orin**, TensorRT fp16/INT8, exported from the bf16-trained checkpoint
- **833 GFLOPs/frame** measured; 15 Hz pose needs a 66 ms budget per lit+dark pair
- Orin at realistic sustained throughput: **42–83 ms fp16** (marginal), **21–42 ms INT8** (comfortable)
- Export is gated by a **parity check** — corner error $\Delta < 0.05$ px and ID accuracy
  $\Delta < 0.1\%$ on 1 k val images — before any engine is trusted
- ONNX opset 17 via the TorchScript path; all ops TensorRT-standard. `grid_sample`
  and all Stage-3 logic live in pipeline code, outside the exported graph

---

## Timing

| # | slide | s |
|---|---|---|
| 1 | title | 20 |
| 2 | architecture | 40 |
| 3 | headline result | 40 |
| 4 | receptive field | 45 |
| 5 | attention gate | 35 |
| 6 | loss | 30 |
| 7 | attention heads | 45 |
| 8 | encoder/decoder taps | 30 |
| 9 | ablations | 30 |
| 10 | deployment | 30 |
| | **total** | **5:05** |

Over by ~35 s against a hard 4:30. Cut `03_headline_distance.png` (fold into slide 3)
and trim slide 8 to the layers figure only.

---

## Three things NOT to claim

Recorded so they don't slip out under time pressure.

1. **Do not say attention helps at far range or under occlusion.** Measured: far is
   level-to-worse, occlusion is within noise (−0.32, −0.96 pp). Only close range.
2. **Do not point at a numbered head** in the attention figure (see slide 7).
3. **Do not present the ablation reference as the production model.** The ablation
   arms are 50 k-budget runs against a shared 50 k reference; the production
   checkpoint is 160 k. Different numbers, same protocol.
