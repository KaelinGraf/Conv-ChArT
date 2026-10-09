# Deferred arms — built or specified, deliberately not queued

Kaelin, 2026-07-30: *"i dont think we need to bother anyway with the really thorough
attribution just yet, if it turns out to be a very strong result we can consider it. we
will keep the xsa addition to the nodilate + width_half in the back pocket for now."*

These are **not** forgotten work and **not** blocked work. Each is a deliberate hold with
a stated trigger. If a trigger fires, the arm is worth running; if it does not, the arm
should stay unrun and that is the correct outcome.

---

## 1. A-XSA on top of nodilate + width_half  — the back-pocket lever

**Status: not built.** Would be three keys off `configs/rev640.yaml`
(`e4_dilated: false`, `width_mult: 0.5`, `xsa: true`) plus the tuned `sigma_hm`, so a
one-line derivation from `configs/abl_nodilate_width_half_{s05,s1}.yaml`.

**Trigger:** A-NODILATE+WIDTH_HALF (882,402 params, queued) comes back materially worse
than the nodilate baseline, AND the suspicion is that the narrowed attention is the
binding constraint rather than the thinner conv trunk.

**Why XSA is the right lever for that specific failure.** XSA is **parameter-free** — it
changes how attention normalises, not how much capacity it has. Measured on the 2x2 grid,
it produced the largest attention-concentration effect of any arm: bottleneck entropy fell
from the reference's ~7.0 nats to **4.96** at step 5k, versus 5.86 for nodilate and 5.25
for the two combined. So if d=128 is too narrow, XSA is the cheapest way to make a narrow
attention spend its capacity better, at zero parameter cost.

**Caveat carried forward:** A-XSA alone measured NULL on the per-regime sweep at full
width (all regimes within +/-0.8 pp), and its 50k arm is only at 24,026 steps pending
resume. A null at full width does not imply a null at half width -- the hypothesis is
precisely that exclusivity matters *more* when capacity is scarce -- but it does mean the
prior is weak.

---

## 2. Decoupled capacity attribution — conv width vs attention width

**Status: needs a small code change, not built.**

`width_mult` couples the two capacities: `dcc/model.py:213` does `d = c(d)`, so halving
the width halves the attention dim too (256 -> 128, head dim 16) and attention params fall
-74.9% (1,580,032 -> 396,800). **`configs/abl_width_half.yaml` has the same coupling**, so
running it does not separate them either.

Decoupling needs an attention-width multiplier independent of `width_mult` (~3 lines in
`model.py` plus a config key). That would give the real 2x2:

| arm | conv channels | attention d | isolates |
|---|---|---|---|
| nodilate (done) | full | 256 | baseline |
| nodilate + width_half (queued) | half | 128 | both at once |
| nodilate, half conv, d=256 | half | **256** | **conv capacity** |
| nodilate, full conv, d=128 | full | **128** | **attention capacity** |

**Trigger:** only if nodilate + width_half is a **very strong result** worth defending in
detail, or a clear loss worth explaining. Kaelin's call: not worth the GPU before that is
known.

---

## 3. A-BETA0 — the clean split of the loss claim

**Status: BUILT and verified** (`configs/abl_beta0.yaml`, one key: `focal_beta: 0`).
Displaced from the queue on 2026-07-30 in favour of nodilate + width_half.

**What only this arm can do.** A-CE beat the reference (+0.52 pp M-04, 9x lower >4 px
tail) but it removes **two** things at once: the Gaussian grading AND focal's alpha
easy-negative modulation. beta=0 removes only the grading -- `(1-Y)^0 = 1` everywhere, so
the graded target is ignored and Y survives only as the binary `Y == 1` mask -- while alpha
stays. So beta=0 is **standard focal loss on a one-hot target**, sitting exactly between
the reference and A-CE:

| arm | alpha | beta | equals | the step it isolates |
|---|---|---|---|---|
| reference | 2 | 4 | CornerNet penalty-reduced focal | --- |
| **A-BETA0** | 2 | 0 | RetinaNet focal, gamma=2 | vs ref: **the Gaussian grading** |
| A-CE | 0 | 0 | plain BCE | vs beta0: **alpha, the imbalance fix** |

**Trigger:** if the loss/target claim needs defending in the paper beyond "the tuned sigma
wins". Right now the sigma ladder (2.0 -> 1.0 -> 0.5) carries that argument on its own and
is more directly useful, since sigma is a hyperparameter we would actually ship.

**Related open oddity worth a sentence if this ever runs:** under plain BCE the background
should hold ~99.8% of the loss on a 640x480 frame (measured), which predicts A-CE should
have trained badly. It did not -- it is the best arm on the board. Either our positives are
dense enough (16 per frame in a fixed lattice, not one rare object) or our background is
learnable enough that 99.8% of nearly-nothing is still nearly nothing. beta=0 is the arm
that would tell us which.

---

## 4. A-GATES-OFF — still queued, not deferred

Listed here only to avoid confusion: `configs/abl_gates_off.yaml` is **in** the queue,
last, after the A-XSA and A-CE resumes. Bypass rather than deletion, so parameter counts
stay comparable.
