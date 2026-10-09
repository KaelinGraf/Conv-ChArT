# Self-supervised fine-tuning from robot video — where the design got to

**Status: DESIGNED, NOT BUILT.** No code exists. Recovered from the session transcript
2026-07-29 and written down here because it had never been committed to permanent
storage — the plan below was agreed with Kaelin and then lived only in conversation.

Nothing here is measured. It is a design with its failure modes named.

---

## The mechanism, in one sentence

**The lattice gate is an automatic ground-truth machine**: on real footage, any
detection that survives geometric consistency is a pseudo-label whose quality is known,
because the board's rigidity is *external* information the network did not supply.

## Why it is self-improvement and not self-confirmation

This is the crux, and it is the reason the design is worth building at all.

Accepted frames take their labels from the **reprojected fitted lattice**, not from the
raw detections. The homography/pose fit denoises corner positions using board structure,
so the pseudo-label is *better than the network's own output on that frame*. Training on
it therefore moves the network toward the geometry rather than toward its own prior.

A self-training scheme that trained on its raw argmax would only sharpen existing
beliefs, including the wrong ones. The fit is what breaks that loop.

## Prerequisite: synthetic pretraining is not optional

Kaelin asked this directly and the answer is yes. The harvest is bootstrapped by the
model's own gated detections, so the model must already detect the deployment board well
enough that *some* frames pass the acceptance gate — otherwise the harvest is empty and
there is nothing to learn from. "Same marker type" means same dictionary family and board
geometry.

For a new board the sequence is: **synthetic pretrain (or class-head retarget via
`--retarget-from`) → deploy → harvest → SSL fine-tune.** The synthetic stage is the
ignition; SSL is the trim. It closes the appearance gap on top of a working detector; it
cannot replace it.

## Tooling as planned

**`tools/harvest_video.py`** — the only substantial new piece. Consumes robot footage +
camera K, runs the existing `dcc.pipeline.detect` per frame, applies a pseudo-label
acceptance gate:

- pose solved and **not** `ambiguous` (the IPPE ratio-1.5 flag — see
  `paper/results_rev5/10_uncertainty/NOTES_what_didnt_work.md`, where ambiguous poses
  measured NEES ~8,185 against an expectation of 6; they are unusable, not down-weightable)
- reprojection RMS under a hard bar
- at least N direct-head IDs at high `p_id`
- **temporal consistency with neighbouring frames** — a fluke that passes geometry once
  will not pass it across a motion sequence

**Output format**: the existing SD-05 record schema + provenance (source frame,
confidences, gate metrics) + a manifest with sha1. `dcc/targets.py` then renders training
targets from harvested frames with zero changes.

**Diversity capping**: bucket accepted frames by (s, tilt, brightness) and cap per
bucket. Without it the harvest concentrates on easy frames and teaches nothing — the
frames most likely to pass the gate are exactly the frames the model already handles.

**Training side**: a small `RealFrameSet` in `dcc/dataset.py` mixed into the stream at a
config ratio (`real_frac` ~0.1–0.3), fine-tuned at low LR from the trained checkpoint via
the existing retarget machinery. Short runs, **two-sided gate**: held-out *real* frames
must improve while synth val must not regress (the forgetting watch).

## The label amplification, stated precisely

The gate is per-FRAME, not per-corner: a frame passes or is discarded wholesale. But a
frame that passes yields a POSE, and a pose plus known board geometry reprojects **all
16 corners** — including the ones the network never found.

The minimum viable frame is 4 confidently-identified corners (PnP's floor) and it yields
a 16-corner label. The 12 undetected corners are new information the network did not
supply: they come from the board's rigidity and the solved geometry. At least 4x
amplification, and the 4 it did find come back improved, because the fit averaged over
all the evidence.

## THE OPEN HOLE: the lattice gives position, not visibility

Reprojection tells you where each corner IS, not whether it is OBSERVABLE. A corner
behind an occluder, or off the frame edge, reprojects to a perfectly good coordinate.

Label those as visible and you teach the network to **hallucinate corners through
occluders** — the exact opposite of the abstention behaviour the synthetic occlusion
training buys, and it would surface as a regression on the occlusion sweeps.

Off-frame is trivial to test. Occlusion is not: it is the same hard problem the synthetic
generator only solves because it KNOWS where its occluders are. Three options, none free:

1. mark reprojected-but-undetected corners `visible=false` — safe, but teaches the
   network to miss them; actively harmful
2. exclude them from the loss via a third "unknown" state — needs a masked loss, which
   `dcc/losses.py` does not currently have
3. accept the label only when enough corners were directly detected that the undetected
   few are plausibly occluded rather than missed

(2) is the right answer and it is a real change to the loss, not a harvest-side detail.
**This must be settled before any harvest code is written.**

## The bootstrap is a ratchet with a ceiling

Each round admits frames slightly harder than the last, because the model running the
gate is better than the one before it. But the expansion is **from the inside outward and
it cannot jump gaps**: a frame where no 4 corners are recoverable never enters the
corpus, however many rounds run.

So SSL closes the APPEARANCE gap (real sensor noise, optics, surfaces) far better than
the GEOMETRY gap. If deployment presents boards at s=200 and synthetic training stopped
at 128, SSL will not rescue that — synthetic coverage has to.

**Re-harvest each round, do not accumulate.** Appending bakes an early round's mistakes
in permanently and every later round trains on them; re-running the harvest with the
improved model lets bad labels fall back out.

## Where video specifically earns its name

Everything above works on a bag of unrelated images. What makes it VIDEO is that a frame
which FAILS the gate can inherit a pose from neighbours that passed — track through the
sequence and interpolate. That converts hard frames into labelled examples **without
ever detecting anything in them**, which is the only route to the genuinely-hard tail.
It is also the strongest confirmation-bias guard in the design, since a hard frame's
label originates in a different frame entirely.

## Confirmation-bias guards, named because they are the failure mode of all self-training

1. Geometric corroboration is **external** information (the board's rigidity, not the
   network's opinion)
2. Labels come from the **reprojected fit**, not the raw detections
3. Real fraction is **capped**, so synthetic diversity keeps anchoring the distribution
4. Synth-val **regression gate** catches drift
5. Optional: classical-detector **agreement audit** on a bright-frame subset — an
   independent second opinion where classical is reliable

## Relationship to the rest of the project

- The **differencing augmentation** discussed in the same conversation WAS built (rev-3
  onward, `differencing_p: 0.10`) — Kaelin's two-frame formulation with inter-frame
  motion shift, which derives the subtraction artifacts rather than imitating them.
- **Uncertainty as a first-class output** was the sibling item from the same wish list.
  Partially built, then parked — see the rev-5 uncertainty notes.
- No real-frame evaluation exists yet. Every result in this project is synthetic, which
  is the standing limitation this work would begin to address.
