# rev-6: integrate the board into the scene (deferred, NOT for Friday)

**Kaelin, 2026-07-28, after reviewing 15 random training draws** (`random_train_samples_15.png`):

> *"The board seems to occupy the mid range, and be very visually obvious, a lot of
> the time. This is fine for now (looks good for results), but the board really
> needs to be degraded into the scene better, have more occlusions, and just
> generally be degraded a bit more. It's very visually obvious that the board is
> like cut into the scene."*

Correct diagnosis. This is **the** limitation of the current results and the main
sim-to-real risk. Explicitly deferred — the numbers stand for the conference, and
this is the first thing to fix afterwards.

## What was ruled OUT first (all firing correctly, measured over 200 draws)

| | measured | config |
|---|---|---|
| negatives | 4.5% | 5% |
| rectangle holes | 42.5% | 40% |
| **object cutouts** | **31.5%** | 35% (bank of 14,721 loaded) |
| s_px | median 38.4, 23% below 20 px, 30% above 64 px | range [12, 128] |

Nothing is silently disabled, and severity back-off (rev-5) did not cause this.
**The problem is the compositing model, not the augmentation rates.**

## Root cause, verified in source (`dcc/synth.py:_composite_board`)

The whole composite is:

```
matched = match_histograms(board_render, bg_crop)      # GLOBAL, whole-crop
warped  = warpPerspective(matched, H)
m       = _warp_mask(H, ...)
work    = bg_crop * (1 - m) + warped * m               # straight alpha paste
```

Four consequences, in descending order of how much they give the game away:

1. **No local relighting.** Brightness is matched to the crop's *global* histogram,
   so a board sitting on a dark region of a brightly-lit scene keeps the scene's
   average brightness, not the local one. This is the single biggest tell.
2. **No contact shadow.** A real board resting on or against a surface darkens it.
   Nothing here ever does.
3. **Effectively hard alpha edge.** `_warp_mask` + INTER_LINEAR gives ~1 px of
   softness and no more. Note the inconsistency: `place_cutout` **explicitly
   feathers** its alpha (`GaussianBlur (3,3)`) and `_composite_board` does not —
   so pasted *objects* blend better than the board does.
4. **No depth-consistent defocus.** Global blur is applied to the whole composite,
   so board and background always share a focal plane.

NOT a problem, checked: sensor noise/grain IS shared, because `_apply_photometric`
runs on the composite rather than on the board alone.

## Proposed rev-6, ordered by realism gained per unit of effort

1. **Local relighting (biggest win, ~10 lines).** Take a heavily blurred luminance
   field of `bg_crop` (sigma ~ 1/8 of frame width), normalise it to its own mean,
   and multiply the warped board by it before compositing. The board then inherits
   the scene's low-frequency illumination gradient — shadowed regions darken it,
   bright regions lift it — without touching geometry or labels.
2. **Feather the board mask** to match `place_cutout`'s treatment. One
   `GaussianBlur` on `warped_mask`. Removes the razor edge. **Verify labels are
   unaffected**: corner geometry comes from `H @ p_render` and never from the mask,
   so this is photometric-only — but re-run the round-trip audit gate to confirm.
3. **Contact shadow.** Darken a soft, offset copy of the board mask into the
   background before compositing (offset direction from the sampled illumination
   lobe where differencing is active, otherwise a random light direction).
4. **Local rather than global tone match.** Match the board to the *region under
   the warped mask* instead of the whole crop.
5. **More occlusion over the board specifically.** Cutout placement is currently
   frame-uniform, so at small `s` the board is rarely occluded. Bias a fraction of
   placements toward the board quad.
6. **Shift scale mass downward.** Median 38.4 px is 30% of frame width; Kaelin notes
   the mid-range dominates visually. Consider log-uniform weighting toward the small
   end, or widening below 12 px.
7. **Depth-consistent defocus** — separate blur for board vs background. Most
   expensive, least certain payoff; do last.

## How to know it worked

Realism is not self-evidently measurable, so use a **discriminator probe**: train a
small CNN to classify "board region vs background region" from local patches. If it
separates them trivially today and struggles after rev-6, the gap has genuinely
closed. Chasing it by eye will not converge.

Expect **all headline numbers to DROP** after rev-6. That is the point — the current
99.62% M-04 partly measures how easy the compositor makes the board to find.

## What defends the current numbers in the meantime

They are **synthetic-to-synthetic**, and that must be stated. The defence is
relative rather than absolute: on the **same** easy data, Deep ChArUco fine-tuned
reaches ~86.7% and classical OpenCV 44.5%. If the data were trivially easy, both
would be near-perfect too. The comparative claim survives; the absolute number
should be presented as an upper bound.
