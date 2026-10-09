# Why focal on the class head moves localisation: the trunk-gradient share (measured 2026-10-09)

Kaelin, 2026-10-09: "so there's no good explanation for why focal loss helps for the lower resolution class head?"

The record established *that* `hm=BCE, cls=focal` gives the best p95 and tail at 882k (0.6874 / 0.0081% vs
0.6970 / 0.0114% for all-BCE, 35k; `PLAN_autonomous_campaign.md` 2026-08-01/02) and that the two heads are
driven by different mechanisms (beta = 0 decomposition), but never measured *why a class-head loss form moves
the heatmap's localisation*. The candidate mechanism is loss balance through the shared trunk: on a dense
16-channel map at ~0.005% positives, plain BCE spends most of its gradient driving ~19,200 easy background
cells per channel toward zero (cells `read_ids` never reads: identity is read only at detected peaks, at
`tau_id` 0.5), whereas focal's `p^alpha` factor discounts them once they are easy. If so, the class head's
claim on the shared trunk's gradient should be smaller under focal, leaving the trunk tuned more by the heatmap.

## Method (`cls_form_gradient_share.json`; script in the session scratchpad, reproduced below)

Same 128 `SynthVal` frames (indices stratified over the 10k set, so over `s`), batches of 16, bf16 autocast
forward as in training, fp32 losses: `L_hm = BCE(hm)/N`, `L_cls = lambda * form(cls)/N`. For each term alone,
the L2 norm of its gradient over every trunk parameter (everything except `hm.*` and `cls.*`, 842,097 params),
and over the attention blocks only; plus the cosine between the two trunk gradients. **The class form is
swapped on the same frozen weights**, so the form's effect is separated from the training trajectory.

## Result

| checkpoint (trained with) | class form evaluated | L_cls / L_hm | trunk grad cls : hm | cls share of trunk grad | on attention blocks cls : hm | cos(g_hm, g_cls) |
|---|---|---:|---:|---:|---:|---:|
| `abl_wh_ce_35k` (BCE / BCE, lambda 1.0) | BCE (own) | 0.53 | 0.98 | 50% | 1.29 | +0.29 |
| same weights | focal | 0.30 | 0.88 | 47% | 1.51 | +0.29 |
| `abl_lx_wh_hmce_clsfocal_35k` (BCE / focal, lambda 1.0) | focal (own) | 0.13 | 0.50 | 33% | 0.74 | +0.16 |
| same weights | BCE | 14.3 | 38.9 | 97% | 234 | +0.03 |
| `rel_w882_c2_100k` (BCE / focal, lambda 2.0, release) | focal (own) | 0.19 | 0.57 | 36% | 0.70 | +0.31 |
| same weights | BCE | 19.7 | 39.0 | 97% | 203 | +0.11 |

## Reading

- **At the end of training, the focal-class models give the heatmap a larger share of the trunk's gradient**:
  the class head's share is 33-36% against 50% in the all-BCE model, and on the attention blocks the class
  head goes from dominating (1.29 : 1) to minority (0.70-0.74 : 1). This is the direction the loss-balance
  mechanism predicts, and it is the first measurement behind the campaign's "cls=focal is a localisation
  lever" reading.
- **A focal-trained class head leaves its easy negatives un-crushed**: scoring it with BCE gives a class loss
  14-20x the heatmap loss and a trunk gradient 39x the heatmap's. Focal stopped pushing those cells once they
  were easy; BCE never does. On the all-BCE model the swap barely matters (0.98 -> 0.88) because its background
  is already near zero. So the form's effect is cumulative over training, not a property of one gradient step.
- **The heads are not in conflict**: the two trunk gradients are nearly orthogonal (cosine 0.03-0.31). The
  mechanism is about which head's update dominates the shared parameters, not about the heads pulling against
  each other.

## Caveats, stated

- A snapshot at the end of training on 128 frames, not the training trajectory; AdamW's per-parameter
  normalisation weakens any argument made from raw gradient magnitudes, so this is consistent with the
  mechanism rather than a proof of it.
- The localisation effect it would explain is small in absolute terms (p95 -0.0096 px, 4.8x the 882k floor; tail
  0.70x), and the identity cost of focal at 882k (-0.42 pp at 35k, recovered to -0.14 pp by lambda 2.0, inside
  1.4x the floor) has no measured mechanism either; the natural hypothesis (less-suppressed off-channel
  probabilities near `tau_id` 0.5) is untested.
- The standard argument for focal on a dense, extremely imbalanced one-vs-rest map (Lin et al. 2017) applies
  to the class head as written in `dcc/losses.py`; what this note adds is the measured consequence for the
  shared trunk.

## Reproduce

```
P="env PYTHONPATH= /home/kaelin/anaconda3/envs/MLWS/bin/python"
# the script lives in tools/ as of this note's commit: tools/cls_form_gradient_share.py
$P tools/cls_form_gradient_share.py
```
