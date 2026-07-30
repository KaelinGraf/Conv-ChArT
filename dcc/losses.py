"""Losses: CornerNet-style penalty-reduced focal (logit-space, sum
reduction, alpha=2, beta=4), shared across the detector's two heads and the
refiner. detector_loss and refiner_loss both call the focal() below unchanged
-- the refiner's differently-shaped target needs no change to the loss form,
only a different normalisation (see refiner_loss)."""
import torch
import torch.nn.functional as F


def focal(logits, y, alpha=2, beta=4):
    """CornerNet penalty-reduced focal, logit space (bf16-safe), sum reduction.
    Positives are the exact-1.0 cells (targets.py forces them reachable)."""
    z = logits.float()
    pos = y == 1.0
    l_pos = (1 - torch.sigmoid(z)) ** alpha * F.logsigmoid(z) * pos
    l_neg = (1 - y) ** beta * torch.sigmoid(z) ** alpha * F.logsigmoid(-z) * ~pos
    return -(l_pos.sum() + l_neg.sum())


def strict_bce(logits, y):
    """ABLATION A-CE counterpart to focal(): the STRICT reading of the same
    targets. Only the exact-1.0 cells are positive; every other cell is an
    equally-wrong negative, with no Gaussian partial credit and no (1-y)^beta
    penalty reduction. Same logit-space, same sum reduction, so the only thing
    that changes between the two arms is how PERMISSIVE the target is.

    This is the direct test of the claim that a permissive Gaussian target
    improves training stability and expressiveness: focal() forgives a
    near-miss in proportion to how near it is, strict_bce() does not."""
    z = logits.float()
    pos = (y == 1.0).float()
    return F.binary_cross_entropy_with_logits(z, pos, reduction="sum")


def detector_loss(hm_logit, cls_logit, hm_t, cls_t, n_vis_batch, lam=1.0, loss_form="focal",
                   beta=4):
    """Batch-normalised by total visible corners, shared N for both heads;
    clamped so N=0 batches (all-negative, no visible corners) divide by 1
    instead of by zero. loss_form="ce" swaps BOTH heads to strict_bce (A-CE).

    beta is the ablation lever for Kaelin's MAIN CLAIM (2026-07-29). focal()'s
    POSITIVE set is `y == 1.0` -- identical to strict_bce's -- so the Gaussian
    target's graded values enter in exactly ONE place: the (1-y)^beta penalty
    reduction on negatives. A-CE therefore removes TWO things at once (focal's
    alpha easy-negative modulation AND the Gaussian grading) and cannot attribute
    a failure to either; on a dense head at 0.005% positives the alpha term is
    the one that dominates, which is standard focal-loss literature and not a
    novel claim. beta=0 makes (1-y)^0 == 1 for every cell, so all negatives take
    full penalty regardless of proximity: the Gaussian grading is gone while the
    imbalance fix stays. THAT is the clean one-key isolation of the claim, and
    unlike A-CE it still trains, so the learning curves are comparable."""
    n = max(float(n_vis_batch), 1.0)
    if loss_form == "ce":
        return (strict_bce(hm_logit, hm_t) + lam * strict_bce(cls_logit, cls_t)) / n
    return (focal(hm_logit, hm_t, beta=beta) + lam * focal(cls_logit, cls_t, beta=beta)) / n


def refiner_loss(logits, targets):
    """Refiner loss: same focal form, normalised by batch size (one
    forced-1.0 positive per crop by construction, so B is the natural N).
    targets (B,64,64) is unsqueezed to logits' (B,1,64,64) before combining --
    without it, elementwise broadcast would pair every logit-crop against
    every target-crop (a (B,B,64,64) cross product) instead of matching them
    one-to-one."""
    b = logits.shape[0]
    return focal(logits, targets.unsqueeze(1)) / max(b, 1)
