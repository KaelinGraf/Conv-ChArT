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
                   beta=4, alpha=2, loss_form_hm=None, loss_form_cls=None):
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
    unlike A-CE it still trains, so the learning curves are comparable.

    alpha is threaded through for the same reason beta is: configs/*.yaml all declare
    top-level `alpha: 2` and `beta: 4`, but until 2026-07-30 NOTHING in training read
    either -- the live lever was the separately-named `focal_beta`, and `alpha`/`beta`
    were read only by tools/loss_explainer_pdf.py, i.e. by the FIGURE. Every config
    carried alpha=2/beta=4, exactly focal()'s defaults, so no run was ever mis-trained;
    but `--set beta=0` would have silently no-opped, and a hand-edit of `beta` would have
    produced a slide describing a loss the model was not trained with. Callers now resolve
    focal_beta -> beta -> 4 so the config keys are live and the figure cannot drift.

    PER-HEAD FORMS (2026-07-31). loss_form_hm / loss_form_cls override loss_form for one head
    each, because the campaign measured the two heads to be driven by DIFFERENT mechanisms:
    beta=0 (Gaussian grading removed, focal's alpha kept) captured 77% of A-CE's localisation
    gain but only 8.5% of its identity gain. So the penalty discount governs the HEATMAP and the
    focal->BCE switch -- which also drops alpha's easy-negative modulation on the CLASS head --
    governs IDENTITY. Switching both heads together, as loss_form alone does, cannot separate
    them or take the better of each. Default None keeps the old single-lever behaviour exactly."""
    n = max(float(n_vis_batch), 1.0)
    hm_form = loss_form_hm or loss_form
    cls_form = loss_form_cls or loss_form
    head = lambda logit, target, form: (strict_bce(logit, target) if form == "ce"
                                        else focal(logit, target, alpha=alpha, beta=beta))
    return (head(hm_logit, hm_t, hm_form) + lam * head(cls_logit, cls_t, cls_form)) / n


def loss_kwargs(cfg):
    """Resolve detector_loss's form/alpha/beta from a config, in ONE place.

    Three trainer call sites (train_detector's train and val loops, train_pair's loop) all
    need the identical resolution, and the `focal_beta` vs `beta` precedence is the kind of
    detail that drifts when it is written out three times. Precedence is
    focal_beta -> beta -> default, so an arm cut with either key name is live."""
    return {"loss_form": cfg.get("loss_form", "focal"),
            "loss_form_hm": cfg.get("loss_form_hm"),
            "loss_form_cls": cfg.get("loss_form_cls"),
            "beta": cfg.get("focal_beta", cfg.get("beta", 4)),
            "alpha": cfg.get("alpha", 2)}


def refiner_loss(logits, targets):
    """Refiner loss: same focal form, normalised by batch size (one
    forced-1.0 positive per crop by construction, so B is the natural N).
    targets (B,64,64) is unsqueezed to logits' (B,1,64,64) before combining --
    without it, elementwise broadcast would pair every logit-crop against
    every target-crop (a (B,B,64,64) cross product) instead of matching them
    one-to-one."""
    b = logits.shape[0]
    return focal(logits, targets.unsqueeze(1)) / max(b, 1)
