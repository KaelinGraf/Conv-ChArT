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
    """ABLATION A-CE counterpart to focal(): the STRICT reading of the same targets."""
    z = logits.float()
    pos = (y == 1.0).float()
    return F.binary_cross_entropy_with_logits(z, pos, reduction="sum")


def detector_loss(hm_logit, cls_logit, hm_t, cls_t, n_vis_batch, lam=1.0, loss_form="focal",
                   beta=4, alpha=2, loss_form_hm=None, loss_form_cls=None):
    """Batch-normalised by total visible corners, shared N for both heads; clamped so N=0 batches
    (all-negative, no visible corners) divide by 1 instead of by zero. loss_form="ce" swaps BOTH
    heads to strict_bce (A-CE).
    """
    n = max(float(n_vis_batch), 1.0)
    hm_form = loss_form_hm or loss_form
    cls_form = loss_form_cls or loss_form
    head = lambda logit, target, form: (strict_bce(logit, target) if form == "ce"
                                        else focal(logit, target, alpha=alpha, beta=beta))
    return (head(hm_logit, hm_t, hm_form) + lam * head(cls_logit, cls_t, cls_form)) / n


def loss_kwargs(cfg):
    """Resolve detector_loss's form/alpha/beta from a config, in ONE place."""
    return {"loss_form": cfg.get("loss_form", "focal"),
            "loss_form_hm": cfg.get("loss_form_hm"),
            "loss_form_cls": cfg.get("loss_form_cls"),
            "beta": cfg.get("focal_beta", cfg.get("beta", 4)),
            "alpha": cfg.get("alpha", 2)}


def refiner_loss(logits, targets):
    """Refiner loss: same focal form, normalised by batch size (one forced-1.0 positive per crop by
    construction, so B is the natural N). targets (B,64,64) is unsqueezed to logits' (B,1,64,64)
    before combining -- without it, elementwise broadcast would pair every logit-crop against
    every target-crop (a (B,B,64,64) cross product) instead of matching them one-to-one.
    """
    b = logits.shape[0]
    return focal(logits, targets.unsqueeze(1)) / max(b, 1)
