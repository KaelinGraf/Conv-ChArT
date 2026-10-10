"""Losses. detector_loss(hm_logits, cls_logits, hm_t, cls_t, n_vis, lam, **loss_kwargs(cfg)) and
refiner_loss(logits, targets, **refiner_loss_kwargs(cfg)). Optional masks (0 = no gradient) carry
real-frame pseudo-labels whose visibility or exact pixel is unknown.
"""
import torch
import torch.nn.functional as F


def focal(logits, y, alpha=2, beta=4, mask=None):
    z = logits.float()
    pos = y == 1.0
    l_pos = (1 - torch.sigmoid(z)) ** alpha * F.logsigmoid(z) * pos
    l_neg = (1 - y) ** beta * torch.sigmoid(z) ** alpha * F.logsigmoid(-z) * ~pos
    if mask is not None:
        l_pos, l_neg = l_pos * mask, l_neg * mask
    return -(l_pos.sum() + l_neg.sum())


def strict_bce(logits, y, mask=None):
    z = logits.float()
    pos = (y == 1.0).float()
    weight = None if mask is None else torch.broadcast_to(mask.float(), z.shape)
    return F.binary_cross_entropy_with_logits(z, pos, weight=weight, reduction="sum")


def detector_loss(hm_logit, cls_logit, hm_t, cls_t, n_vis_batch, lam=1.0, loss_form="focal",
                   beta=4, alpha=2, loss_form_hm=None, loss_form_cls=None, hm_mask=None, cls_mask=None):
    n = max(float(n_vis_batch), 1.0)
    hm_form = loss_form_hm or loss_form
    cls_form = loss_form_cls or loss_form
    assert {hm_form, cls_form} <= {"focal", "ce"}, \
        f"loss_form must be 'focal' or 'ce', got hm={hm_form!r} cls={cls_form!r}"
    head = lambda logit, target, form, mask: (strict_bce(logit, target, mask=mask) if form == "ce"
                                              else focal(logit, target, alpha=alpha, beta=beta, mask=mask))
    return (head(hm_logit, hm_t, hm_form, hm_mask) + lam * head(cls_logit, cls_t, cls_form, cls_mask)) / n


def loss_kwargs(cfg):
    return {"loss_form": cfg.get("loss_form", "focal"),
            "loss_form_hm": cfg.get("loss_form_hm"),
            "loss_form_cls": cfg.get("loss_form_cls"),
            "beta": cfg.get("focal_beta", cfg.get("beta", 4)),
            "alpha": cfg.get("alpha", 2)}


def refiner_loss(logits, targets, loss_form="focal", alpha=2, beta=4):
    b = logits.shape[0]
    assert loss_form in ("focal", "ce"), f"loss_form must be 'focal' or 'ce', got {loss_form!r}"
    t = targets.unsqueeze(1)
    per_batch = strict_bce(logits, t) if loss_form == "ce" else focal(logits, t, alpha=alpha, beta=beta)
    return per_batch / max(b, 1)


def refiner_loss_kwargs(cfg):
    return {"loss_form": cfg.get("refiner_loss_form", "focal"),
            "beta": cfg.get("focal_beta", cfg.get("beta", 4)),
            "alpha": cfg.get("alpha", 2)}
