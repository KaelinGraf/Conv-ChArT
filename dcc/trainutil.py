"""Training-loop primitives shared by tools/train_detector.py and
tools/train_refiner.py: EMA shadow weights, the cosine LR schedule, a
JSON-lines metrics logger, checkpoint save/load, and no-weight-decay param
groups. Model-independent -- no dcc.model/dcc.losses import here.
"""
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_NORM_TYPES = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)


class EMA:
    """Exponential moving average of a model's full state_dict (params +
    buffers), decayed toward the live weights on every update(). Non-floating
    buffers (e.g. BatchNorm's num_batches_tracked) pass through undecayed --
    copied, not averaged. copy_to loads the shadow into a (same-architecture)
    model, e.g. to run validation forward passes on EMA weights."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach(), alpha=1 - d)
            else:
                s.copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}


def cosine_lr(step, total, peak, floor, warmup):
    """LR at optimizer `step`: linear 0->peak over [0, warmup), cosine
    peak->floor over [warmup, total], floor beyond total."""
    if warmup > 0 and step < warmup:
        return peak * step / warmup
    if step >= total:
        return floor
    prog = (step - warmup) / max(total - warmup, 1)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * prog))


class JsonlLogger:
    """Appends dicts as JSON lines to `path` (parent dirs created), each
    stamped with step + wall-clock time. Opens/closes the file per call so a
    crash never loses a buffered line, and a resumed run's append just
    extends the existing history."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, step, **fields):
        rec = {"step": step, "wall": time.time(), **fields}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")


def _git_hash():
    try:
        root = Path(__file__).resolve().parents[1]
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def save_ckpt(path, step, resume_count, model, ema, optim, cfg, last_val):
    """One .pt with everything needed to resume bit-for-bit: step counters,
    model/EMA/optimizer state, the config used, provenance (git_hash), and
    the global torch RNG. The numpy GLOBAL RNG is deliberately NOT captured:
    every numpy draw in this project flows through explicit Generators seeded
    from config (train_seed*1000+resume_count derives the stream on resume),
    so global numpy state is dead weight -- and dcc/ has a zero-global-RNG
    purity rule enforced by test_generator.py's static scan."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "resume_count": resume_count,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optim": optim.state_dict(),
        "cfg": cfg,
        "git_hash": _git_hash(),
        "torch_rng": torch.get_rng_state(),
        "last_val": last_val,
    }, path)


def load_ckpt(path, model, ema, optim, map_location=None, restore_optim=True):
    """Restores model/ema in place, plus optim and both global RNGs; returns
    the full ckpt dict (step, resume_count, cfg, git_hash, last_val, ...).
    restore_optim=False skips optim.load_state_dict -- for the freeze_trunk
    retarget path, where the caller builds a fresh optimizer over cls params
    only, and the checkpoint's optim state (shaped for whichever params were
    trainable in the run that saved it) would otherwise mismatch that fresh
    optimizer's param-group structure."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    ema.load_state_dict(ckpt["ema"])
    if restore_optim:
        optim.load_state_dict(ckpt["optim"])
    torch.set_rng_state(ckpt["torch_rng"].cpu())
    return ckpt


def param_groups(model, wd):
    """Two AdamW param groups: zero weight decay on any bias parameter or
    any parameter owned by a norm module (LayerNorm/BatchNorm*/GroupNorm),
    full `wd` on everything else. Parameters with requires_grad=False (e.g.
    a frozen trunk) are excluded entirely, so this also doubles as "build an
    optimizer over the currently-trainable params only"."""
    no_decay_ids = {id(p) for m in model.modules() if isinstance(m, _NORM_TYPES)
                     for p in m.parameters(recurse=False)}
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if name.endswith(".bias") or id(p) in no_decay_ids else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]
