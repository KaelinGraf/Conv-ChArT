"""Training-loop primitives shared by tools/train_detector.py and
tools/train_refiner.py: EMA shadow weights, the cosine LR schedule, a
JSON-lines metrics logger, checkpoint save/load, no-weight-decay param
groups, and a generator+config fingerprint for the training-guard lock
(tools/preflight.py's generator_lock check). Model-independent -- no
dcc.model/dcc.losses import here.
"""
import hashlib
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


def save_ckpt(path, step, resume_count, model, ema, optim, cfg, last_val, retargeted_from=None):
    """One .pt with everything needed to resume bit-for-bit: step counters,
    model/EMA/optimizer state, the config used, provenance (git_hash), and
    the global torch RNG. The numpy GLOBAL RNG is deliberately NOT captured:
    every numpy draw in this project flows through explicit Generators seeded
    from config (train_seed*1000+resume_count derives the stream on resume),
    so global numpy state is dead weight -- and dcc/ has a zero-global-RNG
    purity rule enforced by test_generator.py's static scan. retargeted_from
    (optional): {"path": str, "board": <base checkpoint's own cfg["board"],
    if present>} -- set only on a --retarget-from run, so a retargeted
    checkpoint is self-describing about the base trunk it came from, on top
    of the board it was retargeted onto (cfg["board"], always present)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": step,
        "resume_count": resume_count,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optim": optim.state_dict(),
        "cfg": cfg,
        "git_hash": _git_hash(),
        "torch_rng": torch.get_rng_state(),
        "last_val": last_val,
    }
    if retargeted_from is not None:
        ckpt["retargeted_from"] = retargeted_from
    torch.save(ckpt, path)


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


def load_retarget_ckpt(path, model, map_location=None):
    """New-board retarget path: loads every non-cls.* tensor from a base
    checkpoint (trained on ANY board -- its own n_cls, hence its cls.*
    shape, may differ from `model`'s) into `model` in place; cls.* -- the
    sole board-specific part, see dcc/model.py's header -- is excluded from
    the load entirely and keeps model's own fresh init. strict=False's
    missing/unexpected keys are asserted to be EXACTLY model's own cls.*
    keys and empty respectively: a base checkpoint whose non-cls.*
    architecture doesn't match `model` (wrong file, incompatible trunk) is a
    misuse to fail loudly on, not a silent partial load. Returns the full
    base checkpoint dict (cfg, git_hash, ... intact) so the caller can read
    its board block for retarget provenance."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    base_sd = {k: v for k, v in ckpt["model"].items() if not k.startswith("cls.")}
    result = model.load_state_dict(base_sd, strict=False)
    expected_missing = {k for k in model.state_dict() if k.startswith("cls.")}
    assert set(result.missing_keys) == expected_missing, \
        f"retarget load: missing keys {set(result.missing_keys) ^ expected_missing} outside/inside cls.*"
    assert not result.unexpected_keys, f"retarget load: unexpected keys in base checkpoint {result.unexpected_keys}"
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


def generator_fingerprint(cfg, root):
    """Content hash of the generator (the four dcc/ files that decide what a
    sample looks like) plus the config knobs that shape their output:
    {"files": {relpath: sha1-hex of its bytes}, "config_sha1": sha1 of the
    relevant cfg subset as canonical JSON}. `root` is the project root
    (Path(__file__).resolve().parents[1] from the caller). Compared
    bit-for-bit by tools/preflight.py's generator_lock check against the
    fingerprint an audit run stamped into report.json -- a training run
    refuses to start on a generator or config that has drifted since the
    last green audit."""
    rels = ("dcc/board.py", "dcc/synth.py", "dcc/targets.py", "dcc/dataset.py")
    files = {rel: hashlib.sha1((root / rel).read_bytes()).hexdigest() for rel in rels}
    keys = ("board", "synth", "input_size", "scale_range_px", "negative_p", "sigma_hm", "sigma_cls",
            "refiner_jitter_px")
    subset = {k: cfg[k] for k in keys}
    config_sha1 = hashlib.sha1(json.dumps(subset, sort_keys=True).encode()).hexdigest()
    return {"files": files, "config_sha1": config_sha1}
