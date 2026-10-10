"""Training utilities: EMA, cosine_lr, JsonlLogger, save_ckpt / load_ckpt / load_retarget_ckpt,
param_groups(model, wd, lr_mult=None), warn_cfg_drift and generator_fingerprint.
"""
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import torch
import torch.nn as nn

_NORM_TYPES = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)


class EMA:
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
    if warmup > 0 and step < warmup:
        return peak * step / warmup
    if step >= total:
        return floor
    prog = (step - warmup) / max(total - warmup, 1)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * prog))


class JsonlLogger:
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
    tmp = path.with_name(path.name + ".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)


def load_ckpt(path, model, ema, optim, map_location=None, restore_optim=True):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    ema.load_state_dict(ckpt["ema"])
    if restore_optim:
        optim.load_state_dict(ckpt["optim"])
    torch.set_rng_state(ckpt["torch_rng"].cpu())
    return ckpt


def flat_cfg(d, pre=""):
    out = {}
    for k, v in d.items():
        kk = f"{pre}{k}"
        if isinstance(v, dict):
            out.update(flat_cfg(v, kk + "."))
        else:
            out[kk] = v
    return out


def warn_cfg_drift(ckpt, live_cfg, steps=None, budget_key="train.steps"):
    old = ckpt.get("cfg")
    if not old:
        return {}
    fo, fn = flat_cfg(old), flat_cfg(live_cfg)
    diff = {k: (fo.get(k, "<absent>"), fn.get(k, "<absent>")) for k in set(fo) | set(fn)
            if fo.get(k) != fn.get(k)}
    if diff:
        print(f"[resume] WARNING: {len(diff)} config key(s) differ from the checkpoint's own cfg:")
        for k, (was, now) in sorted(diff.items()):
            print(f"[resume]   {k}: {was!r} -> {now!r}")
    was = fo.get(budget_key)
    if steps is not None and was not in (None, steps):
        print(f"[resume] WARNING: LR SCHEDULE RE-ANCHORED -- checkpoint {budget_key}={was} vs "
              f"this run's {steps}. cosine_lr anneals to the budget, so resuming under a "
              f"different one puts the LR somewhere the original schedule never visited.")
    return diff


def load_retarget_ckpt(path, model, map_location=None):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    base_sd = {k: v for k, v in ckpt["model"].items() if not k.startswith("cls.")}
    result = model.load_state_dict(base_sd, strict=False)
    expected_missing = {k for k in model.state_dict() if k.startswith("cls.")}
    assert set(result.missing_keys) == expected_missing, \
        f"retarget load: missing keys {set(result.missing_keys) ^ expected_missing} outside/inside cls.*"
    assert not result.unexpected_keys, f"retarget load: unexpected keys in base checkpoint {result.unexpected_keys}"
    return ckpt


def param_groups(model, wd, lr_mult=None):
    import fnmatch
    no_decay_ids = {id(p) for m in model.modules() if isinstance(m, _NORM_TYPES)
                     for p in m.parameters(recurse=False)}
    groups, matched = {}, set()
    for name, p in model.named_parameters():
        mult = 1.0
        if lr_mult is not None:
            hit = next((pat for pat in lr_mult if fnmatch.fnmatchcase(name, pat)), None)
            mult = float(lr_mult[hit]) if hit is not None else 0.0
            p.requires_grad_(mult > 0)
            if hit is not None:
                matched.add(hit)
        if not p.requires_grad:
            continue
        no_decay = name.endswith(".bias") or id(p) in no_decay_ids
        groups.setdefault((mult, no_decay), []).append(p)
    if lr_mult is None:
        return [{"params": groups.get((1.0, False), []), "weight_decay": wd},
                {"params": groups.get((1.0, True), []), "weight_decay": 0.0}]
    unmatched = set(lr_mult) - matched
    assert not unmatched, f"lr_mult patterns matched no parameter: {sorted(unmatched)}"
    return [{"params": ps, "weight_decay": 0.0 if nd else wd, "lr_mult": mult}
            for (mult, nd), ps in sorted(groups.items(), key=lambda kv: (-kv[0][0], kv[0][1]))]


def generator_fingerprint(cfg, root):
    rels = ("dcc/board.py", "dcc/synth.py", "dcc/targets.py", "dcc/dataset.py",
            "dcc/refiner_data.py")
    files = {rel: hashlib.sha1((root / rel).read_bytes()).hexdigest() for rel in rels}
    keys = ("board", "synth", "input_size", "scale_range_px", "negative_p", "sigma_hm", "sigma_cls",
            "refiner_jitter_px")
    subset = {k: cfg[k] for k in keys}
    subset["sigma_ref"] = cfg.get("sigma_ref", 1.5)
    config_sha1 = hashlib.sha1(json.dumps(subset, sort_keys=True).encode()).hexdigest()
    return {"files": files, "config_sha1": config_sha1}
