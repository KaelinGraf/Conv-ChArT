import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from dcc.trainutil import EMA, JsonlLogger, cosine_lr, load_ckpt, param_groups, save_ckpt


class _Toy(nn.Module):
    """Linear + BatchNorm1d -- one bias-bearing layer, one norm layer, so
    param_groups has both exclusion rules to exercise, and EMA/ckpt tests
    have both a floating buffer (running_mean/var) and a non-floating one
    (num_batches_tracked) to round-trip."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x):
        return self.bn(self.lin(x))


def test_cosine_lr_shape():
    total, warmup, peak, floor = 1000, 100, 3e-4, 3e-6
    assert cosine_lr(0, total, peak, floor, warmup) == 0.0
    assert cosine_lr(warmup, total, peak, floor, warmup) == pytest.approx(peak)
    assert cosine_lr(warmup // 2, total, peak, floor, warmup) == pytest.approx(peak / 2)
    assert cosine_lr(total, total, peak, floor, warmup) == pytest.approx(floor)
    assert cosine_lr(total + 500, total, peak, floor, warmup) == pytest.approx(floor)

    warm_vals = [cosine_lr(s, total, peak, floor, warmup) for s in range(0, warmup + 1)]
    assert all(b > a for a, b in zip(warm_vals, warm_vals[1:]))

    decay_vals = [cosine_lr(s, total, peak, floor, warmup) for s in range(warmup, total + 1, 10)]
    assert all(b < a for a, b in zip(decay_vals, decay_vals[1:]))


def test_ema_convergence():
    torch.manual_seed(0)
    model = _Toy()
    ema = EMA(model, decay=0.9)
    with torch.no_grad():
        model.lin.weight.fill_(0.37)
        model.lin.bias.fill_(-0.21)
        model.bn.weight.fill_(1.7)
        model.bn.bias.fill_(0.05)
        model.bn.running_mean.fill_(0.3)
        model.bn.running_var.fill_(2.0)
    target = {k: v.clone() for k, v in model.state_dict().items()}

    for _ in range(100):
        ema.update(model)

    for k, v in ema.state_dict().items():
        assert torch.allclose(v, target[k], atol=1e-3), k


def test_ckpt_roundtrip(tmp_path):
    torch.manual_seed(1)
    model = _Toy()
    optim = torch.optim.AdamW(param_groups(model, wd=0.01), lr=1e-3)
    ema = EMA(model, decay=0.99)

    x = torch.randn(8, 4)
    for _ in range(2):
        optim.zero_grad(set_to_none=True)
        model(x).pow(2).mean().backward()
        optim.step()
        ema.update(model)

    torch.manual_seed(999)
    torch.rand(7)  # move rng state off the trivial post-seed point

    ckpt_path = tmp_path / "ckpt.pt"
    save_ckpt(ckpt_path, step=42, resume_count=3, model=model, ema=ema, optim=optim,
              cfg={"foo": 1, "bar": [1, 2, 3]}, last_val={"m01_mean": 0.7})

    expected_torch = torch.rand(5)

    torch.rand(50)  # perturb further -- load_ckpt below must undo this too
    # (numpy global RNG is deliberately NOT in the ckpt: all numpy draws are
    # Generator-explicit project-wide; see save_ckpt's docstring.)
    model2 = _Toy()
    optim2 = torch.optim.AdamW(param_groups(model2, wd=0.5), lr=5e-2)
    ema2 = EMA(model2, decay=0.5)

    ckpt = load_ckpt(ckpt_path, model2, ema2, optim2, map_location="cpu")

    assert ckpt["step"] == 42 and ckpt["resume_count"] == 3
    assert ckpt["cfg"] == {"foo": 1, "bar": [1, 2, 3]}
    assert ckpt["last_val"] == {"m01_mean": 0.7}
    assert isinstance(ckpt["git_hash"], str) and ckpt["git_hash"]

    for k, v in model.state_dict().items():
        assert torch.equal(v, model2.state_dict()[k]), k
    for k, v in ema.state_dict().items():
        assert torch.equal(v, ema2.state_dict()[k]), k

    assert optim2.param_groups[0]["lr"] == optim.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert optim2.param_groups[0]["weight_decay"] == optim.param_groups[0]["weight_decay"]
    for p_old, p_new in zip(model.parameters(), model2.parameters()):
        st_old, st_new = optim.state[p_old], optim2.state[p_new]
        assert torch.equal(st_old["exp_avg"], st_new["exp_avg"])
        assert torch.equal(st_old["exp_avg_sq"], st_new["exp_avg_sq"])

    # bitwise RNG restore: the very next draw must match what would have
    # come right after save_ckpt, despite the perturbation in between.
    assert torch.equal(torch.rand(5), expected_torch)


def test_ckpt_restore_optim_false(tmp_path):
    """freeze_trunk retarget path: restore_optim=False must still restore
    model/ema and leave the caller's optimizer (built fresh, e.g. over a
    different param subset) completely untouched."""
    torch.manual_seed(2)
    model = _Toy()
    optim = torch.optim.AdamW(param_groups(model, wd=0.01), lr=1e-3)
    ema = EMA(model, decay=0.99)
    ckpt_path = tmp_path / "ckpt.pt"
    save_ckpt(ckpt_path, step=7, resume_count=0, model=model, ema=ema, optim=optim, cfg={}, last_val=None)

    model2 = _Toy()
    optim2 = torch.optim.AdamW(param_groups(model2, wd=0.5), lr=9e-2)
    assert optim2.state_dict()["state"] == {}  # no .step() taken yet -- the untouched baseline
    ckpt = load_ckpt(ckpt_path, model2, EMA(model2), optim2, map_location="cpu", restore_optim=False)

    assert ckpt["step"] == 7
    for k, v in model.state_dict().items():
        assert torch.equal(v, model2.state_dict()[k]), k
    assert optim2.param_groups[0]["lr"] == pytest.approx(9e-2)  # untouched, not optim's 1e-3
    assert optim2.state_dict()["state"] == {}  # still untouched -- restore_optim=False skipped it


def test_param_groups_excludes_bias_and_norm():
    model = _Toy()
    decay, no_decay = param_groups(model, wd=0.05)
    assert decay["weight_decay"] == 0.05
    assert no_decay["weight_decay"] == 0.0

    def _has(lst, p):
        return any(q is p for q in lst)

    assert _has(decay["params"], model.lin.weight)
    assert not _has(decay["params"], model.lin.bias)
    assert not _has(decay["params"], model.bn.weight)
    assert not _has(decay["params"], model.bn.bias)
    assert _has(no_decay["params"], model.lin.bias)
    assert _has(no_decay["params"], model.bn.weight)
    assert _has(no_decay["params"], model.bn.bias)

    total = len(decay["params"]) + len(no_decay["params"])
    assert total == sum(1 for p in model.parameters() if p.requires_grad)


def test_param_groups_excludes_frozen():
    model = _Toy()
    model.bn.requires_grad_(False)
    decay, no_decay = param_groups(model, wd=0.1)
    all_params = decay["params"] + no_decay["params"]
    assert not any(p is model.bn.weight or p is model.bn.bias for p in all_params)
    assert any(p is model.lin.weight for p in all_params)


def test_jsonl_logger(tmp_path):
    path = tmp_path / "run" / "metrics.jsonl"
    logger = JsonlLogger(path)
    logger.log(step=1, loss=0.5)
    logger.log(step=2, loss=0.4, extra="x")

    assert path.exists()
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    rec0, rec1 = json.loads(lines[0]), json.loads(lines[1])
    assert rec0["step"] == 1 and rec0["loss"] == 0.5 and "wall" in rec0
    assert rec1["step"] == 2 and rec1["loss"] == 0.4 and rec1["extra"] == "x"
    assert rec1["wall"] >= rec0["wall"]
