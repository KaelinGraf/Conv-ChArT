"""tools/finetune_cost.py -- isolated GPU cost of each fine-tuning arm: peak training VRAM and ms/step.

The training runs measure wall time on a 24-core machine whose synthetic-data generator is the
bottleneck, so their samples/s says little about what the GPU needs. This measures the GPU side alone,
per arm and per batch size: build the model, apply the arm's freezing/lr_mult exactly as the trainer
does (dcc.trainutil.param_groups), then time forward + loss + backward + clip + AdamW step on random
tensors of the real shapes (640x480 input, bf16 autocast, channels_last -- the trainer's settings).
Random inputs cost exactly what real ones do; only the numbers differ.

Per arm x batch: `peak_mem_mb` (torch.cuda.max_memory_allocated: weights + grads + optimizer state +
saved activations), `reserved_mb` (allocator high-water mark, what nvidia-smi shows), `ms_per_step`
(median of timed iterations, CUDA-synchronised; one micro-batch, no accumulation), `n_trainable`,
`optim_state_mb`. The release recipe is batch 16 x accum 2, so its footprint is the B=16 row.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--batches", nargs="+", type=int, default=[4, 8, 16])
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    import torch
    import torch.nn as nn
    from dcc.board import n_corners
    from dcc.dataset import load_config
    from dcc.losses import detector_loss, loss_kwargs
    from dcc.model import DetectorNet, detector_kwargs
    from dcc.trainutil import param_groups
    dev = torch.device("cuda")
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    # merge into an existing report so arms cut later (ft17, ft19, ...) extend the ladder's file rather than replace it
    rows = json.load(open(a.out))["rows"] if Path(a.out).exists() else {}
    print(f"{'arm':<34} {'B':>3} {'trainable':>10} {'peak MB':>9} {'reserved':>9} {'ms/step':>8}")
    for cfg_path in a.configs:
        cfg = load_config(cfg_path); W, H = cfg["input_size"]; name = Path(cfg_path).stem
        ft = cfg.get("finetune") or {}
        for B in a.batches:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            model = DetectorNet(H, W, n_cls=n_corners(cfg.get("board")), **detector_kwargs(cfg)).to(dev, memory_format=torch.channels_last)
            model.train()
            optim = torch.optim.AdamW(param_groups(model, cfg["train"]["wd"], lr_mult=ft.get("lr_mult")), lr=cfg["train"]["lr"])
            for m in model.modules():
                if isinstance(m, bn_types) and not any(q.requires_grad for q in m.parameters(recurse=False)):
                    m.eval()
            n_tr = sum(q.numel() for q in model.parameters() if q.requires_grad)
            g = torch.Generator(device=dev).manual_seed(0)
            x = torch.rand(B, 1, H, W, device=dev, generator=g).contiguous(memory_format=torch.channels_last)
            n_cls = n_corners(cfg.get("board"))
            hm = torch.zeros(B, 1, H, W, device=dev); cls = torch.zeros(B, n_cls, H // 4, W // 4, device=dev)
            for b in range(B):                      # one forced-1.0 positive per corner, like a fully visible board
                ys, xs = torch.randint(0, H, (n_cls,), generator=g, device=dev), torch.randint(0, W, (n_cls,), generator=g, device=dev)
                hm[b, 0, ys, xs] = 1.0; cls[b, torch.arange(n_cls, device=dev), ys // 4, xs // 4] = 1.0
            nvis = torch.tensor(float(n_cls * B), device=dev)
            times = []
            for it in range(5 + a.iters):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    h, c = model(x)
                    loss = detector_loss(h, c, hm, cls, nvis, cfg["lambda_cls"], **loss_kwargs(cfg))
                loss.backward()
                nn.utils.clip_grad_norm_((q for q in model.parameters() if q.requires_grad), cfg["train"]["clip_norm"])
                optim.step(); optim.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                if it >= 5:
                    times.append((time.perf_counter() - t0) * 1e3)
            opt_mb = sum(v.numel() * v.element_size() for s in optim.state.values() for v in s.values() if torch.is_tensor(v)) / 2**20
            row = {"n_params": sum(q.numel() for q in model.parameters()), "n_trainable": n_tr, "batch": B,
                   "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20, "reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
                   "ms_per_step": statistics.median(times), "ms_per_step_p90": sorted(times)[int(0.9 * (len(times) - 1))],
                   "optim_state_mb": opt_mb, "lr_mult": ft.get("lr_mult")}
            rows[f"{name}@B{B}"] = row
            print(f"{name:<34} {B:>3} {n_tr:>10,} {row['peak_mem_mb']:>9.0f} {row['reserved_mb']:>9.0f} {row['ms_per_step']:>8.1f}", flush=True)
            del model, optim, x, hm, cls, h, c, loss
        # written after EVERY arm, so a crash on arm k keeps arms < k (one did: a bias-only LayerNorm)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "input_size": [W, H],
                   "note": "one micro-batch per step, no grad accumulation; bf16 autocast, channels_last; random tensors of the real shapes",
                   "rows": rows}, open(a.out, "w"), indent=1)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
