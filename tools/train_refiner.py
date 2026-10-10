"""Stage-2 refiner training.
    PYTHONPATH= python tools/train_refiner.py --config C --name N [--steps S] [--resume CKPT] [--workers W]
"""
import argparse
from functools import partial
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BIAS_EDGES = (-4.0, -2.0, 0.0, 2.0, 4.0)
_WARNED = set()


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--name", default="run")
    p.add_argument("--resume", default=None, help="checkpoint .pt to resume from")
    p.add_argument("--steps", type=int, default=None, help="override cfg refiner_train.steps")
    p.add_argument("--workers", type=int, default=None, help="override cfg refiner_train.workers -- cap this "
                   "when another job is running: loader workers live in /dev/shm and the machine has hard-locked "
                   "twice on kswapd shmem_writepage reclaim stalls at high worker counts")
    return p


def _warn_once(key, msg):
    if key not in _WARNED:
        print(msg)
        _WARNED.add(key)


def _worker_init(_):
    import cv2
    cv2.setNumThreads(1)


def _soft_argmax_fallback(ref_sigmoid):
    import torch
    t = ref_sigmoid.reshape(-1, 64, 64)
    B = t.shape[0]
    ys, xs = torch.meshgrid(torch.arange(64.0, device=t.device), torch.arange(64.0, device=t.device),
                             indexing="ij")
    out = torch.zeros(B, 2, device=t.device)
    for i in range(B):
        ay, ax = divmod(int(torch.argmax(t[i])), 64)
        y0, x0 = min(max(ay - 2, 0), 59), min(max(ax - 2, 0), 59)
        w = t[i, y0:y0 + 5, x0:x0 + 5]
        wsum = w.sum().clamp_min(1e-12)
        out[i, 0] = (w * xs[y0:y0 + 5, x0:x0 + 5]).sum() / wsum
        out[i, 1] = (w * ys[y0:y0 + 5, x0:x0 + 5]).sum() / wsum
    return out.detach().cpu().numpy()


def _refiner_val_collate(batch, sigma=1.5):
    import numpy as np
    import torch
    from dcc.targets import render_refiner_target

    crops, targets, ds = [], [], []
    for crop_list in batch:
        for c in crop_list:
            crops.append(torch.from_numpy(c["crop"]).float().unsqueeze(0) / 255.0)
            targets.append(torch.from_numpy(render_refiner_target(c["d"], sigma=sigma)))
            ds.append(c["d"])
    if not crops:
        return None, None, None
    return torch.stack(crops), torch.stack(targets), np.stack(ds)


def _bias_bins(values, errors):
    import numpy as np
    out = {}
    for lo, hi in zip(BIAS_EDGES[:-1], BIAS_EDGES[1:]):
        m = (values >= lo) & (values <= hi if hi == BIAS_EDGES[-1] else values < hi)
        out[f"[{lo:g},{hi:g})"] = float(errors[m].mean()) if m.any() else None
    return out


def run_refiner_validation(model, loader, device, rl_kw=None):
    import numpy as np
    import torch
    from dcc.losses import refiner_loss
    rl_kw = rl_kw or {}

    try:
        from dcc.pipeline import soft_argmax
    except ImportError as e:
        soft_argmax = _soft_argmax_fallback
        _warn_once("pipeline_softargmax", f"[train_refiner] dcc.pipeline.soft_argmax unavailable ({e}); "
                   "using a local fallback (same algorithm) for the refined-error readout.")

    model.eval()
    loss_sum, loss_n = 0.0, 0
    errs, all_dx, all_dy, err_x, err_y = [], [], [], [], []

    with torch.no_grad():
        for crops, targets, d in loader:
            if crops is None:
                continue
            crops = crops.to(device, non_blocking=True, memory_format=torch.channels_last)
            targets_d = targets.to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(crops)
                loss = refiner_loss(logits, targets_d, **rl_kw)
            loss_sum += float(loss) * crops.shape[0]
            loss_n += crops.shape[0]

            prob = torch.sigmoid(logits.float())[:, 0]
            uv = np.asarray(soft_argmax(prob))
            u_gt, v_gt = 31.5 + 8 * d[:, 0], 31.5 + 8 * d[:, 1]
            errs.extend(np.linalg.norm(uv - np.stack([u_gt, v_gt], axis=1), axis=1).tolist())
            pred_dx, pred_dy = (uv[:, 0] - 31.5) / 8, (uv[:, 1] - 31.5) / 8
            all_dx.extend(d[:, 0].tolist())
            all_dy.extend(d[:, 1].tolist())
            err_x.extend((pred_dx - d[:, 0]).tolist())
            err_y.extend((pred_dy - d[:, 1]).tolist())

    errs = np.array(errs)
    m03 = {"mean": float(errs.mean()) if errs.size else None,
           "median": float(np.median(errs)) if errs.size else None,
           "p95": float(np.percentile(errs, 95)) if errs.size else None,
           "mean_px": float(errs.mean() / 8) if errs.size else None,
           "median_px": float(np.median(errs) / 8) if errs.size else None,
           "p95_px": float(np.percentile(errs, 95) / 8) if errs.size else None,
           "frac_lt_0p25px": float((errs < 2.0).mean()) if errs.size else None,
           "frac_lt_0p5px": float((errs < 4.0).mean()) if errs.size else None,
           "frac_lt_1px": float((errs < 8.0).mean()) if errs.size else None,
           "n": int(errs.size)}
    bias = {"x": _bias_bins(np.array(all_dx), np.array(err_x)),
            "y": _bias_bins(np.array(all_dy), np.array(err_y))}
    return {"val_loss": loss_sum / loss_n if loss_n else None, "m03": m03, "bias_vs_jitter": bias}


def main():
    args = build_parser().parse_args()

    import time

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from dcc.dataset import RefinerVal, SynthStream, load_config
    from dcc.losses import refiner_loss, refiner_loss_kwargs
    from dcc.model import Refiner
    from dcc.trainutil import EMA, JsonlLogger, cosine_lr, load_ckpt, param_groups, save_ckpt, warn_cfg_drift

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")

    cfg = load_config(args.config)
    if args.steps is not None:
        cfg["refiner_train"]["steps"] = args.steps
    if args.workers is not None:
        cfg["refiner_train"]["workers"] = args.workers
    rcfg = cfg["refiner_train"]

    rl_kw = refiner_loss_kwargs(cfg)
    rw = cfg.get("refiner_width", 1.0)
    model = Refiner(width_mult=rw).to(device, memory_format=torch.channels_last)
    eval_model = Refiner(width_mult=rw).to(device, memory_format=torch.channels_last)
    print(f"[train_refiner] refiner_width={rw} params={sum(p.numel() for p in model.parameters()):,} "
          f"loss_form={rl_kw['loss_form']} sigma_ref={cfg.get('sigma_ref', 1.5)}")
    model.train()

    ema = EMA(model, decay=rcfg["ema_decay"])
    optim = torch.optim.AdamW(param_groups(model, rcfg["wd"]), lr=rcfg["lr"], betas=(0.9, 0.999))

    run_dir = Path("runs") / args.name
    logger = JsonlLogger(run_dir / "metrics.jsonl")

    step, resume_count, last_val = 0, 0, None
    if args.resume:
        ckpt = load_ckpt(args.resume, model, ema, optim, map_location=device)
        warn_cfg_drift(ckpt, cfg, steps=rcfg["steps"], budget_key="refiner_train.steps")
        step, resume_count, last_val = ckpt["step"], ckpt["resume_count"] + 1, ckpt["last_val"]

    stream_seed = cfg["synth"]["train_seed"] * 1000 + resume_count
    print(f"[train_refiner] stream_seed={stream_seed} resume_count={resume_count} start_step={step}")

    train_ds = SynthStream(cfg, stream="refiner", seed=stream_seed, render_targets=True)
    train_loader = DataLoader(train_ds, batch_size=rcfg["batch"], num_workers=rcfg["workers"],
                               multiprocessing_context="spawn", persistent_workers=True, pin_memory=True,
                               worker_init_fn=_worker_init)

    val_ds = RefinerVal(cfg, n=cfg["synth"]["refiner_val_composites"])
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=8, multiprocessing_context="spawn",
                             persistent_workers=True,
                             collate_fn=partial(_refiner_val_collate,
                                                sigma=cfg.get("sigma_ref", 1.5)))

    total_steps = rcfg["steps"]
    train_iter = iter(train_loader)
    t0, val_time, n_samples = time.time(), 0.0, 0

    while step < total_steps:
        batch = next(train_iter)
        crops = batch["crop"].to(device, non_blocking=True, memory_format=torch.channels_last)
        targets = batch["target"].to(device, non_blocking=True)
        n_samples += crops.shape[0]

        optim.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(crops)
            loss = refiner_loss(logits, targets, **rl_kw)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), rcfg["clip_norm"])

        lr = cosine_lr(step, total_steps, rcfg["lr"], rcfg["lr_floor"], rcfg["warmup_steps"])
        for g in optim.param_groups:
            g["lr"] = lr
        optim.step()
        ema.update(model)

        step += 1
        loss_val = float(loss.detach())
        elapsed = time.time() - t0 - val_time
        logger.log(step=step, loss=loss_val, lr=lr, grad_norm=float(grad_norm),
                   samples_per_s=n_samples / elapsed if elapsed > 0 else 0.0)
        if step % 10 == 0 or step <= 10:
            print(f"[step {step}/{total_steps}] loss={loss_val:.4f} lr={lr:.3e} "
                  f"grad_norm={float(grad_norm):.3f} samples/s={n_samples / elapsed:.2f}")

        if step % rcfg["val_every"] == 0 or step == total_steps:
            v0 = time.time()
            ema.copy_to(eval_model)
            result = run_refiner_validation(eval_model, val_loader, device, rl_kw)
            print(f"[val step {step}] loss={result['val_loss']} m03={result['m03']} "
                  f"bias_vs_jitter={result['bias_vs_jitter']}")
            logger.log(step=step, val=result)
            last_val = result
            ckpt_path = run_dir / f"ckpt_{step:07d}.pt"
            save_ckpt(ckpt_path, step, resume_count, model, ema, optim, cfg, last_val)
            print(f"[train_refiner] wrote checkpoint {ckpt_path}")
            val_time += time.time() - v0

    elapsed = time.time() - t0 - val_time
    print(f"[train_refiner] done: step={step} train_elapsed={elapsed:.1f}s val_elapsed={val_time:.1f}s "
          f"samples/s={n_samples / elapsed if elapsed > 0 else 0.0:.2f}")


if __name__ == "__main__":
    main()
