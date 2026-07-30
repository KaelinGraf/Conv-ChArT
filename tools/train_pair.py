"""tools/train_pair.py -- train TWO (or more) detector arms from ONE data stream.

WHY THIS EXISTS. Measured 2026-07-30: with the small tier models the GPU sits at ~54%
utilisation (sawtooth 8-99%) while the CPU is pinned at load 23/23. Both live arms delivered
73.1 and 73.8 samples/s despite a 2.6x difference in GFLOPs/frame (52.0 vs 19.8) -- identical
throughput across very different compute is the signature of a DATA-bound loop, not a
compute-bound one. Per-sample generation is ~136 ms (147 samples/s over 20 train workers),
and a cProfile attributes ~55% of it to _apply_photometric and its 5.1 GaussianBlur calls.
Sample generation is the entire cost; target rendering is 0.77 ms (0.4%) and JPEG decode
3.4%, so neither is worth moving.

THE FIX, AND WHY IT IS THIS SHAPE. Two trainers generating separate streams do the expensive
work twice. Running both models in ONE process behind ONE DataLoader makes each generated
batch do two model-steps, halving generation cost per model-step. The obvious alternative --
two processes sharing a ring buffer in shared memory -- was rejected deliberately: it needs
refcounted slots and IPC, and its failure mode is SILENT (a refcount bug serves stale or
duplicated samples and shows up as a mysteriously worse model, never as a crash). This file
has no concurrency primitives at all; it is a for-loop over models.

A SECOND, NON-OBVIOUS BENEFIT. Feeding paired arms identical batches is common random
numbers: the data draw cancels between the arms, so the A-vs-B contrast (does parameter-free
XSA earn its keep at this width?) has strictly LOWER variance than two independently-seeded
runs. Sharing is better experimental design here, not merely cheaper.

WHAT IT COSTS. The arms are coupled: one process, shared fate, they start and stop together,
and their GPU work is sequential (affordable only because the GPU is half idle). Arms may
have DIFFERENT step budgets -- an arm that reaches its total stops training while the others
continue, at which point the sharing benefit is gone but correctness is not affected.

Every per-arm mechanism is identical to tools/train_detector.py and is imported from it
rather than reimplemented: run_validation, _val_collate, _worker_init, early stop, EMA,
cosine_lr, save/load_ckpt. If that file's loop changes, this one must be re-checked.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # train_detector is a sibling, not a package


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", action="append", required=True, metavar="NAME:CONFIG:STEPS[:RESUME]",
                   help="repeatable; at least 2. RESUME may be empty for a fresh arm.")
    p.add_argument("--workers", type=int, default=None, help="loader workers for the SHARED stream")
    return p


def parse_arm(spec):
    """NAME:CONFIG:STEPS[:RESUME] -> dict. Split from the left with maxsplit so a resume
    path containing ':' would still survive (paths here never do, but the failure would be
    silent and produce a fresh run instead of a resume, which is exactly the kind of bug
    that costs a night of GPU)."""
    parts = spec.split(":", 3)
    if len(parts) < 3:
        raise SystemExit(f"--arm {spec!r}: need NAME:CONFIG:STEPS[:RESUME]")
    name, config, steps = parts[0], parts[1], parts[2]
    resume = parts[3] if len(parts) > 3 and parts[3] else None
    if not Path(config).is_file():
        raise SystemExit(f"--arm {name}: config {config} not found")
    if resume and not Path(resume).is_file():
        raise SystemExit(f"--arm {name}: resume checkpoint {resume} not found")
    return {"name": name, "config": config, "total": int(steps), "resume": resume}


def shared_signature(cfg):
    """Everything that determines what the SHARED loader emits. Two arms may differ freely in
    model/optimiser keys (width_mult, xsa, e4_dilated, lr...) but must agree here, or one of
    them would silently train on the other's data convention. sigma_hm/sigma_cls are in the
    list because the worker renders TARGETS, not just images -- differing sigma is the one
    plausible mismatch that would still 'work' and quietly poison an arm."""
    t = cfg["train"]
    return {"input_size": cfg["input_size"], "board": cfg.get("board"),
            "sigma_hm": cfg.get("sigma_hm"), "sigma_cls": cfg.get("sigma_cls"),
            "lambda_cls": cfg.get("lambda_cls"),
            "batch": t["batch"], "accum": t["accum"],
            "val_every": t["val_every"], "full_val_every": t["full_val_every"],
            "val_subset": t["val_subset"], "synth": cfg["synth"]}


def main():
    args = build_parser().parse_args()
    specs = [parse_arm(s) for s in args.arm]
    if len(specs) < 2:
        raise SystemExit("train_pair needs at least 2 --arm entries; use train_detector.py for one")

    import json
    import time
    from functools import partial

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    import train_detector as td
    from dcc.board import n_corners
    from dcc.dataset import SynthStream, SynthVal, load_config
    from dcc.losses import detector_loss
    from dcc.model import DetectorNet, detector_kwargs
    from dcc.trainutil import EMA, JsonlLogger, cosine_lr, load_ckpt, param_groups, save_ckpt

    assert torch.cuda.is_available(), "CUDA required"
    assert torch.backends.cuda.flash_sdp_enabled(), "flash SDPA backend is disabled"
    assert torch.backends.cuda.mem_efficient_sdp_enabled(), "mem-efficient SDPA backend is disabled"
    device = torch.device("cuda")

    # ---- load configs and PROVE they may share a stream, before anything expensive ----
    for a in specs:
        a["cfg"] = load_config(a["config"])
        a["cfg"]["train"]["steps"] = a["total"]
        if args.workers is not None:
            a["cfg"]["train"]["workers"] = args.workers
        if a["cfg"].get("freeze_trunk"):
            raise SystemExit(f"{a['name']}: freeze_trunk is not supported in pair mode")
    ref = shared_signature(specs[0]["cfg"])
    for a in specs[1:]:
        sig = shared_signature(a["cfg"])
        bad = sorted(k for k in set(ref) | set(sig) if json.dumps(ref.get(k), sort_keys=True, default=str)
                     != json.dumps(sig.get(k), sort_keys=True, default=str))
        if bad:
            raise SystemExit(f"{specs[0]['name']} and {a['name']} disagree on loader-determining "
                             f"key(s) {bad} -- they cannot share a stream")
    print(f"[train_pair] {len(specs)} arms share a stream; loader-determining keys verified identical")

    cfg0 = specs[0]["cfg"]
    tcfg0 = cfg0["train"]
    W, H = cfg0["input_size"]
    accum = tcfg0["accum"]

    # ---- per-arm state ----
    for a in specs:
        cfg, t = a["cfg"], a["cfg"]["train"]
        n_cls = n_corners(cfg.get("board"))
        a["model"] = DetectorNet(H, W, n_cls=n_cls, **detector_kwargs(cfg)).to(
            device, memory_format=torch.channels_last)
        a["eval_model"] = DetectorNet(H, W, n_cls=n_cls, **detector_kwargs(cfg)).to(
            device, memory_format=torch.channels_last)
        a["model"].train()
        a["ema"] = EMA(a["model"], decay=t["ema_decay"])
        a["optim"] = torch.optim.AdamW(param_groups(a["model"], t["wd"]), lr=t["lr"], betas=(0.9, 0.999))
        a["run_dir"] = Path("runs") / a["name"]
        a["logger"] = JsonlLogger(a["run_dir"] / "metrics.jsonl")
        a["preview_dir"] = a["run_dir"] / "preview"
        a["preview_dir"].mkdir(parents=True, exist_ok=True)
        a["step"], a["resume_count"], a["last_val"] = 0, 0, None
        if a["resume"]:
            ck = load_ckpt(a["resume"], a["model"], a["ema"], a["optim"], map_location=device)
            a["step"], a["resume_count"], a["last_val"] = ck["step"], ck["resume_count"] + 1, ck["last_val"]
        a["micro"], a["accum_loss"], a["n_samples"] = 0, 0.0, 0
        a["es"] = {**td._EARLY_STOP_DEFAULTS, **(t.get("early_stop") or {})}
        a["full_val_history"] = []
        print(f"[train_pair] {a['name']}: start_step={a['step']}/{a['total']} "
              f"resume_count={a['resume_count']} params={sum(p.numel() for p in a['model'].parameters()):,}")

    # ---- ONE stream, ONE val loader (the whole point) ----
    # Seed band: solo runs use train_seed*1000 + resume_count with small resume_count, so the
    # +500 offset guarantees a pair run never replays a sequence any solo run already consumed.
    # Without it, an arm resuming into a pair could re-see its own earlier samples.
    stream_seed = cfg0["synth"]["train_seed"] * 1000 + 500 + sum(a["resume_count"] for a in specs)
    print(f"[train_pair] stream_seed={stream_seed}")
    train_ds = SynthStream(cfg0, stream="detector", seed=stream_seed, render_targets=True)
    train_loader = DataLoader(train_ds, batch_size=tcfg0["batch"], num_workers=tcfg0["workers"],
                              multiprocessing_context="spawn", persistent_workers=True, pin_memory=True,
                              prefetch_factor=tcfg0["prefetch_factor"], worker_init_fn=td._worker_init)
    val_ds = SynthVal(cfg0, n=tcfg0["val_subset"], seed=cfg0["synth"]["val_seed"])
    val_loader = DataLoader(val_ds, batch_size=tcfg0["batch"], num_workers=8,
                            multiprocessing_context="spawn", persistent_workers=True,
                            collate_fn=partial(td._val_collate, cfg0))
    full_val_ds = SynthVal(cfg0, n=cfg0["synth"]["val_size"], seed=cfg0["synth"]["val_seed"])
    full_val_loader = DataLoader(full_val_ds, batch_size=tcfg0["batch"], num_workers=8,
                                 multiprocessing_context="spawn", persistent_workers=True,
                                 collate_fn=partial(td._val_collate, cfg0))

    train_iter = iter(train_loader)
    for a in specs:
        a["optim"].zero_grad(set_to_none=True)
    t0, val_time = time.time(), 0.0
    batches = 0

    while any(a["step"] < a["total"] for a in specs):
        batch = next(train_iter)
        # Moved to device ONCE and reused by every arm -- the H2D copy is shared too.
        images = batch["image"].to(device, non_blocking=True, memory_format=torch.channels_last)
        hms = batch["heatmap"].unsqueeze(1).to(device, non_blocking=True)
        cts = batch["classes"].to(device, non_blocking=True)
        nvis = batch["n_vis"].to(device, non_blocking=True)
        nvis_sum = nvis.sum()
        batches += 1

        for a in specs:
            if a["step"] >= a["total"]:
                continue                      # finished arm: stop training it, keep the others going
            cfg, t = a["cfg"], a["cfg"]["train"]
            a["n_samples"] += images.shape[0]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hm_logits, cls_logits = a["model"](images)
                loss = detector_loss(hm_logits, cls_logits, hms, cts, nvis_sum, cfg["lambda_cls"],
                                     loss_form=cfg.get("loss_form", "focal"),
                                     beta=cfg.get("focal_beta", 4)) / accum
            loss.backward()
            a["accum_loss"] += float(loss.detach()) * accum
            a["micro"] += 1
            if a["micro"] < accum:
                continue

            grad_norm = nn.utils.clip_grad_norm_(
                (p for p in a["model"].parameters() if p.requires_grad), t["clip_norm"])
            lr = cosine_lr(a["step"], a["total"], t["lr"], t["lr_floor"], t["warmup_steps"])
            for g in a["optim"].param_groups:
                g["lr"] = lr
            a["optim"].step()
            a["optim"].zero_grad(set_to_none=True)
            a["ema"].update(a["model"])

            avg_loss = a["accum_loss"] / accum
            elapsed = time.time() - t0 - val_time
            a["step"] += 1
            a["micro"], a["accum_loss"] = 0, 0.0
            a["logger"].log(step=a["step"], loss=avg_loss, lr=lr, grad_norm=float(grad_norm),
                            samples_per_s=a["n_samples"] / elapsed if elapsed > 0 else 0.0)
            if a["step"] % 10 == 0 or a["step"] <= 10:
                print(f"[{a['name']} step {a['step']}/{a['total']}] loss={avg_loss:.4f} lr={lr:.3e} "
                      f"samples/s={a['n_samples'] / elapsed if elapsed > 0 else 0:.2f}")

            if a["step"] % t["val_every"] == 0 or a["step"] == a["total"]:
                v0 = time.time()
                a["ema"].copy_to(a["eval_model"])
                res = td.run_validation(a["eval_model"], val_loader, cfg, device,
                                        t["tau_hm"], t["match_px"], a["preview_dir"], a["step"])
                print(f"[{a['name']} val {a['step']}] m01={res['m01']} m04={res['m04']}")
                a["logger"].log(step=a["step"], val=res)
                a["last_val"] = res
                val_time += time.time() - v0

            if a["step"] % t.get("ckpt_rolling_every", 1000) == 0:
                save_ckpt(a["run_dir"] / "ckpt_latest.pt", a["step"], a["resume_count"],
                          a["model"], a["ema"], a["optim"], cfg, a["last_val"])

            if a["step"] % t["full_val_every"] == 0 or a["step"] == a["total"]:
                v0 = time.time()
                a["ema"].copy_to(a["eval_model"])
                full = td.run_validation(a["eval_model"], full_val_loader, cfg, device,
                                         t["tau_hm"], t["match_px"], a["preview_dir"], a["step"])
                print(f"[{a['name']} FULL val {a['step']}] m01={full['m01']} m04={full['m04']}")
                a["logger"].log(step=a["step"], full_val=full)
                a["last_val"] = full
                save_ckpt(a["run_dir"] / f"ckpt_{a['step']:07d}.pt", a["step"], a["resume_count"],
                          a["model"], a["ema"], a["optim"], cfg, a["last_val"])
                val_time += time.time() - v0
                a["full_val_history"].append({"step": a["step"], **full})
                if td.early_stop_should_trigger(a["full_val_history"], a["step"], a["es"]):
                    print(f"[train_pair] {a['name']} early stop at {a['step']}")
                    a["total"] = a["step"]

            if a["step"] >= a["total"]:
                # Compatibility marker. tools/queue_ablations.sh and tools/resume_after_boot.sh
                # both decide "finished cleanly" by grepping runs/<name>.log for this exact
                # string. In pair mode stdout is one shared log, so write it per-arm as well --
                # otherwise a reboot would resume an arm that had actually completed.
                save_ckpt(a["run_dir"] / "ckpt_latest.pt", a["step"], a["resume_count"],
                          a["model"], a["ema"], a["optim"], cfg, a["last_val"])
                line = (f"[train_detector] done: step={a['step']} (train_pair, shared stream)\n")
                with open(Path("runs") / f"{a['name']}.log", "a") as fh:
                    fh.write(line)
                print(f"[train_pair] {a['name']} COMPLETE at step {a['step']}")

    elapsed = time.time() - t0 - val_time
    tot = sum(a["n_samples"] for a in specs)
    print(f"[train_pair] done: {batches} batches -> {tot} model-samples in {elapsed:.1f}s "
          f"({tot / elapsed if elapsed > 0 else 0:.2f} model-samples/s aggregate, "
          f"{batches * tcfg0['batch'] / elapsed if elapsed > 0 else 0:.2f} generated samples/s)")


if __name__ == "__main__":
    main()
