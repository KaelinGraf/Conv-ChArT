"""tools/train_charuconet.py -- ChArUcoNet (Hu et al. 2019) comparison-arm
training loop. Mirrors tools/train_detector.py's shape (spawn DataLoader,
bf16 autocast, AdamW cosine, EMA, periodic validation + checkpointing) but
targets tools/charuconet.py's SuperPoint-backbone model instead of
dcc.model.DetectorNet -- built as a wholly separate script so this arm's
comparison stays isolated from Conv-ChArT's own training path: no edits to
dcc/*, configs/*, or tools/train_detector.py.

Data path: dcc.dataset.SynthStream/SynthVal always generate at
cfg["input_size"] (640x480 for configs/rev640.yaml, this arm's board/synth
config source of physical-scale truth -- see _val_collate/_train_collate,
which downscale to ChArUcoNet's fixed 320x240 operating point and render its
65-way/17-way cell targets, all INSIDE this file's collate_fn, never inside
dcc/). Validation matches decoded 320x240 detections back at 640x480 (the
inverse of that same downscale) against the native-resolution GT records,
reusing tools/train_detector.py's pure M-01/M-02/M-04 helpers (OCTAVE_BINS,
_octave_bucket, _match_greedy, TAIL_PX) via an importlib load (see
_load_train_detector) -- never DetectorNet-specific parts of that module
(run_validation, the gate/attention hooks, viz previews).
"""
import argparse
import importlib.util
import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from charuconet import (IN_H, IN_W, charuconet_loss_components, dcModel, decode_cells, dihedral_maps,
                        load_reference_state_dict, pre_bgr_normalize, render_cell_targets,
                        verify_corner_id_convention)
from dcc.board import n_corners
from dcc.dataset import SynthStream, SynthVal, load_config
from dcc.trainutil import EMA, JsonlLogger, cosine_lr, param_groups, save_ckpt, warn_cfg_drift

ROOT = Path(__file__).resolve().parents[1]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/rev640.yaml")
    p.add_argument("--name", default="charuconet_ours")
    p.add_argument("--steps", type=int, default=None, help="override cfg train.steps")
    p.add_argument("--resume", default=None, help="OUR checkpoint (.pt from this trainer) to resume "
                   "from -- distinct from --pretrained, which loads JunkyByte's reference format. "
                   "Used to continue a fine-tune onto a REVISED generator without restarting it.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--batch", type=int, default=None, help="override cfg train.batch")
    p.add_argument("--accum", type=int, default=None, help="override cfg train.accum")
    p.add_argument("--val-every", type=int, default=None, help="override cfg train.val_every")
    p.add_argument("--val-subset", type=int, default=None, help="override cfg train.val_subset")
    p.add_argument("--pretrained", default=None,
                    help="JunkyByte/deepcharuco .ckpt to load before training (or, with --steps 0, "
                         "for a zero-training transfer EVAL of their checkpoint on our data/metrics)")
    p.add_argument("--prefetch-factor", type=int, default=2,
                    help="DataLoader prefetch_factor -- kept low (default 2) since this arm runs "
                         "concurrently with the main trainer's own loader pool and must stay CPU-light")
    p.add_argument("--lr", type=float, default=None, help="override cfg train.lr (cosine schedule peak)")
    p.add_argument("--lr-floor", type=float, default=None, help="override cfg train.lr_floor")
    p.add_argument("--warmup-steps", type=int, default=None, help="override cfg train.warmup_steps")
    p.add_argument("--board-dictionary", default=None,
                    help="override cfg board.dictionary (e.g. DICT_4X4_50) -- diagnostic use: isolates "
                         "the ArUco-dictionary variable from photometrics when comparing against a "
                         "pretrained checkpoint of unknown/suspected-different training dictionary; "
                         "board geometry (squares, marker_ratio) is untouched, so corner-ID topology "
                         "(verify_corner_id_convention) is unaffected by this override")
    return p


def _worker_init(_):
    cv2.setNumThreads(1)


def _load_train_detector():
    """tools.train_detector, loaded by absolute path -- an unrelated
    detectron2 'tools' package shadows any `import tools.train_detector`
    (see tests/test_variant640.py's identical fixture). Exists only to reuse
    its pure OCTAVE_BINS/_octave_bucket/_match_greedy/TAIL_PX matching
    helpers below -- never its DetectorNet-specific run_validation."""
    spec = importlib.util.spec_from_file_location("_train_detector_ref", ROOT / "tools" / "train_detector.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _to_320(cfg, record):
    """Downscales a native cfg["input_size"] (640x480 for rev640.yaml)
    record to ChArUcoNet's fixed 320x240 operating point: cv2.INTER_AREA
    image resize + the half-pixel corner-coordinate map x' = (x+0.5)*0.5 -
    0.5 -- the r=2 case of dcc/pipeline.py's own resize-ratio convention
    (x_input=(x_sensor+0.5)/r-0.5), applied here at a fixed ratio rather
    than a config-derived r since this arm's operating point never varies."""
    W, H = cfg["input_size"]
    assert (W, H) == (IN_W * 2, IN_H * 2), f"train_charuconet's 640->320 collate assumes a 640x480 source, got {(W, H)}"
    img320 = cv2.resize(record["image"], (IN_W, IN_H), interpolation=cv2.INTER_AREA)
    corners = record["corners"]
    pts = (np.array([[c["x"], c["y"]] for c in corners], dtype=np.float64).reshape(-1, 2) + 0.5) * 0.5 - 0.5
    vis = np.array([c["visible"] for c in corners], dtype=bool)
    idx = np.array([c["index"] for c in corners], dtype=int)
    return img320, pts, vis, idx


def _render_sample(cfg, record, n_cls):
    """pre_bgr_normalize (image-128)/255, NOT this project's usual /255.0 --
    see charuconet.py's own docstring: their pretrained BatchNorm stats are
    calibrated to this exact distribution. render_cell_targets' rare same-
    cell collision tie-break needs an rng (JunkyByte's own create_label
    coin-flips it); seeded from s_px so it's deterministic per record
    content -- exact reproducibility isn't load-bearing here (the tie-break
    only ever fires for two board corners landing in the same 8px cell of
    the 320x240 frame, an extreme-close-range edge case), just "doesn't
    silently vary run-to-run for no reason", matching this project's general
    seeded-stream ethos without needing a dedicated per-sample counter
    threaded through the collate call chain."""
    img320, pts, vis, idx = _to_320(cfg, record)
    rng = np.random.default_rng(int(record["s_px"] * 1e6) & 0xFFFFFFFF)
    loc_t, id_t = render_cell_targets(pts, vis, idx, n_cls, rng=rng)
    image = torch.from_numpy(pre_bgr_normalize(img320.astype(np.float32))).unsqueeze(0)
    return image, torch.from_numpy(loc_t), torch.from_numpy(id_t)


def _train_collate(cfg, n_cls, batch):
    """batch: list of (image, record) from SynthStream(render_targets=False,
    stream='detector') -- SynthStream itself always renders at
    cfg["input_size"], so the downscale+target step happens entirely here."""
    samples = [_render_sample(cfg, record, n_cls) for _, record in batch]
    images, locs, ids = zip(*samples)
    return {"image": torch.stack(images), "loc": torch.stack(locs), "id": torch.stack(ids)}


def _val_collate(cfg, n_cls, batch):
    """batch: list of (image, record) from SynthVal.__getitem__ -- records
    stay at native 640x480 resolution (returned alongside the 320x240
    tensors) so run_validation can match decoded detections against the
    original GT scale."""
    samples = [_render_sample(cfg, record, n_cls) for _, record in batch]
    images, locs, ids = zip(*samples)
    records = [record for _, record in batch]
    return torch.stack(images), torch.stack(locs), torch.stack(ids), records


def run_validation(model, loader, n_cls, device, match_px, ref):
    """Returns {val_loss, m01, m02, m04} -- M-01/M-02/M-04 per
    tools/train_detector.py's protocol, adapted for ChArUcoNet's own decode:
    decode_cells' 320x240 detections are scaled back to 640x480 (the inverse
    of _to_320's forward map) before matching against native-resolution GT,
    and per-detection identity comes straight from decode_cells' own id
    argmax at the matched cell -- no separate bilinear ID readout (that's
    dcc.pipeline.read_ids, a Conv-ChArT-specific mechanism this head has no
    analogue of: loc and id share one H/8 cell grid here, so a detection's
    identity is whatever the id head predicted for the cell that detected
    it)."""
    model.eval()
    errs, tail_hits = [], 0
    octaves = {k: [0, 0] for k, _lo, _hi in ref.OCTAVE_BINS}
    id_octaves = {k: [0, 0] for k, _lo, _hi in ref.OCTAVE_BINS}
    # All 8 dihedral relabelings scored alongside identity from the same decode
    # (Kaelin, 2026-07-28) -- a wrong corner-ID convention on either side of a
    # comparison is otherwise indistinguishable from genuinely bad id accuracy.
    # "identity" here duplicates id_octaves' own count (kept for a single,
    # self-contained 8-way table rather than a special-cased 9th field).
    maps = dihedral_maps(n_cls)
    map_correct = {name: 0 for name in maps}
    loss_sum, loss_loc_sum, loss_id_sum, loss_n = 0.0, 0.0, 0.0, 0
    # Diagnostics (Kaelin, 2026-07-28): a summed val_loss and a single m04
    # scalar can't distinguish "id_head collapsed to the dustbin class"
    # (~98.7% of cells have no corner, a strong local optimum for plain CE)
    # from "id_head learning but underpowered probe" from "convention
    # mismatch" -- id_histogram/id_dustbin_frac make the collapse case
    # directly visible: near-100% dustbin_frac is diagnostic on its own,
    # independent of m04's accuracy-under-a-possibly-wrong-mapping reading.
    id_histogram = np.zeros(n_cls + 1, dtype=np.int64)  # index n_cls = id-dustbin
    n_matches_total = 0

    with torch.no_grad():
        for images, loc_t, id_t, records in loader:
            images = images.to(device, non_blocking=True)
            loc_t_d, id_t_d = loc_t.to(device, non_blocking=True), id_t.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loc_logits, id_logits = model(images)
                loss_loc, loss_id = charuconet_loss_components(loc_logits, id_logits, loc_t_d, id_t_d)
            loss_sum += float(loss_loc + loss_id) * images.shape[0]
            loss_loc_sum += float(loss_loc) * images.shape[0]
            loss_id_sum += float(loss_id) * images.shape[0]
            loss_n += images.shape[0]

            for b in range(images.shape[0]):
                record = records[b]
                gt = [(c["x"], c["y"], c["index"]) for c in record["corners"] if c["visible"]]
                gt_xy = np.array([(x, y) for x, y, _ in gt], dtype=np.float64).reshape(-1, 2)

                xy320, idc, _loc_conf, _id_conf = decode_cells(loc_logits[b].float(), id_logits[b].float())
                det_xy = (xy320 + 0.5) * 2.0 - 0.5   # 320x240 -> 640x480, inverse of _to_320's map

                matches = ref._match_greedy(gt_xy, det_xy, match_px)
                bucket = ref._octave_bucket(record["s_px"])
                if bucket:
                    octaves[bucket][1] += len(gt_xy)
                    octaves[bucket][0] += len(matches)
                for gi, di, dist in matches:
                    errs.append(dist)
                    if dist > ref.TAIL_PX:
                        tail_hits += 1
                    pred = int(idc[di])
                    id_histogram[pred if pred >= 0 else n_cls] += 1
                    n_matches_total += 1
                    if bucket:
                        id_octaves[bucket][1] += 1
                        id_octaves[bucket][0] += int(pred == gt[gi][2])
                        for name, m in maps.items():
                            pred_mapped = int(m[pred]) if pred >= 0 else -1
                            map_correct[name] += int(pred_mapped == gt[gi][2])

    errs = np.array(errs)
    m01 = {"mean": float(errs.mean()) if errs.size else None,
           "median": float(np.median(errs)) if errs.size else None,
           "p95": float(np.percentile(errs, 95)) if errs.size else None,
           "tail_frac_gt4px": tail_hits / errs.size if errs.size else None,
           "n_matched": int(errs.size)}
    m02 = {k: (v[0] / v[1] if v[1] else None) for k, v in octaves.items()}
    tot_c, tot_n = sum(v[0] for v in id_octaves.values()), sum(v[1] for v in id_octaves.values())
    m04 = {"accuracy": tot_c / tot_n if tot_n else None,
           "by_octave": {k: (v[0] / v[1] if v[1] else None) for k, v in id_octaves.items()}}
    m04_dihedral = {name: (c / tot_n if tot_n else None) for name, c in map_correct.items()}
    id_dustbin_frac = float(id_histogram[n_cls] / n_matches_total) if n_matches_total else None
    id_histogram_dict = {str(i): int(id_histogram[i]) for i in range(n_cls)}
    id_histogram_dict["dustbin"] = int(id_histogram[n_cls])
    model.train()
    return {"val_loss": loss_sum / loss_n if loss_n else None,
            "loss_loc": loss_loc_sum / loss_n if loss_n else None,
            "loss_id": loss_id_sum / loss_n if loss_n else None,
            "m01": m01, "m02": m02, "m04": m04, "m04_dihedral": m04_dihedral,
            "id_dustbin_frac": id_dustbin_frac, "id_histogram": id_histogram_dict}


def main():
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    if args.board_dictionary:
        cfg["board"] = {**(cfg.get("board") or {}), "dictionary": args.board_dictionary}
        print(f"[train_charuconet] board.dictionary overridden to {args.board_dictionary} "
              f"(diagnostic run -- configs/rev640.yaml itself is untouched)")
    tcfg = cfg["train"]
    steps = args.steps if args.steps is not None else tcfg["steps"]
    batch = args.batch or tcfg["batch"]
    accum = args.accum or tcfg["accum"]
    val_every = args.val_every or tcfg["val_every"]
    val_subset = args.val_subset or tcfg["val_subset"]
    lr_peak = args.lr if args.lr is not None else tcfg["lr"]
    lr_floor = args.lr_floor if args.lr_floor is not None else tcfg["lr_floor"]
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else tcfg["warmup_steps"]

    # Concurrent-with-main-trainer loader cap (Kaelin, conditional GO 2026-07-28): this arm may run
    # alongside the live rev640_baseline trainer only under a restricted loader -- workers<=2, batch<=32,
    # prefetch_factor=2 -- so the two generator worker pools can't starve each other. Hard-asserted, not
    # silently clamped: a violation here means a launch-command mistake worth failing loudly on.
    # 2026-07-28 (team-lead): raised 2 -> 6. The cap above was scoped to "alongside the LIVE
    # rev640_baseline trainer", which held 12 loader workers; that run ended at 190k and its
    # checkpoints are retired, so the pool it protected no longer exists. This arm now runs
    # beside the refiner only (8 workers), so 6 + 8 = 14 of 24 cores -- still leaves headroom,
    # and 2 workers would make the 100k-step fine-tune generator-bound past the deadline.
    assert args.workers <= 6, f"--workers {args.workers} exceeds the concurrent-run cap of 6"
    assert batch <= 32, f"--batch {batch} exceeds the concurrent-run cap of 32"
    assert args.prefetch_factor <= 2, f"--prefetch-factor {args.prefetch_factor} exceeds the concurrent-run cap of 2"

    verify_corner_id_convention(cfg.get("board"))  # fail fast: an ID-convention mismatch invalidates M-04 silently otherwise
    print("[train_charuconet] corner-ID convention self-test vs JunkyByte/deepcharuco: PASS")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_cls = n_corners(cfg.get("board"))
    model = dcModel(n_ids=n_cls).to(device)
    eval_model = dcModel(n_ids=n_cls).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train_charuconet] device={device} dcModel params: {n_params} ({n_params / 1e6:.4f} M)")

    if args.resume and args.pretrained:
        raise SystemExit("--resume and --pretrained are mutually exclusive: --resume already carries "
                          "the weights this run reached, --pretrained would overwrite them with "
                          "JunkyByte's initial state and silently restart the fine-tune.")
    if args.pretrained:
        pre_ckpt = load_reference_state_dict(args.pretrained, model, map_location=device)
        print(f"[train_charuconet] loaded pretrained JunkyByte/deepcharuco weights from {args.pretrained} "
              f"(their epoch={pre_ckpt.get('epoch')} global_step={pre_ckpt.get('global_step')})")

    run_dir = Path("runs") / args.name
    logger = JsonlLogger(run_dir / "metrics.jsonl")

    if args.pretrained and steps == 0:
        # Zero-training transfer arm: their weights, our eval set/metrics, no
        # optimizer/EMA/training loop needed at all.
        val_workers = min(args.workers, 4)
        val_ds = SynthVal(cfg, n=val_subset, seed=cfg["synth"]["val_seed"])
        val_loader = DataLoader(val_ds, batch_size=batch, num_workers=val_workers,
                                 multiprocessing_context="spawn" if val_workers > 0 else None,
                                 persistent_workers=val_workers > 0,
                                 prefetch_factor=args.prefetch_factor if val_workers > 0 else None,
                                 collate_fn=partial(_val_collate, cfg, n_cls))
        ref = _load_train_detector()
        result = run_validation(model, val_loader, n_cls, device, tcfg["match_px"], ref)
        print(f"[train_charuconet] ZERO-TRAINING TRANSFER (their weights, our data): "
              f"loss={result['val_loss']} (loc={result['loss_loc']} id={result['loss_id']}) "
              f"m01={result['m01']} m02={result['m02']} m04={result['m04']} m04_dihedral={result['m04_dihedral']} "
              f"id_dustbin_frac={result['id_dustbin_frac']} id_histogram={result['id_histogram']}")
        logger.log(step=0, transfer_eval=result)
        return

    ema = EMA(model, decay=tcfg["ema_decay"])
    optim = torch.optim.AdamW(param_groups(model, tcfg["wd"]), lr=lr_peak, betas=(0.9, 0.999))

    # Optimizer-state confirmation (Kaelin, diagnostic request 2026-07-28): read the id_head's
    # actual param-group membership/LR/requires_grad straight off `optim`, not inferred from config --
    # a silent freeze or a param dropped from every group would otherwise be invisible until m04 stays
    # flat for reasons that have nothing to do with the LR schedule.
    id_head_names = {n for n, _p in model.named_parameters() if n.startswith(("convDa.", "bnDa.", "convDb."))}
    id_head_ids = {id(p) for n, p in model.named_parameters() if n in id_head_names}
    id_head_requires_grad = all(p.requires_grad for n, p in model.named_parameters() if n in id_head_names)
    id_head_group_lrs = {g["lr"] for g in optim.param_groups if any(id(p) in id_head_ids for p in g["params"])}
    print(f"[train_charuconet] id_head params: {len(id_head_names)} tensors, requires_grad={id_head_requires_grad}, "
          f"in optimizer param-group(s) with lr={id_head_group_lrs}")

    val_workers = min(args.workers, 4)
    val_ds = SynthVal(cfg, n=val_subset, seed=cfg["synth"]["val_seed"])
    val_loader = DataLoader(val_ds, batch_size=batch, num_workers=val_workers,
                             multiprocessing_context="spawn" if val_workers > 0 else None,
                             persistent_workers=val_workers > 0,
                             prefetch_factor=args.prefetch_factor if val_workers > 0 else None,
                             collate_fn=partial(_val_collate, cfg, n_cls))

    ref = _load_train_detector()

    optim.zero_grad(set_to_none=True)
    micro, accum_loss, accum_loss_loc, accum_loss_id, n_samples, step, last_val = 0, 0.0, 0.0, 0.0, 0, 0, None
    t0, val_time = time.time(), 0.0

    if args.pretrained:
        # Domain-adaptation curve (Kaelin, 2026-07-28): a fine-tune run's metrics.jsonl should start
        # from the SAME banked zero-shot point (--pretrained --steps 0) rather than only showing
        # post-adaptation values, so the curve reads as "adapting FROM their checkpoint," not
        # in isolation. Uses `model` directly (EMA's shadow is still an exact copy of the freshly
        # loaded pretrained weights at this point, before any optimizer step has run). Run BEFORE the
        # train loader/train_iter exist below -- concurrency-cap discipline: this arm shares CPU with
        # the main trainer's own loader pool, and there's no reason to double this run's own worker
        # count (2 train + 2 val = 4) during an eval-only phase that only needs val's 2.
        v0 = time.time()
        step0_val = run_validation(model, val_loader, n_cls, device, tcfg["match_px"], ref)
        print(f"[val step 0 -- pretrained, pre-finetune] loss={step0_val['val_loss']} "
              f"(loc={step0_val['loss_loc']} id={step0_val['loss_id']}) m01={step0_val['m01']} "
              f"m02={step0_val['m02']} m04={step0_val['m04']} m04_dihedral={step0_val['m04_dihedral']} "
              f"id_dustbin_frac={step0_val['id_dustbin_frac']} "
              f"id_histogram={step0_val['id_histogram']}")
        logger.log(step=0, val=step0_val)
        last_val = step0_val
        val_time += time.time() - v0

    resume_count = 0
    if args.resume:
        from dcc.trainutil import load_ckpt
        rck = load_ckpt(args.resume, model, ema, optim, map_location=device)
        warn_cfg_drift(rck, cfg, steps=steps)
        step = rck["step"]
        resume_count = rck.get("resume_count", 0) + 1
        print(f"[train_charuconet] RESUMED from {args.resume} at step={step} "
              f"(resume_count={resume_count}); continuing to {steps} under {args.config}")

    # resume_count bumps the stream seed so a resumed run draws FRESH samples rather
    # than replaying the sequence it already trained on -- same discipline as
    # tools/train_detector.py and tools/train_refiner.py.
    stream_seed = cfg["synth"]["train_seed"] * 1000 + resume_count
    train_ds = SynthStream(cfg, stream="detector", seed=stream_seed, render_targets=False)
    train_loader = DataLoader(train_ds, batch_size=batch, num_workers=args.workers,
                               multiprocessing_context="spawn" if args.workers > 0 else None,
                               persistent_workers=args.workers > 0, pin_memory=True,
                               prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
                               collate_fn=partial(_train_collate, cfg, n_cls),
                               worker_init_fn=_worker_init if args.workers > 0 else None)
    train_iter = iter(train_loader)

    while step < steps:
        batch_ = next(train_iter)
        images = batch_["image"].to(device, non_blocking=True)
        loc_t = batch_["loc"].to(device, non_blocking=True)
        id_t = batch_["id"].to(device, non_blocking=True)
        n_samples += images.shape[0]

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loc_logits, id_logits = model(images)
            loss_loc, loss_id = charuconet_loss_components(loc_logits, id_logits, loc_t, id_t)
            loss = (loss_loc + loss_id) / accum
        loss.backward()
        accum_loss += float(loss.detach()) * accum
        accum_loss_loc += float(loss_loc.detach())
        accum_loss_id += float(loss_id.detach())
        micro += 1
        if micro < accum:
            continue

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), tcfg["clip_norm"])
        lr = cosine_lr(step, steps, lr_peak, lr_floor, warmup_steps)
        for g in optim.param_groups:
            g["lr"] = lr
        optim.step()
        optim.zero_grad(set_to_none=True)
        ema.update(model)

        avg_loss = accum_loss / accum
        avg_loss_loc, avg_loss_id = accum_loss_loc / accum, accum_loss_id / accum
        elapsed = time.time() - t0 - val_time
        step += 1
        micro, accum_loss, accum_loss_loc, accum_loss_id = 0, 0.0, 0.0, 0.0
        logger.log(step=step, loss=avg_loss, loss_loc=avg_loss_loc, loss_id=avg_loss_id, lr=lr,
                   grad_norm=float(grad_norm), samples_per_s=n_samples / elapsed if elapsed > 0 else 0.0)
        if step % 10 == 0 or step <= 10:
            print(f"[step {step}/{steps}] loss={avg_loss:.4f} (loc={avg_loss_loc:.4f} id={avg_loss_id:.4f}) "
                  f"lr={lr:.3e} grad_norm={float(grad_norm):.3f} samples/s={n_samples / elapsed:.2f}")

        if step % val_every == 0 or step == steps:
            v0 = time.time()
            ema.copy_to(eval_model)
            last_val = run_validation(eval_model, val_loader, n_cls, device, tcfg["match_px"], ref)
            print(f"[val step {step}] loss={last_val['val_loss']} (loc={last_val['loss_loc']} id={last_val['loss_id']}) "
                  f"m01={last_val['m01']} m02={last_val['m02']} m04={last_val['m04']} "
                  f"m04_dihedral={last_val['m04_dihedral']} "
                  f"id_dustbin_frac={last_val['id_dustbin_frac']} id_histogram={last_val['id_histogram']}")
            logger.log(step=step, val=last_val)
            val_time += time.time() - v0

            # Milestone checkpoint: one unique file per val (this arm has no separate full_val
            # concept to hang a wider-spaced milestone off, unlike tools/train_detector.py -- val_every
            # itself is the natural boundary here). Cheap regardless: dcModel is ~1.24M params, ~20MB/file.
            ckpt_path = run_dir / f"ckpt_{step:07d}.pt"
            save_ckpt(ckpt_path, step, resume_count, model, ema, optim, cfg, last_val)
            print(f"[train_charuconet] wrote checkpoint {ckpt_path}")

        # Rolling resume point, overwritten in place (mirrors tools/train_detector.py's own
        # ckpt_rolling_every fix, added 2026-07-28 after a DataLoader worker died mid-run and cost
        # 14.7k steps because the only checkpoints were milestone-spaced): deliberately NOT tied to
        # val cadence -- crash insurance, not a measurement, one file overwritten in place rather than
        # accumulating.
        if step % tcfg.get("ckpt_rolling_every", 1000) == 0:
            # resume_count, NOT 0: the literal meant every checkpoint claimed to be a first
            # run, so resume #2 recomputed resume #1's stream_seed (:374) and REPLAYED its
            # exact sample sequence. This is the Deep ChArUco baseline on a crash-prone box;
            # a silently-replaying baseline is a fairness defect under the "baselines get a
            # fair shot" rule, not just a curiosity.
            save_ckpt(run_dir / "ckpt_latest.pt", step, resume_count, model, ema, optim, cfg, last_val)

    elapsed = time.time() - t0 - val_time
    print(f"[train_charuconet] done: step={step} train_elapsed={elapsed:.1f}s val_elapsed={val_time:.1f}s "
          f"samples/s={n_samples / elapsed if elapsed > 0 else 0.0:.2f}")


if __name__ == "__main__":
    main()
