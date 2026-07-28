"""tools/train_detector.py -- MT-05 Stage 1 (detector) training loop.

Recipe (all sourced from cfg["train"]): AdamW (lr, betas=(0.9,0.999), wd via
dcc.trainutil.param_groups so biases/norm params skip weight decay), micro-
batch cfg.train.batch x cfg.train.accum grad-accumulation steps per
optimizer step, bf16 autocast + channels_last, grad-clip by global norm
after accumulation (before the optimizer step), cosine LR + EMA update per
optimizer step. Validation runs forward passes on a second `eval_model`
instance loaded from EMA weights (ema.copy_to), so the live training model's
mode/grads are never disturbed by validation. Every validation pass also
dumps the first 3 images' draw_overlay/heatmap_overlay panels to
runs/<name>/preview/ -- a numeric gate alone missed a visually-obvious
generator defect once (see tools/preflight.py's generator_lock), so a human
glance at real predictions rides alongside the M-01/M-02/M-04 numbers below.

Metrics per MT-05: M-01 (coarse localisation error vs visible GT, matched by
greedy NN within cfg.train.match_px; TAIL_PX=4 is MT-03's refiner capture
range, the tail-fraction gate -- a fixed spec constant, not a config key),
M-02 (match ratio by s-octave, bins per test_generator.py's convention:
[12,16) [16,32) [32,64) [64,128], last bin closed; the sub-octave low bin is
the corner-only regime -- markers unreadable), and M-04 (ID accuracy on matched
corners, by octave) via a LAZY import of dcc.pipeline.read_ids -- that
module owns the bilinear ID readout (one canonical implementation); if it
isn't importable yet, M-04 is skipped with a one-time warning, never
reimplemented here.

P1/P2 diagnostics (gate3/gate4 alpha stats; per-block attention entropy on
one fp32 val image) are read via forward hooks on model.gate3/gate4/blocks
that recompute the signal from dcc.model's own exposed pieces rather than
reimplementing the gate/attention math: AttnGate.alpha(skip, g) (forward
itself returns only the gated skip) for P1, and Block.n1/qkv_heads (forward
runs attention through F.scaled_dot_product_attention, which never
materialises softmax(QK^T)) for P2. If a hooked module lacks these methods,
its diagnostic is silently omitted after one warning: this is instrumentation
around a dcc.model contract that could still change, not an acceptance gate,
so it must never crash a training run.

wandb mirroring is config-gated (cfg["train"]["wandb"]["enabled"], default
false) and strictly best-effort alongside metrics.jsonl, never a replacement
for it: every _wandb_* call swallows its own exceptions (warn-once via
_warn_once) so a telemetry outage (bad API key, no network, a wandb-side
error) can never crash or stall the run. When enabled, the run ID and URL
print as an unmissable `WANDB_RUN_ID=... WANDB_RUN_URL=...` line right after
init, for a supervising process to parse.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TAIL_PX = 4.0  # MT-01/MT-03: M-01's blocking tail threshold, sensor px (rho=1 here == input px)
OCTAVE_BINS = (("12-16", 12.0, 16.0), ("16-32", 16.0, 32.0), ("32-64", 32.0, 64.0), ("64-128", 64.0, 128.0))
_WARNED = set()


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--name", default="run")
    p.add_argument("--resume", default=None, help="checkpoint .pt to resume from")
    p.add_argument("--steps", type=int, default=None, help="override cfg train.steps")
    p.add_argument("--freeze-trunk", action="store_true")
    p.add_argument("--retarget-from", default=None,
                    help="base checkpoint .pt to retarget onto this config's board (loads every "
                         "non-cls.* tensor; requires --freeze-trunk)")
    return p


def _warn_once(key, msg):
    if key not in _WARNED:
        print(msg)
        _WARNED.add(key)


def _worker_init(_):
    import cv2
    cv2.setNumThreads(1)


def _octave_bucket(s_px):
    for name, lo, hi in OCTAVE_BINS:
        if lo <= s_px < hi or (s_px == hi and name == OCTAVE_BINS[-1][0]):
            return name
    return None


_EARLY_STOP_DEFAULTS = {"enabled": False, "patience": 3, "min_steps": 100000, "tail_gate": 0.01}


def _early_stop_series(window, get, direction):
    """One monitored metric's relative improvement across `window`
    (chronological full-val entries): direction="lower" (m01 mean) or
    "higher" (m02/m04 ratios). None if `get` is missing/None for ANY entry
    in `window` -- an unevaluable metric (empty octave bucket, m04 before
    dcc.pipeline is importable) drops out of the decision below rather than
    blocking or forcing a stop."""
    values = [get(h) for h in window]
    if any(v is None for v in values):
        return None
    start, best = values[0], (min(values) if direction == "lower" else max(values))
    if start == 0:
        return 0.0 if best == start else float("inf")
    return ((start - best) if direction == "lower" else (best - start)) / abs(start)


def early_stop_should_trigger(history, step, es_cfg):
    """Pure decision function behind the config-gated early stop
    (cfg["train"]["early_stop"]) -- factored out so it's testable without a
    training loop. `history` is the chronological list of full-val result
    dicts (run_validation()'s return shape, each additionally carrying its
    own "step") seen so far; `step` is the current global step. Stops iff,
    over the last `patience` consecutive full-vals: (1) m01's
    tail_frac_gt4px stayed <= tail_gate at every one, AND (2) none of the
    monitored metrics (m01 mean, each M-02 octave ratio, m04 accuracy)
    improved >1% relative from the window's first value to its best value
    inside the window -- i.e. training has both plateaued and stayed inside
    the refiner's capture range throughout."""
    cfg = {**_EARLY_STOP_DEFAULTS, **(es_cfg or {})}
    if not cfg["enabled"] or step < cfg["min_steps"]:
        return False
    patience = cfg["patience"]
    if patience < 1 or len(history) < patience:
        return False   # patience < 1: history[-0:] would slice the WHOLE history, not none of it
    window = history[-patience:]

    tails = [h["m01"]["tail_frac_gt4px"] for h in window]
    if any(t is None or t > cfg["tail_gate"] for t in tails):
        return False

    series = [_early_stop_series(window, lambda h: h["m01"]["mean"], "lower")]
    for name, _lo, _hi in OCTAVE_BINS:
        series.append(_early_stop_series(window, lambda h, k=name: h["m02"].get(k), "higher"))
    series.append(_early_stop_series(window, lambda h: (h["m04"] or {}).get("accuracy"), "higher"))
    return not any(s is not None and s > 0.01 for s in series)


def _wandb_init(cfg, wandb_cfg, run_dir, name):
    """Best-effort wandb run construction, gated by cfg["train"]["wandb"]
    ["enabled"] (default false) -- None when disabled. Any failure here
    (missing package, bad API key, no network) is a warn-once and training
    proceeds without telemetry: metrics.jsonl remains the source of truth
    regardless, same contract _wandb_log/_wandb_log_preview/_wandb_finish
    below all share."""
    if not wandb_cfg.get("enabled", False):
        return None
    try:
        import wandb
        run = wandb.init(project=wandb_cfg.get("project", "conv-chart"), name=name,
                          mode=wandb_cfg.get("mode", "online"), dir=str(run_dir), config=cfg)
        # Unmissable and grep-able: a supervising agent locates the run from this line.
        print(f"[train_detector] WANDB_RUN_ID={run.id} WANDB_RUN_URL={run.get_url()}")
        return run
    except Exception as e:
        _warn_once("wandb", f"[train_detector] wandb init failed ({type(e).__name__}: {e}); "
                   "continuing without telemetry.")
        return None


def _wandb_flatten(fields):
    """JsonlLogger-shaped kwargs -> wandb's flat metric namespace: a scalar
    kwarg logs under its own key unchanged; a dict-valued kwarg (val=result,
    full_val=result, early_stop={...}) becomes "<key>/<nested path,
    underscore-joined>" per leaf -- e.g. val={"m01": {"mean": 1.2}} ->
    {"val/m01_mean": 1.2}. None leaves are dropped (an unavailable metric,
    e.g. m04 before dcc.pipeline is importable, shouldn't open a wandb chart
    that would just stay empty)."""
    def walk(prefix, v, sep):
        if isinstance(v, dict):
            out = {}
            for k, vv in v.items():
                out.update(walk(f"{prefix}{sep}{k}", vv, "_"))
            return out
        return {} if v is None else {prefix: v}

    out = {}
    for key, val in fields.items():
        out.update(walk(key, val, "/"))
    return out


def _wandb_log(run, step, **fields):
    """Mirrors exactly what logger.log(step, **fields) receives -- call this
    alongside every JsonlLogger.log call, same kwargs, so the two stay in
    lockstep by construction rather than by separately-maintained call sites."""
    if run is None:
        return
    try:
        run.log(_wandb_flatten(fields), step=step)
    except Exception as e:
        _warn_once("wandb", f"[train_detector] wandb log failed ({type(e).__name__}: {e}); "
                   "continuing without further telemetry.")


def _wandb_log_preview(run, preview_dir, step, group):
    """Mirrors run_validation's first-3 preview PNGs -- already written to
    preview_dir at these exact filenames by run_validation itself -- as
    wandb.Image, keyed <group>/preview_<i>. Reads the files back rather than
    threading a new return value through run_validation, so its existing
    contract (and the tests that exercise it) stay untouched."""
    if run is None:
        return
    try:
        import wandb
        imgs = {f"{group}/preview_{i}": wandb.Image(str(p))
                for i in range(3) if (p := preview_dir / f"step_{step:07d}_val{i}.png").exists()}
        if imgs:
            run.log(imgs, step=step)
    except Exception as e:
        _warn_once("wandb", f"[train_detector] wandb preview log failed ({type(e).__name__}: {e}); "
                   "continuing without further telemetry.")


def _wandb_finish(run):
    if run is None:
        return
    try:
        run.finish()
    except Exception as e:
        _warn_once("wandb", f"[train_detector] wandb finish failed ({type(e).__name__}: {e}).")


def _val_targets(cfg, record):
    """Renders detector targets for one SynthVal (image, record) pair via
    dcc.targets directly -- mirrors dcc.dataset._render_detector_targets
    (private to that module) since SynthVal's raw-record shape differs from
    SynthStream's dict-yield contract."""
    import numpy as np
    import torch
    from dcc.board import n_corners
    from dcc.targets import render_class_targets, render_heatmap

    corners = record["corners"]
    h, w = record["image"].shape
    pts = np.array([[c["x"], c["y"]] for c in corners], dtype=np.float64).reshape(-1, 2)
    vis = np.array([c["visible"] for c in corners], dtype=bool)
    idx = np.array([c["index"] for c in corners], dtype=int)
    hm = render_heatmap(pts, vis, (w, h), sigma=cfg["sigma_hm"])
    ct = render_class_targets(pts, vis, idx, (w, h), sigma=cfg["sigma_cls"], n_cls=n_corners(cfg.get("board")))
    image = torch.from_numpy(record["image"]).float().unsqueeze(0) / 255.0
    return image, torch.from_numpy(hm), torch.from_numpy(ct), int(vis.sum())


def _val_collate(cfg, batch):
    import torch
    imgs, hms, cts, nvis, recs = [], [], [], [], []
    for _, record in batch:
        img, hm, ct, n = _val_targets(cfg, record)
        imgs.append(img)
        hms.append(hm)
        cts.append(ct)
        nvis.append(n)
        recs.append(record)
    return torch.stack(imgs), torch.stack(hms), torch.stack(cts), torch.tensor(nvis), recs


def _match_greedy(gt_xy, det_xy, max_px):
    """One-to-one nearest-neighbour match within max_px, closest pairs first
    (standard greedy keypoint-matching protocol). Returns [(gt_i, det_i,
    dist), ...]."""
    import numpy as np
    if len(gt_xy) == 0 or len(det_xy) == 0:
        return []
    d = np.linalg.norm(gt_xy[:, None, :] - det_xy[None, :, :], axis=-1)
    cand = sorted(((d[i, j], i, j) for i in range(d.shape[0]) for j in range(d.shape[1])
                   if d[i, j] <= max_px))
    used_gt, used_det, out = set(), set(), []
    for dist, i, j in cand:
        if i in used_gt or j in used_det:
            continue
        used_gt.add(i)
        used_det.add(j)
        out.append((i, j, dist))
    return out


def _gate_hook(name, captured):
    """dcc.model.AttnGate exposes alpha(skip, g) separately from forward
    (which returns only the gated skip tensor) -- recompute it from the same
    (skip, g) the hook sees forward receive, rather than guessing at a
    tuple-output contract forward doesn't have."""
    def fn(module, inputs, output):
        captured[name] = module.alpha(*inputs).detach()
    return fn


def _block_hook(name, captured):
    """dcc.model.Block runs attention via F.scaled_dot_product_attention,
    which never materialises softmax(QK^T) -- so recompute it explicitly,
    fp32, from the block's own (public) n1/qkv_heads, using the exact input
    the hook sees forward receive."""
    def fn(module, inputs, output):
        x_normed = module.n1(inputs[0])
        q, k, _v = module.qkv_heads(x_normed)
        scale = q.shape[-1] ** -0.5
        captured[name] = (q.float() @ k.float().transpose(-2, -1) * scale).softmax(dim=-1).detach()
    return fn


def _register_gate_hooks(model):
    """Cheap (1x1 conv) -- safe to leave attached for a whole batch sweep."""
    captured, handles = {}, []
    for gname in ("gate3", "gate4"):
        g = getattr(model, gname, None)
        if g is not None and hasattr(g, "alpha"):
            handles.append(g.register_forward_hook(_gate_hook(gname, captured)))
    return captured, handles


def _register_block_hooks(model):
    """Materialises a full (heads, T, T) attention matrix per block -- T is
    the H/16 bottleneck token count (thousands at this input res), so this is
    ONLY safe on a single image and must never be left attached across a
    batched validation sweep (a B=8 sweep at T~7500 tried to allocate ~14GB
    and OOM'd the whole run the first time this was smoke-tested)."""
    captured, handles = {}, []
    for i, blk in enumerate(getattr(model, "blocks", []) or []):
        if hasattr(blk, "qkv_heads") and hasattr(blk, "n1"):
            handles.append(blk.register_forward_hook(_block_hook(f"block{i}", captured)))
    return captured, handles


def _diag_log_fields(captured):
    fields = {}
    for gname in ("gate3", "gate4"):
        a = captured.get(gname)
        if a is not None:
            a = a.float()  # bf16 autocast tensor -- torch.quantile requires float32/64
            fields[f"{gname}_alpha_mean"] = float(a.mean())
            fields[f"{gname}_alpha_min"] = float(a.min())
            fields[f"{gname}_alpha_max"] = float(a.max())
            fields[f"{gname}_alpha_std"] = float(a.std())
            fields[f"{gname}_alpha_p10"] = float(a.quantile(0.1))
            fields[f"{gname}_alpha_p50"] = float(a.quantile(0.5))
            fields[f"{gname}_alpha_p90"] = float(a.quantile(0.9))
    for key in sorted(k for k in captured if k.startswith("block")):
        p = captured[key].float().clamp_min(1e-12)
        fields[f"{key}_attn_entropy"] = float(-(p * p.log()).sum(-1).mean())
    if not fields:
        _warn_once("diag_probe", "[train_detector] P1/P2 probe: model.gate3/gate4 lack .alpha() or "
                   "model.blocks lack .n1()/.qkv_heads() -- gate-alpha/attention-entropy diagnostics "
                   "skipped (dcc.model contract changed since this was written).")
    return fields


def run_validation(model, loader, cfg, device, tau_hm, match_px, preview_dir=None, step=None):
    """Returns {val_loss, m01, m02, m04, diag}; m04 is None if dcc.pipeline
    isn't importable yet. preview_dir/step (pass both together) additionally
    dump the first 3 images' draw_overlay/heatmap_overlay side-by-side panels
    to preview_dir/step_{step:07d}_val{i}.png -- pure visualisation, excluded
    from every metric above."""
    import cv2
    import numpy as np
    import torch
    from dcc import viz
    from dcc.losses import detector_loss
    from dcc.pipeline import merge_close, peaks

    try:
        from dcc.pipeline import read_ids
    except ImportError as e:
        read_ids = None
        _warn_once("pipeline", f"[train_detector] dcc.pipeline.read_ids unavailable ({e}); skipping M-04.")

    model.eval()
    gate_captured, gate_handles = _register_gate_hooks(model)
    errs, tail_hits = [], 0
    octaves = {k: [0, 0] for k, _, _ in OCTAVE_BINS}     # name -> [matched, visible]
    id_octaves = {k: [0, 0] for k, _, _ in OCTAVE_BINS}  # name -> [correct, total]
    loss_sum, loss_n, first_image, n_preview = 0.0, 0, None, 0

    with torch.no_grad():
        for images, hms, cts, nvis, records in loader:
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            # detector_loss's hm_t must carry the same explicit channel dim as
            # hm_logit (B,1,H,W); dataset.py/render_heatmap store (B,H,W) (no
            # channel axis, by convention -- see render_class_targets, which
            # bakes its 16 channels into the array from the start instead).
            # Unsqueezed here, at the call site, since dcc/losses.py isn't a
            # file this actor owns -- see report.
            hms_d = hms.unsqueeze(1).to(device, non_blocking=True)
            cts_d = cts.to(device, non_blocking=True)
            nvis_d = nvis.to(device, non_blocking=True)
            if first_image is None:
                first_image = images[:1].clone()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                hm_logits, cls_logits = model(images)
                # n_vis_batch is a scalar in detector_loss (float(n_vis_batch)
                # in its body) -- sum the per-sample counts, don't pass the
                # (B,) batch tensor.
                loss = detector_loss(hm_logits, cls_logits, hms_d, cts_d, nvis_d.sum(), cfg["lambda_cls"])
            loss_sum += float(loss) * images.shape[0]
            loss_n += images.shape[0]

            hm_prob = torch.sigmoid(hm_logits.float())
            cls_prob = torch.sigmoid(cls_logits.float())
            for b in range(images.shape[0]):
                record = records[b]

                if preview_dir is not None and n_preview < 3:
                    panel = cv2.hconcat([viz.draw_overlay(record["image"], record),
                                          viz.heatmap_overlay(record["image"], hm_prob[b, 0].cpu().numpy())])
                    cv2.imwrite(str(preview_dir / f"step_{step:07d}_val{n_preview}.png"), panel)
                    n_preview += 1

                gt = [(c["x"], c["y"], c["index"]) for c in record["corners"] if c["visible"]]
                gt_xy = np.array([(x, y) for x, y, _ in gt], dtype=np.float64).reshape(-1, 2)
                det_xy, _p_hm = merge_close(*peaks(hm_prob[b, 0], tau_hm))
                det_xy = det_xy.astype(np.float64)
                matches = _match_greedy(gt_xy, det_xy, match_px)
                bucket = _octave_bucket(record["s_px"])

                if bucket:
                    octaves[bucket][1] += len(gt_xy)
                    octaves[bucket][0] += len(matches)
                for _gi, _di, dist in matches:
                    errs.append(dist)
                    if dist > TAIL_PX:
                        tail_hits += 1

                if read_ids is not None and matches:
                    xy = torch.from_numpy(np.stack([det_xy[di] for _, di, _ in matches])).to(device)
                    pred_idx, _conf = read_ids(cls_prob[b], xy)
                    pred_idx = np.asarray(pred_idx.detach().cpu() if torch.is_tensor(pred_idx) else pred_idx)
                    if bucket:
                        for (gi, _di, _dist), p in zip(matches, pred_idx):
                            id_octaves[bucket][1] += 1
                            id_octaves[bucket][0] += int(int(p) == gt[gi][2])

    diag_fields = _diag_log_fields(gate_captured)
    for h in gate_handles:
        h.remove()

    if first_image is not None:
        block_captured, block_handles = _register_block_hooks(model)
        try:
            with torch.no_grad():
                model(first_image.float())
            diag_fields.update({k: v for k, v in _diag_log_fields(block_captured).items() if "entropy" in k})
        except torch.cuda.OutOfMemoryError as e:
            _warn_once("entropy_oom", f"[train_detector] P2 attention-entropy probe OOM'd ({e}); skipping "
                       "(instrumentation, not an acceptance gate).")
            torch.cuda.empty_cache()
        finally:
            for h in block_handles:
                h.remove()

    errs = np.array(errs)
    m01 = {"mean": float(errs.mean()) if errs.size else None,
           "median": float(np.median(errs)) if errs.size else None,
           "p95": float(np.percentile(errs, 95)) if errs.size else None,
           "tail_frac_gt4px": tail_hits / errs.size if errs.size else None,
           "n_matched": int(errs.size)}
    m02 = {k: (v[0] / v[1] if v[1] else None) for k, v in octaves.items()}
    m04 = None
    if read_ids is not None:
        tot_c, tot_n = sum(v[0] for v in id_octaves.values()), sum(v[1] for v in id_octaves.values())
        m04 = {"accuracy": tot_c / tot_n if tot_n else None,
               "by_octave": {k: (v[0] / v[1] if v[1] else None) for k, v in id_octaves.items()}}
    return {"val_loss": loss_sum / loss_n if loss_n else None, "m01": m01, "m02": m02, "m04": m04,
            "diag": diag_fields}


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.retarget_from and not args.freeze_trunk:
        parser.error("--retarget-from requires --freeze-trunk")

    import time
    from functools import partial

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from dcc.board import n_corners
    from dcc.dataset import SynthStream, SynthVal, load_config
    from dcc.losses import detector_loss  # noqa: F401 -- imported here so a missing dcc.losses fails fast
    from dcc.model import DetectorNet, detector_kwargs
    from dcc.trainutil import EMA, JsonlLogger, cosine_lr, load_ckpt, load_retarget_ckpt, param_groups, save_ckpt

    assert torch.cuda.is_available(), "CUDA required"
    assert torch.backends.cuda.flash_sdp_enabled(), "flash SDPA backend is disabled"
    assert torch.backends.cuda.mem_efficient_sdp_enabled(), "mem-efficient SDPA backend is disabled"
    device = torch.device("cuda")

    cfg = load_config(args.config)
    if args.steps is not None:
        cfg["train"]["steps"] = args.steps
    tcfg = cfg["train"]

    W, H = cfg["input_size"]
    n_cls = n_corners(cfg.get("board"))
    model = DetectorNet(H, W, n_cls=n_cls, **detector_kwargs(cfg)).to(device, memory_format=torch.channels_last)
    eval_model = DetectorNet(H, W, n_cls=n_cls, **detector_kwargs(cfg)).to(device,
                                                                            memory_format=torch.channels_last)

    freeze = args.freeze_trunk or cfg.get("freeze_trunk", False)
    model.train()
    if freeze:
        # Retarget path ONLY: everything except cls.* freezes (P12). This loop
        # must never run unconditionally -- doing so silently trains 2.1% of
        # the network on frozen random features (caught by the docs actor
        # 2026-07-27 after three prior gates missed it).
        for name, p in model.named_parameters():
            p.requires_grad_(name.startswith("cls."))
        bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
        for name, m in model.named_modules():
            if not name.startswith("cls") and isinstance(m, bn_types):
                m.eval()

    retargeted_from = None
    if args.retarget_from:
        # New-board retarget: the base checkpoint's own n_cls (hence cls.*
        # shape) may differ from this run's -- load_retarget_ckpt excludes
        # cls.* from the load entirely, so model's freshly-initialised class
        # head (built at THIS run's n_cls, above) is what actually trains.
        base_ckpt = load_retarget_ckpt(args.retarget_from, model, map_location=device)
        retargeted_from = {"path": str(args.retarget_from), "board": (base_ckpt.get("cfg") or {}).get("board")}

    ema = EMA(model, decay=tcfg["ema_decay"])
    optim = torch.optim.AdamW(param_groups(model, tcfg["wd"]), lr=tcfg["lr"], betas=(0.9, 0.999))

    run_dir = Path("runs") / args.name
    logger = JsonlLogger(run_dir / "metrics.jsonl")
    preview_dir = run_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = _wandb_init(cfg, tcfg.get("wandb", {}), run_dir, args.name)

    step, resume_count, last_val = 0, 0, None
    if args.resume:
        # freeze_trunk builds a fresh optimizer over cls params only (per
        # spec); a checkpoint saved from a full (unfrozen) run has optim
        # state shaped for many more params, so restoring it here would
        # mismatch this run's param_groups -- restore_optim=False for that
        # combination, model/ema/step/RNG still transfer as usual.
        ckpt = load_ckpt(args.resume, model, ema, optim, map_location=device, restore_optim=not freeze)
        step, resume_count, last_val = ckpt["step"], ckpt["resume_count"] + 1, ckpt["last_val"]

    stream_seed = cfg["synth"]["train_seed"] * 1000 + resume_count
    print(f"[train_detector] stream_seed={stream_seed} resume_count={resume_count} start_step={step}")

    train_ds = SynthStream(cfg, stream="detector", seed=stream_seed, render_targets=True)
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch"], num_workers=tcfg["workers"],
                               multiprocessing_context="spawn", persistent_workers=True, pin_memory=True,
                               prefetch_factor=tcfg["prefetch_factor"], worker_init_fn=_worker_init)

    # Matches the training micro-batch deliberately: val runs under no_grad
    # (no backward-pass activations to retain) so this is not the binding
    # memory constraint train_detector.py itself has, but the GPU here is
    # shared with other concurrent processes with fluctuating usage -- a
    # larger val batch bought no required functionality (unpinned choice)
    # and cost a real OOM mid-smoke-test the one time headroom was tight.
    val_batch = tcfg["batch"]
    val_ds = SynthVal(cfg, n=tcfg["val_subset"], seed=cfg["synth"]["val_seed"])
    val_loader = DataLoader(val_ds, batch_size=val_batch, num_workers=8, multiprocessing_context="spawn",
                             persistent_workers=True, collate_fn=partial(_val_collate, cfg))
    full_val_ds = SynthVal(cfg, n=cfg["synth"]["val_size"], seed=cfg["synth"]["val_seed"])
    full_val_loader = DataLoader(full_val_ds, batch_size=val_batch, num_workers=8,
                                  multiprocessing_context="spawn", persistent_workers=True,
                                  collate_fn=partial(_val_collate, cfg))

    total_steps, accum = tcfg["steps"], tcfg["accum"]
    es_cfg = {**_EARLY_STOP_DEFAULTS, **(tcfg.get("early_stop") or {})}
    full_val_history = []
    train_iter = iter(train_loader)
    optim.zero_grad(set_to_none=True)
    micro, accum_loss, n_samples = 0, 0.0, 0
    t0, val_time = time.time(), 0.0  # val_time excluded from samples/s below -- see report

    while step < total_steps:
        batch = next(train_iter)
        images = batch["image"].to(device, non_blocking=True, memory_format=torch.channels_last)
        hms = batch["heatmap"].unsqueeze(1).to(device, non_blocking=True)  # (B,H,W)->(B,1,H,W), see run_validation
        cts = batch["classes"].to(device, non_blocking=True)
        nvis = batch["n_vis"].to(device, non_blocking=True)
        n_samples += images.shape[0]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            hm_logits, cls_logits = model(images)
            loss = detector_loss(hm_logits, cls_logits, hms, cts, nvis.sum(), cfg["lambda_cls"]) / accum
        loss.backward()
        accum_loss += float(loss.detach()) * accum
        micro += 1
        if micro < accum:
            continue

        grad_norm = nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),
                                              tcfg["clip_norm"])
        lr = cosine_lr(step, total_steps, tcfg["lr"], tcfg["lr_floor"], tcfg["warmup_steps"])
        for g in optim.param_groups:
            g["lr"] = lr
        optim.step()
        optim.zero_grad(set_to_none=True)
        ema.update(model)

        avg_loss = accum_loss / accum
        # samples/s excludes val_time throughout -- otherwise every step's
        # throughput after the first validation is diluted by however long
        # that (and every prior) validation call took, since elapsed keeps
        # growing from t0 but n_samples only counts training samples.
        elapsed = time.time() - t0 - val_time
        step += 1
        micro, accum_loss = 0, 0.0
        train_fields = {"loss": avg_loss, "lr": lr, "grad_norm": float(grad_norm),
                         "samples_per_s": n_samples / elapsed if elapsed > 0 else 0.0}
        logger.log(step=step, **train_fields)
        _wandb_log(wandb_run, step, **train_fields)
        if step % 10 == 0 or step <= 10:
            print(f"[step {step}/{total_steps}] loss={avg_loss:.4f} lr={lr:.3e} "
                  f"grad_norm={float(grad_norm):.3f} samples/s={n_samples / elapsed:.2f}")

        if step % tcfg["val_every"] == 0 or step == total_steps:
            v0 = time.time()
            ema.copy_to(eval_model)
            result = run_validation(eval_model, val_loader, cfg, device, tcfg["tau_hm"], tcfg["match_px"],
                                     preview_dir, step)
            print(f"[val step {step}] loss={result['val_loss']} m01={result['m01']} m02={result['m02']} "
                  f"m04={result['m04']} diag={result['diag']}")
            logger.log(step=step, val=result)
            _wandb_log(wandb_run, step, val=result)
            _wandb_log_preview(wandb_run, preview_dir, step, "val")
            last_val = result
            val_time += time.time() - v0

        # Rolling resume point, overwritten in place (2026-07-28): the
        # milestone checkpoints below stay one unique file per
        # full_val_every (25k), but they alone left a 25k-step exposure
        # window -- a DataLoader worker died at step 164,670 and cost 14.7k
        # steps of real training, since the last milestone was 150k. This is
        # deliberately NOT tied to the val cadence: it is crash insurance,
        # not a measurement, and it costs 75 MB total rather than per write.
        if step % tcfg.get("ckpt_rolling_every", 1000) == 0:
            save_ckpt(run_dir / "ckpt_latest.pt", step, resume_count, model, ema, optim, cfg,
                      last_val, retargeted_from=retargeted_from)

        if step % tcfg["full_val_every"] == 0 or step == total_steps:
            v0 = time.time()
            ema.copy_to(eval_model)
            full_result = run_validation(eval_model, full_val_loader, cfg, device, tcfg["tau_hm"],
                                          tcfg["match_px"], preview_dir, step)
            print(f"[FULL val step {step}] loss={full_result['val_loss']} m01={full_result['m01']} "
                  f"m02={full_result['m02']} m04={full_result['m04']}")
            logger.log(step=step, full_val=full_result)
            _wandb_log(wandb_run, step, full_val=full_result)
            _wandb_log_preview(wandb_run, preview_dir, step, "full_val")
            last_val = full_result
            ckpt_path = run_dir / f"ckpt_{step:07d}.pt"
            save_ckpt(ckpt_path, step, resume_count, model, ema, optim, cfg, last_val,
                      retargeted_from=retargeted_from)
            print(f"[train_detector] wrote checkpoint {ckpt_path}")
            val_time += time.time() - v0

            full_val_history.append({"step": step, **full_result})
            if early_stop_should_trigger(full_val_history, step, es_cfg):
                # An exit, not a state: nothing here (or in save_ckpt above) marks the
                # checkpoint or cfg as early-stopped, so a later --resume from ckpt_path
                # just continues the ordinary loop past this step, unaware it happened.
                window = full_val_history[-es_cfg["patience"]:]
                early_stop_record = {"window_steps": [h["step"] for h in window],
                                      "window": [{"step": h["step"], "m01": h["m01"], "m02": h["m02"],
                                                  "m04": h["m04"]} for h in window]}
                logger.log(step=step, early_stop=early_stop_record)
                _wandb_log(wandb_run, step, early_stop=early_stop_record)
                print(f"[train_detector] early stop: flat and tail_frac_gt4px<={es_cfg['tail_gate']} over "
                      f"the last {es_cfg['patience']} full-vals (steps {[h['step'] for h in window]}); "
                      f"stopping at step {step}, checkpoint={ckpt_path}")
                break

    elapsed = time.time() - t0 - val_time
    print(f"[train_detector] done: step={step} train_elapsed={elapsed:.1f}s val_elapsed={val_time:.1f}s "
          f"samples/s={n_samples / elapsed if elapsed > 0 else 0.0:.2f}")
    _wandb_finish(wandb_run)


if __name__ == "__main__":
    main()
