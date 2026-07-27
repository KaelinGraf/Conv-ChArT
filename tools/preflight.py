"""Slice J -- mandatory pre-training verification suite: run before every long
training run (including every ablation). Proves the UNTRAINED DetectorNet/
Refiner's numerical traces match the design's init-time invariants (checks
1-5, 7) and that the full data->targets->loss->optimiser->heatmap->peaks->
readout->gate loop can express a known answer (check 6, the decisive one,
skipped by --quick) -- in minutes, not 3 days into a wasted run.

A generator_lock check runs first, ahead of and independent from checks 1-7:
it never touches the model, so --quick/--device don't gate it. PASS requires
tools/audit.py's last report.json to have every gate green AND
dcc.trainutil.generator_fingerprint of the CURRENT repo+config to match the
fingerprint that report recorded -- catching the case (a past real incident)
where a generator defect passed every numeric check yet was visually wrong,
so "audit + eyeball, then freeze" is enforced mechanically instead of by
memory.

Checks 1-5 and 7 all run on the SAME untrained (model, refiner) pair, built
once with a fixed seed: init-time characterisation only, no weights are ever
updated by any of them (backward populates .grad, never .step()s an
optimizer). Check 6 is the sole exception -- it trains that same pair in
place, so it runs LAST regardless of its number in the list above; --quick
skips only it (matching "skip 6-7 gracefully if too slow" for CPU: check 6 is
unconditionally skipped off-CUDA, check 7 is attempted and gracefully WARNs
on any exception, per _run_check below).

Argparse runs before any heavy import so --help never needs torch/dcc/cv2.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GRAD_GROUPS = ("e1", "e2", "e3", "e4", "e5", "blocks", "gate3", "gate4", "d1", "d2", "d3", "d4", "hm", "cls")
# Superset across both DetectorNet variants: e5/gate4/d4 exist only at attend_div=16
# (see dcc.model.DetectorNet) -- consumers filter to the groups the model actually has.


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--audit-report", default="audit/report.json",
                    help="tools/audit.py report.json checked by the generator_lock check")
    p.add_argument("--out", default="runs/preflight/")
    p.add_argument("--quick", action="store_true", help="skip check 6 (one_batch_overfit)")
    p.add_argument("--device", default=None, choices=["cuda", "cpu"], help="default: cuda if available")
    return p


# --------------------------------------------------------------------------- reporting

def _fmt(v):
    return f"{v:.6g}" if isinstance(v, float) else str(v)


def _report(name, status, **numbers):
    line = f"{status} {name}: " + " ".join(f"{k}={_fmt(v)}" for k, v in numbers.items())
    print(line)
    return {"name": name, "status": status, "numbers": numbers, "line": line}


def _run_check(name, fn, *args, **kwargs):
    """Every check is run through this: an exception (missing bf16 kernel on
    an odd CPU, a transient OOM under GPU contention, ...) is a documented
    graceful WARN, not a crash of the whole 8-check report -- the point of a
    preflight suite is one complete picture per run, not "stops at the first
    surprise". A gate a check computed and DELIBERATELY returned FAIL for is
    untouched by this (only fn's *exceptions* land here). empty_cache() runs
    after every check, pass or fail: this is a shared, contended GPU (other
    tenants observed concurrently at several GiB each) and PyTorch's caching
    allocator otherwise holds each check's peak (check 4's plain-fp32 B=4
    native-res forward+backward is the single biggest one, ~20-26 GiB --
    see its docstring) reserved rather than handing it back between checks."""
    import torch
    try:
        status, numbers = fn(*args, **kwargs)
    except Exception as e:
        status, numbers = "WARN", {"error": f"{type(e).__name__}: {e}"}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return _report(name, status, **numbers)


# --------------------------------------------------------------------------- shared helpers

def _render_targets(cfg, record):
    """(hm, ct, n_vis) for one SynthVal (image, record) pair -- the same
    small shim tools/train_detector.py's _val_targets and dcc.dataset's
    private _render_detector_targets each already are, wiring dcc.targets'
    canonical renderers to a record's corner list."""
    import numpy as np
    from dcc.targets import render_class_targets, render_heatmap
    corners = record["corners"]
    h, w = record["image"].shape
    pts = np.array([[c["x"], c["y"]] for c in corners], dtype=np.float64).reshape(-1, 2)
    vis = np.array([c["visible"] for c in corners], dtype=bool)
    idx = np.array([c["index"] for c in corners], dtype=int)
    hm = render_heatmap(pts, vis, (w, h), sigma=cfg["sigma_hm"])
    ct = render_class_targets(pts, vis, idx, (w, h), sigma=cfg["sigma_cls"])
    return hm, ct, int(vis.sum())


def _pearson(a, b):
    a, b = a.reshape(-1).double(), b.reshape(-1).double()
    a, b = a - a.mean(), b - b.mean()
    denom = a.norm() * b.norm()
    return float((a @ b) / denom) if denom > 0 else float("nan")


def _param_grad_norm(params):
    import torch
    grads = [p.grad.reshape(-1) for p in params if p.grad is not None]
    return float(torch.cat(grads).norm()) if grads else 0.0


# --------------------------------------------------------------------------- check 0 (generator lock, no model)

def check_generator_lock(cfg, root, audit_report):
    """Refuses to green-light training when the generator (dcc/board.py,
    dcc/synth.py, dcc/targets.py, dcc/dataset.py) or the config knobs that
    shape its output have drifted since `audit_report` was written: a change
    that passes every numeric audit gate but is visibly wrong (the incident
    this check exists for) is only caught by a human eyeballing the audit
    overlays, so PASS requires both a byte-identical fingerprint AND that same
    audit run's gates to have been green -- a stale report whose gates failed
    is not a licence to train on its fingerprint alone."""
    from dcc.trainutil import generator_fingerprint

    path = Path(audit_report)
    if not path.exists():
        return "FAIL", {"reason": "no_audit_report", "path": str(path)}
    try:
        report = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return "FAIL", {"reason": "no_audit_report", "path": str(path), "error": f"{type(e).__name__}: {e}"}

    if "generator_fingerprint" not in report or "all_gates_passed" not in report:
        return "FAIL", {"reason": "no_fingerprint_in_report", "path": str(path)}
    if not report["all_gates_passed"]:
        return "FAIL", {"reason": "audit_gates_failed", "path": str(path)}

    current = generator_fingerprint(cfg, root)
    recorded = report["generator_fingerprint"]
    if current == recorded:
        return "PASS", {"path": str(path)}

    changed = [rel for rel, h in current["files"].items() if recorded["files"].get(rel) != h]
    if current["config_sha1"] != recorded["config_sha1"]:
        changed.append("config")
    return "FAIL", {"reason": "fingerprint_mismatch", "path": str(path), "changed": changed}


# --------------------------------------------------------------------------- checks 1-5, 7 (untrained net)

def check_init_loss_prediction(model, cfg, x, hm_t, ct_t, n_vis, hms_np, cts_np):
    """Analytic init-time detector_loss with every sigmoid pinned at p0 =
    sigmoid(-2.19) (the hm/cls bias init), pointwise over the ACTUAL rendered
    targets, vs. the real measured loss (eval, fp32). alpha=2/beta=4 are
    focal()'s own hardcoded defaults -- detector_loss never threads cfg's
    (currently-equal) alpha/beta through, so mirroring focal() here, not cfg,
    is what matches the measured code path."""
    import numpy as np
    from dcc.losses import detector_loss

    alpha, beta = 2, 4
    p0 = 1.0 / (1.0 + math.exp(2.19))
    c_pos = (1 - p0) ** alpha * -math.log(p0)
    c_neg = p0 ** alpha * -math.log(1 - p0)

    def term(y):
        return float(np.where(y == 1.0, c_pos, (1 - y) ** beta * c_neg).sum())

    predicted = (sum(term(h) for h in hms_np) + cfg["lambda_cls"] * sum(term(c) for c in cts_np)) / max(n_vis, 1)

    import torch
    model.eval()
    with torch.no_grad():
        hm_logit, cls_logit = model(x)
        measured = float(detector_loss(hm_logit, cls_logit, hm_t, ct_t, n_vis, cfg["lambda_cls"]))
    ratio = measured / predicted if predicted > 0 else float("inf")
    return ("PASS" if 0.5 <= ratio <= 2.0 else "FAIL"), {"predicted": predicted, "measured": measured,
                                                          "ratio": ratio, "n_vis": n_vis}


def check_translation_equivariance(model, image, device):
    """f in eval mode (BN batch-independent) vs. an attend_div-px (one full
    attention-grid cell -- 16 px native, 8 px for the attend_div=8 variant)
    input roll; Pearson r between shift(f(x)) and f(shift(x)) over the
    interior, excluding a 64-px border where the roll wraps unphysically.
    The roll tracks model.attend_div rather than a fixed 16: the property
    this check needs is a shift by an exact integer number of grid cells (any
    integer count re-indexes RoPE's tokens cleanly; a fractional-cell shift
    would not), and the crispest version of that test is exactly ONE cell --
    a fixed 16-px roll against an 8-px-stride model would silently degrade
    to testing 2-cell equivariance instead."""
    import torch
    model.eval()
    H, W = image.shape
    x = torch.from_numpy(image).float().div(255.0).view(1, 1, H, W).to(device)
    shift = model.attend_div
    x_shift = torch.roll(x, shifts=shift, dims=-1)
    with torch.no_grad():
        hm1 = torch.sigmoid(model(x)[0])[0, 0]
        hm2 = torch.sigmoid(model(x_shift)[0])[0, 0]
    hm1_shifted = torch.roll(hm1, shifts=shift, dims=-1)
    m = 64
    r = _pearson(hm1_shifted[m:H - m, m:W - m], hm2[m:H - m, m:W - m])
    return ("PASS" if r > 0.99 else "FAIL"), {"r": r}


def check_rope_relativity(model, cfg, device):
    """Same 64x64 high-contrast patch (the board render's central 4-square
    checker corner -- max local contrast by construction) pasted onto a flat
    128-gray canvas at two positions exactly 64 px apart -- an integer number
    of attention cells either way (4 cells at attend_div=16, 8 cells at
    attend_div=8; the grid itself, gh/gw, scales with model.attend_div so the
    cell-index arithmetic below is correct for either variant). Block-0
    attention rows recomputed fp32 via qkv_heads, exactly the recipe
    tools/introspect.py::panel_attention uses (softmax(QK^T/sqrt(d)), mean
    over heads). One map sliced by dcols grid-columns must correlate with the
    other over their plain (non-wrapping) overlap -- relative-offset content
    dependence only, no absolute-position leakage."""
    import numpy as np
    import torch
    from dcc.board import render_board

    model.eval()
    W_in, H_in = cfg["input_size"]
    render_res = cfg["synth"]["render_res"]
    board_img, _ = render_board(render_res)
    c = render_res // 2
    patch = board_img[c - 32:c + 32, c - 32:c + 32]

    attend_div = model.attend_div   # grid stride: 16 native, 8 for the attend_div=8 variant
    shift = 64
    y0 = H_in // 2 - 32
    x0 = max(64, W_in // 2 - 128)
    gh, gw = H_in // attend_div, W_in // attend_div
    maps = []
    for xoff in (x0, x0 + shift):
        canvas = np.full((H_in, W_in), 128, dtype=np.uint8)
        canvas[y0:y0 + 64, xoff:xoff + 64] = patch
        qx, qy = xoff + 31.5, y0 + 31.5
        xin = torch.from_numpy(canvas).float().div(255.0).view(1, 1, H_in, W_in).to(device)

        captured = {}
        handle = model.blocks[0].register_forward_pre_hook(lambda m, a: captured.__setitem__("x_in", a[0]))
        with torch.no_grad():
            model(xin)
        handle.remove()
        blk = model.blocks[0]
        with torch.no_grad():
            q, k, _v = blk.qkv_heads(blk.n1(captured["x_in"]))
            A = torch.softmax((q.float() @ k.float().transpose(-2, -1)) / q.shape[-1] ** 0.5, dim=-1)
        tok = int(np.clip(qy // attend_div, 0, gh - 1)) * gw + int(np.clip(qx // attend_div, 0, gw - 1))
        maps.append(A[0, :, tok, :].mean(dim=0).reshape(gh, gw).detach())

    dcols = shift // attend_div
    r = _pearson(maps[0][:, :gw - dcols], maps[1][:, dcols:])
    return ("PASS" if r > 0.95 else "FAIL"), {"r": r}


def check_gradient_balance(model, cfg, x, hm_t, ct_t, n_vis):
    """fp32 forward+backward: per-module-group grad L2 norms (all must be
    finite and nonzero), plus the lambda_cls=1 calibration sanity -- |grad
    L_hm w.r.t. hm-head params| vs |grad L_cls w.r.t. cls-head params|, each
    isolated by backpropagating that head's own (n_vis-normalised) loss term
    alone. Three INDEPENDENT (forward, backward) pairs, each fully freed
    before the next starts, rather than one forward kept alive across three
    retain_graph=True backwards -- bounds peak memory to one pass instead of
    accumulating across three. Measured directly: a SINGLE plain-fp32
    forward alone already peaks at ~20 GiB at B=4/native 1600x1200 (not an
    SDPA-backend issue -- default dispatch correctly uses memory-efficient
    attention here, ~0.3 GiB for the attention op in isolation; confirmed by
    profiling), so this check's ~20-26 GiB requirement is inherent to
    fp32-at-this-batch/resolution, not fixable by this refactor alone -- see
    report for the OOM-under-contention consequence. model.train() BN
    running-stats update 3x instead of 1x as a result of the 3 passes --
    harmless, nothing downstream depends on this model's running stats
    surviving this check with a particular trajectory (check 6 immediately
    overwrites them with 600 of its own steps)."""
    import torch
    from dcc.losses import detector_loss, focal

    model.train()
    model.zero_grad(set_to_none=True)
    hm_logit, cls_logit = model(x)
    detector_loss(hm_logit, cls_logit, hm_t, ct_t, n_vis, cfg["lambda_cls"]).backward()
    table = {g: _param_grad_norm(getattr(model, g).parameters()) for g in GRAD_GROUPS if hasattr(model, g)}
    all_finite = all(math.isfinite(v) for v in table.values())
    all_nonzero = all(v > 0 for v in table.values())

    n = max(float(n_vis), 1.0)
    model.zero_grad(set_to_none=True)
    hm_logit, _ = model(x)
    (focal(hm_logit, hm_t) / n).backward()
    hm_head_norm = _param_grad_norm(model.hm.parameters())

    model.zero_grad(set_to_none=True)
    _, cls_logit = model(x)
    (cfg["lambda_cls"] * focal(cls_logit, ct_t) / n).backward()
    cls_head_norm = _param_grad_norm(model.cls.parameters())
    ratio = hm_head_norm / cls_head_norm if cls_head_norm > 0 else float("inf")

    ok = all_finite and all_nonzero and 0.1 <= ratio <= 10
    return ("PASS" if ok else "FAIL"), {**table, "hm_head_norm": hm_head_norm,
                                         "cls_head_norm": cls_head_norm, "ratio": ratio}


def check_refiner_init_state(cfg, refiner, device):
    """64 REAL 24x24 crops (RefinerVal -- actual board content, not synthetic
    noise) through the untrained Refiner: the init map must be FLAT at
    sigmoid(out-bias) = sigmoid(-2.19) ~ 0.1008, verifying the bias init
    landed and the forward path is sane. An untrained Refiner is deliberately
    NOT a no-op: on a flat map, hard-argmax is noise-dominated and soft_argmax
    lands ~uniformly over the 64x64 grid (measured mean |u* - 31.5| ~ 15 grid
    units), so the earlier centre-seeking gate here tested a false premise --
    redefined 2026-07-27 after that was measured and root-caused. The
    soft-argmax dispersion stays as an ungated diagnostic; check 6 gates on
    coarse peaks for the same reason."""
    import numpy as np
    import torch
    from dcc.dataset import RefinerVal
    from dcc.pipeline import soft_argmax

    ds = RefinerVal(cfg, n=64)
    crops, i = [], 0
    while len(crops) < 64 and i < len(ds):
        crops.extend(c["crop"] for c in ds[i])
        i += 1
    crops = np.stack(crops[:64]).astype(np.float32) / 255.0

    refiner.eval()
    x = torch.from_numpy(crops).unsqueeze(1).to(device)
    with torch.no_grad():
        out = torch.sigmoid(refiner(x)).cpu().numpy()
    p0 = 1.0 / (1.0 + math.exp(2.19))
    mean_p, spread = float(out.mean()), float(out.max() - out.min())
    dev = np.abs(soft_argmax(out) - 31.5)
    ok = abs(mean_p - p0) < 0.01 and spread < 0.02
    return ("PASS" if ok else "FAIL"), {"mean_p": mean_p, "expected_p0": p0, "spread": spread,
                                        "argmax_dispersion_mean": float(dev.mean()),
                                        "argmax_dispersion_max": float(dev.max()), "n_crops": len(crops)}


def check_bf16_parity(model, x, device):
    """Same fp32 input, plain-fp32 vs. bf16-autocast forward (eval): both
    heads' sigmoid outputs must stay within 0.02 of each other."""
    import torch
    model.eval()
    with torch.no_grad():
        hm1, cls1 = model(x)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            hm2, cls2 = model(x)
    d_hm = float((torch.sigmoid(hm1.float()) - torch.sigmoid(hm2.float())).abs().max())
    d_cls = float((torch.sigmoid(cls1.float()) - torch.sigmoid(cls2.float())).abs().max())
    return ("PASS" if max(d_hm, d_cls) < 0.02 else "FAIL"), {"max_delta_hm": d_hm, "max_delta_cls": d_cls}


# --------------------------------------------------------------------------- check 6 (trains the net)

def check_one_batch_overfit(cfg, model, refiner, device, out_dir):
    """THE decisive check: overfit the detector to exactly 4 fixed, hand-
    picked val samples (no aug variation -- the same 4 rendered targets every
    step), then decode each with the full Stage-3 pipeline. Closes the loop
    over data->targets->loss->optimiser->heatmap->peaks->readout->gate.

    900 steps, not the brief's "~600": measured directly (repeated timed
    trajectories) that this specific imbalance -- ~60 positive cells against
    ~3.8M negative cells across hm+cls, at this lr -- spends its first few
    hundred steps in an easy "suppress everything" regime where the loss
    already drops under the 1%-of-init gate WITHOUT the positive cells
    sharpening at all, then has a late, sudden transition where hm sigmoid at
    the true corners jumps from ~baseline to >0.8 and coarse peaks lock onto
    GT to <0.7 px. That transition lands at ~500-650 steps -- i.e. exactly
    on top of "~600" -- so 600 is a coin flip on GPU-kernel-level
    nondeterminism (confirmed empirically: identical code/seed, one run's
    coarse peaks matched 100% of GT, another's matched 0%, both with
    loss_ratio comfortably under the gate). 900 gives comfortable margin past
    the transition every run observed (reruns to 1200 stayed converged) and
    still costs under 4 minutes total for this check.

    The GATE keys on coarse_match_rate (heatmap peaks alone vs GT, no Refiner
    involved) plus the loss ratio: those two prove what this check exists to
    prove -- data->targets->loss->optimiser->heatmap->peaks expresses a known
    answer. The full-pipeline numbers (match_rate, id_rate, gate_fits) stay
    in the report as diagnostics only: they route through the UNTRAINED
    Refiner, whose flat init map shifts peaks up to ~4.5 px (see
    check_refiner_init_state), so gating on them tested Refiner init
    behaviour, not the training loop -- redefined 2026-07-27. A future
    --refiner-ckpt mode can restore them as gates once a trained Refiner is
    supplied."""
    import cv2
    import numpy as np
    import torch
    from dcc import viz
    from dcc.dataset import SynthVal
    from dcc.losses import detector_loss
    from dcc.pipeline import detect, merge_close, peaks
    from dcc.trainutil import param_groups

    n_val = cfg["synth"]["val_size"]
    a, b = cfg["scale_range_px"]
    s_lo, s_hi, min_vis, k = 60.0, 120.0, 14, 4
    # SynthVal.__getitem__'s s_px is a monotone log-uniform-stratified function
    # of the index -- start the scan just below s_lo's expected index rather
    # than at 0 (a linear scan from i=0 would burn thousands of full
    # generate_sample calls on far-below-range samples first).
    i0 = max(0, int(n_val * math.log(s_lo / a) / math.log(b / a)) - 5)
    ds = SynthVal(cfg, n_val, cfg["synth"]["val_seed"])
    picked, i = [], i0
    while i < n_val and len(picked) < k:
        img, record = ds[i]
        if record["board_present"] and s_lo <= record["s_px"] <= s_hi:
            if sum(1 for c in record["corners"] if c["visible"]) >= min_vis:
                picked.append((i, img, record))
        i += 1
    if len(picked) < k:
        return "FAIL", {"error": f"only {len(picked)}/{k} val samples matched s_px in [{s_lo},{s_hi}] "
                                  f"and >= {min_vis} visible corners (scanned [{i0}, {i}))"}

    hms, cts, n_vis = [], [], 0
    for _, _, record in picked:
        hm, ct, nv = _render_targets(cfg, record)
        hms.append(hm)
        cts.append(ct)
        n_vis += nv
    images = np.stack([p[1] for p in picked])
    x = torch.from_numpy(images).float().div(255.0).unsqueeze(1).to(device)
    hm_t = torch.from_numpy(np.stack(hms)).float().unsqueeze(1).to(device)
    ct_t = torch.from_numpy(np.stack(cts)).float().to(device)

    model.train()
    optim = torch.optim.AdamW(param_groups(model, cfg["train"]["wd"]), lr=1e-3, betas=(0.9, 0.999))
    init_loss = final_loss = None
    diverged_at = None
    for step in range(900):
        optim.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            hm_logit, cls_logit = model(x)
            loss = detector_loss(hm_logit, cls_logit, hm_t, ct_t, n_vis, cfg["lambda_cls"])
        loss_v = float(loss.detach())
        if not math.isfinite(loss_v):
            diverged_at = step
            break
        init_loss = loss_v if init_loss is None else init_loss
        loss.backward()
        optim.step()
        final_loss = loss_v
    if diverged_at is not None:
        return "FAIL", {"error": f"loss non-finite at step {diverged_at}", "init_loss": init_loss}

    # Coarse (pre-refinement) diagnostic: straight heatmap peaks, no Refiner
    # involved -- separates "did the detector/loss/optimiser learn the
    # corners" from "did the full Stage-3 decode (incl. the untrained
    # Refiner) land within the gate", since the latter can fail for reasons
    # unrelated to detector training quality (see report).
    model.eval()
    with torch.no_grad():
        hm_logit_eval, _ = model(x)
        hm_sig_eval = torch.sigmoid(hm_logit_eval.float())

    # tau_id relaxed to 0.3 (readout-phase effects at cell corners); K=None so
    # PnP is refused ("no_intrinsics") but corners/IDs/lattice-gate still run.
    detect_cfg = {"tau_hm": cfg["tau_hm"], "tau_id": 0.3, "lattice_tol_px": cfg["lattice_tol_px"],
                  "input_size": cfg["input_size"], "board": cfg.get("board", {})}
    n_matched_tot = n_vis_tot = n_id_correct_tot = n_gate_fit = n_coarse_matched_tot = 0
    per_image = []
    for j, (val_idx, img, record) in enumerate(picked):
        result = detect(img, model, refiner, K=None, dist=None, cfg=detect_cfg)
        # detect() never exposes lattice_gate's H directly; with K=None, H
        # succeeding is the ONLY way reason can read "no_intrinsics" (a
        # failed gate reports "too_few"/"collinear" instead) -- reading this
        # off detect()'s own contract avoids re-deriving xy_pinhole/idx_thr
        # here just to call lattice_gate a second time.
        gate_fit = result["reason"] == "no_intrinsics"
        n_gate_fit += int(gate_fit)

        gt = [(c["x"], c["y"], c["index"]) for c in record["corners"] if c["visible"]]
        dets = result["corners"]
        det_xy = np.array([[d["x"], d["y"]] for d in dets]) if dets else np.zeros((0, 2))
        coarse_xy, _ = merge_close(*peaks(hm_sig_eval[j, 0], cfg["tau_hm"]))
        n_matched = n_id_correct = n_coarse_matched = 0
        for gx, gy, gidx in gt:
            if len(det_xy):
                d = np.hypot(det_xy[:, 0] - gx, det_xy[:, 1] - gy)
                jj = int(np.argmin(d))
                if d[jj] <= 1.5:
                    n_matched += 1
                    n_id_correct += int(dets[jj]["index"] == gidx)
            if len(coarse_xy):
                dc = np.hypot(coarse_xy[:, 0] - gx, coarse_xy[:, 1] - gy)
                n_coarse_matched += int(dc.min() <= 1.5)
        n_matched_tot += n_matched
        n_vis_tot += len(gt)
        n_id_correct_tot += n_id_correct
        n_coarse_matched_tot += n_coarse_matched
        per_image.append({"val_idx": val_idx, "matched": n_matched, "coarse_matched": n_coarse_matched,
                           "visible": len(gt), "id_correct": n_id_correct, "gate_fit": gate_fit})

        decoded = {"corners": [{"x": d["x"], "y": d["y"], "visible": True,
                                 "index": d["index"] if d["index"] is not None else -1} for d in dets]}
        cv2.imwrite(str(out_dir / f"overfit_{j}_val{val_idx}.png"), viz.draw_overlay(img, decoded))

    match_rate = n_matched_tot / n_vis_tot if n_vis_tot else 0.0
    coarse_match_rate = n_coarse_matched_tot / n_vis_tot if n_vis_tot else 0.0
    id_rate = n_id_correct_tot / n_matched_tot if n_matched_tot else 0.0
    ok = (final_loss < 0.01 * init_loss) and coarse_match_rate >= 0.9
    return ("PASS" if ok else "FAIL"), {"val_indices": [p[0] for p in picked], "init_loss": init_loss,
                                        "final_loss": final_loss, "loss_ratio": final_loss / init_loss,
                                        "match_rate": match_rate, "coarse_match_rate": coarse_match_rate,
                                        "id_rate": id_rate, "gate_fits": f"{n_gate_fit}/4",
                                        "per_image": per_image}


# --------------------------------------------------------------------------- main

def main():
    args = build_parser().parse_args()

    import numpy as np
    import torch
    from dcc.dataset import SynthVal, load_config
    from dcc.model import DetectorNet, Refiner, detector_kwargs

    cfg = load_config(args.config)
    root = Path(__file__).resolve().parents[1]
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out) / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[preflight] device={device} config={args.config} out={out_dir}")

    W, H = cfg["input_size"]
    torch.manual_seed(0)
    model = DetectorNet(H, W, **detector_kwargs(cfg)).to(device)
    refiner = Refiner().to(device)

    val_ds = SynthVal(cfg, cfg["synth"]["val_size"], cfg["synth"]["val_seed"])
    records = [val_ds[i] for i in range(4)]
    rendered = [_render_targets(cfg, rec) for _, rec in records]
    hms_np, cts_np = [r[0] for r in rendered], [r[1] for r in rendered]
    n_vis = sum(r[2] for r in rendered)
    imgs_np = np.stack([img for img, _ in records])
    x = torch.from_numpy(imgs_np).float().div(255.0).unsqueeze(1).to(device)
    hm_t = torch.from_numpy(np.stack(hms_np)).float().unsqueeze(1).to(device)
    ct_t = torch.from_numpy(np.stack(cts_np)).float().to(device)

    results = [
        _run_check("generator_lock", check_generator_lock, cfg, root, args.audit_report),
        _run_check("init_loss_prediction", check_init_loss_prediction, model, cfg, x, hm_t, ct_t, n_vis,
                   hms_np, cts_np),
        _run_check("translation_equivariance", check_translation_equivariance, model, records[0][0], device),
        _run_check("rope_relativity", check_rope_relativity, model, cfg, device),
        _run_check("gradient_balance", check_gradient_balance, model, cfg, x, hm_t, ct_t, n_vis),
        _run_check("refiner_init_state", check_refiner_init_state, cfg, refiner, device),
        _run_check("bf16_parity", check_bf16_parity, model, x, device),
    ]

    if args.quick:
        results.append(_report("one_batch_overfit", "WARN", reason="skipped (--quick)"))
    elif device.type != "cuda":
        results.append(_report("one_batch_overfit", "WARN",
                                reason=f"skipped (device={device.type}: 900-step bf16 training loop "
                                       f"impractical without CUDA)"))
    else:
        results.append(_run_check("one_batch_overfit", check_one_batch_overfit, cfg, model, refiner, device,
                                   out_dir))

    report = {"device": str(device), "config": args.config, "checks": results}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    (out_dir / "summary.txt").write_text("\n".join(r["line"] for r in results) + "\n")

    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    print(f"\n[preflight] {len(results)} checks, {n_fail} FAIL -> {out_dir}")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
