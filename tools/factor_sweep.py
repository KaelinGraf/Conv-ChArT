"""tools/factor_sweep.py -- deterministic one-factor-at-a-time evaluation:
a single fixed base scene, one swept factor stepping through a progressive
range while EVERYTHING else stays bit-identical, all three systems (ours /
zero-shot Deep ChArUco / classical) scored per frame. Paper deliverable: the
per-augmentation-effect figures and the three-way side-by-side filmstrip.

Determinism architecture (the hard requirement): generate_sample's own
sequential rng is NOT used for the swept factor -- draw-order coupling would
change unrelated draws between steps. Instead the scene is built ONCE via
dcc.synth's own compositing primitives (imported directly -- the established
project pattern, see tools/gen_eval_pose.py's identical import of
_prep_background/_apply_occlusion/_apply_photometric/visible), with EVERY
affine+perspective degree of freedom pinned through the `components` override
surface dcc.synth.generate_sample itself documents. A pinned component means
the underlying rng draw still happens (dcc.synth's own draw-order-invariance
contract) but its VALUE is discarded in favour of the override, so the
composited geometry is bit-identical regardless of what rng state or object
feeds it. Two scene classes:
  - GEOMETRIC axes (distance/tilt/rotation): the swept DoF (s/tilt/theta)
    changes per step, so _composite_board is called once per step -- the
    geometry itself IS the sweep.
  - PHOTOMETRIC/OCCLUSION axes: the composite is built ONCE at the fixed base
    pose and cached; each step re-applies dcc.synth._apply_photometric to a
    COPY of that same array with a config that zeroes every *_p key except
    the swept effect's own (forced to 1.0) and pins every OTHER numeric range
    that effect owns to a single point too (only the swept parameter varies)
    -- the same isolation technique tools/eval_classical.py's build_tier_cfgs
    already uses and re-exports here. A fresh identically-seeded rng is
    created for every step of a photometric axis (not advanced across steps)
    so any remaining per-effect draws (e.g. a droplet's position) land
    IDENTICALLY at every step -- consecutive droplet-count steps are
    literal prefixes of each other, not independent random layouts. Blur and
    occlusion aren't reachable this way (motion_blur_kmax doesn't map
    1:1 to a single kernel size once >3; occlusion needs an explicit
    board-relative fraction, not dcc.synth's own random-count/size/position
    holes) -- both get small, documented, faithful-to-source inline
    implementations instead (_apply_fixed_motion_blur, _apply_fractional_occlusion).

The three systems per frame:
  1. ours: dcc.pipeline.detect with the Conv-ChArT DetectorNet+Refiner
     (--ckpt/--refiner-ckpt; the checkpoint's OWN saved cfg is used for model
     construction AND scene generation by default -- see build_cfg -- so
     there is no way for a --config mismatch to silently invalidate a run).
  2. zero-shot Deep ChArUco: tools/charuconet.py's dcModel + JunkyByte/
     deepcharuco's published checkpoint (--dc-ckpt), imported directly (this
     script lives in tools/ and is invoked as a script, so `from charuconet
     import ...` resolves to the sibling file, not the detectron2 'tools'
     package shadow -- same pattern tools/train_charuconet.py already uses).
  3. classical: tools/eval_classical.py's own ID'd ChArUco arm, imported the
     same direct way (`from eval_classical import ...`).

GPU coordination (hard rule, Kaelin 2026-07-28): single-process generation
only, GPU use restricted to B=1 eval passes, and every GPU batch is gated on
check_trainer_health() (runs/rev640_baseline/metrics.jsonl mtime < 3 min,
samples_per_s >= 100) -- see main()'s per-axis gate call before any
ours/zero-shot forward pass.
"""
import argparse
import os
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch

from dcc.board import render_board
from dcc.pipeline import detect as dcc_detect
from dcc.synth import _apply_photometric, _composite_board, _prep_background, _warp_mask, list_backgrounds, visible
from dcc.viz import draw_overlay, tile

# tools/*.py is invoked as a script (its own directory is on sys.path[0] by
# Python's normal "script dir" rule) -- `import tools.X` is shadowed
# machine-wide by an unrelated detectron2 package, but a bare top-level
# `from charuconet import ...` is not, exactly like train_charuconet.py.
from charuconet import IN_H, IN_W, dcModel, decode_cells, load_reference_state_dict, pre_bgr_normalize
import eval_classical as ec

# The health gate below guards a LIVE trainer's loader from GPU/generator
# contention. It must point at whichever run is actually training NOW -- with a
# hard-coded path it silently becomes a permanent 300s-then-RuntimeError block
# the moment that run ends (found 2026-07-28 after rev640_baseline finished at
# step 190k and every eval tool inherited a dead gate). Env override lets the
# caller retarget it; empty string disables the gate entirely, which is correct
# when NO trainer is running.
TRAINER_METRICS = os.environ.get("DCC_TRAINER_METRICS", "runs/rev640_baseline/metrics.jsonl")
TRAINER_MAX_AGE_S = 180
TRAINER_MIN_SPS = 100.0
DC_PRETRAINED_DEFAULT = "/home/kaelin/p4p/external/deepcharuco/src/reference/longrun-epoch=99-step=369700.ckpt"
MATCH_TOL_PX = 4.0

BASE_POSE_DEFAULT = {"s": 40.0, "theta": 0.0, "shear_x": 0.0, "shear_y": 0.0,
                      "tx": 0.0, "ty": 0.0, "tilt": 0.0, "psi": 0.0, "fov_scale": 1.0}


# --------------------------------------------------------------- base scene

def build_cfg(ckpt_path, config_override=None):
    """The checkpoint's OWN saved cfg (input_size/board/attend_div/... it
    was trained under) is the single source of truth for both model
    construction and scene generation, unless --config explicitly overrides
    it -- see module docstring. Returns (cfg, ckpt_dict)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if config_override:
        import yaml
        with open(config_override) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = ckpt["cfg"]
    return cfg, ckpt


def build_base_scene(cfg, bg_seed):
    """ONE deterministic background crop, drawn once and never redrawn --
    every axis (geometric or not) composites against this SAME array.
    Returns (bg_crop float32 BGR, w2, h2)."""
    W, H = cfg["input_size"]
    bg_files = list_backgrounds(cfg["synth"]["backgrounds"])
    rng = np.random.default_rng([bg_seed])
    while True:
        idx = int(rng.integers(len(bg_files)))
        bg = cv2.imread(bg_files[idx], cv2.IMREAD_COLOR)
        if bg is not None:
            break
    bg_crop = _prep_background(bg, rng, cfg["synth"], W, H)
    return bg_crop, W, H


def composite_at_pose(cfg, bg_crop, pose, w2, h2):
    """(work float32 BGR, p_img (16,2), board_mask, board_centroid, H) at a
    FULLY pinned `pose` dict (every one of s/theta/shear_x/shear_y/tx/ty/
    tilt/psi/fov_scale present) -- bit-identical regardless of the rng
    passed to _composite_board, since dcc.synth's `components` override
    covers every draw that call makes (see module docstring). H is the
    board->canvas homography, needed by the differencing axes' white/black
    module-contrast covariate (see _module_contrast)."""
    dummy_rng = np.random.default_rng(0)  # never actually consulted: every draw is overridden
    work, p_img, H, _comps = _composite_board(bg_crop, dummy_rng, cfg, w2, h2, None, 1, dict(pose))
    board_mask = _warp_mask(H, cfg["synth"]["render_res"], w2, h2)
    board_centroid = tuple(p_img.mean(axis=0))
    return work, p_img, board_mask, board_centroid, H


def gt_from_p_img(p_img, holes, w2, h2):
    """{index: (x, y)} for corners visible under `holes` -- dcc.synth.visible
    is the SAME geometric visibility test generate_sample itself uses."""
    return {k: (float(x), float(y)) for k, (x, y) in enumerate(p_img) if visible((x, y), holes, (w2, h2))}


def to_gray(work):
    return cv2.cvtColor(np.clip(work, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)


# ------------------------------------------------------- photometric axes

def _photometric_frame(cfg, work0, board_mask, board_centroid, w2, h2, ph_overrides, axis_seed):
    """Apply _apply_photometric to a COPY of the cached base composite: every
    *_p key zeroed except what ph_overrides turns on, a FRESH rng seeded
    identically for every step of the same axis (axis_seed) so any remaining
    per-effect draws (e.g. a droplet's own position) land identically at
    every step of that axis -- see module docstring."""
    base_ph = cfg["synth"]["photometric"]
    ph = ec._zero_all_probabilities(base_ph)
    ph.update(ph_overrides)
    rng = np.random.default_rng([axis_seed])
    return _apply_photometric(work0.copy(), rng, ph, w2, h2, board_mask=board_mask, board_centroid=board_centroid)


def _apply_fixed_motion_blur(work, kernel):
    """Faithful inline copy of dcc.synth._apply_photometric's motion-blur
    kernel construction (module docstring's documented exception: kmax
    doesn't map 1:1 to one kernel size once >3, so the *_p override trick
    can't force an exact k) -- SAME math, fixed kernel size and a fixed
    45-degree angle (not drawn) so only the kernel size varies across this
    axis's steps."""
    if kernel is None or kernel <= 1:
        return work
    k = int(kernel)
    angle = np.pi / 4.0
    c = k // 2
    dx, dy = np.cos(angle) * c, np.sin(angle) * c
    kern = np.zeros((k, k), dtype=np.float32)
    cv2.line(kern, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1.0, 1)
    kern /= kern.sum()
    return cv2.filter2D(work, -1, kern)


def _apply_fractional_occlusion(work, p_img, fraction, w2, h2):
    """Deterministic rect occluder covering `fraction` of the board's own
    bounding-box height, growing from the top edge inward -- NOT
    dcc.synth._apply_occlusion (that's random count/size/position; this
    sweep needs an explicit, board-relative, monotonically-nesting hole so
    consecutive steps are supersets of each other, "like the half-occlusion
    protocol"). Returns (work_occluded, holes) in the (x0,y0,w,h) format
    dcc.synth.visible() already consumes."""
    if fraction <= 0:
        return work, []
    x0b, y0b = p_img.min(axis=0)
    x1b, y1b = p_img.max(axis=0)
    hole = (float(x0b), float(y0b), float(x1b - x0b), float(fraction * (y1b - y0b)))
    out = work.copy()
    xi0, yi0 = max(int(round(hole[0])), 0), max(int(round(hole[1])), 0)
    xi1 = min(int(round(hole[0] + hole[2])), w2)
    yi1 = min(int(round(hole[1] + hole[3])), h2)
    out[yi0:yi1, xi0:xi1] = 128.0
    return out, [hole]


# -------------------------------------------------------------- axis table

def _geomspace_steps(lo, hi, n):
    return [(f"{v:.1f}", float(v)) for v in np.geomspace(lo, hi, n)]


def _linspace_steps(lo, hi, n, fmt="{:.1f}"):
    return [(fmt.format(v), float(v)) for v in np.linspace(lo, hi, n)]


def _board_saturation_frac(work0, board_mask, p_img, ambient, peak, floor, sigma_frac, w2, h2):
    """Fraction of board-region pixels whose (pre-noise) lit-frame value
    I_lit = work0*(ambient+illum_lobe) would clip at the sensor's 255
    ceiling -- computed exactly like dcc.synth._apply_photometric's own
    i_lit formula (including the real per-pixel Gaussian illumination lobe,
    not a uniform peak -- fixed 2026-07-28 alongside the same bug in
    tools/inversion_audit.py, see _module_contrast's docstring), restricted
    to board_mask > 0.5. Reported as an explicit per-step co-variate on
    every differencing axis (team-lead 2026-07-28 gate-4 follow-up): this is
    the confound that produced a false "non-monotonic model" reading on the
    differencing_ratio axis before it was measured directly and fixed -- a
    reader of the figure should be able to see it, not have to trust that it
    was controlled for."""
    mask_bool = board_mask > 0.5
    if not mask_bool.any():
        return 0.0
    cxb, cyb = p_img.mean(axis=0)
    sigma = sigma_frac * min(w2, h2)
    ys_idx, xs_idx = np.nonzero(mask_bool)
    illum_lobe = floor + (peak - floor) * np.exp(-0.5 * ((xs_idx - cxb) ** 2 + (ys_idx - cyb) ** 2) / sigma ** 2)
    i_lit_board = work0[mask_bool] * (ambient + illum_lobe)[:, None]
    return float((i_lit_board >= 255).mean())


_INVERSION_AUDIT_MODULE = None


def _module_contrast(work0, H, p_img, w2, h2, cfg, ambient, peak, floor, sigma_frac, erode_px=5):
    """(module_contrast, inverted) -- the mechanism measurement from the
    2026-07-28 differencing_ambient investigation (team-lead priority 3):
    unlike _board_saturation_frac (EXTENT of clipping), this measures the
    DEPTH -- the actual differenced value of white vs black board regions --
    which is what determines whether marker bit-decoding fails gracefully
    (contrast shrinks) or catastrophically (contrast goes negative, white
    reads darker than black). Uses the REAL per-pixel Gaussian illumination
    lobe (floor/sigma_frac must be the SAME pinned values render_step's own
    overrides dict used for this frame, so the covariate matches what was
    actually rendered -- an earlier version used a uniform peak across the
    whole frame, caught reviewing the team-lead's corroborating prior
    estimate for tools/inversion_audit.py's inverted-fraction measurement,
    which had the identical bug). Reuses tools/inversion_audit.py's own
    white/black mask builder (imported lazily, script-directory sys.path
    trick, same pattern as `import eval_classical as ec` -- see module
    docstring) rather than duplicating it.

    rev-4 fix (rev4-actor, 2026-07-28): this function reconstructs the
    lit/unlit pair from scratch rather than reading back what
    _apply_photometric actually rendered (deliberate -- it needs its own
    white/black masks the main pipeline doesn't expose), which means it has
    to independently mirror EVERY step of dcc/synth.py's differencing math,
    not just the ones that existed when this function was written. It was
    missing the rev-4 auto-exposure gain (dcc/synth.py:748-751) entirely,
    so it was reporting the PRE-gain (rev-3-equivalent) contrast and
    surfacing false "INVERTED" reads on the exposure_clipping/high-ambient
    steps even though the real rendered frames -- gain-normalised -- do
    not invert. Same gain formula, mirrored exactly."""
    global _INVERSION_AUDIT_MODULE
    if _INVERSION_AUDIT_MODULE is None:
        import inversion_audit as _ia
        _INVERSION_AUDIT_MODULE = _ia
    ia = _INVERSION_AUDIT_MODULE
    board_img, _corners = render_board(cfg["synth"]["render_res"], cfg.get("board"))
    white_mask, black_mask = ia._white_black_masks_canvas(board_img, H, w2, h2, erode_px)
    if white_mask.sum() < 20 or black_mask.sum() < 20:
        return None, None
    cxb, cyb = p_img.mean(axis=0)
    sigma = sigma_frac * min(w2, h2)
    ys_grid, xs_grid = np.mgrid[0:h2, 0:w2].astype(np.float64)
    illum_lobe = floor + (peak - floor) * np.exp(-0.5 * ((xs_grid - cxb) ** 2 + (ys_grid - cyb) ** 2) / sigma ** 2)
    lit_raw = work0 * (ambient + illum_lobe)[..., None]
    ph = cfg["synth"]["photometric"]
    if ph.get("differencing_autoexposure", False):
        hi = float(np.percentile(lit_raw, 99.9))
        gain = min(1.0, ph.get("differencing_ae_target", 250.0) / max(hi, 1e-6))
        lit_raw = lit_raw * gain
        ambient_eff = ambient * gain
    else:
        ambient_eff = ambient
    i_lit = np.clip(lit_raw, 0, 255)
    i_unlit = np.clip(work0 * ambient_eff, 0, 255)
    diff = np.clip(i_lit - i_unlit, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY).astype(np.float64)
    contrast = float(np.median(gray[white_mask]) - np.median(gray[black_mask]))
    return contrast, bool(contrast < 0)


AXES = {
    "distance": {"kind": "geometric", "pose_key": "s", "steps": _geomspace_steps(12, 128, 9)},
    "tilt": {"kind": "geometric", "pose_key": "tilt", "base_s": 40.0,
             "steps": _linspace_steps(0, 60, 7, "{:.0f}deg")},
    "rotation": {"kind": "geometric", "pose_key": "theta",
                 "steps": [(f"{v:.0f}deg", float(np.radians(v))) for v in np.linspace(0, 180, 7)]},
    "differencing_ambient": {"kind": "photometric",
                              "steps": [("off", None)] + [(f"{v:.2f}", v) for v in (0.1, 0.3, 0.5, 0.7, 0.9)]},
    # ambient lowered 0.85->0.45 and the ratio range capped at 1.0 (team-lead
    # 2026-07-28 gate-4 fix #2): ambient + peak now stays <= 0.9 (< 255 DN)
    # at every step, so the axis isolates LED-to-daylight ratio instead of
    # crossing the sensor's clipping ceiling partway through (see
    # _board_saturation_frac's per-step covariate, reported alongside every
    # differencing_ratio result to make this explicit rather than assumed).
    "differencing_ratio": {"kind": "photometric", "ambient": 0.45,
                            "steps": [(f"r={v:.2f}", v) for v in (1.0, 0.75, 0.5, 0.25, 0.12)]},
    # NEW axis (team-lead 2026-07-28 gate-4 fix #2): the clipping regime is
    # physically real (a real camera clips) and deployment-relevant -- given
    # its own axis, fixed ratio=1.0 (peak=ambient), sweeping the overall
    # exposure/pedestal THROUGH the clipping threshold, rather than left as
    # contamination inside the ratio axis.
    "exposure_clipping": {"kind": "photometric",
                           "steps": [(f"ambient={v:.2f}", v) for v in (0.2, 0.4, 0.6, 0.8, 1.0)]},
    "sensor_noise_K": {"kind": "photometric",
                        "steps": [("clean", None)] + [(f"K={v:g}", v) for v in (30, 8, 4, 2, 1)]},
    "droplets": {"kind": "photometric", "steps": [(f"n={v}", v) for v in (0, 1, 2, 4, 6)]},
    "specular": {"kind": "photometric",
                 "steps": [("off", None)] + [(f"s={v}", v) for v in (60, 120, 180, 240)]},
    "blur": {"kind": "manual_blur", "steps": [("none", None)] + [(f"k={v}", v) for v in (3, 5, 7, 9)]},
    "occlusion": {"kind": "occlusion", "steps": [(f"{v:.0%}", v) for v in (0.0, 0.2, 0.4, 0.6)]},
}
DIFFERENCING_AXES = ("differencing_ambient", "differencing_ratio", "exposure_clipping")


def render_step(cfg, base_scene, axis_name, base_pose, step_value, axis_seed):
    """(gray uint8, p_img, holes, covariates) for ONE step of ONE axis.
    covariates is {} except on the three differencing axes, where it carries
    {"saturation_frac": float} -- see _board_saturation_frac."""
    bg_crop, w2, h2 = base_scene
    spec = AXES[axis_name]

    if spec["kind"] == "geometric":
        pose = dict(base_pose)
        if axis_name == "tilt":
            pose["s"] = spec["base_s"]
        pose[spec["pose_key"]] = step_value
        work, p_img, _mask, _cent, _H = composite_at_pose(cfg, bg_crop, pose, w2, h2)
        return to_gray(work), p_img, [], {}

    work0, p_img, board_mask, board_centroid, H = composite_at_pose(cfg, bg_crop, base_pose, w2, h2)

    if spec["kind"] == "occlusion":
        work, holes = _apply_fractional_occlusion(work0, p_img, step_value, w2, h2)
        return to_gray(work), p_img, holes, {}

    if spec["kind"] == "manual_blur":
        work = _apply_fixed_motion_blur(work0, step_value)
        return to_gray(work), p_img, [], {}

    # photometric
    covariates = {}
    if axis_name in DIFFERENCING_AXES and step_value is None:
        return to_gray(work0), p_img, [], {}
    if axis_name == "differencing_ambient":
        ambient, peak = step_value, 0.4
    elif axis_name == "differencing_ratio":
        ambient = spec["ambient"]
        peak = step_value * ambient
    elif axis_name == "exposure_clipping":
        ambient, peak = step_value, step_value  # ratio pinned at 1.0
    if axis_name in DIFFERENCING_AXES:
        floor, sigma_frac = 0.1, 0.35  # pinned -- must match overrides below exactly
        overrides = {"differencing_p": 1.0, "differencing_ambient": [ambient, ambient],
                     "differencing_illum_peak": [peak, peak], "differencing_illum_floor": [floor, floor],
                     "differencing_illum_sigma_frac": [sigma_frac, sigma_frac], "differencing_shift_px": [1.0, 1.0]}
        covariates["saturation_frac"] = _board_saturation_frac(work0, board_mask, p_img, ambient, peak,
                                                                 floor, sigma_frac, w2, h2)
        contrast, inverted = _module_contrast(work0, H, p_img, w2, h2, cfg, ambient, peak, floor, sigma_frac)
        if contrast is not None:
            covariates["module_contrast"] = contrast
            covariates["contrast_inverted"] = inverted
    elif axis_name == "sensor_noise_K":
        if step_value is None:
            return to_gray(work0), p_img, [], {}
        overrides = {"gauss_noise_p": 1.0, "sensor_noise_enabled": True,
                     "sensor_noise_electrons_per_dn": [step_value, step_value],
                     "sensor_noise_read_std": [2.0, 2.0]}
    elif axis_name == "droplets":
        overrides = {"droplet_p": 1.0, "droplet_n": [int(step_value), int(step_value)],
                     "droplet_mode_b_p": 0.0,  # mode (a) bokeh only -- mode (b) is a resample, not a clean intensity sweep
                     "droplet_bokeh_radius": [10.0, 10.0], "droplet_bokeh_brightness": [140.0, 140.0]}
    elif axis_name == "specular":
        if step_value is None:
            return to_gray(work0), p_img, [], {}
        overrides = {"specular_p": 1.0, "specular_strength": [float(step_value), float(step_value)],
                     "specular_exponent": [80.0, 80.0], "specular_spread_frac": [0.14, 0.14]}
    else:
        raise ValueError(f"unknown axis {axis_name!r}")

    work = _photometric_frame(cfg, work0, board_mask, board_centroid, w2, h2, overrides, axis_seed)
    return to_gray(work), p_img, [], covariates


# ------------------------------------------------------------- the 3 systems

def check_trainer_health(metrics_path=None, max_age_s=TRAINER_MAX_AGE_S, min_sps=TRAINER_MIN_SPS):
    """Resolved at CALL time, not import time, so DCC_TRAINER_METRICS set after
    import still applies; "" or a missing file means "no trainer to protect"
    and the gate passes."""
    metrics_path = TRAINER_METRICS if metrics_path is None else metrics_path
    if not metrics_path or not os.path.exists(metrics_path):
        return True, 0.0, float("inf")
    """(ok, age_s, samples_per_s) off the LAST line of the live trainer's
    metrics.jsonl -- the hard gate before any GPU eval batch (Kaelin
    2026-07-28: single-process generation only, GPU use restricted to B=1
    eval passes, paused if the main trainer looks degraded)."""
    with open(metrics_path, "rb") as f:
        f.seek(0, 2)
        f.seek(max(f.tell() - 4096, 0))
        lines = [l for l in f.read().decode(errors="ignore").strip().splitlines() if l.strip()]
    # A live trainer can be mid-write on its own last line (memory.md
    # 2026-07-28 10:45's crash postmortem found exactly this: "the torn last
    # line in metrics.jsonl, died mid-write") -- try the last line, fall back
    # to the second-to-last on a parse failure rather than crashing the gate.
    for candidate in reversed(lines[-2:]):
        try:
            d = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    else:
        raise RuntimeError(f"could not parse a trainer metrics line from the tail of {metrics_path}")
    age = time.time() - d["wall"]
    sps = d.get("samples_per_s", 0.0)
    return age < max_age_s and sps >= min_sps, age, sps


def wait_for_trainer_health(max_wait_s=300, poll_s=15):
    """Blocks until check_trainer_health() passes; returns its (ok, age,
    sps) so callers don't need a separate check just for the printout."""
    t0 = time.time()
    while True:
        ok, age, sps = check_trainer_health()
        if ok:
            return ok, age, sps
        if time.time() - t0 > max_wait_s:
            raise RuntimeError(f"trainer unhealthy for >{max_wait_s}s (age={age:.0f}s sps={sps:.1f}) -- aborting GPU eval")
        print(f"  [gpu-gate] trainer degraded (age={age:.0f}s sps={sps:.1f}) -- pausing {poll_s}s")
        time.sleep(poll_s)


def build_ours(cfg, ckpt, refiner_ckpt_path, device):
    from dcc.model import DetectorNet, Refiner, detector_kwargs
    W, H = cfg["input_size"]
    model = DetectorNet(H, W, **detector_kwargs(cfg)).to(device).eval()
    sd = ckpt.get("ema", ckpt.get("model")) if "ema" in ckpt or "model" in ckpt else ckpt
    model.load_state_dict(sd)
    refiner = Refiner().to(device).eval()
    r_obj = torch.load(refiner_ckpt_path, map_location=device, weights_only=False)
    r_sd = r_obj.get("ema", r_obj.get("model")) if isinstance(r_obj, dict) and ("model" in r_obj or "ema" in r_obj) else r_obj
    refiner.load_state_dict(r_sd)
    return model, refiner


def score_ours(model, refiner, gray, cfg):
    result = dcc_detect(gray, model, refiner, K=None, dist=None, cfg=cfg)
    xy = np.array([[c["x"], c["y"]] for c in result["corners"]], dtype=np.float64).reshape(-1, 2)
    idx = np.array([c["index"] if c["index"] is not None else -1 for c in result["corners"]], dtype=int)
    return xy, idx


def build_dc(n_cls, dc_ckpt_path, device):
    model = dcModel(n_ids=n_cls).to(device).eval()
    load_reference_state_dict(dc_ckpt_path, model, map_location=device)
    return model


def score_dc(model, gray, w2, h2, device):
    img320 = cv2.resize(gray, (IN_W, IN_H), interpolation=cv2.INTER_AREA).astype(np.float32)
    x = torch.from_numpy(pre_bgr_normalize(img320)).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        loc_logits, id_logits = model(x)
    xy320, idc, _loc_conf, _id_conf = decode_cells(loc_logits[0], id_logits[0])
    # 320x240 -> native (w2, h2): inverse of the fixed-ratio half-pixel resize,
    # generalised per-axis from train_charuconet.py's fixed 640->320 case.
    xy = np.empty_like(xy320)
    xy[:, 0] = (xy320[:, 0] + 0.5) * (w2 / IN_W) - 0.5
    xy[:, 1] = (xy320[:, 1] + 0.5) * (h2 / IN_H) - 0.5
    return xy, idc


def score_classical(gray, board, dictionary, params):
    xy, ids = ec.classical_charuco_detect(gray, board, dictionary, params)
    if xy is None:
        return np.zeros((0, 2)), np.zeros((0,), dtype=int)
    return xy, ids


def score_frame(gt_visible, det_xy, det_ids, tol=MATCH_TOL_PX):
    """{n_gt, detected, id_correct, id_wrong, loc_err_mean} -- reuses
    eval_classical's own matching helpers (anonymous match for the
    ID-agnostic "detected" count, id match for identity correctness)."""
    n_detected, det_errs = ec._match_anonymous(gt_visible, det_xy, tol=tol) if len(det_xy) else (0, [])
    n_correct, n_wrong, _id_errs = ec._match_ided(gt_visible, det_xy, det_ids, tol=tol) if len(det_ids) else (0, 0, [])
    return {"n_gt": len(gt_visible), "detected": n_detected, "id_correct": n_correct, "id_wrong": n_wrong,
            "loc_err_mean": float(np.mean(det_errs)) if det_errs else None}


def _classify_dets(gt_visible, det_xy, det_ids, tol=MATCH_TOL_PX):
    """Per-detection "correct"/"wrong"/"unidentified" label, aligned with
    det_xy/det_ids -- the same correct-vs-wrong distance test
    eval_classical._match_ided uses (see its docstring), just exposed per
    detection instead of folded into an aggregate count, plus a third
    bucket for detections that never got an ID at all (index < 0, e.g.
    ours' unread corners) which _match_ided's own two-way split lumps into
    "wrong". Used only for filmstrip colouring (tools/factor_sweep.py's
    render_filmstrip) -- score_frame's own counts remain the scored metric."""
    labels = []
    for xy, idx in zip(det_xy, det_ids):
        idx = int(idx)
        if idx < 0:
            labels.append("unidentified")
            continue
        gt = gt_visible.get(idx)
        if gt is None:
            labels.append("wrong")
        else:
            d = float(np.hypot(xy[0] - gt[0], xy[1] - gt[1]))
            labels.append("correct" if d <= tol else "wrong")
    return labels


# ---------------------------------------------------------------- filmstrip

SYSTEM_ORDER = ("ours", "deepcharuco_zeroshot", "classical")
SYSTEM_LABELS = {"ours": "ours", "deepcharuco_zeroshot": "DC zero-shot", "classical": "classical"}
_GT_COLOR = (255, 255, 255)  # hollow white circle: GT position, drawn under every prediction layer
# correct/unidentified/wrong -- same green/orange/red triples as dcc.viz's own (module-private) palette
_DET_COLOR = {"correct": (0, 200, 0), "unidentified": (0, 140, 255), "wrong": (0, 0, 220)}


def render_filmstrip(frames, gts, dets, scores_per_system, out_path, covariates=None, max_sheet_w=4000):
    """Grid filmstrip (Kaelin 2026-07-28): one row per system (SYSTEM_ORDER),
    one column per axis step, so all three systems' actual PREDICTED
    detections are visible side by side rather than only their score text.
    Each cell layers dcc.viz.draw_overlay twice on the step's frame: GT
    first (hollow white, radius 5, no index text -- "where it should have
    fired"), then that system's own predictions (filled dots, radius 3,
    colour-coded green/orange/red for correct-ID/unidentified/wrong-ID via
    _classify_dets, index text on) on top. System name, step label, the
    saturation_frac/module_contrast covariates (when given) and the
    existing per-step score line are baked into every cell's pad strip (not
    just a header row) so any single row is self-describing on its own.
    cv2-only, no matplotlib (mirrors dcc/viz.py's own choice). Downscales
    the assembled sheet (never drops a step or a system) if it would
    exceed max_sheet_w px wide; returns (sheet, scale_used)."""
    n_cov_rows = max((len(c) for c in covariates), default=0) if covariates else 0
    n_cov_rows = min(n_cov_rows, 2)  # saturation_frac + module_contrast (contrast_inverted rides along with it)
    pad_h = 20 + 18 * (3 + n_cov_rows)  # sysname + step label + score line + covariates
    panels = []
    for sysname in SYSTEM_ORDER:
        for i, (gray, label) in enumerate(frames):
            gt_record = {"corners": [{"x": x, "y": y, "index": idx} for idx, (x, y) in gts[i].items()]}
            # dark ring first, white ring just inside it (team-lead 2026-07-28:
            # a plain white hollow circle washed out against the light board
            # squares, where GT visibility matters most) -- reads as a
            # white-with-dark-outline marker against both dark and light ground.
            img = draw_overlay(gray, gt_record, filled=False, radius=6, draw_indices=False, color_fn=lambda c: (0, 0, 0))
            img = draw_overlay(img, gt_record, filled=False, radius=5, draw_indices=False,
                                color_fn=lambda c: _GT_COLOR)
            xy, ids = dets[sysname][i]
            det_labels = _classify_dets(gts[i], xy, ids)
            pred_record = {"corners": [{"x": float(p[0]), "y": float(p[1]), "index": int(k), "_lbl": lbl}
                                        for p, k, lbl in zip(xy, ids, det_labels)]}
            img = draw_overlay(img, pred_record, filled=True, radius=3, draw_indices=True,
                                color_fn=lambda c: _DET_COLOR[c["_lbl"]])

            pad = np.full((pad_h, img.shape[1], 3), 255, dtype=np.uint8)
            panel = cv2.vconcat([img, pad])
            y = img.shape[0] + 16
            cv2.putText(panel, SYSTEM_LABELS[sysname], (4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(panel, label, (4, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1, cv2.LINE_AA)
            y += 14
            cov = covariates[i] if covariates else {}
            if "saturation_frac" in cov:
                y += 18
                cv2.putText(panel, f"sat_frac: {cov['saturation_frac']:.3f}", (4, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1, cv2.LINE_AA)
            if "module_contrast" in cov:
                y += 18
                inv_tag = " INVERTED" if cov.get("contrast_inverted") else ""
                cv2.putText(panel, f"contrast: {cov['module_contrast']:.1f}DN{inv_tag}", (4, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1, cv2.LINE_AA)
            r = scores_per_system[sysname][i]
            y += 18
            cv2.putText(panel, f"{r['detected']}/{r['n_gt']} ({r['id_correct']}c/{r['id_wrong']}w)", (4, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            panels.append(panel)
    sheet = tile(panels, cols=len(frames))
    scale = min(1.0, max_sheet_w / sheet.shape[1])
    if scale < 1.0:
        sheet = cv2.resize(sheet, (int(sheet.shape[1] * scale), int(sheet.shape[0] * scale)),
                            interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_path), sheet)
    return sheet, scale


# --------------------------------------------------------------------- main

def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="runs/rev640_baseline/ckpt_0150000.pt")
    p.add_argument("--refiner-ckpt", default="runs/refiner_v1/ckpt_0010000.pt")
    p.add_argument("--dc-ckpt", default=DC_PRETRAINED_DEFAULT)
    p.add_argument("--config", default=None, help="override the checkpoint's own saved cfg")
    p.add_argument("--axes", nargs="+", default=list(AXES), choices=list(AXES))
    p.add_argument("--bg-seed", type=int, default=7)
    p.add_argument("--out", default="factor_sweep_out")
    p.add_argument("--determinism-check", action="store_true", help="re-run each axis twice and hash-compare frames")
    p.add_argument("--figures-only", default=None, metavar="RESULTS_JSON",
                    help="re-render filmstrips from a previously-saved results.json's stored scores/dets, "
                         "without re-running any model -- no GPU use, no rescoring. Frames/GT are regenerated "
                         "(cheap, deterministic CPU-only scene compositing -- see module docstring) rather than "
                         "stored, to keep results.json small.")
    return p


def _render_axis_frames(cfg, base_scene, axis_name, axis_seed):
    """(frames, gts, covariates) for every step of one axis -- pure CPU
    scene compositing, fully deterministic given (cfg, base_scene,
    axis_name, axis_seed) per the module docstring's pinned-components
    architecture. Shared by the normal scoring path and --figures-only
    (which needs frames/GT back but must never touch a model or the GPU)."""
    spec = AXES[axis_name]
    frames, gts, covariates = [], [], []
    for label, step_value in spec["steps"]:
        gray, p_img, holes, cov = render_step(cfg, base_scene, axis_name, BASE_POSE_DEFAULT, step_value, axis_seed)
        frames.append((gray, label))
        gts.append(gt_from_p_img(p_img, holes, base_scene[1], base_scene[2]))
        covariates.append(cov)
    return frames, gts, covariates


def _dets_to_json(dets):
    """{sysname: [(xy ndarray, ids ndarray), ...]} -> plain-list form for
    json.dump (np.float64/np.int64 aren't JSON-serialisable)."""
    return {sysname: [{"xy": xy.tolist(), "ids": [int(i) for i in ids]} for xy, ids in per_step]
            for sysname, per_step in dets.items()}


def _dets_from_json(dets_json):
    """Inverse of _dets_to_json, back into (xy ndarray, ids ndarray) pairs."""
    return {sysname: [(np.array(d["xy"], dtype=np.float64).reshape(-1, 2), np.array(d["ids"], dtype=int))
                       for d in per_step]
            for sysname, per_step in dets_json.items()}


def main():
    args = build_parser().parse_args()
    cfg, ckpt = build_cfg(args.ckpt, args.config)
    base_scene = build_base_scene(cfg, args.bg_seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.figures_only:
        with open(args.figures_only) as f:
            saved = json.load(f)
        for axis_name, entry in saved.items():
            if "dets" not in entry:
                raise ValueError(f"{args.figures_only}: axis {axis_name!r} has no saved 'dets' "
                                  "(results.json predates the filmstrip-grid feature) -- re-run the sweep")
            axis_seed = int(hashlib.sha1(axis_name.encode()).hexdigest(), 16) % (2 ** 31)
            frames, gts, covariates = _render_axis_frames(cfg, base_scene, axis_name, axis_seed)
            dets = _dets_from_json(entry["dets"])
            sheet, scale = render_filmstrip(frames, gts, dets, entry["scores"],
                                             out_dir / f"filmstrip_{axis_name}.png", covariates)
            print(f"  {axis_name}: {len(SYSTEM_ORDER)} rows x {len(frames)} cols, scale={scale:.3f} "
                  f"-> {out_dir / f'filmstrip_{axis_name}.png'}")
        print(f"\nwrote {len(saved)} filmstrip PNGs to {out_dir} (figures-only, no model/GPU use)")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    board, dictionary, nx = ec._get_board_and_dictionary(cfg.get("board"))
    params = ec._detector_params()
    n_cls = (nx - 1) ** 2

    # Loaded ONCE, reused across every axis -- torch.load'ing three
    # checkpoints per axis was dead weight (models are stateless across
    # frames; nothing about them depends on the axis being swept).
    ok, age, sps = wait_for_trainer_health()
    print(f"trainer health before model load: ok={ok} age={age:.0f}s sps={sps:.1f}")
    model, refiner = build_ours(cfg, ckpt, args.refiner_ckpt, device)
    dc_model = build_dc(n_cls, args.dc_ckpt, device)

    results = {}
    for axis_name in args.axes:
        spec = AXES[axis_name]
        print(f"=== axis {axis_name} ({spec['kind']}) ===")
        # Python's built-in hash() is per-process-randomised for strings
        # (PYTHONHASHSEED) -- NOT safe for a determinism-gated seed (this
        # exact mistake was already caught once in eval_classical.py's own
        # tier seeding). sha1 is stable across processes/runs.
        axis_seed = int(hashlib.sha1(axis_name.encode()).hexdigest(), 16) % (2 ** 31)
        frames, gts, covariates = _render_axis_frames(cfg, base_scene, axis_name, axis_seed)
        if axis_name in DIFFERENCING_AXES:
            # the "off" step (differencing_ambient's first step) has no
            # covariates at all (differencing never fires) -- print "-"
            # rather than crash on the missing key.
            print("  saturation_frac: " + ", ".join(
                f"{c['saturation_frac']:.3f}" if "saturation_frac" in c else "-" for c in covariates))
            print("  module_contrast: " + ", ".join(
                f"{c['module_contrast']:.1f}{'*' if c.get('contrast_inverted') else ''}"
                if "module_contrast" in c else "-" for c in covariates) + "  (* = inverted)")

        if args.determinism_check:
            for label, step_value in spec["steps"]:
                gray2, _p, _h, _c = render_step(cfg, base_scene, axis_name, BASE_POSE_DEFAULT, step_value, axis_seed)
                idx = [lbl for lbl, _ in spec["steps"]].index(label)
                h1 = hashlib.sha1(frames[idx][0].tobytes()).hexdigest()
                h2_ = hashlib.sha1(gray2.tobytes()).hexdigest()
                assert h1 == h2_, f"determinism FAILED on axis {axis_name} step {label}: frame hash differs across two runs"
            print(f"  determinism check: {len(frames)}/{len(frames)} frames bit-identical across two runs")

        ok, age, sps = wait_for_trainer_health()
        print(f"  trainer health: ok={ok} age={age:.0f}s sps={sps:.1f}")

        scores = {"ours": [], "deepcharuco_zeroshot": [], "classical": []}
        dets = {"ours": [], "deepcharuco_zeroshot": [], "classical": []}
        for (gray, _label), gt in zip(frames, gts):
            xy_o, id_o = score_ours(model, refiner, gray, cfg)
            scores["ours"].append(score_frame(gt, xy_o, id_o))
            dets["ours"].append((xy_o, id_o))
            xy_dc, id_dc = score_dc(dc_model, gray, base_scene[1], base_scene[2], device)
            scores["deepcharuco_zeroshot"].append(score_frame(gt, xy_dc, id_dc))
            dets["deepcharuco_zeroshot"].append((xy_dc, id_dc))
            xy_c, id_c = score_classical(gray, board, dictionary, params)
            scores["classical"].append(score_frame(gt, xy_c, id_c))
            dets["classical"].append((xy_c, id_c))

        for i, (label, _v) in enumerate(spec["steps"]):
            row = {sysname: scores[sysname][i] for sysname in scores}
            print(f"  {label:>10}: " + " | ".join(f"{k}: {v['detected']}/{v['n_gt']} "
                  f"({v['id_correct']}c/{v['id_wrong']}w)" for k, v in row.items()))

        sheet, scale = render_filmstrip(frames, gts, dets, scores, out_dir / f"filmstrip_{axis_name}.png",
                                         covariates)
        print(f"  filmstrip: {len(SYSTEM_ORDER)} rows x {len(frames)} cols, scale={scale:.3f} "
              f"-> {out_dir / f'filmstrip_{axis_name}.png'}")
        results[axis_name] = {"steps": [s for s, _ in spec["steps"]], "scores": scores, "covariates": covariates,
                               "dets": _dets_to_json(dets)}

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_dir}/results.json and per-axis filmstrip PNGs")


if __name__ == "__main__":
    main()
