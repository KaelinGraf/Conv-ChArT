"""tools/robustness_sweep.py -- OUR-MODEL-ONLY robustness curves.

For each factor and each step value along it, generate N independent samples
with the swept factor PINNED and everything else (background, pose, all other
augmentations) randomised, then score the detector twice per frame: once with
the refiner and once coarse-only. That marginalises the nuisance parameters,
so each point is an accuracy estimate with a real confidence interval rather
than the single deterministic anecdote tools/factor_sweep.py renders.

Deliberately NOT tools/factor_sweep.py: that tool holds ONE base scene fixed
and varies one parameter for a visually-comparable filmstrip (n=1 frame, <=16
corners per point -- binomial SE ~2.5pp at our accuracy, useless as a curve).
This one trades the visual A/B for statistical power. Both are wanted; they
answer different questions.

Geometry is pinned exactly through generate_sample's `components` override;
range-valued photometric knobs are pinned by collapsing [lo,hi] -> [v,v] and
forcing that augmentation's probability to 1.0; the two blur families are
applied here, post-generation, because their kernel size is drawn from a set
rather than a range and cannot be pinned through config alone.

No third-party or baseline code -- this is our own model only, so unlike the
comparison tooling it belongs in the tracked tree.
"""
import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MATCH_PX = 4.0        # MT-03 capture radius; same value train_detector validates at
N_DEFAULT = 100


def _deg(v):
    import numpy as np
    return float(np.radians(v))


# kind: "component" -> generate_sample(components=...); "range" -> pin cfg [lo,hi]=[v,v]
# and force `prob` to 1.0; "blur" -> applied post-generation here.
FACTORS = {
    "distance":        {"kind": "component", "key": "s",     "unit": "board square s (px)",
                        "steps": [12, 16, 24, 32, 48, 64, 96, 128]},
    # OUTSIDE the trained envelope, deliberately. scale_range_px is [12, 128], and
    # `distance` above samples exactly that range -- so it measures interpolation, never
    # extrapolation. This factor brackets it on BOTH sides (6-10 px below the floor,
    # 160-256 px above the ceiling) with in-envelope anchors kept for continuity, so the
    # curve shows where generalisation actually ends rather than stopping at the edge of
    # what was trained. Separate factor rather than wider `distance` steps ON PURPOSE:
    # the 4-way comparison's distance data is already banked against Deep ChArUco and
    # classical at the original steps, and widening it would silently invalidate them.
    # s is pinned via components, which bypasses scale_range_px entirely, so any value is
    # reachable. Expect partial views at the top end (5*256 = 1280 px board in a 640 px
    # frame) -- legal, and the visible-corner denominator shrinks accordingly.
    "distance_extrap": {"kind": "component", "key": "s",     "unit": "board square s (px), TRAINED 12-128",
                        "steps": [6, 8, 10, 12, 16, 64, 128, 160, 192, 256]},
    "tilt":            {"kind": "component", "key": "tilt",  "unit": "out-of-plane tilt (deg)",
                        "steps": [0, 10, 20, 30, 40, 50, 60], "xform": _deg, "base_s": 40.0},
    "rotation":        {"kind": "component", "key": "theta", "unit": "in-plane rotation (deg)",
                        "steps": [0, 30, 60, 90, 120, 150, 180], "xform": _deg, "base_s": 40.0},

    "sensor_noise_K":  {"kind": "range", "key": "sensor_noise_electrons_per_dn", "prob": "gauss_noise_p",
                        "unit": "sensor gain K (e-/DN, lower = noisier)", "invert_x": True,
                        # Extended BELOW K=1 (Kaelin 2026-07-29: the SNR axis stopped at 21 dB
                        # and "does snr not start at 0db (or at least a bit of a lower floor)").
                        # K < 1 e-/DN is the high-analog-gain / low-light regime -- the ADC step
                        # is finer than the electron noise -- and since the shot variance in DN is
                        # S/K, driving K down is the only way to reach single-digit dB. K=0.02
                        # lands near 4.5 dB at the measured board level. See tools/snr_calibration.py.
                        "steps": [30, 16, 8, 4, 2, 1, 0.5, 0.25, 0.1, 0.05, 0.02]},
    "brightness":      {"kind": "range", "key": "brightness_range", "prob": "brightness_p",
                        "unit": "brightness offset", "steps": [-0.9, -0.7, -0.5, -0.25, 0.0, 0.35]},
    "contrast":        {"kind": "range", "key": "contrast_range", "prob": "contrast_p",
                        "unit": "contrast scale", "steps": [0.6, 0.8, 1.0, 1.2, 1.4]},
    "ink_contrast":    {"kind": "range", "key": "ink_contrast_scale", "prob": "ink_contrast_p",
                        "unit": "NIR ink contrast scale", "steps": [1.0, 1.15, 1.3, 1.45, 1.6]},
    "vignette":        {"kind": "range", "key": "vignette_strength", "prob": "vignette_p",
                        "unit": "vignette strength", "steps": [0.0, 0.05, 0.12, 0.2, 0.25]},
    "specular":        {"kind": "range", "key": "specular_strength", "prob": "specular_p",
                        "unit": "specular lobe strength (DN)", "steps": [0, 60, 120, 180, 220]},
    "droplets":        {"kind": "range", "key": "droplet_n", "prob": "droplet_p", "integer": True,
                        "unit": "adherent droplets (n)", "steps": [0, 1, 2, 4, 6]},

    # LOW LIGHT as a pure multiplicative rescale, matching Deep ChArUco Fig. 10's protocol
    # (factor 0.6^k) and matching tools/filmstrip_lighting.py exactly, so the curve
    # QUANTIFIES what that filmstrip shows rather than measuring a different thing.
    # NOT the same as `brightness`: that factor bottoms out at 0.1x (b = -0.9) and flips a
    # coin between multiplicative img*(1+b) and additive img + b*255, so it mixes two
    # mechanisms and stops four stops before anything interesting happens. This one goes to
    # 0.010x, where the whole image spans 3 DN and the board's contrast is 1 DN -- the
    # quantisation floor. Applied post-generation to the FINISHED frame, sensor noise
    # included, exactly as the reference does to a captured image.
    "darkness":        {"kind": "scale", "unit": "brightness rescale factor", "invert_x": True,
                        "steps": [1.0, 0.6, 0.36, 0.216, 0.13, 0.078, 0.047, 0.028, 0.017, 0.010]},

    "defocus_blur":    {"kind": "blur", "mode": "gauss", "unit": "Gaussian blur kernel (px)",
                        "steps": [0, 3, 5, 7, 9]},
    "motion_blur":     {"kind": "blur", "mode": "motion", "unit": "motion blur kernel (px)",
                        "steps": [0, 3, 5, 7, 9]},

    "diff_ambient":    {"kind": "range", "key": "differencing_ambient", "prob": "differencing_p",
                        "unit": "daylight pedestal (ambient)", "steps": [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]},
    "diff_ratio":      {"kind": "range", "key": "differencing_illum_peak", "prob": "differencing_p",
                        "unit": "LED peak (vs ambient 0.45)", "steps": [0.9, 0.6, 0.4, 0.25, 0.15],
                        "also": {"differencing_ambient": [0.45, 0.45]}},
    "diff_ghosting":   {"kind": "range", "key": "differencing_shift_px", "prob": "differencing_p",
                        "unit": "inter-frame shift (px)", "steps": [0.0, 0.5, 1.0, 2.0, 3.0]},

    # OCCLUSION (Kaelin 2026-07-29): his hypothesis for the attention bottleneck is that it
    # helps at FAR, VERY CLOSE and OCCLUDED -- the regimes where local evidence alone is
    # insufficient and board-scale context should pay. `distance` already spans far/close;
    # these two close the occlusion gap. They were in the old tools/factor_sweep.py and were
    # NOT carried across when this tool was rewritten for statistical power.
    "occlusion":       {"kind": "synth_range", "path": "occlusion.holes", "prob": "occlusion.p",
                        "integer": True, "unit": "rectangular occluders (n)",
                        "steps": [0, 2, 4, 6, 8, 10]},
    "object_occlusion": {"kind": "synth_range", "path": "cutouts.max_objects", "prob": "cutouts.p",
                         "integer": True, "unit": "real object occluders (n, SAM2 bank)",
                         "steps": [0, 1, 2, 3, 4]},
    # BOARD-AREA OCCLUSION (Kaelin, 2026-07-30). Replaces `occlusion`/`object_occlusion`,
    # whose axes were COUNTS -- a step from 1 to 2 occluders could mean 2% or 40% of the
    # board depending on its apparent scale, so the axis was uninterpretable and, measured
    # over all 19 factors, among the least informative (recall moved 3.6%).
    #
    # The x-value is a TARGET fraction; the achieved fraction is measured per frame and the
    # mean is reported alongside, because an occluder clipped by the frame edge covers less
    # than intended and pretending otherwise would put a wrong number on the axis.
    "board_occlusion": {"kind": "board_occlusion", "unit": "board area occluded (%)",
                        "steps": [0, 10, 20, 30, 40, 50]},
}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/rev640.yaml")
    p.add_argument("--ckpt", default="runs/rev640_baseline/ckpt_0150000.pt")
    p.add_argument("--refiner-ckpt", default="runs/refiner_rev4_ft/ckpt_0013000.pt")
    p.add_argument("--factors", nargs="+", default=list(FACTORS), choices=list(FACTORS))
    p.add_argument("--n", type=int, default=N_DEFAULT, help="frames per step")
    p.add_argument("--seed", type=int, default=20260728)
    p.add_argument("--out", default="paper/results_rev4/robustness")
    return p


def _load_match_greedy():
    """train_detector.py's greedy matcher is the canonical one and lives in a
    TRACKED file -- eval_classical's equivalents are gitignored, so importing
    those would make this module depend on code absent from the public tree."""
    spec = importlib.util.spec_from_file_location(
        "_td_match", str(Path(__file__).resolve().parents[1] / "tools" / "train_detector.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._match_greedy


def apply_factor(cfg, name, value):
    """-> (cfg_step, components, post_fn). Only the swept factor is pinned;
    every other augmentation keeps its configured probability and range."""
    import cv2
    import numpy as np

    spec = FACTORS[name]
    cfg_s = copy.deepcopy(cfg)
    comp, post = {}, None
    v = spec.get("xform", float)(value) if spec["kind"] == "component" else value

    if spec["kind"] == "component":
        comp[spec["key"]] = v
        if "base_s" in spec:
            comp["s"] = spec["base_s"]     # hold apparent scale fixed so tilt/rotation
                                            # measure foreshortening, not distance in disguise
    elif spec["kind"] == "range":
        ph = cfg_s["synth"]["photometric"]
        ph[spec["key"]] = [int(v), int(v)] if spec.get("integer") else [float(v), float(v)]
        ph[spec["prob"]] = 1.0
        for k, val in spec.get("also", {}).items():
            ph[k] = val
    elif spec["kind"] == "synth_range":
        # dotted path under cfg["synth"] (e.g. "occlusion.holes"); the swept knob is pinned to
        # [v, v] and its own probability forced to 1.0, same discipline as "range" above.
        node = cfg_s["synth"]
        *parents, leaf = spec["path"].split(".")
        for k in parents:
            node = node[k]
        # write back the SAME TYPE the config already uses: occlusion.holes is a
        # [lo,hi] range, cutouts.max_objects is a bare int that dcc.synth feeds
        # straight into range(). Writing a list into the latter raises TypeError.
        cur = node[leaf]
        if isinstance(cur, (list, tuple)):
            node[leaf] = [int(v), int(v)] if spec.get("integer") else [float(v), float(v)]
        else:
            node[leaf] = int(v) if spec.get("integer") else float(v)
        pnode = cfg_s["synth"]
        *pparents, pleaf = spec["prob"].split(".")
        for k in pparents:
            pnode = pnode[k]
        pnode[pleaf] = 0.0 if v == 0 else 1.0     # n=0 means "no occluders", not "one of size 0"
    elif spec["kind"] == "scale":
        # multiplicative rescale of the finished frame, rounded and clipped like a real
        # sensor read -- the quantisation floor IS the failure mechanism at the dark end
        post = lambda g, f=float(v): np.clip(np.rint(g.astype(np.float64) * f), 0, 255).astype(np.uint8)
    elif spec["kind"] == "blur":
        ph = cfg_s["synth"]["photometric"]
        ph["gauss_blur_p"] = ph["motion_blur_p"] = 0.0   # own the blur entirely
        k = int(v)
        if k >= 3:
            if spec["mode"] == "gauss":
                post = lambda g, k=k: cv2.GaussianBlur(g, (k, k), 0)
            else:
                ker = np.zeros((k, k), np.float32)
                ker[k // 2, :] = 1.0 / k                  # horizontal line kernel, fixed direction
                post = lambda g, ker=ker: cv2.filter2D(g, -1, ker)
    return cfg_s, comp, post


def score_frame(gt_xy, gt_idx, result, match_greedy):
    """-> (n_gt, n_matched, n_id_correct, [errors]) at MATCH_PX."""
    import numpy as np
    det = result["corners"]
    if not det:
        return len(gt_xy), 0, 0, []
    det_xy = np.array([[c["x"], c["y"]] for c in det], dtype=np.float64)
    det_id = [c["index"] for c in det]
    matches = match_greedy(gt_xy, det_xy, MATCH_PX)
    errs = [d for _, _, d in matches]
    n_id = sum(1 for gi, di, _ in matches if det_id[di] is not None and det_id[di] == gt_idx[gi])
    return len(gt_xy), len(matches), n_id, errs


def occlude_board_fraction(gray, rec, target_frac, rng):
    """Occlude `target_frac` of the BOARD's projected area and update corner visibility to
    match the generator's own convention (dcc/synth.py: a corner inside a hole is NOT
    visible). Returns (occluded_gray, achieved_fraction).

    Without the visibility update this would measure "can the model see through paint",
    since occluded corners would stay in the ground truth and count as misses. The
    generator gates visibility on occlusion, so this must too, or the two occlusion paths
    would mean different things.

    BOARD POLYGON: rec carries the 16 INNER corners, whose hull spans only the middle 3x3
    of the 5x5 board -- 9/25 of its area. The hull is therefore scaled 5/3 about its
    centroid to recover the full board quad. Exact under affine, approximate under strong
    perspective; stated rather than hidden, and identical at every step so the axis stays
    monotonic.
    """
    import cv2
    import numpy as np
    pts = np.array([[c["x"], c["y"]] for c in rec["corners"]], dtype=np.float32)
    if len(pts) < 3 or target_frac <= 0:
        return gray, 0.0
    hull = cv2.convexHull(pts)
    cen = hull.reshape(-1, 2).mean(axis=0)
    board = ((hull.reshape(-1, 2) - cen) * (5.0 / 3.0) + cen).astype(np.float32)
    H, W = gray.shape
    bmask = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(bmask, board.astype(np.int32), 1)
    board_area = int(bmask.sum())
    if board_area == 0:
        return gray, 0.0

    # Grow ONE rectangle from a random board-interior anchor until it covers the target
    # fraction OF THE BOARD (not of the frame). A single blob is the honest hard case: k
    # scattered small holes of the same total area leave every corner's neighbourhood
    # partly visible, which is much easier than one contiguous occluder.
    occ = np.zeros((H, W), np.uint8)
    ys, xs = np.nonzero(bmask)
    j = int(rng.integers(len(xs)))
    cx, cy = int(xs[j]), int(ys[j])
    side = 2.0
    for _ in range(200):
        occ[:] = 0
        h = int(side); w = int(side * 1.35)      # mild aspect, so it is not a perfect square
        cv2.rectangle(occ, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), 1, -1)
        if (occ & bmask).sum() / board_area >= target_frac:
            break
        side *= 1.12
    achieved = float((occ & bmask).sum()) / board_area

    fill = int(rng.integers(0, 256))
    gray = gray.copy()
    gray[occ.astype(bool)] = fill
    for c in rec["corners"]:
        xi, yi = int(round(c["x"])), int(round(c["y"]))
        if 0 <= xi < W and 0 <= yi < H and occ[yi, xi]:
            c["visible"] = False
    return gray, achieved


def run_factor(name, cfg, model, refiner, bg_files, n, seed, match_greedy, device):
    import numpy as np
    from dcc.pipeline import detect
    from dcc.synth import generate_sample

    spec, out = FACTORS[name], []
    for si, value in enumerate(spec["steps"]):
        cfg_s, comp, post = apply_factor(cfg, name, value)
        acc = {k: {"n_gt": 0, "n_match": 0, "n_id": 0, "errs": []} for k in ("refined", "coarse")}
        achieved = []          # board_occlusion: per-frame ACHIEVED fraction, mean reported
        for i in range(n):
            rng = np.random.default_rng([seed, si, i])
            # components=comp is LOAD-BEARING: it is what actually pins the swept
            # geometry. Omitting it (bug, 2026-07-29) silently made distance/tilt/
            # rotation draw random poses, so each 'curve' was N repeats of the same
            # distribution and looked flat -- which is exactly what a robustness curve
            # is supposed to look like when the model IS robust. Failed invisibly.
            rec, _meta = generate_sample(cfg_s, rng, bg_files,
                                          components=comp or None, force_negative=False)
            gray = rec["image"]
            if spec["kind"] == "board_occlusion":
                gray, frac = occlude_board_fraction(gray, rec, value / 100.0, rng)
                achieved.append(frac)
            if post is not None:
                gray = post(gray)
            vis = [c for c in rec["corners"] if c["visible"]]
            if not vis:
                continue
            gt_xy = np.array([[c["x"], c["y"]] for c in vis], dtype=np.float64)
            gt_idx = [c["index"] for c in vis]
            for arm, rf in (("refined", refiner), ("coarse", None)):
                res = detect(gray, model, rf, K=None, dist=None, cfg=cfg_s)
                ngt, nm, nid, errs = score_frame(gt_xy, gt_idx, res, match_greedy)
                a = acc[arm]
                a["n_gt"] += ngt; a["n_match"] += nm; a["n_id"] += nid; a["errs"] += errs
        row = {"value": value, "arms": {}}
        if achieved:
            row["achieved_frac_mean"] = float(sum(achieved) / len(achieved))
        for arm, a in acc.items():
            e = np.array(a["errs"]) if a["errs"] else np.array([0.0])
            row["arms"][arm] = {
                "n_gt": a["n_gt"], "n_match": a["n_match"], "n_id": a["n_id"],
                "recall": a["n_match"] / max(a["n_gt"], 1),
                "id_acc": a["n_id"] / max(a["n_match"], 1),
                "err_median": float(np.median(e)), "err_mean": float(e.mean()),
                "err_p95": float(np.percentile(e, 95)),
                # Wilson-free normal-approx SE on the two rates, for the error bars.
                "recall_se": float(np.sqrt(max(a["n_match"] / max(a["n_gt"], 1) * (1 - a["n_match"] / max(a["n_gt"], 1)), 0) / max(a["n_gt"], 1))),
                "id_acc_se": float(np.sqrt(max(a["n_id"] / max(a["n_match"], 1) * (1 - a["n_id"] / max(a["n_match"], 1)), 0) / max(a["n_match"], 1))),
                "errs": [float(x) for x in a["errs"]],   # raw, so figures can be rebuilt without rerunning
            }
        out.append(row)
        r, c = row["arms"]["refined"], row["arms"]["coarse"]
        print(f"  {name:>15} = {value:>7}: recall {r['recall']*100:5.1f}%  id {r['id_acc']*100:5.1f}%  "
              f"err_med refined {r['err_median']:.4f} / coarse {c['err_median']:.4f} px  (n_gt={r['n_gt']})",
              flush=True)
    return out


def main():
    args = build_parser().parse_args()
    import cv2
    import torch
    # Several of these run concurrently (one process per factor group) beside a
    # refiner training job; unbounded cv2/torch thread pools oversubscribe 24
    # cores badly and each process ends up slower than if it ran alone.
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    from dcc.board import n_corners
    from dcc.dataset import load_config
    from dcc.model import DetectorNet, Refiner, detector_kwargs
    from dcc.synth import list_backgrounds

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W, H = cfg["input_size"]

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = DetectorNet(H, W, n_cls=n_corners(cfg.get("board")), **detector_kwargs(cfg)).to(device).eval()
    model.load_state_dict(ck.get("ema", ck.get("model")))
    rk = torch.load(args.refiner_ckpt, map_location="cpu", weights_only=False)
    refiner = Refiner().to(device).eval()
    refiner.load_state_dict(rk.get("ema", rk.get("model")))

    bg_files = list_backgrounds(cfg["synth"]["backgrounds"])
    match_greedy = _load_match_greedy()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    prov = {"ckpt": args.ckpt, "ckpt_step": ck.get("step"), "refiner_ckpt": args.refiner_ckpt,
            "refiner_step": rk.get("step"), "config": args.config, "n_per_step": args.n,
            "seed": args.seed, "match_px": MATCH_PX, "n_backgrounds": len(bg_files)}
    for name in args.factors:
        print(f"[{name}] {FACTORS[name]['unit']}", flush=True)
        rows = run_factor(name, cfg, model, refiner, bg_files, args.n, args.seed, match_greedy, device)
        with open(out_dir / f"{name}.json", "w") as f:
            json.dump({"factor": name, "unit": FACTORS[name]["unit"],
                       "invert_x": FACTORS[name].get("invert_x", False),
                       "provenance": prov, "steps": rows}, f, indent=2)
        print(f"  -> {out_dir / (name + '.json')}", flush=True)


if __name__ == "__main__":
    main()
