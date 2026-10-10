"""tools/video_bootstrap.py -- self-supervised fine-tuning from ordinary video of the board.

The loop (each round re-harvests with the latest model and fine-tunes from the BASE weights again,
so an early round's mistakes fall back out instead of compounding):

  harvest   run the deployed pipeline on every frame; frames whose identified corners fit the lattice
            become ANCHORS and are labelled by the fit (not by the detections); frames between anchors
            are labelled by TRACKING the board -- each frame's homography measured by aligning the
            board's known picture to that frame (dcc.video). Labels carry three states: positive,
            off-frame, and unknown (no gradient: dcc.losses mask=).
  train     tools/train_detector.py's finetune: entry with a real: block -- harvested frames mixed into
            the synthetic stream at real.frac, BatchNorm held on its statistics.
  eval      held-out clips, never trained on: identified-corner recall and ID precision against a
            reference (ground truth for synthetic clips, a base-model harvest otherwise), and the
            fraction of frames that fit, by brightness. Synthetic validation runs inside train_detector
            and must not regress (the other side of the gate).

Subcommands:
  import-onnx      deployed .onnx -> trainer .pt (needs --config for a detector; verified vs onnxruntime)
  make-clip        a synthetic test clip with ground truth (moving board, a dark stretch, an occluder)
  harvest          video(s) -> <out>/<clip>/{images.npy, records.json} + overlay sheet + manifest.json
  finetune-config  write a train_detector config for one round
  eval             score detectors on a clip
  rounds           harvest -> fine-tune -> eval, N times

Run as a script from the repo root (never `python -m tools...`, see README):
  PYTHONPATH= python tools/video_bootstrap.py <subcommand> --help
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("import-onnx", help="deployed .onnx -> trainer-format .pt")
    s.add_argument("--onnx", required=True)
    s.add_argument("--config", help="architecture config (required for a detector graph)")
    s.add_argument("--kind", choices=["detector", "refiner"], default="detector")
    s.add_argument("--out", required=True)

    s = sub.add_parser("make-clip", help="synthetic test clip with ground truth")
    s.add_argument("--config", required=True, help="supplies the board")
    s.add_argument("--out", required=True)
    s.add_argument("--frames", type=int, default=240)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--size", default="1600x1200", help="WxH sensor frame (4:3)")
    s.add_argument("--backgrounds", default=None, help="image directory; default procedural")
    s.add_argument("--dark", type=float, default=0.004, help="exposure gain at the darkest point")
    s.add_argument("--no-occluder", action="store_true")

    s = sub.add_parser("harvest", help="video(s) -> pseudo-labelled frames")
    _model_args(s)
    s.add_argument("--video", nargs="+", required=True, help="video files or image directories")
    s.add_argument("--out", required=True)
    s.add_argument("--camera", default=None, help="YAML/JSON with K (3x3) and dist, in 4:3 sensor px")
    s.add_argument("--step", type=int, default=1, help="use every step-th source frame")
    s.add_argument("--max-frames", type=int, default=None)
    s.add_argument("--stride", type=int, default=1, help="keep every stride-th labelled frame")
    s.add_argument("--cap", type=int, default=0, help="max frames per diversity bucket (0 = no cap)")
    s.add_argument("--sheet", type=int, default=24, help="frames in the overlay sheet")
    s.add_argument("--gt", nargs="*", default=None, help="ground truth per video (make-clip's gt.jsonl): "
                                                          "adds label-accuracy numbers to the manifest")

    s = sub.add_parser("finetune-config", help="write a train_detector config for one round")
    _ft_args(s)
    s.add_argument("--config", required=True, help="the base model's config (architecture + synth block)")
    s.add_argument("--from", dest="src", required=True, help="base trainer .pt (import-onnx makes one)")
    s.add_argument("--records", nargs="+", required=True, help="harvest clip directories")
    s.add_argument("--out", required=True)

    s = sub.add_parser("eval", help="score detectors on a clip")
    s.add_argument("--video", required=True)
    s.add_argument("--detector", nargs="+", required=True)
    s.add_argument("--refiner", required=True)
    s.add_argument("--config", default=None)
    s.add_argument("--gt", default=None, help="make-clip gt.jsonl")
    s.add_argument("--labels", default=None, help="a harvest clip directory used as the reference")
    s.add_argument("--camera", default=None)
    s.add_argument("--out", default=None)
    s.add_argument("--device", default=None)

    s = sub.add_parser("rounds", help="harvest -> fine-tune -> eval, repeated")
    _ft_args(s)
    s.add_argument("--config", required=True)
    s.add_argument("--base", required=True, help="base detector .pt or .onnx")
    s.add_argument("--refiner", required=True)
    s.add_argument("--train-video", nargs="+", required=True)
    s.add_argument("--heldout-video", nargs="+", required=True)
    s.add_argument("--heldout-gt", nargs="*", default=None)
    s.add_argument("--workdir", required=True)
    s.add_argument("--rounds", type=int, default=2)
    s.add_argument("--camera", default=None)
    s.add_argument("--cap", type=int, default=0)
    s.add_argument("--stride", type=int, default=1)
    return p


def _model_args(s):
    s.add_argument("--detector", required=True, help=".pt (its own cfg) or .onnx (needs --config)")
    s.add_argument("--refiner", required=True, help=".pt or .onnx")
    s.add_argument("--config", default=None)
    s.add_argument("--device", default=None)


def _ft_args(s):
    s.add_argument("--steps", type=int, default=1500)
    s.add_argument("--lr", type=float, default=1e-4)
    s.add_argument("--batch", type=int, default=8)
    s.add_argument("--accum", type=int, default=1)
    s.add_argument("--real-frac", type=float, default=0.5)
    s.add_argument("--darken-synthetic", type=float, default=0.0,
                   help="probability a synthetic replay sample is darkened with the clip's own noise model")
    s.add_argument("--backgrounds", default=None, help="image directory for the synthetic replay stream")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--val-size", type=int, default=500, help="synthetic full-val size (regression gate)")
    s.add_argument("--val-subset", type=int, default=200)
    s.add_argument("--lr-mult", nargs="*", default=["*=1.0"], help="fnmatch=mult, see trainutil.param_groups")


# ---------------------------------------------------------------------------------------------

def _device(arg):
    import torch
    return arg or ("cuda" if torch.cuda.is_available() else "cpu")


def _load_cfg(path):
    import yaml
    return yaml.safe_load(open(path)) if path else None


def _camera(path):
    if not path:
        return None, None
    import yaml
    c = yaml.safe_load(open(path))
    c = c.get("CAMERA", c)
    return c.get("K"), c.get("dist")


def load_models(det_path, ref_path, cfg, device):
    from dcc.onnx_import import load_detector, load_refiner
    det, dcfg = load_detector(det_path, cfg, device)
    return det, load_refiner(ref_path, device), dcfg


def cmd_import_onnx(a):
    import torch
    from dcc.onnx_import import ckpt_from_onnx, detector_from_onnx, refiner_from_onnx
    cfg = _load_cfg(a.config)
    if a.kind == "detector":
        assert cfg is not None, "--config is required for a detector graph"
        model, diff = detector_from_onnx(a.onnx, cfg)
    else:
        model, diff = refiner_from_onnx(a.onnx)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt_from_onnx(a.onnx, model, cfg, diff), a.out)
    print(f"[import-onnx] {a.onnx} -> {a.out}: {sum(p.numel() for p in model.parameters()):,} params, "
          f"max |torch - onnxruntime| = {diff:.2e}")


# ---------------------------------------------------------------------------------------------
# make-clip: a synthetic clip with ground truth, for testing the loop end to end
# ---------------------------------------------------------------------------------------------

def _procedural_bg(rng, w, h):
    import cv2
    import numpy as np
    img = np.zeros((h, w), np.float32)
    for s in (3, 12, 48, 150):
        img += cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), s) * s ** 0.5
    img = (img - img.min()) / max(img.max() - img.min(), 1e-6) * 255
    for _ in range(int(rng.integers(8, 25))):
        p1, p2 = rng.integers(0, w, 2), rng.integers(0, h, 2)
        col = float(rng.uniform(0, 255))
        if rng.random() < 0.5:
            cv2.rectangle(img, (int(p1[0]), int(p2[0])), (int(p1[1]), int(p2[1])), col, -1)
        else:
            cv2.line(img, (int(p1[0]), int(p2[0])), (int(p1[1]), int(p2[1])), col, int(rng.integers(2, 12)))
    return img


def cmd_make_clip(a):
    import cv2
    import numpy as np
    from dcc.board import get_board, render_board
    from dcc.pipeline import canon_lattice
    from dcc.synth import _perspective_factor, _sample_affine, list_backgrounds

    cfg = _load_cfg(a.config)
    W, H = (int(v) for v in a.size.lower().split("x"))
    assert W * 3 == H * 4, "--size must be 4:3"
    rng = np.random.default_rng(a.seed)
    T = a.frames
    out = Path(a.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    if a.backgrounds:
        files = list_backgrounds(a.backgrounds)
        bg = cv2.imread(files[int(rng.integers(len(files)))], cv2.IMREAD_GRAYSCALE).astype(np.float32)
        sc = max(W / bg.shape[1], H / bg.shape[0])
        bg = cv2.resize(bg, (int(np.ceil(bg.shape[1] * sc)), int(np.ceil(bg.shape[0] * sc))))[:H, :W]
    else:
        bg = _procedural_bg(rng, W, H)
    bg = bg * 0.6 + 40
    bcfg = cfg.get("board")
    nsq = get_board(bcfg)[1]
    img, p_render = render_board(480, bcfg)
    ink, paper = rng.uniform(25, 50), rng.uniform(170, 225)
    board = (ink + (paper - ink) * img.astype(np.float32) / 255.0)
    SQ = 480 / nsq
    lat = canon_lattice(nsq - 1)
    A = np.array([[SQ, 0, -0.5], [0, SQ, -0.5], [0, 0, 1.0]])
    ph = rng.uniform(0, 2 * np.pi, 8)
    u = np.arange(T) / max(T - 1, 1)
    s = 115 * np.exp(0.75 * np.sin(2 * np.pi * 0.9 * u + ph[0]))               # sensor px per square: ~55-245
    theta = 0.4 * np.sin(2 * np.pi * 0.7 * u + ph[1]) + ph[2]
    tilt = np.clip(np.radians(25 + 22 * np.sin(2 * np.pi * 1.1 * u + ph[3])), 0, np.radians(55))
    psi = ph[4] + 0.6 * np.sin(2 * np.pi * 0.5 * u + ph[5])
    tx = 0.18 * W * np.sin(2 * np.pi * 0.6 * u + ph[6])
    ty = 0.15 * H * np.sin(2 * np.pi * 0.8 * u + ph[7])
    # exposure: bright, fade to `dark`, hold, recover -- the regime a real night-time approach visits
    gain = np.ones(T)
    f0, f1, f2, f3 = (int(T * q) for q in (0.25, 0.42, 0.62, 0.8))
    gain[f0:f1] = np.exp(np.linspace(0, np.log(a.dark), f1 - f0))
    gain[f1:f2] = a.dark
    gain[f2:f3] = np.exp(np.linspace(np.log(a.dark), 0, f3 - f2))
    occ_on = (not a.no_occluder)
    o0, o1 = int(T * 0.06), int(T * 0.2)
    occ_tex = cv2.GaussianBlur(rng.normal(120, 60, (H, W)).astype(np.float32), (0, 0), 3)
    gt_lines, prev_c = [], None
    for t in range(T):
        comps = dict(s=float(s[t]), theta=float(theta[t]), shear_x=0.04, shear_y=-0.03, tx=float(tx[t]), ty=float(ty[t]))
        M, c = _sample_affine(cfg, None, W, H, None, 1, comps, nsq)
        M3 = np.eye(3)
        M3[:2] = M
        Ht = M3 @ _perspective_factor(float(tilt[t]), float(psi[t]), 1.0, c["s"], W, 480, nsq)
        sig = 0.5 * (SQ / s[t] - 1)
        b = cv2.GaussianBlur(board, (0, 0), sig) if sig > 0.1 else board
        warped = cv2.warpPerspective(b, Ht, (W, H), flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.ones_like(board), Ht, (W, H), flags=cv2.INTER_LINEAR)
        frame = bg * (1 - mask) + warped * mask
        corners = project_pts(Ht @ A, lat)
        vis = (corners[:, 0] >= -0.5) & (corners[:, 0] < W - 0.5) & (corners[:, 1] >= -0.5) & (corners[:, 1] < H - 0.5)
        if occ_on and o0 <= t < o1:                  # an object sweeps across the board
            cen = corners.mean(axis=0)
            q = (t - o0) / max(o1 - o0 - 1, 1)
            ox = cen[0] + (q - 0.5) * 3.0 * s[t]
            oy = cen[1] + 0.4 * s[t]
            ax, ay = 0.9 * s[t], 1.6 * s[t]
            yy, xx = np.mgrid[0:H, 0:W]
            occ = (((xx - ox) / ax) ** 2 + ((yy - oy) / ay) ** 2) <= 1.0
            frame[occ] = occ_tex[occ]
            inside = (((corners[:, 0] - ox) / (ax + 2)) ** 2 + ((corners[:, 1] - oy) / (ay + 2)) ** 2) <= 1.0
            vis &= ~inside
        cen = corners.mean(axis=0)
        if prev_c is not None:                       # motion blur along the image motion, half-frame shutter
            frame = _motion_blur(frame, (cen - prev_c) * 0.5)
        prev_c = cen
        frame = frame * gain[t]
        Kc, read = 6.0, 1.2                          # e-/DN, read noise DN
        frame = rng.poisson(np.maximum(frame, 0) * Kc) / Kc + rng.normal(0, read, frame.shape)
        frame = np.clip(np.rint(frame), 0, 255).astype(np.uint8)
        cv2.imwrite(str(out / "frames" / f"{t:06d}.png"), frame)
        gt_lines.append(json.dumps({"frame": t, "gain": float(gain[t]), "s_px": float(s[t]),
                                    "corners": [{"x": float(x), "y": float(y), "index": k, "visible": bool(vv)}
                                                for k, ((x, y), vv) in enumerate(zip(corners, vis))]}))
    (out / "gt.jsonl").write_text("\n".join(gt_lines) + "\n")
    (out / "meta.json").write_text(json.dumps({"frames": T, "size": [W, H], "seed": a.seed, "dark": a.dark,
                                               "dark_frames": [f1, f2], "occluder": [o0, o1] if occ_on else None,
                                               "board": bcfg, "coordinates": "sensor px of the 4:3 frame"}, indent=1))
    print(f"[make-clip] {T} frames {W}x{H} -> {out}/frames; dark {a.dark} over frames {f1}-{f2}; "
          f"occluder {o0}-{o1 if occ_on else 'off'}")


def _motion_blur(frame, v, max_len=30.0):
    """Linear motion blur of extent v (px), CENTRED: the kernel is sampled symmetrically about its own
    centre with bilinear splats, so the blurred corner's centroid stays on its ground-truth position.
    (A cv2.line kernel with integer endpoints is off-centre by up to half a pixel per axis, which
    showed up as a 0.5-0.8 px bias against ground truth on every blurred frame.)"""
    import cv2
    import numpy as np
    v = np.asarray(v, dtype=np.float64)
    L = float(np.linalg.norm(v))
    if L < 1.0:
        return frame
    if L > max_len:
        v, L = v * max_len / L, max_len
    k = int(np.ceil(L / 2.0)) + 1
    ker = np.zeros((2 * k + 1, 2 * k + 1), np.float64)
    for s in np.linspace(-0.5, 0.5, max(3, int(np.ceil(4 * L)) | 1)):
        x, y = k + s * v[0], k + s * v[1]
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        fx, fy = x - x0, y - y0
        ker[y0, x0] += (1 - fx) * (1 - fy)
        ker[y0, x0 + 1] += fx * (1 - fy)
        ker[y0 + 1, x0] += (1 - fx) * fy
        ker[y0 + 1, x0 + 1] += fx * fy
    return cv2.filter2D(frame, -1, (ker / ker.sum()).astype(np.float32))


def project_pts(H, pts):
    import numpy as np
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return p[:, :2] / p[:, 2:3]


# ---------------------------------------------------------------------------------------------
# harvest
# ---------------------------------------------------------------------------------------------

def harvest(videos, det, ref, cfg, out, camera=(None, None), step=1, max_frames=None, stride=1, cap=0,
            sheet=24, gts=None, hp=None, log=print):
    """Harvest each video; returns {clip: stats}. Labels never come from the detections themselves."""
    import cv2
    import numpy as np
    from dcc import video as V
    from dcc.board import n_corners
    from dcc.pipeline import detect
    from dcc.viz import tile

    W, H = cfg["input_size"]
    n = int(round(n_corners(cfg.get("board")) ** 0.5))
    tmpl = V.BoardTemplate(cfg.get("board"), size=(hp or {}).get("template_size", 320))
    K, dist = camera
    manifest = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "clips": {}, "harvest_params": {**V.HARVEST_DEFAULTS, **(hp or {})}}
    for vi, src in enumerate(videos):
        name = Path(src).stem if Path(src).is_file() else Path(src).name
        t0 = time.time()
        raw, und, results, src_idx, r = [], [], [], [], None
        lens = None
        for i, fs in V.iter_frames(src, step=step, max_frames=max_frames):
            if r is None:
                r = W / fs.shape[1]
                lens = V.Lens(K, dist, r, (W, H))
            fi = V.to_input(fs, W, H)
            raw.append(fi)
            und.append(lens.undistort_image(fi))
            results.append(detect(fs, det, ref, K=None if K is None else np.asarray(K, float), dist=None if dist is None else np.asarray(dist, float), cfg=cfg))
            src_idx.append(i)
        t_det = time.time() - t0
        anchors = {}
        for t, res in enumerate(results):
            f = V.anchor_from_detection(res, r, n, lens, **{**V.HARVEST_DEFAULTS, **(hp or {})})
            if f is not None:
                anchors[t] = f
        labels, st = V.harvest_labels(und, anchors, tmpl, n, hp)
        records, images = [], []
        for t in sorted(labels):
            rec = V.build_record(t, raw[t], und[t], labels[t], tmpl, n, lens, hp)
            rec["source_frame"] = src_idx[t]
            records.append(rec)
        kept = V.cap_records(records, cap_per_bucket=cap, stride=stride)
        images = [raw[rec["frame"]] for rec in kept]
        meta = {"clip": str(src), "frames": len(raw), "input_size": [W, H], "r": r, "camera": bool(K is not None),
                "noise": V.noise_model(raw, records)}
        d, sha = V.save_harvest(out, name, images, kept, meta)
        n_src = {k: sum(1 for x in records if x["source"] == k) for k in ("anchor", "tracked")}
        vis = [c["visible"] for x in kept for c in x["corners"]]
        stats = {"frames": len(raw), "anchors_detected": st["anchors_in"], "anchors": st["anchors"],
                 "labelled": len(records), "labelled_anchor": n_src["anchor"], "labelled_tracked": n_src["tracked"],
                 "kept": len(kept), "corners_positive": vis.count(True), "corners_unknown": vis.count(None),
                 "closures_px": [round(e, 3) for e in st["closures"]], "fb_px": [round(e, 3) for e in st["fb"]],
                 "detect_s": round(t_det, 1), "total_s": round(time.time() - t0, 1), "sha1": sha}
        if gts and vi < len(gts) and gts[vi]:
            stats["vs_gt"] = label_accuracy(kept, gts[vi], r, src_idx)
        manifest["clips"][name] = stats
        if sheet and kept:
            pick = np.linspace(0, len(kept) - 1, min(sheet, len(kept))).round().astype(int)
            tiles = [cv2.resize(V.overlay(images[i], kept[i]), (W // 2, H // 2)) for i in sorted(set(pick.tolist()))]
            cv2.imwrite(str(d / "sheet.png"), tile(tiles, cols=6))
        log(f"[harvest] {name}: {len(raw)} frames, {st['anchors']}/{st['anchors_in']} anchors, "
            f"{len(records)} labelled ({n_src['anchor']} fit + {n_src['tracked']} tracked), {len(kept)} kept, "
            f"{stats['corners_positive']} positive / {stats['corners_unknown']} unknown corners "
            f"({stats['total_s']} s)" + (f"\n          vs ground truth: {stats['vs_gt']}" if "vs_gt" in stats else ""))
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def _read_gt(path):
    return {json.loads(l)["frame"]: json.loads(l) for l in open(path) if l.strip()}


def label_accuracy(records, gt_path, r, src_idx):
    """Harvested labels vs make-clip ground truth (sensor px -> input px): position error of positives,
    and how often a 'positive' is actually occluded or out of frame (hallucination rate)."""
    import numpy as np
    from dcc.video import sensor_to_input
    gt = _read_gt(gt_path)
    errs, halluc, pos, missed_vis, unknown_vis = [], 0, 0, 0, 0
    by_src = {"anchor": [], "tracked": []}
    for rec in records:
        g = gt[rec["source_frame"]]
        for c, gc in zip(rec["corners"], g["corners"]):
            xy = sensor_to_input([[gc["x"], gc["y"]]], r)[0]
            if c["visible"] is True:
                pos += 1
                if not gc["visible"]:
                    halluc += 1
                else:
                    e = float(np.hypot(c["x"] - xy[0], c["y"] - xy[1]))
                    errs.append(e)
                    by_src[rec["source"]].append(e)
            elif gc["visible"]:
                missed_vis += 1
                unknown_vis += int(c["visible"] is None)
    q = lambda v, p: round(float(np.percentile(v, p)), 3) if v else None
    return {"positives": pos, "hallucinated": halluc, "err_median_px": q(errs, 50), "err_p95_px": q(errs, 95),
            "err_max_px": round(max(errs), 3) if errs else None,
            "anchor_err_median": q(by_src["anchor"], 50), "tracked_err_median": q(by_src["tracked"], 50),
            "tracked_err_p95": q(by_src["tracked"], 95), "visible_gt_not_positive": missed_vis,
            "of_which_unknown": unknown_vis}


def cmd_harvest(a):
    cfg_arg = _load_cfg(a.config)
    dev = _device(a.device)
    det, ref, cfg = load_models(a.detector, a.refiner, cfg_arg, dev)
    if cfg_arg is not None:
        cfg = {**cfg, **{k: v for k, v in cfg_arg.items() if k not in ("finetune", "real")}}
    harvest(a.video, det, ref, cfg, a.out, camera=_camera(a.camera), step=a.step, max_frames=a.max_frames,
            stride=a.stride, cap=a.cap, sheet=a.sheet, gts=a.gt)


# ---------------------------------------------------------------------------------------------
# fine-tune config
# ---------------------------------------------------------------------------------------------

def finetune_config(base_cfg, src, records, a):
    import copy
    cfg = copy.deepcopy(base_cfg)
    lr_mult = {}
    for kv in a.lr_mult:
        k, v = kv.rsplit("=", 1)
        lr_mult[k] = float(v)
    cfg.pop("finetune", None)
    cfg.pop("real", None)
    cfg["finetune"] = {"from": str(src), "lr_mult": lr_mult, "freeze_bn": True}
    cfg["real"] = {"records": [str(r) for r in records], "frac": a.real_frac,
                   "darken_synthetic_p": getattr(a, "darken_synthetic", 0.0)}
    t = cfg["train"]
    t.update({"steps": a.steps, "lr": a.lr, "lr_floor": a.lr / 100.0, "warmup_steps": min(100, a.steps // 10),
              "batch": a.batch, "accum": a.accum, "workers": a.workers, "prefetch_factor": 2,
              "val_every": max(1, a.steps // 3), "full_val_every": a.steps, "val_subset": a.val_subset,
              "ckpt_rolling_every": a.steps + 1})
    t["early_stop"] = {"enabled": False}
    t["wandb"] = {"enabled": False}
    cfg["synth"]["val_size"] = a.val_size
    if a.backgrounds:
        cfg["synth"]["backgrounds"] = str(a.backgrounds)
    return cfg


def cmd_finetune_config(a):
    import yaml
    cfg = finetune_config(_load_cfg(a.config), a.src, a.records, a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"[finetune-config] {a.out}: {a.steps} steps at lr {a.lr}, real.frac {a.real_frac}, from {a.src}")


# ---------------------------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------------------------

def evaluate(src, det, ref, cfg, gt_path=None, labels_dir=None, camera=(None, None), bins=(0.02, 0.15),
             return_rows=False):
    """Per-frame identified corners vs a reference, grouped by brightness. With ground truth the groups
    are by exposure gain; with a harvest reference, by the board's median level in the frame."""
    import numpy as np
    from dcc import video as V
    from dcc.board import n_corners
    from dcc.pipeline import detect

    W, H = cfg["input_size"]
    n = int(round(n_corners(cfg.get("board")) ** 0.5))
    K, dist = camera
    ref_by_frame, group_key = {}, None
    if gt_path:
        gt = _read_gt(gt_path)
        group_key = "gain"
    if labels_dir:
        recs = json.loads((Path(labels_dir) / "records.json").read_text())["records"]
        ref_by_frame = {rr["source_frame"]: rr for rr in recs}
    rows, r, lens = [], None, None
    for i, fs in V.iter_frames(src):
        if r is None:
            r = W / fs.shape[1]
            lens = V.Lens(K, dist, r, (W, H))
        res = detect(fs, det, ref, K=None if K is None else np.asarray(K, float),
                     dist=None if dist is None else np.asarray(dist, float), cfg=cfg)
        dets = {}
        for c in res["corners"]:
            if c["index"] is not None:
                dets.setdefault(c["index"], []).append(V.sensor_to_input([[c["x"], c["y"]]], r)[0])
        fit = V.anchor_from_detection(res, r, n, lens, **V.HARVEST_DEFAULTS)
        if gt_path:
            g = gt[i]
            refc = [(k, V.sensor_to_input([[c["x"], c["y"]]], r)[0]) for k, c in enumerate(g["corners"]) if c["visible"]]
            level = g["gain"]
        elif i in ref_by_frame:
            rr = ref_by_frame[i]
            refc = [(c["index"], np.array([c["x"], c["y"]])) for c in rr["corners"] if c["visible"] is True]
            level = rr["board_level"]
        else:
            continue
        hits, errs = 0, []
        for k, xy in refc:
            ds = [np.hypot(*(d - xy)) for d in dets.get(k, [])]
            if ds and min(ds) <= 2.0:
                hits += 1
                errs.append(min(ds))
        n_id = sum(len(v) for v in dets.values())
        wrong = sum(1 for k, v in dets.items() for d in v
                    if not any(kk == k and np.hypot(*(d - xy)) <= 2.0 for kk, xy in refc))
        rows.append({"frame": i, "level": level, "ref": len(refc), "hits": hits, "ids": n_id, "wrong": wrong,
                     "fit": fit is not None, "err": errs})
    lo, hi = bins if group_key == "gain" else (20.0, 60.0)
    groups = {"dark": lambda v: v < lo, "dim": lambda v: lo <= v < hi, "bright": lambda v: v >= hi, "all": lambda v: True}
    out = {}
    for gname, f in groups.items():
        rs = [x for x in rows if f(x["level"])]
        if not rs:
            continue
        ref_n, hit_n, id_n = sum(x["ref"] for x in rs), sum(x["hits"] for x in rs), sum(x["ids"] for x in rs)
        errs = [e for x in rs for e in x["err"]]
        out[gname] = {"frames": len(rs), "recall": round(hit_n / ref_n, 4) if ref_n else None,
                      "id_precision": round(1 - sum(x["wrong"] for x in rs) / id_n, 4) if id_n else None,
                      "fit_rate": round(sum(x["fit"] for x in rs) / len(rs), 4),
                      "err_median_px": round(float(np.median(errs)), 3) if errs else None}
    return (out, rows) if return_rows else out


def cmd_eval(a):
    cfg_arg = _load_cfg(a.config)
    dev = _device(a.device)
    report = {}
    for dpath in a.detector:
        det, ref, cfg = load_models(dpath, a.refiner, cfg_arg, dev)
        report[dpath] = evaluate(a.video, det, ref, cfg, a.gt, a.labels, _camera(a.camera))
        print_eval(dpath, report[dpath])
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))


def _sv_line(v):
    m01, m04 = v["m01"], v["m04"]
    return (f"M-01 median {m01['median']:.4f} p95 {m01['p95']:.4f} tail {m01['tail_frac_gt4px']}, "
            f"M-04 {m04['accuracy'] if m04 else None}, n_matched {m01['n_matched']}")


def print_eval(name, rep):
    print(f"[eval] {name}")
    for g, m in rep.items():
        print(f"   {g:7s} frames {m['frames']:4d}  recall {m['recall']}  id_precision {m['id_precision']}  "
              f"fit_rate {m['fit_rate']}  err_median {m['err_median_px']} px")


# ---------------------------------------------------------------------------------------------
# rounds
# ---------------------------------------------------------------------------------------------

def synth_val(det, cfg, n):
    """The OTHER side of the gate: synthetic validation (M-01/M-02/M-04) through train_detector's own
    run_validation, so base and fine-tuned models are scored by one implementation on one set. Loaded by
    file path, never `import tools.*` (README: the machine-wide tools-package shadow). Single-process
    loader: a collate_fn from a path-loaded module cannot be unpickled by spawn workers."""
    import importlib.util
    from functools import partial

    import torch
    from torch.utils.data import DataLoader

    from dcc.dataset import SynthVal
    spec = importlib.util.spec_from_file_location("_train_detector", ROOT / "tools" / "train_detector.py")
    td = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(td)
    loader = DataLoader(SynthVal(cfg, n=n, seed=cfg["synth"]["val_seed"]), batch_size=6, num_workers=0,
                        collate_fn=partial(td._val_collate, cfg))
    res = td.run_validation(det.to(memory_format=torch.channels_last), loader, cfg, "cuda",
                            cfg["train"]["tau_hm"], cfg["train"]["match_px"])
    return {"m01": res["m01"], "m02": res["m02"], "m04": res["m04"]}


def cmd_rounds(a):
    import torch
    import yaml
    from dcc.onnx_import import ckpt_from_onnx, detector_from_onnx

    wd = Path(a.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    base_cfg = _load_cfg(a.config)
    dev = _device(None)
    base = Path(a.base)
    if base.suffix == ".onnx":                          # fine-tuning needs trainer weights
        model, diff = detector_from_onnx(base, base_cfg)
        base = wd / "base.pt"
        torch.save(ckpt_from_onnx(a.base, model, base_cfg, diff), base)
        print(f"[rounds] imported {a.base} -> {base} (max |d| {diff:.2e})")
    camera = _camera(a.camera)
    log = open(wd / "rounds.log", "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    summary = {"base": {}, "synth_val": {}}
    vcfg = finetune_config(base_cfg, base, [], a)     # the run's own synthetic set: same backgrounds, same size
    det, ref, cfg = load_models(base, a.refiner, base_cfg, dev)
    summary["synth_val"]["base"] = synth_val(det, vcfg, a.val_size)
    say(f"[rounds] base synthetic val: {_sv_line(summary['synth_val']['base'])}")
    for vi, v in enumerate(a.heldout_video):
        gt = a.heldout_gt[vi] if a.heldout_gt and vi < len(a.heldout_gt) else None
        summary["base"][v] = evaluate(v, det, ref, cfg, gt, None, camera)
        print_eval(f"base on {v}", summary["base"][v])
    teacher = base
    for k in range(1, a.rounds + 1):
        rd = wd / f"round{k}"
        say(f"[rounds] ---- round {k}: harvest with {teacher}")
        det, ref, cfg = load_models(teacher, a.refiner, base_cfg, dev)
        cfg = {**cfg, **{kk: vv for kk, vv in base_cfg.items() if kk not in ("finetune", "real")}}
        harvest(a.train_video, det, ref, cfg, rd / "harvest", camera=camera, stride=a.stride, cap=a.cap, log=say)
        del det
        torch.cuda.empty_cache()
        clips = sorted(p for p in (rd / "harvest").iterdir() if (p / "records.json").exists())
        fcfg = finetune_config(base_cfg, base, clips, a)
        (rd / "finetune.yaml").write_text(yaml.safe_dump(fcfg, sort_keys=False))
        name = f"vb_{wd.name}_r{k}"
        cmd = [sys.executable, str(ROOT / "tools" / "train_detector.py"), "--config", str(rd / "finetune.yaml"), "--name", name]
        say(f"[rounds] train: {' '.join(cmd)}")
        with open(rd / "train.log", "w") as tl:
            ret = subprocess.run(cmd, cwd=ROOT, stdout=tl, stderr=subprocess.STDOUT)
        if ret.returncode != 0:
            say(f"[rounds] training FAILED (see {rd / 'train.log'})")
            break
        ckpt = ROOT / "runs" / name / f"ckpt_{a.steps:07d}.pt"
        say(f"[rounds] round {k} model: {ckpt}")
        det, ref, cfg = load_models(ckpt, a.refiner, base_cfg, dev)
        summary[f"round{k}"] = {}
        for vi, v in enumerate(a.heldout_video):
            gt = a.heldout_gt[vi] if a.heldout_gt and vi < len(a.heldout_gt) else None
            summary[f"round{k}"][v] = evaluate(v, det, ref, cfg, gt, None, camera)
            print_eval(f"round {k} on {v}", summary[f"round{k}"][v])
        summary["synth_val"][f"round{k}"] = synth_val(det, vcfg, a.val_size)
        say(f"[rounds] round {k} synthetic val: {_sv_line(summary['synth_val'][f'round{k}'])}  "
            f"(base: {_sv_line(summary['synth_val']['base'])})")
        teacher = ckpt
        (wd / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    say(f"[rounds] done -> {wd / 'summary.json'}")


def main():
    a = build_parser().parse_args()
    {"import-onnx": cmd_import_onnx, "make-clip": cmd_make_clip, "harvest": cmd_harvest,
     "finetune-config": cmd_finetune_config, "eval": cmd_eval, "rounds": cmd_rounds}[a.cmd](a)


if __name__ == "__main__":
    main()
