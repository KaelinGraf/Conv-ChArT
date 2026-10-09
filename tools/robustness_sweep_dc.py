"""tools/robustness_sweep_dc.py -- the Deep ChArUco arm of the robustness
comparison, on the SAME factors, steps, seeds and frames as our own sweep.

Kaelin's framing (2026-07-28): the comparison is **coarse vs coarse**. Our
refined pipeline against their unrefined detector would overstate our advantage
by construction -- Deep ChArUco's published system includes a RefineNet that we
do not run. So this scores their detector against OUR COARSE ARM: both
unrefined, both trained on the same data, architecture against architecture.

Kept OUT of the tracked tree (see .gitignore) for the same reason as the rest of
the comparison tooling: it imports `charuconet`, which vendors dcModel from
JunkyByte/deepcharuco (MIT). tools/robustness_sweep.py stays our-model-only and
tracked; this imports its FACTORS and apply_factor rather than restating them,
so the two arms cannot silently drift onto different step values.

RESOLUTION IS A REAL, REPORTABLE ASYMMETRY, not a nuisance to correct away:
their network sees 320x240 where ours sees 640x480, so at the far end of the
distance axis a board occupying 9% of the frame is ~60 px across for us and
~30 px for them. That is a genuine property of the compared systems and belongs
in the discussion of the distance figure -- it is why their curve falls off
earlier, and saying so is more useful than pretending the two are matched.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))     # sibling import of charuconet

import robustness_sweep as rs   # FACTORS / apply_factor / score_frame, shared verbatim

MATCH_PX = rs.MATCH_PX
DC_PRETRAINED = "/home/kaelin/p4p/external/deepcharuco/src/reference/longrun-epoch=99-step=369700.ckpt"


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/rev640.yaml")
    p.add_argument("--system", default="dc", choices=("dc", "classical"),
                    help="dc = Deep ChArUco; classical = OpenCV's own ChArUco detector (no model, no training)")
    p.add_argument("--dc-ckpt", default=None, help="fine-tuned dcModel .pt; omit for ZERO-SHOT (their pretrained)")
    p.add_argument("--factors", nargs="+", default=list(rs.FACTORS), choices=list(rs.FACTORS))
    p.add_argument("--n", type=int, default=rs.N_DEFAULT)
    p.add_argument("--seed", type=int, default=20260728, help="MUST match our sweep's seed -- identical frames")
    p.add_argument("--out", default="paper/results_rev6/05_comparison/robustness_dc")
    return p


def main():
    args = build_parser().parse_args()
    import cv2, numpy as np, torch
    cv2.setNumThreads(1); torch.set_num_threads(2)
    from charuconet import IN_H, IN_W, dcModel, decode_cells, load_reference_state_dict, pre_bgr_normalize
    import eval_classical as ec
    from dcc.board import n_corners
    from dcc.dataset import load_config
    from dcc.synth import generate_sample, list_backgrounds

    cfg = load_config(args.config)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W, H = cfg["input_size"]
    n_cls = n_corners(cfg.get("board"))

    model = None
    if args.system == "classical":
        # OpenCV's detector: no weights, no training, nothing to load. It is the
        # only arm that cannot adapt to the generator, which is exactly what makes
        # it a useful independent read on how hard the data is.
        cboard, cdict, _cnx = ec._get_board_and_dictionary(cfg.get("board"))
        cparams = ec._detector_params()
        step, arm = None, "classical_opencv"
    elif True:
        model = dcModel(n_ids=n_cls).to(dev).eval()
    if args.system == "dc" and args.dc_ckpt:
        obj = torch.load(args.dc_ckpt, map_location=dev, weights_only=False)
        sd = obj.get("ema", obj.get("model")) if isinstance(obj, dict) and ("model" in obj or "ema" in obj) else obj
        model.load_state_dict(sd)
        step = obj.get("step") if isinstance(obj, dict) else None
        arm = "deepcharuco_finetuned"
    elif args.system == "dc":
        load_reference_state_dict(DC_PRETRAINED, model, map_location=dev)
        step, arm = None, "deepcharuco_zeroshot"

    bg = list_backgrounds(cfg["synth"]["backgrounds"])
    match_greedy = rs._load_match_greedy()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    prov = {"arm": arm, "dc_ckpt": None if args.system == "classical" else (args.dc_ckpt or DC_PRETRAINED), "dc_step": step, "config": args.config,
            "n_per_step": args.n, "seed": args.seed, "match_px": MATCH_PX,
            "dc_input_size": [IN_W, IN_H], "our_input_size": [W, H],
            "note": "coarse-vs-coarse: their detector is unrefined and their RefineNet is NOT run; "
                    "compare against our COARSE arm only"}

    for name in args.factors:
        spec = rs.FACTORS[name]
        print(f"[{name}] {spec['unit']}", flush=True)
        rows = []
        for si, value in enumerate(spec["steps"]):
            cfg_s, comp, post = rs.apply_factor(cfg, name, value)
            acc = {"n_gt": 0, "n_match": 0, "n_id": 0, "errs": []}
            achieved = []
            for i in range(args.n):
                # identical seed triple to tools/robustness_sweep.py -> identical frames
                rng = np.random.default_rng([args.seed, si, i])
                rec, _ = generate_sample(cfg_s, rng, bg, components=comp or None, force_negative=False)
                gray = rec["image"]
                # board_occlusion is handled in the LOOP, not in apply_factor, because it
                # needs the record's corners to find the board and to update visibility.
                # Called from rs so both arms occlude identically -- same seed, same frames,
                # same rectangle, same visibility gating as our own sweep.
                if spec["kind"] == "board_occlusion":
                    gray, frac = rs.occlude_board_fraction(gray, rec, value / 100.0, rng)
                    achieved.append(frac)
                if post is not None:
                    gray = post(gray)
                vis = [c for c in rec["corners"] if c["visible"]]
                if not vis:
                    continue
                gt_xy = np.array([[c["x"], c["y"]] for c in vis], dtype=np.float64)
                gt_idx = [c["index"] for c in vis]

                if args.system == "classical":
                    cxy, cids = ec.classical_charuco_detect(gray, cboard, cdict, cparams)
                    if cxy is None:
                        xy, idc = np.zeros((0, 2)), np.zeros((0,), dtype=int)
                    else:
                        xy, idc = np.asarray(cxy, dtype=np.float64).reshape(-1, 2), np.asarray(cids).reshape(-1)
                else:
                    img320 = cv2.resize(gray, (IN_W, IN_H), interpolation=cv2.INTER_AREA).astype(np.float32)
                    x = torch.from_numpy(pre_bgr_normalize(img320)).unsqueeze(0).unsqueeze(0).to(dev)
                    with torch.no_grad():
                        loc_logits, id_logits = model(x)
                    xy320, idc, _lc, _ic = decode_cells(loc_logits[0], id_logits[0])
                    # inverse of the fixed-ratio half-pixel resize, per-axis
                    xy = np.empty_like(xy320, dtype=np.float64)
                    xy[:, 0] = (xy320[:, 0] + 0.5) * (W / IN_W) - 0.5
                    xy[:, 1] = (xy320[:, 1] + 0.5) * (H / IN_H) - 0.5

                res = {"corners": [{"x": float(a), "y": float(b), "index": int(k)}
                                    for (a, b), k in zip(xy, idc)]}
                ngt, nm, nid, errs = rs.score_frame(gt_xy, gt_idx, res, match_greedy)
                acc["n_gt"] += ngt; acc["n_match"] += nm; acc["n_id"] += nid; acc["errs"] += errs

            e = np.array(acc["errs"]) if acc["errs"] else np.array([0.0])
            rec_r = acc["n_match"] / max(acc["n_gt"], 1)
            id_r = acc["n_id"] / max(acc["n_match"], 1)
            rows.append({"value": value, "arms": {arm: {
                "n_gt": acc["n_gt"], "n_match": acc["n_match"], "n_id": acc["n_id"],
                "recall": rec_r, "id_acc": id_r,
                "err_median": float(np.median(e)), "err_mean": float(e.mean()),
                "err_p95": float(np.percentile(e, 95)),
                "recall_se": float(np.sqrt(max(rec_r * (1 - rec_r), 0) / max(acc["n_gt"], 1))),
                "id_acc_se": float(np.sqrt(max(id_r * (1 - id_r), 0) / max(acc["n_match"], 1))),
                "errs": [float(v) for v in acc["errs"]]}}})
            print(f"  {name:>15} = {value:>7}: recall {rec_r*100:5.1f}%  id {id_r*100:5.1f}%  "
                  f"err_med {np.median(e):.4f} px  (n_gt={acc['n_gt']})", flush=True)
        with open(out_dir / f"{name}.json", "w") as f:
            json.dump({"factor": name, "unit": spec["unit"], "invert_x": spec.get("invert_x", False),
                       "provenance": prov, "steps": rows}, f, indent=2)
        print(f"  -> {out_dir / (name + '.json')}", flush=True)


if __name__ == "__main__":
    main()
