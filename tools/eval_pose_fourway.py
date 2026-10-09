"""tools/eval_pose_fourway.py -- pose accuracy for all four arms, side by side.

Ours (coarse), ours + refiner, fine-tuned Deep ChArUco, and classical OpenCV, each
scored on the SAME gen_eval_pose.py set with the SAME intrinsics and the SAME PnP.

THE FAIRNESS DECISION, stated up front because it changes what the table means:
every arm's corners go through **our** PnP path (dcc.pipeline.pnp -> undistort ->
solvePnPGeneric/IPPE). Classical has its own `cv2` pose routine and Deep ChArUco has
none at all, so using each arm's native solver would compare three different PnP
implementations as much as three detectors. Routing every arm through one solver
isolates the thing under test: **the corners**. It also means the classical numbers
here will not match tools/eval_classical.py's own pose output, and that is expected.

REFUSALS ARE NOT ERRORS. dcc.pipeline refuses a pose it cannot corroborate (a vacuous
4-point fit, a degenerate lattice) rather than emitting a fabricated one. A refusal is
a correct outcome. It is reported as its own rate and excluded from the error
statistics -- never silently counted as a miss, and never dropped without being
counted, since an arm that refuses 40% of frames and is accurate on the rest is not
better than one that answers everything at slightly lower accuracy.

UNITS. gen_eval_pose.py fixes square_length_m = 1.0, so translation error is in BOARD
SQUARES, not metres. Rotation is the geodesic angle between R_gt and R_est.

GITIGNORED (imports charuconet) -- same reason as robustness_sweep_dc.py.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


RELEASE_CKPTS = ["rel222=runs/rel_w25_lam2_100k_rev6/ckpt_0100000.pt",
                 "rel502=runs/rel_w375_lam15_100k_rev6/ckpt_0100000.pt",
                 "rel882=runs/rel_w882_c2_100k_rev6/ckpt_0100000.pt"]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pose-set", default="eval_pose_rev6")
    p.add_argument("--config", default="configs/rev640.yaml")
    p.add_argument("--ckpt", action="append", metavar="LABEL=PATH",
                   help="repeatable; every arm is scored on the SAME frames in ONE pass, so the "
                        "baselines are computed once rather than once per tier. Default: the "
                        "three release tiers.")
    p.add_argument("--refiner-ckpt", default="runs/ref_s15_10k_rev6/ckpt_0010000.pt")
    p.add_argument("--dc-ckpt", default="runs/charuconet_rev6_ft/ckpt_latest.pt")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--out", default="paper/results_rev6/13_pose/pose_fourway.json")
    return p


def main():
    args = build_parser().parse_args()
    import cv2, numpy as np, torch
    cv2.setNumThreads(1); torch.set_num_threads(2)
    import eval_classical as ec
    from charuconet import IN_H, IN_W, dcModel, decode_cells, pre_bgr_normalize
    from dcc.board import n_corners
    from dcc.dataset import load_config
    from dcc.model import DetectorNet, detector_kwargs, refiner_for
    from dcc.pipeline import detect, lattice_gate, pnp, recover, undistort

    cfg = load_config(args.config)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W, H = cfg["input_size"]
    ps = Path(args.pose_set)
    meta = json.loads((ps / "meta.json").read_text())
    labels = [json.loads(l) for l in (ps / "labels.jsonl").read_text().splitlines()]
    labels = labels[:args.n]
    n_cls = n_corners(cfg.get("board"))
    sqlen = (cfg.get("board") or {}).get("square_length_m") or 1.0
    n = int(np.sqrt(n_cls))

    # EACH DETECTOR IS BUILT FROM ITS OWN CHECKPOINT'S cfg, not from --config. The release
    # tiers differ in width_mult (0.25 / 0.375 / 0.5-with-attn), gates and lambda_cls, so a
    # single --config-built net either raises a shape error or -- worse -- silently scores the
    # wrong architecture. Same fix already applied to tools/eval_pose_ours.py.
    specs = [tuple(x.split("=", 1)) for x in (args.ckpt or RELEASE_CKPTS)]
    ours_arms = []
    for label, path in specs:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        mcfg = ck.get("cfg") or cfg
        m = DetectorNet(H, W, n_cls=n_corners(mcfg.get("board")), **detector_kwargs(mcfg)).to(dev).eval()
        m.load_state_dict(ck.get("ema", ck.get("model")))
        ours_arms.append((label, m, mcfg, path))
        print(f"[{label}] {sum(q.numel() for q in m.parameters()):,} params  step={ck.get('step')}")
    rk = torch.load(args.refiner_ckpt, map_location="cpu", weights_only=False)
    rsd = rk.get("ema", rk.get("model"))
    # refiner_for infers width from the state dict; a bare Refiner() is the default width and
    # would mis-load any non-default refiner.
    ref = refiner_for(rsd).to(dev).eval(); ref.load_state_dict(rsd)
    dc = dcModel(n_ids=n_cls).to(dev).eval()
    ob = torch.load(args.dc_ckpt, map_location=dev, weights_only=False)
    dc.load_state_dict(ob.get("ema", ob.get("model")))
    cboard, cdict, _ = ec._get_board_and_dictionary(cfg.get("board"))
    cparams = ec._detector_params()

    def rot_err(Rg, Re):
        c = (np.trace(Rg.T @ Re) - 1.0) / 2.0
        return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))

    acc = {k: {"rot": [], "trans": [], "refused": 0, "n": 0}
           for k in [f"{lab}_{v}" for lab, _, _, _ in ours_arms for v in ("coarse", "refined")]
                    + ["deep_charuco", "classical"]}

    for li, lab in enumerate(labels):
        img = cv2.imread(str(ps / lab["file"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        K = np.array(lab["K"], dtype=np.float64)
        Rg = np.array(lab["R"], dtype=np.float64)
        tg = np.array(lab["t"], dtype=np.float64).reshape(3)

        def score(key, rvec, tvec, refused):
            a = acc[key]; a["n"] += 1
            if refused or rvec is None:
                a["refused"] += 1; return
            rv = np.asarray(rvec, dtype=np.float64).reshape(3)
            tv = np.asarray(tvec, dtype=np.float64).reshape(3)
            # A non-finite solve is a FAILURE, not a datum. Letting one through poisons
            # mean and p95 into NaN and silently destroys the whole row -- which is what
            # the classical arm did on the first run.
            if not (np.isfinite(rv).all() and np.isfinite(tv).all()):
                a["refused"] += 1; return
            Re, _ = cv2.Rodrigues(rv.reshape(3, 1))
            a["rot"].append(rot_err(Rg, Re))
            a["trans"].append(float(np.linalg.norm(tv - tg)))

        # --- our two arms: detect() does the whole chain including its own PnP
        for lab, m, mcfg, _ in ours_arms:
            for suffix, r in (("coarse", None), ("refined", ref)):
                o = detect(img, m, r, K=K, cfg=mcfg)   # each arm's OWN cfg (tau_hm, board, ...)
                score(f"{lab}_{suffix}", o["rvec"], o["tvec"], o["rvec"] is None)

        # --- baselines: their corners, OUR PnP (see module docstring)
        def via_our_pnp(key, xy, ids):
            """Baseline corners through the SAME Stage-3 as our arms: lattice gate ->
            recovery -> PnP. Feeding them straight into pnp() was wrong and produced
            nonsense (DC rot p95 122 deg): our arms get the gate's ID filtering via
            detect(), so skipping it for the baselines compared a filtered pipeline
            against an unfiltered one. The gate demotes corners that do not fit the
            board homography, which is exactly what protects PnP from a mis-identified
            correspondence."""
            xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
            ids = np.asarray(ids).reshape(-1).astype(int)
            if len(xy) < 4:
                score(key, None, None, True); return
            idx = -np.ones(len(xy), dtype=int)
            ok = (ids >= 0) & (ids < n_cls)
            idx[ok] = ids[ok]
            xp = undistort(xy, K, None)
            conf = np.ones(len(xy))                       # baselines expose no per-corner score
            Hm, _inl, demoted, degen = lattice_gate(xp, idx, conf, cfg["lattice_tol_px"], n)
            if Hm is None:
                score(key, None, None, True); return
            idx = idx.copy(); idx[demoted] = -1
            idx, _rec, corrob = recover(Hm, xp, idx, conf, cfg["lattice_tol_px"], n)
            if degen == "vacuous" and not corrob:
                score(key, None, None, True); return
            rvec, tvec, _rms, _amb, _, reason, _cov = pnp(xp, idx, K, sqlen, n)
            score(key, rvec, tvec, rvec is None)

        i320 = cv2.resize(img, (IN_W, IN_H), interpolation=cv2.INTER_AREA).astype(np.float32)
        with torch.no_grad():
            ll, il = dc(torch.from_numpy(pre_bgr_normalize(i320)).unsqueeze(0).unsqueeze(0).to(dev))
        xy320, dids, _a, _b = decode_cells(ll[0], il[0])
        dxy = np.empty_like(xy320, dtype=np.float64)
        dxy[:, 0] = (xy320[:, 0] + 0.5) * (W / IN_W) - 0.5
        dxy[:, 1] = (xy320[:, 1] + 0.5) * (H / IN_H) - 0.5
        via_our_pnp("deep_charuco", dxy, dids)

        cxy, cids = ec.classical_charuco_detect(img, cboard, cdict, cparams)
        via_our_pnp("classical", np.zeros((0, 2)) if cxy is None else cxy,
                    np.zeros((0,), int) if cids is None else cids)

        if (li + 1) % 100 == 0:
            print(f"  {li + 1}/{len(labels)}", flush=True)

    rows = {}
    print(f"\n{'arm':>16} {'solve%':>7} {'rot med':>9} {'rot mean':>9} {'rot p95':>9} "
          f"{'tr med':>8} {'tr mean':>9} {'tr p95':>8}")
    for k, a in acc.items():
        r, t = np.array(a["rot"]), np.array(a["trans"])
        if not len(r):
            print(f"{k:>16}   no solves"); rows[k] = {"solve_rate": 0.0, "n": a["n"]}; continue
        rows[k] = {"n": a["n"], "n_solved": int(len(r)), "n_refused": a["refused"],
                   "solve_rate": len(r) / max(a["n"], 1),
                   "rot_deg": {"median": float(np.median(r)), "mean": float(r.mean()),
                                "p95": float(np.percentile(r, 95))},
                   "trans_squares": {"median": float(np.median(t)), "mean": float(t.mean()),
                                      "p95": float(np.percentile(t, 95))}}
        print(f"{k:>16} {100*len(r)/max(a['n'],1):>6.1f}% {np.median(r):>9.4f} {r.mean():>9.4f} "
              f"{np.percentile(r,95):>9.4f} {np.median(t):>8.4f} {t.mean():>9.4f} "
              f"{np.percentile(t,95):>8.4f}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"pose_set": args.pose_set, "n_images": len(labels),
                                "ckpts": {lab: pth for lab, _, _, pth in ours_arms}, "refiner_ckpt": args.refiner_ckpt,
                                "dc_ckpt": args.dc_ckpt,
                                "pnp": "ALL arms use dcc.pipeline.pnp -- isolates the corners, "
                                       "not the PnP implementation; classical numbers therefore "
                                       "differ from eval_classical.py's native output",
                                "units": "rotation deg (geodesic); translation in BOARD SQUARES "
                                         "(square_length_m = 1.0)",
                                "arms": rows}, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
