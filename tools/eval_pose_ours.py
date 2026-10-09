"""tools/eval_pose_ours.py -- M-05 (rotation) and M-06 (translation) for OUR
detector on a tools/gen_eval_pose.py output set.

The classical arm already had a pose evaluator (tools/eval_classical.py
--pose-set); our own model did not, so the pose row of the results table had
no way to be filled. This is that missing piece, and only that: it runs
dcc.pipeline.detect with the set's own K, then compares the returned
(rvec, tvec) against the set's labelled (R, t).

Conventions are taken verbatim from eval_classical._pose_error so the two
arms are directly comparable: rotation error is the geodesic angle between
R_gt and R_est (standard trace formula); translation error is the raw
Euclidean norm, and because gen_eval_pose.py fixes square_length_m = 1.0 it
is unitless -- "board squares", NOT metres. Reported alongside the refusal
rate, which matters more than it looks: dcc.pipeline REFUSES a pose it
cannot corroborate (vacuous 4-point fits, degenerate lattices) rather than
emitting a fabricated one, so a refusal is a correct outcome and must never
be silently counted as an error or dropped from the denominator.

The set's OWN meta.json["config"] drives generation-side conventions (board
geometry must match what actually rendered the images), exactly as
eval_classical.run_pose_set does -- not whatever config is currently on disk.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pose-set", default="eval_pose_rev5")
    p.add_argument("--ckpt", default="runs/rev640_baseline/ckpt_0150000.pt")
    p.add_argument("--refiner-ckpt", default="runs/refiner_rev5/ckpt_0025000.pt")
    p.add_argument("--no-refiner", action="store_true", help="coarse-only arm")
    p.add_argument("--n", type=int, default=None, help="limit images (default: all)")
    p.add_argument("--out", default=None)
    p.add_argument("--per-image", default=None,
                   help="JSONL of per-solve, per-axis errors (camera frame: drot_rad[3], dt_sq[3]) with s_px, "
                        "f_px, ambiguity, rms -- the raw material for a filter's measurement-noise R")
    p.add_argument("--allow-board-mismatch", action="store_true",
                   help="ZERO-SHOT TRANSFER TEST ONLY: score a checkpoint on a pose set rendered with a "
                        "DIFFERENT board of the same corner count. Identity is expected to fail; that is the "
                        "measurement. Off, the mismatch is an error (the class head would be scored against a "
                        "board it was never trained on).")
    return p


def _error_cov(E, amb, s_px):
    """Ensemble measurement-noise statistics of the solved poses: 6-vector [drot_x, drot_y, drot_z
    (rad); dt_x, dt_y, dt_z (board squares)] in the CAMERA frame, R_err = R_est @ R_gt.T and
    dt = t_est - t_gt. THIS is the constant R a Kalman filter can use. It is an ensemble statistic
    over frames, which the parked per-frame `pose_cov` / `sigma_px` are NOT (CLAUDE.md:
    uncertainty calibration is PARKED; pose_cov is ~6x over-confident). Reported for the filter
    contract's accepted set (unambiguous solves -- PIPELINE_SPEC.md says skip the update when
    `ambiguous`), for all solves, and by apparent-scale octave, since pixel error is ~constant
    while pose error grows with range. `var_mad` is (1.4826*MAD)^2, a tail-robust alternative to
    the heavy-tailed sample variance; the gap between the two IS the tail."""
    import numpy as np

    def stats(X):
        med = np.median(X, 0)
        return {"n": int(len(X)), "mean": X.mean(0).tolist(), "std": X.std(0, ddof=1).tolist(),
                "var": X.var(0, ddof=1).tolist(),
                "var_mad": ((1.4826 * np.median(np.abs(X - med), 0)) ** 2).tolist(),
                "cov": np.cov(X.T).tolist()}
    out = {"components": ["drot_x_rad", "drot_y_rad", "drot_z_rad", "dt_x_sq", "dt_y_sq", "dt_z_sq"],
           "frame": "camera; R_err = R_est @ R_gt.T, dt = t_est - t_gt; translation in board squares "
                    "(multiply variances by square_length_m^2 for m^2)"}
    if (~amb).sum() >= 2:
        out["unambiguous"] = stats(E[~amb])
    if len(E) >= 2:
        out["all_solved"] = stats(E)
    edges = [12, 16, 32, 64, 128.0001]      # train_detector.OCTAVE_BINS; top edge inclusive
    out["unambiguous_by_s_px_octave"] = {
        f"{int(lo)}-{int(hi)}": stats(E[m]) for lo, hi in zip(edges[:-1], edges[1:])
        if (m := ~amb & (s_px >= lo) & (s_px < hi)).sum() >= 2}
    return out


def main():
    args = build_parser().parse_args()
    import cv2
    import numpy as np
    import torch
    cv2.setNumThreads(1); torch.set_num_threads(2)

    from dcc.board import n_corners
    from dcc.model import DetectorNet, detector_kwargs, refiner_for
    from dcc.pipeline import detect

    pose_dir = Path(args.pose_set)
    meta = json.loads((pose_dir / "meta.json").read_text())
    cfg = meta["config"]
    labels = [json.loads(l) for l in (pose_dir / "labels.jsonl").read_text().splitlines() if l.strip()]
    if args.n:
        labels = labels[:args.n]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W, H = cfg["input_size"]
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # ARCHITECTURE comes from the CHECKPOINT's own cfg, not the pose set's. meta["config"] records
    # how the IMAGES were rendered -- this module's docstring says so ("generation-side
    # conventions") -- and using it to build the network silently assumed every checkpoint shares
    # the generator config's width. It does not: the release models are width 0.5 and 0.375, and
    # both died on a cls.* shape mismatch. Falls back to the pose set's cfg only for checkpoints
    # too old to carry one.
    mcfg = ck.get("cfg") or cfg
    board_a, board_b = mcfg.get("board"), cfg.get("board")
    if board_a is not None and board_b is not None and board_a != board_b:
        msg = (f"checkpoint board {board_a} != pose-set board {board_b}: the class head would be scored "
               f"against a board it was never trained on")
        assert args.allow_board_mismatch, msg
        print(f"[ZERO-SHOT TRANSFER] {msg} -- proceeding, --allow-board-mismatch given", flush=True)
    model = DetectorNet(H, W, n_cls=n_corners(cfg.get("board")), **detector_kwargs(mcfg)).to(device).eval()
    model.load_state_dict(ck.get("ema", ck.get("model")))

    refiner = None
    rk = None
    if not args.no_refiner:
        rk = torch.load(args.refiner_ckpt, map_location="cpu", weights_only=False)
        # refiner_for infers width from the weights -- Refiner() bare is the 97,056-param default
        # and would mis-build any narrow refiner (audit finding, same class as the line above).
        refiner = refiner_for(rk.get("ema", rk.get("model"))).to(device).eval()
        refiner.load_state_dict(rk.get("ema", rk.get("model")))

    rot, trans, refused, ambiguous, reasons = [], [], 0, 0, {}
    nees, nees_amb = [], []      # chi-square consistency of the reported pose covariance
    errs, amb_flags, s_px = [], [], []      # per solve: 6-vector error, ambiguity flag, apparent scale
    per_image = open(args.per_image, "w") if args.per_image else None
    for i, lab in enumerate(labels):
        img = cv2.imread(str(pose_dir / lab["file"]), cv2.IMREAD_GRAYSCALE)
        K = np.array(lab["K"], dtype=np.float64)
        res = detect(img, model, refiner, K=K, dist=None, cfg=cfg)
        if res["rvec"] is None:
            refused += 1
            reasons[res.get("reason") or "none"] = reasons.get(res.get("reason") or "none", 0) + 1
            continue
        R_est, _ = cv2.Rodrigues(np.asarray(res["rvec"]))
        c = (np.trace(np.array(lab["R"]).T @ R_est) - 1.0) / 2.0
        rot.append(float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))))
        trans.append(float(np.linalg.norm(np.asarray(res["tvec"]).reshape(3) - np.asarray(lab["t"]))))
        ambiguous += bool(res.get("ambiguous"))
        # Camera-frame error pose: R_err = R_est R_gt^T is the small rotation taking the true board
        # orientation to the estimated one, in camera axes; dt is the camera-frame translation error.
        R_err = R_est @ np.array(lab["R"]).T
        dr = cv2.Rodrigues(R_err)[0].reshape(3)
        dt = np.asarray(res["tvec"]).reshape(3) - np.asarray(lab["t"])
        e = np.concatenate([dr, dt])
        errs.append(e); amb_flags.append(bool(res.get("ambiguous"))); s_px.append(float(lab.get("s_px", np.nan)))
        if per_image is not None:
            per_image.write(json.dumps({
                "i": i, "file": lab["file"], "s_px": s_px[-1], "f_px": float(K[0, 0]),
                "n_visible": sum(bool(c_["visible"]) for c_ in lab["corners"]),
                "n_used": sum(c_["index"] is not None for c_ in res["corners"]),
                "rms_px": None if res["rms"] is None else float(res["rms"]), "ambiguous": amb_flags[-1],
                "rot_deg": rot[-1], "trans_sq": trans[-1], "drot_rad": dr.tolist(), "dt_sq": dt.tolist()}) + "\n")
        # NEES: e^T SIGMA^-1 e with e = [delta_rvec; delta_t]. If the reported
        # covariance is honest this is chi-square_6 distributed -> MEAN 6.0.
        # << 6 means over-cautious (a filter would under-weight good fixes);
        # >> 6 means OVER-CONFIDENT, which is how a Kalman filter diverges.
        C = res.get("pose_cov")
        if C is not None:
            try:
                v = float(e @ np.linalg.solve(np.array(C), e))
                (nees_amb if res.get("ambiguous") else nees).append(v)
            except np.linalg.LinAlgError:
                pass
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(labels)} solved={len(rot)} refused={refused}", flush=True)

    if per_image is not None:
        per_image.close()
    r, t = np.array(rot), np.array(trans)
    nz, na = np.array(nees), np.array(nees_amb)
    E, A, S = np.array(errs).reshape(-1, 6), np.array(amb_flags, dtype=bool), np.array(s_px, dtype=np.float64)
    arm = "coarse-only" if args.no_refiner else "with-refiner"
    out = {"pose_set": str(pose_dir), "arm": arm, "ckpt": args.ckpt, "ckpt_step": ck.get("step"),
           "refiner_ckpt": None if args.no_refiner else args.refiner_ckpt,
           "refiner_step": None if rk is None else rk.get("step"),
           "n_images": len(labels), "n_solved": len(rot), "n_refused": refused,
           "refusal_reasons": reasons, "n_ambiguous": int(ambiguous),
           "solve_rate": len(rot) / max(len(labels), 1),
           "m05_rot_deg": {"median": float(np.median(r)), "mean": float(r.mean()),
                            "p95": float(np.percentile(r, 95))} if len(r) else None,
           "m06_trans_squares": {"median": float(np.median(t)), "mean": float(t.mean()),
                                  "p95": float(np.percentile(t, 95))} if len(t) else None,
           "nees": {"expected_chi2_dof": 6, "n_unambiguous": int(len(nz)), "n_ambiguous": int(len(na)),
                     "mean_unambiguous": float(nz.mean()) if len(nz) else None,
                     "median_unambiguous": float(np.median(nz)) if len(nz) else None,
                     "mean_ambiguous": float(na.mean()) if len(na) else None,
                     "frac_within_chi2_95": float((nz <= 12.592).mean()) if len(nz) else None},
           "error_cov": _error_cov(E, A, S) if len(E) >= 2 else None}
    print(f"\n[{arm}] {len(rot)}/{len(labels)} solved ({100*out['solve_rate']:.1f}%), "
          f"{refused} refused {reasons}, {ambiguous} flagged ambiguous")
    if len(nz):
        print(f"  NEES (unambiguous, n={len(nz)}): mean {nz.mean():.3f} median {np.median(nz):.3f} "
              f"vs chi2_6 expectation 6.000 -> {'OVER-confident' if nz.mean() > 6 else 'conservative'} "
              f"by {max(nz.mean()/6, 6/max(nz.mean(),1e-9)):.2f}x")
        print(f"  frac within chi2_6 95% bound (12.592): {(nz <= 12.592).mean()*100:.1f}% (ideal 95%)")
    if len(na):
        print(f"  NEES (AMBIGUOUS poses, n={len(na)}): mean {na.mean():.3f} -- single-Gaussian cov "
              f"cannot describe a bimodal posterior; expect this to be large")
    if out["error_cov"] and "unambiguous" in out["error_cov"]:
        u = out["error_cov"]["unambiguous"]
        print(f"  error std (unambiguous, n={u['n']}) rot xyz deg {np.degrees(u['std'][:3]).round(4).tolist()} "
              f"trans xyz sq {np.round(u['std'][3:], 5).tolist()}  | robust (MAD) rot deg "
              f"{np.degrees(np.sqrt(u['var_mad'][:3])).round(4).tolist()} trans sq {np.round(np.sqrt(u['var_mad'][3:]), 5).tolist()}")
    if len(r):
        print(f"  M-05 rotation    median {np.median(r):.4f} deg  mean {r.mean():.4f}  p95 {np.percentile(r,95):.4f}")
        print(f"  M-06 translation median {np.median(t):.4f} sq   mean {t.mean():.4f}  p95 {np.percentile(t,95):.4f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
