"""tools/eval_classical.py -- A13: classical full-pipeline ChArUco baseline
across the s x SNR envelope (see ablations.md A13). Two independent classical
arms, both cv2-only:
  (a) ID'd arm: cv2.aruco.detectMarkers + interpolateCornersCharuco (+
      solvePnP) -- the standard OpenCV ChArUco pipeline, tuned
      adaptive-threshold parameters (see _detector_params).
  (b) anonymous-corner arm: goodFeaturesToTrack + cornerSubPix -- the sub-9px
      inset probe (ID-less corner localisation persisting below the
      marker-legibility floor, per A13's "honest inset" claim).

Board-convention self-test (verify_board_convention): dcc/board.py's corner
index i comes straight from cv2's own board.getChessboardCorners() (no
reindexing), so classical charucoIds returned against the SAME board object
(via dcc.board.get_board) are automatically in dcc's own row-major
convention -- this function proves that empirically on a clean render rather
than assuming it, and is run as a startup gate here and as
tests/test_eval_classical.py::test_board_convention.

Two input modes:
  --grid            s x SNR-tier sweep on dcc.synth.generate_sample frontal
                     frames (memory.md 2026-07-28 01:10's too-small-sweep
                     protocol: tilt <= TILT_MAX_RAD, FRAMES_PER_POINT frames
                     per point). No camera K in this mode -- ID/localisation
                     only, no pose.
  --pose-set <dir>   a tools/gen_eval_pose.py output directory: full ID,
                     corner-localisation, and pose scoring against its
                     labels.jsonl K/R/t ground truth.
Both may be combined in one invocation; results merge into one JSON.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import yaml

from dcc.board import get_board, n_corners, render_board
from dcc.pipeline import canon_lattice
from dcc.synth import generate_sample, list_backgrounds

S_GRID = [6, 8, 10, 12, 16, 20, 25, 32, 48, 64, 96, 128]
TIERS = ("clean", "mid", "deployment-dark", "overcast")
TILT_MAX_RAD = 0.08          # frontal-sweep bound, memory.md 2026-07-28 01:10
FRAMES_PER_POINT = 15        # 5->15 (team-lead 2026-07-28): cheap smoothing at ~0.29s/frame,
                              # and diagnoses whether small-n noise (e.g. clean s=64's 77.6%) persists
ID_MATCH_TOL_PX = 5.0        # detected charucoID xy vs GT xy at that same index
ANON_MATCH_TOL_PX = 3.0      # anonymous corner vs nearest GT junction


def _detector_params():
    """Adaptive-threshold + refinement tuning for the small/dim markers this
    sweep deliberately pushes into: cv2's defaults (winSize 3-23 step 10,
    minMarkerPerimeterRate 0.03) target well-lit, moderate-scale boards.
    Step tightened 10->4 (finer threshold search across the s range swept in
    one process); minMarkerPerimeterRate loosened 0.03->0.01 so the
    marker-legibility floor this tool measures is the sensor/aliasing limit,
    not an artificial perimeter gate (a marker at s=25px module spans
    single-digit pixels at native 1600x1200, well under the default's
    ~12px/side implied floor). Sub-pixel corner refinement on so both the
    marker-corner seed for interpolateCornersCharuco and this tool's own
    localisation-error metric are refined, not integer-snapped."""
    p = cv2.aruco.DetectorParameters()
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 23
    p.adaptiveThreshWinSizeStep = 4
    p.minMarkerPerimeterRate = 0.01
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.cornerRefinementWinSize = 5
    p.cornerRefinementMaxIterations = 30
    p.cornerRefinementMinAccuracy = 0.05
    return p


def _get_board_and_dictionary(bcfg=None):
    """The board object dcc.synth/dcc.board themselves render from -- NOT a
    freshly constructed one -- so classical charucoIds land in dcc's own
    corner-index convention by construction (see module docstring).
    setLegacyPattern(False) is pinned explicitly rather than relied on as
    cv2 4.10's own default (verified: getLegacyPattern() is already False
    out of the box, matching dcc/board.py, which never calls
    setLegacyPattern -- pinning it here nails the convention down against a
    future cv2 default change, per SD-12)."""
    board, nx = get_board(bcfg)
    board.setLegacyPattern(False)
    return board, board.getDictionary(), nx


def classical_charuco_detect(gray, board, dictionary, params):
    """detectMarkers + interpolateCornersCharuco -- the standard OpenCV
    ChArUco arm. Returns (xy (N,2) float64, ids (N,) int) or (None, None) if
    nothing interpolates. ids are never a "no detection" sentinel -- every
    id interpolateCornersCharuco returns is a genuine detection."""
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    marker_corners, marker_ids, _ = detector.detectMarkers(gray)
    if marker_ids is None or len(marker_ids) == 0:
        return None, None
    n_ch, ch_xy, ch_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray, board)
    if n_ch == 0:
        return None, None
    return ch_xy.reshape(-1, 2).astype(np.float64), ch_ids.reshape(-1).astype(int)


def anonymous_corners(gray, max_corners=200, quality=0.02, min_dist=4):
    """goodFeaturesToTrack + cornerSubPix -- ID-less corner localisation, the
    sub-9px inset arm (A13's "honest inset": classical still finds anonymous
    corners below where marker decoding, and the learned model, both quit)."""
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=quality, minDistance=min_dist)
    if pts is None:
        return np.zeros((0, 2), dtype=np.float64)
    pts = pts.reshape(-1, 2).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    cv2.cornerSubPix(gray, pts, (5, 5), (-1, -1), crit)
    return pts.astype(np.float64)


def verify_board_convention(bcfg=None, render_res=480):
    """Startup self-test: on a clean board render, classical charucoIds must
    land at dcc.board.render_board's own analytic corner_px(i). A mismatch
    here invalidates every ID comparison downstream -- see module docstring.
    Returns (max_err_px, n_matched); raises AssertionError on failure."""
    board, dictionary, nx = _get_board_and_dictionary(bcfg)
    img, analytic_corners = render_board(render_res, bcfg)
    ch_xy, ch_ids = classical_charuco_detect(img, board, dictionary, _detector_params())
    n_expected = n_corners(bcfg)
    assert ch_xy is not None, "convention self-test: classical detector found nothing on a clean board render"
    assert len(ch_ids) == n_expected, (
        f"convention self-test: interpolated {len(ch_ids)}/{n_expected} corners on a clean render "
        "(expected all -- a clean render should be the easiest possible frame)")
    err = np.linalg.norm(ch_xy - analytic_corners[ch_ids], axis=1)
    max_err = float(err.max())
    assert max_err < 1.0, (
        f"convention self-test FAILED: max err {max_err:.3f}px between classical charucoIds and "
        "dcc.board's analytic corner_px(i) -- ID ordering mismatch, do not trust any ID comparison below")
    return max_err, len(ch_ids)


def _match_ided(gt_visible, det_xy, det_ids, tol=ID_MATCH_TOL_PX):
    """gt_visible: {index: (x, y)}. charucoIds already carry identity (no
    nearest-neighbour ambiguity needed, unlike the anonymous arm) -- each
    detection is checked directly against its own index's GT position.
    n_wrong counts a detected id that's either not in gt_visible at all
    (off-board/false marker read, or a GT-occluded index) or whose xy lands
    > tol from that index's GT position (a genuine wrong-ID misread, not
    just localisation noise). Returns (n_correct, n_wrong, loc_errs)."""
    n_correct, n_wrong, loc_errs = 0, 0, []
    if det_ids is None:
        return 0, 0, []
    for xy, idx in zip(det_xy, det_ids):
        gt = gt_visible.get(int(idx))
        if gt is None:
            n_wrong += 1
            continue
        d = float(np.hypot(xy[0] - gt[0], xy[1] - gt[1]))
        if d <= tol:
            n_correct += 1
            loc_errs.append(d)
        else:
            n_wrong += 1
    return n_correct, n_wrong, loc_errs


def _match_anonymous(gt_visible, anon_xy, tol=ANON_MATCH_TOL_PX):
    """Greedy nearest-neighbour match, each GT corner claimed at most once:
    ID-less detections carry no identity, so this is a pure localisation
    check -- does SOME detected point land near this GT junction. Returns
    (n_matched, loc_errs)."""
    gt_pts = np.array(list(gt_visible.values()), dtype=np.float64) if gt_visible else np.zeros((0, 2))
    if len(gt_pts) == 0 or len(anon_xy) == 0:
        return 0, []
    claimed, loc_errs = set(), []
    for gx, gy in gt_pts:
        d = np.hypot(anon_xy[:, 0] - gx, anon_xy[:, 1] - gy)
        for j in np.argsort(d):
            j = int(j)
            if d[j] > tol:
                break
            if j not in claimed:
                claimed.add(j)
                loc_errs.append(float(d[j]))
                break
    return len(loc_errs), loc_errs


def _zero_all_probabilities(ph_cfg):
    return {k: (0.0 if k.endswith("_p") else v) for k, v in ph_cfg.items()}


def build_tier_cfgs(cfg):
    """SNR tiers per A13 (ablations.md: "clean, mid (sigma~=4 + mild blur),
    deployment-dark (post rev-2 generator)"), calibrated against memory.md
    2026-07-28 09:05's three measured rev-3 illumination regimes (NIGHT
    ambient 0.05-0.15, OVERCAST ambient 0.4-0.55/peak 0.3-0.6, MIDDAY
    ambient 0.8-0.95/peak 0.15-0.3) and the team-lead's 2026-07-28 gate-4
    ruling on this tool's first-pass results:

    - clean: photometric off entirely (generate_sample photometric=False).
    - mid: an ISOLATED synthetic probe -- sigma~=4 Gaussian noise + blur,
      every other photometric effect explicitly zeroed, per the spec's own
      parenthetical definition.
    - deployment-dark: "post rev-2 generator" -- the FULL standard
      photometric stack at its normal probabilities (vignette, specular,
      sensor noise, speckle, etc.), differencing forced on
      (differencing_p=1.0) and biased into the MIDDAY regime -- the actual
      daytime rig physics (LED barely above a 90% daylight pedestal,
      residual SNR ~1.5-3), and the regime the A13 collapse expectation was
      written against.
    - overcast: same full-stack construction, differencing forced into the
      OVERCAST regime instead. Kept as its own reported tier rather than
      folded into deployment-dark: OVERCAST's residual SNR (~1, peak
      comparable to ambient) is genuinely survivable for classical --
      measured NOT collapsing here (this tool's first-pass smoke run,
      2026-07-28: 92.4% ID rate at s=64, actually above clean's 77.6% --
      the board-centred illumination lobe acts as a spotlight +
      background-suppression effect, which HELPS classical detection since
      background clutter is its main failure mode). That's a real,
      reportable result, not a miscalibration to paper over -- see that
      commit's report for the two earlier bugs (NIGHT-range confusion, then
      wrongly zeroing the rest of the stack) this tier's construction
      already fixed.
    Returns {tier: (cfg_for_generate_sample, photometric_bool)}."""
    base_ph = cfg["synth"]["photometric"]

    mid_ph = dict(_zero_all_probabilities(base_ph), gauss_noise_p=1.0, sensor_noise_enabled=False,
                  noise_std=[4.0, 4.0], gauss_blur_p=1.0)
    mid_cfg = copy.deepcopy(cfg)
    mid_cfg["synth"]["photometric"] = mid_ph

    dark_ph = dict(base_ph, differencing_p=1.0,
                   differencing_ambient=[0.8, 0.95], differencing_illum_peak=[0.15, 0.3])
    dark_cfg = copy.deepcopy(cfg)
    dark_cfg["synth"]["photometric"] = dark_ph

    overcast_ph = dict(base_ph, differencing_p=1.0,
                        differencing_ambient=[0.4, 0.55], differencing_illum_peak=[0.3, 0.6])
    overcast_cfg = copy.deepcopy(cfg)
    overcast_cfg["synth"]["photometric"] = overcast_ph

    return {"clean": (cfg, False), "mid": (mid_cfg, True),
            "deployment-dark": (dark_cfg, True), "overcast": (overcast_cfg, True)}


def run_grid(cfg, bg_files, s_values, tiers, frames_per_point, seed, params):
    board, dictionary, nx = _get_board_and_dictionary(cfg.get("board"))
    tier_cfgs = build_tier_cfgs(cfg)
    cells = []
    for s in s_values:
        for tier in tiers:
            tier_cfg, use_photometric = tier_cfgs[tier]
            n_gt = n_anon_det = n_id_correct = n_id_wrong = 0
            anon_errs, id_errs = [], []
            for frame_idx in range(frames_per_point):
                rng = np.random.default_rng([seed, s, TIERS.index(tier), frame_idx])
                tilt = float(rng.uniform(0.0, TILT_MAX_RAD))
                record, _ = generate_sample(tier_cfg, rng, bg_files, s=s, force_negative=False,
                                             photometric=use_photometric, occlude=False,
                                             components={"tilt": tilt})
                gray = record["image"]
                gt_visible = {c["index"]: (c["x"], c["y"]) for c in record["corners"] if c["visible"]}
                n_gt += len(gt_visible)

                anon_xy = anonymous_corners(gray)
                n_matched, a_errs = _match_anonymous(gt_visible, anon_xy)
                n_anon_det += n_matched
                anon_errs += a_errs

                det_xy, det_ids = classical_charuco_detect(gray, board, dictionary, params)
                if det_xy is not None:
                    n_correct, n_wrong, i_errs = _match_ided(gt_visible, det_xy, det_ids)
                    n_id_correct += n_correct
                    n_id_wrong += n_wrong
                    id_errs += i_errs

            cells.append({
                "s": s, "tier": tier, "n_frames": frames_per_point, "gt_corners": n_gt,
                "anon_det_rate": n_anon_det / max(n_gt, 1),
                "anon_loc_err_median": float(np.median(anon_errs)) if anon_errs else None,
                "anon_loc_err_p90": float(np.percentile(anon_errs, 90)) if anon_errs else None,
                "id_rate": n_id_correct / max(n_gt, 1),
                "id_wrong": n_id_wrong,
                "id_loc_err_median": float(np.median(id_errs)) if id_errs else None,
                "id_loc_err_p90": float(np.percentile(id_errs, 90)) if id_errs else None,
            })
    return cells


def _solve_pnp_classical(xy, ids, K, square_length_m, n):
    """Plain cv2.solvePnP (SOLVEPNP_ITERATIVE) on the ID'd set -- the
    classical-baseline PnP call named in the A13 spec, deliberately NOT
    dcc.pipeline.pnp's own solvePnPGeneric/IPPE (that call tests OUR solver
    choice; this arm tests the standard OpenCV recipe end-to-end). Object
    points = canon_lattice(n) * square_length_m, z=0 -- identical convention
    to dcc.pipeline.pnp. Returns (rvec, tvec, n_used); rvec is None on <4
    correspondences or solver failure."""
    n_used = 0 if ids is None else len(ids)
    if n_used < 4:
        return None, None, n_used
    lattice = canon_lattice(n)
    obj = (np.hstack([lattice[ids], np.zeros((n_used, 1))]) * square_length_m).reshape(-1, 1, 3)
    img = np.asarray(xy, dtype=np.float64).reshape(-1, 1, 2)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None, n_used
    return rvec, tvec, n_used


def _pose_error(rvec_est, tvec_est, R_gt, t_gt):
    """Rotation error as the geodesic angle between R_gt and R_est (standard
    trace formula); translation error as the raw Euclidean norm -- pose-eval
    sets fix square_length_m=1.0 (tools/gen_eval_pose.py), so this is
    unitless in "board squares", not metres."""
    R_est, _ = cv2.Rodrigues(rvec_est)
    cos_theta = (np.trace(np.asarray(R_gt).T @ R_est) - 1.0) / 2.0
    rot_err_deg = float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))
    trans_err = float(np.linalg.norm(np.asarray(tvec_est).reshape(3) - np.asarray(t_gt).reshape(3)))
    return rot_err_deg, trans_err


def run_pose_set(pose_dir, params):
    """--pose-set mode: score against a tools/gen_eval_pose.py output dir's
    own labels.jsonl (K/R/t/corners) and its own recorded generator config
    (meta.json["config"]) -- NOT the caller's --config, since a pose set may
    have been generated under an older config than the one currently on
    disk (board convention must match what actually rendered the images)."""
    pose_dir = Path(pose_dir)
    with open(pose_dir / "meta.json") as f:
        meta = json.load(f)
    pose_cfg = meta["config"]
    board, dictionary, nx = _get_board_and_dictionary(pose_cfg.get("board"))
    n = nx - 1

    n_gt = n_id_correct = n_id_wrong = n_pose_success = n_images = 0
    id_errs, rot_errs, trans_errs = [], [], []
    with open(pose_dir / "labels.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            n_images += 1
            gray = cv2.imread(str(pose_dir / rec["file"]), cv2.IMREAD_GRAYSCALE)
            gt_visible = {c["index"]: (c["x"], c["y"]) for c in rec["corners"] if c["visible"]}
            n_gt += len(gt_visible)

            det_xy, det_ids = classical_charuco_detect(gray, board, dictionary, params)
            if det_xy is None:
                continue
            n_correct, n_wrong, i_errs = _match_ided(gt_visible, det_xy, det_ids)
            n_id_correct += n_correct
            n_id_wrong += n_wrong
            id_errs += i_errs

            K = np.array(rec["K"])
            rvec, tvec, n_used = _solve_pnp_classical(det_xy, det_ids, K, rec["square_length_m"], n)
            if rvec is None:
                continue
            rot_e, trans_e = _pose_error(rvec, tvec, np.array(rec["R"]), rec["t"])
            rot_errs.append(rot_e)
            trans_errs.append(trans_e)
            n_pose_success += 1

    def med(a):
        return float(np.median(a)) if a else None

    def p90(a):
        return float(np.percentile(a, 90)) if a else None

    return {
        "dir": str(pose_dir), "n_images": n_images, "gt_corners": n_gt,
        "id_rate": n_id_correct / max(n_gt, 1), "id_wrong": n_id_wrong,
        "id_loc_err_median": med(id_errs), "id_loc_err_p90": p90(id_errs),
        "pose_success_rate": n_pose_success / max(n_images, 1),
        "rot_err_deg_median": med(rot_errs), "rot_err_deg_p90": p90(rot_errs),
        "trans_err_median": med(trans_errs), "trans_err_p90": p90(trans_errs),
    }


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--pose-set", default=None, help="tools/gen_eval_pose.py output dir")
    p.add_argument("--grid", action="store_true", help="run the s x SNR-tier sweep")
    p.add_argument("--s-values", type=int, nargs="+", default=S_GRID)
    p.add_argument("--tiers", nargs="+", default=list(TIERS), choices=list(TIERS))
    p.add_argument("--frames-per-point", type=int, default=FRAMES_PER_POINT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="eval_classical/results.json")
    return p


def main():
    args = build_parser().parse_args()
    if not args.grid and not args.pose_set:
        print("nothing to do: pass --grid and/or --pose-set <dir>")
        sys.exit(2)

    t0 = time.time()
    print("running board-convention self-test...")
    max_err, n_matched = verify_board_convention()
    print(f"  convention OK: {n_matched}/{n_matched} corners, max err {max_err:.4f}px")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    params = _detector_params()

    results = {
        "seed": args.seed,
        "convention_self_test": {"max_err_px": max_err, "n_matched": n_matched},
        "detector_params": {
            "adaptiveThreshWinSizeMin": params.adaptiveThreshWinSizeMin,
            "adaptiveThreshWinSizeMax": params.adaptiveThreshWinSizeMax,
            "adaptiveThreshWinSizeStep": params.adaptiveThreshWinSizeStep,
            "minMarkerPerimeterRate": params.minMarkerPerimeterRate,
            "cornerRefinementMethod": "CORNER_REFINE_SUBPIX",
        },
    }

    if args.grid:
        bg_files = list_backgrounds(cfg["synth"]["backgrounds"])
        if not bg_files:
            print(f"background corpus missing/empty at {cfg['synth']['backgrounds']!r}")
            sys.exit(2)
        cells = run_grid(cfg, bg_files, args.s_values, args.tiers, args.frames_per_point, args.seed, params)
        results["grid"] = cells
        print(f"\n{'s':>5} {'tier':>16} {'anon_det%':>10} {'anon_px':>8} {'id%':>7} {'id_px':>8}")
        for c in cells:
            anon_px = c["anon_loc_err_median"]
            id_px = c["id_loc_err_median"]
            print(f"{c['s']:>5} {c['tier']:>16} {c['anon_det_rate'] * 100:>9.1f}% "
                  f"{anon_px if anon_px is not None else float('nan'):>8.3f} "
                  f"{c['id_rate'] * 100:>6.1f}% "
                  f"{id_px if id_px is not None else float('nan'):>8.3f}")

    if args.pose_set:
        pose_result = run_pose_set(args.pose_set, params)
        results["pose_set"] = pose_result
        print(f"\npose-set {args.pose_set}: n={pose_result['n_images']} "
              f"id_rate={pose_result['id_rate'] * 100:.1f}% "
              f"pose_success={pose_result['pose_success_rate'] * 100:.1f}% "
              f"rot_med={pose_result['rot_err_deg_median']} trans_med={pose_result['trans_err_median']}")

    results["wall_clock_s"] = time.time() - t0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
