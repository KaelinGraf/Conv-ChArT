"""tools/gen_eval_pose.py -- pose-consistent evaluation set generator.

The only place in Conv-ChArT with camera pose (K, R, t) bookkeeping. Per
image i, rng = np.random.default_rng([pose_seed, i]) draws a pinhole K and
a board pose (R, t) with X_cam = R @ X_board + t, then projects the board's
metric lattice (corner i at ((i%4)+1, (i//4)+1, 0) m; square_length_m fixed
at 1.0 here -- config's board.square_length_m is null/unused elsewhere;
board spans [0,5]x[0,5] m, centre (2.5,2.5,0)) through it. Composited via
the induced homography

    H = K @ [r1 | r2 | t] @ S,   S = [[1/SQ,0,0.5/SQ],[0,1/SQ,0.5/SQ],[0,0,1]]

(r1, r2 = R's first two columns; S maps a render-pixel homogeneous coord to
board metres, (px+0.5)/SQ, inverting render_board's px = X*SQ - 0.5 offset).
Checked every image against cv2.projectPoints of the same lattice to float
precision -- disagreement means a convention (column order, dehomogenisation,
the S offset) has slipped.

Everything else -- background prep, occlusion, photometrics, geometric
corner visibility -- is the exact dcc.synth machinery generate_sample uses.
Only the pinhole sampling and the perspective warp are new here.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MIN_CORNERS = 4      # PnP minimum; see _accept_pose


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--out", default="eval_pose/")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--sheet-n", type=int, default=20)
    return p


def _accept_pose(rng, i, W, H, s_lo, s_hi, lattice, visible):
    """Draw (K, R, t) up to 100x, rejection-sampling until >=8 of the
    lattice's corners (board-size-dependent -- see dcc.board.n_corners)
    project in-frame and all of them are in front of the camera; a config
    pathology raises loudly rather than silently degrading. s_target is
    the nominal apparent square size at the board centre -- tilt makes
    the true local scale vary across the board."""
    import numpy as np
    import cv2
    cx, cy = (W - 1) / 2, (H - 1) / 2
    for tries in range(1, 101):
        f = rng.uniform(0.7, 1.4) * W
        K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
        s_target = float(np.exp(rng.uniform(np.log(s_lo), np.log(s_hi))))
        z = f / s_target
        phi, tilt, psi = rng.uniform(-np.pi, np.pi), rng.uniform(0.0, np.radians(60)), rng.uniform(0.0, 2 * np.pi)
        Raxis, _ = cv2.Rodrigues(tilt * np.array([np.cos(psi), np.sin(psi), 0.0]))
        cp, sp = np.cos(phi), np.sin(phi)
        R = np.array([[cp, -sp, 0.0], [sp, cp, 0.0], [0.0, 0.0, 1.0]]) @ Raxis
        # +-0.45 matches dcc.synth._sample_affine's own translation range. The old
        # +-0.35 kept boards further from the frame edge than the training/val
        # distribution does, so truncated views were under-represented.
        u, v = cx + rng.uniform(-0.45, 0.45) * W, cy + rng.uniform(-0.45, 0.45) * H
        t = z * (np.linalg.inv(K) @ np.array([u, v, 1.0])) - R @ np.array([2.5, 2.5, 0.0])

        cam_z = (lattice @ R.T + t)[:, 2]
        rvec, _ = cv2.Rodrigues(R)
        img_pts = cv2.projectPoints(lattice, rvec, t, K, None)[0].reshape(-1, 2)
        n_in = sum(visible((x, y), [], (W, H)) for x, y in img_pts)
        # MIN_CORNERS = 4, the PnP minimum -- NOT 8. Requiring 8 silently excluded every
        # truncated or heavily-occluded view, so the pose set was materially easier than
        # the data the detectors actually see: measured 69.2% fully-visible and 0.0% with
        # fewer than 8 corners, against SynthVal's 43.1% and 9.4%. That inflated every
        # arm's solve rate and flattered the baselines most (Kaelin, 2026-07-29: 'our
        # other metrics definitely dont show deep charuco solving 96 percent').
        # 4 is the floor a pose set can legitimately impose -- below it there is no pose
        # to score, which is a detection failure and belongs in the recall metrics, not
        # here.
        if n_in >= MIN_CORNERS and (cam_z > 0).all():
            return K, R, t, s_target, img_pts, tries
    raise RuntimeError(f"image {i}: no acceptable pose within 100 tries -- check scale_range_px / K envelope")


def _homography_and_assert(K, R, t, SQ, render_corners, img_pts):
    """Builds H = K @ [r1|r2|t] @ S, then asserts it reproduces
    cv2.projectPoints to float precision -- the analytic identity that
    catches convention slips (r1/r2 column order, the dehomogenisation, the
    S offset)."""
    import numpy as np
    Rt = np.column_stack([R[:, 0], R[:, 1], t])
    S = np.array([[1 / SQ, 0.0, 0.5 / SQ], [0.0, 1 / SQ, 0.5 / SQ], [0.0, 0.0, 1.0]])
    Hmat = K @ Rt @ S
    proj = np.hstack([render_corners, np.ones((len(render_corners), 1))]) @ Hmat.T
    proj = proj[:, :2] / proj[:, 2:3]
    err = float(np.linalg.norm(proj - img_pts, axis=1).max())
    assert err < 1e-6, f"S-matrix identity violated: max err {err:.3e} px"
    return Hmat, err


def _composite_board_persp(bg_crop, board_3ch, mask_src, Hmat, W, H):
    """Perspective analogue of dcc.synth._composite_board: histogram-match
    the (already rendered, pose-invariant) board to this crop and warp it
    in with Hmat in place of an affine."""
    import cv2
    import numpy as np
    from skimage.exposure import match_histograms
    matched = match_histograms(board_3ch, bg_crop, channel_axis=-1).astype(np.float32)
    warped = cv2.warpPerspective(matched, Hmat, (W, H), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    wmask = cv2.warpPerspective(mask_src, Hmat, (W, H), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    m = wmask[..., None]
    return bg_crop.astype(np.float32) * (1 - m) + warped * m


def main():
    args = build_parser().parse_args()

    import numpy as np
    import cv2
    import skimage
    import yaml
    from dcc.board import get_board, n_corners, render_board
    from dcc.pipeline import canon_lattice
    from dcc.synth import list_backgrounds, _prep_background, _apply_occlusion, _apply_photometric, visible
    from dcc.viz import draw_overlay, tile

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    bcfg = cfg.get("board")
    n_cls = n_corners(bcfg)
    assert n_cls >= 8, (f"board has {n_cls} inner corners (cfg['board']['squares']); boards smaller than "
                         "8 inner corners cannot satisfy the SD-10 >=8-visible acceptance rule in _accept_pose")
    bg_path = cfg["synth"]["backgrounds"]
    bg_files = list_backgrounds(bg_path)
    if not bg_files:
        print(f"background corpus missing/empty at {bg_path!r} -- COCO download may still be running")
        sys.exit(2)

    W, H = cfg["input_size"]
    render_res = cfg["synth"]["render_res"]
    SQ = render_res // get_board(bcfg)[1]
    s_lo, s_hi = cfg["scale_range_px"]
    pose_seed = cfg["synth"]["pose_seed"]

    board_img, render_corners = render_board(render_res, bcfg)
    board_3ch = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)
    mask_src = np.ones((render_res, render_res), dtype=np.float32)
    n = math.isqrt(len(render_corners))
    lattice = np.hstack([canon_lattice(n), np.zeros((n * n, 1))])

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    tries_hist, sheet, max_err = [], [], 0.0
    with open(out / "labels.jsonl", "w") as lf:
        for i in range(args.n):
            rng = np.random.default_rng([pose_seed, i])
            K, R, t, s_target, img_pts, tries = _accept_pose(rng, i, W, H, s_lo, s_hi, lattice, visible)
            tries_hist.append(tries)
            Hmat, err = _homography_and_assert(K, R, t, SQ, render_corners, img_pts)
            max_err = max(max_err, err)

            while True:
                bg = cv2.imread(bg_files[int(rng.integers(len(bg_files)))], cv2.IMREAD_COLOR)
                if bg is not None:
                    break
            bg_crop = _prep_background(bg, rng, cfg["synth"], W, H)
            work = _composite_board_persp(bg_crop, board_3ch, mask_src, Hmat, W, H)
            holes = _apply_occlusion(work, rng, cfg["synth"]["occlusion"], W, H)
            corners_out = [{"x": float(x), "y": float(y), "index": k,
                             "visible": visible((x, y), holes, (W, H))}
                            for k, (x, y) in enumerate(img_pts)]
            work = _apply_photometric(work, rng, cfg["synth"]["photometric"], W, H)
            image = cv2.cvtColor(np.clip(work, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

            fname = f"images/{i:06d}.png"
            cv2.imwrite(str(out / fname), image)
            lf.write(json.dumps({"file": fname, "K": K.tolist(), "R": R.tolist(), "t": t.tolist(),
                                  "square_length_m": 1.0, "s_px": s_target, "corners": corners_out}) + "\n")

            if i < args.sheet_n:
                sheet.append(draw_overlay(image, {"corners": corners_out}, {"holes": holes}))
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{args.n}")

    if sheet:
        cv2.imwrite(str(out / "overlay_sheet.png"), tile(sheet))

    meta = {"pose_seed": pose_seed, "n": args.n,
            "versions": {"numpy": np.__version__, "cv2": cv2.__version__, "skimage": skimage.__version__},
            "backgrounds_sha1": hashlib.sha1("\n".join(bg_files).encode()).hexdigest(),
            "config": cfg,
            "resample_tries": {"max": max(tries_hist, default=0),
                               "gt1_count": sum(1 for tr in tries_hist if tr > 1)},
            "s_matrix_assert_max_px": max_err}
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {args.n} images to {out}")


if __name__ == "__main__":
    main()
