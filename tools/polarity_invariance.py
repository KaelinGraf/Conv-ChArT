"""tools/polarity_invariance.py -- team-lead 2026-07-28 top-priority
follow-up to the contrast-inversion mechanism investigation: does our
learned detector actually READ inverted-contrast (differencing-clipped)
boards, or does it fail on them just like classical (which structurally
cannot, its bit test is hard-wired to "white bit = high value")? The live
trainer's M-04 (98.37% at the in-training step-160k checkpoint -- NOT final
weights, see --ckpt below) is hard to reconcile with an inverted fraction of
10.76% of training samples, p10 contrast -28 DN (tools/inversion_audit.py's
corrected Gaussian-lobe measurement, superseding an earlier 13.6% run that
had a uniform-peak bug) UNLESS the model has learned polarity invariance --
this script settles it directly: partition a val sample by MEASURED board
contrast polarity (same mask method as tools/inversion_audit.py, applied
post-hoc to whatever each sample's own random photometric draw produced --
no forcing), and report M-01/M-02/M-04 separately per partition.

Val sampling note: dcc.dataset.SynthVal.__getitem__ discards generate_
sample's own `meta` dict (only returns image, record) -- meta["M"] (the
board homography) is required to build the polarity mask, so this script
calls dcc.synth.generate_sample directly, reproducing SynthVal's own
stratified-s sampling recipe (dcc/*, forbidden to edit, is unmodified;
this is a read-only, faithful reproduction of its __getitem__ logic).

GPU discipline (team-lead 2026-07-28, standing concurrency rules): B=1
eval passes only, trainer health re-checked before starting and periodically
during the sweep, small subset by default (--n 400).
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

import factor_sweep as fs
import inversion_audit as ia
from dcc.board import render_board
from dcc.pipeline import detect as dcc_detect
from dcc.synth import generate_sample, list_backgrounds


def _load_train_detector():
    """tools.train_detector, loaded by absolute path -- an unrelated
    detectron2 'tools' package shadows any `import tools.train_detector`
    (identical footgun/fixture to tools/train_charuconet.py's own
    _load_train_detector, reused here for the exact same reason: pure
    OCTAVE_BINS/_octave_bucket/_match_greedy/TAIL_PX helpers, never
    run_validation, which is DetectorNet-training-loop-specific)."""
    spec = importlib.util.spec_from_file_location("_train_detector_ref", ROOT / "tools" / "train_detector.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    # Required, NO default (team-lead 2026-07-28): this measurement's whole
    # value depends on which weights produced it -- the M-04=98.37% figure
    # it's being compared against is from the in-training step-160k
    # checkpoint, not final weights, and the paper-facing number MUST come
    # from final weights. A convenient default here is exactly the kind of
    # thing that gets silently reused after the run finishes; forcing
    # --ckpt to be spelled out every invocation removes that failure mode.
    p.add_argument("--ckpt", required=True, help="detector checkpoint -- MUST be the weights this "
                    "measurement is meant to characterise, e.g. the final training checkpoint; no default")
    p.add_argument("--refiner-ckpt", default="runs/refiner_v1/ckpt_0010000.pt")
    p.add_argument("--config", default=None)
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--seed", type=int, default=None, help="default: cfg's own val_seed")
    p.add_argument("--match-px", type=float, default=8.0)
    p.add_argument("--erode-px", type=int, default=5)
    return p


def _val_sample(cfg, bg_files, val_seed, n, i):
    """Faithful, read-only reproduction of dcc.dataset.SynthVal.__getitem__
    (see module docstring) -- returns (record, meta), keeping meta (SynthVal
    itself discards it)."""
    rng = np.random.default_rng([val_seed, i])
    if rng.random() < cfg["negative_p"]:
        return generate_sample(cfg, rng, bg_files, force_negative=True)
    a, b = cfg["scale_range_px"]
    s = a * (b / a) ** ((i + rng.random()) / n)
    return generate_sample(cfg, rng, bg_files, s=s, force_negative=False)


def _measure_polarity(record, meta, board_img, erode_px):
    """None if not a positive/board-present sample or the board is too small
    to measure reliably; else (contrast, inverted)."""
    if not record["board_present"] or meta["M"] is None:
        return None
    H, w2 = meta["M"], record["image"].shape[1]
    h2 = record["image"].shape[0]
    white_mask, black_mask = ia._white_black_masks_canvas(board_img, H, w2, h2, erode_px)
    if white_mask.sum() < 20 or black_mask.sum() < 20:
        return None
    gray = record["image"].astype(np.float64)
    return float(np.median(gray[white_mask]) - np.median(gray[black_mask]))


def _score_sample(model, refiner, cfg, ref, record, match_px):
    """(loc_errs, matched_octave_bucket_hits, id_correct_count, id_total_count)
    for ONE sample -- M-01/M-02/M-04's per-sample contribution, computed
    with ref._octave_bucket/_match_greedy (tools/train_detector.py's pure
    helpers)."""
    gt = [(c["x"], c["y"], c["index"]) for c in record["corners"] if c["visible"]]
    gt_xy = np.array([(x, y) for x, y, _ in gt], dtype=np.float64).reshape(-1, 2)
    result = dcc_detect(record["image"], model, refiner, K=None, dist=None, cfg=cfg)
    det_xy = np.array([[c["x"], c["y"]] for c in result["corners"]], dtype=np.float64).reshape(-1, 2)
    det_idx = np.array([c["index"] if c["index"] is not None else -1 for c in result["corners"]], dtype=int)

    matches = ref._match_greedy(gt_xy, det_xy, match_px)
    bucket = ref._octave_bucket(record["s_px"])
    loc_errs = [dist for _gi, _di, dist in matches]
    octave_hit = (bucket, len(matches), len(gt_xy)) if bucket else None
    id_correct = sum(1 for gi, di, _d in matches if det_idx[di] == gt[gi][2])
    id_total = len(matches)
    return loc_errs, octave_hit, id_correct, id_total


def main():
    args = build_parser().parse_args()
    cfg, ckpt = fs.build_cfg(args.ckpt, args.config)
    val_seed = args.seed if args.seed is not None else cfg["synth"]["val_seed"]
    ref = _load_train_detector()

    ok, age, sps = fs.wait_for_trainer_health()
    print(f"trainer health before model load: ok={ok} age={age:.0f}s sps={sps:.1f}")
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, refiner = fs.build_ours(cfg, ckpt, args.refiner_ckpt, device)

    bg_files = list_backgrounds(cfg["synth"]["backgrounds"])
    board_img, _corners = render_board(cfg["synth"]["render_res"], cfg.get("board"))

    partitions = {"inverted": {"loc_errs": [], "octaves": {}, "id_correct": 0, "id_total": 0, "n": 0},
                  "non_inverted": {"loc_errs": [], "octaves": {}, "id_correct": 0, "id_total": 0, "n": 0}}
    n_negative = n_skipped = 0
    t0 = time.time()

    for i in range(args.n):
        if i > 0 and i % 100 == 0:
            ok, age, sps = fs.check_trainer_health()
            print(f"  {i}/{args.n} -- trainer health: ok={ok} age={age:.0f}s sps={sps:.1f}")
            if not ok:
                fs.wait_for_trainer_health()

        record, meta = _val_sample(cfg, bg_files, val_seed, args.n, i)
        if not record["board_present"]:
            n_negative += 1
            continue
        pol = _measure_polarity(record, meta, board_img, args.erode_px)
        if pol is None:
            n_skipped += 1
            continue
        key = "inverted" if pol < 0 else "non_inverted"

        loc_errs, octave_hit, id_correct, id_total = _score_sample(model, refiner, cfg, ref, record, args.match_px)
        p = partitions[key]
        p["n"] += 1
        p["loc_errs"] += loc_errs
        p["id_correct"] += id_correct
        p["id_total"] += id_total
        if octave_hit:
            bucket, n_matched, n_vis = octave_hit
            m, v = p["octaves"].get(bucket, (0, 0))
            p["octaves"][bucket] = (m + n_matched, v + n_vis)

    elapsed = time.time() - t0
    print(f"\nn={args.n} n_negative={n_negative} n_skipped(board too small)={n_skipped} elapsed={elapsed:.1f}s")
    for key, p in partitions.items():
        print(f"\n=== {key} (n={p['n']}) ===")
        if p["n"] == 0:
            print("  (no samples)")
            continue
        errs = np.array(p["loc_errs"])
        m01_median = float(np.median(errs)) if len(errs) else None
        m01_mean = float(errs.mean()) if len(errs) else None
        m04 = p["id_correct"] / p["id_total"] if p["id_total"] else None
        print(f"  M-01 loc err: median={m01_median} mean={m01_mean} n_matched={len(errs)}")
        print(f"  M-04 ID accuracy: {p['id_correct']}/{p['id_total']} = {m04}")
        print("  M-02 recall by octave:")
        for bucket, (m, v) in sorted(p["octaves"].items()):
            print(f"    {bucket}: {m}/{v} = {m / v if v else None}")


if __name__ == "__main__":
    main()
