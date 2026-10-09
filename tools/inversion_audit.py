"""tools/inversion_audit.py -- quantifies how much of the training
distribution's differencing draws land in the contrast-POLARITY-INVERSION
regime characterized in the 2026-07-28 differencing_ambient investigation
(tools/factor_sweep.py's differencing_ambient/differencing_ratio/
exposure_clipping axes): white board reflectance clips harder than black
under a bright daylight pedestal, so the differenced signal doesn't just
lose contrast -- past a boundary in (ambient, peak) space it goes NEGATIVE
(white reads darker than black), which any marker-reading system decodes
as noise rather than a degraded-but-recognisable pattern.

This measures the LIVE training config's own draw distributions
(differencing_ambient ~ U(*cfg's own range), differencing_illum_peak ~
U(*cfg's own range), read from cfg rather than hardcoded) against REAL
background/pose variability (a fresh random draw per sample, not
factor_sweep.py's single fixed base scene) -- the question is about label
quality in the live campaign, not a paper-figure probe, so it needs the
same background-histogram-matching variability real training samples see
(dcc.synth._composite_board's match_histograms shifts each sample's board
reflectance range depending on which background got drawn).

Sampling design: for each of --n samples, draw a REAL (background, pose)
composite via dcc.synth._composite_board with components=None (full random
affine+perspective, matching what SynthStream itself would draw), then
independently draw (ambient, peak) from the config's own differencing
ranges and measure white-vs-black board contrast -- this is exactly the
distribution of (ambient, peak) CONDITIONAL ON differencing firing
(differencing_p only gates WHETHER a sample uses differencing; given that
it does, ambient/peak are drawn from these same two ranges independently of
differencing_p's own value), so P(inverted | differencing) is measured
directly and P(inverted | any sample) = differencing_p * that fraction.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from dcc.board import render_board
from dcc.synth import _composite_board, list_backgrounds


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/rev640.yaml")
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--seed", type=int, default=9001)
    p.add_argument("--erode-px", type=int, default=5)
    p.add_argument("--boundary-out", default=None,
                    help="also write the illuminator-sizing boundary figure (ambient x peak, "
                         "fraction-inverted heatmap + 50%% contour) to this PNG path")
    p.add_argument("--boundary-backgrounds", type=int, default=20)
    p.add_argument("--boundary-grid", type=int, default=60)
    return p


def _white_black_masks_canvas(board_img, H, w2, h2, erode_px):
    """Warp the analytic board render's pure-white (==255) and pure-black
    (==0) pixels into canvas space via the SAME H the frame was composited
    with, eroded to exclude edge-blur pixels from the smooth (INTER_LINEAR)
    board warp -- identical method to the differencing_ambient investigation
    this tool follows up on."""
    white_render = (board_img == 255).astype(np.float32)
    black_render = (board_img == 0).astype(np.float32)
    white_canvas = cv2.warpPerspective(white_render, H, (w2, h2), flags=cv2.INTER_NEAREST,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    black_canvas = cv2.warpPerspective(black_render, H, (w2, h2), flags=cv2.INTER_NEAREST,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    kernel = np.ones((erode_px, erode_px), np.uint8) if erode_px > 0 else None
    white_mask = cv2.erode((white_canvas > 0.5).astype(np.uint8), kernel) > 0 if kernel is not None else white_canvas > 0.5
    black_mask = cv2.erode((black_canvas > 0.5).astype(np.uint8), kernel) > 0 if kernel is not None else black_canvas > 0.5
    return white_mask, black_mask


def measure_sample(cfg, bg_files, board_img, rng, ambient_range, peak_range, floor_range,
                    sigma_frac_range, erode_px, ys_grid, xs_grid):
    """One (background, pose) draw, one independent (ambient, peak, floor,
    sigma_frac) draw from the config's own differencing ranges -- returns
    (contrast, ambient, peak, n_white_px, n_black_px). contrast =
    median(diff|white) - median(diff|black), computed with the REAL
    per-pixel Gaussian illumination lobe (illum_lobe = floor + (peak-floor)*
    exp(-dist^2/2sigma^2) about the board centroid) that dcc.synth.
    _apply_photometric's differencing branch actually uses -- NOT a uniform
    peak across the frame.

    CORRECTNESS NOTE (2026-07-28, caught reviewing the team-lead's
    corroborating prior estimate): an earlier version of this function
    applied `peak` UNIFORMLY (i_lit = work0*(ambient+peak)) rather than the
    Gaussian falloff -- this overestimates illumination reaching white/black
    regions away from the board centre, inflating the measured inversion
    fraction (the team lead's own analytical "uniform lobe" estimate is
    explicitly flagged as an upper bound for exactly this reason -- their
    Gaussian-lobe estimate is ~5-6x lower). This version matches the real
    pipeline formula exactly, including sampling floor/sigma_frac from their
    own config ranges rather than holding them fixed.

    i_lit/i_unlit are full BGR (the board's per-channel values diverge from
    each other post-match_histograms against a COLOUR background), diff
    clipped in BGR space, THEN converted to grayscale at the very end,
    exactly matching generate_sample's own final step and therefore exactly
    what the detectors actually see. Noise-free (skips the per-frame
    Poissonian-Gaussian draw) -- the differencing_ambient investigation's
    flat noise-sigma column shows noise doesn't move the polarity question,
    so this audit is CPU-cheap by design."""
    W, H_px = cfg["input_size"]
    while True:
        idx = int(rng.integers(len(bg_files)))
        bg = cv2.imread(bg_files[idx], cv2.IMREAD_COLOR)
        if bg is not None:
            break
    from dcc.synth import _prep_background
    bg_crop = _prep_background(bg, rng, cfg["synth"], W, H_px)
    work0, p_img, Hmat, comps = _composite_board(bg_crop, rng, cfg, W, H_px, None, 1, None)

    white_mask, black_mask = _white_black_masks_canvas(board_img, Hmat, W, H_px, erode_px)
    if white_mask.sum() < 20 or black_mask.sum() < 20:
        return None  # board too small/off-frame in this random pose -- skip, not force

    ambient = float(rng.uniform(*ambient_range))
    peak = float(rng.uniform(*peak_range))
    floor = float(rng.uniform(*floor_range))
    sigma = float(rng.uniform(*sigma_frac_range)) * min(W, H_px)
    cxb, cyb = p_img.mean(axis=0)
    illum_lobe = floor + (peak - floor) * np.exp(-0.5 * ((xs_grid - cxb) ** 2 + (ys_grid - cyb) ** 2) / sigma ** 2)
    i_lit = np.clip(work0 * (ambient + illum_lobe)[..., None], 0, 255)
    i_unlit = np.clip(work0 * ambient, 0, 255)
    diff = np.clip(i_lit - i_unlit, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY).astype(np.float64)
    contrast = float(np.median(gray[white_mask]) - np.median(gray[black_mask]))
    return contrast, ambient, peak, int(white_mask.sum()), int(black_mask.sum())


def _representative_reflectance(cfg, bg_files, board_img, rng, erode_px, sigma, ys_grid, xs_grid):
    """One (background, pose) draw's (W_white, W_black, lobe_white,
    lobe_black): median grayscale reflectance per region (BEFORE
    differencing) plus each region's own MEAN Gaussian lobe weight
    (exp(-dist^2/2sigma^2), a per-background, ambient/peak-INDEPENDENT
    scalar) -- reused across an entire (ambient, peak) grid sweep
    (plot_inversion_boundary) so the sweep only needs cheap scalar
    arithmetic per grid point instead of re-warping/re-compositing an image
    per cell, while still respecting the lobe's spatial falloff (fixed at a
    representative `sigma`, the config's own sigma_frac range midpoint --
    the audit's measure_sample is what varies sigma/floor per sample for
    the headline P(inverted) estimate; this is a 2D visualisation
    simplification, disclosed in the figure title)."""
    W, H_px = cfg["input_size"]
    while True:
        idx = int(rng.integers(len(bg_files)))
        bg = cv2.imread(bg_files[idx], cv2.IMREAD_COLOR)
        if bg is not None:
            break
    from dcc.synth import _prep_background
    bg_crop = _prep_background(bg, rng, cfg["synth"], W, H_px)
    work0, p_img, Hmat, comps = _composite_board(bg_crop, rng, cfg, W, H_px, None, 1, None)
    white_mask, black_mask = _white_black_masks_canvas(board_img, Hmat, W, H_px, erode_px)
    if white_mask.sum() < 20 or black_mask.sum() < 20:
        return None
    gray0 = cv2.cvtColor(np.clip(work0, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
    cxb, cyb = p_img.mean(axis=0)
    lobe_shape = np.exp(-0.5 * ((xs_grid - cxb) ** 2 + (ys_grid - cyb) ** 2) / sigma ** 2)
    return (float(np.median(gray0[white_mask])), float(np.median(gray0[black_mask])),
            float(lobe_shape[white_mask].mean()), float(lobe_shape[black_mask].mean()))


def plot_inversion_boundary(cfg, bg_files, board_img, out_path, n_backgrounds=20, seed=4242,
                             erode_px=5, n_grid=60):
    """The deployment-section deliverable (team-lead 2026-07-28 priority 2):
    a 2D (ambient, peak) map of the fraction of REAL sampled backgrounds
    whose contrast inverts at that operating point, with the 50%-inverted
    contour drawn as the illuminator-sizing boundary -- for a given daylight
    pedestal (ambient), the minimum LED increment (peak) that keeps contrast
    positive is everything ABOVE the boundary; equivalently, for a given
    illuminator (peak), the maximum ambient it tolerates is everything to
    the LEFT of the boundary at that peak. Uses the real per-region Gaussian
    lobe weight (see _representative_reflectance), sigma fixed at the
    config's own sigma_frac range midpoint and floor fixed at its range
    midpoint -- both disclosed in the title, not swept here (measure_sample
    is what varies them per-sample for the headline P(inverted) estimate)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ph = cfg["synth"]["photometric"]
    a_lo, a_hi = ph["differencing_ambient"]
    p_lo, p_hi = ph["differencing_illum_peak"]
    sigma_lo, sigma_hi = ph["differencing_illum_sigma_frac"]
    floor_lo, floor_hi = ph["differencing_illum_floor"]
    sigma = 0.5 * (sigma_lo + sigma_hi) * min(cfg["input_size"])
    floor = 0.5 * (floor_lo + floor_hi)
    ambient_grid = np.linspace(max(a_lo, 0.01), a_hi, n_grid)
    peak_grid = np.linspace(p_lo, p_hi, n_grid)
    A, P = np.meshgrid(ambient_grid, peak_grid)

    W, H_px = cfg["input_size"]
    ys_grid, xs_grid = np.mgrid[0:H_px, 0:W].astype(np.float64)
    rng = np.random.default_rng([seed])
    inverted_sum = np.zeros_like(A)
    n_valid_bg = 0
    for _ in range(n_backgrounds):
        rep = _representative_reflectance(cfg, bg_files, board_img, rng, erode_px, sigma, ys_grid, xs_grid)
        if rep is None:
            continue
        W_white, W_black, lobe_white, lobe_black = rep
        illum_white = floor + (P - floor) * lobe_white
        illum_black = floor + (P - floor) * lobe_black
        diff_white = np.clip(W_white * (A + illum_white), 0, 255) - np.clip(W_white * A, 0, 255)
        diff_black = np.clip(W_black * (A + illum_black), 0, 255) - np.clip(W_black * A, 0, 255)
        inverted_sum += ((diff_white - diff_black) < 0)
        n_valid_bg += 1
    frac_inverted = inverted_sum / max(n_valid_bg, 1)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(frac_inverted, origin="lower", extent=[a_lo, a_hi, p_lo, p_hi],
                    aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
    if frac_inverted.min() < 0.5 < frac_inverted.max():
        cs = ax.contour(A, P, frac_inverted, levels=[0.5], colors="black", linewidths=2)
        ax.clabel(cs, fmt={0.5: "50% inverted"})
    ax.set_xlabel("ambient (daylight pedestal)"); ax.set_ylabel("peak (LED increment)")
    ax.set_title(f"Contrast-inversion boundary ({n_valid_bg} backgrounds, "
                 f"sigma={sigma:.0f}px floor={floor:.2f} fixed at range midpoint)", fontsize=10)
    fig.colorbar(im, ax=ax, label="fraction of backgrounds inverted")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path, frac_inverted, ambient_grid, peak_grid


def main():
    args = build_parser().parse_args()
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ph = cfg["synth"]["photometric"]
    differencing_p = ph["differencing_p"]
    ambient_range = ph["differencing_ambient"]
    peak_range = ph["differencing_illum_peak"]
    floor_range = ph["differencing_illum_floor"]
    sigma_frac_range = ph["differencing_illum_sigma_frac"]
    print(f"config: differencing_p={differencing_p} ambient~U{ambient_range} peak~U{peak_range} "
          f"floor~U{floor_range} sigma_frac~U{sigma_frac_range}")

    bg_files = list_backgrounds(cfg["synth"]["backgrounds"])
    board_img, _corners = render_board(cfg["synth"]["render_res"], cfg.get("board"))
    W, H_px = cfg["input_size"]
    ys_grid, xs_grid = np.mgrid[0:H_px, 0:W].astype(np.float64)

    rng = np.random.default_rng([args.seed])
    contrasts, ambients, peaks = [], [], []
    n_skipped = 0
    for i in range(args.n):
        result = measure_sample(cfg, bg_files, board_img, rng, ambient_range, peak_range,
                                 floor_range, sigma_frac_range, args.erode_px, ys_grid, xs_grid)
        if result is None:
            n_skipped += 1
            continue
        contrast, ambient, peak, _nw, _nb = result
        contrasts.append(contrast)
        ambients.append(ambient)
        peaks.append(peak)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{args.n}")

    contrasts = np.array(contrasts)
    ambients = np.array(ambients)
    peaks = np.array(peaks)
    n_valid = len(contrasts)
    n_inverted = int((contrasts < 0).sum())
    frac_inverted_given_differencing = n_inverted / n_valid if n_valid else float("nan")
    frac_inverted_overall = differencing_p * frac_inverted_given_differencing

    print(f"\nn_sampled={args.n} n_valid={n_valid} n_skipped(board too small/off-frame)={n_skipped}")
    print(f"P(inverted | differencing fires) = {n_inverted}/{n_valid} = {frac_inverted_given_differencing:.4f}")
    print(f"P(inverted | any training sample) = differencing_p * above = {frac_inverted_overall:.4f}")
    print(f"contrast stats: median={np.median(contrasts):.1f} p10={np.percentile(contrasts,10):.1f} "
          f"p90={np.percentile(contrasts,90):.1f} min={contrasts.min():.1f} max={contrasts.max():.1f}")

    if n_inverted:
        inv_ambient, inv_peak = ambients[contrasts < 0], peaks[contrasts < 0]
        print(f"inverted region: ambient in [{inv_ambient.min():.3f}, {inv_ambient.max():.3f}] "
              f"(median {np.median(inv_ambient):.3f}), peak in [{inv_peak.min():.3f}, {inv_peak.max():.3f}] "
              f"(median {np.median(inv_peak):.3f})")
        # boundary curve: for each ambient bin, the smallest peak that inverted (empirical, noisy at low n/bin)
        bins = np.linspace(ambient_range[0], ambient_range[1], 13)
        print("empirical inversion boundary (min inverting peak per ambient bin):")
        for lo, hi in zip(bins[:-1], bins[1:]):
            in_bin = (ambients >= lo) & (ambients < hi)
            inv_in_bin = in_bin & (contrasts < 0)
            if inv_in_bin.any():
                print(f"  ambient [{lo:.2f},{hi:.2f}): n={int(in_bin.sum())} min_inverting_peak={peaks[inv_in_bin].min():.3f}")
            elif in_bin.any():
                print(f"  ambient [{lo:.2f},{hi:.2f}): n={int(in_bin.sum())} no inversions in this bin")

    if args.boundary_out:
        out_path, frac_inverted, ambient_grid, peak_grid = plot_inversion_boundary(
            cfg, bg_files, board_img, args.boundary_out,
            n_backgrounds=args.boundary_backgrounds, erode_px=args.erode_px, n_grid=args.boundary_grid)
        print(f"\nwrote {out_path}")
        # the illuminator-sizing rule itself, read straight off the grid: for
        # each ambient column, the smallest peak with <50% of backgrounds inverted
        print("illuminator-sizing rule (min safe peak per ambient, <50% backgrounds inverted):")
        print_idx = sorted(set(range(0, len(ambient_grid), max(1, len(ambient_grid) // 10))) | {len(ambient_grid) - 1})
        for j in print_idx:
            col = frac_inverted[:, j]
            safe = np.nonzero(col < 0.5)[0]
            min_safe_peak = peak_grid[safe[0]] if len(safe) else float("nan")
            print(f"  ambient={ambient_grid[j]:.3f}: min_safe_peak={min_safe_peak:.3f}")


if __name__ == "__main__":
    main()
