"""SD-07 fast-arm refiner crop generation: renders the refiner's 24x24
training crops directly via a local-window warp instead of harvesting them
from a discarded full W*size_mult x H*size_mult composite (dcc.synth's
generate_sample + cut_refiner_crops -- still THE canonical full-frame
contract, see that module's docstring, and RefinerVal's val anchor, both
untouched by this module).

fast_refiner_crops draws ONE board placement (H) per call, via dcc.synth's
own _sample_affine/_sample_perspective/_perspective_factor -- the SAME
samplers generate_sample uses, over the SAME virtual W*size_mult x
H*size_mult canvas -- then renders each accepted corner's crop by warping
only its own (24 + 2*_MARGIN)^2 window through H (H_win =
T(-(cx-12-_MARGIN), -(cy-12-_MARGIN)) @ H, see _render_fast_window), rather
than warping and mostly discarding the full canvas: pixel-for-pixel what
the full warp would produce at those coordinates (verified in
tests/test_refiner_fast.py::test_window_equivalence). _MARGIN=8 gives
blur/motion-blur/ghost their kernel/shift support (max combined extent
~2.8+1+4=7.8px, see dcc.synth._apply_photometric's pinned step order)
without needing full-canvas evaluation; the central 24x24 is taken after
photometrics.

Corner choice + jitter mirror cut_refiner_crops' discipline exactly: pick
up to refiner_max_corners corners (rng.permutation over the visible ones),
j ~ U(-refiner_jitter_px, refiner_jitter_px)^2, resample <=3x to keep the
true offset inside the 64x64@8x (+-3.9375px) support and the crop within a
12px margin of the virtual canvas edge.

Background/histogram-match: match_histograms costs ~2.1ms/call, too slow
to pay per-crop, so _cached_bg_match memoises (decoded image, board render
matched to it) per background path (~32-entry LRU, per-worker since torch
spawns one process per DataLoader worker) -- every crop sharing a
background reuses the match. The bg file CHOICE is still rng-driven per
call (deterministic); the match itself runs against the RAW DECODED image,
not the full path's flip/rotate/crop-prepped canvas -- an accepted
approximation (flip/rotation are histogram-invariant, only the canvas crop
differs). Each crop's own background tile is an independent random
(_PATCH, _PATCH) crop of the cached decoded image, drawn fresh per corner
(background diversity within one H draw, unlike the full arm which shares
one composite's background across all of its harvested crops).

mixed_refiner_crops(cfg, rng, bg_files) is the stream entry point: an
in-worker per-call coin (rng.random() < synth.refiner_full_frac) picks the
full arm -- generate_sample(occlude=False, force_negative=False) +
cut_refiner_crops, verbatim, for distribution insurance (global-photometric
realism the fast arm's local evaluation can't reproduce) -- or the fast arm
otherwise. Both arms return the same [{"crop": uint8 (24,24), "d":
float64[2]}, ...] schema.

Determinism: every draw goes through the passed-in rng, in this order: bg-
file index (+ a fresh draw on each failed decode), _sample_affine's draws,
_sample_perspective's draws (see dcc.synth's module docstring for both),
then per accepted corner: jitter (<=3 tries), then the background-tile crop
position, then photometrics. The fast arm's stream need not bit-match the
full arm's -- it is its own documented sequence.
"""
import cv2
import numpy as np
from skimage.exposure import match_histograms

from dcc.board import get_board, render_board
from dcc.synth import (_apply_photometric, _perspective_factor, _sample_affine,
                        _sample_perspective, cut_refiner_crops, generate_sample, visible)

_MARGIN = 8
_PATCH = 24 + 2 * _MARGIN

_BG_MATCH_CACHE = {}
_BG_CACHE_MAXSIZE = 32


def _cached_bg_match(bg_path, board_3ch):
    """(decoded image, matched-to-it board render) for `bg_path`, memoised
    per path (see module docstring) -- match_histograms is the expensive
    step this amortizes. `decoded` is upscaled (never downscaled, mirrors
    dcc.synth._prep_background's own never-downscale rule) when smaller
    than _PATCH in either dimension, so every cached entry can always
    supply a (_PATCH, _PATCH) window crop. Returns None on a decode
    failure -- the caller retries with a different bg_path, as
    generate_sample does."""
    if bg_path in _BG_MATCH_CACHE:
        entry = _BG_MATCH_CACHE.pop(bg_path)
        _BG_MATCH_CACHE[bg_path] = entry
        return entry
    decoded = cv2.imread(bg_path, cv2.IMREAD_COLOR)
    if decoded is None:
        return None
    h, w = decoded.shape[:2]
    if h < _PATCH or w < _PATCH:
        scale = _PATCH / min(h, w)
        decoded = cv2.resize(decoded, (max(_PATCH, int(np.ceil(w * scale))),
                                        max(_PATCH, int(np.ceil(h * scale)))),
                              interpolation=cv2.INTER_LINEAR)
    matched = match_histograms(board_3ch, decoded, channel_axis=-1).astype(np.float32)
    _BG_MATCH_CACHE[bg_path] = (decoded, matched)
    if len(_BG_MATCH_CACHE) > _BG_CACHE_MAXSIZE:
        _BG_MATCH_CACHE.pop(next(iter(_BG_MATCH_CACHE)))
    return decoded, matched


def _render_fast_window(Hmat, matched, mask_src, bg_tile, cx, cy):
    """Pure per-corner window compositor -- no rng, factored out so it can
    be unit tested directly against the corresponding window of a real
    dcc.synth._composite_board call (see
    tests/test_refiner_fast.py::test_window_equivalence), mirroring
    dcc.synth.place_cutout's role for _apply_cutouts. `matched` is the
    (already prefiltered) render_res^2 board render; `bg_tile` an already-
    selected (_PATCH, _PATCH, 3) background patch; (cx, cy) the jittered
    integer crop centre in virtual-canvas coordinates. Returns (work,
    origin): the float32 BGR (_PATCH, _PATCH, 3) composite (pre-
    photometrics) and its virtual-canvas top-left (wx0, wy0), for
    _apply_photometric's window_origin."""
    wx0, wy0 = cx - 12 - _MARGIN, cy - 12 - _MARGIN
    T = np.array([[1.0, 0.0, -wx0], [0.0, 1.0, -wy0], [0.0, 0.0, 1.0]])
    H_win = T @ Hmat
    warped_board = cv2.warpPerspective(matched, H_win, (_PATCH, _PATCH), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped_mask = cv2.warpPerspective(mask_src, H_win, (_PATCH, _PATCH), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    m = warped_mask[..., None]
    work = bg_tile.astype(np.float32) * (1 - m) + warped_board * m
    return work, (wx0, wy0)


def fast_refiner_crops(cfg, rng, bg_files):
    """The fast arm: one board placement (H), one cached background match,
    up to refiner_max_corners local-window crops. See module docstring for
    the draw order and the caching/approximation contract."""
    syn = cfg["synth"]
    W, H = cfg["input_size"]
    size_mult = syn["refiner_res_mult"]
    w2, h2 = W * size_mult, H * size_mult
    # same whole-pixel coercion as generate_sample: fractional refiner_res_mult
    # (2.5 at 640x480) must still produce an integer virtual canvas
    assert w2 == int(w2) and h2 == int(h2), \
        f"input_size {cfg['input_size']} x size_mult {size_mult} is not a whole-pixel canvas"
    w2, h2 = int(w2), int(h2)
    render_res = syn["render_res"]
    bcfg = cfg.get("board")
    nx = get_board(bcfg)[1]
    jitter = cfg["refiner_jitter_px"]
    max_n = syn["refiner_max_corners"]
    ph = syn["photometric"]

    board_img, p_render = render_board(render_res, bcfg)
    board_3ch = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)
    while True:
        idx = int(rng.integers(len(bg_files)))
        entry = _cached_bg_match(bg_files[idx], board_3ch)
        if entry is not None:
            decoded, matched = entry
            break

    M, comps = _sample_affine(cfg, rng, w2, h2, None, size_mult, None, nx)
    tau, psi, fov_scale = _sample_perspective(cfg, rng, None)
    comps.update(tilt=tau, psi=psi, fov_scale=fov_scale)
    M3 = np.eye(3)
    M3[:2, :] = M
    Hmat = M3 @ _perspective_factor(tau, psi, fov_scale, comps["s"], w2, render_res, nx)

    pf = syn["prefilter"]
    s, SQ = comps["s"], render_res // nx
    if pf["enabled"] and s < SQ:
        # mirrors dcc.synth._composite_board's own prefilter call exactly
        sigma_r = pf["k"] * (SQ / s - 1.0)
        if sigma_r > 0.1:
            matched = cv2.GaussianBlur(matched, (0, 0), sigmaX=sigma_r)

    p_hom = np.hstack([p_render, np.ones((len(p_render), 1))]) @ Hmat.T
    p_img = p_hom[:, :2] / p_hom[:, 2:3]
    pts = np.array([p for p in p_img if visible(tuple(p), [], (w2, h2))],
                    dtype=np.float64).reshape(-1, 2)
    order = rng.permutation(len(pts))[:max_n] if len(pts) > max_n else np.arange(len(pts))

    mask_src = np.ones((render_res, render_res), dtype=np.float32)
    dh, dw = decoded.shape[:2]

    out = []
    for i in order:
        p = pts[i]
        for _try in range(3):
            j = rng.uniform(-jitter, jitter, size=2)
            c = np.rint(p + j)
            d = p - c
            cx, cy = int(c[0]), int(c[1])
            if max(abs(d[0]), abs(d[1])) <= 3.9375 and 12 <= cx <= w2 - 12 and 12 <= cy <= h2 - 12:
                y0 = int(rng.integers(0, dh - _PATCH + 1))
                x0 = int(rng.integers(0, dw - _PATCH + 1))
                bg_tile = decoded[y0:y0 + _PATCH, x0:x0 + _PATCH]
                work, origin = _render_fast_window(Hmat, matched, mask_src, bg_tile, cx, cy)
                work = _apply_photometric(work, rng, ph, w2, h2, window_origin=origin)
                gray = cv2.cvtColor(np.clip(work, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
                out.append({"crop": gray[_MARGIN:_MARGIN + 24, _MARGIN:_MARGIN + 24].copy(), "d": d})
                break
    return out


def mixed_refiner_crops(cfg, rng, bg_files):
    """Stream entry point: an explicit-Generator coin per call against
    cfg["synth"]["refiner_full_frac"] picks the full arm (generate_sample +
    cut_refiner_crops, verbatim) or the fast arm (fast_refiner_crops) --
    the mixed-stream distribution insurance described in this module's
    docstring; see configs/default.yaml's refiner_full_frac comment for why
    it is an in-worker per-sample coin rather than a dedicated worker or a
    cross-process queue."""
    if rng.random() < cfg["synth"]["refiner_full_frac"]:
        size_mult = cfg["synth"]["refiner_res_mult"]
        record, _ = generate_sample(cfg, rng, bg_files, size_mult=size_mult,
                                     occlude=False, force_negative=False)
        pts = [(c["x"], c["y"]) for c in record["corners"] if c["visible"]]
        return cut_refiner_crops(cfg, rng, record["image"], pts)
    return fast_refiner_crops(cfg, rng, bg_files)
