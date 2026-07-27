"""SD-01..SD-11 synthetic sample generator — the one canonical generation path.

Conventions: points are (x, y) float64; arrays are row-major [y][x]. Images
are uint8 on the way in/out; the photometric workspace is float32 0..255,
BGR (3-channel), clipped to uint8 only once, at the very end. size_mult
scales the (W, H) = cfg["input_size"] canvas up (2x stands in for the sensor
frame for the refiner corpus, per SD-07); s_px/tx/ty are always reported at
that generated resolution. Every random draw takes an explicit rng argument
(a numpy Generator instance) — nothing here reaches into the global numpy
random state, the stdlib `random` module, or cv2's RNG.

Algorithm order per sample (SD-02..SD-06): (1) negative decision; (2)
background sample + flip/rotate/pad/crop; (3) [positive only] render board,
sample/accept an affine, histogram-match, anti-alias prefilter, warp,
composite; (4) occlusion
(cutout-object composite, then rect holes -- both gated by `occlude`);
(5) visibility (junction-point, geometric only); (6) photometrics, then clip
to uint8; (7) grayscale, last. Negatives run (2, 4, 6, 7) and skip the board
(step 3) and per-corner visibility (step 5 — there are no corners).

Perspective model (SD-02 Rev C): the board warp is a 3x3 homography H = M3
@ P -- M3 the existing 6-DoF affine lifted to 3x3, P a perspective factor
conjugated about the render centre c_r so that P(c_r) dehomogenises to c_r
exactly: P = T_cr @ [[1,0,0],[0,1,0],[gx,gy,1]] @ T_-cr. (gx, gy) reproduce
the third-row perturbation of the homography induced by tilting a
fronto-parallel board by tau (about in-plane axis angle psi) in front of a
virtual pinhole of focal f = fov_scale * w2 viewing it at apparent scale s:
g = sin(tau) * s / (f * SQ); (gx, gy) = g * (cos psi, sin psi); SQ = the
render square side in px (see tools/gen_eval_pose.py for the from-first-
-principles pinhole construction this is calibrated against, and
tests/test_generator.py::test_perspective_calibration for the numeric
check). tau ~ U(0, tilt_max_deg) is kept (else forced to 0, pure affine)
w.p. perspective_p; psi ~ U(0, 2*pi); fov_scale ~ U(*fov_scale_range). The
coin and all three draws always happen, in that order, regardless of the
coin's outcome or of a `components` override (_sample_perspective) -- a
sample always spends the same four rng calls here, keeping downstream
occlusion/photometric draws aligned across perspective_p/tilt_max_deg
config changes.

Object-cutout occlusion (config-gated, synth.cutouts, additional to
_apply_occlusion's rect holes -- those model dead zones/glare, this models
real clutter sitting in front of the board): an offline SAM2 segmentation
pass (tools/gen_cutouts.py) builds a bank of RGBA PNG cutouts from COCO
images; _apply_cutouts warps (resize, rotate, feather) and alpha-composites
up to cutouts.max_objects of them per sample, max-accumulating their alpha
into an (h2, w2) occ_alpha mask. A junction whose rounded pixel lands on
occ_alpha >= 0.5 counts as covered -- the same kind of geometric,
point-in-region test _apply_occlusion's rect holes already are, so DA-04's
discriminator stays geometric-only. Like _sample_perspective, _apply_cutouts
always spends its full rng budget -- a coin, then max_objects * 6 draws --
regardless of the coin/bank/per-slot outcome, so cutouts.p/max_objects and
bank presence/absence never desync the draws downstream of it. Gated by the
same `occlude` flag as the rect holes (both are "occlusion effects" off in
one switch), so refiner-crop generation and content-check tests that pass
occlude=False stay exactly as clean as before this feature existed.

Slice B5 (final generator slice -- three independent, config-gated
additions): (a) contrast jitter (blend-about-mean: work = m + (work-m)*c)
and multiplicative speckle noise in _apply_photometric -- the reference
pipeline's only contrast source and the Deep ChArUco paper's own Table 1
speckle term, both previously missing from this port; (b) an anti-alias
prefilter in _composite_board (synth.prefilter), band-limiting the board
render before warpPerspective minifies it so small marker modules degrade
the way a real lens/sensor would rather than aliasing under INTER_LINEAR
point-sampling -- pixel-only (blurs the board image, never the mask), spends
no rng draws, and runs strictly after H is fixed and before it is ever
applied to a point, so the corner geometry is provably untouched; (c)
make_generic_crop, an optional
(synth.refiner_generic_frac, default 0 = off) refiner-corpus complement that
swaps a synthetic arbitrary-junction crop (2-4 flat angular sectors meeting
at one point, Deep-ChArUco Fig 6 style) in for a real chessboard-corner
crop, so the refiner also sees non-checkerboard corner shapes;
dataset.py's _maybe_replace_generic is the shared glue for both refiner call
sites (SynthStream and RefinerVal) and draws nothing extra when the frac is
0 or absent. (a) adds mid-photometric rng draws, so every existing val/train
stream realigns one final time from this slice on -- accepted, and the last
such realignment before this file freezes for the training campaign.

Rev-2 augmentation pack (task #29 -- sim-to-real, all config-gated,
individually probability/enabled-gated, all photometric-only except
adherent droplets, see below): _apply_photometric's pinned order is now
composite -> (1) board specular -> (2) droplets -> (3) vignette -> (4) NIR
ink-contrast jitter -> existing blur family -> (5) differencing pair (when
drawn, SUBSUMES the brightness step for that sample -- brightness's own
coin isn't even drawn) or legacy brightness -> dark-regime white grey-out
(task #25, gated on that sample's own OUTPUT exposure landing deep, BEFORE
sensor noise -- the differencing branch reads its own clipped result's 99th
percentile rather than its ambient draw, since ambient alone doesn't
determine output brightness; the brightness branch's own draw IS
output-coupled, so it's used directly, see _apply_photometric) -> (6)
Poissonian-Gaussian sensor noise (replaces the legacy
additive-Gaussian draw when sensor_noise_enabled; gauss_noise_p still gates
WHETHER noise fires either way) -> (7) fixed-pattern noise (full-canvas
only, see _apply_photometric) -> existing speckle/mult/contrast/rgb-shift/
glare/ghost, unchanged relative order (brightness excised from its old spot
between mult and contrast) -> clip/uint8. (1) specular, (3) vignette, (4)
ink-contrast and the differencing pair's illumination lobe are all
board-anchored or canvas-centred FIELDS evaluated at absolute canvas
coordinates -- window-evaluable exactly like the pre-existing glare
implementation, sharing its lazy per-call coordinate grid (_apply_
photometric's abs_grid closure). board_mask (the warped board alpha, `work`'s
own local frame) and board_centroid ((x, y) in ABSOLUTE canvas coordinates)
thread into _apply_photometric from generate_sample (via a new _warp_mask
helper -- _composite_board's own 4-tuple return can't grow without breaking
tests/test_refiner_fast.py's positional unpack, so the mask is re-derived
from the returned H instead of also being returned) and from
dcc.refiner_data.fast_refiner_crops (board_centroid approximated by the
crop's own corner position there, per the module's own approximation
style). Droplets' mode (b) (a refractive patch that resamples a
several-times-larger neighbourhood than its own disc) and fixed-pattern
noise (a fixed-per-frame pattern) are both restricted to window_origin is
None (the full-canvas arm): mode (b) needs pixel data well outside the fast
refiner arm's local-window budget, and FPN's "fixed per frame" contract
would need a hash-reseeded-per-column RNG to stay window-consistent, which
this pack does not attempt -- 24px refiner crops are near-DC for banding
regardless. Droplet mode (b) is the ONLY label-touching new augmentation:
an information-destroying disc (alpha/radius past a config threshold)
registers a bounding-square hole through generate_sample's own `holes` list
(the same rect-hole format/visible() test _apply_occlusion's holes already
use), via an optional holes_out list _apply_photometric appends to in
place -- not a return-value change, for the same tests/test_generator.py
and tools/gen_eval_pose.py single-return-value-callers reason as above.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from skimage.exposure import match_histograms

from dcc.board import get_board, render_board


def list_backgrounds(path):
    """Sorted background image paths. `path` a directory -> recursive glob of
    *.jpg/*.jpeg/*.png (glob order is not deterministic; sorting is). `path`
    a .json -> a COCO annotation file: images are assumed to live in a
    sibling directory named after the JSON's own basename (e.g.
    instances_train2017.json -> ./train2017/); file_name is joined directly,
    no alternate COCO directory layout is probed."""
    p = Path(path)
    if p.suffix == ".json":
        data = json.loads(p.read_text())
        base = p.parent / p.stem
        return sorted(str(base / im["file_name"]) for im in data["images"])
    return sorted(str(f) for pat in ("*.jpg", "*.jpeg", "*.png") for f in p.rglob(pat))


def load_cutouts(path):
    """Sorted cutout-bank RGBA PNG paths (tools/gen_cutouts.py's output dir).
    A missing/falsy/non-directory `path` returns [] with no error -- the
    object-occlusion feature just silently disables (an empty bank makes
    _apply_cutouts a no-op regardless of cutouts.p)."""
    if not path:
        return []
    p = Path(path)
    return sorted(str(f) for f in p.glob("*.png")) if p.is_dir() else []


_CUTOUT_CACHE = {}


def _cached_cutouts(path):
    """Module-level memoised load_cutouts, keyed by `path`: generate_sample's
    fallback for its optional cutout_files=None parameter. dcc/dataset.py
    owns SynthStream/SynthVal/RefinerVal and is a concurrent actor's file in
    this slice, so it has no explicit cutout_files plumbing yet (unlike
    bg_files, which it already loads once and passes in); this cache is what
    keeps that fallback from re-globbing the bank directory on every single
    sample. A follow-up can add explicit cutout_files loading to dataset.py,
    mirroring bg_files, once that file's owner is free to take it."""
    if path not in _CUTOUT_CACHE:
        _CUTOUT_CACHE[path] = load_cutouts(path)
    return _CUTOUT_CACHE[path]


def visible(p, holes, size_wh):
    """SD-04 discriminator, junction-point/geometric only. Half-open pixel
    areas: a pixel-centre coordinate is in-frame/in-hole up to but excluding
    the far edge, so a corner sitting exactly on a hole's or frame's far
    boundary counts as covered/outside."""
    x, y = p
    w, h = size_wh
    if not (-0.5 <= x < w - 0.5 and -0.5 <= y < h - 0.5):
        return False
    for x0, y0, hw, hh in holes:
        if x0 - 0.5 <= x < x0 + hw - 0.5 and y0 - 0.5 <= y < y0 + hh - 0.5:
            return False
    return True


def _prep_background(bg, rng, syn, w2, h2):
    """Flip/rotate the loaded background and random-crop to (w2, h2).

    The source is first UPSCALED (never downscaled) so it covers the canvas
    with a 15% margin before rotating. Without this, any source smaller than
    the canvas gets mirror-tiled by reflect padding -- at a 1600x1200 canvas
    a typical ~640x480 photo would be >80% reflection: a kaleidoscope of
    artificial symmetric junctions that no real frame contains, exactly the
    corner-like structure negatives must NOT carry. With the cover-scale, at
    most thin reflected wedges from the rotation can survive near crop
    edges; the reflect pad below is a never-in-practice safety net.
    Draw order (count fixed): hflip coin, angle, y0, x0."""
    if rng.random() < syn["bg_hflip_p"]:
        bg = cv2.flip(bg, 1)
    h, w = bg.shape[:2]
    cover = max(w2 / w, h2 / h) * 1.15
    if cover > 1.0:
        bg = cv2.resize(bg, (int(np.ceil(w * cover)), int(np.ceil(h * cover))),
                        interpolation=cv2.INTER_LINEAR)
        h, w = bg.shape[:2]
    angle = rng.uniform(-syn["rot_deg"], syn["rot_deg"])
    Mrot = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, 1.0)
    bg = cv2.warpAffine(bg, Mrot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    pad_h, pad_w = max(0, h2 - h), max(0, w2 - w)
    if pad_h or pad_w:
        top, left = pad_h // 2, pad_w // 2
        bg = cv2.copyMakeBorder(bg, top, pad_h - top, left, pad_w - left, cv2.BORDER_REFLECT)
    h, w = bg.shape[:2]
    y0 = int(rng.integers(0, h - h2 + 1))
    x0 = int(rng.integers(0, w - w2 + 1))
    return bg[y0:y0 + h2, x0:x0 + w2]


def _sample_affine(cfg, rng, w2, h2, s_arg, size_mult, components, nx=5):
    """Draw (or take from `components`, the test/audit override surface) the
    six DoF of the board placement: A = (s/SQ) * R(theta) @ Shear(shear_x,
    shear_y); M = [A | c_in + t - A @ c_r] maps render-space -> composite-
    space about the render's own centre, then to c_in + t. The caller lifts
    M to 3x3 and right-multiplies it by a sampled perspective factor (see
    module docstring). nx is the board's per-side square count (SQ =
    render_res // nx); default 5 matches the project board."""
    comp = components or {}
    if "s" in comp:
        s = float(comp["s"])
    elif s_arg is not None:
        s = float(s_arg)
    else:
        a, b = cfg["scale_range_px"]
        s = float(np.exp(rng.uniform(np.log(a * size_mult), np.log(b * size_mult))))
    shear_deg = cfg["synth"]["shear_deg"]
    tf = cfg["synth"]["translate_frac"]
    theta = float(comp["theta"]) if "theta" in comp else float(rng.uniform(-np.pi, np.pi))
    shear_x = float(comp["shear_x"]) if "shear_x" in comp else float(np.radians(rng.uniform(-shear_deg, shear_deg)))
    shear_y = float(comp["shear_y"]) if "shear_y" in comp else float(np.radians(rng.uniform(-shear_deg, shear_deg)))
    tx = float(comp["tx"]) if "tx" in comp else float(rng.uniform(-tf, tf) * w2)
    ty = float(comp["ty"]) if "ty" in comp else float(rng.uniform(-tf, tf) * h2)

    render_res = cfg["synth"]["render_res"]
    SQ = render_res // nx
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    Sh = np.array([[1.0, np.tan(shear_x)], [np.tan(shear_y), 1.0]])
    A = (s / SQ) * (R @ Sh)
    c_r = np.array([(render_res - 1) / 2, (render_res - 1) / 2])
    c_in = np.array([(w2 - 1) / 2, (h2 - 1) / 2])
    M = np.zeros((2, 3), dtype=np.float64)
    M[:, :2] = A
    M[:, 2] = c_in + np.array([tx, ty]) - A @ c_r
    comps_out = {"s": s, "theta": theta, "shear_x": shear_x, "shear_y": shear_y, "tx": tx, "ty": ty}
    return M, comps_out


def _sample_perspective(cfg, rng, components):
    """Draw (or take from `components`) tau/psi/fov_scale -- see module
    docstring for the draw-order contract: coin, tau, psi, fov_scale always
    draw from rng in that order regardless of the coin's outcome or of an
    override; an override only swaps the OUTPUT value, so the number of rng
    calls this step consumes never depends on `components`."""
    comp = components or {}
    syn = cfg["synth"]
    coin = rng.random() < syn["perspective_p"]
    tau_drawn = float(rng.uniform(0.0, np.radians(syn["tilt_max_deg"])))
    psi_drawn = float(rng.uniform(0.0, 2 * np.pi))
    fov_drawn = float(rng.uniform(*syn["fov_scale"]))
    tau = float(comp["tilt"]) if "tilt" in comp else (tau_drawn if coin else 0.0)
    psi = float(comp["psi"]) if "psi" in comp else (psi_drawn if coin else 0.0)
    fov_scale = float(comp["fov_scale"]) if "fov_scale" in comp else fov_drawn
    return tau, psi, fov_scale


def _perspective_factor(tau, psi, fov_scale, s, w2, render_res, nx=5):
    """3x3 P conjugated to fix the render centre c_r (see module docstring):
    calibrated so a tilt tau about axis angle psi matches, to first order,
    the induced homography of a tilted fronto-parallel board at apparent
    scale s in front of a virtual pinhole of focal fov_scale * w2. nx is the
    board's per-side square count (SQ = render_res // nx); default 5
    matches the project board."""
    SQ = render_res // nx
    g = np.sin(tau) * s / (fov_scale * w2 * SQ)
    gx, gy = g * np.cos(psi), g * np.sin(psi)
    cr = (render_res - 1) / 2
    Tcr = np.array([[1.0, 0.0, cr], [0.0, 1.0, cr], [0.0, 0.0, 1.0]])
    Tmcr = np.array([[1.0, 0.0, -cr], [0.0, 1.0, -cr], [0.0, 0.0, 1.0]])
    Pg = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [gx, gy, 1.0]])
    return Tcr @ Pg @ Tmcr


def _warp_mask(H, render_res, w2, h2):
    """The board alpha (render_res^2 all-ones, H-warped to the (w2, h2)
    canvas) -- factored out of _composite_board so generate_sample can
    obtain it a SECOND time (for _apply_photometric's board_mask arg)
    without _composite_board's own 4-tuple return growing a 5th element,
    which would break tests/test_refiner_fast.py::test_window_equivalence's
    positional unpack. The extra warpPerspective this costs per positive
    sample is cheap (single-channel) next to the 3-channel board/background
    warps already paid for in _composite_board."""
    mask_src = np.ones((render_res, render_res), dtype=np.float32)
    return cv2.warpPerspective(mask_src, H, (w2, h2), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _composite_board(bg_crop, rng, cfg, w2, h2, s_arg, size_mult, components):
    """Render the board once, sample its affine + perspective factors,
    histogram-match it to the background crop, anti-alias prefilter it when
    minifying (synth.prefilter -- pixel-only, no rng, runs after H/p_img's
    inputs are already fixed so geometry can't be touched), warp board+mask,
    and alpha-composite. Returns the float32 BGR composite, the board's
    (nx-1)^2 warped analytic corners, the 3x3 homography H, and the
    component dict actually used (post override)."""
    render_res = cfg["synth"]["render_res"]
    bcfg = cfg.get("board")
    board_img, p_render = render_board(render_res, bcfg)
    nx = get_board(bcfg)[1]
    M, comps = _sample_affine(cfg, rng, w2, h2, s_arg, size_mult, components, nx)
    tau, psi, fov_scale = _sample_perspective(cfg, rng, components)
    comps.update(tilt=tau, psi=psi, fov_scale=fov_scale)
    M3 = np.eye(3)
    M3[:2, :] = M
    H = M3 @ _perspective_factor(tau, psi, fov_scale, comps["s"], w2, render_res, nx)

    board_3ch = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)
    matched = match_histograms(board_3ch, bg_crop, channel_axis=-1).astype(np.float32)

    pf = cfg["synth"]["prefilter"]
    s, SQ = comps["s"], render_res // nx
    if pf["enabled"] and s < SQ:
        # Band-limit the render before warpPerspective minifies it (up to
        # 6x at s=16) -- INTER_LINEAR point-samples on the way down, which
        # aliases marker modules harsher than a real lens/sensor would.
        # Mask is untouched: its edge already interpolates smoothly, and H/
        # p_img (below) were fixed before this line runs, so corner geometry
        # can't be affected either way.
        sigma_r = pf["k"] * (SQ / s - 1.0)
        if sigma_r > 0.1:
            matched = cv2.GaussianBlur(matched, (0, 0), sigmaX=sigma_r)

    warped_board = cv2.warpPerspective(matched, H, (w2, h2), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped_mask = _warp_mask(H, render_res, w2, h2)
    m = warped_mask[..., None]
    work = bg_crop.astype(np.float32) * (1 - m) + warped_board * m

    p_hom = np.hstack([p_render, np.ones((len(p_render), 1))]) @ H.T
    p_img = p_hom[:, :2] / p_hom[:, 2:3]
    return work, p_img, H, comps


def place_cutout(work, occ_alpha, rgba, scale, rot, hflip, cx, cy):
    """Pure placement step of _apply_cutouts, factored out so it can be unit
    tested directly with explicit params (no rng/bank involved): given an
    already-loaded BGRA `rgba` cutout, hflip it if drawn, resize so its
    longest side is scale*min(w2,h2) (INTER_AREA when minifying), rotate by
    `rot` degrees about its own centre (expanding the canvas so nothing is
    cropped, transparent-filled corners), feather the alpha with a (3,3)
    GaussianBlur, then alpha-composite it onto `work` (mutated in place, its
    (h2, w2) taken from work.shape) centred at (cx, cy) in work's own frame
    -- clipped to frame; fully off-frame is legal, a no-op. Also
    max-accumulates the placed alpha into `occ_alpha` (mutated in place).
    Returns the placed (x0, y0, w, h) bbox in work's frame, or None if
    nothing landed in-frame (fully off-frame, or a degenerate/wrong-shape
    `rgba` -- e.g. a foreign non-RGBA file dropped into the bank)."""
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
        return None
    if hflip:
        rgba = cv2.flip(rgba, 1)

    h2, w2 = work.shape[:2]
    ch, cw = rgba.shape[:2]
    long_side = max(ch, cw)
    if long_side <= 0:
        return None
    f = (scale * min(w2, h2)) / long_side
    new_w, new_h = max(1, int(round(cw * f))), max(1, int(round(ch * f)))
    rgba = cv2.resize(rgba, (new_w, new_h), interpolation=cv2.INTER_AREA if f < 1.0 else cv2.INTER_LINEAR)

    centre = (new_w / 2, new_h / 2)
    Mrot = cv2.getRotationMatrix2D(centre, rot, 1.0)
    cos_a, sin_a = abs(Mrot[0, 0]), abs(Mrot[0, 1])
    exp_w = int(new_h * sin_a + new_w * cos_a)
    exp_h = int(new_h * cos_a + new_w * sin_a)
    Mrot[0, 2] += exp_w / 2 - centre[0]
    Mrot[1, 2] += exp_h / 2 - centre[1]
    rgba = cv2.warpAffine(rgba, Mrot, (exp_w, exp_h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    rgba[..., 3] = cv2.GaussianBlur(rgba[..., 3], (3, 3), 0)

    ox, oy = int(round(cx - exp_w / 2)), int(round(cy - exp_h / 2))
    x0, y0 = max(ox, 0), max(oy, 0)
    x1, y1 = min(ox + exp_w, w2), min(oy + exp_h, h2)
    if x1 <= x0 or y1 <= y0:
        return None
    lx0, ly0 = x0 - ox, y0 - oy
    patch = rgba[ly0:ly0 + (y1 - y0), lx0:lx0 + (x1 - x0)]
    alpha = patch[..., 3:4].astype(np.float32) / 255.0
    work[y0:y1, x0:x1] = work[y0:y1, x0:x1] * (1 - alpha) + patch[..., :3].astype(np.float32) * alpha
    occ_alpha[y0:y1, x0:x1] = np.maximum(occ_alpha[y0:y1, x0:x1], alpha[..., 0])
    return (x0, y0, x1 - x0, y1 - y0)


def _apply_cutouts(work, rng, cfg, cutout_files, w2, h2):
    """Realistic object occlusion -- see module docstring. Draws each slot's
    params and delegates the actual warp/composite to place_cutout. Returns
    (work, occ_alpha, placed): occ_alpha the (h2, w2) float32 max-accumulated
    placed alpha in [0, 1] (generate_sample's point-in-alpha visibility
    test); placed the per-object [{file, bbox}] list for meta["cutouts"].

    Stream-alignment discipline (mirrors _sample_perspective, see module
    docstring): the coin, then every slot's SIX draws -- idx, scale, rot,
    hflip, (cx, cy) as one size=2 draw, use_slot -- ALWAYS fire in that
    order, regardless of the coin, bank emptiness, or use_slot outcome. (cx,
    cy) are drawn together in one rng.uniform(size=2) call rather than two
    separate ones so the per-slot budget is exactly six Generator calls."""
    cutouts = cfg["synth"]["cutouts"]
    coin = rng.random() < cutouts["p"]
    occ_alpha = np.zeros((h2, w2), dtype=np.float32)
    placed = []
    for _ in range(cutouts["max_objects"]):
        idx = int(rng.integers(2 ** 31))
        scale = float(rng.uniform(*cutouts["scale"]))
        rot = float(rng.uniform(-180.0, 180.0))
        hflip = rng.random() < 0.5
        cx, cy = rng.uniform(-0.1, 1.1, size=2) * (w2, h2)
        use_slot = rng.random() < 0.7
        if not (coin and cutout_files and use_slot):
            continue
        path = cutout_files[idx % len(cutout_files)]
        rgba = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        bbox = place_cutout(work, occ_alpha, rgba, scale, rot, hflip, cx, cy)
        if bbox is not None:
            placed.append({"file": path, "bbox": bbox})

    return work, occ_alpha, placed


def _apply_occlusion(work, rng, occ, w2, h2):
    """SD-04 CoarseDropout-style holes, filled in place; returns the hole
    rects (x0, y0, w, h) so visibility and meta can use them."""
    holes = []
    if rng.random() < occ["p"]:
        lo_n, hi_n = occ["holes"]
        lo_s, hi_s = occ["size"]
        n = int(rng.integers(lo_n, hi_n + 1))
        for _ in range(n):
            hw = int(rng.integers(lo_s, hi_s + 1))
            hh = int(rng.integers(lo_s, hi_s + 1))
            x0 = int(rng.integers(0, w2 - hw + 1))
            y0 = int(rng.integers(0, h2 - hh + 1))
            choice = int(rng.integers(0, 4))
            if choice == 3:
                fill = rng.integers(0, 256, size=(hh, hw, 3)).astype(np.float32)
            else:
                fill = (0.0, 128.0, 255.0)[choice]
            work[y0:y0 + hh, x0:x0 + hw] = fill
            holes.append((x0, y0, hw, hh))
    return holes


def _sample_on_mask(rng, mask, thresh=0.5, tries=5):
    """A point drawn (reject-sampled, `tries` attempts) uniformly over
    `mask`'s own bounding box and accepted only where `mask > thresh` --
    "uniform on the board" for _apply_photometric's specular lobe centre.
    Falls back to the bbox centre after `tries` misses (a warped board mask
    is a single compact convex-ish blob, so a miss run this long is rare)
    or to the array's own centre if `mask` has no pixel above `thresh` at
    all (e.g. an off-board refiner-crop window). Returns LOCAL (x, y) --
    `mask`'s own coordinate frame, not necessarily the canvas's."""
    ys, xs = np.where(mask > thresh)
    h, w = mask.shape[:2]
    if len(xs) == 0:
        return w / 2.0, h / 2.0
    x0, x1, y0, y1 = float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())
    for _ in range(tries):
        x, y = rng.uniform((x0, y0), (x1 + 1.0, y1 + 1.0))
        iy, ix = min(int(y), h - 1), min(int(x), w - 1)
        if mask[iy, ix] > thresh:
            return float(x), float(y)
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def _apply_refractive_droplet(work, cx, cy, radius, minify, alpha, blur_sigma):
    """Mode-(b) adherent droplet (Roser & Geiger ICCV 2009; You et al. CVPR
    2013): the disc of radius `radius` centred at (cx, cy) is replaced by a
    minified, vertically-inverted, blurred crop of the scene around that
    same point -- a cheap stand-in for a water droplet's own tiny lens.
    Full-canvas only (see _apply_photometric's module-docstring note): the
    source crop spans up to radius/minify in side length (~133px at the
    loosest minify), well beyond the fast refiner arm's local-window
    budget. Returns a NEW array (`work` is not mutated in place), matching
    _apply_photometric's own convention -- even though the upstream
    generate_sample steps _apply_cutouts/_apply_occlusion do mutate their
    `work` argument in place."""
    h, w = work.shape[:2]
    diam = max(2, int(round(2 * radius)))
    side = min(max(diam, int(round(diam / minify))), w, h)
    x0 = int(np.clip(round(cx - side / 2), 0, max(0, w - side)))
    y0 = int(np.clip(round(cy - side / 2), 0, max(0, h - side)))
    patch = cv2.resize(work[y0:y0 + side, x0:x0 + side], (diam, diam), interpolation=cv2.INTER_AREA)
    patch = cv2.flip(patch, 0)
    patch = cv2.GaussianBlur(patch, (0, 0), sigmaX=blur_sigma)

    yy, xx = np.mgrid[0:diam, 0:diam].astype(np.float32) - (diam - 1) / 2.0
    disc = ((xx ** 2 + yy ** 2) <= (diam / 2.0) ** 2).astype(np.float32) * alpha

    px0, py0 = int(round(cx - diam / 2.0)), int(round(cy - diam / 2.0))
    tx0, ty0 = max(px0, 0), max(py0, 0)
    tx1, ty1 = min(px0 + diam, w), min(py0 + diam, h)
    if tx1 <= tx0 or ty1 <= ty0:
        return work
    lx0, ly0 = tx0 - px0, ty0 - py0
    lx1, ly1 = lx0 + (tx1 - tx0), ly0 + (ty1 - ty0)
    a = disc[ly0:ly1, lx0:lx1, None]
    out = work.copy()
    out[ty0:ty1, tx0:tx1] = work[ty0:ty1, tx0:tx1] * (1 - a) + patch[ly0:ly1, lx0:lx1] * a
    return out


def _apply_photometric(work, rng, ph, w2, h2, window_origin=None, board_mask=None,
                        board_centroid=None, holes_out=None):
    """SD-03 photometric set, in the pinned order; each gated by its own
    probability. Returns the (possibly reassigned, cv2 ops are not in-place)
    float32 array; caller clips/casts to uint8 afterwards. Slice B5 added
    speckle (the Deep ChArUco paper's own Table 1 term) and contrast (the
    reference pipeline's only contrast source, dropped when this file
    replaced it) -- both, like every step here, conditional draws, so
    turning any one on shifts every rng call downstream of it. See the
    module docstring for the full pinned order (rev-2 pack, task #29,
    reordered speckle/contrast to after the new sensor-noise/FPN steps and
    moved brightness into the differencing/brightness branch below).

    window_origin (dcc.refiner_data's fast refiner-crop arm only; every
    other caller leaves it None) marks `work` as a (w2, h2)-canvas WINDOW
    rather than the full canvas: `work`'s own shape gives the patch's real
    dims (used to size the per-pixel noise/speckle/mult fields and ghost's
    warp output -- byte-identical to before when work.shape == (h2, w2), as
    for every non-None-window_origin caller); w2/h2 stay the FULL virtual
    canvas the position-dependent fields (glare's centre draw, its mgrid) are
    drawn/evaluated over, restricted to the window via window_origin's
    (wx0, wy0) offset. Local operations (blur, ghost's shift) need no such
    restriction: their kernel/shift extent is well within the fast arm's
    window margin, so evaluating them on the window alone already matches
    full-canvas evaluation at those pixels.

    Rev-2 pack (task #29, see module docstring for the full pinned order and
    the window/hole-registration design notes): board_mask (2D float32 in
    [0, 1], `work`'s own local frame) and board_centroid ((x, y) in ABSOLUTE
    canvas coordinates) gate/centre the board-anchored steps (specular,
    ink-contrast, differencing's illumination lobe); both None for negatives
    or callers with no board geometry to hand (e.g. tests/test_generator.py's
    isolated-effect calls) -- the gating coin still always draws in that
    case, only the effect's body is skipped, so rng consumption stays
    self-consistent regardless of board presence. holes_out, if given a
    list, gets mode-(b) droplet holes APPENDED to it in place (see
    generate_sample); every other caller leaves it None."""
    ph_h, ph_w = work.shape[:2]
    wx0, wy0 = (0, 0) if window_origin is None else window_origin

    _grid = [None]

    def abs_grid():
        # Lazy + shared across every position-dependent step below (specular,
        # droplet bokeh, vignette, differencing, glare): computed at most
        # once per call, by whichever fires first. Most samples fire none of
        # them, so unconditionally allocating two (ph_h, ph_w) float32 grids
        # on every call (glare's own previous inline cost, now shared) would
        # be wasted work far more often than not.
        if _grid[0] is None:
            ys, xs = np.mgrid[wy0:wy0 + ph_h, wx0:wx0 + ph_w]
            _grid[0] = (ys.astype(np.float32), xs.astype(np.float32))
        return _grid[0]

    have_board = board_mask is not None and np.any(board_mask > 0.5)
    cxb, cyb = board_centroid if board_centroid is not None else ((w2 - 1) / 2.0, (h2 - 1) / 2.0)

    # (1) board specular -- Blinn-Phong lobe (Blinn 1977): a 2D Gaussian
    # raised to a power n as a proxy for (N.H)^n. No real 3D normal/half-
    # vector is simulated -- there is no per-pixel surface geometry in this
    # 2D pipeline -- the same simplification glare's own Gaussian "hot spot"
    # already makes.
    if rng.random() < ph["specular_p"]:
        if have_board:
            lx, ly = _sample_on_mask(rng, board_mask)
            cx, cy = wx0 + lx, wy0 + ly
            strength = rng.uniform(*ph["specular_strength"])
            n = rng.uniform(*ph["specular_exponent"])
            spread = rng.uniform(*ph["specular_spread_frac"]) * min(w2, h2)
            ys, xs = abs_grid()
            cos_nh = np.clip(np.exp(-0.5 * ((xs - cx) ** 2 + (ys - cy) ** 2) / spread ** 2), 0.0, 1.0)
            work = work + (strength * cos_nh ** n)[..., None].astype(np.float32)

    # (2) adherent droplets (Roser & Geiger ICCV 2009; You et al. CVPR
    # 2013): position drawn canvas-uniform (like glare); mode (b) (the only
    # label-touching new augmentation) is restricted to the full-canvas arm
    # (window_origin is None), see module docstring -- the window arm still
    # gets mode (a) bokeh blobs, just never mode (b).
    if rng.random() < ph["droplet_p"]:
        lo, hi = ph["droplet_n"]
        for _ in range(int(rng.integers(lo, hi + 1))):
            cx, cy = rng.uniform((0.0, 0.0), (float(w2), float(h2)))
            mode_b = window_origin is None and rng.random() < ph["droplet_mode_b_p"]
            if mode_b:
                radius = float(rng.uniform(*ph["droplet_refractive_radius"]))
                minify = rng.uniform(*ph["droplet_refractive_minify"])
                alpha = rng.uniform(*ph["droplet_refractive_alpha"])
                blur_sigma = rng.uniform(*ph["droplet_refractive_blur_sigma"])
                work = _apply_refractive_droplet(work, cx, cy, radius, minify, alpha, blur_sigma)
                if (holes_out is not None and alpha > ph["droplet_hole_alpha_thresh"]
                        and radius > ph["droplet_hole_radius_thresh"]):
                    # Information-destroying: register a bounding-SQUARE hole
                    # (not the disc itself) through the existing rect-hole
                    # machinery (_apply_occlusion's own format, visible()'s
                    # own test) rather than teaching visible() a circular
                    # test -- simpler, and only ever slightly OVER-occludes
                    # (the disc's own four corners), never under-occludes.
                    holes_out.append((cx - radius, cy - radius, 2 * radius, 2 * radius))
            else:
                radius = rng.uniform(*ph["droplet_bokeh_radius"])
                brightness = rng.uniform(*ph["droplet_bokeh_brightness"])
                ys, xs = abs_grid()
                blob = brightness * np.exp(-0.5 * ((xs - cx) ** 2 + (ys - cy) ** 2) / radius ** 2)
                work = work + blob[..., None].astype(np.float32)

    # (3) vignetting (Goldman TPAMI 2010): I * (1 - v*(r/r_max)^2)^2, about
    # the canvas centre.
    if rng.random() < ph["vignette_p"]:
        v = rng.uniform(*ph["vignette_strength"])
        ys, xs = abs_grid()
        ccx, ccy = (w2 - 1) / 2.0, (h2 - 1) / 2.0
        r_max = 0.5 * np.sqrt(w2 ** 2 + h2 ** 2)
        r2 = ((xs - ccx) ** 2 + (ys - ccy) ** 2) / r_max ** 2
        work = work * ((1 - v * r2) ** 2)[..., None].astype(np.float32)

    # (4) NIR ink-contrast jitter -- own placement choice (not pinned by the
    # brief): grouped with the other scene-material effects above, before
    # the camera/sensor-side effects below. Carbon-black toner/ink absorbs
    # strongly across the NIR band even at low dot density, while many
    # dye-based inks are comparatively NIR-transparent -- the basis of
    # Vis-NIR ink discrimination in questioned-document/security-printing
    # forensics (see e.g. "Principal component analysis for the forensic
    # discrimination of black inkjet inks based on the Vis-NIR fibre optics
    # reflection spectra", Forensic Science International 254 (2015)) -- so
    # a printed board's dark ink is expected to read lighter/greyer under
    # NIR illumination than it does under visible light.
    if rng.random() < ph["ink_contrast_p"]:
        if have_board:
            mask_bool = board_mask > 0.5
            # plain per-pixel channel mean, not a BGR2GRAY-weighted luma --
            # mirrors contrast_p's own "plain scalar mean" precedent below.
            luma = work.mean(axis=-1)
            vals = luma[mask_bool]
            median, white = np.median(vals), np.percentile(vals, 95)
            scale = rng.uniform(*ph["ink_contrast_scale"])
            dark = (mask_bool & (luma < median))[..., None]
            work = np.where(dark, np.minimum(work * scale, white), work).astype(np.float32)

    if rng.random() < ph["motion_blur_p"]:
        ks = np.arange(3, ph["motion_blur_kmax"] + 1, 2)
        k = int(rng.choice(ks))
        angle = rng.uniform(0, np.pi)
        c = k // 2
        dx, dy = np.cos(angle) * c, np.sin(angle) * c
        kernel = np.zeros((k, k), dtype=np.float32)
        cv2.line(kernel, (int(round(c - dx)), int(round(c - dy))),
                  (int(round(c + dx)), int(round(c + dy))), 1.0, 1)
        kernel /= kernel.sum()
        work = cv2.filter2D(work, -1, kernel)
    if rng.random() < ph["gauss_blur_p"]:
        k = int(rng.choice([3, 5, 7]))
        work = cv2.GaussianBlur(work, (k, k), 0)
    # (5) differencing pair (Petschnigg et al., "Digital Photography with
    # Flash and No-Flash Image Pairs," SIGGRAPH 2004; Eisemann & Durand,
    # "Flash Photography Enhancement via Intrinsic Relighting," SIGGRAPH
    # 2004 -- companion papers on the same flash/no-flash pair) or legacy
    # brightness -- mutually exclusive: when differencing's coin
    # fires, brightness's own coin is never even drawn ("subsumes" it
    # exactly as the brief specifies; the ambient/illum draws own exposure
    # for that sample). `dark` carries the exposure verdict into the
    # grey-out step right below, whichever branch (or neither) set it.
    dark = False
    if rng.random() < ph["differencing_p"]:
        ambient = rng.uniform(*ph["differencing_ambient"])
        peak = rng.uniform(*ph["differencing_illum_peak"])
        floor = rng.uniform(*ph["differencing_illum_floor"])
        sigma = rng.uniform(*ph["differencing_illum_sigma_frac"]) * min(w2, h2)
        mag = rng.uniform(*ph["differencing_shift_px"])
        ang = rng.uniform(0, 2 * np.pi)
        ys, xs = abs_grid()
        illum = floor + (peak - floor) * np.exp(-0.5 * ((xs - cxb) ** 2 + (ys - cyb) ** 2) / sigma ** 2)
        i_lit = work * illum[..., None].astype(np.float32)
        i_unlit = work * ambient
        dxs, dys = mag * np.cos(ang), mag * np.sin(ang)
        Mshift = np.array([[1, 0, dxs], [0, 1, dys]], dtype=np.float64)
        i_unlit_shifted = cv2.warpAffine(i_unlit, Mshift, (ph_w, ph_h), flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_REFLECT)
        # differencing clips here (not just at the very end): a negative
        # lit-minus-unlit is a physically impossible "negative light"
        # reading, not a workspace value the rest of the pipeline should see.
        work = np.clip(i_lit - i_unlit_shifted, 0, 255)
        # Output-coupled grey-out gate, computed on the CLIPPED result, not
        # on ambient (gate defect found at review: ambient alone doesn't
        # determine output brightness -- a low ambient paired with a high
        # illum peak can still clip to a bright frame, so an ambient-only
        # gate was crushing contrast on ~11% of all frames regardless of
        # their actual exposure). 99th percentile over the flat array (every
        # channel and pixel pooled together, no separate luma reduction --
        # same "plain scalar" precedent as contrast_p's own mean below) reads
        # as "the white point landed below dark_greyout_white_thresh."
        dark = np.percentile(work, 99) < ph["dark_greyout_white_thresh"]
    elif rng.random() < ph["brightness_p"]:
        b = rng.uniform(*ph["brightness_range"])
        if rng.random() < 0.5:
            work = work * (1 + b)
        else:
            # additive clamp: b below the floor turns the whole frame (board
            # included) into exact zeros -- information-free positives that
            # contradict the negative stream. See brightness_add_floor in config.
            work = work + max(b, ph["brightness_add_floor"]) * 255
        # b IS output-coupled here (unlike differencing's ambient): it scales
        # the whole frame directly, so the input draw is a valid proxy for
        # the output's own exposure -- no percentile needed.
        dark = b < ph["dark_greyout_brightness_thresh"]

    # (task #25 remainder) dark-regime white grey-out, BEFORE sensor noise
    # per the brief: residual-contrast collapse when this sample's exposure
    # (whichever branch drew it, if either) landed deep. Own observation --
    # see the frame-000051 finding in the report.
    if dark:
        blend = rng.uniform(*ph["dark_greyout_blend"])
        m = work.mean()
        work = (1 - blend) * work + blend * m

    # (6) sensor noise: Poissonian-Gaussian (Foi et al. 2008 IEEE TIP; EMVA
    # 1288) when sensor_noise_enabled, else the legacy additive-Gaussian
    # draw kept functional for A/B -- gauss_noise_p still gates WHETHER
    # noise fires either way; sensor_noise_enabled only picks the formula.
    if rng.random() < ph["gauss_noise_p"]:
        if ph["sensor_noise_enabled"]:
            K = rng.uniform(*ph["sensor_noise_electrons_per_dn"])
            sigma_read = rng.uniform(*ph["sensor_noise_read_std"])
            rate = np.maximum(work, 0.0) * K  # guard: rng.poisson rejects lam<0; lam=0 is legal
            shot = rng.poisson(rate).astype(np.float32) / K
            read = rng.normal(0, sigma_read, work.shape).astype(np.float32)
            work = shot + read
        else:
            std = rng.uniform(*ph["noise_std"])
            work = work + rng.normal(0, std, work.shape).astype(np.float32)

    # (7) fixed-pattern noise (EMVA 1288; El Gamal & Eltoukhy 2005) --
    # full-canvas only (window_origin is None). A genuinely FIXED pattern
    # needs column/row offsets identical whether evaluated over the full
    # canvas or any sub-window of it; deriving that from ABSOLUTE column/row
    # index would need a per-column deterministic hash-reseed (up to ~1600
    # reseeds/frame, and untested here for statistical quality) rather than
    # drawing a single positionally-indexed array -- restricted to the full
    # arm instead and documented, per the brief's own escape hatch (refiner
    # crops are 24px: banding is near-DC at that scale, so the gap costs
    # little).
    if window_origin is None and rng.random() < ph["fpn_p"]:
        sigma_c = rng.uniform(*ph["fpn_col_std"])
        col = rng.normal(0, sigma_c, size=(1, ph_w, 1)).astype(np.float32)
        row = rng.normal(0, sigma_c * 0.5, size=(ph_h, 1, 1)).astype(np.float32)  # "row the same at halved sigma"
        work = work + col + row
        if rng.random() < ph["fpn_prnu_p"]:
            prnu_std = rng.uniform(*ph["fpn_prnu_std"])
            work = work * rng.normal(1.0, prnu_std, size=(ph_h, ph_w, 1)).astype(np.float32)

    # -- existing speckle/mult/contrast/rgb-shift/glare/ghost, unchanged
    # relative order (brightness excised from its old spot between mult and
    # contrast -- it now lives in the differencing/brightness branch above).
    if rng.random() < ph["speckle_p"]:
        std = rng.uniform(*ph["speckle_std"])
        work = work * (1 + rng.normal(0, std, (ph_h, ph_w, 1)).astype(np.float32))
    if rng.random() < ph["mult_noise_p"]:
        field = rng.uniform(*ph["mult_noise_range"], size=(ph_h, ph_w, 1)).astype(np.float32)
        work = work * field
    if rng.random() < ph["contrast_p"]:
        c = rng.uniform(*ph["contrast_range"])
        m = work.mean()  # plain scalar mean, not a BGR2GRAY-weighted luma
        # one -- simpler, and equally valid: the blend-about-mean pivot only
        # needs to be work's OWN mean, whichever way that scalar is computed.
        work = m + (work - m) * c
    if rng.random() < ph["rgb_shift_p"]:
        shift = rng.uniform(-ph["rgb_shift_limit"], ph["rgb_shift_limit"], size=3).astype(np.float32)
        work = work + shift
    if rng.random() < ph["glare_p"]:
        cx, cy = rng.uniform(0, w2), rng.uniform(0, h2)
        ax, ay = rng.uniform(20, 120), rng.uniform(20, 120)
        ang = rng.uniform(0, np.pi)
        peak = rng.uniform(40, 200)
        ys, xs = abs_grid()
        xr = (xs - cx) * np.cos(ang) + (ys - cy) * np.sin(ang)
        yr = -(xs - cx) * np.sin(ang) + (ys - cy) * np.cos(ang)
        glare = peak * np.exp(-0.5 * ((xr / ax) ** 2 + (yr / ay) ** 2))
        work = work + glare[..., None].astype(np.float32)
    if rng.random() < ph["ghost_p"]:
        alpha = rng.uniform(*ph["ghost_alpha"])
        mag = rng.uniform(*ph["ghost_shift_px"])
        ang = rng.uniform(0, 2 * np.pi)
        dx, dy = mag * np.cos(ang), mag * np.sin(ang)
        blurred = cv2.GaussianBlur(work, (3, 3), 0)
        Mshift = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float64)
        shifted = cv2.warpAffine(blurred, Mshift, (ph_w, ph_h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT)
        work = (1 + alpha) * work - alpha * shifted
    return work


def generate_sample(cfg, rng, bg_files, s=None, size_mult=1, force_negative=None,
                     photometric=True, occlude=True, components=None, cutout_files=None):
    """Generate one SD-05 record + in-memory meta. `components` (test/audit
    surface only) overrides any of the affine's s/theta/shear_x/shear_y/tx/ty
    with a fixed value, skipping that value's rng draw; it can also override
    tilt/psi/fov_scale, but those three always draw from rng regardless (see
    module docstring), so an override there only swaps the output value.
    meta["M"] is the full 3x3 homography (still keyed "M" for compatibility);
    None for negatives, as before. `cutout_files` is the object-occlusion
    bank (see load_cutouts); None (the default) falls back to a module-level
    memoised load of cfg["synth"]["cutouts"]["path"] -- dataset.py is not
    this slice's file, so it can't yet be given the explicit-pass-in
    treatment bg_files already gets (see _cached_cutouts)."""
    W, H = cfg["input_size"]
    # size_mult may be fractional (refiner_res_mult 2.5 at 640x480 -> the
    # 1600x1200 sensor frame); the canvas itself must land on whole pixels.
    w2, h2 = W * size_mult, H * size_mult
    assert w2 == int(w2) and h2 == int(h2), \
        f"input_size {cfg['input_size']} x size_mult {size_mult} is not a whole-pixel canvas"
    w2, h2 = int(w2), int(h2)
    negative = force_negative if force_negative is not None else rng.random() < cfg["negative_p"]
    if cutout_files is None:
        cutout_files = _cached_cutouts(cfg["synth"].get("cutouts", {}).get("path"))

    while True:
        idx = int(rng.integers(len(bg_files)))
        bg = cv2.imread(bg_files[idx], cv2.IMREAD_COLOR)
        if bg is not None:
            break
    bg_crop = _prep_background(bg, rng, cfg["synth"], w2, h2)

    if negative:
        work = bg_crop.astype(np.float32)
        p_img, M, comps, s_px, board_mask = None, None, None, 0.0, None
    else:
        work, p_img, M, comps = _composite_board(bg_crop, rng, cfg, w2, h2, s, size_mult, components)
        s_px = comps["s"]
        # re-derived from H rather than a 5th _composite_board return value --
        # see _warp_mask's own docstring for why.
        board_mask = _warp_mask(M, cfg["synth"]["render_res"], w2, h2)
    board_centroid = tuple(p_img.mean(axis=0)) if p_img is not None else None

    holes, cutouts_meta, occ_alpha = [], [], np.zeros((h2, w2), dtype=np.float32)
    if occlude:
        work, occ_alpha, cutouts_meta = _apply_cutouts(work, rng, cfg, cutout_files, w2, h2)
        holes = _apply_occlusion(work, rng, cfg["synth"]["occlusion"], w2, h2)

    # Photometrics run BEFORE the visibility loop below (moved up from its
    # old post-visibility spot; corners_out's own loop draws no rng at all,
    # so this reorder changes nothing about the existing rng stream) so that
    # holes_out=holes lets a strong mode-(b) droplet register a hole in time
    # for THIS sample's own corners to see it -- see _apply_photometric's
    # module-docstring note.
    if photometric:
        work = _apply_photometric(work, rng, cfg["synth"]["photometric"], w2, h2,
                                   board_mask=board_mask, board_centroid=board_centroid, holes_out=holes)

    corners_out = []
    if not negative:
        for k, (x, y) in enumerate(p_img):
            # DA-04 discriminator stays geometric-only: a junction is now
            # ALSO invisible if its (rounded) pixel lands on cutout alpha
            # >= 0.5 -- point-in-alpha, the same kind of test as the
            # existing point-in-hole-rect check inside visible().
            vis = visible((x, y), holes, (w2, h2))
            if vis:
                iy, ix = int(np.rint(y)), int(np.rint(x))
                if 0 <= iy < h2 and 0 <= ix < w2:
                    vis = bool(occ_alpha[iy, ix] < 0.5)
            corners_out.append({"x": float(x), "y": float(y), "index": k, "visible": vis})

    image = cv2.cvtColor(np.clip(work, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

    record = {"image": image, "board_present": not negative, "s_px": float(s_px), "corners": corners_out}
    meta = {"M": M, "components": comps, "holes": holes, "cutouts": cutouts_meta, "bg_file": bg_files[idx]}
    return record, meta


def cut_refiner_crops(cfg, rng, image2x, corners_visible_xy):
    """SD-07 refiner crops: up to refiner_max_corners rng-chosen corners get
    a jittered 24x24 crop each. j is resampled up to 3 times if it would put
    the true offset outside the 64x64@8x support or the crop outside the
    image; a corner that never lands is skipped, not clamped."""
    jitter = cfg["refiner_jitter_px"]
    max_n = cfg["synth"]["refiner_max_corners"]
    h2, w2 = image2x.shape
    pts = np.asarray(corners_visible_xy, dtype=np.float64).reshape(-1, 2)
    order = rng.permutation(len(pts))[:max_n] if len(pts) > max_n else np.arange(len(pts))

    out = []
    for i in order:
        p = pts[i]
        for _try in range(3):
            j = rng.uniform(-jitter, jitter, size=2)
            c = np.rint(p + j)
            d = p - c
            cx, cy = int(c[0]), int(c[1])
            if max(abs(d[0]), abs(d[1])) <= 3.9375 and 12 <= cx <= w2 - 12 and 12 <= cy <= h2 - 12:
                out.append({"crop": image2x[cy - 12:cy + 12, cx - 12:cx + 12].copy(), "d": d})
                break
    return out


def make_generic_crop(rng, jitter_px=4.0, roughen=True):
    """DA-04 generic-corner refiner-corpus complement (task #6, Deep-ChArUco
    Fig 6 style): a synthetic 24x24 crop of 2-4 flat-shaded angular sectors
    meeting at one point, standing in for the arbitrary (non-checkerboard)
    junction shapes a real scene can present -- cut_refiner_crops' chessboard
    corners are otherwise the only shape SD-07 ever produces. Same
    {"crop", "d"} schema as cut_refiner_crops' output, so it drops straight
    in as a per-crop replacement (see dataset.py's _maybe_replace_generic).

    d is drawn with the same recipe as cut_refiner_crops -- uniform jitter,
    reject-resample against the 64x64@8x target's 3.9375px support, bounded
    at 3 tries -- then clamped into the support as a final (and vanishingly
    rare-to-trigger) guarantee: unlike a real corner, there is no image to
    skip a stubborn draw from, so this crop must always land.

    The true corner sits at local (px, py) = (12+dx, 12+dy). k rays at
    rng-drawn angles partition the crop into k angular sectors radiating OUT
    to the crop edge (straight, infinite lines -- never bounded arcs); each
    is flat-shaded its own gray level, redrawn (up to 20 tries) until it
    differs by >=40 from both its circular neighbours -- so the true corner
    is the only corner-LIKE point in the crop; every other boundary is a
    plain straight edge between two flat regions. Two measured refinements
    on top of that literal reading, both earning their keep against
    test_generic_crops' cornerSubPix check (see that test and the B5 report
    for the numbers): (1) the k angles are reject-resampled (bounded at 30
    tries) against a 35deg minimum gap to every circular neighbour, and,
    for k=2 only, against being within 35deg of exactly antipodal -- a
    sector under ~35deg is a sliver too thin to read as "a corner" at all,
    and two near-antipodal rays (k=2) degenerate to a plain straight edge
    through the point, which is the classical aperture problem: there is no
    unique corner-like point ON a straight edge, so both violate the "only
    corner-like point" invariant above, not just cornerSubPix's comfort.
    (2) the sector fill is rendered at 16x supersampling then
    INTER_AREA-downsampled to 24x24, giving the ray boundaries the same
    kind of anti-aliased sub-pixel edge a real lens/warpPerspective would
    (and that cut_refiner_crops' real corners already have) instead of a
    staircased, pixel-hard one -- this alone was the single biggest lever
    on subpixel accuracy of everything tried.

    roughen (on by default) adds light photometric wear on top -- gaussian
    noise (std U(2,8)) then an optional k=3 blur -- mirroring SD-03's set at
    refiner-crop scale; roughen=False returns the clean piecewise-constant
    crop test_generic_crops' cornerSubPix check needs (noise/blur would
    perturb a classical subpixel refiner for reasons unrelated to whether
    the corner geometry itself is right)."""
    for _try in range(3):
        d = rng.uniform(-jitter_px, jitter_px, size=2)
        if max(abs(d[0]), abs(d[1])) <= 3.9375:
            break
    d = np.clip(d, -3.9375, 3.9375)
    px, py = 12 + d[0], 12 + d[1]

    k = int(rng.integers(2, 5))
    min_gap = np.radians(35.0)
    for _try in range(30):
        angles = np.sort(rng.uniform(0, 2 * np.pi, size=k))
        gaps = np.diff(np.concatenate([angles, angles[:1] + 2 * np.pi]))
        if gaps.min() >= min_gap and not (k == 2 and abs(gaps[0] - np.pi) < min_gap):
            break

    levels = [int(rng.integers(0, 256))]
    for i in range(1, k):
        neighbours = [levels[i - 1]] + ([levels[0]] if i == k - 1 else [])
        for _try in range(20):
            lvl = int(rng.integers(0, 256))
            if all(abs(lvl - n) >= 40 for n in neighbours):
                break
        levels.append(lvl)

    ss = 16
    n = 24 * ss
    coords = (np.arange(n) - (ss - 1) / 2) / ss
    ys, xs = np.meshgrid(coords, coords, indexing="ij")
    ang = np.mod(np.arctan2(ys - py, xs - px), 2 * np.pi)
    sector = (np.searchsorted(angles, ang, side="right") - 1) % k
    hi = np.zeros((n, n), dtype=np.float32)
    for i in range(k):
        hi[sector == i] = levels[i]
    crop = cv2.resize(hi, (24, 24), interpolation=cv2.INTER_AREA).astype(np.uint8)

    if roughen:
        std = rng.uniform(2.0, 8.0)
        noisy = crop.astype(np.float32) + rng.normal(0, std, crop.shape).astype(np.float32)
        bk = int(rng.choice([0, 3]))
        if bk:
            noisy = cv2.GaussianBlur(noisy, (bk, bk), 0)
        crop = np.clip(noisy, 0, 255).astype(np.uint8)

    return {"crop": crop, "d": d}
