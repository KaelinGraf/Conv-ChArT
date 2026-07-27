import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import cv2
import numpy as np
import pytest
import yaml

from dcc.synth import _apply_photometric, _sample_on_mask, generate_sample, visible

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "default.yaml"


@pytest.fixture(scope="module")
def bg_files(tmp_path_factory):
    """4 random-noise 640x640 backgrounds -- no COCO/network dependency
    (mirrors tests/test_generator.py's own fixture)."""
    d = tmp_path_factory.mktemp("bgs")
    rng = np.random.default_rng(0xA096)
    paths = []
    for i in range(4):
        img = rng.integers(0, 256, size=(640, 640, 3), dtype=np.uint8)
        p = d / f"bg{i}.png"
        cv2.imwrite(str(p), img)
        paths.append(str(p))
    return sorted(paths)


@pytest.fixture(scope="module")
def cfg(bg_files):
    with open(CONFIG_PATH) as f:
        c = yaml.safe_load(f)
    c["synth"] = dict(c["synth"])
    c["synth"]["backgrounds"] = str(Path(bg_files[0]).parent)
    return c


def _photometric_only(cfg, **on):
    """cfg["synth"]["photometric"] copy with every *_p gate zeroed except
    `on` -- mirrors tests/test_generator.py's own helper of the same name
    (duplicated rather than cross-imported: each test file owns its
    fixtures/helpers independently, same as test_refiner_fast.py's own
    bg_files fixture)."""
    ph = dict(cfg["synth"]["photometric"])
    for k in ph:
        if k.endswith("_p"):
            ph[k] = 0.0
    ph.update(on)
    return ph


def _all_on(cfg):
    """Every *_p gate at 1.0 plus sensor_noise_enabled -- the "all new augs
    forced on" stress config for the determinism check."""
    ph = dict(cfg["synth"]["photometric"])
    for k in ph:
        if k.endswith("_p"):
            ph[k] = 1.0
    ph["sensor_noise_enabled"] = True
    return ph


# ------------------------------------------------------------- determinism --

def test_determinism_all_on(cfg, bg_files):
    """Same seed, every rev-2 (and legacy) photometric gate forced to 1.0 at
    once -- specular + droplets (incl. mode-b holes) + vignette +
    ink-contrast + differencing + sensor noise + FPN + the pre-existing
    tail all firing on the same sample -> still bit-identical output."""
    c = copy.deepcopy(cfg)
    c["synth"]["photometric"] = _all_on(cfg)

    rec1, meta1 = generate_sample(c, np.random.default_rng(555), bg_files, s=70.0, force_negative=False)
    rec2, meta2 = generate_sample(c, np.random.default_rng(555), bg_files, s=70.0, force_negative=False)

    assert np.array_equal(rec1["image"], rec2["image"])
    assert rec1["corners"] == rec2["corners"]
    assert meta1["holes"] == meta2["holes"]


# ------------------------------------------------------- label preservation --

def test_label_preservation(cfg, bg_files):
    """Every rev-2 aug except mode-(b) droplets is photometric-only: forcing
    one on (in isolation) must change the rendered image but never move a
    corner or flip its visibility. Mode-(b) droplets are the sole exception
    and are covered by the dedicated hole-registration tests below."""
    components = {"s": 90.0, "theta": 0.15, "shear_x": 0.0, "shear_y": 0.0, "tx": 10.0, "ty": -10.0}
    photometric_only_gates = ["specular_p", "vignette_p", "ink_contrast_p", "differencing_p",
                               "gauss_noise_p", "fpn_p"]

    c_off = copy.deepcopy(cfg)
    c_off["synth"]["photometric"] = _photometric_only(cfg)
    rec_off, _ = generate_sample(c_off, np.random.default_rng(101), bg_files, force_negative=False,
                                  occlude=False, components=dict(components))

    for key in photometric_only_gates:
        c_on = copy.deepcopy(cfg)
        c_on["synth"]["photometric"] = _photometric_only(cfg, **{key: 1.0})
        rec_on, _ = generate_sample(c_on, np.random.default_rng(101), bg_files, force_negative=False,
                                     occlude=False, components=dict(components))
        assert not np.array_equal(rec_on["image"], rec_off["image"]), f"{key} did not visibly fire"
        assert rec_on["corners"] == rec_off["corners"], f"{key} moved or hid a corner"


# ----------------------------------------------- droplet hole registration --

def test_droplet_hole_registration_unit(cfg):
    """_apply_photometric's mode-(b) droplet, forced on with alpha/radius
    past both hole thresholds, must append exactly one bounding-square hole
    to holes_out, sized 2*radius, whose own centre visible() reads as
    covered -- the same rect-hole contract _apply_occlusion's holes use."""
    h2 = w2 = 200
    work = np.full((h2, w2, 3), 120.0, dtype=np.float32)
    ph = _photometric_only(cfg, droplet_p=1.0)
    ph = dict(ph, droplet_n=[1, 1], droplet_mode_b_p=1.0,
              droplet_refractive_radius=[15.0, 15.0], droplet_refractive_minify=[0.3, 0.3],
              droplet_refractive_alpha=[0.9, 0.9], droplet_refractive_blur_sigma=[2.0, 2.0])

    holes = []
    _apply_photometric(work.copy(), np.random.default_rng(11), ph, w2, h2, holes_out=holes)

    assert len(holes) == 1
    x0, y0, hw, hh = holes[0]
    assert hw == pytest.approx(30.0) and hh == pytest.approx(30.0)
    cx, cy = x0 + hw / 2, y0 + hh / 2
    assert visible((cx, cy), holes, (w2, h2)) is False


def test_droplet_hole_registration_below_threshold_no_hole(cfg):
    """The same forced mode-(b) droplet, but with alpha/radius BELOW the
    hole thresholds, must register nothing -- the effect is photometric
    only until both thresholds are cleared."""
    h2 = w2 = 200
    # a gradient, not a flat field: a flat work array makes the refractive
    # patch's source crop identical to its destination, so compositing it
    # would be an (invisible) no-op regardless of whether the droplet fired.
    work = (np.arange(w2, dtype=np.float32)[None, :, None] * np.ones((h2, 1, 3), np.float32))
    ph = _photometric_only(cfg, droplet_p=1.0)
    ph = dict(ph, droplet_n=[1, 1], droplet_mode_b_p=1.0,
              droplet_refractive_radius=[6.0, 6.0], droplet_refractive_minify=[0.3, 0.3],
              droplet_refractive_alpha=[0.5, 0.5], droplet_refractive_blur_sigma=[2.0, 2.0])

    holes = []
    out = _apply_photometric(work.copy(), np.random.default_rng(11), ph, w2, h2, holes_out=holes)

    assert holes == []
    assert not np.array_equal(out, work)  # still visually composited


def test_droplet_hole_registration_integration(cfg, bg_files):
    """End-to-end through generate_sample: a forced strong mode-(b) droplet
    stream must, for at least one of a handful of seeds, cover a corner
    that was visible with photometrics off and flip it to invisible --
    exactly like _apply_occlusion's own rect holes already do."""
    components = {"s": 95.0, "theta": 0.0, "shear_x": 0.0, "shear_y": 0.0, "tx": 0.0, "ty": 0.0}
    ph_strong = _photometric_only(cfg, droplet_p=1.0)
    ph_strong.update(droplet_n=[4, 4], droplet_mode_b_p=1.0,
                      droplet_refractive_radius=[20.0, 20.0], droplet_refractive_minify=[0.3, 0.3],
                      droplet_refractive_alpha=[0.9, 0.9], droplet_refractive_blur_sigma=[2.0, 2.0])
    c_on = copy.deepcopy(cfg)
    c_on["synth"]["photometric"] = ph_strong
    c_off = copy.deepcopy(cfg)
    c_off["synth"]["photometric"] = _photometric_only(cfg)

    flipped = False
    for seed in range(30):
        rec_off, _ = generate_sample(c_off, np.random.default_rng(seed), bg_files, force_negative=False,
                                      occlude=False, components=dict(components))
        rec_on, meta_on = generate_sample(c_on, np.random.default_rng(seed), bg_files, force_negative=False,
                                           occlude=False, components=dict(components))
        vis_off = {c["index"]: c["visible"] for c in rec_off["corners"]}
        vis_on = {c["index"]: c["visible"] for c in rec_on["corners"]}
        if any(vis_off[i] and not vis_on[i] for i in vis_off):
            flipped = True
            assert len(meta_on["holes"]) >= 1
            break

    assert flipped, "expected at least one seed (of 30) where a forced strong droplet covered a visible corner"


# ------------------------------------------------------------- differencing --

def test_differencing(cfg, bg_files):
    """Petschnigg-et-al. flash/no-flash differencing: clipped-zero pixels
    appear, and a nonzero inter-frame shift creates a "doubled edge" (a band
    of clipped-zero AND anomalously bright pixels straddling a static
    high-contrast edge) that a perfectly-aligned zero-shift subtraction does
    not produce. A synthetic bright disc on a dark field is rotationally
    symmetric, so the check doesn't depend on the shift's randomly-drawn
    angle. Labels bind to the lit geometry: corner positions (from p_img/H
    alone, never touched by _apply_photometric) are checked unchanged
    end-to-end via generate_sample."""
    h2 = w2 = 240
    yy, xx = np.mgrid[0:h2, 0:w2]
    r = np.sqrt((xx - w2 / 2) ** 2 + (yy - h2 / 2) ** 2)
    disc_r = 60
    work = (np.where(r < disc_r, 220.0, 20.0).astype(np.float32))[..., None] * np.ones((1, 1, 3), np.float32)

    def run(shift_range):
        ph = _photometric_only(cfg, differencing_p=1.0)
        ph.update(differencing_ambient=[0.15, 0.15], differencing_illum_peak=[1.0, 1.0],
                  differencing_illum_floor=[1.0, 1.0], differencing_shift_px=shift_range)
        return _apply_photometric(work.copy(), np.random.default_rng(3), ph, w2, h2)

    zero_shift = run([0.0, 0.0])
    shifted = run([6.0, 6.0])

    assert np.any(shifted == 0)  # clipped-zero pixels present

    annulus = (r > disc_r - 10) & (r < disc_r + 10)
    zero_count = np.count_nonzero((zero_shift[..., 0] == 0) & annulus)
    shift_count = np.count_nonzero((shifted[..., 0] == 0) & annulus)
    print("test_differencing zero-clipped pixels in edge annulus: shift0=%d shift6=%d" % (zero_count, shift_count))
    assert shift_count > zero_count + 20

    components = {"s": 80.0, "theta": 0.1, "shear_x": 0.0, "shear_y": 0.0, "tx": 5.0, "ty": -5.0}
    c_off = copy.deepcopy(cfg)
    c_off["synth"]["photometric"] = _photometric_only(cfg)
    c_diff = copy.deepcopy(cfg)
    c_diff["synth"]["photometric"] = _photometric_only(cfg, differencing_p=1.0)

    rec_off, _ = generate_sample(c_off, np.random.default_rng(19), bg_files, force_negative=False,
                                  occlude=False, components=dict(components))
    rec_diff, _ = generate_sample(c_diff, np.random.default_rng(19), bg_files, force_negative=False,
                                   occlude=False, components=dict(components))
    corners_off = [(c["x"], c["y"]) for c in rec_off["corners"]]
    corners_diff = [(c["x"], c["y"]) for c in rec_diff["corners"]]
    assert corners_off == corners_diff


def test_dark_greyout_output_coupled(cfg):
    """Gate-defect regression (found at review): the differencing branch's
    grey-out gate must key off the CLIPPED OUTPUT's own brightness, not the
    ambient draw alone. A low ambient (0.03, below the OLD ambient-based
    threshold of 0.05) paired with a high illumination peak still clips to a
    bright frame here -- under the old (buggy) ambient-only gate this would
    have been wrongly grey-ed out (crushed toward its own, much dimmer,
    mean); under the fixed output-percentile gate it must not be touched."""
    h2 = w2 = 120
    yy, xx = np.mgrid[0:h2, 0:w2]
    r = np.sqrt((xx - w2 / 2) ** 2 + (yy - h2 / 2) ** 2)
    disc_r = 30
    work = (np.where(r < disc_r, 220.0, 20.0).astype(np.float32))[..., None] * np.ones((1, 1, 3), np.float32)

    ph = _photometric_only(cfg, differencing_p=1.0)
    ph.update(differencing_ambient=[0.03, 0.03], differencing_illum_peak=[2.0, 2.0],
              differencing_illum_floor=[2.0, 2.0], differencing_shift_px=[0.0, 0.0])
    out = _apply_photometric(work.copy(), np.random.default_rng(1), ph, w2, h2)

    # disc region: clip(220*2.0 - 220*0.03*2.0, 0, 255) = clip(433.4, 0, 255)
    # = 255 -- grey-out (blend toward the whole frame's own, much dimmer,
    # mean) would have pulled this well below 255; a bare identity-shift
    # warpAffine can leave a few interpolation-boundary pixels short of the
    # disc's own true radius, so check comfortably inside it, not the edge.
    inner = r < disc_r - 5
    print("test_dark_greyout_output_coupled inner-disc min:", out[..., 0][inner].min())
    assert out[..., 0][inner].min() > 240


# ------------------------------------------------------- sensor noise (P-G) --

def test_poisson_gaussian_shot_noise(cfg):
    """Poissonian-Gaussian sensor noise (Foi et al. 2008): variance must
    grow with signal intensity (the shot-noise signature), unlike the
    legacy constant-variance additive-Gaussian draw -- a regression over two
    flat patches of different brightness, many independent draws each."""
    h2 = w2 = 48

    def var_at(level, sensor_noise_enabled, n=200):
        ph = _photometric_only(cfg, gauss_noise_p=1.0)
        ph = dict(ph, sensor_noise_enabled=sensor_noise_enabled,
                  sensor_noise_electrons_per_dn=[1.0, 1.0], sensor_noise_read_std=[1.5, 1.5],
                  noise_std=[5.0, 5.0])
        work = np.full((h2, w2, 3), float(level), dtype=np.float32)
        vals = [_apply_photometric(work.copy(), np.random.default_rng(1000 + i), ph, w2, h2)[0, 0, 0]
                for i in range(n)]
        return float(np.var(vals))

    pg_dark, pg_bright = var_at(20.0, True), var_at(200.0, True)
    legacy_dark, legacy_bright = var_at(20.0, False), var_at(200.0, False)
    print("test_poisson_gaussian_shot_noise variance: PG dark=%.2f bright=%.2f, legacy dark=%.2f bright=%.2f"
          % (pg_dark, pg_bright, legacy_dark, legacy_bright))

    assert pg_bright > pg_dark * 3  # shot noise: variance scales with intensity
    assert legacy_bright == pytest.approx(legacy_dark, rel=0.5)  # legacy: constant variance regardless of level


# ---------------------------------------------------------------- vignette --

def test_vignette_falloff(cfg):
    """Multiplicative radial falloff about the canvas centre: corners must
    end up darker than the centre."""
    h2 = w2 = 200
    work = np.full((h2, w2, 3), 200.0, dtype=np.float32)
    ph = _photometric_only(cfg, vignette_p=1.0)
    ph = dict(ph, vignette_strength=[0.25, 0.25])
    out = _apply_photometric(work.copy(), np.random.default_rng(3), ph, w2, h2)
    assert out[2, 2, 0] < out[h2 // 2, w2 // 2, 0]


# ------------------------------------------------------------ ink-contrast --

def test_ink_contrast_lightens_dark_board_pixels(cfg):
    """NIR ink-contrast jitter: board pixels darker than the board's own
    median get scaled up (toward grey), capped below the board's own white
    level; pixels at/above the median are untouched."""
    h2 = w2 = 60
    work = np.full((h2, w2, 3), 200.0, dtype=np.float32)
    work[:30] = 20.0
    mask = np.ones((h2, w2), dtype=np.float32)
    ph = _photometric_only(cfg, ink_contrast_p=1.0)
    ph = dict(ph, ink_contrast_scale=[1.5, 1.5])
    out = _apply_photometric(work.copy(), np.random.default_rng(2), ph, w2, h2, board_mask=mask)
    assert np.allclose(out[:30], 30.0, atol=1.0)
    assert np.allclose(out[30:], 200.0)


# --------------------------------------------------------------------- FPN --

def test_fpn_full_canvas_only(cfg):
    """Fixed-pattern noise fires in full-canvas mode (window_origin=None)
    and never in window mode -- the documented fast-arm restriction."""
    h2 = w2 = 80
    work = np.full((h2, w2, 3), 128.0, dtype=np.float32)
    ph = _photometric_only(cfg, fpn_p=1.0)

    out_full = _apply_photometric(work.copy(), np.random.default_rng(6), ph, w2, h2, window_origin=None)
    out_win = _apply_photometric(work.copy(), np.random.default_rng(6), ph, w2, h2, window_origin=(5, 5))

    assert not np.allclose(out_full, 128.0)
    assert np.array_equal(out_win, work)


# ------------------------------------------------------------ board sampling --

def test_sample_on_mask_uniform():
    """_sample_on_mask (specular's lobe-centre reject-sampler): every draw
    must land inside the mask, and over enough draws should span both ends
    of a simple rectangular mask -- a coarse uniformity smoke check."""
    rng = np.random.default_rng(5)
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[20:80, 30:70] = 1.0
    xs, ys = [], []
    for _ in range(500):
        x, y = _sample_on_mask(rng, mask)
        # truncation, not rounding: matches _sample_on_mask's own internal
        # int(y)/int(x) validation, and this codebase's established
        # half-open continuous-to-pixel convention (see visible()'s own
        # docstring).
        assert mask[min(int(y), 99), min(int(x), 99)] > 0.5
        xs.append(x)
        ys.append(y)
    xs, ys = np.array(xs), np.array(ys)
    assert xs.min() < 40 and xs.max() > 60
    assert ys.min() < 30 and ys.max() > 70


# ------------------------------------------------------------------ config --

def test_new_config_keys_present():
    """Both configs carry the full rev-2 key set -- a cheap regression guard
    against a key added to one file and forgotten in the other."""
    required = ("specular_p", "specular_strength", "specular_exponent", "droplet_p", "droplet_n",
                "droplet_mode_b_p", "droplet_hole_alpha_thresh", "droplet_hole_radius_thresh",
                "vignette_p", "vignette_strength", "ink_contrast_p", "ink_contrast_scale",
                "differencing_p", "differencing_ambient", "differencing_illum_peak",
                "differencing_illum_floor", "differencing_shift_px", "dark_greyout_white_thresh",
                "dark_greyout_brightness_thresh", "dark_greyout_blend", "sensor_noise_enabled",
                "sensor_noise_electrons_per_dn", "sensor_noise_read_std", "fpn_p", "fpn_col_std",
                "fpn_prnu_p", "fpn_prnu_std")
    for rel in ("configs/default.yaml", "configs/rev640.yaml"):
        c = yaml.safe_load(open(Path(__file__).parents[1] / rel))
        ph = c["synth"]["photometric"]
        missing = [k for k in required if k not in ph]
        assert not missing, f"{rel} missing keys: {missing}"
