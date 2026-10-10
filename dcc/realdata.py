"""Real-frame training data: harvested video pseudo-labels (dcc.video) as detector samples, mixed
into the synthetic stream at a capped fraction (cfg["real"]["frac"]).

Why the synthetic stream stays in the mix: a video covers one camera, one board and the
conditions that happened to be filmed. Synthetic samples keep every other condition the model
was trained for in front of it, and that fraction is the cap on how far a real clip can pull the
model -- the confirmation-bias guard self-training cannot do without. Any image directory serves
as the background corpus (dcc.synth.list_backgrounds); it does not have to be COCO.

A real record's labels come with what they do NOT know, and the masks carry it (dcc.losses
mask=, 0 = no gradient):
  visible None   position known, observability not -> a disc around it is ignored in the heatmap,
                 and that corner's own class channel is ignored near it (other channels stay
                 negative: no OTHER corner is there either way)
  sigma > tight  the corner is real but its exact pixel is not -> the ring of pixels around the
                 positive is ignored, so a one-pixel label error is not taught as "the true pixel is
                 a negative" (the release heatmaps are one-hot BCE, where that is a full-weight error)
Every positive keeps its own pixel and cell whatever overlaps it.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from dcc.board import n_corners
from dcc.dataset import _render_detector_targets
from dcc.synth import _apply_photometric, generate_sample, list_backgrounds
from dcc.targets import render_class_targets, render_heatmap

REAL_DEFAULTS = {
    "frac": 0.3,              # fraction of samples drawn from real frames
    "darken_p": 0.4,          # darken into the clip's OWN dark regime with its OWN noise (see darken)
    "darken_gain_min": None,  # None: from the clip (its darkest board level / its median bright one)
    "darken_hard_frac": 0.5,  # of darkened draws, the share taken from [gain_min, 4 gain_min]: log-uniform over the
                              #   whole range puts ~6% there, and that band is where the model fails
    "photometric_p": 0.3,     # else dcc.synth's own photometric pack, applied to a real frame
    "darken_synthetic_p": 0.0,  # darken SYNTHETIC samples the same way (the video calibrating the simulator):
                                #   the clip shows how dark and how noisy the camera gets; synthetic scenes
                                #   supply the diversity one clip lacks
    "geometric_p": 0.5,       # random rotation / scale / shift (never a flip: a mirrored board is another board)
    "rot_deg": 20.0,
    "scale": [0.8, 1.25],
    "translate_frac": 0.08,
    "unknown_radius_px": 3.0,  # + 3 sigma: the ignored disc around an unknown corner
    "tight_sigma": 0.15,       # input px; a positive this certain is a strict one-hot target
}


def ring_radius(sigma, tight=0.15):
    """Ignored ring around a positive, by its label sigma (input px)."""
    return 0 if sigma <= tight else 1 if sigma <= 0.5 else 2


def render_real_targets(cfg, image, corners, rcfg=None):
    """Detector targets for one real frame plus the masks (uint8, 1 = trained) that carry each
    label's uncertainty. Same keys as dcc.dataset._render_detector_targets, plus hm_mask/cls_mask."""
    rcfg = {**REAL_DEFAULTS, **(rcfg or {})}
    h, w = image.shape
    n_cls = n_corners(cfg.get("board"))
    pos = [c for c in corners if c["visible"] is True]
    pts = np.array([[c["x"], c["y"]] for c in pos], dtype=np.float64).reshape(-1, 2)
    idx = np.array([c["index"] for c in pos], dtype=int)
    vis = np.ones(len(pos), dtype=bool)
    hm = render_heatmap(pts, vis, (w, h), sigma=cfg["sigma_hm"])
    ct = render_class_targets(pts, vis, idx, (w, h), sigma=cfg["sigma_cls"], n_cls=n_cls)
    hm_mask = np.ones((h, w), np.uint8)
    cls_mask = np.ones((n_cls, h // 4, w // 4), np.uint8)
    for c in corners:
        if c["visible"] is False:
            continue
        x, y, k = c["x"], c["y"], c["index"]
        if c["visible"] is None:
            R = rcfg["unknown_radius_px"] + 3.0 * c["sigma"]
            _zero_disc(hm_mask, x, y, R)
            _zero_disc(cls_mask[k], (x + 0.5) / 4 - 0.5, (y + 0.5) / 4 - 0.5, 1.5)
        else:
            r = ring_radius(c["sigma"], rcfg["tight_sigma"])
            if r:
                jx, jy = int(np.rint(x)), int(np.rint(y))
                hm_mask[max(0, jy - r):jy + r + 1, max(0, jx - r):jx + r + 1] = 0
            if c["sigma"] > 0.5:                    # near a cell edge the forced cell itself may be off by one
                jx, jy = int(np.floor((x + 0.5) / 4)), int(np.floor((y + 0.5) / 4))
                cls_mask[k, max(0, jy - 1):jy + 2, max(0, jx - 1):jx + 2] = 0
    for c in pos:                                   # a positive always keeps its own pixel and cell
        jx, jy = int(np.rint(c["x"])), int(np.rint(c["y"]))
        if 0 <= jx < w and 0 <= jy < h:
            hm_mask[jy, jx] = 1
        jx, jy = int(np.floor((c["x"] + 0.5) / 4)), int(np.floor((c["y"] + 0.5) / 4))
        if 0 <= jx < w // 4 and 0 <= jy < h // 4:
            cls_mask[c["index"], jy, jx] = 1
    return {"image": torch.from_numpy(image).float().unsqueeze(0) / 255.0, "heatmap": torch.from_numpy(hm),
            "classes": torch.from_numpy(ct), "n_vis": len(pos),
            "hm_mask": torch.from_numpy(hm_mask), "cls_mask": torch.from_numpy(cls_mask)}


def _zero_disc(mask, x, y, R):
    h, w = mask.shape
    x0, x1 = max(0, int(np.floor(x - R))), min(w - 1, int(np.ceil(x + R)))
    y0, y1 = max(0, int(np.floor(y - R))), min(h - 1, int(np.ceil(y + R)))
    if x0 > x1 or y0 > y1:
        return
    yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    mask[y0:y1 + 1, x0:x1 + 1][(xx - x) ** 2 + (yy - y) ** 2 <= R * R] = 0


def darken_gain_min(noise, rcfg):
    """Lowest gain the darkening draws: the clip's own range (its darkest board level over its median
    bright one) unless configured; 0.1 for a clip with no dark frames to calibrate from."""
    if rcfg.get("darken_gain_min") is not None:
        return float(rcfg["darken_gain_min"])
    lo, hi = (noise or {}).get("board_level_p5"), (noise or {}).get("board_level_bright_median")
    if not lo or not hi:
        return 0.1
    return float(np.clip(lo / hi, 0.002, 0.5))


def darken(image, rng, noise, gain_min, hard_frac=0.0):
    """A real frame re-exposed at gain g ~ logU(gain_min, 1), with the clip's own Poisson-Gaussian noise
    (dcc.video.noise_model: var = shot * level + read_var) at the new level, then quantised as the
    camera quantises. The noise ADDED is the target variance minus what scaling leaves of the frame's
    own noise, so g = 1 is (statistically) the frame itself.

    WHY: a clip's dark stretch is one board pose sequence over one background -- measured on a synthetic
    clip, fine-tuning on it alone raised dark recall on the clip itself from 19% to 72% and on a held-out
    clip not at all (12% -> 12.5%): it memorised. Every bright frame, re-exposed into the same regime,
    carries the poses and backgrounds the dark stretch lacks."""
    hi = min(1.0, 4.0 * gain_min) if rng.random() < hard_frac else 1.0
    g = float(np.exp(rng.uniform(np.log(gain_min), np.log(hi))))
    I = image.astype(np.float64)
    a, b = noise["shot"], noise["read_var"]
    add = a * g * I + b - g * g * (a * I + b)
    out = g * I + rng.normal(0.0, 1.0, I.shape) * np.sqrt(np.maximum(add, 0.0))
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def augment_real(image, corners, outline, rng, cfg, rcfg=None, noise=None):
    """Augment one real frame; labels move with the pixels. Geometric: rotation/scale/shift about the
    centre (replicated border -- a reflected border would paint a MIRRORED board, which is a different
    board). Then ONE of: darkening into the clip's own dark regime (darken, needs the clip's noise model),
    or dcc.synth._apply_photometric -- the synthetic stream's own pack, with the board mask from the
    record's outline so its board-anchored steps fire on the real board (a droplet that registers an
    occluding hole turns the corners under it to unknown) -- or neither. Never both: each brings its own
    noise. Darkened corners keep their labels, as degraded corners do in the synthetic stream."""
    rcfg = {**REAL_DEFAULTS, **(rcfg or {})}
    h, w = image.shape
    corners = [dict(c) for c in corners]
    outline = np.asarray(outline, dtype=np.float64).reshape(-1, 2)
    if rng.random() < rcfg["geometric_p"]:
        ang = rng.uniform(-rcfg["rot_deg"], rcfg["rot_deg"])
        lo, hi = rcfg["scale"]
        sc = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        tx, ty = rng.uniform(-rcfg["translate_frac"], rcfg["translate_frac"], size=2) * (w, h)
        M = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), ang, sc)
        M[:, 2] += (tx, ty)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        for c in corners:
            c["x"], c["y"] = (M @ np.array([c["x"], c["y"], 1.0])).tolist()
            c["sigma"] = c["sigma"] * sc
            jx, jy = int(np.rint(c["x"])), int(np.rint(c["y"]))
            if c["visible"] is not False and not (0 <= jx < w and 0 <= jy < h):
                c["visible"] = False
        outline = np.hstack([outline, np.ones((len(outline), 1))]) @ M.T
    coin = rng.random()
    if noise is not None and coin < rcfg["darken_p"]:
        image = darken(image, rng, noise, darken_gain_min(noise, rcfg), rcfg["darken_hard_frac"])
    elif rcfg["darken_p"] <= coin < rcfg["darken_p"] + rcfg["photometric_p"]:
        mask = np.zeros((h, w), np.float32)
        cv2.fillConvexPoly(mask, np.rint(outline).astype(np.int32), 1.0)
        work = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR).astype(np.float32)
        holes = []
        work = _apply_photometric(work, rng, cfg["synth"]["photometric"], w, h, board_mask=mask,
                                  board_centroid=tuple(outline.mean(axis=0)), holes_out=holes)
        image = cv2.cvtColor(np.clip(work, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        for x0, y0, hw, hh in holes:
            for c in corners:
                if c["visible"] is True and x0 - 0.5 <= c["x"] < x0 + hw - 0.5 and y0 - 0.5 <= c["y"] < y0 + hh - 0.5:
                    c["visible"] = None
    return image, corners


class RealFrames:
    """Every record of one or more harvest directories (dcc.video.save_harvest), images memory-mapped.
    Build it inside the worker process: a memmap pickled through a spawn DataLoader arrives as a full
    in-memory copy per worker."""

    def __init__(self, dirs):
        self.items = []
        for d in ([dirs] if isinstance(dirs, (str, Path)) else dirs):
            d = Path(d)
            data = json.loads((d / "records.json").read_text())
            recs, noise = data["records"], (data.get("meta") or {}).get("noise")
            imgs = np.load(d / "images.npy", mmap_mode="r")
            assert len(imgs) == len(recs), f"{d}: {len(imgs)} images for {len(recs)} records"
            self.items += [(imgs, i, r, noise) for i, r in enumerate(recs)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        """(image, record, the clip's noise model or None)."""
        imgs, j, rec, noise = self.items[i]
        return np.array(imgs[j]), rec, noise


class MixedStream(IterableDataset):
    """dcc.dataset.SynthStream's detector stream with real frames mixed in: one explicit-Generator coin
    per sample against real.frac (determinism contract as SynthStream: rng = default_rng([seed, worker])).
    Every sample carries hm_mask/cls_mask -- all ones for synthetic samples, whose labels are exact."""

    def __init__(self, cfg, seed=None):
        self.cfg, self.seed = cfg, seed
        self.rcfg = {**REAL_DEFAULTS, **cfg["real"]}

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info is not None else 0
        rng = np.random.default_rng([self.seed, wid]) if self.seed is not None else np.random.default_rng()
        frac = float(self.rcfg["frac"])
        real = RealFrames(self.rcfg["records"])
        assert len(real), f"real.records {self.rcfg['records']} hold no frames"
        noises = [it[3] for it in real.items if it[3]]
        syn_noise = noises[0] if noises else None      # first clip's camera model for the synthetic stream
        bg_files = list_backgrounds(self.cfg["synth"]["backgrounds"]) if frac < 1.0 else []
        assert frac >= 1.0 or bg_files, f"no background images under {self.cfg['synth']['backgrounds']!r}"
        W, H = self.cfg["input_size"]
        n_cls = n_corners(self.cfg.get("board"))
        ones_hm = torch.ones((H, W), dtype=torch.uint8)
        ones_cls = torch.ones((n_cls, H // 4, W // 4), dtype=torch.uint8)
        while True:
            if rng.random() < frac:
                img, rec, noise = real[int(rng.integers(len(real)))]
                assert img.shape == (H, W), f"real frame {img.shape} != input_size {(H, W)}"
                img, corners = augment_real(img, rec["corners"], rec["outline"], rng, self.cfg, self.rcfg, noise)
                yield render_real_targets(self.cfg, img, corners, self.rcfg)
            else:
                record, _ = generate_sample(self.cfg, rng, bg_files)
                if syn_noise is not None and rng.random() < self.rcfg["darken_synthetic_p"]:
                    record["image"] = darken(record["image"], rng, syn_noise, darken_gain_min(syn_noise, self.rcfg),
                                             self.rcfg["darken_hard_frac"])
                t = _render_detector_targets(self.cfg, record)
                t["hm_mask"], t["cls_mask"] = ones_hm, ones_cls
                yield t
