"""Torch datasets over dcc.synth: SynthStream(cfg, stream="detector"|"refiner", seed, render_targets)
for training, SynthVal and RefinerVal for fixed-seed validation; load_config(path) reads a YAML config.
"""
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from dcc.board import n_corners
from dcc.refiner_data import mixed_refiner_crops
from dcc.synth import cut_refiner_crops, generate_sample, list_backgrounds, make_generic_crop
from dcc.targets import render_class_targets, render_heatmap, render_refiner_target


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _render_detector_targets(cfg, record):
    h, w = record["image"].shape
    corners = record["corners"]
    pts = np.array([[c["x"], c["y"]] for c in corners], dtype=np.float64).reshape(-1, 2)
    vis = np.array([c["visible"] for c in corners], dtype=bool)
    idx = np.array([c["index"] for c in corners], dtype=int)
    heatmap = render_heatmap(pts, vis, (w, h), sigma=cfg["sigma_hm"])
    classes = render_class_targets(pts, vis, idx, (w, h), sigma=cfg["sigma_cls"], n_cls=n_corners(cfg.get("board")))
    image = torch.from_numpy(record["image"]).float().unsqueeze(0) / 255.0
    return {"image": image, "heatmap": torch.from_numpy(heatmap), "classes": torch.from_numpy(classes),
            "n_vis": int(vis.sum())}


def _render_refiner_sample(crop, sigma=1.5):
    image = torch.from_numpy(crop["crop"]).float().unsqueeze(0) / 255.0
    return {"crop": image, "target": torch.from_numpy(render_refiner_target(crop["d"], sigma=sigma))}


def _maybe_replace_generic(crop, rng, frac):
    if frac > 0 and rng.random() < frac:
        return make_generic_crop(rng)
    return crop


class SynthStream(IterableDataset):
    def __init__(self, cfg, stream="detector", seed=None, render_targets=False):
        self.cfg = cfg
        self.stream = stream
        self.seed = seed
        self.render_targets = render_targets

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info is not None else 0
        rng = np.random.default_rng([self.seed, wid]) if self.seed is not None else np.random.default_rng()
        bg_files = list_backgrounds(self.cfg["synth"]["backgrounds"])

        if self.stream == "detector":
            while True:
                record, _ = generate_sample(self.cfg, rng, bg_files)
                if self.render_targets:
                    yield _render_detector_targets(self.cfg, record)
                else:
                    yield record["image"], record
        else:
            frac = self.cfg["synth"].get("refiner_generic_frac", 0.0)
            sigma = self.cfg.get("sigma_ref", 1.5)
            while True:
                for crop in mixed_refiner_crops(self.cfg, rng, bg_files):
                    crop = _maybe_replace_generic(crop, rng, frac)
                    yield _render_refiner_sample(crop, sigma) if self.render_targets else crop


class SynthVal(Dataset):
    def __init__(self, cfg, n, seed=None):
        self.cfg = cfg
        self.n = n
        self.val_seed = cfg["synth"]["val_seed"] if seed is None else seed
        self.bg_files = list_backgrounds(cfg["synth"]["backgrounds"])

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng([self.val_seed, i])
        if rng.random() < self.cfg["negative_p"]:
            record, _ = generate_sample(self.cfg, rng, self.bg_files, force_negative=True)
        else:
            a, b = self.cfg["scale_range_px"]
            s = a * (b / a) ** ((i + rng.random()) / self.n)
            record, _ = generate_sample(self.cfg, rng, self.bg_files, s=s, force_negative=False)
        return record["image"], record


class RefinerVal(Dataset):
    def __init__(self, cfg, n, seed=None):
        self.cfg = cfg
        self.n = n
        self.val_seed = cfg["synth"]["val_seed"] + 1 if seed is None else seed
        self.bg_files = list_backgrounds(cfg["synth"]["backgrounds"])

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng([self.val_seed, i])
        size_mult = self.cfg["synth"]["refiner_res_mult"]
        record, _ = generate_sample(self.cfg, rng, self.bg_files, size_mult=size_mult,
                                     occlude=False, force_negative=False)
        pts = [(c["x"], c["y"]) for c in record["corners"] if c["visible"]]
        crops = cut_refiner_crops(self.cfg, rng, record["image"], pts)
        frac = self.cfg["synth"].get("refiner_generic_frac", 0.0)
        return [_maybe_replace_generic(c, rng, frac) for c in crops]
