"""tools/eval_checkpoint.py -- standalone checkpoint evaluation CLI.

Reuses tools/train_detector.py's own run_validation (the canonical M-01/
M-02/M-04 scorer) and tools/train_refiner.py's own run_refiner_validation
(M-03) -- loaded by absolute path via importlib, since an unrelated
detectron2 'tools' package on sys.path shadows `import
tools.train_detector`/`import tools.train_refiner` (see
tests/test_variant640.py and tools/train_charuconet.py's identical fixture).
Reusing the exact scoring code means every number here is directly
comparable to the training-loop validations already on record for the same
metric/config -- no reimplementation, no drift.

--ckpt alone evaluates the detector (M-01/M-02/M-04 on SynthVal); --refiner-
ckpt alone evaluates the refiner (M-03 on RefinerVal); both together run
both, reported side by side. EMA weights are preferred when present
(ckpt["ema"]), else ckpt["model"] -- same preference as tools/introspect.py's
_load_ckpt.

DataLoader workers are forced to 0 (single-process): the collate_fn pulled
off the importlib-loaded module is bound to a synthetic module name a
spawned worker process can't re-import, so multiprocessing spawn would fail
at unpickling. A one-off eval CLI doesn't need the loader parallelism a
training loop does.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default=None, help="detector checkpoint (.pt); runs M-01/M-02/M-04 if given")
    p.add_argument("--refiner-ckpt", default=None, help="refiner checkpoint (.pt); runs M-03 if given")
    p.add_argument("--config", default="configs/rev640.yaml")
    p.add_argument("--n", type=int, default=None,
                    help="val samples; default cfg.synth.val_size (detector) / "
                         "cfg.synth.refiner_val_composites (refiner)")
    p.add_argument("--out", default=None, help="optional JSON output path")
    return p


def _synth_hash(cfg):
    """sha256 (first 12 hex chars) of cfg["synth"], canonicalised via sorted-
    key JSON -- self-describing provenance for which resolved photometric/
    generator block produced a given SynthVal/RefinerVal instance. SynthVal
    is "bit-identical-by-construction" for a given (config, val_seed), but
    its identity is DERIVED FROM THE CONFIG: editing configs/rev640.yaml
    changes which images the val set contains, not just what's scored. Two
    results are numbers-comparable only if this hash (and the config path)
    match; if it doesn't, they were scored on different image sets and any
    delta between them is a distribution change, not a like-for-like gap."""
    import hashlib
    import json
    return hashlib.sha256(json.dumps(cfg["synth"], sort_keys=True).encode()).hexdigest()[:12]


def _load_module(name, relpath):
    """tools/<relpath>, loaded by absolute path -- bypasses the detectron2
    'tools' package that shadows `import tools.<name>` (see
    tests/test_variant640.py's identical fixture)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_ckpt(model, path, device):
    """torch.load; accepts a raw state_dict OR a trainer ckpt dict with
    'model'/'ema' keys (ema preferred) -- same contract as
    tools/introspect.py's _load_ckpt. Returns the full loaded object so the
    caller can pull step/cfg provenance off it."""
    import torch
    obj = torch.load(path, map_location=device, weights_only=False)
    sd = obj.get("ema", obj.get("model")) if isinstance(obj, dict) and ("model" in obj or "ema" in obj) else obj
    model.load_state_dict(sd)
    return obj


def eval_detector(ckpt_path, cfg, config_path, n, device):
    from functools import partial

    import torch
    from torch.utils.data import DataLoader

    from dcc.board import n_corners
    from dcc.dataset import SynthVal
    from dcc.model import DetectorNet, detector_kwargs

    train_detector = _load_module("_train_detector_eval", "tools/train_detector.py")

    W, H = cfg["input_size"]
    model = DetectorNet(H, W, n_cls=n_corners(cfg.get("board")),
                         **detector_kwargs(cfg)).to(device, memory_format=torch.channels_last)
    obj = _load_ckpt(model, ckpt_path, device)
    model.eval()

    n = n if n is not None else cfg["synth"]["val_size"]
    val_ds = SynthVal(cfg, n=n, seed=cfg["synth"]["val_seed"])
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch"], num_workers=0,
                             collate_fn=partial(train_detector._val_collate, cfg))

    tcfg = cfg["train"]
    result = train_detector.run_validation(model, val_loader, cfg, device, tcfg["tau_hm"], tcfg["match_px"])
    result.update(step=obj.get("step") if isinstance(obj, dict) else None, config=str(config_path),
                   ckpt=str(ckpt_path), n=n, synth_hash=_synth_hash(cfg))
    return result


def eval_refiner(ckpt_path, cfg, config_path, n, device):
    import torch
    from torch.utils.data import DataLoader

    from dcc.dataset import RefinerVal
    from dcc.model import Refiner

    train_refiner = _load_module("_train_refiner_eval", "tools/train_refiner.py")

    model = Refiner().to(device, memory_format=torch.channels_last)
    obj = _load_ckpt(model, ckpt_path, device)
    model.eval()

    n = n if n is not None else cfg["synth"]["refiner_val_composites"]
    val_ds = RefinerVal(cfg, n=n)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=0, collate_fn=train_refiner._refiner_val_collate)

    result = train_refiner.run_refiner_validation(model, val_loader, device)
    result.update(step=obj.get("step") if isinstance(obj, dict) else None, config=str(config_path),
                   ckpt=str(ckpt_path), n=n, synth_hash=_synth_hash(cfg))
    return result


def _print_detector(result):
    m01, m02, m04 = result["m01"], result["m02"], result["m04"]
    print(f"\n[detector] ckpt={result['ckpt']} step={result['step']} config={result['config']} n={result['n']} "
          f"synth_hash={result['synth_hash']}")
    print(f"  val_loss={result['val_loss']}")
    if m01["mean"] is not None:
        print(f"  M-01: mean={m01['mean']:.3f}px median={m01['median']:.3f}px p95={m01['p95']:.3f}px "
              f"tail_frac_gt4px={m01['tail_frac_gt4px']:.4f} n_matched={m01['n_matched']}")
    else:
        print("  M-01: no matches")
    print(f"  M-02 by octave: {m02}")
    if m04 is not None:
        print(f"  M-04: accuracy={m04['accuracy']:.4f} by_octave={m04['by_octave']}")
    else:
        print("  M-04: unavailable (dcc.pipeline.read_ids not importable)")


def _print_refiner(result):
    m03 = result["m03"]
    print(f"\n[refiner] ckpt={result['ckpt']} step={result['step']} config={result['config']} n={result['n']} "
          f"synth_hash={result['synth_hash']}")
    print(f"  val_loss={result['val_loss']}")
    print(f"  M-03 (px): median={m03['median_px']:.4f} mean={m03['mean_px']:.4f} "
          f"p95={m03['p95_px']:.4f} n={m03['n']}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not args.ckpt and not args.refiner_ckpt:
        parser.error("at least one of --ckpt/--refiner-ckpt is required")

    import json

    import torch

    from dcc.dataset import load_config

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    cfg = load_config(args.config)

    out = {}
    if args.ckpt:
        out["detector"] = eval_detector(args.ckpt, cfg, args.config, args.n, device)
        _print_detector(out["detector"])
    if args.refiner_ckpt:
        out["refiner"] = eval_refiner(args.refiner_ckpt, cfg, args.config, args.n, device)
        _print_refiner(out["refiner"])

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
