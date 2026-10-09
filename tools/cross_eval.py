"""tools/cross_eval.py -- team-lead 2026-07-28 queue item 5: the rev-2/rev-3
2x2 cross-evaluation (two models x two validation distributions) answering
Kaelin's question -- did the rev-3 retraining lose anything on the old
(rev-2) distribution, and was retraining worth it against the new (rev-3)
one?

                          | rev-2 val (X)        | rev-3 val (Y)
  rev-2 model  (A)        | KNOWN (trainer log)  | MEASURE (this script)
  rev-3 model  (B, final) | MEASURE (this script)| KNOWN (trainer log)

Reuses tools/train_detector.py's own run_validation/_val_collate (imported
by absolute path -- the tools/ package-shadowing footgun, see
_load_train_detector below) rather than reimplementing the M-01/M-02/M-04
protocol a third time: run_validation is what the trainer itself calls, so
this script's numbers are computed identically to the "KNOWN" cells already
in hand (no refiner, raw heatmap-peak + bilinear ID readout -- the same
convention every M-04 number quoted so far in this investigation uses; do
NOT confuse this with dcc.pipeline.detect's full refiner-based pipeline,
which is a different, not-directly-comparable number).

***CRITICAL USAGE NOTE, READ BEFORE RUNNING***
Which git tree this SCRIPT FILE physically lives in determines which dcc/
generator code it imports -- `sys.path.insert(0, str(Path(__file__).
resolve().parents[1]))` below resolves relative to this file's own path on
disk, NOT the process's current working directory. Running this file via an
absolute path that points at the CURRENT (rev-3) tree while `cd`'d into a
rev-2 worktree does NOT make it import the rev-2 generator -- it silently
imports rev-3's dcc/dataset.py + dcc/synth.py regardless. To evaluate the
rev-2 (X) val distribution correctly:
    git worktree add <scratch-path> 8ed5684
    cp tools/cross_eval.py <scratch-path>/tools/cross_eval.py   # THIS COPY STEP IS REQUIRED
    cd <scratch-path>
    PYTHONPATH= <conda-python> tools/cross_eval.py --ckpt <abs path to model B> \\
        --config configs/rev640.yaml --tag modelB_on_X --out <abs out path>
    cd -; git worktree remove <scratch-path>
For the rev-3 (Y) val distribution, just run this file normally from the
current tree with model A's checkpoint:
    PYTHONPATH= <conda-python> tools/cross_eval.py --ckpt <abs path to model A> \\
        --config configs/rev640.yaml --tag modelA_on_Y --out <abs out path>
Model checkpoints are always passed by ABSOLUTE path so this works
regardless of which tree is the current working directory (dcc/model.py,
dcc/losses.py, dcc/pipeline.py are byte-identical between the rev-2 and
rev-3 commits -- team-lead verified diff -- so a checkpoint from either era
loads cleanly in either tree's DetectorNet).

Self-contained trainer-health gate (does NOT import tools/factor_sweep.py):
that file doesn't exist in the rev-2 commit, so any cross-worktree import of
it would break the rev-2 cell. The tiny health-check duplicated below is a
deliberate, documented exception to this project's usual "don't duplicate"
rule, for exactly that reason.

GPU discipline (standing concurrency rules): small val subset, batch<=32,
workers kept light, trainer health re-checked before starting.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
TRAINER_METRICS = "runs/rev640_baseline/metrics.jsonl"


def _check_trainer_health(metrics_path=TRAINER_METRICS, max_age_s=180, min_sps=100.0):
    """Self-contained duplicate of tools/factor_sweep.py's own
    check_trainer_health (see module docstring for why this can't just
    import that file) -- including its torn-last-line fallback (memory.md
    2026-07-28 10:45's crash postmortem: a live trainer can be mid-write on
    its own last line)."""
    with open(metrics_path, "rb") as f:
        f.seek(0, 2)
        f.seek(max(f.tell() - 4096, 0))
        lines = [l for l in f.read().decode(errors="ignore").strip().splitlines() if l.strip()]
    for candidate in reversed(lines[-2:]):
        try:
            d = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    else:
        raise RuntimeError(f"could not parse a trainer metrics line from the tail of {metrics_path}")
    age = time.time() - d["wall"]
    sps = d.get("samples_per_s", 0.0)
    return age < max_age_s and sps >= min_sps, age, sps


def _wait_for_trainer_health(max_wait_s=300, poll_s=15):
    t0 = time.time()
    while True:
        ok, age, sps = _check_trainer_health()
        if ok:
            return ok, age, sps
        if time.time() - t0 > max_wait_s:
            raise RuntimeError(f"trainer unhealthy for >{max_wait_s}s (age={age:.0f}s sps={sps:.1f}) -- aborting")
        print(f"  [gpu-gate] trainer degraded (age={age:.0f}s sps={sps:.1f}) -- pausing {poll_s}s")
        time.sleep(poll_s)


def _load_train_detector():
    """tools.train_detector, loaded by absolute path -- an unrelated
    detectron2 'tools' package shadows any `import tools.train_detector`
    (same footgun/fixture as tools/train_charuconet.py's own
    _load_train_detector). Loaded from ROOT (THIS file's own directory's
    parent -- see module docstring's critical usage note), so it's the
    train_detector.py of whichever tree this script currently lives in."""
    spec = importlib.util.spec_from_file_location("_train_detector_ref", ROOT / "tools" / "train_detector.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, help="ABSOLUTE path to the checkpoint being evaluated")
    p.add_argument("--config", default="configs/rev640.yaml", help="config of THIS tree -- determines the val distribution")
    p.add_argument("--val-subset", type=int, default=None, help="default: cfg train.val_subset")
    p.add_argument("--workers", type=int, default=2, help="kept light per standing concurrency rules")
    p.add_argument("--batch", type=int, default=None, help="default: cfg train.batch, capped at 32")
    p.add_argument("--tag", required=True, help="label for this cell, e.g. 'modelB_on_X' -- goes into the "
                                                  "output so the 2x2 table can be assembled unambiguously")
    p.add_argument("--out", default=None, help="default: cross_eval_<tag>.json in the cwd")
    return p


def main():
    args = build_parser().parse_args()
    import torch
    from functools import partial
    from torch.utils.data import DataLoader

    from dcc.board import n_corners
    from dcc.dataset import SynthVal, load_config
    from dcc.model import DetectorNet, detector_kwargs

    ref = _load_train_detector()
    cfg = load_config(args.config)
    tcfg = cfg["train"]
    val_subset = args.val_subset or tcfg["val_subset"]
    batch = min(args.batch or tcfg["batch"], 32)

    ok, age, sps = _wait_for_trainer_health()
    print(f"[cross_eval tag={args.tag}] trainer health: ok={ok} age={age:.0f}s sps={sps:.1f}")
    print(f"[cross_eval tag={args.tag}] tree_root={ROOT} config={args.config} ckpt={args.ckpt} "
          f"val_subset={val_subset} batch={batch} tau_hm={tcfg['tau_hm']} match_px={tcfg['match_px']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W, H = cfg["input_size"]
    n_cls = n_corners(cfg.get("board"))
    model = DetectorNet(H, W, n_cls=n_cls, **detector_kwargs(cfg)).to(device).eval()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = ckpt.get("ema", ckpt.get("model")) if isinstance(ckpt, dict) and ("model" in ckpt or "ema" in ckpt) else ckpt
    model.load_state_dict(sd)

    val_ds = SynthVal(cfg, n=val_subset, seed=cfg["synth"]["val_seed"])
    val_loader = DataLoader(val_ds, batch_size=batch, num_workers=args.workers,
                             multiprocessing_context="spawn" if args.workers > 0 else None,
                             persistent_workers=args.workers > 0,
                             collate_fn=partial(ref._val_collate, cfg))

    t0 = time.time()
    result = ref.run_validation(model, val_loader, cfg, device, tcfg["tau_hm"], tcfg["match_px"])
    elapsed = time.time() - t0

    report = {"tag": args.tag, "tree_root": str(ROOT), "config": args.config, "ckpt": args.ckpt,
              "val_subset": val_subset, "batch": batch, "tau_hm": tcfg["tau_hm"], "match_px": tcfg["match_px"],
              "elapsed_s": elapsed, "val_loss": result["val_loss"], "m01": result["m01"],
              "m02": result["m02"], "m04": result["m04"]}
    print(f"[cross_eval tag={args.tag}] val_loss={result['val_loss']} "
          f"m01_median={result['m01']['median']} m04={result['m04']['accuracy']} elapsed={elapsed:.1f}s")

    out_path = Path(args.out) if args.out else Path(f"cross_eval_{args.tag}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[cross_eval tag={args.tag}] wrote {out_path}")


if __name__ == "__main__":
    main()
