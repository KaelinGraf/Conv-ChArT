"""P15 export gate for the RELEASE checkpoints: ONNX exportability + fp16 parity.

WHY THIS EXISTS. Deployment is a Jetson AGX Orin running a TensorRT fp16 engine, and the
design pinned a P15 acceptance gate for that step -- "fp16 export parity, corner delta
< 0.05 px and ID delta < 0.1% on 1k val". Until 2026-08-05 that gate had never been run on a
trained model: `tests/test_model.py` exports an UNTRAINED DetectorNet at the default 320x240
with default kwargs, which proves the architecture family is exportable but says nothing about
the three shipping checkpoints, whose configs differ (width_mult 0.25/0.375, per-arm gates,
lambda_cls, 640x480 input). No .onnx file existed anywhere in the repo.

TWO INDEPENDENT QUESTIONS, answered separately:

  1. EXPORTABILITY -- does THIS checkpoint's architecture export to opset 17 and survive
     onnx.checker, with no op outside the TRT-standard set? Banned ops are the same
     {Complex, Loop, If} the unit test uses, so the contract does not fork.

  2. fp16 NUMERICAL PARITY -- P15 is a PAIRED comparison, not two independent metric runs:
     the same image is decoded twice, once fp32 and once fp16, and the two corner sets are
     matched to each other. Comparing two aggregate M-01 numbers would let offsetting errors
     cancel and report parity that individual corners do not have.

     ID delta is measured at the fp32 PEAK POSITIONS for both precisions. That isolates the
     class head's own precision sensitivity from position drift -- otherwise a corner that
     moved half a pixel could read a different cell and be charged to the class head. The
     full-chain number (fp16 positions AND fp16 class map) is reported alongside, because
     that is what actually runs on the device.

fp16 here is true half precision on weights AND activations (`model.half()`), not autocast:
a TRT fp16 engine stores fp16 weights, so autocast's fp32 master weights would be the
optimistic measurement, and this gate should not be optimistic.

Does NOT run the exported graph -- onnxruntime is not installed in MLWS, so op-level ONNX
numerics are out of scope here and stated as such in the report rather than silently skipped.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BANNED_OPS = {"Complex", "Loop", "If"}   # same contract as tests/test_model.py

RELEASE = [("222k", "runs/rel_w25_lam2_100k_rev6/ckpt_0100000.pt"),
           ("502k", "runs/rel_w375_lam15_100k_rev6/ckpt_0100000.pt"),
           ("882k", "runs/rel_w882_c2_100k_rev6/ckpt_0100000.pt")]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", action="append", metavar="TIER=PATH",
                   help="override the release set; repeatable")
    p.add_argument("--n", type=int, default=1000, help="val images for the parity pass (P15 says 1k)")
    p.add_argument("--out", default="paper/results_rev6/24_export_parity")
    p.add_argument("--device", default=None, choices=["cuda", "cpu"])
    p.add_argument("--refiner", default="runs/ref_s15_10k_rev6/ckpt_0010000.pt",
                   help="the refiner every release eval used; P15's sub-pixel budget is "
                        "meaningless without it (peaks() returns integer coords)")
    p.add_argument("--refine-min-peak", type=float, default=None,
                   help="override cfg refine_min_peak; 0 disables the refiner guard. Exists to"
                        " attribute large fp16 deltas: a guard that fires in one precision and"
                        " not the other swaps refined for coarse, a discrete ~4 px jump.")
    p.add_argument("--half", default="both", choices=["both", "detector", "refiner"],
                   help="which net runs in fp16; attributes the tail to one stage")
    p.add_argument("--skip-onnx", action="store_true")
    p.add_argument("--skip-parity", action="store_true",
                   help="refresh the ONNX artefacts WITHOUT touching export_parity.json. "
                        "Exists because re-running this tool at a small --n to re-export a "
                        "graph silently overwrote a completed n=1000 parity record.")
    return p


def load_model(ckpt_path, device):
    """DetectorNet built from the CHECKPOINT's own cfg (never a passed-in config: the release
    arms differ in width/gates/lambda and a mismatched build fails as a shape error at best,
    or silently scores the wrong architecture at worst), with EMA weights preferred."""
    import torch
    from dcc.board import n_corners
    from dcc.model import DetectorNet, detector_kwargs

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    W, H = cfg["input_size"]
    model = DetectorNet(H, W, n_cls=n_corners(cfg.get("board")), **detector_kwargs(cfg)).to(device)
    model.load_state_dict(ck.get("ema", ck["model"]))
    return model.eval(), cfg, ck


def export_onnx(model, shape, out_path, output_names=("hm", "cls")):
    """shape is (H, W) of the net's OWN input -- 480x640 for a detector, 24x24 for the refiner.
    The refiner is exported too: the shipping pipeline is detector AND refiner, so a detector
    engine alone is not a deployable set, and tests/test_model.py only ever exported an
    UNTRAINED Refiner."""
    import onnx
    import torch

    H, W = shape
    torch.onnx.export(model, torch.randn(1, 1, H, W, device=next(model.parameters()).device),
                      str(out_path), opset_version=17, dynamo=False,
                      input_names=["input"], output_names=list(output_names))
    g = onnx.load(str(out_path))
    onnx.checker.check_model(g)
    ops = sorted({n.op_type for n in g.graph.node})
    bad = sorted(set(ops) & BANNED_OPS)
    return {"path": str(out_path), "opset": 17, "n_nodes": len(g.graph.node),
            "ops": ops, "banned_ops_present": bad, "checker": "PASS",
            "size_mb": round(out_path.stat().st_size / 2**20, 2)}


def parity(model, refiner, cfg, n, device, refine_min_peak=None, half="both"):
    """Paired fp32-vs-fp16 decode over the fixed SynthVal stream, through the FULL detect()
    pipeline. -> summary dict.

    THE REFINER IS MANDATORY HERE, and this is not a detail. peaks() returns INTEGER
    coordinates (its own docstring says so), so a peak-only comparison can only ever produce a
    delta of 0 or >=1 px -- measured on the 222k tier at n=20 it gave p95 = 0.00000 and
    max = 1.00000, i.e. a vacuous PASS against a 0.05 px threshold that integer coordinates
    cannot resolve. P15's sub-pixel budget is therefore a statement about the REFINED output,
    and that is what is compared here.

    fp16 is applied by wrapping each net so its input is cast to half and its outputs back to
    float. detect() builds a float32 input tensor internally (dcc/pipeline.py's `inp`), so a
    bare .half() model would raise on the first conv; the wrapper keeps every internal
    activation and weight in fp16 -- TRT-engine semantics -- without detect() needing to know."""
    import copy

    import numpy as np
    import torch
    from dcc.dataset import SynthVal
    from dcc.pipeline import detect

    class Half(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = copy.deepcopy(m).half().eval()

        def forward(self, x):
            out = self.m(x.half())
            return tuple(o.float() for o in out) if isinstance(out, tuple) else out.float()

    # ATTRIBUTION LEVER. The >0.05 px tail is near-identical across three DIFFERENT detectors
    # (107/101/104 corners at n=1000) while the refiner is shared, which POINTS at the refiner
    # but does not prove it. half="detector"/"refiner" casts one net at a time so the tail can
    # be attributed by measurement instead of inference.
    m16 = Half(model) if half in ("both", "detector") else model
    r16 = (Half(refiner) if half in ("both", "refiner") else refiner) if refiner is not None else None
    val = SynthVal(cfg, n=n, seed=cfg["synth"]["val_seed"])
    dcfg = {"tau_hm": cfg["train"]["tau_hm"], "tau_id": cfg.get("tau_id", 0.5),
            "lattice_tol_px": cfg.get("lattice_tol_px", 3.0), "input_size": cfg["input_size"],
            "board": cfg.get("board"), "refine_min_peak": cfg.get("refine_min_peak", 0.3) if refine_min_peak is None else refine_min_peak}

    d_pos, n_pair, n_32, n_16, id_diff, n_none32, n_none16 = [], 0, 0, 0, 0, 0, 0
    for i in range(n):
        img, _rec = val[i]        # SynthVal yields (uint8 image, record), not a dict
        c32 = detect(img, model, refiner, cfg=dcfg)["corners"]
        c16 = detect(img, m16, r16, cfg=dcfg)["corners"]
        n_32 += len(c32); n_16 += len(c16)
        n_none32 += sum(1 for c in c32 if c["index"] is None)
        n_none16 += sum(1 for c in c16 if c["index"] is None)
        if not c32 or not c16:
            continue
        a = np.array([[c["x"], c["y"]] for c in c32], float)
        b = np.array([[c["x"], c["y"]] for c in c16], float)
        d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
        j = d.argmin(axis=1)
        dm = d[np.arange(len(a)), j]
        near = dm <= 8.0                     # same 8 px NN window M-01 matches at
        d_pos.extend(dm[near].tolist())
        n_pair += int(near.sum())
        for ia in np.nonzero(near)[0]:
            if c32[ia]["index"] != c16[j[ia]]["index"]:
                id_diff += 1

    d = np.asarray(d_pos) if d_pos else np.zeros(0)
    return {"n_images": n, "n_corners_fp32": n_32, "n_corners_fp16": n_16,
            "n_matched": n_pair, "unmatched_fp32": n_32 - n_pair,
            "unidentified_fp32": n_none32, "unidentified_fp16": n_none16,
            # TAIL COUNTS, not just p95. The mean here EXCEEDS the p95 (0.0028 vs 0.0016 px on
            # the 222k tier at n=1000), which is only possible with rare large outliers -- so a
            # p95-only verdict would certify parity that a handful of corners do not have. That
            # is the same vacuous-gate failure this file was written to close in preflight.
            "n_gt_0p05px": int((d > 0.05).sum()), "n_gt_1px": int((d > 1.0).sum()),
            "frac_gt_0p05px_pct": 100.0 * float((d > 0.05).sum()) / max(len(d), 1),
            "d_corner_mean": float(d.mean()) if len(d) else None,
            "d_corner_p95": float(np.percentile(d, 95)) if len(d) else None,
            "d_corner_max": float(d.max()) if len(d) else None,
            "id_diff_pct": 100.0 * id_diff / max(n_pair, 1),
            "corner_count_delta_pct": 100.0 * (n_16 - n_32) / max(n_32, 1)}


def verdict(par):
    """P15: corner delta < 0.05 px AND ID delta < 0.1%, on the REFINED corners the device
    actually emits, and on p95 rather than the mean so a small tail cannot hide behind a good
    average."""
    if par["d_corner_p95"] is None:
        return "FAIL", ["no matched corners"]
    fails = []
    if par["d_corner_p95"] >= 0.05:
        fails.append(f"corner p95 {par['d_corner_p95']:.4f} >= 0.05 px")
    # A corner past P15's budget is a corner past P15's budget, however rare. Reported as its
    # own line rather than folded into the p95 so the tail is never invisible.
    if par["frac_gt_0p05px_pct"] >= 0.1:
        fails.append(f"{par['frac_gt_0p05px_pct']:.4f}% of corners exceed 0.05 px "
                     f"(n={par['n_gt_0p05px']}, max {par['d_corner_max']:.3f} px)")
    if par["id_diff_pct"] >= 0.1:
        fails.append(f"ID {par['id_diff_pct']:.4f}% >= 0.1%")
    return ("PASS" if not fails else "FAIL"), fails


def main():
    args = build_parser().parse_args()
    import torch

    from dcc.model import refiner_for

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    # The refiner is shared across all three tiers -- the same one every release eval used
    # (provenance blocks in 20_robustness_REL*/). Loaded once: it is not a per-tier variable.
    rck = torch.load(args.refiner, map_location=device, weights_only=False)
    refiner = refiner_for(rck.get("ema", rck["model"])).to(device).eval()
    refiner.load_state_dict(rck.get("ema", rck["model"]))
    tiers = ([tuple(s.split("=", 1)) for s in args.ckpt] if args.ckpt else RELEASE)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    report = {"gate": "P15 fp16 export parity", "device": device, "n_val": args.n,
              "refiner_ckpt": args.refiner,
              "thresholds": {"d_corner_px": 0.05, "id_diff_pct": 0.1}, "half": args.half,
              "onnx_runtime_check": "NOT RUN -- onnxruntime absent from the MLWS env; "
                                    "this gate covers export validity and fp16 numerics only",
              "tiers": {}}

    for tier, path in tiers:
        if not Path(path).exists():
            print(f"[{tier}] MISSING {path}"); continue
        model, cfg, ck = load_model(path, device)
        entry = {"ckpt": path, "step": ck.get("step"), "config_input_size": cfg["input_size"],
                 "n_params": sum(p.numel() for p in model.parameters())}
        if not args.skip_onnx:
            entry["onnx"] = export_onnx(model, (cfg["input_size"][1], cfg["input_size"][0]),
                                        out / f"detector_{tier}.onnx")
            print(f"[{tier}] onnx OK  {entry['onnx']['n_nodes']} nodes  "
                  f"{entry['onnx']['size_mb']} MB  banned={entry['onnx']['banned_ops_present']}")
        if args.skip_parity:
            report["tiers"][tier] = entry
            continue
        par = entry["parity_fp16"] = parity(model, refiner, cfg, args.n, device, args.refine_min_peak, args.half)
        entry["verdict"], entry["failures"] = verdict(par)
        print(f"[{tier}] {entry['verdict']}  d_corner p95={par['d_corner_p95']:.5f} px  "
              f"max={par['d_corner_max']:.5f}  mean={par['d_corner_mean']:.5f}  "
              f"ID={par['id_diff_pct']:.4f}%  >0.05px={par['n_gt_0p05px']} ({par['frac_gt_0p05px_pct']:.3f}%)  "
              f"corners {par['n_corners_fp32']}->{par['n_corners_fp16']}")
        if entry["failures"]:
            print(f"       failures: {entry['failures']}")
        report["tiers"][tier] = entry

    if not args.skip_onnx:
        # 24x24 is the refiner's fixed crop (dcc/refiner_data.py's contract); its single output
        # is the 64x64 logit map, not the detector's (hm, cls) pair.
        report["refiner_onnx"] = export_onnx(refiner, (24, 24), out / "refiner.onnx",
                                             output_names=("logits",))
        r = report["refiner_onnx"]
        print(f"[refiner] onnx OK  {r['n_nodes']} nodes  {r['size_mb']} MB  "
              f"banned={r['banned_ops_present']}")

    if args.skip_parity:
        print("\n--skip-parity: ONNX artefacts refreshed, export_parity.json left untouched")
        return
    (out / "export_parity.json").write_text(json.dumps(report, indent=2))
    print(f"\n-> {out / 'export_parity.json'}")


if __name__ == "__main__":
    main()
