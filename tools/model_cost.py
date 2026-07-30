"""tools/model_cost.py -- parameters, MACs and measured latency for the deployed pair.

Counts come from torch.utils.flop_counter against the LIVE modules, so they cover the
attention matmuls and the PixelShuffle refiner without a hand-maintained table. The
5090 latency is measured; the Orin figure is an EXTRAPOLATION and is reported as a
range with both of its bounds named -- compute and memory bandwidth -- because a model
this small at 640x480 is not obviously compute-bound.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Jetson AGX Orin 64GB, from NVIDIA's module datasheet. DENSE figures: the 275 TOPS
# headline is INT8 with 2:4 structured sparsity, which we do not use, so it halves.
# FP16 tensor-core rate is half the dense INT8 rate (Ampere).
ORIN = {"fp16_tflops_peak": 68.75, "int8_tops_peak": 137.5, "bw_gbps": 204.8}
# Sustained fraction of datasheet peak reached by TRT on conv-heavy graphs. Wide,
# deliberately: this is the dominant uncertainty in the whole estimate, and no local
# measurement can pin it down -- it is a property of the target's kernels, not ours.
SUSTAINED = (0.15, 0.30)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/rev640.yaml")
    p.add_argument("--crops", type=int, default=16, help="refiner crops per frame")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--out", default="paper/results_rev6/15_cost/model_cost.json")
    return p


def main():
    a = build_parser().parse_args()
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    from dcc.dataset import load_config
    from dcc.model import DetectorNet, Refiner, detector_kwargs

    cfg = load_config(a.config)
    W, H = cfg["input_size"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    det = DetectorNet(H, W, **detector_kwargs(cfg)).to(dev).eval()
    ref = Refiner().to(dev).eval()

    p_det = sum(p.numel() for p in det.parameters())
    p_ref = sum(p.numel() for p in ref.parameters())
    # the transformer stack is named `blocks` + the trailing `norm` (P5)
    p_attn = sum(p.numel() for n, p in det.named_parameters()
                 if n.startswith("blocks") or n.startswith("norm"))
    p_e4 = sum(p.numel() for n, p in det.named_parameters() if n.startswith("e4"))
    p_gate = sum(p.numel() for n, p in det.named_parameters() if n.startswith("gate"))

    def macs(mod, x):
        # FlopCounterMode counts 2 flops per MAC; halve to report MACs.
        with FlopCounterMode(display=False) as f, torch.no_grad():
            mod(x)
        return f.get_total_flops() / 2

    x_det = torch.randn(1, 1, H, W, device=dev)
    x_ref = torch.randn(a.crops, 1, 24, 24, device=dev)
    m_det = macs(det, x_det)
    m_ref = macs(ref, x_ref)

    def time_ms(fn):
        for _ in range(30):
            fn()
        torch.cuda.synchronize()
        e0, e1 = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        e0.record()
        for _ in range(a.iters):
            fn()
        e1.record(); torch.cuda.synchronize()
        return e0.elapsed_time(e1) / a.iters

    def bench(mod, x):
        if dev.type != "cuda":
            return float("nan")
        mod = mod.to(memory_format=torch.channels_last)      # as trained/deployed
        x = x.to(memory_format=torch.channels_last)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            return time_ms(lambda: mod(x))

    t_det, t_ref = bench(det, x_det), bench(ref, x_ref)

    # This machine's OWN achievable ceiling, measured rather than quoted from a spec
    # sheet -- the anchor for "what fraction of peak does this graph actually reach".
    def peak_tflops():
        if dev.type != "cuda":
            return float("nan")
        n = 8192
        A = torch.randn(n, n, device=dev, dtype=torch.float16)
        B = torch.randn(n, n, device=dev, dtype=torch.float16)
        with torch.no_grad():
            ms = time_ms(lambda: A @ B)
        return 2 * n ** 3 / (ms * 1e-3) / 1e12

    gm = (m_det + m_ref) / 1e9
    gflops = 2 * gm
    host_peak = peak_tflops()
    achieved = gflops / ((t_det + t_ref) * 1e-3) / 1e3

    # Two independent ceilings. Whichever is SLOWER governs.
    est = {}
    for lo_hi, frac in zip(("optimistic", "conservative"), SUSTAINED[::-1]):
        compute_ms = 1e3 * gflops / (ORIN["fp16_tflops_peak"] * frac * 1e3)
        est[lo_hi] = {"sustained_frac": frac, "compute_bound_ms": compute_ms,
                      "fps": 1e3 / compute_ms}
    for lo_hi, frac in zip(("int8_optimistic", "int8_conservative"), SUSTAINED[::-1]):
        # INT8 counts the same MACs against the 2x-faster integer pipe.
        ms = 1e3 * gflops / (ORIN["int8_tops_peak"] * frac * 1e3)
        est[lo_hi] = {"sustained_frac": frac, "compute_bound_ms": ms, "fps": 1e3 / ms}
    eff = achieved / host_peak
    # Bandwidth floor: every weight read once per frame + activations. Weights alone
    # at fp16 is a hard lower bound on traffic, not an estimate of it.
    bytes_min = 2 * (p_det + p_ref)
    est["bandwidth_floor_ms"] = 1e3 * bytes_min / (ORIN["bw_gbps"] * 1e9)

    # Latency is worthless if something else is on the GPU. Record the contention
    # rather than emitting a clean-looking number someone quotes six months later.
    import subprocess
    smi = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout
    others = [l for l in smi.strip().splitlines() if l and str(__import__("os").getpid()) not in l]

    out = {"input_size": [W, H], "crops_per_frame": a.crops,
           "stages": [n for n, _ in det.named_children()],
           "latency_contended": bool(others), "other_gpu_procs": others,
           "params": {"detector": p_det, "refiner": p_ref, "total": p_det + p_ref,
                       "attention_blocks": p_attn, "e4": p_e4, "gates": p_gate},
           "macs": {"detector": m_det, "refiner_all_crops": m_ref,
                     "total_G": gm, "total_GFLOPs": gflops},
           "measured_5090_fp16_ms": {"detector": t_det, "refiner": t_ref,
                                      "total": t_det + t_ref,
                                      "fps": 1e3 / (t_det + t_ref)},
           "orin_estimate": est, "orin_spec": ORIN}
    o = Path(a.out); o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(out, indent=2))

    print(f"params   detector {p_det:,}  refiner {p_ref:,}  TOTAL {p_det + p_ref:,}")
    print(f"         attention {p_attn:,} ({100*p_attn/p_det:.1f}%)  e4 {p_e4:,} "
          f"({100*p_e4/p_det:.1f}%)  gate {p_gate:,} ({100*p_gate/p_det:.1f}%)")
    print(f"MACs     detector {m_det/1e9:.2f} G  refiner x{a.crops} {m_ref/1e9:.3f} G  "
          f"TOTAL {gm:.2f} G  ({gflops:.1f} GFLOPs)")
    print(f"5090 fp16  det {t_det:.2f} ms  ref {t_ref:.2f} ms  total {t_det+t_ref:.2f} ms "
          f"= {1e3/(t_det+t_ref):.0f} fps")
    print(f"           graph reaches {achieved:.1f} of this GPU's measured "
          f"{host_peak:.1f} TFLOPS ceiling = {100*eff:.1f}%"
          + ("   [CONTENDED -- both numbers depressed, not a clean benchmark]"
             if others else ""))
    print(f"\nOrin AGX (datasheet roofline, {gflops:.1f} GFLOPs/frame):")
    for k, peak, unit in (("conservative", ORIN["fp16_tflops_peak"], "TFLOPS fp16"),
                          ("optimistic", ORIN["fp16_tflops_peak"], "TFLOPS fp16"),
                          ("int8_conservative", ORIN["int8_tops_peak"], "TOPS int8"),
                          ("int8_optimistic", ORIN["int8_tops_peak"], "TOPS int8")):
        print(f"  {k:18s} {est[k]['compute_bound_ms']:6.1f} ms  {est[k]['fps']:6.1f} fps "
              f"(at {100*est[k]['sustained_frac']:.0f}% of {peak:.1f} {unit})")
    print(f"Orin weight-traffic floor {est['bandwidth_floor_ms']:.2f} ms")
    print(f"-> {o}")


if __name__ == "__main__":
    main()
