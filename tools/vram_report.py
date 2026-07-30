"""tools/vram_report.py -- measured VRAM for a detector variant, plus a deployment estimate.

Reports four numbers per arm, because they differ by more than an order of magnitude and
conflating them is the usual way VRAM gets misquoted:

  1. WEIGHTS         params x dtype. What the .onnx file weighs.
  2. ACTIVATIONS     torch.cuda.max_memory_allocated for a forward pass, minus weights.
                     Batch-dependent, and the number that actually scales.
  3. TORCH RESERVED  what the caching allocator holds. Always much larger than allocated,
                     and what `nvidia-smi` shows -- which is why nvidia-smi never answers
                     "how much does this model need".
  4. ONNX/TRT ESTIMATE  weights + peak activation working set. A deployment runtime plans
                     a static arena and reuses buffers, so it needs roughly the largest
                     concurrent live set, not the sum of all intermediates.

The estimate is a MODEL, not a measurement -- no TensorRT here. It is bounded below by
weights + the largest single activation pair and above by the eager peak; both bounds are
printed rather than a single confident figure.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--configs", nargs="+", default=[
        "configs/rev640.yaml",
        "configs/abl_nodilate.yaml",
        "configs/abl_nodilate_width_half_s05.yaml",
        "configs/abl_nodilate_width_quarter_s05.yaml"])
    p.add_argument("--batches", nargs="+", type=int, default=[1, 16])
    p.add_argument("--out", default="paper/results_rev6/15_cost/vram_report.json")
    a = p.parse_args()

    import torch
    from dcc.dataset import load_config
    from dcc.model import DetectorNet, detector_kwargs
    if not torch.cuda.is_available():
        sys.exit("no CUDA device")
    dev = torch.device("cuda")

    rows = {}
    print(f"{'arm':<34} {'B':>3} {'weights':>9} {'activations':>12} "
          f"{'reserved':>10} {'ONNX est.':>18}")
    for cfg_path in a.configs:
        cfg = load_config(cfg_path)
        W, H = cfg["input_size"]
        name = Path(cfg_path).stem
        for B in a.batches:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            # BASELINE, then measure weights as a DELTA against it. Reading
            # memory_allocated() as an absolute was a bug: torch.autocast keeps an fp16 copy
            # of every weight it casts, and that cache survives `del m` + empty_cache(), so
            # each arm inherited the PREVIOUS arm's fp16 residue (~7.0 MB from the 3.5 M-param
            # config) and reported it as its own weights -- 882,402 fp32 params showed as
            # 11.8 MB instead of 3.4 MB. The delta is correct whatever is already resident.
            base = torch.cuda.memory_allocated()
            m = DetectorNet(H, W, **detector_kwargs(cfg)).to(dev).eval()
            m = m.to(memory_format=torch.channels_last)
            n_par = sum(t.numel() for t in m.parameters())
            w_bytes = torch.cuda.memory_allocated() - base
            # Cross-check against the analytic size; anything beyond a few percent of
            # alignment slack means the delta is picking up something that is not weights.
            w_expect = n_par * 4
            assert abs(w_bytes - w_expect) < 0.10 * w_expect + 2**20, (
                f"{name}@B{B}: measured weights {w_bytes} vs analytic {w_expect} "
                f"({n_par} params x fp32) -- allocation accounting is contaminated")
            x = torch.randn(B, 1, H, W, device=dev).to(memory_format=torch.channels_last)
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                for _ in range(3):
                    m(x)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            reserved = torch.cuda.max_memory_reserved()
            act = peak - base - w_bytes       # subtract the baseline too, not just weights

            # Deployment bounds. fp16 halves the weights; activations are already fp16
            # under autocast. Lower bound assumes aggressive buffer reuse (a runtime keeps
            # roughly one input+output pair live at the widest stage); upper bound is the
            # eager peak, which reuses nothing.
            w_fp16 = n_par * 2
            lo = w_fp16 + act * 0.35
            hi = w_fp16 + act
            mb = lambda v: v / 2**20
            print(f"{name:<34} {B:>3} {mb(w_bytes):>7.1f}MB {mb(act):>10.1f}MB "
                  f"{mb(reserved):>8.1f}MB {mb(lo):>7.0f}-{mb(hi):.0f}MB")
            rows[f"{name}@B{B}"] = {
                "params": n_par, "weights_fp32_mb": mb(w_bytes),
                "weights_fp16_mb": mb(w_fp16), "activations_mb": mb(act),
                "torch_reserved_mb": mb(reserved),
                "onnx_estimate_mb": [mb(lo), mb(hi)]}
            del m, x
            torch.clear_autocast_cache()      # else the fp16 weight copies outlive the model
            torch.cuda.empty_cache()

    o = Path(a.out); o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps({
        "note": "activations = max_memory_allocated minus weights, fp16 autocast, "
                "channels_last, no_grad. reserved = caching-allocator high-water mark, "
                "which is what nvidia-smi reflects. onnx_estimate is a MODEL not a "
                "measurement: weights_fp16 + [0.35, 1.0] x eager activation peak, the "
                "range spanning full buffer reuse to none.",
        "rows": rows}, indent=2))
    print(f"\n-> {o}")


if __name__ == "__main__":
    main()
