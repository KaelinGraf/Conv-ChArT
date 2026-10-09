"""XSA premise diagnostic (Zhai 2026, arXiv:2603.09078) on Conv-ChArT's bottleneck.

The paper's Figure 1 measures three quantities per layer on a trained LM and
argues they justify exclusive self-attention:
  (a) mean cosine sim between value vectors v_i, v_j  (i != j)
  (b) mean DIAGONAL attention weight a_ii
  (c) mean cosine sim between the attention output y_i and the SELF value v_i
      -- "attention similarity bias", their Fig 1 right panel: ~0.2 at the
      first layers rising to ~0.6 by layer 23 of a 24-layer 1.3B model.

XSA removes exactly (c): z_i = y_i - (y_i . v_i) v_i / ||v_i||^2. So (c) IS the
size of what XSA would delete. If our blocks sit at or below their layer-0
value, the premise does not transfer and there is nothing to remove.

Two regime differences to keep in view when reading the output: our attention
is BIDIRECTIONAL (theirs causal, which structurally forces self-weight at early
positions), and we have 2 blocks against their 24 -- their effect grows with
depth, so we sample the weakest end of their own curve.

CPU-only by construction; never touches the GPU or the live trainer.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
ROOT = Path("/home/kaelin/p4p/dense deep charuco")
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(ROOT / "configs/rev640.yaml"))
    p.add_argument("--ckpt", required=True)
    p.add_argument("--indices", default="500,3600,7000,9600")
    args = p.parse_args()

    import numpy as np
    import torch
    import torch.nn.functional as F
    import yaml
    from dcc.model import DetectorNet, detector_kwargs
    from dcc.board import n_corners
    from dcc.dataset import SynthVal

    torch.set_grad_enabled(False)
    cfg = yaml.safe_load(open(args.config))
    w, h = cfg["input_size"]
    model = DetectorNet(h, w, n_cls=n_corners(cfg["board"]), **detector_kwargs(cfg)).eval()
    obj = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = obj.get("ema", obj.get("model")) if isinstance(obj, dict) else obj
    model.load_state_dict(sd)
    print(f"loaded {args.ckpt} (step {obj.get('step')}), {len(model.blocks)} attention blocks")

    ds = SynthVal(cfg, cfg["synth"]["val_size"], cfg["synth"]["val_seed"])
    idxs = [int(i) for i in args.indices.split(",")]

    per_block = {i: {"cos_yv": [], "a_ii": [], "cos_vv": []} for i in range(len(model.blocks))}

    for idx in idxs:
        img, rec = ds[idx]
        x = torch.from_numpy(img).float()[None, None] / 255.0
        blk_in = []
        handles = [b.register_forward_pre_hook(
            lambda m, a, kw: blk_in.append((a[0] if a else next(iter(kw.values()))).detach()),
            with_kwargs=True) for b in model.blocks]
        model(x)
        for hd in handles:
            hd.remove()

        for bi, (blk, xin) in enumerate(zip(model.blocks, blk_in)):
            q, k, v = blk.qkv_heads(blk.n1(xin))          # (1, H, T, dh); RoPE on q,k only
            y = F.scaled_dot_product_attention(q, k, v)   # bidirectional, matches forward()
            # (c) attention-similarity bias: cos(y_i, v_i), per head per token
            cos_yv = F.cosine_similarity(y, v, dim=-1)    # (1, H, T)
            per_block[bi]["cos_yv"].append(float(cos_yv.mean()))
            # (b) diagonal attention weight, computed head by head to bound memory
            dh = q.shape[-1]
            diag = []
            for hh in range(q.shape[1]):
                A = torch.softmax(q[0, hh] @ k[0, hh].transpose(0, 1) / dh ** 0.5, dim=-1)
                diag.append(float(A.diagonal().mean()))
                del A
            per_block[bi]["a_ii"].append(float(np.mean(diag)))
            # (a) value-value similarity over random distinct pairs
            T = v.shape[2]
            g = torch.Generator().manual_seed(0)
            i1 = torch.randint(0, T, (2000,), generator=g)
            i2 = torch.randint(0, T, (2000,), generator=g)
            m = i1 != i2
            per_block[bi]["cos_vv"].append(
                float(F.cosine_similarity(v[0, :, i1[m]], v[0, :, i2[m]], dim=-1).mean()))
        print(f"  frame val{idx} done")

    T = model.blocks[0].qkv_heads(model.blocks[0].n1(blk_in[0]))[2].shape[2]
    print(f"\ntokens per frame T = {T}; uniform attention would give a_ii = 1/T = {1/T:.2e}")
    print(f"{'block':<7}{'cos(y_i,v_i)':>15}{'a_ii':>12}{'a_ii / (1/T)':>15}{'cos(v_i,v_j)':>15}")
    for bi, d in per_block.items():
        print(f"{bi:<7}{np.mean(d['cos_yv']):>15.4f}{np.mean(d['a_ii']):>12.2e}"
              f"{np.mean(d['a_ii'])*T:>15.2f}{np.mean(d['cos_vv']):>15.4f}")
    print("\nPaper's reference (Fig 1, 24-layer 1.3B causal LM): cos(y_i,v_i) ~0.2 at the")
    print("first layers rising to ~0.6 by layer 23. That value IS the magnitude XSA deletes.")


if __name__ == "__main__":
    main()
