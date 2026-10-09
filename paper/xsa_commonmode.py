"""How much of the attention output is a CONSTANT vector? (XSA follow-up.)

xsa_diagnostic.py established that cos(y_i, v_i) is 0.275 / 0.531 in our two
blocks -- the magnitude XSA deletes -- and that it is NOT driven by diagonal
attention (~0.1% of the mass) but by value-vector correlation (0.106 -> 0.543).

The decomposition that implies: v_j = mu + r_j, so y_i = mu + sum_j a_ij r_j.
The mu term is a constant vector added to every token, carrying no positional
information, and LayerNorm does NOT remove it (LN normalises per token across
channels, not across tokens). This script measures how big that constant is:

  common_mode_frac = || mean_i(y_i) || / mean_i(|| y_i ||)

and the decisive control -- cos(y_i, v_i) recomputed after subtracting the
token-mean from BOTH. If the bias is purely common-mode, that residual cosine
collapses toward 0 and XSA is mostly deleting a constant. If it stays high, the
bias is per-token structure and XSA is deleting something content-dependent,
which is a materially different (and riskier) proposition.

CPU-only. Does not touch the GPU or the live trainer.
"""
import argparse
import sys
from pathlib import Path

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
    model.load_state_dict(obj.get("ema", obj.get("model")) if isinstance(obj, dict) else obj)
    print(f"loaded step {obj.get('step')}")

    ds = SynthVal(cfg, cfg["synth"]["val_size"], cfg["synth"]["val_seed"])
    acc = {i: {"cm_y": [], "cm_v": [], "cos_raw": [], "cos_centred": []}
           for i in range(len(model.blocks))}

    for idx in [int(i) for i in args.indices.split(",")]:
        img, _ = ds[idx]
        x = torch.from_numpy(img).float()[None, None] / 255.0
        blk_in = []
        hs = [b.register_forward_pre_hook(
            lambda m, a, kw: blk_in.append((a[0] if a else next(iter(kw.values()))).detach()),
            with_kwargs=True) for b in model.blocks]
        model(x)
        for hd in hs:
            hd.remove()

        for bi, (blk, xin) in enumerate(zip(model.blocks, blk_in)):
            q, k, v = blk.qkv_heads(blk.n1(xin))
            y = F.scaled_dot_product_attention(q, k, v)        # (1, H, T, dh)
            # fraction of the output that is a constant vector across tokens
            acc[bi]["cm_y"].append(float(
                (y.mean(dim=2).norm(dim=-1) / y.norm(dim=-1).mean(dim=-1)).mean()))
            acc[bi]["cm_v"].append(float(
                (v.mean(dim=2).norm(dim=-1) / v.norm(dim=-1).mean(dim=-1)).mean()))
            acc[bi]["cos_raw"].append(float(F.cosine_similarity(y, v, dim=-1).mean()))
            yc, vc = y - y.mean(dim=2, keepdim=True), v - v.mean(dim=2, keepdim=True)
            acc[bi]["cos_centred"].append(float(F.cosine_similarity(yc, vc, dim=-1).mean()))
        print(f"  val{idx} done")

    print(f"\n{'block':<7}{'||mean y||/mean||y||':>22}{'same for v':>13}"
          f"{'cos(y,v)':>11}{'cos after centring':>21}")
    for bi, d in acc.items():
        print(f"{bi:<7}{np.mean(d['cm_y']):>22.4f}{np.mean(d['cm_v']):>13.4f}"
              f"{np.mean(d['cos_raw']):>11.4f}{np.mean(d['cos_centred']):>21.4f}")
    print("\nRead: a high common-mode fraction with a LOW centred cosine means the")
    print("attention-similarity bias is essentially a constant offset -- XSA would be")
    print("deleting dead weight. A centred cosine that stays high means the bias is")
    print("per-token and content-dependent, so XSA removes real signal too.")


if __name__ == "__main__":
    main()
