"""tools/find_attention_frames.py -- pick SynthVal frames where the bottleneck
attention is actually selective, instead of choosing a frame by eye and hoping.

The attention panel is only a useful figure if the attention concentrates
somewhere legible. Frame val4970 was picked for being a clean, easy scene and
turned out to be the opposite: block-0 entropy 8.34 nats against a uniform
ceiling of ln(4800) = 8.48, i.e. essentially no selectivity. Scanning for it is
cheap, so scan.

Two criteria, both reported, because they can disagree and the disagreement is
informative:
  - ENTROPY of the mean-over-heads attention row from the query token. Low =
    concentrated. Uniform is ln(T); the gap from it is what matters, not the
    absolute value.
  - BOARD MASS: the fraction of that row's mass landing inside the GT board
    quad. High = attending to the board specifically rather than merely
    concentrating somewhere.

A frame with low entropy but low board mass is concentrating on background
clutter -- exactly the figure NOT to publish. Rank on board mass, report both.

Computation mirrors introspect.panel_attention verbatim (same query-token
choice, same blk.qkv_heads path so RoPE is applied identically) -- this picks
frames for that panel, so any divergence would select on the wrong quantity.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/rev640.yaml")
    p.add_argument("--ckpt", default="runs/rev640_baseline/ckpt_0150000.pt")
    p.add_argument("--indices", type=int, nargs="+", default=None)
    p.add_argument("--scan", type=int, default=60, help="scan this many evenly-spaced val indices")
    p.add_argument("--out", default="paper/results_rev6/04_introspection/attention_frame_scan.json")
    return p


def main():
    args = build_parser().parse_args()
    import cv2
    import numpy as np
    import torch
    cv2.setNumThreads(1); torch.set_num_threads(2)

    from dcc.dataset import load_config, SynthVal
    from dcc.model import DetectorNet, detector_kwargs

    cfg = load_config(args.config)
    device = torch.device("cpu")            # CPU by design: runs beside GPU training
    W, H = cfg["input_size"]
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = DetectorNet(H, W, **detector_kwargs(cfg)).to(device).eval()
    model.load_state_dict(ck.get("ema", ck.get("model")))

    ds = SynthVal(cfg, n=cfg["synth"]["val_size"] if "val_size" in cfg["synth"] else 10000)
    idxs = args.indices if args.indices else list(np.linspace(0, len(ds) - 1, args.scan, dtype=int))

    blk_in = []
    hooks = [b.register_forward_pre_hook(lambda m, i: blk_in.append(i[0])) for b in model.blocks]

    rows = []
    for i in idxs:
        img, rec = ds[int(i)]        # SynthVal yields (image, record); record holds corners/s_px
        corners = [c for c in rec["corners"] if c.get("visible")]
        if len(corners) < 4:
            continue
        blk_in.clear()
        with torch.no_grad():
            hm_logits, _ = model(torch.from_numpy(img).float()[None, None].to(device) / 255.0)
        prob = torch.sigmoid(hm_logits)[0, 0].numpy()
        qy, qx = np.unravel_index(int(prob.argmax()), prob.shape)
        div = model.attend_div
        gh, gw = H // div, W // div
        tok = int(np.clip(qy // div, 0, gh - 1)) * gw + int(np.clip(qx // div, 0, gw - 1))

        # GT board quad -> token-grid mask, so "board mass" means the board, not a bbox of clutter
        pts = np.array([[c["x"], c["y"]] for c in corners], dtype=np.float32)
        hull = cv2.convexHull(pts)
        m = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(m, hull.astype(np.int32), 1)
        mask = cv2.resize(m, (gw, gh), interpolation=cv2.INTER_AREA).astype(bool)

        ents, masses = [], []
        for blk, x_in in zip(model.blocks, blk_in):
            with torch.no_grad():
                q, k, _ = blk.qkv_heads(blk.n1(x_in))
                A = torch.softmax((q.float() @ k.float().transpose(-2, -1)) / q.shape[-1] ** 0.5, dim=-1)
            a = A[0, :, tok, :].numpy().mean(axis=0).reshape(gh, gw)
            ents.append(float(-(a * np.log(a + 1e-12)).sum()))
            masses.append(float(a[mask].sum()))
        rows.append({"index": int(i), "s_px": float(rec.get("s_px", 0)),
                     "n_visible": len(corners), "entropy": ents, "board_mass": masses,
                     "board_frac_of_grid": float(mask.mean())})

    for h in hooks:
        h.remove()
    uniform = float(np.log((H // model.attend_div) * (W // model.attend_div)))
    rows.sort(key=lambda r: -np.mean(r["board_mass"]))
    print(f"uniform entropy ceiling = {uniform:.2f} nats;  n scanned = {len(rows)}\n")
    print(f"{'idx':>6}{'s_px':>7}{'H blk0':>8}{'H blk1':>8}{'mass0':>8}{'mass1':>8}{'board%grid':>12}{'lift':>7}")
    for r in rows[:12]:
        lift = np.mean(r["board_mass"]) / max(r["board_frac_of_grid"], 1e-6)
        print(f"{r['index']:>6}{r['s_px']:>7.1f}{r['entropy'][0]:>8.2f}{r['entropy'][1]:>8.2f}"
              f"{r['board_mass'][0]:>8.3f}{r['board_mass'][1]:>8.3f}{r['board_frac_of_grid']*100:>11.1f}%{lift:>7.1f}x")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"uniform_entropy": uniform, "ckpt": args.ckpt, "ckpt_step": ck.get("step"),
                   "config": args.config, "rows": rows}, f, indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
