"""tools/attention_head_roles.py -- do the attention heads SPECIALISE, and how many
of them actually read the markers?

Kaelin wants to say on stage that the model "has not only learned to attend to
important parts of the board, but has done so in an extremely efficient manner, as
only 1-2 heads look at the markers themselves, which shows the model has converged
onto the optimal representation of determining ids by process of elimination."

That is a strong claim and it came from an impression on an older checkpoint. This
measures it, per head, on whatever checkpoint you point it at.

For each head, the query token is the board-centre token and the attention row is
split three ways by WHERE its mass lands:

  CORNER  -- within `corner_r` cells of one of the 16 inner corners. Junction
             geometry: where a corner-to-corner head would look.
  MARKER  -- on the board but NOT near a corner. The ArUco code cells live here,
             so a head reading identity directly must put mass here.
  OFF     -- outside the board quad entirely.

A head is called a MARKER READER when its marker mass exceeds `marker_thr` AND
exceeds its own corner mass -- i.e. it is preferentially looking at code cells
rather than junctions. Reported per frame and aggregated, with the count of marker
readers per block, which is the number Kaelin wants to quote.

Mirrors introspect.panel_attention's query-token choice and blk.qkv_heads path, so
RoPE is applied identically and the numbers correspond to that figure.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/rev640.yaml")
    p.add_argument("--ckpt", default="runs/rev640_160k_rev6/ckpt_0160000.pt")
    p.add_argument("--n", type=int, default=40, help="frames")
    p.add_argument("--seed", type=int, default=20260729)
    p.add_argument("--corner-frac", type=float, default=0.35,
                   help="corner neighbourhood radius as a FRACTION OF CORNER SPACING, not a fixed\n"
                        "cell count: spacing in cells scales with board size, so a fixed radius\n"
                        "swallows the whole board at small s and leaves no marker region to measure")
    p.add_argument("--marker-thr", type=float, default=0.25, help="marker-mass fraction to count as a reader")
    p.add_argument("--out", default="paper/results_rev6/07_introspection/attention_head_roles.json")
    return p


def main():
    args = build_parser().parse_args()
    import cv2, numpy as np, torch
    cv2.setNumThreads(1); torch.set_num_threads(2)
    from dcc.dataset import load_config
    from dcc.model import DetectorNet, detector_kwargs
    from dcc.synth import generate_sample, list_backgrounds

    cfg = load_config(args.config)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W, H = cfg["input_size"]
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = DetectorNet(H, W, **detector_kwargs(cfg)).to(dev).eval()
    model.load_state_dict(ck.get("ema", ck.get("model")))
    bg = list_backgrounds(cfg["synth"]["backgrounds"])
    div = cfg.get("attend_div", 16)
    gh, gw = H // div, W // div

    feats, blk_in = {}, []
    model.e4.register_forward_hook(lambda m, i, o: feats.__setitem__("z", o.detach()))
    for blk in model.blocks:
        blk.register_forward_pre_hook(
            lambda m, a, kw: blk_in.append((a[0] if a else next(iter(kw.values()))).detach()),
            with_kwargs=True)

    n_blocks = len(model.blocks)
    per_block = [[] for _ in range(n_blocks)]     # per frame: list of per-head (corner, marker, off)
    for i in range(args.n):
        rng = np.random.default_rng([args.seed, i])
        rec, _ = generate_sample(cfg, rng, bg, force_negative=False)
        vis = [c for c in rec["corners"] if c["visible"]]
        if len(vis) < 8:
            continue
        pts = np.array([[c["x"], c["y"]] for c in vis])
        blk_in.clear()
        with torch.no_grad():
            model(torch.from_numpy(rec["image"]).float()[None, None].to(dev) / 255.0)

        # cell-space corner positions and the board quad
        cell = pts / div
        ys, xs = np.mgrid[0:gh, 0:gw]
        d2 = np.min(((xs[..., None] - cell[:, 0]) ** 2 + (ys[..., None] - cell[:, 1]) ** 2), axis=-1)
        # radius SCALED BY THE BOARD'S OWN CORNER SPACING in cells. A fixed cell radius is
        # scale-confounded: at s=22 px the corners are ~2.8 cells apart, so a 1.5-cell
        # neighbourhood covers essentially the entire board and no "marker" region survives.
        dd = np.linalg.norm(cell[:, None] - cell[None], axis=2)
        np.fill_diagonal(dd, np.inf)
        spacing = float(np.median(dd.min(axis=1)))
        corner_r = max(args.corner_frac * spacing, 0.75)
        corner_m = d2 <= corner_r ** 2
        hull = cv2.convexHull(cell.astype(np.float32)).reshape(-1, 2)
        board_m = np.zeros((gh, gw), np.uint8)
        cv2.fillConvexPoly(board_m, np.round(hull).astype(np.int32), 1)
        board_m = board_m.astype(bool)
        marker_m = board_m & ~corner_m
        # query = the token nearest the board centre, matching panel_attention's intent
        qc = np.clip(np.round(cell.mean(0)).astype(int), [0, 0], [gw - 1, gh - 1])
        qi = int(qc[1] * gw + qc[0])

        for b, blk in enumerate(model.blocks):
            with torch.no_grad():
                q, k, _v = blk.qkv_heads(blk.n1(blk_in[b]))
                # einsum, NOT `q[0,:,qi] @ k[0].transpose(-2,-1)`: that broadcasts
                # (heads,dim) against (heads,dim,T) into a heads x heads CROSS-PRODUCT,
                # silently yielding 64 rows for an 8-head model. Match each head to its own K.
                att = torch.softmax(torch.einsum("hd,htd->ht", q[0, :, qi], k[0]) /
                                     (q.shape[-1] ** 0.5), dim=-1)      # (heads, T)
            a = att.cpu().numpy().reshape(-1, gh, gw)
            tot = a.reshape(len(a), -1).sum(1)
            # BOTH mass fraction AND lift. Mass alone misreads the figure: the panels are
            # per-head normalised, so the eye reads DENSITY. Off-board is ~90% of tokens, so a
            # head with 72% of its mass smeared over background looks dark there while 12%
            # concentrated on a few marker cells looks bright. Lift = mass / (region's share of
            # tokens), i.e. enrichment over uniform -- scale-free, and it is what the figure shows.
            n_tok = corner_m.size
            fr = np.array([corner_m.sum(), marker_m.sum(), (~board_m).sum()]) / n_tok
            mass = np.stack([a[:, corner_m].sum(1) / tot, a[:, marker_m].sum(1) / tot,
                             a[:, ~board_m].sum(1) / tot], axis=1)      # (heads, 3)
            per_block[b].append(np.concatenate([mass, mass / np.maximum(fr, 1e-9)], axis=1))

    out = {"ckpt": args.ckpt, "step": ck.get("step"), "n_frames": len(per_block[0]),
           "corner_frac_of_spacing": args.corner_frac, "marker_thr": args.marker_thr, "blocks": []}
    for b, frames in enumerate(per_block):
        A = np.stack(frames)                                  # (F, heads, 6)
        med = np.median(A, axis=0)                            # (heads, 6) mass x3, lift x3
        # classify on LIFT, not mass: marker-enriched AND more marker- than corner-enriched
        readers = [h for h in range(len(med)) if med[h, 4] > med[h, 3] and med[h, 4] > 1.5]
        print(f"\nBLOCK {b}   ({len(med)} heads, median over {len(A)} frames)")
        print(f"  {'head':>5} | {'mass: CORNER':>12} {'MARKER':>7} {'OFF':>6} "
              f"| {'lift: CORNER':>12} {'MARKER':>7} {'OFF':>6} | role")
        for h in range(len(med)):
            role = "MARKER READER" if h in readers else ("corner" if med[h, 3] >= med[h, 4] else "mixed")
            print(f"  {h:>5} | {med[h,0]:>12.3f} {med[h,1]:>7.3f} {med[h,2]:>6.3f} "
                  f"| {med[h,3]:>12.1f} {med[h,4]:>7.1f} {med[h,5]:>6.2f} | {role}")
        print(f"  -> {len(readers)} of {len(med)} heads are MARKER READERS (by lift): {readers}")
        out["blocks"].append({"block": b, "median_mass_corner": med[:, 0].tolist(),
                              "median_mass_marker": med[:, 1].tolist(), "median_mass_off": med[:, 2].tolist(),
                              "median_lift_corner": med[:, 3].tolist(), "median_lift_marker": med[:, 4].tolist(),
                              "median_lift_off": med[:, 5].tolist(),
                              "marker_readers": readers, "n_heads": int(len(med))})
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
