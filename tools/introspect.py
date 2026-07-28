"""Introspection & visualisation CLI for the Conv-ChArT detector (conference
demo). Seven presentation-grade panels rendered from a single forward pass
over one SynthVal sample or a raw image: pipeline end-to-end, 3D heatmap
landscape, bottleneck-attention maps, decoder skip gates, a gate
selectivity probe (skip/conditioning/gated/suppressed against a GT
board-region mask), an effective-receptive-field probe, and encoder feature
maps. No --ckpt -> an UNTRAINED DetectorNet, flagged on every figure title.
Argparse runs before any heavy import so --help never needs torch/dcc/matplotlib.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PANELS_ALL = ["pipeline", "heatmap3d", "attention", "gates", "gateprobe", "gateflow", "gateablation", "erf", "features"]


def _parse_xy(s):
    x, y = s.split(",")
    return float(x), float(y)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--ckpt", default=None, help="detector checkpoint; absent -> UNTRAINED DetectorNet")
    p.add_argument("--refiner-ckpt", default=None, help="refiner checkpoint; absent -> UNTRAINED Refiner")
    p.add_argument("--ckpt-b", default=None, help="2nd detector ckpt: erf side-by-side ablation compare")
    p.add_argument("--index", type=int, default=None, help="SynthVal[index]; default 0 if --image absent")
    p.add_argument("--image", default=None, help="raw grayscale/colour file instead of a SynthVal sample")
    p.add_argument("--out", default="introspect_out/")
    p.add_argument("--panels", default="all", help="comma list from " + ",".join(PANELS_ALL) + ", or 'all'")
    p.add_argument("--query-xy", type=_parse_xy, default=None, metavar="X,Y",
                   help="attention/erf/heatmap3d query point; default = strongest heatmap (or class) peak")
    p.add_argument("--gif", action="store_true", help="heatmap3d: also write a rotating-azimuth GIF")
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument("--show", action="store_true", help="interactive windows after saving")
    return p


# --------------------------------------------------------------------------- shared helpers

def _load_ckpt(model, path, device):
    """torch.load; accepts a raw state_dict OR a trainer ckpt dict with
    'model'/'ema' keys (ema preferred). weights_only=False: these are the
    user's own local checkpoints, not untrusted downloads."""
    import torch
    if not path:
        return False
    obj = torch.load(path, map_location=device, weights_only=False)
    sd = obj.get("ema", obj.get("model")) if isinstance(obj, dict) and ("model" in obj or "ema" in obj) else obj
    model.load_state_dict(sd)
    return True


def _build_module(cls, ckpt, device, *ctor_args, **ctor_kwargs):
    m = cls(*ctor_args, **ctor_kwargs).to(device).eval()
    return m, _load_ckpt(m, ckpt, device)


def _suptitle(ckpt_path, tag, trained):
    status = Path(ckpt_path).name if trained else "UNTRAINED — pipeline demo"
    return f"{tag} | {status}"


def _sample(cfg, args):
    """(image uint8 (H,W), tag, record|None) -- tag is filename-safe, used in
    every out path; record is generate_sample's own dict (GT "corners" etc),
    None for a raw --image file. Every panel but gateprobe ignores the 3rd
    value; gateprobe's board-region mask needs it."""
    import cv2
    if args.image:
        img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"could not read image: {args.image}")
        w, h = cfg["input_size"]
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA), Path(args.image).stem, None
    from dcc.dataset import SynthVal
    idx = args.index if args.index is not None else 0
    ds = SynthVal(cfg, cfg["synth"]["val_size"], cfg["synth"]["val_seed"])
    img, record = ds[idx]
    return img, f"val{idx}", record


def _peak_or_query(args, prob_hm_2d):
    """(x, y) float: --query-xy if given, else the strongest heatmap peak."""
    import numpy as np
    if args.query_xy is not None:
        return args.query_xy
    y, x = np.unravel_index(int(np.argmax(prob_hm_2d)), prob_hm_2d.shape)
    return float(x), float(y)


def _peaks_only(prob_hm_2d, tau=0.3):
    """Peak decode only (the first stage of the inference pipeline),
    inlined here: 3x3 maxpool-equality peaks on sigmoid(hm). Used only when
    dcc.pipeline hasn't landed (no refine/ID/pose in that case)."""
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(prob_hm_2d)[None, None]
    pooled = F.max_pool2d(t, 3, stride=1, padding=1)
    ys, xs = torch.nonzero((t == pooled) & (t > tau), as_tuple=True)[2:]
    return list(zip(xs.tolist(), ys.tolist()))


def _forward_hooked(model, x):
    """Single no_grad forward pass, capturing e1..e5 outputs, each attention
    block's input (pre-n1), and gate3/gate4's exact call args -- so
    gate.alpha(*args, **kwargs) afterward replays the same computation
    forward used internally (the mechanism the attention/gates panel briefs
    name explicitly). Returns (hm_logits, cls_logits, feats, block_inputs,
    gate_args)."""
    import torch
    feats, blk_in, gate_args, handles = {}, [], {}, []
    for n in ("e1", "e2", "e3", "e4", "e5"):
        mod = getattr(model, n, None)
        if mod is not None:
            handles.append(mod.register_forward_hook(lambda m, i, o, n=n: feats.__setitem__(n, o.detach())))
    for blk in getattr(model, "blocks", []):
        handles.append(blk.register_forward_pre_hook(
            lambda m, a, kw: blk_in.append((a[0] if a else next(iter(kw.values()))).detach()),
            with_kwargs=True))
    for n in ("gate3", "gate4"):
        mod = getattr(model, n, None)
        if mod is not None:
            handles.append(mod.register_forward_pre_hook(
                lambda m, a, kw, n=n: gate_args.__setitem__(
                    n, (tuple(t.detach() for t in a), {k: v.detach() for k, v in kw.items()})),
                with_kwargs=True))
    with torch.no_grad():
        hm_logits, cls_logits = model(x)
    for h in handles:
        h.remove()
    return hm_logits, cls_logits, feats, blk_in, gate_args


def _no_ticks(ax, title):
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _show_plain(ax, image, title):
    ax.imshow(image, cmap="gray")
    _no_ticks(ax, title)


def _show_overlay(ax, image, heat, title, cmap="magma", alpha=0.55, vmin=None, vmax=None, mark=None):
    from dcc import viz
    ax.imshow(viz.overlay_alpha(image, heat, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax))
    if mark is not None:
        ax.plot(*mark, "+", color="cyan", markersize=14, markeredgewidth=2)
    _no_ticks(ax, title)


def _finish(fig, path, dpi, show, rect=None):
    fig.tight_layout(rect=rect) if rect else fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    print(f"wrote {path} ({path.stat().st_size} bytes)")
    if not show:
        import matplotlib.pyplot as plt
        plt.close(fig)


# --------------------------------------------------------------------------- panels

def panel_pipeline(model, x, image, cfg, args, out, tag, suptitle, dpi, show, device):
    import numpy as np
    import torch
    import cv2
    import matplotlib.pyplot as plt
    from dcc.model import Refiner
    from dcc import viz

    H, W = image.shape
    with torch.no_grad():
        hm_logits, cls_logits = model(x)
    prob_hm = torch.sigmoid(hm_logits)[0, 0].cpu().numpy()
    prob_cls = torch.sigmoid(cls_logits)[0].cpu().numpy()
    cls_up = np.repeat(np.repeat(prob_cls.max(axis=0), 4, axis=0), 4, axis=1)   # cell j -> px 4j..4j+3

    refiner, _ = _build_module(Refiner, args.refiner_ckpt, device)
    K = np.array([[1.05 * W, 0, W / 2], [0, 1.05 * W, H / 2], [0, 0, 1]])
    note = None
    try:
        from dcc.pipeline import detect
        result = detect(image, model, refiner, K=K, dist=None, cfg=cfg)
        corners, rvec, tvec = result.get("corners", []), result.get("rvec"), result.get("tvec")
        if rvec is None:
            note = result.get("reason", "pose not computed")
    except ImportError:
        corners = [{"x": px, "y": py, "source": "coarse"} for px, py in _peaks_only(prob_hm)]
        rvec = tvec = None
        note = "dcc.pipeline absent -- peaks-only fallback (no refine/ID/pose)"

    colors = {"head": (0, 200, 0), "recovered": (0, 140, 255), "coarse": (160, 160, 160)}
    disp = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for c in corners:
        cv2.circle(disp, (int(round(c["x"])), int(round(c["y"]))), 4,
                    colors.get(c.get("source", "coarse"), (160, 160, 160)), -1, cv2.LINE_AA)
    if rvec is not None and tvec is not None:
        cv2.drawFrameAxes(disp, K, np.zeros(5), np.asarray(rvec, dtype=np.float64).reshape(3, 1),
                           np.asarray(tvec, dtype=np.float64).reshape(3, 1), 0.5)

    fig = plt.figure(figsize=(24, 5))
    axes = [fig.add_subplot(1, 5, i + 1, projection=("3d" if i == 2 else None)) for i in range(5)]
    _show_plain(axes[0], image, "input")
    _show_overlay(axes[1], image, prob_hm, "predicted heatmap")
    viz.surface3d(axes[2], prob_hm, stride=2)
    axes[2].set_title("heatmap 3D", fontsize=9)
    _show_overlay(axes[3], image, cls_up, "class-map max")
    axes[4].imshow(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
    _no_ticks(axes[4], "corners" + (f"\n{note}" if note else ""))

    fig.suptitle(f"pipeline | {suptitle}")
    _finish(fig, out / f"pipeline_{tag}.png", dpi, show)
    if note:
        print(f"  note: {note}")


def panel_heatmap3d(model, x, image, args, out, tag, suptitle, dpi, show, make_gif):
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from dcc import viz

    H, W = image.shape
    with torch.no_grad():
        hm_logits, _ = model(x)
    prob = torch.sigmoid(hm_logits)[0, 0].cpu().numpy()
    qx, qy = _peak_or_query(args, prob)

    fig = plt.figure(figsize=(14, 6))
    ax_full = fig.add_subplot(1, 2, 1, projection="3d")
    viz.surface3d(ax_full, prob, stride=2)
    ax_full.set_title("full frame", fontsize=10)

    win = 96
    x0 = int(np.clip(qx - win / 2, 0, W - win))
    y0 = int(np.clip(qy - win / 2, 0, H - win))
    ax_zoom = fig.add_subplot(1, 2, 2, projection="3d")
    viz.surface3d(ax_zoom, prob[y0:y0 + win, x0:x0 + win], stride=1)
    ax_zoom.set_title(f"{win}px window @ ({int(qx)},{int(qy)})", fontsize=10)

    fig.suptitle(f"heatmap3d | {suptitle}")
    png = out / f"heatmap3d_{tag}.png"
    fig.tight_layout()
    fig.savefig(png, dpi=dpi)
    print(f"wrote {png} ({png.stat().st_size} bytes)")

    if make_gif:
        from matplotlib.animation import FuncAnimation, PillowWriter
        # A 72-frame rotation re-rasterises the whole 3D surface every frame:
        # at the static plot's stride=2 (~480k polygons at this config's
        # 1600x1200) that hangs/segfaults in mplot3d's pure-Python renderer
        # (confirmed empirically -- fine as a single frame, not x72). Rebuild
        # ax_full at a much coarser stride (~7.5k polygons) just for the
        # animation; the static PNG above already has full detail saved.
        ax_full.clear()
        viz.surface3d(ax_full, prob, stride=16)
        ax_full.set_title("full frame (rotating)", fontsize=10)
        anim = FuncAnimation(fig, lambda i: ax_full.view_init(elev=45, azim=360 * i / 72), frames=72)
        gif = out / f"heatmap3d_{tag}.gif"
        anim.save(gif, writer=PillowWriter(fps=15))
        print(f"wrote {gif} ({gif.stat().st_size} bytes)")
    if not show:
        plt.close(fig)


def panel_attention(model, x, image, args, out, tag, suptitle, dpi, show):
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    H, W = image.shape
    hm_logits, _, _, blk_in, _ = _forward_hooked(model, x)
    if not blk_in:
        print("SKIP attention: model.blocks missing/empty")
        return
    prob = torch.sigmoid(hm_logits)[0, 0].cpu().numpy()
    qx, qy = _peak_or_query(args, prob)
    div = model.attend_div   # grid stride: 16 native, 8 for the attend_div=8 variant
    gh, gw = H // div, W // div
    tok = int(np.clip(qy // div, 0, gh - 1)) * gw + int(np.clip(qx // div, 0, gw - 1))

    n = len(blk_in)
    fig = plt.figure(figsize=(16, 5.5 * n))
    gs = fig.add_gridspec(2 * n, 5)
    for bi, (blk, x_in) in enumerate(zip(model.blocks, blk_in)):
        with torch.no_grad():
            q, k, _ = blk.qkv_heads(blk.n1(x_in))
            A = torch.softmax((q.float() @ k.float().transpose(-2, -1)) / q.shape[-1] ** 0.5, dim=-1)
        a = A[0, :, tok, :].cpu().numpy()                    # (heads, T)
        a_mean = a.mean(axis=0).reshape(gh, gw)
        ent = float(-(a_mean * np.log(a_mean + 1e-12)).sum())
        _show_overlay(fig.add_subplot(gs[2 * bi:2 * bi + 2, 0]), image, a_mean,
                      f"block {bi} | mean-heads | H={ent:.2f} nats", mark=(qx, qy))
        for h in range(a.shape[0]):
            _show_overlay(fig.add_subplot(gs[2 * bi + h // 4, 1 + h % 4]), image,
                          a[h].reshape(gh, gw), f"head {h}", mark=(qx, qy))

    fig.suptitle(f"attention | {suptitle}")
    _finish(fig, out / f"attention_{tag}.png", dpi, show)


def panel_gates(model, x, image, args, out, tag, suptitle, dpi, show):
    import torch
    import cv2
    import matplotlib.pyplot as plt

    H, W = image.shape
    _, _, _, _, gate_args = _forward_hooked(model, x)
    names = [n for n in ("gate3", "gate4") if n in gate_args]
    if not names:
        print("SKIP gates: neither gate3 nor gate4 found")
        return
    fig, axes = plt.subplots(1, len(names), figsize=(7 * len(names), 6), squeeze=False)
    for ax, n in zip(axes[0], names):
        a, kw = gate_args[n]
        with torch.no_grad():
            alpha = getattr(model, n).alpha(*a, **kw)
        up = cv2.resize(alpha[0, 0].cpu().numpy(), (W, H), interpolation=cv2.INTER_LINEAR)
        im = ax.imshow(up, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"{n}.alpha", fontsize=9)
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["0 = vetoed", "1 = passed"])
    fig.suptitle(f"gates | {suptitle}")
    fig.text(0.5, 0.02, "H/2 and full-res skips are ungated by design (only H/4-and-coarser skips are gated)",
              ha="center", fontsize=8, style="italic")
    _finish(fig, out / f"gates_{tag}.png", dpi, show, rect=[0, 0.05, 1, 1])


def _swap_donor_gate_args(model, cfg, args, device):
    """A second, unrelated frame's gate_args -- the conditioning-ablation
    swap test (gate-actor Task C variant 4): wrong-image `g` (conditioning),
    right-image `skip`, is the cleanest probe of whether alpha actually
    depends on its conditioning signal. Always drawn from SynthVal (bit-
    identical, board present with high probability) regardless of whether
    the main sample came from --image, so the swap is available even for a
    raw-file invocation. Donor index is offset from --index (or 0) by a
    third of val_size so it's reliably a different scene, not adjacent."""
    import torch
    from dcc.dataset import SynthVal
    val_size = cfg["synth"]["val_size"]
    donor_idx = (0 if args.index is None else args.index + val_size // 3) % val_size
    ds = SynthVal(cfg, val_size, cfg["synth"]["val_seed"])
    donor_img, _ = ds[donor_idx]
    donor_x = torch.from_numpy(donor_img).float().div(255.0).unsqueeze(0).unsqueeze(0).to(device)
    _, _, _, _, donor_gate_args = _forward_hooked(model, donor_x)
    return donor_gate_args


def panel_gate_probe(model, x, image, out, tag, suptitle, dpi, show, corners=None, donor_gate_args=None):
    """Selectivity probe (Kaelin 2026-07-28): does an AttnGate's alpha
    actually vary spatially with content, or does it just attenuate the
    whole skip uniformly? panel_gates only shows alpha itself; this adds the
    skip (s3) and conditioning signal (z) alpha is computed FROM, the gated
    skip (s3*alpha) and what got thrown away (s3*(1-alpha)), and an alpha
    histogram against the pass-through init value (sigmoid(3)=0.9526). skip
    and z are aggregated over channels two ways -- channel-mean(abs) AND
    channel-max(abs), since they tell different stories (mean can hide a few
    strongly-selective channels; max can hide that most channels are flat).
    `corners` (GT {"x","y"} points, generated frames only) additionally
    builds a convex-hull board-region mask and reports mean alpha inside vs
    outside it -- printed, and marked on the histogram -- the quantitative
    read on whether alpha tracks the board or is diffuse. `donor_gate_args`
    (see _swap_donor_gate_args) adds a 5th column: baseline alpha vs
    swapped-context alpha side by side, plus their |diff| map and
    mean-abs-change/Pearson-r -- the single most direct image for "does the
    gate use conditioning, or only the skip.\""""
    import numpy as np
    import cv2
    import torch
    import matplotlib.pyplot as plt

    INIT_ALPHA = 0.9526   # sigmoid(3.0), AttnGate's pass-through init (dcc/model.py's AttnGate docstring)
    H, W = image.shape
    _, _, _, _, gate_args = _forward_hooked(model, x)
    names = [n for n in ("gate3", "gate4") if n in gate_args]
    if not names:
        print("SKIP gate_probe: neither gate3 nor gate4 found")
        return

    def chan_reduce(feat):
        """(channel-mean(abs), channel-max(abs)) 2D maps from a (1,C,h,w) tensor."""
        f = feat[0].abs()
        return f.mean(dim=0).cpu().numpy(), f.max(dim=0).values.cpu().numpy()

    def up(m):
        return cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)

    ncols = 5 if donor_gate_args else 4
    for n in names:
        a, kw = gate_args[n]
        skip, g = a[0], a[1]
        with torch.no_grad():
            alpha = getattr(model, n).alpha(*a, **kw)[0, 0]   # (h4, w4) -- skip's own res, pre display-upsample
        gated, suppressed = skip * alpha, skip * (1 - alpha)
        alpha_np = alpha.cpu().numpy()
        alpha_up = up(alpha_np)

        fig, axes = plt.subplots(3, ncols, figsize=(5 * ncols, 15), squeeze=False)
        _show_plain(axes[0, 0], image, "input")
        im = axes[0, 1].imshow(alpha_up, cmap="viridis", vmin=0, vmax=1)
        _no_ticks(axes[0, 1], f"{n}.alpha (fixed 0-1 scale)")
        fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
        _show_overlay(axes[0, 2], image, alpha_up, f"{n}.alpha over input", cmap="viridis", vmin=0, vmax=1)

        stats = f"mean={alpha_np.mean():.3f} std={alpha_np.std():.3f}"
        axes[0, 3].hist(alpha_np.ravel(), bins=50, range=(0, 1), color="steelblue")
        axes[0, 3].axvline(INIT_ALPHA, color="red", ls="--", lw=1.5, label=f"init={INIT_ALPHA}")
        if corners:
            pts = np.array([[c["x"], c["y"]] for c in corners], dtype=np.float32)
            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillConvexPoly(mask, cv2.convexHull(pts).astype(np.int32), 1)
            mb = mask.astype(bool)
            m_in, m_out = float(alpha_up[mb].mean()), float(alpha_up[~mb].mean())
            ratio = m_in / m_out if m_out > 1e-9 else float("nan")
            axes[0, 3].axvline(m_in, color="lime", lw=1.5, label=f"inside board={m_in:.3f}")
            axes[0, 3].axvline(m_out, color="orange", lw=1.5, label=f"outside board={m_out:.3f}")
            stats += f" | inside={m_in:.3f} outside={m_out:.3f} ratio={ratio:.2f}"
            print(f"  {n} board-region alpha: inside={m_in:.4f} outside={m_out:.4f} ratio={ratio:.3f}")
        axes[0, 3].legend(fontsize=6, loc="upper left")
        axes[0, 3].set_title(f"{n}.alpha histogram\n{stats}", fontsize=8)

        if donor_gate_args and n in donor_gate_args:
            (_, donor_g), _ = donor_gate_args[n]
            with torch.no_grad():
                alpha_swap = getattr(model, n).alpha(skip, donor_g)[0, 0].cpu().numpy()
            diff = np.abs(alpha_np - alpha_swap)
            r = float(np.corrcoef(alpha_np.ravel(), alpha_swap.ravel())[0, 1])
            axes[0, 4].imshow(up(alpha_swap), cmap="viridis", vmin=0, vmax=1)
            _no_ticks(axes[0, 4], f"{n}.alpha SWAPPED-context g\n(right skip, wrong-frame conditioning)")
            axes[1, 4].imshow(up(diff), cmap="magma", vmin=0, vmax=1)
            _no_ticks(axes[1, 4], f"|alpha - alpha_swap|\nmean_abs_change={diff.mean():.3f}  pearson_r={r:.3f}")
            axes[2, 4].axis("off")
            print(f"  {n} conditioning swap: mean_abs_change={diff.mean():.4f} pearson_r={r:.4f}")

        for col, (label, feat) in enumerate((("skip s3", skip), ("cond z", g),
                                              ("gated s3*alpha", gated), ("suppressed s3*(1-alpha)", suppressed))):
            mean_map, max_map = chan_reduce(feat)
            axes[1, col].imshow(up(mean_map), cmap="viridis")
            _no_ticks(axes[1, col], f"{label} | channel-mean(abs)")
            axes[2, col].imshow(up(max_map), cmap="viridis")
            _no_ticks(axes[2, col], f"{label} | channel-max(abs)")

        fig.suptitle(f"gate_probe {n} | {suptitle}")
        _finish(fig, out / f"gateprobe_{n}_{tag}.png", dpi, show)


def panel_gate_flow(model, x, image, out, tag, suptitle, dpi, show, corners=None):
    """Minimal, explicitly-labelled view of WHAT THE DECODER ACTUALLY EATS
    (Kaelin 2026-07-28, asked for after panel_gate_probe proved too busy):
    exactly the four maps on the path into d3's convolution, plus the alpha
    histogram. Deliberately NOT the full diagnostic -- panel_gate_probe
    keeps the suppressed-content/channel-max/swap columns for when the
    question is "is the gate selective"; this one answers "what is
    concatenated, and in what proportion".

    The concat site is dcc/model.py:248 -- `d3(cat([up2(z), gate3(s3, z)]))`
    -- so the panels are, left to right, the network input; the raw encoder
    skip s3 BEFORE gating; the upsampled attention/bottleneck output up2(z),
    which is the other half of the concat; and the GATED skip s3*alpha,
    which is what physically enters the concat. Every feature map is
    channel-mean(|.|) over its channels (one reduction, stated on the axis,
    rather than probe's two) and shown at its own native resolution with the
    shape in the title, so the H/4-vs-H/8 asymmetry is visible rather than
    hidden by a common resize."""
    import numpy as np
    import cv2
    import torch
    import matplotlib.pyplot as plt

    INIT_ALPHA = 0.9526   # sigmoid(3.0), AttnGate's pass-through init
    H, W = image.shape
    _, _, _, _, gate_args = _forward_hooked(model, x)
    names = [n for n in ("gate3", "gate4") if n in gate_args]
    if not names:
        print("SKIP gate_flow: neither gate3 nor gate4 found")
        return

    def cmean(feat):
        return feat[0].abs().mean(dim=0).cpu().numpy()

    for n in names:
        a, kw = gate_args[n]
        skip, g = a[0], a[1]
        with torch.no_grad():
            alpha = getattr(model, n).alpha(*a, **kw)[0, 0]
        gated = skip * alpha
        alpha_np = alpha.cpu().numpy()
        # The ACTUAL tensor d3 consumes: cat([up2(z), s3*alpha], dim=1) -- dcc/model.py:249.
        # Panels 2-4 each show one ADDITIVE contribution; this is the fused result, and the
        # channel-mean over it is weighted by the 2:1 channel split (256 from z vs 128 from
        # the gated skip), which is stated in the title so the z-dominance is read as
        # arithmetic rather than as a finding.
        from dcc.model import up2 as _up2
        concat = torch.cat([_up2(g), gated], 1)
        sk, gz, gt, cc = cmean(skip), cmean(g), cmean(gated), cmean(concat)
        sh = lambda t: f"{tuple(t.shape[1:])[0]}ch {tuple(t.shape[2:])[0]}x{tuple(t.shape[3:])[0]}"

        fig, axes = plt.subplots(1, 6, figsize=(31, 5.6), squeeze=False)
        _show_plain(axes[0, 0], image, f"1. INPUT\ngrayscale {H}x{W}")
        for ax, m, title in (
            (axes[0, 1], sk, f"2. ENCODER SKIP  s3  (pre-gate)\n{sh(skip)} @ H/4 -- channel-mean|.|"),
            (axes[0, 2], gz, f"3. ATTENTION OUT  z  (post-transformer)\n{sh(g)} @ H/8 -- the OTHER half of the concat"),
            (axes[0, 3], gt, f"4. GATED SKIP  s3 x alpha\n{sh(gated)} @ H/4 -- the gate's additive contribution"),
            (axes[0, 4], cc, f"5. CONCATENATED  cat[up2(z), s3*alpha]\n{sh(concat)} @ H/4 -- what d3 CONVOLVES (256z:128skip)"),
        ):
            im = ax.imshow(m, cmap="magma")
            _no_ticks(ax, title)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        stats = f"mean={alpha_np.mean():.3f}  std={alpha_np.std():.3f}"
        axes[0, 5].hist(alpha_np.ravel(), bins=50, range=(0, 1), color="steelblue")
        axes[0, 5].axvline(INIT_ALPHA, color="red", ls="--", lw=1.5, label=f"pass-through init = {INIT_ALPHA}")
        if corners:
            pts = np.array([[c["x"], c["y"]] for c in corners], dtype=np.float32)
            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillConvexPoly(mask, cv2.convexHull(pts).astype(np.int32), 1)
            au = cv2.resize(alpha_np, (W, H), interpolation=cv2.INTER_LINEAR)
            mb = mask.astype(bool)
            m_in, m_out = float(au[mb].mean()), float(au[~mb].mean())
            axes[0, 5].axvline(m_in, color="lime", lw=1.5, label=f"mean ON board = {m_in:.3f}")
            axes[0, 5].axvline(m_out, color="orange", lw=1.5, label=f"mean OFF board = {m_out:.3f}")
            stats += f"  |  on-board {m_in:.3f} vs off-board {m_out:.3f}"
        axes[0, 5].set_xlabel("alpha  (0 = skip fully suppressed, 1 = passed through)")
        axes[0, 5].set_ylabel("pixel count")
        axes[0, 5].legend(fontsize=7, loc="upper left")
        axes[0, 5].set_title(f"6. GATE WEIGHT alpha -- distribution\n{stats}", fontsize=9)

        fig.suptitle(f"{suptitle}  --  {n} flow into d3:  d3( concat[ up2(z) , s3*alpha ] )", fontsize=12)
        _finish(fig, out / f"gateflow_{n}_{tag}.png", dpi, show)


def panel_gate_ablation(model, x, image, out, tag, suptitle, dpi, show, corners=None, donor_gate_args=None):
    """Conditioning-ablation (Kaelin/team-lead Task C, 2026-07-28): does
    alpha actually depend on its conditioning signal g, or would skip s3
    alone produce the same mask -- i.e. has the gate degenerated into a
    skip-driven saliency filter that ignores global board context? Holds
    skip FIXED and degrades g three ways: spatial-mean (kills spatial/global
    structure, keeps per-channel magnitude), zeroed (W_g contributes only
    its bias), swapped (a DIFFERENT frame's own g entirely -- the cleanest
    single test; needs donor_gate_args, see _swap_donor_gate_args). Also
    splits gate.wx(skip) vs gate.wg(g) BEFORE they're summed+ReLU'd: if
    ||wg(g)|| << ||wx(skip)|| the conditioning is numerically negligible
    regardless of what the alpha ablations show -- a one-number answer to
    the same question. Row 0 is each variant's alpha (fixed 0-1 scale,
    board in/out/ratio in the title); row 1 is |baseline - variant| against
    baseline (magnitude balance as text under baseline itself)."""
    import numpy as np
    import cv2
    import torch
    import matplotlib.pyplot as plt

    H, W = image.shape
    _, _, _, _, gate_args = _forward_hooked(model, x)
    names = [n for n in ("gate3", "gate4") if n in gate_args]
    if not names:
        print("SKIP gate_ablation: neither gate3 nor gate4 found")
        return

    def up(m):
        return cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)

    def board_stats(alpha_up):
        if not corners:
            return None, None, None
        pts = np.array([[c["x"], c["y"]] for c in corners], dtype=np.float32)
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(pts).astype(np.int32), 1)
        mb = mask.astype(bool)
        m_in, m_out = float(alpha_up[mb].mean()), float(alpha_up[~mb].mean())
        return m_in, m_out, (m_in / m_out if m_out > 1e-9 else float("nan"))

    for n in names:
        gate = getattr(model, n)
        a, kw = gate_args[n]
        skip, g = a[0], a[1]
        variants = {"baseline": g, "spatial-mean g": g.mean(dim=(2, 3), keepdim=True).expand_as(g),
                    "zeroed g": torch.zeros_like(g)}
        if donor_gate_args and n in donor_gate_args:
            (_, donor_g), _ = donor_gate_args[n]
            variants["swapped g"] = donor_g

        with torch.no_grad():
            alphas = {name: gate.alpha(skip, gv)[0, 0].cpu().numpy() for name, gv in variants.items()}
            wx, wg = gate.wx(skip), gate.wg(g)
        wx_rms, wg_rms = float(wx.pow(2).mean().sqrt()), float(wg.pow(2).mean().sqrt())
        wx_l2, wg_l2 = float(wx.norm()), float(wg.norm())
        print(f"  {n} magnitude balance: ||wx(skip)||_rms={wx_rms:.4f} ||wg(g)||_rms={wg_rms:.4f} "
              f"(wg/wx ratio={wg_rms / wx_rms:.3f}) | L2 wx={wx_l2:.2f} wg={wg_l2:.2f}")
        base_in, base_out, base_ratio = board_stats(up(alphas["baseline"]))
        if base_in is not None:
            print(f"  {n} baseline: inside={base_in:.4f} outside={base_out:.4f} ratio={base_ratio:.3f}")

        fig, axes = plt.subplots(2, len(variants), figsize=(5 * len(variants), 10), squeeze=False)
        for col, (name, av) in enumerate(alphas.items()):
            av_up = up(av)
            m_in, m_out, ratio = board_stats(av_up)
            title = name if m_in is None else f"{name}\nin={m_in:.3f} out={m_out:.3f} ratio={ratio:.2f}"
            axes[0, col].imshow(av_up, cmap="viridis", vmin=0, vmax=1)
            _no_ticks(axes[0, col], title)
            if name == "baseline":
                axes[1, col].axis("off")
                axes[1, col].text(0.05, 0.5, f"wx(skip) rms={wx_rms:.3f}\nwg(g) rms={wg_rms:.3f}\n"
                                   f"wg/wx ratio={wg_rms / wx_rms:.3f}", fontsize=10, va="center")
            else:
                diff = np.abs(alphas["baseline"] - av)
                r = float(np.corrcoef(alphas["baseline"].ravel(), av.ravel())[0, 1])
                axes[1, col].imshow(up(diff), cmap="magma", vmin=0, vmax=1)
                _no_ticks(axes[1, col], f"|baseline - {name}|\nmean|d|={diff.mean():.3f} r={r:.3f}")
                print(f"  {n} {name}: mean_abs_change={diff.mean():.4f} pearson_r={r:.4f} "
                      f"inside={m_in:.4f} outside={m_out:.4f} ratio={ratio:.3f}")

        fig.suptitle(f"gate_ablation {n} (conditioning degraded, skip fixed) | {suptitle}")
        _finish(fig, out / f"gateablation_{n}_{tag}.png", dpi, show)


def _erf_grad(m, x, args):
    """Unit-gradient effective-receptive-field probe: backward from one
    class-head logit (the strongest ID, or --query-xy's cell) to the input.
    Returns (log10(|grad|+eps) (H,W), the (x,y) point that was targeted)."""
    import numpy as np
    xin = x.clone().requires_grad_(True)
    hm_logits, cls_logits = m(xin)
    if args.query_xy is not None:
        qx, qy = args.query_xy
        gh, gw = cls_logits.shape[-2:]
        cy, cx = int(np.clip(qy // 4, 0, gh - 1)), int(np.clip(qx // 4, 0, gw - 1))
        ch = int(cls_logits[0, :, cy, cx].argmax())
    else:
        ch, cy, cx = np.unravel_index(int(cls_logits[0].argmax()), cls_logits.shape[1:])
        qx, qy = cx * 4.0 + 1.5, cy * 4.0 + 1.5   # cell-centre in input px (targets.py convention)
    cls_logits[0, int(ch), int(cy), int(cx)].backward()
    grad = xin.grad[0, 0].detach().cpu().numpy()
    return np.log10(np.abs(grad) + 1e-12), (float(qx), float(qy))


def panel_erf(model, x, image, args, out, tag, suptitle, dpi, show, model_b=None, suptitle_b=None):
    import matplotlib.pyplot as plt

    g_a, q = _erf_grad(model, x, args)
    panels = [(g_a, suptitle)]
    if model_b is not None:
        g_b, _ = _erf_grad(model_b, x, args)
        panels.append((g_b, suptitle_b))
    vmin = min(g.min() for g, _ in panels)
    vmax = max(g.max() for g, _ in panels)

    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 6.5), squeeze=False)
    for ax, (g, name) in zip(axes[0], panels):
        _show_overlay(ax, image, g, name, cmap="magma", alpha=0.6, vmin=vmin, vmax=vmax, mark=q)
    fig.suptitle(f"Effective receptive field probe | {suptitle}")
    _finish(fig, out / f"erf_{tag}.png", dpi, show)


def panel_features(model, x, image, out, tag, suptitle, dpi, show):
    import cv2
    import matplotlib.pyplot as plt

    H, W = image.shape
    _, _, feats, _, _ = _forward_hooked(model, x)
    names = [n for n in ("e1", "e2", "e3", "e4", "e5") if n in feats]
    if not names:
        print("SKIP features: no e1..e5 encoder stages found")
        return
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4.2), squeeze=False)
    for ax, n in zip(axes[0], names):
        fmap = feats[n][0].abs().mean(dim=0).cpu().numpy()
        ax.imshow(cv2.resize(fmap, (W, H), interpolation=cv2.INTER_LINEAR), cmap="viridis")
        ax.set_title(n, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"encoder features (mean |activation|) | {suptitle}")
    _finish(fig, out / f"features_{tag}.png", dpi, show)


# --------------------------------------------------------------------------- main

def main():
    parser = build_parser()
    args = parser.parse_args()
    requested = PANELS_ALL if args.panels == "all" else [p.strip() for p in args.panels.split(",") if p.strip()]
    unknown = set(requested) - set(PANELS_ALL)
    if unknown:
        parser.error(f"unknown panel(s) {sorted(unknown)}; choose from {PANELS_ALL} or 'all'")

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import torch
    try:
        from dcc.model import DetectorNet, detector_kwargs
    except ImportError as e:
        print(f"dcc.model not importable yet ({e}) -- introspect.py cannot run any panel without it "
              f"(py_compile/--help are unaffected).")
        sys.exit(3)
    from dcc.dataset import load_config

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image, tag, record = _sample(cfg, args)
    W, H = cfg["input_size"]
    model, trained = _build_module(DetectorNet, args.ckpt, device, H, W, **detector_kwargs(cfg))
    x = torch.from_numpy(image).float().div(255.0).unsqueeze(0).unsqueeze(0).to(device)
    suptitle = _suptitle(args.ckpt, tag, trained)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if "pipeline" in requested:
        panel_pipeline(model, x, image, cfg, args, out, tag, suptitle, args.dpi, args.show, device)
    if "heatmap3d" in requested:
        panel_heatmap3d(model, x, image, args, out, tag, suptitle, args.dpi, args.show, args.gif)
    if "attention" in requested:
        panel_attention(model, x, image, args, out, tag, suptitle, args.dpi, args.show)
    if "gates" in requested:
        panel_gates(model, x, image, args, out, tag, suptitle, args.dpi, args.show)
    donor_gate_args = None
    if "gateprobe" in requested or "gateablation" in requested:
        donor_gate_args = _swap_donor_gate_args(model, cfg, args, device)
    if "gateprobe" in requested:
        panel_gate_probe(model, x, image, out, tag, suptitle, args.dpi, args.show,
                          corners=(record.get("corners") if record else None), donor_gate_args=donor_gate_args)
    if "gateflow" in requested:
        panel_gate_flow(model, x, image, out, tag, suptitle, args.dpi, args.show,
                         corners=(record.get("corners") if record else None))
    if "gateablation" in requested:
        panel_gate_ablation(model, x, image, out, tag, suptitle, args.dpi, args.show,
                             corners=(record.get("corners") if record else None), donor_gate_args=donor_gate_args)
    if "erf" in requested:
        model_b, suptitle_b = None, None
        if args.ckpt_b:
            model_b, b_trained = _build_module(DetectorNet, args.ckpt_b, device, H, W, **detector_kwargs(cfg))
            suptitle_b = _suptitle(args.ckpt_b, tag, b_trained)
        panel_erf(model, x, image, args, out, tag, suptitle, args.dpi, args.show, model_b, suptitle_b)
    if "features" in requested:
        panel_features(model, x, image, out, tag, suptitle, args.dpi, args.show)

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
