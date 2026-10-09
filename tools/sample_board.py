"""tools/sample_board.py -- a 3x3 board of deliberately-contrasted training samples.

Nine RANDOM samples would mostly look alike: the augmentation draws are independent,
so the modal frame is mid-scale, mid-brightness, unoccluded. To show the envelope
rather than its centre, this draws a pool from the real training stream and then
fills nine SLOTS, each claiming the pool's most extreme sample on one axis.

Two slots cannot be reached by sampling at a sane pool size (differencing fires on
10% of frames, negatives on 5%, and neither is reliably identifiable after the fact),
so they are FORCED and labelled as such on the panel. Every other panel is an
untouched draw from the same generator the trainer consumes.

Captions state MEASURED properties (scale, visible count, mean level) plus the slot
that selected the frame. "blurred" and "blown highlights" are descriptions of the
measured image, not claims about which augmentation fired -- motion blur and defocus
are not separable after the fact.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# (tag, key) -- key(feat) is maximised over the pool; higher = better fit for the slot.
SLOTS = [
    ("far -- small board", lambda f: -f["s"]),
    ("close range, partial view", lambda f: f["s"] - 40 * f["nvis"] / 16),
    ("steep tilt", lambda f: abs(f["tilt"])),
    ("object occlusion (SAM2)", lambda f: f["ncut"] * 10 + (16 - f["nvis"])),
    ("low light", lambda f: -f["mean"]),
    ("blurred", lambda f: -f["lapvar"]),
    ("blown highlights", lambda f: f["sat"]),
]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/rev640.yaml")
    p.add_argument("--pool", type=int, default=240, help="samples drawn before selection")
    p.add_argument("--seed", type=int, default=20260729)
    p.add_argument("--rect-occlusion", action="store_true",
                   help="keep the rectangular dropout holes (on in training, off in this "
                        "board by default -- they read as a rendering bug on a slide, and "
                        "the SAM2 cutouts make the same point with a real object)")
    p.add_argument("--out", default="paper/results_rev6/14_data/training_samples_3x3.png")
    return p


def main():
    a = build_parser().parse_args()
    import copy
    import cv2, numpy as np
    cv2.setNumThreads(1)
    from dcc.dataset import load_config
    from dcc.synth import generate_sample, list_backgrounds
    from dcc.viz import draw_overlay, tile

    cfg = load_config(a.config)
    if not a.rect_occlusion:
        # Illustration-only: the trained stream DOES carry these (occlusion.p 0.55).
        # Cutout probability is left at its trained value so the object-occlusion
        # panels stay representative rather than staged.
        cfg = copy.deepcopy(cfg)
        cfg["synth"]["occlusion"]["p"] = 0.0
    bg = list_backgrounds(cfg["synth"]["backgrounds"])

    def feats(rec, meta):
        g = rec["image"]
        return {"s": rec["s_px"], "nvis": sum(c["visible"] for c in rec["corners"]),
                # negatives carry no board, hence no affine components
                "tilt": np.degrees((meta["components"] or {}).get("tilt", 0.0)),
                "ncut": len(meta.get("cutouts") or []),
                "mean": float(g.mean()), "sat": float((g >= 250).mean()),
                "lapvar": float(cv2.Laplacian(g, cv2.CV_64F).var())}

    pool = []
    for i in range(a.pool):
        rng = np.random.default_rng([a.seed, i])
        rec, meta = generate_sample(cfg, rng, bg, force_negative=False)
        if sum(c["visible"] for c in rec["corners"]) >= 4:
            pool.append((rec, meta, feats(rec, meta)))
    print(f"pool: {len(pool)} usable of {a.pool}")

    picked, used = [], set()
    for tag, key in SLOTS:
        j = max((k for k in range(len(pool)) if k not in used), key=lambda k: key(pool[k][2]))
        used.add(j)
        picked.append((tag, *pool[j]))

    # The two slots sampling cannot reach. Forced, and said so on the panel.
    dcfg = copy.deepcopy(cfg)
    dcfg["synth"]["photometric"]["differencing_p"] = 1.0
    for tag, c, neg in (("differencing pair [forced]", dcfg, False),
                        ("negative -- no board [forced]", cfg, True)):
        for i in range(200):                      # first draw with enough board to read
            rec, meta = generate_sample(c, np.random.default_rng([a.seed + 7, i]), bg,
                                        force_negative=neg)
            if neg or sum(cc["visible"] for cc in rec["corners"]) >= 10:
                break
        picked.append((tag, rec, meta, feats(rec, meta)))

    cells = []
    for tag, rec, meta, f in picked:
        im = draw_overlay(rec["image"], rec, meta, draw_indices=False, filled=False, radius=5)
        bar = np.zeros((44, im.shape[1], 3), np.uint8)
        sub = ("no board" if not rec["board_present"] else
               f"s={f['s']:.0f}px  {f['nvis']}/16 visible  mean={f['mean']:.0f}")
        cv2.putText(bar, tag, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bar, sub, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
        cells.append(cv2.vconcat([bar, im]))
        print(f"  {tag:32s} {sub}")

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), tile(cells, cols=3))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
