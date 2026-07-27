"""Acceptance gate for the synthetic-data pipeline. Exits 0 iff every gate
passes, 1 if any fails, 2 if the background corpus isn't there yet. Argparse
runs before any heavy import so --help never needs dcc/numpy/cv2/matplotlib.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

_PROJ_DIR = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _PROJ_DIR)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--out", default="audit/")
    p.add_argument("--n-dist", type=int, default=10000)
    p.add_argument("--n-overlay", type=int, default=200)
    p.add_argument("--n-roundtrip", type=int, default=1000)
    p.add_argument("--save", action="store_true", help="also materialise the dist pass to val_set.npz")
    return p


def _bin_gate(vals, edges, tol, log=False):
    """Fraction per bin + whether every bin is within tol (relative) of its
    expected share: bin width over total span, log-space for log-uniform
    quantities (s_px, whose edges include a sub-octave [12,16) bin), linear
    otherwise (refiner d). Equal-width edges reduce to equal shares."""
    import numpy as np
    counts, _ = np.histogram(vals, bins=edges)
    total = int(counts.sum())
    if not total:
        return [0.0] * (len(edges) - 1), False
    e = np.log(edges) if log else np.asarray(edges, dtype=float)
    share = np.diff(e) / (e[-1] - e[0])
    frac = counts / total
    return frac.tolist(), bool(np.all(np.abs(frac - share) <= tol * share))


def _recompose_corners(comp, corner_px, render_res, w2, h2):
    """Audit round-trip: the full 3x3 H rebuilt from meta['components']
    independently of dcc.synth._sample_affine/_perspective_factor, so a
    regression there can't cancel itself out."""
    import numpy as np
    R = np.array([[np.cos(comp["theta"]), -np.sin(comp["theta"])],
                  [np.sin(comp["theta"]), np.cos(comp["theta"])]])
    Sh = np.array([[1.0, np.tan(comp["shear_x"])], [np.tan(comp["shear_y"]), 1.0]])
    A = (comp["s"] / (render_res // 5)) * (R @ Sh)
    c_r = np.array([(render_res - 1) / 2, (render_res - 1) / 2])
    c_in = np.array([(w2 - 1) / 2, (h2 - 1) / 2])
    m2 = c_in + np.array([comp["tx"], comp["ty"]]) - A @ c_r
    M3 = np.eye(3)
    M3[:2, :2], M3[:2, 2] = A, m2

    SQ = render_res // 5
    g = np.sin(comp["tilt"]) * comp["s"] / (comp["fov_scale"] * w2 * SQ)
    gx, gy = g * np.cos(comp["psi"]), g * np.sin(comp["psi"])
    cr = (render_res - 1) / 2
    Tcr = np.array([[1.0, 0.0, cr], [0.0, 1.0, cr], [0.0, 0.0, 1.0]])
    Tmcr = np.array([[1.0, 0.0, -cr], [0.0, 1.0, -cr], [0.0, 0.0, 1.0]])
    Pg = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [gx, gy, 1.0]])
    H = M3 @ (Tcr @ Pg @ Tmcr)

    hom = np.hstack([corner_px, np.ones((len(corner_px), 1))]) @ H.T
    return hom[:, :2] / hom[:, 2:3]


def _val_sample(generate_sample, cfg, bg_files, val_seed, n, i):
    """Reproduces SynthVal(cfg, n, val_seed)[i]'s record bit-for-bit but also
    returns meta (SynthVal discards it) -- mirrors dcc.dataset.SynthVal
    exactly, including its stratified-s pre-draw for positive samples."""
    import numpy as np
    rng = np.random.default_rng([val_seed, i])
    if rng.random() < cfg["negative_p"]:
        return generate_sample(cfg, rng, bg_files, force_negative=True)
    a, b = cfg["scale_range_px"]
    s = a * (b / a) ** ((i + rng.random()) / n)
    return generate_sample(cfg, rng, bg_files, s=s, force_negative=False)


def _hash_pair(img, record):
    """SHA1 of image bytes + a canonical repr of the record. Duplicated
    (not imported) into _REPRO_CODE below: this environment has an unrelated
    top-level `tools` package on sys.path that shadows any local namespace
    package of the same name, so a subprocess `from tools.audit import ...`
    is not reliable -- keep the two copies in sync."""
    corners = sorted((c["index"], float(c["x"]), float(c["y"]), bool(c["visible"]))
                      for c in record["corners"])
    payload = repr((record["board_present"], float(record["s_px"]), corners)).encode()
    return hashlib.sha1(img.tobytes()).hexdigest(), hashlib.sha1(payload).hexdigest()


_REPRO_CODE = """import sys, hashlib
sys.path.insert(0, {pd!r})
from dcc.dataset import load_config, SynthVal
cfg = load_config({cfg!r})
ds = SynthVal(cfg, {n}, cfg["synth"]["val_seed"])
for i in range({n}):
    img, rec = ds[i]
    corners = sorted((c["index"], float(c["x"]), float(c["y"]), bool(c["visible"])) for c in rec["corners"])
    payload = repr((rec["board_present"], float(rec["s_px"]), corners)).encode()
    print(hashlib.sha1(img.tobytes()).hexdigest(), hashlib.sha1(payload).hexdigest())
"""


class _NpEnc(json.JSONEncoder):
    """report.json safety net: cast any stray numpy scalar/array that slipped
    through without an explicit float()/tolist() (e.g. from meta/record
    fields we don't fully control) rather than crashing at the last line."""
    def default(self, o):
        import numpy as np
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def _gate_overlays(SynthVal, viz, cfg, val_seed, n_overlay, out):
    import cv2
    ds = SynthVal(cfg, n_overlay, val_seed)
    imgs = [viz.draw_overlay(*ds[i]) for i in range(n_overlay)]
    for s in range(0, len(imgs), 25):
        cv2.imwrite(str(out / f"overlay_{s // 25:02d}.png"), viz.tile(imgs[s:s + 25]))


def _gate_distributions(generate_sample, cfg, bg_files, val_seed, n_dist, out, save):
    import numpy as np
    import matplotlib.pyplot as plt
    s_pos, vis_counts, neg_count, hole_count = [], [], 0, 0
    images, records = [], []
    for i in range(n_dist):
        record, meta = _val_sample(generate_sample, cfg, bg_files, val_seed, n_dist, i)
        if record["board_present"]:
            s_pos.append(record["s_px"])
            vis_counts.append(sum(c["visible"] for c in record["corners"]))
        else:
            neg_count += 1
        if meta["holes"]:
            hole_count += 1
        if save:
            images.append(record["image"])
            records.append(record)

    s_pos = np.array(s_pos)
    a, b = cfg["scale_range_px"]
    s_edges = sorted({float(e) for e in (a, 16.0, 32.0, 64.0, 128.0, b) if a <= e <= b})
    octave_frac, octave_ok = _bin_gate(s_pos, s_edges, 0.2, log=True)
    neg_frac = neg_count / n_dist
    neg_ok = abs(neg_frac - cfg["negative_p"]) <= 0.005
    occ_p = cfg["synth"]["occlusion"]["p"]
    occ_frac = hole_count / n_dist
    occ_ok = abs(occ_frac - occ_p) <= 0.05

    fig, ax = plt.subplots()
    ax.hist(s_pos, bins=np.logspace(np.log10(s_edges[0]), np.log10(s_edges[-1]), 40))
    ax.set_xscale("log"); ax.set_xlabel("s_px"); ax.set_ylabel("count")
    ax.set_title(f"s_px octave shares {[round(f, 3) for f in octave_frac]}")
    fig.savefig(out / "dist_s_px.png"); plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(vis_counts, bins=np.arange(18) - 0.5)
    ax.set_xlabel("visible corners"); ax.set_ylabel("count")
    fig.savefig(out / "dist_visible.png"); plt.close(fig)

    if save:
        labels = [{k: v for k, v in r.items() if k != "image"} for r in records]
        np.savez_compressed(out / "val_set.npz", images=np.stack(images),
                             records=np.array([json.dumps(r) for r in labels]))

    report = {"octave_frac": octave_frac, "octave_ok": octave_ok, "negative_frac": neg_frac,
              "negative_ok": neg_ok, "occlusion_frac": occ_frac, "occlusion_ok": occ_ok,
              "visible_count_hist": np.bincount(vis_counts, minlength=17).tolist()}
    gates = [("s_px octave flatness", octave_ok), ("negative fraction", neg_ok),
             ("occlusion incidence", occ_ok)]
    return gates, report


def _gate_roundtrip(generate_sample, board, cfg, bg_files, val_seed, n_roundtrip):
    import numpy as np
    render_res = cfg["synth"]["render_res"]
    W, H = cfg["input_size"]
    _, corner_px = board.render_board(render_res)
    worst = 0.0
    for i in range(n_roundtrip):
        rng = np.random.default_rng([val_seed, i])
        record, meta = generate_sample(cfg, rng, bg_files, photometric=False, occlude=False,
                                        force_negative=False)
        corners_img = _recompose_corners(meta["components"], corner_px, render_res, W, H)
        record_xy = np.zeros((16, 2))
        for c in record["corners"]:
            record_xy[c["index"]] = (c["x"], c["y"])
        worst = max(worst, float(np.linalg.norm(corners_img - record_xy, axis=1).max()))
    return worst < 0.01, worst


def _gate_repeatability(SynthVal, cfg, args_config, val_seed):
    n = 50
    ds = SynthVal(cfg, n, val_seed)
    main_hashes = [_hash_pair(*ds[i]) for i in range(n)]
    cfg_abs = str(Path(args_config).resolve())
    code = _REPRO_CODE.format(pd=_PROJ_DIR, cfg=cfg_abs, n=n)
    result = subprocess.run([sys.executable, "-c", code], cwd=_PROJ_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        print("repeatability subprocess FAILED:\n" + result.stderr[-3000:])
        return False
    sub_hashes = [tuple(line.split()) for line in result.stdout.splitlines()]
    return main_hashes == sub_hashes


def _gate_refiner(RefinerVal, generate_sample, cut_refiner_crops, cfg, bg_files, val_seed, out):
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    syn = cfg["synth"]
    ds = RefinerVal(cfg, syn["refiner_val_composites"])  # default seed = val_seed+1: audit the CANONICAL set
    d = np.array([c["d"] for i in range(len(ds)) for c in ds[i]]).reshape(-1, 2)
    edges = np.linspace(-3.9375, 3.9375, 9)
    dx_frac, dx_ok = _bin_gate(d[:, 0], edges, 0.25)
    dy_frac, dy_ok = _bin_gate(d[:, 1], edges, 0.25)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(d[:, 0], bins=edges); axes[0].set_title("refiner dx")
    axes[1].hist(d[:, 1], bins=edges); axes[1].set_title("refiner dy")
    fig.savefig(out / "refiner_d_hist.png"); plt.close(fig)

    CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    flat = []
    for i in range(500):
        # val_seed+2: independent of the val_seed / val_seed+1 streams SynthVal
        # and RefinerVal already consume, so this content-check draws its own
        # composites rather than re-walking samples another gate already used.
        rng = np.random.default_rng([val_seed + 2, i])
        record, _ = generate_sample(cfg, rng, bg_files, size_mult=syn["refiner_res_mult"],
                                     photometric=False, force_negative=False)
        pts = [(c["x"], c["y"]) for c in record["corners"] if c["visible"]]
        flat += cut_refiner_crops(cfg, rng, record["image"], pts)
        if len(flat) >= 100:
            break

    # Gate on median + p90, not max: cornerSubPix (the check's instrument, not
    # the data) is ill-conditioned on small-s crops (11x11 window vs 0.15*s
    # marker margin) and on high-anisotropy wedges (shear+rotation collapse two
    # quadrants toward a knife-edge), so its error tail is heavy on perfectly
    # correct crops. A systematic geometry/label bug (axis swap, half-pixel
    # offset) shifts EVERY crop by >=0.5 px and moves the median; the tail
    # doesn't. Labels themselves are exact by the round-trip gate.
    errs = []
    for rec in flat[:100]:
        crop, dd = rec["crop"], rec["d"]
        seed_pt = np.array([[[12 + dd[0], 12 + dd[1]]]], dtype=np.float32)
        refined = cv2.cornerSubPix(crop, seed_pt, (5, 5), (-1, -1), CRIT).reshape(2)
        errs.append(float(np.linalg.norm(refined - np.array([12 + dd[0], 12 + dd[1]]))))
    content_med, content_p90 = float(np.median(errs)), float(np.percentile(errs, 90))
    content_ok = content_med <= 0.10 and content_p90 <= 0.50

    report = {"crops": len(d), "dx_frac": dx_frac, "dy_frac": dy_frac, "hist_ok": dx_ok and dy_ok,
              "content_median_px": content_med, "content_p90_px": content_p90,
              "content_max_px": float(max(errs)), "content_ok": content_ok, "content_n": len(errs)}
    gates = [("refiner d histogram", dx_ok and dy_ok), ("refiner content check", content_ok)]
    return gates, report


def main():
    args = build_parser().parse_args()

    import numpy as np
    import cv2
    import skimage
    import matplotlib
    matplotlib.use("Agg")
    from dcc.dataset import load_config, SynthVal, RefinerVal
    from dcc.synth import generate_sample, list_backgrounds, cut_refiner_crops
    from dcc.trainutil import generator_fingerprint
    from dcc import board, viz

    cfg = load_config(args.config)
    bg_path = cfg["synth"]["backgrounds"]
    bg_files = list_backgrounds(bg_path)
    if not bg_files:
        print(f"background corpus missing/empty at {bg_path!r} -- COCO download may still be running")
        sys.exit(2)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    val_seed = cfg["synth"]["val_seed"]

    _gate_overlays(SynthVal, viz, cfg, val_seed, args.n_overlay, out)

    gates = []
    dist_gates, dist_report = _gate_distributions(generate_sample, cfg, bg_files, val_seed,
                                                   args.n_dist, out, args.save)
    gates += dist_gates

    rt_ok, rt_max = _gate_roundtrip(generate_sample, board, cfg, bg_files, val_seed, args.n_roundtrip)
    gates.append(("round-trip max px", rt_ok))

    repeat_ok = _gate_repeatability(SynthVal, cfg, args.config, val_seed)
    gates.append(("repeatability (subprocess byte-identical)", repeat_ok))

    refiner_gates, refiner_report = _gate_refiner(RefinerVal, generate_sample, cut_refiner_crops,
                                                   cfg, bg_files, val_seed, out)
    gates += refiner_gates

    for name, ok in gates:
        print(f"{'PASS' if ok else 'FAIL'}: {name}")

    all_gates_passed = all(ok for _, ok in gates)
    report = {
        "config": cfg,
        "distributions": dist_report,
        "roundtrip_max_px": rt_max,
        "repeatability_ok": repeat_ok,
        "refiner": refiner_report,
        "versions": {"numpy": np.__version__, "cv2": cv2.__version__, "skimage": skimage.__version__},
        "backgrounds_sha1": hashlib.sha1("\n".join(bg_files).encode()).hexdigest(),
        "gates": {name: ok for name, ok in gates},
        "generator_fingerprint": generator_fingerprint(cfg, Path(_PROJ_DIR)),
        "all_gates_passed": all_gates_passed,
    }
    with open(out / "report.json", "w") as f:
        json.dump(report, f, indent=2, cls=_NpEnc)

    sys.exit(0 if all_gates_passed else 1)


if __name__ == "__main__":
    main()
