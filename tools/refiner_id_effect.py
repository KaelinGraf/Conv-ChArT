"""tools/refiner_id_effect.py -- WHY does refinement cost ID accuracy?

The class map is sampled AT THE REFINED POSITION (dcc/pipeline.py: read_ids on
(xy_sensor + 0.5) * r - 0.5), so refinement does not merely report where a corner
is -- it chooses where that corner's identity is read from. Two candidate routes:

  (A) MATCH LOSS   -- refinement pushes a corner past the match radius, so it is
                      never matched and cannot contribute an ID at all.
  (B) READOUT LOSS -- the corner still matches, but the shifted sample point
                      reads a lower p_id that falls under tau_id, so it is
                      demoted to unidentified (index -1).

Both are plausible from the code alone; only measurement separates them, and the
split decides what (if anything) is worth doing about it. Detections pair 1:1 by
index between the two arms because peaks/merge_close run BEFORE refinement, so
the same detection can be followed down both paths.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MATCH_PX = 4.0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/rev640.yaml")
    p.add_argument("--ckpt", default="runs/rev640_baseline/ckpt_0150000.pt")
    p.add_argument("--refiner-ckpt", default="runs/refiner_rev5/ckpt_0010000.pt")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=20260728)
    p.add_argument("--out", default="paper/results_rev6/02_refiner/refiner_id_effect.json")
    return p


def main():
    args = build_parser().parse_args()
    import cv2, numpy as np, torch, importlib.util
    cv2.setNumThreads(1); torch.set_num_threads(2)
    from dcc.dataset import load_config
    from dcc.model import DetectorNet, Refiner, detector_kwargs
    from dcc.pipeline import detect
    from dcc.synth import generate_sample, list_backgrounds
    spec = importlib.util.spec_from_file_location(
        "_td", str(Path(__file__).resolve().parents[1] / "tools" / "train_detector.py"))
    td = importlib.util.module_from_spec(spec); spec.loader.exec_module(td)

    cfg = load_config(args.config)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W, H = cfg["input_size"]
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = DetectorNet(H, W, **detector_kwargs(cfg)).to(dev).eval()
    model.load_state_dict(ck.get("ema", ck.get("model")))
    rk = torch.load(args.refiner_ckpt, map_location="cpu", weights_only=False)
    refiner = Refiner().to(dev).eval()
    refiner.load_state_dict(rk.get("ema", rk.get("model")))
    bg = list_backgrounds(cfg["synth"]["backgrounds"])

    # counts per GT corner, classified by what happened in each arm
    c = dict(both_ok=0, both_bad=0, only_coarse_id=0, only_ref_id=0,
             lost_match=0, lost_readout=0, gained=0, n_gt=0)
    pid_pairs = []
    for i in range(args.n):
        rng = np.random.default_rng([args.seed, 777, i])
        rec, _ = generate_sample(cfg, rng, bg, force_negative=False)
        vis = [k for k in rec["corners"] if k["visible"]]
        if not vis:
            continue
        gt = np.array([[k["x"], k["y"]] for k in vis], dtype=np.float64)
        gidx = [k["index"] for k in vis]
        rr = detect(rec["image"], model, refiner, K=None, dist=None, cfg=cfg)
        rc = detect(rec["image"], model, None, K=None, dist=None, cfg=cfg)

        def arm(res):
            d = res["corners"]
            if not d:
                return {}, {}
            xy = np.array([[k["x"], k["y"]] for k in d])
            out, pid = {}, {}
            for gi, di, _ in td._match_greedy(gt, xy, MATCH_PX):
                out[gi] = d[di]["index"]
                pid[gi] = d[di]["p_id"]
            return out, pid

        mr, pr = arm(rr)
        mc, pc = arm(rc)
        for gi in range(len(gt)):
            c["n_gt"] += 1
            ok_r = mr.get(gi) is not None and mr.get(gi, -1) == gidx[gi]
            ok_c = mc.get(gi) is not None and mc.get(gi, -1) == gidx[gi]
            if gi in pr and gi in pc:
                pid_pairs.append((pc[gi], pr[gi]))
            if ok_r and ok_c:
                c["both_ok"] += 1
            elif ok_c and not ok_r:
                c["only_coarse_id"] += 1
                # was it lost because the corner stopped matching, or because the
                # ID read at the shifted point fell under tau_id?
                if gi not in mr:
                    c["lost_match"] += 1
                else:
                    c["lost_readout"] += 1
            elif ok_r and not ok_c:
                c["only_ref_id"] += 1; c["gained"] += 1
            else:
                c["both_bad"] += 1

    p = np.array(pid_pairs)
    lost = c["only_coarse_id"]
    print(f"n_gt={c['n_gt']}  both_ok={c['both_ok']}  both_bad={c['both_bad']}")
    print(f"ID lost by refining : {lost}  ({100*lost/max(c['n_gt'],1):.2f}% of GT corners)")
    print(f"   (A) match loss   : {c['lost_match']}  ({100*c['lost_match']/max(lost,1):.1f}% of losses)")
    print(f"   (B) readout loss : {c['lost_readout']}  ({100*c['lost_readout']/max(lost,1):.1f}% of losses)")
    print(f"ID GAINED by refining: {c['gained']}  ({100*c['gained']/max(c['n_gt'],1):.2f}%)")
    print(f"net ID effect       : {c['gained']-lost:+d} corners "
          f"({100*(c['gained']-lost)/max(c['n_gt'],1):+.2f} pp)")
    if len(p):
        print(f"p_id on corners matched in BOTH arms (n={len(p)}): "
              f"coarse mean {p[:,0].mean():.4f} / refined mean {p[:,1].mean():.4f}  "
              f"(refined lower on {100*(p[:,1]<p[:,0]).mean():.1f}% of them)")
    out = {"counts": c, "p_id_coarse_mean": float(p[:, 0].mean()) if len(p) else None,
           "p_id_refined_mean": float(p[:, 1].mean()) if len(p) else None,
           "frac_pid_lower_when_refined": float((p[:, 1] < p[:, 0]).mean()) if len(p) else None,
           "provenance": {"ckpt": args.ckpt, "ckpt_step": ck.get("step"),
                           "refiner_ckpt": args.refiner_ckpt, "refiner_step": rk.get("step"),
                           "config": args.config, "n_frames": args.n, "match_px": MATCH_PX}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
