"""tools/working_range.py -- physical standoff range for a given board, from the
MEASURED scale sweep rather than from the trained envelope.

The detector's competence is a function of s, the board-square size in INPUT pixels.
Turning that into a distance needs three physical facts and nothing else:

    z = f_px * S / s        S = square edge in metres, f_px = focal length in input px

The scale bands below are read from paper/results_rev6/03_robustness/{distance,
distance_extrap}.json at run time, so the range moves if the measurement does.

f_px is quoted at the NETWORK INPUT resolution, not the sensor's: the See3CAM_20CUG
centre-crops 1600x1200 and downsamples 2.5x to 640x480, so the effective pixel pitch
is 2.5 * 3.0 um = 7.5 um. Using the sensor's 3.0 um here would overstate range by 2.5x.
"""
import argparse
import json
from pathlib import Path

PITCH_UM = 3.0          # OV2311 physical pixel
RHO = 2.5               # sensor 1600x1200 -> network input 640x480
INPUT_W = 640

# Bands, defined on measured ID accuracy + recall. Kept as thresholds rather than
# hardcoded s values so a re-measurement moves the answer.
BANDS = [("core", 0.99, 0.95), ("usable", 0.96, 0.95), ("degraded", 0.80, 0.90)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--board-mm", type=float, default=120.0, help="board OUTER edge")
    p.add_argument("--squares", type=int, default=5)
    p.add_argument("--lenses", default="4,6,8,12,16", help="focal lengths in mm")
    p.add_argument("--sweeps", nargs="+",
                   default=["paper/results_rev6/03_robustness/distance.json",
                            "paper/results_rev6/03_robustness/distance_extrap.json"])
    p.add_argument("--out", default="paper/results_rev6/17_range/working_range.json")
    a = p.parse_args()

    pts = {}
    for f in a.sweeps:
        d = json.loads(Path(f).read_text())
        for st in d["steps"]:
            r = st["arms"]["refined"]
            # both sweeps carry s=12,16,64,128; keep the worse reading, never the flattering one
            s = st["value"]
            prev = pts.get(s)
            cur = (r["id_acc"], r["recall"])
            pts[s] = cur if prev is None else (min(prev[0], cur[0]), min(prev[1], cur[1]))

    S = a.board_mm / a.squares / 1000.0                      # square edge, metres
    f_px = {mm: float(mm) / (PITCH_UM * RHO / 1000.0) for mm in
            (float(x) for x in a.lenses.split(","))}

    # The TRAINED perspective envelope. dcc/synth.py:312 sets the virtual pinhole to
    # f = fov_scale * canvas_width, so synth.fov_scale IS the focal length in units of
    # frame widths -- a lens outside this band is an extrapolation, however good the
    # scale numbers look.
    import yaml
    fov_lo, fov_hi = yaml.safe_load(Path("configs/rev640.yaml").read_text())["synth"]["fov_scale"]
    f_trained = (fov_lo * INPUT_W, fov_hi * INPUT_W)
    mm_trained = tuple(f * PITCH_UM * RHO / 1000.0 for f in f_trained)

    # A band is the longest CONTIGUOUS run of measured scales that all pass. Taking the smallest and largest
    # passing s (as this did until 2026-10-10) spans failing steps in between -- on the 4.7M sweep s = 64
    # fails recall inside the "32-128" core band, and on the 882k release sweep 32, 64 and 128 fail it.
    bands, scales = {}, sorted(pts)
    for name, id_min, rec_min in BANDS:
        best, run = [], []
        for s in scales:
            i, r = pts[s]
            run = run + [s] if (i is not None and i >= id_min and r >= rec_min) else []
            best = run if len(run) > len(best) else best
        bands[name] = (best[0], best[-1]) if best else None

    # Close-range geometry limit: the whole board spans `squares * s` input px, so it
    # stops fitting the frame width beyond s = INPUT_W / squares regardless of accuracy.
    s_fit = INPUT_W / a.squares

    out = {"board_mm": a.board_mm, "squares": a.squares, "square_mm": 1000 * S,
           "pixel_pitch_um": PITCH_UM, "rho": RHO, "input_px_pitch_um": PITCH_UM * RHO,
           "s_full_board_in_frame": s_fit,
           "measured": {str(s): {"id_acc": i, "recall": r} for s, (i, r) in sorted(pts.items())},
           "bands_s": bands, "ranges_m": {}}

    import math
    out["trained_focal_px"] = f_trained
    out["trained_focal_mm"] = mm_trained
    out["trained_hfov_deg"] = [2 * math.degrees(math.atan(INPUT_W / (2 * f)))
                               for f in f_trained]

    print(f"board {a.board_mm:.0f} mm / {a.squares} squares -> square {1000*S:.1f} mm")
    print(f"input pixel pitch {PITCH_UM * RHO:.1f} um   full board fits frame at s <= {s_fit:.0f}")
    print(f"TRAINED lens envelope: f {f_trained[0]:.0f}-{f_trained[1]:.0f} px = "
          f"{mm_trained[0]:.1f}-{mm_trained[1]:.1f} mm, HFOV "
          f"{out['trained_hfov_deg'][0]:.0f}-{out['trained_hfov_deg'][1]:.0f} deg\n")
    for mm, fp in f_px.items():
        hfov = 2 * math.degrees(math.atan(INPUT_W / (2 * fp)))
        inb = f_trained[0] <= fp <= f_trained[1]
        row = {"f_px_input": fp, "hfov_deg": hfov, "in_trained_envelope": inb}
        print(f"f = {mm:g} mm  ({fp:.0f} px at input, HFOV {hfov:.0f} deg)"
              + ("" if inb else "   *** OUTSIDE TRAINED LENS ENVELOPE ***"))
        for name, (lo, hi) in ((k, v) for k, v in bands.items() if v):
            # near limit comes from the LARGER s, far limit from the smaller s
            near, far = fp * S / hi, fp * S / lo
            near_fit = fp * S / min(hi, s_fit)
            row[name] = {"s": [lo, hi], "near_m": near, "far_m": far,
                          "near_m_full_board": near_fit}
            print(f"    {name:9s} s {lo:>3}-{hi:<4} -> {near:.2f} m .. {far:.2f} m"
                  + ("" if hi <= s_fit else f"   (full board in frame from {near_fit:.2f} m)"))
        out["ranges_m"][f"{mm:g}mm"] = row
        print()

    o = Path(a.out); o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(out, indent=2))
    print(f"-> {o}")


if __name__ == "__main__":
    main()
