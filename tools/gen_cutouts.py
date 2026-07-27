"""tools/gen_cutouts.py -- OFFLINE SAM2 segmentation: builds a reusable RGBA
cutout bank from COCO images for dcc.synth's realistic object-occlusion mode
(dcc/synth.py's _apply_cutouts / load_cutouts). Run once, ahead of time; the
generator only ever reads the bank directory back.

RUN UNDER THE EOMT CONDA ENV PYTHON, NOT THE PROJECT'S MLWS ENV -- the two
envs carry different torch builds and must never be imported in the same
interpreter:

    /home/kaelin/anaconda3/envs/eomt/bin/python tools/gen_cutouts.py \\
        --coco /home/kaelin/datasets/coco/train2017 --out /home/kaelin/datasets/cutouts

This process imports only cv2/numpy/torch/sam2/stdlib -- never `dcc`.

Per image: SAM2AutomaticMaskGenerator (default params) proposes masks; each
is kept iff its area fraction falls in [--min-area-frac, --max-area-frac],
predicted_iou >= 0.85, stability_score >= 0.9, its bbox clears the image
border by >= 2px on every side (border-clipped objects have artificial
straight edges), and its solidity (mask-pixel-area / convex-hull-area) is
>= 0.4 (drops wire-frame/hollow junk -- see tests/test_generator.py). Up to
--max-per-image survivors are kept, by descending predicted_iou. Each kept
mask is cropped to its bbox and saved as an RGBA PNG (RGB = image crop, A =
mask*255) directly under --out, named %07d.png in save order.

GPU note: AMG runs ~1-2 s/image on a 5090, so a full --n-images 3000 sweep
takes ~1-1.5 h; smoke-test with a small --n-images first.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
SAM2_CKPT = "/home/kaelin/BinPicking/RealAnnotate/sam2_ckpts/sam2.1_hiera_base_plus.pt"


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coco", default="/home/kaelin/datasets/coco/train2017")
    p.add_argument("--out", default="/home/kaelin/datasets/cutouts")
    p.add_argument("--ckpt", default=SAM2_CKPT)
    p.add_argument("--sam-config", default=SAM2_CONFIG)
    p.add_argument("--n-images", type=int, default=3000)
    p.add_argument("--max-per-image", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-area-frac", type=float, default=0.005)
    p.add_argument("--max-area-frac", type=float, default=0.25)
    p.add_argument("--force", action="store_true", help="allow writing into a non-empty --out")
    return p


def _accept(ann, img_h, img_w, min_area_frac, max_area_frac):
    """SAM2 mask filter predicate: area-fraction bounds, confidence floors,
    border clearance (drops border-clipped masks -- artificial straight
    edges), and solidity (drops wire-frame/hollow masks: a thin outline has
    a small pixel-area relative to the convex hull it encloses)."""
    import cv2
    import numpy as np

    area_frac = ann["area"] / (img_h * img_w)
    if not (min_area_frac <= area_frac <= max_area_frac):
        return False
    if ann["predicted_iou"] < 0.85 or ann["stability_score"] < 0.9:
        return False
    x0, y0, bw, bh = ann["bbox"]
    if x0 < 2 or y0 < 2 or x0 + bw > img_w - 2 or y0 + bh > img_h - 2:
        return False
    mask_u8 = ann["segmentation"].astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    hull_area = cv2.contourArea(cv2.convexHull(np.vstack(contours)))
    if hull_area <= 0 or ann["area"] / hull_area < 0.4:
        return False
    return True


def main():
    args = build_parser().parse_args()
    out = Path(args.out)
    if out.is_dir() and any(out.iterdir()) and not args.force:
        print(f"refusing to write into non-empty --out {out} (pass --force to override)", file=sys.stderr)
        sys.exit(1)
    out.mkdir(parents=True, exist_ok=True)

    import cv2
    import numpy as np
    import torch
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    if not torch.cuda.is_available():
        print("CUDA not available -- SAM2AutomaticMaskGenerator requires a GPU", file=sys.stderr)
        sys.exit(1)

    all_files = sorted(str(f) for f in Path(args.coco).glob("*.jpg"))
    rng = np.random.default_rng(args.seed)
    n = min(args.n_images, len(all_files))
    chosen = [all_files[i] for i in rng.permutation(len(all_files))[:n]]

    model = build_sam2(args.sam_config, args.ckpt, device="cuda")
    mask_gen = SAM2AutomaticMaskGenerator(model)

    count = 0
    manifest_files = []
    for n_swept, fpath in enumerate(chosen):
        img_bgr = cv2.imread(fpath, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]
        anns = mask_gen.generate(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        kept = sorted((a for a in anns if _accept(a, h, w, args.min_area_frac, args.max_area_frac)),
                      key=lambda a: -a["predicted_iou"])

        for ann in kept[:args.max_per_image]:
            x0, y0, bw, bh = (int(round(v)) for v in ann["bbox"])
            x0, y0 = max(x0, 0), max(y0, 0)
            bw, bh = min(bw, w - x0), min(bh, h - y0)
            if bw <= 0 or bh <= 0:
                continue
            rgba = cv2.cvtColor(img_bgr[y0:y0 + bh, x0:x0 + bw], cv2.COLOR_BGR2BGRA)
            rgba[..., 3] = ann["segmentation"][y0:y0 + bh, x0:x0 + bw].astype(np.uint8) * 255
            fname = f"{count:07d}.png"
            cv2.imwrite(str(out / fname), rgba)
            manifest_files.append(fname)
            count += 1

        if (n_swept + 1) % 100 == 0:
            print(f"{n_swept + 1}/{len(chosen)} images swept, {count} cutouts kept")

    manifest_files.sort()
    manifest = {
        "params": vars(args),
        "n_images_swept": len(chosen),
        "n_cutouts": count,
        "files": manifest_files,
        "files_sha1": hashlib.sha1("\n".join(manifest_files).encode()).hexdigest(),
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {count} cutouts from {len(chosen)} images to {out}")


if __name__ == "__main__":
    main()
