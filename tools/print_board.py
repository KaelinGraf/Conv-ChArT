"""tools/print_board.py -- print-ready PDF (+ PNG) of the board a config trains on, at an exact physical size.

The release model's board (configs/abl_c2_wh_clsfocal_lam2.yaml: 5x5 squares, DICT_5X5_50 markers 0-11, marker
ratio 0.7) has NO physical size: the detector works on the square's apparent size s in the 640x480 input (10-160 px
measured, core 32-128; paper/results_rev6/17_range/working_range.json), so the edge length is a deployment choice,
z = f_px * S / s -- pick it with tools/working_range.py --board-mm, and put the same value in the deployment config's
board.square_length_m so poses come out in metres (dcc/pipeline.py). This draws THE SAME cv2 board object the
generator renders (dcc.board.get_board), rastered with a whole number of pixels per square, placed on the page at
exactly --square-mm per square, with a 100 mm scale bar and the dimensions printed underneath: print at 100% /
"actual size" (never "fit to page") and check both with a ruler.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from PIL import Image
from reportlab.lib.pagesizes import A3, A4, letter
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from dcc.board import get_board

PAGES = {"A4": A4, "A3": A3, "letter": letter}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/abl_c2_wh_clsfocal_lam2.yaml", help="the 882k release config")
    p.add_argument("--square-mm", type=float, default=24.0, help="square edge on paper; 24 = the 120 mm board of 17_range")
    p.add_argument("--page", default="A4", choices=PAGES)
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--out", default=None, help="default paper/results_rev6/31_print_board/<dictionary>_<n>x<n>_<square>mm.pdf")
    a = p.parse_args()
    bcfg = yaml.safe_load(open(a.config))["board"]
    board, nx = get_board(bcfg)
    ids = board.getIds().ravel().tolist()
    board_mm, marker_mm = nx * a.square_mm, board.getMarkerLength() * a.square_mm   # the board's square length is 1.0
    sq_px = round(a.square_mm / 25.4 * a.dpi)                   # whole pixels per square: every edge lands on a pixel
    img = Image.fromarray(board.generateImage((nx * sq_px, nx * sq_px)))
    out = Path(a.out or f"paper/results_rev6/31_print_board/{bcfg['dictionary']}_{nx}x{nx}_{a.square_mm:g}mm.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    dpi = sq_px * 25.4 / a.square_mm
    img.save(out.with_suffix(".png"), dpi=(dpi, dpi))           # so the PNG also prints true at "actual size"
    W, H = PAGES[a.page]
    assert board_mm + 30 <= W / mm, f"{board_mm:g} mm board + 15 mm margins does not fit {a.page} ({W / mm:.0f} mm wide)"
    c = canvas.Canvas(str(out), pagesize=(W, H))
    x0, y0 = (W - board_mm * mm) / 2, H - 15 * mm - board_mm * mm
    c.drawImage(ImageReader(img), x0, y0, board_mm * mm, board_mm * mm)
    y = y0 - 12 * mm                                             # 100 mm scale bar with end ticks
    c.setLineWidth(0.5)
    for xa, ya, xb, yb in [(x0, y, x0 + 100 * mm, y), (x0, y - 2 * mm, x0, y + 2 * mm),
                           (x0 + 100 * mm, y - 2 * mm, x0 + 100 * mm, y + 2 * mm)]:
        c.line(xa, ya, xb, yb)
    c.setFont("Helvetica", 9)
    c.drawString(x0 + 102 * mm, y - 1 * mm, "100 mm scale bar")
    first = "black" if img.getpixel((sq_px // 2, sq_px // 2)) == 0 else "white"
    for i, line in enumerate([
        f"Conv-ChArT board, {a.config}",
        f"{nx}x{nx} squares, {bcfg['dictionary']} markers {ids[0]}-{ids[-1]}, marker/square ratio "
        f"{board.getMarkerLength():g}, top-left square {first}",
        f"square {a.square_mm:g} mm, marker {marker_mm:g} mm, board edge {board_mm:g} mm. Print at 100% / actual size, "
        f"never 'fit to page'.",
        f"Ruler check: board edge = {board_mm:g} mm, scale bar = 100 mm. Raster {sq_px} px per square ({dpi:.0f} dpi).",
    ]):
        c.drawString(15 * mm, y - (8 + 5 * i) * mm, line)
    c.save()
    print(f"-> {out} and {out.with_suffix('.png')}  ({a.page}; board {board_mm:g} mm, square {a.square_mm:g} mm, "
          f"marker {marker_mm:g} mm, {sq_px} px/square)")


if __name__ == "__main__":
    main()
