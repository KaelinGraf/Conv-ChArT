# 31 — Print-ready release board

Kaelin, 2026-10-09: "what's the main fiducial marker we use (for the release model)? how can I print it to ensure
it's the right size?"

**The board** (`configs/abl_c2_wh_clsfocal_lam2.yaml`, `board:`; identical in `base_nodilate_s05.yaml` and every
release tier): a cv2 `CharucoBoard`, 5x5 squares, `DICT_5X5_50` markers with ids 0-11 in raster order, marker/square
ratio 0.7, cv2 4.10 non-legacy pattern (top-left square black, markers in the white squares), 16 inner corners.
`board.square_length_m` is null: the model has no physical size, it works on the square's apparent size in the
640x480 input (measured bands in `17_range/working_range.json`: core 32-128 px, usable 16-128, degraded 10-160).

**Files** (`tools/print_board.py --square-mm 24 --page A4`, defaults):
- `DICT_5X5_50_5x5_24mm.pdf` -- A4, the board at exactly 24 mm per square (120 mm edge, 16.8 mm markers), a
  100 mm scale bar and the dimensions printed under it. Print at 100% / actual size, never "fit to page".
- `DICT_5X5_50_5x5_24mm.png` -- the same raster (2835 px = 567 px per square) with a 600.075 dpi tag, in case the
  printer wants an image; at "actual size" it also comes out at 120 mm.

**Verified**: the PDF rasterised with `pdftoppm` at 300 dpi, `cv2.aruco.CharucoDetector` on the page finds markers
0-11 and all 16 charuco corners; the corner lattice measures 72.01 mm between inner corners 0 and 3 (3 squares;
expected 72.00), i.e. 24.003 mm per square and a 120.02 mm board edge; page size 595.276 x 841.89 pt (A4).

**Choosing the size**: `tools/working_range.py --board-mm <edge>` turns an edge length into standoff ranges for
the See3CAM_20CUG (OV2311, 640x480 input) per lens from the measured scale sweep; the banked 120 mm answer is
`17_range/working_range.json` (core band: 3 mm lens 0.08-0.34 m, 6 mm 0.15-0.60 m, 8 mm 0.20-0.80 m, 12 mm
0.30-1.20 m; range scales linearly with the edge). Whatever edge is printed, put `square_length_m: <edge/5 in m>`
in the deployment config's `board:` so `dcc/pipeline.py` reports translation in metres (default 1.0 = board squares).
