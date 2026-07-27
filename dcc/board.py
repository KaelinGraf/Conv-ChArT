"""Conv-ChArT board geometry convention: cv2 CharucoBoard((nx, ny), 1.0,
marker_ratio, dictionary), config-driven via get_board(bcfg) (bcfg is
cfg["board"]-shaped: squares [nx, ny], dictionary a cv2.aruco predefined-
dictionary name, marker_ratio). Only square boards (nx == ny) are supported
-- get_board/n_corners assert it, rectangular boards have no defined corner-
index convention here. A missing/None bcfg, or one missing these keys (e.g.
pipeline.py's PnP-only {"square_length_m": ...} mini-dicts), falls back to
the original 5x5 DICT_5X5_50 board, so every pre-retarget caller is
unaffected.

Pixel-centre convention: an integer pixel coordinate is that pixel's centre
(OpenCV convention); keypoints are (x, y) float64. Inner corner i (row =
i // (nx-1), col = i % (nx-1), top-left origin) renders at ((col+1)*SQ -
0.5, (row+1)*SQ - 0.5), SQ = res // nx the square side in pixels — square
boundaries fall between pixels, hence the -0.5 offset to the pixel-centre
grid. cv2 lives only here.
"""
import functools

import cv2
import numpy as np

_DEFAULT_BCFG = {"squares": [5, 5], "dictionary": "DICT_5X5_50", "marker_ratio": 0.7}


def _nx(bcfg):
    """Per-side square count, falling back through _DEFAULT_BCFG for a
    missing "squares" key (None bcfg, or a partial dict like a PnP-only
    {"square_length_m": ...} mini-dict). Asserts square (nx == ny):
    rectangular boards have no defined corner-index convention here."""
    nx, ny = (bcfg or {}).get("squares", _DEFAULT_BCFG["squares"])
    assert nx == ny, f"rectangular boards are unsupported, got squares [{nx}, {ny}]"
    return nx


def n_corners(bcfg=None):
    """Inner-corner count (nx-1)^2 -- the class head's channel count and the
    canonical-lattice size for bcfg's board."""
    return (_nx(bcfg) - 1) ** 2


@functools.lru_cache(maxsize=None)
def _build_board(nx, dictionary, marker_ratio):
    assert hasattr(cv2.aruco, dictionary), f"unknown cv2.aruco dictionary {dictionary!r}"
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    return cv2.aruco.CharucoBoard((nx, nx), 1.0, marker_ratio, d)


def get_board(bcfg=None):
    """(board, nx) for bcfg (cfg["board"]-shaped); a None/partial bcfg falls
    back through _DEFAULT_BCFG the same way n_corners does. Cached on the
    hashable (nx, dictionary, marker_ratio) triple, since bcfg itself (a
    dict) isn't hashable."""
    bcfg = bcfg or {}
    nx = _nx(bcfg)
    dictionary = bcfg.get("dictionary", _DEFAULT_BCFG["dictionary"])
    marker_ratio = bcfg.get("marker_ratio", _DEFAULT_BCFG["marker_ratio"])
    return _build_board(nx, dictionary, marker_ratio), nx


BOARD = get_board()[0]  # default 5x5 board -- kept for direct-BOARD callers (tests/test_synth.py)


def render_board(res: int = 480, bcfg: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Render the board at res x res; return (image uint8, corners float64
    ((nx-1)^2, 2)). bcfg=None reproduces the original 5x5 board."""
    board, nx = get_board(bcfg)
    if res % nx != 0:
        raise ValueError(f"res must be a multiple of {nx}, got {res}")
    sq = res // nx
    img = board.generateImage((res, res))
    corners = board.getChessboardCorners()[:, :2].astype(np.float64) * sq - 0.5
    return img, corners
