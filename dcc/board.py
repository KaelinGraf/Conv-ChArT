"""Board geometry. get_board(bcfg) returns (cv2.aruco.CharucoBoard, squares per side) for a config's
board block (None gives the default 5x5 DICT_5X5_50 board); n_corners(bcfg) is the inner-corner count;
render_board(res, bcfg) returns the board image and its inner-corner pixel coordinates.
"""
import functools

import cv2
import numpy as np

_DEFAULT_BCFG = {"squares": [5, 5], "dictionary": "DICT_5X5_50", "marker_ratio": 0.7}


def _nx(bcfg):
    nx, ny = (bcfg or {}).get("squares", _DEFAULT_BCFG["squares"])
    assert nx == ny, f"rectangular boards are unsupported, got squares [{nx}, {ny}]"
    return nx


def n_corners(bcfg=None):
    return (_nx(bcfg) - 1) ** 2


@functools.lru_cache(maxsize=None)
def _build_board(nx, dictionary, marker_ratio, marker_id_offset=0):
    assert hasattr(cv2.aruco, dictionary), f"unknown cv2.aruco dictionary {dictionary!r}"
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    if marker_id_offset == 0:
        return cv2.aruco.CharucoBoard((nx, nx), 1.0, marker_ratio, d)
    n_markers = (nx * nx) // 2
    assert marker_id_offset + n_markers <= d.bytesList.shape[0], \
        f"{dictionary} has {d.bytesList.shape[0]} markers; offset {marker_id_offset} + {n_markers} exceeds it"
    ids = np.arange(marker_id_offset, marker_id_offset + n_markers, dtype=np.int32)
    return cv2.aruco.CharucoBoard((nx, nx), 1.0, marker_ratio, d, ids)


def get_board(bcfg=None):
    bcfg = bcfg or {}
    nx = _nx(bcfg)
    dictionary = bcfg.get("dictionary", _DEFAULT_BCFG["dictionary"])
    marker_ratio = bcfg.get("marker_ratio", _DEFAULT_BCFG["marker_ratio"])
    return _build_board(nx, dictionary, marker_ratio, int(bcfg.get("marker_id_offset", 0) or 0)), nx


BOARD = get_board()[0]


def render_board(res: int = 480, bcfg: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    board, nx = get_board(bcfg)
    if res % nx != 0:
        raise ValueError(f"res must be a multiple of {nx}, got {res}")
    sq = res // nx
    img = board.generateImage((res, res))
    corners = board.getChessboardCorners()[:, :2].astype(np.float64) * sq - 0.5
    return img, corners
