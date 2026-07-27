"""Conv-ChArT board geometry convention: cv2 CharucoBoard(5x5, 1.0, 0.7, DICT_5X5_50).

Pixel-centre convention: an integer pixel coordinate is that pixel's centre
(OpenCV convention); keypoints are (x, y) float64. Inner corner i (row = i//4,
col = i%4, top-left origin) renders at ((col+1)*SQ - 0.5, (row+1)*SQ - 0.5),
SQ = res // 5 the square side in pixels — square boundaries fall between
pixels, hence the -0.5 offset to the pixel-centre grid. cv2 lives only here.
"""
import cv2
import numpy as np

_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
BOARD = cv2.aruco.CharucoBoard((5, 5), 1.0, 0.7, _DICT)


def render_board(res: int = 480) -> tuple[np.ndarray, np.ndarray]:
    """Render the board at res x res; return (image uint8, corners float64 (16,2))."""
    if res % 5 != 0:
        raise ValueError(f"res must be a multiple of 5, got {res}")
    sq = res // 5
    img = BOARD.generateImage((res, res))
    corners = BOARD.getChessboardCorners()[:, :2].astype(np.float64) * sq - 0.5
    return img, corners
