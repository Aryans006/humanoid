import cv2
import numpy as np

from ego2libero import se3
from ego2libero.perception.board_pose import T_W_board


def test_board_frame_puts_board_in_positive_quadrant():
    s, g = 0.055, 0.015
    board = cv2.aruco.GridBoard((4, 3), s, g, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
                                np.arange(12))
    T = T_W_board(s, g)
    assert se3.is_rotation(T[:3, :3])
    pts = {int(i): se3.transform_points(T, np.asarray(c, float)) for i, c in zip(board.getIds().ravel(), board.getObjPoints())}
    allp = np.concatenate(list(pts.values()))
    assert np.all(allp[:, :2] >= -1e-6) and np.allclose(allp[:, 2], 0, atol=1e-6)
    np.testing.assert_allclose(pts[8][3], [0, 0, 0], atol=1e-6)  # tag 8 bottom-left corner = origin
    np.testing.assert_allclose(allp.max(0)[:2], [4 * s + 3 * g, 3 * s + 2 * g], atol=1e-6)
    assert pts[0][:, 1].min() > pts[8][:, 1].max()  # tag 0 is "up" (+y) from tag 8
    # z up: board normal in W is +z, so a camera above the table has positive z.
    assert np.allclose(T[:3, :3] @ [0, 0, -1], [0, 0, 1])
