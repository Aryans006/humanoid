import cv2
import numpy as np
import pytest

from ego2libero import se3, video
from ego2libero.calib import calibrate as cal
from conftest import look_at
from synth import charuco_plane, render

SIZE = (960, 540)
K_TRUE = np.array([[760.0, 0, 483.0], [0, 760.0, 266.0], [0, 0, 1]])
DIST_TRUE = np.array([0.09, -0.16, 0.0008, -0.0006, 0.06])


@pytest.fixture(scope="module")
def board():
    b = cv2.aruco.CharucoBoard((7, 5), 0.035, 0.026, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100))
    b.setLegacyPattern(False)
    return b


@pytest.fixture(scope="module")
def calib_clip(tmp_path_factory, board):
    """A ChArUco board seen from 36 poses spread over the image, through a distorted lens."""
    rng = np.random.default_rng(3)
    plane = charuco_plane(board, np.eye(4))
    center = np.array([0.1225, 0.0875, 0])  # board center in its own frame (y down, z into board)
    path = tmp_path_factory.mktemp("cal") / "calib.mp4"
    with video.VideoWriter(path, *SIZE, crf=8) as w:
        k = 0
        for gx in np.linspace(-0.11, 0.11, 4):
            for gy in np.linspace(-0.07, 0.07, 3):
                for tilt in range(3):
                    d = rng.uniform(0.32, 0.45)
                    off = rng.normal(scale=0.12, size=2) * (tilt > 0)
                    eye = center + np.array([off[0], off[1], -d])  # camera on the printed (-z) side
                    T = look_at(eye, center + [gx, gy, 0], up=(0, -1, 0))
                    img = render([plane], T, K_TRUE, SIZE, dist=DIST_TRUE)
                    w.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), k / 30)
                    k += 1
    return path


def test_calibration_recovers_intrinsics(calib_clip, board):
    views, size = cal.collect_views(calib_clip, board, stride=1, min_corners=12)
    assert size == SIZE and len(views) >= 25
    sel = cal.select_views(views, size, n=30)
    c = cal.calibrate_views(sel, size)
    assert c.rms < 0.5, c.rms
    assert abs(c.K[0, 0] / K_TRUE[0, 0] - 1) < 0.01 and abs(c.K[1, 1] / K_TRUE[1, 1] - 1) < 0.01
    assert np.abs(c.K[:2, 2] - K_TRUE[:2, 2]).max() < 4
    # Distortion terms trade off against each other and the principal point against a small
    # rotation, so compare the distortion's *shape* (relative to the image center) over the area
    # the board covered. Uncovered corners are extrapolation; the capture guide says to sweep them.
    allp = np.concatenate([v.img for v in views])
    lo, hi = allp.min(0), allp.max(0)
    g = np.stack(np.meshgrid(np.linspace(lo[0], hi[0], 9), np.linspace(lo[1], hi[1], 7)), -1).reshape(-1, 1, 2)
    ctr = np.array([[[480.0, 270.0]]])
    a = cv2.undistortPoints(g, K_TRUE, DIST_TRUE, P=K_TRUE) - cv2.undistortPoints(ctr, K_TRUE, DIST_TRUE, P=K_TRUE)
    b = cv2.undistortPoints(g, c.K, c.dist, P=K_TRUE) - cv2.undistortPoints(ctr, c.K, c.dist, P=K_TRUE)
    # 1.5 px: includes the allowed focal error (0.15% x ~390 px radius ~ 0.6 px) at the coverage corners
    assert np.abs(a - b).max() < 1.5, np.abs(a - b).max()


def test_centered_K_u_is_valid_and_tight():
    K_u = cal.centered_K_u(K_TRUE, DIST_TRUE, SIZE)
    w, h = SIZE
    assert K_u[0, 0] == K_u[1, 1] and K_u[0, 2] == w / 2 and K_u[1, 2] == h / 2
    mx, my = cal.undistort_maps(K_TRUE, DIST_TRUE, K_u, SIZE)
    # every undistorted pixel samples inside the raw image ...
    assert mx.min() >= -0.5 and my.min() >= -0.5 and mx.max() <= w - 0.5 and my.max() <= h - 0.5
    # ... and the field of view is not wasted: some border pixel reaches the raw image edge
    reach = min(mx.min() + 0.5, my.min() + 0.5, w - 0.5 - mx.max(), h - 0.5 - my.max())
    assert reach < 3


def test_undistortion_matches_ideal_pinhole(board, tmp_path):
    """Undistorting a distorted render must reproduce a pinhole render with K_u."""
    plane = charuco_plane(board, np.eye(4))
    T = look_at([0.12, 0.05, -0.38], [0.1225, 0.0875, 0], up=(0, -1, 0))
    raw = render([plane], T, K_TRUE, SIZE, dist=DIST_TRUE)
    K_u = cal.centered_K_u(K_TRUE, DIST_TRUE, SIZE)
    mx, my = cal.undistort_maps(K_TRUE, DIST_TRUE, K_u, SIZE)
    und = cv2.remap(raw, mx, my, cv2.INTER_LINEAR)
    ideal = render([plane], T, K_u, SIZE)
    det = cv2.aruco.CharucoDetector(board)
    ca, ia, _, _ = det.detectBoard(und)
    cb, ib, _, _ = det.detectBoard(ideal)
    common = np.intersect1d(ia.ravel(), ib.ravel())
    assert len(common) >= 20
    pa = np.array([ca.reshape(-1, 2)[list(ia.ravel()).index(k)] for k in common])
    pb = np.array([cb.reshape(-1, 2)[list(ib.ravel()).index(k)] for k in common])
    assert np.abs(pa - pb).max() < 0.3


def test_undistort_clip_keeps_timestamps(tmp_path):
    src = tmp_path / "raw.mp4"
    ts = [0.0, 0.016, 0.034, 0.05, 0.083]
    with video.VideoWriter(src, 320, 240) as w:
        for t in ts:
            w.write(np.full((240, 320, 3), 128, np.uint8), t)
    K = np.array([[300.0, 0, 160], [0, 300, 120], [0, 0, 1]])
    out = cal.undistort_clip(src, tmp_path / "clip", K, np.zeros(5), K)
    np.testing.assert_allclose(np.load(tmp_path / "clip" / "timestamps.npy"), ts, atol=1e-4)
    np.testing.assert_allclose(video.read_timestamps(out), ts, atol=1e-4)


def test_warped_calibration_recovers_focal_on_bent_board():
    """A board bent by ~4 mm biases plain calibration; the warp-aware fit recovers the lens."""
    rng = np.random.default_rng(7)
    K = np.array([[1350.0, 0, 955], [0, 1350.0, 531], [0, 0, 1]])
    dist = np.array([0.01, -0.02, 0, 0, 0.0])
    gx, gy = np.meshgrid(np.arange(1, 7) * 0.035, np.arange(1, 5) * 0.035)
    xy = np.c_[gx.ravel(), gy.ravel()]
    u, v = ((xy - xy.mean(0)) / 0.1).T
    z = 0.002 * u * v + 0.0012 * u * u - 0.0008 * v**3  # smooth twist, ~4 mm across the sheet
    obj_true = np.c_[xy, z]
    views = []
    for k in range(30):
        T = look_at(xy.mean(0).tolist() + [0] + np.r_[rng.normal(scale=0.15, size=2), -0.45],
                    np.r_[xy.mean(0) + rng.normal(scale=0.04, size=2), 0], up=(0, -1, 0))
        rv, _ = cv2.Rodrigues(se3.inv(T)[:3, :3])
        img, _ = cv2.projectPoints(obj_true, rv, se3.inv(T)[:3, 3], K, dist)
        img = img.reshape(-1, 2) + rng.normal(scale=0.2, size=(len(xy), 2))
        views.append(cal.CalibView(k, np.c_[xy, np.zeros(len(xy))], img, 1.0))
    flat = cal.calibrate_views(views, (1920, 1080))
    warped, info = cal.calibrate_warped(views, (1920, 1080))
    assert abs(warped.K[0, 0] / 1350 - 1) < 0.005, warped.K[0, 0]
    assert np.abs(warped.K[:2, 2] - [955, 531]).max() < 5
    assert warped.rms < 0.35 < flat.rms, (warped.rms, flat.rms)
    assert 2.0 < info["board_warp_range_mm"] < 8.0, info
