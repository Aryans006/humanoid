"""Synthetic renderer for end-to-end tests: textured planes (board, box faces, ChArUco) seen
by a pinhole camera with optional lens distortion. Pixel-center convention matches OpenCV."""

from __future__ import annotations

import cv2
import numpy as np

from ego2libero import se3

PPM = 4000.0  # texture pixels per meter (0.25 mm)


class Plane:
    """A texture on a plane. ``T_W_P`` maps plane coords (x right, y down in the texture,
    meters, origin at the texture's top-left pixel corner) into W."""

    def __init__(self, tex: np.ndarray, T_W_P: np.ndarray):
        self.tex, self.T_W_P = tex, T_W_P

    def homography(self, T_C_W, K):
        # texture pixel center (u, v) -> plane point ((u + .5)/PPM, (v + .5)/PPM, 0)
        S = np.array([[1 / PPM, 0, 0.5 / PPM], [0, 1 / PPM, 0.5 / PPM], [0, 0, 1]])
        T = T_C_W @ self.T_W_P
        return K @ np.c_[T[:3, 0], T[:3, 1], T[:3, 3]] @ S

    def depth_and_facing(self, T_C_W):
        T = T_C_W @ self.T_W_P
        c = T[:3, :3] @ np.array([self.tex.shape[1] / 2 / PPM, self.tex.shape[0] / 2 / PPM, 0]) + T[:3, 3]
        n = T[:3, 2]  # plane normal (texture viewed from -z side is the printed side)
        return c[2], float(n @ c) > 0


def board_planes(board_cfg, T_W_board_):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, board_cfg["dictionary"]))
    s, g = board_cfg["tag_size_m"], board_cfg["gap_m"]
    cols, rows = board_cfg["cols"], board_cfg["rows"]
    pad = 0.01
    W, H = cols * s + (cols - 1) * g + 2 * pad, rows * s + (rows - 1) * g + 2 * pad
    tex = np.full((int(round(H * PPM)), int(round(W * PPM))), 255, np.uint8)
    n = d.markerSize + 2
    for r in range(rows):
        for c in range(cols):
            mid = board_cfg["first_id"] + r * cols + c
            px = int(round(s * PPM / n)) * n
            m = cv2.aruco.generateImageMarker(d, mid, px, borderBits=1)
            m = cv2.resize(m, (int(round(s * PPM)),) * 2, interpolation=cv2.INTER_NEAREST)
            x0, y0 = int(round((pad + c * (s + g)) * PPM)), int(round((pad + r * (s + g)) * PPM))
            tex[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
    # board frame origin (top-left of tag 0) sits at (pad, pad) in the texture; board z points into
    # the table, i.e. the printed side faces -z, as Plane expects.
    T_board_P = se3.make_T(np.eye(3), [-pad, -pad, 0])
    return [Plane(tex, T_W_board_ @ T_board_P)]


def box_planes(model, T_W_O, dictionary="DICT_APRILTAG_36h11"):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    planes = []
    for tid, T_O_G in model.T_O_G.items():
        axis = int(np.argmax(np.abs(T_O_G[:3, 2])))
        face = float(min(np.delete(model.dims, axis)))
        fp, tp = int(round(face * PPM)), int(round(model.tag_size * PPM))
        tex = np.full((fp, fp), 255, np.uint8)
        m = cv2.resize(cv2.aruco.generateImageMarker(d, tid, d.markerSize + 2, borderBits=1), (tp, tp),
                       interpolation=cv2.INTER_NEAREST)
        o = (fp - tp) // 2
        tex[o:o + tp, o:o + tp] = m
        # texture (x right, y down) -> tag frame G (x right, y up, z out): flip y and z
        # (printed side faces -z of the plane frame => +z of G, outward)
        T_G_P = se3.make_T(np.diag([1.0, -1.0, -1.0]), [-face / 2, face / 2, 0])
        planes.append(Plane(tex, T_W_O @ T_O_G @ T_G_P))
    return planes


def charuco_plane(board: "cv2.aruco.CharucoBoard", T_W_B):
    sq = board.getSquareLength()
    nx, ny = board.getChessboardSize()
    size = (int(round(nx * sq * PPM)), int(round(ny * sq * PPM)))
    tex = board.generateImage(size, marginSize=0, borderBits=1)
    # ChArUco board frame: origin top-left, x right, y down, z into the board => printed side -z.
    return Plane(tex, T_W_B)


def render(planes, T_W_C, K, size, dist=None, background=110, ss=2):
    """Render at ``ss``x supersampling, then area-downsample. With ``dist``, the output is
    what a lens with those distortion coefficients would record."""
    w, h = size
    K2 = K.astype(float).copy()
    K2[:2, :2] *= ss
    K2[:2, 2] = K[:2, 2] * ss + (ss - 1) / 2
    T_C_W = se3.inv(T_W_C)
    img = np.full((h * ss, w * ss), background, np.uint8)
    vis = [(p.depth_and_facing(T_C_W), p) for p in planes]
    for (depth, facing), p in sorted(vis, key=lambda x: -x[0][0]):
        if not facing or depth <= 0:
            continue
        Hm = p.homography(T_C_W, K2)
        warped = cv2.warpPerspective(p.tex, Hm, (w * ss, h * ss), flags=cv2.INTER_LINEAR, borderValue=0)
        mask = cv2.warpPerspective(np.full(p.tex.shape, 255, np.uint8), Hm, (w * ss, h * ss),
                                   flags=cv2.INTER_NEAREST, borderValue=0)
        img[mask > 0] = warped[mask > 0]
    if dist is not None:
        # for each distorted output pixel, sample the ideal image at its undistorted location
        uu, vv = np.meshgrid(np.arange(w * ss, dtype=np.float32), np.arange(h * ss, dtype=np.float32))
        pts = np.stack([(uu - (ss - 1) / 2) / ss, (vv - (ss - 1) / 2) / ss], -1).reshape(-1, 1, 2)
        und = cv2.undistortPoints(pts.astype(np.float64), K, dist, P=K).reshape(h * ss, w * ss, 2)
        mx = (und[..., 0] * ss + (ss - 1) / 2).astype(np.float32)
        my = (und[..., 1] * ss + (ss - 1) / 2).astype(np.float32)
        img = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderValue=background)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
