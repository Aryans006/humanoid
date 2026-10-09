"""Camera pose from the table board (plan stage S2) -> T_W_C(t)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ego2libero import se3
from ego2libero.perception.box_pose import reproj_rms, rvec_tvec_to_T


def T_W_board(tag_size: float, gap: float, rows: int = 3) -> np.ndarray:
    """OpenCV GridBoard frame -> table frame W.

    OpenCV: origin top-left corner of the first tag, x right, y down the page, z into the
    table. W: origin bottom-left corner of the bottom-left tag, x right, y up the page, z up.
    """
    H = rows * tag_size + (rows - 1) * gap
    T = np.diag([1.0, -1.0, -1.0, 1.0])
    T[1, 3] = H
    return T


@dataclass
class BoardModel:
    board: cv2.aruco.GridBoard
    T_W_board: np.ndarray
    ids: set[int]

    @classmethod
    def from_config(cls, cfg: dict) -> "BoardModel":
        from ego2libero.config import require

        require(cfg, "tag_size_m", "gap_m", where="board.yaml")
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg["dictionary"]))
        n = cfg["cols"] * cfg["rows"]
        ids = np.arange(cfg["first_id"], cfg["first_id"] + n)
        b = cv2.aruco.GridBoard((cfg["cols"], cfg["rows"]), float(cfg["tag_size_m"]), float(cfg["gap_m"]), d, ids)
        return cls(b, T_W_board(cfg["tag_size_m"], cfg["gap_m"], cfg["rows"]), set(int(i) for i in ids))

    def object_points_W(self, corners_by_id: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Matched (object points in W, image points) for the board tags in a frame."""
        ids = [i for i in sorted(corners_by_id) if i in self.ids]
        if not ids:
            return np.zeros((0, 3)), np.zeros((0, 2))
        c = [np.asarray(corners_by_id[i], np.float32).reshape(1, 4, 2) for i in ids]
        obj, img = self.board.matchImagePoints(c, np.asarray(ids, np.int32))
        return se3.transform_points(self.T_W_board, obj.reshape(-1, 3).astype(np.float64)), img.reshape(-1, 2)


@dataclass
class CameraTrack:
    T_W_C: np.ndarray  # (N, 4, 4), NaN where missing
    rms_px: np.ndarray  # (N,), NaN where not solved
    n_tags: np.ndarray  # (N,)

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.T_W_C).all(axis=(1, 2))

    def save(self, path) -> None:
        np.savez_compressed(path, T_W_C=self.T_W_C, rms_px=self.rms_px, n_tags=self.n_tags)

    @classmethod
    def load(cls, path) -> "CameraTrack":
        d = np.load(path)
        return cls(d["T_W_C"], d["rms_px"], d["n_tags"])


def solve_board_pose(corners_by_id, K, model: BoardModel, dist=None, min_tags: int = 2,
                     max_rms_px: float = 2.0) -> tuple[np.ndarray | None, float, int]:
    """T_W_C for one frame, or None. Returns (T_W_C, rms_px, n_board_tags)."""
    dist = np.zeros(5) if dist is None else dist
    n = sum(1 for i in corners_by_id if i in model.ids)
    if n < min_tags:
        return None, np.nan, n
    obj, img = model.object_points_W(corners_by_id)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None, np.nan, n
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
    T_C_W = rvec_tvec_to_T(rvec, tvec)
    rms = reproj_rms(T_C_W, obj, img, K, dist)
    if rms > max_rms_px:
        return None, rms, n
    return se3.inv(T_C_W), rms, n


def track_camera(per_frame: list[dict], K, model: BoardModel, min_tags: int = 2, max_rms_px: float = 2.0) -> CameraTrack:
    N = len(per_frame)
    T = np.full((N, 4, 4), np.nan)
    rms = np.full(N, np.nan)
    ntag = np.zeros(N, dtype=np.int32)
    for i, c in enumerate(per_frame):
        T_i, r, n = solve_board_pose(c, K, model, min_tags=min_tags, max_rms_px=max_rms_px)
        rms[i], ntag[i] = r, n
        if T_i is not None:
            T[i] = T_i
    # z-up sanity: the camera must be above the table.
    ok = np.isfinite(T).all(axis=(1, 2))
    if ok.any() and np.median(T[ok, 2, 3]) < 0:
        raise RuntimeError("camera below the table in W: board frame convention is wrong")
    return CameraTrack(T, rms, ntag)


def focal_from_board(corners_by_id, model: BoardModel, center) -> float:
    """Focal length (px) implied by one view of the flat board, from its homography alone
    (square pixels, known principal point, no distortion: use undistorted detections).
    Returns NaN when the view is too close to fronto-parallel to constrain it."""
    obj, img = model.object_points_W(corners_by_id)
    if len(obj) < 8:
        return np.nan
    H, _ = cv2.findHomography(obj[:, :2], img - np.asarray(center, float))
    if H is None:
        return np.nan
    h1, h2 = H[:, 0], H[:, 1]
    est = []
    den = h1[2] * h2[2]  # orthogonality of the two rotation columns
    if abs(den) > 1e-12:
        est.append(-(h1[0] * h2[0] + h1[1] * h2[1]) / den)
    den = h1[2] ** 2 - h2[2] ** 2  # equal norms of the two rotation columns
    if abs(den) > 1e-12:
        est.append(-(h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2) / den)
    est = [e for e in est if e > 0]
    # each constraint is ill-conditioned for some tilts; keep the better-conditioned one
    return float(np.sqrt(est[int(np.argmax([abs(h1[2] * h2[2]), abs(h1[2] ** 2 - h2[2] ** 2)][:len(est)]))])) if est else np.nan
