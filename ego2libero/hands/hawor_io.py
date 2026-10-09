"""Read hawor_driver.py output into pipeline conventions (main env; no HaWoR import).

CONVENTION STATUS: UNCONFIRMED. Per plan, HaWoR's camera/world axes must be confirmed by
reprojecting joints on 3 pilot clips before alignment code is trusted. Until then this
module assumes what HaWoR's code implies:
  * ``R_c2w, t_c2w`` from ``load_slam_cam`` map camera-frame points into S, i.e. they are
    ``T_S_C``, with an OpenCV camera (x right, y down, z forward). HaWoR's demo applies
    diag(1, -1, -1) to camera AND hands only for its viewer, so raw outputs are consistent.
  * MANO ``global_orient`` is the wrist (root joint) orientation in S, and joint 0 is the
    wrist position in S.
Replace this paragraph with the confirmed convention after the pilot reprojection check.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ego2libero import se3

THUMB_TIP, INDEX_TIP = 4, 8  # OpenPose hand order


@dataclass
class HandTrack:
    T_S_C: np.ndarray  # (N, 4, 4) HaWoR camera poses
    T_S_H: np.ndarray  # (N, 4, 4) wrist pose
    joints_S: np.ndarray  # (N, 21, 3)
    mano_pose: np.ndarray  # (N, 48) global_orient + 45 finger params, axis-angle
    mano_betas: np.ndarray  # (10,) median over observed frames
    observed: np.ndarray  # (N,) bool; False where HaWoR's infiller made the frame up
    img_focal: float
    img_center: np.ndarray
    image_size: np.ndarray
    slam_scale: float

    @property
    def n(self) -> int:
        return len(self.T_S_C)

    def aperture(self) -> np.ndarray:
        return np.linalg.norm(self.joints_S[:, THUMB_TIP] - self.joints_S[:, INDEX_TIP], axis=-1)


def load_hawor(path: str | Path, hand: str = "right") -> HandTrack:
    d = np.load(path, allow_pickle=False)
    T_S_C = se3.make_T(d["R_c2w"], d["t_c2w"])
    rot = d[f"{hand}_rot"]
    joints = d[f"{hand}_joints"]
    T_S_H = se3.make_T(se3.so3_exp(rot), joints[:, 0])
    observed = d[f"{hand}_observed"].astype(bool)
    betas = d[f"{hand}_betas"]
    betas_ref = betas[observed] if observed.any() else betas
    return HandTrack(
        T_S_C=T_S_C,
        T_S_H=T_S_H,
        joints_S=joints,
        mano_pose=np.concatenate([rot, d[f"{hand}_hand_pose"]], axis=1),
        mano_betas=np.median(betas_ref, axis=0),
        observed=observed,
        img_focal=float(d["img_focal"]),
        img_center=d["img_center"],
        image_size=d["image_size"],
        slam_scale=float(d["slam_scale"]),
    )


def check_intrinsics(track: HandTrack, K_u: np.ndarray, tol_px: float = 0.5) -> None:
    """HaWoR must have run with K_u: same focal, principal point at the image center."""
    K_u = np.asarray(K_u)
    w, h = track.image_size
    problems = []
    if abs(track.img_focal - K_u[0, 0]) > tol_px:
        problems.append(f"focal {track.img_focal} != K_u fx {K_u[0, 0]}")
    if abs(K_u[0, 0] - K_u[1, 1]) > tol_px:
        problems.append("K_u has fx != fy; HaWoR assumes square pixels")
    if np.abs(K_u[:2, 2] - [w / 2, h / 2]).max() > tol_px:
        problems.append(f"K_u principal point {K_u[:2, 2]} != image center {[w / 2, h / 2]}")
    if np.abs(track.img_center - [w / 2, h / 2]).max() > tol_px:
        problems.append(f"HaWoR SLAM center {track.img_center} != image center")
    if problems:
        raise ValueError("; ".join(problems))
