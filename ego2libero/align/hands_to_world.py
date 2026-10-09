"""Hand track from HaWoR's world S into the table frame W (plan stage S5).

Per frame with a board pose, the hand hangs off the tag-derived camera, which is the more
accurate local anchor: ``T_W_H = T_W_C @ T_C_H`` with ``T_C_H = inv(T_S_C) @ T_S_H`` and its
translation scaled by the Sim(3) scale. Frames without a board pose fall back to the global
Sim(3) mapping of ``T_S_H``. Joints are transformed the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ego2libero import se3
from ego2libero.align.sim3 import Sim3, align_sim3, relative_scaled
from ego2libero.hands.hawor_io import INDEX_TIP, THUMB_TIP, HandTrack

HAND_OBSERVED, HAND_INFILLED = 0, 2  # plan's hand_source codes


@dataclass
class HandTrackW:
    T_W_H: np.ndarray  # (N, 4, 4)
    joints_W: np.ndarray  # (N, 21, 3)
    hand_source: np.ndarray  # (N,) uint8: 0 observed, 2 HaWoR-infilled
    anchored_on_board: np.ndarray  # (N,) bool: True = via T_W_C, False = via global Sim(3)
    mano_pose: np.ndarray  # (N, 48)
    mano_betas: np.ndarray  # (10,)
    sim: Sim3

    @property
    def aperture(self) -> np.ndarray:
        return np.linalg.norm(self.joints_W[:, THUMB_TIP] - self.joints_W[:, INDEX_TIP], axis=-1)

    def save(self, path) -> None:
        np.savez_compressed(path, T_W_H=self.T_W_H, joints_W=self.joints_W, hand_source=self.hand_source,
                            anchored_on_board=self.anchored_on_board, mano_pose=self.mano_pose,
                            mano_betas=self.mano_betas, sim_s=self.sim.s, sim_R=self.sim.R, sim_t=self.sim.t,
                            sim_rot_residual_deg=self.sim.rot_residual_deg,
                            sim_center_residual_m=self.sim.center_residual_m, sim_scale_fixed=self.sim.scale_fixed)

    @classmethod
    def load(cls, path) -> "HandTrackW":
        d = np.load(path)
        sim = Sim3(float(d["sim_s"]), d["sim_R"], d["sim_t"], n_frames=len(d["sim_rot_residual_deg"]),
                   scale_fixed=bool(d["sim_scale_fixed"]), rot_residual_deg=d["sim_rot_residual_deg"],
                   center_residual_m=d["sim_center_residual_m"])
        return cls(d["T_W_H"], d["joints_W"], d["hand_source"], d["anchored_on_board"], d["mano_pose"],
                   d["mano_betas"], sim)


def hands_to_world(track: HandTrack, T_W_C: np.ndarray, min_center_spread: float = 0.05) -> HandTrackW:
    if track.n != len(T_W_C):
        raise ValueError(f"HaWoR has {track.n} frames, the clip has {len(T_W_C)}")
    sim = align_sim3(T_W_C, track.T_S_C, min_center_spread)
    on_board = np.isfinite(T_W_C).all(axis=(1, 2))

    T_W_H = sim.apply_pose(track.T_S_H)
    T_C_H = relative_scaled(track.T_S_C[on_board], track.T_S_H[on_board], sim.s)
    T_W_H[on_board] = T_W_C[on_board] @ T_C_H

    joints_W = sim.apply_points(track.joints_S)
    T_C_S = se3.inv(track.T_S_C[on_board])
    joints_C = se3.transform_points(T_C_S, track.joints_S[on_board]) * sim.s  # scale about the camera
    joints_W[on_board] = se3.transform_points(T_W_C[on_board], joints_C)

    src = np.where(track.observed, HAND_OBSERVED, HAND_INFILLED).astype(np.uint8)
    return HandTrackW(T_W_H, joints_W, src, on_board, track.mano_pose, track.mano_betas, sim)
