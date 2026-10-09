"""Similarity alignment of HaWoR's SLAM world S into the table frame W (plan stage S5).

The model is ``x_W = s * R_W_S @ x_S + t_W_S``. Rotation comes from camera orientations
(well conditioned even when the head barely translates); scale and translation from a
linear least-squares fit on camera centers with that rotation fixed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ego2libero import se3

log = logging.getLogger(__name__)


@dataclass
class Sim3:
    s: float
    R: np.ndarray  # R_W_S (3, 3)
    t: np.ndarray  # t_W_S (3,)
    # diagnostics
    n_frames: int = 0
    scale_fixed: bool = False
    rot_residual_deg: np.ndarray = field(default_factory=lambda: np.zeros(0))
    center_residual_m: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def apply_points(self, p_S: np.ndarray) -> np.ndarray:
        return self.s * np.asarray(p_S) @ self.R.T + self.t

    def apply_pose(self, T_S_X: np.ndarray) -> np.ndarray:
        """Map poses expressed in S to W. Orientation rotates; position scales then rotates."""
        T_S_X = np.asarray(T_S_X, dtype=np.float64)
        R = self.R @ T_S_X[..., :3, :3]
        t = self.apply_points(T_S_X[..., :3, 3])
        return se3.make_T(R, t)

    def summary(self) -> dict:
        return {
            "s": float(self.s),
            "R_W_S": self.R.tolist(),
            "t_W_S": self.t.tolist(),
            "n_frames": int(self.n_frames),
            "scale_fixed": bool(self.scale_fixed),
            "rot_residual_median_deg": float(np.median(self.rot_residual_deg)) if self.n_frames else None,
            "rot_residual_p90_deg": float(np.percentile(self.rot_residual_deg, 90)) if self.n_frames else None,
            "center_residual_rms_m": float(np.sqrt(np.mean(self.center_residual_m**2))) if self.n_frames else None,
        }


def align_sim3(
    T_W_C: np.ndarray,
    T_S_C: np.ndarray,
    min_center_spread: float = 0.05,
) -> Sim3:
    """Fit S->W from paired camera poses (N, 4, 4), using only frames where both exist.

    Rows with NaN in either input are ignored. If either track's camera centers spread less
    than ``min_center_spread`` (RMS distance from their mean), scale is ill-conditioned: fix
    s = 1 and log a warning. Checking W matters: with a static camera, SLAM drift can give the
    S track several cm of spurious motion (6.4 cm on demo_002 of 2026-10-04 vs 1.8 mm true),
    and a least-squares fit against a static W track then drives s to ~0.
    """
    T_W_C = np.asarray(T_W_C, dtype=np.float64)
    T_S_C = np.asarray(T_S_C, dtype=np.float64)
    ok = np.isfinite(T_W_C).all(axis=(1, 2)) & np.isfinite(T_S_C).all(axis=(1, 2))
    A, B = T_W_C[ok], T_S_C[ok]
    n = len(A)
    if n < 2:
        raise ValueError(f"align_sim3 needs >= 2 paired frames, got {n}")

    # Rotation: every frame gives an estimate R_W_C @ R_S_C^T of R_W_S.
    R_est = A[:, :3, :3] @ np.swapaxes(B[:, :3, :3], 1, 2)
    R = se3.chordal_mean_rotation(R_est)

    # Scale + translation with R fixed: c_W = s * (R c_S) + t.
    a = B[:, :3, 3] @ R.T
    b = A[:, :3, 3]
    a_mean, b_mean = a.mean(0), b.mean(0)
    da, db = a - a_mean, b - b_mean
    spread_S = float(np.sqrt((da**2).sum(1).mean()))
    spread_W = float(np.sqrt((db**2).sum(1).mean()))
    scale_fixed = min(spread_S, spread_W) < min_center_spread
    if scale_fixed:
        log.warning("camera-center spread W %.3f m / S %.3f m < %.3f m: fixing s = 1",
                    spread_W, spread_S, min_center_spread)
        s = 1.0
    else:
        s = float((da * db).sum() / (da**2).sum())
    t = b_mean - s * a_mean

    rot_res = np.degrees(se3.rotation_angle(np.swapaxes(R_est, 1, 2) @ R))
    ctr_res = np.linalg.norm(s * a + t - b, axis=1)
    return Sim3(s=s, R=R, t=t, n_frames=n, scale_fixed=scale_fixed,
                rot_residual_deg=rot_res, center_residual_m=ctr_res)


def relative_scaled(T_S_C: np.ndarray, T_S_X: np.ndarray, s: float) -> np.ndarray:
    """``T_C_X = inv(T_S_C) @ T_S_X`` with translation multiplied by s (S units -> meters).

    Used to hang a HaWoR-tracked frame X (e.g. the wrist) off the tag-derived camera pose:
    ``T_W_X = T_W_C @ relative_scaled(T_S_C, T_S_X, sim.s)``.
    """
    T_C_X = se3.inv(T_S_C) @ T_S_X
    T_C_X[..., :3, 3] *= s
    return T_C_X
