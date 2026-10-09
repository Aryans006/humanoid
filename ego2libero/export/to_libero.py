"""Resampling to LIBERO's control rate and the fixed W -> LIBERO frame change (plan S9)."""

from __future__ import annotations

import numpy as np

from ego2libero import se3

POSE_KEYS = ("T_W_C", "T_W_O", "T_W_H")
LINEAR_KEYS = ("aperture", "board_reproj_px", "box_reproj_px")
DISCRETE_KEYS = ("phase", "gripper_closed", "cam_source", "obj_source", "hand_source")


def resample(arrays: dict, hz: float = 20.0) -> dict:
    """Arrays at native PTS -> uniform ``hz`` grid over the same span. Poses by decoupled SE(3)
    interpolation, joints / scalars linearly, discrete labels by the previous native frame."""
    t = arrays["t"]
    tq = np.arange(t[0], t[-1] + 1e-9, 1.0 / hz)
    out = {"t": tq}
    prev = np.clip(np.searchsorted(t, tq, side="right") - 1, 0, len(t) - 1)
    for k in POSE_KEYS:
        if k in arrays:
            T = arrays[k]
            ok = np.isfinite(T).all(axis=(1, 2))
            out[k] = se3.interpolate_poses(tq, t[ok], T[ok]) if ok.sum() >= 2 else np.full((len(tq), 4, 4), np.nan)
    if "joints_W" in arrays:
        J = arrays["joints_W"]
        out["joints_W"] = np.stack([np.stack([np.interp(tq, t, J[:, j, a]) for a in range(3)], -1)
                                    for j in range(J.shape[1])], 1)
    for k in LINEAR_KEYS:
        if k in arrays:
            x = arrays[k]
            ok = np.isfinite(x)
            out[k] = np.interp(tq, t[ok], x[ok]) if ok.sum() >= 2 else np.full(len(tq), np.nan)
    for k in DISCRETE_KEYS:
        if k in arrays:
            out[k] = arrays[k][prev]
    for k in ("mano_betas",):
        if k in arrays:
            out[k] = arrays[k]
    if "mano_pose" in arrays:
        out["mano_pose"] = arrays["mano_pose"][prev]
    return out


def to_libero(arrays: dict, T_L_W: np.ndarray) -> dict:
    """Same arrays expressed in the LIBERO world frame L (keys keep their W names + '_L' copies)."""
    T_L_W = np.asarray(T_L_W, float)
    out = dict(arrays)
    for k in POSE_KEYS:
        if k in arrays:
            out[k.replace("_W_", "_L_")] = T_L_W @ arrays[k]
    if "joints_W" in arrays:
        out["joints_L"] = arrays["joints_W"] @ T_L_W[:3, :3].T + T_L_W[:3, 3]
    for k in POSE_KEYS + ("joints_W",):
        out.pop(k, None)
    return out
