"""Hand -> Panda gripper targets in the LIBERO/robosuite world L (numpy only).

The gripper points straight down (the pilots pinch the cube's side faces from above) and only
its yaw is taken from the hand:

* free hand: the tool point (between the finger pads) follows the thumb/index fingertip
  midpoint; yaw closes the fingers along the thumb->index direction.
* held (``gripper_closed``): the tool point sits on the cube centre and closes across the
  pair of cube faces the person pinched, so it follows the accurate tag-based cube pose
  rather than HaWoR's few-cm depth error. The pinch axis is chosen once, at grasp onset.
* the two blend over ``blend_s`` before grasp onset and after release.

HaWoR frames that are infilled, or wildly off even when marked observed (seen on demo_001:
fingertips 1.2 m away, 0.6 m below the table), are replaced by the nearest good frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ego2libero import se3

THUMB_TIP, INDEX_TIP = 4, 8


@dataclass
class Targets:
    t: np.ndarray  # (K,)
    pos_L: np.ndarray  # (K, 3) tool point
    yaw: np.ndarray  # (K,) rad; closing axis = (cos, sin, 0)
    grip: np.ndarray  # (K,) -1 open, +1 closed
    T_L_O: np.ndarray  # (K, 4, 4) recorded cube pose in L
    held: np.ndarray  # (K,) bool
    hand_ok: np.ndarray  # (K,) bool, hand frame used as-is

    def R_L_tool(self, k: int) -> np.ndarray:
        """Gripper orientation: robosuite's top-down frame rotated by yaw about world z."""
        c, s = np.cos(self.yaw[k]), np.sin(self.yaw[k])
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return Rz @ np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


def _fill_from_nearest(x: np.ndarray, ok: np.ndarray) -> np.ndarray:
    x = x.copy()
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        raise ValueError("no usable hand frames")
    near = idx[np.clip(np.searchsorted(idx, np.arange(len(x))), 0, len(idx) - 1)]
    prev = idx[np.clip(np.searchsorted(idx, np.arange(len(x))) - 1, 0, len(idx) - 1)]
    pick = np.where(np.abs(prev - np.arange(len(x))) <= np.abs(near - np.arange(len(x))), prev, near)
    x[~ok] = x[pick[~ok]]
    return x


def _wrap_pi(a):
    return (a + np.pi / 2) % np.pi - np.pi / 2  # closing axis is a line: yaw is defined mod pi


def retarget(d: dict, T_L_W: np.ndarray, table_z: float, blend_s: float = 0.4,
             max_reach_from_cube_m: float = 0.45, min_clearance_m: float = 0.012) -> Targets:
    t = d["t"]
    n = len(t)
    T_L_O = T_L_W @ d["T_W_O"]
    J = d["joints_W"]
    tip = se3.transform_points(T_L_W, (J[:, THUMB_TIP] + J[:, INDEX_TIP]) / 2)
    pinch = (J[:, INDEX_TIP] - J[:, THUMB_TIP]) @ T_L_W[:3, :3].T
    cube = T_L_O[:, :3, 3]
    ok = (d["hand_source"] == 0) & np.isfinite(tip).all(1)
    ok &= (tip[:, 2] > table_z - 0.02) & (np.linalg.norm(tip[:, :2] - cube[:, :2], axis=1) < max_reach_from_cube_m)
    tip = _fill_from_nearest(tip, ok)
    pinch = _fill_from_nearest(pinch, ok)
    tip[:, 2] = np.maximum(tip[:, 2], table_z + min_clearance_m)

    held = np.asarray(d["gripper_closed"], bool)
    w = held.astype(float)
    fps = 1.0 / np.median(np.diff(t))
    nb = max(1, int(round(blend_s * fps)))
    on = np.flatnonzero(np.diff(np.r_[0, held.astype(np.int8)]) == 1)
    off = np.flatnonzero(np.diff(np.r_[held.astype(np.int8), 0]) == -1) + 1
    for a in on:
        k = np.arange(max(0, a - nb), a)
        w[k] = np.maximum(w[k], (k - (a - nb) + 1) / (nb + 1))
    for b in off:
        k = np.arange(b, min(n, b + nb))
        w[k] = np.maximum(w[k], 1 - (k - b + 1) / (nb + 1))

    # free-hand yaw from the thumb->index direction (horizontal part)
    yaw_hand = np.arctan2(pinch[:, 1], pinch[:, 0])
    # held yaw: the cube's horizontal face axis closest to the pinch at grasp onset, then
    # carried with the cube's own yaw
    yaw_held = yaw_hand.copy()
    for a, b in zip(on, off):
        R0 = T_L_O[a, :3, :3]
        horiz = [R0[:, i] for i in range(3) if abs(R0[2, i]) < 0.7]
        p = pinch[a:min(b, a + int(0.3 * fps)) + 1].mean(0)
        ax_i = max(range(len(horiz)), key=lambda i: abs(horiz[i][:2] @ p[:2]))
        axis_O = T_L_O[a, :3, :3].T @ horiz[ax_i]  # that face normal in the cube frame
        for k in range(a, b):
            v = T_L_O[k, :3, :3] @ axis_O
            yaw_held[k] = np.arctan2(v[1], v[0])

    # blend on the circle (mod pi), then unwrap so consecutive yaws never jump by ~pi
    d_yaw = _wrap_pi(yaw_held - yaw_hand)
    yaw = yaw_hand + w * d_yaw
    for k in range(1, n):
        yaw[k] = yaw[k - 1] + _wrap_pi(yaw[k] - yaw[k - 1])
    # The gripper is symmetric, so any yaw + k*pi is the same grasp. Pick the k that centres the
    # sequence on 0 (the Panda's top-down home yaw): demo_001 otherwise ran at 135-196 deg and
    # held the wrist joint at its limit, costing 3-6 cm of tracking.
    yaw -= np.pi * np.round(np.median(yaw) / np.pi)
    pos = (1 - w)[:, None] * tip + w[:, None] * cube
    return Targets(t=t, pos_L=pos, yaw=yaw, grip=np.where(held, 1.0, -1.0), T_L_O=T_L_O, held=held, hand_ok=ok)
