"""Per-frame signals that drive phase segmentation (plan stage S6)."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from ego2libero import se3
from ego2libero.hands.hawor_io import INDEX_TIP, THUMB_TIP


def nan_median_filter(x: np.ndarray, size: int) -> np.ndarray:
    """Median filter that ignores NaNs (NaN stays NaN where the whole window is NaN)."""
    x = np.asarray(x, float)
    n = len(x)
    out = np.full(n, np.nan)
    h = size // 2
    for i in range(n):
        w = x[max(0, i - h):i + h + 1]
        w = w[np.isfinite(w)]
        if len(w):
            out[i] = np.median(w)
    return out


def aperture(joints_W: np.ndarray, smooth: int = 9) -> np.ndarray:
    """Thumb-tip to index-tip distance (m), median-filtered over ``smooth`` frames."""
    a = np.linalg.norm(joints_W[:, THUMB_TIP] - joints_W[:, INDEX_TIP], axis=-1)
    return median_filter(a, size=smooth, mode="nearest") if smooth > 1 else a


def point_box_distance(p_W: np.ndarray, T_W_O: np.ndarray, dims: np.ndarray) -> np.ndarray:
    """Signed distance (m) from points (N, 3) to oriented boxes (N, 4, 4) of size ``dims``:
    positive outside, negative inside. NaN where the box pose is missing."""
    q = np.einsum("nji,nj->ni", T_W_O[:, :3, :3], p_W - T_W_O[:, :3, 3])  # point in box frame
    d = np.abs(q) - np.asarray(dims) / 2
    outside = np.linalg.norm(np.maximum(d, 0), axis=-1)
    inside = np.minimum(d.max(-1), 0)
    return outside + inside


def contact_distance(joints_W: np.ndarray, T_W_O: np.ndarray, dims: np.ndarray) -> np.ndarray:
    """Distance from the thumb/index fingertip midpoint to the box surface (plan's contact signal)."""
    mid = (joints_W[:, THUMB_TIP] + joints_W[:, INDEX_TIP]) / 2
    return point_box_distance(mid, T_W_O, dims)


def robust_speed(p: np.ndarray, t: np.ndarray, half_window: float = 0.15) -> np.ndarray:
    """Speed (m/s) per frame from median positions in the half-windows before and after it.
    Frames with NaN positions are skipped; NaN where either half-window has < 2 samples.
    Frame-to-frame differences turn mm-level pose noise into cm/s of fake motion."""
    p = np.asarray(p, float)
    ok = np.isfinite(p).all(-1)
    v = np.full(len(t), np.nan)
    tt, pp = t[ok], p[ok]
    for i in range(len(t)):
        a = (tt >= t[i] - half_window) & (tt < t[i])
        b = (tt >= t[i]) & (tt <= t[i] + half_window)
        if a.sum() >= 2 and b.sum() >= 2:
            dtm = np.median(tt[b]) - np.median(tt[a])
            if dtm > 1e-6:
                v[i] = np.linalg.norm(np.median(pp[b], 0) - np.median(pp[a], 0)) / dtm
    return v


def angular_speed(T: np.ndarray, t: np.ndarray) -> np.ndarray:
    """deg/s between consecutive valid poses (NaN elsewhere)."""
    ok = np.isfinite(T).all(axis=(1, 2))
    w = np.full(len(t), np.nan)
    idx = np.flatnonzero(ok)
    if len(idx) > 1:
        dr = np.degrees(se3.rotation_distance(T[idx[1:]], T[idx[:-1]]))
        w[idx[1:]] = dr / np.maximum(np.diff(t[idx]), 1e-6)
    return w
