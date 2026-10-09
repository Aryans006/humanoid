"""Outlier removal and smoothing of pose tracks (plan stage S8).

All functions take (N, 4, 4) tracks with NaN for missing frames and return new arrays.
Smoothing never crosses a segment boundary (``segments`` = list of [start, end) frame ranges,
e.g. the phases), because grasp onset and landing are real velocity discontinuities.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from ego2libero import se3


def _valid(T):
    return np.isfinite(T).all(axis=(1, 2))


def _rolling(fn, x: np.ndarray, half: int) -> np.ndarray:
    n = len(x)
    out = np.full(x.shape, np.nan)
    for i in range(n):
        w = x[max(0, i - half):i + half + 1]
        w = w[np.isfinite(w).all(axis=-1)] if w.ndim > 1 else w[np.isfinite(w)]
        if len(w):
            out[i] = fn(w)
    return out


def hampel_outliers(T: np.ndarray, window: int = 7, nsigma: float = 3.0,
                    min_trans_m: float = 0.003, min_rot_deg: float = 1.5) -> np.ndarray:
    """Frames whose position or orientation deviates from the local median by more than
    ``nsigma`` robust sigmas (1.4826 MAD) and by at least the given floors."""
    v = _valid(T)
    out = np.zeros(len(T), bool)
    if v.sum() < window:
        return out
    idx = np.flatnonzero(v)
    p = T[idx, :3, 3]
    half = window // 2
    med = _rolling(lambda w: np.median(w, 0), p, half)
    dev = np.linalg.norm(p - med, axis=1)
    mad = _rolling(np.median, dev, half)
    bad_t = (dev > nsigma * 1.4826 * np.maximum(mad, 1e-9)) & (dev > min_trans_m)
    # orientation: angle to the local chordal mean
    R = T[idx, :3, :3]
    ang = np.empty(len(idx))
    for k in range(len(idx)):
        a, b = max(0, k - half), min(len(idx), k + half + 1)
        ang[k] = np.degrees(se3.rotation_distance(se3.chordal_mean_rotation(R[a:b]), R[k]))
    mad_r = _rolling(np.median, ang, half)
    bad_r = (ang > nsigma * 1.4826 * np.maximum(mad_r, 1e-9)) & (ang > min_rot_deg)
    out[idx[bad_t | bad_r]] = True
    return out


def fill_by_interpolation(T: np.ndarray, t: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Replace ``frames`` (bool mask) by decoupled SE(3) interpolation of the other valid frames."""
    T = T.copy()
    ok = _valid(T) & ~frames
    if ok.sum() >= 2 and frames.any():
        T[frames] = se3.interpolate_poses(t[frames], t[ok], T[ok])
    return T


def _segments_of_valid(v: np.ndarray, a: int, b: int) -> list[tuple[int, int]]:
    d = np.diff(np.r_[0, v[a:b].astype(np.int8), 0])
    return [(a + s, a + e) for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]


def smooth_track(T: np.ndarray, segments: list[tuple[int, int]], window: int = 9, order: int = 2) -> np.ndarray:
    """Savitzky-Golay on translation and on rotation vectors relative to each run's mean
    rotation, applied separately to every contiguous valid run inside every segment."""
    T = T.copy()
    v = _valid(T)
    for a, b in segments:
        for s, e in _segments_of_valid(v, a, b):
            n = e - s
            w = min(window, n if n % 2 else n - 1)
            if w <= order:
                continue
            T[s:e, :3, 3] = savgol_filter(T[s:e, :3, 3], w, order, axis=0)
            R0 = se3.chordal_mean_rotation(T[s:e])
            rv = se3.so3_log(R0.T @ T[s:e, :3, :3])  # small angles about the run's mean
            T[s:e, :3, :3] = R0 @ se3.so3_exp(savgol_filter(rv, w, order, axis=0))
    return T


def replace_with_mean(T: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Set every valid frame in ``frames`` to their robust average pose (resting box)."""
    T = T.copy()
    sel = frames & _valid(T)
    if sel.any():
        T[frames] = se3.average_poses(T[sel])
    return T
