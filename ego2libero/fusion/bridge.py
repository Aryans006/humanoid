"""Occlusion bridging (plan stage S7), geometry core.

While the box is held it moves rigidly with the wrist, so a constant grasp transform
``T_H_O`` carries the box through tag dropouts: ``T_W_O(t) = T_W_H(t) @ T_H_O``.
Missing poses are NaN-filled (N, 4, 4) arrays; per-frame source codes are below.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ego2libero import se3

SRC_TAG, SRC_PROPAGATED, SRC_INTERPOLATED, SRC_MISSING = 0, 1, 2, 3


def valid_mask(T: np.ndarray) -> np.ndarray:
    return np.isfinite(T).all(axis=(-2, -1))


@dataclass
class GraspEstimate:
    T_H_O: np.ndarray
    frames: np.ndarray  # indices actually used (after outlier drop)
    trans_res_m: np.ndarray  # residual per used frame
    rot_res_deg: np.ndarray


def estimate_T_H_O(
    T_W_H: np.ndarray,
    T_W_O: np.ndarray,
    frames: np.ndarray,
    drop_frac: float = 0.2,
    trans_scale_m: float = 0.01,
    rot_scale_deg: float = 5.0,
) -> GraspEstimate:
    """Robust constant hand->box transform over ``frames`` (indices or bool mask).

    Frames lacking either pose are skipped. Residuals are scored as
    ``max(trans / trans_scale_m, rot / rot_scale_deg)``; the worst ``drop_frac`` are
    dropped before the final chordal-mean / median estimate.
    """
    idx = np.flatnonzero(frames) if np.asarray(frames).dtype == bool else np.asarray(frames)
    idx = idx[valid_mask(T_W_H[idx]) & valid_mask(T_W_O[idx])]
    if len(idx) == 0:
        raise ValueError("no frames with both hand and box pose")
    rel = se3.inv(T_W_H[idx]) @ T_W_O[idx]

    def residuals(T):
        return se3.translation_distance(rel, T), np.degrees(se3.rotation_distance(rel, T))

    est = se3.average_poses(rel)
    n_keep = max(1, int(np.ceil(len(idx) * (1 - drop_frac))))
    tr, rr = residuals(est)
    keep = np.argsort(np.maximum(tr / trans_scale_m, rr / rot_scale_deg))[:n_keep]
    keep.sort()
    est = se3.average_poses(rel[keep])
    tr, rr = residuals(est)
    return GraspEstimate(est, idx[keep], tr[keep], rr[keep])


def _ramp(T_from: np.ndarray, D: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """T_from[k] @ interp(I, D, alpha_k): a body-frame correction that fades in."""
    return np.stack([T_from[k] @ se3.interpolate_pair(np.eye(4), D, a) for k, a in enumerate(alphas)])


def bridge_held(
    T_W_O: np.ndarray,
    src: np.ndarray,
    T_W_H: np.ndarray,
    start: int,
    end: int,
    T_H_O: np.ndarray,
    blend_frames: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Fill non-tag frames in [start, end) from the wrist; return (T_W_O, src, jumps).

    At each edge of a gap the propagated track generally disagrees with the tag pose
    beside it. Instead of snapping, the last/first ``blend_frames`` propagated frames get
    a body-frame correction that ramps to the neighbouring tag pose. ``jumps`` logs the
    disagreement at every edge where tags reappear (``kind='reappear'``) or vanish.
    """
    T_W_O = T_W_O.copy()
    src = src.copy()
    prop = T_W_H @ T_H_O
    gap = np.zeros(len(src), dtype=bool)
    gap[start:end] = src[start:end] == SRC_MISSING  # never overwrite tag or interpolated frames
    gap &= valid_mask(T_W_H)
    T_W_O[gap] = prop[gap]
    src[gap] = SRC_PROPAGATED

    jumps = []
    for s, e in _runs(gap):
        for edge, k_tag, kind in ((s, s - 1, "vanish"), (e - 1, e, "reappear")):
            if not (0 <= k_tag < len(src)) or src[k_tag] != SRC_TAG:
                continue
            D = se3.inv(prop[k_tag]) @ T_W_O[k_tag]
            jumps.append({"frame": int(k_tag), "kind": kind,
                          "trans_m": float(np.linalg.norm(D[:3, 3])),
                          "rot_deg": float(np.degrees(se3.rotation_angle(D[:3, :3])))})
            n = min(blend_frames, e - s)
            if kind == "reappear":
                ks = np.arange(e - n, e)
                alphas = (ks - (e - n) + 1) / (n + 1)
            else:
                ks = np.arange(s, s + n)
                alphas = 1 - (ks - s + 1) / (n + 1)
            T_W_O[ks] = _ramp(T_W_O[ks], D, alphas)
    return T_W_O, src, jumps


def interpolate_short_gaps(T_W_O: np.ndarray, src: np.ndarray, t: np.ndarray, start: int, end: int,
                           max_gap_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Inside [start, end), fill gaps no longer than ``max_gap_s`` that have tag poses on both
    sides by SE(3) interpolation between them (source interpolated). On the pilots, short
    gaps are far more accurate this way than propagating from the wrist: a pinched box
    pivots 5-9 deg relative to the HaWoR wrist, about 2 cm at the box's 13 cm lever arm."""
    T_W_O = T_W_O.copy()
    src = src.copy()
    gap = np.zeros(len(src), bool)
    gap[start:end] = ~valid_mask(T_W_O[start:end])
    for s, e in _runs(gap):
        if s == 0 or e >= len(src) or not (valid_mask(T_W_O[s - 1]) and valid_mask(T_W_O[e])):
            continue
        if t[e] - t[s - 1] > max_gap_s:
            continue
        alpha = (t[s:e] - t[s - 1]) / (t[e] - t[s - 1])
        T_W_O[s:e] = se3.interpolate_pair(T_W_O[s - 1], T_W_O[e], alpha)
        src[s:e] = SRC_INTERPOLATED
    return T_W_O, src


def hold_resting(T_W_O: np.ndarray, src: np.ndarray, resting: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill missing poses in resting frames with the last valid pose (next, at clip start)."""
    T_W_O = T_W_O.copy()
    src = src.copy()
    ok = valid_mask(T_W_O)
    if not ok.any():
        return T_W_O, src
    last = np.flatnonzero(ok)[0]
    for k in range(len(src)):
        if ok[k]:
            last = k
        elif resting[k]:
            T_W_O[k] = T_W_O[last]
            src[k] = SRC_INTERPOLATED
    return T_W_O, src


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of each run of True."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))
