"""S6-S8 for one demo: hand cleanup, phases, box gap filling, box smoothing.

Order (deviates slightly from the plan's S6 -> S7 -> S8): the hand is cleaned and smoothed
first, because it does not depend on the box, and phases / grasp transform are better
estimated on a clean wrist. The box is smoothed last, per phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import savgol_filter

from ego2libero.fusion import bridge as br
from ego2libero.fusion import phases as ph
from ego2libero.fusion import signals as sg
from ego2libero.fusion import smooth as sm


@dataclass
class Fused:
    T_W_O: np.ndarray
    obj_source: np.ndarray
    T_W_H: np.ndarray
    joints_W: np.ndarray
    aperture: np.ndarray
    contact: np.ndarray
    box_speed: np.ndarray
    seg: ph.Segmentation
    grasp: br.GraspEstimate | None
    jumps: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def _runs_of_equal(x: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of each run of equal values (e.g. observed vs infilled hand frames)."""
    cut = np.flatnonzero(np.diff(x)) + 1
    edges = np.r_[0, cut, len(x)]
    return list(zip(edges[:-1], edges[1:]))


def _smooth_joints(J: np.ndarray, segments, window: int, order: int) -> np.ndarray:
    J = J.copy()
    ok = np.isfinite(J).all(axis=(1, 2))
    runs = []
    for a, b in segments:
        d = np.diff(np.r_[0, ok[a:b].astype(np.int8), 0])
        runs += [(a + s, a + e) for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]
    for s, e in runs:
        n = e - s
        w = min(window, n if n % 2 else n - 1)
        if w > order:
            J[s:e] = savgol_filter(J[s:e], w, order, axis=0)
    return J


def fuse(t, T_W_O_tags, T_W_H_raw, joints_raw, hand_source, dims, cfg) -> Fused:
    pc, bc, sc = cfg["phases"], cfg["bridge"], cfg["smooth"]
    n = len(t)
    stats = {}

    # hand: drop spikes, refill, smooth. Observed and HaWoR-infilled stretches are smoothed
    # separately: infilled frames off-screen can be wildly off and must not bleed into real ones.
    hand_runs = _runs_of_equal(np.asarray(hand_source))
    out_h = sm.hampel_outliers(T_W_H_raw, sc["hampel_window"], sc["hampel_nsigma"]) & (hand_source == 0)
    T_W_H = sm.smooth_track(sm.fill_by_interpolation(T_W_H_raw, t, out_h), hand_runs,
                            sc["savgol_window"], sc["savgol_order"])
    joints = _smooth_joints(joints_raw, hand_runs, sc["savgol_window"], sc["savgol_order"])
    stats["hand_outliers"] = int(out_h.sum())

    # box tags: spikes become missing (bridged below, like any other gap)
    T_O = T_W_O_tags.copy()
    out_o = sm.hampel_outliers(T_O, sc["hampel_window"], sc["hampel_nsigma"])
    T_O[out_o] = np.nan
    src = np.where(br.valid_mask(T_O), br.SRC_TAG, br.SRC_MISSING).astype(np.uint8)
    stats["box_outliers"] = int(out_o.sum())
    stats["box_tag_coverage"] = float((src == br.SRC_TAG).mean())

    # S6 phases from tag-only box motion and the clean hand
    speed = sg.robust_speed(T_O[:, :3, 3], t)
    contact = sg.contact_distance(joints, T_O, dims)
    seg = ph.segment(t, speed, contact, pc)

    # S7 bridging
    grasp, jumps = None, []
    on = seg.onsets
    if "grasp" in on:
        g0, g1 = on["grasp"], on.get("release", n)
        held = np.zeros(n, bool)
        held[g0:g1] = True
        T_O, src = br.interpolate_short_gaps(T_O, src, t, g0, g1, bc["max_interp_gap_s"])
        use = held & (hand_source == 0) & (src == br.SRC_TAG)
        if use.sum() >= 5:
            grasp = br.estimate_T_H_O(T_W_H, T_O, use, bc["drop_frac"])
            T_O, src, jumps = br.bridge_held(T_O, src, T_W_H, g0, g1, grasp.T_H_O, bc["blend_frames"])
    # everything else (resting box, the short fall after release, clip ends): interpolate / hold
    still_missing = ~br.valid_mask(T_O)
    if still_missing.any() and br.valid_mask(T_O).sum() >= 2:
        T_O = sm.fill_by_interpolation(T_O, t, still_missing)
        src[still_missing] = br.SRC_INTERPOLATED
    stats["box_missing_after_bridge"] = int((~br.valid_mask(T_O)).sum())

    # S8 box smoothing, per phase; resting stretches become their mean pose
    rest = np.zeros(n, bool)
    if "transport" in on:
        rest[:on["transport"]] = True
        if seg.landed is not None:
            rest[seg.landed:] = True
    else:
        rest[:] = True
    T_O = sm.replace_with_mean(T_O, rest & (np.arange(n) < on.get("transport", n)))
    if seg.landed is not None:
        T_O = sm.replace_with_mean(T_O, rest & (np.arange(n) >= seg.landed))
    if "transport" in on:
        moving = [(on["transport"], on.get("release", n))]
        T_O = sm.smooth_track(T_O, moving, sc["savgol_window"], sc["savgol_order"])

    return Fused(T_W_O=T_O, obj_source=src, T_W_H=T_W_H, joints_W=joints, aperture=sg.aperture(joints),
                 contact=sg.contact_distance(joints, T_O, dims), box_speed=speed, seg=seg, grasp=grasp,
                 jumps=jumps, stats=stats)
