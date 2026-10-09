"""Acceptance gates from plan.md ("Validation and acceptance metrics"). Each returns a dict
with the measured values, the threshold and ``pass``; run_demo.py prints and stores them."""

from __future__ import annotations

import numpy as np

from ego2libero import se3
from ego2libero.perception.box_pose import BoxModel, static_window_pose, window_mask


def _gate(name, value, limit, ok, **extra):
    return {"gate": name, "value": value, "limit": limit, "pass": bool(ok), **extra}


def camera_jitter(T_W_C, g) -> list[dict]:
    _, s = static_window_pose(T_W_C, np.ones(len(T_W_C), bool))
    if s["n"] == 0:
        return [_gate("camera_jitter", None, None, False, note="no board poses")]
    return [_gate("camera_center_std_m", s["pos_std_m"], g["cam_jitter_center_m"], s["pos_std_m"] < g["cam_jitter_center_m"]),
            _gate("camera_rot_std_deg", s["rot_std_deg"], g["cam_jitter_rot_deg"], s["rot_std_deg"] < g["cam_jitter_rot_deg"])]


def box_jitter(T_W_O, g) -> list[dict]:
    _, s = static_window_pose(T_W_O, np.ones(len(T_W_O), bool))
    if s["n"] == 0:
        return [_gate("box_jitter", None, None, False, note="no box poses")]
    return [_gate("box_pos_std_m", s["pos_std_m"], g["box_jitter_pos_m"], s["pos_std_m"] < g["box_jitter_pos_m"]),
            _gate("box_rot_std_deg", s["rot_std_deg"], g["box_jitter_rot_deg"], s["rot_std_deg"] < g["box_jitter_rot_deg"])]


def lowest_corner_z(T_W_O: np.ndarray, model: BoxModel) -> float:
    return float(se3.transform_points(T_W_O, model.box_corners())[:, 2].min())


def box_resting(T_W_O, t, model: BoxModel, g, w) -> list[dict]:
    out = []
    for name, m in (("start", window_mask(t, t[0], t[0] + w["static_start_s"])),
                    ("end", window_mask(t, t[-1] - w["static_end_s"], t[-1]))):
        T, s = static_window_pose(T_W_O, m)
        if s["n"] == 0:
            out.append(_gate(f"box_rest_{name}_z_m", None, g["box_rest_z_m"], False, note="no box pose in window"))
            continue
        z = lowest_corner_z(T, model)
        out.append(_gate(f"box_rest_{name}_z_m", z, g["box_rest_z_m"], abs(z) <= g["box_rest_z_m"], n=s["n"]))
    return out


def robust_speed(p: np.ndarray, t: np.ndarray, half_window: float = 0.25) -> np.ndarray:
    """Speed (m/s) at each sample: distance between the median positions of the half-windows
    before and after it, over the time between their median timestamps. Frame-to-frame
    differences are useless here: 2 mm of pose noise at 30-60 fps reads as 6-12 cm/s."""
    n = len(t)
    v = np.full(n, np.nan)
    if n < 4:
        return v
    for i in range(n):
        lo, hi = t[i] - half_window, t[i] + half_window
        shift = max(0.0, t[0] - lo) - max(0.0, hi - t[-1])  # keep the window inside the clip
        lo, hi = lo + shift, hi + shift
        mid = (lo + hi) / 2
        a = (t >= lo) & (t < mid)
        b = (t >= mid) & (t <= hi)
        if a.sum() >= 2 and b.sum() >= 2:
            dtm = np.median(t[b]) - np.median(t[a])
            v[i] = np.linalg.norm(np.median(p[b], 0) - np.median(p[a], 0)) / max(dtm, 1e-6)
    return v


def still_segments(T_W_O, t, v_max=0.01, min_dur=0.8) -> list[np.ndarray]:
    """Frame-index arrays of segments where the box (tag poses) moves slower than v_max."""
    ok = np.isfinite(T_W_O).all(axis=(1, 2))
    idx = np.flatnonzero(ok)
    if len(idx) < 3:
        return []
    still = robust_speed(T_W_O[idx, :3, 3], t[idx]) < v_max
    segs, start = [], None
    for k, s in enumerate(np.r_[still, False]):
        if s and start is None:
            start = k
        elif not s and start is not None:
            seg = idx[start:k]
            if t[seg[-1]] - t[seg[0]] >= min_dur:
                segs.append(seg)
            start = None
    return segs


def known_positions(T_W_O, t, model: BoxModel, spots_W, g, min_still_s: float = 0.8,
                    placement: str = "center") -> list[dict]:
    """Box position while still vs the nearest ruler-measured spot.

    placement "center": the box center sits on the spot. "edge_on_line": the box sits on the
    +y side of the line y = spot_y with its edge touching it and its center at x = spot_x;
    the expected center is then offset by the box footprint's half-extent in y, taken from
    the estimated pose, so it follows the measured box size and any yaw."""
    out = []
    spots = np.asarray(spots_W, float)
    for seg in still_segments(T_W_O, t, min_dur=min_still_s):
        T = se3.average_poses(T_W_O[seg])
        xy = T[:2, 3].copy()
        if placement == "edge_on_line":
            half_y = xy[1] - se3.transform_points(T, model.box_corners())[:, 1].min()
            xy[1] -= half_y  # compare the box's near edge (not its center) with the line
        k = int(np.argmin(np.linalg.norm(spots - xy, axis=1)))
        err = float(np.linalg.norm(spots[k] - xy))
        out.append(_gate(f"known_spot_{'AB'[k] if k < 2 else k}", err, g["known_position_m"], err < g["known_position_m"],
                         box_xy_m=xy.round(4).tolist(), spot_m=spots[k].tolist(), t=[float(t[seg[0]]), float(t[seg[-1]])],
                         bottom_z_m=round(lowest_corner_z(T, model), 4)))
    if not out:
        out.append(_gate("known_spot", None, g["known_position_m"], False, note="no still segment with box pose"))
    return out


def focal_consistency(per_frame, board_model, K_u, tol: float = 0.05, max_frames: int = 300,
                      T_W_C=None, min_tilt_deg: float = 15.0) -> dict:
    """Median focal length implied by the flat table board vs K_u's. A large mismatch means the
    clip was recorded in a different zoom/lens mode than the calibration clip (Blackmagic's 1x
    is a 2x crop of 0.5x), or the calibration's focal length is off."""
    from ego2libero.perception.board_pose import focal_from_board

    idx = [i for i, c in enumerate(per_frame) if sum(k in board_model.ids for k in c) >= 4]
    if T_W_C is not None:
        # A board seen head-on carries no focal-length information (same degeneracy as a flat-on
        # calibration view), and the estimate blows up; keep only tilted views.
        tilt = np.degrees(np.arccos(np.clip(np.abs(T_W_C[:, 2, 2]), 0, 1)))
        idx = [i for i in idx if np.isfinite(tilt[i]) and tilt[i] >= min_tilt_deg]
        if not idx:
            med = float(np.nanmedian(tilt))
            return _gate("board_focal_ratio", None, tol, True,
                         note=f"not measurable: camera {med:.1f} deg from vertical (needs >= {min_tilt_deg:.0f})")
    idx = idx[:: max(1, len(idx) // max_frames)]
    f = np.array([focal_from_board(per_frame[i], board_model, K_u[:2, 2]) for i in idx])
    f = f[np.isfinite(f)]
    if len(f) < 10:
        return _gate("board_focal_ratio", None, tol, True, note=f"only {len(f)} usable board views")
    ratio = float(np.median(f) / K_u[0, 0])
    return _gate("board_focal_ratio", ratio, tol, abs(ratio - 1) < tol, n=int(len(f)),
                 iqr=[round(float(x / K_u[0, 0]), 4) for x in np.percentile(f, [25, 75])])


def sim3_sanity(sim, g) -> list[dict]:
    """HaWoR camera track vs the board camera track (plan: 0.8 < s < 1.2, median rotation
    residual < 3 deg). With a static camera the scale is unobservable and fixed to 1."""
    lo, hi = g["sim3_scale_range"]
    rot = float(np.median(sim.rot_residual_deg)) if sim.n_frames else None
    out = [_gate("sim3_rot_residual_deg", rot, g["sim3_rot_residual_deg"], rot is not None and rot < g["sim3_rot_residual_deg"])]
    if sim.scale_fixed:
        out.append(_gate("sim3_scale", None, [lo, hi], True, note="camera static: scale unobservable, fixed to 1"))
    else:
        out.append(_gate("sim3_scale", float(sim.s), [lo, hi], lo < sim.s < hi))
    return out


def fusion_gates(fused, t, g, labels: dict | None = None, fps: float | None = None) -> list[dict]:
    """M3 gates on a fused demo (ego2libero.fusion.fuse.Fused)."""
    out = []
    seg = fused.seg
    found = [k for k in ("grasp", "transport", "release", "retreat") if k in seg.onsets]
    out.append(_gate("phases_found", len(found), 4, len(found) == 4, onsets_s=seg.onset_times(t), notes=seg.notes))
    held = seg.gripper_closed
    if held.any():
        c = float(np.nanmedian(np.abs(fused.contact[held])))
        out.append(_gate("grasp_contact_m", c, g["grasp_contact_m"], c < g["grasp_contact_m"]))
    if fused.grasp is not None:
        tr, rr = float(np.median(fused.grasp.trans_res_m)), float(np.median(fused.grasp.rot_res_deg))
        out.append(_gate("rigidity_trans_m", tr, g["rigidity_trans_m"], tr < g["rigidity_trans_m"],
                         p90=round(float(np.percentile(fused.grasp.trans_res_m, 90)), 4)))
        out.append(_gate("rigidity_rot_deg", rr, g["rigidity_rot_deg"], rr < g["rigidity_rot_deg"],
                         p90=round(float(np.percentile(fused.grasp.rot_res_deg, 90)), 2)))
    re = [j["trans_m"] for j in fused.jumps if j["kind"] == "reappear"]
    jmax = max(re) if re else 0.0
    out.append(_gate("reanchor_jump_m", jmax, g["reanchor_jump_m"], jmax < g["reanchor_jump_m"], n=len(re)))
    miss = fused.stats["box_missing_after_bridge"]
    out.append(_gate("box_missing_after_bridge", miss, 0, miss == 0))
    if labels:
        fps = fps or 1.0 / np.median(np.diff(t))
        for k, f_lab in labels.items():
            if f_lab is not None and k in seg.onsets:
                err = int(seg.onsets[k] - f_lab)
                out.append(_gate(f"phase_timing_{k}_frames", err, g["phase_timing_frames"],
                                 abs(err) <= g["phase_timing_frames"], label=f_lab, found=int(seg.onsets[k])))
    return out


def coverage(valid: np.ndarray, g) -> dict:
    frac = float(valid.mean()) if len(valid) else 0.0
    return _gate("box_tag_coverage", frac, g["min_tag_coverage"], frac >= g["min_tag_coverage"])
