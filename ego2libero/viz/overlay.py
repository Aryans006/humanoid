"""Overlay video: board axes, box wireframe colored by source, detections, per-frame stats.

Drawn with OpenCV on the undistorted clip, so every projection uses K_u and zero distortion.
Later stages add the hand skeleton, phase and gripper state through the optional arguments.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from ego2libero import se3, video
from ego2libero.fusion.bridge import SRC_INTERPOLATED, SRC_MISSING, SRC_PROPAGATED, SRC_TAG

SRC_COLOR = {SRC_TAG: (60, 220, 60), SRC_PROPAGATED: (0, 160, 255), SRC_INTERPOLATED: (0, 230, 255)}
SRC_NAME = {SRC_TAG: "tag", SRC_PROPAGATED: "propagated", SRC_INTERPOLATED: "interpolated", SRC_MISSING: "missing"}
BOX_EDGES = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
              (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def _project(T_C_X, pts, K):
    p = se3.transform_points(T_C_X, pts)
    ok = p[:, 2] > 1e-3
    uv = (p[:, :2] / np.where(ok, p[:, 2], 1)[:, None]) @ K[:2, :2].T + K[:2, 2]
    return uv, ok


def _line(img, a, b, color, th=2):
    cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), color, th, cv2.LINE_AA)


def _text(img, lines, org=(12, 28), scale=0.6):
    for k, s in enumerate(lines):
        y = org[1] + int(26 * scale / 0.6) * k
        cv2.putText(img, s, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, s, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_frame(img, K, *, T_W_C=None, T_W_O=None, T_C_O=None, box_corners=None, obj_src=SRC_TAG,
               dets=None, joints_W=None, spots_W=None, info=None, axis_len=0.05):
    """Draw one frame in place. Any argument may be None / NaN to skip it."""
    if dets:
        for k, q in dets.items():
            cv2.polylines(img, [np.round(q).astype(np.int32)], True, (200, 120, 0), 1, cv2.LINE_AA)
            cv2.putText(img, str(k), tuple(np.round(q[0]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (200, 120, 0), 1, cv2.LINE_AA)
    cam_ok = T_W_C is not None and np.isfinite(T_W_C).all()
    T_C_W = se3.inv(T_W_C) if cam_ok else None
    if cam_ok:
        uv, ok = _project(T_C_W, np.array([[0, 0, 0], [axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]]), K)
        if ok.all():
            for j, col in zip((1, 2, 3), ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
                _line(img, uv[0], uv[j], col, 3)
        for s in spots_W or []:
            ring = np.c_[0.02 * np.cos(np.linspace(0, 2 * np.pi, 33)), 0.02 * np.sin(np.linspace(0, 2 * np.pi, 33)),
                         np.zeros(33)] + [s[0], s[1], 0]
            uv, ok = _project(T_C_W, ring, K)
            if ok.all():
                cv2.polylines(img, [np.round(uv).astype(np.int32)], True, (255, 0, 255), 2, cv2.LINE_AA)
    T_box = None
    if T_W_O is not None and np.isfinite(T_W_O).all() and cam_ok:
        T_box = T_C_W @ T_W_O
    elif T_C_O is not None and np.isfinite(T_C_O).all():
        T_box = T_C_O
    if T_box is not None and box_corners is not None and obj_src != SRC_MISSING:
        uv, ok = _project(T_box, box_corners, K)
        if ok.all():
            for a, b in BOX_EDGES:
                _line(img, uv[a], uv[b], SRC_COLOR.get(int(obj_src), (128, 128, 128)), 2)
    if joints_W is not None and cam_ok and np.isfinite(joints_W).all():
        uv, ok = _project(T_C_W, joints_W, K)
        for a, b in HAND_BONES:
            if ok[a] and ok[b]:
                _line(img, uv[a], uv[b], (255, 80, 200), 2)
    if info:
        _text(img, info)
    return img


def write_overlay(src_video, dst, K, *, T_W_C=None, T_W_O=None, T_C_O=None, box_corners=None, obj_src=None,
                  dets_per_frame=None, joints_W=None, spots_W=None, extra_info=None, scale: float = 1.0,
                  progress: bool = True) -> Path:
    """Render the overlay for a whole clip. Per-frame arrays are indexed by frame number;
    ``extra_info(i) -> list[str]`` adds text lines. ``scale`` < 1 writes a smaller video."""
    dst = Path(dst)
    info0 = video.probe(src_video)
    w, h = int(info0.width * scale), int(info0.height * scale)
    w, h = w - w % 2, h - h % 2
    Ks = K.copy().astype(float)
    Ks[:2] *= scale
    with video.VideoWriter(dst, w, h, crf=23) as out:
        for i, t, img in tqdm(video.iter_frames(src_video), desc=f"overlay {dst.parent.name}",
                              disable=not progress, unit="f"):
            if scale != 1.0:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            pick = lambda a: None if a is None else a[i]
            dets = None
            if dets_per_frame is not None:
                dets = {k: v * scale for k, v in dets_per_frame[i].items()}
            src = SRC_TAG if obj_src is None else int(obj_src[i])
            lines = [f"frame {i}  t {t:6.3f}s"] + (extra_info(i) if extra_info else [])
            draw_frame(img, Ks, T_W_C=pick(T_W_C), T_W_O=pick(T_W_O), T_C_O=pick(T_C_O), box_corners=box_corners,
                       obj_src=src, dets=dets, joints_W=pick(joints_W), spots_W=spots_W, info=lines)
            out.write(img, t)
    return dst
