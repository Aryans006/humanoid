"""Phone intrinsics from the ChArUco clip, and undistortion of every clip (plan stage S0).

  uv run python -m ego2libero.calib.calibrate data/raw/<session>/calib.mp4

writes K, dist, rms and K_u into configs/camera.yaml and a report next to the clip's
interim folder. K_u has fx == fy and the principal point exactly at the image center,
because HaWoR assumes both; its focal length is chosen so the undistorted image has no
invalid (black) border.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from ego2libero import video
from ego2libero.config import CONFIG_DIR, load_yaml

log = logging.getLogger(__name__)


def charuco_board(cfg: dict | None = None) -> cv2.aruco.CharucoBoard:
    c = (cfg or load_yaml("camera"))["charuco"]
    if c.get("square_m") is None or c.get("marker_m") is None:
        raise ValueError("camera.yaml charuco.square_m / marker_m must be measured")
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, c["dictionary"]))
    b = cv2.aruco.CharucoBoard(tuple(c["squares"]), float(c["square_m"]), float(c["marker_m"]), d)
    b.setLegacyPattern(bool(c.get("legacy_pattern", False)))
    return b


@dataclass
class CalibView:
    frame: int
    obj: np.ndarray  # (k, 3)
    img: np.ndarray  # (k, 2)
    sharpness: float
    motion: float = np.nan  # median corner motion to the next frame, px/frame


def _corner_motion(a_obj, a_img, b_obj, b_img) -> float:
    """Median image displacement of the ChArUco corners two detections share."""
    ka = {tuple(np.round(o, 6)): p for o, p in zip(a_obj, a_img)}
    d = [np.linalg.norm(p - ka[tuple(np.round(o, 6))]) for o, p in zip(b_obj, b_img) if tuple(np.round(o, 6)) in ka]
    return float(np.median(d)) if len(d) >= 4 else np.nan


def collect_views(path, board, stride: int = 3, min_corners: int = 12) -> tuple[list[CalibView], tuple[int, int]]:
    """Detect the board every ``stride`` frames; also detect the frame right after each hit to
    measure how fast the board moves in the image (blur and rolling-shutter skew grow with it)."""
    det = cv2.aruco.CharucoDetector(board)
    views, size = [], None
    for i, _, gray in video.iter_frames(path, fmt="gray"):
        size = gray.shape[::-1]
        if stride > 1 and i % stride == 1 and views and views[-1].frame == i - 1:
            ch_c, ch_ids, _, _ = det.detectBoard(gray)
            if ch_ids is not None and len(ch_ids) >= 4:
                obj, img = board.matchImagePoints(ch_c, ch_ids)
                views[-1].motion = _corner_motion(views[-1].obj, views[-1].img, obj.reshape(-1, 3), img.reshape(-1, 2))
            continue
        if i % stride:
            continue
        ch_c, ch_ids, _, _ = det.detectBoard(gray)
        if ch_ids is None or len(ch_ids) < min_corners:
            continue
        obj, img = board.matchImagePoints(ch_c, ch_ids)
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        views.append(CalibView(i, obj.reshape(-1, 3), img.reshape(-1, 2), sharp))
    if size is None:
        raise ValueError(f"{path}: no frames")
    return views, size


def select_views(views: list[CalibView], size, n: int = 40, max_motion_px_1080p: float = 1.5) -> list[CalibView]:
    """Still, sharp frames spread over position, scale and tilt. Still = board moves less than
    ``max_motion_px_1080p`` px/frame (scaled to the image width); if fewer than 15 frames are
    that still, the slowest half is used. Then sharpness >= median of those."""
    if not views:
        return []
    w, h = size
    lim = max_motion_px_1080p * w / 1920
    mot = np.array([v.motion for v in views])
    if np.isfinite(mot).any():
        still = [v for v in views if np.isfinite(v.motion) and v.motion <= lim]
        if len(still) < 15:
            cut = np.nanmedian(mot)
            still = [v for v in views if np.isfinite(v.motion) and v.motion <= cut]
        views = still or views
    med = np.median([v.sharpness for v in views])
    sharp = [v for v in views if v.sharpness >= med] or views

    def feat(v):
        p = v.img
        span = p.max(0) - p.min(0)
        # perspective cue: ratio of the corner cloud's principal axes (tilt shows as anisotropy)
        ev = np.linalg.eigvalsh(np.cov(p.T)) if len(p) > 2 else np.ones(2)
        return np.array([p[:, 0].mean() / w, p[:, 1].mean() / h, span[0] / w, span[1] / h,
                         np.sqrt(ev[0] / max(ev[1], 1e-9))])

    F = np.stack([feat(v) for v in sharp])
    chosen = [int(np.argmax([len(v.img) for v in sharp]))]
    d = np.linalg.norm(F - F[chosen[0]], axis=1)
    while len(chosen) < min(n, len(sharp)):
        k = int(np.argmax(d))
        chosen.append(k)
        d = np.minimum(d, np.linalg.norm(F - F[k], axis=1))
    return [sharp[k] for k in sorted(chosen)]


@dataclass
class Calibration:
    K: np.ndarray
    dist: np.ndarray
    rms: float
    image_size: tuple[int, int]
    per_view_rms: np.ndarray
    frames: list[int]


def calibrate_views(views: list[CalibView], size) -> Calibration:
    if len(views) < 8:
        raise ValueError(f"only {len(views)} usable calibration views (need >= 8); record a longer, slower calib clip")
    obj = [v.obj.astype(np.float32) for v in views]
    img = [v.img.astype(np.float32) for v in views]
    rms, K, dist, rvecs, tvecs, _, _, per_view = cv2.calibrateCameraExtended(obj, img, size, None, None)
    return Calibration(K=K, dist=dist.ravel()[:5], rms=float(rms), image_size=tuple(size),
                       per_view_rms=per_view.ravel(), frames=[v.frame for v in views])


def _warp_basis(xy: np.ndarray, center: np.ndarray, L: float = 0.1) -> np.ndarray:
    """Smooth out-of-plane shape terms (quadratic + cubic). Constant and linear terms are
    omitted: they are a rigid motion of the board, already covered by each view's pose."""
    u, v = ((xy - center) / L).T
    return np.stack([u * u, u * v, v * v, u**3, u * u * v, u * v * v, v**3], 1)


def calibrate_warped(views: list[CalibView], size, K_init: np.ndarray | None = None) -> tuple[Calibration, dict]:
    """Joint calibration for a target that is not quite flat (paper on a folder).

    The lens is constrained to what a phone lens physically is: fx == fy, no tangential
    distortion, radial k1, k2, k3. The board gets one smooth out-of-plane shape shared by all
    views. OpenCV's calibrateCameraRO instead frees every corner, which on a hand-held sheet
    traded board shape against an implausible lens (principal point 80 px off, strong
    tangential terms). Only views that see every corner are used."""
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    n_all = max(len(v.obj) for v in views)
    full = [v for v in views if len(v.obj) == n_all]
    if len(full) < 8:
        raise ValueError(f"only {len(full)} views see the whole board (need >= 8)")
    ref = full[0].obj
    keys = {tuple(np.round(o, 6)): k for k, o in enumerate(ref)}
    imgs = np.stack([v.img[np.argsort([keys[tuple(np.round(o, 6))] for o in v.obj])] for v in full]).astype(np.float64)
    xy = ref[:, :2].astype(np.float64)
    B = _warp_basis(xy, xy.mean(0))
    n, m, nw = len(full), len(xy), B.shape[1]
    w, h = size
    K0 = K_init if K_init is not None else cv2.calibrateCameraExtended(
        [ref.astype(np.float32)] * n, list(imgs.astype(np.float32)), size, None, None,
        flags=cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST)[1]
    poses = []
    for im in imgs:
        _, rv, tv = cv2.solvePnP(np.c_[xy, np.zeros(m)], im, K0, None)
        poses.append(np.r_[rv.ravel(), tv.ravel()])
    x0 = np.r_[K0[0, 0], K0[0, 2], K0[1, 2], 0.0, 0.0, 0.0, np.zeros(nw), np.ravel(poses)]

    def project(p):
        f, cx, cy, k1, k2, k3 = p[:6]
        P = np.c_[xy, B @ p[6:6 + nw]]
        out = np.empty((n, m, 2))
        for k, q in enumerate(p[6 + nw:].reshape(n, 6)):
            R, _ = cv2.Rodrigues(q[:3])
            X = P @ R.T + q[3:]
            x, y = X[:, 0] / X[:, 2], X[:, 1] / X[:, 2]
            r2 = x * x + y * y
            s = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2**3
            out[k] = np.c_[f * x * s + cx, f * y * s + cy]
        return out

    J = lil_matrix((n * m * 2, len(x0)), dtype=int)
    for k in range(n):
        J[k * m * 2:(k + 1) * m * 2, :6 + nw] = 1
        J[k * m * 2:(k + 1) * m * 2, 6 + nw + 6 * k:12 + nw + 6 * k] = 1
    sol = least_squares(lambda p: (project(p) - imgs).ravel(), x0, jac_sparsity=J, x_scale="jac",
                        loss="soft_l1", f_scale=2.0)
    p = sol.x
    res = project(p) - imgs
    K = np.array([[p[0], 0, p[1]], [0, p[0], p[2]], [0, 0, 1]])
    dist = np.array([p[3], p[4], 0.0, 0.0, p[5]])
    z = B @ p[6:6 + nw]
    cal = Calibration(K=K, dist=dist, rms=float(np.sqrt((res**2).sum(-1).mean())), image_size=tuple(size),
                      per_view_rms=np.sqrt((res**2).sum(-1).mean(-1)), frames=[v.frame for v in full])
    return cal, {"board_warp_range_mm": float(1000 * (z.max() - z.min())), "views": n}


def centered_K_u(K: np.ndarray, dist: np.ndarray, size, n_boundary: int = 400) -> np.ndarray:
    """Pinhole K_u with principal point at the image center and fx == fy, as large a field of
    view as possible without invalid pixels in the undistorted image."""
    w, h = size
    t = np.linspace(0, 1, n_boundary, endpoint=False)
    edge = np.concatenate([np.c_[t * (w - 1), np.zeros_like(t)], np.c_[np.full_like(t, w - 1), t * (h - 1)],
                           np.c_[(1 - t) * (w - 1), np.full_like(t, h - 1)], np.c_[np.zeros_like(t), (1 - t) * (h - 1)]])
    xn = cv2.undistortPoints(edge.reshape(-1, 1, 2).astype(np.float64), K, dist).reshape(-1, 2)
    # Largest centered rectangle with the image's aspect inside the undistorted boundary:
    # each boundary point limits the half-width a to max(|x|, |y| * w / h).
    # The center is (w/2, h/2), HaWoR's convention (OpenCV's pixel-center convention would give
    # (w-1)/2); our own geometry accepts any principal point, so match HaWoR exactly.
    a = np.min(np.maximum(np.abs(xn[:, 0]), np.abs(xn[:, 1]) * w / h))
    f = (w / 2) / a
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])


def undistort_maps(K, dist, K_u, size):
    return cv2.initUndistortRectifyMap(K, dist, None, K_u, tuple(size), cv2.CV_32FC1)


def check_same_camera_mode(clip) -> list[str]:
    """Warn when a clip was not recorded like the calibration clip (lens, resolution, fps).
    A zoom change (e.g. Blackmagic's 1x, which is a 2x crop, vs 0.5x, the full sensor)
    silently invalidates the intrinsics; the field-of-view gate in run_demo catches what
    metadata cannot."""
    try:
        ref = (load_yaml("camera").get("calibrated_with") or {}).get("capture")
    except FileNotFoundError:
        ref = None
    if not ref:
        return []
    cur = video.capture_meta(clip)
    diffs = [f"{k}: calib {ref.get(k)!r} vs clip {cur.get(k)!r}" for k in ("width", "height", "lens", "fps")
             if ref.get(k) is not None and cur.get(k) is not None and str(ref[k]) != str(cur[k])]
    for d in diffs:
        log.warning("%s recorded differently from the calibration clip (%s)", clip, d)
    return diffs


def undistort_clip(src, dst_dir, K, dist, K_u, crf: int = 14, force: bool = False) -> Path:
    """Write ``undistorted.mp4`` (same PTS as the source) and ``timestamps.npy`` into dst_dir."""
    dst_dir = Path(dst_dir)
    out = dst_dir / "undistorted.mp4"
    if out.exists() and (dst_dir / "timestamps.npy").exists() and not force:
        return out
    dst_dir.mkdir(parents=True, exist_ok=True)
    info = video.probe(src)
    check_same_camera_mode(src)
    if info.rotation:
        log.warning("%s has %d deg rotation metadata; frames are processed as decoded (keep every clip "
                    "in the same orientation as the calib clip)", src, info.rotation)
    size = (info.width, info.height)
    mx, my = undistort_maps(K, dist, K_u, size)
    ts = []
    tmp = dst_dir / "undistorted.tmp.mp4"
    with video.VideoWriter(tmp, *size, crf=crf) as w:
        for _, t, img in video.iter_frames(src):
            w.write(cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT), t)
            ts.append(t)
    tmp.replace(out)
    np.save(dst_dir / "timestamps.npy", np.asarray(ts))
    (dst_dir / "timing.json").write_text(json.dumps(video.frame_rate_report(np.asarray(ts)), indent=2))
    return out


CELL_NAMES = [["top-left", "top-mid-left", "top-mid-right", "top-right"],
              ["mid-left", "center-left", "center-right", "mid-right"],
              ["bottom-left", "bottom-mid-left", "bottom-mid-right", "bottom-right"]]


def assess_capture(views: list[CalibView], sel: list[CalibView], cal: Calibration, info: video.VideoInfo,
                   expect_size=(1920, 1080), min_fps: float = 50.0) -> tuple[list[str], list[str], dict]:
    """Problems with the calib clip itself. Returns (blocking, warnings, stats).

    Blocking problems make the intrinsics untrustworthy even when the RMS looks fine: a board
    seen only flat-on cannot separate focal length from distance, and image regions the board
    never visited leave the distortion model extrapolating."""
    w, h = cal.image_size
    allp = np.concatenate([v.img for v in views])
    grid, _, _ = np.histogram2d(allp[:, 1], allp[:, 0], bins=[3, 4], range=[[0, h], [0, w]])
    frac = grid / grid.sum()
    tilts = []
    for v in sel:
        ok, rv, _ = cv2.solvePnP(v.obj, v.img, cal.K, cal.dist)
        R, _ = cv2.Rodrigues(rv)
        tilts.append(float(np.degrees(np.arccos(min(1.0, abs(R[2, 2]))))))
    tilts = np.array(tilts)
    block, warn = [], []
    if np.percentile(tilts, 90) < 25:
        block.append(f"board mostly flat-on to the camera (90th percentile tilt {np.percentile(tilts, 90):.0f} deg, "
                     "need >= 25): tilt it 30-45 deg toward/away and left/right")
    empty = [CELL_NAMES[r][c] for r in range(3) for c in range(4) if frac[r, c] < 0.02]
    if empty:
        block.append(f"board never (or barely) reached these image regions: {', '.join(empty)}")
    k = cal.dist
    if abs(k[1]) > 1.0 or abs(k[4]) > 2.0:
        block.append(f"distortion coefficients implausible (k2={k[1]:.2f}, k3={k[4]:.2f}): overfit from poor coverage/tilt")
    if (w, h) != tuple(expect_size):
        warn.append(f"resolution {w}x{h}, expected {expect_size[0]}x{expect_size[1]}: intrinsics only valid at this "
                    "resolution (re-encoded? record and copy the original file)")
    if info.avg_fps and info.avg_fps < min_fps:
        warn.append(f"{info.avg_fps:.0f} fps, expected 60: check the camera app's frame rate")
    mot = np.array([v.motion for v in views])
    lim = 1.5 * w / 1920
    if np.isfinite(mot).any() and np.mean(mot[np.isfinite(mot)] <= lim) < 0.2:
        warn.append(f"board rarely still: only {100 * np.mean(mot[np.isfinite(mot)] <= lim):.0f}% of frames move "
                    f"< {lim:.1f} px/frame. Pause ~1 s at each pose, and use a fast shutter (1/500)")
    stats = {"tilt_deg": {"min": float(tilts.min()), "median": float(np.median(tilts)), "max": float(tilts.max())},
             "motion_px_per_frame": {"median": float(np.nanmedian(mot)) if np.isfinite(mot).any() else None,
                                     "selected_max": float(np.nanmax([v.motion for v in sel])) if np.isfinite([v.motion for v in sel]).any() else None},
             "corner_share_per_cell_pct": np.round(100 * frac, 1).tolist(),
             "hfov_deg": float(np.degrees(2 * np.arctan(w / 2 / cal.K[0, 0])))}
    return block, warn, stats


def write_camera_yaml(cal: Calibration, K_u: np.ndarray, path: Path = CONFIG_DIR / "camera.yaml",
                      meta: dict | None = None) -> None:
    """Update the calibration fields in camera.yaml, keeping the charuco section and comments
    above the fields intact as far as YAML allows (the file is rewritten)."""
    cfg = yaml.safe_load(open(path)) or {}
    cfg.update(image_size=list(cal.image_size), K=np.round(cal.K, 6).tolist(),
               dist=np.round(cal.dist, 8).tolist(), rms_px=round(cal.rms, 4), K_u=np.round(K_u, 6).tolist())
    if meta:
        cfg["calibrated_with"] = meta
    header = ("# Written by ego2libero/calib/calibrate.py (S0).\n"
              "# K_u: fx == fy and principal point at the image center (HaWoR assumes both).\n")
    with open(path, "w") as f:
        f.write(header)
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path, help="data/raw/<session>/calib.mp4")
    ap.add_argument("--n-views", type=int)
    ap.add_argument("--stride", type=int, default=3, help="check every k-th frame for the board")
    ap.add_argument("--dry-run", action="store_true", help="report only; don't write camera.yaml")
    ap.add_argument("--accept-problems", action="store_true", help="write camera.yaml despite capture problems")
    ap.add_argument("--model", choices=["warped", "flat"], default="warped",
                    help="warped: fit a smooth board shape with a physical lens (default); flat: plain OpenCV")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pcfg = load_yaml("pipeline")["calib"]
    board = charuco_board()
    views, size = collect_views(a.clip, board, a.stride, pcfg["min_corners"])
    log.info("%d frames with >= %d ChArUco corners", len(views), pcfg["min_corners"])
    extra = {"model": a.model}
    if a.model == "warped":
        n_all = max(len(v.obj) for v in views)
        sel = select_views([v for v in views if len(v.obj) == n_all], size, a.n_views or pcfg["n_frames"])
        cal, info = calibrate_warped(sel, size)
        extra.update(info)
        fail_px = pcfg["rms_fail_px_warped"]
        log.info("board warp across the sheet: %.1f mm", info["board_warp_range_mm"])
    else:
        sel = select_views(views, size, a.n_views or pcfg["n_frames"])
        cal = calibrate_views(sel, size)
        fail_px = pcfg["rms_fail_px"]
    K_u = centered_K_u(cal.K, cal.dist, size)
    block, warn, cstats = assess_capture(views, sel, cal, video.probe(a.clip))
    status = "PASS" if cal.rms < pcfg["rms_target_px"] else ("WARN" if cal.rms <= fail_px else "FAIL")
    report = {
        "clip": str(a.clip), "image_size": list(size), "views_used": len(sel), "views_available": len(views),
        "rms_px": cal.rms, "gate": status, "worst_views_px": sorted(cal.per_view_rms.tolist())[-5:],
        "K": cal.K.tolist(), "dist": cal.dist.tolist(), "K_u": K_u.tolist(),
        "fov_kept_deg": float(np.degrees(2 * np.arctan(size[0] / 2 / K_u[0, 0]))),
        "capture_problems": block, "capture_warnings": warn, "capture_stats": cstats, **extra,
        "capture_meta": video.capture_meta(a.clip),
    }
    out_dir = Path("data/interim") / a.clip.parent.name / a.clip.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration_report.json").write_text(json.dumps(report, indent=2))
    log.info("calibration RMS %.3f px  [%s]  f=%.1f  K_u f=%.1f  -> %s", cal.rms, status, cal.K[0, 0], K_u[0, 0],
             out_dir / "calibration_report.json")
    log.info("board tilt %(min).0f/%(median).0f/%(max).0f deg (min/median/max), raw hFOV %(h).1f deg",
             {**cstats["tilt_deg"], "h": cstats["hfov_deg"]})
    for m in warn:
        log.warning("WARNING: %s", m)
    for m in block:
        log.error("PROBLEM: %s", m)
    if status == "FAIL":
        raise SystemExit(f"calibration RMS {cal.rms:.2f} px > {fail_px} px: re-record the calib clip")
    if block and not a.accept_problems and not a.dry_run:
        raise SystemExit("calib clip has capture problems (above); re-record it, or pass --accept-problems")
    if not a.dry_run:
        write_camera_yaml(cal, K_u, meta={"clip": str(a.clip), **extra, "capture": report["capture_meta"]})
        log.info("wrote configs/camera.yaml")


if __name__ == "__main__":
    main()
