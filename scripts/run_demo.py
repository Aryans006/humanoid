"""Run one clip through the pipeline; every stage is cached in data/interim/<session>/<clip>/.

  uv run python scripts/run_demo.py <session> <clip>          # e.g. 2026-10-03_setup demo_000
  uv run python scripts/run_demo.py <session> --all           # every clip except calib
  options: --force (rerun all stages)  --overlay-scale 0.5  --no-overlay

Stages so far (M1): undistort -> detect -> board pose -> box pose -> overlay + gates.
Calibration is separate: uv run python -m ego2libero.calib.calibrate data/raw/<session>/calib.mp4
Gates depend on the clip name: static*, known_pos*, demo_*.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego2libero import gates  # noqa: E402
from ego2libero.calib.calibrate import undistort_clip  # noqa: E402
from ego2libero.config import config_dir, load_yaml, require  # noqa: E402
from ego2libero.perception.board_pose import BoardModel, CameraTrack, track_camera  # noqa: E402
from ego2libero.perception.box_pose import BoxModel, BoxTrack, track_box  # noqa: E402
from ego2libero.perception.detect import Detections, detect_clip  # noqa: E402
from ego2libero.align.hands_to_world import HandTrackW, hands_to_world  # noqa: E402
from ego2libero.hands.hawor_io import check_intrinsics, load_hawor  # noqa: E402

log = logging.getLogger("run_demo")
VIDEO_EXT = (".mp4", ".MP4", ".mov", ".MOV")


def find_clip(session_dir: Path, name: str) -> Path:
    for ext in VIDEO_EXT:
        if (session_dir / f"{name}{ext}").exists():
            return session_dir / f"{name}{ext}"
    raise FileNotFoundError(f"no {name}.mp4/.mov in {session_dir}")


def stage(path: Path, force: bool) -> bool:
    """True if the stage must run."""
    return force or not path.exists()


def run_clip(session: str, clip: str, force=False, overlay=True, overlay_scale=0.5, data=ROOT / "data") -> list[dict]:
    raw = find_clip(data / "raw" / session, clip)
    out = data / "interim" / session / clip
    out.mkdir(parents=True, exist_ok=True)
    cam_cfg, pipe = load_yaml("camera"), load_yaml("pipeline")
    require(cam_cfg, "K", "dist", "K_u", where="camera.yaml (run calibration first)")
    K, dist, K_u = (np.asarray(cam_cfg[k], float) for k in ("K", "dist", "K_u"))

    # S0 undistort
    und = undistort_clip(raw, out, K, dist, K_u, crf=pipe["undistort"]["crf"], force=force)
    t = np.load(out / "timestamps.npy")

    # S1 detect
    p = out / "detections.npz"
    if stage(p, force):
        detect_clip(und, refine=pipe["detect"]["refine"]).save(p)
    dets = Detections.load(p)
    per_frame = dets.per_frame()

    # S2 board -> T_W_C
    board = BoardModel.from_config(load_yaml("board"))
    p = out / "camera_tags.npz"
    if stage(p, force):
        bp = pipe["board_pose"]
        track_camera(per_frame, K_u, board, bp["min_tags"], bp["max_rms_px"]).save(p)
    cam = CameraTrack.load(p)

    # S3 box from face tags -> T_W_O (tag frames only)
    model = BoxModel.from_config(load_yaml("box"))
    p = out / "box_tags.npz"
    if stage(p, force):
        w = pipe["windows"]
        rest = None
        if clip.startswith(("demo", "static", "known_pos")):  # the box starts and ends on the table
            rest = [t <= t[0] + w["static_start_s"], t >= t[-1] - w["static_end_s"]]
        track_box(per_frame, t, K_u, cam.T_W_C, model, pipe["box_pose"], rest_masks=rest).save(p)
    box = BoxTrack.load(p)

    # gates
    g, w = pipe["gates"], pipe["windows"]
    res = [gates.focal_consistency(per_frame, board, K_u, T_W_C=cam.T_W_C),
           {"gate": "board_pose_coverage", "value": float(cam.valid.mean()), "limit": None, "pass": True},
           {"gate": "box_ambiguous_frames", "value": int(box.ambiguous.sum()), "limit": None, "pass": True}]
    if clip.startswith("static"):
        res += gates.camera_jitter(cam.T_W_C, g) + gates.box_jitter(box.T_W_O, g)
    elif clip.startswith("known_pos"):
        task = load_yaml("task")
        spots = task.get("known_spots_W")
        res += (gates.known_positions(box.T_W_O, t, model, spots, g, placement=task.get("known_spot_placement", "center"))
                if spots else [])
    elif clip.startswith("demo"):
        res += gates.box_resting(box.T_W_O, t, model, g, w) + [gates.coverage(box.valid, g)]

    # S5 hands into W (needs hawor.npz from the GPU machine: scripts/remote/pull.sh)
    hands = None
    hawor_p, p = out / "hawor.npz", out / "hands_W.npz"
    if hawor_p.exists():
        if stage(p, force) or p.stat().st_mtime < hawor_p.stat().st_mtime:
            tr = load_hawor(hawor_p)
            check_intrinsics(tr, K_u)
            hands_to_world(tr, cam.T_W_C, pipe["align"]["min_center_spread_m"]).save(p)
        hands = HandTrackW.load(p)
        res += gates.sim3_sanity(hands.sim, g)
        res.append({"gate": "hand_observed_frac", "value": float((hands.hand_source == 0).mean()), "limit": None, "pass": True})

    # S6-S8 fusion (phases, box gap filling, smoothing) and S9 export, for demos with hands
    fused = None
    if clip.startswith("demo") and hands is not None:
        from ego2libero.fusion.fuse import fuse
        from ego2libero.viz.plots import plot_demo

        fused = fuse(t, box.T_W_O, hands.T_W_H, hands.joints_W, hands.hand_source, model.dims, pipe)
        labels = (load_yaml("phase_labels") if (config_dir() / "phase_labels.yaml").exists() else {}) or {}
        res += gates.fusion_gates(fused, t, g, labels.get(session, {}).get(clip))
        plot_demo(out / "plots.png", t, fused, box.T_W_O, hands.T_W_H, pipe["phases"], title=f"{session}/{clip}")
        export_demo(data, session, clip, t, cam, box, hands, fused, K_u, res, pipe)
    (out / "gates.json").write_text(json.dumps(res, indent=2, default=str))

    ov = out / "overlay.mp4"
    stale = hands is not None and ov.exists() and ov.stat().st_mtime < (out / "hands_W.npz").stat().st_mtime
    stale |= fused is not None and ov.exists() and ov.stat().st_mtime < (out / "plots.png").stat().st_mtime
    if overlay and (stage(ov, force) or stale):
        from ego2libero.viz.overlay import write_overlay

        def info(i):
            b = f"board {cam.n_tags[i]} tags" + (f"  {cam.rms_px[i]:.2f}px" if np.isfinite(cam.rms_px[i]) else "  -")
            o = (f"box {box.n_tags[i]} tags {box.rms_px[i]:.2f}px {box.pick_reason[i]}" if box.valid[i] else "box -")
            if hands is None:
                return [b, o]
            src = "observed" if hands.hand_source[i] == 0 else "infilled"
            lines = [b, o, f"hand {src}  aperture {100 * hands.aperture[i]:.1f} cm"]
            if fused is not None:
                from ego2libero.fusion.phases import PHASE_NAMES
                from ego2libero.viz.overlay import SRC_NAME
                lines.append(f"phase {PHASE_NAMES[fused.seg.phase[i]].upper()}  gripper "
                             f"{'CLOSED' if fused.seg.gripper_closed[i] else 'open'}  box {SRC_NAME[int(fused.obj_source[i])]}")
            return lines

        spots = load_yaml("task").get("known_spots_W") if clip.startswith("known_pos") else None
        write_overlay(und, out / "overlay.mp4", K_u, T_W_C=cam.T_W_C,
                      T_W_O=box.T_W_O if fused is None else fused.T_W_O, T_C_O=box.T_C_O if fused is None else None,
                      obj_src=None if fused is None else fused.obj_source,
                      box_corners=model.box_corners(), dets_per_frame=per_frame, spots_W=spots,
                      joints_W=None if hands is None else (hands.joints_W if fused is None else fused.joints_W),
                      extra_info=info, scale=overlay_scale)
    return res


def export_demo(data, session, clip, t, cam, box, hands, fused, K_u, res, pipe):
    """S9: <clip>.npz/.json in W at native PTS, <clip>_20hz, and LIBERO-frame copies once T_L_W is set."""
    from ego2libero.export.to_libero import resample, to_libero
    from ego2libero.export.write_demo import SRC_MISSING, SRC_TAG, write_demo

    cam_src = np.where(cam.valid, SRC_TAG, SRC_MISSING).astype(np.uint8)
    arrays = dict(t=t, T_W_C=cam.T_W_C, T_W_O=fused.T_W_O, T_W_H=fused.T_W_H, joints_W=fused.joints_W,
                  mano_pose=hands.mano_pose, mano_betas=hands.mano_betas, aperture=fused.aperture,
                  gripper_closed=fused.seg.gripper_closed, phase=fused.seg.phase,
                  cam_source=cam_src, obj_source=fused.obj_source, hand_source=hands.hand_source,
                  board_reproj_px=cam.rms_px, box_reproj_px=box.rms_px)
    from ego2libero.sim.frame import load_T_L_W

    box_cfg, T_L_W = load_yaml("box"), load_T_L_W(session).tolist()
    meta = {"stage": "M3", "session": session, "clip": clip, "K_u": K_u, "image_size": [1920, 1080],
            "box_dims_m": box_cfg["dims_m"], "box_tag_size_m": box_cfg["tag_size_m"], "T_L_W": T_L_W,
            "sim3": hands.sim.summary(), "phase_onsets_s": fused.seg.onset_times(t), "phase_notes": fused.seg.notes,
            "fusion_stats": fused.stats,
            "T_H_O": None if fused.grasp is None else fused.grasp.T_H_O.tolist(),
            "thresholds": {k: pipe[k] for k in ("phases", "bridge", "smooth")}, "gates": res,
            "accepted": all(r["pass"] for r in res if r.get("limit") is not None)}
    base = data / "processed" / session / clip
    write_demo(base, arrays, meta)
    a20 = resample(arrays, pipe["export"]["resample_hz"])
    write_demo(base.with_name(f"{clip}_20hz"), a20, {**meta, "resampled_hz": pipe["export"]["resample_hz"]})
    if T_L_W is not None:
        write_demo(base.with_name(f"{clip}_L"), to_libero(arrays, T_L_W), {**meta, "frame": "L"})
        write_demo(base.with_name(f"{clip}_20hz_L"), to_libero(a20, T_L_W), {**meta, "frame": "L", "resampled_hz": 20})
    log.info("exported %s (+ 20 Hz%s)", base.with_suffix(".npz"), ", LIBERO frame" if T_L_W is not None else "; T_L_W not set yet")


def print_gates(session, clip, res):
    print(f"\n== {session}/{clip}")
    for r in res:
        v = r["value"]
        vs = "-" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
        flag = "" if r["limit"] is None else ("n/a" if v is None and r["pass"] else ("PASS" if r["pass"] else "FAIL"))
        extra = {k: v for k, v in r.items() if k not in ("gate", "value", "limit", "pass")}
        print(f"  {flag:4s}  {r['gate']:24s} {vs:>10s}  (limit {r['limit']})  {extra if extra else ''}")


def append_summary(session, clip, res, data=ROOT / "data"):
    path = data / "processed" / session / "summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(["time", "clip", "gate", "value", "limit", "pass"])
        now = dt.datetime.now().isoformat(timespec="seconds")
        for r in res:
            wr.writerow([now, clip, r["gate"], r["value"], r["limit"], r["pass"]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("clip", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--overlay-scale", type=float, default=0.5)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sdir = ROOT / "data/raw" / a.session
    clips = ([p.stem for p in sorted(sdir.iterdir()) if p.suffix in VIDEO_EXT and not p.stem.startswith("calib")]
             if a.all else [a.clip])
    if not clips or clips == [None]:
        ap.error("give a clip name or --all")
    for c in clips:
        res = run_clip(a.session, c, a.force, not a.no_overlay, a.overlay_scale)
        print_gates(a.session, c, res)
        append_summary(a.session, c, res)


if __name__ == "__main__":
    main()
