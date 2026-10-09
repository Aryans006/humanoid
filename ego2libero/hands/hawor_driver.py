"""Run HaWoR on one undistorted clip and dump raw results to an .npz (plan stage S4).

RUNS INSIDE HaWoR's OWN CONDA ENV, with the HaWoR repo as the working directory (its code
uses relative paths such as ./weights and _DATA). Never import this from the main env;
scripts/remote/process.sh calls it, and ego2libero/hands/hawor_io.py reads its output.

Differences from HaWoR's demo.py, all deliberate:
  * Frames are extracted here, one JPEG per decoded frame. HaWoR's own extraction runs
    ``ffmpeg -vf fps=30``, which would resample our 60 fps VFR clips and break the
    frame <-> PTS mapping (timestamps.npy).
  * The calibrated focal length is always passed (HaWoR otherwise assumes 600 px).
    HaWoR assumes the principal point is the image center and fx == fy; S0 builds K_u so.
  * No visualization (aitviewer needs a display); everything is saved instead.

Output keys (S = HaWoR SLAM world, metric after HaWoR's Metric3D scale; N = frames):
  R_c2w (N,3,3), t_c2w (N,3)     camera pose in S as returned by HaWoR's load_slam_cam
  {right,left}_trans (N,3)        MANO transl in S
  {right,left}_rot (N,3)          MANO global_orient, axis-angle, in S
  {right,left}_hand_pose (N,45)   MANO finger pose, axis-angle
  {right,left}_betas (N,10)
  {right,left}_joints (N,21,3)    MANO joints in S, OpenPose order (0 wrist, 4 thumb tip, 8 index tip)
  {right,left}_observed (N,) bool frame was in a detected track (False = infiller output)
  img_focal, img_center (2,), image_size (2,) [w, h], slam_scale, n_frames, hawor_commit
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")  # HaWoR's mask renderer, headless
# torch >= 2.6 defaults torch.load(weights_only=True), which rejects HaWoR's Lightning checkpoint
# (it pickles an omegaconf config). The checkpoints are HaWoR's official release, so load them the
# pre-2.6 way rather than patching HaWoR's code.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
# PyTorch >= 2.6 defaults torch.load(weights_only=True), which rejects HaWoR's official
# checkpoints (they pickle omegaconf configs). They are trusted files from the HaWoR authors.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def extract_frames(video: Path, img_dir: Path) -> int:
    import cv2

    img_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(img_dir.glob("*.jpg"))
    if existing:
        return len(existing)
    cap = cv2.VideoCapture(str(video))
    n = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        cv2.imwrite(str(img_dir / f"{n:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        n += 1
    cap.release()
    if n == 0:
        raise RuntimeError(f"no frames decoded from {video}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True, help="undistorted clip")
    ap.add_argument("--focal", type=float, required=True, help="K_u focal length in pixels")
    ap.add_argument("--out", type=Path, required=True, help="output .npz")
    ap.add_argument("--work", type=Path, required=True, help="scratch dir for HaWoR's caches")
    ap.add_argument("--timestamps", type=Path, help="timestamps.npy; frame count must match")
    ap.add_argument("--hawor-root", type=Path, default=Path.cwd())
    a = ap.parse_args()

    root = a.hawor_root.resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))

    import cv2
    import numpy as np

    from scripts.scripts_test_video.detect_track_video import detect_track_video
    from scripts.scripts_test_video.hawor_video import hawor_motion_estimation, hawor_infiller
    from scripts.scripts_test_video.hawor_slam import hawor_slam
    from hawor.utils.process import run_mano, run_mano_left
    from lib.eval_utils.custom_utils import load_slam_cam

    # HaWoR derives its cache folder from the video path: <dir>/<stem>/...
    a.work.mkdir(parents=True, exist_ok=True)
    stem = "clip"
    vid = a.work / f"{stem}.mp4"
    if not vid.exists():
        vid.symlink_to(a.video.resolve())
    seq = a.work / stem
    n = extract_frames(a.video, seq / "extracted_images")
    if a.timestamps is not None:
        n_ts = len(np.load(a.timestamps))
        if n_ts != n:
            raise RuntimeError(f"decoded {n} frames but timestamps.npy has {n_ts}")
    print(f"[hawor_driver] {n} frames, focal {a.focal:.2f}px")

    args = Namespace(video_path=str(vid), img_focal=a.focal, input_type="file",
                     checkpoint="./weights/hawor/checkpoints/hawor.ckpt",
                     infiller_weight="./weights/hawor/checkpoints/infiller.pt", vis_mode="none")

    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)
    frame_chunks_all, img_focal = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)
    slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    if not os.path.exists(slam_path):
        hawor_slam(args, start_idx, end_idx)
    _, _, R_c2w, t_c2w = load_slam_cam(slam_path)
    slam = dict(np.load(slam_path, allow_pickle=True))
    trans, rot, hand_pose, betas, _ = hawor_infiller(args, start_idx, end_idx, frame_chunks_all)

    out = {
        "R_c2w": R_c2w.numpy().astype(np.float64),
        "t_c2w": t_c2w.numpy().astype(np.float64),
        "img_focal": float(img_focal),
        "img_center": np.asarray(slam["img_center"], dtype=np.float64),
        "slam_scale": float(slam["scale"]),
        "n_frames": int(n),
    }
    h, w = cv2.imread(imgfiles[0]).shape[:2]
    out["image_size"] = np.array([w, h])

    for name, idx, mano_fn in (("left", 0, run_mano_left), ("right", 1, run_mano)):
        sl = slice(idx, idx + 1)
        res = mano_fn(trans[sl], rot[sl], hand_pose[sl], betas=betas[sl])
        observed = np.zeros(n, dtype=bool)
        for ck in frame_chunks_all.get(idx, []):
            observed[np.asarray(ck)] = True
        out.update({
            f"{name}_trans": trans[idx].numpy().astype(np.float64),
            f"{name}_rot": rot[idx].numpy().astype(np.float64),
            f"{name}_hand_pose": hand_pose[idx].numpy().astype(np.float64),
            f"{name}_betas": betas[idx].numpy().astype(np.float64),
            f"{name}_joints": res["joints"][0].detach().cpu().numpy().astype(np.float64),
            f"{name}_observed": observed,
        })
    try:
        out["hawor_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        out["hawor_commit"] = "unknown"

    for k in ("R_c2w", "right_trans", "right_joints"):
        if len(out[k]) != n:
            raise RuntimeError(f"{k} has {len(out[k])} frames, expected {n}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **out)
    print(f"[hawor_driver] wrote {a.out}  (right hand observed in {out['right_observed'].mean():.0%} of frames)")


if __name__ == "__main__":
    main()
