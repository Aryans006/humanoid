"""Replay a processed demo on a Panda in the cube-to-box tabletop task (robosuite 1.4.1, CPU).

The box is placed where the person put the cube down, so a faithful replay drops the cube in it.

  MUJOCO_GL=egl .venv-sim/bin/python scripts/sim_replay.py <session> <demo> [--mode kinematic|physics]

kinematic: the cube is forced to its recorded pose every step (shows the data itself).
physics:   the cube starts at its recorded pose and only moves if the gripper really grasps it.
Grasps wait for the tool to reach the cube, as in sim_record.py; --no-grasp-wait turns that off.
Writes data/processed/<session>/sim/<demo>_<mode>.mp4 (real overlay | sim from the phone's
viewpoint | sim front view) and <demo>_<mode>.json with tracking / outcome metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import robosuite as suite  # noqa: E402
from robosuite import load_controller_config  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from ego2libero import se3, video  # noqa: E402
from ego2libero.export.write_demo import load_demo  # noqa: E402
from ego2libero.sim.execute import GraspSchedule  # noqa: E402
from ego2libero.sim.frame import TABLE_Z, load_T_L_W  # noqa: E402
from ego2libero.sim.retarget import retarget  # noqa: E402
from ego2libero.sim.scene import CubeToBox  # noqa: E402


def _look_at(eye, target, up=(0, 0, 1)):
    """OpenCV-convention camera (x right, y down, z forward) at eye looking at target."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    return dict(pos=eye, R_cv=np.stack([x, np.cross(z, x), z], axis=1))


def label(img, text):
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("demo")
    ap.add_argument("--mode", choices=["kinematic", "physics"], default="kinematic")
    ap.add_argument("--preroll-s", type=float, default=2.0)
    ap.add_argument("--init-noise", type=float, default=0.0,
                    help="std (rad) of robosuite's random start-joint noise; 0 = exactly repeatable (robosuite default 0.02)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-video", action="store_true", help="metrics only (fast, for robustness sweeps)")
    ap.add_argument("--tlw", help="json with a T_L_W to use instead of configs/libero_frame.yaml")
    ap.add_argument("--no-grasp-wait", dest="grasp_wait", action="store_false",
                    help="close the gripper on the demo's own timing (the old behaviour)")
    a = ap.parse_args()

    proc = ROOT / "data/processed" / a.session
    d = load_demo(proc / f"{a.demo}_20hz")
    T_L_W = load_T_L_W(a.session, a.tlw)
    tg = retarget(d, T_L_W, TABLE_Z)
    K_u = np.array(d["meta"]["K_u"], float)

    # the box goes where the person put the cube down
    box_xy = tg.T_L_O[-1][:2, 3]
    cam_ok = np.isfinite(d["T_W_C"]).all(axis=(1, 2))
    T_L_C = T_L_W @ se3.average_poses(d["T_W_C"][cam_ok])
    # The phone itself sits where the robot's wrist moves (~0.55 m above the table), so the sim
    # renders the same view from 0.8 m further back along the phone's optical axis, with the field
    # of view narrowed to frame the same patch of table.
    half_fov = np.arctan(K_u[1, 2] / K_u[1, 1])
    depth = float((T_L_C[:3, 3][2] - TABLE_Z) / max(T_L_C[2, 2] * -1, 0.3))
    back = 0.8
    cams = {"phoneview": dict(pos=T_L_C[:3, 3] - back * T_L_C[:3, 2], R_cv=T_L_C[:3, :3],
                              fovy_deg=float(np.degrees(2 * np.arctan(np.tan(half_fov) * depth / (depth + back))))),
            "observer": dict(**_look_at([0.75, -0.55, 1.35], [-0.02, 0.0, 0.86]), fovy_deg=38.0)}

    ctrl = load_controller_config(default_controller="OSC_POSE")
    ctrl["control_delta"] = False
    np.random.seed(a.seed)
    noise = None if a.init_noise <= 0 else {"magnitude": a.init_noise, "type": "gaussian"}
    env = CubeToBox(robots="Panda", initialization_noise=noise, controller_configs=ctrl, has_renderer=False, has_offscreen_renderer=True,
                         use_camera_obs=False, use_object_obs=False, control_freq=20, horizon=100000, ignore_done=True,
                         box_xy=box_xy, cameras=cams)
    render = not a.no_video
    env.reset()
    env.set_cube_pose(tg.T_L_O[0])
    env.sim.forward()
    eef_site = env.sim.model.site_name2id(env.robots[0].controller.eef_name)
    eef = lambda: env.sim.data.site_xpos[eef_site].copy()

    real = {}
    if render:
        for i, t, img in video.iter_frames(ROOT / "data/interim" / a.session / a.demo / "overlay.mp4"):
            real[round(t, 4)] = img
    real_t = np.array(sorted(real))

    out_dir = proc / "sim"
    out_dir.mkdir(exist_ok=True)
    tag = f"{a.demo}_{a.mode}" + (f"_noise{a.init_noise:g}_seed{a.seed}" if a.init_noise > 0 else "")
    path = out_dir / f"{tag}.mp4"
    H = 360

    class _NoWriter:
        def write(self, *args):
            pass

        def close(self):
            pass

    writer = video.VideoWriter(path, 640 + 640 + 480, H, crf=23, fps_hint=20) if render else _NoWriter()

    def frame(k_data, phase_txt):
        sim_view = env.sim.render(camera_name="phoneview", width=640, height=360)[::-1, :, ::-1].copy()
        front = env.sim.render(camera_name="observer", width=480, height=360)[::-1, :, ::-1].copy()
        tr = tg.t[k_data]
        rimg = real[real_t[np.argmin(np.abs(real_t - tr))]]
        rimg = cv2.resize(rimg, (640, 360), interpolation=cv2.INTER_AREA)
        return np.hstack([label(rimg, f"real  t={tr:5.2f}s"), label(sim_view, f"sim ({a.mode})  {phase_txt}"),
                          label(front, "sim observer view")])

    def action(k, g=None):
        R = tg.R_L_tool(k)
        return np.r_[tg.pos_L[k], Rotation.from_matrix(R).as_rotvec(), tg.grip[k] if g is None else g]

    names = ["approach", "grasp", "transport", "release", "retreat"]
    n_pre = int(round(a.preroll_s * 20))
    start = eef()
    t_video = 0.0
    for j in range(n_pre):  # move from the home pose to the first target
        alpha = min(1.0, (j + 1) / (0.8 * n_pre))
        act = action(0)
        act[:3] = (1 - alpha) * start + alpha * tg.pos_L[0]
        env.step(act)
        if a.mode == "kinematic":
            env.set_cube_pose(tg.T_L_O[0])
            env.sim.forward()
        if render:
            writer.write(frame(0, "moving to start"), t_video)
        t_video += 0.05

    # the schedule may repeat a frame while it waits for the tool to reach the cube, so every
    # per-step metric is indexed by the frames actually executed (ks), not by range(len(tg.t))
    sched = GraspSchedule(tg.grip, tg.pos_L, fps=20, enabled=a.grasp_wait)
    err, cube_z, cube_xy_err, ks = [], [], [], []
    first_success = None
    while True:
        nxt = sched.next(eef(), env.cube_pose()[:3, 3])
        if nxt is None:
            break
        k, g = nxt
        env.step(action(k, g))
        if a.mode == "kinematic":
            env.set_cube_pose(tg.T_L_O[k])
            env.sim.forward()
        if first_success is None and env.success():
            first_success = float(tg.t[k])
        ks.append(k)
        err.append(np.linalg.norm(eef() - tg.pos_L[k]))
        Tc = env.cube_pose()
        cube_z.append(Tc[2, 3] - TABLE_Z)
        cube_xy_err.append(np.linalg.norm(Tc[:2, 3] - tg.T_L_O[k][:2, 3]))
        ph = names[int(d["phase"][k])] + ("  gripper CLOSED" if g > 0 else "  gripper open")
        if first_success is not None:
            ph += "  IN BOX"
        if render:
            writer.write(frame(k, ph), t_video)
        t_video += 0.05
    writer.close()

    err, cube_z, cube_xy_err, ks = map(np.array, (err, cube_z, cube_xy_err, ks))
    held = tg.held[ks]
    rec_z = tg.T_L_O[:, 2, 3] - TABLE_Z
    metrics = {
        "mode": a.mode, "steps": int(len(ks)), "demo_frames": int(len(tg.t)), "video": str(path),
        "grasp_wait": a.grasp_wait, "grasps": sched.grasps,
        "tool_tracking_error_m": {"median": float(np.median(err)), "p95": float(np.percentile(err, 95)),
                                  "while_held_median": float(np.median(err[held]))},
        "cube_max_height_m": {"sim": float(cube_z.max()), "recorded": float(rec_z.max())},
        "cube_final_xy_error_m": float(cube_xy_err[-1]),
        "cube_xy_error_while_held_median_m": float(np.median(cube_xy_err[held])),
        "lifted_while_held_frac": float(np.mean(cube_z[held] > 0.025 + 0.015)),
        "box_xy_L": box_xy.tolist(),
        "success_cube_in_box": bool(env.success()),
        "first_success_t": first_success,
    }
    metrics.update(init_noise=a.init_noise, seed=a.seed)
    if not render:
        metrics["video"] = None
    (out_dir / f"{tag}.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
