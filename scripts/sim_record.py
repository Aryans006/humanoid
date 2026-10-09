"""Record sim episodes of the cube-to-box task from processed human demos (training data).

  MUJOCO_GL=egl .venv-sim/bin/python scripts/sim_record.py <session> [demo ...] [--repeats 1]

Each demo is retargeted and executed in the CubeToBox scene with LIBERO-style DELTA pose actions
(robosuite OSC_POSE, control_delta=True, 20 Hz), and exactly the executed actions are saved.
The workspace placement T_L_W comes from the session's entry in configs/libero_frame.yaml.
At each grasp the demo is held until the tool point actually reaches the cube and the fingers
have seated (ego2libero.sim.execute.GraspSchedule), because the person lifts almost the instant
they touch it; --no-grasp-wait restores the demo's own timing.
The box is at one fixed place for every episode: the mean put-down spot over the session.
Only successful episodes (cube resting in the box) are kept.

Output: data/sim_episodes/<session>/<demo>_r<k>.npz with
  image, wrist_image (T, 256, 256, 3) uint8, upright      # agentview + robot0_eye_in_hand
  state (T, 8)    eef pos (3), eef axis-angle (3), gripper qpos (2)   # LIBERO's state
  action (T, 7)   delta pos (3), delta axis-angle (3) in [-1, 1], gripper -1 open / +1 close
  task (str), success, demo, fps
and an index.json. ``--repeats k`` re-runs each demo with different robot start-joint noise.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from robosuite import load_controller_config  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from ego2libero.export.write_demo import load_demo  # noqa: E402
from ego2libero.sim.execute import GraspSchedule  # noqa: E402
from ego2libero.sim.frame import TABLE_Z, load_T_L_W  # noqa: E402
from ego2libero.sim.retarget import retarget  # noqa: E402
from ego2libero.sim.scene import CubeToBox  # noqa: E402

TASK = "pick up the red cube and place it in the box"
POS_MAX, ROT_MAX = 0.05, 0.5  # robosuite OSC_POSE default output limits (m, rad) for actions in [-1, 1]


def box_location(session: str, T_L_W: np.ndarray) -> np.ndarray:
    """Mean put-down spot (cube's final xy) over all exported demos of the session, in L."""
    ends = []
    for f in sorted(glob.glob(str(ROOT / "data/processed" / session / "demo_*_20hz.npz"))):
        if f.endswith("_L.npz"):
            continue
        T = np.load(f)["T_W_O"][-1]
        ends.append((T_L_W @ T)[:2, 3])
    return np.median(ends, axis=0)


def obs_images(env, size=256):
    img = env.sim.render(camera_name="agentview", width=size, height=size)[::-1].copy()
    wrist = env.sim.render(camera_name="robot0_eye_in_hand", width=size, height=size)[::-1].copy()
    side = env.sim.render(camera_name="sideview", width=size, height=size)[::-1].copy()
    return img, wrist, side


def run_episode(d, T_L_W, box_xy, seed, noise, max_tail_s=1.5, grasp_wait=True):
    tg = retarget(d, T_L_W, TABLE_Z)
    # start where the person's hand is first visible, end shortly after the cube is released
    k0 = int(np.flatnonzero(tg.hand_ok)[0]) if tg.hand_ok.any() else 0
    rel = np.flatnonzero(np.diff(tg.held.astype(np.int8)) == -1)
    k1 = min(len(tg.t), (int(rel[-1]) + int(max_tail_s * 20)) if len(rel) else len(tg.t))
    np.random.seed(seed)
    ctrl = load_controller_config(default_controller="OSC_POSE")  # delta control, LIBERO-style
    env = CubeToBox(robots="Panda", controller_configs=ctrl, has_renderer=False, has_offscreen_renderer=True,
                    use_camera_obs=False, use_object_obs=False, control_freq=20, horizon=100000, ignore_done=True,
                    box_xy=box_xy, initialization_noise=None if noise <= 0 else {"magnitude": noise, "type": "gaussian"})
    env.reset()
    env.set_cube_pose(tg.T_L_O[0])
    env.sim.forward()
    site = env.sim.model.site_name2id(env.robots[0].controller.eef_name)
    grip_idx = env.robots[0]._ref_gripper_joint_pos_indexes

    imgs, wrists, sides, states, actions = [], [], [], [], []

    def step(p_tgt, R_tgt, g):
        p = env.sim.data.site_xpos[site].copy()
        R = env.sim.data.site_xmat[site].reshape(3, 3).copy()
        img, wrist, side = obs_images(env)
        state = np.r_[p, Rotation.from_matrix(R).as_rotvec(), env.sim.data.qpos[grip_idx]]
        dpos = np.clip((p_tgt - p) / POS_MAX, -1, 1)
        drot = np.clip(Rotation.from_matrix(R_tgt @ R.T).as_rotvec() / ROT_MAX, -1, 1)
        a = np.r_[dpos, drot, g]
        env.step(a)
        imgs.append(img), wrists.append(wrist), sides.append(side), states.append(state), actions.append(a)

    # approach the demo's first target from wherever the arm starts, then follow the demo
    start = env.sim.data.site_xpos[site].copy()
    for j in range(30):
        step(start + min(1.0, (j + 1) / 24) * (tg.pos_L[k0] - start), tg.R_L_tool(k0), -1.0)
    sched = GraspSchedule(tg.grip, tg.pos_L, k0=k0, k1=k1, fps=20, enabled=grasp_wait)
    while True:
        nxt = sched.next(env.sim.data.site_xpos[site], env.cube_pose()[:3, 3])
        if nxt is None:
            break
        k, g = nxt
        step(tg.pos_L[k], tg.R_L_tool(k), g)
    ok = env.success()
    env.close()
    return dict(image=np.stack(imgs), wrist_image=np.stack(wrists), side_image=np.stack(sides), state=np.array(states, np.float32),
                action=np.array(actions, np.float32), success=ok), sched.grasps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("demos", nargs="*", help="default: the session's selection.json 'selected' list")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--noise", type=float, default=0.02, help="start-joint noise (rad) for repeats >= 2")
    ap.add_argument("--tlw", help="json with a T_L_W to use instead of configs/libero_frame.yaml")
    ap.add_argument("--no-grasp-wait", dest="grasp_wait", action="store_false",
                    help="close the gripper on the demo's own timing (the old behaviour)")
    a = ap.parse_args()
    proc = ROOT / "data/processed" / a.session
    demos = a.demos or json.loads((proc / "selection.json").read_text())["selected"]
    T_L_W = load_T_L_W(a.session, a.tlw)
    box_xy = box_location(a.session, T_L_W)
    out = ROOT / "data/sim_episodes" / a.session
    out.mkdir(parents=True, exist_ok=True)
    index_p = out / "index.json"
    index = json.loads(index_p.read_text()) if index_p.exists() else {
        "task": TASK, "fps": 20, "box_xy_L": box_xy.tolist(), "T_L_W": T_L_W.tolist(),
        "grasp_wait": a.grasp_wait, "episodes": {}}
    for demo in demos:
        if not (proc / f"{demo}_20hz.npz").exists():
            print(f"{demo}: no export yet, skipped")
            continue
        d = load_demo(proc / f"{demo}_20hz")
        for r in range(a.repeats):
            name = f"{demo}_r{r}"
            ep, grasps = run_episode(d, T_L_W, box_xy, seed=r, noise=0.0 if r == 0 else a.noise,
                                     grasp_wait=a.grasp_wait)
            index["episodes"][name] = {"demo": demo, "repeat": r, "success": bool(ep["success"]),
                                       "frames": len(ep["action"]), "grasps": grasps}
            if ep["success"]:
                np.savez_compressed(out / f"{name}.npz", **ep, task=TASK, demo=demo, fps=20)
            waited = "".join(f"  grasp@{g['frame']}: {g['reason']} after {g['reach_steps']} steps, "
                             f"{g['tool_target_err_m'] * 100:.1f} cm off target, "
                             f"{g['tool_cube_dist_m'] * 100:.1f} cm from the cube" for g in grasps)
            print(f"{name}: {'SUCCESS' if ep['success'] else 'failed'}  {len(ep['action'])} frames{waited}")
            index_p.write_text(json.dumps(index, indent=2))
    n_ok = sum(e["success"] for e in index["episodes"].values())
    print(f"{n_ok}/{len(index['episodes'])} episodes successful -> {out}")


if __name__ == "__main__":
    main()
