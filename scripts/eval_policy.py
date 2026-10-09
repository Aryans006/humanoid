"""Roll a trained SmolVLA checkpoint out in the CubeToBox task and report in-box success
(plan.md's missing M4 step; handoff section 5).

  MUJOCO_GL=egl .venv-sim/bin/python scripts/eval_policy.py <session> <checkpoint_dir> [options]

The policy runs in .venv-eval behind scripts/policy_server.py, because lerobot needs numpy >= 2
and robosuite needs numpy < 2. This process owns the simulator and feeds it exactly what
sim_record.py recorded: camera1/2/3 = agentview / wrist / sideview at 256^2, an 8-D state
(eef pos, eef axis-angle, gripper qpos), and the policy's 7-D delta action straight into
robosuite's OSC_POSE.

**The scene must match the one the checkpoint was trained in.** ``--tlw`` / ``--box-xy`` exist
for exactly that: the 2026-10-07 checkpoints were trained under the pilots' T_L_W with the box at
(-0.022, 0.118), while that session now uses a frame rotated 180 degrees. Evaluated in the wrong
frame a policy sees a workspace rotated away from its training data and scores zero for a reason
that has nothing to do with the policy. ``--as-trained`` reads both back from a recording index.

Writes <out>/eval_<checkpoint>.json with per-position and overall success.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from robosuite import load_controller_config  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from ego2libero.export.write_demo import load_demo  # noqa: E402
from ego2libero.sim.frame import TABLE_Z, load_T_L_W  # noqa: E402
from ego2libero.sim.scene import CubeToBox  # noqa: E402

TASK = "pick up the red cube and place it in the box"


class Policy:
    """The .venv-eval side of the fence."""

    def __init__(self, ckpt: Path, python: Path):
        self.p = subprocess.Popen([str(python), str(ROOT / "scripts/policy_server.py"), str(ckpt)],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  env={**os.environ, "HF_HOME": str(ROOT / "data/.hf_cache")})
        self._recv()  # ready

    def _send(self, msg, payload=b""):
        msg["nbytes"] = len(payload)
        self.p.stdin.write((json.dumps(msg) + "\n").encode())
        self.p.stdin.write(payload)
        self.p.stdin.flush()

    def _recv(self):
        head = self.p.stdout.readline()
        if not head:
            raise RuntimeError("policy server died")
        msg = json.loads(head)
        n = msg.get("nbytes", 0)
        buf = b""
        while len(buf) < n:
            c = self.p.stdout.read(n - len(buf))
            if not c:
                raise RuntimeError("policy server died mid-message")
            buf += c
        return msg, buf

    def reset(self):
        self._send({"cmd": "reset"})
        self._recv()

    def act(self, images: dict, state: np.ndarray, task: str) -> np.ndarray:
        payload = b"".join(np.ascontiguousarray(v, np.uint8).tobytes() for v in images.values())
        payload += np.ascontiguousarray(state, np.float32).tobytes()
        self._send({"cmd": "act", "images": {k: list(v.shape) for k, v in images.items()},
                    "state": [len(state)], "task": task}, payload)
        msg, buf = self._recv()
        return np.frombuffer(buf, np.float32, count=msg["action"][0]).copy()

    def close(self):
        try:
            self._send({"cmd": "quit"})
            self.p.wait(timeout=30)
        except Exception:
            self.p.kill()


def obs_images(env, size=256):
    """Exactly sim_record.py's cameras and orientation."""
    return {"camera1": env.sim.render(camera_name="agentview", width=size, height=size)[::-1].copy(),
            "camera2": env.sim.render(camera_name="robot0_eye_in_hand", width=size, height=size)[::-1].copy(),
            "camera3": env.sim.render(camera_name="sideview", width=size, height=size)[::-1].copy()}


def rollout(policy, T_L_O0, box_xy, max_steps, seed=0):
    ctrl = load_controller_config(default_controller="OSC_POSE")  # delta, as recorded
    np.random.seed(seed)
    env = CubeToBox(robots="Panda", controller_configs=ctrl, has_renderer=False, has_offscreen_renderer=True,
                    use_camera_obs=False, use_object_obs=False, control_freq=20, horizon=100000,
                    ignore_done=True, box_xy=box_xy, initialization_noise=None)
    env.reset()
    env.set_cube_pose(T_L_O0)
    env.sim.forward()
    site = env.sim.model.site_name2id(env.robots[0].controller.eef_name)
    grip_idx = env.robots[0]._ref_gripper_joint_pos_indexes
    policy.reset()

    success_at, max_lift = None, 0.0
    for k in range(max_steps):
        p = env.sim.data.site_xpos[site].copy()
        R = env.sim.data.site_xmat[site].reshape(3, 3).copy()
        state = np.r_[p, Rotation.from_matrix(R).as_rotvec(), env.sim.data.qpos[grip_idx]].astype(np.float32)
        a = policy.act(obs_images(env), state, TASK)
        env.step(np.clip(a, -1.0, 1.0))
        max_lift = max(max_lift, float(env.cube_pose()[2, 3] - TABLE_Z))
        if success_at is None and env.success():
            success_at = k
            break
    ok = bool(env.success())
    final = env.cube_pose()[:3, 3].copy()
    env.close()
    return {"success": ok, "success_step": success_at, "steps": k + 1,
            "max_cube_lift_m": max_lift,
            "final_cube_xy_err_m": float(np.linalg.norm(final[:2] - box_xy))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("checkpoint", help="a .../checkpoints/<step> dir, or its pretrained_model/")
    ap.add_argument("--per-position", type=int, default=2, help="start poses per cube position")
    ap.add_argument("--max-steps", type=int, default=400, help="20 s at 20 Hz")
    ap.add_argument("--tlw", help="json with the T_L_W the checkpoint was trained under")
    ap.add_argument("--box-xy", nargs=2, type=float, help="box centre in L, as trained")
    ap.add_argument("--as-trained", help="a sim_episodes index.json to read T_L_W and box_xy_L from")
    ap.add_argument("--out", default=None)
    ap.add_argument("--eval-python", default=str(ROOT / ".venv-eval/bin/python"))
    a = ap.parse_args()

    ck = Path(a.checkpoint)
    if (ck / "pretrained_model").is_dir():
        ck = ck / "pretrained_model"

    T_L_W = load_T_L_W(a.session, a.tlw)
    box_xy = np.array(a.box_xy, float) if a.box_xy else None
    if a.as_trained:
        idx = json.loads(Path(a.as_trained).read_text())
        if "T_L_W" in idx:
            T_L_W = np.array(idx["T_L_W"], float)
        elif not a.tlw:
            T_L_W = load_T_L_W()  # pre-dates per-session frames: the file's default is what it used
        box_xy = np.array(idx["box_xy_L"], float)
    if box_xy is None:
        raise SystemExit("need --box-xy or --as-trained: the box must sit where the policy was trained")

    # one start pose per demo, grouped by the session's labelled cube positions
    proc = ROOT / "data/processed" / a.session
    rows = [r for r in csv.DictReader(open(proc / "selection.csv")) if r["position"]]
    starts = {}
    for r in sorted(rows, key=lambda r: -float(r["score"])):
        f = proc / f"{r['clip']}_20hz.npz"
        if len(starts.setdefault(r["position"], [])) < a.per_position and f.exists():
            starts[r["position"]].append((r["clip"], T_L_W @ load_demo(proc / f"{r['clip']}_20hz")["T_W_O"][0]))

    print(f"checkpoint {ck}")
    print(f"T_L_W translation {np.round(T_L_W[:3, 3], 4).tolist()}, box_xy {np.round(box_xy, 4).tolist()}")
    print(f"{sum(len(v) for v in starts.values())} rollouts, max {a.max_steps} steps each\n")

    policy = Policy(ck, Path(a.eval_python))
    results, t0 = {}, time.time()
    try:
        for pos in sorted(starts):
            results[pos] = []
            for clip, T0 in starts[pos]:
                t1 = time.time()
                r = rollout(policy, T0, box_xy, a.max_steps)
                r.update(position=pos, clip=clip, wall_s=round(time.time() - t1, 1))
                results[pos].append(r)
                print(f"  {pos} {clip}: {'SUCCESS' if r['success'] else 'failed '} "
                      f"{r['steps']:3d} steps, lift {r['max_cube_lift_m'] * 100:5.1f} cm, "
                      f"final {r['final_cube_xy_err_m'] * 100:5.1f} cm from the box  [{r['wall_s']}s]")
    finally:
        policy.close()

    flat = [r for v in results.values() for r in v]
    summary = {
        "checkpoint": str(ck), "session": a.session, "T_L_W": T_L_W.tolist(), "box_xy_L": box_xy.tolist(),
        "max_steps": a.max_steps, "wall_s": round(time.time() - t0, 1),
        "success_rate": round(float(np.mean([r["success"] for r in flat])), 4),
        "n": len(flat),
        "by_position": {p: {"success": int(sum(r["success"] for r in v)), "n": len(v)} for p, v in results.items()},
        "rollouts": flat,
    }
    out = Path(a.out) if a.out else proc / "eval" / f"eval_{ck.parent.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n{summary['success_rate']:.0%} in the box ({sum(r['success'] for r in flat)}/{len(flat)})  "
          f"-> {out}   [{summary['wall_s']}s]")


if __name__ == "__main__":
    main()
