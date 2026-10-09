"""Residual RL on top of a frozen SmolVLA policy, in the CubeToBox task.

The base policy is left exactly as trained and supplies an action ``a_base`` at 20 Hz; a learned
residual nudges it:

    a = clip(a_base + alpha * a_res, -1, 1)        alpha = 0.2 by default

With robosuite's OSC_POSE output limits that is at most 1 cm of position and 5.7 degrees of
rotation per step, so the base stays in charge while the residual accumulates real authority over
an approach (~10 cm across ten steps). Measured on checkpoint 026000, 16 of 21 failures never got
the cube off the table and missed the grasp by 1-3 cm, which is inside that envelope.

Two things shape the design:

* **The base replays 50-step chunks open-loop.** It only looks at the cameras on a chunk boundary,
  so within a chunk the residual carries all closed-loop correction. That also means the cameras
  only need rendering every 50 steps, which is most of what makes this affordable on CPU.
* **The residual sees privileged sim state, not pixels** (``observation``, 27-D). It is a sim
  policy, a small MLP over it costs microseconds, and it reacts every step where the base cannot.

``observation`` and ``reward`` are pure numpy so they can be tested without robosuite; the env
itself needs .venv-sim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

TABLE_Z = 0.8
CUBE_HALF = 0.025
LIFT_M = 0.04  # cube centre this far above its resting height counts as "off the table"
OBS_DIM = 27


DIM_SETS = {"pos": (0, 1, 2), "pose": (0, 1, 2, 3, 4, 5), "all": tuple(range(7))}


@dataclass
class ResidualConfig:
    # Which of the 7 action dims the residual may touch. Default: position only.
    #
    # The orientation is not the policy's to get wrong -- retargeting fixes the gripper top-down
    # with a yaw taken from the pinched cube faces, and the open/close timing is the base policy's.
    # The diagnosed failure is *positional*: the grasp misses by 1-3 cm. Handing RL the rotation and
    # gripper channels only gives it rope. Measured on the first runs, SAC learned a sustained
    # +0.33 residual on rotation-y, which at alpha = 0.2 is ~1.9 deg of tilt per step accumulating
    # across an approach; the gripper stopped pointing down and lifted the cube in 2% of episodes
    # against the base policy's own ~45%. Position-only cuts the action space from 7 to 3 and
    # removes that failure mode by construction.
    dims: tuple[int, ...] = DIM_SETS["pos"]
    alpha: float = 0.2  # residual authority as a fraction of the action range
    max_steps: int = 400  # 20 s at 20 Hz
    # reward weights, tilted toward the reach/grasp phase where the failures are
    w_reach: float = 1.0  # potential per metre of tool-to-cube distance, before the lift
    w_lift: float = 5.0  # potential gap between "on the table" and "in hand"
    w_place: float = 2.0  # potential per metre of cube-to-box distance
    w_success: float = 20.0
    # no explicit drop penalty: potential-based shaping charges for it automatically
    w_effort: float = 0.05  # per unit of ||a_res||^2, keeps the residual small

    def __post_init__(self):
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")


def observation(tool_pos, tool_rot, grip_qpos, cube_pos, cube_rot, box_xy, a_base, lifted) -> np.ndarray:
    """The residual's 27-D view. Differences are given explicitly so a small MLP does not have to
    learn subtraction, and the base action is included because the residual corrects *it*."""
    tool_pos, cube_pos = np.asarray(tool_pos, float), np.asarray(cube_pos, float)
    box_xy = np.asarray(box_xy, float)
    return np.concatenate([
        tool_pos - [0, 0, TABLE_Z], np.asarray(tool_rot, float), np.asarray(grip_qpos, float),
        cube_pos - [0, 0, TABLE_Z], np.asarray(cube_rot, float),
        cube_pos - tool_pos,  # the grasp signal
        cube_pos[:2] - box_xy,
        np.asarray(a_base, float), [float(lifted)],
    ]).astype(np.float32)


def potential(cfg: ResidualConfig, tool_pos, cube_pos, box_xy, lifted) -> float:
    """Phi(s): higher is closer to done. Reach the cube, get it off the table, carry it to the box."""
    tool_pos, cube_pos = np.asarray(tool_pos, float), np.asarray(cube_pos, float)
    d_box = float(np.linalg.norm(cube_pos[:2] - np.asarray(box_xy, float)))
    phi = -cfg.w_place * d_box
    if not lifted:
        phi -= cfg.w_reach * float(np.linalg.norm(tool_pos - cube_pos)) + cfg.w_lift
    return phi


def reward(cfg: ResidualConfig, phi_prev: float, phi_next: float, a_res,
           success: bool, gamma: float = 0.99) -> tuple[float, dict]:
    """Potential-based shaping plus the sparse task reward.

    The first version was a plain distance penalty that switched from tool-to-cube to cube-to-box on
    the lift. That made progress *cost* money: hovering near the cube ran at w_reach * 0.05 = 0.05
    per step, while carrying ran at w_place * 0.2 = 0.4 per step, so lifting raised the running cost
    eightfold against a one-off +5 bonus. Over a 400-step episode the two strategies came out within
    2.5 points of each other, and SAC duly learned to hover: return climbed from -182 to -119 while
    the lift rate fell from 40% to 5%.

    Potential-based shaping (Ng, Harada & Russell 1999) removes that whole class of bug. The shaping
    term telescopes over an episode to gamma^T * Phi(s_T) - Phi(s_0), so it cannot be farmed by
    standing still, and the optimal policy is provably unchanged -- only the learning signal is
    denser. Dropping the cube needs no special penalty either: ``lifted`` goes false, Phi falls by
    w_lift, and the agent is charged for it automatically."""
    shaped = gamma * phi_next - phi_prev
    terms = {"shaping": shaped, "effort": -cfg.w_effort * float(np.sum(np.square(a_res)))}
    if success:
        terms["success"] = cfg.w_success
    return float(sum(terms.values())), terms


class ResidualCubeToBox:
    """CubeToBox driven by ``base`` with a learned residual. Needs robosuite (.venv-sim).

    ``base`` must expose ``chunk(images: dict, state: np.ndarray, task: str) -> (n, 7)``;
    it is queried only when the current chunk runs out, and the cameras are only rendered then.
    """

    def __init__(self, base, starts, box_xy, cfg: ResidualConfig | None = None, task: str = "",
                 seed: int = 0, render_size: int = 256):
        from robosuite import load_controller_config

        from ego2libero.sim.scene import CubeToBox

        self.base, self.starts, self.cfg = base, list(starts), cfg or ResidualConfig()
        self.box_xy = np.asarray(box_xy, float)
        self.task, self.render_size = task, render_size
        self.rng = np.random.default_rng(seed)
        np.random.seed(seed)
        ctrl = load_controller_config(default_controller="OSC_POSE")
        self.env = CubeToBox(robots="Panda", controller_configs=ctrl, has_renderer=False,
                             has_offscreen_renderer=True, use_camera_obs=False, use_object_obs=False,
                             control_freq=20, horizon=100_000, ignore_done=True, box_xy=self.box_xy,
                             initialization_noise=None)
        self.env.reset()
        self._site = self.env.sim.model.site_name2id(self.env.robots[0].controller.eef_name)
        self._grip = self.env.robots[0]._ref_gripper_joint_pos_indexes
        self._chunk: np.ndarray | None = None

    # --- sim readouts -------------------------------------------------------
    def _tool(self):
        p = self.env.sim.data.site_xpos[self._site].copy()
        R = self.env.sim.data.site_xmat[self._site].reshape(3, 3).copy()
        return p, Rotation.from_matrix(R).as_rotvec()

    def _cube(self):
        T = self.env.cube_pose()
        return T[:3, 3].copy(), Rotation.from_matrix(T[:3, :3]).as_rotvec()

    def _images(self):
        s = self.render_size
        return {"camera1": self.env.sim.render(camera_name="agentview", width=s, height=s)[::-1].copy(),
                "camera2": self.env.sim.render(camera_name="robot0_eye_in_hand", width=s, height=s)[::-1].copy(),
                "camera3": self.env.sim.render(camera_name="sideview", width=s, height=s)[::-1].copy()}

    def _base_state(self):
        p, rv = self._tool()
        return np.r_[p, rv, self.env.sim.data.qpos[self._grip]].astype(np.float32)

    def _next_base_action(self) -> np.ndarray:
        """Pop the next base action, re-querying (and only then rendering) at a chunk boundary."""
        if self._chunk is None or self._k_chunk >= len(self._chunk):
            self._chunk = np.asarray(self.base.chunk(self._images(), self._base_state(), self.task), float)
            self._k_chunk = 0
            self.n_base_calls += 1
        a = self._chunk[self._k_chunk]
        self._k_chunk += 1
        return a

    # --- gym-ish API --------------------------------------------------------
    def reset(self, start: int | None = None) -> np.ndarray:
        i = self.rng.integers(len(self.starts)) if start is None else start
        self.start_idx = int(i)
        self.env.reset()
        self.env.set_cube_pose(self.starts[self.start_idx])
        self.env.sim.forward()
        self.base.reset()
        self._chunk, self._k_chunk = None, 0
        self.t, self.was_lifted, self.n_base_calls, self.succeeded = 0, False, 0, False
        self._a_base = self._next_base_action()
        return self._obs()

    def _obs(self):
        p, rv = self._tool()
        c, crv = self._cube()
        return observation(p, rv, self.env.sim.data.qpos[self._grip], c, crv, self.box_xy,
                           self._a_base, self._lifted())

    def _lifted(self):
        return bool(self.env.cube_pose()[2, 3] - TABLE_Z > CUBE_HALF + LIFT_M)

    @property
    def act_dim(self) -> int:
        return len(self.cfg.dims)

    def step(self, a_res: np.ndarray):
        a_res = np.clip(np.asarray(a_res, float), -1.0, 1.0)
        full = np.zeros(7)
        full[list(self.cfg.dims)] = a_res  # untouched channels stay exactly the base policy's
        a = np.clip(self._a_base + self.cfg.alpha * full, -1.0, 1.0)
        was = self.was_lifted
        p0, _ = self._tool()
        c0, _ = self._cube()
        phi_prev = potential(self.cfg, p0, c0, self.box_xy, was)
        self.env.step(a)
        self.t += 1
        lift = self._lifted()
        first = lift and not was
        self.was_lifted = was or lift
        success = bool(self.env.success())
        self.succeeded |= success
        p, _ = self._tool()
        c, _ = self._cube()
        phi_next = potential(self.cfg, p, c, self.box_xy, self.was_lifted)
        r, terms = reward(self.cfg, phi_prev, phi_next, full, success)
        done = success or self.t >= self.cfg.max_steps
        if not done:
            self._a_base = self._next_base_action()
        info = {"success": success, "lifted": self.was_lifted, "terms": terms,
                "start": self.start_idx, "steps": self.t, "base_calls": self.n_base_calls,
                "cube_box_dist_m": float(np.linalg.norm(c[:2] - self.box_xy))}
        return self._obs(), r, done, info

    def close(self):
        self.env.close()
