"""Turning retargeted targets into what the robot actually executes.

The demos are human: the person lifts within a frame or two of touching the cube. Replaying the
recorded gripper command one-for-one closes the fingers while the arm is still descending, so the
Panda grips air or catches one corner -- the main reason only 34 of 53 accepted demos recorded a
successful sim episode.

``GraspSchedule`` is a closed loop over the target frames. At every frame where the demo asks the
gripper to close it stops advancing, keeps commanding that frame's pose with the fingers still
open until the arm has really arrived, then closes and holds still long enough for the fingers to
seat. Only then does the demo resume. Nothing is faked: the caller records the actions it actually
sent, dwell steps included.

"Arrived" is the arm coming to rest at its commanded pose, not a fixed distance to the cube.
Measured on this dataset, OSC_POSE settles 1.2-1.9 cm short of the commanded tool point (gravity
and joint limits), so a 1 cm absolute test never passes and the wait just burns its timeout; and
the open gripper often nudges the cube 2-4 cm on the way down, so a test against the cube's live
position can be moving away faster than the arm closes on it. Both are visible in the per-grasp
record, which keeps the residuals so a bad grasp can be told from a slow one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GraspSchedule:
    """Drive with ``k, g = sched.next(tool_pos, cube_pos)`` until it returns ``None``; ``k`` is the
    target frame to command and ``g`` the gripper command (-1 open, +1 closed) to send with it."""

    grip: np.ndarray  # (K,) per-frame gripper command from retarget, -1 open / +1 closed
    pos_L: np.ndarray  # (K, 3) per-frame tool-point target, to measure arrival against
    k0: int = 0
    k1: int | None = None  # exclusive; default len(grip)
    fps: float = 20.0
    reach_m: float = 0.01  # arrived once the tool point is this close to its commanded target
    settle_m: float = 0.002  # ... or once the tool has moved less than this over settle_s
    settle_s: float = 0.3
    reach_timeout_s: float = 1.5  # ... and in any case give up waiting after this long
    close_hold_s: float = 0.5  # hold the pose this long while the fingers seat
    enabled: bool = True  # False replays the recorded timing as-is (for A/B comparison)
    grasps: list[dict] = field(default_factory=list)  # one record per grasp, for metrics

    def __post_init__(self) -> None:
        self.grip = np.asarray(self.grip, float)
        self.pos_L = np.asarray(self.pos_L, float)
        if self.k1 is None:
            self.k1 = len(self.grip)
        self._k = self.k0
        self._phase = "run"
        self._n = 0
        self._last_g = -1.0
        self._trail: list[np.ndarray] = []  # recent tool positions, for the settle test
        self._reach_max = max(1, int(round(self.reach_timeout_s * self.fps)))
        self._settle_n = max(1, int(round(self.settle_s * self.fps)))
        self._hold_max = max(1, int(round(self.close_hold_s * self.fps)))

    @property
    def extra_steps(self) -> int:
        """How many steps the dwells added on top of the demo's own frames."""
        return sum(g["reach_steps"] + g["hold_steps"] for g in self.grasps)

    def next(self, tool_pos, cube_pos) -> tuple[int, float] | None:
        if self._phase == "run":
            if self._k >= self.k1:
                return None
            g = float(self.grip[self._k])
            if g > 0 and self._last_g <= 0 and self.enabled:
                self._phase, self._n, self._trail = "reach", 0, []  # grasp onset: stop the clock
            else:
                return self._advance(g)

        if self._phase == "reach":
            tool = np.asarray(tool_pos, float)
            err = float(np.linalg.norm(tool - self.pos_L[self._k]))
            self._trail.append(tool)
            reason = ("reached" if err <= self.reach_m
                      else "settled" if (len(self._trail) > self._settle_n
                                         and np.linalg.norm(tool - self._trail[-self._settle_n - 1]) < self.settle_m)
                      else "timeout" if self._n >= self._reach_max else None)
            if reason is None:
                self._n += 1
                return self._k, -1.0  # same pose target, fingers still open
            self.grasps.append({"frame": int(self._k), "reach_steps": self._n, "reason": reason,
                                "tool_target_err_m": err, "hold_steps": self._hold_max,
                                "tool_cube_dist_m": float(np.linalg.norm(tool - np.asarray(cube_pos, float)))})
            self._phase, self._n = "close", 0

        # "close": hold the pose with the fingers closing, then let the demo run on
        if self._n < self._hold_max:
            self._n += 1
            return self._k, 1.0
        self._phase = "run"
        return self._advance(1.0)

    def _advance(self, g: float) -> tuple[int, float]:
        k, self._k, self._last_g = self._k, self._k + 1, g
        return k, g
