"""Tabletop task: Panda, table, red 50 mm cube, open-top box. Pick the cube, place it in the box.
Runs in .venv-sim (robosuite 1.4.1, the engine and robot/controller setup LIBERO uses).

Built on robosuite's Lift task (the same Panda, table and OSC controller LIBERO uses), with the
cube size fixed to the real cube and deterministic placement.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
from robosuite.environments.manipulation.lift import Lift
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from scipy.spatial.transform import Rotation


def _fmt(v):
    return " ".join(f"{x:.6f}" for x in np.ravel(v))


def mj_quat(R: np.ndarray) -> np.ndarray:
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


class CubeToBox(Lift):
    """Pick the red cube and place it in the open-top box.

    The box is static (welded to the world), so the policy only has to get the cube into it.
    ``box_xy`` is its centre on the table in L; ``success()`` = cube resting inside the walls."""

    def __init__(self, *args, cube_half=0.025, cube_density=100.0, box_xy=(0.0, 0.0), box_inner=0.10,
                 box_wall_h=0.03, box_wall_t=0.005, cameras=None, **kw):
        self.cube_half = cube_half
        self.cube_density = cube_density  # 100 kg/m^3: ~12.5 g, like the hollow paper cube
        self.box_xy = np.asarray(box_xy, float)
        self.box_inner, self.box_wall_h, self.box_wall_t = box_inner, box_wall_h, box_wall_t
        self.extra_cams = cameras or {}  # name -> dict(pos, R_cv (OpenCV camera axes in L), fovy_deg)
        super().__init__(*args, **kw)

    def _load_model(self):
        super(Lift, self)._load_model()
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        arena = TableArena(table_full_size=self.table_full_size, table_friction=self.table_friction,
                           table_offset=self.table_offset)
        arena.set_origin([0, 0, 0])
        h = self.cube_half
        self.cube = BoxObject(name="cube", size=[h, h, h], rgba=[0.85, 0.05, 0.05, 1], density=self.cube_density,
                              friction=[1.0, 0.005, 0.0001])
        self.placement_initializer = UniformRandomSampler(
            name="ObjectSampler", mujoco_objects=self.cube, x_range=[0, 0], y_range=[0, 0], rotation=0.0,
            ensure_object_boundary_in_range=False, ensure_valid_placement=False,
            reference_pos=self.table_offset, z_offset=0.0)
        self.model = ManipulationTask(mujoco_arena=arena, mujoco_robots=[r.robot_model for r in self.robots],
                                      mujoco_objects=self.cube)
        self._add_box()
        for name, c in self.extra_cams.items():
            R_mj = c["R_cv"] @ np.diag([1.0, -1.0, -1.0])  # MuJoCo cameras look along -z with y up
            ET.SubElement(self.model.worldbody, "camera", name=name, mode="fixed",
                          pos=_fmt(c["pos"]), quat=_fmt(mj_quat(R_mj)), fovy=f"{c['fovy_deg']:.4f}")

    def _add_box(self):
        """Open-top box: a thin floor and four walls, as one static body on the table."""
        z0 = self.table_offset[2]
        a, hgt, t = self.box_inner / 2, self.box_wall_h, self.box_wall_t
        body = ET.SubElement(self.model.worldbody, "body", name="target_box",
                             pos=_fmt([self.box_xy[0], self.box_xy[1], z0]))
        rgba = "0.25 0.45 0.75 1"
        parts = {"floor": ([0, 0, t / 2], [a + t, a + t, t / 2]),
                 "wall_px": ([a + t / 2, 0, hgt / 2], [t / 2, a + t, hgt / 2]),
                 "wall_nx": ([-a - t / 2, 0, hgt / 2], [t / 2, a + t, hgt / 2]),
                 "wall_py": ([0, a + t / 2, hgt / 2], [a, t / 2, hgt / 2]),
                 "wall_ny": ([0, -a - t / 2, hgt / 2], [a, t / 2, hgt / 2])}
        for name, (pos, size) in parts.items():
            ET.SubElement(body, "geom", name=f"box_{name}", type="box", pos=_fmt(pos), size=_fmt(size),
                          rgba=rgba, friction="1 0.005 0.0001", group="1")  # group 1 = drawn; still collides

    def success(self, tol: float = 0.002) -> bool:
        """Cube centre within the inner walls, below the rim, and nearly at rest."""
        p = self.cube_pose()[:3, 3]
        d = np.abs(p[:2] - self.box_xy)
        inside = np.all(d < self.box_inner / 2 - self.cube_half * 0.5 + tol)
        low = p[2] < self.table_offset[2] + self.box_wall_t + 2 * self.cube_half + 0.01
        still = np.linalg.norm(self.sim.data.get_joint_qvel(self.cube.joints[0])[:3]) < 0.05
        return bool(inside and low and still)

    def set_cube_pose(self, T_L_O: np.ndarray, zero_vel: bool = True):
        q = np.r_[T_L_O[:3, 3], mj_quat(T_L_O[:3, :3])]
        self.sim.data.set_joint_qpos(self.cube.joints[0], q)
        if zero_vel:
            self.sim.data.set_joint_qvel(self.cube.joints[0], np.zeros(6))

    def cube_pose(self) -> np.ndarray:
        q = self.sim.data.get_joint_qpos(self.cube.joints[0])
        T = np.eye(4)
        T[:3, 3] = q[:3]
        T[:3, :3] = Rotation.from_quat([q[4], q[5], q[6], q[3]]).as_matrix()
        return T
