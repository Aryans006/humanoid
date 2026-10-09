import numpy as np
from scipy.spatial.transform import Rotation

from ego2libero import se3
from ego2libero.sim.retarget import retarget


def _demo(n=200, cube_yaw=0.3):
    t = np.arange(n) / 20
    held = (t >= 3) & (t < 7)
    s = np.clip((t - 3) / 4, 0, 1)
    cube_p = np.c_[0.2 + 0.1 * s, np.full(n, 0.1), 0.025 + 0.08 * np.sin(np.pi * s)]
    T_W_O = se3.make_T(Rotation.from_euler("z", np.full((n, 1), cube_yaw)).as_matrix(), cube_p)
    J = np.zeros((n, 21, 3))
    c = cube_p + [0.0, 0.0, 0.0]
    axis = np.array([np.cos(cube_yaw + 0.15), np.sin(cube_yaw + 0.15), 0])  # pinch ~ along cube x (+noise)
    J[:, 4] = c - 0.03 * axis + [0, -0.02, 0.01]  # fingertips off by a couple of cm, like HaWoR
    J[:, 8] = c + 0.03 * axis + [0, -0.02, 0.01]
    src = np.zeros(n, np.uint8)
    src[:10] = 2
    return dict(t=t, T_W_O=T_W_O, joints_W=J, hand_source=src, gripper_closed=held)


def test_held_tool_point_on_cube_centre_and_pinch_axis():
    d = _demo()
    tg = retarget(d, np.eye(4), table_z=0.0)
    np.testing.assert_allclose(tg.pos_L[tg.held], d["T_W_O"][tg.held, :3, 3], atol=1e-12)
    # yaw closes across the cube's x faces (cube yaw 0.3), not the noisy 0.45 pinch
    ang = np.abs(((tg.yaw[tg.held] - 0.3) + np.pi / 2) % np.pi - np.pi / 2)
    assert ang.max() < 1e-9
    assert (tg.grip[tg.held] == 1).all() and (tg.grip[~tg.held] == -1).all()
    assert not tg.hand_ok[:10].any()  # infilled frames replaced by the nearest good one
    np.testing.assert_allclose(tg.pos_L[:10], np.broadcast_to(tg.pos_L[10], (10, 3)), atol=1e-12)


def test_targets_are_continuous_and_above_table():
    d = _demo()
    d["joints_W"][50:52] = [[5.0, -3.0, -0.6]]  # HaWoR glitch marked observed
    tg = retarget(d, np.eye(4), table_z=0.0)
    step = np.linalg.norm(np.diff(tg.pos_L, axis=0), axis=1)
    assert step.max() < 0.03  # no jumps between 20 Hz samples
    assert (tg.pos_L[:, 2] >= 0.012 - 1e-12).all()
    assert np.abs(np.diff(tg.yaw)).max() < 0.5


def test_yaw_centred_on_home_orientation():
    d = _demo(cube_yaw=3.0)  # pinch axis near 180 deg: equivalent to near 0 for a symmetric gripper
    tg = retarget(d, np.eye(4), table_z=0.0)
    assert abs(np.median(tg.yaw)) <= np.pi / 2


def test_tabletop_frame_places_centre_on_the_table_and_sessions_override():
    from ego2libero.sim.frame import load_T_L_W, tabletop_T_L_W

    T = tabletop_T_L_W([0.3, 0.1], rot_deg=180)
    np.testing.assert_allclose(T @ [0.3, 0.1, 0.0, 1.0], [0, 0, 0.8, 1], atol=1e-12)
    np.testing.assert_allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(T[:3, :3] @ [1, 0, 0], [-1, 0, 0], atol=1e-12)  # W's x -> L's -x

    cfg = load_T_L_W("2026-10-07_dataset")
    assert not np.allclose(cfg, load_T_L_W()), "the dataset session must not use the pilots' frame"
    np.testing.assert_allclose(load_T_L_W("no_such_session"), load_T_L_W(), atol=0)
