import numpy as np
from scipy.spatial.transform import Rotation

from ego2libero import se3
from ego2libero.align.hands_to_world import HandTrackW, hands_to_world
from ego2libero.hands.hawor_io import HandTrack
from conftest import look_at, random_pose


def _track(rng, n=40, s_true=1.0, cam_motion=0.0):
    T_W_S = random_pose(rng)
    eyes = np.c_[0.3 + cam_motion * np.sin(np.linspace(0, 3, n)), np.full(n, 0.1), np.full(n, 0.55)]
    T_W_C = np.stack([look_at(e, [0.3, 0.1, 0.0], up=(0, 1, 0)) for e in eyes])
    # hand in W: moves above the table
    p = np.c_[0.25 + 0.1 * np.linspace(0, 1, n), np.full(n, 0.05), 0.1 + 0.05 * np.sin(np.linspace(0, 3, n))]
    T_W_H = se3.make_T(Rotation.from_rotvec(np.c_[np.zeros(n), np.linspace(0, 1, n), np.zeros(n)]).as_matrix(), p)
    joints_H = rng.normal(scale=0.03, size=(21, 3))
    joints_H[0] = 0
    joints_W = np.einsum("nij,kj->nki", T_W_H[:, :3, :3], joints_H) + T_W_H[:, None, :3, 3]
    # HaWoR's view: S = W mapped by inverse similarity (x_S = R^T (x_W - t) / s)
    to_S = lambda X: (X - T_W_S[:3, 3]) @ T_W_S[:3, :3] / s_true
    T_S_C = se3.make_T(T_W_S[:3, :3].T @ T_W_C[:, :3, :3], to_S(T_W_C[:, :3, 3]))
    T_S_H = se3.make_T(T_W_S[:3, :3].T @ T_W_H[:, :3, :3], to_S(T_W_H[:, :3, 3]))
    obs = np.ones(n, bool)
    obs[10:15] = False
    tr = HandTrack(T_S_C, T_S_H, to_S(joints_W), np.zeros((n, 48)), np.zeros(10), obs, 1384.0,
                   np.array([960.0, 540.0]), np.array([1920, 1080]), 1.0)
    return tr, T_W_C, T_W_H, joints_W


def test_static_camera_recovers_hand_in_world(rng, tmp_path):
    tr, T_W_C, T_W_H, joints_W = _track(rng)
    T_W_C_obs = T_W_C.copy()
    T_W_C_obs[20:25] = np.nan  # board pose missing: falls back to the global Sim(3)
    h = hands_to_world(tr, T_W_C_obs)
    assert h.sim.scale_fixed and h.sim.s == 1.0
    np.testing.assert_allclose(h.T_W_H, T_W_H, atol=1e-9)
    np.testing.assert_allclose(h.joints_W, joints_W, atol=1e-9)
    assert not h.anchored_on_board[20:25].any() and h.anchored_on_board[:20].all()
    assert (h.hand_source[10:15] == 2).all() and (h.hand_source[:10] == 0).all()
    h.save(tmp_path / "h.npz")
    np.testing.assert_allclose(HandTrackW.load(tmp_path / "h.npz").T_W_H, h.T_W_H)


def test_moving_camera_with_scale(rng):
    tr, T_W_C, T_W_H, joints_W = _track(rng, s_true=1.4, cam_motion=0.4)
    h = hands_to_world(tr, T_W_C)
    assert not h.sim.scale_fixed and abs(h.sim.s - 1.4) < 1e-6
    np.testing.assert_allclose(h.T_W_H, T_W_H, atol=1e-8)
    np.testing.assert_allclose(h.joints_W, joints_W, atol=1e-8)
