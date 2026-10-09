import numpy as np
from scipy.spatial.transform import Rotation

from ego2libero import se3
from ego2libero.align.sim3 import align_sim3, relative_scaled
from conftest import random_pose


def _scenario(rng, n=300, s_true=1.37, motion=0.4):
    T_W_S = random_pose(rng)
    R, t = T_W_S[:3, :3], T_W_S[:3, 3]
    # Head-like camera path in W: smooth wander of `motion` meters plus rotation.
    u = np.linspace(0, 2 * np.pi, n)
    c_W = np.c_[motion * np.sin(u), 0.5 * motion * np.sin(2 * u), 0.5 + 0.1 * motion * np.cos(u)]
    R_W_C = Rotation.from_rotvec(np.c_[0.3 * np.sin(u), 0.2 * np.cos(u), 0.5 * u / np.pi]).as_matrix()
    T_W_C = se3.make_T(R_W_C, c_W)
    # Same cameras seen in S: x_W = s R x_S + t  =>  x_S = R^T (x_W - t) / s
    c_S = (c_W - t) @ R / s_true
    T_S_C = se3.make_T(R.T @ R_W_C, c_S)
    return T_W_C, T_S_C, s_true, R, t


def test_recovers_known_similarity_under_noise(rng):
    T_W_C, T_S_C, s, R, t = _scenario(rng)
    noisy = T_W_C.copy()
    noisy[:, :3, 3] += rng.normal(scale=0.003, size=(len(noisy), 3))
    noisy[:, :3, :3] = noisy[:, :3, :3] @ Rotation.from_rotvec(
        rng.normal(scale=np.radians(0.3), size=(len(noisy), 3))).as_matrix()
    noisy[::17] = np.nan  # missing board poses are ignored
    sim = align_sim3(noisy, T_S_C)
    assert abs(sim.s - s) / s < 0.01
    assert np.degrees(se3.rotation_distance(sim.R, R)) < 0.1
    assert np.linalg.norm(sim.t - t) < 0.005
    assert not sim.scale_fixed
    assert sim.summary()["center_residual_rms_m"] < 0.01


def test_exact_mapping_of_poses(rng):
    T_W_C, T_S_C, *_ = _scenario(rng)
    sim = align_sim3(T_W_C, T_S_C)
    np.testing.assert_allclose(sim.apply_pose(T_S_C), T_W_C, atol=1e-9)
    # Hanging a tracked frame off the tag camera equals the Sim(3) mapping when exact.
    T_S_H = T_S_C @ random_pose(rng, 0.3)
    via_cam = T_W_C @ relative_scaled(T_S_C, T_S_H, sim.s)
    np.testing.assert_allclose(via_cam, sim.apply_pose(T_S_H), atol=1e-9)


def test_small_spread_fixes_scale(rng):
    T_W_C, T_S_C, *_ = _scenario(rng, motion=0.01, s_true=1.0)
    sim = align_sim3(T_W_C, T_S_C)
    assert sim.scale_fixed and sim.s == 1.0
    np.testing.assert_allclose(sim.apply_pose(T_S_C), T_W_C, atol=1e-6)


def test_static_camera_with_slam_drift_fixes_scale(rng):
    """Static real camera, but the SLAM track wanders several cm: scale must not collapse."""
    T_W_C, T_S_C, s, R, t = _scenario(rng, motion=0.001, s_true=1.0)
    drift = np.cumsum(rng.normal(scale=0.008, size=(len(T_S_C), 3)), axis=0)
    T_S_C = T_S_C.copy()
    T_S_C[:, :3, 3] += drift
    assert np.sqrt(((drift - drift.mean(0)) ** 2).sum(1).mean()) > 0.05
    sim = align_sim3(T_W_C, T_S_C)
    assert sim.scale_fixed and sim.s == 1.0
