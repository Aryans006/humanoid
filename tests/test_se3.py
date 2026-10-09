import numpy as np
from scipy.spatial.transform import Rotation

from ego2libero import se3
from conftest import random_pose


def test_inverse_and_compose(rng):
    for _ in range(20):
        A, B = random_pose(rng), random_pose(rng)
        np.testing.assert_allclose(se3.inv(A) @ A, np.eye(4), atol=1e-12)
        np.testing.assert_allclose(se3.inv(A @ B), se3.inv(B) @ se3.inv(A), atol=1e-12)
        np.testing.assert_allclose(se3.compose(A, B, se3.inv(B)), A, atol=1e-12)


def test_batched_inverse(rng):
    T = np.stack([random_pose(rng) for _ in range(7)])
    np.testing.assert_allclose(se3.inv(T) @ T, np.tile(np.eye(4), (7, 1, 1)), atol=1e-12)


def test_transform_points(rng):
    T = random_pose(rng)
    p = rng.normal(size=(10, 3))
    ph = np.c_[p, np.ones(10)]
    np.testing.assert_allclose(se3.transform_points(T, p), (ph @ T.T)[:, :3], atol=1e-12)


def test_so3_round_trip(rng):
    w = rng.normal(size=(50, 3))
    w *= (rng.uniform(0, np.pi - 1e-3, 50) / np.linalg.norm(w, axis=1))[:, None]
    np.testing.assert_allclose(se3.so3_log(se3.so3_exp(w)), w, atol=1e-9)


def test_se3_round_trip(rng):
    for _ in range(50):
        T = random_pose(rng)
        if se3.rotation_angle(T[:3, :3]) > np.pi - 1e-3:
            continue
        np.testing.assert_allclose(se3.se3_exp(se3.se3_log(T)), T, atol=1e-9)
    xi = np.r_[rng.normal(size=3), 1e-9 * rng.normal(size=3)]  # small-angle branch
    np.testing.assert_allclose(se3.se3_log(se3.se3_exp(xi)), xi, atol=1e-9)


def test_quaternion_continuity(rng):
    R = Rotation.from_rotvec(np.linspace(0, 1, 30)[:, None] * [0, 0, 2 * np.pi - 0.1]).as_matrix()
    q = se3.to_quat_xyzw(R)
    assert np.all(np.sum(q[1:] * q[:-1], axis=1) > 0)
    np.testing.assert_allclose(se3.from_quat_xyzw(q), R, atol=1e-12)


def test_interpolation_endpoints_and_continuity(rng):
    A, B = random_pose(rng), random_pose(rng)
    np.testing.assert_allclose(se3.interpolate_pair(A, B, 0.0), A, atol=1e-9)
    np.testing.assert_allclose(se3.interpolate_pair(A, B, 1.0), B, atol=1e-9)
    alphas = np.linspace(0, 1, 201)
    path = se3.interpolate_pair(A, B, alphas)
    step_rot = se3.rotation_distance(path[:-1], path[1:])
    step_t = se3.translation_distance(path[:-1], path[1:])
    # Constant angular and linear speed: every step equal, total equals the geodesic.
    np.testing.assert_allclose(step_rot, step_rot.mean(), atol=1e-6)
    np.testing.assert_allclose(step_rot.sum(), se3.rotation_distance(A, B), atol=1e-6)
    np.testing.assert_allclose(step_t, np.linalg.norm(B[:3, 3] - A[:3, 3]) / 200, atol=1e-9)


def test_interpolate_poses_hits_knots(rng):
    tk = np.array([0.0, 0.3, 0.35, 1.0])
    Tk = np.stack([random_pose(rng) for _ in tk])
    np.testing.assert_allclose(se3.interpolate_poses(tk, tk, Tk), Tk, atol=1e-9)
    mid = se3.interpolate_poses([0.15], tk, Tk)[0]
    np.testing.assert_allclose(mid, se3.interpolate_pair(Tk[0], Tk[1], 0.5), atol=1e-9)


def test_chordal_mean_and_average(rng):
    R0 = Rotation.random(random_state=1).as_matrix()
    noise = Rotation.from_rotvec(rng.normal(scale=np.radians(1), size=(500, 3))).as_matrix()
    Rm = se3.chordal_mean_rotation(R0 @ noise)
    assert np.degrees(se3.rotation_distance(Rm, R0)) < 0.2
    assert se3.is_rotation(Rm)
    T = se3.make_T(R0 @ noise, np.r_[1, 2, 3] + rng.normal(scale=0.001, size=(500, 3)))
    T[:10, :3, 3] += 5.0  # outliers barely move the median
    assert np.linalg.norm(se3.average_poses(T)[:3, 3] - [1, 2, 3]) < 1e-3
