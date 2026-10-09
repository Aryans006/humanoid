"""SE(3)/SO(3) utilities.

Convention everywhere: ``T_A_B`` is the pose of frame B expressed in frame A, a
4x4 float64 matrix with ``p_A = T_A_B @ p_B``. Composition reads left to right:
``T_W_O = T_W_C @ T_C_O``.

All functions accept a single pose (4, 4) or a batch (N, 4, 4) unless noted.
Quaternions are scipy order ``xyzw``.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

_EPS = 1e-12


# --------------------------------------------------------------------------- basics


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build homogeneous transforms from rotations (..., 3, 3) and translations (..., 3)."""
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    batch = np.broadcast_shapes(R.shape[:-2], t.shape[:-1])
    T = np.zeros(batch + (4, 4))
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def identity(n: int | None = None) -> np.ndarray:
    return np.eye(4) if n is None else np.tile(np.eye(4), (n, 1, 1))


def inv(T: np.ndarray) -> np.ndarray:
    """Closed-form inverse of rigid transforms."""
    T = np.asarray(T, dtype=np.float64)
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = np.swapaxes(R, -1, -2)
    return make_T(Rt, -np.einsum("...ij,...j->...i", Rt, t))


def compose(*Ts: np.ndarray) -> np.ndarray:
    """``compose(A, B, C) == A @ B @ C`` with broadcasting over batches."""
    out = np.asarray(Ts[0], dtype=np.float64)
    for T in Ts[1:]:
        out = out @ np.asarray(T, dtype=np.float64)
    return out


def transform_points(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Apply T (4, 4) or (N, 4, 4) to points (..., 3). Batched T maps p of shape (N, M, 3)."""
    T = np.asarray(T, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if T.ndim == 2:
        return p @ T[:3, :3].T + T[:3, 3]
    return np.einsum("nij,nmj->nmi", T[:, :3, :3], p) + T[:, None, :3, 3]


def is_rotation(R: np.ndarray, tol: float = 1e-6) -> bool:
    R = np.asarray(R)
    I = np.eye(3)
    ortho = np.abs(np.swapaxes(R, -1, -2) @ R - I).max() < tol
    return bool(ortho and np.all(np.abs(np.linalg.det(R) - 1.0) < tol))


def project_to_SO3(M: np.ndarray) -> np.ndarray:
    """Nearest rotation (Frobenius) to each 3x3 matrix in M."""
    U, _, Vt = np.linalg.svd(M)
    D = np.ones(M.shape[:-2] + (3,))
    D[..., 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ (D[..., :, None] * Vt)


# --------------------------------------------------------------------------- SO(3)


def hat(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    W = np.zeros(w.shape[:-1] + (3, 3))
    W[..., 0, 1], W[..., 0, 2] = -w[..., 2], w[..., 1]
    W[..., 1, 0], W[..., 1, 2] = w[..., 2], -w[..., 0]
    W[..., 2, 0], W[..., 2, 1] = -w[..., 1], w[..., 0]
    return W


def so3_exp(w: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(np.asarray(w, dtype=np.float64).reshape(-1, 3)).as_matrix().reshape(
        np.shape(w)[:-1] + (3, 3)
    )


def so3_log(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    return Rotation.from_matrix(R.reshape(-1, 3, 3)).as_rotvec().reshape(R.shape[:-2] + (3,))


def rotation_angle(R: np.ndarray) -> np.ndarray:
    """Rotation angle in radians of each R, in [0, pi]."""
    return np.linalg.norm(so3_log(R), axis=-1)


def rotation_distance(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    """Geodesic angle (radians) between rotations; accepts poses (4x4) too."""
    R1 = np.asarray(R1)[..., :3, :3]
    R2 = np.asarray(R2)[..., :3, :3]
    return rotation_angle(np.swapaxes(R1, -1, -2) @ R2)


def translation_distance(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(T1)[..., :3, 3] - np.asarray(T2)[..., :3, 3], axis=-1)


# --------------------------------------------------------------------------- SE(3)


def _left_jacobian(w: np.ndarray) -> np.ndarray:
    """SO(3) left Jacobian V(w), used by the SE(3) exp/log translation part."""
    theta = np.linalg.norm(w, axis=-1)[..., None, None]
    W = hat(w)
    W2 = W @ W
    small = theta < 1e-6
    th = np.where(small, 1.0, theta)
    A = np.where(small, 0.5 - theta**2 / 24.0, (1 - np.cos(th)) / th**2)
    B = np.where(small, 1.0 / 6.0 - theta**2 / 120.0, (th - np.sin(th)) / th**3)
    return np.eye(3) + A * W + B * W2


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """xi = (rho, w): translation-part first, rotation vector last. Shape (..., 6)."""
    xi = np.asarray(xi, dtype=np.float64)
    rho, w = xi[..., :3], xi[..., 3:]
    R = so3_exp(w)
    t = np.einsum("...ij,...j->...i", _left_jacobian(w), rho)
    return make_T(R, t)


def se3_log(T: np.ndarray) -> np.ndarray:
    """Inverse of :func:`se3_exp`. Returns (..., 6) as (rho, w)."""
    T = np.asarray(T, dtype=np.float64)
    w = so3_log(T[..., :3, :3])
    rho = np.linalg.solve(_left_jacobian(w), T[..., :3, 3:4])[..., 0]
    return np.concatenate([rho, w], axis=-1)


# --------------------------------------------------------------------------- quaternions


def to_quat_xyzw(R_or_T: np.ndarray, continuous: bool = True) -> np.ndarray:
    """Rotations (or poses) to xyzw quaternions; hemisphere-continuous along axis 0 if batched."""
    M = np.asarray(R_or_T, dtype=np.float64)[..., :3, :3]
    q = Rotation.from_matrix(M.reshape(-1, 3, 3)).as_quat().reshape(M.shape[:-2] + (4,))
    if continuous and q.ndim == 2:
        q = quat_continuous(q)
    return q


def quat_continuous(q: np.ndarray) -> np.ndarray:
    """Flip signs so consecutive quaternions have non-negative dot product."""
    q = np.array(q, dtype=np.float64, copy=True)
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0:
            q[i] = -q[i]
    return q


def from_quat_xyzw(q: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    R = Rotation.from_quat(q.reshape(-1, 4)).as_matrix().reshape(q.shape[:-1] + (3, 3))
    if t is None:
        return R
    return make_T(R, t)


# --------------------------------------------------------------------------- averaging


def chordal_mean_rotation(R: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Chordal L2 mean: projection of the (weighted) arithmetic mean of matrices onto SO(3)."""
    R = np.asarray(R, dtype=np.float64)[..., :3, :3]
    w = np.ones(len(R)) if weights is None else np.asarray(weights, dtype=np.float64)
    M = np.einsum("n,nij->ij", w, R)
    return project_to_SO3(M)


def average_poses(T: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Robust pose average: chordal mean rotation, per-axis median translation."""
    T = np.asarray(T, dtype=np.float64)
    return make_T(chordal_mean_rotation(T, weights), np.median(T[:, :3, 3], axis=0))


# --------------------------------------------------------------------------- interpolation


def interpolate_pair(T0: np.ndarray, T1: np.ndarray, alpha: float | np.ndarray) -> np.ndarray:
    """Decoupled interpolation: slerp on rotation, linear on translation.

    For sampled trajectories this is preferred over screw (SE(3) geodesic) interpolation,
    which makes the translation follow a helix around the screw axis.
    alpha may be a scalar or an array (K,), returning (4, 4) or (K, 4, 4).
    """
    alpha_arr = np.atleast_1d(np.asarray(alpha, dtype=np.float64))
    slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([T0[:3, :3], T1[:3, :3]])))
    R = slerp(np.clip(alpha_arr, 0.0, 1.0)).as_matrix()
    t = (1 - alpha_arr)[:, None] * T0[:3, 3] + alpha_arr[:, None] * T1[:3, 3]
    out = make_T(R, t)
    return out[0] if np.ndim(alpha) == 0 else out


def interpolate_poses(t_query: np.ndarray, t_known: np.ndarray, T_known: np.ndarray) -> np.ndarray:
    """Resample a pose track at t_query (decoupled slerp + linear). Clamps outside the range."""
    t_known = np.asarray(t_known, dtype=np.float64)
    t_query = np.asarray(t_query, dtype=np.float64)
    T_known = np.asarray(T_known, dtype=np.float64)
    if len(t_known) == 0:
        raise ValueError("interpolate_poses needs at least one known pose")
    if len(t_known) == 1:
        return np.repeat(T_known, len(t_query), axis=0)
    if np.any(np.diff(t_known) <= 0):
        raise ValueError("t_known must be strictly increasing")
    tq = np.clip(t_query, t_known[0], t_known[-1])
    R = Slerp(t_known, Rotation.from_matrix(T_known[:, :3, :3]))(tq).as_matrix()
    t = np.stack([np.interp(tq, t_known, T_known[:, :3, 3][:, k]) for k in range(3)], axis=-1)
    return make_T(R, t)
