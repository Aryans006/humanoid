import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ego2libero import se3


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def random_pose(rng, t_scale=1.0):
    return se3.make_T(Rotation.random(random_state=rng.integers(1 << 31)).as_matrix(),
                      rng.normal(scale=t_scale, size=3))


def look_at(eye, target, up=(0, 0, 1)):
    """T_W_C for an OpenCV camera (x right, y down, z forward) at eye looking at target."""
    eye, target, up = map(lambda v: np.asarray(v, float), (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return se3.make_T(np.stack([x, y, z], axis=1), eye)


K_PHONE = np.array([[1450.0, 0, 960.0], [0, 1450.0, 540.0], [0, 0, 1]])
