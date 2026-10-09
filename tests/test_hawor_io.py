import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ego2libero import se3
from ego2libero.hands import hawor_io


def _fake_output(path, n=12):
    rng = np.random.default_rng(0)
    out = {
        "R_c2w": Rotation.random(n, random_state=1).as_matrix(),
        "t_c2w": rng.normal(size=(n, 3)),
        "img_focal": 1400.0,
        "img_center": np.array([960.0, 540.0]),
        "image_size": np.array([1920, 1080]),
        "slam_scale": 1.3,
        "n_frames": n,
    }
    for hand in ("left", "right"):
        out[f"{hand}_rot"] = rng.normal(scale=0.5, size=(n, 3))
        out[f"{hand}_trans"] = rng.normal(size=(n, 3))
        out[f"{hand}_hand_pose"] = rng.normal(size=(n, 45))
        out[f"{hand}_betas"] = rng.normal(size=(n, 10))
        out[f"{hand}_joints"] = rng.normal(size=(n, 21, 3))
        out[f"{hand}_observed"] = np.arange(n) % 3 != 0
    np.savez(path, **out)
    return out


def test_load_hawor(tmp_path):
    raw = _fake_output(tmp_path / "h.npz")
    tr = hawor_io.load_hawor(tmp_path / "h.npz")
    assert tr.n == 12
    np.testing.assert_allclose(tr.T_S_C[:, :3, :3], raw["R_c2w"])
    np.testing.assert_allclose(tr.T_S_H[:, :3, 3], raw["right_joints"][:, 0])
    np.testing.assert_allclose(se3.so3_log(tr.T_S_H[:, :3, :3]), raw["right_rot"], atol=1e-9)
    assert tr.mano_pose.shape == (12, 48) and tr.mano_betas.shape == (10,)
    assert tr.observed.sum() == 8
    assert tr.aperture().shape == (12,)


def test_check_intrinsics(tmp_path):
    _fake_output(tmp_path / "h.npz")
    tr = hawor_io.load_hawor(tmp_path / "h.npz")
    hawor_io.check_intrinsics(tr, [[1400, 0, 960], [0, 1400, 540], [0, 0, 1]])
    with pytest.raises(ValueError, match="principal point"):
        hawor_io.check_intrinsics(tr, [[1400, 0, 955], [0, 1400, 540], [0, 0, 1]])
