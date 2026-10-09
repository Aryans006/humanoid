import numpy as np
from scipy.spatial.transform import Rotation

from ego2libero import se3
from ego2libero.fusion import bridge as br


def _held_demo(n=200, gap=(80, 130), hold=(40, 170)):
    t = np.linspace(0, 1, n)
    # Wrist path: lift, carry, set down, with wrist rotation.
    pos = np.c_[0.3 * t, 0.1 * np.sin(np.pi * t), 0.15 + 0.1 * np.sin(np.pi * t)]
    rot = Rotation.from_rotvec(np.c_[0.2 * np.sin(2 * np.pi * t), 0.1 * t, 0.8 * t]).as_matrix()
    T_W_H = se3.make_T(rot, pos)
    T_H_O = se3.make_T(Rotation.from_rotvec([0.1, -1.2, 0.3]).as_matrix(), [0.02, -0.01, 0.09])
    truth = T_W_H @ T_H_O
    truth[: hold[0]] = truth[hold[0]]  # resting before grasp
    truth[hold[1]:] = truth[hold[1] - 1]  # resting after release
    obs = truth.copy()
    src = np.full(n, br.SRC_TAG)
    obs[gap[0]:gap[1]] = np.nan
    src[gap[0]:gap[1]] = br.SRC_MISSING
    return T_W_H, T_H_O, truth, obs, src, hold, gap


def test_bridging_exact_when_grasp_exact():
    T_W_H, T_H_O, truth, obs, src, hold, gap = _held_demo()
    est = br.estimate_T_H_O(T_W_H, obs, np.arange(*hold))
    np.testing.assert_allclose(est.T_H_O, T_H_O, atol=1e-9)
    assert est.trans_res_m.max() < 1e-9

    out, src2, jumps = br.bridge_held(obs, src, T_W_H, *hold, est.T_H_O)
    np.testing.assert_allclose(out, truth, atol=1e-9)
    assert np.all(src2[gap[0]:gap[1]] == br.SRC_PROPAGATED)
    assert np.all(np.delete(src2, np.arange(*gap)) == br.SRC_TAG)
    assert {j["kind"] for j in jumps} == {"vanish", "reappear"}
    assert max(j["trans_m"] for j in jumps) < 1e-9


def test_outlier_frames_dropped():
    T_W_H, T_H_O, truth, obs, src, hold, gap = _held_demo()
    bad = obs.copy()
    bad[hold[0]:hold[0] + 10, :3, 3] += 0.05  # a few corrupted tag poses
    est = br.estimate_T_H_O(T_W_H, bad, np.arange(*hold))
    assert se3.translation_distance(est.T_H_O, T_H_O) < 1e-9


def test_blend_removes_snap():
    T_W_H, T_H_O, truth, obs, src, hold, gap = _held_demo()
    off = T_H_O.copy()
    off[:3, 3] += [0.01, 0, 0]  # 1 cm grasp error -> 1 cm jump at both gap edges
    out, _, jumps = br.bridge_held(obs, src, T_W_H, *hold, off, blend_frames=5)
    assert all(abs(j["trans_m"] - 0.01) < 1e-9 for j in jumps)
    steps = se3.translation_distance(out[1:], out[:-1])
    nominal = se3.translation_distance(truth[1:], truth[:-1]).max()
    assert steps.max() < nominal + 0.01 / 5 + 1e-6  # no step larger than a ramp increment


def test_hold_resting():
    n = 10
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, 0, 3] = np.arange(n)
    src = np.full(n, br.SRC_TAG)
    T[[0, 1, 5, 6]] = np.nan
    src[[0, 1, 5, 6]] = br.SRC_MISSING
    resting = np.ones(n, bool)
    out, s = br.hold_resting(T, src, resting)
    np.testing.assert_allclose(out[[0, 1, 5, 6], 0, 3], [2, 2, 4, 4])
    assert np.all(s[[0, 1, 5, 6]] == br.SRC_INTERPOLATED)


def test_short_gaps_interpolated_and_not_overwritten():
    T_W_H, T_H_O, truth, obs, src, hold, gap = _held_demo()
    obs2, src2 = obs.copy(), src.copy()
    t = np.arange(len(src)) / 60
    # restore the long gap except a 2-frame hole, plus keep the long one elsewhere
    obs2[gap[0]:gap[1]] = truth[gap[0]:gap[1]]
    src2[gap[0]:gap[1]] = br.SRC_TAG
    obs2[90:92] = np.nan
    src2[90:92] = br.SRC_MISSING
    obs2[140:160] = np.nan
    src2[140:160] = br.SRC_MISSING
    o3, s3 = br.interpolate_short_gaps(obs2, src2, t, *hold, max_gap_s=0.1)
    assert np.all(s3[90:92] == br.SRC_INTERPOLATED) and np.all(s3[140:160] == br.SRC_MISSING)
    o4, s4, _ = br.bridge_held(o3, s3, T_W_H, *hold, T_H_O)
    assert np.all(s4[90:92] == br.SRC_INTERPOLATED)  # bridging must not overwrite them
    assert np.all(s4[140:160] == br.SRC_PROPAGATED)
    np.testing.assert_allclose(o4[140:160], truth[140:160], atol=1e-9)
