import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ego2libero import se3
from ego2libero.config import load_yaml
from ego2libero.perception import box_pose as bp
from conftest import K_PHONE, look_at

DIMS = np.array([0.05, 0.05, 0.09])
TAG = 0.04


@pytest.fixture
def model():
    cfg = load_yaml("box")
    cfg.update(dims_m=DIMS.tolist(), tag_size_m=TAG)  # configs hold nulls until measured
    return bp.BoxModel.from_config(cfg)


def visible_corners(model, T_W_C, T_W_O, K, noise_px, rng, only=None):
    T_C_O = se3.inv(T_W_C) @ T_W_O
    out = {}
    for i, T_O_G in model.T_O_G.items():
        if only is not None and i not in only:
            continue
        T_C_G = T_C_O @ T_O_G
        to_cam = -T_C_G[:3, 3]
        if T_C_G[:3, 2] @ to_cam <= 0.2 * np.linalg.norm(to_cam):  # facing away / grazing
            continue
        uv = bp.project(T_C_G, bp.tag_corners(model.tag_size), K)
        out[i] = uv + rng.normal(scale=noise_px, size=uv.shape)
    return out


def test_face_transforms_are_on_faces(model):
    for i, T in model.T_O_G.items():
        assert se3.is_rotation(T[:3, :3])
        n = T[:3, 2]
        ax = int(np.argmax(np.abs(n)))
        assert np.isclose(abs(T[ax, 3]), DIMS[ax] / 2) and np.allclose(np.delete(T[:3, 3], ax), 0)
        assert np.sign(T[ax, 3]) == np.sign(n[ax])  # z points outward
    assert sorted(model.T_O_G) == list(range(20, 26))


def _random_two_face_view(rng):
    T_W_O = se3.make_T(Rotation.from_euler("z", rng.uniform(-np.pi, np.pi)).as_matrix(),
                       [rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), DIMS[2] / 2])
    az = rng.uniform(-np.pi, np.pi)
    eye = T_W_O[:3, 3] + [0.4 * np.cos(az), 0.4 * np.sin(az), 0.35]  # ~0.53 m, head height
    return T_W_O, look_at(eye, T_W_O[:3, 3])


def test_joint_pnp_exact_without_noise(model, rng):
    for _ in range(20):
        T_W_O, T_W_C = _random_two_face_view(rng)
        res = bp.solve_box_pose(visible_corners(model, T_W_C, T_W_O, K_PHONE, 0.0, rng), K_PHONE, model)
        np.testing.assert_allclose(T_W_C @ res.T_C_O, T_W_O, atol=1e-6)


def test_joint_pnp_recovers_pose_under_noise(model, rng):
    """Plan target: 2 mm / 0.5 deg at 0.5 px noise. Single-frame error at ~0.5 m is almost
    entirely depth along the viewing ray, so the target holds at the 90th percentile, not
    for every frame; that is why static windows are averaged."""
    errs_t, errs_r = [], []
    for _ in range(200):
        T_W_O, T_W_C = _random_two_face_view(rng)
        corners = visible_corners(model, T_W_C, T_W_O, K_PHONE, 0.5, rng)
        assert len(corners) >= 2
        res = bp.solve_box_pose(corners, K_PHONE, model)
        assert res is not None and not res.ambiguous and res.rms_px < 1.5
        T_est = T_W_C @ res.T_C_O
        errs_t.append(se3.translation_distance(T_est, T_W_O))
        errs_r.append(np.degrees(se3.rotation_distance(T_est, T_W_O)))
    assert np.median(errs_t) < 0.001 and np.percentile(errs_t, 90) < 0.002, np.percentile(errs_t, [50, 90])
    assert np.median(errs_r) < 0.25 and np.percentile(errs_r, 90) < 0.5, np.percentile(errs_r, [50, 90])


def _single_tag_case(model, rng, dist=0.7):
    """Top tag only, seen nearly head-on from far away: the classic IPPE flip."""
    T_W_O = se3.make_T(Rotation.from_euler("z", 0.4).as_matrix(), [0.05, 0.02, DIMS[2] / 2])
    eye = T_W_O[:3, 3] + [0.08, -0.05, dist]
    T_W_C = look_at(eye, T_W_O[:3, 3], up=(0, 1, 0))
    corners = visible_corners(model, T_W_C, T_W_O, K_PHONE, 0.5, rng, only=[24])
    return T_W_O, T_W_C, corners


def test_ippe_prior_picks_true_candidate(model, rng):
    T_W_O, T_W_C, corners = _single_tag_case(model, rng)
    res = bp.solve_box_pose(corners, K_PHONE, model, T_W_C=T_W_C)
    assert res.ambiguous and len(res.candidates) == 2
    cand_err = [np.degrees(se3.rotation_distance(T_W_C @ T, T_W_O)) for T in res.candidates]
    assert max(cand_err) > 5  # genuinely different candidates
    prior = T_W_O @ se3.make_T(Rotation.from_rotvec([0.03, -0.02, 0.04]).as_matrix(), [0.003, 0, 0])
    for _ in range(2):  # order of candidates must not matter
        k, reason = bp.disambiguate(res.candidates, res.candidate_rms, model, T_W_C, prior)
        assert reason.startswith("prior")
        assert k == int(np.argmin(cand_err))
        res.candidates.reverse(); res.candidate_rms.reverse(); cand_err.reverse()


def test_ippe_resting_rule_without_prior(model, rng):
    T_W_O, T_W_C, corners = _single_tag_case(model, rng)
    res = bp.solve_box_pose(corners, K_PHONE, model, T_W_C=T_W_C, resting=True)
    assert res.pick_reason == "resting"
    assert np.degrees(se3.rotation_distance(T_W_C @ res.T_C_O, T_W_O)) < 3.0


def test_rejects_small_tags(model, rng):
    T_W_O, T_W_C, corners = _single_tag_case(model, rng, dist=4.0)
    assert bp.solve_box_pose(corners, K_PHONE, model) is None


def test_viterbi_tracks_single_head_on_tag_through_occlusion(model):
    """Overhead camera, only the top tag visible (IPPE mirror ambiguity every frame), a 1 s
    occlusion while the box is in the air, then set down. Mirrors demo_000 from 2026-10-04,
    where greedy picking carried a wrong mirror solution (34 deg tilt) to the end."""
    rng = np.random.default_rng(11)
    K = np.array([[1384.0, 0, 960], [0, 1384.0, 540], [0, 0, 1]])
    T_W_C = look_at([0.30, 0.05, 0.55], [0.31, 0.06, 0.0], up=(0, 1, 0))
    n, fps = 300, 60.0
    t = np.arange(n) / fps
    s = np.clip((np.arange(n) - 60) / 180, 0, 1)
    s = 0.5 - 0.5 * np.cos(np.pi * s)
    truth, per_frame = [], []
    for i in range(n):
        lift = 0.10 * np.sin(np.pi * s[i])
        R = Rotation.from_euler("zy", [0.3 + 1.2 * s[i], 0.5 * np.sin(np.pi * s[i])]).as_matrix()
        # tag 24 (+z face) up, like the real cube's top tag
        T = se3.make_T(R, [0.30 + 0.06 * s[i], 0.05, DIMS[2] / 2 + lift])
        truth.append(T)
        if 100 <= i < 160:  # hand covers the tag
            per_frame.append({})
            continue
        per_frame.append(visible_corners(model, T_W_C, T, K, 0.3, rng, only=[24]))
    truth = np.stack(truth)
    T_cam = np.repeat(T_W_C[None], n, 0)
    box = bp.track_box(per_frame, t, K, T_cam, model, {}, rest_masks=[t <= 0.5, t >= t[-1] - 0.5])
    v = box.valid
    assert v.sum() > 200
    err = np.degrees(se3.rotation_distance(box.T_W_O[v], truth[v]))
    # single head-on tag: ~3-5 deg of rotation noise is expected; a mirror flip mid-lift is ~2x the
    # 29 deg tilt (~58 deg), and at the end it shows as a tilted box resting on the table
    assert err.max() < 20, err.max()
    end = v & (t >= t[-1] - 0.5)
    T_end = se3.average_poses(box.T_W_O[end])
    assert np.degrees(np.arccos(np.abs(T_end[2, :3]).max())) < 4
