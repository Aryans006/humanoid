import numpy as np
from scipy.spatial.transform import Rotation

from ego2libero import se3
from ego2libero.config import load_yaml
from ego2libero.export.to_libero import resample, to_libero
from ego2libero.fusion import phases as ph, signals as sg, smooth as sm


def test_point_box_distance():
    T = np.tile(se3.make_T(Rotation.from_euler("z", 0.7).as_matrix(), [0.3, 0.1, 0.025]), (4, 1, 1))
    dims = np.array([0.05, 0.05, 0.05])
    up = T[0, :3, :3] @ [0, 0, 1]
    p = np.stack([T[0, :3, 3], T[0, :3, 3] + 0.025 * up, T[0, :3, 3] + 0.045 * up, T[0, :3, 3] + 0.01 * up])
    d = sg.point_box_distance(p, T, dims)
    np.testing.assert_allclose(d, [-0.025, 0.0, 0.020, -0.015], atol=1e-12)


def _scripted(fps=60, drop=True):
    """approach 0-2 s, grasp 2-3 s, transport 3-6 s, release at 6 s (drop), retreat after."""
    t = np.arange(0, 9, 1 / fps)
    speed = np.where((t >= 3) & (t < 6), 0.05, 0.002)
    if drop:
        speed = np.where((t >= 6) & (t < 6.15), 0.4, speed)  # falling box
    contact = np.where(t < 2, 0.15 - 0.07 * t, 0.003)
    contact = np.where(t >= 6, 0.02 + 0.4 * (t - 6), contact)  # fingertips leave at release
    return t, speed, contact


def test_phase_segmentation_with_drop():
    cfg = load_yaml("pipeline")["phases"]
    t, speed, contact = _scripted()
    s = ph.segment(t, speed, contact, cfg)
    on = s.onset_times(t)
    assert abs(on["grasp"] - 1.85) < 0.1, on  # contact < 2 cm from t = 1.857 s
    assert abs(on["transport"] - 3.0) < 0.05, on
    assert abs(on["release"] - 6.0) < 0.05, on  # fingertips leave, before the box lands
    assert on["retreat"] >= 6.15 - 1e-9, on  # only once the box has landed
    g = s.gripper_closed
    assert g[(t > 2.0) & (t < 5.9)].all() and not g[t < 1.8].any() and not g[t > 6.05].any()


def test_hand_tracking_dropout_mid_transport_is_not_a_release():
    """HaWoR loses the hand for 0.4 s while the box is still being carried; the box only lands
    at 6 s. The release must be the landing, not the dropout: an early release truncates the
    demo mid-air, which is what cost the 2026-10-07 session most of its sim episodes."""
    cfg = load_yaml("pipeline")["phases"]
    t, speed, contact = _scripted(drop=False)
    contact = np.where((t >= 4.0) & (t < 4.4), 0.3, contact)  # dropout: fingertips "leave"
    s = ph.segment(t, speed, contact, cfg)
    on = s.onset_times(t)
    assert abs(on["release"] - 6.0) < 0.05, on
    assert any("still in flight" in n for n in s.notes), s.notes
    assert s.gripper_closed[(t > 4.0) & (t < 5.9)].all()  # carried straight through the dropout


def test_phase_segmentation_box_never_moves():
    cfg = load_yaml("pipeline")["phases"]
    t = np.arange(0, 3, 1 / 60)
    s = ph.segment(t, np.full(len(t), 0.001), np.full(len(t), 0.1), cfg)
    assert (s.phase == ph.APPROACH).all() and not s.onsets


def _noisy_track(rng, n=300):
    t = np.arange(n) / 60
    p = np.c_[0.1 * t, 0.05 * np.sin(t), np.where(t < 2.5, 0.025, 0.025 + 0.1 * (t - 2.5))]
    R = Rotation.from_rotvec(np.c_[0.2 * t, np.zeros(n), 0.1 * t]).as_matrix()
    clean = se3.make_T(R, p)
    noisy = clean.copy()
    noisy[:, :3, 3] += rng.normal(scale=0.002, size=(n, 3))
    noisy[:, :3, :3] = noisy[:, :3, :3] @ Rotation.from_rotvec(rng.normal(scale=np.radians(1), size=(n, 3))).as_matrix()
    return t, clean, noisy


def test_hampel_and_smoothing(rng):
    t, clean, noisy = _noisy_track(rng)
    spiky = noisy.copy()
    spiky[[50, 120, 200], :3, 3] += 0.05
    out = sm.hampel_outliers(spiky)
    assert {50, 120, 200} <= set(np.flatnonzero(out)) and out.sum() <= 10
    fixed = sm.fill_by_interpolation(spiky, t, out)
    sm_T = sm.smooth_track(fixed, [(0, 150), (150, 300)], window=15)
    err_raw = se3.translation_distance(noisy, clean).mean()
    err_s = se3.translation_distance(sm_T, clean).mean()
    rot_raw = np.degrees(se3.rotation_distance(noisy, clean)).mean()
    rot_s = np.degrees(se3.rotation_distance(sm_T, clean)).mean()
    assert err_s < 0.6 * err_raw and rot_s < 0.6 * rot_raw, (err_raw, err_s, rot_raw, rot_s)


def test_smoothing_does_not_cross_segments():
    t = np.arange(200) / 60
    T = np.tile(np.eye(4), (200, 1, 1))
    T[100:, 2, 3] = 0.1  # a step: two resting segments
    out = sm.smooth_track(T, [(0, 100), (100, 200)], window=31)
    np.testing.assert_allclose(out[:100, 2, 3], 0, atol=1e-12)
    np.testing.assert_allclose(out[100:, 2, 3], 0.1, atol=1e-12)


def test_resample_and_libero(rng):
    t, clean, _ = _noisy_track(rng)
    t = t + rng.uniform(-0.002, 0.002, len(t))  # slightly irregular PTS
    t.sort()
    arrays = dict(t=t, T_W_O=clean, T_W_H=clean, joints_W=np.repeat(clean[:, None, :3, 3], 21, 1),
                  aperture=np.linspace(0.08, 0.05, len(t)), phase=(t > 2).astype(np.uint8),
                  gripper_closed=t > 2, obj_source=np.zeros(len(t), np.uint8))
    r = resample(arrays, 20)
    assert np.allclose(np.diff(r["t"]), 0.05) and r["T_W_O"].shape == (len(r["t"]), 4, 4)
    ref = se3.interpolate_poses(r["t"], t, clean)
    np.testing.assert_allclose(r["T_W_O"], ref, atol=1e-9)
    assert set(np.unique(r["phase"])) == {0, 1}
    T_L_W = se3.make_T(Rotation.from_euler("z", 0.5).as_matrix(), [0.1, -0.2, 0.9])
    L = to_libero(r, T_L_W)
    np.testing.assert_allclose(L["T_L_O"], T_L_W @ r["T_W_O"], atol=1e-12)
    np.testing.assert_allclose(L["joints_L"][:, 0], se3.transform_points(T_L_W, r["joints_W"][:, 0]), atol=1e-12)
    assert "T_W_O" not in L


def test_hand_smoothing_respects_infilled_boundary(rng):
    from ego2libero.fusion.fuse import fuse
    cfg = load_yaml("pipeline")
    n = 240
    t = np.arange(n) / 60
    H = se3.make_T(np.tile(np.eye(3), (n, 1, 1)), np.c_[np.full(n, 0.3), np.full(n, 0.1), np.full(n, 0.15)])
    src = np.zeros(n, np.uint8)
    src[:60] = 2
    H[:60, :3, 3] = [1.2, -0.5, -0.8]  # off-screen garbage, as HaWoR infills
    J = np.repeat(H[:, None, :3, 3], 21, 1).copy()
    O = np.tile(se3.make_T(np.eye(3), [0.3, 0.1, 0.025]), (n, 1, 1))
    f = fuse(t, O, H, J, src, np.array([0.05] * 3), cfg)
    np.testing.assert_allclose(f.T_W_H[60:, :3, 3], H[60:, :3, 3], atol=1e-9)  # untouched by the garbage
    np.testing.assert_allclose(f.joints_W[60:], J[60:], atol=1e-9)
