"""End to end on rendered video: detect -> board pose -> box pose -> gates -> overlay."""

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ego2libero import gates, se3, video
from ego2libero.config import load_yaml
from ego2libero.perception.board_pose import BoardModel, track_camera
from ego2libero.perception.box_pose import BoxModel, track_box
from ego2libero.perception.detect import Detections, detect_clip
from ego2libero.viz.overlay import write_overlay
from conftest import look_at
from synth import board_planes, box_planes, render

SIZE = (640, 480)
K = np.array([[520.0, 0, 320], [0, 520.0, 240], [0, 0, 1]])
BOARD = dict(dictionary="DICT_APRILTAG_36h11", cols=4, rows=3, first_id=0, tag_size_m=0.054, gap_m=0.015)
L = 0.05


def box_traj(n=60):
    """Rest (0-11), lift / carry / turn (12-44), rest (45-59). Returns T_W_O (n, 4, 4)."""
    p0, p1 = np.array([0.33, 0.08, L / 2]), np.array([0.37, -0.03, L / 2])
    T = []
    for i in range(n):
        s = np.clip((i - 12) / 32, 0, 1)
        s = 0.5 - 0.5 * np.cos(np.pi * s)  # smooth
        lift = 0.08 * np.sin(np.pi * s)
        p = (1 - s) * p0 + s * p1 + [0, 0, lift]
        tilt = 0.35 * np.sin(np.pi * s)  # tip the box mid-air so other faces show
        R = Rotation.from_euler("zx", [0.4 + 0.9 * s, tilt]).as_matrix()
        T.append(se3.make_T(R, p))
    return np.stack(T)


def cam_traj(n=60):
    u = np.linspace(0, 1, n)
    eyes = np.c_[0.22 + 0.06 * np.sin(2 * np.pi * u), -0.33 + 0.03 * u, 0.44 + 0.02 * np.cos(3 * u)]
    targets = np.c_[0.24 + 0.03 * u, 0.06 + 0 * u, 0 * u]
    return np.stack([look_at(e, g) for e, g in zip(eyes, targets)])


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    board = BoardModel.from_config(BOARD)
    cfg = load_yaml("box")
    cfg.update(dims_m=[L] * 3, tag_size_m=0.04)
    model = BoxModel.from_config(cfg)
    T_W_O, T_W_C = box_traj(), cam_traj()
    rng = np.random.default_rng(0)
    t = np.r_[0, np.cumsum(rng.choice([1 / 30, 1 / 30, 1 / 25], size=len(T_W_O) - 1))]
    path = tmp_path_factory.mktemp("demo") / "undistorted.mp4"
    with video.VideoWriter(path, *SIZE, crf=12) as w:
        for i in range(len(t)):
            img = render(board_planes(BOARD, board.T_W_board) + box_planes(model, T_W_O[i]), T_W_C[i], K, SIZE)
            w.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), t[i])
    dets = detect_clip(path, progress=False)
    det_path = path.parent / "detections.npz"
    dets.save(det_path)
    return dict(path=path, board=board, model=model, T_W_O=T_W_O, T_W_C=T_W_C, t=t,
                dets=Detections.load(det_path))


def test_timestamps_and_detections(scene):
    np.testing.assert_allclose(scene["dets"].t, scene["t"], atol=1e-4)
    seen = set(scene["dets"].ids.tolist())
    assert set(range(12)) <= seen and len(seen & set(range(20, 26))) >= 3


def test_camera_track(scene):
    cam = track_camera(scene["dets"].per_frame(), K, scene["board"])
    v = cam.valid
    assert v.mean() > 0.95
    e_t = se3.translation_distance(cam.T_W_C[v], scene["T_W_C"][v])
    e_r = np.degrees(se3.rotation_distance(cam.T_W_C[v], scene["T_W_C"][v]))
    assert np.median(e_t) < 0.002 and e_t.max() < 0.006, (np.median(e_t), e_t.max())
    assert np.median(e_r) < 0.2 and e_r.max() < 0.5, (np.median(e_r), e_r.max())
    scene["cam"] = cam


def test_box_track_and_gates(scene, tmp_path):
    cam = scene.get("cam") or track_camera(scene["dets"].per_frame(), K, scene["board"])
    t = scene["t"]
    box = track_box(scene["dets"].per_frame(), t, K, cam.T_W_C, scene["model"],
                    load_yaml("pipeline")["box_pose"], rest_masks=[t <= t[0] + 0.35, t >= t[-1] - 0.4])
    v = box.valid
    assert v.mean() > 0.8, v.mean()
    e_t = se3.translation_distance(box.T_W_O[v], scene["T_W_O"][v])
    e_r = np.degrees(se3.rotation_distance(box.T_W_O[v], scene["T_W_O"][v]))
    assert np.median(e_t) < 0.003 and e_t.max() < 0.012, (np.median(e_t), e_t.max())
    assert np.median(e_r) < 1.0 and e_r.max() < 5.0, (np.median(e_r), e_r.max())  # no IPPE flips

    pipe = load_yaml("pipeline")
    windows = {"static_start_s": 0.35, "static_end_s": 0.4}  # this clip rests ~0.4 s at each end
    rest = gates.box_resting(box.T_W_O, scene["t"], scene["model"], pipe["gates"], windows)
    assert all(r["pass"] for r in rest), rest
    spots = [scene["T_W_O"][0, :2, 3], scene["T_W_O"][-1, :2, 3]]
    kp = gates.known_positions(box.T_W_O, scene["t"], scene["model"], spots, pipe["gates"], min_still_s=0.3)
    assert len(kp) == 2 and all(r["pass"] and r["value"] < 0.004 for r in kp), kp

    out = write_overlay(scene["path"], tmp_path / "overlay.mp4", K, T_W_C=cam.T_W_C, T_W_O=box.T_W_O,
                        T_C_O=box.T_C_O, box_corners=scene["model"].box_corners(),
                        dets_per_frame=scene["dets"].per_frame(), scale=0.5, progress=False)
    assert len(video.read_timestamps(out)) == len(scene["t"])


def test_focal_consistency_detects_zoom_change(scene):
    pf = scene["dets"].per_frame()
    ok = gates.focal_consistency(pf, scene["board"], K, T_W_C=scene["T_W_C"])
    assert ok["pass"] and abs(ok["value"] - 1) < 0.03, ok
    K2 = K.copy()
    K2[0, 0] = K2[1, 1] = K[0, 0] * 2  # calibration from a 2x-different zoom mode
    bad = gates.focal_consistency(pf, scene["board"], K2, T_W_C=scene["T_W_C"])
    overhead = np.repeat(np.eye(4)[None], len(pf), 0)  # camera looking straight down: unmeasurable
    assert gates.focal_consistency(pf, scene["board"], K, T_W_C=overhead)["value"] is None
    assert not bad["pass"] and abs(bad["value"] - 0.5) < 0.03, bad


def test_known_position_edge_on_line(scene):
    """Cube resting with its edge on a line: the gate compares its near edge, not its center."""
    model = scene["model"]
    T = scene["T_W_O"][:1].repeat(80, 0)
    t = np.arange(80) / 60
    near_edge_y = se3.transform_points(T[0], model.box_corners())[:, 1].min()
    spot = [T[0, 0, 3], near_edge_y]
    pipe = load_yaml("pipeline")
    r = gates.known_positions(T, t, model, [spot], pipe["gates"], placement="edge_on_line")
    assert r[0]["pass"] and r[0]["value"] < 1e-6, r
    r = gates.known_positions(T, t, model, [spot], pipe["gates"], placement="center")
    assert not r[0]["pass"]
