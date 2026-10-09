"""Smoke test of scripts/run_demo.py on a rendered, lens-distorted clip with its own configs."""

import importlib.util
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

from ego2libero import video
from ego2libero.calib.calibrate import centered_K_u
from ego2libero.perception.board_pose import T_W_board
from ego2libero.perception.box_pose import BoxModel
from synth import board_planes, box_planes, render
from test_pipeline_synth import BOARD, box_traj, cam_traj

ROOT = Path(__file__).resolve().parents[1]
SIZE = (640, 480)
K = np.array([[520.0, 0, 322], [0, 520.0, 238], [0, 0, 1]])
DIST = np.array([0.05, -0.08, 0.0, 0.0, 0.02])


def test_run_demo_end_to_end(tmp_path, monkeypatch):
    cfg = tmp_path / "configs"
    shutil.copytree(ROOT / "configs", cfg)
    monkeypatch.setenv("EGO2LIBERO_CONFIG_DIR", str(cfg))
    for name, upd in [("board", dict(tag_size_m=BOARD["tag_size_m"], gap_m=BOARD["gap_m"])),
                      ("box", dict(dims_m=[0.05] * 3, tag_size_m=0.04)),
                      ("camera", dict(K=K.tolist(), dist=DIST.tolist(), image_size=list(SIZE),
                                      K_u=centered_K_u(K, DIST, SIZE).tolist()))]:
        y = yaml.safe_load(open(cfg / f"{name}.yaml"))
        y.update(upd)
        yaml.safe_dump(y, open(cfg / f"{name}.yaml", "w"))
    pipe = yaml.safe_load(open(cfg / "pipeline.yaml"))
    pipe["windows"] = {"static_start_s": 0.35, "static_end_s": 0.4}
    yaml.safe_dump(pipe, open(cfg / "pipeline.yaml", "w"))

    box_cfg = yaml.safe_load(open(cfg / "box.yaml"))
    model = BoxModel.from_config(box_cfg)
    T_W_O, T_W_C = box_traj(), cam_traj()
    raw = tmp_path / "data/raw/sess/demo_000.mp4"
    raw.parent.mkdir(parents=True)
    with video.VideoWriter(raw, *SIZE, crf=12) as w:
        for i in range(len(T_W_O)):
            img = render(board_planes(BOARD, T_W_board(BOARD["tag_size_m"], BOARD["gap_m"]))
                         + box_planes(model, T_W_O[i]), T_W_C[i], K, SIZE, dist=DIST)
            w.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), i / 30)

    spec = importlib.util.spec_from_file_location("run_demo", ROOT / "scripts/run_demo.py")
    rd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rd)
    res = rd.run_clip("sess", "demo_000", data=tmp_path / "data", overlay_scale=0.5)
    rd.append_summary("sess", "demo_000", res, data=tmp_path / "data")
    by = {r["gate"]: r for r in res}
    assert by["board_pose_coverage"]["value"] > 0.95
    assert by["box_rest_start_z_m"]["pass"] and by["box_rest_end_z_m"]["pass"], res
    assert by["box_tag_coverage"]["pass"], res
    out = tmp_path / "data/interim/sess/demo_000"
    for f in ("undistorted.mp4", "timestamps.npy", "detections.npz", "camera_tags.npz", "box_tags.npz",
              "overlay.mp4", "gates.json"):
        assert (out / f).exists(), f
    assert (tmp_path / "data/processed/sess/summary.csv").exists()

    # M2: a HaWoR result next to the clip -> hands in W, Sim(3) gate, hand overlay, export
    from scipy.spatial.transform import Rotation
    from ego2libero.export.write_demo import load_demo
    n = len(T_W_O)
    K_u = np.array(yaml.safe_load(open(cfg / "camera.yaml"))["K_u"])
    rng = np.random.default_rng(1)
    T_S_C = T_W_C.copy()  # HaWoR world = W here
    hw = dict(R_c2w=T_S_C[:, :3, :3], t_c2w=T_S_C[:, :3, 3], img_focal=K_u[0, 0], img_center=K_u[:2, 2],
              image_size=np.array(SIZE), slam_scale=1.0, n_frames=n)
    for hand in ("left", "right"):
        hw.update({f"{hand}_rot": Rotation.random(n, random_state=2).as_rotvec(), f"{hand}_trans": np.zeros((n, 3)),
                   f"{hand}_hand_pose": np.zeros((n, 45)), f"{hand}_betas": np.zeros((n, 10)),
                   f"{hand}_joints": rng.normal(scale=0.03, size=(n, 21, 3)) + [0.3, 0.05, 0.1],
                   f"{hand}_observed": np.arange(n) % 4 != 0})
    np.savez(out / "hawor.npz", **hw)
    res = rd.run_clip("sess", "demo_000", data=tmp_path / "data", overlay_scale=0.5)
    by = {r["gate"]: r for r in res}
    assert by["sim3_rot_residual_deg"]["pass"] and by["sim3_rot_residual_deg"]["value"] < 0.5, by["sim3_rot_residual_deg"]
    assert (out / "hands_W.npz").exists()
    d = load_demo(tmp_path / "data/processed/sess/demo_000")
    for k in ("t", "T_W_C", "T_W_O", "T_W_H", "joints_W", "aperture", "cam_source", "obj_source", "hand_source"):
        assert k in d and len(d[k]) == n, k
    assert d["meta"]["stage"] == "M3"
    for k in ("phase", "gripper_closed", "mano_pose"):
        assert k in d and len(d[k]) == n, k
    assert np.isfinite(d["T_W_O"]).all()  # S7 leaves no missing box poses
    d20 = load_demo(tmp_path / "data/processed/sess/demo_000_20hz")
    assert abs(np.median(np.diff(d20["t"])) - 0.05) < 1e-9 and d20["T_W_O"].shape[1:] == (4, 4)
    assert (out / "plots.png").exists()
    from ego2libero.align.hands_to_world import HandTrackW
    hW = HandTrackW.load(out / "hands_W.npz")  # unsmoothed S5 output: exact up to the board-camera error
    on = np.isfinite(d["T_W_C"]).all(axis=(1, 2))
    np.testing.assert_allclose(hW.joints_W[on], hw["right_joints"][on], atol=2e-3)
