# ego2libero — perception stage

Phone videos of a pick-and-place → metric 6D trajectories in a table frame. The design is in [plan.md](plan.md). **Hands-on steps (printing, building, measuring, recording) are in [GUIDE.md](GUIDE.md).** This file covers how to run things and where the code differs from the plan.

## Setup (main env)

```bash
uv sync                      # Python 3.10, numpy/scipy/opencv-contrib/av/...
uv run pytest                # synthetic tests, no video needed
```

## Print the targets

```bash
uv run python tools/print_tags.py --check    # writes tags/{board,box_tags,charuco}_A4.pdf
```

Print at **100% / actual size** on matte paper. Check the 100 mm scale bar first, then measure the tags and the assembled box and fill in [measurements.yaml](measurements.yaml). Those values get copied into `configs/`. Code that needs an unmeasured value raises an error instead of falling back to nominal sizes.

| Sheet | Contents (nominal) |
| --- | --- |
| board_A4 | 3×4 tag36h11, ids 0–11, 55 mm tags, 15 mm gap (shrunk from 60/20 to fit one A4 sheet) |
| box_net_A4 | Fold-up 50 mm paper cube, ids 20–25, 40 mm tags. `--check` simulates the fold against `box.yaml`. (`--box-style separate` gives loose tags for a solid box.) |
| charuco_A4 | 7×5 squares, 36 mm square, 27 mm marker, DICT_5X5_100 |

## HaWoR on a GPU machine

HaWoR can't run on CPU: its SLAM stage (masked DROID-SLAM) only builds CUDA kernels. Workflow:

```bash
# on this machine (MANO_DIR = folder with MANO_RIGHT.pkl + MANO_LEFT.pkl from mano.is.tue.mpg.de)
MANO_DIR=~/mano scripts/remote/push.sh user@GPU_IP          # code, configs, MANO, undistorted clips

# on the GPU machine (first run installs Miniforge, the hawor env, weights, runs a smoke test)
cd ~/ego2libero && tmux new -s hawor scripts/remote/process.sh

# back here
scripts/remote/pull.sh user@GPU_IP                           # -> data/interim/<session>/<demo>/hawor.npz
```

`process.sh --video any.mp4 --focal 1400` runs a single clip, useful before calibration exists. The GPU box needs Linux, an NVIDIA driver, `git`, `curl` and `libegl1`. The CUDA toolkit and gcc come from conda. CUDA 11.8 is picked automatically for GPUs up to Ada/Hopper, and CUDA 12.8 for Blackwell (the 12.8 path is less tested).

## Deviations from plan.md and findings

- **Board size:** the board uses 55 mm tags and 15 mm gaps so it fits on one A4 sheet.
- **W frame is fixed, not detected, and its origin moved.** OpenCV 5's GridBoard frame has its origin at the top-left corner of tag 0, y down the page and z *into* the table. W instead has its origin at the bottom-left corner of tag 8, with x along the bottom edge, y up the page and z up, so every board coordinate is positive. See `perception/board_pose.T_W_board`.
- **HaWoR resamples to 30 fps.** Its own frame extraction runs `ffmpeg -vf fps=30`. `hands/hawor_driver.py` extracts every decoded frame itself, so HaWoR frame *i* is our PTS index *i*. It checks the count against `timestamps.npy`.
- **`K_u` must have its principal point at the image center and fx = fy.** HaWoR hard-codes `(w/2, h/2)` in both motion estimation and SLAM. So S0 must build `K_u` that way, not use `getOptimalNewCameraMatrix` as-is. `hawor_io.check_intrinsics` enforces it.
- **HaWoR's `pred_valid` is all ones after infilling.** The observed/infilled flag (`hand_source`) is rebuilt from HaWoR's per-hand detection tracks instead.
- **The box PnP accuracy target is a percentile, not a worst case.** With 0.5 px noise at about 0.5 m, single-frame error is about 98% depth along the viewing ray. The median is 0.6 mm, the 90th percentile is under 2 mm and the max is about 2.3 mm. The test asserts the plan's 2 mm / 0.5° at the 90th percentile. Static-window averaging takes care of the rest.
- **Interpolation is decoupled.** It uses slerp for rotation and linear interpolation for translation, not screw-motion SE(3), so translation follows straight lines between samples.
- `box_pose.py` (S3 solve plus IPPE disambiguation) and `fusion/bridge.py` (S7 core) were pulled forward into M0 so the plan's synthetic tests could cover them.
- **ROS Humble on `PYTHONPATH`:** its pytest plugins are disabled in `pyproject.toml`.

- **Corner refinement is `CORNER_REFINE_APRILTAG`, not SUBPIX.** On rendered tags, SUBPIX puts corners about 0.3 px *inside* the true corner (cornerSubPix targets saddle points, and a tag's outer corner is an L). That made the cube read 4 mm too far at 0.5 m. APRILTAG refinement measured +0.00 px bias. Configurable via `pipeline.yaml: detect.refine`.
- **`K_u` principal point is `(w/2, h/2)`**, HaWoR's convention, not OpenCV's `(w-1)/2`. Its focal length is the largest that leaves no black border after undistortion.
- **"Still" detection uses windowed medians.** Speed is measured from median positions 0.25 s apart, because frame-to-frame differences turn 2 mm of pose noise into 6+ cm/s.

- **The shutter is 1/250 for demos (1/120 for calib), not 1/500.** At indoor light, 1/500 needs ISO 5000+. At 1/250 a 0.5 m/s hand blurs about 6 px at 1080p and the cube's tags 2–3 px. Raise ISO before slowing the shutter: noise hurts tag detection far less than blur.

- **Calibration uses a warp-aware fit (`--model warped`, the default).** The hand-held ChArUco sheet was bent by about 4 mm. Plain OpenCV calibration then biased the focal length 2.7% low (1321 vs about 1350 px) at 1.5 px RMS. OpenCV's `calibrateCameraRO` frees every corner and traded board shape for an implausible lens (principal point 80 px off). The warped fit uses a physical lens (fx = fy, radial only) plus one smooth board shape: 1.03 px RMS, focal length stable to about 0.7% across view subsets. Residual flex varies pose to pose, so the warped model's fail limit is 1.25 px. Independent check: `run_demo` reports `board_focal_ratio`, the focal length implied by the flat table board in each clip, divided by `K_u`'s.
- **Blackmagic zoom: use 0.5×.** On the Pixel 10, 0.5× is the native main camera (25 mm, about 71° hFOV). "1×" is a 2× digital crop (about 43°). The metadata says "25mm" for both, so `board_focal_ratio` is what catches a mismatch (about 0.5 or 2.0).

- **Single-tag ambiguity is resolved over the whole clip (Viterbi), not frame by frame.** The plan's greedy rules (nearest to the previous pose, then resting, then reprojection) failed on the first real demo: after a 1 s grasp occlusion, with the cube in the air, it locked onto the wrong IPPE mirror solution and carried a 34° tilt to the end. Now every frame keeps both candidates. The sequence minimises total rotation change, plus penalties for table penetration and, in the protocol's still start/end windows, for not resting flat. The end-rest error went from −6.1 to −1.6 mm and static rotation jitter from 3.1° to 1.75°.
- **`board_focal_ratio` is `n/a` for an overhead camera.** A board seen within 15° of head-on carries no focal-length information. With the overhead mount (2.8° from vertical), the resting-height gate is what catches focal or zoom errors.

- **Camera mount: fixed overhead (decided 2026-10-04).** This settles the plan's open "head strap or tripod" decision. The phone looks straight down from about 0.55 m (2.8° from vertical). Consequences: only the cube's top tag is usually visible, so IPPE ambiguity is the norm (handled by Viterbi); raw box rotation jitter is about 1.75°, so the static gate is relaxed to 2°; focal-length errors are nearly invisible, so the `C009` calibration is adequate here, even though a 20° test view showed it about 4% short; HaWoR's camera is static, so Sim(3) uses s = 1. Hand reconstruction from a top-down view is less typical of HaWoR's egocentric training data; check it on the pilots.

- **A release needs the box to land, not just the fingertips to leave (found 2026-10-08).** Release used to fire on 0.1 s of "fingertips off the box". HaWoR loses the hand mid-transport in about a quarter of the 2026-10-07 demos, and each dropout then ended the demo with the cube still 4–18 cm in the air and seconds from being put down — `gripper_closed` ran for as little as 0.2 s. The effect on training data was decisive: of the 53 accepted demos, the 39 whose release landed the cube on a surface recorded 33 successful sim episodes (85%), while the 14 released mid-air recorded 1 (7%). 33 + 1 is exactly the 34 episodes the first training run had. A release candidate now has to be followed by the box coming to rest within 0.5 s — free fall from 18 cm takes 0.2 s, so genuine drops still qualify. Mid-air releases went 14 → 1, and that one is a real 10.9 cm drop.
  The correction also makes the M3 gates mean something: `grasp_contact_m` and `rigidity_trans_m` are medians over the held window, and `reanchor_jump_m` depends on the wrist bridging that the held window drives. Measured over a 4-frame window they were nearly vacuous. Recording all 64 clips showed 15 of the 18 demos the old limits rejected still produce a successful sim episode, so `grasp_contact_m` went 0.015 → 0.020 and `rigidity_trans_m` 0.015 → 0.025, while `reanchor_jump_m` stayed at 0.02 (a jump is a step in the commanded tool position mid-carry, and delta actions clip at 5 cm/step). The pilots pass either way, with the same gate values to 4 decimal places. Sim success over the session went 34/53 → 58/64.

- **The box tag size is an effective, fitted value (38.63 mm; the ruler said 39).** With the overhead camera, the cube's height comes straight from the tag's apparent size, and a 1 mm ruler can't resolve the 0.75% that matters (about 4 mm at 0.53 m). It's fitted so the resting cube touches the table in static + known_pos, and validated on the three pilot demos, which it never saw: resting error went from −2 to −6 mm to within ±2.5 mm.

## Running M1 (S0–S3)

```bash
uv run python -m ego2libero.calib.calibrate data/raw/<session>/calib.mp4   # -> configs/camera.yaml
uv run python scripts/run_demo.py <session> --all                          # every clip except calib
uv run python scripts/run_demo.py <session> demo_000 --force               # one clip, recompute
```

Per clip, `data/interim/<session>/<clip>/` holds `undistorted.mp4`, `timestamps.npy`, `detections.npz`, `camera_tags.npz` (T_W_C), `box_tags.npz` (T_W_O, tag frames only), `gates.json` and `overlay.mp4`. Gates depend on the clip name (`static*`, `known_pos*`, `demo_*`) and are appended to `data/processed/<session>/summary.csv`.

## Sim task: cube to box (beyond plan.md's scope, added 2026-10-06/07)

The task is "pick the red cube and place it in the box": Panda, table, a red 50 mm cube, and an open-top box (10 × 10 cm inside, 3 cm walls, static). Success means the cube is resting inside the walls. It's built in robosuite 1.4.1, the engine LIBERO uses, with the same Panda and OSC pose controller at 20 Hz, rather than as a LIBERO task file. What matters for training SmolVLA is LIBERO's data format (cameras, 8-D state, 7-D actions), not its task definition machinery. It runs on CPU (EGL) in a separate env, `.venv-sim` (numpy < 2):

```bash
uv venv .venv-sim --python 3.10 && uv pip install --python .venv-sim/bin/python "robosuite==1.4.1" "mujoco==2.3.7" "numpy<2" scipy pyyaml av imageio && uv pip install --python .venv-sim/bin/python -e . --no-deps
MUJOCO_GL=egl .venv-sim/bin/python scripts/sim_replay.py <session> <demo> --mode physics [--init-noise 0.02 --seed N] [--no-video]
```

- **`T_L_W` is per session** (`configs/libero_frame.yaml`, loaded by `ego2libero/sim/frame.py`). Each entry puts the middle of that session's cube paths on the table centre and turns W's axes by a chosen angle. It is not cosmetic: under the pilots' placement, the 2026-10-07 cubes sit at L y ≈ −0.19 m, where a straight-down grasp drives Panda joint 6 into its limit in about 99% of frames and tracking error is 3.5–5 cm. Rotating that session's frame 180° brings it to about 1.0 cm. For replays, the box goes where the person put the cube down.
- **Grasps wait for the arm, not the clock** (`ego2libero/sim/execute.py`). The person lifts within a frame or two of touching the cube, so replaying the recorded gripper command closes the fingers while the arm is still descending. `GraspSchedule` stops advancing the demo at each grasp, keeps commanding that pose with the fingers open until the arm has arrived, then closes and holds 0.5 s before the demo resumes; the executed actions, dwell included, are what gets recorded. "Arrived" means the tool has come to rest, not a fixed distance to the cube: OSC_POSE settles 1.2–1.9 cm short of the commanded point, so a 1 cm absolute test never passes, and the open gripper often nudges the cube 2–4 cm on the way down, so a test against the cube's live position can recede faster than the arm closes on it.
- **Retargeting** (`ego2libero/sim/retarget.py`):
  - The gripper points down and takes its yaw from the hand.
  - While held, the tool point sits on the tag-tracked cube centre and closes across the pinched faces. Otherwise it follows the fingertip midpoint.
  - Yaw is centred mod 180° on the Panda's home yaw. Without that, demo_001 pinned the wrist joint at its limit.
- **Pilots, physics:** 3/3 noise-free replays put the cube in the box (0.3 / 1.5 / 1.5 cm from centre). With robosuite's default start-joint noise (0.02 rad), 24/24 succeed over 8 seeds × 3 demos.
- **Outputs:** `data/processed/<session>/sim/<demo>_physics.mp4` (real overlay | sim from the phone's direction | observer view) and `.json` metrics.
- robosuite hides geom group 0 (collision-only) when rendering, so visible task objects must be group 1.

## Status

- **M0, M1, M2:** done (see the deviations above).
- **M3 (fusion):** done on the three pilots, and every gate passes. Results:

  | | grasp / transport / release / retreat (s) | grasp contact | rigidity (trans, rot) | box gaps filled |
  | --- | --- | --- | --- | --- |
  | demo_000 | 4.83 / 6.10 / 11.72 / 12.55 | 0.57 cm | 0.78 cm, 5.0° | 63 frames interpolated |
  | demo_001 | 4.18 / 6.37 / 11.62 / 11.82 | 0.51 cm | 0.51 cm, 7.8° | 6 |
  | demo_002 | 3.88 / 5.35 / 10.67 / 10.82 | 0.87 cm | 0.53 cm, 9.6° | 9 |

  None of the pilots needed hand-based propagation and there are no re-anchoring jumps. The phase-timing gate waits for hand labels in `configs/phase_labels.yaml`.
- **M3 design decisions (from the pilot data):**
  - Phases are anchored on the box's own motion. Grasp onset is the start of fingertip contact before transport; aperture is only a weak cue for a pinch on a 5 cm cube.
  - Release is when the fingertips leave the box *and the box then lands within `release_fall_s`* (0.5 s): the pilots drop the cube, so the plan's "box at rest" rule would put release after landing, but contact alone trusts HaWoR too far. See the finding below.
  - Held-box gaps of up to 0.25 s are interpolated between tag poses. Only longer gaps use the wrist, because a pinched cube pivots 5–9° relative to the HaWoR wrist.
  - The rigidity rotation limit is 12° (was 8°) for the same reason.
  - The hand is smoothed separately over observed and HaWoR-infilled stretches.
- **Exports** per demo, in `data/processed/<session>/`: `demo_NNN.npz/.json` (native PTS), `demo_NNN_20hz.*`, and `_L` copies once `configs/libero_frame.yaml: T_L_W` is set. Load them with `ego2libero.export.write_demo.load_demo`. `plots.png` and `overlay.mp4` (phase, gripper and box source shown) are in `data/interim/<session>/<clip>/`.
- **Unattended run:** `scripts/remote/auto_finish.sh` waits for the GPU machine, pulls, and runs the hand stage. Status is in `data/processed/<session>/AUTO_STATUS.txt`.
