# Ego→Libero Perception Pipeline: Implementation Plan

Oct 3, 2026 · @Aryan

## Context and scope

This stage turns monocular phone videos of a human doing a pick-and-place into metric, time-synced 6D trajectories in one table-anchored world frame: camera pose, box pose, wrist pose, fingertips, grip aperture, and phase labels. Everything downstream consumes one file per demo from this stage.

The larger project drives a Panda arm in LIBERO from this data, then trains a policy (SmolVLA) on simulated rollouts. Because the policy trains on sim renders, not on the phone frames, fiducial tags in the real video cost nothing.

In scope:

- Phone intrinsics calibration and capture protocol
- Table frame from an AprilTag board; box 6D pose from tags on its faces
- Hand reconstruction with HaWoR, aligned into the table frame
- Occlusion bridging, SE(3) smoothing, phase segmentation
- Overlay videos and metrics to validate all of the above
- Export into LIBERO coordinates (a fixed transform; no robot logic)

Out of scope here: hand-to-gripper retargeting, trajectory optimization, sim replay, policy training. Leave a clean interface for them (see Output data format).

## Physical setup

One rigid rectangular box with a tag on every face, plus a flat tag board taped to the table. All tags are AprilTag tag36h11, detected through OpenCV's `DICT_APRILTAG_36h11`.

| Item | Spec | Tag IDs |
| --- | --- | --- |
| Box | \~50 × 50 × 90 mm, rigid (cardboard on foam core is fine). Grasp width must stay under the Panda's 80 mm max opening | 20–25, one per face, centered and edge-aligned |
| Box tags | \~40 mm black-square edge on the 50 mm faces; as large as the face allows on the long faces, same size for simplicity | — |
| Table board | 3 × 4 grid, 60 mm tags, 20 mm gap, on A3 or two A4 sheets glued flat to cardboard | 0–11 |
| Calibration target | ChArUco board printed separately, used only for intrinsics | own dictionary (e.g. `DICT_5X5_100`) to avoid ID clashes |

Generation and measurement rules:

1. `tools/print_tags.py` renders the board (`cv2.aruco.GridBoard`), the six box tags, and the ChArUco target as PDFs at a stated physical size.
2. After printing, measure the black-square edge of the tags with a ruler or calipers. Printers rescale. Put measured values in the configs, never nominal ones.
3. Measure the box's three edge lengths after assembly. Box frame: origin at the geometric center, x/y/z along the three edges, z up when the box stands on its smallest face.
4. `T_box_tag[i]` for each face is derived from box dimensions, assuming the tag is centered and edge-aligned. M2 adds an optional refinement from a short clip where the box is rotated by hand with two faces visible.
5. Board and box need a matte finish. Glossy prints flare under room lights and kill detections.

Placement target: mark the goal region on the table with tape and record its corners in the table frame once (measure from board origin). Its pose goes in `configs/task.yaml`.

## Capture protocol

Record three pilot demos first and push them through the whole pipeline before recording the full set. Capture mistakes are the most expensive bugs here.

Phone settings, identical for every clip including calibration:

- Use a manual camera app (Blackmagic Camera is free on iOS and Android) so nothing changes mid-clip.
- Video stabilization OFF, including any "action" or "enhanced" mode. Stabilization warps frames and breaks the pinhole model.
- Main 1x lens locked. No lens auto-switching, no zoom.
- Focus, exposure, ISO, white balance locked. Shutter 1/500 s or faster; add light rather than slowing the shutter.
- 1920 × 1080 at 60 fps. Same orientation always.
- Phone on a head strap for egocentric view, or a fixed tripod at head height if egocentric is dropped (open decision).

Clips to record, in order:

1. **calib**: ChArUco target moved slowly through the whole frame at varied tilts, \~60 s.
2. **static**: phone, board and box all still for 10 s. Used to measure tag-pose jitter.
3. **box\_refine** (optional): box turned slowly in hand, two faces visible at a time, 20 s.
4. **demo\_NNN**: one pick-and-place per clip.

Rules for every demo:

- Start and end with 1–2 s where hand and box are still and at least 4 board tags plus one box tag are visible. The start anchors the world frame and the initial box pose.
- One hand only (right), bare, short sleeves.
- Grasp so at least one box tag stays visible when possible, e.g. pinch across the two long faces from above.
- Keep the box inside the board area and moves smooth; no throws, no regrasps.
- Vary initial box position and yaw across demos. Target 25+ demos after the pilots pass.

Storage: `data/raw/<session>/{calib,static,box_refine,demo_000,...}.mp4` plus a `session.yaml` noting phone model, app settings and date.

## Frames and conventions

One naming rule everywhere: `T_A_B` is the pose of frame B expressed in frame A, a 4 × 4 float64 matrix with `p_A = T_A_B @ p_B`. Composition reads left to right: `T_W_O = T_W_C @ T_C_O`.

| Frame | Meaning | Definition |
| --- | --- | --- |
| `W` | World / table | Board origin; z up out of the table. Assert camera centers have z > 0, else apply a fixed flip |
| `C` | Camera | OpenCV: x right, y down, z forward |
| `O` | Box (object) | Geometric center, axes along edges (see Physical setup) |
| `G_i` | Tag i | OpenCV marker frame from `solvePnP` |
| `H` | Hand wrist | MANO root joint of the right hand |
| `L` | LIBERO world | Fixed `T_L_W` from `configs/libero_frame.yaml` |
| `S` | HaWoR world | Arbitrary SLAM frame, only used during alignment |

Other conventions:

- Units: meters, radians, seconds.
- Timestamps come from the video's presentation timestamps (PyAV), not `frame_idx / fps`. Phone video is often variable frame rate.
- Exported quaternions are `xyzw` (scipy order), hemisphere-continuous across frames.
- HaWoR's camera and world axis conventions must be confirmed by reprojection on a pilot clip before any alignment code is trusted. Write the confirmed convention into `hands/hawor_io.py` as a docstring.
- Every pose carries a per-frame `source` flag: `tag`, `propagated`, `interpolated`, or `missing`.

## Repo layout and config files

Two Python environments: a light one for this pipeline, and HaWoR's own conda env, called only through a subprocess wrapper that writes results to disk. Never import HaWoR into the main env.

Main env: Python 3.10+, `numpy`, `scipy`, `opencv-contrib-python>=4.8`, `av`, `pyyaml`, `matplotlib`, `tqdm`, `pytest`. Overlays are drawn with OpenCV; no renderer needed for this stage.

```
ego2libero/
  configs/
    camera.yaml        # K, dist, resolution (written by calib)
    board.yaml         # grid shape, measured tag size, gap, ids
    box.yaml           # measured dims, tag ids per face, tag size, T_box_tag overrides
    task.yaml          # goal region in W
    libero_frame.yaml  # T_L_W
    pipeline.yaml      # thresholds (see Pipeline stages)
  tools/print_tags.py
  ego2libero/
    se3.py             # compose, invert, log/exp, interpolate, averaging
    video.py           # PTS-accurate frame iterator
    calib/calibrate.py
    perception/detect.py, board_pose.py, box_pose.py
    hands/run_hawor.py, hawor_io.py
    align/sim3.py
    fusion/bridge.py, smooth.py, phases.py
    viz/overlay.py, plots.py
    export/write_demo.py, to_libero.py
  scripts/run_demo.py   # one demo end to end, every stage cached to disk
  tests/
  data/raw/  data/interim/  data/processed/
```

Each stage reads the previous stage's file from `data/interim/<session>/<demo>/` and is skippable when its output exists (`--force` to rerun). This keeps iteration fast while HaWoR, the slow stage, runs once per demo.

## Pipeline stages

Tags give accurate poses whenever they are visible; HaWoR gives a continuous hand track. The pipeline anchors everything on tags and uses the hand only to fill gaps. Stages run in this order; S4 runs in parallel with S1–S3.

### S0 Calibrate and undistort

- Detect ChArUco corners with `cv2.aruco.CharucoDetector`. Keep \~40 frames that are sharp (Laplacian variance above the clip median) and spread across the image; at least 12 corners each.
- `cv2.calibrateCamera` on matched object/image points, 5-parameter distortion. Target RMS reprojection < 0.5 px; fail loudly above 1.0 px.
- Undistort every clip once into `undistorted.mp4` (CRF ≤ 14) with `K_u` from `getOptimalNewCameraMatrix(alpha=0)`. Save original PTS to `timestamps.npy`. All later stages use `K_u` and zero distortion, including HaWoR.

### S1 Detect tags

- `cv2.aruco.ArucoDetector`, `DICT_APRILTAG_36h11`, `cornerRefinementMethod = CORNER_REFINE_SUBPIX`.
- Per frame, store corners for every detected ID, plus tag pixel area. Cache to `detections.npz`. Everything downstream reads this cache.

### S2 Camera pose from the board → `T_W_C(t)`

- Gather all visible board tags' corners; object points from the board layout (`board.matchImagePoints`).
- Require ≥ 2 tags. `solvePnP` with `SOLVEPNP_SQPNP`, then `solvePnPRefineLM`. Reject the frame if RMS reprojection > 2 px.
- Invert to `T_W_C`. Frames without a valid board pose are marked `missing`; S5 fills them from HaWoR's camera track.

### S3 Box pose from face tags → `T_W_O(t)`, tag frames only

- Object points per box tag = its four corners mapped through `T_box_tag[i]` into the box frame.
- **Two or more box tags visible:** one joint `SOLVEPNP_SQPNP` + LM refine over all their corners. No ambiguity.
- **One tag visible:** `SOLVEPNP_IPPE_SQUARE` returns two candidates. Pick by, in order: (1) closest to the previous accepted `T_W_O` (rotation geodesic), (2) when the box should be resting, the candidate whose lowest corner is within 1 cm of z = 0 and whose one axis is within 10° of W's z, (3) lower reprojection error. Log every ambiguous pick.
- Reject if tag area < 400 px² or RMS reprojection > 2 px.
- `T_W_O = T_W_C @ T_C_O`. Over the static start window, average poses (chordal mean for rotation, median for translation) to get a precise initial pose.

### S4 Hand reconstruction with HaWoR (own env)

- `hands/run_hawor.py` shells out to the HaWoR demo on `undistorted.mp4`, passing the calibrated focal length (look for its focal-length argument; do not let it estimate one). MANO model files are required (register at the MANO site).
- `hands/hawor_io.py` converts outputs into: right-hand MANO params, wrist pose `T_S_H(t)`, 21 joints in S, HaWoR camera poses `T_S_C(t)`, and a per-frame flag for frames HaWoR infilled rather than observed.
- Before anything else uses it, reproject the joints on 3 pilot clips to confirm axis conventions and that the right hand was selected.

### S5 Align HaWoR into W

- Use frames where both `T_W_C` (S2) and `T_S_C` (S4) exist.
- Rotation: `R_W_S` = chordal mean of `R_W_C(t) @ R_S_C(t).T`. This works even when the head barely translates, where position-only Umeyama is ill-conditioned.
- Scale and translation: linear least squares on camera centers, `c_W = s · R_W_S · c_S + t`. If camera-center spread is under 5 cm, fix s = 1 and log a warning.
- Hand in W, per frame:
  - board pose valid → `T_W_H = T_W_C(t) @ T_C_H(t)`, where `T_C_H = inv(T_S_C) @ T_S_H` with translation multiplied by s (tag camera pose is the more accurate local anchor);
  - board pose missing → `T_W_H = Sim3(S→W) applied to T_S_H`.
- Transform joints the same way. Report s, rotation residual spread, and center residual RMS.

### S6 Phase segmentation

Signals: grip aperture = thumb-tip to index-tip distance; contact distance = fingertip midpoint to box surface (box as an oriented box); box speed from S3 tag frames.

| Phase | Starts when |
| --- | --- |
| approach | clip start |
| grasp | aperture < `a_close` and contact distance < `d_contact`, confirmed by box speed > `v_move` within 0.5 s |
| transport | box speed > `v_move` |
| release | box speed < `v_rest` for 0.3 s and aperture rising |
| retreat | contact distance > `d_contact` after release |

Starting thresholds in `pipeline.yaml`, tune on pilots: `a_close` 0.06 m, `d_contact` 0.02 m, `v_move` 0.02 m/s, `v_rest` 0.01 m/s. Emit binary `gripper_closed` = 1 from grasp onset to release.

### S7 Occlusion bridging

- For each grasp-to-release interval, estimate `T_H_O` as a robust mean over frames where both the box tag pose and the hand pose exist: chordal mean rotation, median translation, after dropping the worst 20% by residual.
- Gaps inside the interval: `T_W_O(t) = T_W_H(t) @ T_H_O`, source `propagated`.
- Gaps while the box rests: hold the last resting pose, source `interpolated`.
- When tags reappear, log the jump between propagated and tag pose. Blend over 5 frames with SE(3) interpolation instead of snapping.
- Optional hand-depth fix: if the `T_H_O` residuals are large, fit one per-demo factor λ that scales `T_C_H` translation along the camera ray to minimize contact distance during grasp. Accept only if 0.85 < λ < 1.15.

### S8 Outlier removal and smoothing

- Hampel filter on translational and angular velocity per stream; drop spikes and refill by SE(3) interpolation.
- Box while resting: replace by the window's averaged pose. Box while held and wrist always: Savitzky–Golay on translation; for rotation, smooth the rotation vectors of `R_prev.T @ R(t)` in a local window and map back.
- Never smooth across a phase boundary for the box; grasp onset is a real velocity discontinuity.

### S9 Export

- Write the demo file (next section) in W at native timestamps.
- Write a second copy resampled to 20 Hz (LIBERO's control rate), SE(3)-interpolated.
- `export/to_libero.py` applies `T_L_W` to produce L-frame copies. Nothing robot-specific happens here.

&#91;embedded content: data flow · S0–S9, two branches merging at S5\]

The tag branch and HaWoR run independently and meet at S5; from S6 on, every stage needs both.

## Output data format

Each demo produces `data/processed/<session>/<demo>.npz` (arrays, W frame, native timestamps), `<demo>_20hz.npz` (same keys, resampled), `<demo>_L.npz` and `<demo>_20hz_L.npz` (LIBERO frame), and `<demo>.json` (metadata). `export/write_demo.py` also provides `load_demo(path) -> dict` so the retargeting stage never parses files by hand.

| Key | Shape | Meaning |
| --- | --- | --- |
| `t` | (N,) | Seconds from clip start, from PTS |
| `T_W_C` | (N, 4, 4) | Camera pose |
| `T_W_O` | (N, 4, 4) | Box pose |
| `T_W_H` | (N, 4, 4) | Right wrist pose |
| `joints_W` | (N, 21, 3) | MANO joints |
| `mano_pose`, `mano_betas` | (N, 48), (10,) | Raw MANO params for later re-fitting |
| `aperture` | (N,) | Thumb-to-index tip distance, m |
| `gripper_closed` | (N,) bool | 1 between grasp onset and release |
| `phase` | (N,) uint8 | 0 approach, 1 grasp, 2 transport, 3 release, 4 retreat |
| `cam_source`, `obj_source`, `hand_source` | (N,) uint8 | 0 tag/observed, 1 propagated, 2 interpolated, 3 missing; hand uses 0 observed, 2 HaWoR-infilled |
| `board_reproj_px`, `box_reproj_px` | (N,) | Per-frame RMS, NaN where not measured |

`<demo>.json` holds: `K_u`, image size, box dimensions, `T_L_W`, Sim(3) result (s, residuals), phase boundary timestamps, thresholds used, count of ambiguous single-tag picks, λ if applied, and the git commit hash.

## Validation and acceptance metrics

A demo is accepted only if it passes every gate below; `scripts/run_demo.py` prints a pass/fail line per gate and appends a row to `data/processed/<session>/summary.csv`. Targets are starting values; tighten them once pilots show what the setup can do.

| Gate | Target | Measured on |
| --- | --- | --- |
| Calibration RMS | < 0.5 px | calib clip |
| Camera pose jitter | center std < 2 mm, rotation std < 0.2° | static clip |
| Box pose jitter | position std < 3 mm, rotation std < 1° | static clip |
| Box resting on table | lowest corner z within ±5 mm of 0 | every demo's start and end windows |
| Known-position check | < 1 cm error at two taped, ruler-measured spots | one dedicated clip |
| Sim(3) sanity | 0.8 < s < 1.2; median rotation residual < 3° | every demo |
| Grasp contact | median contact distance < 1.5 cm during grasp | every demo |
| Hand–box rigidity | `T_H_O` residual < 1.5 cm and < 8° while held | every demo |
| Re-anchor jump | < 2 cm when tags reappear | every demo |
| Coverage | ≥ 60% of frames with box pose from tags; 0 missing after S7 | every demo |
| Phase timing | within ±3 frames of hand labels | 3 pilot demos, labeled manually |

Visual artifacts per demo, the fastest debugging tool and good material for the final write-up:

- `overlay.mp4`: board axes, box wireframe colored by `obj_source`, hand skeleton, phase name, gripper state, reprojection error text.
- `plots.png`: box and wrist position and rotation vs time with source shading; aperture, contact distance and box speed with threshold lines and phase boundaries.

Unit tests (`tests/`), all on synthetic data so they run without video:

- [ ] `se3`: inverse, compose, log/exp round trips; interpolation endpoints and continuity
- [ ] Sim(3) alignment recovers a random known similarity transform under noise
- [ ] Box PnP: project a known box pose through `K`, add 0.5 px noise, recover pose within 2 mm and 0.5°
- [ ] IPPE disambiguation picks the right candidate given a prior pose
- [ ] Bridging: a synthetic held trajectory with a gap is reconstructed exactly when `T_H_O` is exact

## Milestones

Three tracks start at once, because the code agent can build and test most of the geometry before any real video exists, and HaWoR setup is the slowest single task.

**M0: in parallel, start immediately**

- [ ] Agent: `se3.py`, `align/sim3.py`, `video.py`, `tools/print_tags.py`, and all synthetic unit tests passing
- [ ] Aryan: print tags, measure them, build box and board, fill `board.yaml` and `box.yaml`; record calib, static and 3 pilot demos
- [ ] HaWoR env: install in its own conda env on the GPU machine, get MANO files, run its stock demo on any sample video

**M1: tag geometry on pilots (S0–S3)**

- [ ] Calibration, undistortion, detection cache, board pose, box pose, overlay video

* Done when: calibration, camera jitter, box jitter, and box-resting gates pass on pilots

**M2: hands into W (S4–S5)**

- [ ] HaWoR wrapper and IO, convention check by reprojection, Sim(3) alignment, hand added to overlay

* Done when: hand skeleton visibly sits on the hand in the overlay, Sim(3) gate passes on all 3 pilots

**M3: fusion (S6–S8)**

- [ ] Phase segmentation, occlusion bridging, outlier removal, smoothing, plots

* Done when: phase timing, grasp contact, rigidity, re-anchor and coverage gates pass on pilots

**M4: export and full dataset (S9)**

- [ ] Export files, `load_demo`, `summary.csv`, LIBERO-frame copies
- [ ] Record the full set (25+ demos), batch-run, drop demos failing any gate

* Done when: the retargeting stage can load every accepted demo with one call

## Known pitfalls and open decisions

Most failures in this kind of pipeline come from capture and conventions, not algorithms. Check these before debugging math.

| Pitfall | Symptom | Guard |
| --- | --- | --- |
| Variable frame rate | Hand and tag streams drift apart in time | PTS timestamps only (S0) |
| Stabilization left on | Board pose jitters even on the static clip | Settings check; static gate |
| Printer scaling | Box floats or sinks a few mm; distances off by a constant % | Measured sizes in configs; known-position gate |
| Tag stuck off-center | Constant offset in box pose, rigidity residual stays high | `box_refine` clip to estimate `T_box_tag` |
| Single-tag pose flip | Box rotation jumps by tens of degrees for a few frames | IPPE disambiguation + Hampel filter; inspect the log |
| Motion blur in transport | Tag dropouts mid-move | Faster shutter, more light, slower motion |
| HaWoR picks the wrong hand or infills off-screen frames | Skeleton on the wrong hand, or a confident wrist with no hand visible | Right-hand selection check; `hand_source` flag; ignore infilled frames when estimating `T_H_O` |
| Monocular hand depth error | Fingertips float 2–5 cm in front of or behind the box at grasp | Contact gate; optional λ fix in S7 |
| Quaternion sign flips | Smoothed rotation spins wildly | Hemisphere fix before any filtering |

Open decisions for Aryan before M4 (the agent should leave config placeholders, not guess):

- [ ] Egocentric head strap or fixed tripod at head height
- [ ] Exact task and goal region (single place-on-target, or more)
- [ ] Which LIBERO scene, and where the board origin maps in it (`T_L_W`), given the Panda's reach
- [ ] Final demo count