# Handoff: ego2libero (phone video → sim → SmolVLA)

State as of 2026-10-08. The design is in [plan.md](plan.md), technical decisions and deviations are in [FINDINGS.md](FINDINGS.md), and the recording how-to is in [GUIDE.md](GUIDE.md). This file covers what was done, where everything lives, and what's next.

---

## 1. What exists and works

**Perception pipeline (plan.md M0–M3), complete and tested (61 tests, `uv run pytest`):**

- **S0, calibration:** warped-board model, since the hand-held ChArUco sheet was bent about 4 mm. `K` is in `configs/camera.yaml` (f ≈ 1344 px, 1080p60, Blackmagic **0.5×** = native main camera).
- **S1–S3:**
  - AprilTag detection uses `CORNER_REFINE_APRILTAG`, because SUBPIX is biased.
  - Board → `T_W_C`.
  - Cube pose with Viterbi disambiguation of single-tag IPPE flips.
  - Effective tag size is fitted (38.63 mm).
- **S4, HaWoR on the GPU box:** a wrapper driver that extracts every frame itself (HaWoR resamples to 30 fps otherwise) and passes the calibrated focal length.
- **S5, hands into W:** Sim(3) fixes scale = 1 when either camera track is static, because SLAM drift broke this once.
- **S6–S8, fusion:**
  - Phases are anchored on cube motion. Release = fingertips leave the cube **and the cube then lands within 0.5 s**; see section 3, this is the fix that mattered.
  - Short tag gaps are interpolated; long gaps are bridged from the wrist.
  - Outlier removal and phase-aware smoothing.
  - Hand smoothing is done separately for observed vs HaWoR-infilled frames.
- **S9, export:** native-rate files, 20 Hz files, and `_L` (sim-frame) files, with an `accepted` flag from all gates.
- **Gates:** M1 (jitter, resting height, known spots) and M3 (contact, rigidity, re-anchor, coverage) all run in `scripts/run_demo.py`.

**Sim (beyond plan.md):** robosuite 1.4.1 cube-to-box task (`ego2libero/sim/scene.py`, class `CubeToBox`): Panda, a red 50 mm cube, an open blue box, and success = cube resting in the box. It runs on CPU with EGL in `.venv-sim`.

- `scripts/sim_replay.py`: replays a demo with absolute OSC, box placed at the demo's own put-down spot. Writes a video and metrics.
- `scripts/sim_record.py`: records training episodes with **delta** OSC actions (LIBERO-style 7-D), box fixed at the session's median put-down spot, 3 cameras (agentview, wrist, sideview, 256²), 8-D state. Keeps successes only.
- `ego2libero/sim/retarget.py`: hand → top-down gripper. The tool point sits on the cube centre while held, and yaw is centred modulo 180°.
- `ego2libero/sim/frame.py`: **`T_L_W` is per session** (`configs/libero_frame.yaml`). The pilots' placement put the 2026-10-07 cubes where a top-down grasp pins Panda joint 6 against its limit; that session now uses a frame rotated 180° about its own cube-path centre.
- `ego2libero/sim/execute.py`: `GraspSchedule`, which holds the demo at each grasp until the arm has actually arrived, then closes and lets the fingers seat before the demo resumes.

**Training:** LeRobot 0.4.4 with SmolVLA (`lerobot/smolvla_base`) on the GPU box.

- `scripts/remote/to_lerobot.py`: episodes → LeRobot datasets, strictly from `split.json`. Cameras map to camera1/2/3 = agent/wrist/side; base expects 3.
- `scripts/remote/train_smolvla.sh`: batch-size probe (64 fits on the 5090), `--save_freq=2000`, `--dataset.video_backend=pyav` (torchcodec is broken with torch 2.7).
- `scripts/remote/val_watch.py`: validation loss per checkpoint into `val_loss.csv`, with a fixed seed.

---

## 2. Data and where it lives

**This machine (`aryan2`, no GPU):**

| What | Path |
| --- | --- |
| Setup session: calib, static, known_pos ×2, pilots demo_000–002 | `data/raw/2026-10-03_setup/` |
| Dataset session: 64 demos (C026–C090), 5 cube start positions | `data/raw/2026-10-07_dataset/` |
| Per-clip intermediates (undistorted, detections, poses, hawor.npz, hands_W, overlays, plots) | `data/interim/<session>/<clip>/` |
| Exports, `selection.csv/json`, `OVERNIGHT.txt`, `AUTO_STATUS.txt`, `sim/` replay metrics | `data/processed/<session>/` |
| **Current sim episodes (all 64 clips recorded, 58 succeeded) + `index.json` + `split.json`** | `data/sim_episodes/2026-10-07_dataset/` (3.0 GB) |
| Episodes from the stopped run, superseded — kept only for its `index.json` (the old `box_xy_L`, needed to eval the old checkpoints faithfully) | `data/sim_episodes/2026-10-07_dataset.pre-graspfix/` (1.8 GB) |
| Loss curves + logs of the stopped run | `data/processed/2026-10-07_dataset/train/` (`loss_curves.png`, `val_loss.csv`, `train.log`) |
| All checkpoints of the stopped run (13 × 1.26 GB, steps 2k–26k; copy finished, `copy.log` says `rc=0`) | `data/checkpoints/smolvla_cube2box_2026-10-07_dataset/checkpoints/` |

**Disk is tight: 4.4 GB free of 184 GB (98% full).** The two obvious candidates if more is needed are the 1.8 GB `.pre-graspfix` episodes (keep its `index.json`) and the 16 GB of checkpoints, of which only `004000` is likely worth keeping once the eval is done.

**GPU box (`aryan@100.94.141.33`, RTX 5090, via Tailscale share):**

- Everything is in `/mnt/ssd2/ego2libero` (= `~/drive2/ego2libero`).
- It has its own Miniforge in `third_party/miniforge3`, with envs `hawor` and `lerobot`, addressed by path. **Never** use the user's own conda: their `~/.condarc` redirects named envs into `~/miniforge3`.
- The user asked that nothing outside this folder be touched.
- MANO files were copied from `~/painting pipeline/phantom/submodules/phantom-hamer/_DATA/data/mano/`.
- **The box is shared.** As of 2026-10-08 18:00 a `pi05` LeRobot run (`Deviant65/so101_redcube_3cam`, 30k steps) is on the GPU under the same `aryan` login but belongs to a different project, and runs from `/home/aryan/miniforge3`. Other users (`kalki`, `rajat`, `rahul`) are logged in but run only desktop sessions. Check `nvidia-smi` before starting anything and leave other people's jobs alone.

---

## 3. The stopped training run: what actually went wrong

It trained on **33 episodes with 1 validation episode** (P2 only) instead of 50/10, and was stopped at about step 27k of 40k. Validation loss bottomed at 0.057 at step 4,000 and rose to 0.111 by 26,000 while training loss fell to 0.015 — clear overfitting. If anything from that run is used, use **`checkpoints/004000`**.

The handoff written before 2026-10-08 blamed two sim-execution problems. Both were real, but **neither was the main cause.** The main cause was a defect in release detection:

- Release fired on 0.1 s of "fingertips off the cube", measured from HaWoR. HaWoR loses the hand mid-transport on about a quarter of this session's demos, and each dropout ended the demo with the cube still **4–18 cm in the air** and seconds from being put down. `gripper_closed` ran for as little as 0.2 s, so `sim_record.py` opened the gripper in mid-air and truncated the episode 1.5 s later. The cube could not reach the box.
- The split across the 53 accepted demos was decisive: the **39** whose release landed the cube on a surface recorded **33 successful** sim episodes (85%); the **14** released mid-air recorded **1** (7%). 33 + 1 is exactly the 34 episodes that run had.

**Fixed** in `ego2libero/fusion/phases.py`: a contact-loss candidate only counts as a release if the cube comes to rest within `release_fall_s` (0.5 s). Free fall from 18 cm takes 0.2 s, so genuine drops still qualify. Mid-air releases went 14 → 1, and that one is a real 10.9 cm drop. The pilots are completely unaffected — same onsets, same gate values to 4 decimal places.

The two execution problems from the earlier handoff were also fixed, and both were worth fixing:

1. **Workspace placement (`T_L_W`).** Adopted the rotated-180° frame from `data/processed/2026-10-07_dataset/tlw_best.json`, now recorded per session in `configs/libero_frame.yaml` and loaded by `ego2libero/sim/frame.py`. `scripts/run_demo.py` uses it for the `_L` exports too.
2. **Grasp timing.** `GraspSchedule` (`ego2libero/sim/execute.py`) holds the demo at each grasp until the arm has arrived, then closes and holds 0.5 s. Note the criterion is **"the arm has stopped moving", not "within 1 cm of the cube"**: OSC_POSE settles 1.2–1.9 cm short of its commanded tool point, so a 1 cm absolute test never passes and just burns the timeout; and the open gripper often nudges the cube 2–4 cm on the way down, so a test against the cube's live position can recede faster than the arm closes on it.

Measured on a 10-demo sample, 2 per start position:

| configuration | sample success |
| --- | --- |
| as the run was stopped | 6/10 |
| release fix only | 8/10 |
| release fix + grasp timing | **9/10** |

---

## 4. Current state of the data (2026-10-08)

All 64 clips were re-recorded after the fixes: **58/64 succeed in sim** (was 34/53).

The release fix also made the M3 gates mean something. `grasp_contact_m` and `rigidity_trans_m` are medians over the held window, and `reanchor_jump_m` depends on the wrist bridging that window drives — measured over a 4-frame window they were nearly vacuous. With correct windows, 7 more demos failed, and 15 of the 18 rejected demos still produced successful sim episodes. The thresholds had been calibrated on 3 pilot demos against the truncated windows, so two of them were re-calibrated (`configs/pipeline.yaml` carries the reasoning):

- `grasp_contact_m` 0.015 → **0.020**
- `rigidity_trans_m` 0.015 → **0.025**
- `reanchor_jump_m` deliberately left at 0.02: a re-anchor jump is a step in the commanded tool position mid-carry, and delta actions clip at 5 cm/step, so it is a real defect in the action trace even when the cube still lands in the box.

Result: **54 accepted, 51 eligible** (accepted and sim-successful).

| | P1 | P2 | P3 | P4 | P5 | total |
| --- | --- | --- | --- | --- | --- | --- |
| eligible | 9 | 7 | 13 | 13 | 9 | 51 |
| train (9/position) | 9 | 7 | 9 | 9 | 9 | **43** |
| validation | 0 | 0 | 4 | 4 | 0 | **8** |

43 train / 8 val, written to `data/sim_episodes/2026-10-07_dataset/split.json`; 15,733 frames, 13.1 min at 20 Hz. Action saturation is low (position 1.4% of steps, rotation 0.7%) and the gripper is closed 55% of steps.

**Known weakness in this split:** validation draws only from P3 and P4. P2 has just 7 eligible demos in total, so no per-position count reaches 9 everywhere with episodes left over. Giving back a few training slots — roughly 20% per position, which would be about 40 train / 11 val with all five positions represented in both — would make validation loss mean something at every start position. That was raised and the 9-per-position split was chosen; it is a one-line change to `scripts/final_split.py` if you want to revisit it.

---

## 5. Next steps (in order)

1. **Evaluate the stopped run's 13 checkpoints in sim** (harness not built yet). Roll out each checkpoint in `CubeToBox` from each start position and report the in-box success rate. Two prerequisites, neither trivial:
   - LeRobot needs torch, which is in neither `.venv` nor `.venv-sim`. Either add a CPU torch/LeRobot env here (CPU inference is viable, since SmolVLA predicts 50-action chunks) or install robosuite on the GPU box.
   - **The eval must reconstruct the old scene.** Those checkpoints were trained under the pilots' `T_L_W` with the box at `(-0.022, 0.118)`; the session now uses the rotated frame, which puts the box at `(0.155, 0.036)`. Rolled out in the current config the policy sees a workspace rotated 180° from its training data and scores ~0. The old frame is still the top-level default in `configs/libero_frame.yaml`, and the old `box_xy_L` is in `data/sim_episodes/2026-10-07_dataset.pre-graspfix/index.json`.
   - Note these numbers describe a policy trained on the defective data, so they will not carry over to the retrained run.
2. **Retrain** on the GPU box: `bash scripts/remote/train_smolvla.sh 2026-10-07_dataset 40000`, in tmux, after uploading `data/sim_episodes/<session>/` and `scripts/`. Expect validation loss to bottom out well before 40k steps; pick the checkpoint by validation loss. **Wait for the GPU to be free** — see section 2.
3. **Evaluate the retrained policy** with the same harness, in the current scene.

Both 1 and 2 are waiting on the GPU being free; the user will say when.

**Automation that already exists:**
- `scripts/overnight.sh`: smoke test → wait → record → split → train. Still valid; it re-reads the accepted list and deletes `data/sim_episodes/<session>/` before recording.
- `scripts/remote/auto_finish.sh`: wait for HaWoR → pull → export → sim replay.
- Both run detached (`setsid nohup` locally, tmux remotely).

---

## 6. Open decisions and caveats for the user

- **P2 is thin and it is a data problem, not a processing one.** Only 7 of its 11 clips are usable; the failures are re-anchor jumps and loose grasps, not gate calibration. If per-position balance matters, P2 (and to a lesser extent P1 and P5) wants a few more recorded demos.
- **plan.md open items still undecided:** final demo count beyond 64, the real LIBERO scene (the sim is a robosuite tabletop, not a LIBERO task file), and the phase-timing gate (needs hand labels in `configs/phase_labels.yaml`).
- **Camera mount** is fixed overhead (decided). The calibration focal length is about 4% uncertain; that's invisible overhead, but it matters if the camera is tilted.
- The **hollow paper cube** was never filled. Rigidity gates still pass, with a 5–9° pinch pivot.
- `demo_C056` is the one sample demo that still fails under `sim_record`. It succeeds under `sim_replay --mode physics` (7.5 mm tracking, 1.4 mm final error), so it is a delta-OSC saturation case rather than bad data.

---

## 7. Gotchas learned the hard way

- **Trust the cube, not the hand.** The tag-tracked cube pose is the reliable signal; HaWoR is not. Any rule keyed on fingertips needs a physical cross-check against the cube — that is exactly what the release bug was.
- **A gate measured over a window that another bug controls is not a gate.** Three M3 gates looked like they were passing for a year of demos because the held window they average over had been truncated to a few frames.
- **robosuite:**
  - Hides geom group 0 when rendering, so visible objects need group 1.
  - Randomises start joints by default; pass `initialization_noise=None` for repeatable runs.
  - OSC_POSE settles 1.2–1.9 cm short of a commanded tool point under gravity and joint limits. Do not write convergence tests against an absolute distance.
- **HaWoR on Blackwell (5090):** needs CUDA 12.8 / torch 2.7. DROID-SLAM needs sm_120 gencode (`scripts/remote/patch_droid.py`) and needs `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`.
- **Conda in scripts:** activation hooks break under `set -u`.
- **Bash scripts:** edit only when not running (rsync replaces the file safely).
- **LeRobot 0.4.4:**
  - Dataset roots must be absolute paths.
  - `torchcodec` fails with torch 2.7, so use `--dataset.video_backend=pyav`.
  - `smolvla_base` expects `observation.images.camera1..3`.
  - The train log rounds loss to 2 significant figures.
- **Transfers:** clips copied through WhatsApp or app transfers get re-encoded or arrive incomplete. The pipeline refuses truncated files (`TruncatedVideoError`).
