Hi team at humanoid!

I'm Aryan currently a founding engineer at a small Robot Learning startup here in Delhi.

After my recent furlough into dealing with Human ego centric data, the problem proposed stands pretty clear and breakable into sub problems.

With ego centric data and training robot policies from it the paths are clear:
1) if the policy is to be deployed in the real world, and ego centric data is the only source, we need express photo realism since VLA policies tend to flunk performance as soon as you go from sim to real, or even just real to real.
    - so to make ego centric robot policies perform in real world using the standard VLA training process (collect data --> convert it into a suitable format(lerobot being wiodely used) --> train the model) we have two options
        - either find a way to make the ego centric data resemble robot data(image wise) or
        - freeze the vision tower of the VLA, train the dedicated action experts(flow matching in pi0.5, dit in gr00t N1.7), then use a small subset of real robot data to train on this checkpoint. This should ideally reduce the number of episodes we need for getting good performance for a fresh robot embodiment while minimizing the requirement for expensive to collect real world data


For training robot policies using ego centric data, the steps are clear
1) we need to extract the actual human trajectory in the video
2) smoothe that human trajectory to optimize it for our robot
3) track the 6dof position of the object being manipulated
4) render everything into robot data in a simulation

Following this checklist, I've implemented the following approach.

1) For extracting human trajectory from the video I've used HaWoR (https://arxiv.org/pdf/2501.02973). HaWoR reconstructs the hand in a *world* frame rather than the camera frame, recovering MANO pose plus a camera trajectory from its own SLAM. That is the output a retargeter needs: a gripper pose has to be expressed in the table frame, not relative to a moving camera. Methods that stop at camera-frame hand pose leave that transform for you to solve separately.
2) Smoothing is in `ego2libero/fusion/smooth.py`, applied in `fusion/fuse.py`. Three stages:
    - **Outlier rejection (Hampel).** A frame is dropped if its position deviates from the local 7-frame median by more than 3 robust sigmas (1.4826 x MAD) *and* by at least 3 mm; same test on orientation against the local chordal-mean rotation with a 1.5 deg floor. The two floors matter: without them the test fires on sub-millimetre noise during the static stretches. Dropped frames are refilled by decoupled SE(3) interpolation (slerp on rotation, linear on translation).
    - **Savitzky-Golay (window 9, order 2)** on translation, and on rotation vectors taken relative to each run's mean rotation so the filter only ever sees small angles.
    - **Segment-aware application.** The filter never runs across a phase boundary. Grasp onset and the box landing are real velocity discontinuities; smoothing through them rounds off exactly the events the retargeter keys on. Resting stretches are collapsed to their robust mean pose instead of filtered.

    The hand and the cube are smoothed separately, and the hand is smoothed *before* phases are estimated, since phase detection keys on a clean wrist. Observed and HaWoR-infilled stretches are smoothed as separate runs so the filter never blends a real measurement into an interpolated one.
3) I opted for a very rudamentary AprilTag based tracking for the cube. I considered an alternate pipeline which was, camera frame --> sam2 for object segmentation --> depthanything v2 for finding the depth.
    - this had 2 concerns. First since I'm going to be rendering the object in sim, the AprilTag accuracy and determinism is nearly free, since i dont need to work on making it look realistic or anything.
    - i would still lack the orientation of the tracked object that might make things finicky. 
4) A similar April Tag board was used as a world frame anchor. 

Now everything that broke:

**1) HaWoR and the world frame**

- **SLAM drift with a static camera poisons the Sim(3) fit.** HaWoR's world S is aligned to my table frame W by a similarity fit on paired camera poses. With a fixed overhead phone, the true camera spread is ~1.8 mm, but SLAM reported 6.4 cm of spurious motion. A least-squares fit of a drifting track against a static one drives the scale toward zero, which collapses the whole hand trajectory. The fix is to check the spread of *both* tracks and fix s = 1 when either is below 5 cm, since scale is simply unobservable from a camera that does not translate.
- **Rotation and scale have very different conditioning.** Rotation is recovered from camera *orientations*, which stay well conditioned even when the camera barely moves; only scale and translation come from camera centres. Fitting all three jointly fails for the same reason as above.
- **The hand is anchored per-frame on the tags, not on SLAM.** Where a board pose exists, the hand is hung off the tag-derived camera (`T_W_H = T_W_C @ T_C_H`) rather than mapped through the global Sim(3). The tag solve is the more accurate local anchor; the global fit is only the fallback for frames with no board.
- **HaWoR silently resamples to 30 fps.** Its frame extraction runs `ffmpeg -vf fps=30`, so frame *i* of its output is not PTS index *i* of a 60 fps clip. I wrote a driver that extracts every decoded frame itself and asserts the count against the recorded timestamps.
- **`K_u` must use HaWoR's convention.** It hard-codes the principal point at `(w/2, h/2)` and assumes `fx == fy` in both motion estimation and SLAM, so undistortion has to build `K_u` that way rather than taking `getOptimalNewCameraMatrix` as-is.
- **`pred_valid` is all ones after infilling.** HaWoR's own validity flag is useless downstream, so the observed-vs-infilled flag is rebuilt from its per-hand detection tracks. This matters because infilled frames must not be smoothed together with real ones.

**2) Action smoothing**

- **The big one was not smoothing but phase segmentation feeding it.** Release was detected as "fingertips leave the cube", confirmed over only 0.1 s. HaWoR loses the hand mid-transport on about a quarter of my demos, and each dropout registered as a release while the cube was still 4-18 cm in the air. The gripper signal then ran for as little as 0.2 s. Measured across 53 demos: the 39 whose release left the cube on a surface produced 33 successful sim episodes (85%), the 14 released mid-air produced 1 (7%). Fixed by requiring the cube to come to rest within 0.5 s of a candidate release, since free fall from 18 cm takes 0.2 s and a genuine drop still qualifies.
- **Smoothing across phase boundaries destroys the events you are detecting.** Grasp onset and landing are real discontinuities, so the filter is applied per segment.
- **HaWoR glitches are large, not subtle.** Some frames put fingertips 1.2 m away and 0.6 m below the table while still flagged observed, so outlier rejection needs robust statistics (MAD) rather than a variance threshold, plus absolute floors so it does not fire on noise during static stretches.

**3) Tracking**

- **Corner refinement choice shifted the cube by 4 mm.** OpenCV's `CORNER_REFINE_SUBPIX` targets saddle points, but a tag's outer corner is an L, so it lands ~0.3 px *inside* the true corner. At 0.5 m that read the cube 4 mm too far. `CORNER_REFINE_APRILTAG` measured +0.00 px bias.
- **A single visible tag is pose-ambiguous, and an overhead camera makes that the normal case.** With the phone looking straight down, usually only the cube's top tag is visible, so IPPE returns two mirror solutions every frame. Greedy frame-by-frame disambiguation failed on the first real demo: after a 1 s grasp occlusion it locked onto the wrong branch and carried a 34 deg tilt to the end of the clip. Replaced with a Viterbi pass over the whole clip that keeps both candidates per frame and minimises total rotation change, with penalties for table penetration and for not resting flat during the protocol's static windows. End-rest error went from -6.1 mm to -1.6 mm, static rotation jitter from 3.1 deg to 1.75 deg.
- **The tag size that matters is an effective, fitted value.** With an overhead view the cube's height comes almost entirely from the tag's apparent size, and a 1 mm ruler cannot resolve the 0.75% that matters (~4 mm at 0.53 m). Fitting it so the resting cube touches the table gave 38.63 mm against a ruler reading of 39, and validated on three held-out pilots: resting error went from -2..-6 mm to within +-2.5 mm.
- **Calibration had to model a bent board.** The hand-held ChArUco sheet was bowed ~4 mm. Plain OpenCV calibration absorbed that into focal length, biasing it 2.7% low at 1.5 px RMS; `calibrateCameraRO` traded board shape for an implausible lens (principal point 80 px off). A warped-board fit with a physical lens model gave 1.03 px RMS and focal length stable to 0.7% across view subsets.
- **An overhead camera cannot self-check its focal length.** A board seen within 15 deg of head-on carries no focal information, so the usual consistency check reads n/a and the resting-height gate is what catches a zoom or calibration error instead.
- **Phone zoom is a trap.** On the Pixel 10, Blackmagic's "0.5x" is the native main camera and "1x" is a 2x digital crop, but the metadata reports 25 mm for both.

**4) Rendering and sim**

- **numpy 2 silently changes what MuJoCo renders.** Same mujoco, same PyOpenGL, byte-identical physics (eef and joint positions matched to 8 decimals), but images came out ~14 levels darker on 0-255. For a vision policy that is off-distribution input: the same checkpoint scored 20/40 under numpy 1.26 and 7-9/40 under numpy 2 on identical start poses. I found this only after an entire residual-RL study had been run against the crippled baseline. The lesson generalises: I had verified that robosuite *ran* under numpy 2, not that it produced the same *output*. A render checksum would have caught it in one command.
- **Workspace placement is not cosmetic.** The frame mapping the table into the simulator decides whether the arm can physically do the task. My first placement put the cubes where a straight-down grasp drove the Panda's joint 6 into its limit on ~99% of frames, giving 3.5-5 cm of tracking error. Rotating that session's frame 180 deg about its own cube-path centre brought it to ~1.0 cm. It is now per session rather than global.
- **Human timing is not robot timing.** People lift within a frame or two of touching the object. Replaying the recorded gripper signal closes the fingers while the arm is still descending, so it grips air or one corner. The executor now holds at each grasp until the arm has actually come to rest, then closes and lets the fingers seat. Worth noting the criterion is "the arm stopped moving", not "within 1 cm of the cube": OSC settles 1.2-1.9 cm short of its commanded point, so an absolute distance test never passes and just burns its timeout.
- **robosuite specifics:** it hides geom group 0 when rendering, so visible task objects must be group 1; and it randomises start joints by default, which has to be disabled for repeatable evaluation.



Improving performance









---

## Running this

### Layout

```
ego2libero/          the library: perception, alignment, fusion, sim, retargeting, residual RL
scripts/             entry points (one stage each) + scripts/remote/ for GPU boxes
configs/             measured constants. Physical sizes are measured values; nulls fail loudly
tests/              70 tests, no GPU or recorded data needed
results/             eval JSON, val-loss curves, demo selection, residual run logs
tags/                printable A4 PDFs: ChArUco calibration board, world board, cube net
notebooks/           Colab notebook for residual RL on a rented GPU
plan.md              the original design, stage by stage (S0-S9)
FINDINGS.md          technical decisions and every deviation from the plan, with the measurement
GUIDE.md             how to record a session with a phone
HANDOFF.md           current state, what is done, what is next
```

### Environments

Three, because they genuinely conflict:

```bash
uv venv .venv  --python 3.11 && uv pip install -e ".[dev]"        # perception + tests
uv venv .venv-sim --python 3.10 && uv pip install --python .venv-sim/bin/python \
    "robosuite==1.4.1" "mujoco==2.3.7" "numpy<2" scipy pyyaml av imageio && \
    uv pip install --python .venv-sim/bin/python -e . --no-deps   # simulator
uv venv .venv-eval --python 3.10 && uv pip install --python .venv-eval/bin/python \
    torch "lerobot[smolvla]==0.4.4" "robosuite==1.4.1" "mujoco==2.3.7" "numpy==1.26.4"
```

**Pin `numpy<2` wherever the simulator runs.** `lerobot` pulls `rerun-sdk`, which wants `numpy>=2`,
but rerun is never imported on the inference path. numpy 2 silently changes what MuJoCo *renders*
(same mujoco, same PyOpenGL, byte-identical physics, images ~14 levels darker on 0-255). For a
vision policy that is off-distribution: the same checkpoint scored 20/40 under numpy 1.26 and
7-9/40 under numpy 2. See `ego2libero/sim/base_policy.py`.

### Perception, one clip or a whole session

```bash
uv run pytest                                              # 70 tests, ~13 s, no data needed
uv run python scripts/run_demo.py <session> <clip>         # one clip, every stage cached
uv run python scripts/run_demo.py <session> --all          # whole session
uv run python scripts/run_demo.py <session> --all --no-overlay    # skip video rendering (fast)
```

Each stage writes to `data/interim/<session>/<clip>/` and is skipped if its output already exists;
`--force` reruns everything. Gate results land in `gates.json`, exports in `data/processed/`.

The hand stage needs HaWoR, which needs a GPU:

```bash
scripts/remote/push.sh user@gpu-host <session>     # send undistorted clips
scripts/remote/process.sh <session>                # on the GPU box: run HaWoR
scripts/remote/pull.sh user@gpu-host <session>     # fetch hawor.npz back
```

### Simulation and training data

```bash
# replay one demo to see the retargeting (writes side-by-side video + metrics)
MUJOCO_GL=egl .venv-sim/bin/python scripts/sim_replay.py <session> <demo> --mode physics

# record training episodes from every accepted demo
MUJOCO_GL=egl .venv-sim/bin/python scripts/sim_record.py <session>

# train/val split (default: every eligible episode trains, nothing held out)
uv run python scripts/final_split.py <session>
```

On the holdout: checkpoints are chosen by **rolling the policy out in the task**, not by validation
loss. On this dataset validation loss ranked checkpoints *backwards* -- success correlated with
val loss at Spearman +0.83 (p = 0.0005), so the run that was stopped early for "overfitting" was
still improving. `scripts/final_split.py --per N` restores a holdout if you want one.

### Training SmolVLA

```bash
scripts/remote/push_cloud.sh user@gpu-host         # stage dataset + scripts, upload
# then on the box:
bash cloud_setup.sh 40000                          # installs, probes batch size, trains in tmux
```

`cloud_setup.sh` probes the largest batch that fits before committing (64 on a 48 GB card) and
checkpoints every 2k steps, so a run can be stopped at any point and still be useful.

### Evaluating a checkpoint

```bash
MUJOCO_GL=egl .venv-sim/bin/python scripts/eval_policy.py <session> <checkpoint> \
    --as-trained <sim_episodes_index.json> --per-position 8
```

`--as-trained` is not optional in spirit: it reads back the workspace frame and box position the
checkpoint was *trained* under. Evaluated in the wrong frame a policy sees a workspace rotated
away from its training data and scores ~0 for reasons that have nothing to do with the policy.

### Residual RL

```bash
MUJOCO_GL=egl .venv-eval/bin/python scripts/train_residual.py <session> <checkpoint> \
    --as-trained <index.json> --in-process --device cuda \
    --residual-dims pos --alpha 0.2 --updates-per-step 0.1 --steps 500000
```

Defaults encode what went wrong on the way here: the residual acts on **position only** (given
rotation authority, SAC learned a sustained tilt that stopped the gripper pointing down),
exploration starts quiet (`init_log_std=-2`), the reward is **potential-based** (the first version
made lifting cost 8x more per step than hovering, and SAC duly learned to hover), and
`--updates-per-step` defaults low because 1.0 put 19k gradient steps behind ~50 episodes and the
critic fit noise. `ego2libero/rl/sac.py` argues why SAC rather than TD3 or PPO.

### Printing the targets

```bash
uv run python tools/print_tags.py        # regenerates the A4 PDFs in tags/
```

Print at 100% scale, measure what actually came out, and put the measured numbers in
`measurements.yaml`. The configs refuse to run on nominal values.


P.s. I'm actually an undergrad and read the minimum requirement for holding a masters or a phd not a lot sooner than the few hours before the deadline on the 9th october, after I had spent the week already building along with making sure I don't flunk my midterms :). If possible I'd love to still be considered for the role though.