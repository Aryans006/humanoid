# ego2libero

**Phone video of a human doing a pick-and-place → LIBERO-format robot demonstrations → a trained VLA.**

Hi team at Humanoid!

I'm Aryan, currently a founding engineer at a small robot learning startup here in Delhi.

After my recent furlough into dealing with human egocentric data, the problem proposed stands pretty clear, and breakable into sub-problems.

## The problem

With egocentric data, and training robot policies from it, the paths are clear.

If the policy is to be deployed in the real world and egocentric data is the only source, we need express photorealism, since VLA policies tend to flunk performance as soon as you go from sim to real — or even just real to real.

So to make egocentric robot policies perform in the real world using the standard VLA training process (collect data → convert it into a suitable format, LeRobot being widely used → train the model), we have two options:

- either find a way to make the egocentric data resemble robot data, image-wise; or
- freeze the vision tower of the VLA, train the dedicated action experts (flow matching in π0.5, DiT in GR00T N1.7), then use a small subset of real robot data to train on this checkpoint. This should ideally reduce the number of episodes we need for getting good performance for a fresh robot embodiment, while minimizing the requirement for expensive-to-collect real-world data.

I took the second path here: the vision tower is frozen and only the action expert is trained.

## The steps

For training robot policies using egocentric data, the steps are clear:

1. extract the actual human trajectory in the video
2. smooth that human trajectory to optimize it for our robot
3. track the 6-DoF position of the object being manipulated
4. render everything into robot data in a simulation

Following this checklist, I've implemented the following approach.

### 1. Extracting the human trajectory — HaWoR

I used [HaWoR](https://arxiv.org/pdf/2501.02973). HaWoR reconstructs the hand in a **world** frame rather than the camera frame, recovering MANO pose plus a camera trajectory from its own SLAM. That is the output a retargeter needs: a gripper pose has to be expressed in the table frame, not relative to a moving camera. Methods that stop at camera-frame hand pose leave that transform for you to solve separately.

### 2. Smoothing

In `ego2libero/fusion/smooth.py`, applied in `fusion/fuse.py`. Three stages:

- **Outlier rejection (Hampel).** A frame is dropped if its position deviates from the local 7-frame median by more than 3 robust sigmas (1.4826 × MAD) *and* by at least 3 mm; the same test on orientation against the local chordal-mean rotation, with a 1.5° floor. The two floors matter: without them the test fires on sub-millimetre noise during the static stretches. Dropped frames are refilled by decoupled SE(3) interpolation (slerp on rotation, linear on translation).

The hand and the cube are smoothed separately, and the hand is smoothed *before* phases are estimated, since phase detection keys on a clean wrist. Observed and HaWoR-infilled stretches are smoothed as separate runs, so the filter never blends a real measurement into an interpolated one.

### 3. Tracking the object — AprilTags

I opted for a very rudimentary AprilTag-based tracking for the cube. I considered an alternate pipeline: camera frame → SAM2 for object segmentation → Depth Anything v2 for finding the depth. This had 2 concerns:

- first, since I'm going to be rendering the object in sim, the AprilTag accuracy and determinism is nearly free — I don't need to work on making it look realistic or anything;
- I would still lack the orientation of the tracked object, which might make things finicky.

### 4. World frame

A similar AprilTag board was used as a world-frame anchor.


---

## Improving performance: residual RL

The retrained policy is good at some start positions and poor at others, and the failures are
lopsided. Rolling checkpoint 026000 out 40 times from 8 start poses at each of 5 cube positions:

| failure mode | n | median cube lift | median final distance to box |
| --- | --- | --- | --- |
| **never got the cube off the table** | **16 of 21** | 3.5 cm | 31.2 cm |
| lifted it but missed the box | 5 of 21 | 12.2 cm | 10.4 cm |

The cube rests with its centre at 2.5 cm, so a 3.5 cm "lift" means it barely budged. **Three
quarters of the failures are grasp failures**, and they miss by 1-3 cm — not by a lot, and not
randomly.

That shape of error is what residual RL is for. The base policy has the right idea and the wrong
offset, so rather than retrain it, freeze it and learn a small correction on top:

```
a = clip(a_base + alpha * a_res, -1, 1)        alpha = 0.2
```

At robosuite's OSC output limits that is at most **1 cm of position and 5.7° of rotation per step**
— the base policy stays in charge, but the residual accumulates ~10 cm of authority across a
ten-step approach, comfortably inside the envelope the failures live in.

Four design choices, each of which I got wrong first and fixed from a measurement:

- **Position only, not all 7 action dims.** Orientation is already correct by construction —
  retargeting fixes the gripper top-down with a yaw taken from the cube faces the person actually
  pinched — and the gripper open/close timing is the base policy's. Given authority over rotation,
  SAC learned a sustained +0.33 residual on rotation-y, which at alpha = 0.2 is ~1.9° of tilt per
  step compounding over an approach. The gripper stopped pointing down and lifted the cube in 2% of
  episodes, against the base policy's own ~45%. Restricting to position removes that by
  construction.
- **Exploration starts quiet.** SAC's log-std head shares the near-zero-initialised output layer,
  so it starts at std 1 and tanh covers most of [-1, 1] — measured as a 0.56 cm random tool
  displacement *resampled every step*, against a grasp needing 1-3 cm precision. Biasing the
  log-std head to -2 brings that to 0.11 cm.
- **Potential-based reward shaping.** My first reward was a plain distance penalty that switched
  from tool-to-cube to cube-to-box on the lift. That made progress *cost* money: hovering near the
  cube ran at 0.05/step, carrying at 0.4/step, so lifting raised the running cost eightfold against
  a one-off +5 bonus. SAC duly learned to hover — return climbed from -182 to -119 while the lift
  rate fell from 40% to 5%. Replaced with `r = γΦ(s') - Φ(s)` (Ng, Harada & Russell 1999), which
  telescopes over an episode and so cannot be farmed by standing still, and provably leaves the
  optimal policy unchanged.
- **A sane update-to-data ratio.** Budget here is counted in *environment steps*, and an RL episode
  runs to success or a 400-step cap, so 100k steps buys only ~280 attempts at the task (measured:
  245 episodes over 88,609 steps). At 1 gradient update per environment step, step 20k had ~50
  episodes of experience behind **19,000 gradient updates** — about 380 updates per episode. The
  critic memorised a tiny, highly correlated replay buffer and the actor exploited its errors:
  success fell from 60% to 15% (paired +1 −10, McNemar p = 0.012). Dropping to 0.1 updates/step
  took it back to parity.

### What it actually did

Parity, not improvement. Across runs in a correct environment the residual converged to roughly a
no-op (60% vs 60% baseline, paired +2 −2 at 80k steps), which is the right safe answer for an agent
that cannot find a real gain.

The open question is whether that is because **no gain exists** or because **the agent never had
enough experience to find one**. Note these are RL rollout episodes — the robot's own attempts at
the task, not the 51 recorded demonstrations that SmolVLA trains on. A 100k-step run is ~280 such
attempts, where SAC on continuous control normally wants thousands. Every piece of evidence points
at the sample budget: the only lever that changed the outcome was the update-to-data ratio, which
is a sample-size symptom, and throughput is capped at ~35 env-steps/s by the frozen policy's
forward pass.

One result worth recording for anyone attempting this on top of a large VLA: **residual RL here is
bottlenecked by the frozen policy's forward pass, not by physics.** MuJoCo costs 4.4 ms/step and
the base policy ~20 ms/step amortised, so a GPU only helps if it is not already busy — sharing a
GPU with a training job gave 37 env-steps/s, barely better than the 35 on CPU. That reframes the
cost model: the budget for residual RL on a VLA is set by inference throughput, and buying it means
either a dedicated accelerator or a distilled base.


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