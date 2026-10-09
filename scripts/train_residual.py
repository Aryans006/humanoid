"""Train the bounded residual on top of a frozen SmolVLA, with SAC.

  # this machine (split envs: sim in .venv-sim, policy behind a pipe)
  MUJOCO_GL=egl .venv-sim/bin/python scripts/train_residual.py <session> <checkpoint> --steps 100000

  # one interpreter, GPU (Colab): everything on numpy 2, policy in-process
  MUJOCO_GL=egl python scripts/train_residual.py <session> <checkpoint> --in-process --device cuda

Why SAC rather than TD3 or PPO is argued in ego2libero/rl/sac.py; the short version is that env
steps cost ~1 s of frozen-base forward per 50, so only an off-policy method is affordable.

Evaluation is **paired**: the base policy resamples flow-matching noise every call (per-step action
std 0.03-0.08, against the residual's 0.2 authority), so an unpaired residual-vs-baseline
comparison needs roughly 4x the rollouts to see the same effect. Both arms are run from the same
start poses with the same base seeds, and the report is the per-start difference.

Checkpoints (actor, critics, temperature, optimiser state) are written every --save-every steps so
a Colab disconnect costs minutes, not the run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ego2libero.export.write_demo import load_demo  # noqa: E402
from ego2libero.sim.frame import load_T_L_W  # noqa: E402
from ego2libero.sim.residual import DIM_SETS, OBS_DIM, ResidualConfig, ResidualCubeToBox  # noqa: E402

TASK = "pick up the red cube and place it in the box"


def load_starts(session: str, T_L_W: np.ndarray, per_position: int, data_root: Path = None):
    """One cube start pose per demo, grouped by the session's labelled positions."""
    proc = Path(data_root or ROOT) / "data/processed" / session
    rows = [r for r in csv.DictReader(open(proc / "selection.csv")) if r["position"]]
    starts, labels = [], []
    for p in sorted({r["position"] for r in rows}):
        g = sorted((r for r in rows if r["position"] == p), key=lambda r: -float(r["score"]))
        for r in g[:per_position]:
            f = proc / f"{r['clip']}_20hz.npz"
            if f.exists():
                starts.append(T_L_W @ load_demo(proc / f"{r['clip']}_20hz")["T_W_O"][0])
                labels.append(p)
    return starts, labels


def eval_subset(labels, per_position: int):
    """Indices of at most ``per_position`` starts from each position, so a mid-training eval costs
    minutes rather than the ~12 min a full 40-start sweep takes on CPU."""
    picked, seen = [], {}
    for i, p in enumerate(labels):
        if seen.get(p, 0) < per_position:
            picked.append(i)
            seen[p] = seen.get(p, 0) + 1
    return picked


def evaluate(env, agent, labels, episodes_per_start=1, base_seed=10_000, idxs=None):
    """Paired: for each start, run the residual and the zero-residual baseline under the same base
    noise, so the difference is not swamped by the base policy's own stochasticity."""
    out = {"residual": [], "baseline": []}
    for i in (range(len(env.starts)) if idxs is None else idxs):
        for rep in range(episodes_per_start):
            seed = base_seed + 1000 * rep + i
            for arm in ("residual", "baseline"):
                if hasattr(env.base, "seed"):
                    env.base.seed = seed  # same flow-matching noise in both arms
                o = env.reset(i)
                done = False
                while not done:
                    a = np.zeros(env.act_dim) if arm == "baseline" else agent.act(o, deterministic=True)
                    o, _, done, info = env.step(a)
                out[arm].append({"success": bool(info["success"]), "position": labels[i],
                                 "lifted": bool(info["lifted"]), "start": i, "seed": seed})
    res = {}
    for arm, rolls in out.items():
        res[arm] = {"success_rate": float(np.mean([r["success"] for r in rolls])),
                    "n": len(rolls),
                    "by_position": {p: float(np.mean([r["success"] for r in rolls if r["position"] == p]))
                                    for p in sorted({r["position"] for r in rolls})}}
    paired = [(r["success"], b["success"]) for r, b in zip(out["residual"], out["baseline"])]
    res["paired"] = {"residual_only_wins": sum(1 for r, b in paired if r and not b),
                     "baseline_only_wins": sum(1 for r, b in paired if b and not r),
                     "both": sum(1 for r, b in paired if r and b), "neither": sum(1 for r, b in paired if not r and not b)}
    res["rollouts"] = out
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("checkpoint", help="the frozen SmolVLA to wrap")
    ap.add_argument("--steps", type=int, default=100_000, help="environment steps")
    ap.add_argument("--alpha", type=float, default=0.2, help="residual authority")
    ap.add_argument("--residual-dims", choices=sorted(DIM_SETS), default="pos",
                    help="which action channels the residual may touch (default: position only)")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--per-position", type=int, default=8, help="distinct cube starts per position")
    ap.add_argument("--as-trained", help="sim_episodes index.json giving the T_L_W and box the base trained in")
    ap.add_argument("--tlw")
    ap.add_argument("--box-xy", nargs=2, type=float)
    ap.add_argument("--in-process", action="store_true", help="run the base policy here (needs numpy 2)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--eval-every", type=int, default=10_000)
    ap.add_argument("--eval-starts", type=int, default=1, help="episodes per start at evaluation")
    ap.add_argument("--eval-per-position", type=int, default=2,
                    help="starts per position used at mid-training evals (the final eval uses all)")
    ap.add_argument("--save-every", type=int, default=5_000)
    ap.add_argument("--start-steps", type=int, default=2_000)
    ap.add_argument("--updates-per-step", type=float, default=1.0,
                    help="gradient updates per env step; fractional is honoured. Keep the "
                         "update-to-data ratio sane: 1.0 over ~50 episodes is ~380 updates per "
                         "episode and the critic overfits a tiny correlated buffer.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--data-root", default=None,
                    help="holds data/processed/<session>/; defaults to the repo root. Set this when "
                         "the code lives somewhere else, as it does in the Colab bundle.")
    ap.add_argument("--resume", help="a run dir to continue from")
    a = ap.parse_args()

    import torch

    from ego2libero.rl.sac import SAC, SACConfig

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    T_L_W = load_T_L_W(a.session, a.tlw)
    box_xy = np.array(a.box_xy, float) if a.box_xy else None
    if a.as_trained:
        idx = json.loads(Path(a.as_trained).read_text())
        T_L_W = np.array(idx["T_L_W"], float) if "T_L_W" in idx else (T_L_W if a.tlw else load_T_L_W())
        box_xy = np.array(idx["box_xy_L"], float)
    if box_xy is None:
        raise SystemExit("need --box-xy or --as-trained: the scene must match the base policy's training")

    data_root = Path(a.data_root) if a.data_root else ROOT
    starts, labels = load_starts(a.session, T_L_W, a.per_position, data_root)
    if not starts:
        raise SystemExit(f"no start poses found under {data_root}/data/processed/{a.session} "
                         f"(need selection.csv and demo_*_20hz.npz); pass --data-root")

    # Printed before the base policy and the simulator are built, and flushed, so that if either
    # dies the output says how far we got. MuJoCo aborts the process at the C level when EGL has no
    # device (a CPU-only machine with MUJOCO_GL=egl), which leaves no Python traceback at all.
    gl = os.environ.get("MUJOCO_GL", "<unset>")
    print(f"start poses : {len(starts)} across {len(set(labels))} positions", flush=True)
    print(f"device      : {a.device}   MUJOCO_GL={gl}", flush=True)
    print(f"T_L_W       : translation {np.round(T_L_W[:3, 3], 4).tolist()}", flush=True)
    print(f"box_xy_L    : {np.round(box_xy, 4).tolist()}", flush=True)
    if a.device.startswith("cuda"):
        import torch as _t
        if not _t.cuda.is_available():
            raise SystemExit("--device cuda but torch reports no CUDA. On Colab: "
                             "Runtime -> Change runtime type -> GPU.")
    elif gl == "egl":
        print("WARNING: MUJOCO_GL=egl on a non-CUDA device. EGL needs a GPU; if this process dies "
              "with no traceback, that is why. Use MUJOCO_GL=osmesa for software rendering "
              "(apt-get install -y libosmesa6-dev).", flush=True)
    print("building base policy ...", flush=True)
    if a.in_process:
        from ego2libero.sim.base_policy import InProcessBasePolicy
        base = InProcessBasePolicy(a.checkpoint, device=a.device)
    else:
        from ego2libero.sim.base_policy import BasePolicy
        base = BasePolicy(a.checkpoint)

    print("building simulator ...", flush=True)
    env = ResidualCubeToBox(base, starts, box_xy,
                            ResidualConfig(dims=DIM_SETS[a.residual_dims], alpha=a.alpha,
                                           max_steps=a.max_steps),
                            task=TASK, seed=a.seed)
    ACT_DIM = env.act_dim
    cfg = SACConfig(start_steps=a.start_steps, updates_per_step=a.updates_per_step)
    agent = SAC(OBS_DIM, ACT_DIM, cfg, device=a.device)

    out = Path(a.out) if a.out else data_root / "data/residual" / f"{a.session}_{Path(a.checkpoint).name}"
    out.mkdir(parents=True, exist_ok=True)
    step0 = 0
    if a.resume:
        sd = torch.load(Path(a.resume) / "agent.pt", map_location=a.device, weights_only=False)
        agent.load_state_dict(sd["agent"])
        step0 = sd["step"]
        print(f"resumed from {a.resume} at step {step0}")

    print(f"training: residual on {a.residual_dims} ({ACT_DIM} dims), alpha={a.alpha}, {a.steps} steps, "
          f"{'in-process' if a.in_process else 'piped'} base\n", flush=True)
    log_p, hist = out / "train_log.jsonl", []
    o = env.reset()
    ep_ret, ep_len, ep_n, t0 = 0.0, 0, 0, time.time()
    update_credit = 0.0
    for step in range(step0, a.steps):
        act = np.random.uniform(-1, 1, ACT_DIM) if step < cfg.start_steps and not a.resume else agent.act(o)
        o2, r, done, info = env.step(act)
        agent.buffer.add(o, act, r, o2, done and info["steps"] < a.max_steps)  # time-limit is not terminal
        o, ep_ret, ep_len = o2, ep_ret + r, ep_len + 1
        if done:
            ep_n += 1
            rec = {"step": step, "episode": ep_n, "return": round(ep_ret, 2), "len": ep_len,
                   "success": bool(info["success"]), "lifted": bool(info["lifted"])}
            hist.append(rec)
            with open(log_p, "a") as f:
                f.write(json.dumps(rec) + "\n")
            if ep_n % 10 == 0:
                last = hist[-20:]
                sps = (step - step0 + 1) / (time.time() - t0)
                print(f"  step {step:7d}  ep {ep_n:4d}  return {np.mean([h['return'] for h in last]):8.1f}  "
                      f"success {np.mean([h['success'] for h in last]):4.0%}  lifted "
                      f"{np.mean([h['lifted'] for h in last]):4.0%}  {sps:4.1f} steps/s", flush=True)
            o, ep_ret, ep_len = env.reset(), 0.0, 0
        if len(agent.buffer) >= cfg.batch_size and step >= cfg.start_steps:
            # Fractional rates must work: at 1.0 update/step this run put 19,000 gradient steps
            # behind ~50 episodes (380:1) and the critic fit noise, so the actor degraded the base
            # from 60% to 15%. int() truncation would have silently floored any value < 1 to one
            # update per step, i.e. no change at all.
            update_credit += cfg.updates_per_step
            while update_credit >= 1.0:
                agent.update()
                update_credit -= 1.0
        if a.save_every and (step + 1) % a.save_every == 0:
            torch.save({"agent": agent.state_dict(), "step": step + 1, "args": vars(a)}, out / "agent.pt")
        if a.eval_every and (step + 1) % a.eval_every == 0:
            ev = evaluate(env, agent, labels, a.eval_starts, idxs=eval_subset(labels, a.eval_per_position))
            ev["step"] = step + 1
            (out / f"eval_{step + 1}.json").write_text(json.dumps(ev, indent=2))
            p = ev["paired"]
            print(f"  EVAL step {step + 1}: residual {ev['residual']['success_rate']:.0%} vs "
                  f"baseline {ev['baseline']['success_rate']:.0%} (n={ev['residual']['n']}) | "
                  f"paired +{p['residual_only_wins']} -{p['baseline_only_wins']}", flush=True)
            o, ep_ret, ep_len = env.reset(), 0.0, 0

    torch.save({"agent": agent.state_dict(), "step": a.steps, "args": vars(a)}, out / "agent.pt")
    ev = evaluate(env, agent, labels, max(2, a.eval_starts))
    (out / "eval_final.json").write_text(json.dumps(ev, indent=2))
    p = ev["paired"]
    print(f"\nFINAL residual {ev['residual']['success_rate']:.0%} vs baseline "
          f"{ev['baseline']['success_rate']:.0%} (n={ev['residual']['n']})")
    print(f"  paired: residual-only {p['residual_only_wins']}, baseline-only {p['baseline_only_wins']}, "
          f"both {p['both']}, neither {p['neither']}")
    print(f"  -> {out}")
    env.close()
    base.close()


if __name__ == "__main__":
    main()
