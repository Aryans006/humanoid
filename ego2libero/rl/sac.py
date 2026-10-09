"""Soft Actor-Critic for the bounded residual (see ego2libero/sim/residual.py).

Why SAC here, rather than TD3 or PPO
------------------------------------
The binding constraint is that **environment steps are expensive**: the frozen SmolVLA base must
run a forward pass every 50 steps, which costs ~1 s on CPU and dominates everything else. Measured
throughput is ~45 env-steps/s on this machine and a projected ~125-170 on a Colab GPU runtime.
That puts us in the low-sample regime, and it is what decides the algorithm.

* **vs PPO.** PPO is on-policy: each batch of experience is used for a handful of gradient steps
  and then discarded. It is the right choice when the simulator is massively parallel and a step
  is nearly free (thousands of envs in GPU physics), which is the opposite of our situation. For
  the same final performance PPO typically needs an order of magnitude more environment steps than
  an off-policy method; at 45 steps/s that is the difference between an hour and a day. SAC's
  replay buffer lets us take many gradient steps per environment step, moving the cost back onto
  the GPU where it is cheap.

* **vs TD3.** TD3 is the real alternative: also off-policy, also sample-efficient, and simpler.
  Two things favour SAC in this particular setup.

  1. *Exploration is already noisy, and by an amount we do not control.* The base policy is
     stochastic -- flow matching resamples noise each call, giving per-step action std of
     0.03-0.08, which is a sizeable fraction of the residual's own 0.2 authority. TD3 explores by
     adding a fixed-sigma Gaussian to a deterministic actor, so sigma has to be hand-tuned against
     that pre-existing noise floor: too small and the residual never finds the correction, too
     large and it destroys the base's competence. SAC learns a stochastic policy and tunes the
     entropy temperature automatically against a target entropy, which adapts instead of guessing.
  2. *Fewer knobs that interact with the residual bound.* TD3 adds exploration sigma, target-policy
     smoothing sigma, smoothing clip and policy delay, and the first three are all in action units
     -- the same units as the residual bound alpha, so they must be retuned whenever alpha changes.
     SAC's temperature is learned, so alpha stays the single authority knob.

  The honest caveat: on standard manipulation benchmarks SAC and TD3 land close together, and TD3
  may well work here too. SAC is chosen for robustness to the base's noise, not for a large
  expected margin. SAC's own risk is the opposite one -- entropy maximisation can keep the residual
  noisier than ideal; ``target_entropy_scale`` below is the lever if that shows up.

* **Why not plain behaviour-cloning more data.** The failures are not a coverage problem. The base
  misses the grasp by 1-3 cm on starts it was trained on; it needs closed-loop correction inside a
  chunk it is replaying open-loop, which is exactly what RL on a residual provides.

Two residual-specific choices
-----------------------------
* Exploration starts **quiet**, not merely centred. ``init_log_std = -2`` biases the log-std head
  so the initial residual moves the tool ~0.11 cm per step rather than 0.56 cm, and
  ``target_entropy_scale = 0.5`` stops SAC pushing it back up. Not a detail: with std 1 the
  residual lifted the cube in 2% of episodes against the base policy's own ~45%.

* The actor's last layer is initialised near zero. Note what this does and does not buy: the
  *deterministic* policy (the mean, which evaluation uses) starts at a no-op residual, so a fresh
  agent evaluates at the base policy's own 45% rather than at a random perturbation of it. It does
  **not** make exploration quiet -- the same near-zero init gives ``log_std ~ 0``, so sampled
  residuals cover the full tanh range from step one. That is intended: alpha already bounds the
  damage to 1 cm and 5.7 deg per step. If early training turns out to be too destructive, lower the
  initial log_std rather than alpha, so evaluation keeps its full authority.
* The policy is a tanh-squashed Gaussian, which matches the residual's bounded box exactly; the
  log-probability correction for the tanh is included (Haarnoja et al. 2018, appendix C).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


@dataclass
class SACConfig:
    hidden: tuple[int, ...] = (256, 256)
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005  # target network Polyak rate
    batch_size: int = 256
    buffer_size: int = 400_000
    start_steps: int = 2_000  # pure base policy (residual = 0) before learning starts
    updates_per_step: float = 1.0
    target_entropy_scale: float = 0.5  # target entropy = -scale * action_dim
    init_last_layer_scale: float = 1e-3  # near-zero residual MEAN at initialisation
    init_log_std: float = -2.0  # ... and a quiet one; see Actor.__init__


def _mlp(sizes, out_dim, last_scale=None):
    layers, prev = [], sizes[0]
    for h in sizes[1:]:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    last = nn.Linear(prev, out_dim)
    if last_scale is not None:
        nn.init.uniform_(last.weight, -last_scale, last_scale)
        nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Tanh-squashed Gaussian over the bounded residual."""

    def __init__(self, obs_dim, act_dim, cfg: SACConfig):
        super().__init__()
        self.net = _mlp((obs_dim, *cfg.hidden), 2 * act_dim, last_scale=cfg.init_last_layer_scale)
        self.act_dim = act_dim
        # The near-zero last layer quiets the *mean*, but log_std comes off the same layer and so
        # starts at ~0 -> std 1 -> tanh covers most of [-1, 1]. At alpha = 0.2 that is a 0.56 cm
        # random tool displacement resampled every step, against a grasp needing 1-3 cm precision.
        # Measured: the first run lifted the cube in 2% of episodes where the base alone lifts it
        # in ~45% -- the residual destroyed the base instead of correcting it. Start quiet.
        with torch.no_grad():
            self.net[-1].bias[act_dim:].fill_(cfg.init_log_std)

    def forward(self, obs):
        mu, log_std = self.net(obs).chunk(2, dim=-1)
        return mu, log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, obs, deterministic=False):
        mu, log_std = self(obs)
        if deterministic:
            return torch.tanh(mu), None
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        x = normal.rsample()
        a = torch.tanh(x)
        # tanh change of variables; the 1e-6 keeps log(0) out of the gradient
        logp = (normal.log_prob(x) - torch.log(1 - a.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return a, logp


class Critic(nn.Module):
    """Twin Q networks; the minimum is used, as in TD3 and SAC."""

    def __init__(self, obs_dim, act_dim, cfg: SACConfig):
        super().__init__()
        self.q1 = _mlp((obs_dim + act_dim, *cfg.hidden), 1)
        self.q2 = _mlp((obs_dim + act_dim, *cfg.hidden), 1)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs = np.zeros((size, obs_dim), np.float32)
        self.next_obs = np.zeros((size, obs_dim), np.float32)
        self.act = np.zeros((size, act_dim), np.float32)
        self.rew = np.zeros((size, 1), np.float32)
        self.done = np.zeros((size, 1), np.float32)
        self.size, self.ptr, self.full = size, 0, False

    def add(self, o, a, r, o2, d):
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i], self.next_obs[i], self.done[i] = o, a, r, o2, float(d)
        self.ptr = (i + 1) % self.size
        self.full |= self.ptr == 0

    def __len__(self):
        return self.size if self.full else self.ptr

    def sample(self, n, device):
        idx = np.random.randint(0, len(self), size=n)
        t = lambda x: torch.as_tensor(x[idx], device=device)
        return t(self.obs), t(self.act), t(self.rew), t(self.next_obs), t(self.done)


class SAC:
    def __init__(self, obs_dim, act_dim, cfg: SACConfig | None = None, device="cpu"):
        self.cfg = cfg or SACConfig()
        self.device, self.act_dim = device, act_dim
        self.actor = Actor(obs_dim, act_dim, self.cfg).to(device)
        self.critic = Critic(obs_dim, act_dim, self.cfg).to(device)
        self.critic_target = Critic(obs_dim, act_dim, self.cfg).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr)
        # automatic temperature: optimise log_alpha against a fixed target entropy
        self.target_entropy = -self.cfg.target_entropy_scale * act_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=self.cfg.lr)
        self.buffer = ReplayBuffer(obs_dim, act_dim, self.cfg.buffer_size)
        self.updates = 0

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            o = torch.as_tensor(np.asarray(obs, np.float32), device=self.device).unsqueeze(0)
            a, _ = self.actor.sample(o, deterministic)
        return a.squeeze(0).cpu().numpy()

    def update(self) -> dict:
        cfg = self.cfg
        o, a, r, o2, d = self.buffer.sample(cfg.batch_size, self.device)
        with torch.no_grad():
            a2, logp2 = self.actor.sample(o2)
            q1t, q2t = self.critic_target(o2, a2)
            target = r + cfg.gamma * (1 - d) * (torch.min(q1t, q2t) - self.alpha * logp2)
        q1, q2 = self.critic(o, a)
        loss_c = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.opt_c.zero_grad(set_to_none=True)
        loss_c.backward()
        self.opt_c.step()

        for p in self.critic.parameters():
            p.requires_grad_(False)
        a_pi, logp = self.actor.sample(o)
        q1pi, q2pi = self.critic(o, a_pi)
        loss_a = (self.alpha * logp - torch.min(q1pi, q2pi)).mean()
        self.opt_a.zero_grad(set_to_none=True)
        loss_a.backward()
        self.opt_a.step()
        for p in self.critic.parameters():
            p.requires_grad_(True)

        loss_alpha = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.opt_alpha.zero_grad(set_to_none=True)
        loss_alpha.backward()
        self.opt_alpha.step()

        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                pt.mul_(1 - cfg.tau).add_(cfg.tau * p)
        self.updates += 1
        return {"loss_critic": float(loss_c), "loss_actor": float(loss_a),
                "alpha": float(self.alpha), "entropy": float(-logp.mean())}

    def state_dict(self):
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(), "log_alpha": self.log_alpha.detach().cpu(),
                "opt_a": self.opt_a.state_dict(), "opt_c": self.opt_c.state_dict(),
                "opt_alpha": self.opt_alpha.state_dict(), "updates": self.updates}

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.critic_target.load_state_dict(sd["critic_target"])
        with torch.no_grad():
            self.log_alpha.copy_(sd["log_alpha"].to(self.device))
        self.opt_a.load_state_dict(sd["opt_a"])
        self.opt_c.load_state_dict(sd["opt_c"])
        self.opt_alpha.load_state_dict(sd["opt_alpha"])
        self.updates = sd.get("updates", 0)
