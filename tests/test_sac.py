import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ego2libero.rl.sac import SAC, Actor, ReplayBuffer, SACConfig

OBS, ACT = 27, 7


def test_actor_starts_as_the_base_policy():
    """Near-zero last layer: the residual must begin at ~0 so training starts at the baseline,
    not at a random perturbation that first has to climb back."""
    a = Actor(OBS, ACT, SACConfig())
    o = torch.randn(64, OBS)
    mean, _ = a.sample(o, deterministic=True)
    assert mean.abs().max() < 0.02, "a fresh residual should be a no-op"
    loose = Actor(OBS, ACT, SACConfig(init_last_layer_scale=1.0))
    assert loose.net[-1].weight.abs().max() > 0.1  # the knob does something


def test_exploration_starts_quiet():
    """The near-zero last layer quiets only the mean; log_std shares that layer and would start at
    ~0 (std 1), measured as 0.56 cm of random tool motion per step. That destroyed the base
    policy's grasp: 2% of episodes lifted the cube, against ~45% for the base alone."""
    a = Actor(OBS, ACT, SACConfig())
    act, _ = a.sample(torch.randn(4096, OBS))
    assert act.abs().mean() < 0.2, "initial exploration should be a nudge, not a random walk"
    loud = Actor(OBS, ACT, SACConfig(init_log_std=0.0))
    assert loud.sample(torch.randn(4096, OBS))[0].abs().mean() > 0.4   # what the first run did
    assert SACConfig().target_entropy_scale < 1.0   # and SAC must not push the noise back up


def test_sampled_actions_are_bounded_and_log_prob_is_finite():
    a = Actor(OBS, ACT, SACConfig(init_last_layer_scale=1.0))
    act, logp = a.sample(torch.randn(256, OBS) * 10)
    assert act.shape == (256, ACT) and logp.shape == (256, 1)
    assert act.abs().max() <= 1.0  # tanh squashing keeps the residual in its box
    assert torch.isfinite(logp).all()


def test_replay_buffer_wraps_and_samples():
    b = ReplayBuffer(OBS, ACT, size=10)
    for i in range(13):
        b.add(np.full(OBS, i, np.float32), np.zeros(ACT, np.float32), float(i),
              np.full(OBS, i + 1, np.float32), i % 5 == 0)
    assert len(b) == 10 and b.full
    o, a, r, o2, d = b.sample(4, "cpu")
    assert o.shape == (4, OBS) and r.shape == (4, 1)
    np.testing.assert_allclose(o2.numpy(), o.numpy() + 1, atol=0)  # transitions stay paired


def test_update_runs_and_moves_the_temperature():
    torch.manual_seed(0)
    np.random.seed(0)
    sac = SAC(OBS, ACT, SACConfig(batch_size=32, buffer_size=1000))
    for _ in range(200):
        sac.buffer.add(np.random.randn(OBS).astype(np.float32), np.random.uniform(-1, 1, ACT).astype(np.float32),
                       float(np.random.randn()), np.random.randn(OBS).astype(np.float32), False)
    a0 = float(sac.alpha)
    stats = [sac.update() for _ in range(20)]
    assert all(np.isfinite(list(s.values())).all() for s in stats)
    assert sac.updates == 20 and float(sac.alpha) != a0  # temperature is being learned
    # target entropy follows the scale; it is deliberately below -ACT so SAC keeps the
    # residual quiet rather than maximising exploration against a competent base policy
    assert sac.target_entropy == -SACConfig().target_entropy_scale * ACT < 0

    sd = sac.state_dict()
    other = SAC(OBS, ACT, SACConfig(batch_size=32, buffer_size=1000))
    other.load_state_dict(sd)  # survives a Colab disconnect
    o = torch.randn(8, OBS)
    np.testing.assert_allclose(sac.actor.sample(o, True)[0].detach().numpy(),
                               other.actor.sample(o, True)[0].detach().numpy(), atol=0)


def test_target_network_tracks_but_lags():
    sac = SAC(OBS, ACT, SACConfig(batch_size=32, tau=0.005))
    for _ in range(100):
        sac.buffer.add(np.random.randn(OBS).astype(np.float32), np.zeros(ACT, np.float32), 1.0,
                       np.random.randn(OBS).astype(np.float32), False)
    before = [p.clone() for p in sac.critic_target.parameters()]
    sac.update()
    after = list(sac.critic_target.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))          # it moved
    assert all((b - a).abs().max() < 0.1 for b, a in zip(before, after))      # but only slightly
