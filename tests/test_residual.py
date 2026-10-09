import numpy as np
import pytest

from ego2libero.sim.residual import DIM_SETS, OBS_DIM, ResidualConfig, observation, potential, reward

TABLE_Z, BOX = 0.8, np.array([0.1, 0.0])


def _obs(tool, cube, a_base=np.zeros(7), lifted=False):
    return observation(tool, [0, 0, 0], [0.04, 0.04], cube, [0, 0, 0], BOX, a_base, lifted)


def test_observation_layout_and_differences():
    o = _obs([0.0, 0.0, 0.9], [0.1, 0.0, 0.825])
    assert o.shape == (OBS_DIM,) and o.dtype == np.float32
    np.testing.assert_allclose(o[0:3], [0, 0, 0.1], atol=1e-6)        # tool, table-relative
    np.testing.assert_allclose(o[8:11], [0.1, 0, 0.025], atol=1e-6)   # cube, table-relative
    np.testing.assert_allclose(o[14:17], [0.1, 0, -0.075], atol=1e-6)  # cube - tool
    np.testing.assert_allclose(o[17:19], [0.0, 0.0], atol=1e-6)       # cube - box, in xy
    assert o[-1] == 0.0


def test_potential_rises_monotonically_toward_the_goal():
    cfg = ResidualConfig()
    cube_far = [0.4, 0.0, 0.825]
    phi = lambda tool, cube, lifted: potential(cfg, tool, cube, BOX, lifted)
    far = phi([0.4, 0.0, 0.95], cube_far, False)
    near = phi([0.41, 0.0, 0.83], cube_far, False)
    held = phi([0.4, 0.0, 0.90], [0.4, 0.0, 0.90], True)
    placed = phi([0.1, 0.0, 0.90], [0.1, 0.0, 0.90], True)
    assert far < near < held < placed, (far, near, held, placed)


def test_shaping_cannot_be_farmed_by_standing_still():
    """The old reward paid 0.05/step to hover and 0.4/step to carry, so hovering won and SAC
    learned exactly that. A potential difference telescopes: standing still earns ~0 forever."""
    cfg = ResidualConfig()
    tool, cube = [0.2, 0.0, 0.84], [0.2, 0.0, 0.825]
    phi = potential(cfg, tool, cube, BOX, False)
    total = 0.0
    for _ in range(400):                       # 400 steps of doing nothing
        r, _ = reward(cfg, phi, phi, np.zeros(3), success=False, gamma=1.0)
        total += r
    assert abs(total) < 1e-9, "hovering must not accumulate reward"

    # whereas actually finishing the task pays the sparse bonus
    done, _ = reward(cfg, phi, phi, np.zeros(3), success=True, gamma=1.0)
    assert done == pytest.approx(cfg.w_success)


def test_lifting_is_rewarded_and_dropping_is_charged_back():
    cfg = ResidualConfig()
    tool, cube = [0.2, 0.0, 0.85], [0.2, 0.0, 0.85]
    low = potential(cfg, tool, cube, BOX, False)
    high = potential(cfg, tool, cube, BOX, True)
    assert high - low == pytest.approx(cfg.w_lift + cfg.w_reach * 0.0, abs=1e-9)
    up, _ = reward(cfg, low, high, np.zeros(3), False, gamma=1.0)
    down, _ = reward(cfg, high, low, np.zeros(3), False, gamma=1.0)
    assert up == pytest.approx(cfg.w_lift) and down == pytest.approx(-cfg.w_lift)


def test_effort_is_penalised():
    cfg = ResidualConfig()
    lazy, _ = reward(cfg, 0.0, 0.0, np.zeros(3), False, gamma=1.0)
    busy, _ = reward(cfg, 0.0, 0.0, np.ones(3), False, gamma=1.0)
    assert lazy - busy == pytest.approx(cfg.w_effort * 3)


def test_residual_authority_is_bounded_by_alpha():
    """a = clip(a_base + alpha * a_res): at most 1 cm and 5.7 deg per step at alpha = 0.2."""
    cfg = ResidualConfig(alpha=0.2)
    a_base = np.zeros(7)
    for a_res in (np.ones(7), -np.ones(7), np.zeros(7)):
        a = np.clip(a_base + cfg.alpha * np.clip(a_res, -1, 1), -1, 1)
        assert np.abs(a).max() <= cfg.alpha + 1e-12
    assert np.isclose(cfg.alpha * 0.05, 0.01)       # POS_MAX = 5 cm  -> 1 cm
    assert np.isclose(np.degrees(cfg.alpha * 0.5), 5.729577, atol=1e-5)  # ROT_MAX = 0.5 rad
    # a saturated base action cannot be pushed past the actuator limit
    assert np.clip(np.ones(7) + cfg.alpha * np.ones(7), -1, 1).max() == 1.0
    with pytest.raises(ValueError):
        ResidualConfig(alpha=0.0)


def test_residual_touches_only_its_own_channels():
    """Position-only by default: rotation and gripper must come through as the base commanded them.
    Letting RL perturb rotation produced a sustained +0.33 tilt residual that stopped the gripper
    pointing down; orientation is already fixed top-down by retargeting, so it is not RL's to set."""
    cfg = ResidualConfig()
    assert cfg.dims == (0, 1, 2)
    a_base = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0])
    full = np.zeros(7)
    full[list(cfg.dims)] = np.ones(len(cfg.dims))
    a = np.clip(a_base + cfg.alpha * full, -1, 1)
    np.testing.assert_allclose(a[3:], a_base[3:], atol=0)     # rotation + gripper untouched
    np.testing.assert_allclose(a[:3], a_base[:3] + 0.2, atol=1e-12)

    assert DIM_SETS["all"] == tuple(range(7)) and len(DIM_SETS["pose"]) == 6
    wide = ResidualConfig(dims=DIM_SETS["all"])
    full = np.zeros(7); full[list(wide.dims)] = np.ones(7)
    assert not np.allclose(np.clip(a_base + wide.alpha * full, -1, 1)[3:], a_base[3:])
