import numpy as np

from ego2libero.sim.execute import GraspSchedule

GRIP = np.where((np.arange(60) >= 20) & (np.arange(60) < 50), 1.0, -1.0)
POS = np.zeros((60, 3))  # every target at the origin, so tool height == tracking error


def _drive(sched, height):
    """Run the schedule to exhaustion; ``height(step)`` is the tool position it sees."""
    out, i = [], 0
    while (nxt := sched.next([0.0, 0.0, height(i)], [0.0, 0.0, -0.5])) is not None:
        out.append(nxt)
        i += 1
        assert i < 10_000
    return out


def test_waits_for_the_tool_to_arrive_then_closes_and_holds():
    sched = GraspSchedule(GRIP, POS, fps=20, reach_m=0.01, reach_timeout_s=1.5, close_hold_s=0.5)
    steps = _drive(sched, lambda i: 0.05 - 0.007 * (i - 19) if i < 25 else 0.005)  # descends, arrives at 25

    ks, gs = np.array([k for k, _ in steps]), np.array([g for _, g in steps])
    assert ks[:20].tolist() == list(range(20)) and (gs[:20] == -1).all()  # demo replayed as-is
    dwell = np.flatnonzero(ks == 20)
    # 5 steps of descending with the fingers open, 0.5 s of closing, then frame 20 itself
    assert len(dwell) == 5 + 10 + 1 and sched.extra_steps == 15
    assert (gs[dwell[:5]] == -1).all() and (gs[dwell[5:]] == 1).all()
    assert ks[dwell[-1] + 1] == 21 and gs[dwell[-1] + 1] == 1  # demo resumes, still closed
    assert ks[-1] == 59 and (np.diff(ks[dwell[-1]:]) == 1).all()  # and never repeats a frame again
    g, = sched.grasps
    assert (g["frame"], g["reach_steps"], g["reason"]) == (20, 5, "reached")
    assert g["tool_target_err_m"] == 0.005 and g["tool_cube_dist_m"] == 0.505


def test_reach_times_out_when_the_arm_never_settles_and_release_is_not_delayed():
    sched = GraspSchedule(GRIP, POS, fps=20, reach_timeout_s=1.5, close_hold_s=0.5)
    steps = _drive(sched, lambda i: 0.05 + 0.005 * i)  # drifting away: neither close nor still

    ks = np.array([k for k, _ in steps])
    assert (ks == 20).sum() == 30 + 10 + 1  # the full 1.5 s timeout, then the close hold
    assert sched.grasps[0]["reason"] == "timeout" and sched.grasps[0]["reach_steps"] == 30
    assert len(sched.grasps) == 1  # only the close is delayed, the release runs on the demo's clock
    assert (np.diff(ks[ks > 20]) == 1).all()


def test_a_stalled_arm_stops_waiting_even_though_it_never_reaches_the_target():
    """OSC settles 1-2 cm short of the commanded tool point, so arrival is "stopped moving"."""
    sched = GraspSchedule(GRIP, POS, fps=20, reach_m=0.01, settle_m=0.002, settle_s=0.3,
                          reach_timeout_s=1.5)
    steps = _drive(sched, lambda i: max(0.019, 0.05 - 0.004 * (i - 19)))  # stalls at 1.9 cm

    ks = np.array([k for k, _ in steps])
    g, = sched.grasps
    assert g["reason"] == "settled" and g["reach_steps"] < 30  # well before the 1.5 s timeout
    assert abs(g["tool_target_err_m"] - 0.019) < 1e-9
    assert (ks == 20).sum() == g["reach_steps"] + 10 + 1


def test_k0_k1_window_and_disabled_passthrough():
    sched = GraspSchedule(GRIP, POS, k0=18, k1=24, fps=20)
    # already at the cube, so no descending steps: just the close hold, then frames 20..23
    assert [k for k, _ in _drive(sched, lambda i: 0.005)] == [18, 19] + [20] * 11 + [21, 22, 23]

    off = GraspSchedule(GRIP, POS, fps=20, enabled=False)
    steps = _drive(off, lambda i: 0.5)
    assert [k for k, _ in steps] == list(range(60)) and not off.grasps
    assert [g for _, g in steps] == GRIP.tolist()
