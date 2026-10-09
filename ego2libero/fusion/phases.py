"""Phase segmentation (plan stage S6): approach, grasp, transport, release, retreat.

Anchored on the box's own motion (from tags, the most reliable signal), then refined with
the hand:

  transport  first sustained box motion (speed > v_move)
  grasp      last time before transport that the fingertips reach the box surface
             (contact < d_contact) and stay; aperture is only a weak cue for a pinch on a
             5 cm cube, so it is not required
  release    first time after transport that the fingertips leave the box (contact >
             d_contact) and the box then lands within release_fall_s, or the box comes to
             rest (speed < v_rest for release_rest_s), whichever comes first. The plan's rule
             (box at rest + aperture rising) places release after the box lands; the pilots
             drop the cube from 6-12 cm, hence the contact rule. The landing test is what
             keeps a HaWoR dropout mid-transport from "releasing" a box still in flight.
  retreat    after release, once the box has landed and the fingertips are clearly away
             (contact > 2 d_contact)

``gripper_closed`` is 1 from grasp onset to release onset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

APPROACH, GRASP, TRANSPORT, RELEASE, RETREAT = range(5)
PHASE_NAMES = ["approach", "grasp", "transport", "release", "retreat"]


@dataclass
class Segmentation:
    phase: np.ndarray  # (N,) uint8
    onsets: dict = field(default_factory=dict)  # phase name -> frame index
    notes: list = field(default_factory=list)
    landed: int | None = None  # first frame the released box is at rest again (not a phase)

    @property
    def gripper_closed(self) -> np.ndarray:
        g = np.zeros(len(self.phase), bool)
        if "grasp" in self.onsets and "release" in self.onsets:
            g[self.onsets["grasp"]:self.onsets["release"]] = True
        return g

    def onset_times(self, t: np.ndarray) -> dict:
        return {k: float(t[v]) for k, v in self.onsets.items()}


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _first_sustained(mask: np.ndarray, start: int, n_min: int) -> int | None:
    for a, b in _runs(mask[start:]):
        if b - a >= n_min:
            return start + a
    return None


def segment(t: np.ndarray, box_speed: np.ndarray, contact: np.ndarray, cfg: dict) -> Segmentation:
    """Phases from box speed (m/s) and fingertip contact distance (m), both NaN-tolerant."""
    n = len(t)
    fps = 1.0 / np.median(np.diff(t))
    v_move, v_rest = cfg["v_move_mps"], cfg["v_rest_mps"]
    d_contact = cfg["d_contact_m"]
    n_move = max(1, int(round(cfg.get("move_confirm_s", 0.2) * fps)))
    n_rest = max(1, int(round(cfg["release_rest_s"] * fps)))
    n_touch = max(1, int(round(cfg.get("contact_confirm_s", 0.1) * fps)))
    seg = Segmentation(np.full(n, APPROACH, np.uint8))

    moving = np.nan_to_num(box_speed, nan=0.0) > v_move
    i_tr = _first_sustained(moving, 0, n_move)
    if i_tr is None:
        seg.notes.append("box never moved: approach only")
        return seg

    # grasp: start of the contact run that is ongoing just before transport
    touching = np.nan_to_num(contact, nan=np.inf) < d_contact
    i_gr = None
    k = i_tr
    lookback = int(round(cfg.get("grasp_lookback_s", 3.0) * fps))
    while k > max(0, i_tr - lookback) and not touching[k]:
        k -= 1  # fingertips may read just off the surface at the very first motion frame
    if touching[k]:
        while k > 0 and touching[k - 1]:
            k -= 1
        i_gr = k
    else:
        seg.notes.append("no fingertip contact before transport: grasp onset = transport onset")
        i_gr = i_tr

    # release: fingertips leave the box, or the box comes to rest, whichever first
    away = np.nan_to_num(contact, nan=-np.inf) > d_contact
    resting = np.nan_to_num(box_speed, nan=np.inf) < v_rest
    i_rest = _first_sustained(resting, i_tr + n_move, n_rest)
    # Losing contact only counts as a release if the box really left the hand: free fall from
    # 18 cm takes 0.2 s, so a genuine release (a drop included) is followed by the box landing
    # almost at once. HaWoR loses the hand mid-transport in about a quarter of this dataset's
    # demos; without this test those dropouts released a box that was still 4-18 cm up and
    # seconds from being put down, which truncated the demo and sank most of the sim episodes.
    n_fall = max(1, int(round(cfg.get("release_fall_s", 0.5) * fps)))
    i_away, n_ignored = None, 0
    for a, _ in (ab for ab in _runs(away[i_tr + n_move:]) if ab[1] - ab[0] >= n_touch):
        cand = i_tr + n_move + a
        i_land = _first_sustained(resting, cand, n_rest)
        if i_land is not None and i_land - cand <= n_fall:
            i_away = cand
            break
        n_ignored += 1
    if n_ignored:
        seg.notes.append(f"{n_ignored} contact loss(es) ignored: the box was still in flight")
    cands = [i for i in (i_away, i_rest) if i is not None]
    if not cands:
        seg.notes.append("no release found: transport until the end")
        seg.phase[i_gr:i_tr] = GRASP
        seg.phase[i_tr:] = TRANSPORT
        seg.onsets = {"grasp": i_gr, "transport": i_tr}
        return seg
    i_rel = min(cands)
    seg.notes.append(f"release by {'contact' if i_rel == i_away else 'box at rest'}")

    # retreat: the box has landed (at rest) and the fingertips are clearly away. Contact alone
    # is not enough: a dropped box falls away from a still hand within a frame or two.
    i_land = _first_sustained(resting, i_rel, n_rest)
    clear = np.nan_to_num(contact, nan=-np.inf) > 2 * d_contact
    i_rt = None if i_land is None else _first_sustained(clear, i_land, n_touch)
    if i_land is not None:
        seg.landed = i_land
    i_rt = n if i_rt is None else i_rt

    seg.phase[i_gr:i_tr] = GRASP
    seg.phase[i_tr:i_rel] = TRANSPORT
    seg.phase[i_rel:i_rt] = RELEASE
    seg.phase[i_rt:] = RETREAT
    seg.onsets = {"grasp": i_gr, "transport": i_tr, "release": i_rel}
    if i_rt < n:
        seg.onsets["retreat"] = i_rt
    return seg
