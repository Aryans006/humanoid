"""plots.png per demo: positions, rotations and phase signals vs time, with phase boundaries
and the box-pose source shaded."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ego2libero import se3  # noqa: E402
from ego2libero.fusion.bridge import SRC_INTERPOLATED, SRC_PROPAGATED  # noqa: E402
from ego2libero.fusion.phases import PHASE_NAMES  # noqa: E402


def _phases(ax, t, seg):
    for name, i in seg.onsets.items():
        ax.axvline(t[i], color="k", lw=0.8, ls="--")
        ax.text(t[i], ax.get_ylim()[1], f" {name}", va="top", fontsize=7)


def _shade(ax, t, mask, color, label):
    if mask.any():
        ax.fill_between(t, 0, 1, where=mask, transform=ax.get_xaxis_transform(), color=color, alpha=0.15, label=label)


def plot_demo(path, t, fused, T_W_O_tags, T_W_H_raw, cfg_phases, title="") -> Path:
    fig, axs = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    src = fused.obj_source
    # 1 positions
    ax = axs[0]
    for k, (c, n) in enumerate(zip("rgb", "xyz")):
        ax.plot(t, 100 * T_W_O_tags[:, k, 3], ".", ms=1.5, color=c, alpha=0.3)
        ax.plot(t, 100 * fused.T_W_O[:, k, 3], color=c, lw=1.5, label=f"box {n}")
        ax.plot(t, 100 * fused.T_W_H[:, k, 3], color=c, lw=1, ls=":", label=f"wrist {n}")
    ax.set_ylabel("position in W (cm)")
    # 2 rotations relative to the first frame
    ax = axs[1]
    for T, lab, ls in ((fused.T_W_O, "box", "-"), (fused.T_W_H, "wrist", ":")):
        ok = np.isfinite(T).all(axis=(1, 2))
        ang = np.full(len(t), np.nan)
        ref = T[np.flatnonzero(ok)[0]]
        ang[ok] = np.degrees(se3.rotation_distance(ref, T[ok]))
        ax.plot(t, ang, ls=ls, label=f"{lab} rotation from start")
    ax.set_ylabel("deg")
    # 3 grasp signals
    ax = axs[2]
    ax.plot(t, 100 * fused.aperture, label="aperture (cm)")
    ax.plot(t, 100 * fused.contact, label="fingertip-box surface (cm)")
    ax.axhline(100 * cfg_phases["d_contact_m"], color="C1", lw=0.6, ls=":")
    ax.set_ylim(-3, 20)
    ax.set_ylabel("cm")
    # 4 box speed
    ax = axs[3]
    ax.plot(t, 100 * fused.box_speed, color="C2", label="box speed (cm/s)")
    for k, c in (("v_move_mps", "C3"), ("v_rest_mps", "C4")):
        ax.axhline(100 * cfg_phases[k], color=c, lw=0.6, ls=":", label=k)
    ax.set_ylim(0, 25)
    ax.set_xlabel("t (s)")
    for ax in axs:
        _shade(ax, t, src == SRC_PROPAGATED, "orange", "box propagated")
        _shade(ax, t, src == SRC_INTERPOLATED, "gold", "box interpolated")
        _phases(ax, t, fused.seg)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=4, loc="upper left")
    axs[0].set_title(title + "  phases: " + ", ".join(PHASE_NAMES))
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return Path(path)
