"""Where the table frame W sits in the simulator world L: ``p_L = T_L_W @ p_W``.

The placement is per session, because it decides whether the Panda can actually reach the cube
in a top-down grasp. The 2026-10-03 pilots and the 2026-10-07 dataset occupy different patches
of the table in W, and the pilots' placement puts the dataset's cubes where a straight-down
grasp pins joint 6 against its limit (3.5-5 cm of tracking error). See configs/libero_frame.yaml.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ego2libero.config import load_yaml

TABLE_Z = 0.8  # robosuite tabletop height in L


def tabletop_T_L_W(centre_W, rot_deg: float, table_z: float = TABLE_Z) -> np.ndarray:
    """Place W on the robosuite tabletop: ``centre_W`` lands on the table centre (L x=y=0) and
    W's axes are turned by ``rot_deg`` about z. This is the family the T_L_W sweep searches."""
    a = np.radians(rot_deg)
    R = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:2, 3] = -R[:2, :2] @ np.asarray(centre_W, float)[:2]
    T[2, 3] = table_z
    return T


def load_T_L_W(session: str | None = None, override: str | Path | None = None) -> np.ndarray:
    """T_L_W for a session: an explicit json (``{"T_L_W": ...}``, as the sweep writes) wins, then
    the session's entry in configs/libero_frame.yaml, then the file's default."""
    if override:
        return np.array(json.loads(Path(override).read_text())["T_L_W"], float)
    cfg = load_yaml("libero_frame")
    entry = (cfg.get("sessions") or {}).get(session or "")
    return np.array((entry or cfg)["T_L_W"], float)
