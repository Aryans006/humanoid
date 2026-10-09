"""Per-demo output files (plan stage S9) and the loader the retargeting stage uses.

``<demo>.npz`` holds arrays in W at native timestamps, ``<demo>.json`` the metadata. Keys follow
plan.md's "Output data format". Until M3 lands (phases, occlusion bridging, smoothing) the
export is marked ``"stage": "M2-preliminary"`` and lacks ``phase`` / ``gripper_closed``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SRC_TAG, SRC_PROPAGATED, SRC_INTERPOLATED, SRC_MISSING = 0, 1, 2, 3


def write_demo(path: str | Path, arrays: dict, meta: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=_jsonable))
    return path.with_suffix(".npz")


def load_demo(path: str | Path) -> dict:
    """All arrays of a demo plus its metadata under ``"meta"``."""
    path = Path(path)
    d = dict(np.load(path.with_suffix(".npz")))
    js = path.with_suffix(".json")
    d["meta"] = json.loads(js.read_text()) if js.exists() else {}
    return d


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    raise TypeError(type(o))
