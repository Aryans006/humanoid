"""Rank a session's demos on tag-based quality and pick the best N, balanced over start positions.

  uv run python scripts/select_demos.py <session> [--n 50] [--positions 5]

Needs run_demo.py to have processed the clips (S0-S3). Writes
data/processed/<session>/selection.csv (every clip, features, verdict) and selection.json.

Hard rejects: a failed M1 gate, no clear lift, or more than one lift (regrasp). The rest are
clustered by the cube's start position (k-means) and ranked inside each cluster; the best
N / positions per cluster are selected, topped up from the best leftovers if a cluster is short.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego2libero.fusion.signals import robust_speed  # noqa: E402
from ego2libero.perception.board_pose import CameraTrack  # noqa: E402
from ego2libero.perception.box_pose import BoxTrack  # noqa: E402


def _runs(mask):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def features(clip_dir: Path) -> dict:
    t = np.load(clip_dir / "timestamps.npy")
    cam = CameraTrack.load(clip_dir / "camera_tags.npz")
    box = BoxTrack.load(clip_dir / "box_tags.npz")
    gates = json.loads((clip_dir / "gates.json").read_text())
    fps = 1 / np.median(np.diff(t))
    v = box.valid
    p = box.T_W_O[:, :3, 3]
    s_win, e_win = v & (t <= t[0] + 1), v & (t >= t[-1] - 1)
    start = np.median(p[s_win], 0) if s_win.any() else np.full(3, np.nan)
    end = np.median(p[e_win], 0) if e_win.any() else np.full(3, np.nan)
    z = p[:, 2]
    # Lift episodes: tag gaps are bridged by interpolating height, and the cube only counts as
    # back on the table within 1 cm of its start height (hysteresis). Thresholding valid frames
    # alone split single smooth lifts into "regrasps" at every brief tag gap.
    z_f = np.interp(t, t[v], z[v]) if v.sum() >= 2 else np.full(len(t), np.nan)
    off_table = z_f > start[2] + 0.01
    lifts = [(a, b) for a, b in _runs(off_table)
             if (b - a) / fps > 0.3 and np.nanmax(z_f[a:b]) > start[2] + 0.04]
    gaps = [(b - a) / fps for a, b in _runs(~v)]
    speed = robust_speed(p, t)
    failed = [g["gate"] for g in gates if g.get("limit") is not None and not g["pass"]]
    return {
        "clip": clip_dir.name, "duration_s": float(t[-1]),
        "board_coverage": float(cam.valid.mean()), "board_rms_px": float(np.nanmedian(cam.rms_px)),
        "box_coverage": float(v.mean()), "box_max_gap_s": float(max(gaps) if gaps else 0.0),
        "start_x": float(start[0]), "start_y": float(start[1]), "end_x": float(end[0]), "end_y": float(end[1]),
        "moved_m": float(np.linalg.norm(end[:2] - start[:2])), "max_lift_m": float(np.nanmax(z) - start[2]) if v.any() else 0.0,
        "n_lifts": len(lifts), "lift_s": float(sum(b - a for a, b in lifts) / fps),
        "peak_speed_mps": float(np.nanpercentile(speed, 99)) if np.isfinite(speed).any() else np.nan,
        "failed_gates": ";".join(failed),
    }


def kmeans(X, k, iters=100, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
        C2 = np.stack([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        if np.allclose(C2, C):
            break
        C = C2
    return lab, C


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--positions", type=int, default=5)
    a = ap.parse_args()
    base = ROOT / "data/interim" / a.session
    rows = [features(d) for d in sorted(base.glob("demo_*")) if (d / "gates.json").exists()]

    for r in rows:
        why = []
        if r["failed_gates"]:
            why.append("gate:" + r["failed_gates"])
        if r["n_lifts"] == 0 or r["max_lift_m"] < 0.04:
            why.append("no clear lift")
        if r["n_lifts"] > 1:
            why.append(f"{r['n_lifts']} lifts (regrasp)")
        if not np.isfinite(r["start_x"]) or not np.isfinite(r["end_x"]):
            why.append("cube not seen at start/end")
        r["reject"] = "; ".join(why)
        # higher is better: tag coverage dominates, then short gaps and clean board fits
        r["score"] = (2.0 * r["box_coverage"] - 0.5 * min(r["box_max_gap_s"], 2.0)
                      + 0.5 * r["board_coverage"] - 0.2 * max(r["board_rms_px"] - 1.0, 0.0))

    ok = [r for r in rows if not r["reject"]]
    X = np.array([[r["start_x"], r["start_y"]] for r in ok])
    lab, C = kmeans(X, a.positions)
    order = np.argsort(C[:, 0] + 10 * np.round(C[:, 1], 2))  # stable position names
    name = {int(j): f"P{i + 1}" for i, j in enumerate(order)}
    for r, l in zip(ok, lab):
        r["position"] = name[int(l)]
        r["dist_to_position_cm"] = float(100 * np.linalg.norm(X[ok.index(r)] - C[l]))
    per = a.n // a.positions
    chosen = []
    for p in sorted(set(name.values())):
        grp = sorted([r for r in ok if r["position"] == p], key=lambda r: -r["score"])
        chosen += grp[:per]
    rest = sorted([r for r in ok if r not in chosen], key=lambda r: -r["score"])
    chosen += rest[:max(0, a.n - len(chosen))]
    for r in rows:
        r["selected"] = r in chosen
        r.setdefault("position", "")

    out = ROOT / "data/processed" / a.session
    keys = ["clip", "selected", "position", "score", "reject", "duration_s", "box_coverage", "box_max_gap_s",
            "board_coverage", "board_rms_px", "start_x", "start_y", "end_x", "end_y", "moved_m", "max_lift_m",
            "n_lifts", "lift_s", "peak_speed_mps", "failed_gates", "dist_to_position_cm"]
    with open(out / "selection.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["position"], -r["score"])):
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in keys})
    summary = {"n_clips": len(rows), "n_rejected": len(rows) - len(ok), "n_selected": len(chosen),
               "positions": {name[int(j)]: {"centre_W_m": C[j].round(3).tolist(), "n_ok": int((lab == j).sum()),
                                            "n_selected": sum(1 for r in chosen if r["position"] == name[int(j)])}
                             for j in range(a.positions)},
               "selected": sorted(r["clip"] for r in chosen),
               "rejected": {r["clip"]: r["reject"] for r in rows if r["reject"]}}
    (out / "selection.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "selected"}, indent=2))


if __name__ == "__main__":
    main()
