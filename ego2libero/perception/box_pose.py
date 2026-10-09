"""Box 6D pose from AprilTags on its faces (plan stage S3), geometry core.

Box frame O: origin at the geometric center, axes along the edges, z up when the box
stands on its smallest face. Each face tag is centered and edge-aligned; its frame G
follows OpenCV's IPPE_SQUARE marker convention: x right, y up (toward the printed tag's
top edge), z out of the face. Corner order matches the detector: TL, TR, BR, BL.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from ego2libero import se3

log = logging.getLogger(__name__)

_AXES = {"+x": (0, 1.0), "-x": (0, -1.0), "+y": (1, 1.0), "-y": (1, -1.0), "+z": (2, 1.0), "-z": (2, -1.0)}


def _axis(name: str) -> tuple[np.ndarray, int]:
    i, sgn = _AXES[name]
    v = np.zeros(3)
    v[i] = sgn
    return v, i


def face_tag_transform(dims: np.ndarray, normal: str, up: str) -> np.ndarray:
    """T_O_G for a tag centered on the face with outward ``normal`` and its top edge toward ``up``."""
    z, i = _axis(normal)
    y, j = _axis(up)
    if i == j:
        raise ValueError(f"tag up axis {up} must differ from face normal {normal}")
    x = np.cross(y, z)
    return se3.make_T(np.stack([x, y, z], axis=1), z * dims[i] / 2.0)


def tag_corners(size: float) -> np.ndarray:
    """Tag corners in its own frame G, detector order TL, TR, BR, BL (IPPE_SQUARE order)."""
    h = size / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


@dataclass
class BoxModel:
    dims: np.ndarray  # (3,) edge lengths along x, y, z in meters
    tag_size: float  # black-square edge in meters
    T_O_G: dict[int, np.ndarray]  # tag id -> 4x4

    @classmethod
    def from_config(cls, cfg: dict) -> "BoxModel":
        from ego2libero.config import require

        require(cfg, "dims_m", "tag_size_m", where="box.yaml")
        dims = np.asarray(cfg["dims_m"], dtype=np.float64)
        T = {int(k): face_tag_transform(dims, v[0], v[1]) for k, v in cfg["faces"].items()}
        for k, M in (cfg.get("T_box_tag_overrides") or {}).items():
            T[int(k)] = np.asarray(M, dtype=np.float64)
        return cls(dims=dims, tag_size=float(cfg["tag_size_m"]), T_O_G=T)

    @property
    def ids(self) -> list[int]:
        return sorted(self.T_O_G)

    def object_points(self, ids) -> np.ndarray:
        """Stacked corners (4k, 3) in the box frame for the given tag ids."""
        c = tag_corners(self.tag_size)
        return np.concatenate([se3.transform_points(self.T_O_G[i], c) for i in ids])

    def box_corners(self) -> np.ndarray:
        """The 8 box vertices (8, 3) in O."""
        h = self.dims / 2.0
        s = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)
        return s * h


# --------------------------------------------------------------------------- PnP helpers


def rvec_tvec_to_T(rvec, tvec) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    return se3.make_T(R, np.asarray(tvec, dtype=np.float64).ravel())


def T_to_rvec_tvec(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    return rvec, T[:3, 3].reshape(3, 1).copy()


def project(T_C_X: np.ndarray, pts_X: np.ndarray, K: np.ndarray, dist=None) -> np.ndarray:
    rvec, tvec = T_to_rvec_tvec(T_C_X)
    uv, _ = cv2.projectPoints(np.asarray(pts_X, dtype=np.float64), rvec, tvec, K,
                              np.zeros(5) if dist is None else dist)
    return uv.reshape(-1, 2)


def reproj_rms(T_C_X, pts_X, uv, K, dist=None) -> float:
    d = project(T_C_X, pts_X, K, dist) - np.asarray(uv).reshape(-1, 2)
    return float(np.sqrt((d**2).sum(1).mean()))


# --------------------------------------------------------------------------- solve


@dataclass
class BoxPoseResult:
    T_C_O: np.ndarray
    rms_px: float
    ids: list[int]
    ambiguous: bool = False  # single-tag IPPE case
    pick_reason: str = ""  # which disambiguation rule decided
    candidates: list[np.ndarray] = field(default_factory=list)  # T_C_O candidates (single tag)
    candidate_rms: list[float] = field(default_factory=list)


def _resting_plausible(T_W_O: np.ndarray, model: BoxModel, z_tol: float, axis_tol_deg: float) -> bool:
    corners_W = se3.transform_points(T_W_O, model.box_corners())
    if abs(corners_W[:, 2].min()) > z_tol:
        return False
    # Some box axis must be (anti)parallel to W's z.
    cosines = np.abs(T_W_O[2, :3])  # z-components of the box axes in W
    return bool(np.degrees(np.arccos(np.clip(cosines.max(), -1, 1))) <= axis_tol_deg)


def disambiguate(
    cands_T_C_O: list[np.ndarray],
    cand_rms: list[float],
    model: BoxModel,
    T_W_C: np.ndarray | None = None,
    prior_T_W_O: np.ndarray | None = None,
    resting: bool = False,
    prior_margin_deg: float = 5.0,
    rest_z_tol: float = 0.01,
    rest_axis_tol_deg: float = 10.0,
) -> tuple[int, str]:
    """Pick among IPPE candidates: (1) nearest prior rotation, (2) resting plausibility,
    (3) lowest reprojection error. Rules 1 and 2 need T_W_C. Returns (index, reason)."""
    n = len(cands_T_C_O)
    if n == 1:
        return 0, "unique"
    if T_W_C is not None:
        cands_W = [T_W_C @ T for T in cands_T_C_O]
        if prior_T_W_O is not None:
            d = np.degrees([se3.rotation_distance(prior_T_W_O, T) for T in cands_W])
            order = np.argsort(d)
            if d[order[1]] - d[order[0]] > prior_margin_deg:
                return int(order[0]), f"prior ({d[order[0]]:.1f} vs {d[order[1]]:.1f} deg)"
        if resting:
            ok = [_resting_plausible(T, model, rest_z_tol, rest_axis_tol_deg) for T in cands_W]
            if sum(ok) == 1:
                return int(np.argmax(ok)), "resting"
    return int(np.argmin(cand_rms)), f"reproj ({min(cand_rms):.2f} px)"


def solve_box_pose(
    corners: dict[int, np.ndarray],
    K: np.ndarray,
    model: BoxModel,
    dist: np.ndarray | None = None,
    *,
    min_area_px2: float = 400.0,
    max_rms_px: float = 2.0,
    T_W_C: np.ndarray | None = None,
    prior_T_W_O: np.ndarray | None = None,
    resting: bool = False,
) -> BoxPoseResult | None:
    """Box pose in the camera from detected tag corners ``{id: (4, 2)}``.

    Two or more box tags: joint SQPnP + LM refine (no ambiguity). One tag: IPPE_SQUARE
    candidates, disambiguated with :func:`disambiguate`. Returns None if rejected.
    """
    dist = np.zeros(5) if dist is None else np.asarray(dist, dtype=np.float64)
    ids = [i for i in sorted(corners) if i in model.T_O_G
           and cv2.contourArea(np.asarray(corners[i], dtype=np.float32).reshape(4, 2)) >= min_area_px2]
    if not ids:
        return None
    uv = np.concatenate([np.asarray(corners[i], dtype=np.float64).reshape(4, 2) for i in ids])

    if len(ids) >= 2:
        obj = model.object_points(ids)
        ok, rvec, tvec = cv2.solvePnP(obj, uv, K, dist, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            return None
        rvec, tvec = cv2.solvePnPRefineLM(obj, uv, K, dist, rvec, tvec)
        T_C_O = rvec_tvec_to_T(rvec, tvec)
        res = BoxPoseResult(T_C_O, reproj_rms(T_C_O, obj, uv, K, dist), ids, pick_reason="joint")
    else:
        (i,) = ids
        obj_G = tag_corners(model.tag_size)
        n, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj_G, uv, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        T_G_O = se3.inv(model.T_O_G[i])
        cands = [rvec_tvec_to_T(r, t) @ T_G_O for r, t in zip(rvecs[:n], tvecs[:n])]
        obj_O = model.object_points([i])
        rms = [reproj_rms(T, obj_O, uv, K, dist) for T in cands]
        k, reason = disambiguate(cands, rms, model, T_W_C, prior_T_W_O, resting)
        log.debug("single-tag %d: picked %d by %s (rms %s)", i, k, reason, np.round(rms, 2))
        res = BoxPoseResult(cands[k], rms[k], ids, ambiguous=n > 1, pick_reason=reason,
                            candidates=cands, candidate_rms=rms)

    if res.rms_px > max_rms_px:
        return None
    return res


# --------------------------------------------------------------------------- whole clip


@dataclass
class BoxTrack:
    T_C_O: np.ndarray  # (N, 4, 4) from tags, NaN where no accepted solve
    T_W_O: np.ndarray  # (N, 4, 4) T_W_C @ T_C_O, NaN where either is missing
    rms_px: np.ndarray  # (N,)
    n_tags: np.ndarray  # (N,) box tags used
    ambiguous: np.ndarray  # (N,) bool, single-tag IPPE frames
    pick_reason: np.ndarray  # (N,) str

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.T_W_O).all(axis=(1, 2))

    def save(self, path) -> None:
        np.savez_compressed(path, T_C_O=self.T_C_O, T_W_O=self.T_W_O, rms_px=self.rms_px, n_tags=self.n_tags,
                            ambiguous=self.ambiguous, pick_reason=self.pick_reason.astype("U64"))

    @classmethod
    def load(cls, path) -> "BoxTrack":
        d = np.load(path)
        return cls(d["T_C_O"], d["T_W_O"], d["rms_px"], d["n_tags"], d["ambiguous"], d["pick_reason"])


def track_box(per_frame: list[dict], t: np.ndarray, K, T_W_C: np.ndarray, model: BoxModel,
              cfg: dict | None = None, rest_masks: list[np.ndarray] | None = None) -> BoxTrack:
    """Per-frame box pose from face tags, with single-tag ambiguity resolved over the whole clip.

    Every frame keeps all its pose candidates (two for a single tag: IPPE's mirror solutions;
    one when 2+ tags give a joint solve). A Viterbi pass then picks the candidate sequence with
    the least total rotation change, plus unary costs: reprojection error, penetrating the
    table, and (inside ``rest_masks``, e.g. the capture protocol's still start/end windows) not
    resting flat on the table. Greedy frame-by-frame picking (nearest to the previous pose) was
    tried first and failed on real data: after an occlusion it can lock onto the wrong mirror
    solution and carry it to the end of the clip."""
    cfg = cfg or {}
    min_area = cfg.get("min_tag_area_px2", 400.0)
    max_rms = cfg.get("max_rms_px", 2.0)
    z_tol, ax_tol = cfg.get("rest_z_tol_m", 0.01), cfg.get("rest_axis_tol_deg", 10.0)
    w_rms, w_rest, w_pen = cfg.get("viterbi_rms_weight", 5.0), 60.0, 60.0
    N = len(per_frame)
    T_C_O = np.full((N, 4, 4), np.nan)
    T_W_O = np.full((N, 4, 4), np.nan)
    rms = np.full(N, np.nan)
    ntag = np.zeros(N, np.int32)
    amb = np.zeros(N, bool)
    reason = np.full(N, "", dtype="U64")
    resting_mask = np.zeros(N, bool)
    for m in rest_masks or []:
        resting_mask |= m
    corners_O = model.box_corners()

    chain = []  # (frame, [T_C_O], [rms], [T_W_O])
    for i, c in enumerate(per_frame):
        box_c = {k: v for k, v in c.items() if k in model.T_O_G}
        if not box_c:
            continue
        res = solve_box_pose(box_c, K, model, min_area_px2=min_area, max_rms_px=max_rms)
        if res is None:
            continue
        cands = res.candidates or [res.T_C_O]
        crms = res.candidate_rms or [res.rms_px]
        keep = [k for k, r in enumerate(crms) if r <= max_rms] or [int(np.argmin(crms))]
        cands, crms = [cands[k] for k in keep], [crms[k] for k in keep]
        ntag[i], amb[i] = len(res.ids), len(cands) > 1
        if not np.isfinite(T_W_C[i]).all():
            k = int(np.argmin(crms))  # no world frame: cannot compare over time
            T_C_O[i], rms[i], reason[i] = cands[k], crms[k], "reproj (no camera pose)"
            continue
        chain.append((i, cands, crms, [T_W_C[i] @ T for T in cands]))

    if chain:
        # unary costs
        U = []
        for i, cands, crms, Tw in chain:
            u = []
            for T, r in zip(Tw, crms):
                c = w_rms * (r - min(crms))
                if se3.transform_points(T, corners_O)[:, 2].min() < -0.015:
                    c += w_pen
                if resting_mask[i] and not _resting_plausible(T, model, z_tol, ax_tol):
                    c += w_rest
                u.append(c)
            U.append(np.array(u))
        # forward pass
        cost, back = [U[0]], []
        for k in range(1, len(chain)):
            prevT, curT = chain[k - 1][3], chain[k][3]
            P = np.degrees([[se3.rotation_distance(a, b) for b in curT] for a in prevT])
            tot = cost[-1][:, None] + P
            back.append(np.argmin(tot, axis=0))
            cost.append(tot.min(axis=0) + U[k])
        # backtrack
        s = int(np.argmin(cost[-1]))
        picks = [s]
        for b in reversed(back):
            s = int(b[s])
            picks.append(s)
        picks.reverse()
        for (i, cands, crms, Tw), s in zip(chain, picks):
            T_C_O[i], T_W_O[i], rms[i] = cands[s], Tw[s], crms[s]
            reason[i] = "joint" if len(cands) == 1 and ntag[i] > 1 else ("viterbi" if len(cands) > 1 else "single")
    return BoxTrack(T_C_O, T_W_O, rms, ntag, amb, reason)


def window_mask(t: np.ndarray, start: float, end: float) -> np.ndarray:
    return (t >= start) & (t <= end)


def static_window_pose(T: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """Averaged pose (chordal-mean rotation, median translation) over valid frames in mask,
    plus jitter stats (translation std per axis -> norm, rotation std about the mean)."""
    sel = mask & np.isfinite(T).all(axis=(1, 2))
    if sel.sum() == 0:
        return np.full((4, 4), np.nan), {"n": 0}
    Ts = T[sel]
    mean = se3.average_poses(Ts)
    dt = Ts[:, :3, 3] - mean[:3, 3]
    drot = np.degrees(se3.rotation_distance(Ts, mean))
    return mean, {"n": int(sel.sum()), "pos_std_m": float(np.linalg.norm(dt.std(0))),
                  "rot_std_deg": float(np.sqrt(np.mean(drot**2)))}
