"""AprilTag detection over a whole clip, cached to detections.npz (plan stage S1).

Everything downstream reads the cache instead of re-decoding video.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from ego2libero import video


REFINE = {"apriltag": cv2.aruco.CORNER_REFINE_APRILTAG, "subpix": cv2.aruco.CORNER_REFINE_SUBPIX,
          "contour": cv2.aruco.CORNER_REFINE_CONTOUR, "none": cv2.aruco.CORNER_REFINE_NONE}


def make_detector(dictionary: str = "DICT_APRILTAG_36h11", refine: str = "apriltag") -> cv2.aruco.ArucoDetector:
    """Corner refinement defaults to AprilTag edge fitting, not plan.md's SUBPIX: on rendered
    tags SUBPIX puts corners ~0.3 px inside the true corner (cornerSubPix targets saddle points,
    and a tag's outer corner is an L), which shrinks tags and pushes single-object poses away
    from the camera (+4 mm at 0.5 m for 37 px tags). APRILTAG is unbiased, just noisier."""
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = REFINE[refine]
    return cv2.aruco.ArucoDetector(d, p)


@dataclass
class Detections:
    t: np.ndarray  # (N,) seconds, from PTS
    frame: np.ndarray  # (M,) frame index of each detection
    ids: np.ndarray  # (M,)
    corners: np.ndarray  # (M, 4, 2) TL, TR, BR, BL in pixels
    area: np.ndarray  # (M,) pixel area

    @property
    def n_frames(self) -> int:
        return len(self.t)

    def at(self, i: int, ids=None) -> dict[int, np.ndarray]:
        """{tag id: (4, 2) corners} in frame i, optionally restricted to ``ids``."""
        sel = np.flatnonzero(self.frame == i)
        out = {int(self.ids[k]): self.corners[k] for k in sel}
        if ids is not None:
            ids = set(ids)
            out = {k: v for k, v in out.items() if k in ids}
        return out

    def per_frame(self, ids=None) -> list[dict[int, np.ndarray]]:
        order = np.argsort(self.frame, kind="stable")
        bounds = np.searchsorted(self.frame[order], np.arange(self.n_frames + 1))
        keep = None if ids is None else set(ids)
        out = []
        for i in range(self.n_frames):
            ks = order[bounds[i]:bounds[i + 1]]
            out.append({int(self.ids[k]): self.corners[k] for k in ks if keep is None or int(self.ids[k]) in keep})
        return out

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, t=self.t, frame=self.frame, ids=self.ids, corners=self.corners, area=self.area)

    @classmethod
    def load(cls, path: str | Path) -> "Detections":
        d = np.load(path)
        return cls(d["t"], d["frame"], d["ids"], d["corners"], d["area"])


def detect_clip(path: str | Path, dictionary: str = "DICT_APRILTAG_36h11", refine: str = "apriltag",
                progress: bool = True) -> Detections:
    det = make_detector(dictionary, refine)
    ts, frames, ids, corners, areas = [], [], [], [], []
    it = video.iter_frames(path, fmt="gray")
    for i, t, gray in tqdm(it, desc=f"detect {Path(path).parent.name}", disable=not progress, unit="f"):
        ts.append(t)
        c, idx, _ = det.detectMarkers(gray)
        if idx is None:
            continue
        for cc, k in zip(c, idx.ravel()):
            q = cc.reshape(4, 2).astype(np.float64)
            frames.append(i)
            ids.append(int(k))
            corners.append(q)
            areas.append(abs(cv2.contourArea(q.astype(np.float32))))
    return Detections(np.asarray(ts), np.asarray(frames, dtype=np.int32), np.asarray(ids, dtype=np.int32),
                      np.asarray(corners, dtype=np.float64).reshape(-1, 4, 2), np.asarray(areas))
