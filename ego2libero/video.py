"""PTS-accurate video I/O (PyAV).

Phone video is often variable frame rate, so timestamps always come from each frame's
presentation timestamp, never from ``frame_idx / fps``. ``t`` is seconds from the first
presented frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import av
import numpy as np


@dataclass
class VideoInfo:
    width: int
    height: int
    n_frames: int  # container's claim; may be 0 or wrong, trust read_timestamps() instead
    avg_fps: float
    time_base: Fraction
    codec: str
    rotation: int  # display-matrix rotation in degrees, 0 if none


def probe(path: str | Path) -> VideoInfo:
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        rot = 0
        try:
            # PyAV >= 13 exposes the display matrix rotation on frames, not streams;
            # read the first frame to get it.
            f = next(c.decode(s))
            rot = int(round(getattr(f, "rotation", 0) or 0))
        except StopIteration:
            pass
        return VideoInfo(
            width=s.codec_context.width,
            height=s.codec_context.height,
            n_frames=int(s.frames or 0),
            avg_fps=float(s.average_rate) if s.average_rate else float("nan"),
            time_base=s.time_base,
            codec=s.codec_context.name,
            rotation=rot,
        )


def capture_meta(path: str | Path) -> dict:
    """Camera mode recorded in the file: size, nominal fps, and the camera app's lens,
    shutter and ISO tags when present (Blackmagic Camera writes them)."""
    info = probe(path)
    with av.open(str(path)) as c:
        m = dict(c.metadata)
    pick = lambda suffix: next((v for k, v in m.items() if k.endswith(suffix)), None)
    return {"width": info.width, "height": info.height, "fps": round(info.avg_fps) if info.avg_fps == info.avg_fps else None,
            "lens": pick("camera.lensType"), "shutter": pick("shutterSpeed"), "iso": pick("camera.iso"),
            "model": pick("quicktime.model")}


def iter_frames(
    path: str | Path, fmt: str = "bgr24", start: int = 0, stop: int | None = None
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield ``(frame_idx, t_seconds, image)`` in presentation order.

    ``fmt`` is any PyAV pixel format (``bgr24`` for OpenCV, ``gray`` for detection).
    Frames before ``start`` are decoded but not converted.
    """
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        t0 = None
        for i, frame in enumerate(_decode_checked(c, s, path)):
            if frame.pts is None:
                raise ValueError(f"{path}: frame {i} has no PTS")
            if t0 is None:
                t0 = frame.pts
            if stop is not None and i >= stop:
                break
            if i < start:
                continue
            yield i, float((frame.pts - t0) * s.time_base), frame.to_ndarray(format=fmt)


def read_timestamps(path: str | Path, decode: bool = False) -> np.ndarray:
    """Presentation timestamps (s, from first frame) of every frame, in presentation order.

    Default demuxes packets only (fast) and sorts their PTS, which equals presentation
    order even with B-frames. ``decode=True`` decodes every frame instead (slow, exact).
    """
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        if decode:
            pts = [f.pts for f in _decode_checked(c, s, path)]
        else:
            pts = [p.pts for p in c.demux(s) if p.pts is not None and p.size > 0]
        _check_count(len(pts), s, path)
    if not pts:
        raise ValueError(f"{path}: no frames")
    pts = np.sort(np.asarray(pts, dtype=np.int64))
    return ((pts - pts[0]) * float(s.time_base)).astype(np.float64)


class TruncatedVideoError(RuntimeError):
    pass


def _check_count(n_read: int, stream, path) -> None:
    """The container header states the frame count; far fewer readable frames means the file
    is cut off (an interrupted or still-running copy), so refuse it instead of using a prefix."""
    claimed = int(stream.frames or 0)
    if claimed and n_read < 0.98 * claimed - 2:
        raise TruncatedVideoError(f"{path}: header says {claimed} frames but only {n_read} are readable; "
                                  "the file is truncated or still being copied")


def _decode_checked(container, stream, path):
    n = 0
    try:
        for frame in container.decode(stream):
            n += 1
            yield frame
    except av.error.InvalidDataError as e:
        raise TruncatedVideoError(f"{path}: decoding failed after {n} frames ({e}); "
                                  "the file is truncated or still being copied") from e
    _check_count(n, stream, path)


def frame_rate_report(t: np.ndarray) -> dict:
    """Summary of timing regularity: median dt, jitter, and suspected dropped frames."""
    dt = np.diff(t)
    med = float(np.median(dt))
    return {
        "n_frames": int(len(t)),
        "duration_s": float(t[-1] - t[0]),
        "median_fps": 1.0 / med if med > 0 else float("nan"),
        "dt_min_ms": float(dt.min() * 1e3),
        "dt_max_ms": float(dt.max() * 1e3),
        "dt_std_ms": float(dt.std() * 1e3),
        "n_gaps_over_1p5x": int((dt > 1.5 * med).sum()),
        "is_vfr": bool(dt.std() > 0.05 * med),
    }


class VideoWriter:
    """H.264 writer that stamps each frame with a caller-supplied time in seconds.

    Pass the source PTS through to keep the written clip's timing identical to the input.
    ``crf`` <= 14 is visually lossless enough for corner detection.
    """

    def __init__(self, path: str | Path, width: int, height: int, crf: int = 14,
                 time_base: Fraction = Fraction(1, 90000), codec: str = "libx264",
                 pix_fmt: str = "yuv420p", fps_hint: int = 60):
        self._c = av.open(str(path), mode="w")
        self._s = self._c.add_stream(codec, rate=fps_hint)
        self._s.width, self._s.height = width, height
        self._s.pix_fmt = pix_fmt
        self._s.time_base = time_base
        self._s.codec_context.time_base = time_base
        if codec == "libx264":
            self._s.options = {"crf": str(crf), "preset": "medium"}
        self._tb = time_base
        self._last_pts = -1

    def write(self, img_bgr: np.ndarray, t: float) -> None:
        pts = int(round(t / self._tb))
        if pts <= self._last_pts:
            raise ValueError(f"non-increasing timestamp {t} (pts {pts} <= {self._last_pts})")
        self._last_pts = pts
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(img_bgr), format="bgr24")
        frame.pts = pts
        frame.time_base = self._tb
        for packet in self._s.encode(frame):
            self._c.mux(packet)

    def close(self) -> None:
        for packet in self._s.encode():
            self._c.mux(packet)
        self._c.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
