import numpy as np

from ego2libero import video


def test_vfr_timestamps_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    dt = rng.choice([1 / 60, 1 / 60, 1 / 60, 1 / 30, 1 / 55], size=59)  # irregular gaps
    t = np.r_[0.0, np.cumsum(dt)]
    path = tmp_path / "vfr.mp4"
    with video.VideoWriter(path, 320, 240, crf=10) as w:
        for i, ti in enumerate(t):
            img = np.full((240, 320, 3), 4 * i, dtype=np.uint8)  # brightness encodes index
            w.write(img, ti)

    ts_fast = video.read_timestamps(path)
    ts_dec = video.read_timestamps(path, decode=True)
    np.testing.assert_allclose(ts_fast, t, atol=1e-4)
    np.testing.assert_allclose(ts_dec, t, atol=1e-4)

    seen = list(video.iter_frames(path))
    assert [i for i, _, _ in seen] == list(range(len(t)))
    np.testing.assert_allclose([ti for _, ti, _ in seen], t, atol=1e-4)
    levels = np.array([img.mean() for *_, img in seen])
    assert np.all(np.diff(levels) > 0)  # presentation order (codec shifts flat levels a little)

    rep = video.frame_rate_report(ts_fast)
    assert rep["is_vfr"] and rep["n_frames"] == len(t)
    info = video.probe(path)
    assert (info.width, info.height) == (320, 240)


def test_iter_frames_window(tmp_path):
    path = tmp_path / "cfr.mp4"
    with video.VideoWriter(path, 64, 48) as w:
        for i in range(20):
            w.write(np.zeros((48, 64, 3), np.uint8), i / 60)
    got = [(i, t) for i, t, _ in video.iter_frames(path, start=5, stop=8)]
    assert [i for i, _ in got] == [5, 6, 7]
    np.testing.assert_allclose([t for _, t in got], np.arange(5, 8) / 60, atol=1e-4)
