import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import print_tags  # noqa: E402

from ego2libero.config import load_yaml


def test_net_folds_to_box_config():
    print_tags.check_net(load_yaml("box")["faces"])


def test_net_check_catches_wrong_orientation(monkeypatch):
    monkeypatch.setattr(print_tags, "NET_HALF_TURN", set())  # top tag printed the wrong way up
    with pytest.raises(AssertionError, match="tag 24"):
        print_tags.check_net(load_yaml("box")["faces"])


def test_net_check_catches_swapped_faces(monkeypatch):
    cells = dict(print_tags.NET_CELLS)
    cells[20], cells[21] = cells[21], cells[20]
    monkeypatch.setattr(print_tags, "NET_CELLS", cells)
    monkeypatch.setattr(print_tags, "NET_PARENT", {20: 22, 21: 22, 23: 20, 24: 22, 25: 22})
    with pytest.raises(AssertionError):
        print_tags.check_net(load_yaml("box")["faces"])
