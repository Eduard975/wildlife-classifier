import json

import numpy as np
from PIL import Image

from rafl.edge.node_io import DirectoryBurstSource, pick_alert_crop, split_jpegs, write_alert
from rafl.fl.real import noisy_labels, standardize


def test_split_jpegs():
    assert split_jpegs(b"x\xff\xd8aa\xff\xd9yy\xff\xd8b\xff\xd9\xff\xd8cut") == [b"\xff\xd8aa\xff\xd9", b"\xff\xd8b\xff\xd9"]


def test_pick_alert_crop():
    crops = [{"det_conf": 0.9, "probs": np.array([0.2, 0.8])}, {"det_conf": 0.5, "probs": np.array([0.0, 1.0])},
             {"det_conf": 0.9, "probs": np.array([0.9, 0.1])}]
    assert pick_alert_crop(crops, 1) is crops[0]
    assert pick_alert_crop([], 0) is None


def test_write_alert(tmp_path):
    im = Image.new("RGB", (1280, 720), (10, 20, 30))
    rec = write_alert(tmp_path / "alerts", "b001", "bear", 0.87654, "f/b001_2.jpg", [0.25, 0.25, 0.75, 0.75], im,
                      wall_time=0.0)
    saved = json.loads((tmp_path / "alerts/b001.json").read_text())
    assert saved == rec and rec["species"] == "bear" and rec["confidence"] == 0.877
    thumb = Image.open(tmp_path / "alerts/b001.jpg")
    assert thumb.size == (320, 180)
    assert thumb.getpixel((80, 90))[0] > 200


def test_directory_source(tmp_path):
    Image.new("RGB", (8, 8)).save(tmp_path / "a.jpg")
    json.dump({"classes": ["bear", "other"], "animal_index": 1,
               "bursts": [{"id": "b0", "truth": "bear", "frames": ["a.jpg"]}]}, open(tmp_path / "frames.json", "w"))
    src = DirectoryBurstSource(tmp_path)
    b = next(iter(src.bursts()))
    assert (b.id, b.truth, b.frames[0].name) == ("b0", "bear", "a.jpg")
    assert b.frames[0].load().size == (8, 8)


def test_noisy_labels_and_standardize():
    y = np.zeros(10000, np.int64)
    flipped = noisy_labels(y, 5, 0.1, np.random.default_rng(0))
    assert 0.06 < (flipped != 0).mean() < 0.10
    x = np.array([[2.0, 4.0]])
    assert np.allclose(standardize(x, np.array([1.0, 2.0]), np.array([1.0, 2.0])), [[1.0, 1.0]])
