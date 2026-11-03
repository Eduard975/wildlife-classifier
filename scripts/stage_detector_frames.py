"""Detector step 0: download full frames for detector calibration (train) and evaluation (test).

Evaluation frames come from test-split locations: frames with LILA MegaDetector boxes
(the reference) per class, plus labelled frames where MegaDetector found no box >= 0.2
(treated as empty). Frames are stored with the long side at 1280 px.

Usage: .venv/bin/python scripts/stage_detector_frames.py
"""
from __future__ import annotations

import hashlib
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from PIL import Image, ImageOps

from rafl.data.split import split_locations
from rafl.data.taxa import to_class, unit_hash

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/frames"
PER_CLASS_EVAL = 60
EMPTY_EVAL = 100
CALIB = 200
LONG_SIDE = 1280
FRACS = {"train": 0.70, "val": 0.15, "test": 0.15}  # same as build_crops.py


def fetch(row):
    path = OUT / row.frame_path
    if path.exists():
        return True
    for _ in range(3):
        try:
            r = requests.get(row.url, timeout=30)
            r.raise_for_status()
            im = ImageOps.exif_transpose(Image.open(io.BytesIO(r.content))).convert("RGB")
            im.thumbnail((LONG_SIDE, LONG_SIDE))
            path.parent.mkdir(parents=True, exist_ok=True)
            im.save(path, quality=92)
            return True
        except Exception:
            continue
    return False


def main():
    cand = pd.read_parquet(ROOT / "data/manifests/candidates.parquet")
    boxes = pd.read_parquet(ROOT / "data/manifests/boxes.parquet")
    cand["label"] = cand.fine.map(to_class)
    split = split_locations(cand.location_id, FRACS)
    cand["split"] = cand.location_id.map(split)
    has_box = set(zip(boxes.dataset, boxes.rel_path))
    cand["has_box"] = [k in has_box for k in zip(cand.dataset, cand.rel_path)]
    cand["h"] = cand.image_id.map(unit_hash)
    cand = cand.sort_values("h")

    test = cand[cand.split == "test"]
    pos = test[test.has_box].groupby("label").head(PER_CLASS_EVAL).assign(role="eval")
    neg = test[~test.has_box].head(EMPTY_EVAL).assign(role="eval")
    calib = cand[(cand.split == "train") & cand.has_box].groupby("label").head(CALIB // 5).assign(role="calib")
    frames = pd.concat([pos, neg, calib])
    frames["frame_path"] = [f"{r}/{hashlib.sha1(i.encode()).hexdigest()[:16]}.jpg"
                            for r, i in zip(frames.role, frames.image_id)]

    with ThreadPoolExecutor(16) as ex:
        frames["ok"] = list(ex.map(fetch, frames.itertuples()))
    frames = frames[frames.ok]
    ref = boxes.merge(frames[["dataset", "rel_path", "frame_path"]], on=["dataset", "rel_path"])
    frames.drop(columns=["h"]).to_parquet(ROOT / "data/manifests/frames.parquet")
    ref.to_parquet(ROOT / "data/manifests/frame_boxes.parquet")
    print(frames.groupby(["role", "has_box", "label"]).size())
    print("reference boxes:", len(ref))


if __name__ == "__main__":
    main()
