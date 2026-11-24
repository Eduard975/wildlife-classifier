"""Download train-split full frames (positives and empties) for detector calibration and QAT.

Frames come from train-split locations only, same split and frame format as
stage_detector_frames.py. Writes data/frames/train/*.jpg and data/manifests/train_frames.parquet
(+ train_frame_boxes.parquet with LILA MegaDetector v5 boxes, normalized xywh top-left).

Usage: .venv/bin/python scripts/calib_detector_frames.py --pos 2400 --empty 600
"""
from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from rafl.data.split import split_locations
from rafl.data.taxa import to_class, unit_hash
from stage_detector_frames import FRACS, OUT, fetch

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", type=int, default=2400)
    ap.add_argument("--empty", type=int, default=600)
    ap.add_argument("--min-conf", type=float, default=0.2)
    args = ap.parse_args()
    cand = pd.read_parquet(ROOT / "data/manifests/candidates.parquet")
    boxes = pd.read_parquet(ROOT / "data/manifests/boxes.parquet")
    boxes = boxes[boxes.conf >= args.min_conf]
    cand["label"] = cand.fine.map(to_class)
    cand["split"] = cand.location_id.map(split_locations(cand.location_id, FRACS))
    has_box = set(zip(boxes.dataset, boxes.rel_path))
    cand["has_box"] = [k in has_box for k in zip(cand.dataset, cand.rel_path)]
    cand["h"] = cand.image_id.map(unit_hash)
    train = cand[cand.split == "train"].sort_values("h")
    assert not set(train.location_id) & set(cand[cand.split == "test"].location_id)
    per = args.pos // train[train.has_box].label.nunique()
    pos = train[train.has_box].groupby("label").head(per)
    neg = train[~train.has_box].head(args.empty)
    frames = pd.concat([pos, neg]).assign(role="train")
    frames["frame_path"] = [f"train/{hashlib.sha1(i.encode()).hexdigest()[:16]}.jpg" for i in frames.image_id]
    with ThreadPoolExecutor(16) as ex:
        frames["ok"] = list(ex.map(fetch, frames.itertuples()))
    frames = frames[frames.ok]
    ref = boxes.merge(frames[["dataset", "rel_path", "frame_path"]], on=["dataset", "rel_path"])
    frames.drop(columns=["h"]).to_parquet(ROOT / "data/manifests/train_frames.parquet")
    ref.to_parquet(ROOT / "data/manifests/train_frame_boxes.parquet")
    print(frames.groupby(["has_box", "label"]).size())
    print("reference boxes:", len(ref), "dir", OUT / "train")


if __name__ == "__main__":
    main()
