"""Stage real camera-trap clips for the board: one long black-bear sequence and two empty sequences,
all from the same held-out test camera (AMMonitor, maine-difw-moose-vol2 COFFEELOSN).

Each clip is split into bursts of --burst frames, the way a camera trap triggers, so the board
makes a decision every few frames. Writes <out>/frames.json (DirectoryBurstSource format) and
<out>/frames/*.jpg (long side 1280 px). The source URLs and timestamps are kept in
<out>/raw/sequences.json; if that file is missing, the LILA label file is streamed (about 5 min).

  .venv/bin/python scripts/stage_board_clips.py
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LOCATION = "AMMonitor Camera Traps : maine-difw-moose-vol2_COFFEELOSN"
PREFIX = "AMMonitor Camera Traps : maine-difw-moose-vol2_location_COFFEELOSN_sequence_index_"
CLIPS = [("bear", "bear", ["00123"]), ("empty", "empty", ["00064", "00069"])]


def load_sequences(raw: Path) -> dict:
    meta = raw / "sequences.json"
    if meta.exists():
        return json.load(open(meta))
    want = {PREFIX + s for _, _, ids in CLIPS for s in ids}
    seqs: dict = {}
    with zipfile.ZipFile(ROOT / "data/lila/lila_image_urls_and_labels.csv.zip") as z, \
            z.open("lila_image_urls_and_labels.csv") as fh:
        for r in csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", newline="")):
            if r["sequence_id"] in want:
                seqs.setdefault(r["sequence_id"], []).append(
                    {k: r[k] for k in ("url_gcp", "image_id", "frame_num", "datetime", "original_label")})
    raw.mkdir(parents=True, exist_ok=True)
    json.dump(seqs, open(meta, "w"))
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/board_stage_clips")
    ap.add_argument("--burst", type=int, default=4, help="frames per burst")
    args = ap.parse_args()
    out = ROOT / args.out
    raw = out / "raw"
    seqs = load_sequences(raw)
    ref = json.load(open(ROOT / "runs/board_stage_frames/frames.json"))
    crops = __import__("pandas").read_parquet(ROOT / "data/manifests/crops.parquet")
    assert LOCATION in set(crops[crops.split == "test"].location_id), "clip camera must be a held-out test location"

    if (out / "frames").exists():
        shutil.rmtree(out / "frames")
    (out / "frames").mkdir(parents=True)
    bursts, timing = [], {}
    for name, truth, ids in CLIPS:
        files = []
        for sid in ids:
            rows = sorted(seqs[PREFIX + sid], key=lambda r: (datetime.strptime(r["datetime"], "%m-%d-%Y %H:%M:%S"),
                                                             int(r["frame_num"])))
            for i, r in enumerate(rows):
                src = raw / f"{sid}_{i:03d}.jpg"
                if not src.exists():
                    im = Image.open(io.BytesIO(urllib.request.urlopen(r["url_gcp"], timeout=30).read())).convert("RGB")
                    im.thumbnail((1280, 1280), Image.BICUBIC)
                    im.save(src, quality=92)
                dst = f"frames/{name}_{len(files):03d}.jpg"
                shutil.copy(src, out / dst)
                files.append(dst)
                timing[dst] = {"datetime": r["datetime"], "image_id": r["image_id"], "label": r["original_label"]}
        for k in range(0, len(files), args.burst):
            bursts.append({"id": f"{name}_{k // args.burst:02d}", "truth": truth, "clip": name,
                           "dataset": "AMMonitor Camera Traps", "location": LOCATION,
                           "frames": files[k:k + args.burst]})
        print(f"{name}: {len(files)} frames, {len(range(0, len(files), args.burst))} bursts")
    json.dump({"classes": ref["classes"], "animal_index": ref["animal_index"], "bursts": bursts},
              open(out / "frames.json", "w"), indent=1)
    json.dump(timing, open(out / "timing.json", "w"), indent=1)


if __name__ == "__main__":
    main()
