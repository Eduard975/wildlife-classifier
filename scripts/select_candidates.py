"""Phase 2 step 1: stream the 21.7 GB unified LILA CSV and keep candidate images.

Streams the zip member (never extracted to disk, never loaded whole). Keeps every
target-species image (bear, wolf, lynx, wild boar subsampled) plus deterministic
whole-sequence samples of hard negatives and other animals. Images carrying more
than one label group are dropped, because a crop could not be assigned a class.

Output: data/manifests/candidates.parquet
"""
from __future__ import annotations

import csv
import io
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd

from rafl.data.taxa import DATASETS, group, has_real_sequence, unit_hash

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "data/lila/lila_image_urls_and_labels.csv.zip"
META = ROOT / "data/lila/lila_camera_trap_datasets.csv"
OUT = ROOT / "data/manifests/candidates.parquet"

# Fraction of sequences kept per fine group (1.0 = keep all).
KEEP = {"bear": 1.0, "wolf": 1.0, "lynx": 1.0, "wild_boar": 0.15,
        "coyote": 0.25, "dog": 0.25, "other_animal": 0.02}


def base_urls() -> dict[str, str]:
    out = {}
    for r in csv.DictReader(open(META)):
        name = r["name"]
        for ds in DATASETS:
            if ds == name or ds.startswith(name):
                out[ds] = r["image_base_url_gcp"].rstrip("/") + "/"
    return out


def main() -> None:
    bases = base_urls()
    missing = set(DATASETS) - set(bases)
    if missing:
        sys.exit(f"no base url for {missing}")
    rows: dict[str, dict] = {}
    multi: set[str] = set()
    t0, n = time.time(), 0
    with zipfile.ZipFile(ZIP) as z, z.open("lila_image_urls_and_labels.csv") as fh:
        reader = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", newline=""))
        for r in reader:
            n += 1
            if n % 2_000_000 == 0:
                print(f"{n/1e6:.0f}M rows, {len(rows)} kept, {time.time()-t0:.0f}s", flush=True)
            ds = r["dataset_name"]
            if ds not in DATASETS:
                continue
            fine = group(r["family"], r["genus"], r["scientific_name"], r["class"], r["kingdom"])
            url = r["url_gcp"]
            if fine is None:
                if url in rows:  # image also has a dropped label (e.g. human): ambiguous
                    multi.add(url)
                continue
            seq = r["sequence_id"] if has_real_sequence(r["sequence_id"]) else r["image_id"]
            if unit_hash(seq) >= KEEP[fine]:
                continue
            prev = rows.get(url)
            if prev is not None:
                if prev["fine"] != fine:
                    multi.add(url)
                continue
            rows[url] = {
                "dataset": ds,
                "url": url,
                "rel_path": url[len(bases[ds]):] if url.startswith(bases[ds]) else None,
                "image_id": r["image_id"],
                "sequence_id": r["sequence_id"],
                "location_id": r["location_id"],
                "frame_num": r["frame_num"],
                "datetime": r["datetime"],
                "annotation_level": r["annotation_level"],
                "scientific_name": r["scientific_name"],
                "fine": fine,
            }
    for u in multi:
        rows.pop(u, None)
    df = pd.DataFrame(rows.values())
    bad = df["rel_path"].isna().sum()
    df = df.dropna(subset=["rel_path"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"done: {n} rows scanned in {time.time()-t0:.0f}s; kept {len(df)} images; "
          f"dropped {len(multi)} multi-label, {bad} unmatched base url")
    print(df.groupby(["fine", "dataset"]).size().to_string())


if __name__ == "__main__":
    main()
