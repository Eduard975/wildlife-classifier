"""Phase 2 step 3: cap per class, split by location, download and crop.

- Caps are applied per sequence (highest-confidence frames first) so a class's
  budget is spread over many sequences instead of a few long bursts.
- Split is by location (rafl.data.split), asserted to have zero overlap.
- Images are fetched from the LILA GCS bucket, cropped to a padded square around
  each MD box, resized to 224x224 and saved as JPEG. Full images are discarded.
- Resumable: existing crops are skipped. Failed downloads are recorded, not fatal.

Output: data/crops/<class>/<id>.jpg and data/manifests/crops.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from PIL import Image, ImageOps

from rafl.data.split import no_location_overlap, split_locations
from rafl.data.taxa import DATASETS, has_real_sequence, to_class, unit_hash

ROOT = Path(__file__).resolve().parents[1]
CAND = ROOT / "data/manifests/candidates.parquet"
BOXES = ROOT / "data/manifests/boxes.parquet"
CROPS = ROOT / "data/crops"
OUT = ROOT / "data/manifests/crops.parquet"

CAP = {"bear": 6000, "wolf": 10**9, "lynx": 5000, "wild_boar": 5000,
       "coyote": 2500, "dog": 1000, "other_animal": 6500}
FRAMES_PER_SEQ = {"wolf": 6}
DEFAULT_FRAMES_PER_SEQ = 3
FRACS = {"train": 0.70, "val": 0.15, "test": 0.15}
SIZE, PAD = 224, 0.15

_local = threading.local()


def session() -> requests.Session:
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s


def select() -> pd.DataFrame:
    cand = pd.read_parquet(CAND)
    cand["rel_path"] = cand["rel_path"].str.replace("\\", "/", regex=False).str.lstrip("/")
    boxes = pd.read_parquet(BOXES)
    df = boxes.merge(cand, on=["dataset", "rel_path"], how="inner")
    df["seq_key"] = df["sequence_id"].where(df["sequence_id"].map(has_real_sequence), df["image_id"])
    # per sequence: keep the best frames (by top box confidence)
    best = df.groupby(["seq_key", "rel_path", "fine"])["conf"].max().reset_index()
    best["rank"] = best.groupby("seq_key")["conf"].rank(ascending=False, method="first")
    lim = best["fine"].map(FRAMES_PER_SEQ).fillna(DEFAULT_FRAMES_PER_SEQ)
    keep_imgs = set(best.loc[best["rank"] <= lim, "rel_path"])
    df = df[df.rel_path.isin(keep_imgs)].copy()
    # per class cap, sampled by sequence in deterministic hashed order
    df["seq_order"] = df["seq_key"].map(unit_hash)
    df = df.sort_values(["fine", "seq_order", "seq_key", "rel_path", "box_idx"])
    df["n"] = df.groupby("fine").cumcount()
    df = df[df["n"] < df["fine"].map(CAP)].drop(columns=["n", "seq_order"])
    df["label"] = df["fine"].map(to_class)
    split = split_locations(df["location_id"], FRACS)
    assert no_location_overlap(split)
    df["split"] = df["location_id"].map(split)
    df["licence"] = df["dataset"].map(lambda d: DATASETS[d][1])
    df["crop_id"] = [hashlib.sha1(f"{u}#{i}".encode()).hexdigest()[:16]
                     for u, i in zip(df.url, df.box_idx)]
    df["crop_path"] = "data/crops/" + df["label"] + "/" + df["crop_id"] + ".jpg"
    return df.reset_index(drop=True)


def square(x, y, w, h, W, H):
    side = min(max(w * W, h * H) * (1 + 2 * PAD), W, H)
    cx, cy = (x + w / 2) * W, (y + h / 2) * H
    left = min(max(cx - side / 2, 0), W - side)
    top = min(max(cy - side / 2, 0), H - side)
    return int(left), int(top), int(left + side), int(top + side)


def fetch_and_crop(url: str, items: list[tuple]) -> tuple[str, str]:
    todo = [it for it in items if not (ROOT / it[0]).exists()]
    if not todo:
        return url, "cached"
    for attempt in range(3):
        try:
            r = session().get(url, timeout=30)
            r.raise_for_status()
            im = ImageOps.exif_transpose(Image.open(io.BytesIO(r.content))).convert("RGB")
            break
        except Exception as e:  # noqa: BLE001 - record and move on
            err = f"{type(e).__name__}: {e}"[:200]
            time.sleep(1 + attempt * 2)
    else:
        return url, "fail " + err
    W, H = im.size
    for path, x, y, w, h in todo:
        p = ROOT / path
        p.parent.mkdir(parents=True, exist_ok=True)
        im.crop(square(x, y, w, h, W, H)).resize((SIZE, SIZE), Image.BICUBIC).save(p, quality=90)
    return url, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    df = select()
    print("crops per class/split:\n", df.groupby(["label", "split"]).size().unstack(fill_value=0))
    print("wolf locations per split:", df[df.label == "wolf"].groupby("split").location_id.nunique().to_dict())
    print(f"{df.url.nunique()} images to fetch for {len(df)} crops")
    if args.dry_run:
        return
    jobs = {u: list(zip(g.crop_path, g.x, g.y, g.w, g.h)) for u, g in df.groupby("url")}
    status, t0 = {}, time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(fetch_and_crop, u, it) for u, it in jobs.items()]
        for i, f in enumerate(as_completed(futs), 1):
            u, s = f.result()
            status[u] = s
            if i % 1000 == 0:
                fails = sum(v.startswith("fail") for v in status.values())
                print(f"{i}/{len(jobs)} images, {fails} failed, {time.time()-t0:.0f}s", flush=True)
    df["fetch_status"] = df.url.map(status)
    df["ok"] = df.crop_path.map(lambda p: (ROOT / p).exists())
    df.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {df.ok.sum()}/{len(df)} crops present; "
          f"{df[~df.ok].url.nunique()} images failed")


if __name__ == "__main__":
    main()
