"""Phase 2 step 2: join candidate images to MegaDetector (MDv5a, RDE-filtered) boxes.

Each results JSON is streamed with ijson so the large files (NACTI is 1.6 GB
uncompressed) never sit in memory. A frame is kept only if MD found an animal box
at or above the confidence floor; this removes the empty frames that
sequence-level labels mark as e.g. "wolf". At most MAX_BOXES boxes per image.

Output: data/manifests/boxes.parquet  (one row per box)
"""
from __future__ import annotations

import time
import zipfile
from pathlib import Path

import ijson
import pandas as pd

from rafl.data.taxa import DATASETS

ROOT = Path(__file__).resolve().parents[1]
MD_DIR = ROOT / "data/lila/md"
CAND = ROOT / "data/manifests/candidates.parquet"
OUT = ROOT / "data/manifests/boxes.parquet"
CONF_FLOOR = 0.2
MIN_BOX_FRAC = 0.0015  # drop boxes smaller than ~0.15% of the frame (a few pixels at 224)
MAX_BOXES = 3


def norm(p: str) -> str:
    return p.replace("\\", "/").lstrip("/")


def main() -> None:
    cand = pd.read_parquet(CAND, columns=["dataset", "rel_path"])
    out = []
    for ds, (fname, _lic) in DATASETS.items():
        want = set(cand.loc[cand.dataset == ds, "rel_path"].map(norm))
        if not want:
            continue
        t0, seen, hit, kept = time.time(), 0, 0, 0
        with zipfile.ZipFile(MD_DIR / fname) as z:
            member = next(m for m in z.namelist() if m.endswith(".json"))
            with z.open(member) as fh:
                for im in ijson.items(fh, "images.item", use_float=True):
                    seen += 1
                    f = norm(im.get("file", ""))
                    if f not in want:
                        continue
                    hit += 1
                    dets = [d for d in (im.get("detections") or [])
                            if d.get("category") == "1" and d.get("conf", 0) >= CONF_FLOOR
                            and d["bbox"][2] * d["bbox"][3] >= MIN_BOX_FRAC]
                    dets.sort(key=lambda d: -d["conf"])
                    for i, d in enumerate(dets[:MAX_BOXES]):
                        x, y, w, h = d["bbox"]
                        out.append((ds, f, i, d["conf"], x, y, w, h))
                        kept += 1
        print(f"{ds}: scanned {seen}, matched {hit}/{len(want)} candidates, "
              f"{kept} boxes, {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(out, columns=["dataset", "rel_path", "box_idx", "conf", "x", "y", "w", "h"])
    df.to_parquet(OUT, index=False)
    print(f"wrote {len(df)} boxes for {df.groupby(['dataset','rel_path']).ngroups} images")


if __name__ == "__main__":
    main()
