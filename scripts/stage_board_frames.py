"""Pick held-out test bursts of full camera frames and stage them for the board.

Writes <out>/frames.json and <out>/frames/*.jpg. Truth 'empty' = frames where
MegaDetector v5 found no box >= 0.2. Frames whose crop was hand-marked 'n' are dropped.
Copy to the board with board/run_frames.sh.

  .venv/bin/python scripts/stage_board_frames.py --per-class 12 --empty 16 --seed 7
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import yaml

from rafl.data.taxa import has_real_sequence

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=12)
    ap.add_argument("--empty", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="runs/board_stage_frames")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))
    classes = cfg["models"]["classifier"]["classes"]
    names = (ROOT / "runs/detector/MDV6-yolov10-c_416/names.txt").read_text().split("\n")
    fr = pd.read_parquet(ROOT / "data/manifests/frames.parquet").query("role == 'eval'").copy()
    crops = pd.read_parquet(ROOT / cfg["paths"]["manifest"])
    bad = crops[crops.verified == "n"][["dataset", "rel_path"]].drop_duplicates()
    bad_keys = set(zip(bad.dataset, bad.rel_path))
    fr = fr[[k not in bad_keys for k in zip(fr.dataset, fr.rel_path)]]
    fr["seq"] = fr.sequence_id.where(fr.sequence_id.map(has_real_sequence), fr.image_id)
    fr["truth"] = fr.label.where(fr.has_box, "empty")

    out = ROOT / args.out
    if out.exists():
        shutil.rmtree(out)
    (out / "frames").mkdir(parents=True)
    bursts = []
    for c, n in [(c, args.per_class) for c in classes] + [("empty", args.empty)]:
        g = fr[fr.truth == c].groupby("seq").size().rename("n").reset_index()
        g = g.sample(frac=1, random_state=args.seed).sort_values("n", ascending=False, kind="stable")
        for seq in g.seq.head(n):
            rows = fr[(fr.truth == c) & (fr.seq == seq)].sort_values("frame_num").head(6)
            bid = f"b{len(bursts):03d}"
            frames = []
            for k, r in enumerate(rows.itertuples()):
                name = f"frames/{bid}_{k}.jpg"
                shutil.copy(ROOT / "data/frames" / r.frame_path, out / name)
                frames.append(name)
            bursts.append({"id": bid, "truth": c, "dataset": rows.dataset.iloc[0],
                           "location": rows.location_id.iloc[0], "frames": frames})
    meta = {"classes": classes, "animal_index": names.index("animal"), "bursts": bursts}
    json.dump(meta, open(out / "frames.json", "w"), indent=2)
    print(pd.Series([b["truth"] for b in bursts]).value_counts().to_dict())
    print(f"{len(bursts)} bursts, {sum(len(b['frames']) for b in bursts)} frames -> {out}")


if __name__ == "__main__":
    main()
