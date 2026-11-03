"""Pick held-out test bursts and stage their crops for the board runtime.

Writes <out>/bursts.json and <out>/crops/*.jpg. Copy the folder to the board with
board/run_poc.sh.

  .venv/bin/python scripts/stage_board_demo.py --per-class 8 --seed 7
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="runs/board_stage")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))
    classes = cfg["models"]["classifier"]["classes"]
    df = pd.read_parquet(ROOT / cfg["paths"]["manifest"])
    t = df[df.ok & (df.split == "test") & df.label.isin(classes) & (df.verified != "n")]
    t = t[t.groupby("seq_key").crop_id.transform("size") >= 2]

    out = ROOT / args.out
    if (out / "crops").exists():
        shutil.rmtree(out / "crops")
    (out / "crops").mkdir(parents=True)
    bursts = []
    for c in classes:
        keys = t[t.label == c].seq_key.drop_duplicates()
        for key in keys.sample(min(args.per_class, len(keys)), random_state=args.seed):
            g = t[t.seq_key == key].sort_values("frame_num")
            bid = f"b{len(bursts):03d}"
            frames = []
            for k, r in enumerate(g.itertuples()):
                name = f"crops/{bid}_{k}.jpg"
                shutil.copy(ROOT / r.crop_path, out / name)
                frames.append(name)
            bursts.append({"id": bid, "truth": c, "dataset": g.dataset.iloc[0],
                           "location": g.location_id.iloc[0], "frames": frames})
    json.dump({"classes": classes, "bursts": bursts}, open(out / "bursts.json", "w"), indent=2)
    print(f"{len(bursts)} bursts, {sum(len(b['frames']) for b in bursts)} crops -> {out}")


if __name__ == "__main__":
    main()
