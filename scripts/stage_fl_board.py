"""Stage one TRAIN-split location's crops, the head standardisation stats and Mac reference
features for the board FL client.

  .venv/bin/python scripts/stage_fl_board.py --location "NACTI : lebec_CA-42"
Writes runs/fl_real/board_stage/{manifest.json,stats.npz,ref_features.npy,crops/}.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="efficientnet_lite0")
    ap.add_argument("--location", default="NACTI : lebec_CA-42")
    ap.add_argument("--out", default="runs/fl_real/board_stage")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))
    classes = cfg["models"]["classifier"]["classes"]
    x = np.load(ROOT / f"data/features/{args.arch}.npy")
    meta = pd.read_parquet(ROOT / f"data/features/{args.arch}_meta.parquet")
    crops = pd.read_parquet(ROOT / cfg["paths"]["manifest"]).set_index("crop_id").crop_path
    tr = (meta.split == "train").to_numpy()
    xt = x[tr].astype(np.float32)
    mu, sd = xt.mean(0), xt.std(0) + 1e-6

    g = meta[(meta.split == "train") & (meta.location_id == args.location)]
    assert len(g), "location not in train split"
    out = ROOT / args.out
    if out.exists():
        shutil.rmtree(out)
    (out / "crops").mkdir(parents=True)
    rows = []
    for r in g.itertuples():
        name = f"{r.crop_id}.jpg"
        shutil.copy(ROOT / crops[r.crop_id], out / "crops" / name)
        rows.append({"file": name, "label": classes.index(r.label)})
    json.dump({"location": args.location, "classes": classes, "crops": rows}, open(out / "manifest.json", "w"))
    np.savez(out / "stats.npz", mu=mu, sd=sd)
    np.save(out / "ref_features.npy", x[g.index.to_numpy()].astype(np.float32))
    print(args.location, len(rows), g.label.value_counts().to_dict())


if __name__ == "__main__":
    main()
