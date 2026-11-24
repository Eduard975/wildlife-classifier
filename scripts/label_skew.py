"""Label skew across FL clients (train-split camera locations with >= 20 crops).

Reports how concentrated each client's labels are, the "positive labels only" situation of
Yu et al. (FedAwS): clients that see essentially one class.

  .venv/bin/python scripts/label_skew.py
Writes runs/fl_hetero/label_skew.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    meta = pd.read_parquet(ROOT / "data/features/qat_block_meta.parquet")
    tr = meta[meta.split == "train"]
    rows = []
    for loc, g in tr.groupby("location_id"):
        if len(g) < 20:
            continue
        vc = g.label.value_counts()
        p = (vc / vc.sum()).to_numpy()
        rows.append({"location": loc, "n": len(g), "classes": int((vc > 0).sum()), "top_class": vc.index[0],
                     "top_share": float(p[0]), "entropy_bits": float(-(p * np.log2(p)).sum())})
    d = pd.DataFrame(rows)
    out = {
        "clients": len(d),
        "samples": int(d.n.sum()),
        "single_class_clients": int((d.classes == 1).sum()),
        "top_share_ge_0.9": int((d.top_share >= 0.9).sum()),
        "top_share_ge_0.75": int((d.top_share >= 0.75).sum()),
        "median_top_share": float(d.top_share.median()),
        "median_classes": float(d.classes.median()),
        "mean_entropy_bits": float(d.entropy_bits.mean()),
        "max_entropy_bits": float(np.log2(5)),
        "single_class_by_label": d[d.classes == 1].top_class.value_counts().to_dict(),
        "clients_with_wolf": int(sum((g.label == "wolf").any() for _, g in tr.groupby("location_id") if len(g) >= 20)),
    }
    json.dump(out, open(ROOT / "runs/fl_hetero/label_skew.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
