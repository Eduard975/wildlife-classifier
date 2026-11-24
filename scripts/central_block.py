"""Central (non-federated) reference for the two-tier model: train the head only, or the 1x1 conv and the
head, on all train-split crops from the same starting point as the FL runs (runs/fl_hetero/init_imagenet.npz).

  PYTHONPATH=src .venv/bin/python scripts/central_block.py
Writes runs/fl_hetero/central.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rafl.fl import block

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["bear", "wolf", "lynx", "wild_boar", "other"]


def main():
    q = json.load(open(ROOT / "runs/fl_hetero/block_quant.json"))
    xq = np.load(ROOT / "data/features/qat_block_q.npy", mmap_mode="r")
    meta = pd.read_parquet(ROOT / "data/features/qat_block_meta.parquet")
    y = meta.label.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    deq = lambda a: (a.astype(np.float32) - q["x"][1]) * q["x"][0]  # noqa: E731
    tr = np.flatnonzero((meta.split == "train").to_numpy())
    te = np.flatnonzero(((meta.split == "test") & (meta.verified != "n")).to_numpy())
    xtr, xte = deq(xq[tr]), deq(xq[te])
    init = dict(np.load(ROOT / "runs/fl_hetero/init_imagenet.npz"))
    out = {}
    for tier in ("head", "block"):
        rng = np.random.default_rng(0)
        p = init
        for ep in range(1, 6):
            p = block.local_train(p, xtr, y[tr], tier, 1, 0.003, 0.9, 32, rng)
            m = block.metrics(p, xte, y[te], CLASSES)
            print(tier, "epoch", ep, "test bacc", round(m["bacc"], 4), flush=True)
        out[tier] = m
    json.dump(out, open(ROOT / "runs/fl_hetero/central.json", "w"), indent=1)


if __name__ == "__main__":
    main()
