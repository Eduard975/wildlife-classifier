"""Sanity checks for the FL-on-NPU features (runs/fl_npu): deployed head accuracy through the int8
emulation, and central / FedAvg head training from scratch at a few learning rates.

  PYTHONPATH=src .venv/bin/python scripts/check_fl_npu_features.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from flwr.server.strategy.aggregate import aggregate

from rafl.fl import npu_head
from rafl.fl.sim import init_head, metrics, sgd

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["bear", "wolf", "lynx", "wild_boar", "other"]

pm = npu_head.load_map(ROOT / "runs/fl_npu/patch_map.json")
fq = np.load(ROOT / "data/features/qat_int8_q.npy")
meta = pd.read_parquet(ROOT / "data/features/qat_int8_meta.parquet")
x = npu_head.dequant_features(fq, pm)
y = meta.label.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
te = ((meta.split == "test") & (meta.verified != "n")).to_numpy()
tr = (meta.split == "train").to_numpy()
calib = np.load(ROOT / "runs/fl_npu/calib_features_q.npy")
h = np.load(ROOT / "runs/fl_npu/init_head.npz")
W0, b0 = h["W"], h["b"]


def int8_bacc(W, b, idx):
    q = npu_head.quantize_head(W, b, pm, calib)
    pred = npu_head.emulate(fq[idx], q, pm).argmax(1)
    return float(np.mean([(pred[y[idx] == i] == i).mean() for i in range(5)]))


print("deployed head  float bacc", round(metrics([W0, b0], x[te], y[te], CLASSES)["bacc"], 4),
      " int8 (re-quantized) bacc", round(int8_bacc(W0, b0, te), 4))
rng = np.random.default_rng(0)
for lr in (0.003, 0.01, 0.03):
    p = sgd(init_head(1280, 5, rng), x[tr], y[tr], 5, 5, lr, 0.9, 32, rng)
    print(f"central from scratch lr {lr}: 5 epochs float bacc {metrics(p, x[te], y[te], CLASSES)['bacc']:.4f}"
          f" int8 {int8_bacc(*p, te):.4f}")

locs = meta[tr].location_id.value_counts()
locs = locs[locs >= 100].index[:9]
for lr in (0.01, 0.03):
    p = init_head(1280, 5, np.random.default_rng(1))
    for r in range(10):
        res = []
        for loc in locs:
            m = (meta.location_id == loc).to_numpy() & tr
            res.append((sgd(p, x[m], y[m], 5, 5, lr, 0.9, 32, rng), int(m.sum())))
        p = [a.astype(np.float32) for a in aggregate(res)]
    print(f"FedAvg 9 clients 10 rounds 5 epochs lr {lr}: float {metrics(p, x[te], y[te], CLASSES)['bacc']:.4f}"
          f" int8 {int8_bacc(*p, te):.4f}")
