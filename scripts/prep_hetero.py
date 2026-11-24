"""Starting point and cost measurements for the two-tier FL experiments (rafl.fl.block).

Starting point: the 1x1 conv of the last block comes from the ImageNet EfficientNet-Lite0
(BatchNorm folded in), not from the centrally fine-tuned model, and the head starts at random.
So federated training has something to learn in both tiers, while the frozen backbone below
(the cached 7x7x320 input) is shared by every policy.

Also times one local epoch of each tier on this machine (308 samples, the board client's size).

  PYTHONPATH=src .venv/bin/python scripts/prep_hetero.py
Writes runs/fl_hetero/init_imagenet.npz and runs/fl_hetero/tier_timing_mac.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import timm
import torch

from rafl.fl import block

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / "runs/fl_hetero"
    m = timm.create_model("efficientnet_lite0.ra_in1k", pretrained=True).eval()
    w = m.conv_head.weight.detach().numpy().reshape(1280, 320)
    bn = m.bn2
    s = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().numpy()
    Wc = (w * s[:, None]).T.astype(np.float32)
    bc = (bn.bias.detach().numpy() - bn.running_mean.detach().numpy() * s).astype(np.float32)
    rng = np.random.default_rng(0)
    p = {"Wc": Wc, "bc": bc, "W": (rng.standard_normal((1280, 5)) * 0.01).astype(np.float32),
         "b": np.zeros(5, np.float32)}
    np.savez(out / "init_imagenet.npz", **p)

    q = json.load(open(out / "block_quant.json"))
    xs = np.load(ROOT / "data/features/qat_block_q.npy", mmap_mode="r")
    x = (xs[:308].astype(np.float32) - q["x"][1]) * q["x"][0]
    y = rng.integers(0, 5, 308)
    timing = {}
    for tier in ("head", "block"):
        t = time.perf_counter()
        for _ in range(3):
            block.local_train(p, x, y, tier, 1, 0.003, 0.9, 32, rng)
        timing[tier] = (time.perf_counter() - t) / 3
        print(f"{tier}: {timing[tier]:.4f} s per epoch over 308 samples, {block.n_params(tier, p)} parameters")
    json.dump(timing, open(out / "tier_timing_mac.json", "w"), indent=1)


if __name__ == "__main__":
    main()
