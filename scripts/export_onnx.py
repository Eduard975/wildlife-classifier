"""Phase 4 step 1 (torch env): checkpoint -> ONNX + int8 calibration set.

Usage: python scripts/export_onnx.py --arch efficientnet_lite0
Writes runs/export/<arch>/model.onnx and calib.npy (NHWC float32, 0..1 range,
drawn from the train split, balanced over classes).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from rafl.models.classifier import build_classifier

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))
    classes = cfg["models"]["classifier"]["classes"]
    size = cfg["models"]["classifier"]["input"]
    n_cal = cfg["quant"]["calibration_crops"]

    ck = torch.load(ROOT / f"runs/phase3/{args.arch}/best.pt", map_location="cpu")
    model = build_classifier(args.arch, len(classes), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    out = ROOT / "runs/export" / args.arch
    out.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, torch.zeros(1, 3, size, size), out / "model.onnx", input_names=["input"],
                      output_names=["logits"], opset_version=17, dynamo=False)

    df = pd.read_parquet(ROOT / cfg["paths"]["manifest"])
    df = df[df.ok & (df.split == "train") & df.label.isin(classes)]
    cal = df.groupby("label", group_keys=False).apply(
        lambda g: g.sample(n_cal // len(classes), random_state=cfg["seed"]))
    arr = np.stack([np.asarray(Image.open(ROOT / p).convert("RGB").resize((size, size)), np.float32) / 255.0
                    for p in cal.crop_path])
    np.save(out / "calib.npy", arr)
    print("onnx + calib", arr.shape, "epoch", ck["epoch"])


if __name__ == "__main__":
    main()
