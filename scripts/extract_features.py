"""Cache frozen ImageNet-backbone features for every crop (input to head-only FL).

The backbone stays frozen in federated training, so features are computed once.
Runs on CPU by default so it does not compete with GPU training.

Usage: python scripts/extract_features.py --arch efficientnet_lite0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from rafl.models.classifier import IMAGENET_MEAN, IMAGENET_STD, TIMM_NAMES

import timm

ROOT = Path(__file__).resolve().parents[1]
META = ["crop_id", "label", "split", "dataset", "location_id", "seq_key", "verified"]


class Imgs(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))
    classes = cfg["models"]["classifier"]["classes"]
    size = cfg["models"]["classifier"]["input"]
    df = pd.read_parquet(ROOT / cfg["paths"]["manifest"])
    df = df[df.ok & df.label.isin(classes)].reset_index(drop=True)

    model = timm.create_model(TIMM_NAMES[args.arch], pretrained=True, num_classes=0).eval().to(args.device)
    tf = T.Compose([T.Resize(size), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    dl = DataLoader(Imgs([str(ROOT / p) for p in df.crop_path], tf), batch_size=args.batch, num_workers=2)

    feats = []
    with torch.inference_mode():
        for i, x in enumerate(dl):
            feats.append(model(x.to(args.device)).float().cpu().numpy().astype(np.float16))
            if i % 50 == 0:
                print(f"{i * args.batch}/{len(df)}", flush=True)

    out = ROOT / "data/features"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{args.arch}.npy", np.concatenate(feats))
    df[META].to_parquet(out / f"{args.arch}_meta.parquet", index=False)
    print("saved", out / f"{args.arch}.npy", np.concatenate(feats).shape)


if __name__ == "__main__":
    main()
