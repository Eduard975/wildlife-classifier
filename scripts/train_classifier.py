"""Phase 3: central FP32 baseline crop classifier (the accuracy ceiling).

Trains on the location-held-out train split, selects the epoch by val balanced
accuracy, and reports on test:
  - test_all:     every test crop except those hand-marked 'n'
  - wolf/bear headline recall on hand-verified 'y' crops only
  - hard:         hand-marked '?' crops
  - burst vote:   mean softmax per seq_key (sequence, or image when unknown)

Usage: python scripts/train_classifier.py --arch efficientnet_lite0 [--epochs 12]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from rafl.models.classifier import IMAGENET_MEAN, IMAGENET_STD, build_classifier

ROOT = Path(__file__).resolve().parents[1]


class Crops(Dataset):
    def __init__(self, df: pd.DataFrame, classes: list[str], tf):
        self.paths = [str(ROOT / p) for p in df.crop_path]
        idx = {c: i for i, c in enumerate(classes)}
        self.y = np.array([idx[c] for c in df.label], dtype=np.int64)
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), self.y[i]


def transforms(size: int):
    norm = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    train = T.Compose([
        T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3, 0.2),
        T.RandomGrayscale(0.25),  # night IR frames are grayscale
        *norm,
    ])
    return train, T.Compose([T.Resize(size), *norm])


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in loader:
        out.append(model(x.to(device)).float().softmax(1).cpu().numpy())
    return np.concatenate(out)


def report(y: np.ndarray, p: np.ndarray, classes: list[str]) -> dict:
    pred = p.argmax(1)
    k = len(classes)
    cm = np.bincount(y * k + pred, minlength=k * k).reshape(k, k)
    per = {}
    for i, c in enumerate(classes):
        tp, n_true, n_pred = cm[i, i], cm[i].sum(), cm[:, i].sum()
        r = tp / n_true if n_true else float("nan")
        pr = tp / n_pred if n_pred else float("nan")
        f1 = 2 * pr * r / (pr + r) if (pr + r) else 0.0
        per[c] = {"n": int(n_true), "recall": float(r), "precision": float(pr), "f1": float(f1)}
    recalls = [v["recall"] for v in per.values() if v["n"]]
    return {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean([v["f1"] for v in per.values() if v["n"]])),
        "per_class": per,
        "confusion": cm.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: rows per split")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))
    classes = cfg["models"]["classifier"]["classes"]
    size = cfg["models"]["classifier"]["input"]
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_parquet(ROOT / cfg["paths"]["manifest"])
    df = df[df.ok & df.label.isin(classes)]
    splits = {s: df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test")}
    splits["test"] = splits["test"][splits["test"].verified != "n"].reset_index(drop=True)
    if args.limit:
        splits = {s: d.sample(min(args.limit, len(d)), random_state=0).reset_index(drop=True)
                  for s, d in splits.items()}

    out = ROOT / "runs/phase3" / args.arch
    out.mkdir(parents=True, exist_ok=True)
    tf_train, tf_eval = transforms(size)
    dl = lambda d, tf, shuf: DataLoader(Crops(d, classes, tf), batch_size=args.batch, shuffle=shuf,
                                        num_workers=args.workers, persistent_workers=args.workers > 0,
                                        drop_last=shuf)
    train_dl = dl(splits["train"], tf_train, True)
    val_dl = dl(splits["val"], tf_eval, False)
    test_dl = dl(splits["test"], tf_eval, False)

    counts = np.bincount(train_dl.dataset.y, minlength=len(classes))
    weights = torch.tensor((counts.max() / np.maximum(counts, 1)) ** 0.5, dtype=torch.float32)
    print(f"device={device} train={len(splits['train'])} val={len(splits['val'])} "
          f"test={len(splits['test'])} class_counts={counts.tolist()} weights={weights.numpy().round(2).tolist()}",
          flush=True)

    model = build_classifier(args.arch, len(classes)).to(device)
    crit = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs * len(train_dl)
    warm = len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))

    best, history = -1.0, []
    for ep in range(args.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
            n += len(y)
        val = report(val_dl.dataset.y, predict(model, val_dl, device), classes)
        history.append({"epoch": ep, "loss": tot / n, "val_bacc": val["balanced_accuracy"],
                        "val_macro_f1": val["macro_f1"], "secs": time.time() - t0})
        print(json.dumps(history[-1]), flush=True)
        if val["balanced_accuracy"] > best:
            best = val["balanced_accuracy"]
            torch.save({"arch": args.arch, "classes": classes, "state_dict": model.state_dict(),
                        "epoch": ep, "val": val}, out / "best.pt")

    ck = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ck["state_dict"])
    test = splits["test"]
    p = predict(model, test_dl, device)
    y = test_dl.dataset.y
    v = test.verified.to_numpy()

    res = {"arch": args.arch, "best_epoch": ck["epoch"], "val": ck["val"], "history": history,
           "test_all": report(y, p, classes)}
    # headline: wolf/bear recall on hand-verified crops only
    res["verified_recall"] = {}
    for c in ("wolf", "bear"):
        m = (test.label.to_numpy() == c) & (v == "y")
        ci = classes.index(c)
        r = float((p[m].argmax(1) == ci).mean()) if m.any() else float("nan")
        res["verified_recall"][c] = {"n": int(m.sum()), "recall": r,
                                     "ci95": 1.96 * math.sqrt(r * (1 - r) / max(1, m.sum()))}
    hard = v == "?"
    res["hard"] = {"n": int(hard.sum()), "correct": int((p[hard].argmax(1) == y[hard]).sum())}
    # burst voting: mean softmax over crops sharing a seq_key
    g = pd.DataFrame(p).groupby(test.seq_key.to_numpy())
    vp = g.mean()
    vy = pd.Series(y).groupby(test.seq_key.to_numpy()).first().loc[vp.index].to_numpy()
    res["test_burst_vote"] = report(vy, vp.to_numpy(), classes)

    json.dump(res, open(out / "metrics.json", "w"), indent=2)
    pd.DataFrame({"crop_id": test.crop_id, "label": test.label, "verified": v,
                  **{f"p_{c}": p[:, i] for i, c in enumerate(classes)}}).to_parquet(out / "test_probs.parquet")
    t = res["test_all"]["per_class"]
    print("\nTEST (per crop)  bacc=%.3f macroF1=%.3f" % (res["test_all"]["balanced_accuracy"], res["test_all"]["macro_f1"]))
    for c in classes:
        print(f"  {c:10s} n={t[c]['n']:5d} recall={t[c]['recall']:.3f} precision={t[c]['precision']:.3f}")
    for c, r in res["verified_recall"].items():
        print(f"  verified {c}: recall={r['recall']:.3f} ±{r['ci95']:.3f} (n={r['n']})")
    print("  burst-vote bacc=%.3f (n=%d bursts)" % (res["test_burst_vote"]["balanced_accuracy"], res["test_burst_vote"]["n"]))


if __name__ == "__main__":
    main()
