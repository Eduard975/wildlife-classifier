"""Quantization-aware fine-tuning for per-tensor int8 export.

BatchNorm is folded into the preceding conv, every conv weight is fake-quantized
per tensor (symmetric int8) with a straight-through estimator, and activations are
fake-quantized (asymmetric int8, EMA min/max) after each conv/activation and each
block output. The trained folded weights are exported to ONNX as plain FP32 convs
and then go through the existing onnx2tf -> per-tensor int8 TFLite path.

Usage: python scripts/qat_classifier.py --arch efficientnet_lite0 [--epochs 8]
Writes runs/qat/<arch>/{best.pt,metrics.json,model.onnx,calib.npy}.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image

from rafl.models.classifier import build_classifier
from train_classifier import Crops, report, transforms, predict
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]


class WeightFQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        s = w.detach().abs().max().clamp_min(1e-8) / 127.0
        return torch.clamp(torch.round(w / s), -127, 127) * s

    @staticmethod
    def backward(ctx, g):
        return g


class ActFQ(nn.Module):
    def __init__(self, momentum=0.01):
        super().__init__()
        self.register_buffer("lo", torch.zeros(()))
        self.register_buffer("hi", torch.zeros(()))
        self.register_buffer("init", torch.zeros((), dtype=torch.bool))
        self.m = momentum
        self.enabled = True
        self.observe = True

    def forward(self, x):
        if not self.enabled:
            return x
        if self.training and self.observe:
            lo, hi = x.detach().min(), x.detach().max()
            if not self.init:
                self.lo.copy_(lo), self.hi.copy_(hi), self.init.fill_(True)
            else:
                self.lo.lerp_(lo, self.m), self.hi.lerp_(hi, self.m)
        if not self.init:
            return x
        lo, hi = self.lo.clamp(max=0), self.hi.clamp(min=0)
        s = ((hi - lo) / 255.0).clamp_min(1e-8)
        z = torch.round(-lo / s)
        q = torch.clamp(torch.round(x / s) + z, 0, 255)
        return ((q - z) * s - x).detach() + x


class QConv(nn.Module):
    def __init__(self, conv: nn.Conv2d, bn: nn.Module):
        super().__init__()
        std = (bn.running_var + bn.eps).sqrt()
        g = bn.weight / std
        self.stride, self.padding, self.dilation, self.groups = conv.stride, conv.padding, conv.dilation, conv.groups
        self.weight = nn.Parameter(conv.weight.detach() * g.view(-1, 1, 1, 1))
        b = conv.bias.detach() if conv.bias is not None else 0.0
        self.bias = nn.Parameter((b - bn.running_mean) * g + bn.bias.detach())
        act = getattr(bn, "act", None)
        self.act = act if isinstance(act, nn.Module) else nn.Identity()
        self.fq = ActFQ()
        self.quant_weights = True

    def forward(self, x):
        w = WeightFQ.apply(self.weight) if self.quant_weights else self.weight
        y = nn.functional.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return self.fq(self.act(y))


def fold(model: nn.Module):
    for name, mod in list(model.named_children()):
        kids = list(mod.named_children())
        for (n1, c), (n2, b) in zip(kids, kids[1:]):
            if isinstance(c, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
                setattr(mod, n1, QConv(c, b))
                setattr(mod, n2, nn.Identity())
        fold(mod)
    top = list(model.named_children())
    for (n1, c), (n2, b) in zip(top, top[1:]):
        if isinstance(c, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
            setattr(model, n1, QConv(c, b))
            setattr(model, n2, nn.Identity())
    return model


def add_block_fq(model: nn.Module):
    for seq in model.blocks:
        for blk in seq:
            blk.out_fq = ActFQ()
            blk.register_forward_hook(lambda m, i, o: m.out_fq(o))


def build_qat(arch, ncls, ckpt, device):
    m = build_classifier(arch, ncls, pretrained=False)
    m.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"])
    m.eval()
    fold(m)
    add_block_fq(m)
    return m.to(device)


def set_quant(model, weights=True, acts=True):
    for mod in model.modules():
        if isinstance(mod, QConv):
            mod.quant_weights = weights
        if isinstance(mod, ActFQ):
            mod.enabled = acts


def bn_free_check(model):
    return not any(isinstance(m, nn.BatchNorm2d) for m in model.modules())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="efficientnet_lite0")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--freeze_obs_epoch", type=int, default=-3, help="stop updating act ranges this many epochs before the end")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="runs/qat")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))
    classes = cfg["models"]["classifier"]["classes"]
    size = cfg["models"]["classifier"]["input"]
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    df = pd.read_parquet(ROOT / cfg["paths"]["manifest"])
    df = df[df.ok & df.label.isin(classes)]
    splits = {s: df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test")}
    splits["test"] = splits["test"][splits["test"].verified != "n"].reset_index(drop=True)
    if args.limit:
        splits = {s: d.sample(min(args.limit, len(d)), random_state=0).reset_index(drop=True) for s, d in splits.items()}
    out = ROOT / args.out / args.arch
    out.mkdir(parents=True, exist_ok=True)
    tf_train, tf_eval = transforms(size)
    dl = lambda d, tf, shuf: DataLoader(Crops(d, classes, tf), batch_size=args.batch, shuffle=shuf,
                                        num_workers=args.workers, persistent_workers=args.workers > 0,
                                        drop_last=shuf, pin_memory=device == "cuda")
    train_dl, val_dl, test_dl = dl(splits["train"], tf_train, True), dl(splits["val"], tf_eval, False), dl(splits["test"], tf_eval, False)

    counts = np.bincount(train_dl.dataset.y, minlength=len(classes))
    weights = torch.tensor((counts.max() / np.maximum(counts, 1)) ** 0.5, dtype=torch.float32)
    model = build_qat(args.arch, len(classes), ROOT / f"runs/phase3/{args.arch}/best.pt", device)
    assert bn_free_check(model)
    print(f"device={device} train={len(splits['train'])} test={len(splits['test'])}", flush=True)

    set_quant(model, weights=False, acts=False)
    fp_val = report(val_dl.dataset.y, predict(model, val_dl, device), classes)["balanced_accuracy"]
    set_quant(model, True, True)
    model.train()
    for x, _ in train_dl:
        with torch.no_grad():
            model(x.to(device))
        break
    q_val = report(val_dl.dataset.y, predict(model, val_dl, device), classes)["balanced_accuracy"]
    print(f"folded fp32 val_bacc={fp_val:.4f}  fake-quant before QAT val_bacc={q_val:.4f}", flush=True)

    crit = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    steps = args.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    stop_obs = args.epochs + args.freeze_obs_epoch if args.freeze_obs_epoch < 0 else args.freeze_obs_epoch

    best, history = -1.0, [{"epoch": -1, "fp32_folded_val_bacc": fp_val, "fakequant_init_val_bacc": q_val}]
    for ep in range(args.epochs):
        model.train()
        if ep >= stop_obs:
            for mod in model.modules():
                if isinstance(mod, ActFQ):
                    mod.observe = False
        t0, tot, n = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
            n += len(y)
        val = report(val_dl.dataset.y, predict(model, val_dl, device), classes)
        history.append({"epoch": ep, "loss": tot / n, "val_bacc": val["balanced_accuracy"], "secs": time.time() - t0})
        print(json.dumps(history[-1]), flush=True)
        if val["balanced_accuracy"] > best:
            best = val["balanced_accuracy"]
            torch.save({"arch": args.arch, "classes": classes, "state_dict": model.state_dict(), "epoch": ep, "val": val}, out / "best.pt")

    model.load_state_dict(torch.load(out / "best.pt", map_location=device)["state_dict"])
    test = splits["test"]
    p = predict(model, test_dl, device)
    y, v = test_dl.dataset.y, test.verified.to_numpy()
    res = {"history": history, "test_fakequant": report(y, p, classes), "verified_recall": {}}
    for c in ("wolf", "bear"):
        m = (test.label.to_numpy() == c) & (v == "y")
        res["verified_recall"][c] = float((p[m].argmax(1) == classes.index(c)).mean())
    json.dump(res, open(out / "metrics.json", "w"), indent=2)
    print("TEST fake-quant bacc=%.4f verified=%s" % (res["test_fakequant"]["balanced_accuracy"], res["verified_recall"]), flush=True)

    set_quant(model, True, False)
    for mod in model.modules():
        if isinstance(mod, QConv):
            mod.weight.data = WeightFQ.apply(mod.weight.data)
    set_quant(model, False, False)
    model.eval().cpu()
    torch.onnx.export(model, torch.zeros(1, 3, size, size), out / "model.onnx", input_names=["input"],
                      output_names=["logits"], opset_version=17, dynamo=False)
    tr = df[df.split == "train"]
    cal = tr.groupby("label", group_keys=False).apply(lambda g: g.sample(cfg["quant"]["calibration_crops"] // len(classes), random_state=cfg["seed"]))
    arr = np.stack([np.asarray(Image.open(ROOT / p_).convert("RGB").resize((size, size)), np.float32) / 255.0 for p_ in cal.crop_path])
    np.save(out / "calib.npy", arr)
    print("exported", out / "model.onnx", arr.shape, flush=True)


if __name__ == "__main__":
    main()
