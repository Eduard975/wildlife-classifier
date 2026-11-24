"""Quantization-aware distillation of the MegaDetector v6 detector for per-tensor int8.

The student is the fused FP32 detector with per-tensor symmetric int8 fake-quant on every
conv weight and asymmetric int8 fake-quant (min/max EMA) on conv outputs, SiLU outputs,
concat/upsample/pool/residual outputs, the input and both output tensors. The FP32 detector
is the teacher: the loss is soft-label BCE on class probabilities plus score-weighted L1 on
normalized boxes over all anchors, so no labels are needed. Trained weights are snapped to
the int8 grid and saved as a plain state_dict for export_detector.py --state.

Usage:
  .venv/bin/python scripts/qat_detector.py prep --size 416
  .venv/bin/python scripts/qat_detector.py train --size 416 --epochs 20     (CUDA machine)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
GREY = 114.0 / 255.0


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
        self.accumulate = False

    def forward(self, x):
        if not self.enabled:
            return x
        if self.observe:
            lo, hi = x.detach().min(), x.detach().max()
            if not self.init:
                self.lo.copy_(lo), self.hi.copy_(hi), self.init.fill_(True)
            elif self.accumulate:
                self.lo.copy_(torch.minimum(self.lo, lo)), self.hi.copy_(torch.maximum(self.hi, hi))
            else:
                self.lo.lerp_(lo, self.m), self.hi.lerp_(hi, self.m)
        if not self.init:
            return x
        lo, hi = self.lo.clamp(max=0), self.hi.clamp(min=0)
        s = ((hi - lo) / 255.0).clamp_min(1e-8)
        z = torch.round(-lo / s)
        q = torch.clamp(torch.round(x / s) + z, 0, 255)
        return ((q - z) * s - x).detach() + x


class QConv2d(nn.Conv2d):
    def forward(self, x):
        w = WeightFQ.apply(self.weight) if self.quant_w else self.weight
        return self.fq(self._conv_forward(x, w, self.bias))


def build_model(weights: str, imgsz: int):
    from export_detector import SplitHead
    from ultralytics import YOLO

    yolo = YOLO(str(ROOT / weights))
    m = yolo.model.fuse().eval()
    head = m.model[-1]
    head.export, head.format = True, "onnx"
    if hasattr(head, "end2end"):
        head.end2end = False
    return SplitHead(m, imgsz).eval()


def to_qat(net: nn.Module):
    from ultralytics.nn.modules import Concat, Conv
    from ultralytics.nn.modules.block import CIB, PSABlock, Bottleneck, Attention, SPPF

    net.in_fq, net.box_fq, net.score_fq = ActFQ(), ActFQ(), ActFQ()
    for m in list(net.modules()):
        if isinstance(m, nn.Conv2d):
            m.__class__ = QConv2d
            m.fq, m.quant_w = ActFQ(), True
        if isinstance(m, Conv) and not isinstance(m.act, nn.Identity):
            m.act = nn.Sequential(nn.SiLU(), ActFQ())
    head = net.model.model[-1]
    for branch in (head.cv2, head.cv3):
        for seq in branch[1:]:
            seq[-1].fq = branch[0][-1].fq
    hooked = (Concat, nn.Upsample, nn.MaxPool2d, CIB, PSABlock, Bottleneck, Attention, SPPF)
    for m in list(net.modules()):
        if isinstance(m, hooked):
            m.out_fq = ActFQ()
            m.register_forward_hook(lambda mod, i, o: mod.out_fq(o))
    return net


def fq_forward(net, x):
    boxes, scores = net(net.in_fq(x))
    return net.box_fq(boxes), net.score_fq(scores)


def set_quant(net, weights=True, acts=True):
    for m in net.modules():
        if isinstance(m, QConv2d):
            m.quant_w = weights
        if isinstance(m, ActFQ):
            m.enabled = acts


def plain_state(net):
    net.eval()
    sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    for name, m in net.named_modules():
        if isinstance(m, QConv2d):
            sd[f"{name}.weight"] = WeightFQ.apply(m.weight.detach()).cpu()
    return {k.removeprefix("model."): v for k, v in sd.items() if ".fq." not in k and "_fq." not in k and not k.endswith((".lo", ".hi", ".init"))
            and ".act.1." not in k}


def letterbox_u8(path: Path, size: int):
    from PIL import Image

    im = Image.open(path).convert("RGB")
    s = size / max(im.size)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(im, (0, 0))
    return np.asarray(canvas, np.uint8)


def prep(args):
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor

    fr = pd.read_parquet(ROOT / "data/manifests/train_frames.parquet")
    paths = [ROOT / "data/frames" / p for p in fr.frame_path.sample(frac=1.0, random_state=0)]
    from PIL import Image

    out = ROOT / "data/frames" / f"train{args.size}"
    out.mkdir(exist_ok=True)

    def one(i_p):
        i, p = i_p
        Image.fromarray(letterbox_u8(p, args.size)).save(out / f"{i:05d}.jpg", quality=92)

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, enumerate(paths)))
    print(out, len(paths), "empty:", int((~fr.has_box).sum()))


def load_cache(size):
    from PIL import Image

    files = sorted((ROOT / "data/frames" / f"train{size}").glob("[0-9]*.jpg"))
    return np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files])


def augment(x):
    n = x.shape[0]
    dev = x.device
    s = torch.empty(n, device=dev).uniform_(0.6, 1.25)
    s = torch.where(torch.rand(n, device=dev) < 0.4, torch.ones_like(s), s)
    flip = (torch.rand(n, device=dev) < 0.5).float() * 2 - 1
    t = (1 - s).abs() * torch.empty(n, 2, device=dev).uniform_(-1, 1).T
    theta = torch.zeros(n, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 1, 1] = s * flip, s
    theta[:, 0, 2], theta[:, 1, 2] = t[0], t[1]
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x - GREY, grid, mode="bilinear", padding_mode="zeros", align_corners=False) + GREY
    a = torch.empty(n, 1, 1, 1, device=dev).uniform_(0.7, 1.3)
    b = torch.empty(n, 1, 1, 1, device=dev).uniform_(-0.1, 0.1)
    x = (x - 0.5) * a + 0.5 + b
    gray = (torch.rand(n, 1, 1, 1, device=dev) < 0.15).float()
    x = gray * x.mean(1, keepdim=True).expand_as(x) + (1 - gray) * x
    return x.clamp(0, 1)


def distill_loss(student, teacher, box_w=10.0):
    sb, ss = student
    tb, ts = teacher
    cls = F.binary_cross_entropy(ss.clamp(1e-6, 1 - 1e-6), ts, reduction="sum") / ts.sum().clamp_min(1.0)
    w = ts.max(1, keepdim=True)[0]
    box = ((sb - tb).abs() * w).sum() / w.sum().clamp_min(1.0) / 4
    return cls + box_w * box, cls.item(), box.item()


def agreement(student, teacher, thr=0.25):
    sb, ss = student
    tb, ts = teacher
    t_max, s_max = ts.max(1)[0], ss.max(1)[0]
    pos = t_max >= thr
    return ((s_max >= thr) & pos).sum().item() / max(pos.sum().item(), 1), ((s_max >= thr) & ~pos).sum().item() / max((~pos).sum().item(), 1)


def train(args):
    torch.backends.cudnn.benchmark = True
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)
    data = torch.from_numpy(load_cache(args.size)).to(dev).permute(0, 3, 1, 2).contiguous()
    n_hold = args.holdout
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(data), generator=g)
    hold, tr = perm[:n_hold].to(dev), perm[n_hold:].to(dev)
    teacher = build_model(args.weights, args.size).to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = to_qat(build_model(args.weights, args.size)).to(dev).eval()
    for n_, p in student.named_parameters():
        p.requires_grad_(".dfl." not in n_)
    batch = lambda idx, aug: (augment(data[idx].float() / 255.0) if aug else data[idx].float() / 255.0)

    def evaluate():
        tot, ag, fp, n = 0.0, 0.0, 0.0, 0
        fqs = [m for m in student.modules() if isinstance(m, ActFQ)]
        obs = [m.observe for m in fqs]
        for m in fqs:
            m.observe = False
        with torch.no_grad():
            for i in range(0, len(hold), 32):
                x = batch(hold[i:i + 32], False)
                t, s = teacher(x), fq_forward(student, x)
                tot += distill_loss(s, t)[0].item() * len(x)
                a, f = agreement(s, t)
                ag, fp, n = ag + a * len(x), fp + f * len(x), n + len(x)
        for m, o in zip(fqs, obs):
            m.observe = o
        return tot / n, ag / n, fp / n

    calib = torch.arange(min(args.calib_imgs, len(data)), device=dev)

    def calibrate(observe_after):
        fqs = [m for m in student.modules() if isinstance(m, ActFQ)]
        for m in fqs:
            m.init.fill_(False)
            m.observe, m.accumulate = True, True
        with torch.no_grad():
            for i in range(0, len(calib), 32):
                fq_forward(student, batch(calib[i:i + 32], False))
        for m in fqs:
            m.observe, m.accumulate = observe_after, False

    set_quant(student, False, False)
    base = evaluate()
    set_quant(student, True, True)
    calibrate(True)
    init = evaluate()
    print(f"holdout loss/teacher-recall/fp  fp32 {base}  fakequant-init {init}", flush=True)

    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps = args.epochs * (len(tr) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.05)
    out = ROOT / args.out / f"{Path(args.weights).stem}_{args.size}"
    out.mkdir(parents=True, exist_ok=True)
    hist = [{"epoch": -1, "fp32": base, "fakequant_init": init}]
    for ep in range(args.epochs):
        if ep == args.epochs - args.freeze_last:
            calibrate(False)
            print("recalibrated and froze activation ranges", evaluate(), flush=True)
        t0, tot, k = time.time(), 0.0, 0
        order = tr[torch.randperm(len(tr), device=dev)]
        for i in range(0, len(order) - args.batch + 1, args.batch):
            x = batch(order[i:i + args.batch], True)
            with torch.no_grad():
                t = teacher(x)
            loss, _, _ = distill_loss(fq_forward(student, x), t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            sched.step()
            tot, k = tot + loss.item(), k + 1
        val = evaluate()
        hist.append({"epoch": ep, "train_loss": tot / k, "hold_loss": val[0], "teacher_recall": val[1], "fp_rate": val[2],
                     "secs": round(time.time() - t0, 1)})
        print(json.dumps(hist[-1]), flush=True)
        torch.save(plain_state(student), out / "state.pt")
    json.dump(hist, open(out / "train_history.json", "w"), indent=1)
    print("saved", out / "state.pt", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "train"])
    ap.add_argument("--size", type=int, default=416)
    ap.add_argument("--weights", default="models/detector/MDV6-yolov10-c.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--holdout", type=int, default=200)
    ap.add_argument("--calib-imgs", type=int, default=512)
    ap.add_argument("--freeze-last", type=int, default=3)
    ap.add_argument("--out", default="runs/qat_detector")
    args = ap.parse_args()
    {"prep": prep, "train": train}[args.cmd](args)


if __name__ == "__main__":
    main()
