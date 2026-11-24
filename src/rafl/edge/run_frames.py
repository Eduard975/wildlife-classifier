"""Board runtime: full camera frame -> int8 detector -> square crops -> int8 classifier -> burst vote -> alert.

Needs only numpy, PIL and tflite_runtime. Reads <stage>/frames.json written by
scripts/stage_board_frames.py; writes events.jsonl and summary.json to <out>.

  python3 run_frames.py --stage ~/rafl/stage_frames --detector det.tflite --classifier cls.tflite \
      --delegate <libvx_delegate.so> --cache-dir ~/rafl/vxcache_frames_<ts> --out ~/rafl/runs/<name>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from rafl.edge.node_io import DirectoryBurstSource, V4L2Source, pick_alert_crop, write_alert
except ImportError:
    from node_io import DirectoryBurstSource, V4L2Source, pick_alert_crop, write_alert

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
DANGEROUS = ("bear", "wolf", "lynx", "wild_boar")
PAD, CROP = 0.15, 224


def temps_c() -> list[float]:
    out = []
    for p in sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp")):
        try:
            out.append(int(open(p).read()) / 1000)
        except OSError:
            pass
    return out


def load1() -> float:
    return float(open("/proc/loadavg").read().split()[0])


def make_interpreter(model, delegate, cache_dir):
    from tflite_runtime.interpreter import Interpreter, load_delegate
    if delegate:
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["VIV_VX_ENABLE_CACHE_GRAPH_BINARY"] = "1"
        os.environ["VIV_VX_CACHE_BINARY_GRAPH_DIR"] = str(cache_dir)
        return Interpreter(str(model), experimental_delegates=[load_delegate(delegate)])
    return Interpreter(str(model), num_threads=4)


def letterbox(im, size):
    s = size / max(im.size)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(im, (0, 0))
    return np.asarray(canvas, np.float32) / 255.0, s


def square(x, y, w, h, W, H):
    side = min(max(w * W, h * H) * (1 + 2 * PAD), W, H)
    cx, cy = (x + w / 2) * W, (y + h / 2) * H
    left = min(max(cx - side / 2, 0), W - side)
    top = min(max(cy - side / 2, 0), H - side)
    return int(left), int(top), int(left + side), int(top + side)


def iou_one(a, b):
    tl, br = np.maximum(a[:2], b[:, :2]), np.minimum(a[2:], b[:, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), 1)
    return inter / (np.prod(a[2:] - a[:2]) + np.prod(b[:, 2:] - b[:, :2], 1) - inter)


def nms(xyxy, conf, iou=0.45, top=20):
    order, keep = np.argsort(-conf), []
    while len(order) and len(keep) < top:
        i, order = order[0], order[1:]
        keep.append(i)
        order = order[iou_one(xyxy[i], xyxy[order]) < iou]
    return keep


def dequant(t, d):
    y = t.astype(np.float32)
    if d["dtype"] == np.int8 or d["dtype"] == np.uint8:
        s, z = d["quantization"]
        y = (y - z) * s
    return y


class Detector:
    def __init__(self, it, size, animal, thr):
        self.it, self.size, self.animal, self.thr = it, size, animal, thr
        self.i = it.get_input_details()[0]
        self.outs = it.get_output_details()

    def preprocess(self, im):
        x, s = letterbox(im, self.size)
        if self.i["dtype"] == np.int8:
            sc, z = self.i["quantization"]
            x = np.clip(np.round(x / sc + z), -128, 127).astype(np.int8)
        return x[None], s

    def invoke(self, x):
        self.it.set_tensor(self.i["index"], x)
        self.it.invoke()
        return [self.it.get_tensor(o["index"])[0].copy() for o in self.outs]

    def postprocess(self, raw, s, W, H):
        outs = {}
        for o, y in zip(self.outs, raw):
            y = dequant(y, o)
            n = y.shape[0] if y.shape[0] < y.shape[-1] else y.shape[-1]
            outs[n] = y if y.shape[0] < y.shape[-1] else y.T
        boxes = outs.pop(4)
        conf = next(iter(outs.values()))[self.animal]
        m = conf >= self.thr
        if not m.any():
            return np.zeros((0, 4)), np.zeros(0)
        cx, cy, w, h = boxes[:, m] * self.size / s / np.array([[W], [H], [W], [H]])
        xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        k = nms(xyxy, conf[m])
        return xyxy[k], conf[m][k]


class Classifier:
    def __init__(self, it, classes):
        self.it, self.classes = it, classes
        self.i, self.o = it.get_input_details()[0], it.get_output_details()[0]

    def prepare(self, im, box):
        W, H = im.size
        x0, y0, x1, y1 = box
        c = im.crop(square(x0, y0, x1 - x0, y1 - y0, W, H)).resize((CROP, CROP), Image.BICUBIC)
        x = ((np.asarray(c, np.float32) / 255.0 - MEAN) / STD)
        si, zi = self.i["quantization"]
        return np.clip(np.round(x / si + zi), -128, 127).astype(np.int8)[None]

    def invoke(self, x):
        self.it.set_tensor(self.i["index"], x)
        self.it.invoke()
        y = dequant(self.it.get_tensor(self.o["index"])[0], self.o)
        p = np.exp(y - y.max())
        return p / p.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=None)
    ap.add_argument("--source", choices=["dir", "v4l2"], default="dir")
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--burst-frames", type=int, default=5)
    ap.add_argument("--n-bursts", type=int, default=1)
    ap.add_argument("--alerts-dir", default=None)
    ap.add_argument("--detector", required=True)
    ap.add_argument("--classifier", required=True)
    ap.add_argument("--delegate", default=None)
    ap.add_argument("--cache-dir", default="~/rafl/vxcache_frames")
    ap.add_argument("--size", type=int, default=416)
    ap.add_argument("--det-thr", type=float, default=0.25)
    ap.add_argument("--max-crops", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    alerts_dir = Path(args.alerts_dir).expanduser() if args.alerts_dir else out / "alerts"
    if args.source == "dir":
        source = DirectoryBurstSource(args.stage)
    else:
        if not V4L2Source.available(args.device):
            raise SystemExit(f"no camera at {args.device}")
        ref = DirectoryBurstSource(args.stage)
        source = V4L2Source(args.device, ref.classes, ref.animal_index, args.burst_frames, args.n_bursts)
    classes = source.classes
    backend = "npu" if args.delegate else "cpu"
    cache = Path(args.cache_dir).expanduser()

    t0 = time.perf_counter()
    dit = make_interpreter(args.detector, args.delegate, cache / "det")
    dit.allocate_tensors()
    cit = make_interpreter(args.classifier, args.delegate, cache / "cls")
    cit.allocate_tensors()
    det = Detector(dit, args.size, source.animal_index, args.det_thr)
    cls = Classifier(cit, classes)
    load_ms = 1000 * (time.perf_counter() - t0)

    first = source.first_image()
    t0 = time.perf_counter()
    for _ in range(3):
        det.invoke(det.preprocess(first)[0])
        cls.invoke(cls.prepare(first, (0.3, 0.3, 0.6, 0.6)))
    warm_ms = 1000 * (time.perf_counter() - t0)

    log = open(out / "events.jsonl", "w")
    start = time.time()
    rows, frames = [], []
    alerts = []
    for b in source.bursts():
        acc, wsum, n_boxes = np.zeros(len(classes), np.float32), 0.0, 0
        cands, imgs = [], {}
        for fr in b.frames:
            f = fr.name
            tf0 = time.perf_counter()
            im = fr.load()
            W, H = im.size
            x, s = det.preprocess(im)
            t1 = time.perf_counter()
            raw = det.invoke(x)
            t2 = time.perf_counter()
            xyxy, conf = det.postprocess(raw, s, W, H)
            t3 = time.perf_counter()
            crops, cls_ms, prep_ms = [], 0.0, 0.0
            for bx, cf in list(zip(xyxy, conf))[: args.max_crops]:
                ta = time.perf_counter()
                xin = cls.prepare(im, bx)
                tb = time.perf_counter()
                p = cls.invoke(xin)
                tc = time.perf_counter()
                prep_ms += 1000 * (tb - ta)
                cls_ms += 1000 * (tc - tb)
                acc += float(cf) * p
                cands.append({"frame": f, "box": [float(v) for v in bx], "det_conf": float(cf), "probs": p})
                imgs[f] = im
                wsum += float(cf)
                crops.append({"box": [round(float(v), 3) for v in bx], "det_conf": round(float(cf), 3),
                              "pred": classes[int(p.argmax())], "conf": round(float(p.max()), 3),
                              "cls_ms": round(1000 * (tc - tb), 2)})
            n_boxes += len(xyxy)
            total = 1000 * (time.perf_counter() - tf0)
            pre_ms = 1000 * (t1 - tf0)
            det_ms = 1000 * (t2 - t1)
            post_ms = 1000 * (t3 - t2)
            ev = {"type": "frame", "t": round(time.time() - start, 3), "burst": b.id, "frame": f,
                  "boxes": int(len(xyxy)), "crops": crops, "det_ms": round(det_ms, 2),
                  "cls_ms": round(cls_ms, 2), "pre_ms": round(pre_ms, 2),
                  "crop_prep_ms": round(prep_ms, 2), "post_ms": round(post_ms, 2),
                  "total_ms": round(total, 2), "temp_c": temps_c(), "load1": load1()}
            frames.append(ev)
            log.write(json.dumps(ev) + "\n")
            log.flush()
            top = max(crops, key=lambda c: c["det_conf"]) if crops else None
            print(f"[{ev['t']:7.2f}s] {b.id} {Path(f).name:<14} boxes={ev['boxes']} "
                  f"{(top['pred'] + ' ' + format(top['conf'], '.2f')) if top else '-':<16} "
                  f"det {det_ms:6.1f} cls {cls_ms:5.1f} total {total:6.1f} ms "
                  f"{ev['temp_c'][0] if ev['temp_c'] else 0:.0f}C", flush=True)
        if wsum > 0:
            vote = acc / wsum
            pred, vconf = classes[int(vote.argmax())], float(vote.max())
        else:
            pred, vconf = "empty", 0.0
        alert = pred in DANGEROUS and vconf >= args.threshold
        row = {"type": "burst", "t": round(time.time() - start, 3), "burst": b.id, "truth": b.truth,
               "pred": pred, "conf": round(vconf, 3), "alert": alert, "detector_fired": n_boxes > 0,
               "correct": pred == b.truth, "frames": len(b.frames), "boxes": n_boxes}
        if alert:
            best = pick_alert_crop(cands, classes.index(pred))
            if best:
                rec = write_alert(alerts_dir, b.id, pred, vconf, best["frame"], best["box"], imgs[best["frame"]],
                                  extra={"truth": b.truth, "frames_in_burst": len(b.frames)})
                row["alert_file"] = rec["thumbnail"]
                alerts.append(rec)
        rows.append(row)
        log.write(json.dumps(row) + "\n")
        print(f"{'ALERT' if alert else '  -  '} burst {b.id} truth={b.truth} vote={pred} {vconf:.2f}",
              flush=True)

    def stat(key, sel=None):
        v = [f[key] for f in frames] if sel is None else sel
        return {"mean": round(float(np.mean(v)), 2), "p95": round(float(np.percentile(v, 95)), 2)}

    ms_total = [f["total_ms"] for f in frames]
    danger = [r for r in rows if r["truth"] in DANGEROUS]
    labelled = [r for r in rows if r["truth"] in classes]
    summary = {
        "backend": backend, "detector": Path(args.detector).parent.name + "/" + Path(args.detector).name,
        "classifier": Path(args.classifier).name, "load_ms": round(load_ms, 1), "warmup_ms": round(warm_ms, 1),
        "frames": len(frames), "bursts": len(rows), "det_thr": args.det_thr, "threshold": args.threshold,
        "max_crops": args.max_crops,
        "ms_per_frame": {"total": stat("total_ms"), "decode_letterbox": stat("pre_ms"),
                         "detector": stat("det_ms"), "det_post": stat("post_ms"),
                         "crop_prep": stat("crop_prep_ms"), "classifier": stat("cls_ms")},
        "fps": round(1000 / float(np.mean(ms_total)), 2),
        "burst_accuracy_5class": round(float(np.mean([r["correct"] for r in labelled])), 3),
        "dangerous_bursts": len(danger), "dangerous_alerted": sum(r["alert"] for r in danger),
        "other_bursts": sum(r["truth"] == "other" for r in rows),
        "false_alerts_on_other": sum(r["alert"] for r in rows if r["truth"] == "other"),
        "empty_bursts": sum(r["truth"] == "empty" for r in rows),
        "false_alerts_on_empty": sum(r["alert"] for r in rows if r["truth"] == "empty"),
        "detector_fired_on_other": sum(r["detector_fired"] for r in rows if r["truth"] == "other"),
        "detector_fired_on_empty": sum(r["detector_fired"] for r in rows if r["truth"] == "empty"),
        "wall_s": round(time.time() - start, 2), "alerts_written": len(alerts),
    }
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
