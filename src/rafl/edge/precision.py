"""Per-burst precision switching for the detector on the board. numpy, PIL and tflite_runtime only.

Precision levels of the detector (measured on the board, report section 9.3):
    L0  int8 per-tensor on the NPU    32 ms per frame   (default)
    L1  int8 per-channel on the NPU   36 ms per frame
    L2  FP32 on the CPU, 4 threads   133 ms per frame
The classifier always runs int8 on the NPU: after QAT its int8 accuracy equals FP32 (0.884 / 0.885), and
it is the model federated learning updates, so it must stay the deployed int8 file.

Every burst runs at L0 first. The burst is re-run at a higher level when its L0 result is uncertain:
  - "near": the best dangerous-class vote lies within `margin` below the alert threshold, or
  - "weak": no box reached the detection threshold, but some box reached `weak_thr`.
The level is chosen from the camera's state: L2 if battery >= soc_l2, temperature < temp_l2 and CPU load
< cpu_busy; otherwise L1 if battery >= soc_l1; otherwise no escalation. Below soc_low, only the first
`low_frames` frames of a burst are processed. The escalated result replaces the L0 decision.

  python3 -m rafl.edge.precision --stage ~/rafl/stage_frames --out ~/rafl/runs/precision_<ts> \
      --l0 det_i8pt.tflite --l1 det_i8pc.tflite --l2 det_fp32.tflite --classifier cls.tflite
runs every burst at every level, then replays the switching rule on the log for several battery states.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from rafl.edge.node_io import DirectoryBurstSource, pick_alert_crop, write_alert
from rafl.edge.run_frames import CROP, DANGEROUS, MEAN, STD, Detector, dequant, square

LEVELS = ("L0", "L1", "L2")


def interpreter(model, delegate, cache):
    from tflite_runtime.interpreter import Interpreter, load_delegate
    if delegate:
        Path(cache).mkdir(parents=True, exist_ok=True)
        os.environ["VIV_VX_ENABLE_CACHE_GRAPH_BINARY"] = "1"
        os.environ["VIV_VX_CACHE_BINARY_GRAPH_DIR"] = str(cache)
        it = Interpreter(str(model), experimental_delegates=[load_delegate(delegate)])
    else:
        it = Interpreter(str(model), num_threads=4)
    it.allocate_tensors()
    return it


class Classifier:
    """int8 classifier; works with the plain deployed file and with the FL file that has extra outputs."""

    def __init__(self, it):
        self.it = it
        self.i = it.get_input_details()[0]
        self.o = next(o for o in it.get_output_details() if len(o["shape"]) == 2 and o["shape"][-1] < 16)

    def prepare(self, im, box):
        W, H = im.size
        x0, y0, x1, y1 = box
        c = im.crop(square(x0, y0, x1 - x0, y1 - y0, W, H)).resize((CROP, CROP), Image.BICUBIC)
        x = (np.asarray(c, np.float32) / 255.0 - MEAN) / STD
        si, zi = self.i["quantization"]
        return np.clip(np.round(x / si + zi), -128, 127).astype(np.int8)[None]

    def invoke(self, x):
        self.it.set_tensor(self.i["index"], x)
        self.it.invoke()
        y = dequant(self.it.get_tensor(self.o["index"])[0], self.o)
        p = np.exp(y - y.max())
        return p / p.sum()


@dataclass
class Policy:
    det_thr: float = 0.25
    weak_thr: float = 0.10
    threshold: float = 0.6
    margin: float = 0.25
    soc_l2: float = 0.5
    soc_l1: float = 0.3
    soc_low: float = 0.25
    low_frames: int = 2
    temp_l2: float = 75.0
    cpu_busy: float = 0.7

    def trigger(self, r):
        if r["danger_vote"] < self.threshold and r["danger_vote"] >= self.threshold - self.margin:
            return "near"
        if r["boxes"] == 0 and r["max_weak"] >= self.weak_thr:
            return "weak"
        return None

    def level(self, state):
        if state["battery_soc"] >= self.soc_l2 and state["temp_c"] < self.temp_l2 and state["cpu_load"] < self.cpu_busy:
            return "L2"
        if state["battery_soc"] >= self.soc_l1:
            return "L1"
        return None

    def frames(self, state, n):
        return min(n, self.low_frames) if state["battery_soc"] < self.soc_low else n


class Pipeline:
    def __init__(self, classes, animal, detectors: dict, classifier: Classifier, policy: Policy, max_crops=5,
                 lock=None, npu_levels=("L0", "L1")):
        self.classes, self.policy, self.max_crops = classes, policy, max_crops
        self.det = {k: Detector(it, 416, animal, policy.weak_thr) for k, it in detectors.items()}
        self.cls = classifier
        self.lock = lock or nullcontext()
        self.npu_levels = set(npu_levels)
        self.swap_lock = threading.Lock()

    def set_classifier(self, classifier: Classifier):
        """Hot swap, used when federated learning deploys a new int8 model."""
        with self.swap_lock:
            self.cls = classifier

    def run(self, level, images):
        """images: list of (name, PIL image). Returns the burst result at one precision level."""
        n = len(self.classes)
        acc, wsum, boxes, max_weak = np.zeros(n, np.float32), 0.0, 0, 0.0
        cands, det_ms, cls_ms = [], 0.0, 0.0
        det = self.det[level]
        with self.swap_lock:
            cls = self.cls
        t0 = time.perf_counter()
        for name, im in images:
            W, H = im.size
            x, s = det.preprocess(im)
            ta = time.perf_counter()
            with (self.lock if level in self.npu_levels else nullcontext()):
                raw = det.invoke(x)
            det_ms += 1000 * (time.perf_counter() - ta)
            xyxy, conf = det.postprocess(raw, s, W, H)
            if len(conf):
                max_weak = max(max_weak, float(conf.max()))
            keep = [(b, c) for b, c in zip(xyxy, conf) if c >= self.policy.det_thr][: self.max_crops]
            boxes += len(keep)
            for bx, cf in keep:
                xin = cls.prepare(im, bx)
                tb = time.perf_counter()
                with self.lock:
                    p = cls.invoke(xin)
                cls_ms += 1000 * (time.perf_counter() - tb)
                acc += float(cf) * p
                wsum += float(cf)
                cands.append({"frame": name, "box": [float(v) for v in bx], "det_conf": float(cf), "probs": p})
        vote = acc / wsum if wsum > 0 else np.zeros(n)
        pred = self.classes[int(vote.argmax())] if wsum > 0 else "empty"
        vconf = float(vote.max()) if wsum > 0 else 0.0
        dk = [i for i, c in enumerate(self.classes) if c in DANGEROUS]
        danger_vote = float(vote[dk].max()) if wsum > 0 else 0.0
        return {"level": level, "pred": pred, "conf": round(vconf, 4), "danger_vote": round(danger_vote, 4),
                "alert": pred in DANGEROUS and vconf >= self.policy.threshold, "boxes": boxes,
                "max_weak": round(max_weak, 4), "frames": len(images), "det_ms": round(det_ms, 1),
                "cls_ms": round(cls_ms, 1), "ms": round(1000 * (time.perf_counter() - t0), 1), "cands": cands}

    def burst(self, images, state):
        """L0 first; escalate when uncertain and the camera's state allows. Returns (final, l0, escalated)."""
        images = images[: self.policy.frames(state, len(images))]
        r0 = self.run("L0", images)
        why = self.policy.trigger(r0)
        lvl = self.policy.level(state) if why else None
        if lvl is None:
            return r0, r0, {"trigger": why, "to": None}
        r1 = self.run(lvl, images)
        return r1, r0, {"trigger": why, "to": lvl}


def load_burst(b):
    return [(fr.name, fr.load()) for fr in b.frames]


def strip(r):
    return {k: v for k, v in r.items() if k != "cands"}


def replay(log, policy: Policy, state):
    """Decisions and cost of the switching rule for a fixed camera state, from a log of all levels."""
    out = {"alerts_dangerous": 0, "false_alerts": 0, "escalated": 0, "ms": 0.0, "by_level": {}}
    for row in log:
        r0 = row["levels"]["L0"]
        why = policy.trigger(r0)
        lvl = policy.level(state) if why else None
        final = row["levels"][lvl] if lvl else r0
        cost = r0["ms"] + (row["levels"][lvl]["ms"] if lvl else 0.0)
        out["ms"] += cost
        out["escalated"] += bool(lvl)
        out["by_level"][lvl or "L0"] = out["by_level"].get(lvl or "L0", 0) + 1
        if final["alert"] and row["truth"] in DANGEROUS:
            out["alerts_dangerous"] += 1
        if final["alert"] and row["truth"] not in DANGEROUS:
            out["false_alerts"] += 1
    out["ms_per_burst"] = round(out["ms"] / len(log), 1)
    out["ms"] = round(out["ms"], 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--l0", required=True)
    ap.add_argument("--l1", required=True)
    ap.add_argument("--l2", required=True)
    ap.add_argument("--classifier", required=True)
    ap.add_argument("--delegate", default="/usr/lib/libvx_delegate.so")
    ap.add_argument("--cache-dir", default="~/rafl/vxcache_precision")
    ap.add_argument("--alerts", action="store_true", help="also write alert thumbnails for the L0 decisions")
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir).expanduser()
    src = DirectoryBurstSource(args.stage)
    t0 = time.perf_counter()
    dets = {"L0": interpreter(args.l0, args.delegate, cache / "l0"), "L1": interpreter(args.l1, args.delegate, cache / "l1"),
            "L2": interpreter(args.l2, None, None)}
    cls = Classifier(interpreter(args.classifier, args.delegate, cache / "cls"))
    pol = Policy()
    pipe = Pipeline(src.classes, src.animal_index, dets, cls, pol)
    first = [("warm", src.first_image())]
    for lvl in LEVELS:
        pipe.run(lvl, first)
    load_s = time.perf_counter() - t0
    log = []
    with open(out / "levels.jsonl", "w") as f:
        for b in src.bursts():
            imgs = load_burst(b)
            row = {"burst": b.id, "truth": b.truth, "levels": {lvl: strip(pipe.run(lvl, imgs)) for lvl in LEVELS}}
            row["trigger"] = pol.trigger(row["levels"]["L0"])
            if args.alerts and row["levels"]["L0"]["alert"]:
                r = pipe.run("L0", imgs)
                best = pick_alert_crop(r["cands"], src.classes.index(r["pred"]))
                if best:
                    write_alert(out / "alerts", b.id, r["pred"], r["conf"], best["frame"], best["box"], dict(imgs)[best["frame"]])
            log.append(row)
            f.write(json.dumps(row) + "\n")
            lv = row["levels"]
            print(f"{b.id} truth={b.truth:<9} " + "  ".join(f"{k}:{lv[k]['pred']:<9}{lv[k]['conf']:.2f}{'!' if lv[k]['alert'] else ' '}"
                                                          f"{lv[k]['ms']:6.0f}ms" for k in LEVELS) + f"  trigger={row['trigger']}", flush=True)
    fixed = {}
    for lvl in LEVELS:
        rs = [row["levels"][lvl] for row in log]
        fixed[lvl] = {"alerts_dangerous": sum(r["alert"] for r, row in zip(rs, log) if row["truth"] in DANGEROUS),
                      "false_alerts": sum(r["alert"] for r, row in zip(rs, log) if row["truth"] not in DANGEROUS),
                      "ms_per_burst": round(float(np.mean([r["ms"] for r in rs])), 1)}
    states = {"battery 0.80": {"battery_soc": 0.8, "temp_c": 40.0, "cpu_load": 0.1},
              "battery 0.40": {"battery_soc": 0.4, "temp_c": 40.0, "cpu_load": 0.1},
              "battery 0.20": {"battery_soc": 0.2, "temp_c": 40.0, "cpu_load": 0.1},
              "battery 0.80, CPU busy": {"battery_soc": 0.8, "temp_c": 40.0, "cpu_load": 0.9}}
    summary = {"bursts": len(log), "dangerous_bursts": sum(r["truth"] in DANGEROUS for r in log),
               "load_s": round(load_s, 1), "fixed_levels": fixed,
               "switching": {k: replay(log, pol, s) for k, s in states.items()},
               "triggers": {k: sum(r["trigger"] == k for r in log) for k in ("near", "weak")}, "policy": pol.__dict__}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
