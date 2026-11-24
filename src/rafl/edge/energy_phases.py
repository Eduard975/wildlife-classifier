"""Phase 7 energy workload for the board: steady phases of one operation each, read with a USB power meter.

Each phase repeats one operation for --phase-s seconds and counts how often it ran. A gap
with nothing running separates phases so the meter settles. Energy per operation is then
  (P_phase - P_idle) * duration / count   (incremental)   and   P_phase * duration / count   (whole board).

Console output goes to stderr. stdout is used only by the wifi_tx phase, which streams random
bytes over the SSH session to the Mac, so the caller must send stdout to /dev/null.

  python3 energy_phases.py --out ~/rafl/runs/phase7_<ts> --delegate <libvx_delegate.so> \
      --detector models/det_ld/int8_per_tensor.tflite --classifier models/efficientnet_lite0_int8_per_tensor.tflite
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(Path.home() / "rafl/pylib")]

from run_frames import Classifier, Detector, make_interpreter  # noqa: E402
from node_io import DirectoryBurstSource  # noqa: E402
from telemetry import Recorder  # noqa: E402

PHASES = ["idle", "npu_classifier", "npu_detector", "npu_pipeline", "cpu_pipeline",
          "fl_features", "fl_train", "wifi_tx"]


def say(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def stamp(t: float) -> str:
    return datetime.fromtimestamp(t).strftime("%H:%M:%S")


class Pipeline:
    """Frame -> detector -> crops -> classifier, as in run_frames.py, without the burst vote."""

    def __init__(self, det: Detector, cls: Classifier, frames, max_crops=5):
        self.det, self.cls, self.max_crops = det, cls, max_crops
        self.frames = itertools.cycle(frames)
        self.crops = 0

    def __call__(self):
        im = next(self.frames).load()
        W, H = im.size
        x, s = self.det.preprocess(im)
        xyxy, _ = self.det.postprocess(self.det.invoke(x), s, W, H)
        for bx in xyxy[: self.max_crops]:
            self.cls.invoke(self.cls.prepare(im, bx))
            self.crops += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="~/rafl/stage_frames")
    ap.add_argument("--detector", required=True)
    ap.add_argument("--classifier", required=True)
    ap.add_argument("--delegate", required=True)
    ap.add_argument("--cache-dir", default="~/rafl/vxcache_phase7")
    ap.add_argument("--fl-dir", default="~/rafl/fl")
    ap.add_argument("--phases", default=",".join(PHASES))
    ap.add_argument("--phase-s", type=float, default=90.0)
    ap.add_argument("--gap-s", type=float, default=15.0)
    ap.add_argument("--hz", type=float, default=2.0)
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    phases = [p for p in args.phases.split(",") if p]
    unknown = set(phases) - set(PHASES)
    if unknown:
        raise SystemExit(f"unknown phases: {sorted(unknown)}")
    cache = Path(args.cache_dir).expanduser()
    fl = Path(args.fl_dir).expanduser()

    say("setup: loading models and compiling NPU graphs (not measured)")
    src = DirectoryBurstSource(str(Path(args.stage).expanduser()))
    frames = [f for b in src.bursts() for f in b.frames]
    ops, units = {}, {}

    def npu_models():
        dit = make_interpreter(args.detector, args.delegate, cache / "det")
        dit.allocate_tensors()
        cit = make_interpreter(args.classifier, args.delegate, cache / "cls")
        cit.allocate_tensors()
        return Detector(dit, 416, src.animal_index, 0.25), Classifier(cit, src.classes)

    if {"npu_classifier", "npu_detector", "npu_pipeline"} & set(phases):
        det, cls = npu_models()
        im0 = frames[0].load()
        xd, _ = det.preprocess(im0)
        xc = cls.prepare(im0, (0.3, 0.3, 0.6, 0.6))
        for _ in range(3):
            det.invoke(xd)
            cls.invoke(xc)
        ops["npu_classifier"], units["npu_classifier"] = (lambda: cls.invoke(xc)), "inference"
        ops["npu_detector"], units["npu_detector"] = (lambda: det.invoke(xd)), "inference"
        npu_pipe = Pipeline(det, cls, frames)
        ops["npu_pipeline"], units["npu_pipeline"] = npu_pipe, "frame"
    if "cpu_pipeline" in phases:
        dc = make_interpreter(args.detector, None, None)
        dc.allocate_tensors()
        cc = make_interpreter(args.classifier, None, None)
        cc.allocate_tensors()
        cpu_pipe = Pipeline(Detector(dc, 416, src.animal_index, 0.25), Classifier(cc, src.classes), frames)
        cpu_pipe()
        ops["cpu_pipeline"], units["cpu_pipeline"] = cpu_pipe, "frame"
    if "fl_features" in phases:
        from PIL import Image
        from run_frames import MEAN, STD
        from tflite_runtime.interpreter import Interpreter
        man = json.load(open(fl / "data/manifest.json"))
        files = itertools.cycle([fl / "data/crops" / r["file"] for r in man["crops"]])
        bb = Interpreter(str(fl / "backbone_fp32.tflite"), num_threads=4)
        bb.allocate_tensors()
        bi, bo = bb.get_input_details()[0], bb.get_output_details()[0]

        def feature():
            im = Image.open(next(files)).convert("RGB")
            if im.size != (224, 224):
                im = im.resize((224, 224), Image.BILINEAR)
            bb.set_tensor(bi["index"], (((np.asarray(im, np.float32) / 255.0 - MEAN) / STD)[None]).astype(np.float32))
            bb.invoke()
            return bb.get_tensor(bo["index"])
        feature()
        ops["fl_features"], units["fl_features"] = feature, "crop"
    if "fl_train" in phases:
        from rafl.fl.real import standardize
        from rafl.fl.sim import init_head, sgd
        man = json.load(open(fl / "data/manifest.json"))
        st = np.load(fl / "data/stats.npz")
        x = standardize(np.load(fl / "board_features.npy"), st["mu"], st["sd"])
        y = np.array([r["label"] for r in man["crops"]], np.int64)
        k = len(man["classes"])
        rng = np.random.default_rng(0)
        head = init_head(x.shape[1], k, rng)
        ops["fl_train"] = lambda: sgd(head, x, y, k, 1, 0.01, 0.9, 32, rng)
        units["fl_train"] = f"round (1 epoch, {len(y)} samples)"
    if "wifi_tx" in phases:
        buf = os.urandom(64 * 1024)
        stream = sys.stdout.buffer

        def send():
            stream.write(buf)
            stream.flush()
        ops["wifi_tx"], units["wifi_tx"] = send, "64 KiB chunk"
    ops["idle"], units["idle"] = None, "second"

    rec = Recorder(out / "telemetry.jsonl", args.hz)
    rec.start()
    results = []
    say(f"\n{len(phases)} phases of {args.phase_s:.0f} s with {args.gap_s:.0f} s gaps, "
        f"about {len(phases) * (args.phase_s + args.gap_s) / 60:.1f} min. Read the meter during each phase.")
    try:
        for i, name in enumerate(phases, 1):
            rec.phase = "gap"
            time.sleep(args.gap_s)
            op = ops[name]
            rec.phase = name
            t0 = time.time()
            say(f"\n=== [{i}/{len(phases)}] {name:<15} START {stamp(t0)}  ({args.phase_s:.0f} s) ===")
            n, p0, next_note = 0, time.perf_counter(), 30.0
            while (el := time.perf_counter() - p0) < args.phase_s:
                if op is None:
                    time.sleep(min(1.0, args.phase_s - el))
                else:
                    op()
                    n += 1
                if el >= next_note:
                    say(f"    {name}: {el:4.0f} s, {n} x {units[name]}")
                    next_note += 30.0
            dur = time.perf_counter() - p0
            t1 = time.time()
            r = {"phase": name, "start": t0, "end": t1, "start_hms": stamp(t0), "end_hms": stamp(t1),
                 "duration_s": round(dur, 3), "count": n if op is not None else round(dur, 3), "unit": units[name],
                 "ms_per_op": round(1000 * dur / n, 3) if op is not None and n else None}
            if name in ("npu_pipeline", "cpu_pipeline"):
                r["crops"] = ops[name].crops
                ops[name].crops = 0
            if name == "wifi_tx":
                r["bytes"] = n * len(buf)
                r["throughput_kbps"] = round(n * len(buf) * 8 / dur / 1000, 1)
            results.append(r)
            say(f"=== [{i}/{len(phases)}] {name:<15} END   {stamp(t1)}  count={r['count']} "
                f"{('ms/op=' + str(r['ms_per_op'])) if r['ms_per_op'] else ''} ===")
        rec.phase = "gap"
        time.sleep(args.gap_s)
    finally:
        rec.stop()
        json.dump({"phases": results, "phase_s": args.phase_s, "gap_s": args.gap_s,
                   "detector": args.detector, "classifier": args.classifier}, open(out / "phases.json", "w"), indent=1)
    say(f"\ndone: {out}/phases.json, {out}/telemetry.jsonl")


if __name__ == "__main__":
    main()
