"""Board runtime POC: classify staged camera bursts on the NPU, vote per burst, alert.

Needs only numpy, PIL and tflite_runtime (all on the board image). Reads
<stage>/bursts.json written by scripts/stage_board_demo.py and writes one JSON line
per frame and per burst to <out>/events.jsonl, plus <out>/summary.json.

  python3 run_bursts.py --stage ~/rafl/stage --model ~/rafl/models/<m>.tflite \
      --delegate <path to libvx_delegate.so> --out ~/rafl/runs/<name>
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

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
DANGEROUS = ("bear", "wolf", "lynx", "wild_boar")


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


def make_interpreter(model: str, delegate: str | None, cache_dir: Path):
    from tflite_runtime.interpreter import Interpreter, load_delegate
    if delegate:
        os.environ["VIV_VX_ENABLE_CACHE_GRAPH_BINARY"] = "1"
        os.environ["VIV_VX_CACHE_BINARY_GRAPH_DIR"] = str(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return Interpreter(model, experimental_delegates=[load_delegate(delegate)])
    return Interpreter(model, num_threads=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--delegate", default=None, help="VX delegate path; omit for CPU")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    stage, out = Path(args.stage).expanduser(), Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(stage / "bursts.json"))
    classes = meta["classes"]
    backend = "npu" if args.delegate else "cpu"

    t0 = time.perf_counter()
    it = make_interpreter(args.model, args.delegate, Path("~/rafl/vxcache_demo").expanduser())
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    si, zi = i["quantization"]
    so, zo = o["quantization"]
    load_ms = 1000 * (time.perf_counter() - t0)

    log = open(out / "events.jsonl", "w")
    start = time.time()
    lat, rows = [], []
    for b in meta["bursts"]:
        acc = np.zeros(len(classes), np.float32)
        for f in b["frames"]:
            img = np.asarray(Image.open(stage / f).convert("RGB"), np.float32) / 255.0
            x = np.clip(np.round(((img - MEAN) / STD) / si + zi), -128, 127).astype(np.int8)[None]
            t = time.perf_counter()
            it.set_tensor(i["index"], x)
            it.invoke()
            ms = 1000 * (time.perf_counter() - t)
            y = (it.get_tensor(o["index"])[0].astype(np.float32) - zo) * so
            p = np.exp(y - y.max())
            p /= p.sum()
            acc += p
            lat.append(ms)
            ev = {"type": "frame", "t": round(time.time() - start, 3), "burst": b["id"], "frame": f,
                  "pred": classes[int(p.argmax())], "conf": round(float(p.max()), 3),
                  "infer_ms": round(ms, 2), "temp_c": temps_c(), "load1": load1()}
            log.write(json.dumps(ev) + "\n")
            log.flush()
            print(f"[{ev['t']:7.2f}s] {b['id']:<4} {Path(f).name:<22} {ev['pred']:<9} {ev['conf']:.2f} "
                  f"{ms:6.1f} ms  {ev['temp_c'][0] if ev['temp_c'] else 0:.0f}C", flush=True)
        vote = acc / len(b["frames"])
        pred = classes[int(vote.argmax())]
        alert = pred in DANGEROUS and float(vote.max()) >= args.threshold
        row = {"type": "burst", "t": round(time.time() - start, 3), "burst": b["id"], "truth": b["truth"],
               "pred": pred, "conf": round(float(vote.max()), 3), "alert": alert,
               "correct": pred == b["truth"], "frames": len(b["frames"])}
        rows.append(row)
        log.write(json.dumps(row) + "\n")
        print(f"{'ALERT' if alert else '  -  '} burst {b['id']} truth={b['truth']} vote={pred} {vote.max():.2f}",
              flush=True)

    danger = [r for r in rows if r["truth"] in DANGEROUS]
    steady = lat[1:] or lat
    summary = {
        "backend": backend, "model": Path(args.model).name, "load_ms": round(load_ms, 1),
        "first_infer_ms": round(lat[0], 1), "mean_infer_ms": round(float(np.mean(steady)), 2),
        "p95_infer_ms": round(float(np.percentile(steady, 95)), 2),
        "bursts": len(rows), "frames": len(lat),
        "burst_accuracy": round(float(np.mean([r["correct"] for r in rows])), 3),
        "dangerous_bursts": len(danger), "dangerous_alerted": sum(r["alert"] for r in danger),
        "false_alerts_on_other": sum(r["alert"] for r in rows if r["truth"] == "other"),
        "wall_s": round(time.time() - start, 2), "threshold": args.threshold,
    }
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
