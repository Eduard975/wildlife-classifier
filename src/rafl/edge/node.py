"""The camera node: detection with per-burst precision switching and federated learning in one program.

Runs standalone on the board (numpy, PIL, tflite_runtime, flwr). Two threads share one NPU (a lock
serialises every NPU call) and one battery model:
  - detection: bursts from a frame source go through the int8 detector and classifier with the
    switching rule of rafl.edge.precision; alerts are written as thumbnails;
  - federated learning: rafl.fl.board_block_client in the main thread. After each round's
    re-quantized model is loaded on the NPU, the detection thread switches to it before its next burst.
Battery: the FL client's model (one FL round = one hour); escalations to the FP32 CPU detector are
charged at the assumed CPU power for their measured time.

  PYTHONPATH=~/rafl:~/rafl/pylib python3 -m rafl.edge.node --server 127.0.0.1:<port> --stage ~/rafl/stage_frames \
      --l0 L0.tflite --l1 L1.tflite --l2 L2.tflite --out ~/rafl/runs/node_<ts> [board_block_client options]
  add --no-fl to run detection only.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

from rafl.edge.node_io import DirectoryBurstSource, pick_alert_crop, write_alert
from rafl.edge.precision import Classifier, Pipeline, Policy, interpreter, load_burst, strip
from rafl.edge.run_frames import DANGEROUS
from rafl.edge.telemetry import Telemetry
from rafl.fl import board_block_client as fl


class Detection(threading.Thread):
    def __init__(self, args, pipe: Pipeline, source, battery, log, stop: threading.Event):
        super().__init__(daemon=True)
        self.args, self.pipe, self.source, self.battery, self.log, self.stop_ev = args, pipe, source, battery, log, stop
        self.tel = Telemetry(iface=args.iface)
        self.model = "deployed"
        self.done = threading.Event()
        self.stats = {"bursts": 0, "escalated": 0, "alerts": 0, "dangerous": 0, "dangerous_alerted": 0, "false_alerts": 0}

    def swap(self, it, rnd, digest):
        self.pipe.set_classifier(Classifier(it))
        self.model = digest
        self.log.write(json.dumps({"type": "model_swap", "t": round(time.time(), 3), "round": rnd, "digest": digest}) + "\n")

    def state(self):
        s = self.tel.sample()
        return {"battery_soc": self.battery.soc, "cpu_load": s["cpu_util"], "temp_c": max(s["temp_c"].values(), default=0.0)}

    def run(self):
        try:
            for loop in range(self.args.loops):
                for b in self.source.bursts():
                    if self.stop_ev.is_set():
                        return
                    st = self.state()
                    imgs = load_burst(b)
                    t0 = time.perf_counter()
                    final, r0, esc = self.pipe.burst(imgs, st)
                    wall_ms = 1000 * (time.perf_counter() - t0)
                    if esc["to"] == "L2":
                        self.battery.spend(final["ms"] / 1000, 0.0)  # CPU FP32 run, at the assumed CPU power
                    row = {"type": "burst", "t": round(time.time(), 3), "loop": loop, "burst": b.id, "truth": b.truth,
                           "model": self.model, "state": {k: round(v, 3) for k, v in st.items()}, "trigger": esc["trigger"],
                           "escalated_to": esc["to"], "final": strip(final), "l0": strip(r0) if esc["to"] else None,
                           "wall_ms": round(wall_ms, 1)}
                    if final["alert"]:
                        best = pick_alert_crop(final["cands"], self.pipe.classes.index(final["pred"]))
                        if best:
                            rec = write_alert(Path(self.args.out).expanduser() / "alerts", f"{b.id}_l{loop}", final["pred"],
                                              final["conf"], best["frame"], best["box"], dict(imgs)[best["frame"]],
                                              extra={"truth": b.truth, "level": final["level"], "model": self.model})
                            row["alert_file"] = rec["thumbnail"]
                    self.log.write(json.dumps(row) + "\n")
                    s = self.stats
                    s["bursts"] += 1
                    s["escalated"] += bool(esc["to"])
                    s["alerts"] += final["alert"]
                    s["dangerous"] += b.truth in DANGEROUS
                    s["dangerous_alerted"] += final["alert"] and b.truth in DANGEROUS
                    s["false_alerts"] += final["alert"] and b.truth not in DANGEROUS
                    tag = "ALERT" if final["alert"] else "  -  "
                    print(f"[det] {tag} {b.id} truth={b.truth:<9} {final['pred']:<9} {final['conf']:.2f} level={final['level']}"
                          f"{' (' + esc['trigger'] + ')' if esc['trigger'] else ''} {wall_ms:6.0f} ms  battery {st['battery_soc']:.2f}"
                          f"  cpu {st['cpu_load']:.2f}  model {self.model}", flush=True)
                    self.stop_ev.wait(self.args.interval_s)
        finally:
            self.done.set()


def main():
    ap = argparse.ArgumentParser()
    fl.add_args(ap)
    ap.set_defaults(server="")
    for a in ap._actions:
        if a.dest == "server":
            a.required = False
    ap.add_argument("--no-fl", action="store_true")
    ap.add_argument("--stage", required=True)
    ap.add_argument("--l0", required=True)
    ap.add_argument("--l1", required=True)
    ap.add_argument("--l2", required=True)
    ap.add_argument("--classifier", default="~/rafl/flblock/classifier_block.tflite")
    ap.add_argument("--cache-dir", default="~/rafl/vxcache_node")
    ap.add_argument("--interval-s", type=float, default=1.0, help="pause between bursts")
    ap.add_argument("--loops", type=int, default=1000, help="passes over the staged bursts")
    ap.add_argument("--after-fl-s", type=float, default=30.0, help="keep detecting this long after FL ends")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.load = json.loads(args.load)
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir).expanduser()
    npu_lock = threading.Lock()

    src = DirectoryBurstSource(args.stage)
    with npu_lock:
        dets = {"L0": interpreter(Path(args.l0).expanduser(), args.delegate, cache / "l0"),
                "L1": interpreter(Path(args.l1).expanduser(), args.delegate, cache / "l1"),
                "L2": interpreter(Path(args.l2).expanduser(), None, None)}
        cls = Classifier(interpreter(Path(args.classifier).expanduser(), args.delegate, cache / "cls"))
    pipe = Pipeline(src.classes, src.animal_index, dets, cls, Policy(), lock=npu_lock)
    for lvl in ("L0", "L1", "L2"):
        pipe.run(lvl, [("warm", src.first_image())])

    log = open(out / "events.jsonl", "w", buffering=1)
    stop = threading.Event()
    client = None
    if args.no_fl:
        battery = fl.Battery(args.battery_soc, args.battery_wh, args.start_hour, args.solar_factor, args.seed)
    else:
        fl.patch_grpc_timeouts()
        client = fl.BoardBlockClient(args, npu_lock=npu_lock)
        battery = client.battery
    det = Detection(args, pipe, src, battery, log, stop)
    if client:
        client.on_deploy = det.swap
    det.start()
    try:
        if client:
            from flwr.client import start_client
            start_client(server_address=args.server, client=client.to_client(), insecure=True)
            log.write(json.dumps({"type": "fl_done", "t": round(time.time(), 3)}) + "\n")
            det.done.wait(args.after_fl_s)
        else:
            det.done.wait()
    finally:
        stop.set()
        det.join(timeout=30)
        if client:
            client.close()
        json.dump(det.stats, open(out / "detection_summary.json", "w"), indent=1)
        print("[node] detection", json.dumps(det.stats), flush=True)


if __name__ == "__main__":
    main()
