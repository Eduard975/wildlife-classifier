"""Board Flower client for resource-aware FL with dynamic quantization, running standalone on the board.

Each round the board, with no help from the server beyond the FP32 global model:
  1. re-quantizes the received block and head into its int8 classifier, calibrated on its own crops
     (rafl.fl.npu_block), loads the file on the NPU and evaluates it on held-out bursts;
  2. applies this round's link limit to its own uplink (board/police.sh, an egress policer on the
     connection to the server); the limit stands in for the deployment link and is not shown to the
     controller;
  3. observes its state: real CPU load and temperature, its own measurement of the previous upload's
     throughput (tx bytes on the interface after the update was handed to Flower), its measured
     training time per sample per tier, and a battery model that drains with measured times;
  4. lets the controller (rafl.fl.hetero.resource_aware) choose skip, or the tier (head or block) and
     the upload bit width (4 or 2), trains on the A53 cores and uploads a packed update with error
     feedback.

  python3 -m rafl.fl.board_block_client --server 127.0.0.1:<port> --root ~/rafl/flblock \
      --police-dst <server ip> --police-dport <port of the tunnel connection>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path.home() / "rafl/pylib"))

from flwr.client import NumPyClient  # noqa: E402

from rafl.edge import requant  # noqa: E402
from rafl.edge.telemetry import Telemetry  # noqa: E402
from rafl.fl import block, hetero, npu_block  # noqa: E402
from rafl.fl.quant import encode  # noqa: E402
from rafl.fl.resources import EnvConfig  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_rgb(f, size=224):
    im = Image.open(f).convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.BICUBIC)
    return np.asarray(im)


class Npu:
    """int8 classifier on the NPU; returns logits and the 7x7x320 block input."""

    def __init__(self, model: bytes, path: Path, delegate: str, cache: Path):
        from tflite_runtime.interpreter import Interpreter, load_delegate
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(model)
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["VIV_VX_ENABLE_CACHE_GRAPH_BINARY"] = "1"
        os.environ["VIV_VX_CACHE_BINARY_GRAPH_DIR"] = str(cache)
        t0 = time.perf_counter()
        self.it = Interpreter(str(path), experimental_delegates=[load_delegate(delegate)])
        self.it.allocate_tensors()
        self.i = self.it.get_input_details()[0]
        outs = self.it.get_output_details()
        self.o_logits = next(o["index"] for o in outs if o["shape"][-1] < 16 and len(o["shape"]) == 2)
        self.o_block = next(o["index"] for o in outs if list(o["shape"][1:]) == [7, 7, 320])
        self.run(np.zeros((224, 224, 3), np.uint8))  # graph compile happens on the first invoke
        self.load_s = time.perf_counter() - t0

    def run(self, rgb):
        x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        s, z = self.i["quantization"]
        self.it.set_tensor(self.i["index"], np.clip(np.round(x / s + z), -128, 127).astype(np.int8)[None])
        self.it.invoke()
        return self.it.get_tensor(self.o_logits).reshape(-1).copy(), self.it.get_tensor(self.o_block).reshape(49, 320).copy()


class Battery:
    """Board-side battery model. One FL round stands for one hour of camera operation. Power constants
    are the ASSUMED values of rafl.fl.resources.EnvConfig; the times they multiply are measured."""

    def __init__(self, soc, wh, start_hour, solar_factor, seed):
        self.c = EnvConfig(battery_wh=wh)
        self.soc, self.start_hour, self.solar_factor = soc, start_hour, solar_factor
        self.rng = np.random.default_rng(seed)

    def hour(self, rnd):
        return (self.start_hour + rnd) % 24

    def step_hour(self, rnd):
        h = self.hour(rnd)
        sun = max(0.0, np.sin(np.pi * (h - 6) / 12)) if 6 <= h <= 18 else 0.0
        solar = self.c.solar_peak_w * self.solar_factor * sun * self.rng.uniform(0.3, 1.0)
        self.soc = float(np.clip(self.soc + (solar - self.c.base_draw_w) / self.c.battery_wh, 0, 1))

    def spend(self, train_s, upload_s):
        e = self.c.wake_overhead_wh + (train_s * self.c.train_w + upload_s * self.c.radio_w) / 3600
        self.soc = float(max(0.0, self.soc - e / self.c.battery_wh))
        return e


class UploadMeter:
    """Measures the board's own upload: after the update is handed to Flower, the time until the
    interface's tx byte counter has grown by the transported size."""

    def __init__(self, iface):
        self.path = Path(f"/sys/class/net/{iface}/statistics/tx_bytes")
        self.last = None  # (bytes, seconds, kbit/s)
        self._t = None

    def _tx(self):
        return int(self.path.read_text())

    def start(self, nbytes, timeout=600.0):
        tx0, t0 = self._tx(), time.monotonic()

        def watch():
            while time.monotonic() - t0 < timeout:
                if self._tx() - tx0 >= nbytes:
                    dt = time.monotonic() - t0
                    self.last = (nbytes, dt, nbytes * 8 / dt / 1000)
                    return
                time.sleep(0.02)
        self._t = threading.Thread(target=watch, daemon=True)
        self._t.start()

    def wait(self, timeout=600.0):
        if self._t is not None:
            self._t.join(timeout)


PING_TIMEOUT_OPTS = [("grpc.keepalive_timeout_ms", 600000), ("grpc.http2.ping_timeout_ms", 600000)]


def patch_grpc_timeouts():
    """On a slow uplink an HTTP/2 ping reply waits behind the queued upload in the same TCP stream, and
    gRPC's default 20 s ping timeout closes a working connection. Give the client channel 10 minutes."""
    import grpc
    orig = grpc.insecure_channel

    def channel(target, options=None, compression=None):
        return orig(target, options=list(options or []) + PING_TIMEOUT_OPTS, compression=compression)
    grpc.insecure_channel = channel


class Police:
    def __init__(self, script, iface, dst, dport):
        self.script, self.iface, self.dst, self.dport = script, iface, dst, dport
        self.kbit = None

    def set(self, kbit):
        if self.dst and kbit and kbit != self.kbit:
            subprocess.run(["sh", self.script, "set", self.iface, self.dst, str(self.dport), str(int(kbit))], check=True)
            self.kbit = kbit

    def clear(self):
        if self.dst:
            subprocess.run(["sh", self.script, "clear", self.iface], check=False)
            self.kbit = None


class BoardBlockClient(NumPyClient):
    def __init__(self, args, npu_lock=None, on_deploy=None):
        self.args = args
        self.npu_lock = npu_lock or threading.Lock()
        self.on_deploy = on_deploy  # called with (interpreter, round, digest) after each new model is on the NPU
        root = self.root = Path(args.root).expanduser()
        self.tmpl = (root / "classifier_block.tflite").read_bytes()
        self.qm = requant.load_map(root / "quant_map.json")
        self.xp = npu_block.block_input_params(self.tmpl, self.qm)
        man = json.load(open(root / "data/manifest.json"))
        self.train_files = [root / "data/crops" / r["file"] for r in man["crops"]]
        k = len(man["classes"])
        y = np.array([r["label"] for r in man["crops"]], np.int64)
        rng = np.random.default_rng(args.seed)
        flip = rng.random(len(y)) < args.noise
        self.y = np.where(flip, rng.integers(0, k, len(y)), y)
        ev = json.load(open(root / "eval/bursts.json"))
        cls = ev["classes"]
        self.eval_items = [(root / "eval" / f, cls.index(b["truth"]), b["id"]) for b in ev["bursts"] for f in b["frames"]]
        self.rng = np.random.default_rng(args.seed + 1)
        self.tel = Telemetry(iface=args.iface)
        self.battery = Battery(args.battery_soc, args.battery_wh, args.start_hour, args.solar_factor, args.seed)
        self.meter = UploadMeter(args.iface)
        self.police = Police(str(Path.home() / "rafl/police.sh"), args.iface, args.police_dst, args.police_dport)
        self.sps = dict(hetero.S_PER_SAMPLE)  # replaced by the board's own measurements after the first round of each tier
        self.sps_measured = {}
        self.policy = hetero.resource_aware(margin=args.margin, s_per_sample=self.sps)
        self.residual = {}
        self._unbilled_upload = False
        self._sizes = None
        self.x_train = self.x_eval = None
        self.load_proc = []
        self.log = open(root / "rounds.jsonl", "a", buffering=1)

    def sizes(self, p):
        """Payload bytes per (tier, bits); depends only on the parameter shapes."""
        if self._sizes is None:
            self._sizes = {(t, b): hetero.payload_size(p, t, b, self.args.group) for t in block.TIERS for b in (2, 4)}
        return self._sizes

    def _run(self, npu, rgb):
        with self.npu_lock:
            return npu.run(rgb)

    # ---- background workload standing in for detection activity ----
    def start_load(self, rnd):
        kinds = self.args.load.get(str(rnd), [])
        for kind in kinds:
            cmd = [sys.executable, str(Path.home() / "rafl/energy_phases.py"), "--out", str(self.root / f"load_r{rnd}"),
                   "--delegate", self.args.delegate, "--detector", str(Path.home() / "rafl/models/det_ld/int8_per_tensor.tflite"),
                   "--classifier", str(Path.home() / "rafl/models/efficientnet_lite0_int8_per_tensor.tflite"),
                   "--phases", kind, "--phase-s", "900", "--gap-s", "0", "--hz", "1"]
            self.load_proc.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        if kinds:
            time.sleep(self.args.load_warmup_s)
        return ",".join(kinds) or "none"

    def stop_load(self):
        for p in self.load_proc:
            p.terminate()
        for p in self.load_proc:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        self.load_proc = []

    def deploy(self, p, rnd):
        """Re-quantize the global block and head on the board and run it on the NPU."""
        if self.x_train is None:  # block inputs do not depend on the federated layers: compute once
            with self.npu_lock:
                npu0 = Npu(self.tmpl, self.root / "models/template.tflite", self.args.delegate, self.root / "vxcache/template")
            t0 = time.perf_counter()
            self.x_train = np.array([self._run(npu0, load_rgb(f))[1] for f in self.train_files])
            self.block_ms = 1000 * (time.perf_counter() - t0) / len(self.train_files)
            self.x_eval = np.array([self._run(npu0, load_rgb(f))[1] for f, _, _ in self.eval_items])
            self.xf_train = npu_block.dequant(self.x_train, self.xp)
            self.xf_eval = npu_block.dequant(self.x_eval, self.xp)
            del npu0
        t0 = time.perf_counter()
        model = npu_block.deployable(p, self.xf_train, self.qm, self.tmpl)
        quant_s = time.perf_counter() - t0
        digest = requant.digest(model)
        with self.npu_lock:
            npu = Npu(model, self.root / f"models/classifier_r{rnd}.tflite", self.args.delegate, self.root / f"vxcache/r{rnd}_{digest}")
        t0 = time.perf_counter()
        logits = np.array([self._run(npu, load_rgb(f))[0] for f, _, _ in self.eval_items]).astype(np.float32)
        eval_ms = 1000 * (time.perf_counter() - t0) / len(self.eval_items)
        if self.on_deploy:
            self.on_deploy(npu.it, rnd, digest)
        y = np.array([t for _, t, _ in self.eval_items])
        pf = block.logits(p, self.xf_eval).argmax(1)
        votes, truth = {}, {}
        e = np.exp(logits - logits.max(1, keepdims=True))
        for (_, t, bid), pi in zip(self.eval_items, e / e.sum(1, keepdims=True)):
            votes[bid] = votes.get(bid, 0) + pi
            truth[bid] = t
        return {"model_digest": digest, "requant_s": round(quant_s, 3), "npu_load_s": round(npu.load_s, 2),
                "npu_eval_ms_per_crop": round(eval_ms, 2), "npu_crop_acc": float((logits.argmax(1) == y).mean()),
                "npu_burst_acc": float(np.mean([votes[k].argmax() == truth[k] for k in votes])),
                "npu_vs_float_agree": float((logits.argmax(1) == pf).mean()),
                "block_input_ms_per_crop": round(self.block_ms, 2)}

    def fit(self, parameters, config):
        rnd = int(config["round"])
        self.meter.wait(timeout=5)
        p = {k: np.asarray(v, np.float32) for k, v in zip(block.NAMES, parameters)}
        m = {"round": rnd, **self.deploy(p, rnd)}
        last = self.meter.last
        if last and self._unbilled_upload:
            self.battery.spend(0.0, last[1])  # radio energy of the previous upload, at its measured duration
            self._unbilled_upload = False
        m.update({"prev_upload_bytes": last[0] if last else 0, "prev_upload_s": round(last[1], 2) if last else 0.0,
                  "measured_kbps": round(last[2], 1) if last else 0.0})
        if config.get("deploy_only"):
            m.update({"tier": "deploy", "bits": 0, "skipped": 1})
            self.log.write(json.dumps(m) + "\n")
            return [np.zeros(0, np.uint8)], 0, m
        self.police.set(config.get("link_kbit"))
        load = self.start_load(rnd)
        try:
            self.battery.step_hour(rnd)
            self.tel.sample()
            time.sleep(1.0)
            obs = self.tel.observe(battery_soc=self.battery.soc)
            kbps = last[2] if last else self.args.uplink_prior_kbps
            s = {"battery_soc": self.battery.soc, "cpu_load": obs["cpu_load"], "temp_c": obs["temp_c"],
                 "uplink_kbps": kbps, "deadline_s": float(config["deadline_s"]), "n_samples": len(self.y)}
            sizes = self.sizes(p)
            plan = self.policy(rnd, "board", s, sizes)
            m.update({"link_kbit": float(config.get("link_kbit") or 0), "background_load": load, "hour": self.battery.hour(rnd),
                      "battery_soc": round(self.battery.soc, 4), "cpu_load": round(obs["cpu_load"], 3),
                      "temp_c": round(obs["temp_c"], 1), "uplink_kbps_used": round(kbps, 1),
                      "tier": plan.tier, "bits": plan.bits, "why": plan.why,
                      "s_per_sample_head": self.sps["head"], "s_per_sample_block": self.sps["block"]})
            if plan.tier == "skip":
                m["skipped"] = 1
                self.log.write(json.dumps(m) + "\n")
                return [np.zeros(0, np.uint8)], 0, m
            t0 = time.perf_counter()
            new = block.local_train(p, self.xf_train, self.y, plan.tier, plan.epochs, self.args.lr, self.args.momentum,
                                    self.args.batch, self.rng)
            train_s = time.perf_counter() - t0
            self.sps[plan.tier] = train_s / len(self.y)
            self.sps_measured[plan.tier] = self.sps[plan.tier]
            delta = hetero.split({k: new[k] - p[k] for k in block.TIERS[plan.tier]}, plan.tier, self.args.group)
            res = self.residual.get(plan.tier)
            if res is not None:
                delta = [d + r for d, r in zip(delta, res)]
            payload, deq = encode(delta, plan.bits, self.rng)
            self.residual[plan.tier] = [d - q for d, q in zip(delta, deq)]
            upload_s_est = len(payload) * 8 / (kbps * 1000)
            m.update({"skipped": 0, "train_s": round(train_s, 3), "payload_bytes": len(payload),
                      "upload_s_predicted": round(upload_s_est, 2), "battery_wh_train": self.battery.spend(train_s, 0.0)})
            self.log.write(json.dumps(m) + "\n")
            arr = np.frombuffer(payload, np.uint8).copy()
            self.meter.last = None
            self.meter.start(arr.nbytes + 128)  # numpy header added by Flower's serialization
            self._unbilled_upload = True
            return [arr], len(self.y), m
        finally:
            self.stop_load()

    def close(self):
        self.stop_load()
        self.police.clear()


def add_args(ap):
    ap.add_argument("--server", required=True)
    ap.add_argument("--root", default="~/rafl/flblock")
    ap.add_argument("--delegate", default="/usr/lib/libvx_delegate.so")
    ap.add_argument("--iface", default="mlan0")
    ap.add_argument("--police-dst", default="")
    ap.add_argument("--police-dport", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--margin", type=float, default=0.6)
    ap.add_argument("--uplink-prior-kbps", type=float, default=8.0, help="assumed before the first measured upload")
    ap.add_argument("--battery-soc", type=float, default=0.72)
    ap.add_argument("--battery-wh", type=float, default=12.0)
    ap.add_argument("--start-hour", type=int, default=17)
    ap.add_argument("--solar-factor", type=float, default=0.5)
    ap.add_argument("--load", default="{}", help="JSON: round -> list of energy_phases phases run in the background")
    ap.add_argument("--load-warmup-s", type=float, default=12.0)


def main():
    ap = argparse.ArgumentParser()
    add_args(ap)
    args = ap.parse_args()
    args.load = json.loads(args.load)

    from flwr.client import start_client
    patch_grpc_timeouts()
    client = BoardBlockClient(args)
    try:
        start_client(server_address=args.server, client=client.to_client(), insecure=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
