"""Board Flower client that closes the FL-to-NPU loop.

Each round the board:
  1. writes the received global head into its int8 classifier (rafl.fl.npu_head) and loads the
     new file on the NPU (fresh graph cache);
  2. evaluates that NPU model on held-out test crops and checks that its logits equal the int8
     emulation of the head it was given;
  3. reads its real telemetry (CPU load, temperature) and asks the controller whether to train
     and at which bit width; battery and uplink come from the server's simulated environment;
  4. trains the head on features computed by the same NPU model, and uploads a packed update.

  python3 -m rafl.fl.board_npu_client --server 127.0.0.1:<port> --root ~/rafl/flnpu
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path.home() / "rafl/pylib"))

from rafl.edge.telemetry import Telemetry  # noqa: E402
from rafl.fl import npu_head  # noqa: E402
from rafl.fl.controller import ControllerConfig, ResourceAwareController  # noqa: E402
from rafl.fl.packed_client import PackedHeadClient  # noqa: E402
from rafl.fl.real import noisy_labels  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class NpuModel:
    def __init__(self, model_bytes: bytes, path: Path, delegate: str, cache: Path):
        from tflite_runtime.interpreter import Interpreter, load_delegate
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(model_bytes)
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["VIV_VX_ENABLE_CACHE_GRAPH_BINARY"] = "1"
        os.environ["VIV_VX_CACHE_BINARY_GRAPH_DIR"] = str(cache)
        t0 = time.perf_counter()
        self.it = Interpreter(str(path), experimental_delegates=[load_delegate(delegate)])
        self.it.allocate_tensors()
        self.i = self.it.get_input_details()[0]
        self.outs = self.it.get_output_details()
        self.run(np.zeros((224, 224, 3), np.uint8))  # graph compile happens on the first invoke
        self.load_s = time.perf_counter() - t0

    def run(self, rgb: np.ndarray):
        x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        s, z = self.i["quantization"]
        self.it.set_tensor(self.i["index"], np.clip(np.round(x / s + z), -128, 127).astype(np.int8)[None])
        self.it.invoke()
        return (self.it.get_tensor(self.outs[0]["index"]).reshape(-1).copy(),
                self.it.get_tensor(self.outs[1]["index"]).reshape(-1).copy())


def load_rgb(f, size=224):
    im = Image.open(f).convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.BICUBIC)
    return np.asarray(im)


class BoardNpuClient(PackedHeadClient):
    def __init__(self, root: Path, delegate: str, x_unused, y, k, args, policy, tel: Telemetry):
        self.root, self.delegate, self.args, self.tel = root, delegate, args, tel
        self.base = (root / "classifier_fl.tflite").read_bytes()
        self.pm = npu_head.load_map(root / "patch_map.json")
        self.calib = np.load(root / "calib_features_q.npy")
        man = json.load(open(root / "data/manifest.json"))
        self.train_files = [root / "data/crops" / r["file"] for r in man["crops"]]
        ev = json.load(open(root / "eval/bursts.json"))
        self.classes = ev["classes"]
        self.eval_items = [(root / "eval" / f, self.classes.index(b["truth"]), b["id"])
                           for b in ev["bursts"] for f in b["frames"]]
        self.load_proc: list[subprocess.Popen] = []
        self.feats_ready = False
        super().__init__("board", np.zeros((len(y), 1), np.float32), y, k, args.lr, args.momentum, args.batch,
                         args.seed, policy, self.observe_real, args.epochs)

    # ---- background workload standing in for detection activity ----
    def start_load(self, rnd):
        kinds = self.args.load.get(str(rnd), [])
        for kind in kinds:
            cmd = [sys.executable, str(Path.home() / "rafl/energy_phases.py"), "--out", str(self.root / f"load_r{rnd}"),
                   "--delegate", self.delegate, "--detector", str(Path.home() / "rafl/models/det_ld/int8_per_tensor.tflite"),
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

    def observe_real(self, config):
        self.tel.sample()
        time.sleep(1.0)
        obs = self.tel.observe(battery_soc=float(config["battery_soc"]))
        obs["uplink_kbps"] = float(config["uplink_kbps"])  # simulated deployment uplink, not the LAN
        obs["deadline_s"] = float(config.get("deadline_s", obs["deadline_s"]))
        return obs

    def before_round(self, parameters, config):
        rnd = int(config["round"])
        W, b = [np.asarray(p, np.float32) for p in parameters]
        t0 = time.perf_counter()
        q = npu_head.quantize_head(W, b, self.pm, self.calib)
        model = npu_head.patch(self.base, self.pm, q)
        patch_s = time.perf_counter() - t0
        npu = NpuModel(model, self.root / f"models/classifier_r{rnd}.tflite", self.delegate,
                       self.root / f"vxcache/r{rnd}_{npu_head.digest(model)}")
        # evaluate the deployed NPU model on held-out crops, and check it runs the head it was sent
        t0 = time.perf_counter()
        logits, feats = zip(*(npu.run(load_rgb(f)) for f, _, _ in self.eval_items))
        eval_ms = 1000 * (time.perf_counter() - t0) / len(self.eval_items)
        logits, feats = np.array(logits), np.array(feats)
        agree = float((npu_head.emulate(feats, q, self.pm) == logits).mean())
        within1 = float((np.abs(npu_head.emulate(feats, q, self.pm).astype(int) - logits.astype(int)) <= 1).mean())
        y = np.array([t for _, t, _ in self.eval_items])
        pred = logits.argmax(1)
        crop_acc = float((pred == y).mean())
        votes, truth = {}, {}
        p = np.exp(logits.astype(np.float32))
        for (_, t, bid), pi in zip(self.eval_items, p / p.sum(1, keepdims=True)):
            votes[bid] = votes.get(bid, 0) + pi
            truth[bid] = t
        burst_acc = float(np.mean([votes[k].argmax() == truth[k] for k in votes]))
        if not self.feats_ready:
            t0 = time.perf_counter()
            fq = np.array([npu.run(load_rgb(f))[1] for f in self.train_files])
            self.feat_ms = 1000 * (time.perf_counter() - t0) / len(self.train_files)
            np.save(self.root / "board_features_q.npy", fq)
            self.x = npu_head.dequant_features(fq, self.pm)
            ref = self.root / "data/ref_features_q.npy"
            self.ref_match = float((np.load(ref) == fq).mean()) if ref.exists() else float("nan")
            self.feats_ready = True
        load = "none" if config.get("deploy_only") else self.start_load(rnd)
        return {"model_digest": npu_head.digest(model), "npu_load_s": npu.load_s, "patch_s": patch_s, "npu_eval_ms_per_crop": eval_ms,
                "npu_crop_acc": crop_acc, "npu_burst_acc": burst_acc, "npu_vs_emulation": agree,
                "npu_vs_emulation_1lsb": within1, "feat_ms_per_crop": self.feat_ms,
                "feat_match_mac": self.ref_match, "background_load": load, "eval_crops": len(y)}

    def fit(self, parameters, config):
        try:
            return super().fit(parameters, config)
        finally:
            self.stop_load()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--root", default="~/rafl/flnpu")
    ap.add_argument("--delegate", default="/usr/lib/libvx_delegate.so")
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--load", default="{}", help='JSON: round -> list of energy_phases phases run in the background')
    ap.add_argument("--load-warmup-s", type=float, default=12.0)
    args = ap.parse_args()
    args.load = json.loads(args.load)

    from flwr.client import start_client

    root = Path(args.root).expanduser()
    man = json.load(open(root / "data/manifest.json"))
    y = np.array([r["label"] for r in man["crops"]], np.int64)
    k = len(man["classes"])
    y = noisy_labels(y, k, args.noise, np.random.default_rng(args.seed))
    policy = ResourceAwareController(ControllerConfig(), n_params=1280 * k + k)
    client = BoardNpuClient(root, args.delegate, None, y, k, args, policy, Telemetry(train_s_per_sample_epoch=3.8e-5))
    try:
        start_client(server_address=args.server, client=client.to_client(), insecure=True)
    finally:
        client.stop_load()


if __name__ == "__main__":
    main()
