"""Real Flower client for the board. Computes frozen-backbone features of the local crops with TFLite
(FP32, CPU), then trains the softmax head as a flwr NumPyClient against the server.

  python3 -m rafl.fl.board_client --server 127.0.0.1:<port> --data ~/rafl/fl/data --backbone ~/rafl/fl/backbone_fp32.tflite
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path.home() / "rafl/pylib"))

from rafl.fl.real import TimedHeadClient, noisy_labels, standardize

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def backbone_features(model, files, size=224, threads=4):
    from tflite_runtime.interpreter import Interpreter
    it = Interpreter(str(model), num_threads=threads)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    out = []
    for f in files:
        im = Image.open(f).convert("RGB")
        if im.size != (size, size):
            im = im.resize((size, size), Image.BILINEAR)
        x = ((np.asarray(im, np.float32) / 255.0 - MEAN) / STD)[None]
        it.set_tensor(i["index"], x.astype(np.float32))
        it.invoke()
        out.append(it.get_tensor(o["index"])[0].copy())
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--features-out", default=None)
    args = ap.parse_args()

    from flwr.client import start_client

    data = Path(args.data).expanduser()
    man = json.load(open(data / "manifest.json"))
    stats = np.load(data / "stats.npz")
    files = [data / "crops" / r["file"] for r in man["crops"]]
    y = np.array([r["label"] for r in man["crops"]], np.int64)
    k = len(man["classes"])

    t0 = time.perf_counter()
    feats = backbone_features(args.backbone, files)
    feat_s = time.perf_counter() - t0
    if args.features_out:
        np.save(args.features_out, feats)
    print(f"features {feats.shape} in {feat_s:.1f}s ({1000 * feat_s / len(files):.1f} ms/crop)", flush=True)

    rng = np.random.default_rng(args.seed)
    y = noisy_labels(y, k, args.noise, rng)
    x = standardize(feats, stats["mu"], stats["sd"])
    client = TimedHeadClient(man["location"], x, y, k, args.lr, args.momentum, args.batch, args.seed, True)
    start_client(server_address=args.server, client=client.to_client(), insecure=True)


if __name__ == "__main__":
    main()
