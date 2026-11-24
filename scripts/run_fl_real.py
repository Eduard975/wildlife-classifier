"""Mixed real + simulated federated round(s): Flower server on the Mac (loopback only), N simulated
clients in-process, the board as a real Flower client through an SSH reverse tunnel.

  .venv/bin/python scripts/run_fl_real.py --sim-clients 8 --rounds 4 --bits 4
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from flwr.common import (DisconnectRes, EvaluateRes, FitRes, GetParametersRes, GetPropertiesRes, Status, Code,
                         ndarrays_to_parameters, parameters_to_ndarrays)
from flwr.server import ServerConfig, SimpleClientManager, start_server
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from rafl.fl.real import TimedHeadClient, standardize
from rafl.fl.sim import SimConfig, build_clients, init_head, metrics

ROOT = Path(__file__).resolve().parents[1]
OK = Status(code=Code.OK, message="ok")


class InProcessProxy(ClientProxy):
    def __init__(self, cid, client):
        super().__init__(cid)
        self.client = client

    def get_properties(self, ins, timeout, group_id):
        return GetPropertiesRes(status=OK, properties={})

    def get_parameters(self, ins, timeout, group_id):
        return GetParametersRes(status=OK, parameters=ndarrays_to_parameters([]))

    def fit(self, ins, timeout, group_id):
        try:
            up, n, m = self.client.fit(parameters_to_ndarrays(ins.parameters), ins.config)
        except Exception:
            import traceback
            traceback.print_exc()
            raise
        return FitRes(status=OK, parameters=ndarrays_to_parameters(up), num_examples=n, metrics=m)

    def evaluate(self, ins, timeout, group_id):
        return EvaluateRes(status=OK, loss=0.0, num_examples=0, metrics={})

    def reconnect(self, ins, timeout, group_id):
        return DisconnectRes(reason="")


class LoggingFedAvg(FedAvg):
    def __init__(self, names, rows, **kw):
        super().__init__(**kw)
        self.names, self.rows = names, rows

    def aggregate_fit(self, server_round, results, failures):
        for proxy, res in results:
            m = res.metrics
            self.rows.append({"round": server_round, "client": self.names.get(proxy.cid, "board (real)"),
                              "real": proxy.cid not in self.names, "samples": res.num_examples,
                              "upload_bytes": m.get("upload_bytes"), "train_s": m.get("train_s"),
                              "cpu_s": m.get("cpu_s"), "sys_cpu_util": m.get("sys_cpu_util"),
                              "temp_start_c": m.get("temp_start_c"), "temp_end_c": m.get("temp_end_c")})
        return super().aggregate_fit(server_round, results, failures)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="efficientnet_lite0")
    ap.add_argument("--board-location", default="NACTI : lebec_CA-42")
    ap.add_argument("--sim-clients", type=int, default=8)
    ap.add_argument("--min-sim-samples", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--local-epochs", type=int, default=5)
    ap.add_argument("--port", type=int, default=39671)
    ap.add_argument("--host", default=f"root@{os.environ.get('BOARD_HOST', '192.168.1.12')}",
                    help="board ssh target; default root@$BOARD_HOST")
    ap.add_argument("--no-board", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))
    classes = cfg["models"]["classifier"]["classes"]
    x = np.load(ROOT / f"data/features/{args.arch}.npy")
    meta = pd.read_parquet(ROOT / f"data/features/{args.arch}_meta.parquet").reset_index(drop=True)
    tr = (meta.split == "train").to_numpy()
    xt = x[tr].astype(np.float32)
    mu, sd = xt.mean(0), xt.std(0) + 1e-6
    xs = standardize(x, mu, sd)

    sc = SimConfig(seed=1337, local_epochs=args.local_epochs)
    clients = build_clients(xs, meta, classes, sc, exclude={args.board_location})
    cand = sorted(c for c, cl in clients.items() if len(cl.y) >= args.min_sim_samples)
    rng = np.random.default_rng(sc.seed)
    chosen = sorted(rng.choice(cand, size=args.sim_clients, replace=False))
    idx = {c: i for i, c in enumerate(classes)}
    ev = {s: (xs[(meta.split == s).to_numpy()], meta[meta.split == s].label.map(idx).to_numpy())
          for s in ("val", "test")}

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = ROOT / (args.out or f"runs/fl_real/run_{ts}")
    out.mkdir(parents=True, exist_ok=True)
    rows, evals = [], []
    names, cm = {}, SimpleClientManager()
    for j, c in enumerate(chosen):
        cl = clients[c]
        timed = TimedHeadClient(c, cl.x, cl.y, cl.k, cl.lr, cl.momentum, cl.batch, sc.seed + 100 + j, True)
        names[f"sim{j:02d}"] = c
        cm.register(InProcessProxy(f"sim{j:02d}", timed))

    def eval_fn(rnd, params, _):
        row = {"round": rnd}
        for s in ("val", "test"):
            m = metrics(params, *ev[s], classes)
            row.update({f"{s}_acc": m["acc"], f"{s}_bacc": m["bacc"]})
        row.update({f"test_recall_{c}": m[f"recall_{c}"] for c in classes})
        evals.append(row)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
        return 0.0, {}

    n_clients = args.sim_clients + (0 if args.no_board else 1)
    strat = LoggingFedAvg(names, rows, fraction_fit=1.0, fraction_evaluate=0.0, min_fit_clients=n_clients,
                          min_available_clients=n_clients, min_evaluate_clients=0, evaluate_fn=eval_fn,
                          on_fit_config_fn=lambda r: {"bits": args.bits, "local_epochs": args.local_epochs},
                          initial_parameters=ndarrays_to_parameters(init_head(xs.shape[1], len(classes), rng)))

    addr = f"127.0.0.1:{args.port}"
    (out / "lsof.txt").write_text("")

    def snap():
        r = subprocess.run(["lsof", "-nP", "-iTCP", "-sTCP:LISTEN"], capture_output=True, text=True)
        (out / "lsof.txt").write_text("\n".join(l for l in r.stdout.splitlines()
                                                if l.startswith("COMMAND") or f":{args.port} " in l) + "\n")

    threading.Timer(3.0, snap).start()
    proc = None
    if not args.no_board:
        cmd = ("cd ~/rafl && PYTHONPATH=$HOME/rafl python3 -u -m rafl.fl.board_client "
               f"--server {addr} --data ~/rafl/fl/data --backbone ~/rafl/fl/backbone_fp32.tflite "
               "--features-out ~/rafl/fl/board_features.npy --noise 0.1")
        proc = subprocess.Popen(
            ["ssh", "-o", "ProxyCommand=nc -b en0 %h %p", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
             "-o", "ExitOnForwardFailure=yes", "-R", f"127.0.0.1:{args.port}:127.0.0.1:{args.port}",
             args.host, cmd], stdout=open(out / "board_client.log", "w"), stderr=subprocess.STDOUT)
    hist = start_server(server_address=addr, config=ServerConfig(num_rounds=args.rounds), strategy=strat,
                        client_manager=cm)
    if proc:
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.terminate()
    with open(out / "clients.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    pd.DataFrame(evals).to_csv(out / "global.csv", index=False)
    json.dump({"args": vars(args), "sim_clients": list(chosen), "bits": args.bits}, open(out / "config.json", "w"),
              indent=2)
    print("results", out)


if __name__ == "__main__":
    main()
