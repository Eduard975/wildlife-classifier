"""Resource-aware FL with dynamic quantization, with every client decision made on the board.

The Mac plays the cloud: a Flower server on loopback, simulated peer cameras in-process (rafl.fl.hetero
policy and simulated resources), and HeteroFL aggregation (each parameter averaged over the clients that
trained it). The board joins through an SSH reverse tunnel as rafl.fl.board_block_client, which
re-quantizes the global block and head into its own int8 model, measures its own upload, CPU load and
temperature, keeps its own battery model, chooses tier and bit width, and limits its own uplink to the
round's link rate with an egress policer on the tunnel connection.

The server only measures what it can see: each board result's arrival time against the round deadline.

  .venv/bin/python scripts/run_fl_board.py --rounds 8 --sim-clients 8 \
      --link '{"1": 512, "3": 64, "5": 24, "7": 8}' --load '{"4": ["cpu_pipeline", "fl_features"]}'
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import tarfile
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from flwr.common import FitIns, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerConfig, SimpleClientManager, start_server
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from rafl.fl import block, hetero, npu_block
from rafl.fl.quant import decode, encode
from rafl.fl.resources import EnvConfig, ResourceEnv

ROOT = Path(__file__).resolve().parents[1]
import run_fl_real  # noqa: E402  (InProcessProxy)

TUNNEL_SRC_PORT = 52022


class SimBlockClient:
    """A simulated peer camera: same policy, training and packed upload as the board, simulated resources."""

    def __init__(self, cid, x, y, args, env, seed):
        self.cid, self.x, self.y, self.args, self.env = cid, x, y, args, env
        self.rng = np.random.default_rng(seed)
        self.policy = hetero.resource_aware(margin=args.margin)
        self.residual = {}
        self._sizes = None

    def fit(self, parameters, config):
        p = {k: np.asarray(v, np.float32) for k, v in zip(block.NAMES, parameters)}
        if config.get("deploy_only") or not self.env.online(self.cid):
            return [np.zeros(0, np.uint8)], 0, {"tier": "offline" if not config.get("deploy_only") else "deploy", "skipped": 1}
        if self._sizes is None:
            self._sizes = {(t, b): hetero.payload_size(p, t, b, self.args.group) for t in block.TIERS for b in (2, 4)}
        s = {**self.env.observe(self.cid), "n_samples": len(self.y), "deadline_s": float(config["deadline_s"])}
        plan = self.policy(int(config["round"]), self.cid, s, self._sizes)
        m = {"tier": plan.tier, "bits": plan.bits, "why": plan.why, "battery_soc": s["battery_soc"],
             "cpu_load": s["cpu_load"], "uplink_kbps_used": s["uplink_kbps"]}
        if plan.tier == "skip":
            return [np.zeros(0, np.uint8)], 0, {**m, "skipped": 1}
        new = block.local_train(p, self.x, self.y, plan.tier, plan.epochs, self.args.lr, 0.9, 32, self.rng)
        delta = hetero.split({k: new[k] - p[k] for k in block.TIERS[plan.tier]}, plan.tier, self.args.group)
        res = self.residual.get(plan.tier)
        if res is not None:
            delta = [d + r for d, r in zip(delta, res)]
        payload, deq = encode(delta, plan.bits, self.rng)
        self.residual[plan.tier] = [d - q for d, q in zip(delta, deq)]
        return [np.frombuffer(payload, np.uint8).copy()], len(self.y), {**m, "skipped": 0, "payload_bytes": len(payload)}


class TimedProxy(ClientProxy):
    """Wraps the board's gRPC proxy to record when each fit result arrives."""

    def __init__(self, inner, times):
        super().__init__(inner.cid)
        self.inner, self.times = inner, times

    def fit(self, ins, timeout, group_id):
        t0 = time.monotonic()
        res = self.inner.fit(ins, timeout, group_id)
        self.times[int(ins.config["round"])] = time.monotonic() - t0
        return res

    def get_properties(self, ins, timeout, group_id):
        return self.inner.get_properties(ins, timeout, group_id)

    def get_parameters(self, ins, timeout, group_id):
        return self.inner.get_parameters(ins, timeout, group_id)

    def evaluate(self, ins, timeout, group_id):
        return self.inner.evaluate(ins, timeout, group_id)

    def reconnect(self, ins, timeout, group_id):
        return self.inner.reconnect(ins, timeout, group_id)


class TimedClientManager(SimpleClientManager):
    def __init__(self, times):
        super().__init__()
        self.times = times

    def register(self, client):
        if not isinstance(client, (run_fl_real.InProcessProxy, TimedProxy)):
            client = TimedProxy(client, self.times)
        return super().register(client)


class HeteroStrategy(FedAvg):
    def __init__(self, names, env, rows, args, times, init, **kw):
        super().__init__(**kw)
        self.names, self.env, self.rows, self.args, self.times = names, env, rows, args, times
        self.g = {k: v.copy() for k, v in init.items()}
        self.adam = {k: [np.zeros_like(v), np.zeros_like(v)] for k, v in init.items()}
        link = {int(k): float(v) for k, v in json.loads(args.link).items()}
        self.link = lambda r: link[max([k for k in link if k <= r], default=min(link))]

    def cid_of(self, proxy):
        return self.names.get(proxy.cid, "board")

    def configure_fit(self, server_round, parameters, client_manager):
        deploy_only = server_round > self.args.rounds
        if not deploy_only:
            self.env.step(server_round)
        pairs = super().configure_fit(server_round, parameters, client_manager)
        out = []
        for proxy, ins in pairs:
            cfg = {"round": server_round, "deploy_only": int(deploy_only), "deadline_s": self.args.deadline_s}
            if self.cid_of(proxy) == "board":
                cfg["link_kbit"] = self.link(min(server_round, self.args.rounds))
            out.append((proxy, FitIns(ins.parameters, cfg)))
        return out

    def aggregate_fit(self, server_round, results, failures):
        if failures:
            raise RuntimeError(f"{len(failures)} client(s) failed in round {server_round}: {failures}; see board_client.log")
        sums = {k: np.zeros_like(v) for k, v in self.g.items()}
        wts = {k: 0.0 for k in self.g}
        for proxy, res in results:
            cid, m = self.cid_of(proxy), dict(res.metrics)
            arr = parameters_to_ndarrays(res.parameters)[0]
            wire = sum(len(t) for t in res.parameters.tensors)
            row = {"round": server_round, "client": cid, "real": cid == "board", "samples": res.num_examples, "wire_bytes": wire, **m}
            if res.num_examples and arr.size:
                tier = m["tier"]
                if cid == "board":
                    elapsed = self.times.get(server_round, 0.0)
                    ok = elapsed <= self.args.deadline_s
                    row.update({"server_elapsed_s": round(elapsed, 2)})
                else:
                    st = self.env.settle(cid, res.num_examples, 1, arr.size, hetero.S_PER_SAMPLE[tier])
                    ok = st.accepted
                    row["upload_s_sim"] = round(st.upload_s, 2)
                row["accepted"] = ok
                if ok:
                    parts, bits = decode(arr.tobytes())
                    d = hetero.join(parts, tier, self.g)
                    for k, v in d.items():
                        sums[k] += v * res.num_examples
                        wts[k] += res.num_examples
            self.rows.append(row)
            if cid == "board":
                print(">>> " + board_line(server_round, row), flush=True)
        if server_round <= self.args.rounds:
            for k in self.g:
                if wts[k] == 0:
                    continue
                upd = sums[k] / wts[k]
                if self.args.server == "adam":
                    mm, vv = self.adam[k]
                    mm[:] = 0.9 * mm + 0.1 * upd
                    vv[:] = 0.99 * vv + 0.01 * upd ** 2
                    self.g[k] = (self.g[k] + self.args.server_lr * mm / (np.sqrt(vv) + 1e-3)).astype(np.float32)
                else:
                    self.g[k] = (self.g[k] + upd).astype(np.float32)
        return ndarrays_to_parameters([self.g[k] for k in block.NAMES]), {}


def board_line(rnd, r):
    if r.get("tier") == "deploy":
        return f"round {rnd}: board deployed the final model on the NPU (burst acc {r.get('npu_burst_acc', 0):.2f}, model {r.get('model_digest')})"
    head = (f"round {rnd}: hour {r.get('hour')}  link {r.get('link_kbit', 0):.0f} kbit/s  measured {r.get('measured_kbps', 0):.0f}  "
            f"battery {r.get('battery_soc', 0):.2f}  cpu {r.get('cpu_load', 0):.2f}  load={r.get('background_load')}")
    dec = "SKIP" if r.get("skipped") else f"{r.get('tier')} {r.get('bits')}-bit, {r.get('wire_bytes')} B"
    acc = ("accepted" if r.get("accepted") else "MISSED DEADLINE") if "accepted" in r else ""
    return f"{head}  -> {dec} {acc} [{r.get('why', '')}]  NPU burst acc {r.get('npu_burst_acc', 0):.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board-location", default="NACTI : lebec_CA-42")
    ap.add_argument("--sim-clients", type=int, default=8)
    ap.add_argument("--min-sim-samples", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--margin", type=float, default=0.6)
    ap.add_argument("--deadline-s", type=float, default=120.0)
    ap.add_argument("--sim-bw-kbps", type=float, default=48.0)
    ap.add_argument("--link", default='{"1": 512, "3": 64, "5": 24, "7": 8}', help="JSON: first round -> board uplink limit, kbit/s")
    ap.add_argument("--load", default="{}", help="JSON: round -> background phases on the board")
    ap.add_argument("--server", choices=("avg", "adam"), default="avg")
    ap.add_argument("--server-lr", type=float, default=0.01)
    ap.add_argument("--board-args", default="", help="extra arguments for board_block_client (battery, start hour, ...)")
    ap.add_argument("--node", action="store_true",
                    help="run the combined camera node on the board (detection with precision switching + FL), rafl.edge.node")
    ap.add_argument("--node-args", default="--stage ~/rafl/stage_frames --l0 ~/rafl/levels/models/det_i8pt.tflite "
                                           "--l1 ~/rafl/levels/models/det_i8pc.tflite --l2 ~/rafl/levels/models/det_fp32.tflite",
                    help="arguments for rafl.edge.node besides the FL ones")
    ap.add_argument("--port", type=int, default=39673)
    ap.add_argument("--host", default=f"root@{os.environ.get('BOARD_HOST', '192.168.1.12')}")
    ap.add_argument("--iface", default="en0", help="Mac interface towards the board")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--init", choices=("deployed", "imagenet"), default="deployed",
                    help="starting block and head: the deployed ones, or the ImageNet 1x1 conv with a random head "
                         "(scripts/prep_hetero.py), so the federated model has something to learn")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))
    classes = cfg["models"]["classifier"]["classes"]
    idx = {c: i for i, c in enumerate(classes)}
    fb = ROOT / "runs/fl_block"
    qm = json.loads((fb / "quant_map.json").read_text())
    tmpl = (fb / "classifier_block.tflite").read_bytes()
    from rafl.edge import requant
    xp = npu_block.block_input_params(tmpl, requant.load_map(fb / "quant_map.json"))
    xq = np.load(ROOT / "data/features/qat_block_q.npy", mmap_mode="r")
    meta = pd.read_parquet(ROOT / "data/features/qat_block_meta.parquet").reset_index(drop=True)
    y_all = meta.label.map(idx).to_numpy()
    te = np.flatnonzero(((meta.split == "test") & (meta.verified != "n")).to_numpy())
    x_te = npu_block.dequant(xq[te], xp)
    tr = (meta.split == "train").to_numpy()
    d = np.load(ROOT / ("runs/fl_hetero/init_block.npz" if args.init == "deployed" else "runs/fl_hetero/init_imagenet.npz"))
    init = {k: d[k].astype(np.float32) for k in block.NAMES}

    out = ROOT / f"runs/fl_board/run_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    rng = np.random.default_rng(args.seed)
    counts = meta[tr].location_id.value_counts()
    cand = sorted(c for c, n in counts.items() if n >= args.min_sim_samples and c != args.board_location)
    chosen = sorted(rng.choice(cand, size=args.sim_clients, replace=False))
    env = ResourceEnv(list(chosen), EnvConfig(bw_median_kbps=args.sim_bw_kbps, deadline_s=args.deadline_s, seed=args.seed))
    times, rows, evals = {}, [], []
    cm = TimedClientManager(times)
    names = {}
    for j, c in enumerate(chosen):
        m = np.flatnonzero((meta.location_id == c).to_numpy() & tr)
        yy = y_all[m]
        flip = rng.random(len(yy)) < 0.1
        yy = np.where(flip, rng.integers(0, len(classes), len(yy)), yy)
        names[f"sim{j:02d}"] = c
        cm.register(run_fl_real.InProcessProxy(f"sim{j:02d}", SimBlockClient(c, npu_block.dequant(xq[m], xp), yy, args, env, args.seed + j)))

    # ---- stage the board: model template, quant map, its own crops, eval bursts, code ----
    ssh = ["ssh", "-o", f"ProxyCommand=nc -b {args.iface} %h %p", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    stage = out / "board_stage"
    (stage / "data").mkdir(parents=True)
    shutil.copy(fb / "classifier_block.tflite", stage / "classifier_block.tflite")
    shutil.copy(fb / "quant_map.json", stage / "quant_map.json")
    bs = ROOT / "runs/fl_npu/board_stage_src"
    shutil.copytree(bs / "crops", stage / "data/crops")
    shutil.copy(bs / "manifest.json", stage / "data/manifest.json")
    shutil.copytree(ROOT / "runs/board_stage", stage / "eval")
    tarball = out / "board_stage.tar"
    with tarfile.open(tarball, "w") as t:
        t.add(stage, arcname="flblock", filter=lambda ti: None if Path(ti.name).name.startswith("._") else ti)
        for f in ("__init__.py", "fl/__init__.py", "fl/block.py", "fl/hetero.py", "fl/quant.py", "fl/resources.py",
                  "fl/npu_block.py", "fl/board_block_client.py", "edge/__init__.py", "edge/requant.py", "edge/telemetry.py",
                  "edge/precision.py", "edge/node.py", "edge/run_frames.py", "edge/node_io.py"):
            t.add(ROOT / "src/rafl" / f, arcname=f"rafl/{f}")
        t.add(ROOT / "board/police.sh", arcname="police.sh")
        t.add(ROOT / "src/rafl/edge/energy_phases.py", arcname="energy_phases.py")
    subprocess.run(ssh + [args.host, "rm -rf ~/rafl/flblock && tar -C ~/rafl -xf -"], stdin=open(tarball, "rb"), check=True)
    tarball.unlink()
    shutil.rmtree(stage / "data/crops")

    def eval_fn(rnd, params, _):
        p = {k: np.asarray(v, np.float32) for k, v in zip(block.NAMES, params)}
        mt = block.metrics(p, x_te, y_all[te], classes)
        row = {"round": rnd, "test_bacc_float": mt["bacc"], **{k: v for k, v in mt.items() if k.startswith("recall")}}
        evals.append(row)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
        return 0.0, {}

    n = len(chosen) + 1
    strat = HeteroStrategy(names, env, rows, args, times, init, fraction_fit=1.0, fraction_evaluate=0.0, min_fit_clients=n,
                           min_available_clients=n, min_evaluate_clients=0, evaluate_fn=eval_fn,
                           initial_parameters=ndarrays_to_parameters([init[k] for k in block.NAMES]))
    mac_ip = subprocess.run(["ipconfig", "getifaddr", args.iface], capture_output=True, text=True).stdout.strip()
    addr = f"127.0.0.1:{args.port}"
    node_out = f"~/rafl/runs/node_{out.name.removeprefix('run_')}"
    module = "rafl.edge.node" if args.node else "rafl.fl.board_block_client"
    extra = f"{args.node_args} --out {node_out} > {node_out}.log 2>&1" if args.node else ""
    cmd = (f"cd ~/rafl && PYTHONPATH=$HOME/rafl:$HOME/rafl/pylib python3 -u -m {module} "
           f"--server {addr} --root ~/rafl/flblock --lr {args.lr} --group {args.group} --margin {args.margin} "
           f"--police-dst {mac_ip} --police-dport {TUNNEL_SRC_PORT} --load '{args.load}' {args.board_args} {extra}")
    proc = subprocess.Popen(
        ["ssh", "-o", f"ProxyCommand=nc -b {args.iface} -p {TUNNEL_SRC_PORT} %h %p", "-o", "BatchMode=yes",
         "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=20",
         "-o", "ExitOnForwardFailure=yes",
         "-R", f"127.0.0.1:{args.port}:127.0.0.1:{args.port}", args.host, cmd],
        stdout=open(out / "board_client.log", "w"), stderr=subprocess.STDOUT)
    import grpc
    orig_server = grpc.server

    def server(*a, options=None, **kw):  # same ping-timeout fix as the board client (board_block_client.patch_grpc_timeouts)
        return orig_server(*a, options=list(options or []) + [("grpc.keepalive_timeout_ms", 600000),
                                                              ("grpc.http2.ping_timeout_ms", 600000)], **kw)
    grpc.server = server
    try:
        start_server(server_address=addr, config=ServerConfig(num_rounds=args.rounds + 1, round_timeout=900),
                     strategy=strat, client_manager=cm)
    finally:
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.terminate()
        subprocess.run(ssh + [args.host, "sh ~/rafl/police.sh clear mlan0"], check=False)
        with open(out / "board_rounds.jsonl", "wb") as f:
            subprocess.run(ssh + [args.host, "cat ~/rafl/flblock/rounds.jsonl"], stdout=f, check=False)
        if args.node:
            (out / "node").mkdir(exist_ok=True)
            with open(out / "node/node.tar", "wb") as f:
                subprocess.run(ssh + [args.host, f"tar -C {node_out} -cf - ."], stdout=f, check=False)
            subprocess.run(["tar", "-C", str(out / "node"), "-xf", str(out / "node/node.tar")], check=False)
            (out / "node/node.tar").unlink(missing_ok=True)
            with open(out / "node/node.log", "wb") as f:
                subprocess.run(ssh + [args.host, f"cat {node_out}.log"], stdout=f, check=False)
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("round", "client"), k))
        with open(out / "clients.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        pd.DataFrame(evals).to_csv(out / "global.csv", index=False)
        json.dump({"args": vars(args), "sim_clients": list(chosen), "env_summary": env.summary(), "mac_ip": mac_ip},
                  open(out / "config.json", "w"), indent=2)
        print("results", out)


if __name__ == "__main__":
    main()
