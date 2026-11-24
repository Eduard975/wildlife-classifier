"""FL that ends on the NPU: the federated head is the head of the deployed int8 classifier.

Flower server on the Mac (loopback only), N simulated clients in-process, the board as a real
client through an SSH reverse tunnel (rafl.fl.board_npu_client). Every client runs the
resource-aware controller each round and uploads a packed update (rafl.fl.quant.encode), so the
bytes counted are the bytes Flower transports. Battery and uplink come from the simulated
environment (rafl.fl.resources); the board's CPU load and temperature are its real telemetry.
After each round the global head is re-quantized into the int8 file; the board loads that file on
its NPU at the start of the next round. A final deploy-only round puts the last head on the NPU.

  .venv/bin/python scripts/run_fl_npu.py --rounds 6 --sim-clients 8 --scenario constrained \
      --load '{"3": ["npu_pipeline"], "4": ["cpu_pipeline"], "5": ["cpu_pipeline", "fl_features"]}'
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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
from flwr.server.strategy import FedAvg

from rafl.fl import npu_head
from rafl.fl.controller import ControllerConfig, ResourceAwareController
from rafl.fl.packed_client import PackedHeadClient
from rafl.fl.quant import decode
from rafl.fl.resources import EnvConfig, ResourceEnv
from rafl.fl.sim import metrics

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = {"cellular": 48.0, "constrained": 8.0}
import run_fl_real  # noqa: E402  (InProcessProxy)


class LiveTelemetry:
    """Streams the board's telemetry over its own ssh session and writes one readable line per second
    to a local file, tagged with the current FL round. Other terminals can `tail -f` that file without
    any network access of their own."""

    def __init__(self, host, path: Path):
        self.path, self.round = path, 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "w", buffering=1)
        self.f.write(f"# live board telemetry from {host}, started {time.strftime('%H:%M:%S')}\n")
        self.proc = subprocess.Popen(
            ["ssh", "-o", "ProxyCommand=nc -b en0 %h %p", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host,
             "python3 ~/rafl/telemetry.py --hz 1 --duration 3600 --out /dev/stdout"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.t = threading.Thread(target=self._pump, daemon=True)
        self.t.start()

    def _pump(self):
        for line in self.proc.stdout:
            try:
                s = json.loads(line)
            except ValueError:
                continue
            r = f"round {self.round}" if self.round else "setup  "
            self.f.write(f"{time.strftime('%H:%M:%S')}  {r}  cpu {s['cpu_util']:.2f}  "
                         f"temp {max(s['temp_c'].values()):.0f} C  freq {max(s['freq_mhz'])} MHz\n")

    def note(self, text):
        self.f.write(f"{time.strftime('%H:%M:%S')}  >>> {text}\n")

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.t.join(timeout=2)
        self.f.close()


def board_line(server_round, m, wire):
    if m.get("deploy_only"):
        return f"round {server_round}: board loaded the final head on the NPU (model {m.get('model_digest', '')})"
    dec = f"skip ({m.get('reason')})" if m.get("skipped") else f"train, {int(m.get('bits', 0))}-bit"
    return (f"round {server_round}: board load={m.get('background_load', 'none')}  cpu {m.get('cpu_load', 0):.2f}  "
            f"temp {m.get('temp_c', 0):.0f} C  -> {dec}, {wire} B on the wire  "
            f"(NPU burst acc {m.get('npu_burst_acc', 0):.2f}, model {m.get('model_digest', '')})")


class NpuFedAvg(FedAvg):
    def __init__(self, names, env, rows, out, base, pm, calib, rounds, live=None, **kw):
        super().__init__(**kw)
        self.names, self.env, self.rows, self.out = names, env, rows, out
        self.base, self.pm, self.calib, self.rounds = base, pm, calib, rounds
        self.global_params = None
        self.live = live

    def cid_of(self, proxy):
        return self.names.get(proxy.cid, "board")

    def configure_fit(self, server_round, parameters, client_manager):
        self.global_params = parameters_to_ndarrays(parameters)
        deploy_only = server_round > self.rounds
        if not deploy_only:
            self.env.step(server_round)
        if self.live:
            self.live.round = server_round
            self.live.note(f"round {server_round} started" + (" (deploy only)" if deploy_only else ""))
        pairs = super().configure_fit(server_round, parameters, client_manager)
        out = []
        for proxy, ins in pairs:
            cfg = dict(ins.config)
            cfg["round"] = server_round
            cfg["deploy_only"] = int(deploy_only)
            if self.cid_of(proxy) == "board":
                s = self.env.observe("board")
                cfg.update({"battery_soc": s["battery_soc"], "uplink_kbps": s["uplink_kbps"], "deadline_s": s["deadline_s"]})
            out.append((proxy, FitIns(ins.parameters, cfg)))
        return out

    def aggregate_fit(self, server_round, results, failures):
        if failures:
            print("failures", failures, flush=True)
            raise RuntimeError(f"{len(failures)} client(s) failed in round {server_round}; see board_client.log")
        g = self.global_params
        accepted = []
        for proxy, res in results:
            cid, m = self.cid_of(proxy), dict(res.metrics)
            wire = sum(len(t) for t in res.parameters.tensors)
            row = {"round": server_round, "client": cid, "real": cid == "board", "samples": res.num_examples,
                   "wire_bytes": wire, **m}
            arr = parameters_to_ndarrays(res.parameters)[0]
            if res.num_examples and arr.size:
                delta, bits = decode(arr.tobytes())
                st = self.env.settle(cid, res.num_examples, int(m.get("local_epochs", 1)), arr.size)
                row.update({"decoded_bits": bits, "upload_s_sim": round(st.upload_s, 3), "accepted": st.accepted})
                if st.accepted:
                    accepted.append(([p + d for p, d in zip(g, delta)], res.num_examples))
            self.rows.append(row)
            if cid == "board":
                line = board_line(server_round, m, wire)
                print(">>> " + line, flush=True)
                if self.live:
                    self.live.note(line)
        if server_round > self.rounds or not accepted:
            return ndarrays_to_parameters(g), {}
        tot = sum(n for _, n in accepted)
        new = [sum(p[i] * n for p, n in accepted) / tot for i in range(len(g))]
        new = [a.astype(np.float32) for a in new]
        q = npu_head.quantize_head(new[0], new[1], self.pm, self.calib)
        (self.out / "models").mkdir(exist_ok=True)
        (self.out / f"models/classifier_r{server_round}.tflite").write_bytes(npu_head.patch(self.base, self.pm, q))
        return ndarrays_to_parameters(new), {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board-location", default="NACTI : lebec_CA-42")
    ap.add_argument("--sim-clients", type=int, default=8)
    ap.add_argument("--min-sim-samples", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), default="constrained")
    ap.add_argument("--load", default="{}", help="JSON: round -> background phases on the board")
    ap.add_argument("--port", type=int, default=39672)
    ap.add_argument("--host", default=f"root@{os.environ.get('BOARD_HOST', '192.168.1.12')}",
                    help="board ssh target; default root@$BOARD_HOST")
    ap.add_argument("--no-board", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--live-file", default="runs/fl_npu/live.txt",
                    help="local file with live board telemetry and per-round decisions; follow it with tail -f")
    ap.add_argument("--no-live", action="store_true", help="do not stream live board telemetry")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))
    classes = cfg["models"]["classifier"]["classes"]
    idx = {c: i for i, c in enumerate(classes)}
    fl = ROOT / "runs/fl_npu"
    pm = npu_head.load_map(fl / "patch_map.json")
    base = (fl / "classifier_fl.tflite").read_bytes()
    calib = np.load(fl / "calib_features_q.npy")
    fq = np.load(ROOT / "data/features/qat_int8_q.npy")
    meta = pd.read_parquet(ROOT / "data/features/qat_int8_meta.parquet").reset_index(drop=True)
    x = npu_head.dequant_features(fq, pm)
    y_all = meta.label.map(idx).to_numpy()
    te = ((meta.split == "test") & (meta.verified != "n")).to_numpy()
    tr = (meta.split == "train").to_numpy()
    h = np.load(fl / "init_head.npz")
    init = [h["W"].astype(np.float32), h["b"].astype(np.float32)]

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = ROOT / f"runs/fl_npu/run_{ts}"
    out.mkdir(parents=True)
    rng = np.random.default_rng(args.seed)
    counts = meta[tr].location_id.value_counts()
    cand = sorted(c for c, n in counts.items() if n >= args.min_sim_samples and c != args.board_location)
    chosen = sorted(rng.choice(cand, size=args.sim_clients, replace=False))
    ids = list(chosen) + ([] if args.no_board else ["board"])
    env = ResourceEnv(ids, EnvConfig(bw_median_kbps=SCENARIOS[args.scenario], train_s_per_sample_epoch=3.8e-5,
                                     seed=args.seed))
    n_params = 1280 * len(classes) + len(classes)
    names, cm = {}, SimpleClientManager()
    for j, c in enumerate(chosen):
        m = (meta.location_id == c).to_numpy() & tr
        yy = y_all[m]
        flip = rng.random(len(yy)) < 0.1
        yy = np.where(flip, rng.integers(0, len(classes), len(yy)), yy)
        pol = ResourceAwareController(ControllerConfig(), n_params)
        cl = PackedHeadClient(c, x[m], yy, len(classes), args.lr, 0.9, 32, args.seed + j, pol,
                         (lambda cfg_, cid=c: env.observe(cid)))
        names[f"sim{j:02d}"] = c
        cm.register(run_fl_real.InProcessProxy(f"sim{j:02d}", cl))

    if not args.no_board:
        stage = out / "board_stage"
        bs = ROOT / "runs/fl_npu/board_stage_src"  # scripts/stage_fl_board.py --out runs/fl_npu/board_stage_src
        (stage / "data").mkdir(parents=True)
        for f in ("classifier_fl.tflite", "patch_map.json", "calib_features_q.npy"):
            shutil.copy(fl / f, stage / f)
        shutil.copytree(bs / "crops", stage / "data/crops")
        shutil.copy(bs / "manifest.json", stage / "data/manifest.json")
        man = json.load(open(bs / "manifest.json"))
        pos = {cid: i for i, cid in enumerate(meta.crop_id)}
        np.save(stage / "data/ref_features_q.npy", fq[[pos[r["file"][:-4]] for r in man["crops"]]])
        shutil.copytree(ROOT / "runs/board_stage", stage / "eval")
        ssh = ["ssh", "-o", "ProxyCommand=nc -b en0 %h %p", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
        tarball = out / "board_stage.tar"
        with tarfile.open(tarball, "w") as t:
            t.add(stage, arcname="flnpu", filter=lambda ti: None if Path(ti.name).name.startswith("._") else ti)
        subprocess.run(ssh + [args.host, "mkdir -p ~/rafl/rafl/edge && tar -C ~/rafl -xf -"],
                       stdin=open(tarball, "rb"), check=True)
        code = out / "code.tar"
        with tarfile.open(code, "w") as t:
            for f in ("__init__.py", "fl/__init__.py", "fl/sim.py", "fl/real.py", "fl/quant.py", "fl/controller.py",
                      "fl/npu_head.py", "fl/packed_client.py", "fl/board_npu_client.py", "edge/__init__.py",
                      "edge/telemetry.py"):
                t.add(ROOT / "src/rafl" / f, arcname=f"rafl/{f}")
            for f in ("telemetry.py", "energy_phases.py", "run_frames.py", "node_io.py"):
                t.add(ROOT / "src/rafl/edge" / f, arcname=f)
        subprocess.run(ssh + [args.host, "tar -C ~/rafl -xf -"], stdin=open(code, "rb"), check=True)
        tarball.unlink()
        code.unlink()

    rows, evals = [], []

    def eval_fn(rnd, params, _):
        W, b = [np.asarray(p, np.float32) for p in params]
        q = npu_head.quantize_head(W, b, pm, calib)
        pred = npu_head.emulate(fq[te], q, pm).argmax(1)
        yt = y_all[te]
        rec = {c: float((pred[yt == i] == i).mean()) for i, c in enumerate(classes)}
        row = {"round": rnd, "test_bacc_float": metrics([W, b], x[te], yt, classes)["bacc"],
               "test_bacc_int8": float(np.mean(list(rec.values()))), **{f"int8_recall_{c}": v for c, v in rec.items()},
               "head_digest": npu_head.digest(npu_head.patch(base, pm, q))}
        evals.append(row)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
        return 0.0, {}

    n = len(ids)
    live = None
    if not args.no_board and not args.no_live:
        live = LiveTelemetry(args.host, ROOT / args.live_file)
        print(f"live board telemetry: tail -f {args.live_file}", flush=True)
    strat = NpuFedAvg(names, env, rows, out, base, pm, calib, args.rounds, live=live, fraction_fit=1.0, fraction_evaluate=0.0,
                      min_fit_clients=n, min_available_clients=n, min_evaluate_clients=0, evaluate_fn=eval_fn,
                      initial_parameters=ndarrays_to_parameters(init))
    addr = f"127.0.0.1:{args.port}"
    proc = None
    if not args.no_board:
        subprocess.run(["ssh", "-o", "ProxyCommand=nc -b en0 %h %p", "-o", "BatchMode=yes", args.host,
                        "rm -f ~/rafl/flnpu/board_features_q.npy"], check=True)
        cmd = ("cd ~/rafl && PYTHONPATH=$HOME/rafl:$HOME/rafl/pylib python3 -u -m rafl.fl.board_npu_client "
               f"--server {addr} --root ~/rafl/flnpu --lr {args.lr} --load '{args.load}'")
        proc = subprocess.Popen(
            ["ssh", "-o", "ProxyCommand=nc -b en0 %h %p", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
             "-o", "ExitOnForwardFailure=yes", "-R", f"127.0.0.1:{args.port}:127.0.0.1:{args.port}",
             args.host, cmd], stdout=open(out / "board_client.log", "w"), stderr=subprocess.STDOUT)

    def snap():
        r = subprocess.run(["lsof", "-nP", "-iTCP", "-sTCP:LISTEN"], capture_output=True, text=True)
        (out / "lsof.txt").write_text("\n".join(l for l in r.stdout.splitlines()
                                                if l.startswith("COMMAND") or f":{args.port} " in l) + "\n")
    threading.Timer(3.0, snap).start()
    try:
        start_server(server_address=addr, config=ServerConfig(num_rounds=args.rounds + 1, round_timeout=900),
                     strategy=strat, client_manager=cm)
    finally:
        if live:
            live.note("run finished")
            live.close()
    if proc:
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.terminate()
        with open(out / "board_features_q.npy", "wb") as f:
            subprocess.run(["ssh", "-o", "ProxyCommand=nc -b en0 %h %p", "-o", "BatchMode=yes", args.host,
                            "cat ~/rafl/flnpu/board_features_q.npy"], stdout=f, check=True)
        ref = np.load(out / "board_stage/data/ref_features_q.npy").astype(int)
        bf = np.load(out / "board_features_q.npy").astype(int)
        board_digests = {int(r["round"]): r.get("model_digest") for r in rows if r["client"] == "board"}
        server_digests = {int(e["round"]): e["head_digest"] for e in evals}
        check = {"features_equal": float((bf == ref).mean()), "features_within_1lsb": float((abs(bf - ref) <= 1).mean()),
                 "board_runs_server_head": {r: board_digests[r] == server_digests.get(r - 1) for r in sorted(board_digests)}}
        last = max(board_digests)
        mac_sha = hashlib.sha256((out / f"models/classifier_r{args.rounds}.tflite").read_bytes()).hexdigest()
        r = subprocess.run(["ssh", "-o", "ProxyCommand=nc -b en0 %h %p", "-o", "BatchMode=yes", args.host,
                            f"sha256sum ~/rafl/flnpu/models/classifier_r{last}.tflite"], capture_output=True, text=True)
        board_sha = r.stdout.split()[0] if r.stdout else ""
        check["final_sha256"] = {"mac": mac_sha, "board": board_sha, "equal": mac_sha == board_sha}
        json.dump(check, open(out / "loop_check.json", "w"), indent=1)
        print("loop check", json.dumps({k: v for k, v in check.items() if k != "final_sha256"}), flush=True)
        print(f"final model sha256  Mac   classifier_r{args.rounds}.tflite: {mac_sha}\n"
              f"                    board classifier_r{last}.tflite: {board_sha}\n"
              f"                    {'IDENTICAL' if mac_sha == board_sha else 'DIFFERENT'}", flush=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("round", "client"), k))
    with open(out / "clients.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    pd.DataFrame(evals).to_csv(out / "global.csv", index=False)
    json.dump({"args": vars(args), "sim_clients": list(chosen), "env_summary": env.summary()},
              open(out / "config.json", "w"), indent=2)
    print("results", out)


if __name__ == "__main__":
    main()
