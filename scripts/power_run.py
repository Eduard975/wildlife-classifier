"""Measured energy per operation on the board: NXP BCU reads the PEVK's power monitors over the debug USB
while the board runs one operation per phase (src/rafl/edge/energy_phases.py).

The board must be connected to the Mac with the debug USB (micro-USB "DEBUG" port; the Mac then sees an FTDI
"Quad RS232-HS"). BCU is built for Apple Silicon in tools/bcu/ (tools/bcu/VERSION.txt).

  .venv/bin/python scripts/power_run.py                       # all phases, 60 s each
  .venv/bin/python scripts/power_run.py --phases idle,npu_detector,cpu_detector_fp32 --phase-s 30
  .venv/bin/python scripts/power_run.py --sudo                # if BCU cannot claim the FTDI device without root
  .venv/bin/python scripts/power_run.py --analyze runs/board/power_<ts>   # recompute the tables only

Power: vsys_5v is the board's 5 V input (the whole board); vdd_arm the Cortex-A53 cores; vdd_soc the rest of
the SoC; the DRAM rails are summed as dram. Energy per operation = (phase power - idle power) x phase time /
operations, so it is the extra energy of the operation over an idle board. Writes bcu.csv, phases.json,
energy.json and energy.md to runs/board/power_<ts>/.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BCU = ROOT / "tools/bcu/bcu_mac"
RAILS = {"board_5v": ["vsys_5v"], "arm": ["vdd_arm"], "soc": ["vdd_soc"], "dram": ["lpd4_vdd1", "lpd4_vdd2", "lpd4_vddq", "nvcc_dram_1v1"]}
ALL = ("idle,npu_classifier,npu_detector,npu_detector_pc,cpu_detector_fp32,npu_pipeline,cpu_pipeline,"
       "fl_head_train,fl_block_train,requant_block,npu_compile,wifi_tx")


def ssh_cmd(host, iface):
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15",
            "-o", f"ProxyCommand=nc -b {iface} %h %p", host]


def clock_offset(ssh):
    """board clock - Mac clock, from the midpoint of three round trips (seconds)."""
    best = None
    for _ in range(3):
        t0 = time.time()
        b = float(subprocess.run(ssh + ["date +%s.%N"], capture_output=True, text=True, check=True).stdout)
        t1 = time.time()
        if best is None or t1 - t0 < best[0]:
            best = (t1 - t0, b - (t0 + t1) / 2)
    return best[1], best[0]


def read_bcu(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    head, data = rows[0], [r for r in rows[1:] if r and r[0].strip().lstrip("-").isdigit()]
    t = np.array([float(r[0]) for r in data]) / 1000.0
    rails = {}
    for j, h in enumerate(head):
        if h.endswith(" voltage(V)") and j + 1 < len(head) and head[j + 1].endswith(" current(mA)"):
            name = h[: -len(" voltage(V)")]
            v = np.array([float(r[j]) for r in data])
            a = np.array([float(r[j + 1]) for r in data])
            rails[name] = v * a / 1000.0  # W
    return t, rails


def analyze(run: Path, margin_s=3.0):
    meta = json.load(open(run / "power_meta.json"))
    phases = json.load(open(run / "phases.json"))["phases"]
    t, rails = read_bcu(run / "bcu.csv")
    t_board = meta["bcu_start_mac"] + t + meta["clock_offset_s"]
    groups = {g: sum(rails[r] for r in rs if r in rails) for g, rs in RAILS.items() if any(r in rails for r in rs)}
    res = []
    for p in phases:
        m = (t_board >= p["start"] + margin_s) & (t_board <= p["end"] - 1.0)
        row = {"phase": p["phase"], "samples": int(m.sum()), "duration_s": p["duration_s"], "count": p["count"],
               "unit": p["unit"], "ms_per_op": p.get("ms_per_op")}
        for g, w in groups.items():
            row[f"{g}_w"] = round(float(w[m].mean()), 4) if m.any() else None
        res.append(row)
    idle = next((r for r in res if r["phase"] == "idle"), None)
    for r in res:
        if idle and r["phase"] != "idle" and r.get("board_5v_w") is not None and r["ms_per_op"]:
            extra = r["board_5v_w"] - idle["board_5v_w"]
            r["extra_w"] = round(extra, 4)
            r["mj_per_op_extra"] = round(extra * r["ms_per_op"], 3)          # W x ms = mJ
            r["mj_per_op_total"] = round(r["board_5v_w"] * r["ms_per_op"], 3)
    json.dump({"meta": meta, "phases": res}, open(run / "energy.json", "w"), indent=1)
    cols = ["phase", "ms_per_op", "board_5v_w", "arm_w", "soc_w", "dram_w", "extra_w", "mj_per_op_extra", "mj_per_op_total", "unit"]
    with open(run / "energy.md", "w") as f:
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in res:
            f.write("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |\n")
    print((run / "energy.md").read_text())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=f"root@{os.environ.get('BOARD_HOST', '192.168.1.12')}")
    ap.add_argument("--iface", default="en0")
    ap.add_argument("--board-type", default="auto", help="BCU board name (imx8mpevkpwra0, imx8mpevkpwra1, ...) or auto")
    ap.add_argument("--phases", default=ALL)
    ap.add_argument("--phase-s", type=float, default=60.0)
    ap.add_argument("--gap-s", type=float, default=10.0)
    ap.add_argument("--sudo", action="store_true", help="run BCU with sudo (asks for your password in this terminal)")
    ap.add_argument("--analyze", default=None, help="only recompute energy.json/energy.md for this run folder")
    args = ap.parse_args()
    if args.analyze:
        analyze(Path(args.analyze))
        return

    pre = ["sudo"] if args.sudo else []
    ls = subprocess.run(pre + [str(BCU), "lsftdi"], capture_output=True, text=True)
    print(ls.stdout.strip().splitlines()[-2] if ls.stdout.strip() else ls.stderr)
    if "found: 0" in ls.stdout:
        sys.exit("BCU sees no board: connect the PEVK's debug USB to this Mac")
    board = ["-auto"] if args.board_type == "auto" else [f"-board={args.board_type}"]

    ssh = ssh_cmd(args.host, args.iface)
    run = ROOT / f"runs/board/power_{time.strftime('%Y%m%d_%H%M%S')}"
    run.mkdir(parents=True)
    remote = f"~/rafl/runs/{run.name}"
    files = ["src/rafl/edge/energy_phases.py", "src/rafl/edge/run_frames.py", "src/rafl/edge/node_io.py", "src/rafl/edge/telemetry.py"]
    pkg = ["rafl/__init__.py", "rafl/edge/__init__.py", "rafl/edge/requant.py", "rafl/edge/node_io.py", "rafl/fl/__init__.py",
           "rafl/fl/block.py", "rafl/fl/npu_block.py"]
    env = {**os.environ, "COPYFILE_DISABLE": "1"}
    tar = subprocess.run(["tar", "-C", str(ROOT / "src/rafl/edge"), "-cf", "-"] + [Path(f).name for f in files],
                         capture_output=True, check=True, env=env).stdout
    subprocess.run(ssh + ["tar -C ~/rafl -xf -"], input=tar, check=True)
    tar = subprocess.run(["tar", "-C", str(ROOT / "src"), "-cf", "-"] + pkg, capture_output=True, check=True, env=env).stdout
    subprocess.run(ssh + ["tar -C ~/rafl -xf -"], input=tar, check=True)
    busy = subprocess.run(ssh + ['ps -eo args | grep -E "[p]ython3|[b]enchmark_model" || true'], capture_output=True, text=True).stdout
    if busy.strip():
        sys.exit(f"board is busy, not starting:\n{busy}")

    offset, rtt = clock_offset(ssh)
    print(f"board clock - Mac clock = {offset:+.3f} s (round trip {1000 * rtt:.0f} ms)")
    log = open(run / "bcu_console.log", "w")
    t_start = time.time()
    bcu = subprocess.Popen(pre + [str(BCU), "monitor", *board, f"-dump={run / 'bcu.csv'}", "-nodisplay"],
                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(5)
    if bcu.poll() is not None:
        sys.exit(f"BCU exited early, see {run / 'bcu_console.log'}" + ("" if args.sudo else "; try --sudo"))
    json.dump({"bcu_start_mac": t_start, "clock_offset_s": offset, "rtt_s": rtt, "board": board, "phases": args.phases,
               "phase_s": args.phase_s, "gap_s": args.gap_s}, open(run / "power_meta.json", "w"), indent=1)
    cmd = (f"cd ~/rafl && D=$(find / -xdev -maxdepth 4 -name libvx_delegate.so 2>/dev/null | head -1) && "
           f"PYTHONPATH=$HOME/rafl:$HOME/rafl/pylib python3 energy_phases.py --out {remote} --delegate $D "
           f"--detector models/det_ld/int8_per_tensor.tflite --classifier models/efficientnet_lite0_int8_per_tensor.tflite "
           f"--phase-s {args.phase_s} --gap-s {args.gap_s} --phases {args.phases}")
    try:
        with open(run / "console.log", "w") as con:
            # stdout carries the wifi_tx byte stream and is discarded; progress arrives on stderr
            subprocess.run(ssh + [cmd], stdout=subprocess.DEVNULL, stderr=con, check=True)
    finally:
        os.killpg(bcu.pid, signal.SIGINT)
        try:
            bcu.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(bcu.pid, signal.SIGKILL)
    tar = subprocess.run(ssh + [f"tar -C {remote} -cf - ."], capture_output=True, check=True).stdout
    subprocess.run(["tar", "-C", str(run), "-xf", "-"], input=tar, check=True)
    analyze(run)
    print("results", run)


if __name__ == "__main__":
    main()
