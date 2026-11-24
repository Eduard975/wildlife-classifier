"""Read-only resource telemetry for the board (Phase 7). Standard library only.

Reads /proc and /sys, never writes to them:
  CPU utilisation (from /proc/stat deltas), load average, per-zone temperature,
  CPU frequency, available memory, and bytes sent/received on the network interface.

`Telemetry.observe()` returns the same keys as `rafl.fl.resources.ResourceEnv.observe()`,
so `ResourceAwareController` can consume a real board reading unchanged. The board runs
on mains power, so `battery_soc` comes from the caller (a simulated battery), and
`uplink_kbps` is the throughput of the last real upload, reported with `record_upload()`.

  python3 telemetry.py --hz 1 --duration 60 --out trace.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import threading
import time
from pathlib import Path


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def cpu_times() -> list[list[int]]:
    """Per-line jiffies from /proc/stat: index 0 is the total, then one entry per core."""
    rows = []
    for line in (_read("/proc/stat") or "").splitlines():
        if line.startswith("cpu"):
            rows.append([int(v) for v in line.split()[1:]])
    return rows


def _util(a: list[int], b: list[int]) -> float:
    idle = (b[3] + b[4]) - (a[3] + a[4])
    total = sum(b) - sum(a)
    return 1.0 - idle / total if total > 0 else 0.0


def temps_c() -> dict[str, float]:
    out = {}
    for z in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        t, name = _read(z + "/temp"), _read(z + "/type")
        if t is not None:
            out[(name or Path(z).name).strip()] = int(t) / 1000
    return out


def cpu_freq_mhz() -> list[int]:
    out = []
    for p in sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq")):
        v = _read(p)
        if v is not None:
            out.append(int(v) // 1000)
    return out


def mem_available_mb() -> float:
    for line in (_read("/proc/meminfo") or "").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024
    return float("nan")


def net_bytes(iface: str) -> tuple[int, int]:
    for line in (_read("/proc/net/dev") or "").splitlines():
        name, _, rest = line.partition(":")
        if name.strip() == iface:
            v = rest.split()
            return int(v[0]), int(v[8])
    return 0, 0


def load_avg() -> float:
    v = _read("/proc/loadavg")
    return float(v.split()[0]) if v else float("nan")


class Telemetry:
    def __init__(self, iface: str = "mlan0", deadline_s: float = 12.0,
                 train_s_per_sample_epoch: float = 7.5e-4):
        self.iface = iface
        self.deadline_s = deadline_s
        self.train_s_per_sample_epoch = train_s_per_sample_epoch
        self.last_uplink_kbps: float | None = None
        self._cpu = cpu_times()
        self._net = net_bytes(iface)
        self._t = time.monotonic()

    def record_upload(self, nbytes: int, seconds: float) -> None:
        if seconds > 0:
            self.last_uplink_kbps = nbytes * 8 / seconds / 1000

    def sample(self) -> dict:
        """One reading. CPU utilisation and network rates cover the time since the previous call."""
        now, cpu, net = time.monotonic(), cpu_times(), net_bytes(self.iface)
        dt = max(now - self._t, 1e-6)
        util = [_util(a, b) for a, b in zip(self._cpu, cpu)]
        s = {"wall": time.time(), "cpu_util": round(util[0], 4) if util else 0.0,
             "cpu_util_cores": [round(u, 3) for u in util[1:]], "load1": load_avg(),
             "temp_c": temps_c(), "freq_mhz": cpu_freq_mhz(), "mem_avail_mb": round(mem_available_mb(), 1),
             "rx_kbps": round((net[0] - self._net[0]) * 8 / dt / 1000, 2),
             "tx_kbps": round((net[1] - self._net[1]) * 8 / dt / 1000, 2)}
        self._cpu, self._net, self._t = cpu, net, now
        return s

    def observe(self, battery_soc: float = 1.0, n_samples: int = 0) -> dict:
        """Controller input in the format of ResourceEnv.observe()."""
        s = self.sample()
        uplink = self.last_uplink_kbps if self.last_uplink_kbps is not None else s["tx_kbps"]
        return {"battery_soc": battery_soc, "uplink_kbps": uplink, "cpu_load": s["cpu_util"],
                "temp_c": max(s["temp_c"].values(), default=0.0), "online": battery_soc > 0.0,
                "deadline_s": self.deadline_s, "train_s_per_sample_epoch": self.train_s_per_sample_epoch,
                "n_samples": n_samples, "mem_avail_mb": s["mem_avail_mb"]}


class Recorder(threading.Thread):
    """Background sampler writing one JSON line per sample, tagged with the current phase."""

    def __init__(self, path, hz: float = 1.0, iface: str = "mlan0"):
        super().__init__(daemon=True)
        self.f = open(path, "w")
        self.period = 1.0 / hz
        self.tel = Telemetry(iface)
        self.phase = "start"
        self._halt = threading.Event()

    def run(self):
        while not self._halt.wait(self.period):
            s = self.tel.sample()
            s["phase"] = self.phase
            self.f.write(json.dumps(s) + "\n")
            self.f.flush()

    def stop(self):
        self._halt.set()
        self.join(timeout=2 * self.period + 1)
        self.f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hz", type=float, default=1.0)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--iface", default="mlan0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rec = Recorder(Path(args.out).expanduser(), args.hz, args.iface)
    rec.start()
    try:
        time.sleep(args.duration)
    finally:
        rec.stop()


if __name__ == "__main__":
    main()
