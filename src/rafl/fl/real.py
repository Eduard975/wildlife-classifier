"""Pieces shared by the real-device FL deployment: instrumented head client, feature standardisation, label noise."""
from __future__ import annotations

import glob
import time

import numpy as np

from rafl.fl.sim import HeadClient


def temps_c() -> list[float]:
    out = []
    for p in sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp")):
        try:
            out.append(int(open(p).read()) / 1000)
        except OSError:
            pass
    return out


def cpu_busy() -> tuple[float, float]:
    try:
        v = [float(a) for a in open("/proc/stat").readline().split()[1:]]
    except OSError:
        return 0.0, 0.0
    return v[3] + v[4], sum(v)


def standardize(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((x.astype(np.float32) - mu) / sd).astype(np.float32)


def noisy_labels(y: np.ndarray, k: int, rate: float, rng: np.random.Generator) -> np.ndarray:
    flip = rng.random(len(y)) < rate
    return np.where(flip, rng.integers(0, k, len(y)), y)


class TimedHeadClient(HeadClient):
    """HeadClient that reports local wall time, process CPU time, system CPU utilisation and temperature."""

    def fit(self, parameters, config):
        i0, t0 = cpu_busy()
        w0, c0 = time.perf_counter(), time.process_time()
        temp0 = max(temps_c() or [0.0])
        upload, n, m = super().fit(parameters, config)
        w1, c1 = time.perf_counter(), time.process_time()
        i1, t1 = cpu_busy()
        m = dict(m)
        m.update({"train_s": w1 - w0, "cpu_s": c1 - c0,
                  "sys_cpu_util": 1.0 - (i1 - i0) / (t1 - t0) if t1 > t0 else float("nan"),
                  "temp_start_c": temp0, "temp_end_c": max(temps_c() or [0.0])})
        return upload, n, m
