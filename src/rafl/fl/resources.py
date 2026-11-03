"""Simulated per-camera resource environment for the FL simulation.

One round = one simulated hour. Each client is a remote camera with:
  - a battery charged by a solar panel (diurnal, per-site weather) and drained
    by the always-on detection workload plus any FL work it does;
  - an uplink whose bandwidth is site-specific (lognormal) and varies per round;
  - CPU load from the detection workload (bursty: animal activity);
  - SoC temperature from ambient (diurnal) plus load.

The environment is applied identically to every policy: a client whose battery
is empty is offline (cannot detect, cannot train), and an upload that does not
finish within the round deadline is discarded by the server (its bytes and
energy are still spent). Policies only see `observe()`, which reports the
previous round's measured bandwidth, not the true bandwidth of this round.

Numbers marked ASSUMED are placeholders until Phase 7 measures them on the
board. Measured: idle SoC temperature 38-41 C (Phase 4).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EnvConfig:
    battery_wh: float = 20.0            # ASSUMED small camera pack
    base_draw_w: float = 0.40           # ASSUMED duty-cycled detection average
    solar_peak_w: float = 2.5           # ASSUMED small panel, before site/weather factor
    wake_overhead_wh: float = 0.004     # ASSUMED CPU wake + model load per FL round (~5 s at 3 W)
    train_w: float = 2.0                # ASSUMED extra A53 power while training
    train_s_per_sample_epoch: float = 2e-4  # ASSUMED head-only SGD on cached features
    radio_w: float = 1.2                # ASSUMED cellular/sat modem while transmitting
    bw_median_kbps: float = 48.0        # ASSUMED remote-site uplink, lognormal across sites
    bw_site_sigma: float = 1.2
    bw_round_sigma: float = 0.5
    deadline_s: float = 12.0            # round upload deadline
    ambient_c: float = 15.0
    idle_temp_c: float = 39.0           # MEASURED on the board (Phase 4)
    throttle_c: float = 85.0
    start_hour: int = 6
    seed: int = 7


@dataclass
class ClientRes:
    soc: float                 # 0..1
    site_bw_kbps: float
    solar_factor: float        # site shading x weather, 0..1
    heat_factor: float         # enclosure: extra degrees at full load
    last_bw_kbps: float
    cpu: float = 0.1
    temp_c: float = 39.0
    offline_rounds: int = 0
    energy_fl_wh: float = 0.0


@dataclass
class Settle:
    accepted: bool
    upload_s: float
    energy_wh: float


class ResourceEnv:
    def __init__(self, client_ids, cfg: EnvConfig | None = None):
        self.cfg = cfg or EnvConfig()
        c = self.cfg
        rng = np.random.default_rng(c.seed)
        self.rng = np.random.default_rng(c.seed + 1)
        self.c: dict[str, ClientRes] = {}
        for cid in sorted(client_ids):
            bw = float(c.bw_median_kbps * np.exp(rng.normal(0, c.bw_site_sigma)))
            self.c[cid] = ClientRes(soc=float(rng.uniform(0.4, 0.9)), site_bw_kbps=bw,
                                    solar_factor=float(rng.beta(2.0, 1.5)),
                                    heat_factor=float(rng.uniform(15, 45)), last_bw_kbps=bw)
        self.rnd = 0
        self._bw: dict[str, float] = {}

    # ---- time ----
    def hour(self, rnd: int) -> int:
        return (self.cfg.start_hour + rnd) % 24

    def _solar_w(self, rnd: int, r: ClientRes) -> float:
        h = self.hour(rnd)
        sun = max(0.0, np.sin(np.pi * (h - 6) / 12)) if 6 <= h <= 18 else 0.0
        weather = self.rng.uniform(0.3, 1.0)
        return self.cfg.solar_peak_w * r.solar_factor * sun * weather

    def step(self, rnd: int):
        """Advance every camera by one hour of background operation."""
        c = self.cfg
        self.rnd = rnd
        h = self.hour(rnd)
        ambient = c.ambient_c + 8 * np.sin(np.pi * (h - 9) / 12)
        for cid, r in self.c.items():
            # bursty detection load: higher at dawn/dusk (animal activity)
            act = 0.5 if h in (5, 6, 7, 18, 19, 20, 21) else 0.15
            r.cpu = float(np.clip(self.rng.beta(2, 8) + (self.rng.random() < act) * self.rng.uniform(0.3, 0.7), 0, 1))
            r.temp_c = float(max(c.idle_temp_c, ambient + 20) + r.heat_factor * r.cpu)
            r.soc = float(np.clip(r.soc + (self._solar_w(rnd, r) - c.base_draw_w) / c.battery_wh, 0, 1))
            if r.soc <= 0.0:
                r.offline_rounds += 1
            self._bw[cid] = float(r.site_bw_kbps * np.exp(self.rng.normal(0, c.bw_round_sigma)))

    # ---- what a policy (and the real board in Phase 7) can observe ----
    def observe(self, cid: str) -> dict:
        r = self.c[cid]
        return {"battery_soc": r.soc, "uplink_kbps": r.last_bw_kbps, "cpu_load": r.cpu,
                "temp_c": r.temp_c, "online": r.soc > 0.0, "deadline_s": self.cfg.deadline_s,
                "train_s_per_sample_epoch": self.cfg.train_s_per_sample_epoch}

    def online(self, cid: str) -> bool:
        return self.c[cid].soc > 0.0

    # ---- account for an FL round the client actually ran ----
    def settle(self, cid: str, n_samples: int, local_epochs: int, nbytes: int) -> Settle:
        c, r = self.cfg, self.c[cid]
        slow = 2.0 if r.temp_c >= c.throttle_c else 1.0
        train_s = n_samples * local_epochs * c.train_s_per_sample_epoch * slow
        bw = self._bw[cid]
        upload_s = nbytes * 8 / (bw * 1000)
        r.last_bw_kbps = bw  # measured during this upload, seen by the policy next time
        e = c.wake_overhead_wh + (train_s * c.train_w + upload_s * c.radio_w) / 3600
        r.soc = float(max(0.0, r.soc - e / c.battery_wh))
        r.energy_fl_wh += e
        return Settle(accepted=(train_s + upload_s) <= c.deadline_s, upload_s=upload_s, energy_wh=e)

    def summary(self) -> dict:
        n = len(self.c) * max(self.rnd, 1)
        return {"camera_offline_pct": 100 * sum(r.offline_rounds for r in self.c.values()) / n,
                "fl_energy_wh": sum(r.energy_fl_wh for r in self.c.values()),
                "mean_soc_end": float(np.mean([r.soc for r in self.c.values()]))}
