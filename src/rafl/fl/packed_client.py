"""Flower client that trains the classifier head of the deployed int8 model and uploads a packed,
quantized update (rafl.fl.quant.encode). The bit width and the decision to skip come from a
policy (rafl.fl.controller) applied to the client's resource state for this round.

Upload format: one uint8 array holding the packed payload, or an empty array when skipping.
"""
from __future__ import annotations

import time

import numpy as np
from flwr.client import NumPyClient

from rafl.fl.quant import ErrorFeedback
from rafl.fl.sim import RoundPlan, sgd


def skip_reason(s: dict, cfg) -> str:
    if s["battery_soc"] < cfg.soc_reserve:
        return "battery"
    if s["cpu_load"] > cfg.cpu_busy:
        return "cpu_busy"
    if s["temp_c"] >= cfg.temp_limit_c:
        return "temperature"
    return "deadline"


class PackedHeadClient(NumPyClient):
    def __init__(self, cid, x, y, k, lr, momentum, batch, seed, policy, observe, epochs=1):
        self.cid, self.x, self.y, self.k = cid, x, y, k
        self.lr, self.momentum, self.batch, self.epochs = lr, momentum, batch, epochs
        self.rng = np.random.default_rng(seed)
        self.ef = ErrorFeedback()
        self.policy, self.observe = policy, observe

    def before_round(self, parameters, config) -> dict:
        """Hook for the board: deploy the received head on the NPU, evaluate it."""
        return {}

    def fit(self, parameters, config):
        rnd = int(config.get("round", 0))
        extra = self.before_round(parameters, config)
        if config.get("deploy_only"):
            return [np.zeros(0, np.uint8)], 0, {**extra, "skipped": 1, "deploy_only": 1, "bits": 0, "payload_bytes": 0}
        state = {**self.observe(config), "n_samples": len(self.y)}
        plan: RoundPlan = self.policy(rnd, self.cid, state)
        m = {**extra, "cpu_load": float(state["cpu_load"]), "temp_c": float(state["temp_c"]),
             "battery_soc": float(state["battery_soc"]), "uplink_kbps": float(state["uplink_kbps"])}
        if not plan.participate:
            m.update({"skipped": 1, "bits": 0, "payload_bytes": 0,
                      "reason": skip_reason(state, self.policy.cfg) if hasattr(self.policy, "cfg") else "policy"})
            return [np.zeros(0, np.uint8)], 0, m
        t0 = time.perf_counter()
        new = sgd(parameters, self.x, self.y, self.k, plan.local_epochs * self.epochs, self.lr,
                  self.momentum, self.batch, self.rng)
        train_s = time.perf_counter() - t0
        delta = [n - p for n, p in zip(new, parameters)]
        payload = self.ef.compress_packed(delta, plan.bits, self.rng)
        m.update({"skipped": 0, "bits": plan.bits, "payload_bytes": len(payload), "train_s": train_s,
                  "local_epochs": plan.local_epochs * self.epochs})
        return [np.frombuffer(payload, np.uint8).copy()], len(self.y), m
