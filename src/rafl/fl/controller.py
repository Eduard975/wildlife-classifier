"""Resource-aware dynamic quantization controller (Phase 6).

Per client per round, from observed resources only:
  1. Skip if the camera must protect its detection duty: battery below the
     reserve, CPU busy with detections, or SoC at the thermal throttle point.
  2. Choose the update bit width: the largest width in `bit_widths` (capped at
     `max_bits`) whose predicted train + upload time fits the round deadline
     with a safety margin, using the bandwidth measured on the last upload.
  3. Skip if even the smallest width is predicted to miss the deadline: a
     discarded upload costs bytes and energy and contributes nothing.
  4. Optionally run more local epochs when the battery is high and the CPU idle.

`max_bits=4` follows Phase 5: 4-bit updates matched FP32 accuracy for the
classifier head, so wider widths only cost bytes.
"""
from __future__ import annotations

from dataclasses import dataclass

from rafl.fl.sim import RoundPlan


@dataclass
class ControllerConfig:
    bit_widths: tuple[int, ...] = (2, 4, 8)
    max_bits: int = 4
    soc_reserve: float = 0.25
    cpu_busy: float = 0.7
    temp_limit_c: float = 85.0
    margin: float = 0.6            # use at most this fraction of the deadline
    use_skip: bool = True
    use_bits: bool = True
    adapt_epochs: bool = False
    fixed_bits: int = 4            # used when use_bits is False


def payload_bytes(n_params: int, bits: int) -> int:
    if bits >= 32:
        return n_params * 4
    if bits == 16:
        return n_params * 2
    return -(-n_params * bits // 8) + 8


class ResourceAwareController:
    def __init__(self, cfg: ControllerConfig, n_params: int, n_tensors: int = 2):
        self.cfg, self.n_params, self.n_tensors = cfg, n_params, n_tensors

    def _bytes(self, bits: int) -> int:
        # one (lo, scale) header per tensor; payload_bytes adds one already
        return payload_bytes(self.n_params, bits) + 8 * (self.n_tensors - 1) * (bits < 16)

    def __call__(self, rnd: int, cid: str, s: dict) -> RoundPlan:
        c = self.cfg
        if not s:
            return RoundPlan(True, c.fixed_bits, 1)
        n_samples = s.get("n_samples", 0)
        if c.use_skip and (s["battery_soc"] < c.soc_reserve or s["cpu_load"] > c.cpu_busy
                           or s["temp_c"] >= c.temp_limit_c):
            return RoundPlan(False, 0, 0)
        epochs = 2 if (c.adapt_epochs and s["battery_soc"] > 0.7 and s["cpu_load"] < 0.3) else 1
        budget = c.margin * s["deadline_s"] - n_samples * epochs * s["train_s_per_sample_epoch"]
        kbps = max(s["uplink_kbps"], 1e-3)
        if not c.use_bits:
            fits = self._bytes(c.fixed_bits) * 8 / (kbps * 1000) <= budget
            return RoundPlan(fits or not c.use_skip, c.fixed_bits, epochs)
        widths = sorted(b for b in c.bit_widths if b <= c.max_bits)
        for b in reversed(widths):
            if self._bytes(b) * 8 / (kbps * 1000) <= budget:
                return RoundPlan(True, b, epochs)
        return RoundPlan(not c.use_skip, widths[0], epochs)
