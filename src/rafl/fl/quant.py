"""Update quantization for client -> server uploads.

Per-tensor stochastic uniform quantization of the model *delta* to b bits, with
optional error feedback (the quantization residual is carried to the next round
so the bias does not accumulate). b=32 means no quantization (fp32 upload).
"""
from __future__ import annotations

import numpy as np


def quantize(x: np.ndarray, bits: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Return (dequantized x, payload bytes). Unbiased: E[q(x)] = x."""
    if bits >= 32:
        return x.astype(np.float32), x.size * 4
    if bits == 16:
        return x.astype(np.float16).astype(np.float32), x.size * 2
    lo, hi = float(x.min()), float(x.max())
    levels = (1 << bits) - 1
    if hi == lo:
        return np.full_like(x, lo, dtype=np.float32), 8
    scale = (hi - lo) / levels
    v = (x - lo) / scale
    q = np.floor(v + rng.random(x.shape))  # stochastic rounding
    q = np.clip(q, 0, levels)
    # payload: packed codes + lo/scale as two fp32
    return (q * scale + lo).astype(np.float32), int(np.ceil(x.size * bits / 8)) + 8


class ErrorFeedback:
    """Per-client residual memory, one array per parameter tensor."""

    def __init__(self):
        self.residual: list[np.ndarray] | None = None

    def compress(self, delta: list[np.ndarray], bits: int, rng: np.random.Generator,
                 enabled: bool = True) -> tuple[list[np.ndarray], int]:
        if enabled and self.residual is not None:
            delta = [d + r for d, r in zip(delta, self.residual)]
        out, nbytes = [], 0
        for d in delta:
            q, b = quantize(d, bits, rng)
            out.append(q)
            nbytes += b
        if enabled:
            self.residual = [d - q for d, q in zip(delta, out)]
        return out, nbytes
