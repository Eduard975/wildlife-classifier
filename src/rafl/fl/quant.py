"""Update quantization for client -> server uploads.

Per-tensor stochastic uniform quantization of the model *delta* to b bits, with
optional error feedback (the quantization residual is carried to the next round
so the bias does not accumulate). b=32 means no quantization (fp32 upload).
"""
from __future__ import annotations

import struct

import numpy as np

MAGIC = 0xA7


def quantize_codes(x: np.ndarray, bits: int, rng: np.random.Generator) -> tuple[np.ndarray, float, float]:
    """Stochastic uniform codes in [0, 2^bits - 1] with (lo, scale). Unbiased: E[codes*scale+lo] = x."""
    lo, hi = float(x.min()), float(x.max())
    levels = (1 << bits) - 1
    if hi == lo:
        return np.zeros(x.shape, np.uint8), lo, 0.0
    scale = (hi - lo) / levels
    v = (x - lo) / scale
    q = np.floor(v + rng.random(x.shape))  # stochastic rounding
    return np.clip(q, 0, levels).astype(np.uint8), lo, scale


def quantize(x: np.ndarray, bits: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Return (dequantized x, payload bytes). Unbiased: E[q(x)] = x."""
    if bits >= 32:
        return x.astype(np.float32), x.size * 4
    if bits == 16:
        return x.astype(np.float16).astype(np.float32), x.size * 2
    q, lo, scale = quantize_codes(x, bits, rng)
    if scale == 0.0:
        return np.full_like(x, lo, dtype=np.float32), 8
    # payload: packed codes + lo/scale as two fp32
    return (q * scale + lo).astype(np.float32), int(np.ceil(x.size * bits / 8)) + 8


def pack_codes(q: np.ndarray, bits: int) -> bytes:
    per = 8 // bits
    flat = q.reshape(-1).astype(np.uint8)
    flat = np.concatenate([flat, np.zeros((-len(flat)) % per, np.uint8)]).reshape(-1, per)
    out = np.zeros(len(flat), np.uint8)
    for j in range(per):
        out |= flat[:, j] << (bits * j)
    return out.tobytes()


def unpack_codes(b: bytes, bits: int, n: int) -> np.ndarray:
    per, mask = 8 // bits, (1 << bits) - 1
    a = np.frombuffer(b, np.uint8)
    return np.stack([(a >> (bits * j)) & mask for j in range(per)], 1).reshape(-1)[:n]


def encode(tensors: list[np.ndarray], bits: int, rng: np.random.Generator) -> tuple[bytes, list[np.ndarray]]:
    """Serialize quantized tensors. Returns (payload, dequantized tensors exactly as the server will decode them).

    Layout: magic, bits, n_tensors (uint8 each); per tensor: ndim (uint8), dims (uint32 each), then
    fp32 raw (bits 32), fp16 raw (bits 16), or lo, scale (fp32) + packed codes (bits 1, 2, 4, 8)."""
    if bits not in (1, 2, 4, 8, 16, 32):
        raise ValueError(f"unsupported width {bits}")
    parts, deq = [struct.pack("<BBB", MAGIC, bits, len(tensors))], []
    for x in tensors:
        x = np.asarray(x, np.float32)
        parts.append(struct.pack("<B", x.ndim) + struct.pack(f"<{x.ndim}I", *x.shape))
        if bits == 32:
            parts.append(x.astype("<f4").tobytes())
            deq.append(x.copy())
        elif bits == 16:
            h = x.astype("<f2")
            parts.append(h.tobytes())
            deq.append(h.astype(np.float32))
        else:
            q, lo, scale = quantize_codes(x, bits, rng)
            parts.append(struct.pack("<ff", lo, scale) + pack_codes(q, bits))
            deq.append(decode_tensor(q, np.float32(lo), np.float32(scale)))
    return b"".join(parts), deq


def decode_tensor(q, lo, scale):
    return (q.astype(np.float32) * np.float32(scale) + np.float32(lo)).astype(np.float32)


def decode(payload: bytes) -> tuple[list[np.ndarray], int]:
    magic, bits, n = struct.unpack_from("<BBB", payload, 0)
    if magic != MAGIC:
        raise ValueError("not a packed update")
    pos, out = 3, []
    for _ in range(n):
        nd = payload[pos]
        shape = struct.unpack_from(f"<{nd}I", payload, pos + 1)
        pos += 1 + 4 * nd
        size = int(np.prod(shape)) if nd else 1
        if bits == 32:
            out.append(np.frombuffer(payload, "<f4", size, pos).reshape(shape).astype(np.float32))
            pos += 4 * size
        elif bits == 16:
            out.append(np.frombuffer(payload, "<f2", size, pos).reshape(shape).astype(np.float32))
            pos += 2 * size
        else:
            lo, scale = struct.unpack_from("<ff", payload, pos)
            nb = -(-size * bits // 8)
            q = unpack_codes(payload[pos + 8:pos + 8 + nb], bits, size)
            out.append(decode_tensor(q, np.float32(lo), np.float32(scale)).reshape(shape))
            pos += 8 + nb
    if pos != len(payload):
        raise ValueError("trailing bytes in packed update")
    return out, bits


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

    def compress_packed(self, delta: list[np.ndarray], bits: int, rng: np.random.Generator,
                        enabled: bool = True) -> bytes:
        """Like compress, but returns the serialized payload that is actually sent."""
        if enabled and self.residual is not None:
            delta = [d + r for d, r in zip(delta, self.residual)]
        payload, deq = encode(delta, bits, rng)
        if enabled:
            self.residual = [d - q for d, q in zip(delta, deq)]
        return payload
