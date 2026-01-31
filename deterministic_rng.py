from __future__ import annotations

import hashlib
import math


def _u64(seed: bytes, counter: int) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(seed)
    h.update(counter.to_bytes(8, "little", signed=False))
    return int.from_bytes(h.digest(), "little", signed=False)


def uniform01(seed: bytes, counter: int) -> float:
    """
    Deterministic uniform float in (0, 1), derived from (seed, counter).
    """
    # Map 64-bit integer to (0, 1) with clamping away from endpoints
    x = _u64(seed, counter)
    u = (x + 1.0) / (2**64 + 1.0)
    # Defensive clamp (should already be in (0, 1))
    if u <= 0.0:
        return 1.0 / (2**64 + 1.0)
    if u >= 1.0:
        return 1.0 - (1.0 / (2**64 + 1.0))
    return float(u)


def uniform(seed: bytes, counter: int, low: float, high: float) -> float:
    return low + (high - low) * uniform01(seed, counter)


def normal(seed: bytes, counter: int, mean: float = 0.0, std: float = 1.0) -> float:
    """
    Deterministic normal via Box-Muller transform.

    Uses counters (counter, counter+1) internally.
    """
    u1 = uniform01(seed, counter)
    u2 = uniform01(seed, counter + 1)
    r = math.sqrt(-2.0 * math.log(u1))
    theta = 2.0 * math.pi * u2
    z0 = r * math.cos(theta)
    return mean + std * z0


def uint31(seed: bytes, counter: int) -> int:
    return int(_u64(seed, counter) & 0x7FFF_FFFF)
