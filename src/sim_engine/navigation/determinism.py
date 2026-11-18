"""Frozen deterministic per-candidate sampling stream."""

import hashlib
import math
import random


def perturbation(seed, candidate_id, radius, yaw_degrees):
    # A per-candidate stream; scheduling, exclusion and worker count cannot change a draw.
    value = int.from_bytes(hashlib.sha256(f"{seed}:{candidate_id}".encode()).digest(), "big")
    rng = random.Random(value)
    r = radius * math.sqrt(rng.random())
    theta = rng.uniform(-math.pi, math.pi)
    return {
        "offset": [r * math.cos(theta), 0.0, r * math.sin(theta)],
        "yaw_delta": rng.uniform(-math.radians(yaw_degrees), math.radians(yaw_degrees)),
    }
