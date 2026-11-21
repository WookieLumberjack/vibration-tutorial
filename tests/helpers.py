"""Helpers the test modules share (#409 S-14).

`advance_for(sim, duration, piece)` runs a simulator on by `duration` s past its cap on steps a call
(`MAX_STEPS_PER_ADVANCE`): it calls `advance` until the time is reached, in pieces of at most `piece` s if
given, and returns each call's samples. It stops early if a call takes no step (under a step left).
"""

from __future__ import annotations

import math


def advance_for(sim, duration: float, piece: float = math.inf) -> list:
    """Each `sim.advance` call's samples until `duration` s have passed (see the module)."""
    out: list = []
    end = sim.t + duration
    while sim.t < end - 1e-12:
        r = sim.advance(min(piece, end - sim.t))
        t = r.t if hasattr(r, "t") else r[0]
        if not t.size:
            break
        out.append(r)
    return out
