"""Grips on a drawn body (#478 item 4): how a drag on the mass, the spring or the damper sets m, k or c.

A grip changes its value by a factor for a distance dragged, not by a step: e times for every `per` display
units, so a drag halves or doubles a value as easily at 0.1 kg as at 10 kg, and the value moves as the hand
does. It stays within its `low`…`high`, the range of the box beside it, which it moves too. A value that can be
zero (a damper, friction) snaps there below `zero_below`, and grows again from `low` when dragged up from it.

Each step of a drag is a live change: the page passes the new value to its simulator, which carries the motion
on by `Simulator.set_system`'s rules (a mass grip's drag is a run of small captures or releases).

`body_size(m, reference)` is how big the mass is drawn: its side goes as the cube root of the mass (as a block
of one material would), so 8 times the mass is twice as wide, within `SIZE_RANGE` of the reference's.

Working on it: Qt-free (the package's docstring); `tests/test_live_parts.py` checks the mapping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

SIZE_RANGE = (0.5, 2.0)  # the mass's side, as drawn, within these times the reference mass's


@dataclass(frozen=True)
class Grip:
    """A value set by dragging (see the module): e times its value every `per` display units of drag."""

    name: str
    low: float
    high: float
    per: float = 0.5
    zero_below: float = 0.0  # a value under this is zero (0: it never is)

    def dragged(self, start: float, distance: float) -> float:
        """The value a drag of `distance` (display units, + to grow) gives, from `start` where it began."""
        base = max(start, self.low) if self.zero_below > 0 else start
        value = min(max(base * math.exp(distance / self.per), self.low), self.high)
        if self.zero_below > 0 and value < self.zero_below:
            return 0.0
        return value

    def distance(self, start: float, value: float) -> float:
        """How far a drag from `start` takes the value to `value` (the inverse of `dragged`, where it's on the scale)."""
        base = max(start, self.low) if self.zero_below > 0 else start
        return self.per * math.log(max(value, self.low) / base)


def body_size(mass: float, reference: float) -> float:
    """The mass's side as drawn, relative to the reference mass's (see the module)."""
    lo, hi = SIZE_RANGE
    return min(max((mass / reference) ** (1.0 / 3.0), lo), hi)
