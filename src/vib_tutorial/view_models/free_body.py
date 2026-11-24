"""A live free-body diagram (#478 item 2): the forces on a body, as arrows to draw on it, worked out each frame.

`FreeBody` is what a page draws: each force an `Arrow` (its name, its label with its size, its value in newtons,
+ along the body's positive direction, and its colour as a theme role's name), and the net force, which is the
mass times its acceleration. A force of nought is still listed (its arrow drawn at no length), so the arrows
keep their places while one comes and goes.

`ArrowScale` keeps the arrows' scale steady while the forces change: it grows at once to fit the largest force
(an arrow never runs off the body's space) and shrinks only when the largest has stayed under a quarter of the
room for `SHRINK_AFTER` seconds, so a swing's arrows don't breathe with every cycle and a new, smaller motion is
still drawn big enough to read.

`one_mass` is the forces on one mass on a spring and a damper, with friction, a drive and a hand holding it (the
Single oscillator, Damping models); a page with another body builds its own `FreeBody` from its forces the same
way (Pendulums', Isolation's later).

Working on it: Qt-free (the package's docstring); `tests/test_live_parts.py` checks the forces add up to m·a, the
roles name theme colours and the scale's growing and shrinking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .text import fmt

SHRINK_AFTER = 3.0  # s the forces must stay small before the scale shrinks to them
ROOM = 0.8  # the largest force fills this much of the arrows' room


@dataclass(frozen=True)
class Arrow:
    name: str  # what pushes: "spring", "damper", "friction", "drive", "hand", "net"
    label: str  # what's printed by it: its symbol and size
    value: float  # N, + along the body's positive direction
    role: str  # its colour, as a theme role's name (`style.role`)


@dataclass(frozen=True)
class FreeBody:
    arrows: tuple[Arrow, ...]
    net: Arrow  # the sum of the forces: the mass times its acceleration
    scale: float  # display units a newton

    def length(self, arrow: Arrow) -> float:
        return arrow.value * self.scale


class ArrowScale:
    """Newtons to display units, steady while the forces change (see the module)."""

    def __init__(self, room: float) -> None:
        self.room = room  # display units the largest arrow may take
        self.largest = 0.0  # N the scale fits now
        self._small_since: float | None = None

    def update(self, t: float, forces: float) -> float:
        """The scale (display units a newton) at time t, the largest force now `forces` (N)."""
        forces = abs(forces)
        if forces > self.largest:
            self.largest = forces
            self._small_since = None
        elif forces < 0.25 * self.largest:
            if self._small_since is None or t < self._small_since:
                self._small_since = t
            elif t - self._small_since >= SHRINK_AFTER and forces > 0:
                self.largest = forces
                self._small_since = None
        else:
            self._small_since = None
        return ROOM * self.room / self.largest if self.largest > 0 else 0.0

    def reset(self) -> None:
        self.largest = 0.0
        self._small_since = None


def _arrow(name: str, symbol: str, value: float, role: str) -> Arrow:
    return Arrow(name, f"{symbol} = {fmt(abs(value), 'N')}", value, role)


def one_mass(
    stiffness: float,
    damping: float,
    x: float,
    v: float,
    drive: float = 0.0,
    friction: float | None = None,
    hand: float | None = None,
    ground: float = 0.0,
    scale: ArrowScale | None = None,
    t: float = 0.0,
) -> FreeBody:
    """The forces on one mass at displacement x and velocity v: the spring's −k(x − x_g), the damper's −cv, the
    drive's, friction's (as the simulator found it: None, no friction) and a hand's holding it (None: let go)."""
    arrows = [
        _arrow("spring", "kx", -stiffness * (x - ground), "energy.potential"),
        _arrow("damper", "cẋ", -damping * v, "energy.dissipated"),
    ]
    if friction is not None:
        arrows.append(_arrow("friction", "F<sub>f</sub>", friction, "energy.friction"))
    arrows.append(_arrow("drive", "f", drive, "force"))
    if hand is not None:
        arrows.append(_arrow("hand", "hand", hand, "muted"))
    total = math.fsum(a.value for a in arrows)
    net = _arrow("net", "mẍ", total, "strong")
    largest = max(abs(a.value) for a in arrows)
    s = scale.update(t, largest) if scale is not None else (1.0 / largest if largest > 0 else 0.0)
    return FreeBody(tuple(arrows), net, s)
