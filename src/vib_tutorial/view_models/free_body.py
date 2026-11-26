"""A live free-body diagram (#478 item 2): the forces on a body, as arrows to draw, worked out each frame.

`FreeBody` is what a page draws (`gui/free_body_view.py`, beside the moving body): each force an `Arrow` (its
name, its label with its size, its symbol alone, its value in newtons, + along the body's positive direction, and
its color as a theme role's name), and the net force, which is the mass times its acceleration. A force of zero is still listed (its arrow drawn at no length), so the arrows
keep their places while one comes and goes. A turning body's diagram is the same in moments about its pivot (N·m,
the net J θ''), with `aside` rows that aren't in the sum (a pendulum's tension, through the pivot: no moment) on a
scale of their own (`aside_scale`).

`ArrowScale` sizes the arrows to the largest force of the last `SHRINK_AFTER` seconds: it grows at once to fit a
larger force (an arrow never runs off its room), and a force leaves it that long after it passed, so a swing's
peak fills the room, its arrows grow and shrink through each cycle on a scale that holds still while it, and a
smaller motion (once the hand has let go, or the swing has died down) is redrawn as large within seconds.

`one_mass` is the forces on one mass on a spring and a damper, with friction, a drive and a hand holding it (the
Single oscillator); a page with another body builds its own `FreeBody` from its forces the same
way (`view_models/pendulum.py`'s `free_body`, in moments; Isolation's later).

Working on it: Qt-free (the package's docstring); `tests/test_live_parts.py` checks the forces add up to m·a, the
roles name theme colors and the scale's growing and shrinking.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from .text import fmt

SHRINK_AFTER = 2.0  # s a force stays in the scale after it passed
ROOM = 0.8  # the largest force fills this much of the arrows' room


@dataclass(frozen=True)
class Arrow:
    name: str  # what pushes: "spring", "damper", "friction", "drive", "hand", "net"
    label: str  # what's printed by it, as plain text (no markup): its symbol and size
    value: float  # N, + along the body's positive direction
    role: str  # its color, as a theme role's name (`style.role`)
    symbol: str = ""  # its symbol alone, short enough to write on the body ("arm" for the Zener arm's k₁(x − y))
    unit: str = "N"  # its size's ("N·m" for a moment)


@dataclass(frozen=True)
class FreeBody:
    arrows: tuple[Arrow, ...]
    net: Arrow  # the sum of the forces: the mass times its acceleration
    scale: float  # display units a newton (a newton-meter, in moments)
    aside: tuple[Arrow, ...] = ()  # rows drawn under the net, not in its sum (a pendulum's tension)
    aside_scale: float = 0.0  # their display units a newton

    def length(self, arrow: Arrow) -> float:
        return arrow.value * self.scale


class ArrowScale:
    """Newtons to display units: the largest force of the last `SHRINK_AFTER` s fills the room (see the module)."""

    def __init__(self, room: float) -> None:
        self.room = room  # display units the largest arrow may take
        self.largest = 0.0  # N the scale fits now
        self._recent: deque[tuple[float, float]] = deque()  # (t, the largest force then), its peaks decreasing

    def update(self, t: float, forces: float) -> float:
        """The scale (display units a newton) at time t, the largest force now `forces` (N)."""
        forces = abs(forces)
        recent = self._recent
        if recent and t < recent[-1][0]:  # back in time (scrubbed): start again from here
            recent.clear()
        while recent and recent[-1][1] <= forces:  # outdone: it can never be the largest again
            recent.pop()
        recent.append((t, forces))
        while recent[0][0] < t - SHRINK_AFTER:
            recent.popleft()
        self.largest = recent[0][1]
        return ROOM * self.room / self.largest if self.largest > 0 else 0.0

    def reset(self) -> None:
        self.largest = 0.0
        self._recent.clear()


def arrow(name: str, symbol: str, value: float, role: str, short: str | None = None, unit: str = "N") -> Arrow:
    return Arrow(name, f"{symbol} = {fmt(abs(value), unit)}", value, role, short or symbol, unit)


def one_mass(
    stiffness: float,
    damping: float,
    x: float,
    v: float,
    drive: float = 0.0,
    friction: float | None = None,
    hand: float | None = None,
    arm: float | None = None,
    ground: float = 0.0,
    scale: ArrowScale | None = None,
    t: float = 0.0,
) -> FreeBody:
    """The forces on one mass at displacement x and velocity v: the spring's −k(x − x_g), the damper's −cv, the
    drive's, friction's (as the simulator found it: None, no friction), a Zener arm's (None: none) and a hand's
    holding it (None: let go)."""
    arrows = [arrow("spring", "kx", -stiffness * (x - ground), "energy.potential")]
    if damping > 0 or (friction is None and arm is None):  # a friction or Zener mass has no damper of its own
        arrows.append(arrow("damper", "cẋ", -damping * v, "energy.dissipated"))
    if friction is not None:
        arrows.append(arrow("friction", "friction", friction, "energy.friction"))
    if arm is not None:  # a Zener element's arm: its spring k₁ pulling, its dashpot relaxing behind it
        arrows.append(arrow("arm", "k₁(x − y)", arm, "energy.added", short="arm"))
    arrows.append(arrow("drive", "f", drive, "force"))
    if hand is not None:
        arrows.append(arrow("hand", "hand", hand, "muted"))
    total = math.fsum(a.value for a in arrows)
    net = arrow("net", "mẍ", total, "strong")
    largest = max(abs(a.value) for a in (*arrows, net))  # the net too: it can outgrow every force
    s = scale.update(t, largest) if scale is not None else (1.0 / largest if largest > 0 else 0.0)
    return FreeBody(tuple(arrows), net, s)
