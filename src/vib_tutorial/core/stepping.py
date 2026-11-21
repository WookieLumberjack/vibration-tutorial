"""Run pacing in one place (#409 S-8): how many steps a stretch of time makes, and how much a frame may compute.

`StepClock` turns the simulated time a frame asks for into whole steps of size h and carries the part of a
step left over to the next call. Rounding each call on its own loses it: a slow playback's frame is often under
half a step, so a simulator that rounded stood still at 0.02× (the top), or ran at a speed set by the rounding.
A simulator whose page asks for `target − t` each frame (the chain's, the rotor's, Duffing's) carries the time
itself, in its target, and rounds; one asked for a bare duration (the top, the sliding top, the rigid body, the
torsional start-up, the CMS model in time) keeps a StepClock. Past `most` steps a call drops the rest rather
than owing it, so a slow machine falls behind instead of spiralling (#409 N-7).

`Budget` is the frame's computing allowance at the cost measured so far: `allows()` how many units (steps, or
simulated seconds) fit, `spent(seconds, units)` after doing them. A dearer cost counts at once, a cheaper one
gradually (0.2 a frame), so one quick frame doesn't let the next overrun. Pages share `FRAME_BUDGET`.

`Stepper` is what every simulator is to a page (#409 S-16): a clock `t` and `advance(duration)`, which steps on
and returns the samples of that stretch, of the simulator's own type. It is the shape a Rust kernel takes when a
simulator moves there (as `native.LinearSteps` did for the chain's), and the trait a WebAssembly front end would
drive:

    pub trait Stepper {
        type Samples;
        fn t(&self) -> f64;
        fn advance(&mut self, duration: f64) -> Self::Samples;  // whole steps, the rest carried (StepClock)
    }

The page keeps the frame loop and the `Budget`; the stepper only steps. Every simulator in `core/` is one
(`tests/test_stepping.py` checks the eight).

Working on it: `tests/test_stepping.py`; the simulators' own tests check their clocks, `test_frame_stats` the
pages' pace.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

FRAME_BUDGET = 0.012  # s of computing a frame may spend on its simulation: past it a page runs slower than asked
EASE = 0.2  # a cheaper measured cost's weight a frame


class StepClock:
    """Whole steps for each stretch of simulated time, the part of a step left over carried to the next."""

    def __init__(self) -> None:
        self.owed = 0.0  # s asked for and not yet stepped (under a step)
        self.dropped = 0.0  # s the last call dropped past `most`

    def steps(self, duration: float, h: float, most: int) -> int:
        """The whole steps of size `h` that `duration` and what's owed make, at most `most` (the rest dropped)."""
        owed = self.owed + duration
        want = int(owed / h + 1e-9)  # a whole number of steps asked for stays whole
        n = max(0, min(want, most))
        self.owed = owed - n * h if n == want else 0.0
        self.dropped = 0.0 if n == want else owed - n * h
        return n

    def reset(self) -> None:
        self.owed = self.dropped = 0.0


class Budget:
    """A frame's computing allowance, `seconds`, at the measured cost of a unit of work (see the module)."""

    def __init__(self, seconds: float = FRAME_BUDGET) -> None:
        self.seconds = seconds
        self.cost = 0.0  # s of computing a unit; 0: not measured yet

    def allows(self, unknown: float, seconds: float | None = None) -> float:
        """The units the allowance (or `seconds`, this frame's) buys, or `unknown` (a probe) before any cost is
        measured."""
        return (self.seconds if seconds is None else seconds) / self.cost if self.cost > 0 else unknown

    def spent(self, seconds: float, units: float) -> None:
        """`units` of work took `seconds`: a dearer cost counts at once, a cheaper one gradually."""
        if units <= 0:
            return
        cost = seconds / units
        self.cost = cost if cost > self.cost else (1 - EASE) * self.cost + EASE * cost

    def forget(self) -> None:
        """The work changed (a new chain, friction): measure its cost afresh."""
        self.cost = 0.0


@runtime_checkable
class Stepper[S](Protocol):
    """A simulator as a page drives it (see the module): its clock and `advance`, giving that stretch's samples."""

    t: float

    def advance(self, duration: float) -> S: ...
