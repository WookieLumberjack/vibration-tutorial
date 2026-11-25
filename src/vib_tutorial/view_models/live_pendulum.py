"""One live pendulum (#478 Phase 4): a pendulum stepped as it swings, and changed as it swings.

The Pendulums page's body, as `live_mass.LiveMass` is the Single oscillator's. It steps the whole equation
J θ'' + c θ' + k_t θ ± m g d sin θ = 0 with its linearized ghost beside it (`core/pendulum.py`'s `rk4`), and a
change while it swings is a physical event:

- **The inertia changes, the angular momentum Jθ' is kept** (m, L, the rod's pivot, the kind): a string pulled
  in through the pivot, or a mass caught at the pivot's distance, exerts no moment about the pivot, so Jθ' can't
  jump. The speed does: shortened, it swings faster. Shortening at the bottom of each swing and letting out at
  the ends pumps it up, as on a playground swing; the hand doing it does the work.
- **The moments change, the angle and speed are kept** (k<sub>t</sub>, g): the stored energy jumps by
  V'(θ) − V(θ).
- **The damping** (ζ, as the linearized oscillator's) changes the damper c = 2ζ√(|k|J) with no jump.

Each change's work (the energy after it less the energy before) is booked as `changes`, each its time in
`edits`; the ghost changes with it by the same rule.

What the reader does (the interaction logic, here rather than in Qt handlers):

- `set_pendulum(p)`: a box or a grip changed the pendulum (a live change, above).
- `hold(θ)` then `let_go()`: a hand turns the pendulum to θ and holds it there, at rest (the clock stops while
  it's held, and the energy the hand puts in is `released`); let go, it swings from there, its ghost with it.
  That's a pluck.
- `advance(duration)`: step on (nothing while held), the samples into `record`. A frame asks for its own stretch,
  so the part of a step left over is carried to the next (`StepClock`): it's a `stepping.Stepper`.

`changed_at` is the time of the last change or let-go: the period recorder (`view_models/pendulum.py`) times only
the swings after it. The record (`COLUMNS`) keeps each sample's θ, θ', the ghost's θ, the kinetic and potential
energy (with the pendulum of the time) and the ledger: the damper's work, the hand's and the changes', so
`energy_view` shows the balance closing through every change.

Working on it: Qt-free (the package's docstring); `tests/test_live_pendulum.py` checks a pluck, the live changes'
rules (pumping a swing up included) and the balance closing.
"""

from __future__ import annotations

import math

import numpy as np

from ..core.pendulum import Pendulum, rk4, step_size
from ..core.simulator import MAX_STEPS_PER_ADVANCE
from ..core.stepping import StepClock
from .live_mass import EnergyRow, EnergyView
from .record import SampleBuffer
from .text import fmt, number

COLUMNS = ("theta", "omega", "ghost", "kinetic", "potential", "dissipated", "released", "changes")
COL = {name: i for i, name in enumerate(COLUMNS)}


def energy(p: Pendulum, theta: float, omega: float) -> float:
    """Stored: ½Jθ'² + V(θ)."""
    return 0.5 * p.inertia * omega * omega + float(p.potential(theta))


class LivePendulum:
    """One pendulum, stepped live and changed as it swings (see the module)."""

    def __init__(self, pendulum: Pendulum | None = None, capacity: int = 200_000) -> None:
        self.pendulum = pendulum or Pendulum()
        self.record = SampleBuffer(len(COLUMNS), capacity)
        self.clock = StepClock()
        self.t = 0.0
        self.theta = self.omega = 0.0
        self.ghost = self.ghost_omega = 0.0  # the linearized swing's
        self.held: float | None = None
        self.changed_at = 0.0
        self.edits: list[float] = []
        self.dissipated = self.released = self.changes = 0.0

    # ------------------------------------------------------------- what the reader does
    @property
    def stored(self) -> float:
        return energy(self.pendulum, self.theta, self.omega)

    @property
    def acceleration(self) -> float:
        """θ'' now (0 while a hand holds it)."""
        if self.held is not None:
            return 0.0
        p = self.pendulum
        return -(p.damping * self.omega + float(p.moment(self.theta))) / p.inertia

    def set_pendulum(self, p: Pendulum) -> None:
        """A live change (see the module): Jθ' kept, the angle kept, the work booked."""
        old = self.pendulum
        if p == old:
            return
        before = self.stored
        ratio = old.inertia / p.inertia
        self.omega *= ratio
        self.ghost_omega *= ratio
        self.pendulum = p
        self.changes += self.stored - before
        self.edits.append(self.t)
        self.changed_at = self.t

    def hold(self, theta: float) -> None:
        """A hand holds the pendulum still at θ (see the module); the ghost is set there too."""
        before = self.stored
        self.held = self.theta = self.ghost = theta
        self.omega = self.ghost_omega = 0.0
        self.released += self.stored - before

    def let_go(self) -> None:
        self.held = None
        self.changed_at = self.t

    def reset(self) -> None:
        """Back to t = 0 hanging (or standing) still, the record and the books cleared, the pendulum as it is."""
        self.record.clear()
        self.clock.reset()
        self.t = self.theta = self.omega = self.ghost = self.ghost_omega = 0.0
        self.held = None
        self.changed_at = 0.0
        self.edits = []
        self.dissipated = self.released = self.changes = 0.0

    def advance(self, duration: float) -> int:
        """Step on by about `duration` s (nothing while held): the number of samples recorded."""
        if self.held is not None or duration <= 0:
            return 0
        p = self.pendulum
        h = step_size(p)
        n = self.clock.steps(duration, h, MAX_STEPS_PER_ADVANCE)
        if n == 0:
            return 0
        y = rk4(p, (self.theta, self.omega, self.ghost, self.ghost_omega), h, n)
        ts = self.t + h * np.arange(1, n + 1)
        th, om = y[:, 0], y[:, 1]
        rows = np.empty((n, len(COLUMNS)))
        rows[:, COL["theta"]] = th
        rows[:, COL["omega"]] = om
        rows[:, COL["ghost"]] = y[:, 2]
        rows[:, COL["kinetic"]] = 0.5 * p.inertia * om * om
        rows[:, COL["potential"]] = p.potential(th)
        rows[:, COL["dissipated"]] = self.dissipated + y[:, 4]
        rows[:, COL["released"]] = self.released
        rows[:, COL["changes"]] = self.changes
        self.record.extend(ts, rows)
        self.t = float(ts[-1])
        self.theta, self.omega, self.ghost, self.ghost_omega = (float(v) for v in y[-1, :4])
        self.dissipated += float(y[-1, 4])
        return n

    def since_change(self) -> tuple[np.ndarray, np.ndarray]:
        """The record's samples after the last change (`changed_at`): (t, columns)."""
        t, y = self.record.t[: self.record.size], self.record.y[: self.record.size]
        i0 = int(np.searchsorted(t, self.changed_at))
        return t[i0:], y[i0:]


def energy_view(t: np.ndarray, y: np.ndarray) -> EnergyView | None:
    """The energy balance over the record `(t, y)` (its `COLUMNS`) and the ledger at its end, or None if empty: the
    Energy tab's (`live_parts.EnergyTab`), as the Single oscillator's."""
    if t.size == 0:
        return None
    c = COL
    stored = y[:, c["kinetic"]] + y[:, c["potential"]]
    put_in = y[:, c["released"]]
    taken_out = y[:, c["dissipated"]]
    changes = y[:, c["changes"]]
    end = y[-1]
    residual = float(put_in[-1] + changes[-1] - taken_out[-1] - stored[-1])
    rows = (
        EnergyRow("Put in by your hand", float(end[c["released"]]), "muted"),
        EnergyRow("Changing the pendulum", float(end[c["changes"]]), "energy.added"),
        EnergyRow("Taken out by the damper", -float(end[c["dissipated"]]), "energy.dissipated"),
        EnergyRow("Stored now", float(stored[-1]), "energy.stored"),
    )
    scale = max(abs(r.value) for r in rows)
    text = (
        f"Stored now {fmt(float(stored[-1]), 'J')}: what your hand put in, less what the damper took out, with "
        f"the changes' own work. It closes to {fmt(abs(residual), 'J', 2, smallest='p')}"
        + (f" ({number(abs(residual) / scale, 2)} of the largest term)." if scale > 0 else ".")
    )
    return EnergyView(t, stored, put_in, taken_out, changes, rows, residual, text)


def ghost_angle(ghost: float) -> float | None:
    """The ghost's angle to draw, or None once it has run away (an unstable pendulum's linearization)."""
    return ghost if math.isfinite(ghost) and abs(ghost) < 4 * math.pi else None
