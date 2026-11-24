"""One live mass (#478 item 1): a mass on a spring and a damper (and friction), stepped as it runs, changed as it runs.

The Single oscillator's and Damping models' body. It is the chain's `Simulator` with one mass, so it steps by the
exact FOH update (friction's stick–slip by its events), and a change of m, k, c or friction while it runs is a
physical event by `Simulator.set_system`'s rules: a spring change books ½Δk·x², a heavier mass captures its extra
at rest (momentum kept), a lighter one drops a piece at its speed. Nothing restarts, and the record keeps what came
before, with the change's time in `edits`.

What the reader does, and what each does to the run (the interaction logic, here rather than in Qt handlers):

- `set_params(params)`: a box or a grip changed m, k, c or friction (a live change).
- `set_drive(drive)`: the drive switched on or off, or its kind, size or frequency changed. The phase runs on
  (`ForceController` integrates it), so a new frequency joins on with no jump; an unbalance's force is m<sub>e</sub>e Ω²
  at the frequency of the time.
- `hold(x)` then `let_go()`: a hand grabs the mass and holds it at x, at rest (the clock stops while it's held,
  and the energy the hand puts in is `released`); let go, it runs on from there. That's a pluck.
- `advance(duration)`: step on (nothing while held), the samples into `record`. A frame asks for its own
  stretch, so the part of a step left over is carried to the next (`StepClock`): it's a `stepping.Stepper`.

`changed_at` is the time of the last change that starts a new steady state (the system, the drive's frequency or
switching, a let-go): the recorder measures only after it, and `ring_down` looks for peaks only after it.

The record (`COLUMNS`, a `SampleBuffer`) keeps each sample's x, v and force, its kinetic and potential energy (with
the parameters of the time) and the ledger: what the drive put in (`work`), what the damper and friction took out,
and the live changes' terms (`spring_work`, `capture_loss`, `carried_off`) and the hand's (`released`), so
`energy_view` shows the balance closing through every change.

Working on it: Qt-free (the package's docstring); `tests/test_live_parts.py` checks a pluck, live changes keeping
the run going, the drive's kinds and the balance closing. The physics of a change is `core/simulator.py`'s.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core.arm_mass import ArmMass, ArmSystem
from ..core.forcing import ForceController, ForceKind, ForceSettings
from ..core.model import ChainSystem
from ..core.sdof import TWO_PI, Oscillator, peaks
from ..core.simulator import MAX_STEPS_PER_ADVANCE, Simulator
from ..core.stepping import StepClock
from .record import SampleBuffer
from .text import fmt, number

FORCE, UNBALANCE = 0, 1  # the drive's kinds
COLUMNS = (
    "x",
    "v",
    "f",
    "kinetic",
    "potential",
    "added",
    "work",
    "dissipated",
    "friction",
    "spring_work",
    "capture_loss",
    "carried_off",
    "released",
)
COL = {name: i for i, name in enumerate(COLUMNS)}


@dataclass(frozen=True)
class Params:
    """The mass on its spring and damper: SI units (kg, N/m, N·s/m, N of sliding friction)."""

    mass: float = 1.0
    stiffness: float = 400.0
    damping: float = 2.0
    friction: float = 0.0
    static_ratio: float = 1.0
    arm_stiffness: float = 0.0  # a Zener arm beside the spring: its spring k₁ (0: none) ...
    arm_damping: float = 0.0  # ... in series with its dashpot c₁

    @property
    def has_arm(self) -> bool:
        return self.arm_stiffness > 0

    def system(self) -> ChainSystem | ArmSystem:
        if self.has_arm:
            if self.friction > 0:
                raise ValueError("a Zener arm and friction together aren't modelled")
            return ArmSystem(self.mass, self.stiffness, self.damping, self.arm_stiffness, self.arm_damping)
        return ChainSystem(
            np.array([self.mass]),
            np.array([self.stiffness]),
            np.array([self.damping]),
            np.array([self.friction]),
            self.static_ratio,
        )

    def oscillator(self) -> Oscillator:
        """The linear oscillator (friction and the arm left out)."""
        return Oscillator(self.mass, self.stiffness, self.damping)

    def potential(self, x: float, y: float) -> float:
        """The springs' energy at x, the arm's inner end at y."""
        return 0.5 * self.stiffness * x * x + 0.5 * self.arm_stiffness * (x - y) ** 2


@dataclass(frozen=True)
class Drive:
    """The drive on the mass: a force F sin Ωt, or a rotating unbalance m<sub>e</sub>e whose force is m<sub>e</sub>eΩ²."""

    on: bool = False
    kind: int = FORCE
    amplitude: float = 1.0  # N, or the unbalance m_e e in kg·m
    freq_hz: float = 3.0

    @property
    def force(self) -> float:
        """The force's amplitude at its frequency (N)."""
        return self.amplitude * (TWO_PI * self.freq_hz) ** 2 if self.kind == UNBALANCE else self.amplitude


class LiveMass:
    """One mass, stepped live and changed as it runs (see the module)."""

    def __init__(self, params: Params | None = None, drive: Drive | None = None, capacity: int = 200_000) -> None:
        self.params = params or Params()
        self.drive = drive or Drive()
        self.sim = self._make(self.params, ForceController(ForceSettings(kind=ForceKind.HARMONIC)))
        self.record = SampleBuffer(len(COLUMNS), capacity)
        self.held: float | None = None
        self.changed_at = 0.0
        self.clock = StepClock()  # a frame's part of a step carried to the next (`core/stepping.py`)
        self._apply_drive(self.drive)

    # ------------------------------------------------------------- what the reader does
    @property
    def t(self) -> float:
        return self.sim.t

    @property
    def x(self) -> float:
        return float(self.sim.displacement[0])

    @property
    def v(self) -> float:
        return float(self.sim.velocity[0])

    @property
    def force_now(self) -> float:
        """The drive's force at this instant (N)."""
        return self.sim.force.value()

    @property
    def edits(self) -> list[float]:
        return self.sim.edits

    @staticmethod
    def _make(params: Params, force: ForceController) -> Simulator | ArmMass:
        system = params.system()
        return ArmMass(system, force) if isinstance(system, ArmSystem) else Simulator(system, force)

    def set_params(self, params: Params) -> None:
        if params == self.params:
            return
        if params.has_arm == self.params.has_arm:
            self.sim.set_system(params.system())  # type: ignore[arg-type]  # the same kind: its own live change
        else:
            self._swap(params)
        self.params = params
        self.changed_at = self.t

    def _swap(self, params: Params) -> None:
        """A change between a plain mass and one with a Zener arm: the same live change, made across two steppers.
        x carries on, the mass by the capture rules, an arm added starts relaxed (y = x), and one taken away takes
        its stretch's energy with it: the springs' change is booked as spring work, and the books carry over."""
        old, new = self.sim, self._make(params, self.sim.force)
        x, v = self.x, self.v
        y_old = old.arm if isinstance(old, ArmMass) else x
        m_old, m_new = self.params.mass, params.mass
        if m_new > m_old:
            v_new, capture, carried = m_old * v / m_new, 0.5 * m_old * v * v * (m_new - m_old) / m_new, 0.0
        else:
            v_new, capture, carried = v, 0.0, 0.5 * (m_old - m_new) * v * v
        spring = params.potential(x, x) - self.params.potential(x, y_old)
        new.t, new._u = old.t, old._u
        for name in ("work", "dissipated", "friction_loss", "released", "spring_work", "capture_loss", "carried_off"):
            setattr(new, name, getattr(old, name))
        new.energy_added = old.energy_added
        new.edits = list(old.edits)
        new.state = np.array([x, v_new, x]) if isinstance(new, ArmMass) else np.array([x, v_new])
        new.spring_work += spring
        new.capture_loss += capture
        new.carried_off += carried
        new.energy_added += spring - capture - carried
        new.edits.append(new.t)
        self.sim = new

    @property
    def arm_force(self) -> float:
        """The Zener arm's pull on the mass (N; 0 without one)."""
        return self.sim.arm_force if isinstance(self.sim, ArmMass) else 0.0

    @property
    def friction_force(self) -> float:
        """Friction's force on the mass now (N), as the free-body diagram draws it: against the motion while it
        slides, and while it sticks what holds it, up to the static friction."""
        p = self.params
        if p.friction <= 0:
            return 0.0
        if abs(self.v) > 1e-9:
            return -math.copysign(p.friction, self.v)
        pull = -p.stiffness * self.x + self.force_now
        hold = p.friction * p.static_ratio
        return -min(max(pull, -hold), hold)

    def set_drive(self, drive: Drive) -> None:
        if drive == self.drive:
            return
        restarts = (drive.on, drive.kind, drive.freq_hz) != (self.drive.on, self.drive.kind, self.drive.freq_hz)
        self._apply_drive(drive)
        if restarts:
            self.changed_at = self.t

    def _apply_drive(self, drive: Drive) -> None:
        force = self.sim.force
        force.settings.amplitude = drive.force
        force.settings.freq_hz = drive.freq_hz
        if drive.on and not force.on:
            force.switch_on()
        elif not drive.on and force.on:
            force.switch_off()
        self.drive = drive

    def hold(self, x: float) -> None:
        """A hand holds the mass still at x (see the module)."""
        self.held = x
        self.sim.set_state(np.array([x]), np.zeros(1))

    def let_go(self) -> None:
        self.held = None
        self.changed_at = self.t

    def reset(self) -> None:
        """Back to t = 0 at rest, the record cleared, the parameters and the drive as they are."""
        self.sim.reset()
        self.record.clear()
        self.held = None
        self.changed_at = 0.0
        self.clock.reset()
        self._apply_drive(self.drive)

    def advance(self, duration: float) -> int:
        """Step on by about `duration` s (nothing while held): the number of samples recorded."""
        if self.held is not None or duration <= 0:
            return 0
        h = self.sim.step_size()
        steps = self.clock.steps(duration, h, MAX_STEPS_PER_ADVANCE)
        if steps == 0:
            return 0
        ts, xs, vs, fs = self.sim.advance(steps * h)
        if ts.size == 0:
            return 0
        p, s = self.params, self.sim
        x, v = xs[:, 0], vs[:, 0]
        rows = np.empty((ts.size, len(COLUMNS)))
        rows[:, COL["x"]] = x
        rows[:, COL["v"]] = v
        rows[:, COL["f"]] = fs
        rows[:, COL["kinetic"]] = 0.5 * p.mass * v * v
        if isinstance(s, ArmMass):  # and the arm's stretch
            rows[:, COL["potential"]] = 0.5 * p.stiffness * x * x + 0.5 * p.arm_stiffness * (x - s.arm_track) ** 2
        else:
            rows[:, COL["potential"]] = 0.5 * p.stiffness * x * x
        rows[:, COL["added"] : COL["friction"] + 1] = s.ledger
        rows[:, COL["spring_work"]] = s.spring_work
        rows[:, COL["capture_loss"]] = s.capture_loss
        rows[:, COL["carried_off"]] = s.carried_off
        rows[:, COL["released"]] = s.released
        self.record.extend(ts, rows)
        return ts.size

    def since_change(self) -> tuple[np.ndarray, np.ndarray]:
        """The record's samples after the last change (`changed_at`): (t, columns)."""
        t, y = self.record.t[: self.record.size], self.record.y[: self.record.size]
        i0 = int(np.searchsorted(t, self.changed_at))
        return t[i0:], y[i0:]


# ------------------------------------------------------------------ what the page shows from the record


@dataclass(frozen=True)
class RingDown:
    """A free decay's peaks and what they say (`ring_down`)."""

    times: np.ndarray
    heights: np.ndarray
    ratio: float  # each peak over the one after it, on average
    decrement: float  # δ = ln ratio
    zeta: float  # δ/√(4π² + δ²)
    text: str


def ring_down(t: np.ndarray, x: np.ndarray) -> RingDown | None:
    """The peaks of a free decay (the samples since its release) and the damping they give, or None with fewer than
    three peaks (two ratios), or a swing that isn't dying away."""
    if t.size < 8:
        return None
    times, heights = peaks(t, x)
    keep = heights > 1e-9 * (heights.max() if heights.size else 1.0)
    times, heights = times[keep], heights[keep]
    if heights.size < 3 or heights[-1] >= heights[0]:
        return None
    decrement = math.log(heights[0] / heights[-1]) / (heights.size - 1)
    zeta = decrement / math.sqrt(TWO_PI**2 + decrement**2)
    ratio = math.exp(decrement)
    text = (
        f"Each peak is {number(1.0 / ratio)} of the one before: δ = ln {number(ratio)} = {number(decrement)}, so "
        f"ζ = {number(zeta)}."
    )
    return RingDown(times, heights, ratio, decrement, zeta, text)


@dataclass(frozen=True)
class EnergyRow:
    name: str
    value: float  # J
    role: str  # its colour, as a theme role's name


@dataclass(frozen=True)
class EnergyView:
    t: np.ndarray
    stored: np.ndarray  # kinetic + potential
    put_in: np.ndarray  # the drive's work and the hand's
    taken_out: np.ndarray  # by the damper and friction
    changes: np.ndarray  # the live changes' net: spring work − capture loss − carried off
    rows: tuple[EnergyRow, ...]  # the ledger now
    residual: float  # what the balance misses by (J): rounding
    text: str


def energy_view(t: np.ndarray, y: np.ndarray) -> EnergyView | None:
    """The energy balance over the record `(t, y)` (its `COLUMNS`) and the ledger at its end, or None if empty."""
    if t.size == 0:
        return None
    c = COL
    stored = y[:, c["kinetic"]] + y[:, c["potential"]]
    put_in = y[:, c["work"]] + y[:, c["released"]]
    taken_out = y[:, c["dissipated"]] + y[:, c["friction"]]
    changes = y[:, c["spring_work"]] - y[:, c["capture_loss"]] - y[:, c["carried_off"]]
    end = y[-1]
    residual = float(put_in[-1] + changes[-1] - taken_out[-1] - stored[-1])
    rows = (
        EnergyRow("Put in by the drive", float(end[c["work"]]), "force"),
        EnergyRow("Put in by your hand", float(end[c["released"]]), "muted"),
        EnergyRow("Changing the spring", float(end[c["spring_work"]]), "energy.potential"),
        EnergyRow("Lost capturing mass", -float(end[c["capture_loss"]]), "energy.added"),
        EnergyRow("Carried off by mass", -float(end[c["carried_off"]]), "energy.added"),
        EnergyRow("Taken out by the damper", -float(end[c["dissipated"]]), "energy.dissipated"),
        EnergyRow("Taken out by friction", -float(end[c["friction"]]), "energy.friction"),
        EnergyRow("Stored now", float(stored[-1]), "energy.stored"),
    )
    scale = max(abs(r.value) for r in rows)
    text = (
        f"Stored now {fmt(float(stored[-1]), 'J')}: what was put in, less what was taken out, with the changes' "
        f"own terms. It closes to {fmt(abs(residual), 'J', 2, smallest='p')}"
        + (f" ({number(abs(residual) / scale, 2)} of the largest term)." if scale > 0 else ".")
    )
    return EnergyView(t, stored, put_in, taken_out, changes, rows, residual, text)
