"""The two masses live (#478 Phase 4, Coordinates & bases): stepped as they run, changed as they run, on a rail or not.

The run steps `core/bases.Steps`, exactly, in coordinates s with x = Bs: free, B is the identity and s = x; on a
rail along the unit vector u, B = u and s is the distance along it, so the rail is a one-mass oscillator with
uᵀMu, uᵀCu and uᵀKu. Every change happens at a step boundary and is a physical event (#478 item 4), its energy
booked as its own term:

- `set_system`: a spring changed keeps x and v, booking ½xᵀΔKx (`spring_work`). A lighter mass drops a piece at its
  speed (`carried_off`), then a heavier one captures its extra at rest, keeping the momentum Bᵀ M v (each mass's
  free, the rail's uᵀMv on it), and loses the capture's energy (`capture_loss`). A damper changes only the loss.
- `set_rail(u)`: put on a rail, the point is moved to the nearest place on it (s = uᵀx) and its velocity across
  the rail taken away by the rail's impulse, which keeps uᵀMv: s' = uᵀMv / uᵀMu. Turned to a new u′, the hand
  carries the point round at the same s and the impulse keeps u′ᵀMv. Taken away (`set_rail(None)`), x = us and
  v = us' run on free, with no jump. The rail's jumps are the hand's work on it (`rail_work`).
- `hold(x)` then `let_go()`: a hand holds the masses still at x (on a rail, at the nearest place on it); the clock
  stops while held, and its work is `released`.
- `advance(duration)`: step on (nothing while held), the samples into `record`. The part of a step left over is
  carried to the next (`Steps`' `StepClock`): it's a `stepping.Stepper`.

`changed_at` is the time of the last change (the system, the rail, a let-go): the rail's frequency is measured only
after it. `edits` keeps the changes' times, for the time plot's marks: a drag's run of changes is one mark, at its
last.

The record (`COLUMNS`) keeps each sample's x and v, the energies with the parameters of the time, and the ledger,
so `energy_view` shows the balance closing through every change.

Working on it: Qt-free (the package's docstring); `tests/test_live_bases.py` checks the rules and the balance.
"""

from __future__ import annotations

import numpy as np

from ..core.bases import Steps, TwoMass, transformed, unit
from ..core.sdof import peaks
from .live_mass import EnergyRow, EnergyView
from .record import SampleBuffer
from .text import fmt, number

COLUMNS = (
    "x1",
    "x2",
    "v1",
    "v2",
    "kinetic",
    "potential",
    "dissipated",
    "released",
    "spring_work",
    "capture_loss",
    "carried_off",
    "rail_work",
)
COL = {name: i for i, name in enumerate(COLUMNS)}
MARK_GAP = 0.25  # s: changes closer than this (a grip's or the rail's drag) share one mark on the time plot
BOOKS = ("released", "spring_work", "capture_loss", "carried_off", "rail_work")


class LiveBases:
    """The two masses, stepped live and changed as they run (see the module)."""

    def __init__(self, system: TwoMass | None = None, capacity: int = 200_000) -> None:
        self.system = system or TwoMass()
        self.rail: np.ndarray | None = None  # the rail's unit direction, or None: free
        self.record = SampleBuffer(len(COLUMNS), capacity)
        self.held: np.ndarray | None = None
        self.changed_at = 0.0
        self.edits: list[float] = []
        self.books = dict.fromkeys(BOOKS, 0.0)
        self.steps = self._stepper(np.zeros(4 if self.rail is None else 2), 0.0, 0.0)

    # ------------------------------------------------------------- the state
    @property
    def basis(self) -> np.ndarray:
        """B: x = Bs (the identity, free; the rail's direction as a column on one)."""
        return np.eye(2) if self.rail is None else self.rail[:, None]

    def _stepper(self, z: np.ndarray, t: float, lost: float) -> Steps:
        M, C, K = self.system.matrices()
        B = self.basis
        steps = Steps(transformed(B, M), transformed(B, C), transformed(B, K), z, t)
        steps.dissipated = lost
        return steps

    @property
    def t(self) -> float:
        return self.steps.t

    @property
    def s(self) -> np.ndarray:
        return self.steps.z[: self.steps.n]

    @property
    def s_dot(self) -> np.ndarray:
        return self.steps.z[self.steps.n :]

    @property
    def x(self) -> np.ndarray:
        return self.basis @ self.s

    @property
    def v(self) -> np.ndarray:
        return self.basis @ self.s_dot

    def stored(self, x: np.ndarray | None = None, v: np.ndarray | None = None) -> float:
        M, _, K = self.system.matrices()
        x = self.x if x is None else x
        v = self.v if v is None else v
        return 0.5 * float(v @ M @ v) + 0.5 * float(x @ K @ x)

    def _restart(self, s: np.ndarray, s_dot: np.ndarray) -> None:
        """The stepper rebuilt for the system and rail now, from (s, s'), the clock and the loss carried on."""
        self.steps = self._stepper(np.concatenate([s, s_dot]), self.t, self.steps.dissipated)

    def _changed(self) -> None:
        """A change now: measured from, and marked (a drag's run of changes, within MARK_GAP, as one mark)."""
        self.changed_at = self.t
        if self.edits and self.t - self.edits[-1] < MARK_GAP:
            self.edits[-1] = self.t
        else:
            self.edits.append(self.t)

    # ------------------------------------------------------------- what the reader does
    def set_system(self, system: TwoMass) -> None:
        """A live change of m, k or c (see the module)."""
        if system == self.system:
            return
        old = self.system
        B = self.basis
        x, v = self.x, self.v
        m_old = np.array([old.m1, old.m2])
        m_new = np.array([system.m1, system.m2])
        kept = np.minimum(m_old, m_new)  # a lighter mass's piece leaves at its speed ...
        self.books["carried_off"] += 0.5 * float(((m_old - kept) * v * v).sum())
        s_dot = self.s_dot
        middle = transformed(B, np.diag(kept))  # ... then a heavier one captures its extra, keeping BᵀMv
        heavier = transformed(B, np.diag(m_new))
        s_dot_new = np.linalg.solve(heavier, middle @ s_dot)
        self.books["capture_loss"] += 0.5 * float(s_dot @ middle @ s_dot) - 0.5 * float(s_dot_new @ heavier @ s_dot_new)
        _, _, K_old = old.matrices()
        _, _, K_new = system.matrices()
        self.books["spring_work"] += 0.5 * float(x @ (K_new - K_old) @ x)
        self.system = system
        self._restart(self.s, s_dot_new)
        self._changed()

    def set_rail(self, u: np.ndarray | None) -> None:
        """Put the masses on a rail along `u`, turn the rail to `u`, or take it away (None): see the module."""
        if u is None:
            if self.rail is None:
                return
            x, v = self.x, self.v
            self.rail = None
            self._restart(x, v)
            self._changed()
            return
        u = unit(u)
        if self.rail is not None and np.allclose(u, self.rail, atol=1e-12):
            return
        before = self.stored()
        M, _, _ = self.system.matrices()
        v = self.v
        s = float(self.s[0]) if self.rail is not None else float(u @ self.x)
        if self.rail is not None and float(u @ self.rail) < 0:
            s = -s  # the rail's direction flipped over (`unit`'s sign): the point stays on the same side
        s_dot = float(u @ M @ v) / float(u @ M @ u)  # the rail's impulse keeps uᵀMv
        self.rail = u
        if self.held is not None:
            self.held = u * s
            s_dot = 0.0
        self._restart(np.array([s]), np.array([s_dot]))
        self.books["rail_work"] += self.stored() - before
        self._changed()

    def hold(self, x: np.ndarray) -> None:
        """A hand holds the masses still at x (on a rail, at its nearest place on it)."""
        x = np.asarray(x, dtype=float)
        before = self.stored()
        s = x.copy() if self.rail is None else np.array([float(self.rail @ x)])
        self.held = self.basis @ s
        self._restart(s, np.zeros_like(s))
        self.books["released"] += self.stored() - before

    def let_go(self) -> None:
        self.held = None
        self.changed_at = self.t

    def reset(self) -> None:
        """Back to t = 0 at rest, the record and books cleared; the system and rail as they are."""
        self.record.clear()
        self.held = None
        self.changed_at = 0.0
        self.edits = []
        self.books = dict.fromkeys(BOOKS, 0.0)
        n = 2 if self.rail is None else 1
        self.steps = self._stepper(np.zeros(2 * n), 0.0, 0.0)

    def advance(self, duration: float) -> int:
        """Step on by about `duration` s (nothing while held): the number of samples recorded."""
        if self.held is not None or duration <= 0:
            return 0
        ts, zs, lost = self.steps.advance(duration)
        if ts.size == 0:
            return 0
        n = self.steps.n
        B = self.basis
        xs, vs = zs[:, :n] @ B.T, zs[:, n:] @ B.T
        M, _, K = self.system.matrices()
        rows = np.empty((ts.size, len(COLUMNS)))
        rows[:, COL["x1"] : COL["x2"] + 1] = xs
        rows[:, COL["v1"] : COL["v2"] + 1] = vs
        rows[:, COL["kinetic"]] = 0.5 * np.einsum("ij,jk,ik->i", vs, M, vs)
        rows[:, COL["potential"]] = 0.5 * np.einsum("ij,jk,ik->i", xs, K, xs)
        rows[:, COL["dissipated"]] = lost
        for name in BOOKS:
            rows[:, COL[name]] = self.books[name]
        self.record.extend(ts, rows)
        return ts.size

    def since_change(self) -> tuple[np.ndarray, np.ndarray]:
        """The record's samples after the last change (`changed_at`): (t, columns)."""
        t, y = self.record.t[: self.record.size], self.record.y[: self.record.size]
        i0 = int(np.searchsorted(t, self.changed_at, side="right"))
        return t[i0:], y[i0:]

    def rail_hz(self) -> float | None:
        """The rail's frequency as measured since the last change: from the peaks of s, maximum to maximum (Hz), or
        None before two peaks (or off the rail)."""
        if self.rail is None:
            return None
        t, y = self.since_change()
        if t.size < 8:
            return None
        s = y[:, COL["x1"] : COL["x2"] + 1] @ self.rail
        times, _ = peaks(t, s)
        if times.size < 2:
            times, _ = peaks(t, -s)  # plucked the other way: its first maximum is a minimum of s
        if times.size < 2:
            return None
        return (times.size - 1) / float(times[-1] - times[0])


def energy_view(t: np.ndarray, y: np.ndarray) -> EnergyView | None:
    """The energy balance over the record `(t, y)` (its `COLUMNS`) and the ledger at its end, or None if empty."""
    if t.size == 0:
        return None
    c = COL
    stored = y[:, c["kinetic"]] + y[:, c["potential"]]
    put_in = y[:, c["released"]] + y[:, c["rail_work"]]
    taken_out = y[:, c["dissipated"]]
    changes = y[:, c["spring_work"]] - y[:, c["capture_loss"]] - y[:, c["carried_off"]]
    end = y[-1]
    residual = float(put_in[-1] + changes[-1] - taken_out[-1] - stored[-1])
    rows = (
        EnergyRow("Put in by your hand", float(end[c["released"]]), "muted"),
        EnergyRow("Moving the rail", float(end[c["rail_work"]]), "force"),
        EnergyRow("Changing the springs", float(end[c["spring_work"]]), "energy.potential"),
        EnergyRow("Lost capturing mass", -float(end[c["capture_loss"]]), "energy.added"),
        EnergyRow("Carried off by mass", -float(end[c["carried_off"]]), "energy.added"),
        EnergyRow("Taken out by the dampers", -float(end[c["dissipated"]]), "energy.dissipated"),
        EnergyRow("Stored now", float(stored[-1]), "energy.stored"),
    )
    scale = max(abs(r.value) for r in rows)
    text = (
        f"Stored now {fmt(float(stored[-1]), 'J')}: what was put in, less what was taken out, with the changes' "
        f"own terms. It closes to {fmt(abs(residual), 'J', 2, smallest='p')}"
        + (f" ({number(abs(residual) / scale, 2)} of the largest term)." if scale > 0 else ".")
    )
    return EnergyView(t, stored, put_in, taken_out, changes, rows, residual, text)
