"""One mass on a spring, a damper and a relaxing arm, stepped live: the Zener (standard linear solid) element in time.

The arm is a spring k₁ in series with a dashpot c₁, beside the spring k₀ and the damper c (Damping models' viscoelastic
element; the chain's `Simulator` has no such arm, its elements being a spring and a damper side by side). The arm's
inner end, between its spring and its dashpot, is at y, a state of its own with no mass:

    m ẍ = −k₀x − cẋ − k₁(x − y) + f,      c₁ẏ = k₁(x − y)

so z = [x, ẋ, y] is linear, z' = A z + b f, and steps by the exact first-order-hold update, as the chain's do. The
energy books are exact the same way (Van Loan's integrals, `simulator.foh_quadratic_integrals`): the drive's work
∫fẋ, and what the dampers take out, ∫(cẋ² + c₁ẏ²). Stored is ½mẋ² + ½k₀x² + ½k₁(x − y)².

`ArmMass` is shaped as `Simulator` is to `view_models/live_mass.LiveMass`: the clock `t`, a `ForceController`, the
state's `displacement`/`velocity`, `set_state`, `set_system` (a live change), `advance(duration)` and the ledger
(`ledger`, `work`, `dissipated`, `friction_loss`, and the change terms `released`, `spring_work`, `capture_loss`,
`carried_off`, summed in `energy_added`), and `arm_track`, y at each sample of the last advance. A live change keeps by `Simulator.set_system`'s rules: x carries on, a
heavier mass captures its extra at rest (momentum kept), a lighter one drops a piece at its speed; and the arm keeps
its stretch (y carries on), so a change of k₀ or k₁ books ½Δk times its stretch squared as spring work. Set to a
place by a hand (`set_state`), the arm is relaxed there (y = x): held long, as a pluck is.

Working on it: `tests/test_live_parts.py` checks it against `damping_models.Matched.free_decay`'s Zener decay, the
books closing through its changes, and a plain viscous mass against the chain's `Simulator`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .forcing import ForceController
from .simulator import MAX_STEP, STEPS_PER_PERIOD, foh_input_step, foh_quadratic_integrals


@dataclass(frozen=True)
class ArmSystem:
    """m, k₀, c and the arm's k₁ and c₁ (SI units)."""

    mass: float
    stiffness: float  # k₀
    damping: float  # c, beside it
    arm_stiffness: float  # k₁
    arm_damping: float  # c₁

    def __post_init__(self) -> None:
        if not (self.mass > 0 and self.stiffness >= 0 and self.damping >= 0):
            raise ValueError("the mass must be positive, the spring and the damper not negative")
        if not (self.arm_stiffness > 0 and self.arm_damping > 0):
            raise ValueError("the arm's spring and dashpot must be positive")

    def matrices(self) -> tuple[np.ndarray, np.ndarray]:
        m, k0, c, k1, c1 = self.mass, self.stiffness, self.damping, self.arm_stiffness, self.arm_damping
        A = np.array([[0.0, 1.0, 0.0], [-(k0 + k1) / m, -c / m, k1 / m], [k1 / c1, 0.0, -k1 / c1]])
        return A, np.array([0.0, 1.0 / m, 0.0])

    def potential(self, x: float, y: float) -> float:
        return 0.5 * self.stiffness * x * x + 0.5 * self.arm_stiffness * (x - y) ** 2

    def fastest_hz(self) -> float:
        lam = np.linalg.eigvals(self.matrices()[0])
        return float(np.abs(lam).max()) / (2.0 * math.pi)


class ArmMass:
    """One mass with a Zener arm, stepped live (see the module)."""

    def __init__(self, system: ArmSystem, force: ForceController | None = None) -> None:
        self.force = force or ForceController()
        self.system = system
        self.t = 0.0
        self.state = np.zeros(3)  # x, ẋ, y
        self._u = 0.0  # the input where the last step ended
        self.work = self.dissipated = self.friction_loss = 0.0
        self.energy_added = self.released = self.spring_work = self.capture_loss = self.carried_off = 0.0
        self.edits: list[float] = []
        self.ledger = np.empty((0, 4))
        self.arm_track = np.empty(0)  # y at each sample of the last advance
        self._cache: dict[float, tuple] = {}

    # ------------------------------------------------------------- as Simulator is
    @property
    def displacement(self) -> np.ndarray:
        return self.state[:1]

    @property
    def velocity(self) -> np.ndarray:
        return self.state[1:2]

    @property
    def arm(self) -> float:
        """y: where the arm's spring meets its dashpot."""
        return float(self.state[2])

    @property
    def arm_force(self) -> float:
        """The arm's pull on the mass, −k₁(x − y) (N)."""
        return -self.system.arm_stiffness * (self.state[0] - self.state[2])

    @property
    def stored_energy(self) -> float:
        x, v, y = self.state
        return 0.5 * self.system.mass * v * v + self.system.potential(x, y)

    def step_size(self) -> float:
        fmax = max(self.system.fastest_hz(), self.force.settings.max_freq_hz)
        h = MAX_STEP if fmax <= 0 else min(MAX_STEP, 1.0 / (STEPS_PER_PERIOD * fmax))
        return 2.0 ** math.floor(math.log2(h))

    def reset(self) -> None:
        self.t = 0.0
        self.state = np.zeros(3)
        self.force.reset()
        self._u = 0.0
        self.work = self.dissipated = self.friction_loss = 0.0
        self.energy_added = self.released = self.spring_work = self.capture_loss = self.carried_off = 0.0
        self.edits = []

    def set_state(self, x: np.ndarray, v: np.ndarray) -> None:
        """Set x and ẋ (a hand's), the arm relaxed there: the jump in stored energy is `released`."""
        before = self.stored_energy
        self.state = np.array([float(x[0]), float(v[0]), float(x[0])])
        jump = self.stored_energy - before
        self.released += jump
        self.energy_added += jump

    def set_system(self, system: ArmSystem) -> None:
        """A live change (see the module)."""
        old = self.system
        x, v, y = (float(q) for q in self.state)
        if system.mass > old.mass:
            v_new = old.mass * v / system.mass
            capture, carried = 0.5 * old.mass * v * v * (system.mass - old.mass) / system.mass, 0.0
        else:
            v_new = v
            capture, carried = 0.0, 0.5 * (old.mass - system.mass) * v * v
        spring = system.potential(x, y) - old.potential(x, y)
        self.state = np.array([x, v_new, y])
        self.system = system
        self._cache.clear()
        self.spring_work += spring
        self.capture_loss += capture
        self.carried_off += carried
        self.energy_added += spring - capture - carried
        self.edits.append(self.t)

    # ------------------------------------------------------------- stepping
    def _maps(self, h: float) -> tuple:
        if h not in self._cache:
            A, b = self.system.matrices()
            Phi, g0, g1 = foh_input_step(A, b, None, h)
            s = self.system
            Qd = np.zeros((5, 5))  # w = [x, ẋ, y, u, du]: c ẋ² + c₁ ẏ², ẏ = (k₁/c₁)(x − y)
            Qd[1, 1] = s.damping
            a = np.array([1.0, 0.0, -1.0, 0.0, 0.0])
            Qd += (s.arm_stiffness**2 / s.arm_damping) * np.outer(a, a)
            Qw = np.zeros((5, 5))
            Qw[3, 1] = Qw[1, 3] = 0.5  # u ẋ
            Wd, Ww = foh_quadratic_integrals(A, b, h, [Qd, Qw])
            self._cache[h] = (Phi, g0, g1, Wd, Ww)
        return self._cache[h]

    def advance(self, duration: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Step on by about `duration` s: the samples (t, x, ẋ, f), shaped as `Simulator.advance`'s."""
        h = self.step_size()
        steps = max(0, round(duration / h))
        if steps == 0:
            self.ledger, self.arm_track = np.empty((0, 4)), np.empty(0)
            return np.empty(0), np.empty((0, 1)), np.empty((0, 1)), np.empty(0)
        Phi, g0, g1, Wd, Ww = self._maps(h)
        fs = self.force.samples(h, steps)
        ts = np.cumsum(np.concatenate([[self.t], np.full(steps, h)]))[1:]
        zs = np.empty((steps, 3))
        z, f0 = self.state.copy(), self._u
        for k, f1 in enumerate(fs):
            z = Phi @ z + g0 * f0 + g1 * f1
            zs[k] = z
            f0 = f1
        z0s = np.vstack([self.state, zs[:-1]])
        f0s = np.concatenate([[self._u], fs[:-1]])
        w = np.column_stack([z0s, f0s, (fs - f0s) / h])
        dissipated = self.dissipated + np.cumsum(((w @ Wd) * w).sum(axis=1))
        work = self.work + np.cumsum(((w @ Ww) * w).sum(axis=1))
        self.dissipated, self.work = float(dissipated[-1]), float(work[-1])
        self.ledger = np.column_stack([np.full(steps, self.energy_added), work, dissipated, np.zeros(steps)])
        self.arm_track = zs[:, 2].copy()
        self.state = zs[-1].copy()
        self._u = float(fs[-1])
        self.t = float(ts[-1])
        return ts, zs[:, :1], zs[:, 1:2], fs
