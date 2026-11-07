"""The rotor's time response: the exact FOH update of `RotorSystem.state_space`, held over chunks at a speed,
with the films' nonlinear remainder (`_Films`), the stator ring's rub and the thermal bow solved over each
step. The model is core.rotor's.

Working on it
-------------
With `nonlinear_films` the simulator steps the films' whole force instead (`whole_films`: on a round shaft; a
turning shaft keeps them linear). `pi_film` gives it in closed form with its stiffness, damping and
dissipation Ψ (tests check it against `film_force` and finite differences). `RotorSimulator._film_steps`
(`core/rotor_simulator.py`, beside the model in `core/rotor.py` and the whirl, Campbell and steady states in
`core/rotor_modes.py`) keeps the exact FOH update, with each film's K_f and p but its damping replaced by the
bound C_b (`full_film_damping`, the uncavitated film's, taken afresh as the journal moves), and adds the
remainder as a load on the journals, found each step by Newton (`_Films.solve`, cut back on the convex Q).
Don't put C_f back in the update: near the wall the remainder would be stiffer than the update, the step
rings, and the journal is thrown through the wall. The Rust kernel (`native.FilmSteps`, one per held speed and
step from `_Films.stepper`) runs `_film_steps_reference`'s loop and takes the bound itself as the journals
drift, with its own FOH exponential (`linalg.rs`'s `expm_foh`: Al-Mohy–Higham on the augmented matrix's
blocks, balanced first, since A h runs from h to K/M h; more accurate than SciPy's here). ~1.6 µs a step, ~6
µs on Oil whirl, where the journals circle near the wall and the bound is retaken every ~15 steps (the Python:
45–120 µs). Keep the Rust in step with `_film_steps_reference` and `_Films`; `tests/test_native.py` compares
them while ramping and through retakes, and the FOH with `foh_discretize`. The step only resolves the spin and
the modes coarsely (`FILM_MODE_STEPS`), and the page caps each frame's computing (`step(…, budget)`).
Everything else (Campbell, stability, steady states) stays linearized.

A stator ring round the disc (`RotorSystem.stator`, a `Stator`; the *Rub* tab, `gui/rotor_rub.py`) is in the
time response only: `stator_force` gives the ring's push and smoothed-Coulomb drag with its derivatives, and
`RotorSimulator._rub_steps` adds it to the exact FOH update as a load on the disc, linear over the step,
solved by Newton (`_rub_solve`), as the films' remainder is. The Rust kernel (`native.RubSteps`, built per
call from the chunk's Φ, Γ0, Γ1) runs `_rub_steps_reference`'s loop, `stator_force` and the heat included:
~0.6 µs a step against 10–60 in the Python; keep them in step (`tests/test_native.py` compares them).
`step_size` resolves the disc on the ring's mount. Only on a round shaft with the films linearized
(`__post_init__` raises; the page drops the ring and disables its checkbox). Campbell, stability and the
steady states leave it out. Each advance's `RotorSamples.rub` holds the push; the page's history keeps the
largest since the last sample kept (`PUSH`). A forward full rub's steady state is a linear one with the ring
as a cross-coupled spring at the disc; a test checks it. With `thermal_gain` the rub's heat (column 3 of
`rub`) bows the shaft (Newkirk): `RotorSimulator.thermal_bow`, complex from the mark, held over chunks of at
most τ/`THERMAL_STEPS` and moved on by `_warm` (exact first-order lag), loads the shaft as the residual bow
does (`bow_load`, in `force`). The page records the spiral (`_record_spiral`) at a steady speed only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import native
from .fluid_bearing import full_film_damping, pi_film
from .rotor import (
    N_DOF,
    SLIP,
    XA,
    XB,
    XD,
    YD,
    RotorSystem,
    Stator,
    stator_force,
)
from .simulator import MAX_STEP, MAX_STEPS_PER_ADVANCE, STEPS_PER_PERIOD, foh_discretize

CHUNK_TIME = 2e-3  # s; while the speed ramps, A(Ω) is held over chunks this long
RPM = 60.0 / (2.0 * math.pi)  # rpm per rad/s
FILM_MODE_STEPS = 10  # steps per period of the fastest mode (the spin keeps STEPS_PER_PERIOD)
FILM_DRIFT = 0.3  # the films' damping bound is taken afresh once it has moved this much (of its largest term)
FILM_TOL = 1e-7  # Newton stops when the remainder's residual is this small, over (1 N + the largest load)
FILM_ITERATIONS = 12  # and after this many iterations at most
FILM_CORRECTION = 1e-4  # a Newton correction this small (over the same) is taken without checking: it converges
RUB_TOL = 1e-9  # Newton stops when the contact force's residual is this small, over (1 N + k clearance)
RUB_ITERATIONS = 30
THERMAL_STEPS = 200  # a thermal bow is updated at least this often per time constant


# ------------------------------------------------------------------ time stepping


@dataclass
class RotorSamples:
    """What one advance returns, one row per step."""

    t: np.ndarray  # (k,) s
    q: np.ndarray  # (k, n_dof) displacements
    omega: np.ndarray  # (k,) spin speed (rad/s)
    phase: np.ndarray  # (k,) angle φ of the keyphasor mark (rad, unwrapped)
    rub: np.ndarray | None = None  # (k, 4) with a stator: its force on the disc (f_x, f_y), the normal force N
    # and the friction's heat (W)
    bow: np.ndarray | None = None  # (k,) complex with a thermal bow: the bow at the disc (m), from the mark


class _Films:
    """The oil films' whole force on the journals, for RotorSimulator._film_steps.

    The exact update carries each film linearized: its stiffness K_f and load p
    at the held speed (as the linear films have), and in place of its damping
    C_f the bound C_b, the whole uncavitated film's damping where the journal
    was when it was last taken. The rest of the film's force, the remainder r_f
    = f - (p - K_f q - C_b q'), goes in as a force on the journals.
    """

    def __init__(self, system: RotorSystem) -> None:
        self.system = system
        n = system.n_dof
        films = [(j, b.film) for j, b in ((XA, system.bearing_a), (XB, system.bearing_b)) if b.film is not None]
        loads = dict(zip((XA, XB), system.bearing_loads))
        self.dofs = [d for j, _ in films for d in (j, j + 4)]  # x and y of each film's journal
        self.rates = [d + n for d in self.dofs]
        # (k, c, the speed below which the film is held): per film.
        self.films = [(f.film_scale, f.clearance, f.with_load(abs(loads[j])).slowest()) for j, f in films]
        self.scale = 1.0 + max(abs(w) for w in system.bearing_loads)
        self.held = math.nan
        self.key: tuple | None = None
        self.kernel: native.FilmSteps | None = (
            None  # the Rust film steps at the held speed (`stepper`), and their (h, Ω)
        )
        self.kernel_key: tuple | None = None

    def __getstate__(self) -> dict:
        # The Rust kernel doesn't copy or pickle: a copy builds its own, and takes the bound afresh.
        return {**self.__dict__, "kernel": None, "kernel_key": None}

    def hold(self, omega: float) -> None:
        """Take the films' K_f and p at the speed Ω the exact update holds."""
        if omega == self.held:
            return
        Kf, Cf, p = self.system._film_terms(omega)
        d = self.dofs
        self.held = omega
        self.Kf, self.Cf = Kf[np.ix_(d, d)], Cf[np.ix_(d, d)]
        self.p = p[d] if self.system.gravity else np.zeros(len(d))
        self.A = self.system.state_matrix(omega)  # each take changes only the films' damping in it

    def bound(self, z: np.ndarray) -> list[tuple[float, float, float]]:
        """Each film's whole damping (c_xx, c_xy, c_yy) where its journal is in the state z."""
        x = z[self.dofs].tolist()
        return [full_film_damping(k, c, x[2 * i], x[2 * i + 1]) for i, (k, c, _) in enumerate(self.films)]

    def take(self, z: np.ndarray, h: float, omega: float) -> None:
        """Bound the damping where the journals are in z, and discretize over h at the held speed Ω."""
        system, n, d = self.system, self.system.n_dof, self.dofs
        self.hold(omega)
        self.taken = self.bound(z)
        self.size = max(abs(c) for b in self.taken for c in b)
        self.Cb = np.zeros((len(d), len(d)))
        for i, (cxx, cxy, cyy) in enumerate(self.taken):
            self.Cb[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = ((cxx, cxy), (cxy, cyy))
        A = self.A.copy()
        A[n : 2 * n, self.rates] -= system.mass_inverse[:, d] @ (self.Cb - self.Cf)
        self.Phi, self.G0, self.G1 = foh_discretize(A, system.input_matrix(), h)
        self.R0, self.R1 = self.G0[:, d], self.G1[:, d]
        self.Ex, self.Ev = self.R1[d], self.R1[self.rates]  # the journals' x and x' at the step's end, per force
        self.E = np.vstack([self.Ex, self.Ev])
        self.Es = 0.5 * (self.Ev + self.Ev.T)  # ½ Δvᵀ sym(E_v⁻¹) Δv = ½ rᵀ sym(E_v) r
        self.lin = np.hstack([self.Kf, self.Cb])  # the update's films: p - K_f x - C_b x'
        self.eye = np.eye(len(d))
        self.key = (h, omega)

    def stepper(self, h: float):
        """The Rust kernel's steps over h at the held speed (`native.FilmSteps`): it takes the
        bound itself, as `take` does, where it first steps and as the journals drift."""
        if self.kernel_key != (h, self.held):
            assert native.module is not None  # the caller checked
            d = self.dofs
            self.kernel = native.FilmSteps(
                self.A,
                self.system.input_matrix(),
                self.system.mass_inverse[:, d],
                self.Cf,
                self.Kf,
                self.p,
                h,
                d + self.rates,
                self.films,
                FILM_DRIFT,
                FILM_TOL * self.scale,
                FILM_CORRECTION * self.scale,
                FILM_ITERATIONS,
            )
            self.kernel_key = (h, self.held)
        return self.kernel

    def drifted(self, z: np.ndarray) -> bool:
        """True once a journal has moved far enough that the damping bound should be taken afresh."""
        limit = FILM_DRIFT * self.size
        return any(abs(a - b) > limit for new, old in zip(self.bound(z), self.taken) for a, b in zip(new, old))

    def forces(self, y: np.ndarray, omega: float) -> tuple[np.ndarray, np.ndarray, float]:
        """(f, D, Ψ) with the journals at y = [x, x']: the films' force, their stiffness and damping
        [K C] (each film's rows), and their dissipation."""
        m = len(self.dofs)
        f, D = np.empty(m), np.zeros((m, 2 * m))
        psi = 0.0
        for i, (k, c, slowest) in enumerate(self.films):
            j = 2 * i
            o = pi_film(k, c, y[j], y[j + 1], y[m + j], y[m + j + 1], max(abs(omega), slowest))
            f[j : j + 2] = o[:2]
            D[j : j + 2, j : j + 2] = (o[2:4], o[4:6])
            D[j : j + 2, m + j : m + j + 2] = ((o[6], o[7]), (o[7], o[8]))
            psi += o[9]
        return f, D, psi

    def remainder(self, z: np.ndarray, omega: float) -> np.ndarray:
        """r at the state z: the films' force less its part in the exact update."""
        y = z[self.dofs + self.rates]
        return self.forces(y.tolist(), omega)[0] - self.p + self.lin @ y

    def solve(self, zb: np.ndarray, r: np.ndarray, omega: float) -> np.ndarray:
        """r at the step's end, where z = zb + R1 r: Newton from the guess r.

        The journals' velocities settle where a convex function is least, Q =
        their inertia over the step, ½ rᵀ E_v r, plus the films' dissipation Ψ, less
        what the update carries: a Newton step that does not lower it, nor halve
        the residual, is cut back.
        """
        m = len(self.dofs)
        yb = zb[self.dofs + self.rates]
        tol = FILM_TOL * self.scale

        def at(r: np.ndarray) -> tuple:
            y = yb + self.E @ r
            f, D, psi = self.forces(y.tolist(), omega)
            x, v = y[:m], y[m:]
            q = 0.5 * r @ self.Es @ r + psi - 0.5 * v @ self.Cb @ v + (self.p - self.Kf @ x) @ v
            return r - f + self.p - self.lin @ y, D, q

        R, D, q = at(r)
        for _ in range(FILM_ITERATIONS):
            size = np.abs(R).max()
            if size <= tol:
                break
            dr = -np.linalg.solve(self.eye + (D - self.lin) @ self.E, R)
            if np.abs(dr).max() <= FILM_CORRECTION * self.scale:  # what is left is far smaller still
                return r + dr
            slope = R @ (self.Ev @ dr)
            t = 1.0
            while True:
                Rt, Dt, qt = at(r + t * dr)
                if qt <= q + 1e-4 * t * slope or np.abs(Rt).max() <= 0.5 * size or t < 1e-3:
                    break
                t *= 0.5
            r, R, D, q = r + t * dr, Rt, Dt, qt
        return r


def _rub_solve(
    stator: Stator, radius: float, yb: np.ndarray, E: np.ndarray, r: np.ndarray, omega: float, tol: float
) -> tuple[np.ndarray, float]:
    """(r, N): the ring's force on the disc at the step's end, where the disc's (x, y, x', y') = yb + E r, by
    Newton from the guess r (each step cut back until it shrinks the residual), and its normal force."""

    def at(r: np.ndarray) -> tuple[np.ndarray, list[float], float]:
        x, y, vx, vy = (yb + E @ r).tolist()
        fx, fy, normal, J = stator_force(stator, radius, x, y, vx, vy, omega)
        return r - (fx, fy), J, normal

    R, J, normal = at(r)
    for _ in range(RUB_ITERATIONS):
        size = np.abs(R).max()
        if size <= tol:
            break
        dr = -np.linalg.solve(np.eye(2) - np.reshape(J, (2, 4)) @ E, R)
        t = 1.0
        while True:
            Rt, Jt, nt = at(r + t * dr)
            if np.abs(Rt).max() < size or t < 1e-3:
                break
            t *= 0.5
        r, R, J, normal = r + t * dr, Rt, Jt, nt
    return r, normal


class RotorSimulator:
    """Steps the rotor exactly (FOH matrix exponential) while its speed follows a target."""

    def __init__(self, system: RotorSystem | None = None) -> None:
        self.system = system or RotorSystem()
        self.t = 0.0
        self.phase = 0.0  # φ: angle of the keyphasor mark (rad, unwrapped)
        self.state = self._rest()
        self.omega = 0.0  # spin speed now (rad/s)
        self.target = 0.0  # the speed it ramps toward (rad/s)
        self.accel = 500.0 / RPM  # ramp rate (rad/s²)
        self._alpha = 0.0  # Ω' over the last step
        self._cache: dict[tuple[float, float, float | None], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._gravity_at: tuple[float, np.ndarray] = (math.nan, np.empty(0))  # gravity's load, and its speed
        self._f_at: dict[float, float] = {}  # with oil films: the highest frequency at each top speed asked
        self._f_still = self._highest_freq(0.0)
        self._films = _Films(self.system) if self.system.whole_films else None
        self.thermal_bow = 0j  # the rub's thermal bow at the disc (m), turning with the shaft, from the mark

    # ------------------------------------------------------------- setup
    def set_system(self, system: RotorSystem) -> None:
        """New parameters, keeping the motion (live editing).

        If the shaft's modal coordinates change in number, the stations keep their
        motion and the shaft starts again from its static shape between them. If
        the magnets' controllers change their states, those start where they hold
        the rotor's weight.
        """
        old = self.system
        if system.n_dof != old.n_dof or system.n_control != old.n_control:
            n0, n1 = old.n_dof, system.n_dof
            state = np.zeros(system.n_states)
            if n1 == n0:
                state[: 2 * n1] = self.state[: 2 * n0]
            else:
                state[:N_DOF], state[n1 : n1 + N_DOF] = self.state[:N_DOF], self.state[n0 : n0 + N_DOF]
            if system.n_control == old.n_control:
                state[2 * n1 :] = self.state[2 * n0 :]
            else:
                state[2 * n1 :] = system.static_state(self.omega)[2 * n1 :]
            self.state = state
        self.system = system
        self._cache.clear()
        self._gravity_at = (math.nan, np.empty(0))
        self._f_at.clear()
        self._f_still = self._highest_freq(0.0)
        self._films = _Films(system) if system.whole_films else None
        if not self.thermal:
            self.thermal_bow = 0j

    @property
    def thermal(self) -> bool:
        """True if the rub bows the shaft (a stator with a thermal gain)."""
        return self.system.stator is not None and self.system.stator.thermal_gain > 0.0

    def _rest(self) -> np.ndarray:
        """The state at rest: the shaft in its bowed shape (which loads nothing), sagging under gravity."""
        s = self.system
        q = s.bow_displacement(self.phase) + s.rest_deflection(self.phase)
        z = np.concatenate([q, np.zeros_like(q), np.zeros(s.n_control)])
        if s.magnets:  # their controllers already holding the weight
            z[2 * s.n_dof :] = s.static_state(0.0, self.phase if s.periodic else None)[2 * s.n_dof :]
        return z

    def reset(self) -> None:
        """Back to rest, not spinning (the target speed is kept)."""
        self.t = self.phase = self.omega = self._alpha = 0.0
        self.thermal_bow = 0j
        self.state = self._rest()

    def tap(self, impulse_x: float = 0.0, impulse_y: float = 0.0) -> None:
        """Hit the disc: an impulse (N·s) in x and y changes the velocities at once, by M^{-1} times it."""
        Minv = self.system.mass_inverse
        n = self.system.n_dof
        self.state[n : 2 * n] += Minv[:, XD] * impulse_x + Minv[:, YD] * impulse_y

    @property
    def q(self) -> np.ndarray:
        return self.state[: self.system.n_dof]

    @property
    def qdot(self) -> np.ndarray:
        return self.state[self.system.n_dof : 2 * self.system.n_dof]

    @property
    def control(self) -> np.ndarray:
        """The magnets' controllers' states (see RotorSystem._control); empty without magnets."""
        return self.state[2 * self.system.n_dof :]

    def energy(self) -> tuple[float, float]:
        """(kinetic, potential) energy of the vibration (J); the spin's own energy left out."""
        M = self.system.matrices()[0]
        # Measured from the equilibrium: the shaft's strain from its bow (zero at the journals, where
        # the supports act), the springs' from where gravity holds them (gravity's own potential energy
        # cancels the rest). Only K's symmetric part stores energy: the cross-coupling does work instead.
        # An asymmetric shaft's strain energy is on its stiffness at this angle, about its mean sag. An oil
        # film's is on its stiffness at this speed (its skew part drops out of q K q).
        K = self.system.stiffness_at(self.omega, self.phase)
        q = self.q - self.system.bow_displacement(self.phase) - self.system.static_deflection(self.omega)
        return 0.5 * float(self.qdot @ M @ self.qdot), 0.5 * float(q @ K @ q)

    # ------------------------------------------------------------- stepping
    def _highest_freq(self, omega: float) -> float:
        """The fastest root at speed Ω (Hz), |λ|/2π; with oil films, the fastest oscillation, |Im λ|/2π.

        A film damps its journal so hard that its roots decay at once without
        oscillating, far faster than anything else: the exact update needs no
        short step for them, only for what oscillates. So with a magnet's
        controller, whose filter and amplifier are fast real roots.
        """
        lam = np.linalg.eigvals(self.system.state_matrix(omega))
        fast = np.abs(lam.imag) if self.system.films or self.system.magnets else np.abs(lam)
        return float(fast.max()) / (2.0 * math.pi)

    def step_size(self) -> float:
        """h resolving the fastest mode at any speed up to the target, and the spin.

        A forward mode's frequency rises with speed by at most (Ip/Id) Ω (the shaft's
        own sections: 2Ω). An oil film's stiffness changes with the speed, so with
        films the modes at the top speed are looked at too. With the films' whole
        force the exact update still carries every mode, and only the force on the
        journals is stepped: the step resolves the spin, and the modes only to
        FILM_MODE_STEPS a period. Rounded down to a power of two so the
        discretization is reused.
        """
        top = max(abs(self.omega), abs(self.target))
        spin = top / (2.0 * math.pi)
        gyro = max(self.system.ip / self.system.id, 2.0 if self.system.density else 0.0)
        fmax = max(self._f_still + spin * gyro, spin)
        if self.system.films:
            if top not in self._f_at:
                if len(self._f_at) > 64:
                    self._f_at.clear()
                self._f_at[top] = self._highest_freq(top)
            fmax = max(fmax, self._f_at[top])
        if self._films is not None:
            fmax = max(spin, fmax * FILM_MODE_STEPS / STEPS_PER_PERIOD)
        if self.system.stator is not None:  # the disc bouncing on the ring
            fmax = math.hypot(fmax, math.sqrt(self.system.stator.stiffness / self.system.disc_mass) / (2.0 * math.pi))
        h = MAX_STEP if fmax <= 0 else min(MAX_STEP, 1.0 / (STEPS_PER_PERIOD * fmax))
        return 2.0 ** math.floor(math.log2(h))

    def _discrete(
        self, h: float, omega: float, cache: bool, phase: float | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The FOH update over h at speed Ω; with `phase`, an asymmetric shaft standing still at that angle."""
        key = (h, omega, phase)
        if key in self._cache:
            return self._cache[key]
        A = self.system.state_matrix(omega)
        if phase is not None:
            n = self.system.n_dof
            A[n : 2 * n, :n] = -self.system.mass_inverse @ self.system.stiffness_at(omega, phase)
        out = foh_discretize(A, self.system.input_matrix(), h)
        if cache:
            if len(self._cache) > 8:
                self._cache.clear()
            self._cache[key] = out
        return out

    def force(
        self, phase: np.ndarray | float, omega: np.ndarray | float, alpha: np.ndarray | float, held: float = 0.0
    ) -> np.ndarray:
        """The load f (..., n_dof) on q: the 1X unbalance, skew moment and bow force, and gravity.

        Gravity's load with oil films is taken at the speed `held`, as A(Ω) is.
        """
        p = self.system.rotating_load(omega, alpha)
        if self.thermal_bow:  # the thermal bow loads the shaft as the residual bow does
            p = p + self.thermal_bow * self.system.bow_load
        p = p * np.exp(1j * np.asarray(phase))[..., None]
        return self.system.plane_load(p) + self._gravity(held)

    def _gravity(self, omega: float) -> np.ndarray:
        if not self.system.films:
            omega = 0.0
        if self._gravity_at[0] != omega:
            self._gravity_at = (omega, self.system.gravity_load(omega))
        return self._gravity_at[1]

    def _speeds(self, h: float, n: int) -> np.ndarray:
        """Ω at the end of each of the next n steps, ramping toward the target at self.accel."""
        gap = self.target - self.omega
        if gap == 0.0:
            return np.full(n, self.target)
        ramp = self.omega + math.copysign(self.accel * h, gap) * np.arange(1, n + 1)
        return np.minimum(ramp, self.target) if gap > 0 else np.maximum(ramp, self.target)

    def advance(self, duration: float) -> RotorSamples:
        """Advance by about `duration` s of simulated time (at most MAX_STEPS_PER_ADVANCE steps)."""
        h = self.step_size()
        steps = min(MAX_STEPS_PER_ADVANCE, max(0, round(duration / h)))
        n_dof = self.system.n_dof
        ts, qs = np.empty(steps), np.empty((steps, n_dof))
        ws, phis = np.empty(steps), np.empty(steps)
        rubs = np.zeros((steps, 4)) if self.system.stator is not None else None
        bows = np.empty(steps, dtype=complex) if self.thermal else None
        chunk = max(1, int(CHUNK_TIME / h))
        # A thermal bow is held over each chunk, then moved on by the chunk's heat.
        stator = self.system.stator
        warm = max(1, int(stator.thermal_lag / (THERMAL_STEPS * h))) if self.thermal and stator else steps
        k = 0
        while k < steps:
            steady = self.omega == self.target
            n = min(steps - k, warm) if steady else min(steps - k, chunk, warm)
            w1 = self._speeds(h, n)
            w0 = np.concatenate([[self.omega], w1[:-1]])
            alpha = (w1 - w0) / h
            phi1 = self.phase + np.cumsum(0.5 * h * (w0 + w1))  # exact for Ω linear over each step
            # A is held at the chunk's mean speed (exactly the speed when it is steady).
            held = float(np.mean(w1)) if not steady else self.omega
            u0 = self.force(self.phase, self.omega, self._alpha, held)
            u = self.force(phi1, w1, alpha, held)
            # A turning shaft standing still has a fixed stiffness: stepped exactly, like a round one.
            still = self.system.periodic and steady and self.omega == 0.0
            if self._films is not None:
                z = self._film_steps(h, held, w1, u0, u, qs[k : k + n])
            elif self.system.periodic and not still:
                z = self._split_steps(h, held, steady, w0, w1, alpha, u0, u, qs[k : k + n])
            else:
                Phi, G0, G1 = self._discrete(h, held, steady, self.phase if still else None)
                drive = G0 @ np.vstack([u0, u[:-1]]).T + G1 @ u.T
                if rubs is not None:
                    z = self._rub_steps(Phi, G0, G1, drive, w1, qs[k : k + n], rubs[k : k + n])
                else:
                    z = self.state
                    for j in range(n):
                        z = Phi @ z + drive[:, j]
                        qs[k + j] = z[:n_dof]
            self.state = z
            if bows is not None:
                assert rubs is not None  # the heat comes from a rub
                bows[k : k + n] = self.thermal_bow
                self._warm(n * h, rubs[k : k + n, 3], qs[k : k + n], phi1)
            ts[k : k + n] = self.t + h * np.arange(1, n + 1)
            ws[k : k + n], phis[k : k + n] = w1, phi1
            self.t = float(ts[k + n - 1])
            self.omega, self.phase, self._alpha = float(w1[-1]), float(phi1[-1]), float(alpha[-1])
            k += n
        return RotorSamples(ts, qs, ws, phis, rubs, bows)

    def _warm(self, span: float, heat: np.ndarray, q: np.ndarray, phase: np.ndarray) -> None:
        """Move the thermal bow on over `span` s of the rub's `heat` (W) at each step, the disc at q.

        The heat goes into the shaft where it touches the ring: toward the disc's
        centre's direction from the ring's, seen on the shaft (from the mark). Its
        mean over the span drives τ b' = β H - b, exact for it held.
        """
        stator = self.system.stator
        assert stator is not None  # only rubbing heats
        cx, cy = stator.centre
        hot = heat * np.exp(1j * (np.arctan2(q[:, YD] - cy, q[:, XD] - cx) - phase))
        decay = math.exp(-span / stator.thermal_lag)
        self.thermal_bow = self.thermal_bow * decay + stator.thermal_gain * (1.0 - decay) * complex(np.mean(hot))

    def _rub_steps(
        self,
        Phi: np.ndarray,
        G0: np.ndarray,
        G1: np.ndarray,
        drive: np.ndarray,
        w1: np.ndarray,
        out: np.ndarray,
        rubs: np.ndarray,
    ) -> np.ndarray:
        """`_rub_steps_reference`'s steps, by the Rust kernel (`native.RubSteps`) where it's built."""
        if native.module is None:
            return self._rub_steps_reference(Phi, G0, G1, drive, w1, out, rubs)
        system = self.system
        stator = system.stator
        assert stator is not None  # only called with a ring
        kernel = native.RubSteps(
            Phi,
            G0,
            G1,
            system.n_dof,
            XD,
            YD,
            (stator.clearance, stator.stiffness, stator.damping, stator.friction, *stator.centre),
            SLIP,
            system.disc_radius,
            RUB_TOL * (1.0 + stator.stiffness * stator.clearance),
            RUB_ITERATIONS,
        )
        return np.asarray(kernel.steps(self.state, drive, np.ascontiguousarray(w1, dtype=float), self.omega, out, rubs))

    def _rub_steps_reference(
        self,
        Phi: np.ndarray,
        G0: np.ndarray,
        G1: np.ndarray,
        drive: np.ndarray,
        w1: np.ndarray,
        out: np.ndarray,
        rubs: np.ndarray,
    ) -> np.ndarray:
        """Steps with the disc rubbing on the stator; fills out (n, n_dof) and rubs (n, 3), returns the last state.

        The ring's force r on the disc goes in as a load, linear over the step as
        the FOH takes the loads:

            z1 = Φ z0 + Γ0 (u0 + r0) + Γ1 (u1 + r1),

        r1 found by Newton, since it depends on z1 (the trapezoidal rule on r: the
        friction's slope, μ N / SLIP, would ring an explicit step). The step resolves
        the disc bouncing on the ring's stiffness (step_size).
        """
        system = self.system
        n, stator, radius = system.n_dof, system.stator, system.disc_radius
        assert stator is not None  # only called with a ring
        rows = [XD, YD, n + XD, n + YD]
        R0, R1 = G0[:, [XD, YD]], G1[:, [XD, YD]]
        E = R1[rows]
        tol = RUB_TOL * (1.0 + stator.stiffness * stator.clearance)
        z = self.state
        x, y, vx, vy = z[rows].tolist()
        r = np.array(stator_force(stator, radius, x, y, vx, vy, self.omega)[:2])
        for j in range(w1.size):
            zb = Phi @ z + drive[:, j] + R0 @ r
            r, normal = _rub_solve(stator, radius, zb[rows], E, r, float(w1[j]), tol)
            z = zb + R1 @ r
            out[j] = z[:n]
            heat = 0.0
            if normal:  # the drag's work on the slip, μ N sat(s) s
                x, y, vx, vy = z[rows].tolist()
                cx, cy = stator.centre
                rho = math.hypot(x - cx, y - cy)
                slip = radius * w1[j] + ((x - cx) * vy - (y - cy) * vx) / rho
                heat = stator.friction * normal * slip * slip / math.sqrt(slip * slip + SLIP * SLIP)
            rubs[j] = r[0], r[1], normal, heat
        return z

    def _film_steps(
        self, h: float, held: float, w1: np.ndarray, u0: np.ndarray, u: np.ndarray, out: np.ndarray
    ) -> np.ndarray:
        """`_film_steps_reference`'s steps, by the Rust kernel (`_Films.stepper`) where it's built."""
        if native.module is None:
            return self._film_steps_reference(h, held, w1, u0, u, out)
        films = self._films
        assert films is not None  # only called with whole films
        films.hold(held)
        u0, u = np.ascontiguousarray(u0, dtype=float), np.ascontiguousarray(u, dtype=float)
        return np.asarray(films.stepper(h).steps(self.state, u0, u, w1, self.omega, out))

    def _film_steps_reference(
        self, h: float, held: float, w1: np.ndarray, u0: np.ndarray, u: np.ndarray, out: np.ndarray
    ) -> np.ndarray:
        """Steps with the oil films' whole force (`whole_films`); fills out (n, n_dof), returns the last state.

        The film's force depends on where the journal is and how it moves, far from
        linearly once it moves a fair part of its clearance. Each step is the exact
        update of the films linearized (see _Films), plus their remainder r_f as a
        force on the journals, linear over the step (as the FOH takes the loads):

            z1 = Φ z0 + Γ0 (u0 + r_f0) + Γ1 (u1 + r_f1),

        r_f1 found by Newton, since it depends on z1. The films damp their journals
        far faster than the step, so r1 must be implicit; and their damping, the
        stiff part of the remainder, is no more than the bound C_b in the update
        carries, so the remainder is never stiffer than the exact part (the
        trapezoidal rule would ring if it were). C_b is taken afresh, and the
        update with it, as the journals move round; second order in h.
        """
        films, n_dof = self._films, self.system.n_dof
        assert films is not None  # only called with whole films
        films.hold(held)
        z = self.state
        if films.key != (h, held):
            films.take(z, h, held)
        r = films.remainder(z, self.omega)
        last = older = r
        for j in range(w1.size):
            if films.drifted(z):
                films.take(z, h, held)
                r = last = older = films.remainder(z, w1[j - 1] if j else self.omega)
            zb = films.Phi @ z + films.G0 @ (u[j - 1] if j else u0) + films.G1 @ u[j] + films.R0 @ r
            guess = 3.0 * (r - last) + older  # r on the parabola through the last three steps
            r, last, older = films.solve(zb, guess, w1[j]), r, last
            z = zb + films.R1 @ r
            out[j] = z[:n_dof]
        return z

    def _split_steps(
        self,
        h: float,
        held: float,
        steady: bool,
        w0: np.ndarray,
        w1: np.ndarray,
        alpha: np.ndarray,
        u0: np.ndarray,
        u: np.ndarray,
        out: np.ndarray,
    ) -> np.ndarray:
        """Steps with an asymmetric or cracked shaft (Strang splitting); fills out (n, n_dof), returns the last state.

        Its stiffness turns with it, so no one matrix exponential steps it. Each step
        is half a step of everything else (the exact FOH update, as for a round
        shaft, on the mean stiffness), then the turning part at mid-step as a kick
        to the velocities (exact for it alone: v -= h M^{-1} ΔK(φ) q), then the
        other half step. Second order in h, and stable: ΔK is smaller than the
        shaft's stiffness, whose fastest mode the step already resolves.
        """
        system, n_dof = self.system, self.system.n_dof
        n = w1.size
        Phi, G0, G1 = self._discrete(0.5 * h, held, steady)
        wm = 0.5 * (w0 + w1)
        phi0 = np.concatenate([[self.phase], self.phase + np.cumsum(0.5 * h * (w0 + w1))[:-1]])
        phim = phi0 + 0.25 * h * (w0 + wm)  # exact for Ω linear over the step
        um = self.force(phim, wm, alpha, held)
        before = np.vstack([u0, u[:-1]])
        first = G0 @ before.T + G1 @ um.T
        second = G0 @ um.T + G1 @ u.T
        ks, S = system._kick
        angle = np.outer(phim, ks)
        weights = h * np.hstack([np.cos(angle), np.sin(angle)])  # (n, 2m)
        z = self.state
        for j in range(n):
            z = Phi @ z + first[:, j]
            z[n_dof : 2 * n_dof] -= weights[j] @ (S @ z[:n_dof])
            z = Phi @ z + second[:, j]
            out[j] = z[:n_dof]
        return z
