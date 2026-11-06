"""Detecting nonlinearity in a test of the chain: stepped sine up and down at several force levels.

The chain (its masses, springs, dampers and its own Coulomb friction) gets one more element of the page's: a
cubic spring k₃δ³ alongside one of its springs, or a gap, a spring k_g that only engages once the stretch passes
a clearance g either way (`Element`). A linear structure is *homogeneous*: double the force and every response
doubles, so the FRF X/F measured at any level is the same. A nonlinear one isn't: a hardening spring's peak
leans to higher frequencies as the level rises, friction's damping falls (it is a fixed force, a smaller share
of a larger motion) and its stiffness rises as it slips less. Swept slowly up and then down, a leaning peak
jumps, at different frequencies each way: the two sweeps disagree, which a linear structure's can't.

Stepped sine by harmonic balance
--------------------------------
Each step's steady state is found by harmonic balance with one harmonic, x(t) = Re(X e^{iωt}): every element's
force replaced by its first harmonic (its describing function), (K − ω²M + iωC)X + N(X) = F. A cubic spring on
a stretch D gives ¾k₃|D|²D; a gap, k_g[1 − (2/π)(asin r + r√(1 − r²))]D with r = g/|D| (nothing while
|D| ≤ g); friction F_c on a mass moving at V = iωX gives (4F_c/π)V/|V|. `stepped_sine` steps the frequency and
solves by Newton from the last step's X, as a real stepped sine starts each step from where the structure is.
Past a fold that branch ends; the solution found from other starts that is stable (the real Jacobian's
determinant positive, as on the linear FRF; it changes sign at each fold, so the middle branch has it negative)
and nearest the last one is where the structure jumps to.

Distortion
----------
One harmonic leaves out the others a nonlinear element makes: a symmetric one (cubic, gap, friction) adds odd
harmonics, 3X, 5X. `steady_response` steps the chain in time with the exact FOH update, the nonlinear forces
as an input linear over the step whose end value is found by Newton (friction smoothed as F_c tanh(v/v_s)), from
harmonic balance's state until a cycle repeats itself, and gives the response's harmonics.

Working on it
-------------
`core/nonlinear_test.py` and `core/force_id.py` (the *Nonlinearity & forces* page, `gui/nonlinear_forces.py`,
`nonlinear_page`, after Test planning; follows the chain, its friction included). The page adds one `Element`
(a cubic spring or a gap alongside a spring). `stepped_sine` is one-harmonic harmonic balance (`describing`:
¾k₃|D|²D, the gap's `gap_stiffness`, friction (4F/π)V/|V| smoothed below `STUCK`), Newton on Re/Im with a
central-difference Jacobian and a line search (friction's near-step makes full steps oscillate; the whole step
is taken when no halving helps, or Newton sits in the dip a vanished fold leaves), each step from the last;
where it fails or the Jacobian's determinant is negative (the middle branch), the stable solution nearest the
last from other starts (`SEEDS`, `RESCUE` iterations) is the jump. `steady_response` is the exact FOH update
with the nonlinear forces as an input linear over the step, end value by Newton on the whole state, from
harmonic balance's state until a cycle repeats. Force identification: `markov` builds y = Gf from the FOH
update (a displacement, or an acceleration with its direct term), `set_up` adds seeded noise and takes G's SVD
once; `tikhonov`, `tsvd`, `inverse`, `l_curve` (its corner searched only where the curve moves: below the
singular values that matter it stands still and round-off makes false corners), `discrepancy`. The trials set
the controls before sending their chain, so the page works out once. `tests/test_nonlinear_forces.py` checks
the describing functions by quadrature, every sweep step's residual, the time stepper against `solve_ivp`, G
against the FOH simulation, Tikhonov against the normal equations, and the trials' notes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .model import ChainSystem
from .simulator import foh_discretize

TWO_PI = 2.0 * math.pi
TOL = 1e-10  # Newton: the residual's norm below this × the force
ITERATIONS = 40
RESCUE = 200  # Newton's iterations from the other starts, where a branch has ended: it wanders before it lands
HALVINGS = 30
LEAP = 0.2  # a step that moves the response this share of itself, and
STEEPER = 5.0  # this many times further than the step before, is a jump
SEEDS = (0.1, 0.3, 2.0, 3.0, 10.0)  # multiples of the linear response, and the last, tried when a branch ends
STEPS_PER_CYCLE = 64
MAX_CYCLES = 300
SETTLED = 1e-5  # one cycle's largest change below this × its largest response: steady
SMOOTH = 1e-3  # friction's v_s, as a fraction of the drive's velocity scale
STUCK = 1e-4  # friction's describing function is smoothed below this × the linear response's velocity
HARMONICS = 9


class Kind(Enum):
    NONE = "None (linear)"
    CUBIC = "Cubic spring"
    GAP = "Gap (clearance)"


KINDS = tuple(Kind)


@dataclass(frozen=True)
class Element:
    """The page's nonlinear element, alongside spring `spring` (from 0: between mass spring − 1, or the ground,
    and mass spring)."""

    kind: Kind = Kind.NONE
    spring: int = 0
    k3: float = 0.0  # N/m³
    gap: float = 0.0  # m
    k_gap: float = 0.0  # N/m

    @property
    def active(self) -> bool:
        return (self.kind is Kind.CUBIC and self.k3 != 0) or (self.kind is Kind.GAP and self.k_gap > 0)


def _stretch(X: np.ndarray, spring: int) -> np.ndarray:
    return X[..., spring] - X[..., spring - 1] if spring > 0 else X[..., 0]


def gap_stiffness(k: float, gap: float, amplitude: np.ndarray) -> np.ndarray:
    """A gap's describing function: the equivalent stiffness at stretch amplitude A (0 while A ≤ g)."""
    A = np.asarray(amplitude, dtype=float)
    r = np.divide(gap, A, out=np.ones_like(A), where=A > gap)
    return np.where(A > gap, k * (1 - (2 / math.pi) * (np.arcsin(r) + r * np.sqrt(1 - r * r))), 0.0)


def describing(system: ChainSystem, element: Element, X: np.ndarray, omega: float, scale: float) -> np.ndarray:
    """N(X): the nonlinear forces' first harmonics on each mass (X may be a stack of shapes, (..., n))."""
    N = np.zeros_like(X)
    if element.active:
        D = _stretch(X, element.spring)
        A = np.abs(D)
        if element.kind is Kind.CUBIC:
            force = 0.75 * element.k3 * A * A * D
        else:
            force = gap_stiffness(element.k_gap, element.gap, A) * D
        N[..., element.spring] += force
        if element.spring > 0:
            N[..., element.spring - 1] -= force
    if np.any(system.friction > 0):
        V = 1j * omega * X
        eps = STUCK * omega * scale
        N += (4 / math.pi) * system.friction * V / np.sqrt(np.abs(V) ** 2 + eps * eps)
    return N


def nonlinear(system: ChainSystem, element: Element) -> bool:
    return element.active or bool(np.any(system.friction > 0))


@dataclass(frozen=True)
class Solution:
    X: np.ndarray
    converged: bool
    stable: bool


def _solve(
    system: ChainSystem,
    element: Element,
    Z: np.ndarray,
    f: np.ndarray,
    omega: float,
    X0: np.ndarray,
    scale: float,
    iterations: int = ITERATIONS,
) -> Solution:
    """Newton on the real and imaginary parts, the Jacobian by central differences (every column at once)."""
    n = f.size
    size = float(np.abs(f).max())

    def residual(Xs: np.ndarray) -> np.ndarray:
        r = Xs @ Z.T + describing(system, element, Xs, omega, scale) - f
        return np.concatenate([r.real, r.imag], axis=-1)

    X = X0.astype(complex)
    for _ in range(iterations):
        step = 1e-6 * max(float(np.abs(X).max()), scale)
        shift = np.concatenate([np.eye(n), 1j * np.eye(n)]) * step
        R = residual(np.concatenate([X[None], X + shift, X - shift]))
        r0 = R[0]
        J = (R[1 : 2 * n + 1] - R[2 * n + 1 :]).T / (2 * step)  # central differences
        if np.linalg.norm(r0) <= TOL * size:
            return Solution(X, True, bool(np.linalg.det(J) > 0))
        try:
            d = np.linalg.solve(J, -r0)
        except np.linalg.LinAlgError:
            break
        d = d[:n] + 1j * d[n:]
        cap = 0.5 * max(float(np.abs(X).max()), scale)  # no step longer than half the motion so far
        big = float(np.abs(d).max())
        if big > cap:
            d *= cap / big
        # a line search: halve the step until the residual falls (friction's near-step makes full steps
        # overshoot back and forth)
        now = float(np.linalg.norm(r0))
        trial = d
        for _ in range(HALVINGS):
            if float(np.linalg.norm(residual(X + trial))) < now:
                d = trial
                break
            trial = 0.5 * trial
        X = X + d  # (the whole step if no part of it helps: near a fold that has gone, the residual has a
        # shallow dip a line search would stay in)
        if big <= 1e-12 * scale:  # standing still: as converged as round-off allows
            return Solution(X, True, bool(np.linalg.det(J) > 0))
    return Solution(X, False, False)


def harmonic_balance(
    system: ChainSystem,
    element: Element,
    force: int,
    F: float,
    hz: float,
    start: np.ndarray | None = None,
    iterations: int = ITERATIONS,
) -> Solution:
    """The one-harmonic steady state at hz under F on mass `force`, from `start` (the linear response if None)."""
    M, C, K = system.matrices()
    omega = TWO_PI * hz
    Z = K - omega * omega * M + 1j * omega * C
    f = np.zeros(system.n, dtype=complex)
    f[force] = F
    linear = np.linalg.solve(Z, f)
    scale = float(np.abs(linear).max())
    if not nonlinear(system, element):
        return Solution(linear, True, True)
    return _solve(system, element, Z, f, omega, linear if start is None else start, scale, iterations)


@dataclass(frozen=True)
class Sweep:
    hz: np.ndarray  # in the order stepped
    X: np.ndarray  # (steps, n): each step's steady state
    jumps: tuple[int, ...]  # steps where the last step's branch had ended, and the response jumped to another


def stepped_sine(
    system: ChainSystem, element: Element, force: int, F: float, freqs: np.ndarray, up: bool = True
) -> Sweep:
    """Step through freqs (ascending; down if not `up`), each steady state from the last one's."""
    hz = np.asarray(freqs, dtype=float)
    if not up:
        hz = hz[::-1]
    out = np.zeros((hz.size, system.n), dtype=complex)
    jumps: list[int] = []
    last: np.ndarray | None = None
    moved = 0.0  # how far the last step moved the response
    for k, f in enumerate(hz):
        s = harmonic_balance(system, element, force, F, float(f), last)
        lost = last is not None and not (s.converged and s.stable)
        if last is not None and lost:
            linear = harmonic_balance(system, Element(), force, F, float(f)).X
            here = last
            starts = [linear * m for m in (1.0, *SEEDS)] + [here * m for m in SEEDS]
            options = [harmonic_balance(system, element, force, F, float(f), x, RESCUE) for x in starts]
            good = [o for o in options if o.converged and o.stable]
            if good:
                s = min(good, key=lambda o: float(np.linalg.norm(o.X - here)))
        if last is not None:
            step = float(np.linalg.norm(s.X - last))
            # a jump: the branch ended, or Newton landed on another one (a leap far longer than the last step)
            if (lost or (step > LEAP * float(np.linalg.norm(last)) and step > STEEPER * moved)) and (
                not jumps or jumps[-1] != k - 1  # a run of them is one jump, settling over a few steps
            ):
                jumps.append(k)
            moved = step
        out[k] = s.X
        last = s.X
    return Sweep(hz, out, tuple(jumps))


# -------------------------------------------------------------------- distortion, in time
def _forces(system: ChainSystem, element: Element, x: np.ndarray, v: np.ndarray, vs: float):
    """The nonlinear forces g(x, v) on the masses (as restoring forces) and their derivatives dg/dx, dg/dv."""
    n = system.n
    g = np.zeros(n)
    gx = np.zeros((n, n))
    gv = np.zeros((n, n))
    if element.active:
        s = element.spring
        d = x[s] - (x[s - 1] if s > 0 else 0.0)
        if element.kind is Kind.CUBIC:
            force, slope = element.k3 * d**3, 3 * element.k3 * d * d
        else:
            over = abs(d) - element.gap
            force = element.k_gap * over * math.copysign(1.0, d) if over > 0 else 0.0
            slope = element.k_gap if over > 0 else 0.0
        g[s] += force
        gx[s, s] += slope
        if s > 0:
            g[s - 1] -= force
            gx[s, s - 1] -= slope
            gx[s - 1, s] -= slope
            gx[s - 1, s - 1] += slope
    F = system.friction
    if np.any(F > 0):
        t = np.tanh(v / vs)
        g += F * t
        gv += np.diag(F * (1 - t * t) / vs)
    return g, gx, gv


@dataclass(frozen=True)
class Waveform:
    t: np.ndarray  # one drive cycle, s
    x: np.ndarray  # (samples, n): the steady response over it
    harmonics: np.ndarray  # (HARMONICS, n): each mass's amplitude at 1X, 2X, …
    cycles: int  # how many it took to settle


def steady_response(
    system: ChainSystem, element: Element, force: int, F: float, hz: float, start: np.ndarray | None = None
) -> Waveform:
    """The chain's steady response to F cos ωt on mass `force`, stepped in time from harmonic balance's state
    (or `start`, its complex amplitudes) until one cycle repeats the last."""
    n = system.n
    M, C, K = system.matrices()
    Minv = np.linalg.inv(M)
    A = np.zeros((2 * n, 2 * n))
    A[:n, n:] = np.eye(n)
    A[n:, :n] = -Minv @ K
    A[n:, n:] = -Minv @ C
    B = np.zeros((2 * n, n))
    B[n:] = Minv
    omega = TWO_PI * hz
    h = 1.0 / (hz * STEPS_PER_CYCLE)
    Phi, G0, G1 = foh_discretize(A, B, h)
    X = start if start is not None else harmonic_balance(system, element, force, F, hz).X
    z = np.concatenate([X.real, (1j * omega * X).real])
    vs = SMOOTH * omega * max(float(np.abs(X).max()), 1e-12)
    drive = np.zeros(n)
    drive[force] = F
    eye = np.eye(2 * n)

    def load(z_: np.ndarray, s: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        g, gx, gv = _forces(system, element, z_[:n], z_[n:], vs)
        return drive * math.cos(TWO_PI * s / STEPS_PER_CYCLE) - g, gx, gv

    u0, _, _ = load(z, 0)
    previous: np.ndarray | None = None
    cycle = np.zeros((STEPS_PER_CYCLE, n))
    cycles = 0
    while cycles < MAX_CYCLES:
        cycles += 1
        for s in range(STEPS_PER_CYCLE):
            base = Phi @ z + G0 @ u0
            z1 = base + G1 @ u0  # a guess: the load held
            for _ in range(20):
                u1, gx, gv = load(z1, s + 1)
                r = z1 - base - G1 @ u1
                if np.linalg.norm(r) <= 1e-13 * (np.linalg.norm(z1) + 1e-300):
                    break
                J = eye + G1 @ np.hstack([gx, gv])
                z1 = z1 - np.linalg.solve(J, r)
            z, u0 = z1, load(z1, s + 1)[0]
            cycle[s] = z[:n]
        top = float(np.abs(cycle).max())
        if previous is not None and float(np.abs(cycle - previous).max()) <= SETTLED * top:
            break
        previous = cycle.copy()
    x = np.roll(cycle, 1, axis=0)  # sample s is the state at step s (the last one ends the cycle where it began)
    spectrum = np.fft.rfft(x, axis=0) * 2 / STEPS_PER_CYCLE
    t = np.arange(STEPS_PER_CYCLE) * h
    return Waveform(t, x, np.abs(spectrum[1 : HARMONICS + 1]), cycles)


def homogeneity(sweeps: list[Sweep], out: int, levels: list[float]) -> float:
    """The largest difference between the levels' FRFs |X/F| at `out`, as a share of the peak."""
    curves = [np.abs(s.X[:, out]) / F for s, F in zip(sweeps, levels, strict=True)]
    top = max(float(c.max()) for c in curves)
    lo, hi = np.min(curves, axis=0), np.max(curves, axis=0)
    return float((hi - lo).max() / top) if top > 0 else 0.0
