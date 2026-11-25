"""The Pendulums page's view model (#478 Phase 4): its state, and what each panel shows, without Qt.

The page's state is a `PendulumTrial` (its trials, `practice_state()` and `load_setup` all use one). The page runs
a `live_pendulum.LivePendulum` built from it (`pendulum`), and each panel has a function here:

- `info(p)`: J, the stiffness k<sub>θ</sub>, f<sub>n</sub> and T₀ under the boxes.
- `free_body(p, θ, θ', ...)`: the moments about the pivot, a row each (the weight's, the spring's, the damper's, a
  hand's holding it; the net Jθ''): the single oscillator's diagram turned. Under them, aside, the pull along the
  string, rod or arm (`pull`), which passes through the pivot and has no moment.
- The **Period** tab is built by hand, as the Single oscillator's frequency response is: each whole swing the
  pendulum makes lands a point, its measured period at its amplitude (`cycles`, `PeriodRecorder`), and the
  exact, Duffing and linear curves can be shown over the points once there are some (`period_view`). A damped
  swing from high up lands a point a swing as it dies away, tracing the curve by itself.
- `phase_view(p, θ, θ')`: the Phase plane tab, the live path over the undamped orbits of constant energy, with
  the separatrix where there is one.
- `moment_view(p, θ)`: the restoring moment against the angle beside its tangent at 0, the pendulum on it.
- The Energy tab is `live_pendulum.energy_view`'s.
- `GRIPS` and `pivot_dragged`: how the drawn pendulum's grips set L, m, k<sub>t</sub> and the rod's pivot.

Working on it: Qt-free (the package's docstring). `tests/test_live_pendulum.py` checks the views.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from ..core.pendulum import Kind, Pendulum, duffing_period, orbit, periods, phase_curve
from ..core.practice import Placement, Prediction
from .free_body import Arrow, ArrowScale, FreeBody, arrow
from .grips import Grip
from .plot import Curve
from .text import fmt
from .text import number as sig

TAB_PERIOD, TAB_PHASE, TAB_ENERGY = range(3)
MIN_PIVOT = 0.02  # the rod's pivot at least this much of its length from its centre (at its centre it can't swing)


@dataclass(frozen=True)
class PendulumTrial:
    name: str
    note: str
    kind: Kind = Kind.SIMPLE
    mass: float = 1.0
    length: float = 1.0  # the string, the rod, the disc's radius or the inverted arm (m)
    pivot: float = 0.5  # the rod's pivot from its centre (m)
    spring: float = 15.0  # k_t (N·m/rad): the wire's, or the inverted pendulum's at its foot
    gravity: float = 9.81
    zeta: float = 0.0  # the damper at the pivot, as the linearized oscillator's damping ratio
    angle: float = 10.0  # plucked from here (°) and let go (0: hangs at rest)
    theory: bool = False  # the Period tab's curves shown over its points
    tab: int = TAB_PERIOD
    predict: Prediction | Placement | None = None  # asked once the reader has played (Things to try)


def pendulum(t: PendulumTrial) -> Pendulum:
    return Pendulum(t.kind, t.mass, t.length, max(MIN_PIVOT * t.length, t.pivot), t.spring, t.zeta, t.gravity)


def key(p: Pendulum) -> Hashable:
    """What a measured period belongs to: the pendulum, its damping aside (a swing's period hardly moves with it)."""
    return replace(p, zeta=0.0)


def info(p: Pendulum) -> str:
    j = f"J = {sig(p.inertia, 4)} kg·m²"
    if p.stable:
        extra = f", L<sub>eq</sub> = {fmt(p.equivalent_length, 'm')}" if p.kind in (Kind.SIMPLE, Kind.ROD) else ""
        return (
            f"{j}, k<sub>θ</sub> = {sig(p.stiffness, 4)} N·m/rad: f<sub>n</sub> = {sig(p.natural_hz, 4)} Hz, "
            f"T₀ = {sig(p.period0, 4)} s{extra}."
        )
    if p.stiffness == 0:
        return f"{j}, k<sub>θ</sub> = 0: no restoring moment."
    return f"{j}, k<sub>θ</sub> = {sig(p.stiffness, 4)} N·m/rad: negative, so it can't stay at θ = 0."


# ------------------------------------------------------------------ the grips


GRIPS = {
    "length": Grip("length", 0.05, 20.0, per=0.5),  # the bob's (rod's end's, arm's) drag out from the pivot
    "mass": Grip("mass", 0.01, 1e4, per=0.4),
    "spring": Grip("spring", 0.01, 1e5, per=0.3),
}


def pivot_dragged(length: float, start: float, along: float) -> float:
    """The rod's pivot (m from its centre) after its grip is slid `along` m toward the rod's end from `start`."""
    return min(max(start + along, MIN_PIVOT * length), 0.5 * length)


# ------------------------------------------------------------------ the free-body diagram


def pull(p: Pendulum, theta: float, omega: float) -> tuple[str, str, float]:
    """The pull along the string, rod or arm on the body (N, + toward the pivot): what holds it on its circle. Its
    name, symbol and size: a string's tension m(g cos θ + Lθ'²) (the most at the bottom, where it also turns the
    bob), the rod's pivot's pull, the inverted arm's push (negative) or the wire's, holding up the disc's weight."""
    m, g = p.mass, p.gravity
    if p.kind is Kind.TORSION:
        return "wire", "T", m * g
    if p.kind is Kind.INVERTED:
        return "arm", "arm", -m * (g * math.cos(theta) - p.length * omega * omega)
    name = "string" if p.kind is Kind.SIMPLE else "pivot"
    return name, "T" if p.kind is Kind.SIMPLE else "R", m * (g * math.cos(theta) + p.arm * omega * omega)


def free_body(
    p: Pendulum,
    theta: float,
    omega: float,
    held: bool = False,
    scale: ArrowScale | None = None,
    aside_scale: ArrowScale | None = None,
    t: float = 0.0,
) -> FreeBody:
    """The moments about the pivot at θ, θ' (+ along θ), N·m: the weight's ∓mgd sin θ (none on the disc, whose
    weight hangs on the wire), the spring's −k<sub>t</sub>θ (the disc and the inverted pendulum), the damper's −cθ',
    and while `held` the hand's, which balances them; the net Jθ''. Aside, the pull along the string (`pull`)."""
    rows: list[Arrow] = []
    if p.kind is not Kind.TORSION:
        d = "L" if p.kind is not Kind.ROD else "d"
        rows.append(
            arrow("weight", f"mg{d} sin θ", -p.weight_moment * math.sin(theta), "energy.potential", "mg", "N·m")
        )
    if p.kind in (Kind.TORSION, Kind.INVERTED):
        rows.append(arrow("spring", "kₜθ", -p.torsion * theta, "energy.added", unit="N·m"))
    rows.append(arrow("damper", "cθ̇", -p.damping * omega, "energy.dissipated", unit="N·m"))
    if held:
        rows.append(arrow("hand", "hand", float(p.moment(theta)), "muted", unit="N·m"))
    total = math.fsum(a.value for a in rows)
    net = arrow("net", "Jθ̈", total, "strong", unit="N·m")
    largest = max(abs(a.value) for a in (*rows, net))
    s = scale.update(t, largest) if scale is not None else (1.0 / largest if largest > 0 else 0.0)
    name, symbol, value = pull(p, theta, omega)
    aside = arrow(name, symbol, value, "structure")
    a = aside_scale.update(t, abs(value)) if aside_scale is not None else (1.0 / abs(value) if value else 0.0)
    return FreeBody(tuple(rows), net, s, (aside,), a)


# ------------------------------------------------------------------ the period, built by hand


@dataclass(frozen=True)
class PeriodPoint:
    amplitude: float  # rad: half the swing's peak to peak
    period: float  # s
    key: Hashable  # the pendulum it was measured on
    t: float  # when the swing ended


def _extremes(t: np.ndarray, theta: np.ndarray, sign: float) -> tuple[np.ndarray, np.ndarray]:
    """The times and heights of θ's maxima (`sign` 1) or minima (−1), each by a parabola through three samples."""
    y = sign * theta
    i = np.flatnonzero((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])) + 1
    if i.size == 0:
        return np.empty(0), np.empty(0)
    a, b, c = y[i - 1], y[i], y[i + 1]
    den = a - 2 * b + c
    with np.errstate(divide="ignore", invalid="ignore"):
        shift = np.where(den != 0, 0.5 * (a - c) / den, 0.0)
    h = t[i + 1] - t[i]
    return t[i] + shift * h, sign * (b - 0.25 * (a - c) * shift)


def cycles(t: np.ndarray, theta: np.ndarray) -> list[tuple[float, float, float]]:
    """Each whole swing in the samples, maximum to maximum with a minimum between: (when it ended, its amplitude
    (rad, half its peak to peak), its period (s)). A pendulum going over the top has none."""
    if t.size < 5:
        return []
    tmax, hmax = _extremes(t, theta, 1.0)
    tmin, hmin = _extremes(t, theta, -1.0)
    out = []
    for k in range(tmax.size - 1):
        between = np.flatnonzero((tmin > tmax[k]) & (tmin < tmax[k + 1]))
        if between.size != 1:
            continue
        low = hmin[between[0]]
        amplitude = 0.25 * (hmax[k] + hmax[k + 1]) - 0.5 * low
        out.append((float(tmax[k + 1]), float(amplitude), float(tmax[k + 1] - tmax[k])))
    return out


SAME_AMPLITUDE = math.radians(0.5)  # a swing this near one already measured on the same pendulum replaces it


class PeriodRecorder:
    """The Period tab's points (see the module): each whole swing after the last change lands one, a swing at an
    amplitude already measured on the same pendulum replacing it; those from another pendulum stay, faded."""

    def __init__(self) -> None:
        self.points: list[PeriodPoint] = []
        self._after = -math.inf  # the last landed swing's end: the next starts from its maximum

    def update(self, t: np.ndarray, theta: np.ndarray, since: float, key: Hashable) -> int:
        """Land the swings in (t, θ) after `since` (the last change or let-go) not landed yet: how many."""
        start = max(since, self._after)
        i0 = max(int(np.searchsorted(t, start)) - 2, 0)
        landed = 0
        for end, amplitude, period in cycles(t[i0:], theta[i0:]):
            if end <= self._after or end - period < since:
                continue
            self.points = [
                q for q in self.points if not (q.key == key and abs(q.amplitude - amplitude) < SAME_AMPLITUDE)
            ]
            self.points.append(PeriodPoint(amplitude, period, key, end))
            self._after = end
            landed += 1
        return landed

    def current(self, key: Hashable) -> list[PeriodPoint]:
        return [q for q in self.points if q.key == key]

    def earlier(self, key: Hashable) -> list[PeriodPoint]:
        return [q for q in self.points if q.key != key]

    def restart(self) -> None:
        """A reset: the run's time starts again, the points stay."""
        self._after = -math.inf

    def clear(self) -> None:
        self.points = []
        self._after = -math.inf


@dataclass(frozen=True)
class PeriodView:
    points: Curve  # period (s) against amplitude (°): the swings measured on this pendulum
    earlier: Curve  # those from another pendulum, faded
    exact: Curve  # the theory's curves, with `theory` (empty otherwise)
    duffing: Curve
    linear: Curve
    x_span: tuple[float, float]  # °
    y_span: tuple[float, float]  # s: the theory's over x_span and every point, so it stays put as the theory shows
    note: str


THEORY_DEGREES = np.linspace(0.5, 179.0, 180)


def _theory(p: Pendulum) -> tuple[np.ndarray, np.ndarray]:
    a = np.radians(THEORY_DEGREES)
    return periods(p, a), np.array([duffing_period(p, float(x)) for x in a])


def _as_curve(points: Sequence[PeriodPoint]) -> Curve:
    q = sorted(points, key=lambda q: q.amplitude)
    return Curve(np.degrees([x.amplitude for x in q]), np.array([x.period for x in q]))


def period_view(
    p: Pendulum, current: Sequence[PeriodPoint], earlier: Sequence[PeriodPoint], theory: bool
) -> PeriodView:
    """The Period tab (see the module). The axes span the swings measured and the theory's curve over them, shown
    or not, so showing it moves nothing."""
    points, faded = _as_curve(current), _as_curve(earlier)
    empty = Curve.empty()
    if not p.stable:
        note = "Upright is unstable: there is no swing about it to time." if p.stiffness else "No restoring moment."
        return PeriodView(points, faded, empty, empty, empty, (0.0, 180.0), (0.0, 1.0), note)
    exact, duff = _theory(p)
    widest = max([90.0, *points.x, *faded.x])
    x_span = (0.0, min(180.0, 1.1 * widest))
    seen = np.concatenate([exact[THEORY_DEGREES <= x_span[1]], points.y, faded.y, [p.period0]])
    seen = seen[np.isfinite(seen)]
    lo, hi = float(seen.min()), min(float(seen.max()), 3.0 * p.period0)
    pad = 0.08 * max(hi - lo, 0.05 * p.period0)
    y_span = (max(0.0, lo - pad), hi + pad)
    if theory:
        curves = (
            Curve(THEORY_DEGREES, exact),
            Curve(THEORY_DEGREES, duff),
            Curve(np.array([0.0, 180.0]), np.full(2, p.period0)),
        )
    else:
        curves = (empty, empty, empty)
    if not current:
        note = "Pluck it: each whole swing lands its period here, at its amplitude."
    else:
        last = max(current, key=lambda q: q.t)
        note = (
            f"The last swing, ±{sig(math.degrees(last.amplitude))}°, took {sig(last.period, 4)} s: "
            f"{sig(100 * (last.period / p.period0 - 1), plus=True)} % on the small swing's T₀ = {sig(p.period0, 4)} s."
        )
        if p.cubic == 0:
            note += " Its moment is linear (no gravity in it): the same period at any swing."
    return PeriodView(points, faded, *curves, x_span, y_span, note)


# ------------------------------------------------------------------ the phase plane


@dataclass(frozen=True)
class PhaseView:
    orbits: Curve  # undamped orbits of constant energy (°, °/s), NaN between them
    separatrix: Curve  # the orbit through the upside-down balance, where there is one
    path: Curve  # the live swing's
    x_span: tuple[float, float]  # °
    note: str


ORBITS = 6


def phase_reach(p: Pendulum, amplitude: float) -> float:
    """The phase plane's half-width (rad): past the upside-down balance for a hanging pendulum, else the swing's."""
    if p.weight_moment > 0 and p.torsion == 0:
        return 1.05 * math.pi
    return max(1.3 * amplitude, 1.3 * abs(p.leaning()), 0.3)


def phase_view(p: Pendulum, theta: np.ndarray, omega: np.ndarray, reach: float) -> PhaseView:
    """The Phase plane tab over ±`reach` rad: the path (θ, θ') of the samples given over the orbits."""
    x = np.linspace(-reach, reach, 801)
    V = p.potential(x)
    hanging = p.weight_moment > 0 and p.torsion == 0
    top = float(p.potential(math.pi)) if hanging else float(np.max(V))
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    gap = np.array([np.nan])
    for e in np.linspace(float(np.min(V)), top, ORBITS + 2)[1:-1]:
        ox, oy = orbit(p, float(e), x)
        xs += [ox, gap]
        ys += [oy, gap]
    orbits = Curve(np.degrees(np.concatenate(xs)), np.degrees(np.concatenate(ys))) if xs else Curve.empty()
    if hanging:
        inside = np.linspace(-math.pi, math.pi, 801)
        y = phase_curve(p, top, inside)
        y[[0, -1]] = 0.0
        sep = Curve(np.degrees(np.concatenate([inside, inside[::-1]])), np.degrees(np.concatenate([y, -y[::-1]])))
        note = (
            "Grey: the undamped orbits of constant energy; bold: the separatrix through the upside-down balance "
            "(±180°). Inside it the pendulum swings; outside it, it goes round."
        )
    else:
        sep = Curve.empty()
        note = "Grey: the undamped orbits of constant energy. Colour: the swing as it runs."
    path = Curve(np.degrees(theta), np.degrees(omega))
    return PhaseView(orbits, sep, path, (-math.degrees(reach), math.degrees(reach)), note)


# ------------------------------------------------------------------ the restoring moment


@dataclass(frozen=True)
class MomentView:
    moment: Curve  # the whole restoring moment against θ (°)
    tangent: Curve  # its tangent at 0: the linearized stiffness
    now: Curve  # where the pendulum is on it
    x_span: tuple[float, float]


def moment_view(p: Pendulum, theta: float, reach: float) -> MomentView:
    x = np.linspace(-reach, reach, 400)
    return MomentView(
        Curve(np.degrees(x), np.asarray(p.moment(x))),
        Curve(np.degrees(x), p.stiffness * x),
        Curve.point(math.degrees(theta), float(p.moment(theta))),
        (-math.degrees(reach), math.degrees(reach)),
    )
