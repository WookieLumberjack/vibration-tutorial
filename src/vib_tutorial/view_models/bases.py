"""Coordinates & bases' view model (#478 Phase 4): its state, and what each panel shows, without Qt.

The page's state is a `BasesTrial` (its trials, `practice_state()` and `load_setup` all use one): the two masses
(`system`), the basis's two arrows v₁, v₂ (`basis`: x = Vq), whether the masses run on a rail along v₁, and what's
shown. The page runs a `live_bases.LiveBases` built from it, and each panel has a function here:

- `plane_view(V, rail, modes, shown)`: the (x₁, x₂) plane's still parts (the arrows, the grid of the coordinates
  they make, the mode lines, the rail), and `projection(V, x)` the point's parallelogram, x = q₁v₁ + q₂v₂, as it
  moves. All in mm.
- `matrices_view(system, V, rail)`: VᵀMV, VᵀKV (and VᵀCV with dampers) with their off-diagonal coupling marked,
  and what they say: coupled statically, dynamically, both, or not at all (the modes), and the frequencies, the
  same in every basis.
- `coordinates(V, rail, x)`: the coordinates q against time, and `shares(...)`: each coordinate's own energy and
  the coupling's, the **Coupling** tab.
- The **Rail** tab is built by hand: each rail the reader tries lands its measured frequency at its angle
  (`RailRecorder`), with Rayleigh's quotient over them on request (`rail_view`).
- The **Energy** tab is `live_bases.energy_view`'s.
- `arrow_dragged(tip, targets)`: an arrow's drag, snapping onto a mode or an axis within `SNAP`°; `GRIPS`: how the
  drawn chain's grips set m and k.

Working on it: Qt-free (the package's docstring). `tests/test_live_bases.py` checks the views.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np

from ..core.bases import TwoMass, angle, direction, frequencies, rayleigh_hz, transformed
from ..core.practice import Placement, Prediction
from .grips import Grip
from .text import fmt
from .text import number as sig

TAB_COUPLING, TAB_RAIL, TAB_ENERGY = range(3)
MM = 1e3  # the plane is drawn in mm
ARROW = 0.04  # m: an arrow of length 1 is drawn this long
LENGTHS = (0.25, 2.0)  # an arrow's shortest and longest, dragged
SNAP = 3.0  # ° from a mode or an axis, an arrow dragged snaps onto it
REACH = 0.06  # m: the plane's half-width
PLUCK = 0.05  # m: the farthest a pluck pulls either mass
GRID = 0.02  # m: the coordinates' grid lines are this far apart in q
COUPLED = 1e-3  # an off-diagonal under this of its diagonal's geometric mean is nought


@dataclass(frozen=True)
class BasesTrial:
    name: str
    note: str
    m1: float = 1.0
    m2: float = 1.0
    k1: float = 400.0
    k2: float = 100.0
    k3: float = 400.0
    c: float = 0.0
    v1: tuple[float, float] = (1.0, 0.0)  # the basis's arrows, dimensionless: x = q₁v₁ + q₂v₂
    v2: tuple[float, float] = (0.0, 1.0)
    rail: bool = False  # on a rail along v₁ (v₂ taken away)
    modes: bool = False  # the mode lines drawn on the plane
    theory: bool = False  # Rayleigh's quotient drawn on the Rail tab
    pluck: tuple[float, float] = (0.0, 0.0)  # m: plucked from here and let go ((0, 0): at rest, as it opens)
    tab: int = TAB_COUPLING
    predict: Prediction | Placement | None = None  # asked once the reader has played (Things to try)


def system(t: BasesTrial) -> TwoMass:
    return TwoMass(t.m1, t.m2, t.k1, t.k2, t.k3, t.c)


def basis(t: BasesTrial) -> np.ndarray:
    """V, the arrows as its columns."""
    return np.column_stack([np.array(t.v1, dtype=float), np.array(t.v2, dtype=float)])


def key(s: TwoMass) -> Hashable:
    """What a rail's measured frequency belongs to: the masses and springs (the damping hardly moves it)."""
    return (s.m1, s.m2, s.k1, s.k2, s.k3)


def singular(V: np.ndarray) -> bool:
    """The two arrows (nearly) along one line: no basis."""
    a, b = V[:, 0], V[:, 1]
    return abs(float(np.linalg.det(V))) < 1e-3 * float(np.hypot(*a) * np.hypot(*b))


# ------------------------------------------------------------------ the grips and arrows

GRIPS = {
    "m1": Grip("m1", 0.05, 100.0, per=0.4),
    "m2": Grip("m2", 0.05, 100.0, per=0.4),
    "k1": Grip("k1", 1.0, 1e5, per=0.3),
    "k2": Grip("k2", 1.0, 1e5, per=0.3),
    "k3": Grip("k3", 1.0, 1e5, per=0.3),
}


def snap_targets(s: TwoMass) -> tuple[np.ndarray, ...]:
    """The directions a dragged arrow snaps onto: the modes, then the axes."""
    _, shapes = s.modes()
    return shapes[:, 0], shapes[:, 1], np.array([1.0, 0.0]), np.array([0.0, 1.0])


def arrow_dragged(tip: Sequence[float], targets: Sequence[np.ndarray] = ()) -> tuple[tuple[float, float], int | None]:
    """The arrow a drag to `tip` (m, in the plane) makes, its length within `LENGTHS`, and onto which of `targets`
    (directions) it snapped, within `SNAP`° (keeping the side the hand is on), or None."""
    v = np.asarray(tip, dtype=float) / ARROW
    n = float(np.hypot(*v))
    if n < 1e-9:
        v, n = np.array([1.0, 0.0]), 1.0
    length = min(max(n, LENGTHS[0]), LENGTHS[1])
    u = v / n
    snapped = None
    for i, d in enumerate(targets):
        off = abs((angle(u) - angle(d) + 90.0) % 180.0 - 90.0)
        if off < SNAP:
            d = np.asarray(d, dtype=float) / float(np.hypot(*d))
            u = d if float(d @ u) >= 0 else -d
            snapped = i
            break
    w = u * length
    return (float(w[0]), float(w[1])), snapped


# ------------------------------------------------------------------ the plane


@dataclass(frozen=True)
class PlaneView:
    arrows: tuple[tuple[float, float], ...]  # the arrows' tips (mm), one on a rail
    labels: tuple[str, ...]
    grid: tuple[np.ndarray, np.ndarray]  # the coordinates' lines (mm, NaN-broken)
    modes: tuple[tuple[np.ndarray, np.ndarray], ...]  # each mode's line through the origin (mm), if shown
    mode_tips: tuple[
        tuple[float, float], ...
    ]  # where each mode's label goes (mm): its line's far end, from the arrows'

    rail: tuple[np.ndarray, np.ndarray]  # the rail (mm), or empty
    span: float  # the plane's half-width (mm)
    note: str


def _line(d: np.ndarray, reach: float) -> tuple[np.ndarray, np.ndarray]:
    u = d / float(np.hypot(*d))
    return np.array([-u[0], u[0]]) * reach * MM, np.array([-u[1], u[1]]) * reach * MM


def _grid(V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lines of constant q₁ and q₂, GRID apart, across the plane."""
    xs: list[float] = []
    ys: list[float] = []
    far = 4 * REACH
    for i in range(2):
        a, b = V[:, i], V[:, 1 - i]
        step = GRID
        span = int(far / (step * max(float(np.hypot(*a)), 1e-6))) + 1
        bn = b / float(np.hypot(*b))
        for k in range(-min(span, 12), min(span, 12) + 1):
            if k == 0:
                continue
            p = k * step * a
            for end in (p - far * bn, p + far * bn):
                xs.append(float(end[0]) * MM)
                ys.append(float(end[1]) * MM)
            xs.append(math.nan)
            ys.append(math.nan)
    return np.array(xs), np.array(ys)


def plane_view(s: TwoMass, V: np.ndarray, rail: bool, show_modes: bool) -> PlaneView:
    """The plane's still parts (see the module): redrawn when the arrows, the system or what's shown change."""
    tips = tuple((float(V[0, i] * ARROW * MM), float(V[1, i] * ARROW * MM)) for i in range(1 if rail else 2))
    labels = ("v₁",) if rail else ("v₁", "v₂")
    flat = rail or singular(V)
    grid = (np.empty(0), np.empty(0)) if flat else _grid(V)
    _, shapes = s.modes()
    modes = tuple(_line(shapes[:, i], REACH * 1.2) for i in range(2)) if show_modes else ()
    mode_tips = (
        tuple((float(-shapes[0, i] * 0.85 * REACH * MM), float(-shapes[1, i] * 0.85 * REACH * MM)) for i in range(2))
        if show_modes
        else ()
    )
    rail_line = _line(V[:, 0], REACH * 1.2) if rail else (np.empty(0), np.empty(0))
    if rail:
        note = "On a rail along v₁: drag the arrow to turn it as it runs."
    elif singular(V):
        note = "The arrows lie along one line: they don't make a basis, and most motions can't be written in them."
    else:
        note = "Drag the arrows' tips to change the basis as it runs; drag anywhere else to pluck the masses."
    return PlaneView(tips, labels, grid, modes, mode_tips, rail_line, REACH * MM, note)


def projection(V: np.ndarray, rail: bool, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The point's parallelogram in the plane (mm, NaN-broken): from x along v₂ to q₁v₁ and along v₁ to q₂v₂; on a
    rail, nothing (the point is on it)."""
    if rail or singular(V):
        return np.empty(0), np.empty(0)
    q = np.linalg.solve(V, x)
    a, b = q[0] * V[:, 0], q[1] * V[:, 1]
    xs = np.array([a[0], x[0], b[0], math.nan, 0.0, a[0], math.nan, 0.0, b[0]]) * MM
    ys = np.array([a[1], x[1], b[1], math.nan, 0.0, a[1], math.nan, 0.0, b[1]]) * MM
    return xs, ys


# ------------------------------------------------------------------ the matrices


@dataclass(frozen=True)
class Matrix:
    symbol: str  # "VᵀMV"
    unit: str
    entries: tuple[tuple[str, ...], ...]
    coupled: bool  # its off-diagonal isn't nought


@dataclass(frozen=True)
class MatricesView:
    matrices: tuple[Matrix, ...]
    hz: tuple[float, ...]  # the frequencies the matrices give (Hz): one on a rail
    note: str


def _coupled(A: np.ndarray) -> bool:
    if A.shape[0] < 2:
        return False
    scale = math.sqrt(abs(float(A[0, 0] * A[1, 1]))) or 1.0
    return abs(float(A[0, 1])) > COUPLED * scale


def _matrix(symbol: str, unit: str, A: np.ndarray) -> Matrix:
    scale = float(np.abs(A).max()) or 1.0
    entries = tuple(tuple("0" if abs(float(a)) < COUPLED * scale else sig(float(a), 4) for a in row) for row in A)
    return Matrix(symbol, unit, entries, _coupled(A))


def matrices_view(s: TwoMass, V: np.ndarray, rail: bool) -> MatricesView:
    """The matrices in the basis (see the module), and what they say."""
    M, C, K = s.matrices()
    B = V[:, :1] if rail else V
    if not rail and singular(V):
        return MatricesView((), (), "The arrows lie along one line: they don't make a basis.")
    Mt, Ct, Kt = (transformed(B, A) for A in (M, C, K))
    names = ("v₁ᵀMv₁", "v₁ᵀKv₁", "v₁ᵀCv₁") if rail else ("VᵀMV", "VᵀKV", "VᵀCV")
    shown = [_matrix(names[0], "kg", Mt), _matrix(names[1], "N/m", Kt)]
    if s.c > 0:
        shown.append(_matrix(names[2], "N·s/m", Ct))
    if rail:
        f = rayleigh_hz(V[:, 0], M, K)
        note = (
            f"On the rail, one coordinate: m̃ = v₁ᵀMv₁ = {sig(float(Mt[0, 0]), 4)} kg and k̃ = v₁ᵀKv₁ = "
            f"{sig(float(Kt[0, 0]), 4)} N/m, so it swings at √(k̃/m̃)/2π = {sig(f, 4)} Hz: Rayleigh's quotient."
        )
        return MatricesView(tuple(shown), (f,), note)
    hz = frequencies(Mt, Kt)
    both = " and ".join(f"{sig(float(f), 4)}" for f in hz)
    m_c, k_c = _coupled(Mt), _coupled(Kt)
    if not (m_c or k_c):
        what = (
            "Both diagonal: the coordinates are uncoupled, each moving on its own at one frequency. "
            "These arrows lie along the modes."
        )
    elif k_c and not m_c:
        what = "VᵀMV is diagonal and VᵀKV isn't: the coordinates are coupled through the springs (static coupling)."
    elif m_c and not k_c:
        what = "VᵀKV is diagonal and VᵀMV isn't: the coordinates are coupled through the masses (dynamic coupling)."
    else:
        what = "Neither is diagonal: the coordinates are coupled through both the springs and the masses."
    note = f"{what} Their frequencies: {both} Hz, the same in every basis."
    return MatricesView(tuple(shown), tuple(float(f) for f in hz), note)


# ------------------------------------------------------------------ the coordinates and their energies


def coordinates(V: np.ndarray, rail: bool, x: np.ndarray) -> np.ndarray:
    """q (a row a sample) for the displacements `x` (a row a sample): x = Vq; on a rail, q₁ along v₁ alone."""
    if rail:
        v = V[:, 0]
        return (x @ v / float(v @ v))[:, None]
    if singular(V):
        return np.full((x.shape[0], 2), np.nan)
    return np.linalg.solve(V, x.T).T


@dataclass(frozen=True)
class Shares:
    own: tuple[np.ndarray, ...]  # J: each coordinate's ½m̃ᵢᵢq̇ᵢ² + ½k̃ᵢᵢqᵢ²
    coupling: np.ndarray  # J: m̃₁₂q̇₁q̇₂ + k̃₁₂q₁q₂, the rest of the stored energy
    note: str


def shares(s: TwoMass, V: np.ndarray, rail: bool, x: np.ndarray, v: np.ndarray) -> Shares:
    """Each coordinate's own energy and the coupling's (see the module), for the samples' x and v."""
    M, _, K = s.matrices()
    B = V[:, :1] if rail else V
    q, qd = coordinates(V, rail, x), coordinates(V, rail, v)
    Mt, Kt = transformed(B, M), transformed(B, K)
    own = tuple(0.5 * Mt[i, i] * qd[:, i] ** 2 + 0.5 * Kt[i, i] * q[:, i] ** 2 for i in range(B.shape[1]))
    if B.shape[1] == 1:
        return Shares(own, np.zeros(x.shape[0]), "On a rail there is one coordinate, and all the energy is its own.")
    coupling = Mt[0, 1] * qd[:, 0] * qd[:, 1] + Kt[0, 1] * q[:, 0] * q[:, 1]
    if not (_coupled(Mt) or _coupled(Kt)):
        note = "Uncoupled: each coordinate keeps its own energy, and nothing passes between them."
    else:
        note = "Coupled: the energy passes back and forth between the coordinates, through the coupling's term."
    return Shares(own, coupling, note)


# ------------------------------------------------------------------ the rail, built by hand


@dataclass(frozen=True)
class RailPoint:
    angle: float  # ° of the rail
    hz: float  # its frequency, measured
    key: Hashable
    t: float


class RailRecorder:
    """Each rail tried, its measured frequency at its angle: the Rail tab's points (see the module). A rail within
    half a degree of one already landed replaces it."""

    def __init__(self) -> None:
        self.points: list[RailPoint] = []

    def clear(self) -> None:
        self.points = []

    def update(self, rail_angle: float, hz: float, k: Hashable, t: float) -> bool:
        """Land `hz` at `rail_angle`; True if the points changed."""
        for i, p in enumerate(self.points):
            if p.key == k and abs((p.angle - rail_angle + 90.0) % 180.0 - 90.0) < 0.5:
                if abs(p.hz - hz) <= 1e-6 * hz:
                    return False
                self.points[i] = RailPoint(rail_angle, hz, k, t)
                return True
        self.points.append(RailPoint(rail_angle, hz, k, t))
        return True

    def current(self, k: Hashable) -> list[RailPoint]:
        return [p for p in self.points if p.key == k]

    def earlier(self, k: Hashable) -> list[RailPoint]:
        return [p for p in self.points if p.key != k]


@dataclass(frozen=True)
class RailView:
    points: tuple[np.ndarray, np.ndarray]  # (°, Hz)
    earlier: tuple[np.ndarray, np.ndarray]
    quotient: tuple[np.ndarray, np.ndarray]  # Rayleigh's, if shown
    levels: tuple[float, ...]  # f₁ and f₂ (Hz), if shown
    mode_angles: tuple[float, ...]  # the modes' (°), if shown
    now: float | None  # the rail's angle now (°), or None off a rail
    x_span: tuple[float, float]
    y_span: tuple[float, float]
    note: str


def rail_view(
    s: TwoMass, current: Sequence[RailPoint], earlier: Sequence[RailPoint], theory: bool, now: float | None
) -> RailView:
    """The Rail tab (see the module): its axes the same with the theory shown or not."""
    M, _, K = s.matrices()
    hz, shapes = s.modes()
    pts = (np.array([p.angle for p in current]), np.array([p.hz for p in current]))
    old = (np.array([p.angle for p in earlier]), np.array([p.hz for p in earlier]))
    y_top = 1.15 * max(float(hz[1]), max((p.hz for p in earlier), default=0.0))
    levels: tuple[float, ...] = ()
    modes: tuple[float, ...] = ()
    quotient: tuple[np.ndarray, np.ndarray] = (np.empty(0), np.empty(0))
    if theory:
        a = np.linspace(0.0, 180.0, 361)
        f = np.array([rayleigh_hz(direction(d), M, K) for d in a])
        quotient = (a, f)
        levels = (float(hz[0]), float(hz[1]))
        modes = (angle(shapes[:, 0]), angle(shapes[:, 1]))
    if not current:
        note = (
            "Put the masses on a rail (One arrow), pluck them along it and turn the rail: each rail lands its "
            "frequency here, at its angle."
        )
    else:
        low = min(current, key=lambda p: p.hz)
        high = max(current, key=lambda p: p.hz)
        note = (
            f"{len(current)} rail{'s' if len(current) > 1 else ''} so far: the slowest at {sig(low.angle, 3)}°, "
            f"{fmt(low.hz, 'Hz', 4)}; the fastest at {sig(high.angle, 3)}°, {fmt(high.hz, 'Hz', 4)}."
        )
        if theory:
            note += (
                f" Rayleigh's quotient is never below f₁ = {sig(float(hz[0]), 4)} Hz nor above f₂ = "
                f"{sig(float(hz[1]), 4)} Hz, and touches each along its mode."
            )
    return RailView(pts, old, quotient, levels, modes, now, (-4.0, 184.0), (0.0, y_top), note)
