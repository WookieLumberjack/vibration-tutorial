"""State-space modes' view model (#478 Phase 4, the page after Coordinates & bases): its state, and what each panel
shows, without Qt.

The page's state is a `StateTrial` (its trials, `practice_state()` and `load_setup` all use one): the same two masses
as the last page (`system`), which basis of the state the planes describe them in (`basis`, an index into
`core/state_modes.BASES`), where a trial starts them (`start`, one of `STARTS`) and the tab showing. The page runs
the last page's `live_bases.LiveBases` (free, no rail), and each panel has a function here:

- `labels(s, basis)`: what the two coordinate planes are called and their axes: each mass's (x, ẋ/ω̂), each
  undamped mode's (q, q̇/ω), or each state-space mode's pair (w).
- `pairs(B, x, v)`: the four coordinates w = B⁻¹z for each sample, in mm, a plane a pair.
- `x_plane(s)`: the (x₁, x₂) plane's still parts: each state-space mode's displacements, Re ψ and Im ψ as arrows,
  and the ellipse a mode alone traces (a line when ψ is real).
- `matrices_view(s, basis)`: the steps, M, C and K, then A, then the basis T, then T⁻¹AT with its coupling (the
  off-diagonal blocks' entries that aren't zero) marked, and the modes' λ, f and ζ, the same in every basis.
- `sizes(s, x, v)`: the **Sizes** tab: each pair's size, √(w₁² + w₂²), against time, in the undamped modes and in the
  state-space modes, so the dampers' coupling of the undamped modes shows beside the state-space modes' clean
  decays (straight lines on its log axis).
- `shapes_view(s)`: the **Shapes** tab: each mode's ψ as phasors, one a mass, turned so m₁'s lies along the real
  axis (then m₂'s angle is the phase between them), beside the undamped mode's φ.
- `start_state(s, start)`: where a start puts the masses (x, v).
- `own_view(s, own)`: the fourth basis, the reader's own (`core/state_modes.shape_basis`): each mode's guessed
  shape as m₂'s phasor over m₁'s, ρ = ψ₂/ψ₁, dragged on the Matrices tab (`snap`: within `SNAP` of the plot's
  width of A's own shape, onto it, so the red can be seen to go), with how much that pair still passes to the
  other. A mode of A that doesn't swing keeps A's own pair. `own` is (ρ₁, ρ₂), or None for the modes of M and K's.

Working on it: Qt-free (the package's docstring). `tests/test_state_space_view.py` checks the views.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core.bases import TwoMass, state_matrix
from ..core.practice import Placement, Prediction
from ..core.state_modes import (
    BASES,
    StateMode,
    basis,
    coordinates,
    coupling,
    in_basis,
    own_frequency,
    passed,
    ratio,
    shape_columns,
    state_modes,
    undamped_basis,
    undamped_modes,
)
from .bases import GRIPS, MM
from .text import number as sig

__all__ = ["GRIPS", "MM", "StateTrial"]

TAB_MATRICES, TAB_SIZES, TAB_SHAPES = range(3)
MASS, UNDAMPED, STATE, OWN = range(4)  # the bases, as BASES, then the reader's own
BASIS_NAMES = ("Each mass: x and ẋ", "Modes of M and K: q and q̇", "State-space modes", "Your own: shapes you drag")
STARTS = ("Mode 1 of M and K", "Mode 2 of M and K", "State-space mode 1", "State-space mode 2", "m₁ pulled")
BUTTONS = 4  # the page's Start from buttons are STARTS' first four; m₁ pulled is a trial's own (the reader can pull m₁)
START_REACH = 0.03  # m: a start pulls the farther mass this far
REACH = 0.045  # m: the planes' half-width
ARROW = 0.035  # m: a unit displacement drawn this long on the (x₁, x₂) plane
ZERO = 1e-6  # an entry under this of its matrix's largest is zero
FLOOR = 1e-6  # m: the Sizes tab's log axis stops here
EDGE = 0.9  # m₁'s arrow, when longer than the plot, is drawn to this of its half-width
SNAP = 0.02  # a dragged shape within this of its plot's half-width of A's own snaps onto it


@dataclass(frozen=True)
class StateTrial:
    name: str
    note: str
    m1: float = 1.0
    m2: float = 2.0
    k1: float = 400.0
    k2: float = 100.0
    k3: float = 400.0
    c: float = 1.0
    basis: int = UNDAMPED  # the planes' basis, an index into BASES
    start: int | None = None  # where the trial starts the masses, an index into STARTS (None: at rest)
    tab: int = TAB_MATRICES
    own1: tuple[float, float] | None = None  # your own basis: mode 1's ψ₂/ψ₁ as (Re, Im); None: the mode of M and K's
    own2: tuple[float, float] | None = None
    predict: Prediction | Placement | None = None  # asked once the reader has played (Things to try)


def system(t: StateTrial) -> TwoMass:
    return TwoMass(t.m1, t.m2, t.k1, t.k2, t.k3, t.c)


Own = tuple[complex, complex] | None  # your own basis's ψ₂/ψ₁ for each mode, or None: the modes of M and K's


def own_of(t: StateTrial) -> Own:
    """A trial's own shapes, as the view's functions take them."""
    if t.own1 is None or t.own2 is None:
        return None
    return complex(*t.own1), complex(*t.own2)


def own_fields(own: Own) -> dict[str, tuple[float, float] | None]:
    """The trial's fields for `own`."""
    if own is None:
        return {"own1": None, "own2": None}
    return {f"own{r + 1}": (float(own[r].real), float(own[r].imag)) for r in range(2)}


def undamped_ratios(s: TwoMass) -> tuple[complex, complex]:
    """Each mode of M and K's φ₂/φ₁: where your own shapes start."""
    M, _, K = s.matrices()
    _, V = undamped_modes(M, K)
    return complex(V[1, 0] / V[0, 0]), complex(V[1, 1] / V[0, 1])


def own_ratios(s: TwoMass, own: Own) -> tuple[complex, complex]:
    return undamped_ratios(s) if own is None else own


def basis_of(s: TwoMass, which: int, own: Own = None) -> np.ndarray:
    M, C, K = s.matrices()
    if which != OWN:
        return basis(BASES[which], M, C, K)
    ratios = own_ratios(s, own)
    return np.column_stack(
        [shape_columns(M, C, K, ratios[r]) if m.swings else m.columns for r, m in enumerate(modes(s))]
    )


def modes(s: TwoMass) -> tuple[StateMode, ...]:
    M, C, K = s.matrices()
    return state_modes(M, C, K)


# ------------------------------------------------------------------ the planes


@dataclass(frozen=True)
class PlaneLabels:
    title: str
    x: str
    y: str


def labels(s: TwoMass, which: int) -> tuple[PlaneLabels, PlaneLabels]:
    """The two coordinate planes' titles and axes in the basis `which`."""
    sub = "₁₂₃₄"
    if which == MASS:
        M, _, K = s.matrices()
        w = [own_frequency(M, K, i) for i in range(2)]
        one, two = (
            PlaneLabels(f"Mass {i + 1}", f"x{sub[i]}", f"ẋ{sub[i]}/ω̂{sub[i]}  (ω̂ = {sig(w[i], 3)} rad/s)")
            for i in range(2)
        )
    elif which == UNDAMPED:
        one, two = (PlaneLabels(f"Mode {r + 1} of M and K", f"q{sub[r]}", f"q̇{sub[r]}/ω{sub[r]}") for r in range(2))
    elif which == OWN:
        one, two = (PlaneLabels(f"Your mode {r + 1}", f"w{sub[2 * r]}", f"w{sub[2 * r + 1]}") for r in range(2))
    else:
        one, two = (
            PlaneLabels(
                f"State-space mode {r + 1}" + ("" if m.swings else ": overdamped"),
                f"w{sub[2 * r]}",
                f"w{sub[2 * r + 1]}",
            )
            for r, m in enumerate(modes(s))
        )
    return one, two


def pairs(B: np.ndarray, x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """The coordinates w = B⁻¹z (mm) of each sample (rows of x and v, m and m/s): a column each, pairs adjacent."""
    z = np.hstack([np.atleast_2d(x), np.atleast_2d(v)])
    return coordinates(B, z) * MM


def place(B: np.ndarray, w: np.ndarray, pair: int, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    """The state (x, v) with pair `pair`'s coordinates set to (a, b) (mm) and the other pairs' kept as in `w` (mm)."""
    w = np.array(w, dtype=float)
    w[2 * pair], w[2 * pair + 1] = a, b
    z = B @ (w / MM)
    return z[:2], z[2:]


@dataclass(frozen=True)
class ModeShape:
    re: tuple[float, float]  # Re ψ's tip (mm, drawn ARROW long a unit)
    im: tuple[float, float]  # Im ψ's tip (mm)
    ellipse: tuple[np.ndarray, np.ndarray]  # the path a mode alone traces, at its size as drawn (mm)
    swings: bool


@dataclass(frozen=True)
class XPlane:
    shapes: tuple[ModeShape, ...]
    span: float  # mm
    note: str


def x_plane(s: TwoMass) -> XPlane:
    """The (x₁, x₂) plane's still parts: each mode's Re ψ and Im ψ, and its ellipse (see the module)."""
    out = []
    theta = np.linspace(0.0, 2.0 * math.pi, 121)
    for m in modes(s):
        p = m.shape * ARROW * MM
        if m.swings:
            ex = p[0].real * np.cos(theta) - p[0].imag * np.sin(theta)
            ey = p[1].real * np.cos(theta) - p[1].imag * np.sin(theta)
        else:
            ex, ey = np.array([-p[0].real, p[0].real]), np.array([-p[1].real, p[1].real])
        out.append(
            ModeShape((float(p[0].real), float(p[1].real)), (float(p[0].imag), float(p[1].imag)), (ex, ey), m.swings)
        )
    phases = [abs(_off_line(m.phase)) for m in modes(s) if m.swings]
    if not phases:
        note = "Both modes overdamped: no swing, each a pair of decays along its two real shapes."
    elif max(phases) < 0.05:
        note = "Each mode's shape is real: alone, a mode swings back and forth along a line, the masses in step."
    else:
        note = (
            f"The shapes are complex: Re ψ and Im ψ point apart, and a mode alone traces an ellipse, the masses "
            f"{sig(max(phases), 2)}° out of step at most."
        )
    return XPlane(tuple(out), REACH * MM, note)


def _off_line(phase: float) -> float:
    """How far (°) a phase between the masses is from in step (0) or opposed (180)."""
    return (phase + 90.0) % 180.0 - 90.0


# ------------------------------------------------------------------ the matrices


@dataclass(frozen=True)
class Grid:
    symbol: str
    unit: str
    entries: tuple[tuple[str, ...], ...]
    marked: tuple[tuple[bool, ...], ...]  # a coupling term (off the diagonal blocks, not zero)
    blocks: bool  # drawn as 2×2 blocks (a matrix in a paired basis)
    rows: tuple[str, ...] = ()  # what each row and column is (a coordinate's name)
    cols: tuple[str, ...] = ()
    row_groups: tuple[str, ...] = ()  # the block each row and column is in, for its tint (`matrix_table`'s groups)
    col_groups: tuple[str, ...] = ()


@dataclass(frozen=True)
class MatricesView:
    given: tuple[Grid, ...]  # M, C, K
    state: Grid  # A
    basis: Grid  # T (the notes' name for the code's B: B is the input matrix on later pages)
    product: Grid  # T⁻¹AT
    modes: tuple[str, ...]  # a line a mode: λ, f, ζ
    coupled: bool
    note: str


def _grid(
    symbol: str,
    unit: str,
    A: np.ndarray,
    rows: tuple[str, ...],
    cols: tuple[str, ...],
    groups: tuple[tuple[str, ...], tuple[str, ...]],
    blocks: bool = False,
    coupling: bool = True,
) -> Grid:
    scale = float(np.abs(A).max()) or 1.0
    n = A.shape[0]
    entries = tuple(tuple("0" if abs(float(a)) < ZERO * scale else sig(float(a), 3) for a in row) for row in A)
    marked = tuple(
        tuple(coupling and blocks and i // 2 != j // 2 and abs(float(A[i, j])) >= ZERO * scale for j in range(n))
        for i in range(n)
    )
    return Grid(symbol, unit, entries, marked, blocks, rows, cols, *groups)


X = ("x₁", "x₂")
STATE_NAMES = ("x₁", "x₂", "ẋ₁", "ẋ₂")
BY_MASS = ("i", "b")  # M, C, K: a block a mass
BY_HALF = ("i", "i", "b", "b")  # A: displacements, then velocities; T⁻¹AT: a block a pair
PLAIN = ("q", "q", "q", "q")  # T: one tint, it's a basis, not a map


def names(s: TwoMass, which: int) -> tuple[str, ...]:
    """The four coordinates' short names in the basis `which`, pairs adjacent (the planes' axes)."""
    return tuple(n for p in labels(s, which) for n in (p.x, p.y.split("  ")[0]))


def mode_line(r: int, m: StateMode) -> str:
    if m.swings:
        return (
            f"Mode {r + 1}: λ = {sig(m.sigma, 3)} ± {sig(m.omega, 4)}i /s: f = {sig(m.omega / (2 * math.pi), 4)} Hz, "
            f"ζ = {sig(100 * m.zeta, 3)} %"
        )
    a, b = (float(e.real) for e in m.eigenvalues)
    return f"Mode {r + 1}: λ = {sig(a, 3)} and {sig(b, 3)} /s: overdamped, two decays, no swing"


def matrices_view(s: TwoMass, which: int, own: Own = None) -> MatricesView:
    """The steps from M, C, K to A in the basis `which` (see the module), and what they say."""
    M, C, K = s.matrices()
    A = state_matrix(M, C, K)
    B = basis_of(s, which, own)
    Ab = in_basis(B, A)
    mass_groups = (BY_MASS, BY_MASS)
    given = (
        _grid("M", "kg", M, X, X, mass_groups),
        _grid("C", "N·s/m", C, X, X, mass_groups),
        _grid("K", "N/m", K, X, X, mass_groups),
    )
    w = names(s, which)
    coupled = coupling(Ab) >= ZERO
    ms = modes(s)
    if which == MASS:
        note = "Each mass's own coordinates: the springs and dampers both couple them (in red)."
        if not coupled:
            note = "Each mass's own coordinates, and nothing couples them: no coupling spring, no dampers between."
    elif which == UNDAMPED:
        if coupled:
            note = (
                "The modes of M and K: the springs no longer couple them, but the dampers still do (in red): "
                "the damping isn't proportional: the dampers' forces in one mode push on the other."
            )
        elif s.c > 0:
            note = (
                "The modes of M and K, and uncoupled: the damping is proportional, so the dampers' forces in one mode "
                "don't push on the other."
            )
        else:
            note = "The modes of M and K, and uncoupled: no dampers to couple them."
    elif which == OWN:
        note = (
            "Your own shapes: each mode's pair is A's own (no red under its columns) only where its shape is."
            if coupled
            else "Your own shapes are A's own modes: T⁻¹AT is two blocks, nothing red."
        )
    else:
        swinging = all(m.swings for m in ms)
        note = (
            "A's own modes: uncoupled whatever the dampers, one 2×2 block a mode, σ on its diagonal and ω off it."
            if swinging
            else "A's own modes: uncoupled whatever the dampers; an overdamped mode's block is its two decay rates."
        )
    return MatricesView(
        given,
        _grid("A", "", A, STATE_NAMES, STATE_NAMES, (BY_HALF, BY_HALF)),
        _grid("T", "", B, STATE_NAMES, w, (PLAIN, PLAIN), blocks=True, coupling=False),
        _grid("T⁻¹AT", "", Ab, w, w, (BY_HALF, BY_HALF), blocks=True),
        tuple(mode_line(r, m) for r, m in enumerate(ms)),
        coupled,
        note,
    )


# ------------------------------------------------------------------ the Sizes tab


@dataclass(frozen=True)
class Sizes:
    undamped: tuple[np.ndarray, np.ndarray]  # each undamped mode's pair's size (mm) a sample
    state: tuple[np.ndarray, np.ndarray]  # each state-space mode's
    note: str


def sizes(s: TwoMass, x: np.ndarray, v: np.ndarray) -> Sizes:
    """Each pair's size, √(a² + b²) (mm, at least FLOOR), in both modal bases (see the module)."""
    M, _, K = s.matrices()
    floor = FLOOR * MM
    if x.shape[0] == 0:
        empty = (np.empty(0), np.empty(0))
        return Sizes(empty, empty, sizes_note(s))

    def each(B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w = pairs(B, x, v)
        one, two = (np.maximum(np.hypot(w[:, 2 * r], w[:, 2 * r + 1]), floor) for r in range(2))
        return one, two

    return Sizes(each(undamped_basis(M, K)), each(basis_of(s, STATE)), sizes_note(s))


def sizes_note(s: TwoMass) -> str:
    """What the Sizes tab shows for this system."""
    M, C, K = s.matrices()
    ms = modes(s)
    rates = ", ".join(f"mode {r + 1} at σ = {sig(m.sigma, 3)} /s" for r, m in enumerate(ms) if m.swings)
    coupled = coupling(in_basis(undamped_basis(M, K), state_matrix(M, C, K))) >= ZERO
    first = (
        "In the modes of M and K the dampers pass motion between the pairs: start one alone and the other's size "
        "climbs from nothing."
        if coupled
        else "In the modes of M and K nothing passes between the pairs here: the damping is proportional."
    )
    second = (
        f" In the state-space modes each pair shrinks on its own as e^(σt), a straight line on this log axis: {rates}."
        if rates
        else " In the state-space modes each pair decays on its own."
    )
    return first + second


# ------------------------------------------------------------------ the Shapes tab


@dataclass(frozen=True)
class Phasors:
    title: str
    masses: tuple[tuple[float, float], tuple[float, float]]  # each mass's ψ as (Re, Im)
    undamped: tuple[tuple[float, float], tuple[float, float]]  # each mass's φ (real)
    note: str


def shapes_view(s: TwoMass) -> tuple[Phasors, ...]:
    """Each mode's ψ, a phasor a mass, beside the undamped mode's φ (see the module)."""
    M, _, K = s.matrices()
    _, V = undamped_modes(M, K)
    out = []
    for r, m in enumerate(modes(s)):
        p = m.shape * np.exp(-1j * np.angle(m.shape[0]))  # turned so m₁'s phasor is along +Re: m₂'s angle is the phase
        masses = ((float(p[0].real), float(p[0].imag)), (float(p[1].real), float(p[1].imag)))
        sign = 1.0 if V[0, r] >= 0 else -1.0
        phi = ((float(sign * V[0, r]), 0.0), (float(sign * V[1, r]), 0.0))
        if not m.swings:
            note = "Overdamped: no swing, so no phase. Its shape is real, and so is its second's."
        else:
            off = _off_line(m.phase)
            ideal, kind = ("0°", "in step") if abs(m.phase) < 90 else ("180°", "opposed")
            if abs(off) < 0.05:
                note = f"The masses move exactly {kind}: ψ is real, the undamped mode's own shape."
            else:
                way = "lags" if m.phase < 0 else "leads"
                note = (
                    f"m₂ {way} m₁ by {sig(abs(m.phase), 3)}°, not the {ideal} of moving {kind}: ψ is complex, so the masses don't "
                    "reach their peaks together."
                )
        out.append(Phasors(f"Mode {r + 1}", masses, phi, note))
    return tuple(out)


# ------------------------------------------------------------------ the starts


def start_state(s: TwoMass, start: int) -> tuple[np.ndarray, np.ndarray]:
    """Where a start (`STARTS`) puts the masses: (x m, v m/s), the farther mass START_REACH out. A mode of M and K
    starts from rest, displaced along its shape; a state-space mode along Re ψ and already moving, Re(λψ): what it
    takes for the motion to stay in that one mode when the dampers aren't proportional."""
    M, C, K = s.matrices()
    if start == 4:
        return np.array([START_REACH, 0.0]), np.zeros(2)
    if start in (0, 1):
        _, V = undamped_modes(M, K)
        x = V[:, start]
        return x * START_REACH / float(np.abs(x).max()), np.zeros(2)
    m = state_modes(M, C, K)[start - 2]
    z = m.columns[:, 0]
    z = z * START_REACH / float(np.abs(z[:2]).max())
    return z[:2], z[2:]


def _pair(values: np.ndarray, scale: float) -> str:
    """Two numbers to 3 figures, a rounding error's (under a nanometer) printed as 0."""
    a, b = (float(u) * scale for u in values[:2])
    return f"({sig(a if abs(a) > 1e-6 else 0.0, 3)}, {sig(b if abs(b) > 1e-6 else 0.0, 3)})"


def start_conditions(s: TwoMass, start: int) -> str:
    """A start's initial conditions, x (mm) and ẋ (mm/s), and how it sets the masses going (the buttons' tips)."""
    x, v = start_state(s, start)
    how = (
        "from rest"
        if start == 4
        else "displaced along the mode's shape, from rest"
        if start in (0, 1)
        else "displaced along Re ψ and already moving, as the damped mode moves"
        if float(np.abs(v).max()) > 1e-9
        else "displaced along the mode's shape, from rest: with no damping, the mode of M and K itself"
    )
    return f"x = {_pair(x, MM)} mm, ẋ = {_pair(v, MM)} mm/s: {how}."


def start_note(s: TwoMass, start: int, which: int, own: Own = None) -> str:
    """The last start: its name, its initial conditions, and where it puts each pair in the planes' basis (mm), so
    two starts that look alike on the masses (a mode of M and K and the state-space mode it's near) tell apart."""
    x, v = start_state(s, start)
    w = pairs(basis_of(s, which, own), x, v)[0]
    return (
        f"{STARTS[start]}: {start_conditions(s, start)} On the planes: {_pair(w[:2], 1.0)} and {_pair(w[2:], 1.0)} mm."
    )


# ------------------------------------------------------------------ your own basis


@dataclass(frozen=True)
class OwnShape:
    title: str
    span: float  # the plot's half-width (ρ's units: m₂ over m₁)
    ratio: tuple[float, float]  # your ψ₂/ψ₁ (Re, Im): m₂'s phasor, m₁'s the unit along Re
    undamped: float  # the mode of M and K's φ₂/φ₁, where yours starts (real)
    m1: float  # m₁'s unit arrow as drawn: cut at the plot's edge (`EDGE`) when 1 is off it
    movable: bool  # False: A's mode doesn't swing, and keeps its own pair
    found: bool  # your shape is A's own
    note: str


def own_span(s: TwoMass, r: int) -> float:
    """Mode r's drag plot's half-width: room round the mode of M and K's ratio (m₁'s unit arrow may run off it, so
    a mode whose m₂ moves little is still dragged finely)."""
    return 1.5 * abs(undamped_ratios(s)[r])


def snap(s: TwoMass, r: int, at: complex) -> complex:
    """Where a drag to `at` puts mode r's ratio: kept within its plot, and onto A's own if within SNAP of it."""
    span = own_span(s, r)
    at = complex(max(-span, min(span, at.real)), max(-span, min(span, at.imag)))
    true = ratio(modes(s)[r])
    return true if abs(at - true) <= SNAP * span else at


def own_view(s: TwoMass, own: Own) -> tuple[OwnShape, ...]:
    """Each mode's own shape on its drag plot, and what it does to T⁻¹AT (see the module)."""
    M, C, K = s.matrices()
    ratios = own_ratios(s, own)
    phi = undamped_ratios(s)
    Ab = in_basis(basis_of(s, OWN, own), state_matrix(M, C, K))
    out = []
    for r, m in enumerate(modes(s)):
        rho = ratios[r]
        span = own_span(s, r)
        m1 = min(1.0, EDGE * span)
        other = 2 - r
        size = math.degrees(math.atan2(rho.imag, rho.real))
        where = f"m₂/m₁ = {sig(abs(rho), 3)} at {sig(size, 3)}°"
        if not m.swings:
            out.append(
                OwnShape(
                    f"Mode {r + 1}",
                    span,
                    (1.0, 0.0),
                    phi[r].real,
                    m1,
                    False,
                    True,
                    f"Mode {r + 1} is overdamped: it keeps A's own pair, two real decays.",
                )
            )
            continue
        share = passed(Ab, r)
        found = share < ZERO
        note = (
            f"{where}: A's own mode {r + 1}. Its pair passes nothing to mode {other}'s: its columns' red is gone."
            if found
            else f"{where}: its pair passes motion to mode {other}'s, up to {sig(100 * share, 2)} % of A's largest "
            "term (the red in its columns)."
        )
        out.append(OwnShape(f"Mode {r + 1}", span, (rho.real, rho.imag), phi[r].real, m1, True, found, note))
    return tuple(out)
