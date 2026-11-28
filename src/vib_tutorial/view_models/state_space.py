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
- `start_state(s, start)`: where a start puts the masses (x, v).
- `pair_shapes(s, basis, own)`: each pair of the basis as a shape on the Matrices tab, m₂'s part as a phasor
  beside m₁'s (a state-space mode's note says how far m₂ lags or leads), with the ψ and λ whose Re and Im of [ψ; λψ] are its columns of T (every basis's are: a mass alone
  and iω̂, a mode of M and K and iω, A's own) and what it passes to the other pair. In the fourth basis, the
  reader's own (`core/state_modes.shape_basis`), m₂'s phasor ρ = ψ₂/ψ₁ is dragged (`snap`: within `SNAP` of the
  plot's half-width of A's own shape, onto it, so the red can be seen to go), with a gauge of how near A's it is.
  A mode of A that doesn't swing keeps A's own pair. `own` is (ρ₁, ρ₂), or None for the modes of M and K's.

Working on it: Qt-free (the package's docstring). `tests/test_state_space_view.py` checks the views.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from ..core.bases import TwoMass, state_matrix
from ..core.practice import Placement, Prediction
from ..core.state_modes import (
    BASES,
    StateMode,
    basis,
    complex_columns,
    coordinates,
    coupling,
    guessed_mode,
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

TAB_MATRICES, TAB_SIZES = range(2)
MASS, UNDAMPED, STATE, OWN = range(4)  # the bases, as BASES, then the reader's own
BASIS_NAMES = ("Each mass: x and ẋ", "Modes of M and K: q and q̇", "State-space modes", "Your own: shapes you drag")
REAL, COMPLEX = range(2)  # how the Matrices tab writes the basis: T's Re y and Im y, or Y's y and ȳ
WRITTEN = ("Real: T, each pair Re y and Im y", "Complex: Y, each pair y and ȳ")
STARTS = ("Mode 1 of M and K", "Mode 2 of M and K", "State-space mode 1", "State-space mode 2", "m₁ pulled")
BUTTONS = 4  # the page's Start from buttons are STARTS' first four; m₁ pulled is a trial's own (the reader can pull m₁)
START_REACH = 0.03  # m: a start pulls the farther mass this far
REACH = 0.045  # m: the planes' half-width
ARROW = 0.035  # m: a unit displacement drawn this long on the (x₁, x₂) plane
ZERO = 1e-6  # an entry under this of its matrix's largest is zero
EMPTY = 0.01  # the gauge is empty at your start's coupling, or this of A's largest term if that's more
FLOOR = 1e-6  # m: the Sizes tab's log axis stops here
EDGE = 0.9  # m₁'s arrow, when longer than the plot, is drawn to this of its half-width
SNAP = 0.05  # a dragged shape within this of its plot's half-width of A's own snaps onto it


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
    written: int = REAL  # the Matrices tab's basis written real (T) or complex (Y), an index into WRITTEN
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
    lit: int | None = None  # the pair whose columns are lit (the shape being dragged), drawn in its plane's color


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
    entries = tuple(tuple(_entry(complex(a), scale) for a in row) for row in A)
    marked = tuple(
        tuple(coupling and blocks and i // 2 != j // 2 and abs(complex(A[i, j])) >= ZERO * scale for j in range(n))
        for i in range(n)
    )
    return Grid(symbol, unit, entries, marked, blocks, rows, cols, *groups)


def _entry(a: complex, scale: float) -> str:
    """A matrix's entry to 3 figures, a part under ZERO of the matrix's largest entry dropped (0 if both are)."""
    re, im = (v if abs(v) >= ZERO * scale else 0.0 for v in (a.real, a.imag))
    return _complex(complex(re, im)) if re or im else "0"


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


def complex_names(s: TwoMass, which: int) -> tuple[str, ...]:
    """The coordinates η = Y⁻¹z of the basis `which` written complex: η and η̄ a swinging pair, an overdamped mode's
    two real ones as they were."""
    real = names(s, which)
    return tuple(
        n
        for r, swings in enumerate(pairs_swing(s, which))
        for n in ((f"η{'₁₂'[r]}", f"η̄{'₁₂'[r]}") if swings else real[2 * r : 2 * r + 2])
    )


def pairs_swing(s: TwoMass, which: int) -> tuple[bool, ...]:
    """Each pair of the basis `which` is Re and Im of a complex [ψ; λψ]: all but an overdamped mode's, two real ones."""
    return tuple(which not in (STATE, OWN) or m.swings for m in modes(s))


def matrices_view(s: TwoMass, which: int, own: Own = None, lit: int | None = None, written: int = REAL) -> MatricesView:
    """The steps from M, C, K to A in the basis `which` (see the module), written real (T) or complex (Y), and what
    they say."""
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
    T, Tb, symbol = B, Ab, "T"
    if written == COMPLEX:
        T = complex_columns(B, pairs_swing(s, which))
        Tb, symbol, w = in_basis(T, A), "Y", complex_names(s, which)
        note = note.replace("T⁻¹AT", "Y⁻¹AY")
        if which == STATE:
            note = (
                "A's own modes, written complex: Y⁻¹AY is diagonal whatever the dampers, each mode's λ and λ̄ (an "
                "overdamped mode's two decay rates)."
            )
    return MatricesView(
        given,
        _grid("A", "", A, STATE_NAMES, STATE_NAMES, (BY_HALF, BY_HALF)),
        dataclasses.replace(_grid(symbol, "", T, STATE_NAMES, w, (PLAIN, PLAIN), blocks=True, coupling=False), lit=lit),
        dataclasses.replace(_grid(f"{symbol}⁻¹A{symbol}", "", Tb, w, w, (BY_HALF, BY_HALF), blocks=True), lit=lit),
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


# ------------------------------------------------------------------ a mode's phase


def phase_note(m: StateMode) -> str:
    """How a swinging mode's masses move together: exactly in step or opposed (ψ real), or m₂ lagging or leading m₁."""
    ideal, kind = ("0°", "in step") if abs(m.phase) < 90 else ("180°", "opposed")
    if abs(_off_line(m.phase)) < 0.05:
        return f"Real: the masses move exactly {kind}, as the mode of M and K."
    way = "lags" if m.phase < 0 else "leads"
    return (
        f"Complex: m₂ {way} m₁ by {sig(abs(m.phase), 3)}°, not the {ideal} of moving {kind}, so the masses don't "
        "reach their peaks together."
    )


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


# ------------------------------------------------------------------ the pairs as shapes


@dataclass(frozen=True)
class PairShape:
    """One pair of the basis drawn as a shape: m₂'s part as a phasor beside m₁'s, on the Matrices tab."""

    title: str
    span: float  # the plot's half-width (m₁'s part's units)
    m1: tuple[float, float]  # m₁'s arrow's tip as drawn: the unit along Re, cut at `EDGE` of the plot if 1 is off it
    m1_cut: bool  # m₁'s arrow runs on off the plot (to 1)
    m2: tuple[float, float]  # m₂'s: ψ₂/ψ₁ (or the unit along Re, where m₁ doesn't move)
    start: float | None  # dashed: the mode of M and K's ψ₂/ψ₁, where yours start, beside A's and yours
    movable: bool  # yours, and A's mode swings: m₂'s arrow is dragged
    closeness: float | None  # yours: how near A's own, 0 to 1 by the coupling it leaves (3 decades); else None
    columns: str  # the λ, ψ (T's x rows) and λψ (its ẋ rows) whose Re and Im are this pair's columns of T
    note: str


def own_span(s: TwoMass, r: int) -> float:
    """Mode r's plot's half-width: room round the mode of M and K's ψ₂/ψ₁ (m₁'s unit arrow may run off it, so a
    mode whose m₂ moves little is still dragged finely)."""
    return max(1.5 * abs(undamped_ratios(s)[r]), 0.1)


def snap(s: TwoMass, r: int, at: complex) -> complex:
    """Where a drag to `at` puts mode r's ratio: kept within its plot, and onto A's own if within SNAP of it."""
    span = own_span(s, r)
    at = complex(max(-span, min(span, at.real)), max(-span, min(span, at.imag)))
    true = ratio(modes(s)[r])
    return true if abs(at - true) <= SNAP * span else at


def _complex(z: complex) -> str:
    """A complex number to 3 figures: 0.937 - 0.073i, 15.2i, 0.349."""
    re, im = (v if abs(v) > 1e-9 * max(abs(z), 1e-300) else 0.0 for v in (z.real, z.imag))
    if not im:
        return sig(re, 3)
    if not re:
        return f"{sig(im, 3)}i"
    return f"{sig(re, 3)} {'+' if im > 0 else '-'} {sig(abs(im), 3)}i"


SHAPE_TITLES = (
    "Each pair as a shape: m₁'s part along Re, m₂'s beside it",
    "Each pair as a shape: m₁'s part along Re, m₂'s beside it",
    "Each pair as a shape: m₁'s part along Re, m₂'s beside it (dashed: the mode of M and K's)",
    "Your own shapes: drag m₂'s arrow (dashed: where it started)",
)


def pair_shapes(s: TwoMass, which: int, own: Own = None, written: int = REAL) -> tuple[PairShape, ...]:
    """Each pair of the basis `which` as a shape, the ψ and λ that make its columns, and what it passes on (see the
    module). Every basis's pair is Re and Im of [ψ; λψ]: a mass alone and iω̂, a mode of M and K and iω, A's own,
    or yours with the λ it asks for."""
    M, C, K = s.matrices()
    Ab = in_basis(basis_of(s, which, own), state_matrix(M, C, K))
    names_ = names(s, which) if written == REAL else complex_names(s, which)
    phi = undamped_ratios(s)
    omega, V = undamped_modes(M, K)
    ms = modes(s)
    first = in_basis(basis_of(s, OWN, None), state_matrix(M, C, K)) if which == OWN else None
    out = []
    for r in range(2):
        other = 2 - r
        cols = f"{names_[2 * r]} and {names_[2 * r + 1]}"
        share = passed(Ab, r)
        found = share < ZERO
        span = own_span(s, r)
        start: float | None = None
        closeness: float | None = None
        movable = False
        if which == MASS:
            psi = np.eye(2)[r].astype(complex)
            lam = complex(0.0, own_frequency(M, K, r))
            span = 1.2
            title = f"Mass {r + 1}"
            note = (
                f"m{'₁₂'[r]} alone: the other mass doesn't move in this pair, so it isn't a mode. The springs and "
                f"dampers pass its motion to the other pair, up to {sig(100 * share, 2)} % of A's largest term."
            )
        elif which == UNDAMPED:
            psi, lam = V[:, r].astype(complex), complex(0.0, omega[r])
            title = f"Mode {r + 1} of M and K"
            kind = "in step" if phi[r].real > 0 else "opposed"
            note = f"Real: the masses exactly {kind}, as the last page's mode. " + (
                f"Its pair passes nothing to mode {other}'s: the damping is proportional, or there is none."
                if found
                else f"Through the dampers it passes motion to mode {other}'s pair, up to {sig(100 * share, 2)} % "
                "of A's largest term (the red in its columns)."
            )
        else:
            m = ms[r]
            if which == STATE or not m.swings:
                psi, lam = m.shape, m.eigenvalues[0]
                start = phi[r].real
                title = f"State-space mode {r + 1}" if which == STATE else f"Your mode {r + 1}"
                note = (
                    f"{phase_note(m)} A's own mode {r + 1}: its pair passes nothing to mode {other}'s."
                    if m.swings
                    else f"Mode {r + 1} is overdamped: its pair is A's own, two real decays, and passes nothing on."
                )
            else:
                psi, lam = guessed_mode(M, C, K, own_ratios(s, own)[r])
                start, movable, title = phi[r].real, True, f"Your mode {r + 1}"
                assert first is not None, "worked out for your own basis"
                share0 = passed(first, r)
                empty = max(share0, EMPTY)  # undamped, the start is A's own: empty a way from it, not at it
                closeness = 1.0 if found else min(max(math.log10(empty / share) / 3.0, 0.0), 1.0)
                note = (
                    f"A's own mode {r + 1}: its pair passes nothing to mode {other}'s, and the red in its columns "
                    "is gone."
                    if found
                    else f"Its pair passes motion to mode {other}'s, up to {sig(100 * share, 2)} % of A's largest "
                    "term (the red in its columns)."
                )
        if abs(psi[0]) > 1e-12 and abs(psi[1]) > 1e-12:
            rho = complex(psi[1] / psi[0])
            m1, m2 = min(1.0, EDGE * span), (rho.real, rho.imag)
            angle = math.degrees(math.atan2(rho.imag, rho.real))
            where = f"m₂/m₁ = {sig(abs(rho), 3)} at {sig(angle, 3)}°. "
        else:  # one mass doesn't move: the other's part is the unit
            m1, m2, where = 0.0, (min(1.0, EDGE * span), 0.0), ""
        if abs(psi[1]) <= 1e-12:  # m₂ doesn't move: m₁'s part alone
            m1, m2 = min(1.0, EDGE * span), (0.0, 0.0)
        if which in (STATE, OWN) and not ms[r].swings:
            a, b = ms[r].eigenvalues
            columns = (
                f"{'TY'[written]}'s {cols}: its two real eigenvectors [ψ; λψ], one for each decay rate, λ = {_complex(a)} and "
                f"{_complex(b)} /s: rows x₁, x₂ are ψ, rows ẋ₁, ẋ₂ are λψ."
            )
        else:
            v = lam * psi  # the velocities' rows: ẋ = λx along the mode
            are = "are Re and Im of [ψ; λψ]" if written == REAL else "are y = [ψ; λψ] and its conjugate ȳ"
            columns = (
                f"{'TY'[written]}'s {cols} {are}, with λ = {_complex(lam)} /s. Rows x₁, x₂: ψ = "
                f"({_complex(psi[0])}, {_complex(psi[1])}). Rows ẋ₁, ẋ₂: λψ = ({_complex(v[0])}, {_complex(v[1])})."
            )
        out.append(
            PairShape(
                title,
                span,
                (m1, 0.0),
                0 < m1 < 1.0,
                m2,
                start,
                movable,
                closeness,
                columns,
                where + note,
            )
        )
    return tuple(out)
