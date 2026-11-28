"""Vibration isolation's view model (#478 Phase 4): its state, and what each panel shows, without Qt.

The page's state is an `IsolationTrial` (its trials, `practice_state()` and `load_setup` all use one): the machine
(its mass, speed and unbalance, an inertia block), its mounts (their kind, sag and damping, the relaxation's
spring, the intermediate mass), what shakes (`shaking`: the machine's unbalance or the floor), the run-up's time,
the isolation wanted, the tab showing and whether the theory is drawn over what's measured. `isolator(trial)` is
the machine on its mounts. The page runs it live (`live_isolation.LiveIsolation`), and each panel has a function
here:

- `mount_info(iso)`: the mounts' k, c and f<sub>n</sub>, under their boxes.
- `t_view(state, points, theory)`: the **Transmissibility** tab, built by hand: each steady speed's measured T
  against f/f<sub>n</sub>, the machine's points and the floor's in their own markers (they fall on one curve), points
  from before a change faded, and the theory (this isolator, and plain mounts at `FAMILY`'s ζ) only when asked
  for. Its x axis is fixed, `RATIOS`' span, so the points don't move it.
- `force_view(state, rpm, peaks, theory)`: the **Floor force** tab, built as the speed runs up and down: each whole
  turn's largest force on the floor against the speed (or, the floor shaking, the machine's motion), beside the
  unbalance's own force, what a rigid mount would pass (or the floor's motion), with the steady theory on request.
- `steady_note(iso, T, shaking)`: the readout once a speed's T has landed: the forces (or the motions), T and the
  isolation in % and dB.
- `design_note(iso, target)` and `chart(iso)`: the **Design chart** tab, the sag each isolation needs.
- `key(state)`: what a measured point belongs to: the isolator, its speed and unbalance aside (T doesn't depend on
  them).

Working on it: keep it free of Qt (see the package's docstring). `tests/test_view_models.py` checks the panels and
ties the readouts to the numbers the notes quote, as printed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from ..core.isolation import G, Isolator, Mount, deflection_for
from ..core.practice import Placement, Prediction
from ..core.sdof import TWO_PI
from .live_isolation import FLOOR, FLOOR_AMPLITUDE, MACHINE
from .plot import Curve
from .recorder import Point
from .text import db, fmt
from .text import number as sig

MOUNTS = list(Mount)
FAMILY = (0.0, 0.1, 0.3, 0.7)  # the damping ratios drawn behind the isolator's own curve
RATIOS = np.geomspace(0.1, 20.0, 1500)
CHART_SPEEDS = np.geomspace(5.0, 200.0, 200)  # Hz on the design chart
CHART_LEVELS = (0.5, 0.8, 0.9, 0.95, 0.99)
TAB_T, TAB_FORCE, TAB_CHART = range(3)
T_RANGE = (-4.0, math.log10(30.0))  # the transmissibility plot's y range, as log10
START_RUN, START_RUN_UP = range(2)  # a trial's start: at speed from rest, or run up from standstill over `ramp`
SWEEP_SPAN = (0.4, 5.0)  # × f_n: a sweep's speeds
SWEEP_POINTS = 12


@dataclass(frozen=True)
class IsolationTrial:
    name: str
    note: str
    deflection_mm: float = 5.0
    zeta: float = 0.05
    mount: Mount = Mount.VISCOUS
    stiffness_ratio: float = 3.0
    stage_mass: float = 0.5
    block: float = 0.0
    rpm: float = 1500.0
    ramp: float = 2.0  # s: a run up's or down's time
    tab: int = TAB_T
    mass: float = 100.0
    unbalance: float = 0.01
    shaking: int = MACHINE  # what shakes: the machine's unbalance or the floor
    target: float = 90.0  # %: the isolation wanted (the Design chart's)
    theory: bool = False  # the theory drawn over what's measured
    start: int | None = None  # how the trial sets it going (`START_RUN`, `START_RUN_UP`), None: at rest
    predict: Prediction | Placement | None = None  # asked once the reader has played (Things to try)


def isolator(t: IsolationTrial) -> Isolator:
    """The machine on its mounts as a trial (or the page's `practice_state`) sets them."""
    return Isolator(
        t.mass,
        t.rpm,
        t.unbalance,
        1e-3 * t.deflection_mm,
        t.zeta,
        t.mount,
        t.stiffness_ratio,
        t.stage_mass,
        t.block,
    )


def key(t: IsolationTrial) -> Isolator:
    """What a measured point belongs to: the isolator, its speed and unbalance aside."""
    return replace(isolator(t), rpm=0.0, unbalance=0.0)


def mount_info(iso: Isolator) -> str:
    fn = iso.natural_hz
    return (
        f"k = {fmt(iso.stiffness, 'N/m', 4)}, c = {fmt(iso.damping, 'N·s/m', 4)}, f<sub>n</sub> = {sig(fn, 4)} Hz "
        f"({60 * fn:.0f} rpm); running at r = {sig(iso.speed_hz / fn)}."
    )


def _verdict(T: float) -> str:
    return (
        f"isolation {sig(100 * (1 - T))} % ({sig(db(T))} dB)"
        if T < 1
        else f"amplified {sig(T)} times ({sig(db(T), plus=True)} dB)"
    )


def steady_note(iso: Isolator, T: float, shaking: int = MACHINE) -> str:
    """The readout at a steady speed whose T was measured (see the module)."""
    if shaking == FLOOR:
        return (
            f"Steady at {iso.rpm:.0f} rpm: the floor moves ±{fmt(FLOOR_AMPLITUDE, 'm')} and the machine "
            f"±{fmt(T * FLOOR_AMPLITUDE, 'm')}: T = {sig(T)}, {_verdict(T)}."
        )
    F = float(iso.force_at(iso.speed_hz))
    X = abs(complex(iso.response(np.array([iso.speed_hz]))[0][0])) * F
    return (
        f"At {iso.rpm:.0f} rpm the unbalance pulls with {fmt(F, 'N')}; {fmt(T * F, 'N')} reaches the floor: T = "
        f"{sig(T)}, {_verdict(T)}. The machine moves ±{fmt(X, 'm')}."
    )


# ------------------------------------------------------------------ the Transmissibility tab


@dataclass(frozen=True)
class TView:
    machine: Curve  # the points measured with the machine shaking: T against f/f_n
    floor: Curve  # with the floor shaking
    earlier: Curve  # points from before a change, against their own f/f_n
    isolator: Curve  # this isolator's theory (empty unless shown)
    family: tuple[Curve, ...]  # plain mounts at FAMILY's ζ (empty unless shown)
    speed: float  # the running speed over f_n (the drag line's)
    note: str


def t_view(state: IsolationTrial, points: Sequence[Point], theory: bool) -> TView:
    """The Transmissibility tab (see the module). `points` are every measured point, each keyed (isolator, shaking)."""
    iso = isolator(state)
    fn = iso.natural_hz
    now = key(state)
    mine: dict[int, list[Point]] = {MACHINE: [], FLOOR: []}
    earlier: list[tuple[float, float]] = []
    for p in points:
        assert isinstance(p.key, tuple), "a point's key is (the isolator, what shook)"
        k: Isolator = p.key[0]
        shaking: int = p.key[1]
        if k == now:
            mine[shaking].append(p)
        else:
            earlier.append((p.freq_hz / k.natural_hz, abs(p.response)))

    def curve(ps: list[Point]) -> Curve:
        ps = sorted(ps, key=lambda p: p.freq_hz)
        return Curve(np.array([p.freq_hz / fn for p in ps]), np.array([abs(p.response) for p in ps]))

    family: tuple[Curve, ...] = ()
    own = Curve.empty()
    if theory:
        f = RATIOS * fn
        f = f[np.abs(f / fn - 1) > 1e-9] if iso.zeta == 0 else f
        own = Curve(f / fn, iso.transmissibility(f))
        family = tuple(
            Curve(RATIOS, _finite(replace(iso, zeta=z, mount=Mount.VISCOUS, block=0.0).transmissibility(RATIOS * fn)))
            for z in FAMILY
        )
    n = len(mine[MACHINE]) + len(mine[FLOOR])
    if n == 0:
        note = (
            "Run the machine at a steady speed: once the mounts' own ring has died away, its T lands here. "
            "Sweep steps the speed across the resonance and dwells at each."
        )
    else:
        both = mine[MACHINE] and mine[FLOOR]
        note = f"{n} speed{'s' if n > 1 else ''} measured on these mounts" + (
            ": the machine's (●) and the floor's (◆) fall on one curve." if both else "."
        )
    if theory:
        note += " The theory: bold, this isolator; thin, plain mounts at ζ = 0, 0.1, 0.3 and 0.7."

    e = sorted(earlier)
    return TView(
        curve(mine[MACHINE]),
        curve(mine[FLOOR]),
        Curve(np.array([a for a, _ in e]), np.array([b for _, b in e])),
        own,
        family,
        iso.speed_hz / fn,
        note,
    )


def _finite(T: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(T), T, np.nan)


def sweep_speeds(state: IsolationTrial) -> tuple[float, ...]:
    """A sweep's speeds (rpm), from SWEEP_SPAN[0] to [1] × f<sub>n</sub>, log-spaced."""
    fn = isolator(state).natural_hz
    return tuple(float(60 * f) for f in np.geomspace(SWEEP_SPAN[0] * fn, SWEEP_SPAN[1] * fn, SWEEP_POINTS))


# ------------------------------------------------------------------ the Floor force tab


@dataclass(frozen=True)
class ForceView:
    built: Curve  # each whole turn's largest force on the floor (N) against its speed (rpm), or the machine's motion
    rigid: Curve  # what a rigid mount would pass (the unbalance's own force), or the floor's motion
    theory: Curve  # the steady amplitude through these mounts (empty unless shown)
    rpm_range: tuple[float, float]
    title: str  # the y axis
    unit: str
    names: tuple[str, str]  # the built curve's and the rigid's, for the legend
    note: str


def force_view(state: IsolationTrial, rpm: np.ndarray, peaks: np.ndarray, theory: bool) -> ForceView:
    """The Floor force tab (see the module): `rpm` and `peaks`, each turn's, from `live_isolation.by_speed`."""
    iso = isolator(state)
    top = max(1.25 * iso.rpm, float(rpm.max()) * 1.05 if rpm.size else 0.0, 2.5 * 60 * iso.natural_hz)
    grid = np.linspace(top / 400, top, 400)
    fr = grid / 60.0
    if iso.zeta == 0:
        keep = np.abs(fr / iso.natural_hz - 1) > 1e-6
        grid, fr = grid[keep], fr[keep]
    floor = state.shaking == FLOOR
    rigid = np.full(grid.size, FLOOR_AMPLITUDE) if floor else iso.force_at(fr)
    steady = Curve(grid, iso.transmissibility(fr) * rigid) if theory else Curve.empty()
    if rpm.size:
        i = int(np.argmax(peaks))
        unit = "m" if floor else "N"
        note = f"Largest so far: {fmt(float(peaks[i]), unit)} at {rpm[i]:.0f} rpm (f<sub>n</sub> is at {60 * iso.natural_hz:.0f} rpm)."
    else:
        note = "Run the machine up (or down) through its mounts' resonance: each turn's largest swing lands here."
    return ForceView(
        Curve(rpm, peaks),
        Curve(grid, rigid),
        steady,
        (0.0, top),
        "Machine's motion" if floor else "Force on the floor",
        "m" if floor else "N",
        ("the machine", "the floor's motion") if floor else ("on these mounts", "rigid mount (mₑeΩ²)"),
        note,
    )


# ------------------------------------------------------------------ the Design chart tab


@dataclass(frozen=True)
class Chart:
    levels: tuple[Curve, ...]  # the sag each of CHART_LEVELS needs against the speed (rpm)
    now: Curve  # this machine
    rpm_range: tuple[float, float]
    note: str


def design_note(iso: Isolator, target: float) -> str:
    """The sag undamped one-stage mounts need to isolate `target` (a fraction) at the machine's speed."""
    try:
        d = deflection_for(iso.speed_hz, target)
    except ValueError:
        return ""
    return (
        f"{sig(100 * target, 4)} % at {iso.rpm:.0f} rpm needs f<sub>n</sub> ≤ {sig(math.sqrt(G / d) / TWO_PI)} Hz: a sag "
        f"of {sig(1e3 * d)} mm or more (undamped, one stage)."
    )


def chart(iso: Isolator) -> Chart:
    rpm = 60.0 * CHART_SPEEDS
    T = iso.at_speed()[2]
    return Chart(
        levels=tuple(
            Curve(rpm, np.array([1e3 * deflection_for(f, level) for f in CHART_SPEEDS])) for level in CHART_LEVELS
        ),
        now=Curve.point(iso.rpm, 1e3 * iso.deflection),
        rpm_range=(float(rpm[0]), float(rpm[-1])),
        note=(
            f"This machine: {iso.rpm:.0f} rpm on {sig(1e3 * iso.deflection)} mm mounts, "
            + (f"isolating {sig(100 * (1 - T))} %" if T < 1 else f"amplified {sig(T)} times")
            + " with its damping and kind. "
            "Above a line, better isolated than it."
        ),
    )
