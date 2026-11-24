"""The Single oscillator's view model (#478 Phase 3's pilot): its state, and what each panel shows, without Qt.

The page's state is an `SdofTrial` (its trials, `practice_state()` and `load_setup` all use one, bound to the
controls by `gui/bindings.py`). The page runs a `live_mass.LiveMass` built from it (`params`, `drive`), and each
panel has a function here:

- `info(params)`: the oscillator's f<sub>n</sub>, ζ and the rest, under its boxes.
- `frf_view(params, drive, current, earlier, theory)`: the Frequency response tab: the points the reader's dwells
  and sweeps have landed (`recorder`), as amplification and lag against frequency and as a Nyquist plot, those
  from before a change faded, and the theory's curves and the half-power band once asked for.
- `poles_view(params)` and `pole_drag(...)`: the Poles tab, and what dragging a pole does to k and c (the mass
  kept): off the real axis the pair goes there; near it, a real pole, the other staying put (critical damping where
  two meet). The phase plane beside it is the live record's.
- `ring_down` (`live_mass`): the strip chart's readout after a pluck.

Working on it: Qt-free (the package's docstring). `tests/test_sdof_page.py` checks the views and the page;
`tests/test_tutorial_numbers.py` the trials' notes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..core.practice import Placement, Prediction
from ..core.sdof import TWO_PI, Oscillator, frequency_grid, receptance
from .live_mass import FORCE, UNBALANCE, Drive, Params
from .plot import Curve
from .recorder import Point
from .text import number as sig

TAB_FRF, TAB_ENERGY, TAB_POLES = range(3)
RAYS = (0.05, 0.2, 0.5, 0.8)  # the damping ratios' rays on the pole plot
POLE_SNAP = 0.02  # a pole dragged this close to the real axis (of the plot's height) lands on it


@dataclass(frozen=True)
class SdofTrial:
    name: str
    note: str
    mass: float = 1.0
    stiffness: float = 400.0
    damping: float = 2.0
    x0_mm: float = 0.0  # plucked from here and let go (0: starts at rest)
    drive: bool = False  # the drive switched on
    kind: int = FORCE
    amplitude: float = 1.0  # N, or the unbalance m_e e in kg·m
    freq_hz: float = 3.0
    tune: bool = False  # drive at f_n
    sweep: bool = False  # a stepped sweep, building the frequency response
    tab: int = TAB_FRF
    predict: Prediction | Placement | None = None  # asked once the reader has played (Things to try)


def params(t: SdofTrial) -> Params:
    return Params(t.mass, t.stiffness, t.damping)


def drive_freq(t: SdofTrial) -> float:
    """The drive's frequency: f<sub>n</sub> for a tuned trial."""
    return round(params(t).oscillator().natural_hz, 6) if t.tune else t.freq_hz


def drive(t: SdofTrial) -> Drive:
    return Drive(t.drive, t.kind, t.amplitude, drive_freq(t))


def force_amplitude(t: SdofTrial) -> float:
    """The drive's force amplitude at its frequency: F, or m<sub>e</sub>eΩ² for the unbalance."""
    return drive(t).force


def info(p: Params) -> str:
    o = p.oscillator()
    z = o.zeta
    parts = [
        f"f<sub>n</sub> = {sig(o.natural_hz, 4)} Hz, ζ = {sig(z, 4)} (c<sub>c</sub> = {sig(o.critical, 4)} N·s/m)."
    ]
    if 0 < z < 1:
        parts.append(
            f"f<sub>d</sub> = {sig(o.damped_hz, 4)} Hz, τ = {sig(o.time_constant)} s, δ = {sig(o.log_decrement)}, "
            f"Q = {sig(1 / (2 * z))}."
        )
    elif z == 0:
        parts.append("Undamped: it rings for ever.")
    else:
        what = "Critically damped" if abs(z - 1) < 1e-6 else "Overdamped"
        parts.append(f"{what}: no oscillation; slowest time constant {sig(o.time_constant)} s.")
    return " ".join(parts)


# ------------------------------------------------------------------ the frequency response, built by hand


@dataclass(frozen=True)
class FrfView:
    points: Curve  # amplification against frequency: the points landed on this system
    lags: Curve  # their lag (°)
    earlier: Curve  # those from before a change, faded
    earlier_lags: Curve
    theory: Curve  # the theory's curves (empty unless asked for)
    theory_lag: Curve
    band: tuple[float, float] | None  # the half-power band, with the theory
    natural_hz: float
    drive: Curve  # where the drive is now, from the theory (empty while it's off)
    drive_lag: Curve
    nyquist: Curve  # the points as Re H, Im H (m/N)
    nyquist_theory: Curve
    amplification: str  # the left axis's name
    note: str


def _amplification(p: Params, kind: int, freqs: np.ndarray, h: np.ndarray) -> np.ndarray:
    """|X|k/F for a force; m|X|/m<sub>e</sub>e for an unbalance (X = H m<sub>e</sub>eΩ²)."""
    if kind == UNBALANCE:
        return p.mass * np.abs(h) * (TWO_PI * freqs) ** 2
    return p.stiffness * np.abs(h)


def _lag(h: np.ndarray) -> np.ndarray:
    return -np.degrees(np.angle(h)) % 360.0


def _curves(p: Params, kind: int, points: Sequence[Point]) -> tuple[Curve, Curve]:
    if not points:
        return Curve.empty(), Curve.empty()
    f = np.array([q.freq_hz for q in points])
    h = np.array([q.response for q in points])
    return Curve(f, _amplification(p, kind, f, h)), Curve(f, _lag(h))


def frf_view(p: Params, d: Drive, current: Sequence[Point], earlier: Sequence[Point], theory: bool) -> FrfView:
    """The Frequency response tab (see the module)."""
    o = p.oscillator()
    points, lags = _curves(p, d.kind, current)
    earlier_points, earlier_lags = _curves(p, d.kind, earlier)
    h_now = np.array([q.response for q in current])
    nyquist = Curve(h_now.real, h_now.imag) if current else Curve.empty()
    if theory:
        freqs = frequency_grid(o)
        h = receptance(o, freqs)
        curve, lag = Curve(freqs, _amplification(p, d.kind, freqs, h)), Curve(freqs, _lag(h))
        nyq = Curve(h.real, h.imag)
        band = o.half_power_hz() if d.kind == FORCE else None
    else:
        curve = lag = nyq = Curve.empty()
        band = None
    drive_point = drive_lag = Curve.empty()
    if d.on:
        fd = np.array([d.freq_hz])
        hd = receptance(o, fd)
        drive_point, drive_lag = Curve(fd, _amplification(p, d.kind, fd, hd)), Curve(fd, _lag(hd))
    name = "Amplification m|X|/mₑe" if d.kind == UNBALANCE else "Amplification |X|k/F"
    n = len(current)
    if n == 0:
        note = (
            "Nothing measured yet. Switch the drive on and let it run: once the swing is steady, a point lands here "
            "at the drive's frequency. Change the frequency and wait again, or press Sweep."
        )
    else:
        note = f"{n} point{'s' if n != 1 else ''} measured on this oscillator"
        note += f", {len(earlier)} faded from before a change." if earlier else "."
        if band is not None:
            lo, hi = band
            note += f" Half-power band {sig(lo)}–{sig(hi)} Hz: ζ ≈ Δf/2f<sub>n</sub> = {sig((hi - lo) / (hi + lo))}."
    return FrfView(
        points,
        lags,
        earlier_points,
        earlier_lags,
        curve,
        lag,
        band,
        o.natural_hz,
        drive_point,
        drive_lag,
        nyquist,
        nyq,
        name,
        note,
    )


# ------------------------------------------------------------------ the poles


@dataclass(frozen=True)
class PolesView:
    poles: Curve
    rays: tuple[Curve, ...]  # ζ = RAYS from the origin, out to the arc
    arc: Curve  # |s| = ω_n
    note: str


def poles_view(p: Params) -> PolesView:
    o = p.oscillator()
    s = o.poles()
    wn = o.omega_n
    reach = 1.1 * max(wn, float(np.abs(s).max()))
    rays = tuple(Curve(np.array([0.0, -reach * z]), np.array([0.0, reach * math.sqrt(1 - z * z)])) for z in RAYS)
    angle = np.linspace(math.pi / 2, 3 * math.pi / 2, 91)
    arc = Curve(wn * np.cos(angle), wn * np.sin(angle))
    if o.zeta < 1:
        note = (
            f"s = −{sig(-s[0].real)} ± {sig(abs(s[0].imag))}i s⁻¹: the pair sits ω<sub>n</sub> = {sig(wn)} rad/s "
            f"from the origin, at sin⁻¹ζ = {sig(math.degrees(math.asin(o.zeta)))}° from the imaginary axis."
        )
    else:
        note = f"Two real poles, at {sig(s[0].real)} and {sig(s[1].real)} s⁻¹: two decays, no oscillation."
    return PolesView(Curve(s.real, s.imag), rays, arc, note)


def pole_pair(mass: float, sigma: float, w: float) -> tuple[float, float]:
    """k and c for the poles σ ± iw, the mass kept."""
    return mass * (sigma * sigma + w * w), -2.0 * sigma * mass


def pole_drag(
    p: Params,
    x: float,
    y: float,
    held: complex,
    spans: tuple[float, float],
) -> tuple[float, float]:
    """k and c for a pole dragged to x + iy (see the module): `held` is the other pole as the drag began, `spans`
    the plot's (width, height), which set how near the axis snaps."""
    width, height = spans
    x = min(x, 0.0)
    if abs(y) > POLE_SNAP * height:
        return pole_pair(p.mass, x, abs(y))
    tiny = 1e-3 * width
    x = min(x, -tiny)
    other = held.real if abs(held.imag) <= POLE_SNAP * height else x
    other = min(other, -tiny)
    return p.mass * x * other, -p.mass * (x + other)


def phase_plane(x: np.ndarray, v: np.ndarray, omega_n: float) -> Curve:
    """The live path in the phase plane: velocity over ω<sub>n</sub> against displacement."""
    return Curve(x, v / omega_n)


def oscillator(t: SdofTrial) -> Oscillator:
    return params(t).oscillator()
