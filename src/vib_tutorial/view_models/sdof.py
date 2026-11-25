"""The Single oscillator's view model (#478 Phase 3's pilot): its state, and what each panel shows, without Qt.

The page's state is an `SdofTrial` (its trials, `practice_state()` and `load_setup` all use one, bound to the
controls by `gui/bindings.py`). The page runs a `live_mass.LiveMass` built from it (`params`, `drive`). Its damper
follows a law (`model`): viscous, c itself; or hysteretic, Coulomb friction or the Zener element, each matched to
c at f<sub>n</sub> as `core/damping_models.Matched` matches them (the same energy a cycle there as the viscous
damper c = 2ζ√(km); friction's at the amplitude X; the Zener element's storage stiffness k there too):

- **Hysteretic**: it has no causal model in time (a loss η at every frequency, negative ones too), so it runs as
  its stand-in, the viscous damper with its loss at the frequency it runs at, c = ηk/Ω: the drive's while it
  drives, f<sub>n</sub>'s when it rings free. A change of the drive's frequency is then a damper change, live.
- **Coulomb friction**: F<sub>f</sub> = πcω<sub>n</sub>X/4, the chain's stick–slip.
- **Zener**: k₀ and the arm k₁–c₁ (`core/arm_mass.py`).

Under another law than viscous, a *ghost*, the viscous mass with the same start and drive, runs beside it, in grey
(`ghost_params`). Each panel has a function here (the damping laws' own panels are `damping.py`'s):

- `info(state)`: the oscillator's f<sub>n</sub>, ζ and the rest, and its damping law's, under its boxes.
- `frf_view(state, current, earlier, theory)`: the Frequency response tab: the points the reader's dwells and
  sweeps have landed (`recorder`), as amplification and lag against frequency and as a Nyquist plot, those
  from before a change faded, and the theory's curves (the law's, and the viscous ghost's) and the half-power band
  once asked for.
- `poles_view(params)` and `pole_drag(...)`: the Poles tab (the viscous damper's poles), and what dragging a pole
  does to k and c (the mass kept): off the real axis the pair goes there; near it, a real pole, the other staying
  put (critical damping where two meet). The phase plane beside it is the live record's.
- `ring_down` (`live_mass`): the strip chart's readout after a pluck (friction's, `damping.decay_readout`).

Working on it: Qt-free (the package's docstring). `tests/test_sdof_page.py` checks the views and the page;
`tests/test_live_parts.py` the laws run as matched; `tests/test_tutorial_numbers.py` the trials' notes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from ..core.damping_models import Matched, Model
from ..core.practice import Placement, Prediction
from ..core.sdof import TWO_PI, Oscillator, frequency_grid, receptance
from .live_mass import FORCE, UNBALANCE, Drive, Params
from .plot import Curve
from .recorder import Point
from .text import fmt
from .text import number as sig

TAB_ENERGY, TAB_FRF, TAB_LOOP, TAB_POLES = range(4)
SHORT = {Model.VISCOUS: "Viscous", Model.HYSTERETIC: "Hysteretic", Model.COULOMB: "Coulomb", Model.ZENER: "Zener"}
RAYS = (0.05, 0.2, 0.5, 0.8)  # the damping ratios' rays on the pole plot
POLE_SNAP = 0.02  # a pole dragged this close to the real axis (of the plot's height) lands on it


@dataclass(frozen=True)
class SdofTrial:
    name: str
    note: str
    mass: float = 1.0
    stiffness: float = 400.0
    damping: float = 2.0  # the viscous damper c, which the other laws are matched to
    model: Model = Model.VISCOUS  # the damper's law
    friction_mm: float = 10.0  # friction is matched at this amplitude X
    relax: float = 1.0  # the Zener element's loss peak, × f_n
    ghost: bool = True  # under another law, the viscous mass beside it
    x0_mm: float = 0.0  # plucked from here and let go (0: starts at rest)
    drive: bool = False  # the drive switched on
    kind: int = FORCE
    amplitude: float = 1.0  # N, or the unbalance m_e e in kg·m
    freq_hz: float = 3.0
    tune: bool = False  # drive at f_n
    sweep: bool = False  # a stepped sweep, building the frequency response
    damping_only: bool = True  # the loop's force: the damping alone, or with the spring
    tab: int = TAB_ENERGY
    predict: Prediction | Placement | None = None  # asked once the reader has played (Things to try)


def viscous(t: SdofTrial) -> Params:
    """The oscillator with its viscous damper c: the page's under that law, the ghost's under the others."""
    return Params(t.mass, t.stiffness, t.damping)


def oscillator(t: SdofTrial) -> Oscillator:
    """The viscous oscillator: its f<sub>n</sub>, ζ and poles are the page's, whatever the law."""
    return viscous(t).oscillator()


def zeta(t: SdofTrial) -> float:
    return t.damping / (2.0 * math.sqrt(t.stiffness * t.mass))


def matched(t: SdofTrial) -> Matched:
    """The four laws matched to the damper c at f<sub>n</sub> (no damper: matched to a vanishing one)."""
    return Matched(t.mass, t.stiffness, max(zeta(t), 1e-9), 1e-3 * t.friction_mm, t.relax)


def drive_freq(t: SdofTrial) -> float:
    """The drive's frequency: f<sub>n</sub> for a tuned trial."""
    return round(oscillator(t).natural_hz, 6) if t.tune else t.freq_hz


def hysteretic_c(m: Matched, freq_hz: float | None) -> float:
    """The hysteretic loss's stand-in in time: c = ηk/Ω at the frequency it runs at (f<sub>n</sub> free)."""
    omega = TWO_PI * freq_hz if freq_hz else m.omega_n
    return m.eta * m.stiffness / omega


def params(t: SdofTrial, model: Model | None = None) -> Params:
    """The live mass under law `model` (the state's own unless given), matched as the module says."""
    model = t.model if model is None else model
    if model is Model.VISCOUS or t.damping <= 0:
        return viscous(t)
    m = matched(t)
    if model is Model.HYSTERETIC:
        return Params(t.mass, t.stiffness, hysteretic_c(m, drive_freq(t) if t.drive else None))
    if model is Model.COULOMB:
        return Params(t.mass, t.stiffness, 0.0, friction=m.friction)
    k0, k1, tau = m.zener()
    return Params(t.mass, k0, 0.0, arm_stiffness=k1, arm_damping=k1 * tau)


def ghosted(t: SdofTrial) -> bool:
    """Whether the viscous ghost runs beside it: asked for, under another law."""
    return t.ghost and t.model is not Model.VISCOUS


def key(t: SdofTrial) -> tuple:
    """The system the recorder's points belong to: the law and what it's matched with (not the hysteretic
    stand-in's c, which follows the drive's frequency)."""
    return (t.model, t.mass, t.stiffness, t.damping, t.friction_mm, t.relax)


def drive(t: SdofTrial) -> Drive:
    return Drive(t.drive, t.kind, t.amplitude, drive_freq(t))


def force_amplitude(t: SdofTrial) -> float:
    """The drive's force amplitude at its frequency: F, or m<sub>e</sub>eΩ² for the unbalance."""
    return drive(t).force


def info(t: SdofTrial) -> str:
    """Under the oscillator's boxes: its f<sub>n</sub> and ζ, then what its damping law is now."""
    o = oscillator(t)
    z = o.zeta
    parts = [
        f"f<sub>n</sub> = {sig(o.natural_hz, 4)} Hz, ζ = {sig(z, 4)} (c<sub>c</sub> = {sig(o.critical, 4)} N·s/m)."
    ]
    if t.model is not Model.VISCOUS and t.damping > 0:
        return " ".join([*parts, _law(t)])
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


def _law(t: SdofTrial) -> str:
    """What the damping law is, matched to c at f<sub>n</sub>."""
    m = matched(t)
    if t.model is Model.HYSTERETIC:
        c = hysteretic_c(m, drive_freq(t) if t.drive else None)
        at = f"the drive's {sig(drive_freq(t))} Hz" if t.drive else "f<sub>n</sub>, free"
        return f"Loss factor η = 2ζ = {sig(m.eta)}; in time, its stand-in c = ηk/Ω = {sig(c, 4)} N·s/m at {at}."
    if t.model is Model.COULOMB:
        return (
            f"Friction F<sub>f</sub> = πcω<sub>n</sub>X/4 = {fmt(m.friction, 'N')}, matched at X = "
            f"{sig(t.friction_mm)} mm; it sticks within F<sub>f</sub>/k = {fmt(m.friction / m.stiffness, 'm', largest='')}."
        )
    k0, k1, tau = m.zener()
    return (
        f"k₀ = {sig(k0, 4)} N/m beside an arm k₁ = {sig(k1, 4)} N/m in series with c₁ = {sig(k1 * tau, 4)} N·s/m "
        f"(τ<sub>r</sub> = {fmt(tau, 's')}): its loss peaks at {sig(t.relax)} f<sub>n</sub>."
    )


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
    ghost: Curve  # the viscous ghost's amplification, dashed, under another law (with the theory)


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


def _model_receptance(t: SdofTrial, freqs: np.ndarray) -> np.ndarray:
    """The law's H at `freqs` (friction's by its describing function, at the drive's force there: magnitude only)."""
    m = matched(t)
    if t.model is not Model.COULOMB:
        return m.receptance(t.model, freqs, 1.0)
    forces = [force_amplitude(replace(t, freq_hz=float(f), tune=False)) for f in freqs]
    return np.array(
        [complex(m.receptance(Model.COULOMB, np.array([f]), max(F, 1e-12))[0]) for f, F in zip(freqs, forces)]
    )


def frf_view(t: SdofTrial, current: Sequence[Point], earlier: Sequence[Point], theory: bool) -> FrfView:
    """The Frequency response tab (see the module)."""
    p, d, o = viscous(t), drive(t), oscillator(t)
    law = t.model is not Model.VISCOUS and t.damping > 0
    points, lags = _curves(p, d.kind, current)
    earlier_points, earlier_lags = _curves(p, d.kind, earlier)
    h_now = np.array([q.response for q in current])
    nyquist = Curve(h_now.real, h_now.imag) if current else Curve.empty()
    ghost = Curve.empty()
    if theory:
        freqs = frequency_grid(o)
        viscous_h = receptance(o, freqs)
        h = _model_receptance(t, freqs) if law else viscous_h
        curve = Curve(freqs, _amplification(p, d.kind, freqs, h))
        exact = t.model is not Model.COULOMB  # friction's describing function has no phase
        lag = Curve(freqs, _lag(h)) if exact else Curve.empty()
        nyq = Curve(h.real, h.imag) if exact else Curve.empty()
        band = o.half_power_hz() if d.kind == FORCE and not law else None
        if law:
            ghost = Curve(freqs, _amplification(p, d.kind, freqs, viscous_h))
    else:
        curve = lag = nyq = Curve.empty()
        band = None
    drive_point = drive_lag = Curve.empty()
    if d.on and not law:
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
        what = f" with {SHORT[t.model].lower()} damping" if law else " on this oscillator"
        note = f"{n} point{'s' if n != 1 else ''} measured{what}"
        note += f", {len(earlier)} faded from before a change." if earlier else "."
        if band is not None:
            lo, hi = band
            note += f" Half-power band {sig(lo)}–{sig(hi)} Hz: ζ ≈ Δf/2f<sub>n</sub> = {sig((hi - lo) / (hi + lo))}."
    if law and theory:
        note += " The dashed curve is the viscous ghost's."
        if t.model is Model.COULOMB:
            note += (
                " Friction's curve is its describing function's: none where it would stick, unbounded at "
                "f<sub>n</sub>, and no phase."
            )
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
        ghost,
    )


# ------------------------------------------------------------------ the poles


@dataclass(frozen=True)
class PolesView:
    poles: Curve
    rays: tuple[Curve, ...]  # ζ = RAYS from the origin, out to the arc
    arc: Curve  # |s| = ω_n
    note: str


def poles_view(p: Params, law: Model = Model.VISCOUS) -> PolesView:
    """The viscous oscillator's poles (under another `law`, the matched damper's, said so in the note)."""
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
    if law is not Model.VISCOUS:
        note = (
            f"These are the poles of the viscous damper c that the {SHORT[law].lower()} law is matched to: "
            "its own motion isn't a pair of poles like these (the phase plane below shows what it does). "
            "Dragging them still sets k and c. " + note
        )
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
