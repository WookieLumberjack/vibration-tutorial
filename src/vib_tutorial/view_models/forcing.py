"""Forcing an oscillator's view model (#478 Phase 4): its state, and what each panel shows, without Qt.

The page's state is a `ForcingTrial` (its trials, `practice_state()` and `load_setup` all use one): the oscillator
(m, k, c), the one-off force the *Apply* button fires (its kind, size and length), the periodic force the *Wave*
button switches on (its shape, size, frequency or a tune to f<sub>n</sub>/n, and duty), how many harmonics the
Harmonics tab shows, the tab, whether the theory is drawn, and how a trial sets it going (`start`). The page runs
it live (`live_forcing.LiveForcing`), and each panel has a function here:

- `info(osc)`: the oscillator's f<sub>n</sub>, ζ, ω<sub>d</sub> and T<sub>n</sub>, under its boxes.
- `drive_note(live)`: what acts on the mass now, under the time plots.
- `conv_view(conv, osc)`: the **Convolution** tab: the force over the record, h flipped and slid to the instant
  (scaled to the force's plot), their product, and the area against the run's own motion.
- `pulse_view(points, zeta, theory)`: the **Pulse spectrum** tab, built by hand: each pulse's landed peak, its kind's
  markers, against its length over the natural period; the theory's curves (`transient.peak_ratio`) on request.
- `harmonics_view(h, osc, forcing, count, theory)`: the **Harmonics** tab: the force's and the response's line
  spectra, measured from the steady run, with |H| and the series' lines on request, and each harmonic's share of
  the swing's power.

Working on it: Qt-free (the package's docstring); `tests/test_view_models.py` checks the panels and ties the
readouts to the numbers the notes quote, as printed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core.periodic import Forcing, Wave, response_coefficients
from ..core.practice import Placement, Prediction
from ..core.sdof import Oscillator, receptance
from ..core.transient import Pulse, PulseKind, peak_ratio
from .live_forcing import Convolution, Harmonics, LiveForcing, PulsePeak
from .plot import Curve
from .text import fixed, fmt
from .text import number as sig

KINDS = tuple(PulseKind)
WAVES = tuple(Wave)
TAB_CONV, TAB_PULSE, TAB_HARM = range(3)
APPLY, WAVE = range(2)  # a trial's start: fire the one-off force from rest, or switch the wave on from rest
SPECTRUM_KINDS = (PulseKind.RECTANGLE, PulseKind.HALF_SINE, PulseKind.RAMP)  # the ones with a length
RATIOS = np.geomspace(0.05, 4.0, 160)  # a pulse's length over the natural period, on the Pulse spectrum
SHARE_SHOWN = 0.0005  # the smallest share of the swing's power the Harmonics note lists (to one decimal: 0.1 %)


@dataclass(frozen=True)
class ForcingTrial:
    name: str
    note: str
    mass: float = 1.0
    stiffness: float = 400.0
    damping: float = 2.0
    kind: PulseKind = PulseKind.STEP  # the one-off force Apply fires
    amplitude: float = 1.0  # N, or for a hit its impulse, N·s
    duration: float = 0.1  # s: a pulse's length, a ramp's rise time
    wave: Wave = Wave.SQUARE  # the periodic force Wave switches on
    wave_amplitude: float = 1.0  # N, or for hammer blows each blow's impulse, N·s
    freq_hz: float = 1.0
    duty: float = 0.25  # the rectangular pulses' fraction of the period
    harmonics: int = 5  # shown on the Harmonics tab
    tune: int = 0  # set the wave's frequency to f_n / tune (0: freq_hz as it is)
    tab: int = TAB_CONV
    theory: bool = False  # the theory drawn over what's measured
    start: int | None = None  # how a trial sets it going (`APPLY`, `WAVE`), None: at rest
    predict: Prediction | Placement | None = None  # asked once the reader has played (Things to try)


def oscillator(t: ForcingTrial) -> Oscillator:
    return Oscillator(t.mass, t.stiffness, t.damping)


def pulse(t: ForcingTrial) -> Pulse:
    return Pulse(t.kind, t.amplitude, t.duration)


def wave_hz(t: ForcingTrial) -> float:
    """The wave's frequency: the box's, or f<sub>n</sub>/tune (rounded as the box shows it)."""
    return round(oscillator(t).natural_hz / t.tune, 4) if t.tune else t.freq_hz


def forcing(t: ForcingTrial) -> Forcing:
    return Forcing(t.wave, t.wave_amplitude, wave_hz(t), t.duty)


def info(osc: Oscillator) -> str:
    """The oscillator under its boxes (the worked example's first step)."""
    z = osc.zeta
    damped = f", ω<sub>d</sub> = {sig(osc.omega_d, 4)} rad/s" if z < 1 else " (overdamped: no ω<sub>d</sub>)"
    return (
        f"f<sub>n</sub> = {sig(osc.natural_hz)} Hz, T<sub>n</sub> = {sig(osc.period)} s, ζ = {sig(z)}{damped}; the "
        f"static deflection under 1 N is {fmt(osc.static, 'm')}."
    )


def drive_note(live: LiveForcing) -> str:
    """What acts on the mass now, and what's held."""
    if live.held_at is not None:
        return f"Held at x = {fmt(live.held_at, 'm', largest='')}: let go to pluck it."
    acting = [p for p in live.pushes if p.finish > live.t]
    parts = []
    for p in acting:
        if p.pulse.kind is PulseKind.IMPULSE:
            continue
        name = {PulseKind.STEP: "a push", PulseKind.RAMP: "a ramp", PulseKind.RECTANGLE: "a rectangular pulse",
                PulseKind.HALF_SINE: "a half-sine pulse"}[p.pulse.kind]  # fmt: skip
        held = (
            " (held: Let go ends it)" if p.end == math.inf and p.pulse.kind in (PulseKind.STEP, PulseKind.RAMP) else ""
        )
        parts.append(f"{name} of {fmt(p.pulse.amplitude, 'N')}{held}")
    if live.wave is not None:
        w = live.wave
        size = fmt(w.amplitude, "N·s") + " blows" if w.wave is Wave.IMPULSES else "±" + fmt(w.amplitude, "N")
        parts.append(f"{w.wave.value.lower()} {size} at {sig(w.freq_hz)} Hz")
    if not parts:
        return "Nothing acting: Apply a force or switch on a Wave (or pull the mass and let go)."
    return "Acting: " + "; ".join(parts) + "."


# ------------------------------------------------------------------ the Convolution tab


@dataclass(frozen=True)
class ConvView:
    force: Curve  # F(τ) against τ, since the integral's start
    h: Curve  # h(t − τ), scaled to the force's plot (`h_scale` N a m/N·s)
    product: Curve  # F(τ) h(t − τ), m/s
    hits: Curve  # the impulses' times, at their sizes' heights (N·s, drawn on the force's axis as marks)
    at: float  # the instant (s)
    note: str


def conv_view(conv: Convolution | None, osc: Oscillator) -> ConvView:
    if conv is None:
        e = Curve.empty()
        return ConvView(e, e, e, e, 0.0, "Apply a force: the integral is drawn from the run's own record.")
    big_f = float(np.max(np.abs(conv.force))) if conv.force.size else 0.0
    big_f = max(big_f, max((abs(i) for _, i in conv.hits), default=0.0) * osc.omega_n)
    big_h = float(np.max(np.abs(conv.h))) or 1.0
    scale = (big_f or 1.0) / big_h
    after = conv.at - conv.since
    note = (
        f"At {sig(after, 3)} s after the integral's start: ∫F h dτ"
        + (" plus the hits' I h" if conv.hits else "")
        + f" = {fmt(conv.area, 'm', largest='')}; the run: {fmt(conv.x, 'm', largest='')}."
    )
    hits = Curve(np.array([t for t, _ in conv.hits]), np.array([i * osc.omega_n for _, i in conv.hits]))
    return ConvView(
        Curve(conv.tau, conv.force), Curve(conv.tau, conv.h * scale), Curve(conv.tau, conv.product), hits, conv.at, note
    )


# ------------------------------------------------------------------ the Pulse spectrum tab


@dataclass(frozen=True)
class PulseView:
    points: dict[str, Curve]  # by the kind's name: each landed peak against its length over T_n
    theory: dict[str, Curve]  # by the kind's name, on request
    note: str


def pulse_view(points: list[PulsePeak], zeta: float, theory: bool) -> PulseView:
    mine = {k.name: [p for p in points if p.kind is k and abs(p.zeta - zeta) < 1e-9] for k in SPECTRUM_KINDS}
    curves = {name: Curve(np.array([p.ratio for p in ps]), np.array([p.peak for p in ps])) for name, ps in mine.items()}
    lines = {k.name: Curve(RATIOS, peak_ratio(zeta, k, RATIOS)) for k in SPECTRUM_KINDS} if theory else {}
    n = sum(len(ps) for ps in mine.values())
    if n == 0:
        note = (
            "Apply a rectangular, half-sine or ramp pulse with the mass at rest: a period after it ends, its peak "
            "lands here against its length over the natural period."
        )
    else:
        last = points[-1]
        note = (
            f"{n} pulse{'s' if n > 1 else ''} at ζ = {sig(zeta)}. The last: a {last.kind.value.lower()} "
            f"{sig(last.ratio)} periods long peaked at {sig(last.peak)} times its static deflection."
        )
    return PulseView(curves, lines, note)


# ------------------------------------------------------------------ the Harmonics tab


@dataclass(frozen=True)
class HarmonicsView:
    force: Curve  # each harmonic's height |C_j| against its frequency (Hz): the measured force
    response: Curve  # |X_j| (m)
    force_series: Curve  # the series' own |C_j|, on request
    response_series: Curve  # H(jω₀) C_j, on request
    receptance: Curve  # |H| against frequency, scaled to the response's largest line, on request
    natural_hz: float
    note: str


def harmonics_view(h: Harmonics | None, osc: Oscillator, wave: Forcing, count: int, theory: bool) -> HarmonicsView:
    j = np.arange(count + 1)
    f = j * wave.freq_hz
    C = wave.coefficients(count)
    X = response_coefficients(osc, wave, count)
    e = Curve.empty()
    fs, xs, rc = (Curve(f, np.abs(C)), Curve(f, np.abs(X)), e) if theory else (e, e, e)
    if theory:
        grid = np.linspace(0.0, max(f[-1], 1.5 * osc.natural_hz) * 1.05, 400)
        H = np.abs(receptance(osc, grid))
        big = float(np.max(np.abs(X))) or 1.0
        rc = Curve(grid, H * big / float(np.max(H)))
    if h is None:
        return HarmonicsView(e, e, fs, xs, rc, osc.natural_hz, "Switch on a Wave: its harmonics land here once steady.")
    if not h.steady:
        note = "Waiting for the start's free vibration to die away (a few of 1/ζω<sub>n</sub>)…"
        return HarmonicsView(e, e, fs, xs, rc, osc.natural_hz, note)
    power = np.abs(h.response[1:]) ** 2
    total = float(power.sum()) or 1.0
    order = np.argsort(power)[::-1]
    shares = [
        f"{_ordinal(int(k) + 1)} {fixed(100 * power[k] / total, 1)} %" for k in order if power[k] / total >= SHARE_SHOWN
    ]
    note = "The swing's power by harmonic: " + ", ".join(shares) + "."
    return HarmonicsView(Curve(f, np.abs(h.force)), Curve(f, np.abs(h.response)), fs, xs, rc, osc.natural_hz, note)


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def stems(x: np.ndarray, y: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray]:
    """Vertical lines from `floor` up to each y, as pairs for connect="pairs"."""
    return np.repeat(x, 2), np.column_stack([np.full(len(y), floor), y]).ravel()
