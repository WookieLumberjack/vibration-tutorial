"""Damping models' view model (#478 Phase 3's pilot, its second page): one live mass with a damping model to pick.

The page's state is a `DampingTrial` (its trials, `practice_state()` and `load_setup`). The four models are matched
to the oscillator at its natural frequency as `core/damping_models.Matched` matches them (the same energy a cycle
as a viscous damper of ratio ζ there; friction's at the amplitude X; the Zener element's storage stiffness k
there too), and each runs live as a `live_mass.LiveMass` (`params`):

- **Viscous**: the damper c.
- **Hysteretic**: it has no causal model in time (a loss η at every frequency, negative ones too), so it runs as
  its stand-in, the viscous damper with its loss at the frequency it runs at, c = ηk/Ω: the drive's while it
  drives, f<sub>n</sub>'s when it rings free. A change of the drive's frequency is then a damper change, live.
- **Coulomb friction**: F<sub>f</sub> = πcω<sub>n</sub>X/4, the chain's stick–slip.
- **Zener**: k₀ and the arm k₁–c₁ (`core/arm_mass.py`).

A *ghost*, the viscous mass with the same start and drive, runs beside it, faint (`ghost_params`), so the models
are compared as they run, as the old side-by-side decays were.

Each panel has a function here: `info` (what the model is now, under its box), `loop_view` (the hysteresis loop
the run draws, the damping force against x over its last cycle, beside the four models' loops at that
frequency and amplitude, and the energy it encloses), `loss_view` (storage stiffness and loss factor against
frequency), `phasors` (the driven steady state's forces as rotating phasors, the spring's, the damper's and
the mass's adding to the drive's) and `frf_view` (the points the reader's dwells and sweeps landed, over the
model's theory).

Working on it: Qt-free (the package's docstring). `tests/test_live_parts.py` checks the models run as matched (the
Zener decay against `Matched.free_decay`, friction's stop against `coulomb_stop`), the loop's energy and the phasors
closing; `tests/test_damping_page.py` the page.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from ..core.damping_models import MODELS, Matched, Model
from ..core.practice import Placement, Prediction
from ..core.sdof import TWO_PI
from .live_mass import Drive, Params
from .plot import Curve
from .recorder import Point
from .text import fmt
from .text import number as sig

TAB_LOOP, TAB_FRF, TAB_ENERGY = range(3)
RATIOS = np.geomspace(0.05, 20.0, 400)  # frequency over f_n on the loss plots
SHORT = {Model.VISCOUS: "Viscous", Model.HYSTERETIC: "Hysteretic", Model.COULOMB: "Coulomb", Model.ZENER: "Zener"}
ROLES = {Model.VISCOUS: "mode.0", Model.HYSTERETIC: "mode.1", Model.COULOMB: "mode.2", Model.ZENER: "mode.3"}


@dataclass(frozen=True)
class DampingTrial:
    name: str
    note: str
    model: Model = Model.VISCOUS
    mass: float = 1.0
    stiffness: float = 400.0
    zeta: float = 0.05
    amplitude_mm: float = 10.0  # friction is matched at this amplitude
    relax: float = 1.0  # the Zener element's loss peak, × f_n
    ghost: bool = True  # the viscous mass beside it
    x0_mm: float = 0.0  # plucked from here and let go (0: starts at rest)
    drive: bool = False
    force: float = 1.0
    freq_hz: float = 3.0
    tune: bool = False  # drive at f_n
    sweep: bool = False
    damping_only: bool = True  # the loop's force: the damping alone, or with the spring
    tab: int = TAB_LOOP
    predict: Prediction | Placement | None = None  # asked once the reader has handled it


def matched(t: DampingTrial) -> Matched:
    return Matched(t.mass, t.stiffness, t.zeta, 1e-3 * t.amplitude_mm, t.relax)


def drive_freq(t: DampingTrial) -> float:
    return round(matched(t).natural_hz, 6) if t.tune else t.freq_hz


def drive(t: DampingTrial) -> Drive:
    return Drive(t.drive, 0, t.force, drive_freq(t))


def hysteretic_c(m: Matched, freq_hz: float | None) -> float:
    """The hysteretic loss's stand-in in time: c = ηk/Ω at the frequency it runs at (f<sub>n</sub> free)."""
    omega = TWO_PI * freq_hz if freq_hz else m.omega_n
    return m.eta * m.stiffness / omega


def params(t: DampingTrial, model: Model | None = None) -> Params:
    """The live mass for model `model` (the trial's own unless given), matched as the module says."""
    model = t.model if model is None else model
    m = matched(t)
    if model is Model.VISCOUS:
        return Params(t.mass, t.stiffness, m.c)
    if model is Model.HYSTERETIC:
        return Params(t.mass, t.stiffness, hysteretic_c(m, drive_freq(t) if t.drive else None))
    if model is Model.COULOMB:
        return Params(t.mass, t.stiffness, 0.0, friction=m.friction)
    k0, k1, tau = m.zener()
    return Params(t.mass, k0, 0.0, arm_stiffness=k1, arm_damping=k1 * tau)


def ghost_params(t: DampingTrial) -> Params:
    return params(t, Model.VISCOUS)


def info(t: DampingTrial) -> str:
    m = matched(t)
    head = f"f<sub>n</sub> = {sig(m.natural_hz, 4)} Hz. "
    if t.model is Model.VISCOUS:
        return head + f"The damper c = 2ζ√(km) = {sig(m.c, 4)} N·s/m."
    if t.model is Model.HYSTERETIC:
        c = hysteretic_c(m, drive_freq(t) if t.drive else None)
        at = f"the drive's {sig(drive_freq(t))} Hz" if t.drive else "f<sub>n</sub>, free"
        return head + (
            f"Loss factor η = 2ζ = {sig(m.eta)}; in time, its stand-in c = ηk/Ω = {sig(c, 4)} N·s/m at {at}."
        )
    if t.model is Model.COULOMB:
        return head + (
            f"Friction F<sub>f</sub> = πcω<sub>n</sub>X/4 = {fmt(m.friction, 'N')}, matched at X = "
            f"{sig(t.amplitude_mm)} mm; it sticks within F<sub>f</sub>/k = {fmt(m.friction / m.stiffness, 'm', largest='')}."
        )
    k0, k1, tau = m.zener()
    return head + (
        f"k₀ = {sig(k0, 4)} N/m beside an arm k₁ = {sig(k1, 4)} N/m in series with c₁ = {sig(k1 * tau, 4)} N·s/m "
        f"(τ = {fmt(tau, 's')}): its loss peaks at {sig(t.relax)} f<sub>n</sub>."
    )


# ------------------------------------------------------------------ the loop the run draws


@dataclass(frozen=True)
class LoopView:
    live: Curve  # the damping force (or the whole element's) against x over the run's last cycle
    theory: tuple[tuple[Model, Curve], ...]  # each model's steady loop at that frequency and amplitude
    spring: Curve  # kx, dashed, when the spring's force is in
    energy: float  # J, the energy the live loop encloses
    note: str


def element_force(t: np.ndarray, v: np.ndarray, f: np.ndarray, mass: float) -> np.ndarray:
    """What the spring and the damping element push back with, f − mẍ, from the record (ẍ by differences)."""
    if t.size < 3:
        return np.zeros(t.size)
    return f - mass * np.gradient(v, t)


def loop_view(
    trial: DampingTrial, t: np.ndarray, x: np.ndarray, v: np.ndarray, f: np.ndarray, period: float | None
) -> LoopView:
    """The last cycle's loop (`period`: the drive's, or the ringing's; None: nothing to draw yet)."""
    m = matched(trial)
    empty = LoopView(Curve.empty(), (), Curve.empty(), 0.0, "Pluck the mass or drive it: its loop is drawn here.")
    if period is None or t.size < 8 or t[-1] - t[0] < period:
        return empty
    i0 = int(np.searchsorted(t, t[-2] - period))
    total = element_force(t, v, f, trial.mass)[i0:-1]  # ẍ by central differences, so the window's ends left out:
    xc = x[i0:-1]  # a whole period to the last sample but one
    if np.ptp(xc) < 1e-9:
        return replace(empty, note="It's still: no cycle, no loop.")
    damping = total - trial.stiffness * xc
    shown = damping if trial.damping_only else total
    enclosed = float(np.trapezoid(damping, xc))  # ∮F dx round the cycle: the work done against the damping
    amplitude = float(np.abs(xc).max())
    omega = TWO_PI / period
    theory = []
    if amplitude > 0:
        for model in MODELS:
            xs, whole, part = m.loop(model, omega, 200, amplitude)
            theory.append((model, Curve(xs, part if trial.damping_only else whole)))
    spring = (
        Curve.empty()
        if trial.damping_only
        else Curve(np.array([-amplitude, amplitude]), trial.stiffness * np.array([-amplitude, amplitude]))
    )
    viscous = math.pi * m.c * omega * amplitude**2
    note = (
        f"The last cycle, ±{fmt(amplitude, 'm', largest='')} at {sig(1 / period)} Hz, encloses "
        f"{fmt(abs(enclosed), 'J', smallest='µ')}: what the damping took out that cycle (a viscous damper's "
        f"πcΩX² there is {fmt(viscous, 'J', smallest='µ')})."
    )
    return LoopView(Curve(xc, shown), tuple(theory), spring, enclosed, note)


# ------------------------------------------------------------------ stiffness and loss, the phasors, the FRF


@dataclass(frozen=True)
class LossView:
    storage: tuple[tuple[Model, Curve], ...]  # K′/k against f/f_n
    loss: tuple[tuple[Model, Curve], ...]  # η against f/f_n
    now: float | None  # the drive's f/f_n, while it drives


def loss_view(t: DampingTrial) -> LossView:
    m = matched(t)
    w = RATIOS * m.omega_n
    storage, loss = [], []
    for model in MODELS:
        K = m.dynamic_stiffness(model, w)
        storage.append((model, Curve(RATIOS, K.real / m.stiffness)))
        loss.append((model, Curve(RATIOS, K.imag / K.real)))
    return LossView(tuple(storage), tuple(loss), drive_freq(t) / m.natural_hz if t.drive else None)


@dataclass(frozen=True)
class Phasor:
    name: str
    start: tuple[float, float]  # N, (real, imaginary), head to tail: the spring's, the damper's, the mass's
    end: tuple[float, float]  # ending at the drive's
    role: str


def _phasor(name: str, start: complex, end: complex, role: str) -> Phasor:
    return Phasor(name, (start.real, start.imag), (end.real, end.imag), role)


def phasors(t: DampingTrial, time: float) -> tuple[Phasor, ...]:
    """The steady state's forces at `time` as phasors (N), rotating at the drive's Ω: kX, then iΩc<sub>eq</sub>X (the
    model's loss), then −Ω²mX (the mass's), head to tail, which add up to the drive F. Empty while it doesn't drive.
    Friction's by its describing function (none where it would stick)."""
    if not t.drive:
        return ()
    m = matched(t)
    f = drive_freq(t)
    omega = TWO_PI * f
    h = complex(m.receptance(t.model, np.array([f]), t.force)[0])
    if t.model is Model.COULOMB:  # magnitude only: put the phase where the describing function's stiffness does
        K = complex(m.dynamic_stiffness(Model.COULOMB, np.array([omega]), max(abs(h) * t.force, 1e-12))[0])
        h = abs(h) * np.exp(-1j * np.angle(K - omega * omega * t.mass)) if abs(h) > 0 else 0j
    X = h * t.force * np.exp(1j * omega * time)
    K = complex(m.dynamic_stiffness(t.model, np.array([omega]), max(abs(X), 1e-12))[0])
    spring = K.real * X
    loss = 1j * K.imag * X
    inertia = -omega * omega * t.mass * X
    return (
        _phasor("spring", 0j, spring, "energy.potential"),
        _phasor("damping", spring, spring + loss, "energy.dissipated"),
        _phasor("mass", spring + loss, spring + loss + inertia, "energy.kinetic"),
        _phasor("drive", 0j, spring + loss + inertia, "force"),
    )


@dataclass(frozen=True)
class FrfView:
    points: Curve  # |X|k/F against frequency, landed on this system
    earlier: Curve
    theory: Curve  # the model's (empty unless asked for)
    ghost: Curve  # the viscous one's
    natural_hz: float
    note: str


def _amplification(k: float, points: Sequence[Point]) -> Curve:
    if not points:
        return Curve.empty()
    return Curve(np.array([p.freq_hz for p in points]), k * np.abs(np.array([p.response for p in points])))


def frf_view(t: DampingTrial, current: Sequence[Point], earlier: Sequence[Point], theory: bool) -> FrfView:
    m = matched(t)
    if theory:
        freqs = RATIOS * m.natural_hz
        model = Curve(freqs, m.stiffness * np.abs(m.receptance(t.model, freqs, t.force)))
        ghost = Curve(freqs, m.stiffness * np.abs(m.receptance(Model.VISCOUS, freqs, t.force)))
    else:
        model = ghost = Curve.empty()
    n = len(current)
    note = (
        f"{n} point{'s' if n != 1 else ''} measured with {SHORT[t.model].lower()} damping."
        if n
        else "Nothing measured yet: drive it and let it settle, or press Sweep."
    )
    if t.model is Model.COULOMB and theory:
        note += " Friction's curve is its describing function's: none where it would stick, unbounded at f<sub>n</sub>."
    return FrfView(
        _amplification(t.stiffness, current), _amplification(t.stiffness, earlier), model, ghost, m.natural_hz, note
    )


def decay_readout(t: DampingTrial, ts: np.ndarray, x: np.ndarray, still: bool) -> str:
    """The readout under the time plot after a pluck (the samples since it): friction's swing falls by the same
    amount each cycle and sticks; the others' by the same fraction (`live_mass.ring_down`)."""
    from .live_mass import ring_down

    if t.model is not Model.COULOMB:
        decay = ring_down(ts, x)
        return decay.text if decay is not None else ""
    from ..core.sdof import peaks

    _, heights = peaks(ts, x)
    m = matched(t)
    drop = 4.0 * m.friction / m.stiffness
    if heights.size >= 2:
        measured = float(np.mean(-np.diff(heights)))
        text = (
            f"Each peak is {fmt(measured, 'm', largest='')} lower than the one before: 4F<sub>f</sub>/k = "
            f"{fmt(drop, 'm', largest='')}, the same amount each cycle."
        )
    else:
        text = ""
    if still and ts.size:
        text += f" Stuck {fmt(abs(float(x[-1])), 'm', largest='')} off centre: the spring can't beat the friction."
    return text.strip()
