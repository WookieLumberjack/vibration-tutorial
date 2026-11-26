"""The Single oscillator's damping laws' panels (#478 Phase 3's pilot), without Qt: the Loop & loss tab, the
force phasors, and friction's readout.

The state is `sdof.SdofTrial`, its damper following a law (`model`) matched to its c at f<sub>n</sub>
(`sdof.matched`, `sdof.params`: the module says how each runs live). Each panel has a function here: `loop_view`
(the hysteresis loop the run draws, the damping force against x over its last cycle, beside the four laws' loops
at that frequency and amplitude, and the energy it encloses), `loss_view` (storage stiffness and loss factor
against frequency), `phasors` (the driven steady state's forces as rotating phasors, the spring's, the damper's and
the mass's adding to the drive's) and `decay_readout` (the readout after a pluck: friction's falls by the same
amount a cycle).

Working on it: Qt-free (the package's docstring). `tests/test_live_parts.py` checks the laws run as matched (the
Zener decay against `Matched.free_decay`, friction's stop against `coulomb_stop`), the loop's energy and the phasors
closing; `tests/test_sdof_page.py` the page.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from ..core.damping_models import MODELS, Model
from ..core.sdof import TWO_PI
from .plot import Curve
from .sdof import SHORT, SdofTrial, drive_freq, force_amplitude, matched
from .text import fmt
from .text import number as sig

__all__ = ["RATIOS", "ROLES", "SHORT", "decay_readout", "loop_view", "loss_view", "phasors"]

RATIOS = np.geomspace(0.05, 20.0, 400)  # frequency over f_n on the loss plots
ROLES = {Model.VISCOUS: "mode.0", Model.HYSTERETIC: "mode.1", Model.COULOMB: "mode.2", Model.ZENER: "mode.3"}


# ------------------------------------------------------------------ the loop the run draws


@dataclass(frozen=True)
class LoopView:
    live: Curve  # the damping force (or the whole element's) against x over the run's last cycle
    theory: tuple[tuple[Model, Curve], ...]  # each model's steady loop at that frequency and amplitude
    spring: Curve  # kx, dashed, when the spring's force is in
    energy: float  # J, the energy the live loop encloses
    note: str
    axis: str = "Damping force"  # what the loops plot: the damping's force, or the spring's with it


def element_force(t: np.ndarray, v: np.ndarray, f: np.ndarray, mass: float) -> np.ndarray:
    """What the spring and the damping element push back with, f − mẍ, from the record (ẍ by differences)."""
    if t.size < 3:
        return np.zeros(t.size)
    return f - mass * np.gradient(v, t)


def loop_view(
    trial: SdofTrial, t: np.ndarray, x: np.ndarray, v: np.ndarray, f: np.ndarray, period: float | None
) -> LoopView:
    """The last cycle's loop (`period`: the drive's, or the ringing's; None: nothing to draw yet)."""
    m = matched(trial)
    axis = "Damping force" if trial.damping_only else "Spring + damping force"
    empty = LoopView(Curve.empty(), (), Curve.empty(), 0.0, "Pluck the mass or drive it: its loop is drawn here.", axis)
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
    if not trial.damping_only:
        note += " With the spring's kx in, the loop leans along its dashed line: the lean is the stiffness, its width the damping."
    return LoopView(Curve(xc, shown), tuple(theory), spring, enclosed, note, axis)


# ------------------------------------------------------------------ stiffness and loss, the phasors, the FRF


@dataclass(frozen=True)
class LossView:
    storage: tuple[tuple[Model, Curve], ...]  # K′/k against f/f_n
    loss: tuple[tuple[Model, Curve], ...]  # η against f/f_n
    now: float | None  # the drive's f/f_n, while it drives


def loss_view(t: SdofTrial) -> LossView:
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


def phasors(t: SdofTrial, time: float) -> tuple[Phasor, ...]:
    """The steady state's forces at `time` as phasors (N), rotating at the drive's Ω: kX, then iΩc<sub>eq</sub>X (the
    model's loss), then −Ω²mX (the mass's), head to tail, which add up to the drive F. Empty while it doesn't drive.
    Friction's by its describing function (none where it would stick)."""
    if not t.drive:
        return ()
    m = matched(t)
    f = drive_freq(t)
    omega = TWO_PI * f
    force = force_amplitude(t)
    h = complex(m.receptance(t.model, np.array([f]), force)[0])
    if t.model is Model.COULOMB:  # magnitude only: put the phase where the describing function's stiffness does
        K = complex(m.dynamic_stiffness(Model.COULOMB, np.array([omega]), max(abs(h) * force, 1e-12))[0])
        h = abs(h) * np.exp(-1j * np.angle(K - omega * omega * t.mass)) if abs(h) > 0 else 0j
    X = h * force * np.exp(1j * omega * time)
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


def decay_readout(t: SdofTrial, ts: np.ndarray, x: np.ndarray, still: bool) -> str:
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
        text += f" Stuck {fmt(abs(float(x[-1])), 'm', largest='')} off center: the spring can't beat the friction."
    return text.strip()
