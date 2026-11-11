"""Virtual modal test page: measure the chain's FRF as in a lab and compare it with the exact one.

Left: the test (excitation, acquisition, noise, processing). Centre: the
latest recorded block and force spectrum above the estimated FRF and its
coherence. Right: a check of the settings against the chain's modes, and
theory. The data are measured a few blocks per frame, so the average can be
watched settling, or at a playback speed that draws each record as it is
recorded; changing only noise or processing reuses the same data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core import (
    Acquisition,
    ChainSystem,
    Estimate,
    Estimator,
    Excitation,
    FrfEstimator,
    MeasurementSettings,
    ModalResult,
    Processing,
    Response,
    Window,
    frf,
    friction_frf,
    from_receptance,
    impact_spectrum,
    modal_analysis,
    to_receptance,
    transfer,
)
from ..core.identification import (
    Identification,
    Method,
    Stabilization,
    auto_select,
    band_mask,
    circle_fit,
    lscf,
    lsfd,
    mac_matrix,
    match_modes,
    mode_indicator,
    peak_picking,
)
from ..core.measurement import AA_CUTOFF, FORCE_RMS, IMPACT_PEAK, SINE_AMPLITUDE
from ..core.practice import Problem, Question, Requirement, at_least, at_most, number, pick
from . import frame_stats, settle_plots
from .axes import log_axes
from .layouts import ColumnScroll, form_layout
from .lesson import TrialCombo, trials_html
from .links import load_notes, page_link
from .modal_extraction import EXTRACTION_THEORY_HTML, ExtractionControls, ResultsView, StabilizationPlot
from .panels import spin
from .style import MAX_DOF, colors
from .theming import add_legend, mute, settings

FRAME_MS = 30
STEP_BUDGET_MS = 25  # measuring per frame, so the page stays responsive
# Playback: how fast each record is drawn, as a multiple of real time (None: all at once).
SPEEDS = [
    ("Instant", None),
    ("100× real time", 100.0),
    ("30× real time", 30.0),
    ("10× real time", 10.0),
    ("3× real time", 3.0),
    ("Real time", 1.0),
]
BLOCK_SIZES = [256, 512, 1024, 2048, 4096, 8192]


@dataclass(frozen=True)
class ModalTestTrial:
    """A test of the uniform 4-mass chain (with friction if given), or of the chain `masses`, `springs` and
    `dampers` give: what each control is set to."""

    name: str
    note: str
    input: int = 3  # the mass the force is on
    excitation: Excitation = Excitation.IMPACT
    response: Response = Response.DISPLACEMENT
    window: Window | None = None  # None: the excitation's usual one
    block: int = 1024
    fs: float | None = None  # None: Auto
    anti_alias: bool = True
    force_noise: float = 0.0  # %
    response_noise: float = 0.0
    method: Method = Method.LSCF
    friction: float = 0.0  # N on every mass
    tab: int = 1  # Setup, Modes, Stabilization
    masses: tuple[float, ...] = ()  # empty: the uniform 4-mass chain
    springs: tuple[float, ...] = ()
    dampers: tuple[float, ...] = ()
    averages: int = 10
    tip: float = 50.0  # ms, the hammer's pulse
    exp_end: float = 100.0  # %, the exponential window's value at the block's end

    def system(self) -> ChainSystem:
        if not self.masses:
            return ChainSystem.uniform(4, friction=self.friction)
        n = len(self.masses)
        return ChainSystem(
            np.array(self.masses), np.array(self.springs), np.array(self.dampers), np.full(n, self.friction)
        )


TRIALS = (
    ModalTestTrial(
        "A force at a node",
        "The hammer hits m3, which doesn't move in mode 2 of the uniform chain. Mode 2 isn't excited, drops "
        "out of every FRF, and no method can find it: the Modes tab lists three. Move the "
        "force to m4 and it's back.",
        input=2,
    ),
    ModalTestTrial(
        "Force noise: H1 against H2",
        "5% noise on the force channel. H1 reads low at the resonances, where the structure barely resists the "
        "force and the force signal is smallest, and the coherence dips there. Switch the estimator to H2: "
        "the peaks come back.",
        force_noise=5.0,
    ),
    ModalTestTrial(
        "Response noise: the antiresonances",
        "5% noise on the response channels. H1 stays unbiased; switch to H2 and it reads high, most at the "
        "antiresonances, where the response is small. Look at m1's FRF (Show response of).",
        response_noise=5.0,
    ),
    ModalTestTrial(
        "A short block: leakage",
        "Nb = 256 with no window: the hit is still ringing when the block ends, so the peaks leak, low and "
        "wide, and the damping comes out too high. Pick the Force + exponential window: the leakage goes, but "
        "every mode gains Δζ = 1/(τ<sub>w</sub>ω) (the dashed grey curve), which the extraction takes off "
        "again unless you untick <i>Remove the exponential window's damping</i>.",
        block=256,
        window=Window.RECTANGULAR,
    ),
    ModalTestTrial(
        "Peak picking on coarse lines",
        "Peak picking with Nb = 256: each frequency snaps to a spectral line and ζ, from the half-power "
        "points, comes out too high. Lengthen the block, or switch to the circle fit or LSCF.",
        block=256,
        method=Method.PEAK,
    ),
    ModalTestTrial(
        "Acceleration evens out the modes",
        "Periodic random with 2% response noise, measuring displacement: the top modes sink into the noise "
        "and LSCF can miss one. Switch the response sensor to Acceleration: every mode comes through, but "
        "the FRF below the first mode turns to noise.",
        excitation=Excitation.PERIODIC_RANDOM,
        response_noise=2.0,
        tab=2,
    ),
    ModalTestTrial(
        "Friction: the linearity check",
        "0.5 N of friction on every mass, hammer at 100%. Press <i>Hold for comparison</i>, then set the force "
        "level to 1000%: the peaks grow taller and sharper, towards the exact (frictionless) FRF, and the "
        "coherence drops even with no noise. See <i>Nonlinearity: friction</i>.",
        friction=0.5,
    ),
)


# ------------------------------------------------------------------ practice problems
# Each question's numbers are printed round, and its answer is worked from them as printed.
TAB_SETUP, TAB_MODES, TAB_STABILIZATION = range(3)
TWO_PI = 2.0 * math.pi


def _half_power(rng: np.random.Generator) -> Question:
    fn, zeta = pick(rng, (1.105, 3.183, 4.877, 5.982)), pick(rng, (0.01, 0.02, 0.025, 0.05))
    f1, f2 = (float(number(fn * (1 + sign * zeta), 4)) for sign in (-1, 1))
    z = (f2 - f1) / (2 * fn)
    return Question(
        f"A measured FRF peaks at {number(fn, 4)} Hz, and its magnitude is 1/√2 of the peak at {number(f1, 4)} Hz and "
        f"{number(f2, 4)} Hz. What damping ratio do the half-power points give?",
        z,
        f"ζ ≈ (f₂ − f₁)/2f<sub>n</sub> = ({number(f2, 4)} − {number(f1, 4)})/(2 × {number(fn, 4)}) = {number(z)}.",
    )


def _lines(rng: np.random.Generator) -> Question:
    fn, zeta = pick(rng, (1.105, 3.183, 4.877)), pick(rng, (0.005, 0.01, 0.02, 0.05))
    fs, block = pick(rng, (10, 15, 20, 25)), int(pick(rng, (256, 512, 1024, 2048)))
    width = 2 * zeta * fn
    lines = width * block / fs
    return Question(
        f"A mode at {number(fn, 4)} Hz with ζ = {zeta:g}, measured at f<sub>s</sub> = {fs:g} Hz in blocks of "
        f"N<sub>b</sub> = {block}. How many spectral lines fall across its half-power bandwidth?",
        lines,
        f"The lines are Δf = f<sub>s</sub>/N<sub>b</sub> = {number(fs / block)} Hz apart (the block lasts "
        f"{number(block / fs)} s); the bandwidth is 2ζf<sub>n</sub> = {number(width)} Hz, so {number(lines)} lines. "
        "Under two, the peak comes out too low and looks too damped.",
        unit="lines",
        tolerance=0.02,
        setup=dict(fs=float(fs), block=block, tab=TAB_SETUP),
    )


def _exp_window(rng: np.random.Generator) -> Question:
    fs, block = pick(rng, (10, 20, 25)), int(pick(rng, (512, 1024, 2048)))
    end, fn = pick(rng, (1, 5, 10, 20)), pick(rng, (1.105, 3.183, 5.982))
    T = block / fs
    tau = T / math.log(100 / end)
    dz = 1 / (tau * TWO_PI * fn)
    return Question(
        f"An impact test at f<sub>s</sub> = {fs:g} Hz, N<sub>b</sub> = {block}, with an exponential window that has "
        f"fallen to {end:g} % at the block's end. How much damping ratio does it add to the mode at {number(fn, 4)} Hz?",
        100 * dz,
        f"T = N<sub>b</sub>/f<sub>s</sub> = {number(T)} s and e<sup>−T/τ<sub>w</sub></sup> = {end / 100:g}, so "
        f"τ<sub>w</sub> = T/ln({number(100 / end)}) = {number(tau)} s. Every pole moves 1/τ<sub>w</sub> left: "
        f"Δζ = 1/(τ<sub>w</sub>ω) = 1/({number(tau)} × {number(TWO_PI * fn)}) = {number(100 * dz)} %.",
        unit="%",
        tolerance=0.02,
        setup=dict(fs=float(fs), block=block, exp_end=float(end), window=Window.FORCE_EXPONENTIAL, tab=TAB_MODES),
    )


def _tip(rng: np.random.Generator) -> Question:
    tip, f = (  # u = 2fτ from 0.6 to 1.2: well down the roll-off, short of its first zero at 1.5
        (10, 40),
        (20, 20),
        (20, 30),
        (50, 6),
        (50, 10),
        (50, 12),
        (100, 4),
        (100, 6),
        (200, 2),
        (200, 3),
    )[int(rng.integers(10))]
    down = -20 * math.log10(float(impact_spectrum(tip / 1000, np.array([float(f)]))[0]))
    u = 2 * f * tip / 1000
    return Question(
        f"A hammer's tip makes a {tip:g} ms half-sine pulse. How many dB below its level at 0 Hz is the force's spectrum "
        f"at {f:g} Hz?",
        down,
        f"A half-sine of length τ<sub>p</sub> has |F(f)|/F(0) = |cos(πu/2)/(1 − u²)|, u = 2fτ<sub>p</sub> = {number(u)}: "
        f"{number(10 ** (-down / 20))}, {number(down)} dB down. It first reaches zero at 1.5/τ<sub>p</sub> = "
        f"{number(1500 / tip)} Hz.",
        unit="dB",
        tolerance=0.03,
        setup=dict(tip=float(tip), tab=TAB_SETUP),
    )


def _coherence(rng: np.random.Generator) -> Question:
    h1, ratio = pick(rng, (0.5, 1.2, 2.0, 3.5)), pick(rng, (1.02, 1.05, 1.1, 1.25))
    h2 = float(number(h1 * ratio))
    g = h1 / h2
    return Question(
        f"At one frequency, H1 reads {h1:g} mm/N and H2 reads {number(h2)} mm/N (same phase). What is the coherence "
        "there?",
        g,
        f"γ² = |G<sub>xf</sub>|²/(G<sub>ff</sub>G<sub>xx</sub>) = H1/H2 = {h1:g}/{number(h2)} = {number(g)}.",
    )


def _alias(rng: np.random.Generator) -> Question:
    f, fs = pick(rng, (4.877, 5.982)), pick(rng, (7, 8, 9))
    seen = abs(f - fs * round(f / fs))
    return Question(
        f"A mode at {number(f, 4)} Hz, sampled at {fs:g} Hz with the anti-alias filter off. At what frequency does it "
        "appear?",
        seen,
        f"It's above the Nyquist frequency f<sub>s</sub>/2 = {fs / 2:g} Hz, so it folds back to |f − kf<sub>s</sub>| "
        f"= |{number(f, 4)} − {fs:g}| = {number(seen)} Hz, looking like a real mode there.",
        unit="Hz",
        setup=dict(fs=float(fs), anti_alias=False, tab=TAB_SETUP),
    )


def _node(rng: np.random.Generator) -> Question:
    return Question(
        "The uniform 4-mass chain is hit at m3, which doesn't move in mode 2. How many modes can the test find?",
        0,
        "Three: a force at a mode's node puts nothing into it, so mode 2 drops out of every FRF in the column. Hit "
        "m4 and it's back.",
        choices=("Three", "All four", "Two"),
        setup=dict(input=2, tab=TAB_MODES),
        predict=True,
    )


def _force_noise(rng: np.random.Generator) -> Question:
    return Question(
        "5 % noise on the force channel, the FRF estimated by H1. What does H1 do at the resonances?",
        0,
        "It reads low: noise on the force inflates G<sub>ff</sub>, most where the force is smallest, at the "
        "resonances, and the coherence dips there. H2 leaves it out.",
        choices=("It reads low", "It reads high", "It's unaffected"),
        setup=dict(force_noise=5.0, tab=TAB_MODES),
        predict=True,
    )


def _short_block(rng: np.random.Generator) -> Question:
    return Question(
        "An impact test with N<sub>b</sub> = 256 and no window: the response is still ringing when the block ends. "
        "How does the identified damping come out?",
        0,
        "Too high: the block cuts the ringing off, the peaks leak, low and wide, and a wide peak looks damped.",
        choices=("Too high", "Too low", "About right"),
        setup=dict(block=256, window=Window.RECTANGULAR, tab=TAB_MODES),
        predict=True,
    )


def _one_average(rng: np.random.Generator) -> Question:
    return Question(
        "A test with 5 % noise on the response, but only one average. What does the coherence read?",
        0,
        "1 at every frequency: with one block, |XF*|² = |X|²|F|² exactly, whatever the noise. It takes several "
        "averages before noise can show.",
        choices=("1 at every frequency", "Low at the antiresonances", "Low everywhere"),
        setup=dict(averages=1, response_noise=5.0, tab=TAB_SETUP),
        predict=True,
    )


def _modes_of(t: ModalTestTrial) -> tuple[float, float, float]:
    """The chain's narrowest half-power width and highest damped frequency (Hz), and Auto's sample rate."""
    result = modal_analysis(t.system())
    modes = [m.damped for m in result.modes if m.damped is not None]
    return min(2 * d.zeta * d.fn_hz for d in modes), max(d.fd_hz for d in modes), suggested_fs(result)


def _acquisition_spec(rng: np.random.Generator) -> Question:
    c = pick(rng, (1.0, 2.0, 4.0))
    chain: dict[str, Any] = dict(masses=(1.0,) * 4, springs=(400.0,) * 4, dampers=(c,) * 4)
    given = ModalTestTrial("", "", **chain).system()
    width, top, fs = _modes_of(ModalTestTrial("", "", **chain))
    block = min(b for b in BLOCK_SIZES if b >= 4 * fs / width)
    longest = math.ceil(1.2 * 10 * block / fs / 60)  # minutes: a fifth over 10 averages of the design's blocks

    def goal(t: ModalTestTrial):
        w, f_top, auto = _modes_of(t)
        rate = t.fs if t.fs is not None else auto
        band = AA_CUTOFF * rate / 2
        return (
            Requirement(
                f"The uniform 4-mass chain, {c:g} N·s/m dampers",
                f"{len(t.masses)} masses",
                _same_chain(t.system(), given),
            ),
            Requirement("The anti-alias filter on", "on" if t.anti_alias else "off", t.anti_alias),
            at_least("The usable band (80 % of Nyquist)", band, f_top, "Hz"),
            at_least("Lines across the narrowest half-power width", w * t.block / rate, 4),
            at_least("Averages", t.averages, 10),
            at_most("The test's time, averages × T", t.averages * t.block / rate / 60, longest, "min"),
        )

    return Question(
        f"The uniform 4-mass chain (1 kg, 400 N/m, {c:g} N·s/m each) is to be tested with every mode inside the usable "
        f"band (the anti-alias filter on), four or more lines across each mode's half-power width and at least 10 "
        f"averages, in no more than {longest:g} minutes. Choose the sample rate and the block size.",
        0,
        f"The top mode is at {number(top)} Hz, so f<sub>s</sub> ≥ 2 × {number(top)}/0.8 = {number(2.5 * top)} Hz. "
        f"Mode 1 is the narrowest, 2ζf<sub>n</sub> = {number(width)} Hz, so a block must last 4/{number(width)} = "
        f"{number(4 / width)} s or more. At Auto's {fs:g} Hz that's N<sub>b</sub> ≥ {number(4 * fs / width)}: "
        f"{block}, {number(block / fs)} s a block, {number(10 * block / fs / 60)} minutes for 10 averages. A faster "
        "rate needs a longer block for the same lines.",
        setup=dict(**chain, block=256, tab=TAB_SETUP),
        goal=goal,
        design=dict(**chain, block=block, tab=TAB_SETUP),
    )


PROBLEMS = tuple(
    Problem(key, title, "test_page", f"See {page_link('test_page', anchor, section)}.", make)
    for key, title, anchor, section, make in (
        ("half-power", "Damping from the half-power points", "extraction", "Modal parameter extraction", _half_power),
        ("lines", "Lines across a peak", "leakage", "Blocks, the DFT and leakage", _lines),
        ("exp-window", "The exponential window's damping", "excitation", "Excitation", _exp_window),
        ("tip", "A hammer tip's spectrum", "excitation", "Excitation", _tip),
        ("coherence", "Coherence from H1 and H2", "estimators", "Averaging and estimators", _coherence),
        ("alias", "Where a mode aliases", "sampling", "Sampling", _alias),
        ("node", "Predict: a force at a node", "measured", "What is measured, and where", _node),
        ("force-noise", "Predict: force noise", "estimators", "Averaging and estimators", _force_noise),
        ("short-block", "Predict: a short block", "leakage", "Blocks, the DFT and leakage", _short_block),
        ("one-average", "Predict: one average", "estimators", "Averaging and estimators", _one_average),
        (
            "acquisition-spec",
            "Spec: a sample rate and a block",
            "leakage",
            "Blocks, the DFT and leakage",
            _acquisition_spec,
        ),
    )
)


def saved_playback() -> str:
    """The playback speed picked last (its label), kept in the user's settings like the theme."""
    return str(settings().value("playback", SPEEDS[0][0]))


def save_playback(label: str) -> None:
    settings().setValue("playback", label)


OVERLAPS = [0.0, 0.5, 0.75]

# The window each excitation is normally measured with; picked when the excitation changes.
RECOMMENDED_WINDOW = {
    Excitation.IMPACT: Window.FORCE_EXPONENTIAL,
    Excitation.RANDOM: Window.HANN,
    Excitation.BURST_RANDOM: Window.RECTANGULAR,
    Excitation.PERIODIC_RANDOM: Window.RECTANGULAR,
    Excitation.CHIRP: Window.RECTANGULAR,
    Excitation.STEPPED_SINE: Window.RECTANGULAR,
}

EXCITATION_TIPS = {
    Excitation.IMPACT: "A hammer hit: a short half-sine pulse at the start of each block. Quick, and "
    "no shaker needed. The tip sets how short the pulse is, and so how high it excites.",
    Excitation.RANDOM: "A shaker driven by continuous band-limited noise. Never periodic in the "
    "block, so it leaks without a window (Hann). Averaging reduces noise, and random excitation averages any non-linearity into a best "
    "linear estimate.",
    Excitation.BURST_RANDOM: "Noise for the first part of each block, then silence while the "
    "response dies away. If it dies away inside the block, no window is needed.",
    Excitation.PERIODIC_RANDOM: "A random signal exactly one block long, repeated until the "
    "response is steady, so both signals are periodic in the block: no leakage.",
    Excitation.CHIRP: "A sine sweeping from 0 to the band edge in one block, repeated. Periodic "
    "in the block, like periodic random.",
    Excitation.STEPPED_SINE: "One frequency at a time: wait for steady state, then fit a sine to "
    "force and response. Slow, but the most accurate, with no windows or leakage.",
}

WINDOW_TIPS = {
    Window.RECTANGULAR: "No window. Right when the signals are periodic in the block or have died "
    "away by its end; otherwise it leaks.",
    Window.HANN: "Tapers both ends to zero, which cuts leakage for random signals. It also blurs "
    "sharp peaks a little, and it wipes out a hammer hit at the start of the block.",
    Window.FLAT_TOP: "Very wide main lobe: accurate amplitudes of pure tones, but blurs resonances.",
    Window.FORCE_EXPONENTIAL: "For impact tests. Force: keeps the hit and zeroes the noise after "
    "it. Exponential on both channels: makes the response die away inside the block, which adds "
    "known extra damping to every mode.",
}


FORCE_LEVEL_TIP = (
    f"<p>Scales the force. At 100% the hammer hits with a {IMPACT_PEAK:g} N peak, the shaker drives "
    f"{FORCE_RMS:g} N RMS and the stepped sine has a {SINE_AMPLITUDE:g} N amplitude.</p>"
    "<p>A linear chain's FRF does not depend on it: the response scales with the force, and so "
    "does the noise, which is sized to each channel's range. With <b>friction</b> on a mass "
    "(F<sub>f</sub> on the Simulation page) it does. A lab checks linearity this way: measure at "
    "two force levels and overlay the FRFs (<i>Hold for comparison</i>).</p>"
    "<p>With friction and a stepped sine, the plot also shows the describing-function FRF at "
    "this level (dash-dot): the frequency response worked out with each friction force "
    "replaced by its first harmonic, the same approximation as the dash-dot curves on the "
    "Simulation page's Frequency response tab. A stepped sine reads that first harmonic, so "
    "the two should agree while every mass slides; a random or impact test has no single "
    "amplitude to compare with.</p>"
)
HOLD_TIP = (
    "<p>Keep the FRF measured now on the plot, in grey, while you change the force level or the "
    "excitation and measure again. It is drawn while the force is at the same mass and the same "
    "quantity is measured.</p>"
    "<p>For a linear chain the two agree, apart from noise. If they differ, the structure is "
    "nonlinear; friction damps the response to a light force much more than to a strong one.</p>"
)


def _same_chain(a: ChainSystem, b: ChainSystem) -> bool:
    return a.n == b.n and all(
        np.allclose(x, y)
        for x, y in ((a.masses, b.masses), (a.stiffness, b.stiffness), (a.damping, b.damping), (a.friction, b.friction))
    )


def friction_masses(system: ChainSystem) -> str:
    """'m1, m3' for the masses with friction ('' if none)."""
    return ", ".join(f"m{i + 1}" for i in np.flatnonzero(system.friction > 0))


def suggested_fs(result: ModalResult) -> float:
    """A sample rate whose passband (AA_CUTOFF x Nyquist) clears the highest mode by 25%."""
    f_top = max((m.fn_hz for m in result.complex_modes if m.is_oscillatory), default=1.0)
    need = 1.25 * f_top * 2.0 / AA_CUTOFF
    scale = 10.0 ** (math.floor(math.log10(need)) - 1)
    return math.ceil(need / scale) * scale


def percent(fraction: float) -> str:
    return "&lt; 0.01%" if fraction < 1e-4 else f"{100 * fraction:.2g}%"


def phase_deg(h: np.ndarray) -> np.ndarray:
    """Phase in degrees, wrapped to (-190, 170] so a phase near -180 does not flicker to +180."""
    return (np.degrees(np.angle(h)) + 190.0) % 360.0 - 190.0


class SignalView(pg.GraphicsLayoutWidget):
    """The latest block: force and one response against time (with the windows), and the force spectrum."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.force = self.addPlot(row=0, col=0)
        self.force.setLabel("left", "Force", units="N")
        self.spectrum = self.addPlot(row=0, col=1, rowspan=2)
        self.spectrum.setLabel("left", "Force spectrum", units="dB")
        self.spectrum.setLabel("bottom", "Frequency", units="Hz")
        self.spectrum.showGrid(x=True, y=True, alpha=0.3)
        self.response = self.addPlot(row=1, col=0)
        self.response.setLabel("bottom", "Time in block", units="s")
        self.response.setXLink(self.force)
        self.zero_lines = [pg.InfiniteLine(pos=0, angle=0) for _ in range(2)]
        for p, line in zip((self.force, self.response), self.zero_lines):
            p.addItem(line)
        self.ci.layout.setColumnStretchFactor(0, 3)
        self.ci.layout.setColumnStretchFactor(1, 2)
        self.f_curve = self.force.plot()
        self.f_window = self.force.plot()
        self.f_fit = self.force.plot()
        self.x_curve = self.response.plot()
        self.x_window = self.response.plot()
        self.x_fit = self.response.plot()
        self.s_curve = self.spectrum.plot()
        self.s_tip = self.spectrum.plot()
        self.band = pg.LinearRegionItem(movable=False, pen=pg.mkPen(None))
        self.spectrum.addItem(self.band)
        self.apply_theme()

    def apply_theme(self) -> None:
        """Every curve but the response and the measured force, which set_data styles."""
        dash = QtCore.Qt.PenStyle.DashLine
        for line in self.zero_lines:
            line.setPen(pg.mkPen(colors.faint, width=1))
        self.f_fit.setPen(pg.mkPen(colors.force, width=1))
        self.s_curve.setPen(pg.mkPen(colors.force, width=1))
        for curve in (self.f_window, self.x_window):
            curve.setPen(pg.mkPen(colors.grey, width=1, style=dash))
        self.s_tip.setPen(pg.mkPen(colors.strong, width=1, style=dash))
        self.band.setBrush(pg.mkBrush(*colors.band_shade))

    def clear_measurement(self) -> None:
        for curve in (self.f_curve, self.f_window, self.f_fit, self.x_curve, self.x_window, self.x_fit, self.s_curve):
            curve.clear()

    def set_data(self, est: Estimate, j: int, settings: MeasurementSettings, stepped: bool) -> None:
        self.set_time(est, j, stepped)
        self.set_spectrum(est, settings, stepped)

    def set_time(self, est: Estimate, j: int, stepped: bool, shown: float | None = None) -> None:
        """The latest record against time; while it is being recorded, only its first `shown` s."""
        n = est.t.size if shown is None else int(np.searchsorted(est.t, shown, side="right"))
        # Block excitations: the samples joined by lines. Stepped sine: the samples as dots
        # (as few as 2.5 per cycle at the band edge) and the fitted sine through them.
        for plot, curve, fit, y, y_fit, color in (
            (self.force, self.f_curve, self.f_fit, est.f, None if est.fit is None else est.fit[1], colors.force),
            (
                self.response,
                self.x_curve,
                self.x_fit,
                est.x[:, j],
                None if est.fit is None else est.fit[2][:, j],
                colors.mass[j],
            ),
        ):
            if y_fit is None:
                curve.setData(est.t[:n], y[:n], pen=pg.mkPen(color, width=1), symbol=None)
                fit.setData([], [])
            else:
                curve.setData(est.t[:n], y[:n], pen=None, symbol="o", symbolSize=4, symbolPen=None, symbolBrush=color)
                assert est.fit is not None, "a fitted curve comes from the fit"
                m = est.fit[0].size if shown is None else int(np.searchsorted(est.fit[0], shown, side="right"))
                fit.setData(est.fit[0][:m], y_fit[:m])
            # Time spans the record. (Auto-ranging it padded by the widget's size when the range was
            # worked out, which could be before the layout settled.)
            plot.setXRange(0.0, float(est.t[-1]), padding=0.02)
            if shown is None:
                plot.enableAutoRange(axis="y")
            else:  # the whole record's range, so the axes hold still while it is drawn
                plot.setYRange(float(np.min(y)), float(np.max(y)))
        self.x_fit.setPen(pg.mkPen(colors.mass[j], width=1))
        self.response.setLabel("left", f"{est.response.symbol}{j + 1}", units=est.response.unit)
        for curve, window, signal in (
            (self.f_window, est.force_window, est.f),
            (self.x_window, est.response_window, est.x[:, j]),
        ):
            if window is None:
                curve.setData([], [])
            else:
                curve.setData(est.t, window * float(np.abs(signal).max() or 1.0))
        what = (f"{est.freqs[-1]:.3g} Hz" if stepped else f"Block {est.count}") + (
            " (recording…)" if shown is not None else ""
        )
        self.force.setTitle(
            what
            + (
                ": dots as measured · line: fitted sine"
                if stepped
                else ": signals as measured · dashed: window (scaled)"
            ),
            size="9pt",
        )

    def set_spectrum(self, est: Estimate, settings: MeasurementSettings, stepped: bool) -> None:
        spec = est.force_spectrum
        with np.errstate(divide="ignore"):
            db = 10.0 * np.log10(spec / spec.max()) if spec.size and spec.max() > 0 else spec
        self.s_curve.setData(
            est.freqs,
            db,
            connect="finite",
            symbol="o" if stepped else None,
            symbolSize=4,
            symbolPen=None,
            symbolBrush=colors.force,
        )
        nyq = settings.fs / 2
        if settings.excitation is Excitation.IMPACT:
            f = np.linspace(0.0, nyq, 400)
            with np.errstate(divide="ignore"):
                self.s_tip.setData(f, 20.0 * np.log10(impact_spectrum(settings.tip_width, f)), connect="finite")
        else:
            self.s_tip.setData([], [])
        self.band.setRegion((settings.band, nyq))
        self.spectrum.setXRange(0.0, nyq, padding=0)
        self.spectrum.setYRange(-60.0, 5.0, padding=0)
        self.spectrum.setTitle(
            "Averaged |F|² (dB re peak)"
            + (" · dashed: ideal hammer pulse" if settings.excitation is Excitation.IMPACT else ""),
            size="9pt",
        )


class FrfView(pg.GraphicsLayoutWidget):
    """Estimated receptance of one response against the exact one: magnitude, phase, coherence.

    A draggable band on the magnitude plot sets the frequencies the modal fit uses.
    """

    fit_band_changed = QtCore.Signal(float, float)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.mag = self.addPlot(row=0, col=0, axisItems=log_axes())
        self.mag.setLogMode(x=False, y=True)
        self.mag.setLabel("left", "|H|  [m/N]")
        add_legend(self.mag, offset=(-5, 5))
        self.phase = self.addPlot(row=1, col=0)
        self.phase.setLabel("left", "Phase", units="deg")
        self.phase.getAxis("left").setTickSpacing(90.0, 45.0)
        self.coh = self.addPlot(row=2, col=0)
        self.coh.setLabel("left", "Coherence γ²")
        self.coh.setLabel("bottom", "Frequency", units="Hz")
        self.coh.setYRange(0.0, 1.05, padding=0)
        for p in (self.mag, self.phase, self.coh):
            p.showGrid(x=True, y=True, alpha=0.3)
            if p is not self.mag:
                p.setXLink(self.mag)
        self.phase.setYRange(-195.0, 175.0, padding=0)
        self.ci.layout.setRowStretchFactor(0, 3)
        self.ci.layout.setRowStretchFactor(1, 2)
        self.ci.layout.setRowStretchFactor(2, 1)
        self._key: tuple | None = None
        self.measured: list[pg.PlotDataItem] = []  # magnitude, phase, coherence
        self.fit_curves: list[pg.PlotDataItem] = []  # magnitude, phase
        self.held_curves: list[pg.PlotDataItem] = []  # magnitude, phase
        self.fit_region = pg.LinearRegionItem()
        self.fit_region.setZValue(-5)
        self.fit_region.sigRegionChangeFinished.connect(self._on_region)
        self.apply_theme()

    def apply_theme(self) -> None:
        """The fit band now; the curves are drawn again with the next data."""
        shade = pg.mkColor(colors.fit)
        shade.setAlpha(26 if colors.dark else 28)
        self.fit_region.setBrush(pg.mkBrush(shade))
        for line in self.fit_region.lines:
            line.setPen(pg.mkPen(colors.fit, width=1))
        self._key = None

    def _on_region(self) -> None:
        lo, hi = self.fit_region.getRegion()
        self.fit_band_changed.emit(max(0.0, float(lo)), float(hi))

    def set_fit_band(self, lo: float, hi: float) -> None:
        self.fit_region.blockSignals(True)
        self.fit_region.setRegion((lo, hi))
        self.fit_region.blockSignals(False)

    def clear_measurement(self) -> None:
        """Hide the measurement (a new one has started); the exact curves stay."""
        for curve in self.measured:
            curve.clear()

    def set_fit(self, freqs: np.ndarray | None, h: np.ndarray | None) -> None:
        """The FRF rebuilt from the identified modes (None: hide it)."""
        if not self.fit_curves:
            return
        mag_curve, phase_curve = self.fit_curves
        legend = self.mag.legend
        shown = legend.getLabel(mag_curve) is not None
        if h is None:
            mag_curve.setData([], [])
            phase_curve.setData([], [])
            if shown:
                legend.removeItem(mag_curve)
            return
        with np.errstate(invalid="ignore"):
            mag_curve.setData(freqs, np.where(np.abs(h) > 0, np.abs(h), np.nan), connect="finite")
        phase_curve.setData(freqs, phase_deg(h), connect="finite")
        if not shown:
            legend.addItem(mag_curve, "Fitted modal model")

    def set_held(self, freqs: np.ndarray | None, h: np.ndarray | None, label: str = "") -> None:
        """An FRF held for comparison (None: hide it)."""
        if not self.held_curves:
            return
        mag_curve, phase_curve = self.held_curves
        legend = self.mag.legend
        if legend.getLabel(mag_curve) is not None:
            legend.removeItem(mag_curve)
        if h is None or freqs is None:
            mag_curve.setData([], [])
            phase_curve.setData([], [])
            return
        keep = freqs > 0
        with np.errstate(invalid="ignore"):
            mag_curve.setData(freqs[keep], np.where(np.abs(h[keep]) > 0, np.abs(h[keep]), np.nan), connect="finite")
        phase_curve.setData(freqs[keep], phase_deg(h[keep]), connect="finite")
        legend.addItem(mag_curve, label)

    def set_data(
        self,
        est: Estimate,
        j: int,
        system: ChainSystem,
        result: ModalResult,
        settings: MeasurementSettings,
        processing: Processing,
        stepped: bool,
    ) -> None:
        # The exact curves, mode lines and bands change only with the test; while it runs,
        # only the measured curves are updated.
        key = (
            id(system),
            id(result),
            settings.fs,
            settings.input_dof,
            settings.response,
            j,
            stepped,
            processing.window,
            processing.exp_end,
            processing.estimator,
            settings.force_level,
        )
        if key != self._key:
            self._key = key
            self._draw_reference(j, system, result, settings, processing, stepped)
        keep = est.freqs > 0
        fm, Hm = est.freqs[keep], est.H[keep, j]
        with np.errstate(invalid="ignore"):
            mag = np.where(np.abs(Hm) > 0, np.abs(Hm), np.nan)
        mag_curve, phase_curve, coh_curve = self.measured
        mag_curve.setData(fm, mag, connect="finite")
        phase_curve.setData(fm, phase_deg(Hm), connect="finite")
        if est.coherence is not None:
            coh_curve.setData(fm, est.coherence[keep, j], connect="finite")
        noun = "frequencies" if stepped else "averages"
        self.mag.legend.getLabel(mag_curve).setText(f"Measured {processing.estimator.name} ({est.count} {noun})")

    def _draw_reference(
        self,
        j: int,
        system: ChainSystem,
        result: ModalResult,
        settings: MeasurementSettings,
        processing: Processing,
        stepped: bool,
    ) -> None:
        for p in (self.mag, self.phase, self.coh):
            p.clear()
        self.mag.legend.clear()
        nyq = settings.fs / 2
        f = np.linspace(nyq / 2000, nyq, 2000)
        peaks = [m.damped.fd_hz for m in result.modes if m.damped is not None and m.damped.fd_hz < nyq]
        f = np.unique(np.concatenate([f, peaks]))
        response = settings.response
        exact = from_receptance(f, frf(system, f, settings.input_dof)[:, j], response)
        dash = QtCore.Qt.PenStyle.DashLine
        exact_name = "Exact, without friction" if np.any(system.friction > 0) else "Exact"
        self.mag.plot(f, np.abs(exact), pen=pg.mkPen(colors.strong, width=1.5), name=exact_name)
        self.phase.plot(f, phase_deg(exact), pen=pg.mkPen(colors.strong, width=1.5))
        if processing.window is Window.FORCE_EXPONENTIAL and processing.exp_end < 1.0 and not stepped:
            s = 1j * 2 * np.pi * f + 1.0 / processing.exp_tau(settings)
            damped = s**response.power * transfer(system, s, settings.input_dof)[:, j]
            pen = pg.mkPen(colors.grey, width=1, style=dash)
            self.mag.plot(f, np.abs(damped), pen=pen, name="Exact + window damping")
            self.phase.plot(f, phase_deg(damped), pen=pen)
        if stepped and np.any(system.friction > 0):
            # A stepped sine reads the first harmonic at each frequency, which is what the
            # describing function approximates. A mass held by friction leaves a gap.
            amplitude = settings.force_level * SINE_AMPLITUDE
            fd = np.unique(np.concatenate([np.linspace(nyq / 600, nyq, 600), peaks]))
            rf = friction_frf(system, fd, settings.input_dof, amplitude)
            h = rf.X[:, j] / amplitude
            h[rf.stuck[:, j]] = np.nan
            described = from_receptance(fd, h, response)
            pen = pg.mkPen(colors.strong, width=1.5, style=QtCore.Qt.PenStyle.DashDotLine)
            self.mag.plot(
                fd, np.abs(described), pen=pen, connect="finite", name=f"Describing function at {amplitude:.3g} N"
            )
            self.phase.plot(fd, phase_deg(described), pen=pen, connect="finite")

        held_pen = pg.mkPen(colors.grey, width=1.5)
        self.held_curves = [self.mag.plot(pen=held_pen), self.phase.plot(pen=held_pen)]
        color = colors.mass[j]
        symbol = dict(symbol="o", symbolSize=5, symbolPen=None, symbolBrush=color) if stepped else {}
        self.measured = [
            self.mag.plot(pen=pg.mkPen(color, width=1.5), name="Measured", **symbol),
            self.phase.plot(pen=pg.mkPen(color, width=1), **symbol),
            self.coh.plot(pen=pg.mkPen(color, width=1)),
        ]
        fit_pen = pg.mkPen(colors.fit, width=2, style=QtCore.Qt.PenStyle.DashLine)
        self.fit_curves = [self.mag.plot(pen=fit_pen), self.phase.plot(pen=fit_pen)]
        self.mag.addItem(self.fit_region)
        if stepped:
            label = pg.TextItem("A stepped sine has no coherence (one reading per frequency)", color=colors.muted)
            label.setPos(0.02 * nyq, 0.6)
            self.coh.addItem(label)

        for r, m in enumerate(result.modes):
            if m.damped is None or m.damped.fd_hz >= nyq:
                continue
            pen = pg.mkPen(colors.mode[r % len(colors.mode)], width=1, style=QtCore.Qt.PenStyle.DotLine)
            for p in (self.mag, self.phase, self.coh):
                p.addItem(pg.InfiniteLine(pos=m.damped.fd_hz, angle=90, pen=pen))
        for p in (self.mag, self.phase, self.coh):
            band = pg.LinearRegionItem(
                (settings.band, nyq), movable=False, brush=pg.mkBrush(*colors.band_shade), pen=pg.mkPen(None)
            )
            band.setZValue(-10)
            p.addItem(band)
        # An accelerance falls as ω² towards 0 Hz: range it on the band, not on the first lines.
        shown = np.isfinite(exact) & ((f >= 0.02 * nyq) if response is Response.ACCELERATION else True)
        finite = np.abs(exact[shown])
        if finite.size:
            lo, hi = math.log10(finite.min()), math.log10(finite.max())
            self.mag.setYRange(lo - 1.0, hi + 0.5, padding=0)
        self.mag.setXRange(0.0, nyq, padding=0)
        drive = "driving-point" if j == settings.input_dof else "transfer"
        name, unit = ("H", "m/N") if response is Response.DISPLACEMENT else ("A", "(m/s²)/N")
        self.mag.setLabel("left", f"|{name}|  [{unit}]")
        self.mag.setTitle(
            f"{name}<sub>{j + 1}{settings.input_dof + 1}</sub> = {response.symbol}{j + 1} / F "
            f"at m{settings.input_dof + 1} ({drive} {response.frf_name}) · dotted: f<sub>d</sub> of "
            "each mode · grey: above the passband · green: fit band (drag it)",
            size="9pt",
        )


class ModalTestPage(QtWidgets.QWidget):
    """Controls on the left, signals and FRF in the centre, setup check and theory on the right."""

    dof_requested = QtCore.Signal(int)  # the user changed N here; the main window owns the system
    edit_parameters = QtCore.Signal()  # the user wants the Simulation page's parameter panel
    system_requested = QtCore.Signal(object)  # a trial's chain, for the Simulation page

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.system: ChainSystem | None = None
        self.result: ModalResult | None = None
        self.acq: Acquisition | None = None
        self.estimator: FrfEstimator | None = None
        self.seed: int | None = None  # fixed random signals and noise (tests); None: fresh each test
        self.estimate: Estimate | None = None
        self.stab: Stabilization | None = None
        self.poles: list[tuple[int, int]] = []  # (order, index) of the LSCF poles used
        self.picked: list[tuple[int, float]] | None = None  # (order, Hz) picked by hand; None: automatic
        self.ident: Identification | None = None
        self._dirty = False
        self.held: tuple[np.ndarray, np.ndarray, MeasurementSettings] | None = None  # (freqs, H, settings)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(FRAME_MS)
        self._timer.timeout.connect(self._measure_frame)
        self._clock = QtCore.QElapsedTimer()
        # Paced playback: the record being drawn, and how much of it (s) is shown so far.
        self._live: Estimate | None = None
        self._shown: float | None = None
        self._frame = QtCore.QElapsedTimer()

        # --- things to try
        trials = QtWidgets.QGroupBox("Things to try")
        tv = QtWidgets.QVBoxLayout(trials)
        self.trial = TrialCombo(
            [t.name for t in TRIALS],
            "<p>Each sets up a test of the uniform 4-mass chain that shows one thing that can "
            "go wrong (or right). Loading one replaces the Simulation page's chain.</p>",
        )
        self.trial.picked.connect(self.load_trial)
        tv.addWidget(self.trial)
        self.trial_note = QtWidgets.QLabel()
        self.trial_note.setWordWrap(True)
        mute(self.trial_note)
        self.trial_note.setVisible(False)
        tv.addWidget(self.trial_note)

        # --- test setup
        setup = QtWidgets.QGroupBox("Test")
        form = form_layout(setup)
        self.dof = QtWidgets.QSpinBox()
        self.dof.setRange(1, MAX_DOF)
        self.dof.setToolTip("Same chain as the Simulation page; changing it here changes it there too.")
        self.dof.valueChanged.connect(self.dof_requested)
        form.addRow("Number of masses:", self.dof)
        self.excitation = QtWidgets.QComboBox()
        for i, e in enumerate(Excitation):
            self.excitation.addItem(e.value, e)
            self.excitation.setItemData(i, EXCITATION_TIPS[e], QtCore.Qt.ItemDataRole.ToolTipRole)
        form.addRow("Excitation:", self.excitation)
        self.input = QtWidgets.QComboBox()
        self.input.setToolTip("Where the force is applied. Every mass's response is measured.")
        form.addRow("Force at:", self.input)
        self.response = QtWidgets.QComboBox()
        for r in Response:
            self.response.addItem(f"{r.value} ({r.frf_name})", r)
        self.response.setToolTip(
            "<p>What the sensor on each mass measures. <b>Displacement</b> gives the receptance "
            "H = X/F directly. <b>Acceleration</b>, as with the accelerometers of a real test, gives "
            "the accelerance A = −ω²H.</p><p>Modes are extracted from the receptance, so an "
            "accelerance is divided by −ω² first; that amplifies the low-frequency noise.</p>"
        )
        form.addRow("Response sensor:", self.response)
        level_row = QtWidgets.QHBoxLayout()
        self.force_level = spin(1.0, 10000.0, 100.0, 0, " %")
        self.force_level.setToolTip(FORCE_LEVEL_TIP)
        self.force_level_info = QtWidgets.QLabel()
        mute(self.force_level_info)
        level_row.addWidget(self.force_level, 1)
        level_row.addWidget(self.force_level_info)
        form.addRow("Force level:", level_row)
        self.tip = spin(1.0, 2000.0, 50.0, 1, " ms")
        self.tip.setToolTip(
            "<p>Duration of the hammer's half-sine pulse. A hard (metal) tip gives a short pulse, "
            "a soft (rubber) tip a long one.</p><p>The pulse's spectrum is flat at low frequency, "
            "rolls off above about 1/τ<sub>p</sub> and first reaches zero at 1.5/τ<sub>p</sub>. Modes above that are "
            "barely excited, so noise swamps them.</p>"
        )
        self.tip_label = QtWidgets.QLabel("Hammer pulse τₚ:")
        form.addRow(self.tip_label, self.tip)
        self.burst = spin(5.0, 100.0, 50.0, 0, " %")
        self.burst.setToolTip("How much of each block the shaker is on. The rest lets the response die away.")
        self.burst_label = QtWidgets.QLabel("Burst length:")
        form.addRow(self.burst_label, self.burst)
        self.overlap = QtWidgets.QComboBox()
        for o in OVERLAPS:
            self.overlap.addItem(f"{o:.0%}", o)
        self.overlap.setCurrentIndex(1)
        self.overlap.setToolTip(
            "Consecutive blocks share this much data. With a Hann window, 50% "
            "overlap uses the data the window tapers away and gives more "
            "averages from the same measuring time."
        )
        self.overlap_label = QtWidgets.QLabel("Block overlap:")
        form.addRow(self.overlap_label, self.overlap)
        self.points = QtWidgets.QSpinBox()
        self.points.setRange(10, 400)
        self.points.setValue(100)
        self.points.setKeyboardTracking(False)
        self.points.setToolTip("Frequencies measured, evenly spaced up to the band edge.")
        self.points_label = QtWidgets.QLabel("Frequencies:")
        form.addRow(self.points_label, self.points)
        link = QtWidgets.QLabel("Masses, springs and dampers are edited on the <a href='#sim'>Simulation page</a>.")
        mute(link)
        link.setWordWrap(True)
        link.linkActivated.connect(self.edit_parameters)
        form.addRow(link)

        # --- acquisition
        daq = QtWidgets.QGroupBox("Acquisition")
        form = form_layout(daq)
        fs_row = QtWidgets.QHBoxLayout()
        self.fs = spin(0.5, 10000.0, 20.0, 2, " Hz")
        self.fs.setToolTip(
            "Samples per second on every channel. Nyquist frequency fs/2 is the "
            "highest frequency the samples can represent."
        )
        self.fs_auto = QtWidgets.QCheckBox("Auto")
        self.fs_auto.setChecked(True)
        self.fs_auto.setToolTip(
            "Follow the chain: choose fs so the highest mode is well inside "
            f"the anti-alias passband ({AA_CUTOFF:.0%} of Nyquist)."
        )
        fs_row.addWidget(self.fs, 1)
        fs_row.addWidget(self.fs_auto)
        form.addRow("Sample rate fs:", fs_row)
        self.block = QtWidgets.QComboBox()
        for b in BLOCK_SIZES:
            self.block.addItem(str(b), b)
        self.block.setCurrentIndex(BLOCK_SIZES.index(1024))
        self.block.setToolTip(
            "Samples per block Nb. The block lasts T = Nb/fs and the FRF lines "
            "are Δf = 1/T apart: longer blocks resolve narrower peaks."
        )
        self.block_label = QtWidgets.QLabel("Block size Nb:")
        form.addRow(self.block_label, self.block)
        self.averages = QtWidgets.QSpinBox()
        self.averages.setRange(1, 100)
        self.averages.setValue(10)
        self.averages.setKeyboardTracking(False)
        self.averages.setToolTip("Blocks averaged. Noise in the estimate falls as 1/√(averages).")
        self.averages_label = QtWidgets.QLabel("Averages:")
        form.addRow(self.averages_label, self.averages)
        self.anti_alias = QtWidgets.QCheckBox("Anti-alias filter")
        self.anti_alias.setChecked(True)
        self.anti_alias.setToolTip(
            "<p>A steep low-pass on every channel before sampling, with its passband edge at "
            f"{AA_CUTOFF:.0%} of Nyquist (the shaded band on the plots is above it).</p>"
            "<p>Without it, response above fs/2 is not lost but folds back: a mode at f "
            "appears at |f − k·fs|, mixed in with the real response.</p>"
        )
        form.addRow(self.anti_alias)
        self.daq_info = QtWidgets.QLabel()
        self.daq_info.setWordWrap(True)
        mute(self.daq_info)
        form.addRow(self.daq_info)

        # --- noise
        noise = QtWidgets.QGroupBox("Measurement noise")
        form = form_layout(noise)
        noise_tip = (
            "Random noise added to the channel, as a percentage of the channel's peak in "
            "each block (the range an analyzer would set). Changing it reuses the same "
            "measurement, so you can compare like with like."
        )
        self.force_noise = spin(0.0, 100.0, 0.0, 2, " %")
        self.force_noise.setToolTip(noise_tip + " Noise on the force biases H1 low.")
        form.addRow("Force channel:", self.force_noise)
        self.response_noise = spin(0.0, 100.0, 0.0, 2, " %")
        self.response_noise.setToolTip(noise_tip + " Noise on the response biases H2 high.")
        form.addRow("Response channels:", self.response_noise)

        # --- processing
        proc = QtWidgets.QGroupBox("Processing")
        form = form_layout(proc)
        self.taper = QtWidgets.QComboBox()
        for i, kind in enumerate(Window):
            self.taper.addItem(kind.value, kind)
            self.taper.setItemData(i, WINDOW_TIPS[kind], QtCore.Qt.ItemDataRole.ToolTipRole)
        self.taper.setToolTip(
            "Multiplies each block before the FFT. Choosing an excitation picks its "
            "usual window; change it to see what the others do to the same data."
        )
        form.addRow("Window:", self.taper)
        self.exp_end = spin(0.1, 100.0, 100.0, 1, " %")
        self.exp_end.setToolTip(
            "<p>The exponential window's value at the end of the block; 100% = no exponential "
            "(force window only).</p><p>e<sup>−t/τ<sub>w</sub></sup> multiplies the response, so every "
            "pole moves 1/τ<sub>w</sub> to the left: each mode looks more damped by Δζ = 1/(τ<sub>w</sub>ω). The dashed "
            "grey curve is the exact FRF with that extra damping.</p>"
        )
        self.exp_label = QtWidgets.QLabel("Exponential at block end:")
        form.addRow(self.exp_label, self.exp_end)
        self.estimator_combo = QtWidgets.QComboBox()
        for estimator in Estimator:
            self.estimator_combo.addItem(estimator.value, estimator)
        self.estimator_combo.setToolTip(
            "<p><b>H1</b> = G<sub>xf</sub>/G<sub>ff</sub>: noise on the response averages out; "
            "noise on the force inflates G<sub>ff</sub> and biases H1 low.</p>"
            "<p><b>H2</b> = G<sub>xx</sub>/G<sub>fx</sub>: noise on the force averages out; "
            "noise on the response inflates G<sub>xx</sub> and biases H2 high.</p>"
            "<p>When uncorrelated noise is the only error, the true FRF lies between them; leakage or "
            "aliasing can push both the same way. Where coherence is 1 they agree.</p>"
        )
        form.addRow("Estimator:", self.estimator_combo)

        run_row = QtWidgets.QHBoxLayout()
        self.again = QtWidgets.QPushButton("Measure again")
        self.again.setToolTip("Repeat the test with new random signals and noise.")
        self.again.clicked.connect(self.restart)
        self.speed = QtWidgets.QComboBox()
        for label, speed in SPEEDS:
            self.speed.addItem(label, speed)
        self.speed.setCurrentIndex(max(0, self.speed.findText(saved_playback())))  # the last one picked
        self.speed.setToolTip(
            "<p>How fast the test plays. <b>Instant</b> measures it all at once. The others draw "
            "each block (or stepped-sine frequency) as it is recorded, at that multiple of real "
            "time, and update the spectrum and FRF when it is complete, as an analyzer does.</p>"
            "<p>The waits for steady state, and for a hit to die away, are skipped.</p>"
        )
        self.progress = QtWidgets.QLabel()
        self.progress.setWordWrap(True)
        run_row.addWidget(self.again)
        run_row.addWidget(self.progress, 1)
        self.hold = QtWidgets.QPushButton("Hold for comparison")
        self.hold.setCheckable(True)
        self.hold.setToolTip(HOLD_TIP)
        self.hold.toggled.connect(self._on_hold)
        speed_row = QtWidgets.QHBoxLayout()
        speed_row.addWidget(QtWidgets.QLabel("Playback:"))
        speed_row.addWidget(self.speed, 1)

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        self.extract = ExtractionControls()
        for w in (trials, setup, daq, noise, proc):
            lv.addWidget(w)
        lv.addLayout(speed_row)
        lv.addLayout(run_row)
        lv.addWidget(self.hold)
        lv.addWidget(self.extract)
        lv.addStretch(1)
        left_scroll = ColumnScroll(left)
        left_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # --- centre
        self.signals = SignalView()
        self.frf_view = FrfView()
        self.output = QtWidgets.QComboBox()
        self.output.currentIndexChanged.connect(self.redraw)
        out_row = QtWidgets.QHBoxLayout()
        out_row.setContentsMargins(6, 0, 0, 0)
        out_row.addWidget(QtWidgets.QLabel("Show response of:"))
        out_row.addWidget(self.output)
        out_row.addStretch(1)
        frf_box = QtWidgets.QWidget()
        fv = QtWidgets.QVBoxLayout(frf_box)
        fv.setContentsMargins(0, 0, 0, 0)
        fv.addLayout(out_row)
        fv.addWidget(self.frf_view, 1)
        centre = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        centre.addWidget(self.signals)
        centre.addWidget(frf_box)
        centre.setSizes([260, 560])

        # --- right
        self.check = QtWidgets.QTextBrowser()
        self.theory = QtWidgets.QTextBrowser()
        self.theory.setHtml(THEORY_HTML)
        self.results = ResultsView()
        self.stab_plot = StabilizationPlot()
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.check, "Setup")
        self.tabs.addTab(self.results, "Modes")
        self.tabs.addTab(self.stab_plot, "Stabilization")
        self.tabs.addTab(self.theory, "Theory")

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(centre)
        splitter.addWidget(self.tabs)
        splitter.setSizes([360, 850, 480])
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        # --- wiring: test settings measure again; processing and noise reuse the data.
        self.excitation.currentIndexChanged.connect(self._on_excitation)
        for choice in (self.input, self.response, self.block, self.overlap):
            choice.currentIndexChanged.connect(self.restart)
        for field in (self.tip, self.burst, self.fs, self.force_level):
            field.valueChanged.connect(self.restart)
        self.points.valueChanged.connect(self.restart)
        self.anti_alias.toggled.connect(self.restart)
        self.fs_auto.toggled.connect(self._on_fs_auto)
        self.averages.valueChanged.connect(self._on_averages)
        self.speed.currentIndexChanged.connect(self._on_speed)
        for choice in (self.taper, self.estimator_combo):
            choice.currentIndexChanged.connect(self._reprocess)
        for field in (self.exp_end, self.force_noise, self.response_noise):
            field.valueChanged.connect(self._reprocess)
        self.extract.changed.connect(self._on_extraction_changed)
        self.extract.auto_requested.connect(self._on_auto_poles)
        self.extract.clear_requested.connect(self._on_clear_poles)
        self.frf_view.fit_band_changed.connect(self._on_fit_band)
        self.stab_plot.pole_clicked.connect(self._on_pole_clicked)
        self._select_window(Excitation.IMPACT)
        self._show_rows()

    def apply_theme(self) -> None:
        """Redraw the current measurement and its results in the current theme (without measuring again)."""
        for w in (self.signals, self.frf_view, self.stab_plot, self.results.mac):
            w.apply_theme()
        if self.acq is not None and not self._dirty:  # a new chain while hidden: measured again when it shows
            self.check.setHtml(self._setup_check())
            self.redraw()

    # ------------------------------------------------------------ inputs
    def set_system(self, system: ChainSystem, result: ModalResult | None = None) -> None:
        """New chain parameters. Measured now if visible, else when the page is shown."""
        self.system = system
        self.result = result
        self._dirty = True
        self._fill_dofs(system.n)
        if self.isVisible():
            self.restart()

    def _fill_dofs(self, n: int) -> None:
        """One entry per mass in the force's and the response's combos (kept if they already have them)."""
        if self.input.count() == n:
            return
        for combo in (self.input, self.output):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems([f"m{j + 1}" for j in range(n)])
            combo.setCurrentIndex(n - 1)
            combo.blockSignals(False)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._dirty:
            self.restart()
        elif self.running:
            self._frame.start()  # resume the record where it was, not after the time hidden
            self._timer.start()

    @property
    def running(self) -> bool:
        """Still measuring, or still drawing the last record."""
        return self.acq is not None and (not self.acq.done or self._shown is not None)

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._timer.stop()

    def settings(self) -> MeasurementSettings:
        n = self.system.n if self.system else 1
        return MeasurementSettings(
            excitation=self.excitation.currentData(),
            input_dof=min(max(0, self.input.currentIndex()), n - 1),
            response=self.response.currentData(),
            fs=self.fs.value(),
            block=self.block.currentData(),
            averages=self.averages.value(),
            overlap=self.overlap.currentData(),
            anti_alias=self.anti_alias.isChecked(),
            tip_width=self.tip.value() * 1e-3,
            burst=self.burst.value() / 100.0,
            sine_points=self.points.value(),
            force_level=self.force_level.value() / 100.0,
            seed=self.seed,
        )

    def processing(self) -> Processing:
        return Processing(
            window=self.taper.currentData(),
            exp_end=self.exp_end.value() / 100.0,
            estimator=self.estimator_combo.currentData(),
            force_noise=self.force_noise.value() / 100.0,
            response_noise=self.response_noise.value() / 100.0,
        )

    def _on_hold(self, on: bool) -> None:
        """Hold the finished FRF now shown (or let go of it)."""
        if not on:
            self.held = None
        elif self.estimate is None or self.acq is None or not self.acq.done:
            self.hold.setChecked(False)  # nothing complete to hold yet
            return
        else:
            self.held = (self.estimate.freqs, self.estimate.H, self.acq.settings)
        self.redraw()

    def _held_label(self) -> str:
        assert self.held is not None, "asked for only while one is held"
        s = self.held[2]
        return f"Held: {s.excitation.value.lower()}, force level {100 * s.force_level:.0f}%"

    def _show_held(self, j: int) -> None:
        assert self.acq is not None, "a test is set up"
        s = self.acq.settings
        if self.held is None or (self.held[2].input_dof, self.held[2].response) != (s.input_dof, s.response):
            self.frf_view.set_held(None, None)
        else:
            self.frf_view.set_held(self.held[0], self.held[1][:, j], self._held_label())

    def _level_text(self, s: MeasurementSettings) -> str:
        e = s.excitation
        if e is Excitation.IMPACT:
            return f"{s.force_level * IMPACT_PEAK:.4g} N peak"
        if e is Excitation.STEPPED_SINE:
            return f"{s.force_level * SINE_AMPLITUDE:.4g} N amplitude"
        return f"{s.force_level * FORCE_RMS:.4g} N RMS"

    def _on_excitation(self) -> None:
        self._select_window(self.excitation.currentData())
        self._show_rows()
        self.restart()

    def _select_window(self, excitation: Excitation) -> None:
        self.taper.blockSignals(True)
        self.taper.setCurrentIndex(list(Window).index(RECOMMENDED_WINDOW[excitation]))
        self.taper.blockSignals(False)

    def _show_rows(self) -> None:
        e = self.excitation.currentData()
        rows = {
            (self.tip_label, self.tip): e is Excitation.IMPACT,
            (self.burst_label, self.burst): e is Excitation.BURST_RANDOM,
            (self.overlap_label, self.overlap): e is Excitation.RANDOM,
            (self.points_label, self.points): e is Excitation.STEPPED_SINE,
            (self.averages_label, self.averages): e is not Excitation.STEPPED_SINE,
            (self.block_label, self.block): e is not Excitation.STEPPED_SINE,
            (self.exp_label, self.exp_end): self.taper.currentData() is Window.FORCE_EXPONENTIAL,
        }
        for widgets, on in rows.items():
            for w in widgets:
                w.setVisible(on)
        self.taper.setEnabled(e is not Excitation.STEPPED_SINE)

    def load_trial(self, index: int) -> None:
        """Set up TRIALS[index]: its chain (through the Simulation page) and every control, then measure."""
        self.apply_trial(TRIALS[index])

    def load_setup(self, setup: dict) -> None:
        """Set the page up as a practice problem's `setup` says (a trial's fields, with no note). It measures
        when shown, not here."""
        self.apply_trial(ModalTestTrial("", "", **setup), measure_hidden=False)

    def practice_state(self) -> ModalTestTrial:
        """The chain and every control, as a trial: what a practice spec checks (from the settings alone)."""
        s = self.system
        return ModalTestTrial(
            "",
            "",
            input=self.input.currentIndex(),
            excitation=self.excitation.currentData(),
            response=self.response.currentData(),
            window=self.taper.currentData(),
            block=self.block.currentData(),
            fs=None if self.fs_auto.isChecked() else self.fs.value(),
            anti_alias=self.anti_alias.isChecked(),
            force_noise=self.force_noise.value(),
            response_noise=self.response_noise.value(),
            method=self.extract.method.currentData(),
            friction=float(s.friction.max()) if s is not None else 0.0,
            tab=self.tabs.currentIndex(),
            masses=tuple(float(v) for v in s.masses) if s is not None else (),
            springs=tuple(float(v) for v in s.stiffness) if s is not None else (),
            dampers=tuple(float(v) for v in s.damping) if s is not None else (),
            averages=self.averages.value(),
            tip=self.tip.value(),
            exp_end=self.exp_end.value(),
        )

    def apply_trial(self, trial: ModalTestTrial, measure_hidden: bool = True) -> None:
        """Set every control as `trial` says and its chain (through the Simulation page), then measure: now, or,
        if hidden and not `measure_hidden`, once shown."""
        self.trial_note.setText(trial.note)
        self.trial_note.setVisible(bool(trial.note))
        values = (
            (self.excitation, trial.excitation),
            (self.response, trial.response),
            (self.block, trial.block),
            (self.estimator_combo, Estimator.H1),
            (self.extract.method, trial.method),
        )
        widgets = [w for w, _ in values] + [
            self.input,
            self.taper,
            self.fs,
            self.fs_auto,
            self.anti_alias,
            self.force_noise,
            self.response_noise,
            self.force_level,
            self.exp_end,
            self.averages,
            self.tip,
        ]
        for w in widgets:
            w.blockSignals(True)
        for combo, value in values:
            combo.setCurrentIndex(combo.findData(value))
        if self.input.count() > trial.input:
            self.input.setCurrentIndex(trial.input)
        self._select_window(trial.excitation)
        if trial.window is not None:
            self.taper.setCurrentIndex(self.taper.findData(trial.window))
        self.fs_auto.setChecked(trial.fs is None)
        if trial.fs is not None:
            self.fs.setValue(trial.fs)
        self.anti_alias.setChecked(trial.anti_alias)
        self.force_noise.setValue(trial.force_noise)
        self.response_noise.setValue(trial.response_noise)
        self.force_level.setValue(100.0)
        self.exp_end.setValue(trial.exp_end)
        self.averages.setValue(trial.averages)
        self.tip.setValue(trial.tip)
        for w in widgets:
            w.blockSignals(False)
        if self.hold.isChecked():
            self.hold.setChecked(False)
        self.extract.changed.emit()
        self._show_rows()
        measured = self.acq
        system = trial.system()
        self.system_requested.emit(system)
        if self.system is None or not _same_chain(self.system, system):
            self.set_system(system)  # nobody passed it back: test it here alone
        if self.input.currentIndex() != trial.input:  # the chain's size changed and reset it
            self.input.blockSignals(True)
            self.input.setCurrentIndex(trial.input)
            self.input.blockSignals(False)
            measured = self.acq  # measure again below, with the force where the trial puts it
        if self.acq is measured:  # nothing measured since the controls changed
            if self.isVisible() or measure_hidden:
                self.restart()
            else:
                self._dirty = True
        self.tabs.setCurrentIndex(trial.tab)

    def _on_fs_auto(self, on: bool) -> None:
        self.fs.setEnabled(not on)
        if on:
            self.restart()

    def _on_averages(self) -> None:
        if self.acq is None:
            return
        self.acq.settings.averages = self.averages.value()
        self.redraw()
        if self.running and self.isVisible():
            self._timer.start()

    def _on_speed(self) -> None:
        save_playback(self.speed.currentText())
        if self.speed.currentData() is None and self._shown is not None:  # finish the record now
            self._shown = None
            self.redraw()
        if self.running and self.isVisible():
            self._frame.start()
            self._timer.start()

    def _reprocess(self) -> None:
        self._show_rows()
        if self.acq is not None:
            self.estimator = FrfEstimator(self.acq, self.processing())
            if self._shown is not None:  # the record being drawn, with the new noise
                self._live = self.estimator.estimate()
        self.redraw()

    # ----------------------------------------------------------- measure
    def restart(self) -> None:
        """Start a new measurement with the current settings."""
        if self.system is None:
            return
        self._dirty = False
        system = self.system
        n = system.n
        if self.result is None:
            self.result = modal_analysis(system)
        self.dof.blockSignals(True)
        self.dof.setValue(n)
        self.dof.blockSignals(False)
        self._fill_dofs(n)
        self.fs.setEnabled(not self.fs_auto.isChecked())
        if self.fs_auto.isChecked():
            self.fs.blockSignals(True)
            self.fs.setValue(suggested_fs(self.result))
            self.fs.blockSignals(False)
        self.acq = Acquisition(system, self.settings(), self.result)
        self.estimator = FrfEstimator(self.acq, self.processing())
        s = self.acq.settings
        self.force_level_info.setText(self._level_text(s))
        self.extract.set_band(0.0, s.band, top=s.fs / 2)
        self.frf_view.set_fit_band(0.0, s.band)
        self.picked = None
        self._live, self._shown = None, None
        self.signals.clear_measurement()
        self.frf_view.clear_measurement()
        self.identify()
        self._update_info()
        self.check.setHtml(self._setup_check())
        self._measure_some()
        if self.running and self.isVisible():
            self._timer.start()

    def _measure_frame(self) -> None:
        with frame_stats.frame("Virtual modal test", target_ms=FRAME_MS):
            self._measure_some()
            settle_plots(self)

    def _measure_some(self) -> None:
        if self.acq is None:
            return
        speed = self.speed.currentData()
        if speed is not None:
            self._play(speed)
            return
        self._clock.start()
        before = self.acq.progress[0]
        while not self.acq.done:
            left = STEP_BUDGET_MS - self._clock.elapsed()
            if left <= 0 or not self.acq.work(left * 1e-3):
                break  # a slow (friction) record carries on next frame
        if self.acq.done:
            self._timer.stop()
        if self.acq.progress[0] != before or self.acq.done or self.estimate is None:
            self.redraw()
        else:
            done, wanted = self.acq.progress
            self.progress.setText(f"{done} / {wanted} {self._noun} · measuring {done + 1} …")

    def _play(self, speed: float) -> None:
        """Paced playback: measure one record, then draw it over its duration / speed."""
        assert self.acq is not None and self.estimator is not None, "a test is set up"
        if self._shown is None:
            if self.acq.done:
                self._timer.stop()
                return
            first = self.acq.progress[0] == 0
            if not self.acq.work(STEP_BUDGET_MS * 1e-3):  # a slow (friction) record: carry on next frame
                done, wanted = self.acq.progress
                self.progress.setText(f"{done} / {wanted} {self._noun} · measuring {done + 1} …")
                return
            self._live = self.estimator.estimate()
            s = self.acq.settings
            # Continuous random: the part a block shares with the one before was drawn already.
            overlap = s.block - self.acq.hop if s.excitation is Excitation.RANDOM and not first else 0
            self._shown = overlap / s.fs
            self._frame.start()
        else:
            self._shown += self._frame.restart() * 1e-3 * speed
        est = self._live
        assert est is not None, "the record drawn was estimated first"
        if self._shown >= est.t[-1]:
            self._shown = None
            self.redraw()
            if self.acq.done:
                self._timer.stop()
            return
        self.signals.set_time(est, max(0, self.output.currentIndex()), self.acq.stepped, shown=self._shown)
        done, wanted = self.acq.progress
        self.progress.setText(f"{done - 1} / {wanted} {self._noun} · recording {done} …")

    @property
    def _noun(self) -> str:
        assert self.acq is not None, "a test is set up"
        return "frequencies" if self.acq.stepped else "averages"

    def redraw(self) -> None:
        if self.acq is None or self.estimator is None:
            return
        done, wanted = self.acq.progress
        self.progress.setText(f"{done} / {wanted} {self._noun}" + ("" if done >= wanted else " …"))
        est = self.estimator.estimate()
        if est is None:
            return
        j = max(0, self.output.currentIndex())
        s = self.acq.settings
        self.signals.set_data(est, j, s, self.acq.stepped)
        self.frf_view.set_data(est, j, self.acq.system, self.acq.result, s, self.processing(), self.acq.stepped)
        self._show_held(j)
        self.estimate = est
        self.identify()

    # ------------------------------------------------------------ extract
    def _on_extraction_changed(self) -> None:
        lo, hi = self.extract.band
        self.frf_view.set_fit_band(lo, hi)
        self.picked = None
        self.identify()

    def _on_fit_band(self, lo: float, hi: float) -> None:
        self.extract.set_band(lo, hi)
        self.picked = None
        self.identify()

    def _on_auto_poles(self) -> None:
        self.picked = None
        self.identify()

    def _on_clear_poles(self) -> None:
        self.picked = []
        self.identify()

    def _on_pole_clicked(self, order: int, i: int) -> None:
        if self.stab is None:
            return
        chosen = list(self.poles)
        if (order, i) in chosen:
            chosen.remove((order, i))
        else:
            chosen.append((order, i))
        self.picked = [(o, abs(self.stab.pole(o, k)) / (2 * math.pi)) for o, k in chosen]
        self.identify()

    def _pick(self, stab: Stabilization) -> list[tuple[int, int]]:
        """The poles to use: automatic, or the ones picked by hand found again in a new diagram."""
        if self.picked is None:
            return auto_select(stab)
        out = []
        for order, f in self.picked:
            # The same order if the pole is still there, else the nearest order that has it.
            for o in sorted(stab.orders, key=lambda o: abs(o - order)):
                fs = np.abs(stab.poles[stab.orders.index(o)]) / (2 * math.pi)
                if fs.size and abs(fs - f).min() <= 0.02 * f:
                    out.append((o, int(np.argmin(abs(fs - f)))))
                    break
        return out

    def identify(self) -> None:
        """Extract modal parameters from the finished measurement and compare them with the exact modes."""
        est, acq = self.estimate, self.acq
        p = self.processing()
        exp_window = p.window is Window.FORCE_EXPONENTIAL and p.exp_end < 1.0 and not (acq and acq.stepped)
        self.extract.set_window_correction(exp_window)
        if est is None or acq is None or not acq.done:
            self.stab, self.poles, self.ident = None, [], None
            self.frf_view.set_fit(None, None)
            self.results.set_results(
                [], np.zeros((0, 0)), "Modal parameters are extracted when the measurement is complete."
            )
            self.stab_plot.set_data(None, [], np.zeros(0), np.zeros(0), [])
            return
        method = self.extract.current
        band = self.extract.band
        n = acq.n
        response = acq.settings.response
        H = to_receptance(est.freqs, est.H, response)  # the methods fit receptance
        self.stab, self.poles = None, []
        if not np.isfinite(H[band_mask(est.freqs, band)]).all():  # e.g. a hit shorter than a sample, unfiltered
            self.ident = None
            self.frf_view.set_fit(None, None)
            self.results.set_results(
                [],
                np.zeros((0, 0)),
                "No FRF to fit: the force measured nothing at "
                "some lines in the band (a hammer pulse shorter than a sample, with no "
                "anti-alias filter to spread it, misses every sample).",
            )
            self.stab_plot.set_data(None, [], np.zeros(0), np.zeros(0), [])
            return
        if method is Method.PEAK:
            ident = peak_picking(est.freqs, H, band, n)
        elif method is Method.CIRCLE:
            ident = circle_fit(est.freqs, H, band, n)
        else:
            self.stab = lscf(est.freqs, H, band, self.extract.order.value())
            self.poles = self._pick(self.stab)
            ident = lsfd(est.freqs, H, band, [self.stab.pole(o, i) for o, i in self.poles])
        self.ident = ident
        notes = list(ident.notes)
        if response is Response.ACCELERATION:
            notes.append("Fitted to the receptance: the measured accelerance divided by −ω².")
        reported = ident
        if exp_window and self.extract.correct.isChecked():
            sigma = 1.0 / p.exp_tau(acq.settings)
            reported = ident.shifted(sigma)
            notes.append(f"Poles moved {sigma:.3g} 1/s to the right to remove the exponential window's damping.")
        elif exp_window:
            notes.append("<b>Not corrected</b> for the exponential window: every ζ includes its extra damping.")

        exact = [m.damped for m in acq.result.modes if m.damped is not None]
        modes = sorted(reported.modes, key=lambda m: m.fn_hz)
        rows = match_modes(modes, acq.result, band[1])
        found = sum(1 for r in rows if r.mode and r.identified)
        head = (
            f"<b>{method.value}</b>: {len(modes)} mode{'s' if len(modes) != 1 else ''} identified, "
            f"{found} matched to exact modes."
        )
        self.results.set_results(rows, mac_matrix(modes, acq.result), head + "<br>" + "<br>".join(notes))
        self.results.mac.set_matrix(
            mac_matrix(modes, acq.result), [f"{m.fn_hz:.3g} Hz" for m in modes], [str(m.index) for m in exact]
        )

        j = max(0, self.output.currentIndex())
        lo, hi = band
        if self.extract.show_fit.isChecked() and ident.modes:
            f = np.linspace(max(lo, hi / 1000), hi, 800)
            self.frf_view.set_fit(f, from_receptance(f, ident.synthesize(f)[:, j], response))
        else:
            self.frf_view.set_fit(None, None)
        mask = band_mask(est.freqs, band)
        self.stab_plot.set_data(
            self.stab, self.poles, est.freqs[mask], mode_indicator(H[mask]), [d.fn_hz for d in exact]
        )

    # -------------------------------------------------------------- text
    def _update_info(self) -> None:
        assert self.acq is not None, "a test is set up"
        s, a = self.acq.settings, self.acq
        if a.stepped:
            text = f"{s.sine_points} frequencies, {s.band / s.sine_points:.4g} Hz apart<br>"
        else:
            text = f"T = Nb/fs = {s.duration:.4g} s · Δf = 1/T = {s.resolution:.4g} Hz<br>"
        text += f"Nyquist fs/2 = {s.fs / 2:.4g} Hz · passband and excitation to {s.band:.4g} Hz"
        if s.excitation.settles:
            text += f"<br>Waits {a.settle:.3g} s for steady state (transients down to 0.1%)"
        elif s.excitation.waits:
            text += f"<br>Waits {a.settle:.3g} s between hits for the response to die away (to 0.1%)"
        self.daq_info.setText(text)

    def _setup_check(self) -> str:
        """The settings against the chain's modes: what each mode will suffer from."""
        assert self.acq is not None, "a test is set up"
        s, a = self.acq.settings, self.acq
        T, nyq = s.duration, s.fs / 2
        df = s.band / s.sine_points if a.stepped else s.resolution
        rows = []
        warnings = []
        columns = 5 + (0 if a.stepped else 1) + (1 if s.excitation is Excitation.IMPACT else 0)
        for m in a.result.modes:
            d = m.damped
            if d is None:
                continue
            color = colors.mode[(m.index - 1) % len(colors.mode)]
            bw = 2.0 * d.zeta * d.fn_hz
            lines = bw / df
            left = math.exp(d.eigenvalue.real * T)
            notes = []
            if d.fd_hz > nyq:
                alias = abs(d.fd_hz - s.fs * round(d.fd_hz / s.fs))
                notes.append(
                    "above Nyquist: "
                    + ("removed by the filter" if s.anti_alias else f"<b>aliases to {alias:.3g} Hz</b>")
                )
            elif d.fd_hz > s.band:
                notes.append(
                    "in the filter's roll-off"
                    if s.anti_alias
                    else f"above the usable band ({AA_CUTOFF:.0%} of Nyquist)"
                )
            if lines < 2.0 and d.fd_hz <= nyq:
                notes.append(
                    "<b>under-resolved</b>: "
                    + ("the peak falls between frequencies" if a.stepped else "peak too low, looks too damped")
                )
            if s.excitation in (Excitation.IMPACT, Excitation.BURST_RANDOM) and left > 0.01:
                notes.append("<b>still ringing at block end</b>: leakage unless a window is used")
            tip = ""
            if s.excitation is Excitation.IMPACT:
                level = 20.0 * math.log10(max(float(impact_spectrum(s.tip_width, np.array([d.fd_hz]))[0]), 1e-12))
                tip = f"{round(level) + 0:.0f} dB"
                if level < -20.0:
                    notes.append("<b>hardly excited by this tip</b>: noise will dominate")
            rows.append(
                f"<tr><td style='white-space:nowrap; color:{color}'><b>{m.index}</b></td><td style='white-space:nowrap'>{d.fd_hz:.4g}</td>"
                f"<td style='white-space:nowrap'>{d.zeta:.4f}</td><td style='white-space:nowrap'>{bw:.3g}</td><td style='white-space:nowrap'>{lines:.3g}</td>"
                + ("" if a.stepped else f"<td style='white-space:nowrap'>{percent(left)}</td>")
                + (f"<td style='white-space:nowrap'>{tip}</td>" if s.excitation is Excitation.IMPACT else "")
                + "</tr>"
                + (f"<tr><td></td><td colspan='{columns - 1}'>{'; '.join(notes)}</td></tr>" if notes else "")
            )
        # Numbers and headers don't wrap (in a narrow pane they broke mid-number); a mode's notes, if
        # it has any, take a row of their own under it, across the table.
        head = (
            "<tr><th style='white-space:nowrap'>Mode</th><th style='white-space:nowrap'>f<sub>d</sub><br>[Hz]</th><th style='white-space:nowrap'>ζ</th>"
            "<th style='white-space:nowrap'>2ζf<sub>n</sub><br>[Hz]</th>"
            + (
                "<th style='white-space:nowrap'>Points<br>in it</th>"
                if a.stepped
                else "<th style='white-space:nowrap'>Lines<br>in it</th><th style='white-space:nowrap'>Left<br>at T</th>"
            )
            + ("<th style='white-space:nowrap'>Hammer<br>level</th>" if s.excitation is Excitation.IMPACT else "")
            + "</tr>"
        )
        e = s.excitation
        w = self.taper.currentData()
        if e is Excitation.IMPACT and not s.anti_alias and s.tip_width < 2.0 / s.fs:
            warnings.append(
                "The hit lasts less than 2 samples. Without the anti-alias filter to spread it "
                "out, the samples can miss most of it, and the measured force is wrong."
            )
        if e is Excitation.IMPACT and w is Window.HANN:
            warnings.append(
                "A Hann window is zero at the start of the block, where the hit is: it throws the force away."
            )
        if e is Excitation.RANDOM and w is Window.RECTANGULAR:
            warnings.append("Continuous random is not periodic in the block: without a window it leaks.")
        if e in (Excitation.PERIODIC_RANDOM, Excitation.CHIRP) and w is not Window.RECTANGULAR:
            warnings.append("The signals are periodic in the block, so no window is needed; one only blurs the peaks.")
        rubbing = friction_masses(a.system)
        if rubbing:
            warnings.append(
                f"Friction on {rubbing}: the chain is nonlinear, so what is measured is a linear "
                "approximation that depends on the force level and the excitation. <i>Exact</i> is the "
                "chain without friction. Friction damps small motions heavily, so a light force gives low, "
                "flat peaks (or none, if the masses hardly slide); a strong one approaches the exact FRF. "
                "Measure at two force levels with <i>Hold for comparison</i>, and watch the coherence. "
                "The response is simulated step by step, so measuring takes longer."
            )
        warn = "".join(f"<p style='color:{colors.fair}'>{t}</p>" for t in warnings)
        return (
            f"<p><b>{e.value}</b>, force at m{s.input_dof + 1}, {s.response.value.lower()} measured "
            f"({s.response.frf_name}). {EXCITATION_TIPS[e]}</p>{warn}"
            f"<table border='1' cellspacing='0' cellpadding='3' width='100%'>{head}{''.join(rows)}</table>"
            f"<p style='color:{colors.muted}'>2ζf<sub>n</sub> is the half-power bandwidth of each peak; "
            "with fewer than about 2 lines (Δf apart, or stepped-sine frequencies) across it, the peak is missed. "
            "<i>Left at T</i> is how much of a free decay is left at the end of a block "
            "(e<sup>σT</sup>): a hit or burst that has not died away is cut off, which leaks. A problem with a "
            "mode is noted on a row under it; a mode with nothing under it should measure well.</p>"
        )


THEORY_HTML = load_notes("modal_test", extraction=EXTRACTION_THEORY_HTML, trials=trials_html(TRIALS))
