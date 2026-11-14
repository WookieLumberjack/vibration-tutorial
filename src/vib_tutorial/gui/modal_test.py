"""Virtual modal test page: measure the chain's FRF as in a lab and compare it with the exact one.

Left: the test (excitation, acquisition, noise, processing). Centre: the
latest recorded block and force spectrum above the estimated FRF and its
coherence. Right: a check of the settings against the chain's modes, and
theory. The data are measured a few blocks per frame, so the average can be
watched settling, or at a playback speed that draws each record as it is
recorded; changing only noise or processing reuses the same data.

Working on it: the test's settings (the Test, Acquisition, Measurement noise and Processing panes, and the
Setup tab's check) are `modal_test_setup.TestSetup`, which says `remeasure` or `reprocess`; the two plots are
`modal_test_views`' `SignalView` and `FrfView`. The page keeps the trials, the measuring loop (its own timer,
not a `FrameLoop`: it paces the measurement within `STEP_BUDGET_MS` a frame, and a playback speed draws each
record as it's recorded), hold, and the mode extraction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
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
    from_receptance,
    impact_spectrum,
    modal_analysis,
    to_receptance,
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
from ..core.measurement import AA_CUTOFF
from ..core.practice import Problem, Question, Requirement, at_least, at_most, number, pick
from . import frame_stats, modal_test_figures, settle_plots
from .layouts import ColumnScroll
from .lesson import TrialCombo, trials_html
from .links import load_notes, page_link
from .modal_extraction import ExtractionControls, ResultsView, StabilizationPlot
from .modal_test_setup import BLOCK_SIZES, TestSetup, setup_check, suggested_fs
from .modal_test_views import FrfView, SignalView
from .notes_view import NotesBrowser
from .theming import mute, settings

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
        "5 % noise on the force channel. H1 reads low at the resonances, where the structure barely resists the "
        "force and the force signal is smallest, and the coherence dips there. Switch the estimator to H2: "
        "the peaks come back.",
        force_noise=5.0,
    ),
    ModalTestTrial(
        "Response noise: the antiresonances",
        "5 % noise on the response channels. H1 stays unbiased; switch to H2 and it reads high, most at the "
        "antiresonances, where the response is small. Look at m1's FRF (Show response of).",
        response_noise=5.0,
    ),
    ModalTestTrial(
        "A short block: leakage",
        "<m>N<sub>b</sub></m> = 256 with no window: the hit is still ringing when the block ends, so the peaks leak, low and "
        "wide, and the damping comes out too high. Pick the Force + exponential window: the leakage goes, but "
        "every mode gains <m>Δζ = 1/τ<sub>w</sub>ω<sub>n</sub></m> (the dashed grey curve), which the extraction takes off "
        "again unless you untick <i>Remove the exponential window's damping</i>.",
        block=256,
        window=Window.RECTANGULAR,
    ),
    ModalTestTrial(
        "Peak picking on coarse lines",
        "Peak picking with <m>N<sub>b</sub></m> = 256: each frequency snaps to a spectral line and <m>ζ</m>, from the half-power "
        "points, comes out too high. Lengthen the block, or switch to the circle fit or LSCF.",
        block=256,
        method=Method.PEAK,
    ),
    ModalTestTrial(
        "Acceleration evens out the modes",
        "Periodic random with 2 % response noise, measuring displacement: the top modes sink into the noise "
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

        # --- the test's settings: the Test, Acquisition, Measurement noise and Processing panes
        self.test_setup = TestSetup()
        self.test_setup.dof_requested.connect(self.dof_requested)
        self.test_setup.edit_parameters.connect(self.edit_parameters)

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
        for w in (trials, self.test_setup):
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
        self.theory = NotesBrowser()
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
        self.test_setup.remeasure.connect(self.restart)
        self.test_setup.reprocess.connect(self._reprocess)
        self.test_setup.averages_changed.connect(self._on_averages)
        self.speed.currentIndexChanged.connect(self._on_speed)
        self.extract.changed.connect(self._on_extraction_changed)
        self.extract.auto_requested.connect(self._on_auto_poles)
        self.extract.clear_requested.connect(self._on_clear_poles)
        self.frf_view.fit_band_changed.connect(self._on_fit_band)
        self.stab_plot.pole_clicked.connect(self._on_pole_clicked)

    def apply_theme(self) -> None:
        """Redraw the current measurement and its results in the current theme (without measuring again)."""
        for w in (self.signals, self.frf_view, self.stab_plot, self.results.mac):
            w.apply_theme()
        if self.acq is not None and not self._dirty:  # a new chain while hidden: measured again when it shows
            self.check.setHtml(self._check())
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
        self.test_setup.fill_inputs(n)
        if self.output.count() == n:
            return
        self.output.blockSignals(True)
        self.output.clear()
        self.output.addItems([f"m{j + 1}" for j in range(n)])
        self.output.setCurrentIndex(n - 1)
        self.output.blockSignals(False)

    def settings(self) -> MeasurementSettings:
        """The test the controls set up, on the chain shown."""
        return self.test_setup.settings(self.system.n if self.system else 1, self.seed)

    def processing(self) -> Processing:
        return self.test_setup.processing()

    def _check(self) -> str:
        """The Setup tab: the settings against the chain's modes."""
        assert self.acq is not None, "a test is set up"
        return setup_check(self.acq, self.test_setup.taper.currentData())

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
            input=self.test_setup.input.currentIndex(),
            excitation=self.test_setup.excitation.currentData(),
            response=self.test_setup.response.currentData(),
            window=self.test_setup.taper.currentData(),
            block=self.test_setup.block.currentData(),
            fs=None if self.test_setup.fs_auto.isChecked() else self.test_setup.fs.value(),
            anti_alias=self.test_setup.anti_alias.isChecked(),
            force_noise=self.test_setup.force_noise.value(),
            response_noise=self.test_setup.response_noise.value(),
            method=self.extract.method.currentData(),
            friction=float(s.friction.max()) if s is not None else 0.0,
            tab=self.tabs.currentIndex(),
            masses=tuple(float(v) for v in s.masses) if s is not None else (),
            springs=tuple(float(v) for v in s.stiffness) if s is not None else (),
            dampers=tuple(float(v) for v in s.damping) if s is not None else (),
            averages=self.test_setup.averages.value(),
            tip=self.test_setup.tip.value(),
            exp_end=self.test_setup.exp_end.value(),
        )

    def apply_trial(self, trial: ModalTestTrial, measure_hidden: bool = True) -> None:
        """Set every control as `trial` says and its chain (through the Simulation page), then measure: now, or,
        if hidden and not `measure_hidden`, once shown."""
        self.trial_note.setText(trial.note)
        self.trial_note.setVisible(bool(trial.note))
        values = (
            (self.test_setup.excitation, trial.excitation),
            (self.test_setup.response, trial.response),
            (self.test_setup.block, trial.block),
            (self.test_setup.estimator_combo, Estimator.H1),
            (self.extract.method, trial.method),
        )
        widgets = [w for w, _ in values] + [
            self.test_setup.input,
            self.test_setup.taper,
            self.test_setup.fs,
            self.test_setup.fs_auto,
            self.test_setup.anti_alias,
            self.test_setup.force_noise,
            self.test_setup.response_noise,
            self.test_setup.force_level,
            self.test_setup.exp_end,
            self.test_setup.averages,
            self.test_setup.tip,
        ]
        for w in widgets:
            w.blockSignals(True)
        for combo, value in values:
            combo.setCurrentIndex(combo.findData(value))
        if self.test_setup.input.count() > trial.input:
            self.test_setup.input.setCurrentIndex(trial.input)
        self.test_setup._select_window(trial.excitation)
        if trial.window is not None:
            self.test_setup.taper.setCurrentIndex(self.test_setup.taper.findData(trial.window))
        self.test_setup.fs_auto.setChecked(trial.fs is None)
        if trial.fs is not None:
            self.test_setup.fs.setValue(trial.fs)
        self.test_setup.anti_alias.setChecked(trial.anti_alias)
        self.test_setup.force_noise.setValue(trial.force_noise)
        self.test_setup.response_noise.setValue(trial.response_noise)
        self.test_setup.force_level.setValue(100.0)
        self.test_setup.exp_end.setValue(trial.exp_end)
        self.test_setup.averages.setValue(trial.averages)
        self.test_setup.tip.setValue(trial.tip)
        for w in widgets:
            w.blockSignals(False)
        if self.hold.isChecked():
            self.hold.setChecked(False)
        self.extract.changed.emit()
        self.test_setup._show_rows()
        measured = self.acq
        system = trial.system()
        self.system_requested.emit(system)
        if self.system is None or not _same_chain(self.system, system):
            self.set_system(system)  # nobody passed it back: test it here alone
        if self.test_setup.input.currentIndex() != trial.input:  # the chain's size changed and reset it
            self.test_setup.input.blockSignals(True)
            self.test_setup.input.setCurrentIndex(trial.input)
            self.test_setup.input.blockSignals(False)
            measured = self.acq  # measure again below, with the force where the trial puts it
        if self.acq is measured:  # nothing measured since the controls changed
            if self.isVisible() or measure_hidden:
                self.restart()
            else:
                self._dirty = True
        self.tabs.setCurrentIndex(trial.tab)

    def _on_averages(self) -> None:
        if self.acq is None:
            return
        self.acq.settings.averages = self.test_setup.averages.value()
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
        self._fill_dofs(n)
        self.test_setup.follow(n, self.result)
        self.acq = Acquisition(system, self.settings(), self.result)
        self.estimator = FrfEstimator(self.acq, self.processing())
        s = self.acq.settings
        self.test_setup.show_level(s)
        self.extract.set_band(0.0, s.band, top=s.fs / 2)
        self.frf_view.set_fit_band(0.0, s.band)
        self.picked = None
        self._live, self._shown = None, None
        self.signals.clear_measurement()
        self.frf_view.clear_measurement()
        self.identify()
        self.test_setup.show_info(self.acq)
        self.check.setHtml(self._check())
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


THEORY_HTML = load_notes("modal_test", figures=modal_test_figures.FIGURES, trials=trials_html(TRIALS))
