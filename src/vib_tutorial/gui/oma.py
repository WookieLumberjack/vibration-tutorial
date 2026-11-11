"""Operational modal analysis page: the chain's modes from its response to ambient forces alone.

Left: the test (where the ambient forces act and their spectrum, a machine running, the
sensors), the acquisition (sample rate, record length, sensor noise) and the processing
(FDD + EFDD or SSI-cov, and each one's settings), and a few things to try. Centre: the
singular values of the responses' cross-spectral matrix against frequency, the peaks picked
(click to add or drop one) and their EFDD bells, over the exact natural frequencies, with
the peaks whose kurtosis says harmonic crossed; below, the responses. Right: the identified
modes against the exact ones, the chosen mode's autocorrelation with the decay fitted, SSI's
stabilization diagram, the three methods side by side (with the virtual modal test's hammer
test), and the notes. The page follows the chain (core/oma.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.identification import auto_select, mac_matrix, match_modes
from ..core.measurement import AA_CUTOFF, Response
from ..core.modal import ModalResult, modal_analysis
from ..core.model import ChainSystem
from ..core.oma import (
    HAMMER_AVERAGES,
    MAC_MIN,
    MAX_SAMPLES,
    AmbientRecord,
    AmbientSettings,
    Colour,
    Fdd,
    FddMode,
    SsiStabilization,
    ambient_record,
    cross_spectra,
    fdd,
    hammer_modes,
    harmonic_flags,
    harmonics,
    identify,
    kurtosis,
    pick_peaks,
    ssi_cov,
    ssi_modes,
)
from ..core.practice import Problem, Question, Requirement, at_least, at_most, kept, number, pick, rounded
from .layouts import ColumnScroll, FlowLayout, form_layout, group, plot_height
from .lesson import TrialCombo
from .links import load_notes, page_link
from .modal_extraction import ResultsView, StabilizationPlot, colored
from .modal_test import suggested_fs
from .notes_view import NotesBrowser
from .panels import spin
from .style import MAX_DOF, colors
from .theming import add_legend, mute

BLOCKS = (256, 512, 1024, 2048, 4096, 8192, 16384)
CYCLES = 2000  # the record's default length, in periods of the slowest mode
SHOWN_S = 60.0  # s of the record drawn
RECOMPUTE_MS = 150  # the parameters settle this long before the record is made again
FDD, SSI = "FDD + EFDD", "SSI-cov"

TIPS = {
    "input": "<p>Where the ambient forces act: independent random forces on every mass (wind, traffic, "
    "people: what OMA assumes), or on one mass only. A mode with a node there isn't excited, and "
    "no method finds it.</p>",
    "colour": "<p>The forces' spectrum. OMA assumes it is flat (white) over the modes, so the responses' "
    "spectra are the structure's. A low-pass input tilts them, and the modes above its corner "
    "fade into the noise.</p>",
    "response": "<p>What each sensor measures: displacement, or acceleration as accelerometers do (the "
    "usual in OMA). The shapes and frequencies don't depend on it; the higher modes stand out "
    "more in acceleration.</p>",
    "fs": "<p>Samples per second on every channel. Set from the chain when it changes, so the highest mode "
    f"is well inside the anti-alias filter's passband ({AA_CUTOFF:.0%} of Nyquist).</p>",
    "duration": "<p>How long the structure is recorded. OMA needs long records, a thousand periods of the "
    "slowest mode or more: the input is unknown and random, and only averaging many blocks makes "
    "the spectra smooth. Set from the chain when it changes.</p>",
    "noise": "<p>Random noise on each sensor, as a percentage of its channel's RMS. It fills the troughs "
    "between the peaks, and the weakest modes sink into it.</p>",
    "block": "<p>Samples per block in Welch's averaging. A long block resolves narrow peaks (the lines are "
    "fs/N<sub>b</sub> apart) and leaks less, so the damping comes out right; a short one gives more "
    "blocks to average, so the spectra are smoother. Set from the chain when it changes.</p>",
    "mac": "<p>EFDD: the lines round a peak whose first singular vector has a MAC with the peak's above "
    "this are the mode's bell. Higher keeps less of the bell; lower lets in the neighbours.</p>",
    "machine": "<p>A machine running on m1: a sine force at this frequency (0: none) and amplitude (the random "
    "forces' RMS is 1 N). Its response is a sharp peak, a pole with almost no damping: it looks like "
    "a lightly damped mode. A peak whose filtered response has a kurtosis near 1.5 (a sine's; a mode's "
    "random response has 3) is crossed out as a harmonic.</p>",
    "method": "<p><b>FDD + EFDD</b>: peaks of the cross-spectral matrix's first singular value, the damping from "
    "each peak's bell. <b>SSI-cov</b>: a state-space model fitted to the responses' correlations at "
    "every model order; stable poles are picked from the <i>Stabilization</i> tab.</p>",
    "rows": "<p>SSI: block rows of the correlation Toeplitz matrix, the lags it uses (1 to twice this). They "
    "should span a period or two of the slowest mode. Set from the chain when it changes.</p>",
    "order": "<p>SSI: the highest model order (states). Each mode takes two; the extra ones soak up noise.</p>",
    "new": "<p>Record again with fresh random forces and noise: the identified values scatter from record to "
    "record, damping most.</p>",
}
SV_TIP = (
    "<p>The singular values of the responses' cross-spectral matrix at each frequency line, largest "
    "(s₁, bold) first. Near a mode the matrix is that mode's alone, so s₁ peaks at every mode and the "
    "rest stay low (two close modes lift s₂ too). Dotted lines: the exact natural frequencies. The "
    "shaded bells are each mode's lines for EFDD. Click near a peak to add or drop it.</p>"
)

# Things to try: (name, what it sets, note)
TRIALS: tuple[tuple[str, dict[str, Any], str], ...] = (
    (
        "Long record, every mass",
        {},
        "Random forces at every mass, two thousand periods of the slowest mode: every mode found. The lightly "
        "damped ones' damping comes out within a few per cent; on the default chain, modes 3 and 4 are close "
        "and heavily damped, their bells overlap and cut each other short, and mode 4's damping comes out low.",
    ),
    (
        "Short record",
        {"cycles": 100},
        "A hundred periods: few blocks to average, so the spectra are rough and the damping scatters. "
        "Press New record a few times.",
    ),
    (
        "Short blocks",
        {"block": 1 / 8},
        "Blocks an eighth as long: smooth spectra but coarse lines, and the window's leakage widens every "
        "peak: the damping comes out too high, the lightest damped mode's most.",
    ),
    (
        "Forces at m3 only",
        {"input": 2},
        "On the default chain m3 sits on a node of mode 2: nothing excites it, so it is missing from s₁.",
    ),
    (
        "Coloured forces",
        {"corner": 1.5, "response": Response.DISPLACEMENT},
        "Forces low-passed at 1.5× the first mode, measured as displacements: the higher modes are barely "
        "driven and sink toward the noise, and the input's slope across their bells (OMA assumes it flat) "
        "spoils their damping, or loses the mode.",
    ),
    (
        "Noisy sensors",
        {"noise": 200.0},
        "Noise at twice each channel's RMS fills the troughs between the peaks: the more damped modes' bells "
        "drown in it, and their damping comes out far too low, or the mode is lost.",
    ),
    (
        "SSI on close, damped modes",
        {"method": SSI},
        "SSI fits every mode at once, so close, heavily damped modes don't cut each other's bells short: on the "
        "default chain, modes 3 and 4's damping comes out far closer than EFDD's (compare the Methods tab).",
    ),
    (
        "A machine running",
        {"machine": ("between", 1.0)},
        "A sine force on m1 halfway between modes 1 and 2: a sharp peak with almost no damping, like a lightly "
        "damped mode. Its filtered response has a kurtosis of about 1.5, a sine's, so it is crossed out.",
    ),
    (
        "A machine near a mode",
        {"machine": ("near", 3.0)},
        "The sine 2 % above mode 2: FDD's peak there is the sine's (kurtosis 1.5, no damping), and mode 2 hides "
        "under it. Switch to SSI-cov: it fits mode 2 beside the sine.",
    ),
)


@dataclass(frozen=True)
class OmaSetup:
    """A practice problem's setup: a chain (none: the one there is) and the page's settings. `input` is the
    forces' box: 0 every mass, i + 1 mass i."""

    masses: tuple[float, ...] = ()
    springs: tuple[float, ...] = ()
    dampers: tuple[float, ...] = ()
    input: int = 0
    colour: Colour = Colour.WHITE
    corner_hz: float = 5.0
    response: Response = Response.ACCELERATION
    machine_hz: float = 0.0
    machine_force: float = 1.0
    fs: float = 19.0
    duration: float = 1809.0
    noise: float = 2.0
    method: str = FDD
    block: int = 4096
    mac_min: float = MAC_MIN
    rows: int = 34
    max_order: int = 60
    tab: int = 0

    def system(self) -> ChainSystem | None:
        if not self.masses:
            return None
        return ChainSystem(np.array(self.masses), np.array(self.springs), np.array(self.dampers))


PRACTICE_SETUP = OmaSetup  # what a practice problem's setup and the page's practice_state are

# ------------------------------------------------------------------ practice problems
# Each question's numbers are printed round, and its answer is worked from them as printed.
TAB_MODES, TAB_EFDD, TAB_STABILIZATION, TAB_METHODS = range(4)
DEFAULT = dict(masses=(1.0,) * 4, springs=(400.0,) * 4, dampers=(2.0,) * 4)  # the uniform chain: 1.11 Hz first


def _uniform(n: int, c: float) -> dict:
    return dict(masses=(1.0,) * n, springs=(400.0,) * n, dampers=(c,) * n)


def _welch_blocks(samples: int, block: int) -> int:
    """Welch's half-overlapping blocks in `samples`, as `cross_spectra` counts them."""
    hop = max(1, round(block * 0.5))
    return (samples - block) // hop + 1 if samples >= block else 0


def _line_spacing(rng: np.random.Generator) -> Question:
    fs, block = pick(rng, (10, 19, 25, 50, 100)), int(pick(rng, BLOCKS[1:]))
    return Question(
        f"Responses sampled at {fs:g} Hz are cut into Welch blocks of {block} samples. How far apart are the "
        "lines of their cross-spectral matrix?",
        1000 * fs / block,
        f"Δf = f<sub>s</sub>/N<sub>b</sub> = {fs:g}/{block} = {number(fs / block)} Hz, {number(1000 * fs / block)} "
        "mHz: the record's block lasts N<sub>b</sub>/f<sub>s</sub>, and the lines are its reciprocal apart.",
        unit="mHz",
        setup=dict(**DEFAULT, fs=19.0, block=block),
    )


def _blocks(rng: np.random.Generator) -> Question:
    fs, block = pick(rng, (10, 19, 20, 50)), int(pick(rng, (1024, 2048, 4096)))
    minutes = pick(rng, (10, 20, 30, 60))
    samples = round(60 * minutes * fs)
    count = _welch_blocks(samples, block)
    return Question(
        f"A {minutes:g}-minute record sampled at {fs:g} Hz, Welch's blocks {block} samples long, overlapping by "
        "half. How many blocks does the estimate average?",
        count,
        f"{samples} samples; each block starts {block // 2} after the last, so (N − N<sub>b</sub>)/(N<sub>b</sub>/2) "
        f"+ 1 = ({samples} − {block})/{block // 2} + 1, rounded down: {count}.",
        unit="blocks",
        tolerance=0.005,
    )


def _record_length(rng: np.random.Generator) -> Question:
    fn, periods = pick(rng, (0.2, 0.5, 1.1, 2, 4)), pick(rng, (1000, 2000))
    return Question(
        f"A structure's slowest mode is at {fn:g} Hz. How many minutes of ambient record does OMA's rule of thumb, "
        f"{periods:g} periods of the slowest mode, ask for?",
        periods / fn / 60,
        f"{periods:g} periods of {number(1 / fn)} s: {number(periods / fn)} s, {number(periods / fn / 60)} min. The "
        "input is random and unknown, so only many averages make the spectra smooth.",
        unit="min",
    )


def _line_error(rng: np.random.Generator) -> Question:
    fn, fs, block = pick(rng, (1.1, 2.5, 4.0, 7.5)), pick(rng, (19, 25, 50)), int(pick(rng, (512, 1024, 2048)))
    worst = 100 * fs / block / 2 / fn
    return Question(
        f"FDD picks a mode at {fn:g} Hz to the nearest line, sampling at {fs:g} Hz with {block}-sample blocks. "
        "What is the largest error in its frequency, as a percentage of it?",
        worst,
        f"The lines are {number(fs / block)} Hz apart, so the nearest is at most half that, {number(fs / block / 2)} "
        f"Hz, off: {number(worst)} % of {fn:g} Hz. (EFDD's zero crossings refine it.)",
        unit="%",
    )


def _efdd_damping(rng: np.random.Generator) -> Question:
    zeta, n = pick(rng, (0.005, 0.01, 0.02, 0.05)), int(pick(rng, (3, 5, 10)))
    ratio = rounded(math.exp(math.tau * zeta * n / math.sqrt(1 - zeta * zeta)), 3)
    delta = math.log(ratio) / n
    z = delta / math.sqrt(4 * math.pi**2 + delta * delta)
    return Question(
        f"A mode's EFDD autocorrelation falls from one peak to the peak {n} periods later by a factor of "
        f"{number(ratio)}. What damping ratio does that give?",
        100 * z,
        f"As in a free decay: δ = ln({number(ratio)})/{n} = {number(delta)} per period, and ζ = δ/√(4π² + δ²) = "
        f"{number(100 * z)} % (δ/2π = {number(100 * delta / math.tau)} % for light damping).",
        unit="%",
        setup=dict(**DEFAULT, tab=TAB_EFDD),
    )


def _kurtosis(rng: np.random.Generator) -> Question:
    sine, noise = pick(rng, (0.5, 1, 2, 3)), pick(rng, (0.5, 1, 2))
    p = sine * sine / (noise * noise)
    k = (1.5 * p * p + 6 * p + 3) / (p + 1) ** 2
    return Question(
        f"Filtered round a peak, the response is a sine of {sine:g} RMS plus independent Gaussian noise of {noise:g} "
        "RMS. What is its kurtosis E[x⁴]/E[x²]²?",
        k,
        f"A sine's E[s⁴] is 1.5 σ<sub>s</sub>⁴ and a Gaussian's 3 σ<sub>g</sub>⁴; with the cross term 6 σ<sub>s</sub>²"
        f"σ<sub>g</sub>², and p = σ<sub>s</sub>²/σ<sub>g</sub>² = {number(p)}: K = (1.5p² + 6p + 3)/(p + 1)² = "
        f"{number(k)}, between the sine's 1.5 and a mode's 3.",
        setup=dict(**DEFAULT, machine_hz=2.144, machine_force=1.0),
    )


def _node(rng: np.random.Generator) -> Question:
    return Question(
        "On the uniform four-mass chain, mode 2 has a node at m3. With the ambient forces on m3 alone, what does FDD "
        "find of mode 2?",
        0,
        "Nothing: a force at a node does no work on the mode, so it isn't excited and s₁ has no peak there. "
        "OMA needs the forces spread over the structure.",
        choices=("Nothing: it isn't excited", "It, with too much damping", "It, as usual", "It, twice over"),
        setup=dict(**DEFAULT, input=3),
        predict=True,
    )


def _short_blocks(rng: np.random.Generator) -> Question:
    return Question(
        "EFDD's blocks are made short (512 samples at 19 Hz) on the uniform chain, its first mode lightly damped. "
        "What happens to that mode's damping?",
        0,
        "It comes out too high (about 3.3 % against 1.7 %): the lines are coarse beside the peak, and the Hann "
        "window's leakage widens it. Long blocks resolve it.",
        choices=("It comes out too high", "It comes out too low", "It comes out right, only noisier"),
        setup=dict(**DEFAULT, block=512, tab=TAB_EFDD),
        predict=True,
    )


def _machine(rng: np.random.Generator) -> Question:
    return Question(
        "A machine runs on m1, a sine between the uniform chain's first two modes. What does FDD make of it?",
        0,
        "A sharp peak with almost no damping, just like a very lightly damped mode. Its filtered response's "
        "kurtosis, about 1.5 (a sine's; a mode's is 3), gives it away, and the page crosses it out.",
        choices=(
            "A sharp, nearly undamped peak, like a mode",
            "Nothing: a sine isn't random",
            "A broad bump",
            "It shifts the modes beside it",
        ),
        setup=dict(**DEFAULT, machine_hz=2.144, machine_force=1.0),
        predict=True,
    )


def _harmonic_block(rng: np.random.Generator) -> Question:
    return Question(
        "With a machine's sine on the structure, Welch's block is doubled. What happens to the sine's peak in s₁, "
        "against a mode's?",
        0,
        "The sine's about doubles (3 → 5 → 12 → 23 → 38 from 512 to 8192 samples here): its power sits in one line "
        "however fine, so finer lines make it taller. A mode's is a density, and grows only until its peak is "
        "resolved.",
        choices=(
            "The sine's grows about twice; the mode's much less",
            "Both stay the same",
            "Both double",
            "The mode's grows; the sine's stays",
        ),
        setup=dict(**DEFAULT, machine_hz=2.144, machine_force=1.0, block=8192),
        predict=True,
    )


def _acquisition_spec(rng: np.random.Generator) -> Question:
    n, c = int(pick(rng, (2, 3, 4))), pick(rng, (0.5, 1.0, 2.0))
    averages = int(pick(rng, (15, 31)))
    chain = _uniform(n, c)
    result = modal_analysis(
        ChainSystem(np.array(chain["masses"]), np.array(chain["springs"]), np.array(chain["dampers"]))
    )
    slow = min(result.modes, key=lambda m: m.fn_hz)
    fs = suggested_fs(result)
    width = 2 * slow.zeta_modal * slow.fn_hz
    block = min(b for b in BLOCKS if fs / b <= width / 4)
    need = max(1000 / slow.fn_hz, (averages + 1) * (block // 2) / fs)
    need = math.ceil(need)
    while _welch_blocks(round(need * fs), block) < averages:
        need += 1
    longest = 10 * math.ceil(1.3 * need / 10)

    def goal(t: OmaSetup):
        system = t.system()
        modes = modal_analysis(system).modes if system is not None else []
        first = min(modes, key=lambda m: m.fn_hz) if modes else None
        f1 = first.fn_hz if first else math.nan
        lines = 2 * first.zeta_modal * f1 * t.block / t.fs if first else math.nan
        return (
            Requirement(f"{n} masses", f"{len(t.masses)}", len(t.masses) == n),
            *(
                kept(what, now[i] if len(now) > i else math.nan, given[i], unit)
                for what, now, given, unit in (
                    ("m", t.masses, chain["masses"], "kg"),
                    ("k", t.springs, chain["springs"], "N/m"),
                    ("c", t.dampers, chain["dampers"], "N·s/m"),
                )
                for i in range(n)
            ),
            kept("The sample rate", t.fs, fs, "Hz"),
            at_least("Lines across mode 1's half-power width", lines, 4),
            at_least("Periods of mode 1 recorded", t.duration * f1, 1000),
            at_least("Welch's averages", _welch_blocks(round(t.duration * t.fs), t.block), averages),
            at_most("The record's length", t.duration, longest, "s"),
        )

    return Question(
        f"The uniform chain of {n} masses (1 kg, 400 N/m, {c:g} N·s/m each): its first mode is at "
        f"{number(slow.fn_hz)} Hz with ζ = {number(100 * slow.zeta_modal)} %, sampled at {fs:g} Hz. Choose the "
        f"block and the record's length so that four or more lines fall across that mode's half-power width, the "
        f"record holds a thousand of its periods and {averages} or more Welch blocks, in no more than {longest:g} s.",
        0,
        f"The half-power width is 2ζf₁ = {number(width)} Hz, so the lines must be {number(width / 4)} Hz apart or "
        f"closer: a block of {number(4 * fs / width)} samples or more, {block} the shortest on offer. Half "
        f"overlapping, {averages} blocks need ({averages} + 1) × {block // 2} samples, "
        f"{number((averages + 1) * (block // 2) / fs)} s, and a thousand periods {number(1000 / slow.fn_hz)} s: "
        f"{need} s does both. A longer block needs twice the record.",
        setup=dict(**chain, fs=fs, duration=float(longest), block=BLOCKS[0]),
        goal=goal,
        design=dict(**chain, fs=fs, duration=float(need), block=block),
    )


PROBLEMS = tuple(
    Problem(key, title, "oma_page", f"See {page_link('oma_page', anchor, section)}.", make)
    for key, title, anchor, section, make in (
        ("line-spacing", "FDD's lines", "fdd", "Frequency domain decomposition", _line_spacing),
        ("blocks", "Welch's averages", "fdd", "Frequency domain decomposition", _blocks),
        ("record-length", "How long a record", "limits", "What OMA can't do", _record_length),
        ("line-error", "A frequency to the line", "fdd", "Frequency domain decomposition", _line_error),
        ("efdd-damping", "Damping from the bell", "fdd", "Frequency domain decomposition", _efdd_damping),
        ("kurtosis", "A sine in the noise", "harmonics", "Harmonics", _kurtosis),
        ("node", "Predict: forces at a node", "oma", "Operational modal analysis", _node),
        ("short-blocks", "Predict: short blocks", "limits", "What OMA can't do", _short_blocks),
        ("machine", "Predict: a machine running", "harmonics", "Harmonics", _machine),
        ("harmonic-block", "Predict: a longer block", "harmonics", "Harmonics", _harmonic_block),
        ("acquisition-spec", "Spec: a record and a block", "limits", "What OMA can't do", _acquisition_spec),
    )
)

NOTES_HTML = load_notes("oma")


def _plot(left: str, left_units: str | None, bottom: str, bottom_units: str | None) -> pg.PlotWidget:
    p = pg.PlotWidget()
    p.setMenuEnabled(False)
    p.hideButtons()
    p.showGrid(x=True, y=True, alpha=0.15)
    p.setLabel("left", left, units=left_units)
    p.setLabel("bottom", bottom, units=bottom_units)
    return p


def _form_row(form: QtWidgets.QFormLayout, text: str, box: QtWidgets.QWidget, tip: str) -> None:
    box.setToolTip(tip)
    label = QtWidgets.QLabel(text)
    label.setToolTip(tip)
    form.addRow(label, box)


def _cell(fn: float, zeta: float) -> str:
    return f"{fn:.4g} Hz\nζ {100 * zeta:.2g} %"


class OmaPage(QtWidgets.QWidget):
    """Controls on the left, singular values and records in the centre, results and notes on the right."""

    dof_requested = QtCore.Signal(int)  # the user changed N here; the main window owns the system
    system_requested = QtCore.Signal(object)  # a practice problem's chain, for the Simulation page
    edit_parameters = QtCore.Signal()  # the user wants the Simulation page's parameter panel

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.system: ChainSystem | None = None
        self.result: ModalResult | None = None
        self.record: AmbientRecord | None = None
        self.fdd: Fdd | None = None
        self.modes: list[FddMode] = []
        self.peaks: list[int] | None = None  # lines picked by hand; None: automatic
        self.kurt: list[float] = []  # each FDD peak's filtered kurtosis
        self.harmonic: list[bool] = []  # and whether it's a harmonic (core.oma.harmonics)
        self.stab: SsiStabilization | None = None
        self.ssi_picked: list[tuple[int, float]] | None = None  # SSI poles (order, Hz) picked by hand; None: automatic
        self.ssi_poles: list[tuple[int, int]] = []  # (order, index) of the SSI poles used
        self.ssi: list = []  # SSI's modes (IdentifiedMode)
        self.ssi_kurt: list[float] = []
        self.ssi_harmonic: list[bool] = []
        self._hammer: tuple[tuple, list] | None = None  # (the chain and fs it was for, its modes)
        self.seed = 1  # the record's random forces and noise; New record moves it on
        self.selected = 0  # the mode the EFDD tab shows
        self._loading = False
        self._dirty = False  # the record is stale (made again when the page shows)
        self._remeasure = True  # the next recompute makes a new record (not just reprocesses)
        self._recompute = QtCore.QTimer(self)
        self._recompute.setSingleShot(True)
        self._recompute.setInterval(RECOMPUTE_MS)
        self._recompute.timeout.connect(self.recompute)

        # --- left: the test, the acquisition, the processing
        test = QtWidgets.QGroupBox("Test")
        form = form_layout(test)
        self.dof = QtWidgets.QSpinBox()
        self.dof.setRange(1, MAX_DOF)
        self.dof.setToolTip("Same chain as the Simulation page; changing it here changes it there too.")
        self.dof.valueChanged.connect(self.dof_requested)
        form.addRow("Number of masses:", self.dof)
        self.input = QtWidgets.QComboBox()
        _form_row(form, "Ambient forces on:", self.input, TIPS["input"])
        self.colour = QtWidgets.QComboBox()
        for c in Colour:
            self.colour.addItem(c.value, c)
        self.corner = spin(0.01, 1e4, 5.0, 2, " Hz")
        _form_row(form, "Their spectrum:", group(self.colour, "corner", self.corner), TIPS["colour"])
        self.response = QtWidgets.QComboBox()
        for r in (Response.ACCELERATION, Response.DISPLACEMENT):
            self.response.addItem(r.value, r)
        _form_row(form, "Sensors:", self.response, TIPS["response"])
        self.machine_hz = spin(0.0, 1e4, 0.0, 3, " Hz")
        self.machine_force = spin(0.0, 1e6, 1.0, 2, " N")
        _form_row(form, "Machine on m1:", group(self.machine_hz, "at", self.machine_force), TIPS["machine"])
        link = QtWidgets.QLabel("Masses, springs and dampers are edited on the <a href='#sim'>Simulation page</a>.")
        mute(link)
        link.setWordWrap(True)
        link.linkActivated.connect(self.edit_parameters)
        form.addRow(link)

        daq = QtWidgets.QGroupBox("Acquisition")
        form = form_layout(daq)
        self.fs = spin(0.5, 10000.0, 20.0, 2, " Hz")
        self.duration = spin(1.0, 1e6, 900.0, 0, " s")
        self.noise = spin(0.0, 500.0, 2.0, 1, " %")
        _form_row(form, "Sample rate fs:", self.fs, TIPS["fs"])
        _form_row(form, "Record length:", self.duration, TIPS["duration"])
        _form_row(form, "Sensor noise:", self.noise, TIPS["noise"])
        self.new_record = QtWidgets.QPushButton("New record")
        self.new_record.setToolTip(TIPS["new"])
        form.addRow(self.new_record)
        self.daq_info = QtWidgets.QLabel()
        self.daq_info.setWordWrap(True)
        mute(self.daq_info)
        form.addRow(self.daq_info)

        proc = QtWidgets.QGroupBox("Processing")
        form = self.proc_form = form_layout(proc)
        self.method = QtWidgets.QComboBox()
        self.method.addItems([FDD, SSI])
        _form_row(form, "Method:", self.method, TIPS["method"])
        self.block = QtWidgets.QComboBox()
        for b in BLOCKS:
            self.block.addItem(str(b), b)
        _form_row(form, "Block length:", self.block, TIPS["block"])
        self.mac_min = spin(0.1, 0.999, MAC_MIN, 3, "")
        _form_row(form, "EFDD bell MAC ≥", self.mac_min, TIPS["mac"])
        self.rows = QtWidgets.QSpinBox()
        self.rows.setRange(2, 400)
        self.rows.setKeyboardTracking(False)
        _form_row(form, "SSI block rows:", self.rows, TIPS["rows"])
        self.max_order = QtWidgets.QSpinBox()
        self.max_order.setRange(2, 400)
        self.max_order.setSingleStep(2)
        self.max_order.setKeyboardTracking(False)
        _form_row(form, "SSI max order:", self.max_order, TIPS["order"])
        self.auto_peaks = QtWidgets.QPushButton("Pick automatically")
        self.auto_peaks.setToolTip(
            "<p>Forget the peaks (FDD) or poles (SSI) clicked by hand and pick them "
            "again: s₁'s most prominent peaks, at most one per mass, or one pole from "
            "each column of the stabilization diagram stable over five orders.</p>"
        )
        form.addRow(self.auto_peaks)

        trials = QtWidgets.QGroupBox("Things to try")
        tv = QtWidgets.QVBoxLayout(trials)
        self.trial = TrialCombo([name for name, _, _ in TRIALS], prompt=None)
        self.trial.setToolTip(
            "<p>Settings that show one of OMA's strengths or pitfalls, set from the chain as it is now.</p>"
        )
        tv.addWidget(self.trial)
        self.trial_note = QtWidgets.QLabel()
        self.trial_note.setWordWrap(True)
        mute(self.trial_note)
        tv.addWidget(self.trial_note)

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        for box in (test, daq, proc, trials):
            lv.addWidget(box)
        lv.addStretch(1)
        left_scroll = ColumnScroll(left)

        # --- centre: the singular values, the records
        self.sv = _plot("Singular value", None, "Frequency", "Hz")
        self.sv.setLogMode(x=False, y=True)
        self.sv.setToolTip(SV_TIP)
        self.sv.setMouseEnabled(x=False, y=False)
        self.sv.scene().sigMouseClicked.connect(self._on_sv_click)
        plot_height(self.sv, 220)
        self.sv_legend = add_legend(self.sv.getPlotItem(), offset=(-10, 10))
        self.readout = QtWidgets.QLabel()
        self.readout.setWordWrap(True)
        self.readout.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.records = _plot("Response", None, "Time", "s")
        self.records.setToolTip(
            "<p>The start of the record: every mass's response to the ambient forces "
            "(no force is measured). It looks like noise; the modes are in its spectra.</p>"
        )
        self.records.setClipToView(True)
        self.records.setDownsampling(auto=True, mode="peak")
        plot_height(self.records, 120)
        center = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(center)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.addWidget(self.sv, 3)
        cv.addWidget(self.readout)
        cv.addWidget(self.records, 1)

        # --- right: the results, the chosen mode's EFDD, the notes
        self.results = ResultsView()
        self.mode_pick = QtWidgets.QComboBox()
        self.mode_pick.setToolTip("<p>The identified mode whose EFDD fit is shown.</p>")
        self.correlation = _plot("Autocorrelation", None, "Lag", "s")
        self.correlation.setToolTip(
            "<p>The chosen mode's bell transformed back to the time domain: its autocorrelation, a decaying "
            "cosine, normalized to 1 at zero lag. The circled extremes give the decay (the dashed envelope "
            "e<sup>−ζω<sub>n</sub>τ</sup>), the zero crossings the damped frequency.</p>"
        )
        plot_height(self.correlation, 180)
        self.bell_plot = _plot("s₁", None, "Frequency", "Hz")
        self.bell_plot.setLogMode(x=False, y=True)
        self.bell_plot.setToolTip(
            "<p>The chosen mode's bell: the lines round its peak whose first singular "
            "vector looks like the peak's (MAC above the threshold).</p>"
        )
        plot_height(self.bell_plot, 140)
        self.efdd_note = QtWidgets.QLabel()
        self.efdd_note.setWordWrap(True)
        self.efdd_note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        efdd_tab = QtWidgets.QWidget()
        ev = QtWidgets.QVBoxLayout(efdd_tab)
        row = FlowLayout()
        row.add(group("Mode:", self.mode_pick))
        ev.addLayout(row)
        ev.addWidget(self.bell_plot, 2)
        ev.addWidget(self.correlation, 3)
        ev.addWidget(self.efdd_note)
        self.stab_plot = StabilizationPlot()
        self.stab_plot.setToolTip(
            "<p>SSI's poles at every model order, over s₁ (grey). Physical modes stay put "
            "as the order rises and form columns; noise poles scatter. Click a pole to "
            "add it or drop it.</p>"
        )
        self.methods = QtWidgets.QTableWidget(0, 5)
        self.methods.setHorizontalHeaderLabels(["Mode", "Exact", FDD, SSI, "Hammer test"])
        self.methods.verticalHeader().setVisible(False)
        self.methods.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.methods.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.methods.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.methods.setToolTip(
            "<p>Each exact mode as each method finds it: f<sub>n</sub> and ζ, coloured by the "
            "damping's error; hover for the frequency's error and the MAC.</p>"
        )
        self.methods_note = QtWidgets.QLabel()
        self.methods_note.setWordWrap(True)
        self.methods_note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        methods_tab = QtWidgets.QWidget()
        mv = QtWidgets.QVBoxLayout(methods_tab)
        mv.addWidget(self.methods, 1)
        mv.addWidget(self.methods_note)
        self.notes = NotesBrowser()
        self.notes.setHtml(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.efdd_tab = efdd_tab
        self.methods_tab = methods_tab
        self.tabs.addTab(self.results, "Modes")
        self.tabs.addTab(efdd_tab, "EFDD")
        self.tabs.addTab(self.stab_plot, "Stabilization")
        self.tabs.addTab(methods_tab, "Methods")
        self.tabs.addTab(self.notes, "Theory")

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(center)
        splitter.addWidget(self.tabs)
        splitter.setSizes([330, 740, 630])
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        # --- wiring
        for choice in (self.input, self.colour, self.response):
            choice.currentIndexChanged.connect(self._on_measure)
        for field in (self.corner, self.fs, self.duration, self.machine_hz, self.machine_force):
            field.valueChanged.connect(self._on_measure)
        self.noise.valueChanged.connect(self._on_process)
        self.block.currentIndexChanged.connect(self._on_process)
        self.mac_min.valueChanged.connect(self._on_process)
        self.rows.valueChanged.connect(self._on_process)
        self.max_order.valueChanged.connect(self._on_process)
        self.method.currentIndexChanged.connect(self._on_method)
        self.stab_plot.pole_clicked.connect(self._on_pole_clicked)
        self.tabs.currentChanged.connect(self._draw_methods)
        self._on_method(redraw=False)
        self.new_record.clicked.connect(self._on_new_record)
        self.auto_peaks.clicked.connect(self._on_auto_peaks)
        self.trial.picked.connect(self.apply_trial)
        self.mode_pick.currentIndexChanged.connect(self._on_mode_pick)

    # ------------------------------------------------------------- inputs
    def set_system(self, system: ChainSystem, result: ModalResult | None = None) -> None:
        """New chain parameters: the settings follow its modes, and it's recorded again (now if shown)."""
        n_changed = self.system is None or system.n != self.system.n
        self.system = system
        self.result = result or modal_analysis(system)
        self._loading = True
        self.dof.blockSignals(True)
        self.dof.setValue(system.n)
        self.dof.blockSignals(False)
        if n_changed:
            current = self.input.currentIndex()
            self.input.clear()
            self.input.addItem("Every mass")
            self.input.addItems([f"m{i + 1}" for i in range(system.n)])
            self.input.setCurrentIndex(current if 0 <= current <= system.n else 0)
        self._loading = False
        self.apply_trial()

    def _suggested(self) -> tuple[float, float, int]:
        """(fs, record length, block) for the chain's modes: fs clear of the top mode, a thousand periods
        of the slowest, and lines fine enough to put several across its half-power width."""
        assert self.result is not None, "worked out for the chain first"
        modes = [m for m in self.result.modes if m.damped is not None and m.fn_hz > 0]
        fs = suggested_fs(self.result)
        if not modes:
            return fs, 600.0, 2048
        slow = min(modes, key=lambda m: m.fn_hz)
        zeta = max(min(m.zeta_modal for m in modes if math.isfinite(m.zeta_modal)), 1e-3)
        duration = min(CYCLES / slow.fn_hz, MAX_SAMPLES / fs / 2)
        lines = fs / (0.5 * zeta * slow.fn_hz)  # samples per block for a line spacing of ζ f / 2
        block = min((b for b in BLOCKS if b >= lines), default=BLOCKS[-1])
        block = min(block, max(BLOCKS[0], 2 ** int(math.log2(duration * fs / 16))))
        return fs, round(duration), block

    def apply_trial(self) -> None:
        """Set the settings the chosen trial asks for, from the chain as it is."""
        if self.result is None or self.system is None:
            return
        _name, changes, note = TRIALS[self.trial.currentIndex()]
        fs, duration, block = self._suggested()
        freqs = sorted(m.fn_hz for m in self.result.modes if m.fn_hz > 0) or [1.0]
        slow = freqs[0]
        if "cycles" in changes:
            duration = round(changes["cycles"] / slow)
        if "block" in changes:
            block = max(BLOCKS[0], int(block * changes["block"]))
        self._loading = True
        self.fs.setValue(fs)
        self.duration.setValue(duration)
        self.block.setCurrentIndex(BLOCKS.index(block) if block in BLOCKS else 0)
        self.noise.setValue(changes.get("noise", 2.0))
        n = self.system.n
        self.input.setCurrentIndex(changes["input"] + 1 if "input" in changes and changes["input"] < n else 0)
        self.colour.setCurrentIndex(1 if "corner" in changes else 0)
        self.response.setCurrentIndex(self.response.findData(changes.get("response", Response.ACCELERATION)))
        self.corner.setValue(changes.get("corner", 1.5) * slow)
        self.corner.setEnabled("corner" in changes)
        machine_hz, force = 0.0, 1.0
        if "machine" in changes:
            where, force = changes["machine"]
            second = freqs[1] if len(freqs) > 1 else 2.0 * slow
            machine_hz = 0.5 * (slow + second) if where == "between" else 1.02 * second
        self.machine_hz.setValue(machine_hz)
        self.machine_force.setValue(force)
        self.method.setCurrentText(changes.get("method", FDD))
        self.rows.setValue(max(4, min(100, round(2.0 * fs / slow))))  # lags over two periods of the slowest
        self.max_order.setValue(min(15 * self.system.n, 120))
        self._loading = False
        self._on_method(redraw=False)
        self.trial_note.setText(note)
        self.peaks = None
        self._schedule(measure=True)

    def load_setup(self, setup: dict) -> None:
        """Set the page up as a practice problem's `setup` says: its chain (through the Simulation page), then
        the settings. The record is made when the page shows, not now."""
        t = OmaSetup(**setup)
        system = t.system()
        if system is not None:
            self.system_requested.emit(system)
            if self.system is None or not _same_chain(self.system, system):
                self.set_system(system)  # nobody passed it back: show it here alone
        if self.system is None:
            return
        self._loading = True
        self.input.setCurrentIndex(t.input if 0 <= t.input <= self.system.n else 0)
        self.colour.setCurrentIndex(self.colour.findData(t.colour))
        self.corner.setValue(t.corner_hz)
        self.corner.setEnabled(t.colour is Colour.LOW_PASS)
        self.response.setCurrentIndex(self.response.findData(t.response))
        self.machine_hz.setValue(t.machine_hz)
        self.machine_force.setValue(t.machine_force)
        self.fs.setValue(t.fs)
        self.duration.setValue(t.duration)
        self.noise.setValue(t.noise)
        self.method.setCurrentText(t.method)
        self.block.setCurrentIndex(BLOCKS.index(t.block))
        self.mac_min.setValue(t.mac_min)
        self.rows.setValue(t.rows)
        self.max_order.setValue(t.max_order)
        self._loading = False
        self._on_method(redraw=False)
        self.trial_note.setText("")
        self.tabs.setCurrentIndex(t.tab)
        self.peaks = None
        self.ssi_picked = None
        self._schedule(measure=True)

    def practice_state(self) -> OmaSetup:
        """The chain and the page's settings, as a setup: what a practice spec checks."""
        s = self.system
        return OmaSetup(
            tuple(float(v) for v in s.masses) if s is not None else (),
            tuple(float(v) for v in s.stiffness) if s is not None else (),
            tuple(float(v) for v in s.damping) if s is not None else (),
            self.input.currentIndex(),
            self.colour.currentData(),
            self.corner.value(),
            self.response.currentData(),
            self.machine_hz.value(),
            self.machine_force.value(),
            self.fs.value(),
            self.duration.value(),
            self.noise.value(),
            self.method.currentText(),
            self.block.currentData(),
            self.mac_min.value(),
            self.rows.value(),
            self.max_order.value(),
            self.tabs.currentIndex(),
        )

    def settings(self) -> AmbientSettings:
        i = self.input.currentIndex()
        return AmbientSettings(
            fs=self.fs.value(),
            duration=self.duration.value(),
            input_dof=None if i <= 0 else i - 1,
            colour=self.colour.currentData(),
            corner_hz=self.corner.value(),
            response=self.response.currentData(),
            harmonic_hz=self.machine_hz.value(),
            harmonic_force=self.machine_force.value(),
            seed=self.seed,
        )

    def _on_measure(self) -> None:
        if not self._loading:
            self.corner.setEnabled(self.colour.currentData() is Colour.LOW_PASS)
            self.peaks = None
            self.ssi_picked = None
            self._schedule(measure=True)

    def _on_method(self, *_, redraw: bool = True) -> None:
        ssi = self.method.currentText() == SSI
        self.proc_form.setRowVisible(self.mac_min, not ssi)  # the whole row, so it leaves no gap
        for w in (self.rows, self.max_order):
            self.proc_form.setRowVisible(w, ssi)
        # Not while a trial loads: it records again, and the results so far may be of another chain.
        if redraw and self.fdd is not None and not self._loading:
            self._show_results()

    def _on_process(self) -> None:
        if not self._loading:
            self._schedule(measure=False)

    def _on_new_record(self) -> None:
        self.seed += 1
        self._schedule(measure=True, now=True)

    def _on_auto_peaks(self) -> None:
        if self.method.currentText() == SSI:
            self.ssi_picked = None
        else:
            self.peaks = None
        self._process()

    def _on_pole_clicked(self, order: int, i: int) -> None:
        if self.stab is None:
            return
        picked = [(o, abs(self.stab.pole(o, k)) / (2 * math.pi)) for o, k in self.ssi_poles]
        f = abs(self.stab.pole(order, i)) / (2 * math.pi)
        near = [p for p in picked if abs(p[1] - f) <= 0.01 * f]
        if near:
            picked.remove(near[0])
        else:
            picked.append((order, f))
        self.ssi_picked = picked
        self.method.setCurrentText(SSI)
        self._process()

    def _pick(self, stab: SsiStabilization) -> list[tuple[int, int]]:
        """The SSI poles to use: automatic, or the ones picked by hand found again (the virtual modal test's way)."""
        if self.ssi_picked is None:
            return auto_select(stab)
        out = []
        for order, f in self.ssi_picked:
            for o in sorted(stab.orders, key=lambda o: abs(o - order)):
                fs = np.abs(stab.poles[stab.orders.index(o)]) / (2 * math.pi)
                if fs.size and abs(fs - f).min() <= 0.02 * f:
                    out.append((o, int(np.argmin(abs(fs - f)))))
                    break
        return out

    def _schedule(self, measure: bool, now: bool = False) -> None:
        self._remeasure |= measure
        if not self.isVisible():
            self._dirty = True
            return
        if now:
            self.recompute()
        else:
            self._recompute.start()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._dirty:
            self.recompute()

    # ------------------------------------------------------------- computing
    def recompute(self) -> None:
        """Record again if asked for (or stale), then process; the plots and results follow."""
        self._recompute.stop()
        self._dirty = False
        if self.system is None:
            return
        if self._remeasure or self.record is None:
            try:
                self.record = ambient_record(self.system, self.settings(), self.result)
            except ValueError as e:
                self.record = None
                self.readout.setText(f"<b>{e}</b>.")
                return
            self._remeasure = False
            self._draw_records()
        self._process()

    def _process(self) -> None:
        if self.record is None or self.system is None:
            return
        y = self.record.measured(self.noise.value() / 100.0)
        spectra = cross_spectra(y, self.record.settings.fs, self.block.currentData())
        self.fdd = fdd(spectra)
        band = self._band()
        most = self.system.n + (self.machine_hz.value() > 0)  # one per mass, and one for the machine
        lines = pick_peaks(self.fdd, band, most) if self.peaks is None else self.peaks
        self.modes = identify(self.fdd, lines, self.mac_min.value())
        fs = self.record.settings.fs
        self.kurt = harmonic_flags(y, fs, self.fdd, self.modes)
        self.harmonic = harmonics([m.zeta for m in self.modes], self.kurt)
        self.stab = ssi_cov(y, fs, self.rows.value(), self.max_order.value(), band)
        self.ssi_poles = self._pick(self.stab)
        self.ssi = ssi_modes(self.stab, self.ssi_poles)
        self.ssi_kurt = [kurtosis(y, fs, m.shape, m.fn_hz, 2.0 * m.zeta * m.fn_hz) for m in self.ssi]
        self.ssi_harmonic = harmonics([m.zeta for m in self.ssi], self.ssi_kurt)
        self._show_daq_info(spectra)
        self._fill_mode_pick()
        self._draw_sv()
        self._show_results()
        self._draw_efdd()
        self._draw_stabilization()
        self._draw_methods()

    def _band(self) -> tuple[float, float]:
        assert self.record is not None, "worked out for the chain first"
        return 0.0, AA_CUTOFF * self.record.settings.fs / 2.0

    def _on_sv_click(self, event) -> None:
        if self.fdd is None:
            return
        vb = self.sv.getPlotItem().getViewBox()
        if not vb.sceneBoundingRect().contains(event.scenePos()):
            return
        f = vb.mapSceneToView(event.scenePos()).x()
        freqs = self.fdd.freqs
        lines = [m.line for m in self.modes]
        near = [line for line in lines if abs(freqs[line] - f) <= 0.02 * max(f, freqs[1])]
        if near:
            lines.remove(near[0])  # drop the peak clicked
        else:
            # the highest s₁ within a few lines of the click
            k = int(np.argmin(np.abs(freqs - f)))
            lo, hi = max(1, k - 3), min(freqs.size, k + 4)
            lines.append(lo + int(np.argmax(self.fdd.s[lo:hi, 0])))
        self.peaks = sorted(set(lines))
        self._process()

    def _on_mode_pick(self, index: int) -> None:
        if index >= 0 and not self._loading:
            self.selected = index
            self._draw_efdd()
            self._draw_sv()

    # ------------------------------------------------------------- showing
    def apply_theme(self) -> None:
        self.results.mac.apply_theme()
        self.stab_plot.apply_theme()
        if self._dirty:  # a new chain while hidden: the record is the old chain's, made again when it shows
            return
        if self.record is not None:
            self._draw_records()
        if self.fdd is not None:
            self._draw_sv()
            self._show_results()
            self._draw_efdd()
            self._draw_stabilization()
            self._draw_methods()

    def _show_daq_info(self, spectra) -> None:
        assert self.record is not None and self.result is not None, "worked out for the chain first"
        s = self.record.settings
        slow = min((m.fn_hz for m in self.result.modes if m.fn_hz > 0), default=math.nan)
        self.daq_info.setText(
            f"{s.samples:,} samples a channel; {s.duration * slow:,.0f} periods of mode 1. "
            f"{spectra.blocks} blocks of {spectra.block} averaged (half overlapping), lines "
            f"{spectra.resolution:.3g} Hz apart."
        )

    def _fill_mode_pick(self) -> None:
        self._loading = True
        self.mode_pick.clear()
        self.mode_pick.addItems([f"{k + 1}: peak at {m.peak_hz:.4g} Hz" for k, m in enumerate(self.modes)])
        self.selected = min(self.selected, len(self.modes) - 1) if self.modes else 0
        self.mode_pick.setCurrentIndex(self.selected if self.modes else -1)
        self._loading = False

    def _draw_records(self) -> None:
        assert self.record is not None, "worked out for the chain first"
        p = self.records
        p.clear()
        rec = self.record
        y = rec.measured(self.noise.value() / 100.0)
        k = min(y.shape[0], int(SHOWN_S * rec.settings.fs))
        t = rec.t[:k]
        for j in range(y.shape[1]):
            p.plot(t, y[:k, j], pen=pg.mkPen(colors.mass[j % len(colors.mass)], width=1))
        p.setLabel("left", rec.settings.response.value, units=rec.settings.response.unit)
        p.setXRange(0.0, t[-1] if t.size else 1.0, padding=0)

    def _draw_sv(self) -> None:
        assert self.fdd is not None and self.result is not None, "worked out for the chain first"
        p = self.sv
        p.clear()
        self.sv_legend.clear()
        d = self.fdd
        f = d.freqs
        _lo, hi = self._band()
        keep = (f > 0) & (f <= hi)
        s = d.s[keep]
        floor = max(s[:, 0].max() * 1e-8, 1e-300)
        for i in range(d.s.shape[1] - 1, -1, -1):
            first = i == 0
            item = p.plot(
                f[keep],
                np.maximum(s[:, i], floor),
                pen=pg.mkPen(colors.strong if first else colors.grey, width=2 if first else 1),
            )
            if i < 2:
                self.sv_legend.addItem(item, "s₁" if first else "s₂, s₃, …")
        for k, m in enumerate(self.modes):
            color = colors.mode[k % len(colors.mode)]
            bell = m.bell & keep
            c = pg.mkColor(color)
            c.setAlpha(110 if k == self.selected else 60)
            # filled down to the floor (in the view's log10 units: the plot is log in y)
            p.plot(f[bell], np.maximum(d.s[bell, 0], floor), pen=None, fillLevel=math.log10(floor), brush=pg.mkBrush(c))
            if self.harmonic[k]:
                p.plot(
                    [f[m.line]],
                    [d.s[m.line, 0]],
                    pen=None,
                    symbol="x",
                    symbolSize=14,
                    symbolPen=pg.mkPen(colors.force, width=2),
                    symbolBrush=pg.mkBrush(colors.force),
                )
                right = f[m.line] > 0.6 * hi  # the label on the side with room
                text = pg.TextItem(
                    f"harmonic (kurtosis {self.kurt[k]:.2g})", color=colors.force, anchor=(1 if right else 0, 1)
                )
                text.setPos(f[m.line], math.log10(d.s[m.line, 0]))
                p.addItem(text)
            else:
                p.plot(
                    [f[m.line]],
                    [d.s[m.line, 0]],
                    pen=None,
                    symbol="o",
                    symbolSize=9,
                    symbolPen=pg.mkPen(color, width=2),
                    symbolBrush=pg.mkBrush(colors.background),
                )
        for md in self.result.modes:
            if md.damped is not None and md.damped.fn_hz <= hi:
                line = pg.InfiniteLine(
                    md.damped.fn_hz,
                    angle=90,
                    label=str(md.index),
                    labelOpts={"position": 0.95, "color": colors.muted},
                    pen=pg.mkPen(colors.muted, width=1, style=QtCore.Qt.PenStyle.DotLine),
                )
                p.addItem(line)
        p.setXRange(0.0, hi, padding=0)
        p.setYRange(math.log10(floor), math.log10(s[:, 0].max() * 3.0), padding=0)

    def _found(self, method: str) -> tuple[list, list[tuple[float, float]]]:
        """A method's modes (IdentifiedMode), and its peaks crossed out as harmonics: (Hz, kurtosis)."""
        if method == SSI:
            modes, flags, kurt = self.ssi, self.ssi_harmonic, self.ssi_kurt
        else:
            modes, flags, kurt = [m.identified for m in self.modes], self.harmonic, self.kurt
        kept = [m for m, h in zip(modes, flags) if not h]
        found = [(m.fn_hz, k) for m, h, k in zip(modes, flags, kurt) if h]
        return kept, found

    def _show_results(self) -> None:
        assert self.result is not None, "worked out for the chain first"
        f_top = self._band()[1]
        method = self.method.currentText()
        modes, harmonics = self._found(method)
        rows = match_modes(modes, self.result, f_top)
        macs = mac_matrix(modes, self.result)
        found = sum(1 for r in rows if r.mode is not None and r.identified is not None)
        missed = [r.mode for r in rows if r.mode is not None and r.identified is None]
        extra = sum(1 for r in rows if r.mode is None)
        notes = [f"{method}: {found} mode{'s' if found != 1 else ''} found"]
        if missed:
            notes.append("missed: " + ", ".join(f"mode {m}" for m in missed))
        if extra:
            notes.append(f"{extra} {'pole' if method == SSI else 'peak'}{'s' if extra > 1 else ''} matching no mode")
        if harmonics:
            notes.append("left out as harmonics: " + ", ".join(f"{f:.4g} Hz (kurtosis {k:.2g})" for f, k in harmonics))
        failed = sum(1 for m in self.modes if not math.isfinite(m.zeta)) if method == FDD else 0
        if failed:
            notes.append(
                f"{failed} bell{'s' if failed > 1 else ''} too narrow to fit a decay (the frequency is "
                f"the peak's line, ζ unknown)"
            )
        text = "; ".join(notes) + "."
        self.results.set_results(rows, macs, text)
        exact = [m for m in self.result.modes if m.damped is not None]
        self.results.mac.set_matrix(macs, [f"{m.fn_hz:.3g} Hz" for m in modes], [str(m.index) for m in exact])
        self.readout.setText(self._readout(method, rows, harmonics))

    def _readout(self, method: str, rows, harmonics) -> str:
        parts = []
        for r in rows:
            if r.mode is None or r.identified is None:
                continue
            fe = abs(r.exact) / (2 * math.pi)
            ze = -r.exact.real / abs(r.exact)
            ident = r.identified
            parts.append(
                f"mode {r.mode} {ident.fn_hz:.4g} Hz ({100 * (ident.fn_hz / fe - 1):+.1f} %), "
                f"ζ {100 * ident.zeta:.2g} % (exact {100 * ze:.2g} %)"
            )
        text = f"{method} found: " + "; ".join(parts) + "." if parts else f"{method} found no mode."
        if harmonics:
            text += (
                " <b>Harmonic"
                + ("s" if len(harmonics) > 1 else "")
                + "</b> at "
                + ", ".join(f"{f:.4g} Hz (kurtosis {k:.2g}; a mode's is 3)" for f, k in harmonics)
                + "."
            )
        return text

    def _draw_stabilization(self) -> None:
        if self.stab is None or self.fdd is None or self.result is None:
            return
        exact = [m.damped.fn_hz for m in self.result.modes if m.damped is not None]
        self.stab_plot.set_data(self.stab, self.ssi_poles, self.fdd.freqs, self.fdd.s[:, 0], exact)

    def _hammer_modes(self) -> list:
        assert self.system is not None and self.record is not None and self.result is not None, (
            "worked out for the chain first"
        )
        s = self.system
        key = (tuple(s.masses), tuple(s.stiffness), tuple(s.damping), self.record.settings.fs)
        if self._hammer is None or self._hammer[0] != key:
            modes, _ = hammer_modes(s, self.result, self.record.settings.fs, seed=1)
            self._hammer = (key, modes)
        return self._hammer[1]

    def _draw_methods(self) -> None:
        """The Methods tab, when it shows (the hammer test is measured then, once per chain)."""
        if self.tabs.currentWidget() is not self.methods_tab or self.fdd is None or self.result is None:
            return
        f_top = self._band()[1]
        exact = [(m, m.damped) for m in self.result.modes if m.damped is not None and m.damped.fn_hz <= f_top]
        columns = [self._found(FDD)[0], self._found(SSI)[0], self._hammer_modes()]
        matches = [
            {r.mode: r for r in match_modes(modes, self.result, f_top) if r.mode is not None} for modes in columns
        ]
        t = self.methods
        t.setRowCount(len(exact))
        for row, (m, damped) in enumerate(exact):
            fe, ze = damped.fn_hz, damped.zeta
            cells = [(str(m.index), colors.mode[(m.index - 1) % len(colors.mode)], ""), (_cell(fe, ze), None, "")]
            for found in matches:
                r = found.get(m.index)
                if r is None or r.identified is None:
                    cells.append(("missed", colors.poor, ""))
                    continue
                ident = r.identified
                text, color = colored(100 * (ident.zeta / ze - 1), 10.0, 30.0, _cell(ident.fn_hz, ident.zeta))
                tip = (
                    f"fₙ {100 * (ident.fn_hz / fe - 1):+.2f} %, ζ {100 * (ident.zeta / ze - 1):+.1f} %, MAC {r.mac:.3f}"
                )
                cells.append((text, color, tip))
            for c, (text, color, tip) in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                if color:
                    item.setForeground(pg.mkColor(color))
                if tip:
                    item.setToolTip(tip)
                t.setItem(row, c, item)
        t.resizeRowsToContents()
        self.methods_note.setText(
            "FDD + EFDD and SSI-cov from this record; the hammer test from the virtual modal test's way: "
            f"{HAMMER_AVERAGES} hits on m1, force and displacements measured, a force-exponential window, LSCF's "
            "stable poles and LSFD (on the chain alone: no machine, no ambient forces). Knowing the force, only the "
            "hammer test could scale its shapes to modal masses. Colour: the damping's error (green within 10 %, "
            "amber 30 %)."
        )

    def _draw_efdd(self) -> None:
        c, b = self.correlation, self.bell_plot
        c.clear()
        b.clear()
        if not self.modes:
            self.efdd_note.setText("No peak picked.")
            return
        m = self.modes[self.selected]
        color = colors.mode[self.selected % len(colors.mode)]
        d = self.fdd
        assert d is not None, "modes are found in the FDD"
        span = np.flatnonzero(m.bell)
        lo, hi = max(1, span[0] - span.size), min(d.freqs.size - 1, span[-1] + span.size)
        floor = max(d.s[lo : hi + 1, 0].max() * 1e-6, 1e-300)
        b.plot(d.freqs[lo : hi + 1], np.maximum(d.s[lo : hi + 1, 0], floor), pen=pg.mkPen(colors.grey, width=1))
        b.plot(d.freqs[m.bell], np.maximum(d.s[m.bell, 0], floor), pen=pg.mkPen(color, width=2))
        show = m.lag <= (m.lag[m.extremes[-1]] * 1.3 if m.extremes.size else m.lag[-1] * 0.1)
        c.plot(m.lag[show], m.correlation[show], pen=pg.mkPen(colors.strong, width=1.5))
        c.addItem(pg.InfiniteLine(0.0, angle=0, pen=pg.mkPen(colors.faint, width=1)))
        if m.extremes.size:
            c.plot(
                m.lag[m.extremes],
                m.correlation[m.extremes],
                pen=None,
                symbol="o",
                symbolSize=7,
                symbolPen=pg.mkPen(color, width=1.5),
                symbolBrush=pg.mkBrush(colors.background),
            )
        if math.isfinite(m.zeta):
            sigma = m.zeta * 2 * math.pi * m.fn_hz
            t = m.lag[show]
            env = np.exp(-sigma * t)
            # through the extremes' fitted line, not from 1: the bell's edges shift it a little
            a = float(np.exp(np.mean(np.log(np.abs(m.correlation[m.extremes])) + sigma * m.lag[m.extremes])))
            pen = pg.mkPen(color, width=1, style=QtCore.Qt.PenStyle.DashLine)
            c.plot(t, a * env, pen=pen)
            c.plot(t, -a * env, pen=pen)
            self.efdd_note.setText(
                f"Bell of {int(m.bell.sum())} lines; {m.extremes.size} extremes and {m.crossings.size} zero "
                f"crossings fitted: f<sub>n</sub> = {m.fn_hz:.4g} Hz, ζ = {100 * m.zeta:.3g} % (the peak's line: "
                f"{m.peak_hz:.4g} Hz)."
            )
        else:
            self.efdd_note.setText(
                f"Bell of {int(m.bell.sum())} lines: too few extremes to fit a decay. Lower the "
                f"MAC threshold, or lengthen the blocks."
            )
        c.setXRange(0.0, float(m.lag[show][-1]), padding=0)


def _same_chain(a: ChainSystem, b: ChainSystem) -> bool:
    return a.n == b.n and all(
        np.allclose(x, y) for x, y in ((a.masses, b.masses), (a.stiffness, b.stiffness), (a.damping, b.damping))
    )
