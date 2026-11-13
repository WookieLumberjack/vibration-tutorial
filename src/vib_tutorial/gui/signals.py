"""Signals & spectra page: sampling, aliasing, the DFT's lines, leakage, windows, averaging and transducers.

Left: the signal (two tones, a chirp or a struck mode, plus noise) and how it is acquired (the sample rate,
the anti-alias filter, the block, the window, the averages). Centre: the first block, the analog signal behind
its samples and the window over them, and the averaged spectrum, as a peak amplitude or a PSD, with each
tone's true amplitude where it should show. Right: the windows compared, a spectrogram of the record, the
transducers (a seismic mass in a case), and the Theory. The page has its own model (core/signals.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.practice import Problem, Question, Requirement, at_least, at_most, number, pick, rounded
from ..core.signals import (
    FILTER_RATIO,
    WINDOWS,
    Kind,
    Source,
    Window,
    apparent,
    base_excited,
    blocks_needed,
    kernel,
    mass_loaded,
    nearest_line_reading,
    sample,
    spectrogram,
    spectrum,
    usable_band,
    window,
    window_props,
)
from . import settle_plots, signals_figures
from .layouts import ColumnScroll
from .lesson import (
    Form,
    TrialBox,
    connect_all,
    db,
    fmt,
    note_label,
    plot,
    readout_label,
    set_quietly,
    trials_html,
)
from .links import load_notes, page_link
from .notes_view import NotesBrowser
from .style import colors
from .theming import add_legend

KINDS = list(Kind)
BLOCKS = tuple(2**k for k in range(6, 15))  # 64 … 16384 samples
SCALES = ("Peak amplitude", "PSD")
TAB_WINDOWS, TAB_SPECTROGRAM, TAB_SENSORS = range(3)
ANALOG_POINTS = 20000  # most points drawn for the analog signal behind a block
ZOOM_SAMPLES = 64  # the time plot's close-up: the block's first samples
SPAN = 12.0  # lines either side drawn on the Windows tab
SENSOR_RATIOS = np.geomspace(0.01, 100.0, 1200)


@dataclass(frozen=True)
class SignalsTrial:
    name: str
    note: str
    kind: Kind = Kind.TONES
    f1: float = 50.0
    a1: float = 1.0
    f2: float = 120.0
    a2: float = 0.1
    noise: float = 0.0
    zeta: float = 0.01
    sweep: float = 4.0
    fs: float = 1024.0
    block: int = 1024
    window: Window = Window.HANN
    averages: int = 1
    overlap: float = 50.0
    filtered: bool = True
    scale: int = 0
    zoom: bool = False
    tab: int = TAB_WINDOWS
    sensor_hz: float = 30000.0
    sensor_zeta: float = 0.01


TRIALS = (
    SignalsTrial(
        "A tone on a line",
        "1024 samples at 1024 Hz: a 1 s block, lines <m>Δf</m> = 1 Hz apart, the highest at the Nyquist frequency, "
        "512 Hz. A 50 Hz tone fits 50 whole cycles in the block, so it lands on a line and, without a window, "
        "reads its amplitude exactly, 1.000, with nothing on the other lines: no leakage.",
        f2=120.0,
        a2=0.0,
        window=Window.RECTANGULAR,
    ),
    SignalsTrial(
        "Between lines: leakage",
        "The tone moved to 50.5 Hz, half-way between lines, still unwindowed. It reads 0.64 (−3.9 dB, the "
        "rectangular window's scalloping loss), and its energy leaks across the whole spectrum, falling only as "
        "1/f: at 70 Hz the leak reads 0.015, fifteen times a 0.001 tone that is there and can't be seen.",
        f1=50.5,
        f2=70.0,
        a2=0.001,
        window=Window.RECTANGULAR,
    ),
    SignalsTrial(
        "A Hann window",
        "The same two tones through a Hann window. The big one reads 0.85 (−1.4 dB) and spreads over a few "
        "lines, but its skirts fall as 1/f³: the 0.001 tone at 70 Hz now stands clear, reading 0.0010.",
        f1=50.5,
        f2=70.0,
        a2=0.001,
        window=Window.HANN,
    ),
    SignalsTrial(
        "A flat-top window",
        "A flat-top window reads 0.999 for the 50.5 Hz tone: its scalloping loss is 0.01 dB, so a tone's "
        "amplitude is right wherever it falls. The price is a main lobe 5 lines wide each side, and a noise "
        "bandwidth of 3.8 lines. Calibrations use it; separating close tones doesn't.",
        f1=50.5,
        f2=70.0,
        a2=0.001,
        window=Window.FLAT_TOP,
    ),
    SignalsTrial(
        "Aliasing",
        "A 900 Hz tone sampled at 1024 Hz with the anti-alias filter off: above Nyquist (512 Hz), it folds back "
        "and shows at 1024 − 900 = 124 Hz with its whole amplitude, indistinguishable from a real 124 Hz tone. "
        "The close-up of the first 64 samples shows why: the dots trace a slow sine through the fast one.",
        f1=900.0,
        a1=1.0,
        a2=0.0,
        filtered=False,
        zoom=True,
    ),
    SignalsTrial(
        "The anti-alias filter",
        "The same 900 Hz tone with the filter on: an 8-pole low-pass at <m>f<sub>s</sub>/2.56</m> = 400 Hz cuts it to "
        "0.0015 (−56 dB) before it is sampled. The filter also takes 3 dB off a tone at 400 Hz, which is why only "
        "the lines up to <m>f<sub>s</sub>/2.56</m> are kept (the shaded band is not to be trusted).",
        f1=900.0,
        a1=1.0,
        a2=0.0,
        filtered=True,
        zoom=True,
    ),
    SignalsTrial(
        "A tone in noise",
        "A 0.05 tone at 100 Hz in noise of RMS 0.5 (ten times its amplitude). As a PSD the noise reads its "
        "density, 0.5²/512 = 4.9 × 10⁻⁴ (m/s²)² per Hz, whatever the block. As an amplitude, each line holds the noise "
        "in a band about <m>Δf</m> wide: in blocks of 4096 the noise sits at 0.019 and the tone stands 2.6 times above it. "
        "It falls as <m>1/√N</m>: in blocks of 256 it is at 0.077 and the tone drowns.",
        f1=100.0,
        a1=0.05,
        a2=0.0,
        noise=0.5,
        block=4096,
        averages=8,
        scale=0,
    ),
    SignalsTrial(
        "Averaging smooths",
        "Noise alone, of RMS 0.5, as a PSD from one block of 256: each line scatters by 100 %. Raise the averages "
        "to 64 and the scatter falls to an eighth (<m>1/√n<sub>d</sub></m>), but the level stays at 4.9 × 10⁻⁴: averaging doesn't "
        "lower the noise, it makes its level certain. The PSD's area, 0.25, is the RMS².",
        a1=0.0,
        a2=0.0,
        noise=0.5,
        block=256,
        averages=1,
        scale=1,
    ),
    SignalsTrial(
        "A chirp's spectrogram",
        "A sine swept from 10 to 400 Hz in 4 s, in blocks of 256 (0.25 s). The averaged spectrum is a flat "
        "smear: it can't say when each frequency was there. The spectrogram, one spectrum per block, shows the "
        "sweep as a line. A shorter block places it better in time and worse in frequency: <m>Δf Δt</m> ≈ 1.",
        kind=Kind.CHIRP,
        f1=10.0,
        a1=1.0,
        f2=400.0,
        sweep=4.0,
        block=256,
        averages=31,
        tab=TAB_SPECTROGRAM,
    ),
    SignalsTrial(
        "A ringing mode",
        "A mode at 50 Hz with <m>ζ</m> = 0.01 struck once: its half-power width is <m>2ζf<sub>n</sub></m> = 1 Hz. In a 4 s "
        "block (<m>Δf</m> = 0.25 Hz) it has four lines across it and its peak shows, without a window since it dies "
        "away inside the block. A Hann window here would throw away its start, where most of it is.",
        kind=Kind.RINGING,
        f1=50.0,
        a1=1.0,
        zeta=0.01,
        block=4096,
        window=Window.RECTANGULAR,
    ),
    SignalsTrial(
        "An accelerometer",
        "A piezoelectric accelerometer is a seismic mass on a stiff crystal: <m>f<sub>n</sub></m> = 30 kHz, <m>ζ</m> ≈ 0.01. "
        "Below its resonance its output follows the acceleration, within 5 % up to 0.22 <m>f<sub>n</sub></m>, 6.5 kHz. "
        "A geophone is the other way round: soft, <m>f<sub>n</sub></m> = 4.5 Hz, and with <m>ζ</m> = 0.7 it follows the "
        "displacement within 5 % from 1.7 <m>f<sub>n</sub></m>, 7.6 Hz, up.",
        tab=TAB_SENSORS,
    ),
)

# ------------------------------------------------------------------ practice problems
# Each question's numbers are printed round, and its answer is worked from them as printed.


def _alias(rng: np.random.Generator) -> Question:
    fs = pick(rng, (500, 1000, 1024, 2000, 2048))
    f = rounded(fs * pick(rng, (0.6, 0.7, 0.85, 1.15, 1.3, 1.6, 2.2)), 3)
    shown = apparent(f, fs)
    return Question(
        f"A {number(f)} Hz tone is sampled at {fs:g} Hz with no anti-alias filter. At what frequency does it show "
        "in the spectrum?",
        shown,
        f"Nyquist is f<sub>s</sub>/2 = {number(fs / 2)} Hz, below the tone, so it folds back: the samples fit a "
        f"sine at |f − kf<sub>s</sub>| for the k that lands in 0 … f<sub>s</sub>/2, here {number(shown)} Hz.",
        unit="Hz",
        setup=dict(f1=f, a2=0.0, fs=float(fs), filtered=False, window=Window.HANN),
    )


def _line_spacing(rng: np.random.Generator) -> Question:
    fs = pick(rng, (512, 1000, 1024, 2048, 4096, 5000))
    n = int(pick(rng, BLOCKS[2:]))
    return Question(
        f"An analyzer takes blocks of {n} samples at {fs:g} Hz. How far apart are the spectrum's lines?",
        fs / n,
        f"The block lasts T = N/f<sub>s</sub> = {number(n / fs)} s, and the lines are Δf = 1/T = "
        f"f<sub>s</sub>/N = {number(fs / n)} Hz apart.",
        unit="Hz",
        setup=dict(fs=float(fs), block=n),
    )


def _record_length(rng: np.random.Generator) -> Question:
    fs = pick(rng, (1024, 2048, 4096))
    n = int(pick(rng, (1024, 2048, 4096, 8192)))
    averages = int(pick(rng, (4, 8, 10, 20)))
    overlap = pick(rng, (0, 50, 75))
    samples = blocks_needed(n, averages, overlap / 100.0)
    return Question(
        f"{averages} averages of {n}-sample blocks at {fs:g} Hz, overlapping by {overlap:g} %. How long a record "
        "does that take?",
        samples / fs,
        f"Each new block starts {100 - overlap:g} % of a block, {round(n * (1 - overlap / 100))} samples, "
        f"after the last: N + (n − 1) × {round(n * (1 - overlap / 100))} = {samples} samples, "
        f"{number(samples / fs)} s.",
        unit="s",
        setup=dict(fs=float(fs), block=n, averages=averages, overlap=float(overlap), noise=0.5, scale=1),
    )


def _scalloping(rng: np.random.Generator) -> Question:
    kind = (Window.RECTANGULAR, Window.HANN)[int(rng.integers(2))]
    a = pick(rng, (1, 2, 5, 10))
    loss = -window_props(kind).scalloping_db  # dB
    reads = a * 10 ** (-loss / 20)
    name = "no window" if kind is Window.RECTANGULAR else f"a {kind.value} window"
    return Question(
        f"A tone of amplitude {a:g} lies exactly half-way between two lines. With {name}, what amplitude does "
        "the largest line read?",
        reads,
        f"Half-way between lines is the worst place: the window's spectrum half a line from its centre, the "
        f"scalloping loss, is {number(loss)} dB for {name}: {a:g} × 10<sup>−{number(loss)}/20</sup> = "
        f"{number(reads)}. (No window: 2/π, 3.92 dB; Hann 1.42 dB; a flat top, almost none: that's what it's for.)",
        setup=dict(f1=50.5, a1=a, a2=0.0, window=kind),
    )


def _accelerometer_band(rng: np.random.Generator) -> Question:
    fn = pick(rng, (10000, 20000, 30000, 50000))
    zeta = 0.01
    r = usable_band(zeta)
    return Question(
        f"A piezoelectric accelerometer resonates at {fn / 1000:g} kHz, almost undamped (ζ = 0.01). Up to what "
        "frequency does it read the acceleration within 5 %?",
        r * fn,
        f"It reads 1/|1 − r² + 2iζr| of the truth; with ζ small, 1/(1 − r²) = 1.05 at r = √(1 − 1/1.05) = "
        f"{number(r)}: {number(r * fn)} Hz, about a fifth of its f<sub>n</sub>.",
        unit="Hz",
        setup=dict(sensor_hz=fn, sensor_zeta=zeta, tab=TAB_SENSORS),
    )


def _mass_loading(rng: np.random.Generator) -> Question:
    f = pick(rng, (100, 200, 500, 1000))
    m = pick(rng, (0.2, 0.5, 1, 2))
    ms = pick(rng, (0.005, 0.01, 0.03, 0.05))
    shift = f - mass_loaded(f, m, ms)
    return Question(
        f"A {1000 * ms:g} g accelerometer is fixed where a {f:g} Hz mode's modal mass is {m:g} kg. How far does it "
        "lower the mode's frequency?",
        shift,
        f"Its mass adds to the modal mass, the stiffness unchanged: f √(m/(m + m<sub>s</sub>)) = {f:g} × "
        f"√({m:g}/{number(m + ms)}) = {number(f - shift, 5)} Hz, {number(shift)} Hz lower.",
        unit="Hz",
        tolerance=0.03,
        setup=dict(tab=TAB_SENSORS),
    )


def _faster_sampling(rng: np.random.Generator) -> Question:
    return Question(
        "An analyzer's sample rate is doubled, its block still the same number of samples. What happens to the "
        "spacing between lines?",
        0,
        "It doubles: Δf = f<sub>s</sub>/N, or 1/T with the block now half as long. A higher f<sub>s</sub> reaches "
        "higher frequencies, but only a longer block resolves finer.",
        choices=("It doubles (coarser)", "It halves (finer)", "It stays the same"),
        setup=dict(fs=2048.0, f2=51.0, a2=0.5),
        predict=True,
    )


def _more_averages(rng: np.random.Generator) -> Question:
    n = int(pick(rng, (4, 16)))
    return Question(
        f"A noise PSD is averaged over {n} blocks instead of one. What happens to it?",
        0,
        f"Its level stays (the density doesn't depend on the averaging), but each line's scatter falls to "
        f"1/√{n} of the single block's 100 %.",
        choices=(
            f"The level stays; the scatter falls to 1/√{n}",
            f"The level falls to 1/{n}",
            f"The level falls to 1/√{n}",
            "Nothing changes",
        ),
        setup=dict(a1=0.0, a2=0.0, noise=1.0, averages=n, scale=1),
        predict=True,
    )


def _longer_block(rng: np.random.Generator) -> Question:
    return Question(
        "A tone sits in white noise, read on a peak amplitude spectrum. The block is made 4 times longer (same "
        "sample rate). What happens to the tone's line and to the noise around it?",
        0,
        "The tone stays at its amplitude; the noise in each line falls by half (√4), since each line now holds "
        "the noise in a band a quarter as wide. That is how a tone is dug out of noise.",
        choices=(
            "The tone stays; the noise halves",
            "Both stay the same",
            "The tone grows 4 times; the noise stays",
            "Both halve",
        ),
        setup=dict(a2=0.0, noise=2.0, block=4096),
        predict=True,
    )


def _acquisition_spec(rng: np.random.Generator) -> Question:
    top = pick(rng, (100, 200, 400))
    gap = pick(rng, (0.5, 1, 2))
    averages = int(pick(rng, (4, 8)))
    fs = rounded(1.1 * FILTER_RATIO * top, 2)
    n = next(b for b in BLOCKS if fs / b <= gap / 4)
    longest = rounded(1.2 * blocks_needed(n, averages, 0.5) / fs, 2)

    def goal(t: SignalsTrial):
        record = blocks_needed(t.block, t.averages, t.overlap / 100.0) / t.fs
        return (
            Requirement("The anti-alias filter on", "on" if t.filtered else "off", t.filtered),
            at_least("The good band, f<sub>s</sub>/2.56", t.fs / FILTER_RATIO, float(top), "Hz"),
            at_most("The lines' spacing Δf", t.fs / t.block, gap / 4, "Hz"),
            at_least("Averages", t.averages, averages),
            at_most("The record", record, longest, "s"),
        )

    return Question(
        f"Two tones {gap:g} Hz apart near {top:g} Hz are to be measured with an anti-alias filter, at least 4 lines "
        f"apart (Δf ≤ {number(gap / 4)} Hz), averaged {averages} times, all from a record no longer than "
        f"{number(longest, 2)} s. Set up the acquisition.",
        0,
        f"The filter keeps f<sub>s</sub>/2.56, so f<sub>s</sub> ≥ 2.56 × {top:g} = {number(2.56 * top, 4)} Hz; "
        f"no higher than needed, since Δf = f<sub>s</sub>/N and the block's length T = 1/Δf ≥ {number(4 / gap)} s "
        f"is fixed by the resolution. With f<sub>s</sub> = {number(fs, 2)} Hz, N = {n} gives Δf = "
        f"{number(fs / n)} Hz. {averages} blocks end to end take {number(averages * n / fs)} s, too long; "
        f"overlapping them by 50 % takes {number(blocks_needed(n, averages, 0.5) / fs)} s.",
        setup=dict(f1=top - gap, f2=float(top), a2=0.5, filtered=False),
        goal=goal,
        design=dict(f1=top - gap, f2=float(top), a2=0.5, fs=fs, block=n, averages=averages, overlap=50.0),
    )


PROBLEMS = tuple(
    Problem(key, title, "signals_page", f"See {page_link('signals_page', anchor, section)}{more}.", make)
    for key, title, anchor, section, more, make in (
        ("alias", "Where an alias shows", "aliasing", "Aliasing", "", _alias),
        ("line-spacing", "Line spacing", "dft-lines", "Blocks and lines", "", _line_spacing),
        (
            "record-length",
            "A record's length",
            "averaging",
            "Averaging",
            ": how far apart do the blocks start?",
            _record_length,
        ),
        ("scalloping", "Between two lines", "leakage", "Leakage", ": the scalloping loss", _scalloping),
        ("accelerometer-band", "An accelerometer's band", "transducers", "Transducers", "", _accelerometer_band),
        ("mass-loading", "Mass loading", "mass-loading", "Mass loading", "", _mass_loading),
        ("faster-sampling", "Predict: a faster sample rate", "dft-lines", "Blocks and lines", "", _faster_sampling),
        ("more-averages", "Predict: more averages", "averaging", "Averaging", "", _more_averages),
        ("longer-block", "Predict: a longer block", "scaling", "Amplitude or density", "", _longer_block),
        (
            "acquisition-spec",
            "Spec: set up an acquisition",
            "dft-lines",
            "Blocks and lines",
            ", the anti-alias filter and averaging",
            _acquisition_spec,
        ),
    )
)

TIPS = {
    "kind": "<p><b>Two tones:</b> two sines. <b>Chirp:</b> a sine swept from f₁ to f₂, again and again. "
    "<b>Ringing:</b> a mode at f₁ with damping ζ, struck once at t = 0.</p>",
    "f1": "<p>The first tone's frequency (the chirp's start, the mode's frequency).</p>",
    "a1": "<p>Its amplitude (an acceleration, m/s²).</p>",
    "f2": "<p>The second tone's frequency (the chirp's end).</p>",
    "a2": "<p>The second tone's amplitude.</p>",
    "noise": "<p>White noise added at the samples, by its RMS (spread evenly up to the Nyquist frequency).</p>",
    "zeta": "<p>The struck mode's damping ratio.</p>",
    "sweep": "<p>How long each sweep of the chirp takes.</p>",
    "fs": "<p>The sample rate f<sub>s</sub>. Only frequencies up to f<sub>s</sub>/2 (Nyquist) can be represented.</p>",
    "filter": "<p>An 8-pole Butterworth low-pass at f<sub>s</sub>/2.56 before the sampler, as analyzers have, so "
    "nothing above Nyquist folds back.</p>",
    "block": "<p>Samples per block N: the block lasts N/f<sub>s</sub> and its lines are f<sub>s</sub>/N apart.</p>",
    "window": "<p>The taper each block is multiplied by before its DFT.</p>",
    "averages": "<p>How many blocks are averaged (as power).</p>",
    "overlap": "<p>How much each block overlaps the one before.</p>",
    "scale": "<p><b>Peak amplitude:</b> scaled so a tone reads its amplitude. <b>PSD:</b> scaled so noise reads "
    "its density (units²/Hz), whatever the block or the window.</p>",
    "sensor_hz": "<p>The transducer's own natural frequency: its seismic mass on its spring.</p>",
    "sensor_zeta": "<p>Its damping ratio.</p>",
    "structure_hz": "<p>A mode of the structure being measured.</p>",
    "modal_mass": "<p>The mode's modal mass at the measuring point (scaled to a unit motion there): for a small "
    "part, a few grams to a kilogram.</p>",
    "sensor_mass": "<p>The sensor's mass, mounting included.</p>",
}
TIME_TIP = (
    "<p>The first block: the analog signal (thin), its samples (dots, joined), and the window it is multiplied by "
    "(dashed, scaled to the block's largest value). Drag or wheel to zoom in time.</p>"
)
SPECTRUM_TIP = (
    "<p>The averaged spectrum. Circles: each tone's true amplitude, where it should show (folded back if it is "
    "above Nyquist). Dashed: the noise's expected level. Shaded: above f<sub>s</sub>/2.56, where the anti-alias "
    "filter cuts in.</p>"
)
WINDOWS_TIP = (
    "<p>Each window's spectrum about a tone's line, against the offset in lines: the main lobe, how wide it is, "
    "and the sidelobes, how far they leak. A tone between lines is read at ±0.5.</p>"
)
SPECTROGRAM_TIP = "<p>One spectrum per block (the acquisition's block and window, 75 % overlap), in dB.</p>"
SENSOR_TIP = (
    "<p>A seismic mass in a case on the structure, read by its motion relative to the case. Blue: its reading "
    "over the acceleration it should read (an accelerometer). Orange: over the displacement (a seismometer). "
    "The band is ±5 %.</p>"
)


def window_table() -> str:
    rows = "".join(
        f"<tr><td>{w.value}</td><td align='right'>{p.main_lobe:.0f}</td><td align='right'>{p.sidelobe_db:.0f}</td>"
        f"<td align='right'>{p.scalloping_db:.2f}</td><td align='right'>{p.enbw:.2f}</td>"
        f"<td align='right'>{p.coherent_gain:.3f}</td></tr>"
        for w, p in ((w, window_props(w)) for w in WINDOWS)
    )
    return (
        "<table cellspacing='0' cellpadding='3' border='1'><tr><th>Window</th><th>main lobe (lines each "
        "side)</th><th>highest sidelobe (dB)</th><th>scalloping loss (dB)</th><th>noise bandwidth (lines)</th>"
        f"<th>coherent gain</th></tr>{rows}</table>"
    )


NOTES_HTML = load_notes(
    "signals", figures=signals_figures.FIGURES, window_table=window_table(), trials=trials_html(TRIALS)
)


class SignalsPage(QtWidgets.QWidget):
    """Controls on the left, the block and its spectrum in the centre, windows, spectrogram and sensors on the
    right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.source = Source()
        self.samples = np.zeros(0)

        self.trials = TrialBox(
            [t.name for t in TRIALS],
            "<p>Each sets up the signal and its acquisition to show one idea from the Theory tab.</p>",
        )
        self.trials.picked.connect(self.load_trial)

        sig = Form("Signal")
        self.kind = QtWidgets.QComboBox()
        self.kind.addItems([k.value for k in KINDS])
        self.kind.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        sig.add("Kind:", self.kind, TIPS["kind"])
        self.f1 = sig.number("f₁:", 0.0, 1e5, 50.0, 2, " Hz", TIPS["f1"])
        self.a1 = sig.number("Amplitude a₁:", 0.0, 1e3, 1.0, 4, " m/s²", TIPS["a1"])
        self.f2 = sig.number("f₂:", 0.0, 1e5, 120.0, 2, " Hz", TIPS["f2"])
        self.a2 = sig.number("Amplitude a₂:", 0.0, 1e3, 0.1, 4, " m/s²", TIPS["a2"])
        self.zeta = sig.number("Damping ζ:", 0.0001, 0.5, 0.01, 4, "", TIPS["zeta"])
        self.sweep = sig.number("Sweep time:", 0.01, 100.0, 4.0, 2, " s", TIPS["sweep"])
        self.noise = sig.number("Noise RMS:", 0.0, 1e3, 0.0, 4, " m/s²", TIPS["noise"])

        acq = Form("Acquisition")
        self.fs = acq.number("Sample rate fₛ:", 1.0, 1e6, 1024.0, 1, " Hz", TIPS["fs"])
        self.filtered = QtWidgets.QCheckBox("Anti-alias filter")
        self.filtered.setChecked(True)
        self.filtered.setToolTip(TIPS["filter"])
        acq.row(self.filtered)
        self.block = QtWidgets.QComboBox()
        self.block.addItems([str(b) for b in BLOCKS])
        self.block.setCurrentIndex(BLOCKS.index(1024))
        acq.add("Block N:", self.block, TIPS["block"])
        self.taper = QtWidgets.QComboBox()
        self.taper.addItems([w.value for w in WINDOWS])
        self.taper.setCurrentIndex(WINDOWS.index(Window.HANN))
        self.taper.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        acq.add("Window:", self.taper, TIPS["window"])
        self.averages = acq.number("Averages:", 1, 1000, 1, 0, "", TIPS["averages"])
        self.overlap = acq.number("Overlap:", 0.0, 90.0, 50.0, 0, " %", TIPS["overlap"])
        self.acq_info = note_label()
        acq.row(self.acq_info)

        connect_all(
            (
                self.kind,
                self.f1,
                self.a1,
                self.f2,
                self.a2,
                self.zeta,
                self.sweep,
                self.noise,
                self.fs,
                self.filtered,
                self.block,
                self.taper,
                self.averages,
                self.overlap,
            ),
            self._on_inputs,
        )

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        for w in (self.trials, sig, acq):
            lv.addWidget(w)
        lv.addStretch(1)

        # --- centre
        self.time_plot = plot("Signal", "m/s²", "Time", "s", height=150)
        self.time_plot.setToolTip(TIME_TIP)
        self.time_plot.setMouseEnabled(x=True, y=False)
        self.analog_item = pg.PlotDataItem()
        self.samples_item = pg.PlotDataItem(symbol="o", symbolSize=4)
        self.window_items = (pg.PlotDataItem(), pg.PlotDataItem())
        for item in (self.analog_item, *self.window_items, self.samples_item):
            self.time_plot.addItem(item)

        self.scale = QtWidgets.QComboBox()
        self.scale.addItems(SCALES)
        self.scale.setToolTip(TIPS["scale"])
        self.scale.currentIndexChanged.connect(self._draw_spectrum)
        self.spec_plot = plot("Amplitude", "m/s²", "Frequency", "Hz", log_y=True, height=200)
        self.spec_plot.setToolTip(SPECTRUM_TIP)
        self.spec_plot.setMouseEnabled(x=True, y=False)
        self.spec_legend = add_legend(self.spec_plot.getPlotItem(), offset=(-10, 10))
        self.band = pg.LinearRegionItem((0, 1), movable=False)
        self.band.setZValue(-10)
        for line in self.band.lines:
            line.setPen(pg.mkPen(None))
        self.spec_item = pg.PlotDataItem(name="spectrum")
        self.noise_line = pg.PlotDataItem()
        self.noise_line.setZValue(5)
        self.tone_items = pg.PlotDataItem(pen=None, symbol="o", symbolSize=11)
        self.tone_items.setZValue(6)
        self.legend_shown: tuple[bool, ...] = ()
        for item in (self.band, self.spec_item, self.noise_line, self.tone_items):
            self.spec_plot.addItem(item)
        self.spec_note = readout_label()

        center = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(center)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.addWidget(self.time_plot, 2)
        self.zoom = QtWidgets.QCheckBox(f"First {ZOOM_SAMPLES} samples only")
        self.zoom.setToolTip(
            "<p>Show the block's first samples close up, to see what the sampler sees, rather "
            "than the whole block and its window.</p>"
        )
        self.zoom.toggled.connect(self._zoom_time)
        scale_row = QtWidgets.QHBoxLayout()
        scale_row.addWidget(self.zoom)
        scale_row.addStretch(1)
        scale_row.addWidget(QtWidgets.QLabel("Spectrum as:"))
        scale_row.addWidget(self.scale)
        cv.addLayout(scale_row)
        cv.addWidget(self.spec_plot, 3)
        cv.addWidget(self.spec_note)

        # --- right: windows
        wins = QtWidgets.QWidget()
        wv = QtWidgets.QVBoxLayout(wins)
        self.kernel_plot = plot("Level", "dB", "Offset from the tone", "lines", height=200)
        self.kernel_plot.setToolTip(WINDOWS_TIP)
        self.kernel_plot.setMouseEnabled(x=False, y=False)
        add_legend(self.kernel_plot.getPlotItem(), offset=(-10, 10))
        self.kernel_items = [pg.PlotDataItem(name=w.value) for w in WINDOWS]
        self.half_lines = [pg.InfiniteLine(angle=90, pos=p) for p in (-0.5, 0.5)]
        for item in (*self.half_lines, *self.kernel_items):
            self.kernel_plot.addItem(item)
        wv.addWidget(self.kernel_plot, 1)
        self.shape_plot = plot("w", None, "Sample / N", None, height=110)
        self.shape_plot.setMouseEnabled(x=False, y=False)
        self.shape_items = [pg.PlotDataItem() for _ in WINDOWS]
        for item in self.shape_items:
            self.shape_plot.addItem(item)
        wv.addWidget(self.shape_plot)
        self.window_note = note_label()
        wv.addWidget(self.window_note)

        # --- right: spectrogram
        sgram = QtWidgets.QWidget()
        sgv = QtWidgets.QVBoxLayout(sgram)
        self.sgram_plot = plot("Frequency", "Hz", "Time", "s", height=200)
        self.sgram_plot.setToolTip(SPECTROGRAM_TIP)
        self.sgram_plot.setMouseEnabled(x=False, y=False)
        self.sgram_image = pg.ImageItem(axisOrder="row-major")
        self.sgram_plot.addItem(self.sgram_image)
        sgv.addWidget(self.sgram_plot, 1)
        self.sgram_note = note_label()
        sgv.addWidget(self.sgram_note)

        # --- right: transducers
        sensors = QtWidgets.QWidget()
        sv = QtWidgets.QVBoxLayout(sensors)
        sensor = Form("The sensor")
        self.sensor_hz = sensor.number("Its fₙ:", 0.01, 1e6, 30000.0, 2, " Hz", TIPS["sensor_hz"])
        self.sensor_zeta = sensor.number("Its damping ζ:", 0.0, 2.0, 0.01, 3, "", TIPS["sensor_zeta"])
        sensor.changed.connect(self._draw_tab)
        sv.addWidget(sensor)
        self.ratio_plot = plot(
            "Reading / truth", None, "f / fₙ (the sensor's)", None, log_x=True, log_y=True, height=150
        )
        self.ratio_plot.setToolTip(SENSOR_TIP)
        self.ratio_plot.setMouseEnabled(x=False, y=False)
        add_legend(self.ratio_plot.getPlotItem(), offset=(10, 10))
        self.flat_band = pg.LinearRegionItem(
            (math.log10(0.95), math.log10(1.05)), orientation="horizontal", movable=False
        )
        for line in self.flat_band.lines:
            line.setPen(pg.mkPen(None))
        self.acc_item = pg.PlotDataItem(name="accelerometer")
        self.seis_item = pg.PlotDataItem(name="seismometer")
        for item in (self.flat_band, self.acc_item, self.seis_item):
            self.ratio_plot.addItem(item)
        sv.addWidget(self.ratio_plot, 2)
        self.phase_plot = plot("Accelerometer's lag", "°", "f / fₙ (the sensor's)", None, log_x=True, height=100)
        self.phase_plot.setMouseEnabled(x=False, y=False)
        self.phase_item = pg.PlotDataItem()
        self.phase_plot.addItem(self.phase_item)
        self.phase_plot.setXLink(self.ratio_plot)
        sv.addWidget(self.phase_plot, 1)
        self.sensor_note = note_label()
        sv.addWidget(self.sensor_note)
        loading = Form("Mass loading")
        self.structure_hz = loading.number("A mode at:", 0.1, 1e5, 200.0, 1, " Hz", TIPS["structure_hz"])
        self.modal_mass = loading.number("Its modal mass:", 0.001, 1e4, 0.5, 3, " kg", TIPS["modal_mass"])
        self.sensor_mass = loading.number("Sensor's mass:", 0.0, 100.0, 0.03, 4, " kg", TIPS["sensor_mass"])
        self.loading_note = note_label()
        loading.row(self.loading_note)
        loading.changed.connect(self._show_loading)
        sv.addWidget(loading)

        self.notes = NotesBrowser()
        self.notes.setHtml(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(wins, "Windows")
        self.tabs.addTab(sgram, "Spectrogram")
        self.tabs.addTab(ColumnScroll(sensors), "Transducers")
        self.tabs.addTab(self.notes, "Theory")
        self.tabs.currentChanged.connect(self._draw_tab)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(ColumnScroll(left))
        splitter.addWidget(center)
        splitter.addWidget(self.tabs)
        splitter.setSizes([320, 760, 560])
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self._on_inputs()
        self.apply_theme()

    # ------------------------------------------------------------- inputs
    def load_trial(self, index: int) -> None:
        self.apply_trial(TRIALS[index])

    def load_setup(self, setup: dict) -> None:
        """Set the page up as a practice problem's `setup` says (a trial's fields, with no note)."""
        self.apply_trial(SignalsTrial("", "", **setup))

    def practice_state(self) -> SignalsTrial:
        """The page's settings, as a trial: what a practice spec checks."""
        return SignalsTrial(
            "",
            "",
            kind=KINDS[self.kind.currentIndex()],
            f1=self.f1.value(),
            a1=self.a1.value(),
            f2=self.f2.value(),
            a2=self.a2.value(),
            noise=self.noise.value(),
            zeta=self.zeta.value(),
            sweep=self.sweep.value(),
            fs=self.fs.value(),
            block=self.n_block,
            window=self.window_kind,
            averages=int(self.averages.value()),
            overlap=self.overlap.value(),
            filtered=self.filtered.isChecked(),
            scale=self.scale.currentIndex(),
            zoom=self.zoom.isChecked(),
            tab=self.tabs.currentIndex(),
            sensor_hz=self.sensor_hz.value(),
            sensor_zeta=self.sensor_zeta.value(),
        )

    def apply_trial(self, t: SignalsTrial) -> None:
        for box, value in (
            (self.kind, KINDS.index(t.kind)),
            (self.f1, t.f1),
            (self.a1, t.a1),
            (self.f2, t.f2),
            (self.a2, t.a2),
            (self.noise, t.noise),
            (self.zeta, t.zeta),
            (self.sweep, t.sweep),
            (self.fs, t.fs),
            (self.block, BLOCKS.index(t.block)),
            (self.taper, WINDOWS.index(t.window)),
            (self.averages, t.averages),
            (self.overlap, t.overlap),
            (self.filtered, t.filtered),
            (self.scale, t.scale),
            (self.zoom, t.zoom),
            (self.sensor_hz, t.sensor_hz),
            (self.sensor_zeta, t.sensor_zeta),
        ):
            set_quietly(box, value)
        self.trials.show_note(t.note)
        self.tabs.setCurrentIndex(t.tab)
        self._on_inputs()

    @property
    def n_block(self) -> int:
        return BLOCKS[self.block.currentIndex()]

    @property
    def window_kind(self) -> Window:
        return WINDOWS[self.taper.currentIndex()]

    def _on_inputs(self) -> None:
        kind = KINDS[self.kind.currentIndex()]
        self.source = Source(
            kind,
            self.f1.value(),
            self.a1.value(),
            self.f2.value(),
            self.a2.value(),
            self.noise.value(),
            self.zeta.value(),
            self.sweep.value(),
        )
        self.zeta.setEnabled(kind is Kind.RINGING)
        self.sweep.setEnabled(kind is Kind.CHIRP)
        self.f2.setEnabled(kind is not Kind.RINGING)
        self.a2.setEnabled(kind is Kind.TONES)
        fs, n = self.fs.value(), self.n_block
        self.n_record = blocks_needed(n, int(self.averages.value()), self.overlap.value() / 100.0)
        _, self.samples = sample(self.source, fs, self.n_record, self.filtered.isChecked())
        self.spec = spectrum(self.samples, fs, n, self.window_kind, self.overlap.value() / 100.0)
        self.acq_info.setText(
            f"Block T = {fmt(n / fs, 's')}, lines Δf = {fmt(fs / n, 'Hz')} apart up to Nyquist, "
            f"{fmt(fs / 2, 'Hz')}"
            + (f"; good to f<sub>s</sub>/2.56 = {fmt(fs / FILTER_RATIO, 'Hz')}" if self.filtered.isChecked() else "")
            + f". The record: {self.n_record} samples, {fmt(self.n_record / fs, 's')}, {self.spec.blocks} "
            f"block{'s' if self.spec.blocks > 1 else ''}."
        )
        self.redraw()

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        self.analog_item.setPen(pg.mkPen(c.grey, width=1))
        self.samples_item.setPen(pg.mkPen(c.mass[0], width=1.5))
        self.samples_item.setSymbolPen(pg.mkPen(c.mass[0], width=1))
        self.samples_item.setSymbolBrush(pg.mkBrush(c.mass[0]))
        for item in self.window_items:
            item.setPen(pg.mkPen(c.fit, width=1.2, style=QtCore.Qt.PenStyle.DashLine))
        self.band.setBrush(pg.mkBrush(*c.band_shade))
        self.spec_item.setPen(pg.mkPen(c.mass[0], width=1.5))
        self.noise_line.setPen(pg.mkPen(c.strong, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        self.tone_items.setSymbolPen(pg.mkPen(c.force, width=2))
        self.tone_items.setSymbolBrush(None)
        for item, color in zip(self.kernel_items, (c.grey, c.mass[0], c.mode[3])):
            item.setPen(pg.mkPen(color, width=1.5))
        for item, color in zip(self.shape_items, (c.grey, c.mass[0], c.mode[3])):
            item.setPen(pg.mkPen(color, width=1.5))
        for line in self.half_lines:
            line.setPen(pg.mkPen(c.faint, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self.sgram_image.setLookupTable(
            pg.ColorMap(
                [0.0, 0.5, 1.0], [pg.mkColor(c.background), pg.mkColor(c.mode[1]), pg.mkColor(c.force)]
            ).getLookupTable(nPts=256)
        )
        self.flat_band.setBrush(pg.mkBrush(*c.band_shade))
        self.acc_item.setPen(pg.mkPen(c.mass[0], width=2))
        self.seis_item.setPen(pg.mkPen(c.mass[1], width=2))
        self.phase_item.setPen(pg.mkPen(c.mass[0], width=1.5))
        self.redraw()

    def redraw(self) -> None:
        self._draw_time()
        self._draw_spectrum()
        self._draw_tab()
        self._show_loading()
        settle_plots(self)

    def _draw_time(self) -> None:
        fs, n = self.fs.value(), self.n_block
        t = np.arange(n) / fs
        x = self.samples[:n]
        fine = np.linspace(0.0, t[-1], min(ANALOG_POINTS, 20 * n))
        self.analog_item.setData(fine, self.source.analog(fine, fs, self.filtered.isChecked()))
        self.samples_item.setData(t, x)
        self.samples_item.setSymbolSize(4 if n <= 4096 else 0)
        top = float(np.max(np.abs(x))) if x.size else 1.0
        w = window(self.window_kind, n) * (top or 1.0)
        self.window_items[0].setData(t, w)
        self.window_items[1].setData(t, -w)
        self._zoom_time()

    def _zoom_time(self) -> None:
        fs, n = self.fs.value(), self.n_block
        shown = min(n, ZOOM_SAMPLES) if self.zoom.isChecked() else n
        self.time_plot.setXRange(0.0, (shown - 1) / fs, padding=0.02)
        settle_plots(self)

    def _draw_spectrum(self) -> None:
        fs, n, spec = self.fs.value(), self.n_block, self.spec
        psd = self.scale.currentIndex() == 1
        values = spec.psd if psd else spec.amplitude
        self.spec_plot.setLabel("left", "PSD" if psd else "Amplitude", units="(m/s²)²/Hz" if psd else "m/s²")
        floor = values[values > 0]
        low = float(np.min(floor)) if floor.size else 1e-12
        self.spec_item.setData(spec.freqs, np.maximum(values, 1e-300))
        nyquist = fs / 2
        self.band.setRegion((fs / FILTER_RATIO, nyquist))
        self.band.setVisible(self.filtered.isChecked())
        noise = self.source.noise
        props = window_props(self.window_kind, n)
        if noise > 0:
            level = noise**2 / nyquist if psd else 2 * noise * math.sqrt(props.enbw / n)
            self.noise_line.setData([0.0, nyquist], [level, level])
        else:
            self.noise_line.setData([], [])
        lines, texts = [], []
        for f, a in self.source.tones():
            shown = apparent(f, fs)
            # in a PSD a tone's power a²/2 shows spread over the window's noise bandwidth
            true = a * a / 2 / (props.enbw * spec.df) if psd else a
            lines.append((shown, true))
            at, reads = nearest_line_reading(spec, shown)
            where = f"{fmt(f, 'Hz')}" + (f" shows at {fmt(shown, 'Hz')}" if abs(shown - f) > 1e-9 else "")
            if psd:
                texts.append(f"{where}: its power {a * a / 2:.3g} (m/s²)² over {fmt(spec.df, 'Hz')} lines")
            else:
                texts.append(
                    f"{where} (amplitude {a:.4g}): reads {reads:.4g} at {fmt(at, 'Hz')} "
                    f"({round(db(reads / a), 2) + 0.0 if a > 0 else 0.0:+.2f} dB)"
                )
        self.tone_items.setData([p[0] for p in lines], [p[1] for p in lines])
        self._legend(noise > 0, bool(lines))
        mean_square = float(np.mean(self.samples**2)) if self.samples.size else 0.0
        area = float(np.sum(spec.psd) * spec.df)
        texts.append(f"the record's mean square {mean_square:.4g}, the PSD's area {area:.4g}")
        text = "; ".join(texts)
        self.spec_note.setText(text[0].upper() + text[1:] + ".")
        top = max(float(np.max(values)) if values.size else 1.0, max((p[1] for p in lines), default=0.0))
        bottom = max(low, top * 1e-7)
        self.spec_plot.setXRange(0.0, nyquist, padding=0)
        self.spec_plot.setYRange(math.log10(bottom) - 0.3, math.log10(top) + 0.5, padding=0)

    def _legend(self, noise: bool, tones: bool) -> None:
        """List the noise's level and the true tones only when there are some."""
        shown = (noise, tones)
        if shown == self.legend_shown:
            return
        self.legend_shown = shown
        for item in (self.noise_line, self.tone_items):
            self.spec_legend.removeItem(item)
        if noise:
            self.spec_legend.addItem(self.noise_line, "noise's expected level")
        if tones:
            self.spec_legend.addItem(self.tone_items, "true tones")

    def _draw_tab(self) -> None:
        i = self.tabs.currentIndex()
        if i == TAB_WINDOWS:
            self._draw_windows()
        elif i == TAB_SPECTROGRAM:
            self._draw_spectrogram()
        elif i == TAB_SENSORS:
            self._draw_sensors()

    def _draw_windows(self) -> None:
        current = self.window_kind
        for item, shape, w in zip(self.kernel_items, self.shape_items, WINDOWS):
            bins, level = kernel(w, SPAN)
            item.setData(bins, np.maximum(level, -140.0))
            u = np.linspace(0.0, 1.0, 257)
            shape.setData(u, window(w, 257) if w is not Window.RECTANGULAR else np.ones(257))
            width = 2.5 if w is current else 1.0
            for it in (item, shape):
                pen = pg.mkPen(it.opts["pen"])
                pen.setWidthF(width)
                it.setPen(pen)
        self.kernel_plot.setXRange(-SPAN, SPAN, padding=0)
        self.kernel_plot.setYRange(-120.0, 5.0, padding=0)
        p = window_props(current, self.n_block)
        self.window_note.setText(
            f"{current.value}: main lobe {p.main_lobe:.0f} lines each side, highest sidelobe {p.sidelobe_db:.0f} dB, "
            f"scalloping loss {p.scalloping_db:.2f} dB, noise bandwidth {p.enbw:.2f} lines "
            f"({fmt(p.enbw * self.spec.df, 'Hz')} here)."
        )

    def _draw_spectrogram(self) -> None:
        fs, n = self.fs.value(), self.n_block
        if self.samples.size < 2 * n:
            self.sgram_image.clear()
            self.sgram_note.setText("The record holds one block: raise the averages to see it change in time.")
            return
        t, f, level = spectrogram(self.samples, fs, n, self.window_kind)
        top = float(np.max(level))
        image = np.clip((level.T - (top - 60.0)) / 60.0, 0.0, 1.0)
        self.sgram_image.setImage(image, autoLevels=False, levels=(0.0, 1.0))
        dt = t[1] - t[0] if t.size > 1 else n / fs
        self.sgram_image.setRect(QtCore.QRectF(t[0] - dt / 2, 0.0, dt * t.size, f[-1] + fs / n))
        self.sgram_plot.setXRange(0.0, self.samples.size / fs, padding=0)
        self.sgram_plot.setYRange(0.0, fs / 2, padding=0)
        self.sgram_note.setText(
            f"{t.size} blocks of {fmt(n / fs, 's')}, {fmt(dt, 's')} apart (75 % overlap), lines {fmt(fs / n, 'Hz')} "
            "apart; the colour spans 60 dB below the strongest."
        )

    def _draw_sensors(self) -> None:
        z = self.sensor_zeta.value()
        acc, seis, lag = base_excited(SENSOR_RATIOS, z)
        self.acc_item.setData(SENSOR_RATIOS, acc)
        self.seis_item.setData(SENSOR_RATIOS, seis)
        self.phase_item.setData(SENSOR_RATIOS, lag)
        self.ratio_plot.setXRange(-2.0, 2.0, padding=0)
        self.ratio_plot.setYRange(-2.0, min(math.log10(float(max(acc.max(), seis.max()))) + 0.2, 3.0), padding=0)
        self.phase_plot.setYRange(0.0, 180.0, padding=0.05)
        fn = self.sensor_hz.value()
        up, down = usable_band(z), usable_band(z, accelerometer=False)
        self.sensor_note.setText(
            f"Within ±5 %: as an accelerometer up to r = {up:.3g} ({fmt(up * fn, 'Hz')}); as a seismometer from "
            f"r = {down:.3g} ({fmt(down * fn, 'Hz')}) up."
        )

    def _show_loading(self) -> None:
        f = self.structure_hz.value()
        loaded = mass_loaded(f, self.modal_mass.value(), self.sensor_mass.value())
        self.loading_note.setText(
            f"With the sensor on, the mode is at {loaded:.4g} Hz, {100 * (loaded / f - 1):+.2f} %."
        )
