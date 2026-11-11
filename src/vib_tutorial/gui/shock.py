"""Shock response spectrum page: a pulse on the base of a bank of single-degree-of-freedom oscillators.

Left: the pulse (a classical shape; a measured record: the absolute acceleration of a mass on
the chain under a base half-sine, or a pyrotechnic-like transient, with noise and an offset; or
damped sines or wavelets synthesized to a target spectrum), the spectrum's Q and the oscillator
looked at, and things to try. Centre: the pulse, its velocity, and the SRS (maximax, primary and
residual, positive and negative) with the chain's modes marked. Right: the oscillator picked on
the spectrum; the chain's own peaks from the SRS mode by mode (ABS, SRSS, NRL) against its exact
response; the tripartite plot; the classical pulses' spectra side by side; and the notes.
Follows the chain from the Simulation page (friction left out: all of this is linear).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.modal import TWO_PI, ModalResult, modal_analysis
from ..core.model import ChainSystem
from ..core.practice import Problem, Question, Requirement, at_least, at_most, kept, number, pick, rounded
from ..core.random_vibration import G, linear
from ..core.shock import (
    CLASSICAL,
    SYNTH_Q,
    ChainPeaks,
    Pulse,
    Record,
    Response,
    Rule,
    Shape,
    Spectrum,
    bank,
    chain_peaks,
    classical,
    oscillator,
    record,
    sample_rate,
    srs,
    target,
    target_band,
)
from . import weak
from .axes import log_axes
from .layouts import ColumnScroll, FlowLayout, form_layout, group, plot_height
from .lesson import TrialCombo
from .links import load_notes, page_link
from .notes_view import NotesBrowser
from .panels import spin
from .style import MAX_DOF, colors
from .tab_groups import LazyTabs
from .theming import add_legend, mute

RECOMPUTE_MS = 150  # the parameters settle this long before the spectrum is worked out again
DEFAULT_Q = 10.0
TAIL_PERIODS = 2.0  # of the picked oscillator, shown after the record
TOLERANCE_DB = 3.0  # the band drawn round a synthesis's target
HISTORY_PERIODS = 1.5  # of mode 1, shown after the record on the Chain tab
RULE_SYMBOLS = {Rule.ABS: "t1", Rule.SRSS: "t", Rule.NRL: "d"}

TIPS = {
    "shape": "<p>The base's acceleration. <b>Classical</b> pulses, as shock tests specify them: half-sine, "
    "haversine, the two sawtooths, a trapezoid, a square. <b>Measured at a mass</b>: the absolute "
    "acceleration of a mass on the chain, its base given a half-sine (the shock a component mounted "
    "there sees). <b>Measured transient</b>: damped sines at scattered frequencies, as a pyrotechnic "
    "shock is. <b>Damped sines</b> or <b>wavelets to a spectrum</b>: a record built to meet a target SRS (a "
    "plateau above a knee, rising 6 dB an octave below it), as a shaker test is.</p>",
    "peak": "<p>The pulse's peak acceleration, in g (a measured transient is scaled to it; a measured mass's "
    "record comes from a base half-sine this high).</p>",
    "duration": "<p>The pulse's duration T (a measured transient's length; a measured mass's base half-sine). "
    "The spectrum's shape depends on f T.</p>",
    "mass": "<p>The mass whose acceleration is the measured record.</p>",
    "noise": "<p>White noise added to a measured record, its RMS a fraction of the peak, as an accelerometer's "
    "and its amplifier's.</p>",
    "offset": "<p>A constant added to a measured record, a fraction of the peak: an accelerometer's zero shift, "
    "common in pyrotechnic shocks. Integrated, it is a velocity change the shock didn't have.</p>",
    "knee": "<p>The target spectrum's knee: a plateau at the peak above it, rising 6 dB an octave below it, "
    "from a tenth of it to ten times it.</p>",
    "again": "<p>The same record with other noise (a synthesis: other delays between its components).</p>",
    "q": "<p>The oscillators' Q = 1/2ζ: the same for every one. Shock specifications use Q = 10 (ζ = 5 %).</p>",
    "split": "<p>Show the <b>primary</b> spectrum (the peaks while the pulse lasts) and the <b>residual</b> one "
    "(after it, the free decay), and the <b>positive</b> and <b>negative</b> peaks apart. The maximax "
    "is the largest of them. For a measured record, primary means within the record.</p>",
    "f": "<p>The oscillator shown on the right. Drag the line on the spectrum, or click it.</p>",
}
PULSE_TIP = "<p>The base's acceleration (for a measured mass, the base's half-sine is grey).</p>"
VELOCITY_TIP = (
    "<p>The base's velocity, ∫a dt. The change Δv over the pulse decides the spectrum's low end: a soft "
    "oscillator takes the pulse as an impulse.</p>"
)
SRS_TIP = (
    "<p>The shock response spectrum: each oscillator's largest absolute acceleration, against its natural "
    "frequency. Dashed: the pulse's peak (the stiff oscillators' asymptote) and 2π f Δv (the soft ones'). "
    "Dotted: the chain's modes.</p>"
)
OSC_TIP = "<p>The picked oscillator's absolute acceleration, the pulse behind it; the dot marks its largest value.</p>"
DISP_TIP = "<p>The oscillator's displacement relative to the base: its spring's stretch.</p>"
PEAKS_TIP = (
    "<p>The chain's peak at each mass (or spring) on this base: exact (bars), and from the SRS at each mode's "
    "own frequency and damping, combined by ABS (the sum), SRSS (root of the sum of squares) and NRL (the "
    "largest plus the root of the others' squares).</p>"
)
HISTORY_TIP = (
    "<p>The exact response at the output picked (thick), and each mode's part of it (thin), Γ φ times "
    "its oscillator's response; the dot marks the exact peak, the dashes each mode's peak from the SRS. "
    "The modes peak together, early, which the SRSS doesn't allow for.</p>"
)
TRIPARTITE_TIP = (
    "<p>The pseudo-velocity ω |z|max against frequency, on four-way paper: the diagonals rising to the right are "
    "lines of constant relative displacement |z|max, those falling are constant pseudo-acceleration ω² |z|max "
    "(which is close to the SRS at light damping). Dashed: the base's Δv and peak acceleration.</p>"
)
SHAPES_TIP = (
    "<p>The classical pulses at the same peak and duration (or the same Δv), their spectra over f T at "
    "the page's Q.</p>"
)

# Things to try: (name, what it sets, note). The duration is set from the chain (`duration_for`).
TRIALS: tuple[tuple[str, dict[str, Any], str], ...] = (
    (
        "Half-sine",
        {},
        "A 10 g half-sine. At Q = 10 the spectrum peaks at 1.65 × the pulse's peak, at f T = 0.83: an oscillator "
        "whose period is a little longer than the pulse. Stiff oscillators follow the pulse (1 ×); soft ones take it as "
        "an impulse and ring with ω Δv, the dashed line at the left.",
    ),
    (
        "Square",
        {"shape": Shape.SQUARE},
        "A square pulse: a step up and one down. Every stiff oscillator overshoots the step, to 1.85 × at Q = 10 "
        "(2 × undamped), so the spectrum stays high across the top. Its Δv is the largest of the pulses (the area "
        "A T), so it is highest at the left too.",
    ),
    (
        "Terminal-peak sawtooth",
        {"shape": Shape.TERMINAL_PEAK},
        "Rising slowly, dropping at once: the drop makes stiff oscillators ring the other way, so above f T ≈ 2 "
        "the negative spectrum is 0.85–0.86 × the positive one. Tests like it for that: a part is loaded both ways.",
    ),
    (
        "Haversine",
        {"shape": Shape.HAVERSINE},
        "A smooth pulse: no corner, so stiff oscillators don't ring and the negative spectrum at the top is "
        "nearly nothing. Its peak, 1.60 × at f T = 1.0, is close to the half-sine's.",
    ),
    (
        "Light damping, Q = 50",
        {"q": 50.0},
        "The half-sine at Q = 50: the spectrum hardly changes (1.74 × at the peak against 1.65 ×). A shock is over "
        "before damping has time to act on the primary peak, unlike a steady sine, whose response grows with Q.",
    ),
    (
        "A component on the last mass",
        {"shape": Shape.AT_A_MASS},
        "The last mass's acceleration with the base given the half-sine: it rings at the chain's modes, and its "
        "spectrum peaks there, 3.3 × the record's own peak at mode 1. A component tuned near the chain's mode "
        "would be hit far harder than the record's peak suggests.",
    ),
    (
        "Pyrotechnic transient",
        {"shape": Shape.TRANSIENT, "noise": 0.02},
        "A burst of damped sines with a little noise: the spectrum peaks among its frequencies, at 4.5 × the "
        "record's peak (f T ≈ 38). It ends at rest, with no net Δv, so the soft oscillators at the left feel only "
        "their dampers' push: at f T = 0.02 the spectrum is 10⁻⁴ of the peak.",
    ),
    (
        "Zero shift",
        {"shape": Shape.TRANSIENT, "noise": 0.02, "offset": 0.01},
        "The same record with a 1 % offset, an accelerometer's zero shift: integrated over the record it is a "
        "velocity change the shock didn't have, and the spectrum's low end climbs to the 2π f Δv line, ten times "
        "what it was at f T = 0.02. A measured spectrum whose low end follows that line is suspect.",
    ),
    (
        "Modes combined",
        {"tab": "chain_tab"},
        "The half-sine on the chain. Each mode is an oscillator at its own f and ζ, driven by Γ times the base, so "
        "the SRS gives each mode's peak but not when it comes. On m4, exactly 9.14 g; ABS (their sum) 11 g, +20 %; "
        "SRSS 6.62 g, −28 %; NRL 9.27 g, +1 %. The SRSS assumes the modes peak at unrelated times, but a shock sets "
        "them all off at once: each one's part peaks within the first 0.3 s, a third of mode 1's period.",
    ),
    (
        "Four-way paper",
        {"tab": "tri_tab"},
        "The half-sine's spectrum as a pseudo-velocity, read against diagonals of constant displacement and "
        "acceleration. At the left it tends to the base's Δv = 2AT/π, 6.24 m/s; at the right it falls along the "
        "10 g line. Between, it says how far each oscillator's spring stretches: 4.6 m at the softest, the rattle "
        "space a soft mount would need.",
    ),
    (
        "Pulse shapes compared",
        {"tab": "shapes_tab"},
        "The six pulses at the same peak and duration. The square has the most area, so it is highest at the left "
        "(0.12 × at f T = 0.02) and, overshooting, near the top (1.86 ×). The terminal-peak sawtooth rises gently "
        "and peaks at only 1.19 ×. Tick Same velocity change: the low ends meet, and the narrower pulses, now "
        "taller, take the top: the initial-peak sawtooth reaches 2.33 ×, the square only 1.18 ×.",
    ),
    (
        "Damped sines to a spectrum",
        {"shape": Shape.SINES},
        "A test is specified by its spectrum: here 10 g above a 5 Hz knee, rising 6 dB an octave below it. 28 "
        "damped sines, scaled round by round, meet it within ±3 dB over the band, though the record's own peak is "
        "only 4.8 g: each oscillator rings up on the sine at its frequency. But each sine carries a net velocity, so "
        "the record ends at 0.51 m/s, 6 m from where it began, which no shaker can follow.",
    ),
    (
        "Wavelets to a spectrum",
        {"shape": Shape.WAVELETS},
        "The same target met by 28 wavelets, each sin(ωt/N) sin ωt over N half-cycles, N odd: each starts and ends at "
        "rest, so the record does too, within a micrometre, and a shaker can play it: the table moves at most "
        "0.54 m and comes back.",
    ),
)

NOTES_HTML = load_notes("shock")


TAB_OSCILLATOR, TAB_CHAIN, TAB_TRIPARTITE, TAB_SHAPES = range(4)


@dataclass(frozen=True)
class ShockSetup:
    """A practice problem's setup: a chain (none: the one there is) and the page's settings, in the boxes' units
    (the duration in ms, noise and offset in %). `mass` counts from the end when negative."""

    masses: tuple[float, ...] = ()
    springs: tuple[float, ...] = ()
    dampers: tuple[float, ...] = ()
    shape: Shape = Shape.HALF_SINE
    peak: float = 10.0
    duration: float = 100.0
    knee: float = 5.0
    mass: int = -1
    noise: float = 0.0
    offset: float = 0.0
    q: float = DEFAULT_Q
    tab: int = TAB_OSCILLATOR

    def system(self) -> ChainSystem | None:
        if not self.masses:
            return None
        return ChainSystem(np.array(self.masses), np.array(self.springs), np.array(self.dampers))


PRACTICE_SETUP = ShockSetup  # what a practice problem's setup and the page's practice_state are

# ------------------------------------------------------------------ practice problems
# Each question's numbers are printed round, and its answer is worked from them as printed.
AREA = {Shape.HALF_SINE: 2 / math.pi, Shape.HAVERSINE: 0.5, Shape.TERMINAL_PEAK: 0.5, Shape.SQUARE: 1.0}
UNIFORM = dict(masses=(1.0,) * 4, springs=(400.0,) * 4, dampers=(2.0,) * 4)  # the default chain


def _peak_srs(pulse: Pulse, f: np.ndarray, q: float = DEFAULT_Q) -> np.ndarray:
    """The maximax (g) of the pulse's SRS at f, at Q."""
    rec = record(pulse, sample_rate(pulse, float(np.max(f))))
    return srs(rec, f, 1 / (2 * q)).maximax / G


@functools.cache
def _half_sine_peak() -> tuple[float, float]:
    """(f T, SRS over the pulse's peak) where a half-sine's spectrum peaks, at Q = 10."""
    x = np.linspace(0.6, 1.1, 101)
    s = _peak_srs(Pulse(Shape.HALF_SINE, 1.0, 1.0), x)
    i = int(np.argmax(s))
    return float(x[i]), float(s[i])


def _velocity_change(rng: np.random.Generator) -> Question:
    shape = (Shape.HALF_SINE, Shape.HAVERSINE, Shape.TERMINAL_PEAK, Shape.SQUARE)[int(rng.integers(4))]
    A, T = pick(rng, (10, 20, 50, 100)), pick(rng, (5, 6, 11, 20))
    dv = AREA[shape] * A * G * T / 1000
    area = {Shape.HALF_SINE: "2AT/π", Shape.SQUARE: "AT"}.get(shape, "AT/2")
    return Question(
        f"A {shape.value.lower()} pulse, {A:g} g for {T:g} ms. What velocity change does it give the base?",
        dv,
        f"Δv is the pulse's area, {area}: A = {A:g} × 9.807 = {number(A * G)} m/s², T = {T / 1000:g} s, so Δv = "
        f"{number(dv)} m/s.",
        unit="m/s",
        setup=dict(shape=shape, peak=A, duration=T),
    )


def _soft_asymptote(rng: np.random.Generator) -> Question:
    A, T = pick(rng, (10, 20, 50)), pick(rng, (5, 10, 20))
    f = pick(rng, (0.25, 0.5, 1)) * 10 / T  # f T = 0.0025 to 0.01
    dv = 2 * A * G * T / 1000 / math.pi
    peak = TWO_PI * f * dv / G
    page = float(_peak_srs(Pulse(Shape.HALF_SINE, A, T / 1000), np.array([f]))[0])
    return Question(
        f"A {A:g} g, {T:g} ms half-sine. What peak acceleration does a soft oscillator at {number(f)} Hz see, by the "
        "spectrum's low-frequency asymptote?",
        peak,
        f"f T = {number(f * T / 1000)} ≪ 1: it takes the pulse as an impulse, Δv = 2AT/π = {number(dv)} m/s, and "
        f"rings at ω Δv = 2π × {number(f)} × {number(dv)} = {number(peak * G)} m/s², {number(peak)} g. (The page, "
        f"at Q = 10, reads {number(page)} g: the damping takes e<sup>−ζπ/2</sup> off before the first peak.)",
        unit="g",
        setup=dict(peak=A, duration=float(T)),
    )


def _rattle_space(rng: np.random.Generator) -> Question:
    A, T = pick(rng, (10, 20)), pick(rng, (5, 10, 20))
    f = pick(rng, (1, 2)) * 10 / T  # f T = 0.01 or 0.02
    dv = 2 * A * G * T / 1000 / math.pi
    z = dv / (TWO_PI * f)
    return Question(
        f"A soft mount at {number(f)} Hz on a base given a {A:g} g, {T:g} ms half-sine. About how far does its spring "
        "stretch?",
        1000 * z,
        f"The mount stays put while the base moves off at Δv = 2AT/π = {number(dv)} m/s, then rings at that speed: "
        f"|z| = Δv/ω = {number(dv)}/(2π × {number(f)}) = {number(1000 * z)} mm, the rattle space it needs (the "
        "tripartite plot's flat left end; at Q = 10 the damping takes a few percent off its first swing).",
        unit="mm",
        setup=dict(peak=A, duration=float(T), tab=TAB_TRIPARTITE),
    )


def _hardest_hit(rng: np.random.Generator) -> Question:
    T = pick(rng, (2, 5, 6, 11, 20))
    x, _ = _half_sine_peak()
    return Question(
        f"A {T:g} ms half-sine, Q = 10. A component of which natural frequency does it hit hardest?",
        1000 * x / T,
        f"A half-sine's spectrum peaks at f T ≈ {number(x)}, an oscillator whose period is a little longer than the "
        f"pulse: f = {number(x)}/{T / 1000:g} s = {number(1000 * x / T)} Hz.",
        unit="Hz",
        tolerance=0.05,
        setup=dict(duration=float(T)),
    )


def _component_force(rng: np.random.Generator) -> Question:
    m, S = pick(rng, (0.2, 0.5, 2, 5)), pick(rng, (15, 30, 80, 150))
    F = m * S * G
    return Question(
        f"A {m:g} kg component's natural frequency is where a shock's SRS reads {S:g} g. What peak force does its "
        "mount carry?",
        F,
        f"The SRS is the mass's peak absolute acceleration, and the mount's spring and damper push it there: "
        f"F = m × SRS = {m:g} × {S:g} × 9.807 = {number(F)} N.",
        unit="N",
    )


def _combine(rng: np.random.Generator) -> Question:
    a, b, c = pick(rng, (3, 4, 5, 6)), pick(rng, (1.5, 2, 2.5)), pick(rng, (0.5, 1))
    rule = (Rule.ABS, Rule.SRSS, Rule.NRL)[int(rng.integers(3))]
    value = {
        Rule.ABS: a + b + c,
        Rule.SRSS: math.sqrt(a * a + b * b + c * c),
        Rule.NRL: a + math.sqrt(b * b + c * c),
    }[rule]
    how = {
        Rule.ABS: f"ABS adds them: {a:g} + {b:g} + {c:g}",
        Rule.SRSS: f"SRSS is √({a:g}² + {b:g}² + {c:g}²)",
        Rule.NRL: f"NRL is the largest plus the root of the others' squares: {a:g} + √({b:g}² + {c:g}²)",
    }[rule]
    return Question(
        f"Three modes' peak terms at a mass, φΓ times the SRS at each mode: {a:g}, {b:g} and {c:g} g. What peak "
        f"does the {rule.value} rule give?",
        value,
        f"{how} = {number(value)} g.",
        unit="g",
        setup=dict(**UNIFORM, tab=TAB_CHAIN),
    )


def _more_q(rng: np.random.Generator) -> Question:
    return Question(
        "A half-sine's spectrum at Q = 10 peaks at 1.65 × the pulse's peak. What does it peak at with Q = 50?",
        0,
        "Hardly more, 1.74 ×: the shock is over before the damping has acted on the primary peak. A steady sine's "
        "resonance would be five times higher.",
        choices=("Hardly more", "About five times as much", "About √5 times as much", "Less"),
        setup=dict(q=50.0),
        predict=True,
    )


def _square_top(rng: np.random.Generator) -> Question:
    return Question(
        "A square pulse, a step up and a step down. At the top of its spectrum (stiff oscillators), how high is the "
        "SRS against the pulse's peak?",
        0,
        "Nearly twice it, 1.85 × at Q = 10: a stiff oscillator overshoots a step, to 2 × undamped. A half-sine's top "
        "falls to 1 ×.",
        choices=("Nearly twice it", "About the same", "Far below it"),
        setup=dict(shape=Shape.SQUARE, duration=11.0),
        predict=True,
    )


def _srss_short(rng: np.random.Generator) -> Question:
    return Question(
        "The default chain on a base given a 10 g, 100 ms half-sine, its modes' peaks from the SRS combined by SRSS. "
        "Against the exact peak acceleration at the last mass, the SRSS comes out…",
        0,
        "…short: 6.62 g against 9.14 g exactly, −28 %. A shock sets every mode off at once and they peak nearly "
        "together, early; the SRSS assumes unrelated times. ABS gives 11.0 g, NRL 9.27 g.",
        choices=("Short", "About right", "Over"),
        setup=dict(**UNIFORM, duration=100.0, tab=TAB_CHAIN),
        predict=True,
    )


def _zero_shift(rng: np.random.Generator) -> Question:
    return Question(
        "A pyrotechnic transient's record gets a 1 % offset, an accelerometer's zero shift. What happens to its "
        "spectrum's low end?",
        0,
        "It rises, about ten times at 0.2 Hz: integrated over the record the offset is a velocity change the shock "
        "didn't have, and the soft oscillators ring with it (the 2π f Δv line).",
        choices=("It rises a lot", "It hardly changes", "It falls"),
        setup=dict(shape=Shape.TRANSIENT, duration=100.0, noise=2.0, offset=1.0),
        predict=True,
    )


def _test_spec(rng: np.random.Generator) -> Question:
    fc = pick(rng, (50, 100, 200, 500))
    A = pick(rng, (20, 50, 100))
    x, top = _half_sine_peak()
    T = round(1000 * x / fc, 2)  # ms, as the box holds it
    reach = float(_peak_srs(Pulse(Shape.HALF_SINE, A, T / 1000), np.array([fc]))[0])
    lowest = rounded(0.95 * reach, 2)
    highest = rounded(1.25 * A, 2)
    dv = AREA[Shape.HALF_SINE] * A * G * T / 1000
    most_dv = rounded(1.1 * dv, 2)

    def goal(t: ShockSetup):
        pulse = Pulse(t.shape, t.peak, 1e-3 * t.duration)
        rec = record(pulse, sample_rate(pulse, fc))
        at = float(srs(rec, np.array([fc]), 1 / (2 * t.q)).maximax[0]) / G
        return (
            Requirement("A half-sine", t.shape.value, t.shape is Shape.HALF_SINE),
            kept("Q", t.q, DEFAULT_Q),
            at_least(f"The SRS at {number(fc)} Hz", at, lowest, "g"),
            at_most("The peak", t.peak, highest, "g"),
            at_most("The velocity change", rec.velocity_change, most_dv, "m/s"),
        )

    return Question(
        f"A shock test on a half-sine machine must put {number(lowest, 2)} g or more (SRS, Q = 10) into a component at "
        f"{fc:g} Hz. The machine gives at most {number(highest, 2)} g and a velocity change of at most "
        f"{number(most_dv, 2)} m/s. Choose the pulse's peak and duration.",
        0,
        f"The spectrum is A times a curve in f T, highest ({number(top)} ×) at f T ≈ {number(x)}: T = "
        f"{number(x)}/{fc:g} Hz = {T:g} ms. Then A = {A:g} g gives {number(reach)} g at {fc:g} Hz, with "
        f"Δv = 2AT/π = {number(dv)} m/s. A shorter pulse at the peak limit does too, its Δv lower; a longer one soon "
        "runs out of velocity change.",
        setup=dict(peak=float(A), duration=round(10 * T, 2)),
        goal=goal,
        design=dict(peak=float(A), duration=T),
    )


PROBLEMS = tuple(
    Problem(key, title, "shock_page", f"See {page_link('shock_page', anchor, section)}.", make)
    for key, title, anchor, section, make in (
        ("velocity-change", "A pulse's velocity change", "regions", "Three regions", _velocity_change),
        ("soft-asymptote", "A soft oscillator's peak", "regions", "Three regions", _soft_asymptote),
        ("rattle-space", "A soft mount's stretch", "tripartite", "Four-way paper", _rattle_space),
        ("hardest-hit", "Which component it hits hardest", "regions", "Three regions", _hardest_hit),
        ("component-force", "A component's mount force", "srs", "The shock response spectrum", _component_force),
        ("combine", "Modes' peaks combined", "chain", "The chain's peak from the spectrum", _combine),
        ("more-q", "Predict: lighter damping", "damping", "Damping", _more_q),
        ("square-top", "Predict: a square pulse's top", "regions", "Three regions", _square_top),
        ("srss-short", "Predict: SRSS on a shock", "chain", "The chain's peak from the spectrum", _srss_short),
        ("zero-shift", "Predict: a zero shift", "measured", "Measured shocks", _zero_shift),
        ("test-spec", "Spec: a shock test's pulse", "regions", "Three regions", _test_spec),
    )
)


def _plot(left: str, left_units: str | None, bottom: str, bottom_units: str | None, log: bool = False) -> pg.PlotWidget:
    p = pg.PlotWidget(axisItems=log_axes()) if log else pg.PlotWidget()
    p.setMenuEnabled(False)
    p.hideButtons()
    p.showGrid(x=True, y=True, alpha=0.15)
    p.setLabel("left", left, units=left_units)
    p.setLabel("bottom", bottom, units=bottom_units)
    return p


def _form_row(form: QtWidgets.QFormLayout, text: str, box: QtWidgets.QWidget, tip: str) -> QtWidgets.QLabel:
    box.setToolTip(tip)
    label = QtWidgets.QLabel(text)
    label.setToolTip(tip)
    form.addRow(label, box)
    return label


def duration_for(result: ModalResult) -> float:
    """A pulse that spreads the chain's modes across the spectrum: 0.6 of its fastest mode's period, rounded to
    1, 2 or 5 ms times a power of ten."""
    top = max((m.fn_hz for m in result.modes if m.fn_hz > 0), default=6.0)
    return _nice(0.6 / top)


def knee_for(result: ModalResult) -> float:
    """A synthesis's knee at the chain's fastest mode, rounded as `duration_for` does, so its modes sit on the
    target's slope and the band reaches below mode 1."""
    return _nice(max((m.fn_hz for m in result.modes if m.fn_hz > 0), default=5.0))


def _same_chain(a: ChainSystem, b: ChainSystem) -> bool:
    return a.n == b.n and all(
        np.allclose(x, y) for x, y in ((a.masses, b.masses), (a.stiffness, b.stiffness), (a.damping, b.damping))
    )


def _nice(x: float) -> float:
    e = 10.0 ** math.floor(math.log10(x))
    return min((m * e for m in (1, 2, 5, 10)), key=lambda v: abs(math.log(v / x)))


class ShockPage(QtWidgets.QWidget):
    """Controls on the left, the pulse and its spectrum in the centre, the picked oscillator and notes on the right."""

    dof_requested = QtCore.Signal(int)  # the user changed N here; the main window owns the system
    system_requested = QtCore.Signal(object)  # a practice problem's chain, for the Simulation page
    edit_parameters = QtCore.Signal()  # the user wants the Simulation page's parameter panel

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.system: ChainSystem | None = None
        self.result: ModalResult | None = None
        self.record: Record | None = None
        self.spectrum: Spectrum | None = None
        self.chain: ChainPeaks | str | None = None  # the chain's peaks, or why they couldn't be worked out
        self.seed = 1
        self._loading = False
        self._dirty = False
        self._rerecord = True
        self._recompute = QtCore.QTimer(self)
        self._recompute.setSingleShot(True)
        self._recompute.setInterval(RECOMPUTE_MS)
        self._recompute.timeout.connect(self.recompute)

        # --- left: the pulse, the spectrum, things to try
        box = QtWidgets.QGroupBox("Pulse")
        form = form_layout(box)
        self.dof = QtWidgets.QSpinBox()
        self.dof.setRange(1, MAX_DOF)
        self.dof.setToolTip("Same chain as the Simulation page; changing it here changes it there too.")
        self.dof.valueChanged.connect(self.dof_requested)
        form.addRow("Number of masses:", self.dof)
        self.shape = QtWidgets.QComboBox()
        for s in Shape:
            self.shape.addItem(s.value, s)
        self.shape.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        _form_row(form, "Shape:", self.shape, TIPS["shape"])
        self.peak = spin(0.001, 1e5, 10.0, 3, " g")
        self.peak_label = _form_row(form, "Peak:", self.peak, TIPS["peak"])
        self.duration = spin(0.01, 1e5, 100.0, 2, " ms")
        self.duration_label = _form_row(form, "Duration:", self.duration, TIPS["duration"])
        self.knee = spin(1e-3, 1e6, 5.0, 3, " Hz")
        self.knee_label = _form_row(form, "Knee:", self.knee, TIPS["knee"])
        self.mass = QtWidgets.QComboBox()
        self.mass_label = _form_row(form, "Measured on:", self.mass, TIPS["mass"])
        self.noise = spin(0.0, 50.0, 0.0, 2, " %")
        self.noise_label = _form_row(form, "Noise:", self.noise, TIPS["noise"])
        self.offset = spin(-50.0, 50.0, 0.0, 2, " %")
        self.offset_label = _form_row(form, "Offset:", self.offset, TIPS["offset"])
        self.again = QtWidgets.QPushButton("New noise")
        self.again.setToolTip(TIPS["again"])
        form.addRow(self.again)
        self.pulse_info = QtWidgets.QLabel()
        self.pulse_info.setWordWrap(True)
        mute(self.pulse_info)
        form.addRow(self.pulse_info)
        link = QtWidgets.QLabel("Masses, springs and dampers are edited on the <a href='#sim'>Simulation page</a>.")
        mute(link)
        link.setWordWrap(True)
        link.linkActivated.connect(self.edit_parameters)
        form.addRow(link)

        spec_box = QtWidgets.QGroupBox("Spectrum")
        form = form_layout(spec_box)
        self.q = spin(1.0, 500.0, DEFAULT_Q, 1, "")
        _form_row(form, "Q:", self.q, TIPS["q"])
        self.split_time = QtWidgets.QCheckBox("Primary and residual")
        self.split_time.setToolTip(TIPS["split"])
        self.split_sign = QtWidgets.QCheckBox("Positive and negative")
        self.split_sign.setToolTip(TIPS["split"])
        form.addRow(group(self.split_time))
        form.addRow(group(self.split_sign))
        self.f_pick = spin(1e-4, 1e7, 1.0, 3, " Hz")
        _form_row(form, "Oscillator:", self.f_pick, TIPS["f"])

        trials = QtWidgets.QGroupBox("Things to try")
        tv = QtWidgets.QVBoxLayout(trials)
        self.trial = TrialCombo([name for name, _, _ in TRIALS], prompt=None)
        self.trial.setToolTip(
            "<p>Pulses that show something about shock spectra, the duration set from the chain as it is now.</p>"
        )
        tv.addWidget(self.trial)
        self.trial_note = QtWidgets.QLabel()
        self.trial_note.setWordWrap(True)
        mute(self.trial_note)
        tv.addWidget(self.trial_note)

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        for b in (box, spec_box, trials):
            lv.addWidget(b)
        lv.addStretch(1)
        left_scroll = ColumnScroll(left)

        # --- centre: the pulse, its velocity, the spectrum
        self.pulse_plot = _plot("Base acceleration", None, "Time", "s")
        self.pulse_plot.setToolTip(PULSE_TIP)
        plot_height(self.pulse_plot, 110)
        self.velocity_plot = _plot("v (m/s)", None, "Time", "s")
        self.velocity_plot.setToolTip(VELOCITY_TIP)
        plot_height(self.velocity_plot, 90)
        self.velocity_plot.getAxis("left").enableAutoSIPrefix(False)
        self.velocity_plot.setXLink(self.pulse_plot)
        self.srs_plot = _plot("Peak acceleration (g)", None, "Natural frequency", "Hz", log=True)
        self.srs_plot.setLogMode(x=True, y=True)
        self.srs_plot.getAxis("left").enableAutoSIPrefix(False)
        self.srs_plot.setToolTip(SRS_TIP)
        plot_height(self.srs_plot, 220)
        self.srs_legend = add_legend(self.srs_plot.getPlotItem(), offset=(-10, -10))
        self.pick_line = pg.InfiniteLine(0.0, angle=90, movable=True)
        self.pick_line.setZValue(10)
        self.srs_plot.addItem(self.pick_line)
        self.readout = QtWidgets.QLabel()
        self.readout.setWordWrap(True)
        self.readout.setTextFormat(QtCore.Qt.TextFormat.RichText)
        center = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(center)
        cv.addWidget(self.pulse_plot, 2)
        cv.addWidget(self.velocity_plot, 1)
        cv.addWidget(self.srs_plot, 5)
        cv.addWidget(self.readout)

        # --- right: the oscillator, the notes
        self.osc_plot = _plot("Absolute acceleration (g)", None, "Time", "s")
        self.osc_plot.setToolTip(OSC_TIP)
        plot_height(self.osc_plot, 160)
        self.osc_legend = add_legend(self.osc_plot.getPlotItem(), offset=(-10, 10))
        self.disp_plot = _plot("Relative displacement", "m", "Time", "s")
        self.disp_plot.setToolTip(DISP_TIP)
        plot_height(self.disp_plot, 110)
        self.disp_plot.setXLink(self.osc_plot)
        self.osc_note = QtWidgets.QLabel()
        self.osc_note.setWordWrap(True)
        self.osc_note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        osc_tab = QtWidgets.QWidget()
        ov = QtWidgets.QVBoxLayout(osc_tab)
        ov.addWidget(self.osc_plot, 3)
        ov.addWidget(self.disp_plot, 2)
        ov.addWidget(self.osc_note)
        self.osc_tab = osc_tab

        # the chain's peaks: response and output, peaks per output, the output's history, the table
        self.response = QtWidgets.QComboBox()
        for r in Response:
            self.response.addItem(r.value, r)
        self.output = QtWidgets.QComboBox()
        chain_controls = QtWidgets.QWidget()
        flow = FlowLayout(chain_controls)
        flow.addWidget(group("Response:", self.response))
        flow.addWidget(group("Output:", self.output))
        self.peaks_plot = _plot("Peak", None, "Mass", None)
        self.peaks_plot.setToolTip(PEAKS_TIP)
        self.peaks_plot.getAxis("left").enableAutoSIPrefix(False)
        plot_height(self.peaks_plot, 130)
        self.peaks_legend = add_legend(self.peaks_plot.getPlotItem(), offset=(10, 5), colCount=4)
        self.history_plot = _plot("Response", None, "Time", "s")
        self.history_plot.setToolTip(HISTORY_TIP)
        self.history_plot.getAxis("left").enableAutoSIPrefix(False)
        plot_height(self.history_plot, 130)
        self.history_legend = add_legend(self.history_plot.getPlotItem(), offset=(10, 5), colCount=3)
        self.chain_table = QtWidgets.QTableWidget()
        self.chain_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.chain_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.chain_table.setColumnCount(4)
        self.chain_table.setHorizontalHeaderLabels(["Exact"] + [r.value for r in Rule])
        self.chain_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.chain_table.setMinimumHeight(90)
        self.chain_note = QtWidgets.QLabel()
        self.chain_note.setWordWrap(True)
        self.chain_note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.chain_tab = QtWidgets.QWidget()
        chv = QtWidgets.QVBoxLayout(self.chain_tab)
        chv.addWidget(chain_controls)
        chv.addWidget(self.peaks_plot, 3)
        chv.addWidget(self.history_plot, 3)
        chv.addWidget(self.chain_table, 2)
        chv.addWidget(self.chain_note)

        self.tri_plot = _plot("Pseudo-velocity (m/s)", None, "Natural frequency", "Hz", log=True)
        self.tri_plot.setLogMode(x=True, y=True)
        self.tri_plot.getAxis("left").enableAutoSIPrefix(False)
        self.tri_plot.setToolTip(TRIPARTITE_TIP)
        plot_height(self.tri_plot, 240)
        self.tri_note = QtWidgets.QLabel()
        self.tri_note.setWordWrap(True)
        self.tri_note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.tri_tab = QtWidgets.QWidget()
        tv = QtWidgets.QVBoxLayout(self.tri_tab)
        tv.addWidget(self.tri_plot, 1)
        tv.addWidget(self.tri_note)

        self.same_dv = QtWidgets.QCheckBox("Same velocity change")
        self.same_dv.setToolTip("<p>Scale each pulse to the half-sine's Δv instead of its peak.</p>")
        self.shapes_plot = _plot("a / A", None, "t / T", None)
        self.shapes_plot.setToolTip(SHAPES_TIP)
        plot_height(self.shapes_plot, 100)
        self.compare_plot = _plot("Peak / A", None, "f T", None, log=True)
        self.compare_plot.setLogMode(x=True, y=True)
        self.compare_plot.getAxis("left").enableAutoSIPrefix(False)
        self.compare_plot.setToolTip(SHAPES_TIP)
        plot_height(self.compare_plot, 200)
        self.compare_legend = add_legend(self.compare_plot.getPlotItem(), offset=(-10, -10))
        self.shapes_note = QtWidgets.QLabel()
        self.shapes_note.setWordWrap(True)
        self.shapes_note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.shapes_tab = QtWidgets.QWidget()
        sv = QtWidgets.QVBoxLayout(self.shapes_tab)
        sv.addWidget(self.same_dv)
        sv.addWidget(self.shapes_plot, 2)
        sv.addWidget(self.compare_plot, 5)
        sv.addWidget(self.shapes_note)
        self._shapes_key: tuple | None = None

        self.notes = NotesBrowser()
        self.notes.setHtml(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(osc_tab, "Oscillator")
        self.tabs.addTab(self.chain_tab, "Chain")
        self.tabs.addTab(self.tri_tab, "Tripartite")
        self.tabs.addTab(self.shapes_tab, "Pulse shapes")
        self.tabs.addTab(self.notes, "Theory")
        self.lazy = LazyTabs(  # tabs drawn only while they show
            self.tabs,
            {self.chain_tab: self._draw_chain, self.tri_tab: self._draw_tripartite, self.shapes_tab: self._draw_shapes},
        )

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(center)
        splitter.addWidget(self.tabs)
        splitter.setSizes([340, 810, 550])
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        # --- wiring
        self.shape.currentIndexChanged.connect(self._on_pulse)
        self.mass.currentIndexChanged.connect(self._on_pulse)
        for field in (self.peak, self.duration, self.knee, self.noise, self.offset):
            field.valueChanged.connect(self._on_pulse)
        self.q.valueChanged.connect(self._on_spectrum)
        for c in (self.split_time, self.split_sign):
            c.toggled.connect(self._draw_srs)
        self.again.clicked.connect(self._on_again)
        self.trial.picked.connect(self.apply_trial)
        self.pick_line.sigPositionChanged.connect(self._on_line)
        self.f_pick.valueChanged.connect(self._on_pick)
        self.srs_plot.scene().sigMouseClicked.connect(self._on_click)
        self.response.currentIndexChanged.connect(self._on_response)
        self.output.currentIndexChanged.connect(weak.slot(self, lambda self, _: self.lazy.stale(self.chain_tab)))
        self.same_dv.toggled.connect(weak.slot(self, lambda self, _: self.lazy.stale(self.shapes_tab)))

    # ------------------------------------------------------------- inputs
    def set_system(self, system: ChainSystem, result: ModalResult | None = None) -> None:
        """New chain parameters: the trial's settings follow its modes, and it's worked out again (now if shown)."""
        n_changed = self.system is None or system.n != self.system.n
        self.system = linear(system)
        self.result = modal_analysis(self.system) if result is None or system.friction.any() else result
        self._loading = True
        self.dof.blockSignals(True)
        self.dof.setValue(system.n)
        self.dof.blockSignals(False)
        if n_changed:
            mass = self.mass.currentIndex()
            self.mass.clear()
            self.mass.addItems([f"m{i + 1}" for i in range(system.n)])
            self.mass.setCurrentIndex(mass if 0 <= mass < system.n else system.n - 1)
            self._fill_outputs(system.n - 1)
        self.chain = None
        self._loading = False
        self.apply_trial()

    def apply_trial(self) -> None:
        """Set the pulse the chosen trial asks for, its duration from the chain as it is; the seed goes back to 1
        so the notes' numbers hold."""
        if self.result is None or self.system is None:
            return
        _, changes, note = TRIALS[self.trial.currentIndex()]
        self._loading = True
        self.shape.setCurrentIndex(self.shape.findData(changes.get("shape", Shape.HALF_SINE)))
        self.peak.setValue(10.0)
        self.duration.setValue(1e3 * duration_for(self.result))
        self.knee.setValue(knee_for(self.result))
        self.mass.setCurrentIndex(self.system.n - 1)
        self.noise.setValue(100.0 * changes.get("noise", 0.0))
        self.offset.setValue(100.0 * changes.get("offset", 0.0))
        self.q.setValue(changes.get("q", DEFAULT_Q))
        self.seed = 1
        self._loading = False
        self._pick_peak = True
        self._show_controls()
        self.trial_note.setText(note)
        if "tab" in changes:
            self.tabs.setCurrentWidget(getattr(self, changes["tab"]))
        self._schedule(rerecord=True)

    def load_setup(self, setup: dict) -> None:
        """Set the page up as a practice problem's `setup` says: its chain (through the Simulation page), then the
        pulse and the spectrum's Q; the seed goes back to 1."""
        t = ShockSetup(**setup)
        system = t.system()
        if system is not None:
            self.system_requested.emit(system)
            if self.system is None or not _same_chain(self.system, system):
                self.set_system(system)  # nobody passed it back: show it here alone
        if self.system is None:
            return
        self._loading = True
        self.shape.setCurrentIndex(self.shape.findData(t.shape))
        self.peak.setValue(t.peak)
        self.duration.setValue(t.duration)
        self.knee.setValue(t.knee)
        self.mass.setCurrentIndex(t.mass % self.system.n)
        self.noise.setValue(t.noise)
        self.offset.setValue(t.offset)
        self.q.setValue(t.q)
        self.seed = 1
        self._loading = False
        self._pick_peak = True
        self._show_controls()
        self.trial_note.setText("")
        self.tabs.setCurrentIndex(t.tab)
        self._schedule(rerecord=True)

    def practice_state(self) -> ShockSetup:
        """The chain and the page's settings, as a setup: what a practice spec checks."""
        s = self.system
        return ShockSetup(
            tuple(float(v) for v in s.masses) if s is not None else (),
            tuple(float(v) for v in s.stiffness) if s is not None else (),
            tuple(float(v) for v in s.damping) if s is not None else (),
            self.shape.currentData(),
            self.peak.value(),
            self.duration.value(),
            self.knee.value(),
            self.mass.currentIndex(),
            self.noise.value(),
            self.offset.value(),
            self.q.value(),
            self.tabs.currentIndex(),
        )

    def pulse(self) -> Pulse:
        return Pulse(
            self.shape.currentData(),
            self.peak.value(),
            1e-3 * self.duration.value(),
            max(self.mass.currentIndex(), 0),
            0.01 * self.noise.value(),
            0.01 * self.offset.value(),
            self.seed,
            self.knee.value(),
        )

    @property
    def zeta(self) -> float:
        return 1.0 / (2.0 * self.q.value())

    def _show_controls(self) -> None:
        shape = self.shape.currentData()
        w: QtWidgets.QWidget
        for w in (self.mass, self.mass_label):
            w.setVisible(shape is Shape.AT_A_MASS)
        for w in (self.noise, self.noise_label, self.offset, self.offset_label):
            w.setVisible(shape.measured)
        for w in (self.duration, self.duration_label):
            w.setVisible(not shape.synthesized)
        for w in (self.knee, self.knee_label):
            w.setVisible(shape.synthesized)
        self.peak_label.setText("Plateau:" if shape.synthesized else "Peak:")
        self.again.setVisible(shape.measured or shape.synthesized)
        self.again.setText("New delays" if shape.synthesized else "New noise")
        self.again.setEnabled(shape.synthesized or self.noise.value() > 0)

    def _on_pulse(self, *_args) -> None:
        if self._loading:
            return
        if self.sender() is self.shape:
            self._pick_peak = True
        self._show_controls()
        self._schedule(rerecord=True)

    def _on_spectrum(self, *_args) -> None:
        if not self._loading:
            self._schedule(rerecord=False)

    def _on_again(self) -> None:
        self.seed += 1
        self._schedule(rerecord=True, now=True)

    def _schedule(self, rerecord: bool, now: bool = False) -> None:
        self._rerecord |= rerecord
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

    def _fill_outputs(self, keep: int) -> None:
        """The masses or springs, for the chain's peaks."""
        n = self.system.n if self.system is not None else self.mass.count()
        self.output.blockSignals(True)
        self.output.clear()
        name = "m" if self.response.currentData() is Response.ACCELERATION else "k"
        self.output.addItems([f"{name}{i + 1}" for i in range(n)])
        self.output.setCurrentIndex(min(max(keep, 0), n - 1))
        self.output.blockSignals(False)

    def _on_response(self) -> None:
        self._fill_outputs(self.output.currentIndex())
        self.lazy.stale(self.chain_tab)

    # the picked oscillator: the line, the spin box and a click on the spectrum
    def _on_line(self) -> None:
        f = 10.0 ** self.pick_line.value()
        if abs(f / self.f_pick.value() - 1.0) > 1e-6:
            self.f_pick.setValue(f)

    def _on_pick(self, f: float) -> None:
        if f > 0 and abs(10.0 ** self.pick_line.value() / f - 1.0) > 1e-6:
            self.pick_line.blockSignals(True)
            self.pick_line.setValue(math.log10(f))
            self.pick_line.blockSignals(False)
        self._draw_oscillator()

    def _on_click(self, event) -> None:
        vb = self.srs_plot.getPlotItem().getViewBox()
        if not vb.sceneBoundingRect().contains(event.scenePos()):
            return
        self.f_pick.setValue(10.0 ** vb.mapSceneToView(event.scenePos()).x())

    # ------------------------------------------------------------- computing
    def recompute(self) -> None:
        """Record the pulse again if asked for (or stale), then its spectrum, and draw."""
        self._recompute.stop()
        self._dirty = False
        if self.system is None or self.result is None:  # set together
            return
        pulse = self.pulse()
        self.freqs = bank(pulse, self.result)
        if self._rerecord or self.record is None:
            try:
                self.record = record(pulse, sample_rate(pulse, self.freqs[-1]), self.system, self.result)
            except ValueError as e:
                self.record = None
                self.readout.setText(f"<b>{e}</b>.")
                return
            self._rerecord = False
            self.chain = None
        self.spectrum = srs(self.record, self.freqs, self.zeta)
        if getattr(self, "_pick_peak", True):
            self._pick_peak = False
            if pulse.shape is Shape.AT_A_MASS:
                f = min(m.fn_hz for m in self.result.modes if m.fn_hz > 0)
            elif pulse.shape.synthesized:
                f = pulse.knee
            else:
                f = float(self.freqs[np.argmax(self.spectrum.maximax)])
            self.f_pick.blockSignals(True)
            self.f_pick.setValue(f)
            self.f_pick.blockSignals(False)
        self.redraw()

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        if self.spectrum is not None:
            self.redraw()

    def redraw(self) -> None:
        self._draw_pulse()
        self._draw_srs()
        self._draw_oscillator()
        self._shapes_key = None
        self.lazy.stale()

    def _draw_pulse(self) -> None:
        assert self.record is not None and self.system is not None, "drawn after recompute"
        rec = self.record
        t = rec.t
        p = self.pulse_plot
        p.clear()
        p.setLabel(
            "left",
            "Base acceleration (g)"
            if rec.pulse.shape is not Shape.AT_A_MASS
            else f"m{rec.pulse.mass % self.system.n + 1}'s acceleration (g)",
        )
        if rec.pulse.shape is Shape.AT_A_MASS:
            k = round(rec.pulse.duration * rec.fs)
            base = np.zeros(t.size)
            base[: k + 1] = rec.pulse.peak * np.sin(np.pi * np.arange(k + 1) / k)
            p.plot(t, base, pen=pg.mkPen(colors.grey, width=1))
        p.plot(
            np.concatenate([[t[0] - rec.h], t, [t[-1] + rec.h]]),
            np.concatenate([[0.0], rec.a / G, [0.0]]),
            pen=pg.mkPen(colors.force, width=2),
        )
        p.addItem(pg.InfiniteLine(0.0, angle=0, pen=pg.mkPen(colors.faint, width=1)))
        v = rec.velocity()
        p = self.velocity_plot
        p.clear()
        p.setLabel("left", "v (m/s)")
        p.plot(t, v, pen=pg.mkPen(colors.mass[0], width=1.5))
        p.addItem(pg.InfiniteLine(0.0, angle=0, pen=pg.mkPen(colors.faint, width=1)))
        self.pulse_plot.setXRange(-0.02 * rec.duration, 1.02 * rec.duration, padding=0.0)
        self.pulse_info.setText(
            f"Δv = {_minus(f'{rec.velocity_change:.4g}')} m/s over the record ({_time(rec.duration)}); "
            f"{rec.a.size:,} samples at {rec.fs:.4g} Hz; {self.freqs.size} oscillators, {self.freqs[0]:.3g} to "
            f"{self.freqs[-1]:.4g} Hz."
        )

    def _draw_srs(self) -> None:
        assert self.result is not None and self.spectrum is not None, "drawn after recompute"
        sp, rec = self.spectrum, self.record
        if sp is None or rec is None:
            return
        f = sp.f
        p = self.srs_plot
        for item in list(p.getPlotItem().items):
            if item is not self.pick_line:
                p.removeItem(item)
        self.srs_legend.clear()
        floor = 1e-6 * float(np.max(sp.maximax))

        def curve(values, color, width=1.5, style=QtCore.Qt.PenStyle.SolidLine, name=None):
            item = p.plot(f, np.maximum(values / G, floor / G), pen=pg.mkPen(color, width=width, style=style))
            if name:
                self.srs_legend.addItem(item, name)

        dash, dot = QtCore.Qt.PenStyle.DashLine, QtCore.Qt.PenStyle.DotLine
        if self.split_time.isChecked() and self.split_sign.isChecked():
            curve(sp.primary[0], colors.mass[0], name="Primary +")
            curve(sp.primary[1], colors.mass[0], style=dash, name="Primary −")
            curve(sp.residual[0], colors.mass[1], name="Residual +")
            curve(sp.residual[1], colors.mass[1], style=dash, name="Residual −")
        elif self.split_time.isChecked():
            curve(sp.primary.max(axis=0), colors.mass[0], name="Primary")
            curve(sp.residual.max(axis=0), colors.mass[1], name="Residual")
        elif self.split_sign.isChecked():
            curve(sp.positive, colors.mass[0], name="Positive")
            curve(sp.negative, colors.mass[1], style=dash, name="Negative")
        curve(sp.maximax, colors.strong, width=2.5, name=f"Maximax, Q = {sp.q:.3g}")
        synthesis = rec.synthesis
        if synthesis is not None:
            lo, hi = target_band(rec.pulse)
            ft = np.geomspace(lo, hi, 200)
            goal = target(rec.pulse, ft) / G
            item = p.plot(ft, goal, pen=pg.mkPen(colors.force, width=2, style=dot))
            self.srs_legend.addItem(item, f"Target, Q = {SYNTH_Q:.3g}, ±{TOLERANCE_DB:.3g} dB")
            band = pg.mkPen(colors.force, width=1)
            for sign in (-1.0, 1.0):
                p.plot(ft, goal * 10.0 ** (sign * TOLERANCE_DB / 20.0), pen=band)
        peak = float(np.max(np.abs(rec.a)))
        line = pg.mkPen(colors.muted, width=1, style=dash)
        p.plot(f[[0, -1]], np.full(2, peak / G), pen=line)
        dv = abs(rec.velocity_change)
        if dv > 0:
            low = TWO_PI * f * dv
            keep = low < 3.0 * float(np.max(sp.maximax))
            if keep.any():
                p.plot(f[keep], low[keep] / G, pen=line)
        for i, m in enumerate(self.result.modes):
            if m.fn_hz > 0:
                p.addItem(
                    pg.InfiniteLine(
                        math.log10(m.fn_hz),
                        angle=90,
                        pen=pg.mkPen(colors.mode[i % len(colors.mode)], width=1, style=dot),
                    )
                )
        self.pick_line.setPen(pg.mkPen(colors.force, width=2))
        self.pick_line.blockSignals(True)
        self.pick_line.setValue(math.log10(self.f_pick.value()))
        self.pick_line.blockSignals(False)
        top = float(np.max(sp.maximax)) / G
        p.setXRange(math.log10(f[0]), math.log10(f[-1]), padding=0.0)
        bottom = min(max(float(np.min(sp.maximax)) / G / 1.5, top * 1e-5), top * 1e-3)
        p.setYRange(math.log10(bottom), math.log10(top) + 0.3, padding=0.0)

        k = int(np.argmax(sp.maximax))
        modes = ", ".join(f"{m.fn_hz:.3g} Hz {sp.at(m.fn_hz) / G:.3g} g" for m in self.result.modes if m.fn_hz > 0)
        text = (
            f"<b>Maximax</b> {sp.maximax[k] / G:.3g} g at {f[k]:.3g} Hz ({self._scaled(f[k], 2)}), "
            f"{sp.maximax[k] / peak:.3g} × the record's peak {peak / G:.3g} g. At the chain's modes: {modes}."
        )
        if synthesis is not None:
            kind = "damped sines" if rec.pulse.shape is Shape.SINES else "wavelets"
            text += (
                f" <b>Synthesis</b>: {synthesis.f.size} {kind}, within "
                f"{synthesis.worst:.2g} dB of the target at their frequencies after {synthesis.iterations} "
                f"rounds, {_worst(rec, self.spectrum) if abs(sp.q - SYNTH_Q) < 1e-9 else 'n/a'} dB over the "
                f"bank; it ends at {_minus(f'{rec.velocity_change:.3g}')} m/s, "
                f"{_length(abs(rec.displacement()[-1]))} "
                f"from where it started."
            )
        self.readout.setText(text)

    def _scaled(self, f: float, digits: int = 3) -> str:
        """The frequency on the pulse's own scale: f T, or a synthesis's f / knee."""
        assert self.record is not None, "drawn after recompute"
        pulse = self.record.pulse
        if pulse.shape.synthesized:
            return f"f / knee = {f / pulse.knee:.{digits}g}"
        return f"f T = {f * pulse.duration:.{digits}g}"

    def _draw_oscillator(self) -> None:
        rec = self.record
        if rec is None or self.spectrum is None:
            return
        f = self.f_pick.value()
        tail = max(TAIL_PERIODS / f, 0.25 * rec.duration)
        osc = oscillator(rec, f, self.zeta, min(tail, 4.0 * rec.duration + TAIL_PERIODS / f))
        p = self.osc_plot
        p.clear()
        self.osc_legend.clear()
        a = np.zeros(osc.t.size)
        a[: rec.a.size] = rec.a
        base = p.plot(osc.t, a / G, pen=pg.mkPen(colors.grey, width=1))
        y = p.plot(osc.t, osc.acceleration / G, pen=pg.mkPen(colors.mass[0], width=2))
        self.osc_legend.addItem(base, "The pulse")
        self.osc_legend.addItem(y, f"The oscillator, {f:.3g} Hz")
        k = int(np.argmax(np.abs(osc.acceleration)))
        p.plot(
            [osc.t[k]],
            [osc.acceleration[k] / G],
            pen=None,
            symbol="o",
            symbolSize=8,
            symbolBrush=colors.force,
            symbolPen=None,
        )
        end = pg.mkPen(colors.faint, width=1, style=QtCore.Qt.PenStyle.DashLine)
        p.addItem(pg.InfiniteLine(osc.end, angle=90, pen=end))
        p.addItem(pg.InfiniteLine(0.0, angle=0, pen=pg.mkPen(colors.faint, width=1)))
        p.setXRange(0.0, osc.t[-1], padding=0.01)
        d = self.disp_plot
        d.clear()
        d.plot(osc.t, osc.displacement, pen=pg.mkPen(colors.mass[1], width=1.5))
        d.addItem(pg.InfiniteLine(osc.end, angle=90, pen=end))
        d.addItem(pg.InfiniteLine(0.0, angle=0, pen=pg.mkPen(colors.faint, width=1)))

        during = osc.t <= osc.end
        y = osc.acceleration / G
        when = "record" if rec.end is None else "pulse"
        self.osc_note.setText(
            f"f = {f:.4g} Hz, {self._scaled(f)}, Q = {self.q.value():.3g}. During the {when}: "
            f"+{_positive(y[during].max()):.3g} / −{_positive(-y[during].min()):.3g} g; after it (residual): "
            f"+{_positive(y[~during].max()):.3g} / −{_positive(-y[~during].min()):.3g} g. The dot marks the largest, "
            f"{abs(y[k]):.3g} g. Its spring stretches up to {_length(np.max(np.abs(osc.displacement)))}."
        )

    # ------------------------------------------------------------- the chain, the tripartite plot, the pulses
    def _draw_chain(self) -> None:
        assert self.system is not None and self.result is not None, "drawn after recompute"
        if self.record is None or self.spectrum is None:
            return
        if self.chain is None:
            try:
                self.chain = chain_peaks(self.system, self.record, self.result)
            except ValueError as e:
                self.chain = str(e)
        for p in (self.peaks_plot, self.history_plot):
            p.clear()
        self.peaks_legend.clear()
        self.history_legend.clear()
        cp = self.chain
        if isinstance(cp, str):
            self.chain_table.setRowCount(0)
            self.chain_note.setText(f"<b>{cp}</b>.")
            return
        response = self.response.currentData()
        acc = response is Response.ACCELERATION
        scale, unit = (1.0 / G, "g") if acc else (1.0, "N")
        exact = cp.peak(response)
        rules = {rule: cp.estimate(response, rule) for rule in Rule}
        n = exact.size
        x = np.arange(1, n + 1)
        p = self.peaks_plot
        p.setLabel("left", f"Peak ({unit})")
        p.setLabel("bottom", "Mass" if acc else "Spring")
        bars = pg.BarGraphItem(
            x=x, height=exact * scale, width=0.6, brush=pg.mkBrush(colors.grey), pen=pg.mkPen(colors.grey)
        )
        p.addItem(bars)
        self.peaks_legend.addItem(bars, "Exact")
        for i, (rule, values) in enumerate(rules.items()):
            item = p.plot(
                x,
                values * scale,
                pen=None,
                symbol=RULE_SYMBOLS[rule],
                symbolSize=11,
                symbolBrush=colors.mode[i % len(colors.mode)],
                symbolPen=None,
            )
            self.peaks_legend.addItem(item, rule.value)
        p.getAxis("bottom").setTicks([[(int(i), str(i)) for i in x]])
        p.setXRange(0.4, n + 0.6, padding=0.0)
        top = max(float(np.max(exact)), max(float(np.max(v)) for v in rules.values())) * scale
        p.setYRange(0.0, 1.3 * top, padding=0.0)

        j = min(max(self.output.currentIndex(), 0), n - 1)
        h = self.history_plot
        h.setLabel("left", f"{'m' if acc else 'k'}{j + 1} ({unit})")
        parts = cp.contributions(response, j)
        terms = cp.terms(response)[j]
        for r in range(parts.shape[0]):
            color = colors.mode[r % len(colors.mode)]
            item = h.plot(cp.t, parts[r] * scale, pen=pg.mkPen(color, width=1))
            self.history_legend.addItem(item, f"Mode {r + 1}")
            dashes = pg.mkPen(color, width=1, style=QtCore.Qt.PenStyle.DashLine)
            h.addItem(pg.InfiniteLine(abs(terms[r]) * scale, angle=0, pen=dashes))
        y = cp.exact[response][:, j] * scale
        item = h.plot(cp.t, y, pen=pg.mkPen(colors.strong, width=2))
        self.history_legend.addItem(item, "Exact")
        k = int(np.argmax(np.abs(y)))
        h.plot([cp.t[k]], [y[k]], pen=None, symbol="o", symbolSize=8, symbolBrush=colors.force, symbolPen=None)
        reach = max(float(np.max(np.abs(y))), float(np.max(np.abs(terms), initial=0.0)) * scale) or 1.0
        h.setYRange(-1.1 * reach, 1.45 * reach, padding=0.0)
        h.addItem(pg.InfiniteLine(0.0, angle=0, pen=pg.mkPen(colors.faint, width=1)))
        for item in h.getPlotItem().listDataItems():
            item.setDownsampling(auto=True, method="peak")
            item.setClipToView(True)
        h.setXRange(
            0.0, min(cp.t[-1], self.record.duration + HISTORY_PERIODS / float(cp.f.min(initial=math.inf))), padding=0.0
        )

        t = self.chain_table
        t.setRowCount(n)
        t.setVerticalHeaderLabels([f"{'m' if acc else 'k'}{i + 1}" for i in range(n)])
        for i in range(n):
            t.setItem(i, 0, _cell(f"{exact[i] * scale:.3g}"))
            for c, values in enumerate(rules.values(), start=1):
                err = 100.0 * (values[i] / exact[i] - 1.0) if exact[i] > 0 else math.nan
                t.setItem(i, c, _cell(_minus(f"{values[i] * scale:.3g} ({err:+.0f} %)")))
        modes = "; ".join(
            f"{r + 1}: {cp.f[r]:.3g} Hz, ζ {cp.zeta[r]:.2g}, Γ {_minus(f'{cp.gamma[r]:.3g}')}, "
            f"{100.0 * cp.effective[r]:.0f} % of the "
            f"mass, SRS {cp.spectrum[Response.ACCELERATION][r] / G:.3g} g"
            for r in range(cp.f.size)
        )
        self.chain_note.setText(
            f"In {unit}, the error from the exact in brackets. The SRS at each mode's own f and ζ (not the page's Q). "
            f"<b>Modes</b>: {modes}."
            + (
                ""
                if self.result.is_proportional
                else " The damping isn't proportional, so the modes only approximate the chain."
            )
        )

    def _draw_tripartite(self) -> None:
        assert self.record is not None, "drawn after recompute"
        if self.spectrum is None:
            return
        sp, rec = self.spectrum, self.record
        p = self.tri_plot
        p.clear()
        f = sp.f
        pv = sp.pseudo_velocity
        good = pv[pv > 0]
        if good.size == 0:
            return
        top = float(np.max(pv))
        bottom = max(float(np.min(good)), top * 1e-4)
        lo, hi = math.log10(f[0]), math.log10(f[-1])
        y0, y1 = math.log10(bottom) - 0.2, math.log10(top) + 0.4
        grid = pg.mkPen(colors.faint, width=1)
        ends = np.array([f[0], f[-1]])
        # lines of constant displacement (pv = ω d) and pseudo-acceleration (pv = A / ω), a decade apart
        w = TWO_PI * ends
        for d in 10.0 ** np.arange(math.floor(y0 - math.log10(w[1])), math.ceil(y1 - math.log10(w[0])) + 1):
            p.plot(ends, w * d, pen=grid)
            self._grid_label(p, d * w, ends, _length(d), y0, y1, rising=True)
        for a in 10.0 ** np.arange(math.floor(y0 + math.log10(w[0] / G)), math.ceil(y1 + math.log10(w[1] / G)) + 1):
            p.plot(ends, a * G / w, pen=grid)
            self._grid_label(p, a * G / w, ends, f"{a:g} g", y0, y1, rising=False)
        dash = pg.mkPen(colors.muted, width=1, style=QtCore.Qt.PenStyle.DashLine)
        dv = abs(rec.velocity_change)
        if dv > 0:
            p.plot(ends, [dv, dv], pen=dash)
        peak = float(np.max(np.abs(rec.a)))
        p.plot(ends, peak / w, pen=dash)
        if rec.synthesis is not None:
            band = np.geomspace(*target_band(rec.pulse), 200)
            p.plot(
                band,
                target(rec.pulse, band) / (TWO_PI * band),
                pen=pg.mkPen(colors.force, width=2, style=QtCore.Qt.PenStyle.DotLine),
            )
        p.plot(f, np.maximum(pv, bottom * 1e-3), pen=pg.mkPen(colors.strong, width=2.5))
        p.setXRange(lo, hi, padding=0.0)
        p.setYRange(y0, y1, padding=0.0)
        d = sp.displacement
        k = int(np.argmax(d))
        if abs(pv[0] / dv - 1.0) < 0.2 if dv > 0 else False:
            left = (
                f"At the left the pseudo-velocity tends to the base's Δv, {dv:.3g} m/s (dashed; {pv[0]:.3g} m/s "
                f"at {f[0]:.3g} Hz): a soft oscillator stays put while the base moves off, then rings at about "
                f"that speed."
            )
        else:
            left = (
                f"At the left it is {pv[0]:.3g} m/s at {f[0]:.3g} Hz, not the base's Δv, "
                f"{_minus(f'{dv:.3g}')} m/s: an oscillator that slow isn't soft next to a "
                f"{_time(rec.duration)} record, which moves the base and brings it back while it swings."
            )
        self.tri_note.setText(
            f"Q = {sp.q:.3g}. {left} At the right it falls along the line of the peak acceleration, "
            f"{peak / G:.3g} g (dashed): a stiff oscillator follows the base. The relative displacement is largest, "
            f"{_length(float(d[k]))}, at {f[k]:.3g} Hz: the room a mount that soft needs."
        )

    @staticmethod
    def _grid_label(
        p: pg.PlotWidget, pv: np.ndarray, ends: np.ndarray, text: str, y0: float, y1: float, rising: bool
    ) -> None:
        """Label a diagonal where it leaves the plot: at the top for a rising line, at the bottom for a falling one."""
        x0, x1 = np.log10(ends)
        v0, v1 = np.log10(pv)
        y = y1 - 0.05 if rising else y0 + 0.05
        if not min(v0, v1) <= y <= max(v0, v1):
            return
        x = x0 + (y - v0) / (v1 - v0) * (x1 - x0)
        margin = 0.04 * (x1 - x0)
        if not x0 + margin <= x <= x1 - margin:
            return
        label = pg.TextItem(text, color=colors.muted, anchor=(0.5, 0.0 if rising else 1.0))
        label.setPos(x, y)
        p.addItem(label)

    def _draw_shapes(self) -> None:
        if self.spectrum is None:
            return
        q, same = self.q.value(), self.same_dv.isChecked()
        if self._shapes_key == (q, same, colors.strong):
            return
        self._shapes_key = (q, same, colors.strong)
        spectra = shape_spectra(1.0 / (2.0 * q), same)
        s = np.linspace(0.0, 1.0, 201)
        sp, cp = self.shapes_plot, self.compare_plot
        sp.clear()
        cp.clear()
        self.compare_legend.clear()
        for i, (shape, (scale, f, values)) in enumerate(spectra.items()):
            pen = pg.mkPen(colors.mass[i % len(colors.mass)], width=1.5)
            sp.plot(
                np.concatenate([[-0.05, 0.0], s, [1.0, 1.05]]),
                np.concatenate([[0.0, 0.0], scale * classical(shape, s), [0.0, 0.0]]),
                pen=pen,
            )
            item = cp.plot(f, values, pen=pg.mkPen(colors.mass[i % len(colors.mass)], width=2))
            self.compare_legend.addItem(item, shape.value)
        sp.addItem(pg.InfiniteLine(0.0, angle=0, pen=pg.mkPen(colors.faint, width=1)))
        sp.setXRange(-0.05, 1.05, padding=0.0)
        f = next(iter(spectra.values()))[1]
        cp.setXRange(math.log10(f[0]), math.log10(f[-1]), padding=0.0)
        top = max(float(np.max(v)) for _, _, v in spectra.values())
        cp.setYRange(math.log10(top) - 2.5, math.log10(top) + 0.15, padding=0.0)
        rows = []
        for shape, (_scale, f, values) in spectra.items():
            k = int(np.argmax(values))
            rows.append(f"{shape.value} {values[k]:.3g} (f T {f[k]:.2g}), {values[0]:.2g}, {values[-1]:.3g}")
        basis = "the same Δv as the half-sine (their peaks then differ)" if same else "the same peak A"
        self.shapes_note.setText(
            f"Q = {q:.3g}, {basis}; over A, the half-sine's peak. Largest, at f T = "
            f"{f[0]:.2g}, at the top: " + "; ".join(rows) + "."
        )


@functools.lru_cache(maxsize=8)
def shape_spectra(zeta: float, same_dv: bool) -> dict[Shape, tuple[float, np.ndarray, np.ndarray]]:
    """The classical pulses at T = 1 s, each at peak 1 (or scaled to the half-sine's Δv): (its peak, the f T of the
    bank, the SRS over the half-sine's peak)."""
    out = {}
    dv0 = None
    for shape in CLASSICAL:
        pulse = Pulse(shape, 1.0 / G, 1.0)
        f = bank(pulse)
        rec = record(pulse, sample_rate(pulse, f[-1]))
        dv = rec.velocity_change
        dv0 = dv if dv0 is None else dv0
        scale = dv0 / dv if same_dv else 1.0
        out[shape] = (scale, f, scale * srs(rec, f, zeta).maximax)
    return out


def _cell(text: str) -> QtWidgets.QTableWidgetItem:
    item = QtWidgets.QTableWidgetItem(text)
    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
    return item


def _worst(rec: Record, sp: Spectrum) -> str:
    """The SRS's largest distance from a synthesis's target over the bank, in dB."""
    goal = target(rec.pulse, sp.f)
    ok = np.isfinite(goal)
    return f"{np.max(np.abs(20.0 * np.log10(sp.maximax[ok] / goal[ok]))):.2g}"


def _minus(text: str) -> str:
    return text.replace("-", "−")


def _time(x: float) -> str:
    return f"{x:.4g} s" if x >= 1.0 else f"{1e3 * x:.4g} ms"


def _length(x: float) -> str:
    if x >= 1.0:
        return f"{x:.3g} m"
    return f"{1e3 * x:.3g} mm" if x >= 1e-3 else f"{1e6 * x:.3g} µm"


def _positive(x: float) -> float:
    return max(float(x), 0.0) + 0.0  # no "−-0"
