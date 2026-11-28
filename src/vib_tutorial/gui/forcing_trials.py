"""Forcing an oscillator's Things to try, practice problems, tips and notes, without Qt (#478 item 7).

The page merges Transient response and Periodic forcing (#478 Phase 4, decided 2026-11-28). Its state is
`view_models/forcing.py`'s `ForcingTrial`; `gui/forcing.py` binds the page's controls to it and runs it live. A
trial is a setup to go back to: it sets the oscillator and the force, and from rest fires the one-off force
(`start=APPLY`) or switches the wave on (`start=WAVE`). Its prediction is asked only once the reader has handled
the system (`lesson.TrialBox(ask_after_play=True)`), about a case they haven't tried. The problems are the two old
pages' eighteen, their setups in the merged state's fields, each starting what it sets up.

Working on it: `tests/test_tutorial_numbers.py` checks the trials' numbers, `tests/test_practice.py` the problems,
`tests/test_qt_free.py` that this loads no Qt.
"""

from __future__ import annotations

import math

import numpy as np

from ..core.periodic import Wave, periodic_steady
from ..core.practice import Placement, Prediction, Problem, Question, Requirement, at_most, kept, number, pick, rounded
from ..core.sdof import TWO_PI, Oscillator, receptance
from ..core.transient import PulseKind, peak_ratio, peak_ratio_undamped, step_metrics
from ..view_models.forcing import APPLY, TAB_CONV, TAB_HARM, TAB_PULSE, WAVE, ForcingTrial, forcing, wave_hz
from ..view_models.text import number as sig
from . import forcing_figures
from .links import load_notes, page_link
from .trials_text import trials_html

TRIALS = (
    ForcingTrial(
        "A blow",
        "A blow of 0.1 N·s on the 1 kg oscillator (<m>f<sub>n</sub></m> = 3.18 Hz, <m>ζ = 0.05</m>) starts it at "
        "0.1 m/s from where it is: the free vibration <m>h(t)</m> times 0.1. Its first swing, 4.6 mm, is a little "
        "under <m>I/mω<sub>n</sub></m> = 5 mm: the damping takes some in the first quarter period. Hit it again as it "
        "rings: the second blow's ring adds to the first's.",
        kind=PulseKind.IMPULSE,
        amplitude=0.1,
        start=APPLY,
        predict=Prediction(
            "A blow of 0.1 N·s on the 1 kg oscillator: I/mωₙ is 5 mm. How far does its first swing go?",
            ("Exactly 5 mm", "A little under 5 mm", "About 10 mm"),
            1,
            why=(
                "That's undamped. With ζ = 0.05 the damper takes some of the energy in the first quarter period: 4.6 mm.",
                "",
                "Twice static is what a step does. A blow just starts the mass at I/m, and I/mω<sub>n</sub> is as far as that speed carries it.",
            ),
        ),
    ),
    ForcingTrial(
        "A step",
        "1 N pushed on and held. The mass overshoots its static 2.5 mm by "
        "<m>M<sub>p</sub> = e<sup>−ζπ/√(1 − ζ²)</sup></m> = 85 %, to 4.64 mm, at <m>t<sub>p</sub> = π/ω<sub>d</sub></m> "
        "= 0.157 s, and settles within 2 % after 3.8 s (the rule <m>4/ζω<sub>n</sub></m> says 4 s). Let go and it "
        "rings back down about zero: the same step, the other way.",
        kind=PulseKind.STEP,
        start=APPLY,
        predict=Placement(
            "1 N pushed on and held, with a static deflection of 2.5 mm. How far does the mass get? Put the line "
            "at its farthest.",
            plot="x_plot",
            answer=0.00464,
            span=(0.0, 0.008),
            low="It reaches 2.5 mm with speed to spare and overshoots: light damping lets it go 85 % past, to 4.64 mm.",
            high="Twice static, 5 mm, is the undamped limit. With ζ = 0.05 it stops short, at 4.64 mm.",
            tolerance=0.06,
            scale=1000.0,
            unit="mm",
        ),
    ),
    ForcingTrial(
        "Watch the convolution build",
        "A 0.2 s rectangular pulse of 1 N. The Convolution tab is the integral at the instant shown: the force, "
        "the impulse response flipped and slid to that instant, and their product, whose signed area is the "
        "response there. Pause and drag the time line back: the area follows the response.",
        kind=PulseKind.RECTANGLE,
        duration=0.2,
        start=APPLY,
    ),
    ForcingTrial(
        "A pulse half a period long",
        "A 1 N rectangle 0.16 s long, half the period. Undamped, a rectangle of half a period or more throws "
        "the mass to twice its static deflection, as a step would: the most a rectangle can do. Its "
        "pulse-spectrum curve rises to 2 at <m>T = T<sub>n</sub>/2</m> and stays there; with <m>ζ</m> = 0.05 this one "
        "lands at 1.85.",
        kind=PulseKind.RECTANGLE,
        duration=0.157,
        tab=TAB_PULSE,
        start=APPLY,
        predict=Prediction(
            "A 1 N rectangle half a period long (static: 2.5 mm). How far does the mass swing?",
            ("About 2.5 mm", "About twice that", "About half that"),
            1,
            why=(
                "That's the force applied slowly. Suddenly, the mass overshoots as after a step, and the pulse ends with it at its peak.",
                "",
                "A pulse this long is no impulse: the mass reaches a step's full overshoot, twice static, as the force ends.",
            ),
        ),
    ),
    ForcingTrial(
        "A half-sine's worst length",
        "A 1 N half-sine 0.26 s long, about 0.8 of the period: undamped, its peak would be 1.77 times "
        "static, the half-sine curve's top; with <m>ζ = 0.05</m>, 1.65. Shorter, it is an impulse; much longer, "
        "the mass follows it statically.",
        kind=PulseKind.HALF_SINE,
        duration=0.26,
        tab=TAB_PULSE,
        start=APPLY,
        predict=Prediction(
            "A 1 N half-sine about 0.8 of a period long, the worst length. Is its peak above twice the static 2.5 mm?",
            ("Yes, above twice", "No: between 1.5 and 2 times", "No: under static"),
            1,
            why=(
                "Twice is the most even a rectangle, with its sudden start, can do. A half-sine starts gently: 1.77 times static undamped, 1.65 here.",
                "",
                "At 0.8 of a period it's near its worst. Under static is for pulses far shorter than the period.",
            ),
        ),
    ),
    ForcingTrial(
        "A slow ramp",
        "1 N reached over 0.87 s, 2.75 periods, and held. The mass follows the ramp, lagging, and overshoots the end "
        "by only <m>|sin(πT/T<sub>n</sub>)|/(πT/T<sub>n</sub>)</m> = 8 % of the static 2.5 mm (undamped; 6 % with "
        "<m>ζ = 0.05</m>). A ramp of a whole number of periods leaves no swing at all.",
        kind=PulseKind.RAMP,
        duration=0.864,
        tab=TAB_PULSE,
        start=APPLY,
        predict=Prediction(
            "1 N reached over 2.75 periods. How far past the static 2.5 mm does the mass overshoot?",
            ("Twice it, as a step does", "A few percent", "About half of it"),
            1,
            why=(
                "A step jumps. Over 2.75 periods the mass has time to follow, which leaves only |sin πr|/πr = 8 %.",
                "",
                "|sin πr|/πr at r = 2.75 is 0.08. Half would take a ramp only about 0.6 of a period long.",
            ),
        ),
    ),
    ForcingTrial(
        "A square wave at a third",
        "A ±1 N square wave at <m>f<sub>n</sub>/3</m> = 1.06 Hz. Its fundamental is far below resonance, but its third "
        "harmonic, <m>4/3π</m> = 0.42 N, sits on <m>f<sub>n</sub></m>, where <m>|H| = 1/cω<sub>n</sub></m> = "
        "25 mm/N: it alone swings the mass 10.6 mm, three times the fundamental's 3.6 mm. The response rings at "
        "three times the drive: slow machines excite fast modes this way.",
        tune=3,
        harmonics=7,
        tab=TAB_HARM,
        start=WAVE,
        predict=Prediction(
            "The square wave at a third of the natural frequency. Which harmonic swings the mass most?",
            ("The fundamental, the largest", "The third, on resonance", "Neither: it's far below resonance"),
            1,
            why=(
                "It's the largest force, 1.27 N, but far below resonance: 3.6 mm. The third's 0.42 N, on resonance, is amplified ten times.",
                "",
                "The fundamental is, but the square wave's third harmonic, at 3 × f<sub>n</sub>/3, sits right on f<sub>n</sub>.",
            ),
        ),
    ),
    ForcingTrial(
        "Hammer blows",
        "A 0.1 N·s blow every second. An impulse train has every harmonic at the same height, <m>2I/T<sub>0</sub></m> "
        "= 0.2 N, so the response's lines are <m>H</m> itself, sampled at 1 Hz steps. In time, each blow sets the mass "
        "ringing and the next arrives before it has died away (<m>1/ζω<sub>n</sub></m> = 1 s): the steady state "
        "swings from −4.5 to 5.2 mm about the mean <m>I/kT<sub>0</sub></m> = 0.25 mm.",
        wave=Wave.IMPULSES,
        wave_amplitude=0.1,
        harmonics=20,
        tab=TAB_HARM,
        start=WAVE,
        predict=Prediction(
            "A blow every second. How do the impulse train's harmonics fall with their number j?",
            ("As 1/j", "As 1/j²", "They don't: all the same height"),
            2,
            why=(
                "That's a square wave's, with its jumps. A blow is sharper than any jump: its harmonics don't fall at all.",
                "That's a triangle's, with its kinks. A train of blows has every harmonic at 2I/T.",
                "",
            ),
        ),
    ),
)

# ------------------------------------------------------------------ practice problems
# Each question's numbers are printed round, and its answer is worked from them as printed.


def _impulse_peak(rng: np.random.Generator) -> Question:
    m = pick(rng, (0.5, 1, 2, 5))
    k = pick(rng, (200, 400, 1000, 2500))
    impulse = pick(rng, (0.05, 0.1, 0.2, 0.5))
    wn = math.sqrt(k / m)
    x = impulse / (m * wn)
    return Question(
        f"A {sig(m, 6)} kg mass on a spring of {sig(k, 6)} N/m, undamped and at rest, is struck a blow of {sig(impulse, 6)} N·s. "
        "How far does it swing?",
        1e3 * x,
        f"The blow leaves it at rest where it was with velocity v = I/m = {number(impulse / m)} m/s, and it "
        f"vibrates freely from there: x = (I/mω<sub>n</sub>) sin ω<sub>n</sub>t, with ω<sub>n</sub> = "
        f"√({sig(k, 6)}/{sig(m, 6)}) = {number(wn)} rad/s. It swings ±{number(1e3 * x)} mm.",
        unit="mm",
        setup=dict(start=APPLY, kind=PulseKind.IMPULSE, amplitude=impulse, mass=m, stiffness=k, damping=0.0),
    )


def _step_peak(rng: np.random.Generator) -> Question:
    m = pick(rng, (1, 2, 5))
    k = pick(rng, (400, 1000, 2500))
    c = rounded(2.0 * pick(rng, (0.02, 0.05, 0.1, 0.2, 0.3)) * math.sqrt(k * m))
    force = pick(rng, (1, 2, 5, 10))
    osc = Oscillator(m, k, c)
    over = step_metrics(osc).overshoot
    peak = force / k * (1.0 + over)
    return Question(
        f"m = {sig(m, 6)} kg, k = {sig(k, 6)} N/m and c = {number(c)} N·s/m, at rest. A force of {sig(force, 6)} N is switched on "
        "and left on. How far does the mass get at most?",
        1e3 * peak,
        f"ζ = c/2√(km) = {number(osc.zeta)}. It overshoots the static {number(1e3 * force / k)} mm by "
        f"M<sub>p</sub> = e<sup>−ζπ/√(1 − ζ²)</sup> = {number(over)}: (F/k)(1 + M<sub>p</sub>) = "
        f"{number(1e3 * peak)} mm.",
        unit="mm",
        setup=dict(start=APPLY, kind=PulseKind.STEP, amplitude=force, mass=m, stiffness=k, damping=c, tab=TAB_CONV),
    )


def _peak_time(rng: np.random.Generator) -> Question:
    m = pick(rng, (1, 2, 5))
    k = pick(rng, (400, 1000, 2500))
    c = rounded(2.0 * pick(rng, (0.05, 0.1, 0.2, 0.3, 0.5)) * math.sqrt(k * m))
    osc = Oscillator(m, k, c)
    tp = math.pi / osc.omega_d
    return Question(
        f"m = {sig(m, 6)} kg, k = {sig(k, 6)} N/m and c = {number(c)} N·s/m. A force is switched on and left on. How long "
        "after does the mass reach its farthest?",
        tp,
        f"The step response peaks after half a damped period, π/ω<sub>d</sub>: ζ = {number(osc.zeta)}, "
        f"ω<sub>n</sub> = {number(osc.omega_n)} rad/s, ω<sub>d</sub> = ω<sub>n</sub>√(1 − ζ²) = "
        f"{number(osc.omega_d)} rad/s, so t<sub>p</sub> = {number(tp)} s.",
        unit="s",
        setup=dict(start=APPLY, kind=PulseKind.STEP, mass=m, stiffness=k, damping=c, tab=TAB_CONV),
    )


def _rectangle_peak(rng: np.random.Generator) -> Question:
    k = pick(rng, (100, 400, 1000))
    tn = TWO_PI / math.sqrt(k)
    length = rounded(pick(rng, (0.05, 0.1, 0.2, 0.3, 0.6, 1.0, 1.5)) * tn)
    force = pick(rng, (1, 5, 10, 20))
    r = length / tn
    ratio = peak_ratio_undamped(PulseKind.RECTANGLE, r)
    rule = (
        f"Shorter than half a period, it is nearly an impulse: 2 sin(πT/T<sub>n</sub>) = {number(ratio)} times static"
        if r < 0.5
        else "Half a period or longer, it throws the mass to twice the static deflection, as a step would"
    )
    return Question(
        f"A 1 kg mass on a spring of {sig(k, 6)} N/m, undamped and at rest, is pushed with {sig(force, 6)} N for {number(length)} "
        "s, then let go. How far does it swing at most?",
        1e3 * force / k * ratio,
        f"T<sub>n</sub> = 2π/√{sig(k, 6)} = {number(tn)} s, so the pulse lasts r = {number(r)} periods. {rule}: "
        f"{number(ratio)} × {number(1e3 * force / k)} mm = {number(1e3 * force / k * ratio)} mm.",
        unit="mm",
        setup=dict(start=APPLY, kind=PulseKind.RECTANGLE, amplitude=force, duration=length, stiffness=k, damping=0.0),
    )


def _ramp_overshoot(rng: np.random.Generator) -> Question:
    k = pick(rng, (100, 400, 1000))
    tn = TWO_PI / math.sqrt(k)
    rise = rounded(pick(rng, (0.5, 0.75, 1.5, 2.5, 3.25)) * tn)
    r = rise / tn
    over = abs(math.sin(math.pi * r)) / (math.pi * r)
    return Question(
        f"A 1 kg mass on a spring of {sig(k, 6)} N/m, undamped, has a load ramped onto it in a straight line over "
        f"{number(rise)} s, then held. By what percentage of the static deflection does it overshoot?",
        100.0 * over,
        f"T<sub>n</sub> = 2π/√{sig(k, 6)} = {number(tn)} s, so the ramp takes r = {number(r)} periods, and it "
        f"overshoots by |sin πr|/πr = {number(over)}: {number(100 * over)} %.",
        unit="%",
        tolerance=0.03,
        setup=dict(start=APPLY, kind=PulseKind.RAMP, duration=rise, stiffness=k, damping=0.0),
    )


def _sudden_load(rng: np.random.Generator) -> Question:
    return Question(
        "A weight is let go onto an undamped spring all at once, instead of being lowered onto it gently. How "
        "far does the spring go at most, against how far it sags under the weight lowered gently?",
        0,
        "Twice as far. A force switched on suddenly overshoots its static deflection by e<sup>−ζπ/√(1 − ζ²)</sup>, "
        "all of it with no damping: x = (F/k)(1 − cos ω<sub>n</sub>t) swings between 0 and 2F/k.",
        choices=("Twice as far", "As far", "√2 times as far", "Half as far again"),
        why=(
            "",
            "That's lowering it gently, the mass never moving fast. Let go at once, it reaches the static deflection "
            "moving, and carries on.",
            "No √2 here: with no damping the mass swings as far past the static deflection as it started above it.",
            "That would need damping to take some of the energy out; with none, the overshoot is the whole static "
            "deflection.",
        ),
        setup=dict(start=APPLY, kind=PulseKind.STEP, damping=0.0, tab=TAB_CONV),
        predict=True,
    )


def _short_pulse(rng: np.random.Generator) -> Question:
    n = int(pick(rng, (2, 3)))
    return Question(
        f"A rectangular push lasting a twentieth of the natural period is made {n} times as long, with the same "
        "force. What happens to the largest swing it leaves?",
        0,
        f"A pulse much shorter than the period is an impulse: only its area I = FT counts, and the swing grows "
        f"with it, 2 sin(πT/T<sub>n</sub>) ≈ 2πT/T<sub>n</sub> times static. {n} times the length is about {n} "
        f"times the swing (exactly {number(math.sin(n * math.pi / 20) / math.sin(math.pi / 20))} times, undamped).",
        choices=(
            f"About {n} times as large",
            "About the same",
            f"About 1/{n} as large",
            f"About {n * n} times as large",
        ),
        why=(
            "",
            "A push much longer than the period leaves the same swing whatever its length. One much shorter is an "
            "impulse, and its swing grows with its area FT.",
            "A longer push gives more impulse, not less: the area FT grows with it.",
            "The swing grows with the impulse FT, which grows as the length itself; it's the energy that goes as "
            "its square.",
        ),
        setup=dict(start=APPLY, kind=PulseKind.RECTANGLE, amplitude=10.0, duration=rounded(n * 0.0157)),
        predict=True,
    )


def _whole_periods(rng: np.random.Generator) -> Question:
    n = int(pick(rng, (1, 2, 3)))
    periods = "one natural period" if n == 1 else f"{n} natural periods"
    return Question(
        f"An undamped mass has a load ramped onto it in a straight line over exactly {periods}, then held. How "
        "far past the static deflection does it swing?",
        0,
        f"Not at all. During the ramp the mass lags it and swings about it; the overshoot afterwards is "
        f"|sin πr|/πr of the static, with r = {n} the ramp's length in periods, and sin {n}π = 0. The ramp ends "
        "just as the mass has swung back onto it, at rest.",
        choices=("It doesn't overshoot", f"By about 1/{n}π of the static", "To twice the static, as a step would"),
        why=(
            "",
            "That's the envelope, the most a ramp of that length can leave: |sin πr|/πr. With r a whole number, "
            "sin πr = 0 and nothing is left.",
            "A step is a ramp of no length. A ramp over whole periods lets the mass catch up: it ends just as the "
            "mass is back on the load, at rest.",
        ),
        setup=dict(start=APPLY, kind=PulseKind.RAMP, duration=rounded(n * TWO_PI / 20.0, 4), damping=0.0),
        predict=True,
    )


def _ramp_spec(rng: np.random.Generator) -> Question:
    k = pick(rng, (100, 400, 1000))
    tn = TWO_PI / math.sqrt(k)
    force = pick(rng, (1, 5, 10))
    most = pick(rng, (1, 2))  # % overshoot
    longest = rounded(pick(rng, (2.5, 3.5)) * tn, 2)
    design = rounded(2.0 * tn, 4)

    def goal(t: ForcingTrial):
        osc = Oscillator(t.mass, t.stiffness, t.damping)
        ramp = t.kind is PulseKind.RAMP
        over = 100.0 * (peak_ratio(osc.zeta, PulseKind.RAMP, np.array([t.duration / osc.period]))[0] - 1.0)
        return (
            kept("The mass", t.mass, 1.0, "kg"),
            kept("The spring", t.stiffness, k, "N/m"),
            Requirement("No damper", f"{number(t.damping)} N·s/m", t.damping == 0),
            Requirement(
                f"A ramp up to {sig(force, 6)} N", t.kind.value, ramp and abs(t.amplitude - force) <= 0.005 * force
            ),
            at_most("Its rise time", t.duration, longest, "s"),
            at_most("The overshoot", max(over, 0.0) if ramp else 100.0, most, "%"),
        )

    return Question(
        f"A crane lowers {sig(force, 6)} N onto a 1 kg mass on an undamped spring of {sig(k, 6)} N/m (a stand-in for a "
        f"lightly damped structure), ramping the load on in a straight line. It must be on within {number(longest, 2)} "
        f"s, and the structure mustn't swing more than {sig(most, 6)} % past its static deflection. Choose the ramp's "
        "rise time.",
        0,
        f"The overshoot is |sin πr|/πr of the static, r the rise time in periods. Its envelope 1/πr would need "
        f"r ≥ {number(100 / (most * math.pi))}, far too slow; but it vanishes at every whole number of periods. "
        f"T<sub>n</sub> = 2π/√{sig(k, 6)} = {number(tn)} s: ramp over one, two or three periods ({number(design)} s for "
        f"two), to within about ±{number(2 * most / 100 * tn, 2)} s of it.",
        setup=dict(
            start=APPLY, kind=PulseKind.RAMP, amplitude=force, duration=rounded(0.5 * tn, 4), stiffness=k, damping=0.0
        ),
        goal=goal,
        design=dict(start=APPLY, kind=PulseKind.RAMP, amplitude=force, duration=design, stiffness=k, damping=0.0),
    )


_TRANSIENT = tuple(
    Problem(key, title, "forcing_page", hint, make)
    for key, title, hint, make in (
        (
            "impulse-peak",
            "Swing after a blow",
            f"See {page_link('forcing_page', 'impulse-response', 'An impulse and the impulse response')}.",
            _impulse_peak,
        ),
        (
            "step-peak",
            "Overshoot of a step",
            f"See {page_link('forcing_page', 'step-response', 'The step response')}.",
            _step_peak,
        ),
        (
            "peak-time",
            "When a step peaks",
            f"See {page_link('forcing_page', 'step-response', 'The step response')}: in rad/s, ω<sub>d</sub>.",
            _peak_time,
        ),
        (
            "rectangle-peak",
            "A rectangular pulse",
            f"See {page_link('forcing_page', 'pulses', 'Pulses')}: first, its length in periods.",
            _rectangle_peak,
        ),
        (
            "ramp-overshoot",
            "Overshoot of a ramp",
            f"See {page_link('forcing_page', 'pulses', 'Pulses')}.",
            _ramp_overshoot,
        ),
        (
            "sudden-load",
            "Predict: a load let go",
            f"See {page_link('forcing_page', 'step-response', 'The step response')}.",
            _sudden_load,
        ),
        (
            "short-pulse",
            "Predict: a longer short pulse",
            f"See {page_link('forcing_page', 'pulses', 'Pulses')}.",
            _short_pulse,
        ),
        (
            "whole-periods",
            "Predict: a ramp of whole periods",
            f"See {page_link('forcing_page', 'pulses', 'Pulses')}.",
            _whole_periods,
        ),
        (
            "ramp-spec",
            "Spec: put a load on gently",
            f"See {page_link('forcing_page', 'pulses', 'Pulses')}: where is |sin πr|/πr zero?",
            _ramp_spec,
        ),
    )
)


def _swing(t: ForcingTrial) -> float:
    """The steady swing's largest |x| (m), exactly."""
    wave = forcing(t)
    times = np.linspace(0.0, wave.period, 600, endpoint=False)
    return float(np.max(np.abs(periodic_steady(Oscillator(t.mass, t.stiffness, t.damping), wave, times))))


def _square_harmonic(rng: np.random.Generator) -> Question:
    F = pick(rng, (1, 2, 5, 10))
    k = int(pick(rng, (1, 3, 5, 7, 9)))
    f0 = pick(rng, (0.5, 1, 2))
    a = 4 * F / (math.pi * k)
    return Question(
        f"A ±{sig(F, 6)} N square wave repeats at {sig(f0, 6)} Hz. How large is its harmonic at {number(k * f0)} Hz?",
        a,
        f"That is the {k}{'st' if k == 1 else 'th'} harmonic. A square wave is (4F/π) Σ sin jω₀t / j over odd j: "
        f"4 × {sig(F, 6)}/({k}π) = {number(a)} N.",
        unit="N",
        setup=dict(start=WAVE, wave=Wave.SQUARE, wave_amplitude=float(F), freq_hz=f0, harmonics=max(k, 5)),
    )


def _pulse_harmonic(rng: np.random.Generator) -> Question:
    F = pick(rng, (1, 2, 5))
    d = pick(rng, (0.1, 0.2, 0.3, 0.4))
    k = int(pick(rng, (1, 2, 3)))
    a = 2 * F / (math.pi * k) * abs(math.sin(math.pi * k * d))
    return Question(
        f"A train of rectangular pulses of {sig(F, 6)} N, each on for {sig(100 * d, 6)} % of the period. How large is "
        f"its harmonic number {k}?",
        a,
        f"|C<sub>j</sub>| = (2F/πj)|sin πjd| = (2 × {sig(F, 6)}/{k}π) |sin({k}π × {sig(d, 6)})| = {number(a)} N. (It is zero at "
        "every multiple of 1/d.)",
        unit="N",
        setup=dict(start=WAVE, wave=Wave.PULSES, wave_amplitude=float(F), duty=d, harmonics=8),
    )


def _rectified_sag(rng: np.random.Generator) -> Question:
    F = pick(rng, (1, 2, 5, 10))
    k = pick(rng, (100, 400, 1000))
    sag = 2 * F / (math.pi * k)
    return Question(
        f"A rectified sine force F|sin πt/T| with F = {sig(F, 6)} N acts on a {sig(k, 6)} N/m spring. About what mean deflection "
        "does the mass swing around?",
        1e3 * sag,
        f"The mean of F|sin| is 2F/π = {number(2 * F / math.pi)} N, and the mean only sags the spring "
        f"(H(0) = 1/k): {number(1e3 * sag)} mm.",
        unit="mm",
        setup=dict(start=WAVE, wave=Wave.RECTIFIED, wave_amplitude=float(F), stiffness=float(k), tab=TAB_HARM),
    )


def _harmonic_resonance(rng: np.random.Generator) -> Question:
    F = pick(rng, (1, 2, 5))
    zeta = pick(rng, (0.01, 0.02, 0.05))
    n = int(pick(rng, (3, 5)))
    osc = Oscillator(1.0, 400.0, rounded(2 * zeta * 20.0))
    f0 = rounded(osc.natural_hz / n)
    H = float(abs(receptance(osc, np.array([n * f0]))[0]))
    x = 4 * F / (math.pi * n) * H
    ordinal = "3rd" if n == 3 else f"{n}th"
    return Question(
        f"A ±{sig(F, 6)} N square wave at {number(f0)} Hz drives a 1 kg mass on 400 N/m with c = {number(osc.damping)} "
        f"N·s/m (f<sub>n</sub> = {number(osc.natural_hz)} Hz). How large is the response at its {ordinal} harmonic?",
        1e3 * x,
        f"The {ordinal} harmonic, 4F/{n}π = {number(4 * F / (math.pi * n))} N, lands on f<sub>n</sub>, where |H| ≈ "
        f"1/cω<sub>n</sub> = {number(H)} m/N: {number(1e3 * x)} mm. The force runs at a {n}th of the natural "
        "frequency and still resonates it.",
        unit="mm",
        tolerance=0.03,
        setup=dict(
            start=WAVE, wave=Wave.SQUARE, wave_amplitude=float(F), freq_hz=f0, damping=osc.damping, harmonics=n + 2
        ),
    )


def _impulse_lines(rng: np.random.Generator) -> Question:
    blow = pick(rng, (0.1, 0.2, 0.5, 1))
    f0 = pick(rng, (1, 2, 5, 10))
    a = 2 * blow * f0
    return Question(
        f"A hammer strikes a blow of {sig(blow, 6)} N·s {sig(f0, 6)} times a second. How large is each of its harmonics?",
        a,
        f"An impulse train is I/T + (2I/T) Σ cos jω₀t: every harmonic the same, 2I/T = 2 × {sig(blow, 6)} × {sig(f0, 6)} = "
        f"{number(a)} N (the mean is half that).",
        unit="N",
        setup=dict(start=WAVE, wave=Wave.IMPULSES, wave_amplitude=blow, freq_hz=float(f0), harmonics=8),
    )


def _square_at_half(rng: np.random.Generator) -> Question:
    return Question(
        "A square wave drives an oscillator at exactly half its natural frequency. Does any harmonic resonate it?",
        0,
        "No: a square wave has odd harmonics only (it repeats upside down every half period), so nothing lands on "
        "2f₀ = f<sub>n</sub>. A sawtooth there would resonate through its second harmonic.",
        choices=("No: it has no even harmonics", "Yes: its 2nd harmonic", "Yes: every harmonic adds up"),
        why=(
            "",
            "A square wave has no 2nd harmonic: it repeats upside down every half period, which cancels its even "
            "harmonics. A sawtooth's 2nd would resonate it.",
            "Only a harmonic at f<sub>n</sub> resonates it, and the one that would land there, the 2nd, a square "
            "wave doesn't have.",
        ),
        setup=dict(start=WAVE, wave=Wave.SQUARE, tune=2, harmonics=9),
        predict=True,
    )


def _fastest_falling(rng: np.random.Generator) -> Question:
    return Question(
        "Of these waves at the same height, which one's 9th harmonic is the smallest?",
        0,
        "The triangle's: it has no jump, only kinks, so its harmonics fall as 1/k² (1/81 here). The square's and "
        "the sawtooth's jumps make theirs fall as 1/k, and an impulse train's don't fall at all.",
        choices=("Triangle", "Square wave", "Sawtooth", "Impulse train"),
        why=(
            "",
            "A square wave jumps, and a jump's harmonics fall only as 1/k (1/9 here). The triangle has kinks, not "
            "jumps, so its fall as 1/k².",
            "A sawtooth jumps once a period, so its harmonics fall as 1/k, as the square's do; the triangle's fall "
            "as 1/k².",
            "An impulse train's harmonics don't fall at all: every one is the same height.",
        ),
        setup=dict(start=WAVE, wave=Wave.TRIANGLE, harmonics=9),
        predict=True,
    )


def _gibbs(rng: np.random.Generator) -> Question:
    return Question(
        "A square wave's Fourier series is cut at 5 harmonics, then at 50. What happens to its overshoot at the jumps?",
        0,
        "It stays about 9 % of the jump (Gibbs), only narrower: the series never matches a jump, it just squeezes "
        "the ripple toward it.",
        choices=("It stays about 9 %, only narrower", "It shrinks to nothing", "It shrinks to a tenth"),
        why=(
            "",
            "The series matches the wave everywhere but at the jump, yet the overshoot beside the jump never goes: "
            "it stays about 9 % of the jump, only squeezed closer (Gibbs).",
            "The ripple gets ten times narrower, not lower: its height stays about 9 % of the jump.",
        ),
        setup=dict(start=WAVE, wave=Wave.SQUARE, harmonics=50),
        predict=True,
    )


def _speed_spec(rng: np.random.Generator) -> Question:
    F = pick(rng, (1, 2, 5))
    zeta = pick(rng, (0.01, 0.02))
    c = rounded(2 * zeta * 20.0)
    fn = Oscillator(1.0, 400.0, c).natural_hz
    lo, hi = rounded(fn / 3.6), rounded(fn / 1.6)
    design = rounded(fn / 2)
    most = rounded(1.15 * 1e3 * _swing(ForcingTrial("", "", wave_amplitude=float(F), freq_hz=design, damping=c)), 2)

    def goal(t: ForcingTrial):
        return (
            kept("The mass", t.mass, 1.0, "kg"),
            kept("The spring", t.stiffness, 400.0, "N/m"),
            kept("The damper", t.damping, c, "N·s/m"),
            Requirement(
                f"A ±{sig(F, 6)} N square wave",
                f"{t.wave.value}, {number(t.wave_amplitude)}",
                t.wave is Wave.SQUARE and abs(t.wave_amplitude - F) <= 0.005 * F,
            ),
            Requirement(
                f"Its frequency between {number(lo)} and {number(hi)} Hz",
                f"{number(wave_hz(t))} Hz",
                lo <= wave_hz(t) <= hi,
            ),
            at_most("The steady swing", 1e3 * _swing(t), most, "mm"),
        )

    return Question(
        f"A machine's ±{sig(F, 6)} N square-wave force can run anywhere from {number(lo)} to {number(hi)} Hz. It shakes "
        f"a 1 kg mass on 400 N/m with c = {number(c)} N·s/m (f<sub>n</sub> = {number(fn)} Hz). Pick a speed where "
        f"the mass's steady swing stays within ±{number(most, 2)} mm.",
        0,
        f"Each odd harmonic jf₀ must keep off f<sub>n</sub>. f₀ = f<sub>n</sub>/3 = {number(fn / 3)} Hz puts the 3rd "
        "on it. Below that the jumps come slowly and each one rings the mass (the 5th, 7th, … harmonics crowd "
        "round f<sub>n</sub>); above it the 3rd is past f<sub>n</sub>, and only the 1st matters, still below "
        f"resonance. So anywhere from about 0.4f<sub>n</sub> up to {number(hi)} Hz does: f<sub>n</sub>/2 = "
        f"{number(fn / 2)} Hz, say, where a square wave has no harmonic at all.",
        setup=dict(start=WAVE, wave_amplitude=float(F), freq_hz=rounded(fn / 3), damping=c, harmonics=9, tab=TAB_HARM),
        goal=goal,
        design=dict(start=WAVE, wave_amplitude=float(F), freq_hz=design, damping=c, harmonics=9, tab=TAB_HARM),
    )


_PERIODIC = tuple(
    Problem(key, title, "forcing_page", f"See {page_link('forcing_page', anchor, section)}{more}.", make)
    for key, title, anchor, section, more, make in (
        ("square-harmonic", "A square wave's harmonic", "shapes", "Shapes and their harmonics", "", _square_harmonic),
        ("pulse-harmonic", "A pulse train's harmonic", "shapes", "Shapes and their harmonics", "", _pulse_harmonic),
        ("rectified-sag", "The mean of a force", "superposition", "Each harmonic through H", "", _rectified_sag),
        (
            "harmonic-resonance",
            "A harmonic on resonance",
            "superposition",
            "Each harmonic through H",
            "",
            _harmonic_resonance,
        ),
        ("impulse-lines", "A hammer's harmonics", "shapes", "Shapes and their harmonics", "", _impulse_lines),
        (
            "square-at-half",
            "Predict: half the natural frequency",
            "shapes",
            "Shapes and their harmonics",
            "",
            _square_at_half,
        ),
        (
            "fastest-falling",
            "Predict: the smoothest wave",
            "shapes",
            "Shapes and their harmonics",
            "",
            _fastest_falling,
        ),
        ("gibbs", "Predict: more harmonics", "shapes", "Shapes and their harmonics", "", _gibbs),
        (
            "speed-spec",
            "Spec: a running speed",
            "superposition",
            "Each harmonic through H",
            ": which harmonics does a square wave have?",
            _speed_spec,
        ),
    )
)


PROBLEMS = _TRANSIENT + _PERIODIC

TIPS = {
    "mass": "<p>The mass m. Changed as it runs, a heavier mass captures its extra at rest (the momentum kept).</p>",
    "stiffness": "<p>The spring k.</p>",
    "damping": "<p>The damper c.</p>",
    "kind": "<p>The force <i>Apply</i> fires, from the moment it's pressed, on whatever the mass is doing. <b>Hit:</b> "
    "a blow of impulse I (N·s), over in an instant. <b>Push and hold:</b> F switched on and held until <i>Let go</i>. "
    "<b>Ramp, then hold:</b> F reached in a straight line over the time T, held until Let go. <b>Rectangular "
    "pulse:</b> F for T. <b>Half-sine pulse:</b> F sin(πt/T) for T.</p>",
    "amplitude": "<p>The force F, or a hit's impulse I = ∫F dt.</p>",
    "duration": "<p>A pulse's length T, or a ramp's rise time.</p>",
    "apply": "<p>Fire the force from now. Applied with the mass at rest, a pulse lands its peak on the Pulse spectrum "
    "tab a period after it ends; applied while it moves, its response adds to the motion (superposition).</p>",
    "let_go": "<p>End a push or a ramp being held: the force drops to zero, and the mass rings about its rest.</p>",
    "wave": "<p>The periodic force <i>Wave</i> switches on: each has its Fourier series in closed form.</p>",
    "wave_amplitude": "<p>The wave's height F (for hammer blows, each blow's impulse I, N·s).</p>",
    "freq": "<p>How often it repeats, f₀ = 1/T₀. Changed as it runs, the wave carries on from where it is in its period.</p>",
    "duty": "<p>The fraction d of each period the rectangular pulses are on.</p>",
    "harmonics": "<p>How many harmonics N the Harmonics tab shows.</p>",
    "tune": "<p>Set f₀ to f<sub>n</sub> divided by this, so that harmonic sits on resonance (0: the frequency box's).</p>",
    "wave_on": "<p>Switch the wave on from now, or off.</p>",
    "theory": "<p>Draw the theory over what's measured: the pulse spectrum's curves; the series' lines and |H|.</p>",
}
X_TIP = (
    "<p>The mass's displacement, the last 8 s. A dashed line marks a change of m, k or c, or a pluck. Paused, drag "
    "the line at the end back: the Convolution tab draws the integral at that instant.</p>"
)
F_TIP = "<p>The force on the mass, the last 8 s; a hit as a mark (its impulse I is a force for an instant).</p>"
CONV_TIP = (
    "<p>Duhamel's integral x(t) = ∫ f(τ) h(t − τ) dτ at the instant shown, from the run's record since the mass was "
    "last let go or changed: each slice f(τ)dτ of the force is a small impulse, which rings as h from then on; at t "
    "it has rung for t − τ. Their product (shaded) summed, with each hit's I h(t − tᵢ), is x(t). h is scaled to the "
    "force's plot.</p>"
)
PULSE_TIP = (
    "<p>The largest |x| over its static F/k against the pulse's length over the natural period, one point per pulse "
    "applied with the mass at rest. Short pulses are impulses (the peak grows with the area); long ones are "
    "static, a rectangle's at twice and a ramp's at once the static deflection. <i>Show the theory</i> draws each "
    "kind's curve for this damping.</p>"
)
HARM_TIP = (
    "<p>Each harmonic's height at its frequency, measured from the steady run: the force's (top) and the response's "
    "(bottom). <i>Show the theory</i> adds the series' own lines and |H| (dashed, scaled to the response's largest "
    "line); f<sub>n</sub> dotted.</p>"
)

NOTES_HTML = load_notes("forcing", figures=forcing_figures.FIGURES, trials=trials_html(TRIALS))
