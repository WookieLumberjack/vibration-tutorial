"""The Single oscillator's Things to try, practice problems, tips and notes, without Qt (#478 item 7).

Its state is `view_models/sdof.py`'s `SdofTrial`; `gui/sdof.py` binds the page's controls to it and runs it live.
A trial is a setup to go back to: it sets the oscillator and the drive, plucks the mass from `x0_mm` or switches
the drive on (or starts a sweep), and says what to look for. Its prediction is asked only once the reader has
handled the system (`lesson.TrialBox(ask_after_play=True)`), about a case they haven't tried.

Working on it: `tests/test_tutorial_numbers.py` checks the trials' numbers, `tests/test_practice.py` the problems,
`tests/test_qt_free.py` that this loads no Qt.
"""

from __future__ import annotations

import math

import numpy as np

from ..core.modal import log_decrement
from ..core.practice import Placement, Prediction, Problem, Question, Requirement, at_most, kept, number, pick, rounded
from ..core.sdof import TWO_PI, Oscillator, receptance, settling_time
from ..view_models.sdof import FORCE, TAB_ENERGY, TAB_FRF, TAB_POLES, UNBALANCE, SdofTrial
from ..view_models.text import number as sig
from . import sdof_figures
from .links import course_map, keys_html, load_notes, page_link, paths_html
from .trials_text import trials_html

TRIALS = (
    SdofTrial(
        "Ringing down: the log decrement",
        "<m>m</m> = 1 kg, <m>k</m> = 400 N/m, <m>c</m> = 2 N·s/m: <m>f<sub>n</sub></m> = 3.18 Hz and <m>ζ = 0.05</m>. "
        "Plucked from 10 mm, each peak is 1.37 times the next: the log decrement <m>δ = ln 1.37 = 0.315</m>, and "
        "<m>ζ = δ/2π = 0.050</m>. The readout under the time plot measures it from the peaks as they come; the "
        "phase plane winds in, a turn a cycle. Drag the damper's handle while it rings and the decay changes there "
        "and then.",
        x0_mm=10.0,
        tab=TAB_POLES,
        predict=Prediction(
            "Plucked from 10 mm with ζ = 0.05, how much smaller is each peak than the one before?",
            ("5 % smaller", "About 27 % smaller", "Half the size"),
            1,
            ordered=True,
            why=(
                "ζ isn't the fraction lost a cycle: each peak is e<sup>−2πζ</sup> = 0.73 of the one before, so about 2π times as much goes.",
                "",
                "That would take ζ ≈ 0.11, twice this. At ζ = 0.05 each peak is e<sup>−2πζ</sup> = 0.73 of the one before.",
            ),
        ),
    ),
    SdofTrial(
        "Critical damping: c = 40 N·s/m",
        "<m>c</m> = 40 N·s/m, the critical damping <m>2√(km)</m>: plucked, the mass comes back without crossing "
        "zero, the fastest that does. Hold the curves (H), set <m>c</m> to 80 N·s/m (<m>ζ = 2</m>) and pluck it "
        "again: it creeps back, held up by its slow root at <m>−0.27ω<sub>n</sub></m> (a time constant of 0.19 s "
        "against critical's 0.05 s).",
        damping=40.0,
        x0_mm=10.0,
        tab=TAB_POLES,
        predict=Prediction(
            "The same mass and spring, plucked from the same place: ζ = 1 or ζ = 2, which is back at rest sooner?",
            ("ζ = 1", "ζ = 2", "They settle together"),
            0,
            why=(
                "",
                "More damping takes energy out faster, but it also holds the mass back: past critical, the slow root moves towards zero and the creep back takes longer.",
                "Their slowest roots differ: −ω<sub>n</sub> for ζ = 1 and −0.27ω<sub>n</sub> for ζ = 2.",
            ),
        ),
    ),
    SdofTrial(
        "Overdamped: two real poles",
        "<m>c</m> = 80 N·s/m, <m>ζ = 2</m>. The poles have left the imaginary axis for two points on the real one, at "
        "−5.4 and −74.6 s<sup>−1</sup>: two decays and no oscillation. Drag either along the real axis, or off it, "
        "and <m>k</m> and <m>c</m> follow, the motion carrying on.",
        damping=80.0,
        x0_mm=10.0,
        tab=TAB_POLES,
    ),
    SdofTrial(
        "Resonance: a 90° lag",
        "A 1 N force at <m>f<sub>n</sub></m> = 3.18 Hz, switched on at rest. The swing grows over a few time "
        "constants (1 s) to <m>(F/k)/2ζ</m> = 25 mm, ten times the static 2.5 mm: the quality factor "
        "<m>Q = 1/2ζ = 10</m>. Once it's steady its point lands on the Frequency response tab, lagging the force by "
        "90°. On the body, the spring's and the mass's forces cancel: the damper's arrow alone balances the drive's.",
        drive=True,
        tune=True,
        tab=TAB_FRF,
        predict=Placement(
            "1 N on a 400 N/m spring is 2.5 mm, pushed slowly. Driven at f<sub>n</sub> with ζ = 0.05, how big does the steady "
            "swing grow? Put the line at its peak.",
            plot="strip",
            answer=0.025,
            span=(0.0, 0.05),
            low="Short of 25 mm the force still puts in more each cycle than the damper takes out, so the swing "
            "grows on, to (F/k)/2ζ: ten times static.",
            high="The damper's loss grows as the swing squared, the force's input only as the swing: they balance at "
            "(F/k)/2ζ = 25 mm.",
            scale=1000.0,
            unit="mm",
        ),
    ),
    SdofTrial(
        "Building the frequency response: a sweep",
        "The drive steps from 1.06 to 9.55 Hz, dwelling at each frequency until the swing is steady, and each "
        "point lands as it's measured: the peak builds up near <m>f<sub>n</sub></m>, and the lag climbs through 90° "
        "there. Show the theory's curve over the points: <m>|H|</m> falls to <m>1/√2</m> of its peak at 3.01 and "
        "3.33 Hz, 0.32 Hz apart, so <m>ζ ≈ Δf/2f<sub>n</sub> = 0.050</m>. On the Nyquist plot below, the points near "
        "resonance trace nearly a circle of diameter <m>1/cω<sub>n</sub></m> = 25 mm/N.",
        drive=True,
        sweep=True,
        tab=TAB_FRF,
    ),
    SdofTrial(
        "Undamped, just off resonance",
        "<m>c = 0</m>, a 1 N force at 3.0 Hz, 0.18 Hz below <m>f<sub>n</sub></m>. The free vibration the start sets off "
        "never dies, and it drifts in and out of step with the forced one: the swing beats at "
        "<m>|f − f<sub>n</sub>|</m> = 0.18 Hz, every 5.5 s, between zero and twice the steady 22 mm. It never "
        "settles, so no point lands on the Frequency response tab.",
        damping=0.0,
        drive=True,
        freq_hz=3.0,
        tab=TAB_ENERGY,
        predict=Prediction(
            "No damping, driven from rest 0.18 Hz below f<sub>n</sub>. What does the swing do?",
            ("Grows for ever", "Settles to a steady swing", "Swells and fades, over and over"),
            2,
            why=(
                "Only exactly at f<sub>n</sub>. 0.18 Hz off, the forced swing drifts out of step with the free one, and they cancel every 5.5 s.",
                "Settling needs damping to kill the free vibration the start sets off. With none it rings on, beating against the forced swing.",
                "",
            ),
        ),
    ),
    SdofTrial(
        "Exactly at resonance, undamped",
        "<m>c = 0</m> and the force at <m>f<sub>n</sub></m>. There is no steady state: the swing grows in a straight "
        "line, <m>F/2mω<sub>n</sub></m> = 25 mm a second, for ever, and the Energy tab shows the drive's work all "
        "going into store. Drag the damper's handle up while it grows and it levels off.",
        damping=0.0,
        drive=True,
        tune=True,
        tab=TAB_ENERGY,
        predict=Prediction(
            "No damping, driven from rest exactly at f<sub>n</sub>. What does the swing do?",
            ("Grows steadily, for ever", "Settles to a steady swing", "Swells and fades, over and over"),
            0,
            why=(
                "",
                "With no damping nothing takes out the energy the force puts in, and at resonance it puts some in every cycle.",
                "Beating needs two frequencies apart. Here they're the same, so the free and forced swings never drift out of step.",
            ),
        ),
    ),
    SdofTrial(
        "A rotating unbalance",
        "An unbalance <m>m<sub>e</sub>e</m> = 1×10<sup>−4</sup> kg·m (10 g at 10 mm) on a 1 kg machine. Its force "
        "grows as <m>Ω²</m>, so the response starts at zero, peaks at resonance at <m>(m<sub>e</sub>e/m)/2ζ</m> = 1 mm, "
        "and levels off far above it at <m>m<sub>e</sub>e/m</m> = 0.1 mm, 180° out of phase: the machine turns round "
        "the shared centre of mass. Sweep it to see the curve build.",
        drive=True,
        kind=UNBALANCE,
        amplitude=1e-4,
        tune=True,
        tab=TAB_FRF,
        predict=Prediction(
            "An unbalanced machine run faster and faster, far past its resonance: its swing…",
            ("keeps growing, as the speed squared", "levels off", "falls back to nothing"),
            1,
            why=(
                "The force does, but far above resonance the mass's inertia takes it all, x ≈ −F/mΩ²: the Ω² cancel, leaving m<sub>e</sub>e/m.",
                "",
                "That's a fixed force's response, falling as 1/Ω². The unbalance's force grows as Ω², and the two cancel: it levels off.",
            ),
        ),
    ),
)


# ------------------------------------------------------------------ practice problems
# Each question's numbers are printed round (or to 3–4 figures), and its answer is worked from them as printed.


def _natural_frequency(rng: np.random.Generator) -> Question:
    m = pick(rng, (0.5, 1, 2, 4, 5, 10))
    k = pick(rng, (50, 100, 200, 400, 800, 1000, 2500, 5000))
    osc = Oscillator(m, k, 0.0)
    return Question(
        f"A {sig(m, 6)} kg mass sits on a spring of {sig(k, 6)} N/m. What is its natural frequency f<sub>n</sub>?",
        osc.natural_hz,
        f"ω<sub>n</sub> = √(k/m) = √({sig(k, 6)}/{sig(m, 6)}) = {number(osc.omega_n)} rad/s, and f<sub>n</sub> = "
        f"ω<sub>n</sub>/2π = {number(osc.natural_hz)} Hz.",
        unit="Hz",
        setup=dict(mass=m, stiffness=k, damping=rounded(0.1 * math.sqrt(k * m)), x0_mm=10.0, tab=TAB_POLES),
    )


def _critical_damping(rng: np.random.Generator) -> Question:
    m = pick(rng, (0.5, 1, 2, 4, 5, 10))
    k = pick(rng, (50, 100, 200, 400, 800, 1000, 2500, 5000))
    cc = Oscillator(m, k, 0.0).critical
    return Question(
        f"A {sig(m, 6)} kg mass on a spring of {sig(k, 6)} N/m. What damping coefficient c makes it critically damped?",
        cc,
        f"c<sub>c</sub> = 2√(km) = 2√({sig(k, 6)} × {sig(m, 6)}) = {number(cc)} N·s/m (that is, 2mω<sub>n</sub>).",
        unit="N·s/m",
        setup=dict(mass=m, stiffness=k, damping=rounded(cc), x0_mm=10.0, tab=TAB_POLES),
    )


def _log_decrement(rng: np.random.Generator) -> Question:
    k = pick(rng, (100, 400, 900, 1600))
    zeta = pick(rng, (0.01, 0.02, 0.03, 0.05, 0.08, 0.1))
    n = int(pick(rng, (1, 2, 3, 5)))
    x0 = pick(rng, (5, 10, 20))
    delta = log_decrement(zeta)
    xn = rounded(x0 * math.exp(-n * delta))
    d = math.log(x0 / xn) / n  # from the printed peaks
    answer = d / math.sqrt(TWO_PI**2 + d * d)
    cycles = "the next peak" if n == 1 else f"the peak {n} cycles later"
    return Question(
        f"A mass released from rest rings down. One peak is {sig(x0, 6)} mm and {cycles} is {number(xn)} mm. What is "
        "its damping ratio ζ?",
        answer,
        f"The log decrement is δ = (1/{n}) ln({sig(x0, 6)}/{number(xn)}) = {number(d)}, and ζ = δ/√(4π² + δ²) = "
        f"{number(answer)} (δ/2π = {number(d / TWO_PI)} is near enough when ζ is small).",
        tolerance=0.03,
        setup=dict(mass=1.0, stiffness=k, damping=rounded(2.0 * zeta * math.sqrt(k)), x0_mm=x0, tab=TAB_POLES),
    )


def _resonant_amplitude(rng: np.random.Generator) -> Question:
    m = pick(rng, (1, 2, 5))
    k = pick(rng, (200, 400, 800, 2000))
    c = rounded(pick(rng, (0.02, 0.05, 0.1)) * 2.0 * math.sqrt(k * m))
    force = pick(rng, (0.5, 1, 2, 5))
    osc = Oscillator(m, k, c)
    x = force / (c * osc.omega_n)
    return Question(
        f"m = {sig(m, 6)} kg, k = {sig(k, 6)} N/m and c = {number(c)} N·s/m, driven by {sig(force, 6)} N at exactly its natural "
        "frequency. What is the steady amplitude?",
        1e3 * x,
        f"At ω<sub>n</sub> = {number(osc.omega_n)} rad/s the spring and the mass cancel, and only the damper "
        f"holds the force: X = F/(cω<sub>n</sub>) = {sig(force, 6)}/({number(c)} × {number(osc.omega_n)}) = "
        f"{number(1e3 * x)} mm. That's (F/k)/2ζ: Q = 1/2ζ = {number(1.0 / (2.0 * osc.zeta))} times the static "
        f"{number(1e3 * force / k)} mm.",
        unit="mm",
        setup=dict(mass=m, stiffness=k, damping=c, drive=True, amplitude=force, tune=True, tab=TAB_FRF),
    )


def _half_power(rng: np.random.Generator) -> Question:
    k = pick(rng, (400, 1000, 2500, 10000))
    zeta = pick(rng, (0.02, 0.03, 0.05, 0.08, 0.1))
    osc = Oscillator(1.0, k, 2.0 * zeta * math.sqrt(k))
    band = osc.half_power_hz()
    assert band is not None, "ζ ≤ 0.1 has a peak"
    lo, hi = (rounded(f, 4) for f in band)
    estimate = (hi - lo) / (hi + lo)
    return Question(
        f"A receptance's peak has its half-power points at {number(lo, 4)} Hz and {number(hi, 4)} Hz. What is "
        "the damping ratio ζ?",
        zeta,
        f"ζ ≈ Δf/2f<sub>n</sub>, with f<sub>n</sub> ≈ ({number(lo, 4)} + {number(hi, 4)})/2 = "
        f"{number((lo + hi) / 2, 4)} Hz and Δf = {number(hi - lo)} Hz: ζ ≈ {number(estimate)}. (Exactly, "
        f"ζ = {number(zeta)}.)",
        tolerance=0.03,
        setup=dict(
            mass=1.0,
            stiffness=k,
            damping=rounded(2.0 * zeta * math.sqrt(k)),
            drive=True,
            sweep=True,
            tab=TAB_FRF,
        ),
    )


def _more_damping(rng: np.random.Generator) -> Question:
    n = int(pick(rng, (2, 3, 5)))
    return Question(
        f"An oscillator driven at its natural frequency has settled to its steady swing. Its damper is made {n} "
        "times stronger. What happens to the steady amplitude?",
        0,
        f"At resonance the spring's and the mass's forces cancel, so only the damper holds the force: X = "
        f"F/(cω<sub>n</sub>), and {n} times the c gives 1/{n} of the swing. (Off resonance damping matters far "
        "less.)",
        choices=(f"It falls to 1/{n}", f"It falls to 1/√{n}", "It stays the same", f"It grows {n} times"),
        why=(
            "",
            "A square root comes in where the frequency depends on stiffness or mass; the amplitude at resonance "
            "is inversely proportional to the damping itself.",
            "That holds well away from resonance, where the spring or the mass carries the force; at resonance "
            "they cancel, and the damper alone holds it.",
            "Damping takes energy out every cycle: more of it can only lower the swing.",
        ),
        setup=dict(damping=2.0 * n, drive=True, tune=True, tab=TAB_FRF),
        predict=True,
    )


def _heavier(rng: np.random.Generator) -> Question:
    n = int(pick(rng, (4, 9)))
    root = math.isqrt(n)
    return Question(
        f"The mass on a spring is made {n} times heavier, the spring the same. What happens to its natural frequency?",
        0,
        f"f<sub>n</sub> = √(k/m)/2π: {n} times the mass gives 1/√{n} = 1/{root} of the frequency.",
        choices=(f"It falls to 1/{root}", f"It falls to 1/{n}", "It stays the same", f"It rises {root} times"),
        why=(
            "",
            "The frequency goes as the square root of k/m, not as k/m itself.",
            "A pendulum's frequency doesn't depend on its mass, as gravity pulls the heavier bob harder in step; "
            "a spring's force doesn't grow with the mass.",
            "The heavier mass is slower to move: the same spring swings it less often, not more.",
        ),
        setup=dict(mass=float(n), x0_mm=10.0, tab=TAB_POLES),
        predict=True,
    )


def _quickest_return(rng: np.random.Generator) -> Question:
    return Question(
        "The same mass and spring, released from the same place, once critically damped (ζ = 1) and once with "
        "twice that damping (ζ = 2). Which settles back to rest sooner?",
        0,
        "Critical damping. More damping slows the return: ζ = 2's slow root at −0.27ω<sub>n</sub> holds it back "
        "(a time constant of 0.19 s here, against critical's 0.05 s).",
        choices=("ζ = 1", "ζ = 2", "They settle together"),
        why=(
            "",
            "More damping takes energy out faster, but it also holds the mass back: past critical, the slow root "
            "moves towards zero and the creep back takes longer.",
            "Their slowest roots differ: −ω<sub>n</sub> for ζ = 1 and −0.27ω<sub>n</sub> for ζ = 2.",
        ),
        setup=dict(damping=40.0, x0_mm=10.0, tab=TAB_POLES),
        predict=True,
    )


def _drive_force(t: SdofTrial) -> float:
    """The drive's force amplitude at its frequency: F, or m_e e Ω² for the unbalance."""
    return t.amplitude * (TWO_PI * t.freq_hz) ** 2 if t.kind == UNBALANCE else t.amplitude


def _settle_spec(rng: np.random.Generator) -> Question:
    m = pick(rng, (1, 2, 5))
    k = pick(rng, (100, 400, 1000, 2500))
    c = rounded(0.8 * 2.0 * math.sqrt(k * m))  # ζ = 0.8: an overshoot that lands just inside the band
    best = settling_time(Oscillator(m, k, c))
    most = rounded(best * pick(rng, (1.2, 1.5)), 2)

    def goal(t: SdofTrial):
        return (
            kept("The mass", t.mass, m, "kg"),
            kept("The spring", t.stiffness, k, "N/m"),
            at_most("Time to settle within 2 %", settling_time(Oscillator(t.mass, t.stiffness, t.damping)), most, "s"),
        )

    return Question(
        f"A {sig(m, 6)} kg mass on a spring of {sig(k, 6)} N/m is a pointer that must come to rest quickly when released. "
        f"Choose the damper so that, released at rest from x<sub>0</sub>, it stays within 2 % of x<sub>0</sub> "
        f"of its rest position after {number(most, 2)} s.",
        0,
        f"Too little damping rings on; too much creeps back on its slow root. The quickest into a ±2 % band is "
        f"near ζ ≈ 0.8, which overshoots just inside the band: c = 2ζ√(km) = {number(c)} N·s/m settles in "
        f"{number(best)} s. (Critically damped, c = {number(2 * math.sqrt(k * m))} N·s/m, it takes "
        f"{number(settling_time(Oscillator(m, k, 2 * math.sqrt(k * m))))} s.)",
        setup=dict(mass=m, stiffness=k, damping=rounded(0.1 * math.sqrt(k * m)), x0_mm=10.0, tab=TAB_POLES),
        goal=goal,
        design=dict(mass=m, stiffness=k, damping=c, x0_mm=10.0, tab=TAB_POLES),
    )


def _detune_spec(rng: np.random.Generator) -> Question:
    m = pick(rng, (1, 2, 5))
    k = pick(rng, (400, 1000, 2500))
    c = rounded(2.0 * pick(rng, (0.02, 0.05)) * math.sqrt(k * m))
    force = pick(rng, (1, 2, 5))
    f = rounded(Oscillator(m, k, c).natural_hz)
    w2 = m * (TWO_PI * f) ** 2
    stiffer = rounded(pick(rng, (1.5, 2.0)) * k)
    most = rounded(1.1 * 1e3 * force * abs(receptance(Oscillator(m, stiffer, c), np.array([f]))[0]), 2)

    def goal(t: SdofTrial):
        osc = Oscillator(t.mass, t.stiffness, t.damping)
        x = _drive_force(t) * abs(receptance(osc, np.array([t.freq_hz]))[0])
        return (
            kept("The mass", t.mass, m, "kg"),
            kept("The damper", t.damping, c, "N·s/m"),
            Requirement(
                f"The drive kept at {sig(force, 6)} N",
                f"{number(_drive_force(t), 4)} N",
                t.kind == FORCE and abs(t.amplitude - force) <= 0.005 * force,
            ),
            kept("Its frequency", t.freq_hz, f, "Hz"),
            at_most("The steady amplitude", 1e3 * x, most, "mm"),
        )

    return Question(
        f"A {sig(m, 6)} kg mass on a spring of {sig(k, 6)} N/m, with a {number(c)} N·s/m damper, is shaken by {sig(force, 6)} N at "
        f"{number(f)} Hz: right on its resonance. The mass, the damper and the drive can't change. Change the "
        f"spring so the steady amplitude is no more than {number(most, 2)} mm.",
        0,
        f"At resonance only the damper holds the force: X = F/(cω) = {number(1e3 * force / (c * TWO_PI * f))} mm. "
        f"Move the resonance away from the drive: X ≈ F/|k − mω²| (the damper hardly matters off resonance), so "
        f"|k − mω²| ≥ F/X = {number(force / (1e-3 * most))} N/m with mω² = {number(w2)} N/m: k ≥ "
        f"{number(w2 + force / (1e-3 * most))} N/m, or k ≤ {number(w2 - force / (1e-3 * most))} N/m. A spring of "
        f"{number(stiffer)} N/m does it.",
        setup=dict(mass=m, stiffness=k, damping=c, drive=True, amplitude=force, freq_hz=f, tab=TAB_FRF),
        goal=goal,
        design=dict(mass=m, stiffness=stiffer, damping=c, drive=True, amplitude=force, freq_hz=f, tab=TAB_FRF),
    )


PROBLEMS = (
    Problem(
        "natural-frequency",
        "Natural frequency",
        "sdof_page",
        f"See {page_link('sdof_page', 'free-vibration', 'Free vibration')}: ω<sub>n</sub> = √(k/m), in rad/s.",
        _natural_frequency,
        figure="fig:oscillator",
    ),
    Problem(
        "critical-damping",
        "Critical damping",
        "sdof_page",
        f"See {page_link('sdof_page', 'damping-regimes', 'Under, critical and over')}.",
        _critical_damping,
    ),
    Problem(
        "log-decrement",
        "Damping from a decay",
        "sdof_page",
        f"See {page_link('sdof_page', 'log-decrement', 'Measuring damping from a decay')}: over n cycles, "
        "δ = (1/n) ln(x<sub>0</sub>/x<sub>n</sub>).",
        _log_decrement,
    ),
    Problem(
        "resonant-amplitude",
        "Amplitude at resonance",
        "sdof_page",
        f"See {page_link('sdof_page', 'forced-response', 'Forced response')}: what's left of k − ω²m + iωc "
        "at ω<sub>n</sub>?",
        _resonant_amplitude,
    ),
    Problem(
        "half-power",
        "Damping from the half-power points",
        "sdof_page",
        f"See {page_link('sdof_page', 'half-power', 'The half-power bandwidth')}.",
        _half_power,
    ),
    Problem(
        "more-damping",
        "Predict: more damping at resonance",
        "sdof_page",
        f"See {page_link('sdof_page', 'forced-response', 'Forced response')}.",
        _more_damping,
    ),
    Problem(
        "heavier",
        "Predict: a heavier mass",
        "sdof_page",
        f"See {page_link('sdof_page', 'free-vibration', 'Free vibration')}.",
        _heavier,
    ),
    Problem(
        "quickest-return",
        "Predict: the quickest return",
        "sdof_page",
        f"See {page_link('sdof_page', 'damping-regimes', 'Under, critical and over')}.",
        _quickest_return,
    ),
    Problem(
        "settle-spec",
        "Spec: come to rest quickly",
        "sdof_page",
        f"See {page_link('sdof_page', 'damping-regimes', 'Under, critical and over')}: is critical damping "
        "really the quickest into a band?",
        _settle_spec,
    ),
    Problem(
        "detune-spec",
        "Spec: off the resonance",
        "sdof_page",
        f"See {page_link('sdof_page', 'forced-response', 'Forced response')}.",
        _detune_spec,
    ),
)

TIPS = {
    "mass": "<p>The mass m. Drag the handle on the mass's corner to change it as it runs: added mass is "
    "caught at rest (the momentum kept), removed mass leaves at the mass's speed.</p>",
    "stiffness": "<p>The spring's stiffness k: the force per metre of stretch. Drag the handle on the spring "
    "(up: stiffer) to change it as it runs.</p>",
    "damping": "<p>The viscous damper's coefficient c: the force per m/s of velocity. Its critical value "
    "2√(km) is the least that stops the oscillator overshooting. Drag the handle on the damper to change it.</p>",
    "drive": "<p>Switch the drive on or off. It joins in at the frequency and phase it has, the mass carrying on "
    "from where it is.</p>",
    "kind": "<p><b>Force:</b> F sin Ωt on the mass.<br><b>Rotating unbalance:</b> a mass m<sub>e</sub> turning "
    "at radius e inside the machine, whose force m<sub>e</sub>eΩ² sin Ωt grows with the speed.</p>",
    "amplitude": "<p>The force's amplitude F, or the unbalance m<sub>e</sub>e.</p>",
    "freq": "<p>The drive's frequency Ω/2π. A change joins on with no jump in the force; the swing then settles "
    "to its new steady state, and a point lands on the Frequency response tab once it has.</p>",
}

NOTES_HTML = load_notes(
    "sdof",
    figures=sdof_figures.FIGURES,
    course=course_map(),
    paths=paths_html(),
    keys=keys_html(),
    trials=trials_html(TRIALS),
)
