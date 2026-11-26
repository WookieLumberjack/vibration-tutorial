"""The Single oscillator's damping laws: their Things to try, practice problems and tips, without Qt (#478 item 7).

They were the Damping models page's, merged into the Single oscillator (one example, one page: the same mass and
spring, its damper's law picked). Their state is the page's, `view_models/sdof.py`'s `SdofTrial`; `sdof_trials`
puts them after the oscillator's own, and its notes hold theirs. A trial is a setup to go back to: it picks the
law, plucks the mass from `x0_mm` or drives it (or sweeps), and says what to look for; its prediction is asked once
the reader has handled the system.

Working on it: `tests/test_tutorial_numbers.py` checks the trials' numbers, `tests/test_practice.py` the problems,
`tests/test_qt_free.py` that this loads no Qt.
"""

from __future__ import annotations

import math

import numpy as np

from ..core.damping_models import Matched, Model, coulomb_stop
from ..core.practice import Prediction, Problem, Question, at_most, kept, number, pick, rounded
from ..core.sdof import TWO_PI
from ..view_models.sdof import TAB_ENERGY, TAB_FRF, TAB_LOOP, SdofTrial, matched
from ..view_models.text import number as sig
from .links import page_link

TRIALS = (
    SdofTrial(
        "Four loops, one energy",
        "<m>m</m> = 1 kg, <m>k</m> = 400 N/m (<m>f<sub>n</sub></m> = 3.18 Hz), each damping law matched to the "
        "damper <m>c</m> = 2 N·s/m (<m>ζ</m> = 0.05) there. Driven at <m>f<sub>n</sub></m> by 0.4 N, the viscous mass settles to ±10 mm, and its "
        "loop of damping force against <m>x</m> encloses <m>πcω<sub>n</sub>X²</m> = 12.6 mJ a cycle: the work the "
        "damper takes. Every model's loop at that cycle, drawn faint, encloses the same: that is what matched means. "
        "Viscous, hysteretic and Zener damping make ellipses, their force a quarter cycle ahead of the motion; "
        "friction makes a rectangle, <m>±F<sub>f</sub></m> = ±0.31 N. Pick each law in the Damper box in turn, as it runs.",
        drive=True,
        tune=True,
        amplitude=0.4,
        tab=TAB_LOOP,
        predict=Prediction(
            "Viscous, hysteretic, friction and Zener damping, matched at fₙ and cycled there through ±10 mm. Which takes out the most energy a cycle?",
            ("Viscous", "Friction", "All the same"),
            2,
            why=(
                "Matched means taking the same energy a cycle, at f<sub>n</sub> and this amplitude: every loop encloses πcω<sub>n</sub>X² = 12.6 mJ.",
                "Friction's loop is a rectangle and viscous damping's an ellipse, but matched they enclose the same 12.6 mJ.",
                "",
            ),
        ),
    ),
    SdofTrial(
        "Twice the frequency",
        "Driven at <m>2f<sub>n</sub></m> = 6.37 Hz by 12 N, the swing is about ±10 mm again. The viscous loop is twice "
        "as fat (25.1 mJ: its force grows with the velocity), the hysteretic and friction loops are unchanged "
        "(12.6 mJ), and the Zener element's, past its loss peak, is thinner (10.7 mJ).",
        drive=True,
        freq_hz=6.366,
        amplitude=12.0,
        tab=TAB_LOOP,
        predict=Prediction(
            "The same cycle at twice fₙ. Which loops grow?",
            ("All of them", "Only the viscous one", "None of them"),
            1,
            why=(
                "Hysteretic and friction loops don't depend on frequency, and the Zener element's thins past its loss peak: only the viscous loop grows.",
                "",
                "The viscous damper's force grows with the velocity: at twice the frequency its loop is twice as fat, 25.1 mJ.",
            ),
        ),
    ),
    SdofTrial(
        "With the spring: what a test machine draws",
        "The whole force, spring and damping together, as a test machine measures it on a sample: the loop is the "
        "spring's line <m>kx</m> (dashed) opened out by the damping. The tilt is the stiffness (4 N at 10 mm), the "
        "width the loss (0.4 N): a lightly damped part draws a thin loop.",
        drive=True,
        tune=True,
        amplitude=0.4,
        damping_only=False,
        tab=TAB_LOOP,
    ),
    SdofTrial(
        "Friction at resonance",
        "Friction alone, driven by 1 N at <m>f<sub>n</sub></m>. Friction takes at most <m>4F<sub>f</sub>X</m> a cycle, "
        "which grows only as <m>X</m>, while the force puts in <m>πFX</m>: since <m>F</m> = 1 N &gt; "
        "<m>4F<sub>f</sub>/π</m> = 0.4 N, the swing grows and grows, where the viscous ghost settles at 25 mm. Set "
        "the force below 0.4 N and friction holds it.",
        model=Model.COULOMB,
        drive=True,
        tune=True,
        tab=TAB_ENERGY,
        predict=Prediction(
            "1 N at resonance, friction matched to the viscous damper at 10 mm. Where does friction's swing stop?",
            ("At 25 mm, as viscous does", "Lower: friction holds it", "It doesn't: it runs away"),
            2,
            why=(
                "That's where viscous damping's loss, growing as X², catches the force's input. Friction's grows only as X, as the input does: it never catches up.",
                "Friction holds the mass only under a force of 4F<sub>f</sub>/π = 0.4 N. At 1 N it can't.",
                "",
            ),
        ),
    ),
    SdofTrial(
        "Released against friction",
        "Plucked from 10 mm, beside the viscous ghost. The ghost's swing falls by the same fraction each cycle, "
        "<m>e<sup>−δ</sup></m> = 0.73, for ever. Friction's falls by the same amount, <m>4F<sub>f</sub>/k</m> = 3.1 mm, "
        "and stops dead after three cycles, 0.58 mm off center, where the spring can no longer beat the friction.",
        model=Model.COULOMB,
        x0_mm=10.0,
        tab=TAB_LOOP,
        predict=Prediction(
            "Released from 10 mm against friction. How does its swing die away?",
            (
                "By the same fraction each cycle, for ever",
                "By the same amount each cycle, then it stops",
                "It doesn't die away",
            ),
            1,
            why=(
                "That's viscous damping, whose force shrinks with the motion. Friction's doesn't: it takes the same 4F<sub>f</sub>/k off each cycle.",
                "",
                "Friction takes 4F<sub>f</sub>/k = 3.1 mm off every cycle, until the spring can no longer beat it.",
            ),
        ),
    ),
    SdofTrial(
        "A viscoelastic loss peak",
        "The Zener element: a spring <m>k<sub>0</sub></m> beside a spring <m>k<sub>1</sub></m> and a dashpot in series. "
        "Slowly the dashpot gives and only <m>k<sub>0</sub></m> = 364 N/m holds; fast it locks and "
        "<m>k<sub>0</sub> + k<sub>1</sub></m> = 444 N/m does. In between, at its relaxation frequency (here "
        "<m>f<sub>n</sub></m>), the loss factor peaks at 0.1 (the lower plot). Plucked, it rings down beside the "
        "viscous ghost almost as one: matched at <m>f<sub>n</sub></m>, they lose the same there. Rubber mounts and "
        "polymers behave like this.",
        model=Model.ZENER,
        x0_mm=10.0,
        tab=TAB_LOOP,
        predict=Prediction(
            "A spring beside a spring and dashpot in series (Zener). Where is its loss factor largest?",
            ("At low frequency", "At its relaxation frequency", "It grows for ever with frequency"),
            1,
            why=(
                "Slowly, the dashpot gives and only the spring k<sub>0</sub> works: nothing is lost.",
                "",
                "Fast, the dashpot locks and the two springs work as one: nothing is lost then either. The loss peaks in between.",
            ),
        ),
    ),
    SdofTrial(
        "Relaxing well above resonance",
        "The Zener element's peak moved to <m>10f<sub>n</sub></m>. Below it, its loss grows in proportion to "
        "frequency, as a viscous damper's does: through the oscillator's band it is a viscous damper. It needs a "
        "stiffer arm for the same damping at <m>f<sub>n</sub></m>.",
        model=Model.ZENER,
        relax=10.0,
        x0_mm=10.0,
        tab=TAB_LOOP,
    ),
    SdofTrial(
        "Hysteretic damping's stand-in, swept",
        "Hysteretic damping (<m>η</m> = 0.1) has no model in time that respects cause and effect, so the page runs "
        "its stand-in: a viscous damper with the same loss at the frequency it runs at, <m>c = ηk/Ω</m>, changed "
        "as the drive's frequency is. Swept, its points land on the hysteretic curve: over the viscous ghost's above "
        "<m>f<sub>n</sub></m>, where viscous damping's loss grows with the frequency, and under it below.",
        model=Model.HYSTERETIC,
        drive=True,
        sweep=True,
        tab=TAB_FRF,
    ),
)


# ------------------------------------------------------------------ practice problems
# Each question's numbers are printed round, and its answer is worked from them as printed.


def _viscous_energy(rng: np.random.Generator) -> Question:
    c = pick(rng, (0.5, 1, 2, 5, 10))
    f = pick(rng, (2, 5, 10, 20, 50))
    x = pick(rng, (1, 2, 5, 10))
    energy = math.pi * c * TWO_PI * f * (1e-3 * x) ** 2
    return Question(
        f"A viscous damper of {sig(c, 6)} N·s/m is cycled at {sig(f, 6)} Hz through ±{sig(x, 6)} mm. How much energy does it take "
        "out per cycle?",
        1e3 * energy,
        f"W<sub>d</sub> = πcΩX² with Ω = 2π × {sig(f, 6)} = {number(TWO_PI * f)} rad/s and X = {number(1e-3 * x)} m: "
        f"π × {sig(c, 6)} × {number(TWO_PI * f)} × {number(1e-3 * x)}² = {number(1e3 * energy)} mJ.",
        unit="mJ",
    )


def _friction_from_decay(rng: np.random.Generator) -> Question:
    k = pick(rng, (200, 400, 1000, 2500))
    drop = pick(rng, (0.5, 1, 2, 3, 4))
    friction = k * 1e-3 * drop / 4.0
    return Question(
        f"A mass on a spring of {sig(k, 6)} N/m rings down in a straight line, its swing falling by {sig(drop, 6)} mm every "
        "cycle. What friction force is it sliding against?",
        friction,
        f"Each half cycle swings about the point F<sub>f</sub>/k off center on the side friction pushes to, so the "
        f"swing loses 2F<sub>f</sub>/k a half cycle, 4F<sub>f</sub>/k a cycle: F<sub>f</sub> = k × drop/4 = "
        f"{sig(k, 6)} × {number(1e-3 * drop)}/4 = {number(friction)} N.",
        unit="N",
        setup=dict(model=Model.COULOMB, stiffness=k, x0_mm=10.0, tab=TAB_LOOP),
    )


def _friction_swings(rng: np.random.Generator) -> Question:
    k = pick(rng, (200, 400, 1000))
    friction = pick(rng, (0.2, 0.5, 1, 2))
    d = 1e3 * friction / k  # mm
    n = int(pick(rng, (2, 3, 4, 5, 6)))
    x0 = rounded(d * (2 * n + pick(rng, (-0.6, -0.3, 0.3, 0.6))), 2)
    swings, _ = coulomb_stop(k, friction, 1e-3 * x0)
    return Question(
        f"A mass on a spring of {sig(k, 6)} N/m slides against a friction force of {sig(friction, 6)} N. Released from rest "
        f"{number(x0, 2)} mm from center, how many half cycles (swings from one turn to the next) does it make "
        "before it sticks?",
        swings,
        f"F<sub>f</sub>/k = {number(d)} mm. Each swing takes 2F<sub>f</sub>/k = {number(2 * d)} mm off the distance "
        f"from center, and it sticks at the first turn within {number(d)} mm of it: {number(x0, 2)} mm takes "
        f"{swings} swings.",
        tolerance=0.01,
    )


def _equivalent_viscous(rng: np.random.Generator) -> Question:
    friction = pick(rng, (0.5, 1, 2, 5))
    f = pick(rng, (2, 5, 10, 20))
    x = pick(rng, (1, 2, 5, 10))
    w = TWO_PI * f
    c = 4.0 * friction / (math.pi * w * 1e-3 * x)
    return Question(
        f"A friction force of {sig(friction, 6)} N is cycled at {sig(f, 6)} Hz through ±{sig(x, 6)} mm. What viscous damper takes the "
        "same energy per cycle?",
        c,
        f"Friction takes 4F<sub>f</sub>X, a viscous damper πcΩX²; equal when c<sub>eq</sub> = 4F<sub>f</sub>/πΩX = "
        f"4 × {sig(friction, 6)}/(π × {number(w)} × {number(1e-3 * x)}) = {number(c)} N·s/m. Half the amplitude, twice "
        "the c<sub>eq</sub>: friction is heavy damping for small motions.",
        unit="N·s/m",
    )


def _zener_peak(rng: np.random.Generator) -> Question:
    k0 = pick(rng, (100, 200, 400, 1000))
    a = pick(rng, (0.1, 0.25, 0.5, 1, 2, 3))
    k1 = rounded(a * k0)
    a = k1 / k0
    eta = a / (2.0 * math.sqrt(1.0 + a))
    return Question(
        f"A rubber mount behaves as a Zener element: {sig(k0, 6)} N/m slowly (relaxed), {number(k0 + k1)} N/m fast "
        "(unrelaxed). What is its loss factor at its peak?",
        eta,
        f"k<sub>1</sub> = {number(k1)} N/m, so a = k<sub>1</sub>/k<sub>0</sub> = {number(a)}. η(Ωτ<sub>r</sub>) = "
        f"aΩτ<sub>r</sub>/(1 + (1 + a)Ω²τ<sub>r</sub>²) peaks at Ωτ<sub>r</sub> = 1/√(1 + a), where η = a/(2√(1 + a)) = {number(eta)}. The "
        "further apart the relaxed and unrelaxed stiffnesses, the more it can damp.",
        setup=dict(model=Model.ZENER, x0_mm=10.0, tab=TAB_LOOP),
    )


def _frequency_doubled(rng: np.random.Generator) -> Question:
    return Question(
        "Four dampers, viscous, hysteretic, Zener (its loss peak at f<sub>n</sub>) and friction, each take the "
        "same energy per cycle at f<sub>n</sub>. Cycled through the same amplitude at 2f<sub>n</sub>, which takes "
        "the most?",
        0,
        "Viscous: its force grows with the velocity, so it takes twice as much. Hysteretic and friction take the "
        "same as before, and the Zener element, past its peak, a little less.",
        choices=("Viscous", "Hysteretic", "Friction", "Zener"),
        why=(
            "",
            "Hysteretic damping takes the same energy per cycle at any frequency: its force is sized by the "
            "displacement, not the velocity.",
            "Friction's force is the same at any speed, so over the same stroke it takes 4F<sub>f</sub>X a cycle at "
            "any frequency.",
            "Past its loss peak at f<sub>n</sub>, the Zener element's loss falls a little; the viscous damper's "
            "doubles.",
        ),
        setup=dict(drive=True, freq_hz=6.366, amplitude=12.0, tab=TAB_LOOP),
        predict=True,
    )


def _amplitude_halved(rng: np.random.Generator) -> Question:
    return Question(
        "A friction damper is cycled through half the amplitude it was, at the same frequency. What happens to "
        "the energy it takes per cycle?",
        0,
        "It halves: W<sub>d</sub> = 4F<sub>f</sub>X, the same force over half the stroke. A viscous or hysteretic "
        "damper's (∝ X²) falls to a quarter; so friction is relatively heavier damping for small motions.",
        choices=("It halves", "It falls to a quarter", "It stays the same", "It doubles"),
        why=(
            "",
            "That's a viscous or hysteretic damper, whose force grows with the motion (∝ X²). Friction's force "
            "stays the same, over half the stroke.",
            "The force is the same, but it acts over half the distance: 4F<sub>f</sub>X halves with X.",
            "Less motion can't take more energy: the same friction force acts over half the stroke.",
        ),
        setup=dict(model=Model.COULOMB, friction_mm=5.0, x0_mm=5.0, tab=TAB_LOOP),
        predict=True,
    )


def _friction_resonance(rng: np.random.Generator) -> Question:
    return Question(
        "A mass on a spring with friction (no other damping) is driven by a harmonic force at its natural "
        "frequency, the force larger than 4/π times the friction. What becomes of its swing?",
        0,
        "It grows without bound (until something else stops it). At resonance the force puts in πFX a cycle and "
        "friction takes 4F<sub>f</sub>X: both grow as X, so if F &gt; 4F<sub>f</sub>/π the input wins at every "
        "amplitude.",
        choices=("It grows without bound", "It settles where friction balances the force", "It stays stuck"),
        why=(
            "",
            "A balance needs the loss to grow faster than the input. At resonance both grow as X (πFX in, "
            "4F<sub>f</sub>X out), so past 4/π times the friction no swing balances.",
            "It sticks only while the force can't beat the friction. This force is more than 4/π times it, so the "
            "mass slips and the swing builds.",
        ),
        setup=dict(model=Model.COULOMB, drive=True, tune=True, amplitude=1.0, tab=TAB_FRF),
        predict=True,
    )


def _friction_spec(rng: np.random.Generator) -> Question:
    k = pick(rng, (200, 400, 1000))
    x0 = pick(rng, (5, 10, 20))
    swings = int(pick(rng, (3, 4, 5)))
    # friction that stops it after `swings` half cycles with room either side
    d = x0 / (2 * swings + 0.5)  # mm
    wn = math.sqrt(k)  # on 1 kg
    c = rounded(4.0 * k * d / (math.pi * wn * x0), 3)  # F_f/k = d (mm), F_f = πcω_nX/4
    design = Matched(1.0, k, c / (2.0 * wn), 1e-3 * x0)
    n_d, end = coulomb_stop(k, design.friction, 1e-3 * x0)
    off = rounded(1.25 * 1e3 * abs(end), 2)

    def goal(t: SdofTrial):
        m = matched(t)
        n, stop = coulomb_stop(t.stiffness, m.friction, 1e-3 * t.friction_mm)
        return (
            kept("The mass", t.mass, 1.0, "kg"),
            kept("The spring", t.stiffness, k, "N/m"),
            kept("The release, X", t.friction_mm, float(x0), "mm"),
            at_most("Swings before it sticks", n, swings),
            at_most("Where it sticks, off center", 1e3 * abs(stop), off, "mm"),
        )

    return Question(
        f"A 1 kg mass on a spring of {sig(k, 6)} N/m, released from {sig(x0, 6)} mm, is to be stopped by friction alone (the "
        f"Coulomb friction law, plucked from X). It must stick within {swings} swings (half cycles), no more than "
        f"{number(off, 2)} mm off center. The page sets the friction by matching it to the damper c at the "
        "amplitude X: choose c.",
        0,
        f"More friction stops it sooner, but further off center (anywhere within F<sub>f</sub>/k), and too much "
        f"never lets it move. Each swing takes 2F<sub>f</sub>/k off, so {swings} swings need about "
        f"2F<sub>f</sub>/k ≈ {sig(x0, 6)}/{swings} mm. F<sub>f</sub> = πcω<sub>n</sub>X/4 with ω<sub>n</sub> = "
        f"{number(wn)} rad/s: c = {number(c)} N·s/m gives F<sub>f</sub> = {number(design.friction)} N, {n_d} swings, "
        f"stuck {number(1e3 * abs(end))} mm off center. Much more friction does too: with F<sub>f</sub>/k just under "
        "X/2 it makes one swing and sticks near the center.",
        setup=dict(
            model=Model.COULOMB,
            stiffness=k,
            damping=rounded(0.01 * wn, 3),
            friction_mm=float(x0),
            x0_mm=float(x0),
            tab=TAB_LOOP,
        ),
        goal=goal,
        design=dict(model=Model.COULOMB, stiffness=k, damping=c, friction_mm=float(x0), x0_mm=float(x0), tab=TAB_LOOP),
    )


PROBLEMS = tuple(
    Problem(key, title, "sdof_page", f"See {page_link('sdof_page', anchor, section)}{more}.", make)
    for key, title, anchor, section, more, make in (
        ("viscous-energy", "Energy per cycle", "viscous", "Viscous damping", "", _viscous_energy),
        ("friction-decay", "Friction from a decay", "coulomb", "Coulomb friction", "", _friction_from_decay),
        ("friction-swings", "When friction stops it", "coulomb", "Coulomb friction", "", _friction_swings),
        (
            "equivalent-viscous",
            "Equivalent viscous damping",
            "coulomb",
            "Coulomb friction",
            ": equal energies per cycle",
            _equivalent_viscous,
        ),
        ("zener-peak", "A rubber mount's loss peak", "viscoelastic", "The Zener element", "", _zener_peak),
        (
            "frequency-doubled",
            "Predict: twice the frequency",
            "loss-factor",
            "The loss factor, compared",
            "",
            _frequency_doubled,
        ),
        ("amplitude-halved", "Predict: half the amplitude", "coulomb", "Coulomb friction", "", _amplitude_halved),
        (
            "friction-resonance",
            "Predict: friction at resonance",
            "coulomb",
            "Coulomb friction",
            "",
            _friction_resonance,
        ),
        (
            "friction-spec",
            "Spec: stop it with friction",
            "coulomb",
            "Coulomb friction",
            ": how far does each swing fall, and where can it stick?",
            _friction_spec,
        ),
    )
)

TIPS = {
    "model": "<p>The damper's law, matched at f<sub>n</sub> to the damper c: viscous, c itself; hysteretic (by its "
    "stand-in in time, c = ηk/Ω at the frequency it runs at); Coulomb friction (matched at the amplitude X); or "
    "the viscoelastic Zener element. Change it as it runs.</p>",
    "friction_mm": "<p>The amplitude X where friction is matched to c: F<sub>f</sub> = πcω<sub>n</sub>X/4.</p>",
    "relax": "<p>Where the Zener element's loss factor peaks, as a multiple of f<sub>n</sub>.</p>",
    "ghost": "<p>A viscous mass with the damper c, plucked and driven as this one is: drawn in gray beside it (a gray line on the time plot, a dashed outline on the mass, a dashed curve on the Frequency response).</p>",
    "damping_only": "<p>Ticked, the loops are the damper's force alone. Untick it to add the spring's force kx, "
    "as a test machine measures a part: the loop then leans along the spring's dashed line, the lean the "
    "stiffness and the width the damping.</p>",
}
