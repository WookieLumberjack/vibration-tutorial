"""Pendulums' Things to try, practice problems, tips and notes, without Qt (#478 Phase 4, page standard item 7).

Its state is `view_models/pendulum.py`'s `PendulumTrial`; `gui/pendulum.py` binds the page's controls to it and
runs it live. A trial is a setup to go back to: it sets the pendulum, plucks it from `angle` (or leaves it at
rest), and says what to look for. Its prediction is asked only once the reader has handled the pendulum
(`lesson.TrialBox(ask_after_play=True)`), about a case they haven't tried.

Working on it: `tests/test_tutorial_numbers.py` checks the trials' numbers, `tests/test_practice.py` the problems,
`tests/test_qt_free.py` that this loads no Qt.
"""

from __future__ import annotations

import math

import numpy as np

from ..core.pendulum import Kind, Pendulum, conjugate_pivot, period
from ..core.practice import Placement, Prediction, Problem, Question, Requirement, kept, number, pick, rounded
from ..view_models.pendulum import TAB_ENERGY, TAB_PERIOD, TAB_PHASE, PendulumTrial
from ..view_models.text import number as sig
from . import pendulum_figures
from .links import load_notes, page_link
from .trials_text import trials_html

TRIALS = (
    PendulumTrial(
        "A quarter of the string",
        "A 1 kg bob on a 0.25 m string, plucked from 10°. Gravity is its spring: the restoring moment is "
        "<m>mgL sin θ ≈ mgLθ</m> and the moment of inertia <m>mL²</m>, so <m>ω<sub>n</sub> = √(g/L)</m>, and a "
        "quarter of the length halves the period: 1.003 s for small swings, where the 1 m string took 2.006 s. "
        "Each swing lands its period on the Period tab, 1.005 s at 10°. The mass isn't in it: drag the ring on the "
        "bob up for a heavier one, and the swing carries on at the same pace.",
        length=0.25,
        angle=10.0,
        predict=Placement(
            "The same bob on a string a quarter as long, 0.25 m, plucked from 10°. Where will its swings' periods "
            "land? Put the line at the period.",
            plot="period_plot",
            answer=1.005,
            span=(0.0, 3.0),
            axis="y",
            low="Not a quarter: the period goes as √L, so a quarter of the string halves it, to 1.0 s.",
            high="A shorter string swings faster: the period goes as √L, so a quarter of it halves the 2.0 s, to 1.0 s.",
            tolerance=0.15,
            unit="s",
        ),
    ),
    PendulumTrial(
        "Let go from 90°",
        "The 1 m pendulum from 90°. <m>sin θ</m> falls below <m>θ</m>, so the spring softens as the swing grows: "
        "the period is 2.368 s, 18 % longer than the small swing's, and the linearized ghost drifts ahead. Its "
        "point lands high on the Period tab; show the theory for the exact curve (an elliptic integral) and "
        "Duffing's first approximation, <m>T ≈ T<sub>0</sub>(1 + A²/16)</m>, good to 1 % up to about 75°.",
        angle=90.0,
        predict=Prediction(
            "The 1 m pendulum let go from 90° instead of 10°. Its period is…",
            ("shorter", "the same", "longer"),
            2,
            ordered=True,
            why=(
                "sin θ falls below θ as the swing grows: gravity pulls back less than the linear spring would, so "
                "the period lengthens.",
                "Only while sin θ ≈ θ, up to about 10°. From 90° the period is 18 % longer.",
                "",
            ),
        ),
    ),
    PendulumTrial(
        "Nearly over the top, dying away",
        "From 175°, with a little damping (<m>ζ</m> = 0.003). The bob nearly balances upside down and lingers "
        "there, and so near the top even a little damping takes a lot: its first whole swing is down to about 150° "
        "and takes 3.5 s, 1.76 times the small swing's 2.006 s. Then each swing lands its period on the Period tab "
        "as it dies away, and the points trace the period's curve down towards <m>T<sub>0</sub></m> by themselves "
        "(speed it up to see more). The Phase plane tab shows why it lingers: the path starts close to the "
        "separatrix, the orbit through the upside-down balance.",
        angle=175.0,
        zeta=0.003,
        tab=TAB_PERIOD,
    ),
    PendulumTrial(
        "Pump the swing",
        "The 1 m pendulum swinging from 10°. Slide the ring on the string in as the bob sweeps through the bottom, "
        "and out again at each end of its swing. Shortened, it swings faster at once: nothing turns it about the "
        "pivot, so its angular momentum <m>Jθ̇</m> is kept as <m>J = mL²</m> shrinks. You pull in against the "
        "tension, greatest at the bottom, and let out against the least, at the ends, so each pump leaves energy "
        "in the swing: the Energy tab books it as <i>Changing the pendulum</i>, and the swing grows. A child on a "
        "swing pumps it so, standing up at the bottom and crouching at the ends.",
        angle=10.0,
        tab=TAB_ENERGY,
        predict=Prediction(
            "The string pulled in to half its length as the bob passes the bottom of its swing. The bob…",
            ("slows down", "keeps its speed", "speeds up"),
            2,
            ordered=True,
            why=(
                "Pulling the string in has no moment about the pivot, so the angular momentum mL²θ̇ is kept. With "
                "L halved, θ̇ goes up four times and the bob's speed Lθ̇ doubles.",
                "Its angular momentum mL²θ̇ is kept, not its speed: half the length, twice the speed.",
                "",
            ),
        ),
    ),
    PendulumTrial(
        "A rod: where to pivot it",
        "A 1 m uniform rod pivoted at its end, 0.5 m from its centre: <m>J = mL²/3</m> and the weight's arm "
        "<m>d</m> = 0.5 m, so it swings as a simple pendulum <m>L<sub>eq</sub></m> = 0.667 m long, in 1.638 s. "
        "Slide the ring on its pivot towards the middle as it swings: 0.289 m from the centre (<m>L/√12</m>, the "
        "radius of gyration) it swings fastest, 1.524 s; 0.167 m from the centre (<m>k<sub>G</sub>²/d</m>) it "
        "swings at 1.638 s again. That's Kater's reversible pendulum, which measured <m>g</m>.",
        kind=Kind.ROD,
        angle=10.0,
        tab=TAB_PERIOD,
        predict=Prediction(
            "A 1 m rod swinging from one end, or a bob on a 1 m string: which swings faster?",
            ("The rod", "The bob on the string", "Neither: the same"),
            0,
            why=(
                "",
                "The rod's mass is spread along it, some of it near the pivot: it swings as a pendulum 2L/3 = "
                "0.667 m long, so faster.",
                "Only a point mass at 1 m swings as a 1 m pendulum. The rod's mass is spread along it: it swings as "
                "one 0.667 m long.",
            ),
        ),
    ),
    PendulumTrial(
        "A disc on a wire, twisted 120°",
        "A 1 kg disc of radius 0.1 m on a wire of torsional stiffness <m>k<sub>t</sub></m> = 0.5 N·m/rad: "
        "<m>J = mr²/2</m>, <m>f<sub>n</sub> = √(k<sub>t</sub>/J)/2π</m> = 1.59 Hz. The wire's twist is linear, "
        "so its point on the Period tab lands at the small twist's 0.628 s even at 120°: the softening came from "
        "gravity's <m>sin θ</m>, not from turning. Slide the ring on its rim outwards as it turns, as a skater "
        "spreads their arms: twice the radius is four times <m>J</m>, and it turns a quarter as fast at once.",
        kind=Kind.TORSION,
        length=0.1,
        spring=0.5,
        angle=120.0,
        predict=Prediction(
            "The disc twisted 120° instead of 10° and let go. Its period is…",
            ("longer", "the same", "shorter"),
            1,
            ordered=True,
            why=(
                "A pendulum softens because of gravity's sin θ. The wire's torque is k<sub>t</sub>θ at any twist, "
                "so nothing softens.",
                "",
                "Nothing hardens either: the wire's torque is k<sub>t</sub>θ at any twist, and a linear spring's "
                "period doesn't depend on the amplitude.",
            ),
        ),
    ),
    PendulumTrial(
        "Inverted, let go from 60°",
        "A 1 kg mass on a 1 m arm balanced upright on a 15 N·m/rad spring. Gravity now pushes it over: the "
        "stiffness is <m>k<sub>θ</sub> = k<sub>t</sub> − mgL</m> = 5.19 N·m/rad, <m>f<sub>n</sub></m> = 0.363 Hz. "
        "And it hardens: from 60° the period is 10 % shorter than the small swing's.",
        kind=Kind.INVERTED,
        spring=15.0,
        angle=60.0,
        predict=Prediction(
            "Upright on its spring, let go from 60° rather than a few degrees. Its period is…",
            ("longer, as a hanging pendulum's", "the same", "shorter"),
            2,
            ordered=True,
            why=(
                "Hanging, gravity pulls back and sin θ falling below θ softens it. Upright, gravity pushes over, "
                "and the same shortfall leaves the spring more to win by: it hardens.",
                "Gravity's moment, mgL sin θ, isn't linear: at 60° it's 17 % short of mgLθ, so the spring wins by "
                "more and the swing is quicker.",
                "",
            ),
        ),
    ),
    PendulumTrial(
        "Inverted, on a weaker spring",
        "The spring at 9 N·m/rad, under <m>mgL</m> = 9.81 N·m/rad: the stiffness is negative and upright is "
        "unstable. Let go from 1° with a little damping, it falls over and settles leaning at 40.8°, where the "
        "spring's <m>9θ</m> balances the weight's <m>9.81 sin θ</m>; its linearized ghost just runs away. A column "
        "under a load past its buckling load does the same. Drag the ring on the spring up past 9.81 N·m/rad and "
        "it stands up again.",
        kind=Kind.INVERTED,
        spring=9.0,
        angle=1.0,
        zeta=0.1,
        tab=TAB_PHASE,
        predict=Prediction(
            "A spring of 9 N·m/rad under a weight's 9.81 N·m/rad, let go from 1° with a little damping. It…",
            ("swings about upright", "falls over and settles leaning part-way", "falls flat, to 90°"),
            1,
            why=(
                "Upright needs the spring to beat the weight, 9 against 9.81 N·m/rad. It doesn't, so upright is "
                "unstable.",
                "",
                "As it leans, the weight's moment, 9.81 sin θ, grows more slowly than the spring's 9θ: they "
                "balance at 40.8°, short of 90°.",
            ),
        ),
    ),
)

# ------------------------------------------------------------------ practice problems
# Each question's numbers are printed round, and its answer is worked from them as printed.


def _simple_period(rng: np.random.Generator) -> Question:
    length = pick(rng, (0.25, 0.5, 1, 2, 5, 10))
    p = Pendulum(Kind.SIMPLE, length=length)
    return Question(
        f"A small bob hangs on a {sig(length, 6)} m string (g = 9.81 m/s²). What is its period for small swings?",
        p.period0,
        f"Gravity is the spring: k<sub>θ</sub> = mgL, J = mL², so ω<sub>n</sub> = √(g/L) = "
        f"{number(p.omega0)} rad/s and T = 2π/ω<sub>n</sub> = {number(p.period0)} s, whatever the bob weighs.",
        unit="s",
        setup=dict(kind=Kind.SIMPLE, length=length, angle=5.0),
    )


def _rod_period(rng: np.random.Generator) -> Question:
    length = pick(rng, (0.5, 1, 1.5, 2))
    d = rounded(pick(rng, (0.1, 0.2, 0.3, 0.4, 0.5)) * length)
    p = Pendulum(Kind.ROD, length=length, pivot=d)
    return Question(
        f"A uniform rod {sig(length, 6)} m long swings about a pivot {number(d)} m from its centre. What is its period "
        "for small swings?",
        p.period0,
        f"J = m(L²/12 + d²) about the pivot (parallel axes) and k<sub>θ</sub> = mgd: T = 2π√((L²/12 + d²)/gd) = "
        f"2π√(({number(length**2 / 12)} + {number(d * d)})/(9.81 × {number(d)})) = {number(p.period0)} s, the mass "
        f"cancelling again. It swings like a simple pendulum {number(p.equivalent_length)} m long.",
        unit="s",
        setup=dict(kind=Kind.ROD, length=length, pivot=d, angle=10.0, tab=TAB_PERIOD),
    )


def _large_swing(rng: np.random.Generator) -> Question:
    degrees = pick(rng, (20, 30, 45, 60, 90, 120))
    p = Pendulum(Kind.SIMPLE)
    longer = 100.0 * (period(p, math.radians(degrees)) / p.period0 - 1.0)
    a = math.radians(degrees)
    duffing = 100 * (1 + a * a / 16) * p.period0 / period(p, a) - 100
    return Question(
        f"A simple pendulum is let go from {sig(degrees, 6)}° instead of a small angle. By what percentage is its period "
        "longer?",
        longer,
        f"Exactly, T = T<sub>0</sub>(2/π)K(sin²(A/2)), the energy integral: {number(longer)} % longer. Duffing's "
        f"estimate, T ≈ T<sub>0</sub>(1 + A²/16), says {number(100 * a * a / 16)} %, the period itself "
        f"{number(abs(duffing), 2)} % {'short' if duffing < 0 else 'long'}: it keeps the period within 0.14 % up "
        "to 45° and 1 % up to about 75°, though the lengthening it gives is off by more.",
        unit="%",
        tolerance=0.03,
        setup=dict(kind=Kind.SIMPLE, angle=float(degrees), tab=TAB_PERIOD),
    )


def _inverted_spring(rng: np.random.Generator) -> Question:
    m = pick(rng, (0.5, 1, 2, 5))
    length = pick(rng, (0.2, 0.5, 1))
    least = m * 9.81 * length
    return Question(
        f"A {sig(m, 6)} kg mass on a light arm {sig(length, 6)} m long stands upright on a pivot, held up by a torsion "
        "spring. How stiff must the spring be for upright to be stable?",
        least,
        f"Leaning by θ, the weight pushes it over with mgL sin θ ≈ mgLθ, so the stiffness is k<sub>t</sub> − mgL: "
        f"stable only if k<sub>t</sub> &gt; mgL = {sig(m, 6)} × 9.81 × {sig(length, 6)} = {number(least)} N·m/rad. A "
        "weaker spring lets it fall to a leaning balance, as a column buckles.",
        unit="N·m/rad",
        setup=dict(kind=Kind.INVERTED, mass=m, length=length, spring=rounded(1.2 * least), angle=10.0),
    )


def _disc_on_wire(rng: np.random.Generator) -> Question:
    m = pick(rng, (0.5, 1, 2, 5))
    r = pick(rng, (0.05, 0.1, 0.2))
    k = pick(rng, (0.5, 1, 2, 5))
    p = Pendulum(Kind.TORSION, mass=m, length=r, spring=k)
    return Question(
        f"A {sig(m, 6)} kg disc of radius {sig(r, 6)} m hangs on a wire of torsional stiffness {sig(k, 6)} N·m/rad. At what "
        "frequency does it twist back and forth?",
        p.natural_hz,
        f"J = mr²/2 = {number(p.inertia)} kg·m², so ω<sub>n</sub> = √(k<sub>t</sub>/J) = {number(p.omega0)} rad/s "
        f"and f<sub>n</sub> = {number(p.natural_hz)} Hz: the single oscillator, turned.",
        unit="Hz",
        setup=dict(kind=Kind.TORSION, mass=m, length=r, spring=k, angle=30.0),
    )


def _end_pivot(rng: np.random.Generator) -> Question:
    length = pick(rng, (0.3, 0.6, 0.9, 1.2, 1.5))
    p = Pendulum(Kind.ROD, length=length, pivot=length / 2)
    return Question(
        f"A uniform rod {sig(length, 6)} m long swings about one end. How long a simple pendulum has the same period?",
        p.equivalent_length,
        f"L<sub>eq</sub> = J/md with J = m(L²/12 + L²/4) = mL²/3 and d = L/2: L<sub>eq</sub> = 2L/3 = "
        f"{number(p.equivalent_length)} m.",
        unit="m",
        setup=dict(kind=Kind.ROD, length=length, pivot=length / 2, angle=10.0, tab=TAB_PERIOD),
    )


def _heavier_bob(rng: np.random.Generator) -> Question:
    n = int(pick(rng, (2, 4, 10)))
    return Question(
        f"A simple pendulum's bob is swapped for one {n} times heavier, on the same string. What happens to its "
        "period?",
        0,
        "Nothing: the restoring moment mgL and the inertia mL² grow together, so ω<sub>n</sub> = √(g/L) has no "
        "mass in it.",
        choices=("It stays the same", f"It grows √{n} times", f"It falls to 1/√{n}"),
        why=(
            "",
            "That's a mass on a spring, whose spring stays the same as the mass grows. A pendulum's spring is the "
            "bob's own weight, which grows with it.",
            "The heavier bob is pulled back harder, but it is harder to move in the same proportion: the mass "
            "cancels out of g/L.",
        ),
        setup=dict(kind=Kind.SIMPLE, mass=float(n), angle=10.0),
        predict=True,
    )


def _on_the_moon(rng: np.random.Generator) -> Question:
    return Question(
        "A pendulum clock is taken to the Moon, where gravity is about a sixth of Earth's. What happens to its "
        "pendulum's period?",
        0,
        "It grows √6 ≈ 2.4 times: gravity is the spring, ω<sub>n</sub> = √(g/L). (A disc on a torsion wire "
        "wouldn't change: its spring isn't gravity.)",
        choices=("It grows about 2.4 times", "It grows 6 times", "It stays the same", "It falls to a sixth"),
        why=(
            "",
            "The period goes as 1/√g, not as 1/g: a sixth of the gravity gives √6 ≈ 2.4 times the period.",
            "That holds for a mass on a spring or a disc on a wire, whose springs aren't gravity; a pendulum's "
            "spring is gravity.",
            "Weaker gravity pulls the bob back more gently, so it swings more slowly, not faster.",
        ),
        setup=dict(kind=Kind.SIMPLE, gravity=1.62, angle=10.0),
        predict=True,
    )


def _wire_large_twist(rng: np.random.Generator) -> Question:
    return Question(
        "A disc on a torsion wire is twisted to 90° and let go, instead of 10°. What happens to its period?",
        0,
        "Nothing: a wire's moment is k<sub>t</sub>θ at any twist, so it is linear and its period doesn't depend "
        "on the swing. Only gravity's sin θ makes a pendulum's lengthen.",
        choices=("It stays the same", "It lengthens, as a pendulum's does", "It shortens"),
        why=(
            "",
            "A pendulum's lengthens because its restoring moment, mgL sin θ, grows more slowly than θ at a large "
            "swing. A wire's, k<sub>t</sub>θ, keeps in proportion at any twist.",
            "That would take a spring that stiffens as it's twisted; a wire's moment stays in proportion to the "
            "twist (until it yields).",
        ),
        setup=dict(kind=Kind.TORSION, angle=90.0, tab=TAB_PERIOD),
        predict=True,
    )


def _pivot_spec(rng: np.random.Generator) -> Question:
    length = pick(rng, (0.5, 1, 1.5, 2))
    d = rounded(pick(rng, (0.12, 0.18, 0.4, 0.45)) * length)
    target = rounded(Pendulum(Kind.ROD, length=length, pivot=d).period0)
    other = conjugate_pivot(length, d)

    def goal(t: PendulumTrial):
        now = Pendulum(t.kind, t.mass, t.length, max(1e-3, t.pivot), t.spring, t.zeta, t.gravity).period0
        return (
            Requirement("A rod on a pivot", t.kind.value, t.kind is Kind.ROD),
            kept("Its length", t.length, length, "m"),
            kept("Gravity", t.gravity, 9.81, "m/s²"),
            Requirement(
                f"Its period within ±0.5 % of {number(target)} s",
                f"{number(now, 4)} s",
                t.kind is Kind.ROD and abs(now / target - 1.0) <= 0.005,
            ),
            Requirement(
                f"The pivot on the rod (≤ {number(length / 2)} m from its centre)",
                f"{number(t.pivot)} m",
                t.pivot <= length / 2,
            ),
        )

    return Question(
        f"A uniform rod {sig(length, 6)} m long is to be a clock's pendulum with a period of {number(target)} s. "
        "Where along it should the pivot go?",
        0,
        f"T = 2π√((L²/12 + d²)/gd) is least at d = L/√12 = {number(length / math.sqrt(12))} m and longer either "
        f"side, so a longer period is reached at two pivots, d and k<sub>G</sub>²/d: here {number(d)} m and "
        f"{number(other)} m from the centre"
        + ("" if other <= length / 2 else " (off the rod, so only the first)")
        + ". That is Kater's reversible pendulum: the two pivots are L<sub>eq</sub> apart.",
        setup=dict(kind=Kind.ROD, length=length, pivot=rounded(length / math.sqrt(12)), angle=10.0, tab=TAB_PERIOD),
        goal=goal,
        design=dict(kind=Kind.ROD, length=length, pivot=d, angle=10.0, tab=TAB_PERIOD),
    )


PROBLEMS = tuple(
    Problem(key, title, "pendulum_page", f"See {page_link('pendulum_page', anchor, section)}{more}.", make)
    for key, title, anchor, section, more, make in (
        ("simple-period", "A simple pendulum", "gravity-stiffness", "Gravity as a spring", "", _simple_period),
        ("rod-period", "A rod on a pivot", "gravity-stiffness", "Gravity as a spring", ": parallel axes", _rod_period),
        ("large-swing", "A large swing", "large-swings", "Large swings", "", _large_swing),
        ("inverted-spring", "Hold it upright", "gravity-stiffness", "Gravity as a spring", "", _inverted_spring),
        ("disc-on-wire", "A disc on a wire", "rotation", "Turning instead of sliding", "", _disc_on_wire),
        ("end-pivot", "Equivalent length", "gravity-stiffness", "Gravity as a spring", "", _end_pivot),
        ("heavier-bob", "Predict: a heavier bob", "gravity-stiffness", "Gravity as a spring", "", _heavier_bob),
        ("on-the-moon", "Predict: on the Moon", "gravity-stiffness", "Gravity as a spring", "", _on_the_moon),
        ("wire-large-twist", "Predict: a large twist", "large-swings", "Large swings", "", _wire_large_twist),
        (
            "pivot-spec",
            "Spec: a clock's pendulum",
            "gravity-stiffness",
            "Gravity as a spring",
            ": where is T least?",
            _pivot_spec,
        ),
    )
)

TIPS = {
    "kind": "<p>Simple: a point mass on a massless string. Rod: a uniform rod on a pivot. Disc: a disc twisting on "
    "a wire (no gravity in it). Inverted: a mass on an upright arm held by a torsion spring at its foot. Changed as "
    "it swings, the new body takes over at the same angle, its angular momentum kept.</p>",
    "mass": "<p>The bob's, the rod's or the disc's mass. Drag the ring on the bob (the rod's middle, the disc's "
    "rim) up for more, as it swings: the added mass is caught at rest, the angular momentum kept.</p>",
    "length": "<p>The string's length, the rod's length, the disc's radius or the inverted arm. Slide the ring on "
    "it (the disc's on its rim) to change it as it swings: the angular momentum Jθ̇ is kept, so a shorter one "
    "swings faster at once.</p>",
    "pivot": "<p>How far from the rod's centre the pivot is (0.5 L: at its end). Slide the ring on the pivot "
    "along the rod to move it as it swings.</p>",
    "spring": "<p>The torsion spring: the wire's twist stiffness, or the inverted pendulum's spring at its foot. "
    "Drag the ring on it up for a stiffer one, as it swings.</p>",
    "gravity": "<p>g: 9.81 m/s² on Earth, 1.62 on the Moon. Changed as it swings, the angle and speed are kept "
    "and the stored energy jumps.</p>",
    "zeta": "<p>A viscous damper at the pivot, as the linearized oscillator's damping ratio.</p>",
    "theory": "<p>Draw the curves the points should lie on: the exact period (by the energy integral), Duffing's "
    "first harmonic balance, and the small swing's T₀.</p>",
}

NOTES_HTML = load_notes("pendulum", figures=pendulum_figures.FIGURES, trials=trials_html(TRIALS))
