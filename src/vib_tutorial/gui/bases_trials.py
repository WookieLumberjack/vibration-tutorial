"""Coordinates & bases' Things to try, practice problems, tips and notes, without Qt (#478 Phase 4, page standard
item 7).

Its state is `view_models/bases.py`'s `BasesTrial`; `gui/bases.py` binds the page's controls to it and runs it live.
A trial is a setup to go back to: it sets the masses, springs and arrows, plucks the masses from `pluck` (or
leaves them at rest), and says what to look for. Its prediction is asked only once the reader has handled the
masses (`lesson.TrialBox(ask_after_play=True)`), about a case they haven't tried.

Working on it: `tests/test_tutorial_numbers.py` checks the trials' numbers, `tests/test_practice.py` the problems,
`tests/test_qt_free.py` that this loads no Qt.
"""

from __future__ import annotations

import math

import numpy as np

from ..core.bases import TwoMass, rayleigh_hz
from ..core.practice import Prediction, Problem, Question, number, pick
from ..view_models.bases import TAB_COUPLING, TAB_RAIL, BasesTrial
from . import bases_figures
from .links import load_notes, page_link
from .trials_text import trials_html

R = math.sqrt(0.5)
IN_PHASE, OPPOSED = (R, R), (R, -R)


def _modes(s: TwoMass) -> tuple[tuple[float, float], tuple[float, float]]:
    """The modes' arrows, of unit length, rounded so a trial's numbers print as its note says."""
    _, P = s.modes()
    return (round(float(P[0, 0]), 6), round(float(P[1, 0]), 6)), (round(float(P[0, 1]), 6), round(float(P[1, 1]), 6))


UNEQUAL = TwoMass(m2=2.0)
STIFFER = TwoMass(m2=2.0, k2=400.0)

TRIALS = (
    BasesTrial(
        "Pluck one mass",
        "Only <m>m<sub>1</sub></m> pulled 50 mm and let go. The point traces a box on the plane, and the masses "
        "trade their swing: after 0.699 s <m>m<sub>1</sub></m> has nearly stopped and <m>m<sub>2</sub></m> swings "
        "with the energy, and it comes back every 1.398 s, <m>1/(f<sub>2</sub> − f<sub>1</sub>)</m>. The "
        "Coupling tab shows the energy pass between the coordinates <m>x<sub>1</sub></m> and "
        "<m>x<sub>2</sub></m>, through the coupling spring.",
        pluck=(0.05, 0.0),
        tab=TAB_COUPLING,
        predict=Prediction(
            "The coupling spring made much weaker, 10 N/m instead of 100, and m₁ plucked again. The energy reaches m₂…",
            ("sooner", "just as soon", "later"),
            2,
            ordered=True,
            why=(
                "A weaker coupling spring passes the energy across more slowly: the two modes' frequencies come "
                "closer, and the beat, 1/(f₂ − f₁), grows long.",
                "The beat's length is set by how far apart the two frequencies are, and a weaker coupling brings "
                "them together: 3.183 and 3.262 Hz, a beat of 12.7 s.",
                "",
            ),
        ),
    ),
    BasesTrial(
        "The arrows along the modes",
        "The same pluck, described along the modes: <m>v<sub>1</sub></m> at 45°, in phase, and "
        "<m>v<sub>2</sub></m> at 135°, opposed. Both matrices are diagonal, <m>VᵀMV</m> = 1 kg and <m>VᵀKV</m> = "
        "400 and 600 N/m on the diagonal, and each coordinate's trace is one clean sine: 3.183 Hz and 3.898 Hz. "
        "On the Coupling tab each keeps its energy, and nothing passes between them. The motion is the same; "
        "only its description changed.",
        v1=IN_PHASE,
        v2=OPPOSED,
        pluck=(0.05, 0.0),
        tab=TAB_COUPLING,
    ),
    BasesTrial(
        "Pluck it along a mode",
        "Both masses pulled 35 mm the same way: the point is let go on mode 1's line, and it stays there, "
        "swinging back and forth along it at 3.183 Hz. The masses move together, the coupling spring never "
        "stretches, and nothing beats.",
        pluck=(0.035, 0.035),
        predict=Prediction(
            "Both masses pulled 35 mm, but in opposite ways (one out, one in). Compared with pulling them the same "
            "way, it swings…",
            ("slower", "at the same rate", "faster"),
            2,
            ordered=True,
            why=(
                "Pulled opposite ways, the coupling spring is stretched by both: it adds to each mass's stiffness, "
                "so it swings faster, at 3.898 Hz.",
                "Pulled the same way the coupling spring is idle; pulled opposite ways it works on both masses: "
                "a stiffer mode, at 3.898 Hz.",
                "",
            ),
        ),
    ),
    BasesTrial(
        "Unequal masses",
        "<m>m<sub>2</sub></m> = 2 kg, the mode lines drawn and the arrows on them. Both matrices are still "
        "diagonal, but the modes are no longer at right angles: mode 1 at 69.6°, mode 2 at 169.5°, 80.1° apart. "
        "They are orthogonal through <m>M</m>, <m>v<sub>1</sub>ᵀMv<sub>2</sub> = 0</m>, not in the plane. The "
        "frequencies are 2.421 and 3.624 Hz.",
        m2=2.0,
        v1=_modes(UNEQUAL)[0],
        v2=_modes(UNEQUAL)[1],
        modes=True,
        pluck=(0.05, 0.0),
    ),
    BasesTrial(
        "Coupled through the masses",
        "<m>m<sub>2</sub></m> = 2 kg again, the arrows at 45° and 135°. With the walls' springs equal these are "
        "the springs' own directions: <m>VᵀKV</m> is diagonal, 400 and 600 N/m, but <m>VᵀMV</m> isn't, "
        "−0.5 kg off its diagonal, and the coordinates are coupled through the masses (dynamic coupling). Turn "
        "the arrows to 69.6° and 169.5° to uncouple both.",
        m2=2.0,
        v1=IN_PHASE,
        v2=OPPOSED,
        pluck=(0.05, 0.0),
    ),
    BasesTrial(
        "On a rail along x₁",
        "One arrow, along <m>x<sub>1</sub></m>, and the masses held to its line: only <m>m<sub>1</sub></m> "
        "moves, on a stiffness of 500 N/m, at 3.559 Hz. That is Rayleigh's quotient for <m>v</m> = (1, 0), and it "
        "lands on the Rail tab. Turn the rail by dragging its arrow and let each new angle land its frequency.",
        rail=True,
        pluck=(0.05, 0.0),
        tab=TAB_RAIL,
        predict=Prediction(
            "The rail turned from along x₁ to 45°, so both masses move together on it. It swings…",
            ("slower", "at the same rate", "faster"),
            0,
            ordered=True,
            why=(
                "",
                "On the rail the stiffness is vᵀKv over the mass vᵀMv, and both change with the rail's angle: at "
                "45° the coupling spring isn't stretched.",
                "Along x₁ the coupling spring is stretched by m₁ alone; at 45° the masses move together and it's "
                "idle: less stiffness for the mass, so slower, 3.183 Hz, the slowest any rail can be.",
            ),
        ),
    ),
    BasesTrial(
        "A stiffer coupling turns the modes",
        "<m>m<sub>2</sub></m> = 2 kg and the coupling spring 400 N/m, the mode lines drawn. Mode 1 has turned "
        "from 69.6° to 53.8°: a stiffer coupling spring makes the masses move more nearly together. The "
        "frequencies are 2.534 and 4.896 Hz. Drag the ring on the coupling spring and watch the mode lines "
        "turn as it runs.",
        m2=2.0,
        k2=400.0,
        modes=True,
        pluck=(0.05, 0.0),
    ),
)


# ------------------------------------------------------------------ practice problems
# Each question's numbers are printed round, and its answer is worked from them as printed.


def _symmetric_pair(rng: np.random.Generator) -> Question:
    m = pick(rng, (0.5, 1, 2, 5))
    k = pick(rng, (200, 400, 1000))
    kc = pick(rng, (20, 50, 100, 300))
    s = TwoMass(m, m, k, kc, k)
    hz, _ = s.modes()
    return Question(
        f"Two {number(m)} kg masses are each tied to a wall by a {number(k)} N/m spring and to each other by a "
        f"{number(kc)} N/m spring. At what frequency do they swing in opposite directions?",
        float(hz[1]),
        f"Opposed, each mass stretches the coupling spring by twice its own motion: its stiffness is k + 2k<sub>c</sub> "
        f"= {number(k + 2 * kc)} N/m, so f₂ = √({number(k + 2 * kc)}/{number(m)})/2π = {number(float(hz[1]))} Hz. "
        f"In phase the coupling spring is idle: f₁ = √(k/m)/2π = {number(float(hz[0]))} Hz.",
        unit="Hz",
        setup=dict(m1=m, m2=m, k1=k, k2=kc, k3=k, pluck=(0.0, 0.0), v1=IN_PHASE, v2=OPPOSED, modes=True),
    )


def _beat_time(rng: np.random.Generator) -> Question:
    k = pick(rng, (200, 400, 1000))
    kc = pick(rng, (10, 20, 50, 100))
    s = TwoMass(1.0, 1.0, k, kc, k)
    hz, _ = s.modes()
    half = 1.0 / (2.0 * float(hz[1] - hz[0]))
    return Question(
        f"Two 1 kg masses are each tied to a wall by a {number(k)} N/m spring and to each other by a {number(kc)} N/m "
        "spring. One is pulled aside and let go. How long until the other swings with all the energy?",
        half,
        f"f₁ = √({number(k)})/2π = {number(float(hz[0]))} Hz, f₂ = √({number(k + 2 * kc)})/2π = "
        f"{number(float(hz[1]))} Hz. The two modes drift out of step at f₂ − f₁, so the energy is all across after "
        f"half a beat: 1/(2(f₂ − f₁)) = {number(half)} s.",
        unit="s",
        tolerance=0.03,
        setup=dict(k1=k, k2=kc, k3=k, pluck=(0.05, 0.0), tab=TAB_COUPLING),
    )


def _rail_frequency(rng: np.random.Generator) -> Question:
    m2 = pick(rng, (1, 2, 3))
    k2 = pick(rng, (50, 100, 200))
    r = pick(rng, (-1, 0.5, 1, 2))
    s = TwoMass(1.0, m2, 400.0, k2, 400.0)
    M, _, K = s.matrices()
    v = np.array([1.0, r])
    f = rayleigh_hz(v, M, K)
    kv, mv = float(v @ K @ v), float(v @ M @ v)
    return Question(
        f"Two masses, 1 kg and {number(m2)} kg, are tied to the walls by 400 N/m springs and to each other by "
        f"{number(k2)} N/m. They are held to move along the line x₂ = {number(r)} x₁. At what frequency do they swing?",
        f,
        f"Along v = (1, {number(r)}): vᵀKv = {number(kv)} N/m and vᵀMv = {number(mv)} kg, so ω² = "
        f"{number(kv / mv)} (Rayleigh's quotient) and f = {number(f)} Hz, between the free modes' "
        f"{number(float(s.modes()[0][0]))} and {number(float(s.modes()[0][1]))} Hz.",
        unit="Hz",
        setup=dict(m2=m2, k2=k2, rail=True, v1=(1.0 / math.hypot(1, r), r / math.hypot(1, r)), pluck=_along(r)),
    )


def _along(r: float) -> tuple[float, float]:
    """A 30 mm pluck along the line x₂ = r x₁."""
    h = math.hypot(1.0, r)
    return 0.03 / h, 0.03 * r / h


COUPLINGS = ("Static: through the springs", "Dynamic: through the masses", "Both", "Neither: they're uncoupled")


def _coupling_kind(rng: np.random.Generator) -> Question:
    case = int(rng.integers(3))
    m2 = pick(rng, (2, 3, 4))
    setup: dict[str, object]
    if case == 0:
        prompt, answer = "each mass's own displacement, x₁ and x₂", 0
        setup = dict(m2=m2)
        working = "The masses' matrix is diagonal, M itself; the coupling spring couples K: static coupling."
    elif case == 1:
        prompt, answer = "the arrows (1, 1) and (1, −1)", 1
        setup = dict(m2=m2, v1=IN_PHASE, v2=OPPOSED)
        working = (
            "With equal walls' springs these are K's own directions, so VᵀKV is diagonal; but with unequal "
            f"masses VᵀMV has (1 − {number(m2)})/2 off its diagonal: dynamic coupling."
        )
    else:
        prompt, answer = "the system's two mode shapes", 3
        a, b = _modes(TwoMass(m2=m2))
        setup = dict(m2=m2, v1=a, v2=b, modes=True)
        working = "The modes are orthogonal through both M and K, so both matrices are diagonal: uncoupled."
    return Question(
        f"Two masses, 1 kg and {number(m2)} kg, between walls with equal 400 N/m springs and a 100 N/m spring between "
        f"them, are described in {prompt}. How are the coordinates coupled?",
        answer,
        working,
        choices=COUPLINGS,
        why=tuple(
            ""
            if i == answer
            else {
                0: "Look at VᵀMV: is it diagonal in these coordinates?",
                1: "Look at VᵀKV: does the coupling spring still show off its diagonal?",
                2: "One of the two matrices is diagonal here: work out each.",
                3: "Only the modes make both matrices diagonal at once.",
            }[i]
            for i in range(4)
        ),
        setup=setup | dict(pluck=(0.05, 0.0)),
    )


PROBLEMS = tuple(
    Problem(key, title, "bases_page", f"See {page_link('bases_page', anchor, section)}.", make)
    for key, title, anchor, section, make in (
        ("symmetric-pair", "A symmetric pair", "modes", "The modes", _symmetric_pair),
        ("beat-time", "The energy passing across", "energy", "The energy in each coordinate", _beat_time),
        ("rail-frequency", "On a rail", "rail", "On a rail: Rayleigh's quotient", _rail_frequency),
        ("coupling-kind", "Which coupling?", "coupling", "Coupling belongs to the description", _coupling_kind),
    )
)

TIPS = {
    "m1": "<p>The left mass. Drag the ring on it up for a heavier one as it runs: the extra is caught at rest, the "
    "momentum kept.</p>",
    "m2": "<p>The right mass.</p>",
    "k1": "<p>The spring from the left wall to m₁. Drag the ring on it up for a stiffer one as it runs.</p>",
    "k2": "<p>The coupling spring between the masses: with none, each mass would swing on its own.</p>",
    "k3": "<p>The spring from m₂ to the right wall.</p>",
    "c": "<p>A viscous damper beside each of the three springs, each this strong.</p>",
    "rail": "<p>Take v₂ away and hold the masses to the line along v₁, as if the point slid on a rail. Put on as it "
    "runs, the point is moved to the nearest place on the rail; drag the arrow to turn the rail.</p>",
    "modes": "<p>Draw the two mode directions on the plane. Arrows dragged near a mode (or an axis) snap onto it "
    "either way.</p>",
    "theory": "<p>Draw Rayleigh's quotient for every rail's angle over the points, with f₁ and f₂ and the modes' "
    "angles.</p>",
}

NOTES_HTML = load_notes("bases", figures=bases_figures.FIGURES, trials=trials_html(TRIALS))
