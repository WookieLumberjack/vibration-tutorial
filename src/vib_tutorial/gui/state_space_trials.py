"""State-space modes' Things to try, practice problems, tips and notes, without Qt (#478 Phase 4, page standard item 7).

Its state is `view_models/state_space.py`'s `StateTrial`; `gui/state_space.py` binds the page's controls to it and
runs it live. A trial is a setup to go back to: it sets the masses, springs, dampers and the planes' basis, starts the
masses from one of `STARTS` (or leaves them at rest), and says what to look for. Its prediction is asked only once the
reader has handled the masses (`lesson.TrialBox(ask_after_play=True)`), about a case they haven't tried.

Working on it: `tests/test_tutorial_numbers.py` checks the trials' numbers, `tests/test_practice.py` the problems,
`tests/test_qt_free.py` that this loads no Qt.
"""

from __future__ import annotations

import math

import numpy as np

from ..core.bases import TwoMass
from ..core.practice import Prediction, Problem, Question, number, pick
from ..core.state_modes import state_modes
from ..view_models.state_space import (
    MASS,
    STATE,
    TAB_MATRICES,
    TAB_SHAPES,
    TAB_SIZES,
    UNDAMPED,
    StateTrial,
)
from . import state_space_figures
from .links import load_notes, page_link
from .trials_text import trials_html

M1_PULLED, UNDAMPED_TWO, STATE_ONE, STATE_TWO = range(4)  # the starts, as STARTS

TRIALS = (
    StateTrial(
        "Each mass's own coordinates",
        "<m>m<sub>1</sub></m> pulled 30 mm and let go, each plane one mass's <m>x</m> and <m>ẋ/ω̂</m>. Both planes "
        "move, and in step with each other: the coupling spring and the dampers pass the motion from one mass to the "
        "other, and <m>T<sup>−1</sup>AT</m> shows it in red, in both off-diagonal blocks. This is the state as the "
        "equations first give it.",
        basis=MASS,
        start=M1_PULLED,
    ),
    StateTrial(
        "The modes of M and K leak",
        "Started in mode 2 of <m>M</m> and <m>K</m>, the last page's mode, with mode 1's pair at zero. On the Sizes "
        "tab, mode 1's pair (dashed) climbs from nothing to 1.48 mm in 0.41 s: the dampers pass motion from "
        "mode 2 to mode 1, through the one red term left in each velocity row of <m>T<sup>−1</sup>AT</m>. The springs "
        "no longer couple these coordinates; the dampers still do.",
        basis=UNDAMPED,
        start=UNDAMPED_TWO,
        tab=TAB_SIZES,
        predict=Prediction(
            "The dampers twice as strong, c = 2 N·s/m, and started in mode 2 of M and K again. Mode 1's pair climbs…",
            ("less high", "just as high", "higher"),
            2,
            ordered=True,
            why=(
                "The leak is the dampers' doing: the term that passes the motion across is c's, so stronger dampers "
                "pass more, even though everything dies away faster.",
                "The red term in T⁻¹AT grows with c, so the leak does too: 2.28 mm instead of 1.48.",
                "",
            ),
        ),
    ),
    StateTrial(
        "The same start, in the state-space modes",
        "The same start, now in <m>A</m>'s own modes. Mode 1's pair starts at 0.99 mm, not zero, since the last "
        "page's mode 2 isn't quite one of these, but nothing passes between the pairs: on the Sizes tab both are "
        "straight lines on the log axis, each shrinking as <m>e<sup>σt</sup></m>, mode 1 at <m>σ</m> = −0.358 /s and "
        "mode 2 at −1.14 /s. <m>T<sup>−1</sup>AT</m> is two blocks, and nothing else.",
        basis=STATE,
        start=UNDAMPED_TWO,
        tab=TAB_SIZES,
    ),
    StateTrial(
        "One state-space mode alone",
        "Started in state-space mode 2 itself, the masses moving as well as pulled. Mode 1's plane stays still at "
        "its center, and mode 2's point spirals in, clockwise, turning at 3.619 Hz and shrinking at <m>σ</m> = "
        "−1.14 /s. Switch the basis to the modes of <m>M</m> and <m>K</m> as it runs: now mode 1's pair wobbles, "
        "at mode 2's frequency, where the dampers drag it.",
        basis=STATE,
        start=STATE_TWO,
    ),
    StateTrial(
        "A symmetric pair",
        "Both masses 1 kg, with the same dampers. Now the damping is proportional, <m>C = −3M + 0.01K</m>, and the "
        "modes of <m>M</m> and <m>K</m> uncouple them too: started in mode 2, mode 1's pair stays at zero on the Sizes "
        "tab, and <m>T<sup>−1</sup>AT</m> has no red. The shapes are real (the Shapes tab): the masses move exactly "
        "together or exactly opposed.",
        m2=1.0,
        basis=UNDAMPED,
        start=UNDAMPED_TWO,
        tab=TAB_SIZES,
    ),
    StateTrial(
        "Heavier dampers: complex shapes",
        "Each damper 5 N·s/m. In mode 1, <m>m<sub>2</sub></m> lags <m>m<sub>1</sub></m> by 18.8°: the masses don't "
        "reach their peaks together, so on the plane of <m>x<sub>1</sub></m> and <m>x<sub>2</sub></m> the mode "
        "alone traces an ellipse, not a line, spanned by <m>Re ψ</m> and <m>Im ψ</m>. The Shapes tab draws each "
        "mass's part of <m>ψ</m> as an arrow: they point apart by the lag.",
        c=5.0,
        basis=STATE,
        start=STATE_ONE,
        tab=TAB_SHAPES,
    ),
    StateTrial(
        "Mode 2 overdamped",
        "Each damper 20 N·s/m: past 18.8 N·s/m mode 2 no longer swings. Its two eigenvalues are real, −15.4 and "
        "−31.4 /s, and its block in <m>T<sup>−1</sup>AT</m> is just those two decay rates: its plane shows its point "
        "sliding into the center without turning. Mode 1 still spirals, at 2.276 Hz.",
        c=20.0,
        basis=STATE,
        start=STATE_TWO,
    ),
)


# ------------------------------------------------------------------ the practice problems


def _damping_ratio(rng: np.random.Generator) -> Question:
    sigma = pick(rng, (-0.5, -1.0, -2.0, -4.0))
    omega = pick(rng, (10.0, 15.0, 20.0, 30.0))
    zeta = -sigma / math.hypot(sigma, omega)
    return Question(
        f"A mode's eigenvalues are λ = {number(sigma)} ± {number(omega)}i /s. What is its damping ratio ζ, in %?",
        100 * zeta,
        f"|λ| = √({number(sigma)}² + {number(omega)}²) = {number(math.hypot(sigma, omega))} rad/s, the mode's "
        f"undamped-like frequency, and ζ = −σ/|λ| = {number(-sigma)}/{number(math.hypot(sigma, omega))} = "
        f"{number(100 * zeta)} %.",
        unit="%",
    )


def _halving(rng: np.random.Generator) -> Question:
    m2 = pick(rng, (1.5, 2.0, 3.0))
    c = pick(rng, (1.0, 2.0, 4.0))
    s = TwoMass(m2=m2, c=c)
    one = state_modes(*s.matrices())[0]
    sigma = round(one.sigma, 3)
    t = math.log(2.0) / -sigma
    return Question(
        f"Two masses, 1 kg and {number(m2)} kg, with dampers of {number(c)} N·s/m. Mode 1's eigenvalues are "
        f"λ = {number(sigma)} ± {number(one.omega, 4)}i /s. Started in that state-space mode alone, how long until "
        "its pair's size has halved?",
        t,
        f"Alone, a state-space mode's pair shrinks as e^(σt), whatever the other mode does: it halves when "
        f"e^(σt) = 1/2, t = ln 2/|σ| = 0.6931/{number(-sigma)} = {number(t)} s.",
        unit="s",
        setup=dict(m2=m2, c=c, basis=STATE, start=STATE_ONE, tab=TAB_SIZES),
    )


UNCOUPLES = ("Yes: they uncouple the dampers too", "No: the dampers still couple them")


def _proportional(rng: np.random.Generator) -> Question:
    case = int(rng.integers(3))
    c = pick(rng, (1.0, 2.0, 5.0))
    if case == 0:
        setup: dict[str, object] = dict(m2=1.0)
        about, answer = "two 1 kg masses, 400 N/m to each wall", 0
        working = (
            "Equal masses and equal walls: symmetry sets all three matrices' directions at ±45°, so the modes of M and "
            "K make VᵀCV diagonal too, and T⁻¹AT in the modes of M and K has no coupling."
        )
    elif case == 1:
        setup = dict(m2=2.0)
        about, answer = "a 1 kg and a 2 kg mass, 400 N/m to each wall", 1
        working = (
            "Unequal masses turn the modes of M and K away from ±45°, but the dampers are still "
            "[[2, −1], [−1, 2]], no mix of M and K: the damping isn't proportional, and VᵀCV keeps a term off its diagonal, and the dampers couple the modes."
        )
    else:
        setup = dict(m2=1.0, k3=900.0)
        about, answer = "two 1 kg masses, 400 N/m to one wall and 900 N/m to the other", 1
        working = (
            "Unequal walls turn the modes of M and K, and the dampers, [[2, −1], [−1, 2]], don't follow: "
            "the damping isn't proportional, and VᵀCV isn't diagonal in them."
        )
    return Question(
        f"Dampers of {number(c)} N·s/m sit beside each of the three springs of {about}, 100 N/m between them. Do the "
        "modes of M and K uncouple the equations, dampers and all?",
        answer,
        working,
        choices=UNCOUPLES,
        why=(
            ""
            if answer == 0
            else "Only proportional damping (C = αM + βK, say, or a symmetry that fixes all three matrices' "
            "directions) is uncoupled by the modes of M and K. Is this?",
            ""
            if answer == 1
            else "Here the damping is proportional: the symmetry that fixes the modes fixes C's directions too.",
        ),
        setup=setup | dict(c=c, basis=UNDAMPED, start=UNDAMPED_TWO, tab=TAB_MATRICES),
    )


PROBLEMS = tuple(
    Problem(key, title, "state_page", f"See {page_link('state_page', anchor, section)}.", make)
    for key, title, anchor, section, make in (
        ("damping-ratio", "A mode's damping ratio", "eigen", "A's modes come in pairs", _damping_ratio),
        ("halving", "A spiral halving", "real-basis", "A real basis: one plane a mode", _halving),
        ("proportional", "Do the modes of M and K do?", "compare", "Three bases compared", _proportional),
    )
)

TIPS = {
    "m1": "<p>The left mass. Drag the ring on it up for a heavier one as it runs.</p>",
    "m2": "<p>The right mass. At 2 kg, the page's start, the damping isn't proportional: the modes of M "
    "and K can't uncouple them.</p>",
    "k1": "<p>The spring from the left wall to m₁.</p>",
    "k2": "<p>The coupling spring between the masses.</p>",
    "k3": "<p>The spring from m₂ to the right wall.</p>",
    "c": "<p>A viscous damper beside each of the three springs, each this strong. Past about 19 N·s/m (with m₂ = "
    "2 kg) mode 2 is overdamped.</p>",
    "basis": "<p>Which coordinates the two planes show the four numbers of the state in, a pair a plane: each mass's "
    "displacement and velocity; each mode of M and K's; or A's own modes, which uncouple any dampers. Switch it as "
    "it runs: the motion doesn't change, only its description.</p>",
    "start": "<p>Start the masses from here and let them go: m₁ alone pulled; the last page's mode 2 (the modes of M "
    "and K); or one of A's own modes, which also sets the masses moving as the mode needs.</p>",
}

NOTES_HTML = load_notes("state_space", figures=state_space_figures.FIGURES, trials=trials_html(TRIALS))
