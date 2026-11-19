"""The Frame builder page's Things to try and practice problems: its trial dataclass and the tabs, matrices and
swings it names, the trials' frames, and the problems, each worked from its printed numbers with core/frame.py and
core/frame_response.py.

`FramePage.apply_trial` applies a trial; the page re-exports `TRIALS` and `PROBLEMS`, where the practice window and
the Theory tab look for them. `tests/test_tutorial_numbers.py` checks the trials' numbers, `tests/test_practice.py`
the problems.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..core.frame import (
    TWO_PI,
    Element,
    Frame,
    FrameError,
    Kind,
    Node,
    Section,
    Support,
    Theory,
    assemble,
    cantilever,
    modes,
    portal,
    simply_supported,
    storeys,
    warren,
)
from ..core.frame_response import Load, LoadKind, formulations, rayleigh
from ..core.practice import Prediction, Problem, Question, Requirement, at_least, kept, number, pick
from .links import page_link

TAB_ASSEMBLY, TAB_ELEMENT, TAB_RESPONSE, TAB_FORMULATIONS, TAB_THEORY = range(5)
EMPTY = Frame()


class Matrix(Enum):
    K = "Stiffness K"
    M = "Mass M"
    C = "Damping C"


MATRICES = tuple(Matrix)


class Swing(Enum):
    MODE = "Undamped mode"
    COMPLEX = "Damped mode (complex)"
    RESPONSE = "Response to the load"


SWINGS = tuple(Swing)


@dataclass(frozen=True)
class FrameTrial:
    name: str
    note: str
    frame: Frame = EMPTY
    mode: int = 0
    tab: int = TAB_ASSEMBLY
    picked: int | None = None  # the element shown on the Element tab (and tinted on the Assembly tab)
    matrix: Matrix = Matrix.K
    grid: float = 0.25
    load: Load | None = None  # None: a vertical impulse at the last node free to move so (`default_load`)
    out: int | None = None  # the DOF the Response tab plots (None: the load's)
    swing: Swing = Swing.MODE
    predict: Prediction | None = None  # asked before it runs (Things to try)


SQUARE = Frame(
    (Node(0.0, 0.0, Support.PINNED), Node(1.0, 0.0, Support.PINNED), Node(1.0, 1.0), Node(0.0, 1.0)),
    tuple(Element(Kind.BAR, i, j) for i, j in ((0, 1), (1, 2), (2, 3), (3, 0))),
)
DEEP = Section(depth=0.2, width=0.05)
FIRST_FLOOR_DAMPER = Element(Kind.DAMPER, 2, None, value=5000.0, anchor=(-0.5, 1.0))
# Den Hartog's absorber, 0.25 kg: μ = 0.25/4.91 = 5.1 % of the four-beam cantilever's first mode's modal mass at the tip
TUNED = (
    dataclasses.replace(cantilever(1.0, 4), zeta=0.01)
    .with_node(1.0, 0.25, Support.GUIDED_Y)
    .with_element(Element(Kind.SPRING, 4, 5, value=15.6e3))
    .with_element(Element(Kind.DAMPER, 4, 5, value=16.0))
    .with_element(Element(Kind.MASS, 5, value=0.25))
)

TRIALS = (
    FrameTrial(
        "A cantilever in four beams",
        "A steel beam 50 × 50 mm, 1 m long, clamped at its left end and cut into four beam elements: 41.78, 262.1 "
        "and 738.7 Hz, against Euler's exact 41.78, 261.8 and 733.1. Each is a little high, the higher ones more: "
        "the mesh is a Rayleigh–Ritz model, so it can only be too stiff. Mode 4 (1301 Hz) stretches it: all its "
        "strain energy is axial. The first beam is picked: its matrices are on the Element tab, tinted where they "
        "land in K.",
        frame=cantilever(1.0, 4),
        picked=0,
        grid=0.125,
    ),
    FrameTrial(
        "One beam element",
        "The same cantilever as a single beam: 41.97 Hz, only 0.5 % high, but 413.6 Hz for the second mode, 58 % "
        "high: one element's two cubics can't bend into the second mode's shape. Select the beam and divide it "
        "(Divide into, on the left) to watch the frequencies come down onto Euler's.",
        frame=cantilever(1.0, 1),
        picked=0,
        tab=TAB_ELEMENT,
    ),
    FrameTrial(
        "Lumped mass",
        "The four-beam cantilever with each beam's mass lumped at its ends: 40.61, 238.7 and 632.1 Hz, now below "
        "Euler's, and further below than the consistent mass was above. The turns have no mass, so their four DOFs "
        "are condensed out (the readout): 8 DOFs with mass are left, and 8 modes.",
        frame=dataclasses.replace(cantilever(1.0, 4), lumped=True),
        matrix=Matrix.M,
        grid=0.125,
        predict=Prediction(
            "The four-beam cantilever's mass switched from consistent to lumped (half each beam's at each end). What "
            "happens to its frequencies?",
            (
                "Only the axial modes change",
                "They don't change",
                "They rise",
                "They fall",
            ),
            3,
            why=(
                "The bending modes change most: lumping gives the sections' turning no inertia and puts each element's mass at its ends.",
                'The mass matrix sets the frequencies as much as the stiffness does: lumping it moves them, by several percent at the higher modes.',
                'Consistent masses make the mesh a Rayleigh–Ritz model, whose frequencies are upper bounds; lumped masses at the nodes come out below them.',
                '',
            ),
        ),
    ),
    FrameTrial(
        "A deep beam: Timoshenko",
        "A steel beam 1 m long and 200 mm deep (<m>ℓ/h</m> = 5), pinned and on a roller, in eight Timoshenko beams: "
        "441.0, 1539 and 2992 Hz in bending (1295 Hz stretches it); Euler–Bernoulli (Beams, on the left) gives "
        "469.1, 1877 and 4227. Shear and the sections' turning inertia soften it, the higher modes more: −6 %, "
        "−18 %, −29 %. The Beams & plates page's closed forms are 440.8, 1529 and 2921 Hz.",
        frame=dataclasses.replace(simply_supported(1.0, 8, DEEP), theory=Theory.TIMOSHENKO),
        grid=0.125,
        predict=Prediction(
            "A deep beam (its length 5 times its depth) switched from Euler–Bernoulli to Timoshenko beams. What "
            "happens to its bending frequencies?",
            (
                "They don't change",
                "They rise, the higher modes most",
                "They fall, all by the same fraction",
                "They fall, the higher modes most",
            ),
            3,
            why=(
                "At five depths long, shear and the sections' turning matter: −6 % for mode 1, more above.",
                "Shear flexibility and rotary inertia both lower a frequency, and Timoshenko's beam adds both.",
                'Shear and rotary inertia matter more as the half waves shorten towards the depth, so the higher modes fall most.',
                '',
            ),
        ),
    ),
    FrameTrial(
        "A portal frame: sway",
        "Two columns 1.5 m high and a girder 2 m long, 50 × 50 mm steel, the bases clamped: the first mode, 14.90 "
        "Hz, sways the girder sideways on its columns, which bend in double curvature. Pin the bases (Select a "
        "base, Support on the left) and it drops to 6.78 Hz: the columns now bend in single curvature, like "
        "cantilevers held at the top by the girder.",
        frame=portal(2.0, 1.5, 4),
        predict=Prediction(
            "The portal frame, its bases clamped, sways at 14.90 Hz. If its bases were pinned instead, what would the"
            " sway mode do?",
            (
                "Fall to 0 Hz: a mechanism",
                "Hardly change",
                "Rise",
                "Fall to about half",
            ),
            3,
            why=(
                "Pinned bases let the columns' feet turn, but the girder's rigid joints still hold the frame square: it sways more easily, not freely.",
                'Clamped, the columns bend in double curvature; pinned, as cantilevers held at the top, far softer: 6.78 Hz.',
                "Freeing the bases to turn takes stiffness away; it can't add any.",
                '',
            ),
        ),
    ),
    FrameTrial(
        "A Warren truss: bars only",
        "Four bays of 1 m, 0.8 m high, 50 × 50 mm steel bars, pinned at the left and on a roller at the right. A "
        "bar carries no moment, so nothing turns its nodes: their nine turns are unused, struck out with the "
        "supports' three DOFs, leaving 2 × 9 − 3 = 15. The first mode, 112.6 Hz, bends the truss as a whole, but "
        "every bar only stretches: its strain energy is all axial.",
        frame=warren(),
        tab=TAB_ASSEMBLY,
        grid=0.1,
    ),
    FrameTrial(
        "A square of bars",
        "Four bars, the bottom two corners pinned: four free DOFs, but mode 1 is at 0 Hz, a parallelogram folding "
        "with no bar stretching. Maxwell's count, <m>2j = b + r</m> (2 × 4 = 4 + 4), says it might be rigid, but the "
        "bottom bar joins two pins and holds nothing (a state of self-stress): the count is needed, not enough. Add a diagonal (the Bar "
        "tool, node 1 to node 3) and the first mode is at 276.9 Hz.",
        frame=SQUARE,
        predict=Prediction(
            'Four bars make a square, its bottom two corners pinned. How many of its modes are at 0 Hz?',
            (
                "Four",
                "Two",
                "None",
                "One",
            ),
            3,
            why=(
                "The three bars that work hold three of the top joints' four DOFs; only one, the fold into a parallelogram, is free.",
                "The three bars that work leave one of the top joints' four DOFs free: a single fold into a parallelogram.",
                "Maxwell's count (8 = 4 + 4) is needed but not enough: the bottom bar between two pins does nothing, and the square folds into a parallelogram.",
                '',
            ),
        ),
    ),
    FrameTrial(
        "Two storeys: the chain again",
        "Columns 50 × 20 mm, 1 m high, clamped; floors 1.5 m long, 100 × 20 mm, with 100 kg on each. The two sway "
        "modes are at 3.44 and 9.14 Hz. As a chain (the Simulation page's) each storey is <m>k = 2 × 12EI/ℓ<sup>3</sup></m> = 168 "
        "kN/m (<m>ℓ</m> = 1 m, a column's height), and each floor 131.4 kg (its 100 kg, the 23.6 kg floor beam, and half the columns, 7.9 kg): "
        "3.52 and 9.21 Hz. The frame's are a little lower: its floors bend, so the columns' tops turn.",
        frame=storeys(),
        grid=0.25,
    ),
    FrameTrial(
        "Rayleigh damping: one answer",
        "The two-storey frame with Rayleigh damping, <m>C = αM + βK</m>, set to <m>ζ</m> = 2 % at its two sway modes (3.44 and "
        "9.14 Hz), and an impulse of 10 N·s across its top floor. The state space (solid) and the conventional "
        "real modes (dashed) lie on each other, FRF and time response alike: <m>C</m> is proportional, <m>Φ<sup>T</sup>CΦ</m> is diagonal, "
        "and the conventional model drops nothing. On the Formulations tab every MPC is 1.000. Beyond mode 2, "
        "Rayleigh's <m>βω</m> takes over: the floors' bending at 92.02 Hz has <m>ζ</m> = 14.7 %.",
        frame=dataclasses.replace(storeys(), zeta=0.02),
        tab=TAB_RESPONSE,
        load=Load(6, 0, LoadKind.IMPULSE, 10.0),
        swing=Swing.RESPONSE,
    ),
    FrameTrial(
        "A damper in the first storey",
        "The same frame, undamped, with a 5000 N·s/m damper from its first floor to the ground. The conventional "
        "model keeps <m>Φ<sup>T</sup>CΦ</m>'s diagonal: modes at 3.44 and 9.14 Hz with <m>ζ</m> = 23.9 and 23.2 %. The state space's are "
        "at 3.64 and 8.65 Hz, with <m>ζ</m> = 24.6 and 23.7 %, and complex: MPC 0.796 and 0.833 (the floors swing out of "
        "step, and their points spread on the Argand plot). On the Response tab, at the damper's floor, the "
        "conventional FRF is off by 25.8 % of the exact peak.",
        frame=storeys().with_element(FIRST_FLOOR_DAMPER),
        tab=TAB_FORMULATIONS,
        load=Load(2, 0, LoadKind.IMPULSE, 10.0),
        swing=Swing.COMPLEX,
    ),
    FrameTrial(
        "A step at the tip",
        "The four-beam cantilever, undamped, and a step of 1000 N up at its tip. Held there slowly, the tip would "
        "rise <m>Fℓ<sup>3</sup>/3EI</m> = 3.05 mm; put on at once, it overshoots to 6.08 mm, 1.997 times as far, and swings about "
        "3.05 mm for ever. Give it some damping (Damping ζ, on the left) and it settles there.",
        frame=cantilever(1.0, 4),
        tab=TAB_RESPONSE,
        load=Load(4, 1, LoadKind.STEP, 1000.0),
        swing=Swing.RESPONSE,
        grid=0.125,
        predict=Prediction(
            "1000 N put on the cantilever's tip at once, undamped. Held there slowly, the tip would rise 3.05 mm. How"
            " far does it go?",
            (
                "3.05 mm, and it settles there",
                "About twice as far, and it swings on",
                "About ten times as far",
            ),
            1,
            ordered=True,
            why=(
                "That's the force put on slowly, or with damping. Put on at once, undamped, the tip overshoots to twice and swings on.",
                '',
                'A sudden step at most doubles the static deflection: 6.08 mm.',
            ),
        ),
    ),
    FrameTrial(
        "Resonance from rest",
        "The cantilever with 1 % Rayleigh damping, driven at its first mode, 41.78 Hz, by 100 N at its tip. From "
        "rest the swing grows toward 14.8 mm, 48.5 times the static 0.305 mm (<m>1/2ζ</m> = 50 for one mode on its "
        "own). After the record's 20 cycles (0.48 s) it has reached 10.6 mm: it grows as <m>1 − e<sup>−ζω<sub>1</sub>t</sup></m>, "
        "and <m>1/ζω<sub>1</sub></m> is 0.38 s.",
        frame=dataclasses.replace(cantilever(1.0, 4), zeta=0.01),
        tab=TAB_RESPONSE,
        load=Load(4, 1, LoadKind.HARMONIC, 100.0, 41.78),
        swing=Swing.RESPONSE,
        grid=0.125,
    ),
    FrameTrial(
        "A tuned mass damper",
        "A 0.25 kg mass on a guide at the cantilever's tip, held by a 15.6 kN/m spring and a 16.0 N·s/m damper, "
        "with 1 % Rayleigh damping: Den Hartog's tuning for <m>μ</m> = 0.25/4.91 = 5.1 % of the first mode's modal "
        "mass at the tip, tuned to <m>f<sub>1</sub>/(1 + μ)</m> = 39.75 Hz with <m>ζ</m> = 12.8 %. The tip's FRF "
        "has two nearly equal peaks, at 37.4 and 44.0 Hz, 5.61 and 5.53 times the static deflection, where the "
        "bare beam's is 48.5. The conventional model gives 3.74 and 3.86: the damper couples the two close modes "
        "strongly (MPC 0.833 and 0.596), the conventional model drops that, and its peaks are a third too low, "
        "on the unsafe side.",
        frame=TUNED,
        tab=TAB_RESPONSE,
        load=Load(4, 1, LoadKind.HARMONIC, 100.0, 40.0),
        out=13,
        grid=0.125,
    ),
)

# ------------------------------------------------------------------ practice problems
# Each question's numbers are printed round, and its answer is worked from them as printed.


def _bar_entry(rng: np.random.Generator) -> Question:
    area, length = pick(rng, (100.0, 200.0, 500.0)), pick(rng, (0.5, 1.0, 2.0))
    angle = pick(rng, (30.0, 45.0, 60.0))
    a = math.radians(angle)
    frame = Frame(
        (Node(0.0, 0.0, Support.PINNED), Node(length * math.cos(a), length * math.sin(a))),
        (Element(Kind.BAR, 0, 1, section=Section(depth=math.sqrt(area) / 1000, width=math.sqrt(area) / 1000)),),
    )
    part = assemble(frame).locals[0]
    k = float(part.stiffness()[2, 2]) / 1e6  # u2's diagonal
    return Question(
        f"A steel bar (E = 210 GPa) of section {area:g} mm² and length {length:g} m runs from node 1 up at "
        f"{angle:g}° to x. What does it add to K at row and column u<sub>2</sub>, in MN/m?",
        k,
        f"Its own k' is (EA/L)[[1, −1], [−1, 1]] on the stretch, EA/L = {number(210e9 * area * 1e-6 / length / 1e6)} "
        f"MN/m, and T takes u<sub>2</sub> into it as cos γ. So Tᵀk′T has (EA/L) cos²γ there: "
        f"{number(k)} MN/m.",
        unit="MN/m",
        setup=dict(frame=frame, picked=0, tab=TAB_ELEMENT),
    )


def _truss_dofs(rng: np.random.Generator) -> Question:
    bays = int(pick(rng, (2, 3, 4, 5, 6)))
    frame = warren(bays)
    free = int(assemble(frame).free.sum())
    joints = len(frame.nodes)
    return Question(
        f"A Warren truss of {bays} bays (bars only: {bays + 1} joints along the bottom, {bays} along the top) is "
        "pinned at one end and on a roller at the other. How many free DOFs has it?",
        free,
        f"Each of the {joints} joints has u, v and θ, but bars don't turn the joints: the {joints} θ's are unused. "
        f"The pin holds two DOFs and the roller one: 2 × {joints} − 3 = {free}.",
        setup=dict(frame=frame, grid=0.1),
    )


def _tip_mass(rng: np.random.Generator) -> Question:
    side, length = pick(rng, (10.0, 15.0, 20.0)), pick(rng, (0.5, 0.8, 1.0))
    mass = pick(rng, (5.0, 10.0, 20.0))
    I = (side / 1000) ** 4 / 12
    k = 3 * 210e9 * I / length**3
    f = math.sqrt(k / mass) / TWO_PI
    sec = Section(depth=side / 1000, width=side / 1000)
    frame = cantilever(length, 4, sec)
    frame = frame.with_element(Element(Kind.MASS, len(frame.nodes) - 1, value=mass))
    return Question(
        f"A steel cantilever (E = 210 GPa), {side:g} × {side:g} mm and {length:g} m long, carries a {mass:g} kg mass "
        "at its tip, much heavier than the beam. About what frequency does it bounce at, the beam's own mass "
        "left out?",
        f,
        f"The tip's stiffness is 3EI/L³, I = s⁴/12 = {number(I * 1e12)} mm⁴: k = {number(k)} N/m, and "
        f"f = √(k/m)/2π = {number(f)} Hz. (The page's model counts the beam's mass too, so it comes out a little "
        "lower.)",
        unit="Hz",
        setup=dict(frame=frame, grid=0.1),
    )


def _square(rng: np.random.Generator) -> Question:
    return Question(
        "Four bars make a square, its bottom two corners pinned. How many of its modes are at 0 Hz?",
        0,
        "One: with the bottom bar between two pins doing nothing, three bars can't hold the top two joints' four "
        "DOFs, and the square folds into a parallelogram with no bar stretched. Maxwell's 2j = b + r holds (8 = 4 "
        "+ 4), but it's needed, not enough.",
        choices=("One", "None", "Two", "Four"),
        why=(
            "",
            "Maxwell's count (8 = 4 + 4) is needed but not enough: the bottom bar between two pins does nothing, "
            "and the square folds into a parallelogram.",
            "The three bars that work leave one of the top joints' four DOFs free: a single fold into a parallelogram.",
            "The three bars that work hold three of the top joints' four DOFs; only one, the fold into a "
            "parallelogram, is free.",
        ),
        setup=dict(frame=SQUARE),
        predict=True,
    )


def _lumped(rng: np.random.Generator) -> Question:
    return Question(
        "A beam model's mass is switched from consistent to lumped (half each element's at each end). What "
        "happens to its frequencies?",
        0,
        "They fall: the consistent mass makes the mesh a Rayleigh–Ritz model, whose frequencies are upper bounds; "
        "lumped puts the mass at the nodes, where it moves most, and the turns get none. For a cantilever in four "
        "beams: 41.78 → 40.61 Hz, 262.1 → 238.7, 738.7 → 632.1.",
        choices=("They fall", "They rise", "They don't change", "Only the axial modes change"),
        why=(
            "",
            "Consistent masses make the mesh a Rayleigh–Ritz model, whose frequencies are upper bounds; lumped "
            "masses at the nodes come out below them.",
            "The mass matrix sets the frequencies as much as the stiffness does: lumping it moves them, by several "
            "percent at the higher modes.",
            "The bending modes change most: lumping gives the sections' turning no inertia and puts each element's "
            "mass at its ends.",
        ),
        setup=dict(frame=dataclasses.replace(cantilever(1.0, 4), lumped=True), matrix=Matrix.M, grid=0.125),
        predict=True,
    )


def _pinned_bases(rng: np.random.Generator) -> Question:
    return Question(
        "A portal frame's two column bases are clamped. They're changed to pins. What happens to its sway mode?",
        0,
        "Its frequency drops to about half (14.90 → 6.78 Hz for the Things to try one): pinned at the foot, each "
        "column bends in single curvature, like a cantilever held by the girder at the top, instead of double. "
        "A cantilever's tip is 4 times as soft as a column clamped at both ends, so f falls by about √4 = 2.",
        choices=(
            "It falls to about half",
            "It rises",
            "It hardly changes",
            "It falls to 0 Hz: a mechanism",
        ),
        why=(
            "",
            "Pins free the columns' feet to turn, which can only make the frame softer.",
            "Pinned at the foot, each column bends in single curvature, like a cantilever, instead of double: 4 "
            "times as soft, half the frequency.",
            "The girder still holds the columns' tops, so the frame still resists sway: softer, not a mechanism.",
        ),
        setup=dict(frame=portal(2.0, 1.5, 4, base=Support.PINNED)),
        predict=True,
    )


def _timoshenko(rng: np.random.Generator) -> Question:
    return Question(
        "A deep beam (its length 5 times its depth) is switched from Euler–Bernoulli to Timoshenko beams. What "
        "happens to its bending frequencies?",
        0,
        "They fall, the higher modes most: shear adds to the bending's deflection, and the sections' turning adds "
        "inertia, both more as the half waves shorten toward the depth (−6 %, −18 % and −29 % for the first three "
        "here).",
        choices=(
            "They fall, the higher modes most",
            "They fall, all by the same fraction",
            "They rise, the higher modes most",
            "They don't change",
        ),
        why=(
            "",
            "Shear and rotary inertia matter more as the half waves shorten towards the depth, so the higher modes "
            "fall most.",
            "Shear flexibility and rotary inertia both lower a frequency, and Timoshenko's beam adds both.",
            "At five depths long, shear and the sections' turning matter: −6 % for mode 1, more above.",
        ),
        setup=dict(frame=dataclasses.replace(simply_supported(1.0, 8, DEEP), theory=Theory.TIMOSHENKO), grid=0.125),
        predict=True,
    )


def _brace_spec(rng: np.random.Generator) -> Question:
    span, height = pick(rng, (2.0, 3.0)), pick(rng, (1.5, 2.0))
    start = portal(span, height, 2, base=Support.PINNED)
    f0 = float(modes(start).hz[0])
    target = 3 * float(number(f0))

    def goal(t: FrameTrial):
        try:
            f1 = float(modes(t.frame).hz[0])
        except FrameError:
            f1 = 0.0
        same = t.frame.nodes[: len(start.nodes)] == start.nodes
        return (
            Requirement("The nodes and their supports kept", "kept" if same else "changed", same),
            at_least("The first frequency", f1, target, "Hz"),
        )

    braced = start.with_element(Element(Kind.BAR, 0, 4))  # a base to the far top corner (2 members a side)
    return Question(
        f"This portal frame ({span:g} m wide, {height:g} m high, 50 × 50 mm steel) stands on pins and sways at "
        f"{number(f0)} Hz. Without moving its nodes or changing its supports, raise its first frequency to at "
        f"least {number(target)} Hz.",
        0,
        "Brace it: a bar across the diagonal (the Bar tool, from a base to the opposite top corner) stops the "
        "sway, which only bent the columns, by stretching instead. The first mode is then the girder's bending, "
        f"{number(float(modes(braced).hz[0]))} Hz. Thicker columns would do it too, but slowly: f grows only as "
        "the depth to the 1.5.",
        setup=dict(frame=start),
        goal=goal,
        design=dict(frame=braced),
    )


def _rayleigh(rng: np.random.Generator) -> Question:
    f1 = pick(rng, (2.0, 5.0, 10.0, 20.0))
    f2 = f1 * pick(rng, (2.5, 3.0, 4.0))
    f3 = f2 * pick(rng, (3.0, 5.0, 10.0))
    zeta = pick(rng, (0.01, 0.02, 0.05))
    alpha, beta = rayleigh(zeta, np.array([f1, f2]))
    w3 = TWO_PI * f3
    z3 = 0.5 * (alpha / w3 + beta * w3)
    return Question(
        f"A frame's Rayleigh damping C = αM + βK is set to ζ = {100 * zeta:g} % at its first two modes, {f1:g} and "
        f"{f2:g} Hz. What is ζ at a mode at {f3:g} Hz, in %?",
        100 * z3,
        f"ζ = (α/ω + βω)/2, and α = 2ζω₁ω₂/(ω₁ + ω₂) = {number(alpha)} 1/s, β = 2ζ/(ω₁ + ω₂) = {number(beta)} s "
        f"meet it at both. At ω = 2π × {f3:g}: ζ = {number(100 * z3)} %. Above mode 2 the βω term grows "
        "without end: Rayleigh damping kills high modes.",
        unit="%",
    )


def _step(rng: np.random.Generator) -> Question:
    force, length = pick(rng, (200.0, 500.0, 1000.0)), pick(rng, (0.5, 0.8, 1.0))
    side = pick(rng, (30.0, 40.0, 50.0))
    I = (side / 1000) ** 4 / 12
    static = force * length**3 / (3 * 210e9 * I)
    sec = Section(depth=side / 1000, width=side / 1000)
    frame = cantilever(length, 4, sec)
    return Question(
        f"An undamped steel cantilever (E = 210 GPa), {side:g} × {side:g} mm and {length:g} m long, has a "
        f"{force:g} N force put on its tip at once, and left there. How far does the tip go at most, in mm?",
        2000 * static,
        f"Twice the static deflection FL³/3EI = {number(1000 * static)} mm (I = s⁴/12 = {number(I * 1e12)} mm⁴): "
        f"{number(2000 * static)} mm. A suddenly applied load does as much work on the way as the spring stores "
        "when it stops, F x = ½ k x², so x = 2F/k; the frame's other modes change it by a fraction of a percent.",
        unit="mm",
        setup=dict(frame=frame, load=Load(4, 1, LoadKind.STEP, force), tab=TAB_RESPONSE, grid=0.125),
    )


def _damper_spec(rng: np.random.Generator) -> Question:
    target = pick(rng, (8.0, 10.0, 12.0))
    start = storeys()

    def goal(t: FrameTrial):
        try:
            first = formulations(t.frame).modes[0]
        except FrameError:
            first = None
        zeta = 100 * first.zeta_exact if first is not None and first.pole is not None else 0.0
        mpc = first.measures.mpc if first is not None and first.measures is not None else 0.0
        same = t.frame.nodes[: len(start.nodes)] == start.nodes
        return (
            Requirement("The nodes and their supports kept", "kept" if same else "changed", same),
            kept("Rayleigh damping ζ", 100 * t.frame.zeta, 0.0, "%"),
            at_least("Mode 1's ζ (the state space's)", zeta, target, "%"),
            at_least("Mode 1's MPC", mpc, 0.95),
        )

    damped = start.with_element(Element(Kind.DAMPER, 6, None, value=1000.0, anchor=(-0.5, 2.0)))
    return Question(
        f"Damp this two-storey frame's first mode to ζ ≥ {target:g} % with dampers (Rayleigh damping left at 0), "
        "without moving its nodes, and keep the mode nearly real: MPC ≥ 0.95.",
        0,
        "Put a damper where the mode moves most: the top floor sways furthest in mode 1, so 1000 N·s/m from it "
        f"to the ground gives ζ = {number(100 * formulations(damped).modes[0].zeta_exact)} % with MPC 0.992. On "
        "the first floor it takes twice that, and the mode goes more complex (2000 N·s/m: 9.61 %, MPC 0.969).",
        setup=dict(frame=start, tab=TAB_FORMULATIONS),
        goal=goal,
        design=dict(frame=damped, tab=TAB_FORMULATIONS),
    )


PROBLEMS = tuple(
    Problem(key, title, "frame_page", f"See {page_link('frame_page', anchor, section)}.", make)
    for key, title, anchor, section, make in (
        ("bar-entry", "A bar's share of K", "transformation", "From the element's axes to the frame's", _bar_entry),
        ("truss-dofs", "A truss's free DOFs", "constraints", "Supports and struck-out DOFs", _truss_dofs),
        ("tip-mass", "A mass on a cantilever", "elements", "The elements' matrices", _tip_mass),
        ("square", "Predict: a square of bars", "mechanisms", "Mechanisms", _square),
        ("lumped", "Predict: lumped mass", "modes", "The modes", _lumped),
        ("pinned-bases", "Predict: pinned bases", "elements", "The elements' matrices", _pinned_bases),
        ("timoshenko", "Predict: Timoshenko beams", "timoshenko", "Timoshenko beams", _timoshenko),
        ("brace-spec", "Spec: brace a portal frame", "mechanisms", "Mechanisms", _brace_spec),
        ("rayleigh", "Rayleigh damping's ζ", "damping", "Damping", _rayleigh),
        ("step", "A step on a cantilever", "forced", "Loads and responses", _step),
        ("damper-spec", "Spec: damp a two-storey frame", "complex", "Complex modes", _damper_spec),
    )
)
