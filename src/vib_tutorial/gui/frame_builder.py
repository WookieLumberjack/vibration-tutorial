"""Frame builder page: build a plane frame or truss on a grid, watch its matrices assemble, see its modes.

Left: Things to try, the selected item's properties (a node's position, support, point mass and torsion spring;
a member's kind, material and section; a spring's or damper's value), the model's settings (consistent or lumped
mass, Euler–Bernoulli or Timoshenko beams) and what to show. Centre: the tools, the canvas (click to place nodes,
click two nodes to join them, drag a node to move it), swinging in the mode picked, and the modes' table. Right:
the global matrix with the picked element's entries tinted and the struck-out rows and columns greyed, the
element's own matrices and its transformation, and the Theory. The page has its own model (core/frame.py) and
animates the mode on its own frame loop while it shows.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.frame import (
    MATERIALS,
    MAX_NODES,
    SHAPES,
    SUPPORTS,
    THEORIES,
    TWO_PI,
    Assembly,
    Element,
    Frame,
    FrameError,
    Kind,
    Modes,
    Node,
    Section,
    Shape,
    Support,
    Theory,
    assemble,
    cantilever,
    dof_label,
    member_curve,
    modes,
    portal,
    simply_supported,
    storeys,
    warren,
)
from ..core.frame_response import (
    LOAD_KINDS,
    Formulations,
    Load,
    LoadKind,
    Responses,
    formulations,
    rayleigh,
    responses,
)
from ..core.practice import Problem, Question, Requirement, at_least, kept, number, pick
from . import frame_figures, frame_stats, settle_plots, weak
from .layouts import FlowLayout, group
from .lesson import (
    Form,
    RunButton,
    TrialBox,
    column,
    combo,
    fmt,
    note_label,
    plot,
    readout_label,
    set_quietly,
    theory,
    three_panes,
    trials_html,
)
from .links import load_notes, page_link
from .style import colors, rgb
from .theming import add_legend, mute

TAB_ASSEMBLY, TAB_ELEMENT, TAB_RESPONSE, TAB_FORMULATIONS, TAB_THEORY = range(5)
EMPTY = Frame()
FRAME_MS = 16
SHOW_HZ = 0.6  # the animation's own pace: every mode swings at it
SWING = 0.08  # the largest displacement drawn, as a fraction of the frame's size
MODE_ROWS = 20  # modes listed at most
NODE_PX, ELEMENT_PX = 10, 7  # how near a click has to be to pick a node or an element
GRIDS = (0.05, 0.1, 0.125, 0.25, 0.5, 1.0)  # m
MATRIX_LIMIT = 90  # DOFs: beyond this the Assembly tab lists the free ones only
PLAY_S = 8.0  # s: a response's record plays over this long on the canvas
DOF_CHOICES = ("u, along x", "v, along y", "θ, a turn")


class Tool(Enum):
    SELECT = "Select"
    NODE = "Node"
    BEAM = "Beam"
    BAR = "Bar"
    SPRING = "Spring"
    DAMPER = "Damper"
    TORSION = "Torsion spring"
    MASS = "Mass"
    SUPPORT = "Support"


TOOLS = tuple(Tool)
TOOL_HINTS = {
    Tool.SELECT: "Click a node or an element to edit it on the left; drag a node to move it. Delete removes it.",
    Tool.NODE: "Click the grid to place a node.",
    Tool.BEAM: "Click two nodes (or grid points, which become nodes) to join them with a beam; keep clicking to "
    "go on from the last. Esc or a right click stops.",
    Tool.BAR: "Click two nodes (or grid points) to join them with a bar (pinned ends: it only stretches). Esc or "
    "a right click stops.",
    Tool.SPRING: "Click a node, then another node, or a grid point to anchor it to the ground there.",
    Tool.DAMPER: "Click a node, then another node, or a grid point to anchor it to the ground there.",
    Tool.TORSION: "Click a node to add a torsion spring from its turn to the ground.",
    Tool.MASS: "Click a node to add a point mass there.",
    Tool.SUPPORT: "Pick a support, then click a node to give it that support (click again to free it).",
}


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
    ),
    FrameTrial(
        "A deep beam: Timoshenko",
        "A steel beam 1 m long and 200 mm deep (<m>ℓ/h</m> = 5), pinned and on a roller, in eight Timoshenko beams: "
        "441.0, 1539 and 2992 Hz in bending (1295 Hz stretches it); Euler–Bernoulli (Beams, on the left) gives "
        "469.1, 1877 and 4227. Shear and the sections' turning inertia soften it, the higher modes more: −6 %, "
        "−18 %, −29 %. The Beams & plates page's closed forms are 440.8, 1529 and 2921 Hz.",
        frame=dataclasses.replace(simply_supported(1.0, 8, DEEP), theory=Theory.TIMOSHENKO),
        grid=0.125,
    ),
    FrameTrial(
        "A portal frame: sway",
        "Two columns 1.5 m high and a girder 2 m long, 50 × 50 mm steel, the bases clamped: the first mode, 14.90 "
        "Hz, sways the girder sideways on its columns, which bend in double curvature. Pin the bases (Select a "
        "base, Support on the left) and it drops to 6.78 Hz: the columns now bend in single curvature, like "
        "cantilevers held at the top by the girder.",
        frame=portal(2.0, 1.5, 4),
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
        "A square of bars: a mechanism",
        "Four bars, the bottom two corners pinned: four free DOFs, but mode 1 is at 0 Hz, a parallelogram folding "
        "with no bar stretching. Maxwell's count, <m>2j = b + r</m> (2 × 4 = 4 + 4), says it might be rigid, but the "
        "bottom bar joins two pins and holds nothing (a state of self-stress): the count is needed, not enough. Add a diagonal (the Bar "
        "tool, node 1 to node 3) and the first mode is at 276.9 Hz.",
        frame=SQUARE,
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
        "A step: twice as far",
        "The four-beam cantilever, undamped, and a step of 1000 N up at its tip. Held there slowly, the tip would "
        "rise <m>Fℓ<sup>3</sup>/3EI</m> = 3.05 mm; put on at once, it overshoots to 6.08 mm, 1.997 times as far, and swings about "
        "3.05 mm for ever. Give it some damping (Damping ζ, on the left) and it settles there.",
        frame=cantilever(1.0, 4),
        tab=TAB_RESPONSE,
        load=Load(4, 1, LoadKind.STEP, 1000.0),
        swing=Swing.RESPONSE,
        grid=0.125,
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

TIPS = {
    "x": "<p>The node's position, x across.</p>",
    "y": "<p>The node's position, y up.</p>",
    "support": "<p>Which of the node's DOFs (u, v, θ) a support holds: pinned holds u and v, a roller one of them, "
    "clamped all three, guided one translation and the turn.</p>",
    "point_mass": "<p>A point mass at the node (0: none). It adds m to its u and v on M's diagonal.</p>",
    "point_inertia": "<p>The point mass's rotary inertia J about the node, on its θ.</p>",
    "torsion": "<p>A torsion spring from the node's turn to the ground (0: none), on its θ in K.</p>",
    "torsion_c": "<p>A torsion damper from the node's turn to the ground (0: none), on its θ in C.</p>",
    "kind": "<p>A beam bends and stretches (three DOFs a node); a bar only stretches: its ends are pinned.</p>",
    "material": "<p>Steel (210 GPa, 7850 kg/m³), aluminium (70 GPa, 2700) or titanium (114 GPa, 4430).</p>",
    "shape": "<p>The section: a rectangle, or a round tube (solid when its wall is half its diameter).</p>",
    "depth": "<p>The section's depth in the frame's plane (a tube's outer diameter): I grows as its cube.</p>",
    "width": "<p>The rectangle's width, out of the plane.</p>",
    "wall": "<p>The tube's wall thickness.</p>",
    "divide": "<p>Cut the member into this many equal members, adding nodes between.</p>",
    "value": "<p>The spring's stiffness or the damper's coefficient.</p>",
    "lumped": "<p>Consistent mass comes from the same shape functions as the stiffness (Rayleigh–Ritz: the "
    "frequencies are upper bounds); lumped puts half each member's mass at each end.</p>",
    "theory": "<p>Euler–Bernoulli beams bend only; Timoshenko's also shear, and their sections' turning has "
    "inertia, which matters for deep beams and high modes.</p>",
    "grid": "<p>The spacing new nodes snap to.</p>",
    "matrix": "<p>The matrix the Assembly and Element tabs show.</p>",
    "free_only": "<p>List only the free DOFs: the matrix the eigenproblem solves.</p>",
    "animate": "<p>Swing the frame (slowed to the same pace whatever its frequency, and exaggerated): a mode "
    "on its own, the response to the load from Start. Space.</p>",
    "swing": "<p>What the frame swings in: the undamped mode picked in the table; the state space's mode beside "
    "it, each node with its own phase; or the exact response to the load, slowed so its record lasts "
    f"{PLAY_S:g} s.</p>",
    "zeta": "<p>Rayleigh damping C = αM + βK, with this ζ at the first two modes (between them it's less, "
    "beyond them more). It's proportional: on its own it leaves the modes real. The dampers come on top.</p>",
    "load_kind": "<p>The load, from rest: a harmonic force F sin Ωt, an impulse (F × 1 s, in N·s), or a step F.</p>",
    "load_at": "<p>The node and the DOF the load acts on (a moment on a turn). The arrow on the canvas.</p>",
    "load_size": "<p>The load's size: N, N·m on a turn; an impulse's N·s.</p>",
    "load_hz": "<p>The harmonic force's frequency.</p>",
    "out_at": "<p>The node and the DOF whose response the Response tab plots (the ring on the canvas).</p>",
}
CANVAS_TIP = (
    "<p>The frame on its grid. Pick a tool above. Dotted: the frame at rest, while it swings in the mode picked. "
    "Numbers are the nodes'.</p>"
)
TABLE_TIP = (
    "<p>The modes: each one's frequency and how its strain energy splits between the members' stretching, their "
    "bending (and shear), and the springs. Click a row to show it.</p>"
)
FORMS_TIP = (
    "<p>Each undamped mode (its f and the conventional ζ = φᵀCφ/2ω) beside the state space's mode that pairs "
    "with it (|λ|/2π and −Re λ/|λ|), and how complex that is: MPC 1 for a real mode. Click a row to show it.</p>"
)
ASSEMBLY_TIP = (
    "<p>The global matrix over every DOF: tinted, the picked element's share Tᵀk'T; greyed and struck out, the "
    "DOFs a support holds or no element touches. A dot is an exact zero.</p>"
)

NOTES_HTML = load_notes("frame_builder", figures=frame_figures.FIGURES, trials=trials_html(TRIALS))


# ------------------------------------------------------------------ drawing helpers
NAN2 = np.full(2, np.nan)


def _zigzag(a: np.ndarray, b: np.ndarray, width: float, turns: int = 5) -> list:
    d = b - a
    length = float(np.hypot(*d))
    if length == 0:
        return [a, b]
    n = np.array([-d[1], d[0]]) / length
    s = np.linspace(0.2, 0.8, 2 * turns + 1)
    wiggle = np.array([0.0] + [(-1) ** i for i in range(1, 2 * turns)] + [0.0])
    return [a, *(a + np.outer(s, d) + np.outer(wiggle * width, n)), b]


def _dashpot(a: np.ndarray, b: np.ndarray, width: float) -> list:
    d = b - a
    length = float(np.hypot(*d))
    if length == 0:
        return [a, b]
    n = np.array([-d[1], d[0]]) / length
    bottom, mouth, plate = a + 0.35 * d, a + 0.65 * d, a + 0.5 * d
    return [
        a,
        bottom,
        NAN2,
        mouth + width * n,
        bottom + width * n,
        bottom - width * n,
        mouth - width * n,
        NAN2,
        plate + 0.75 * width * n,
        plate - 0.75 * width * n,
        NAN2,
        plate,
        b,
    ]


def _ground(at: np.ndarray, toward: np.ndarray, size: float) -> list:
    """A hatched ground line through `at`, across `toward` (a unit vector pointing into the ground)."""
    n = np.array([-toward[1], toward[0]])
    out = [at - 0.8 * size * n, at + 0.8 * size * n, NAN2]
    for s in np.linspace(-0.8, 0.8, 5):
        p = at + s * size * n
        out += [p, p + 0.35 * size * (toward - n), NAN2]
    return out


def _circle(c: np.ndarray, r: float) -> list:
    t = np.linspace(0.0, TWO_PI, 13)
    return [*(c + r * np.column_stack([np.cos(t), np.sin(t)])), NAN2]


def _away(frame: Frame, i: int) -> np.ndarray:
    """The axis direction pointing away from the elements on node i (down if there are none)."""
    p = frame.point(i)
    total = np.zeros(2)
    for e in frame.elements:
        if (e.kind.member or e.kind.link) and i in (e.i, e.j):
            other = frame.point(e.j if e.i == i else e.i) if e.j is not None else np.array(e.anchor)
            d = other - p
            norm = float(np.hypot(*d))
            if norm > 0:
                total += d / norm
    if float(np.hypot(*total)) < 1e-9:
        return np.array([0.0, -1.0])
    axis = int(np.argmax(np.abs(total)))
    out = np.zeros(2)
    out[axis] = -math.copysign(1.0, total[axis])
    return out


def _support(frame: Frame, i: int, at: np.ndarray, h: float) -> list:
    """The node's support symbol at `at`, size h, pointing into the ground."""
    s = frame.nodes[i].support
    if s is Support.FREE:
        return []
    if s is Support.NO_TURN:
        r = 0.45 * h
        return [at + [-r, -r], at + [r, -r], at + [r, r], at + [-r, r], at + [-r, -r], NAN2]
    if s in (Support.PINNED, Support.ROLLER_X, Support.GUIDED_X):
        down = np.array([0.0, -1.0])
    elif s in (Support.ROLLER_Y, Support.GUIDED_Y):
        side = _away(frame, i)
        down = side if side[0] != 0 else np.array([-1.0, 0.0])
    else:
        down = _away(frame, i)
    right = np.array([-down[1], down[0]])

    def p(x: float, y: float) -> np.ndarray:  # local: y negative into the ground
        return at + x * h * right - y * h * down

    if s is Support.CLAMPED:
        return [p(-0.9, 0), p(0.9, 0), NAN2] + _ground(p(0, 0), down, h)[3:]
    if s in (Support.GUIDED_X, Support.GUIDED_Y):
        return (
            [p(-0.7, 0), p(0.7, 0), NAN2]
            + _circle(p(-0.4, -0.22), 0.2 * h)
            + _circle(p(0.4, -0.22), 0.2 * h)
            + _ground(p(0, -0.44), down, h)
        )
    out = [p(0, 0), p(-0.55, -0.9), p(0.55, -0.9), p(0, 0), NAN2]
    if s is Support.PINNED:
        return out + _ground(p(0, -0.9), down, h)
    return out + _circle(p(-0.3, -1.08), 0.17 * h) + _circle(p(0.3, -1.08), 0.17 * h) + _ground(p(0, -1.26), down, h)


def _spiral(at: np.ndarray, h: float) -> list:
    t = np.linspace(0.0, 2.5 * TWO_PI, 60)
    r = h * (0.25 + 0.55 * t / t[-1])
    pts = at + np.column_stack([r * np.cos(t), r * np.sin(t)])
    end = pts[-1]
    return [*pts, end + [0.0, -0.4 * h], NAN2] + _ground(end + [0.0, -0.4 * h], np.array([0.0, -1.0]), 0.6 * h)


def _arrow(at: np.ndarray, dof: int, sign: float, h: float) -> list:
    """A load's arrow ending at `at`: along x or y, or round it for a moment (anticlockwise if sign > 0)."""
    if dof == 2:
        a = np.linspace(-0.6, 0.9, 16) * math.pi * sign
        r = 1.2 * h
        arc = [at + r * np.array([math.cos(t), math.sin(t)]) for t in a]
        end, tangent = arc[-1], np.array([-math.sin(a[-1]), math.cos(a[-1])]) * sign
        tip = [end - 0.5 * h * (tangent + 0.6 * np.array([-tangent[1], tangent[0]]) * k) for k in (1, -1)]
        return [*arc, NAN2, tip[0], end, tip[1], NAN2]
    d = np.zeros(2)
    d[dof] = 1.0 if sign >= 0 else -1.0
    side = np.array([-d[1], d[0]])
    tail = at - 2.5 * h * d
    return [tail, at, NAN2, at - 0.6 * h * d + 0.35 * h * side, at, at - 0.6 * h * d - 0.35 * h * side, NAN2]


def _segments(points: list) -> tuple[np.ndarray, np.ndarray]:
    if not points:
        return np.array([]), np.array([])
    a = np.array([np.asarray(p, dtype=float) for p in points])
    return a[:, 0], a[:, 1]


def _segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    d = b - a
    t = float(np.clip((p - a) @ d / max(float(d @ d), 1e-30), 0.0, 1.0))
    return float(np.hypot(*(p - a - t * d)))


# ------------------------------------------------------------------ the canvas
class FrameCanvas(pg.PlotWidget):
    """The frame on its grid; builds it with the tool picked. Emits `edited(frame, picked)` for every change and
    `picked(selection)` when the selection changes: ("node", i), ("element", k) or None."""

    edited = QtCore.Signal(object, object)
    selected = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setToolTip(CANVAS_TIP)
        self.setAspectLocked(True)
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.hideButtons()
        self.showGrid(x=True, y=True, alpha=0.25)
        for side in ("left", "bottom"):
            self.getAxis(side).enableAutoSIPrefix(False)
        self.setLabel("bottom", "x", units="m")
        self.setLabel("left", "y", units="m")
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.viewport().setMouseTracking(True)
        self.setMouseTracking(True)
        self.frame = Frame()
        self.tool = Tool.SELECT
        self.support = Support.PINNED
        self.grid = 0.25
        self.selection: tuple[str, int] | None = None
        self.pending: int | None = None  # the first node of a two-click tool
        self.cursor: np.ndarray | None = None
        self.q: np.ndarray | None = None  # the displacement drawn
        self.load: Load | None = None  # drawn as an arrow
        self.probe: int | None = None  # the DOF whose response is plotted: a square round its node
        self.unit = 0.1  # m: symbols' size, from the view's
        self._drag: int | None = None
        self._dragged: Frame | None = None
        self.template = {  # what a new element of each kind is made as: the last one edited
            Kind.BEAM: Element(Kind.BEAM, 0, 1),
            Kind.BAR: Element(Kind.BAR, 0, 1),
            Kind.SPRING: Element(Kind.SPRING, 0, 1, value=100e3),
            Kind.DAMPER: Element(Kind.DAMPER, 0, 1, value=200.0),
            Kind.TORSION_SPRING: Element(Kind.TORSION_SPRING, 0, value=10e3),
            Kind.MASS: Element(Kind.MASS, 0, value=10.0),
        }
        plot = self.getPlotItem()
        self.ghost = pg.PlotCurveItem(connect="finite")
        self.bars = pg.PlotCurveItem(connect="finite")
        self.beams = pg.PlotCurveItem(connect="finite")
        self.links = pg.PlotCurveItem(connect="finite")
        self.supports = pg.PlotCurveItem(connect="finite")
        self.highlight = pg.PlotCurveItem(connect="finite")
        self.rubber = pg.PlotCurveItem()
        self.masses = pg.PlotDataItem(pen=None, symbol="s", symbolSize=15)
        self.nodes = pg.PlotDataItem(pen=None, symbol="o", symbolSize=8)
        self.ring = pg.PlotDataItem(pen=None, symbol="o", symbolSize=16)
        self.arrow = pg.PlotCurveItem(connect="finite")
        self.probe_mark = pg.PlotDataItem(pen=None, symbol="s", symbolSize=18)
        for item in (
            self.ghost,
            self.supports,
            self.links,
            self.bars,
            self.beams,
            self.highlight,
            self.rubber,
            self.masses,
            self.nodes,
            self.ring,
            self.arrow,
            self.probe_mark,
        ):
            plot.addItem(item)
        self.labels: list[pg.TextItem] = []
        self.apply_theme()

    # --------------------------------------------------------- state
    def set_frame(self, frame: Frame, fit: bool = True) -> None:
        self.frame = frame
        if self.pending is not None and self.pending >= len(frame.nodes):
            self.pending = None
        if fit:
            self.fit()
        self.draw()

    def set_tool(self, tool: Tool) -> None:
        self.tool = tool
        self.pending = None
        self.draw()

    def fit(self) -> None:
        x0, x1, y0, y1 = self.frame.extent()
        if not self.frame.nodes:
            x0, x1, y0, y1 = 0.0, 3.0, 0.0, 2.0
        size = max(x1 - x0, y1 - y0, 4 * self.grid)
        pad = max(0.25 * size, 2 * self.grid)
        self.setRange(xRange=(x0 - pad, x1 + pad), yRange=(y0 - pad, y1 + pad), padding=0)
        self.unit = 0.035 * (size + 2 * pad)

    def snap(self, p: np.ndarray) -> np.ndarray:
        g = self.grid
        return np.round(p / g) * g

    def _pixel(self) -> float:
        sx, sy = self.getPlotItem().getViewBox().viewPixelSize()
        return max(sx, sy)

    def node_near(self, p: np.ndarray) -> int | None:
        best, reach = None, NODE_PX * self._pixel()
        for i in range(len(self.frame.nodes)):
            d = float(np.hypot(*(self.frame.point(i) - p)))
            if d <= reach:
                best, reach = i, d
        return best

    def element_near(self, p: np.ndarray) -> int | None:
        best, reach = None, ELEMENT_PX * self._pixel()
        f = self.frame
        for k, e in enumerate(f.elements):
            if not (e.kind.member or e.kind.link):
                continue
            b = f.point(e.j) if e.j is not None else np.array(e.anchor)
            d = _segment_distance(p, f.point(e.i), b)
            if d <= reach:
                best, reach = k, d
        return best

    def select(self, selection: tuple[str, int] | None) -> None:
        self.selection = selection
        self.draw()
        self.selected.emit(selection)

    # --------------------------------------------------------- editing
    def _commit(self, frame: Frame, selection: tuple[str, int] | None) -> None:
        self.selection = selection
        self.edited.emit(frame, selection)

    def _node(self, frame: Frame, p: np.ndarray) -> tuple[Frame, int | None]:
        """The node at p (snapped), made if there isn't one; None if the frame is full."""
        found = self.node_near(p)
        if found is None:
            found = frame.node_at(*p)
        if found is not None:
            return frame, found
        if len(frame.nodes) >= MAX_NODES:
            return frame, None
        return frame.with_node(float(p[0]), float(p[1])), len(frame.nodes)

    def press(self, p: np.ndarray, right: bool = False) -> None:
        """A click at p (view coordinates): what the tool does."""
        if right:
            self.pending = None
            self.draw()
            return
        f = self.frame
        hit = self.node_near(p)
        s = self.snap(p)
        tool = self.tool
        if tool is Tool.SELECT:
            if hit is not None:
                self._drag, self._dragged = hit, None
                self.select(("node", hit))
            else:
                k = self.element_near(p)
                self.select(("element", k) if k is not None else None)
        elif tool is Tool.NODE:
            if hit is not None:
                self.select(("node", hit))
            elif len(f.nodes) < MAX_NODES:
                self._commit(f.with_node(float(s[0]), float(s[1])), ("node", len(f.nodes)))
        elif tool in (Tool.BEAM, Tool.BAR, Tool.SPRING, Tool.DAMPER):
            kind = {Tool.BEAM: Kind.BEAM, Tool.BAR: Kind.BAR, Tool.SPRING: Kind.SPRING, Tool.DAMPER: Kind.DAMPER}[tool]
            if self.pending is None:
                f, i = self._node(f, s if hit is None else f.point(hit))
                if i is None:
                    return
                self.pending = i
                if f is not self.frame:
                    self._commit(f, ("node", i))
                else:
                    self.draw()
                return
            a = self.pending
            if kind.link and hit is None:  # to the ground at s
                if float(np.hypot(*(s - f.point(a)))) > 0:
                    e = dataclasses.replace(self.template[kind], i=a, j=None, anchor=(float(s[0]), float(s[1])))
                    self.pending = None
                    self._commit(f.with_element(e), ("element", len(f.elements)))
                return
            f, b = self._node(f, s if hit is None else f.point(hit))
            if b is None or b == a:
                return
            if any({e.i, e.j} == {a, b} and e.kind is kind for e in f.elements):
                self.pending = b if kind.member else None
                self.draw()
                return
            e = dataclasses.replace(self.template[kind], i=a, j=b, anchor=None)
            self.pending = b if kind.member else None
            self._commit(f.with_element(e), ("element", len(f.elements)))
        elif hit is not None and tool in (Tool.TORSION, Tool.MASS):
            kind = Kind.MASS if tool is Tool.MASS else Kind.TORSION_SPRING
            if not any(e.kind is kind and e.i == hit and e.j is None for e in f.elements):
                f = f.with_element(dataclasses.replace(self.template[kind], i=hit, j=None))
                self._commit(f, ("node", hit))
            else:
                self.select(("node", hit))
        elif hit is not None and tool is Tool.SUPPORT:
            node = f.nodes[hit]
            support = Support.FREE if node.support is self.support else self.support
            self._commit(f.replace_node(hit, dataclasses.replace(node, support=support)), ("node", hit))

    def move(self, p: np.ndarray) -> None:
        self.cursor = p
        if self._drag is not None:
            s = self.snap(p)
            other = self.frame.node_at(*s)
            if other is None or other == self._drag:
                node = self.frame.nodes[self._drag]
                self._dragged = self.frame.replace_node(
                    self._drag, dataclasses.replace(node, x=float(s[0]), y=float(s[1]))
                )
                self.draw(self._dragged)
        elif self.pending is not None:
            self.draw()

    def release(self) -> None:
        if self._drag is not None and self._dragged is not None:
            self._commit(self._dragged, ("node", self._drag))
        self._drag, self._dragged = None, None

    def delete_selected(self) -> None:
        if self.selection is None:
            return
        what, index = self.selection
        f = self.frame.without_node(index) if what == "node" else self.frame.without_element(index)
        self.pending = None
        self._commit(f, None)

    def _view_point(self, event) -> np.ndarray:
        scene = self.mapToScene(event.position().toPoint())
        p = self.getPlotItem().getViewBox().mapSceneToView(scene)
        return np.array([p.x(), p.y()])

    def mousePressEvent(self, event) -> None:
        self.setFocus()
        self.press(self._view_point(event), right=event.button() == QtCore.Qt.MouseButton.RightButton)
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        self.move(self._view_point(event))
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self.release()
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        event.accept()

    def wheelEvent(self, event) -> None:
        event.ignore()  # the page scrolls; the view fits the frame

    def keyPressEvent(self, event) -> None:
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.pending = None
            self.draw()
        elif event.key() in (QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace):
            self.delete_selected()
        else:
            super().keyPressEvent(event)

    # --------------------------------------------------------- drawing
    def draw(self, frame: Frame | None = None) -> None:
        f = frame if frame is not None else self.frame
        q = self.q if frame is None and self.q is not None and self.q.size == f.n_dof else None
        h = self.unit

        def at(i: int) -> np.ndarray:
            p = f.point(i)
            return p + q[3 * i : 3 * i + 2] if q is not None else p

        bars: list[np.ndarray] = []
        beams: list[np.ndarray] = []
        links: list[np.ndarray] = []
        ghost: list[np.ndarray] = []
        supports: list[np.ndarray] = []
        for k, e in enumerate(f.elements):
            if e.kind.member and e.j is not None:
                if q is not None:
                    x, y = member_curve(f, k, q)
                    line = [*np.column_stack([x, y]), NAN2]
                    ghost += [f.point(e.i), f.point(e.j), NAN2]
                else:
                    line = [f.point(e.i), f.point(e.j), NAN2]
                (beams if e.kind is Kind.BEAM else bars).extend(line)
            elif e.kind.link:
                a = at(e.i)
                b = at(e.j) if e.j is not None else np.array(e.anchor)
                shape = _zigzag(a, b, 0.3 * h) if e.kind is Kind.SPRING else _dashpot(a, b, 0.3 * h)
                links += [*shape, NAN2]
                if e.j is None:
                    d = b - a
                    links += _ground(b, d / float(np.hypot(*d)), 0.7 * h)
            elif e.kind is Kind.TORSION_SPRING and e.j is None:
                links += _spiral(at(e.i), 0.8 * h)
            elif e.kind is Kind.TORSION_DAMPER and e.j is None:
                links += _circle(at(e.i), 0.6 * h) + _ground(at(e.i) + [0.0, -0.6 * h], np.array([0.0, -1.0]), 0.6 * h)
        for i in range(len(f.nodes)):
            supports += _support(f, i, at(i), h)
        self.bars.setData(*_segments(bars))
        self.beams.setData(*_segments(beams))
        self.links.setData(*_segments(links))
        self.ghost.setData(*_segments(ghost))
        self.supports.setData(*_segments(supports))
        points = np.array([at(i) for i in range(len(f.nodes))]).reshape(-1, 2)
        self.nodes.setData(points[:, 0], points[:, 1])
        heavy = sorted({e.i for e in f.elements if e.kind is Kind.MASS and e.value > 0})
        self.masses.setData(points[heavy, 0] if heavy else [], points[heavy, 1] if heavy else [])
        self._label(points)
        # the selection, and the line from the pending node to the cursor
        lit: list = []
        ring: list[int] = []
        if self.selection is not None:
            what, index = self.selection
            if what == "node" and index < len(f.nodes):
                ring = [index]
            elif what == "element" and index < len(f.elements):
                e = f.elements[index]
                if e.kind.member and e.j is not None:
                    x, y = (
                        member_curve(f, index, q)
                        if q is not None
                        else (np.array([at(e.i)[0], at(e.j)[0]]), np.array([at(e.i)[1], at(e.j)[1]]))
                    )
                    lit = [*np.column_stack([x, y]), NAN2]
                elif e.kind.link:
                    b = at(e.j) if e.j is not None else np.array(e.anchor)
                    lit = [at(e.i), b, NAN2]
        self.highlight.setData(*_segments(lit))
        self.ring.setData([at(i)[0] for i in ring], [at(i)[1] for i in ring])
        load = self.load
        arrow = _arrow(at(load.node), load.dof, load.size, h) if load is not None and load.node < len(f.nodes) else []
        self.arrow.setData(*_segments(arrow))
        probe = self.probe if self.probe is not None and self.probe // 3 < len(f.nodes) else None
        self.probe_mark.setData(*(([at(probe // 3)[0]], [at(probe // 3)[1]]) if probe is not None else ([], [])))
        if self.pending is not None and self.cursor is not None and self.pending < len(f.nodes):
            hit = self.node_near(self.cursor)
            end = f.point(hit) if hit is not None else self.snap(self.cursor)
            a = f.point(self.pending)
            self.rubber.setData([a[0], end[0]], [a[1], end[1]])
        else:
            self.rubber.setData([], [])

    def _label(self, points: np.ndarray) -> None:
        while len(self.labels) < len(points):
            label = pg.TextItem(anchor=(0, 1))
            label.setColor(colors.muted)
            self.getPlotItem().addItem(label, ignoreBounds=True)
            self.labels.append(label)
        offset = 0.25 * self.unit
        for i, label in enumerate(self.labels):
            if i < len(points):
                label.setText(str(i + 1))
                label.setPos(points[i, 0] + offset, points[i, 1] + offset)
                label.show()
            else:
                label.hide()

    def apply_theme(self) -> None:
        c = colors
        self.ghost.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self.bars.setPen(pg.mkPen(c.mass[1], width=2))
        self.beams.setPen(pg.mkPen(c.mass[0], width=4))
        self.links.setPen(pg.mkPen(c.structure, width=1.5))
        self.supports.setPen(pg.mkPen(c.structure, width=1.5))
        self.highlight.setPen(pg.mkPen(c.force, width=3))
        self.rubber.setPen(pg.mkPen(c.grey, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        self.masses.setSymbolBrush(pg.mkBrush(c.structure))
        self.masses.setSymbolPen(None)
        self.nodes.setSymbolBrush(pg.mkBrush(c.background))
        self.nodes.setSymbolPen(pg.mkPen(c.structure, width=1.5))
        self.ring.setSymbolBrush(None)
        self.ring.setSymbolPen(pg.mkPen(c.force, width=2.5))
        self.arrow.setPen(pg.mkPen(c.force, width=2.5))
        self.probe_mark.setSymbolBrush(None)
        self.probe_mark.setSymbolPen(pg.mkPen(c.strong, width=1.5))
        for label in self.labels:
            label.setColor(c.muted)
        self.draw()


# ------------------------------------------------------------------ the matrices as tables
def _blend(a: str, b: str, t: float) -> str:
    """Colour a mixed with t of colour b, as #rrggbb (rich text has no alpha)."""
    ra, rb = rgb(a), rgb(b)
    return "#" + "".join(f"{round((1 - t) * x + t * y):02x}" for x, y in zip(ra, rb))


def _scale(A: np.ndarray) -> int:
    top = float(np.abs(A).max()) if A.size else 0.0
    return math.floor(math.log10(top)) if top > 0 else 0


def _entry(v: float, scale: int, top: float) -> str:
    if v == 0 or abs(v) < 1e-12 * top:
        return "·"
    return f"{v / 10.0**scale:.3g}".replace("-", "−")


def _power(scale: int) -> str:
    return "" if scale == 0 else f"×10<sup>{scale}</sup>".replace("-", "−")


def matrix_html(
    A: np.ndarray,
    rows: list[str],
    cols: list[str],
    tint: np.ndarray | None = None,
    struck_rows: np.ndarray | None = None,
    struck_cols: np.ndarray | None = None,
    title: str = "",
) -> str:
    """A matrix as a rich-text table, its common power of ten in the corner; `tint` (bool, A's shape) tints
    cells, struck rows and columns are greyed and struck out."""
    scale = _scale(A)
    top = float(np.abs(A).max()) if A.size else 0.0
    lit, dim = _blend(colors.background, colors.force, 0.22), _blend(colors.background, colors.grey, 0.18)
    sr = struck_rows if struck_rows is not None else np.zeros(len(rows), dtype=bool)
    sc = struck_cols if struck_cols is not None else np.zeros(len(cols), dtype=bool)

    def head(text: str, struck: bool) -> str:
        if struck:
            return f"<th style='color:{colors.grey}; white-space:nowrap'><s>{text}</s></th>"
        return f"<th style='white-space:nowrap'>{text}</th>"

    out = [
        "<table border='1' cellspacing='0' cellpadding='2' style='border-color:" + colors.cell_border + "'>",
        f"<tr><th>{title} {_power(scale)}</th>" + "".join(head(c, bool(s)) for c, s in zip(cols, sc)) + "</tr>",
    ]
    for r, label in enumerate(rows):
        cells = []
        for c in range(len(cols)):
            text = _entry(float(A[r, c]), scale, top)
            struck = bool(sr[r] or sc[c])
            style = f" style='color:{colors.grey}; white-space:nowrap'" if struck or text == "·" else ""
            if struck and text != "·":
                text = f"<s>{text}</s>"
            bg = f" bgcolor='{lit}'" if tint is not None and tint[r, c] else f" bgcolor='{dim}'" if struck else ""
            nowrap = "" if style else " style='white-space:nowrap'"
            cells.append(f"<td align='right'{nowrap}{bg}{style}>{text}</td>")
        out.append("<tr>" + head(label, bool(sr[r])) + "".join(cells) + "</tr>")
    out.append("</table>")
    return "".join(out)


def element_html(frame: Frame, assembly: Assembly, k: int, matrix: Matrix) -> str:
    """Element k: what it is, its own matrix, its T, and Tᵀ(matrix)T with the global DOFs it lands on."""
    e = frame.elements[k]
    part = assembly.locals[k]
    name = f"{e.kind.value} {k + 1}"
    ends = f"node {e.i + 1}" + (f" to node {e.j + 1}" if e.j is not None else " to the ground")
    lines = [f"<p><b>{name}</b>: {ends}."]
    if e.kind.member or e.kind.link:
        c, s = math.cos(part.angle), math.sin(part.angle)
        lines.append(
            f" L = {fmt(part.length, 'm')}, γ = {math.degrees(part.angle):.4g}° (cos γ = {c:.3f}, sin γ = {s:.3f})."
        )
    if e.kind.member:
        sec, mat = e.section, e.material
        lines.append(
            f" {mat.name}, A = {number(sec.area * 1e6, 4)} mm², I = {number(sec.inertia * 1e12, 4)} mm⁴: EA/L = "
            f"{fmt(mat.E * sec.area / part.length, 'N/m')}"
        )
        if e.kind is Kind.BEAM:
            lines.append(f", EI/L³ = {fmt(mat.E * sec.inertia / part.length**3, 'N/m')}")
            if frame.theory is Theory.TIMOSHENKO:
                lines.append(f", φ<sub>s</sub> = 12EI/κ<sub>s</sub>GAL² = {part.phi:.3g}")
        lines.append(".")
    elif e.kind is Kind.MASS:
        lines.append(f" m = {fmt(e.value, 'kg')}, J = {e.inertia:.4g} kg·m².")
    else:
        unit = {Kind.SPRING: "N/m", Kind.DAMPER: "N·s/m", Kind.TORSION_SPRING: "N·m/rad"}.get(e.kind, "N·m·s/rad")
        lines.append(f" {'c' if e.kind.damper else 'k'} = {fmt(e.value, unit)}.")
    lines.append("</p>")
    own = {Matrix.K: part.k, Matrix.M: part.m, Matrix.C: part.c}[matrix]
    whole = part.T.T @ own @ part.T
    sym = {Matrix.K: "k'", Matrix.M: "m'", Matrix.C: "c'"}[matrix]
    glob = [dof_label(d) for d in part.dofs]
    tables = [
        f"<p>In its own axes ({', '.join(part.local)}):</p>",
        matrix_html(own, list(part.local), list(part.local), title=sym),
    ]
    if not np.allclose(part.T, np.eye(*part.T.shape)):
        tables += [
            "<p>d = T q<sub>e</sub>, from the global DOFs to its own:</p>",
            matrix_html(part.T, list(part.local), glob, title="T"),
        ]
    tables += [
        f"<p>Tᵀ{sym}T, added to {matrix.name} at rows and columns {', '.join(glob)} (tinted on the Assembly tab):</p>",
        matrix_html(whole, glob, glob, title=f"Tᵀ{sym}T"),
    ]
    return "".join(lines) + "".join(tables)


def node_html(frame: Frame, assembly: Assembly, i: int) -> str:
    node = frame.nodes[i]
    dofs = [3 * i + d for d in range(3)]
    names = ", ".join(dof_label(d) for d in dofs)
    held = [dof_label(d) for d in dofs if assembly.fixed[d]]
    unused = [dof_label(d) for d in dofs if assembly.unused[d]]
    out = [f"<p><b>Node {i + 1}</b> at ({node.x:.4g}, {node.y:.4g}) m: DOFs {names}, rows {dofs[0] + 1}–{dofs[2] + 1}."]
    if held:
        out.append(f" {node.support.value}: {', '.join(held)} held, struck out.")
    if unused:
        out.append(f" No element touches {', '.join(unused)}: struck out.")
    on = [k for k, e in enumerate(frame.elements) if i in (e.i, e.j)]
    if on:
        out.append(" On it: " + ", ".join(f"{frame.elements[k].kind.value.lower()} {k + 1}" for k in on) + ".")
    out.append("</p><p>Pick an element (the Select tool) to see its matrices.</p>")
    return "".join(out)


# ------------------------------------------------------------------ the page
class Properties(QtWidgets.QGroupBox):
    """The selected item's settings: a node's, a member's or a spring's (or nothing). One shows at a time."""

    def __init__(self) -> None:
        super().__init__("Selected")
        v = QtWidgets.QVBoxLayout(self)
        self.empty = note_label("Nothing picked: pick the Select tool and click a node or an element.")
        v.addWidget(self.empty)

        self.node = Form("")
        self.node.setFlat(True)
        n = self.node
        self.node_x = n.number("x:", -100.0, 100.0, 0.0, 3, " m", TIPS["x"])
        self.node_y = n.number("y:", -100.0, 100.0, 0.0, 3, " m", TIPS["y"])
        self.support = combo([s.value for s in SUPPORTS], TIPS["support"])
        n.add("Support:", self.support, TIPS["support"])
        self.mass = n.number("Point mass m:", 0.0, 1e6, 0.0, 3, " kg", TIPS["point_mass"])
        self.inertia = n.number("…its J:", 0.0, 1e5, 0.0, 4, " kg·m²", TIPS["point_inertia"])
        self.torsion = n.number("Torsion spring:", 0.0, 1e9, 0.0, 3, " kN·m/rad", TIPS["torsion"])
        self.torsion_c = n.number("Torsion damper:", 0.0, 1e9, 0.0, 3, " N·m·s/rad", TIPS["torsion_c"])
        v.addWidget(self.node)

        self.member = Form("")
        self.member.setFlat(True)
        m = self.member
        self.kind = combo([Kind.BEAM.value, Kind.BAR.value], TIPS["kind"])
        m.add("Kind:", self.kind, TIPS["kind"])
        self.material = combo([x.name for x in MATERIALS], TIPS["material"])
        m.add("Material:", self.material, TIPS["material"])
        self.shape = combo([s.value for s in SHAPES], TIPS["shape"])
        m.add("Section:", self.shape, TIPS["shape"])
        self.section_depth = m.number("Depth (in plane):", 0.1, 5000.0, 50.0, 1, " mm", TIPS["depth"])
        self.section_width = m.number("Width:", 0.1, 5000.0, 50.0, 1, " mm", TIPS["width"])
        self.wall = m.number("Wall:", 0.1, 2500.0, 25.0, 1, " mm", TIPS["wall"])
        self.member_info = note_label()
        m.row(self.member_info)
        self.parts = QtWidgets.QSpinBox()
        self.parts.setRange(2, 16)
        self.parts.setValue(2)
        self.divide = QtWidgets.QPushButton("Divide")
        self.divide.setToolTip(TIPS["divide"])
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Into:"))
        row.addWidget(self.parts)
        row.addWidget(self.divide, 1)
        m.row(row)
        v.addWidget(self.member)

        self.link = Form("")
        self.link.setFlat(True)
        self.value = self.link.number("Stiffness k:", 0.0, 1e12, 100.0, 3, " kN/m", TIPS["value"])
        self.link_info = note_label()
        self.link.row(self.link_info)
        v.addWidget(self.link)

        self.remove = QtWidgets.QPushButton("Delete")
        self.remove.setToolTip("<p>Remove the picked item (a node takes its elements with it). Del does it too.</p>")
        v.addWidget(self.remove)
        self.show_form(self.empty)

    def show_form(self, form: QtWidgets.QWidget) -> None:
        for w in (self.empty, self.node, self.member, self.link):
            w.setVisible(w is form)

    def current(self) -> QtWidgets.QWidget:
        return next(w for w in (self.empty, self.node, self.member, self.link) if not w.isHidden())


def default_load(frame: Frame, assembly: Assembly | None = None) -> Load:
    """An impulse along y at the last node free to move so (else on the first free DOF)."""
    free = (assembly if assembly is not None else assemble(frame)).free
    for i in reversed(range(len(frame.nodes))):
        if free[3 * i + 1]:
            return Load(node=i, dof=1)
    dofs = np.flatnonzero(free)
    return Load(node=int(dofs[0]) // 3, dof=int(dofs[0]) % 3) if dofs.size else Load()


def _percent(zeta: float) -> str:
    """ζ in %, an axial mode's next to nothing said so rather than in full."""
    return f"{100 * zeta:.3g} %" if zeta >= 1e-4 else "< 0.01 %"


def _reach(q: np.ndarray) -> float:
    """The largest node translation in q (complex or real; rows of every DOF, or one column)."""
    return float(np.max(np.hypot(np.abs(q[0::3]), np.abs(q[1::3])), initial=0.0))


class FramePage(QtWidgets.QWidget):
    """Controls on the left, the canvas and its modes in the centre, the matrices on the right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.frame = Frame()
        self.assembly: Assembly = assemble(self.frame)
        self.result: Modes | None = None
        self.error = ""
        self.mode = 0
        self.picked: int | None = None  # the element on the Element tab
        self.load = Load()
        self.out = self.load.index
        self.forms: Formulations | None = None  # worked out while the Response or Formulations tab, or a
        self.runs: Responses | None = None  # damped swing, needs them (None: not yet, since the last change)
        self.response_error = ""
        self.running = True  # a mode swings on its own; the response waits for Start (_on_swing)
        self._phase = math.pi / 2  # a mode starts at its full swing
        self._played = 0.0  # s into the response's record
        self._history: list[Frame] = []
        self._future: list[Frame] = []

        self.trials = TrialBox(
            [t.name for t in TRIALS],
            "<p>Each loads a frame, to show one idea from the Theory tab. It's then yours to change.</p>",
        )
        self.trials.picked.connect(self.load_trial)

        self.props = Properties()
        p = self.props
        for box in (p.node_x, p.node_y, p.mass, p.inertia, p.torsion, p.torsion_c):
            box.editingFinished.connect(self._on_node_box)
        p.support.currentIndexChanged.connect(self._on_node_box)
        for box in (p.section_depth, p.section_width, p.wall):
            box.editingFinished.connect(self._on_member_box)
        for choice in (p.kind, p.material, p.shape):
            choice.currentIndexChanged.connect(self._on_member_box)
        p.value.editingFinished.connect(self._on_link_box)
        p.divide.clicked.connect(self._on_divide)
        p.remove.clicked.connect(weak.slot(self, lambda self: self.canvas.delete_selected()))

        model = Form("Model")
        self.lumped = combo(["Consistent", "Lumped"], TIPS["lumped"])
        model.add("Mass:", self.lumped, TIPS["lumped"])
        self.theory = combo([t.value for t in THEORIES], TIPS["theory"])
        model.add("Beams:", self.theory, TIPS["theory"])
        self.zeta = model.number("Damping ζ:", 0.0, 50.0, 0.0, 2, " %", TIPS["zeta"])
        self.grid = combo([f"{g:g} m" for g in GRIDS], TIPS["grid"])
        model.add("Grid:", self.grid, TIPS["grid"])
        self.lumped.currentIndexChanged.connect(self._on_model)
        self.theory.currentIndexChanged.connect(self._on_model)
        self.zeta.valueChanged.connect(self._on_model)
        self.grid.currentIndexChanged.connect(self._on_grid)

        loading = Form("Load")
        self.load_kind = combo([k.value for k in LOAD_KINDS], TIPS["load_kind"])
        loading.add("Kind:", self.load_kind, TIPS["load_kind"])
        self.load_node, self.load_dof = self._dof_boxes(loading, "At node:", TIPS["load_at"])
        self.load_size = loading.number("Size:", -1e7, 1e7, 100.0, 3, " N", TIPS["load_size"])
        self.load_hz = loading.number("Frequency:", 0.01, 1e5, 10.0, 3, " Hz", TIPS["load_hz"])
        self.out_node, self.out_dof = self._dof_boxes(loading, "Response at:", TIPS["out_at"])
        self.load_form = loading
        for choice in (self.load_kind, self.load_dof, self.out_dof):
            choice.currentIndexChanged.connect(self._on_load)
        for field in (self.load_node, self.out_node, self.load_size, self.load_hz):
            field.valueChanged.connect(self._on_load)

        show = Form("Show")
        self.matrix = combo([m.value for m in MATRICES], TIPS["matrix"])
        show.add("Matrix:", self.matrix, TIPS["matrix"])
        self.free_only = QtWidgets.QCheckBox("Free DOFs only")
        self.free_only.setToolTip(TIPS["free_only"])
        show.row(self.free_only)
        self.swing = combo([w.value for w in SWINGS], TIPS["swing"])
        show.add("Swing:", self.swing, TIPS["swing"])
        self.run_button = RunButton()
        self.run_button.set_running(True)
        self.run_button.setToolTip(TIPS["animate"])
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.setToolTip("<p>Back to the start of the swing; the response waits for Start.</p>")
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.reset_button)
        show.row(buttons)
        self.matrix.currentIndexChanged.connect(self._draw_tabs)
        self.free_only.toggled.connect(self._draw_tabs)
        self.run_button.toggled.connect(self._on_animate)
        self.reset_button.clicked.connect(self.reset)
        self.swing.currentIndexChanged.connect(self._on_swing)

        # --- centre: the tools, the canvas, the modes
        self.canvas: FrameCanvas = FrameCanvas()
        self.canvas.edited.connect(self._on_edit)
        self.canvas.selected.connect(self._on_selected)
        tools = QtWidgets.QWidget()
        flow = FlowLayout(tools)
        flow.setContentsMargins(0, 0, 0, 0)
        self.tool_buttons = QtWidgets.QButtonGroup(self)
        for k, tool in enumerate(TOOLS):
            b = QtWidgets.QToolButton()
            b.setText(tool.value)
            b.setCheckable(True)
            b.setToolTip(f"<p>{TOOL_HINTS[tool]}</p>")
            self.tool_buttons.addButton(b, k)
            flow.add(b)
        self.tool_buttons.button(0).setChecked(True)
        self.tool_buttons.idClicked.connect(self._on_tool)
        self.support_kind = combo([s.value for s in SUPPORTS[1:]], TIPS["support"])
        self.support_kind.currentIndexChanged.connect(self._on_support_kind)
        self.undo_button = QtWidgets.QToolButton()
        self.undo_button.setText("Undo")
        self.undo_button.setToolTip("<p>Undo the last change (Ctrl+Z).</p>")
        self.redo_button = QtWidgets.QToolButton()
        self.redo_button.setText("Redo")
        self.redo_button.setToolTip("<p>Redo it (Ctrl+Shift+Z).</p>")
        self.clear_button = QtWidgets.QToolButton()
        self.clear_button.setText("Clear")
        self.clear_button.setToolTip("<p>Start again from an empty grid (Undo brings it back).</p>")
        self.undo_button.clicked.connect(self.undo)
        self.redo_button.clicked.connect(self.redo)
        self.clear_button.clicked.connect(
            weak.slot(
                self,
                lambda self: self._set_frame(
                    Frame(lumped=self.frame.lumped, theory=self.frame.theory, zeta=self.frame.zeta)
                ),
            )
        )
        flow.add(group(self.support_kind), self.undo_button, self.redo_button, self.clear_button)
        for keys, slot in (("Ctrl+Z", self.undo), ("Ctrl+Shift+Z", self.redo), ("Ctrl+Y", self.redo)):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(keys), self)
            shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)
        self.hint = note_label(TOOL_HINTS[Tool.SELECT])

        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setToolTip(TABLE_TIP)
        self.table.setHorizontalHeaderLabels(["Mode", "f (Hz)", "Axial", "Bending", "Springs"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._on_row)
        self.readout = readout_label()

        # --- right
        self.assembly_view = QtWidgets.QTextBrowser()
        self.assembly_view.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.NoWrap)
        self.assembly_view.setToolTip(ASSEMBLY_TIP)
        self.assembly_note = note_label()
        self.element_view = QtWidgets.QTextBrowser()
        self.element_view.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.NoWrap)
        self.notes = theory(NOTES_HTML)

        self.response_note = note_label()
        self.frf_mag = plot("|H|", None, "Frequency", "Hz", log_x=True, log_y=True)
        self.frf_phase = plot("Phase", "°", "Frequency", "Hz", log_x=True)
        self.time_plot = plot("Displacement", "m", "Time", "s")
        self.frf_legend = add_legend(self.frf_mag, offset=(-10, 10))
        self.playhead = pg.InfiniteLine(angle=90)
        self.time_plot.addItem(self.playhead, ignoreBounds=True)
        self.drive_line = pg.InfiniteLine(angle=90)
        self.frf_mag.addItem(self.drive_line, ignoreBounds=True)
        self.curves = {name: (p.plot(), p.plot()) for name, p in (("mag", self.frf_mag), ("phase", self.frf_phase))}
        self.curves["time"] = (self.time_plot.plot(), self.time_plot.plot())
        self.frf_legend.addItem(self.curves["mag"][0], "State space (exact)")
        self.frf_legend.addItem(self.curves["mag"][1], "Conventional (real modes)")

        self.forms_note = note_label()
        self.forms_table = QtWidgets.QTableWidget(0, 6)
        self.forms_table.setToolTip(FORMS_TIP)
        self.forms_table.setHorizontalHeaderLabels(["Mode", "f (Hz)", "ζ", "f exact", "ζ exact", "MPC"])
        self.forms_table.verticalHeader().setVisible(False)
        self.forms_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.forms_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.forms_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.forms_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.forms_table.itemSelectionChanged.connect(self._on_forms_row)
        self.argand = plot("Im ψ", None, "Re ψ", None, height=200)
        self.argand.setAspectLocked(True)
        self.argand_lines = pg.PlotCurveItem(connect="finite")
        self.argand_dots = pg.PlotDataItem(pen=None, symbol="o", symbolSize=7)
        self.argand.addItem(self.argand_lines)
        self.argand.addItem(self.argand_dots)
        self.argand_note = note_label()

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(column(self.assembly_note, self.assembly_view, stretch=False), "Assembly")
        self.tabs.addTab(self.element_view, "Element")
        self.tabs.addTab(
            column(self.response_note, self.frf_mag, self.frf_phase, self.time_plot, stretch=False), "Response"
        )
        self.tabs.addTab(
            column(self.forms_note, self.forms_table, self.argand_note, self.argand, stretch=False), "Formulations"
        )
        self.tabs.addTab(self.notes, "Theory")
        self.tabs.currentChanged.connect(self._draw_tabs)

        centre = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(centre)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(tools)
        v.addWidget(self.hint)
        v.addWidget(self.canvas, 4)
        v.addWidget(self.table, 0)
        v.addWidget(self.readout, 0)
        three_panes(self, column(self.trials, self.props, model, loading, show), centre, self.tabs)

        self._clock = QtCore.QElapsedTimer()
        self._last = 0.0
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._frame)
        self._on_tool(0)
        self.apply_trial(TRIALS[0], keep_history=False)
        self.apply_theme()

    # ------------------------------------------------------------- trials and practice
    def load_trial(self, index: int) -> None:
        self.apply_trial(TRIALS[index])

    def load_setup(self, setup: dict) -> None:
        """Set the page up as a practice problem's `setup` says (a trial's fields, with no note)."""
        self.apply_trial(FrameTrial("", "", **setup))

    def practice_state(self) -> FrameTrial:
        """The page's settings, as a trial: what a practice spec checks."""
        return FrameTrial(
            "",
            "",
            frame=self.frame,
            mode=self.mode,
            tab=self.tabs.currentIndex(),
            picked=self.picked,
            matrix=MATRICES[self.matrix.currentIndex()],
            grid=self.canvas.grid,
            load=self.load,
            out=self.out,
            swing=SWINGS[self.swing.currentIndex()],
        )

    def apply_trial(self, t: FrameTrial, keep_history: bool = True) -> None:
        set_quietly(self.matrix, MATRICES.index(t.matrix))
        self.canvas.grid = t.grid
        set_quietly(self.grid, int(np.argmin([abs(g - t.grid) for g in GRIDS])))
        self.mode = t.mode
        set_quietly(self.swing, SWINGS.index(t.swing))
        self.load = t.load if t.load is not None else default_load(t.frame)
        self.out = t.out if t.out is not None else self.load.index
        self._played = 0.0
        self.trials.show_note(t.note)
        self._set_frame(t.frame, record=keep_history, selection=("element", t.picked) if t.picked is not None else None)
        self.tabs.setCurrentIndex(t.tab)
        self._phase = math.pi / 2
        self.running = True  # a trial starts it
        self.run_button.set_running(True)
        self._show_mode()
        self._start_timer()

    # ------------------------------------------------------------- editing
    def _set_frame(
        self, frame: Frame, record: bool = True, selection: tuple[str, int] | None = None, fit: bool = True
    ) -> None:
        if record and frame != self.frame:
            self._history.append(self.frame)
            self._future.clear()
        self.frame = frame
        set_quietly(self.lumped, int(frame.lumped))
        set_quietly(self.theory, THEORIES.index(frame.theory))
        set_quietly(self.zeta, 100 * frame.zeta)
        self.assembly = assemble(frame)
        self.forms = self.runs = None
        self.response_error = ""
        self._fit_load()
        try:
            self.result = modes(frame, self.assembly)
            self.error = ""
        except FrameError as error:
            self.result, self.error = None, str(error)
        count = self.result.hz.size if self.result is not None else 0
        self.mode = min(self.mode, max(count - 1, 0))
        self.canvas.selection = selection
        self.canvas.load, self.canvas.probe = self.load, self.out
        if selection is not None and selection[0] == "element":
            self.picked = selection[1]
        elif self.picked is not None and self.picked >= len(frame.elements):
            self.picked = None
        self.canvas.set_frame(frame, fit=fit)
        self._fill_props()
        self._fill_table()
        self._fill_readout()
        self._draw_tabs()
        self._show_mode()
        self.undo_button.setEnabled(bool(self._history))
        self.redo_button.setEnabled(bool(self._future))
        self._start_timer()

    def undo(self) -> None:
        if self._history:
            self._future.append(self.frame)
            self._set_frame(self._history.pop(), record=False)

    def redo(self) -> None:
        if self._future:
            self._history.append(self.frame)
            self._set_frame(self._future.pop(), record=False)

    def _on_edit(self, frame: Frame, selection) -> None:
        self._set_frame(frame, selection=selection)

    def _on_selected(self, selection) -> None:
        if selection is not None and selection[0] == "element":
            self.picked = selection[1]
        self._fill_props()
        self._draw_tabs()

    def _on_tool(self, index: int) -> None:
        tool = TOOLS[index]
        self.canvas.set_tool(tool)
        self.support_kind.setVisible(tool is Tool.SUPPORT)
        self.hint.setText(TOOL_HINTS[tool])

    def _on_support_kind(self, index: int) -> None:
        self.canvas.support = SUPPORTS[index + 1]

    def _on_grid(self, index: int) -> None:
        self.canvas.grid = GRIDS[index]
        self.canvas.fit()
        self.canvas.draw()

    def _on_model(self) -> None:
        frame = dataclasses.replace(
            self.frame,
            lumped=bool(self.lumped.currentIndex()),
            theory=THEORIES[self.theory.currentIndex()],
            zeta=self.zeta.value() / 100,
        )
        self._set_frame(frame, selection=self.canvas.selection, fit=False)

    # ------------------------------------------------------------- the load
    def _dof_boxes(self, form: Form, label: str, tip: str) -> tuple[QtWidgets.QSpinBox, QtWidgets.QComboBox]:
        node = QtWidgets.QSpinBox()
        node.setRange(1, MAX_NODES)
        node.setKeyboardTracking(False)
        dof = combo(DOF_CHOICES, tip)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(node)
        row.addWidget(dof, 1)
        holder = QtWidgets.QWidget()
        holder.setLayout(row)
        row.setContentsMargins(0, 0, 0, 0)
        form.add(label, holder, tip)
        return node, dof

    def _fit_load(self) -> None:
        """Keep the load and the response on nodes that exist (else back to the default)."""
        n = len(self.frame.nodes)
        if self.load.node >= n:
            self.load = default_load(self.frame, self.assembly)
        if self.out // 3 >= n:
            self.out = self.load.index
        self._fill_load()

    def _fill_load(self) -> None:
        load = self.load
        top = max(len(self.frame.nodes), 1)
        for box, node in ((self.load_node, load.node), (self.out_node, self.out // 3)):
            box.blockSignals(True)
            box.setMaximum(top)
            box.setValue(node + 1)
            box.blockSignals(False)
        set_quietly(self.load_kind, LOAD_KINDS.index(load.kind))
        set_quietly(self.load_dof, load.dof)
        set_quietly(self.out_dof, self.out % 3)
        set_quietly(self.load_size, load.size)
        set_quietly(self.load_hz, load.hz)
        unit = "N·m" if load.dof == 2 else "N"
        self.load_size.setSuffix(f" {unit}·s" if load.kind is LoadKind.IMPULSE else f" {unit}")
        self.load_form.form.setRowVisible(self.load_hz, load.kind is LoadKind.HARMONIC)

    def _on_load(self) -> None:
        self.load = Load(
            self.load_node.value() - 1,
            self.load_dof.currentIndex(),
            LOAD_KINDS[self.load_kind.currentIndex()],
            self.load_size.value(),
            self.load_hz.value(),
        )
        self.out = 3 * (self.out_node.value() - 1) + self.out_dof.currentIndex()
        self.runs = None
        self._played = 0.0
        self._fill_load()
        self.canvas.load, self.canvas.probe = self.load, self.out
        self._draw_tabs()
        self._show_mode()

    def _on_swing(self) -> None:
        self.reset()
        self._draw_tabs()

    def reset(self) -> None:
        """A mode back at its full swing, still swinging; the response back at t = 0, waiting for Start."""
        self._played, self._phase = 0.0, math.pi / 2
        self.running = SWINGS[self.swing.currentIndex()] is not Swing.RESPONSE
        if self.running:
            self.run_button.set_running(True)
        else:
            self.run_button.rest()
        self._show_mode()
        self._start_timer()

    def _work(self, runs: bool) -> bool:
        """Work out the two formulations (and, if `runs`, their responses) if they aren't yet: True if they are."""
        if self.result is None or self.response_error:
            return False
        if self.forms is None:
            try:
                self.forms = formulations(self.frame, self.assembly, self.result)
            except FrameError as error:
                self.response_error = str(error)
                return False
        if runs and self.runs is None:
            self.runs = responses(self.forms, self.load, self.out)
        return True

    def _selected(self) -> tuple[str, int] | None:
        return self.canvas.selection

    def _node_extras(self, i: int) -> dict[Kind, int | None]:
        """The indices of node i's point mass, torsion spring and torsion damper to the ground (None: none)."""
        found: dict[Kind, int | None] = {Kind.MASS: None, Kind.TORSION_SPRING: None, Kind.TORSION_DAMPER: None}
        for k, e in enumerate(self.frame.elements):
            if e.i == i and e.j is None and e.kind in found and found[e.kind] is None:
                found[e.kind] = k
        return found

    def _on_node_box(self) -> None:
        sel = self._selected()
        if sel is None or sel[0] != "node":
            return
        i = sel[1]
        p = self.props
        f = self.frame
        node = Node(p.node_x.value(), p.node_y.value(), SUPPORTS[p.support.currentIndex()])
        other = f.node_at(node.x, node.y)
        if other is not None and other != i:  # two nodes can't share a point
            node = dataclasses.replace(node, x=f.nodes[i].x, y=f.nodes[i].y)
        f = f.replace_node(i, node)
        wanted = {
            Kind.MASS: (p.mass.value(), p.inertia.value()),
            Kind.TORSION_SPRING: (p.torsion.value() * 1e3, 0.0),
            Kind.TORSION_DAMPER: (p.torsion_c.value(), 0.0),
        }
        extras = self._node_extras(i)
        drop = []
        for kind, (value, inertia) in wanted.items():
            k = extras[kind]
            if value > 0 or (kind is Kind.MASS and inertia > 0):
                e = Element(kind, i, None, value=value, inertia=inertia)
                if k is None:
                    f = f.with_element(e)
                elif f.elements[k] != e:
                    f = f.replace_element(k, e)
                if kind in self.canvas.template:
                    self.canvas.template[kind] = e
            elif k is not None:
                drop.append(k)
        for k in sorted(drop, reverse=True):
            f = f.without_element(k)
        self._set_frame(f, selection=("node", i), fit=False)

    def _on_member_box(self) -> None:
        sel = self._selected()
        if sel is None or sel[0] != "element":
            return
        p = self.props
        e = self.frame.elements[sel[1]]
        shape = SHAPES[p.shape.currentIndex()]
        depth = p.section_depth.value() / 1000
        section = Section(shape, depth, p.section_width.value() / 1000, min(p.wall.value() / 1000, depth / 2))
        kind = Kind.BEAM if p.kind.currentIndex() == 0 else Kind.BAR
        new = dataclasses.replace(e, kind=kind, material=MATERIALS[p.material.currentIndex()], section=section)
        self.canvas.template[kind] = new
        if new != e:
            self._set_frame(self.frame.replace_element(sel[1], new), selection=sel, fit=False)

    def _on_link_box(self) -> None:
        sel = self._selected()
        if sel is None or sel[0] != "element":
            return
        e = self.frame.elements[sel[1]]
        value = self.props.value.value() * (1e3 if e.kind is Kind.SPRING else 1.0)
        new = dataclasses.replace(e, value=value)
        self.canvas.template[e.kind] = new
        if new != e:
            self._set_frame(self.frame.replace_element(sel[1], new), selection=sel, fit=False)

    def _on_divide(self) -> None:
        sel = self._selected()
        if sel is None or sel[0] != "element":
            return
        parts = self.props.parts.value()
        if len(self.frame.nodes) + parts - 1 > MAX_NODES:
            return
        self._set_frame(self.frame.divide(sel[1], parts), selection=None)

    # ------------------------------------------------------------- filling in
    def _fill_props(self) -> None:
        p = self.props
        sel = self._selected()
        if sel is None:
            p.show_form(p.empty)
            p.remove.setEnabled(False)
            return
        p.remove.setEnabled(True)
        what, index = sel
        if what == "node":
            node = self.frame.nodes[index]
            extras = self._node_extras(index)
            mass_at = extras[Kind.MASS]
            mass = self.frame.elements[mass_at] if mass_at is not None else None
            spring = extras[Kind.TORSION_SPRING]
            damper = extras[Kind.TORSION_DAMPER]
            for box, value in (
                (p.node_x, node.x),
                (p.node_y, node.y),
                (p.support, SUPPORTS.index(node.support)),
                (p.mass, mass.value if mass else 0.0),
                (p.inertia, mass.inertia if mass else 0.0),
                (p.torsion, self.frame.elements[spring].value / 1e3 if spring is not None else 0.0),
                (p.torsion_c, self.frame.elements[damper].value if damper is not None else 0.0),
            ):
                set_quietly(box, value)
            p.node.setTitle(f"Node {index + 1}")
            p.show_form(p.node)
            return
        e = self.frame.elements[index]
        if e.kind.member:
            assert e.j is not None, "a member joins two nodes"
            sec = e.section
            for box, value in (
                (p.kind, 0 if e.kind is Kind.BEAM else 1),
                (p.material, [m.name for m in MATERIALS].index(e.material.name)),
                (p.shape, SHAPES.index(sec.shape)),
                (p.section_depth, sec.depth * 1000),
                (p.section_width, sec.width * 1000),
                (p.wall, sec.wall * 1000),
            ):
                set_quietly(box, value)
            tube = sec.shape is Shape.TUBE
            p.member.form.setRowVisible(p.section_width, not tube)
            p.member.form.setRowVisible(p.wall, tube)
            length = self.frame.length(e)
            r = math.sqrt(sec.inertia / sec.area)
            p.member_info.setText(
                f"L = {length:.4g} m, A = {number(sec.area * 1e6, 4)} mm², I = {number(sec.inertia * 1e12, 4)} mm⁴, L/r = "
                f"{length / r:.3g} (slenderness)."
            )
            p.member.setTitle(f"{e.kind.value} {index + 1}: nodes {e.i + 1}–{e.j + 1}")
            p.show_form(p.member)
            return
        if e.kind.link:
            spring = e.kind is Kind.SPRING
            label = p.link.label(p.value)
            label.setText("Stiffness k:" if spring else "Damping c:")
            p.value.setSuffix(" kN/m" if spring else " N·s/m")
            set_quietly(p.value, e.value / 1e3 if spring else e.value)
            end = f"node {e.j + 1}" if e.j is not None else "the ground"
            p.link_info.setText(f"From node {e.i + 1} to {end}, {self.frame.length(e):.4g} m long.")
            p.link.setTitle(f"{e.kind.value} {index + 1}")
            p.show_form(p.link)
            return
        p.show_form(p.empty)

    def _fill_table(self) -> None:
        self.table.blockSignals(True)
        r = self.result
        count = min(r.hz.size, MODE_ROWS) if r is not None else 0
        self.table.setRowCount(count)
        for i in range(count):
            assert r is not None
            texts = [str(i + 1), f"{r.hz[i]:.4g}" if r.hz[i] > 0 else "0 (mechanism)"]
            texts += [f"{100 * r.energy[j, i]:.0f} %" if r.hz[i] > 0 else "" for j in range(3)]
            for col, text in enumerate(texts):
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(i, col, item)
        if count:
            self.table.selectRow(min(self.mode, count - 1))
        self.table.blockSignals(False)
        rows = sum(self.table.rowHeight(i) for i in range(min(count, 5))) if count else 0
        self.table.setFixedHeight(self.table.horizontalHeader().height() + rows + 2 * self.table.frameWidth() + 2)

    def _fill_readout(self) -> None:
        a = self.assembly
        f = self.frame
        n = f.n_dof
        text = (
            f"{len(f.nodes)} nodes, {len(f.elements)} elements: {n} DOFs, {int(a.fixed.sum())} held by supports, "
            f"{int(a.unused.sum())} unused, {int(a.free.sum())} free"
        )
        if self.result is not None:
            r = self.result
            if r.condensed:
                text += f", {r.condensed} of them massless (condensed out)"
            text += f": {r.hz.size} modes."
            if r.mechanisms:
                text += f" <b>{r.mechanisms} at 0 Hz</b>: a mechanism, or the frame isn't held."
        else:
            text += "." + (f" <span style='color:{colors.force}'>{self.error}</span>" if self.error else "")
        self.readout.setText(text)

    def _draw_tabs(self) -> None:
        i = self.tabs.currentIndex()
        if i == TAB_ASSEMBLY:
            self._draw_assembly()
        elif i == TAB_ELEMENT:
            self._draw_element()
        elif i == TAB_RESPONSE:
            self._draw_response()
        elif i == TAB_FORMULATIONS:
            self._draw_formulations()

    def _draw_assembly(self) -> None:
        a = self.assembly
        matrix = MATRICES[self.matrix.currentIndex()]
        A = {Matrix.K: a.K, Matrix.M: a.M, Matrix.C: a.C}[matrix]
        n = A.shape[0]
        if n == 0:
            self.assembly_view.setHtml("<p>Nothing built yet.</p>")
            self.assembly_note.setText("")
            return
        tint = np.zeros((n, n), dtype=bool)
        if self.picked is not None and self.picked < len(a.locals):
            part = a.locals[self.picked]
            own = {Matrix.K: part.stiffness, Matrix.M: part.mass, Matrix.C: part.damping}[matrix]()
            dofs = list(part.dofs)
            tint[np.ix_(dofs, dofs)] = own != 0
        struck = ~a.free
        keep = np.arange(n)
        if self.free_only.isChecked() or n > MATRIX_LIMIT:
            keep = a.free_dofs
        labels = [dof_label(int(d)) for d in keep]
        idx = np.ix_(keep, keep)
        self.assembly_view.setHtml(
            matrix_html(A[idx], labels, labels, tint[idx], struck[keep], struck[keep], title=matrix.name)
        )
        picked = (
            f" Tinted: {self.frame.elements[self.picked].kind.value.lower()} {self.picked + 1}'s share."
            if self.picked is not None and self.picked < len(self.frame.elements)
            else " Pick an element to tint its share."
        )
        unit = {Matrix.K: "N/m, N/rad and N·m/rad", Matrix.M: "kg and kg·m²", Matrix.C: "N·s/m and N·m·s/rad"}[matrix]
        self.assembly_note.setText(
            f"{matrix.value} over {'the free' if len(keep) < n else 'all ' + str(n)} DOFs ({unit}).{picked} "
            "Greyed and struck: held by a support, or unused."
        )

    def _draw_element(self) -> None:
        matrix = MATRICES[self.matrix.currentIndex()]
        sel = self._selected()
        if self.picked is not None and self.picked < len(self.frame.elements) and (sel is None or sel[0] == "element"):
            html = element_html(self.frame, self.assembly, self.picked, matrix)
        elif sel is not None and sel[0] == "node":
            html = node_html(self.frame, self.assembly, sel[1])
        else:
            html = "<p>Pick an element (the Select tool) to see its matrices here.</p>"
        self.element_view.setHtml(html)

    def _draw_response(self) -> None:
        mag, phase, time = self.curves["mag"], self.curves["phase"], self.curves["time"]
        load = self.load
        if not self._work(runs=True) or not self.assembly.free[load.index]:
            for exact, conventional in (mag, phase, time):
                exact.setData([], [])
                conventional.setData([], [])
            self.drive_line.hide()
            self.playhead.hide()
            if self.result is None:
                self.response_note.setText(self.error or "Nothing to drive yet.")
            elif self.response_error:
                self.response_note.setText(self.response_error)
            else:
                self.response_note.setText(
                    f"The load's DOF, {dof_label(load.index)}, is held by a support or unused: the frame doesn't "
                    "feel it. Move the load (on the left)."
                )
            return
        r = self.runs
        assert r is not None
        f = r.freqs
        for (exact, conventional), (a, b) in (
            (mag, (np.maximum(np.abs(r.frf_exact), 1e-300), np.maximum(np.abs(r.frf_conventional), 1e-300))),
            (phase, (np.degrees(np.angle(r.frf_exact)), np.degrees(np.angle(r.frf_conventional)))),
        ):
            exact.setData(f, a)
            conventional.setData(f, b)
        self.frf_phase.setYRange(-185, 185, padding=0)
        for p in (self.frf_mag, self.frf_phase):
            p.setXRange(math.log10(f[0]), math.log10(f[-1]), padding=0)
        peak = float(np.max(np.abs(r.frf_exact)))
        if peak > 0:
            self.frf_mag.setYRange(math.log10(peak) - 5, math.log10(peak) + 0.5, padding=0)
        turn = self.out % 3 == 2
        units = ("rad" if turn else "m") + "/" + ("N·m" if load.dof == 2 else "N")
        self.frf_mag.setLabel("left", f"|{dof_label(self.out)} / F on {dof_label(load.index)}| ({units})")
        harmonic = load.kind is LoadKind.HARMONIC
        self.drive_line.setVisible(harmonic)
        if harmonic:
            self.drive_line.setPos(math.log10(load.hz))
        time[0].setData(r.t, r.exact[:, self.out])
        time[1].setData(r.t, r.conventional[:, self.out])
        self.time_plot.setLabel("left", dof_label(self.out), units="rad" if turn else "m")
        self.time_plot.setXRange(0.0, float(r.t[-1]), padding=0)
        self._place_playhead()
        top = max(peak, 1e-300)
        frf_gap = float(np.max(np.abs(r.frf_exact - r.frf_conventional))) / top
        reach = float(np.max(np.abs(r.exact[:, self.out]), initial=0.0))
        time_gap = float(np.max(np.abs(r.exact[:, self.out] - r.conventional[:, self.out]))) / reach if reach else 0.0
        kind = {
            LoadKind.HARMONIC: f"F sin Ωt at {load.hz:.4g} Hz (the red line)",
            LoadKind.IMPULSE: "an impulse",
            LoadKind.STEP: "a step",
        }[load.kind]
        gap = (
            "The two models agree (the damping is proportional)."
            if frf_gap < 1e-6 and time_gap < 1e-6
            else f"They part: the conventional FRF is off by up to <b>{100 * frf_gap:.3g} %</b> of the exact peak, "
            f"its time response by <b>{100 * time_gap:.3g} %</b> of the exact one's largest."
        )
        undamped = (
            " No damping: each peak is infinite (the grid steps round them), and a harmonic load's response "
            "never settles."
            if self.frame.zeta == 0 and not self.assembly.C.any()
            else ""
        )
        self.response_note.setText(
            f"From {dof_label(load.index)} (the arrow) to {dof_label(self.out)} (the square): the receptance, and "
            f"the response to {kind} from rest. Solid: the state space, exact; dashed: the conventional real modes. "
            f"{gap}{undamped}"
        )

    def _place_playhead(self) -> None:
        on = SWINGS[self.swing.currentIndex()] is Swing.RESPONSE and self.runs is not None
        self.playhead.setVisible(on)
        if on:
            self.playhead.setPos(self._played)

    def _draw_formulations(self) -> None:
        t = self.forms_table
        t.blockSignals(True)
        if not self._work(runs=False):
            t.setRowCount(0)
            t.blockSignals(False)
            self.forms_note.setText(self.response_error or self.error or "Nothing built yet.")
            self.argand_lines.setData([], [])
            self.argand_dots.setData([], [])
            self.argand_note.setText("")
            return
        forms = self.forms
        assert forms is not None
        rows = forms.modes[:MODE_ROWS]
        t.setRowCount(len(rows))
        for i, m in enumerate(rows):
            if m.hz == 0:
                texts = [str(i + 1), "0 (mechanism)", "", "", "", ""]
            elif m.pole is None:
                texts = [str(i + 1), f"{m.hz:.4g}", _percent(m.zeta_conventional), "overdamped", "", ""]
            else:
                assert m.measures is not None
                texts = [
                    str(i + 1),
                    f"{m.hz:.4g}",
                    _percent(m.zeta_conventional),
                    f"{m.hz_exact:.4g}",
                    _percent(m.zeta_exact),
                    f"{m.measures.mpc:.3f}",
                ]
            for col, text in enumerate(texts):
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                t.setItem(i, col, item)
        if rows:
            t.selectRow(min(self.mode, len(rows) - 1))
        t.blockSignals(False)
        n_conv, n_exact = forms.conventional.size, forms.exact.size
        first = n_exact - n_conv
        text = (
            "Proportional: ΦᵀCΦ is diagonal, so the state space's modes are the undamped ones, real, and its poles "
            "are the conventional model's."
            if forms.proportional
            else "Not proportional: the dampers' ΦᵀCΦ has terms off its diagonal, which the conventional model drops. "
            "The state space keeps them: its modes are complex, and its f and ζ move."
        )
        text += f" The conventional model has {n_conv // 2} oscillators; the state space {n_exact} states"
        text += f" ({first} first-order: a damper on a massless DOF)." if first else "."
        if forms.roots.size:
            tau = -1.0 / forms.roots[0]
            text += (
                f" {forms.roots.size} of its roots are real (overdamped modes, or relaxations), the slowest "
                f"dying away as e<sup>−t/τ</sup>, τ = {fmt(tau, 's')}."
            )
        self.forms_note.setText(text)
        self._draw_argand()

    def _draw_argand(self) -> None:
        forms = self.forms
        m = forms.modes[self.mode] if forms is not None and self.mode < len(forms.modes) else None
        if m is None or m.shape is None or m.measures is None:
            self.argand_lines.setData([], [])
            self.argand_dots.setData([], [])
            self.argand_note.setText(
                f"Mode {self.mode + 1} has no oscillating counterpart in the state space." if m is not None else ""
            )
            return
        moves = np.concatenate([m.shape[0::3], m.shape[1::3]])
        moves = moves[np.abs(moves) > 1e-9]
        lines = [p for z in moves for p in ([0.0, 0.0], [z.real, z.imag], NAN2)]
        self.argand_lines.setData(*_segments(lines))
        self.argand_dots.setData(moves.real, moves.imag)
        self.argand.setXRange(-1.05, 1.05, padding=0)
        self.argand.setYRange(-1.05, 1.05, padding=0)
        self.argand_note.setText(
            f"Mode {self.mode + 1}'s shape ψ in the state space: each node's u and v as a point in the complex "
            f"plane. A real mode's lie on one line through 0. MPC = {m.measures.mpc:.3f}, MPD = "
            f"{m.measures.mpd:.3g}°."
        )

    def _on_forms_row(self) -> None:
        rows = self.forms_table.selectionModel().selectedRows()
        if rows:
            self.mode = rows[0].row()
            self.table.blockSignals(True)
            if self.mode < self.table.rowCount():
                self.table.selectRow(self.mode)
            self.table.blockSignals(False)
            self._draw_argand()
            self._show_mode()

    # ------------------------------------------------------------- the modes
    def _on_row(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.mode = rows[0].row()
            self._show_mode()
            if self.tabs.currentIndex() == TAB_FORMULATIONS:
                self._draw_formulations()

    def _on_animate(self, on: bool) -> None:
        self.running = on
        self._show_mode()
        self._start_timer()

    def mode_shape(self) -> np.ndarray | None:
        """The mode picked, scaled so its largest displacement is SWING of the frame's size (None if none)."""
        r = self.result
        if r is None or self.mode >= r.hz.size or not self.frame.nodes:
            return None
        phi = r.shapes[:, self.mode]
        moves = np.hypot(phi[0::3], phi[1::3])
        reach = float(moves.max())
        x0, x1, y0, y1 = self.frame.extent()
        size = max(x1 - x0, y1 - y0, 1e-3)
        if reach < 1e-12:  # a mode that only turns nodes: scale by the turns
            reach = float(np.abs(phi[2::3]).max()) * size
        return phi * (SWING * size / reach) if reach > 0 else None

    def _size(self) -> float:
        x0, x1, y0, y1 = self.frame.extent()
        return max(x1 - x0, y1 - y0, 1e-3)

    def swing_shape(self) -> np.ndarray | None:
        """The displacement drawn now (every DOF), as Swing says: the undamped mode at the animation's phase,
        the state space's (complex) mode at it, or the response at the record's time played; None for none."""
        swing = SWINGS[self.swing.currentIndex()]
        phase = self._phase
        if swing is Swing.RESPONSE:
            if not self._work(runs=True) or not self.assembly.free[self.load.index]:
                return None
            r = self.runs
            assert r is not None
            reach = _reach(r.exact.T)
            if reach <= 0:
                return None
            k = min(round(float(self._played / r.t[-1]) * (r.t.size - 1)), r.t.size - 1)
            return r.exact[k] * (SWING * self._size() / reach)
        if swing is Swing.COMPLEX and self._work(runs=False):
            forms = self.forms
            assert forms is not None
            m = forms.modes[self.mode] if self.mode < len(forms.modes) else None
            if m is not None and m.shape is not None:
                reach = _reach(m.shape)
                if reach > 0:
                    return (m.shape * np.exp(1j * phase)).imag * (SWING * self._size() / reach)
        shape = self.mode_shape()
        return shape * math.sin(phase) if shape is not None else None

    def _show_mode(self) -> None:
        self.canvas.q = self.swing_shape()
        self.canvas.draw()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._start_timer()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._timer.stop()

    def _start_timer(self) -> None:
        if self.isVisible() and self.running and self.result is not None and not self._timer.isActive():
            self._clock.restart()
            self._last = 0.0
            self._timer.start(FRAME_MS)

    def _frame(self) -> None:
        if not self.running:  # paused or reset since this frame was queued
            return
        began = self._clock.elapsed()
        now = began / 1000.0
        dt = min(now - self._last, 0.1)
        self._last = now
        with frame_stats.frame("Frame builder", lambda: self._phase, FRAME_MS):
            self.step(dt)
            settle_plots(self)
        if self.isVisible() and self.running and self.result is not None:
            self._timer.start(max(1, FRAME_MS - (self._clock.elapsed() - began)))

    def frame_context(self) -> str:
        swing = SWINGS[self.swing.currentIndex()].name.lower()
        return f"{len(self.frame.nodes)} nodes, {len(self.frame.elements)} elements, mode {self.mode + 1}, {swing}"

    def step(self, dt: float) -> None:
        """Swing the mode on by dt seconds at the animation's own pace (not while a node is dragged)."""
        if self.canvas._drag is not None:
            return
        if SWINGS[self.swing.currentIndex()] is Swing.RESPONSE and self.runs is not None:
            end = float(self.runs.t[-1])
            self._played = (self._played + dt * end / PLAY_S) % end
            if self.tabs.currentIndex() == TAB_RESPONSE:
                self._place_playhead()
        else:
            self._phase = (self._phase + TWO_PI * SHOW_HZ * dt) % TWO_PI
        self._show_mode()

    def apply_theme(self) -> None:
        self.canvas.apply_theme()
        mute(self.hint)
        dash = QtCore.Qt.PenStyle.DashLine
        for exact, conventional in self.curves.values():
            exact.setPen(pg.mkPen(colors.strong, width=2))
            conventional.setPen(pg.mkPen(colors.fit, width=2, style=dash))
        for line in (self.playhead, self.drive_line):
            line.setPen(pg.mkPen(colors.force, width=1.5, style=dash))
        self.argand_lines.setPen(pg.mkPen(colors.mode[0], width=1.5))
        self.argand_dots.setSymbolBrush(pg.mkBrush(colors.mode[0]))
        self.argand_dots.setSymbolPen(None)
        self._draw_tabs()
