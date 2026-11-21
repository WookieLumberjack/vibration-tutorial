"""3D rigid body page: a body tumbling in space, its inertia tensor, principal axes and Euler's equations.

Left: the body (a cuboid, a cylinder or a sphere, plus a point mass anywhere on it), how it is spun (about one
of its principal axes, nudged off it), a torque (held in the body's axes or in space's, for a set time from the
start or from a *Kick*), and the playback. Centre: the body in 3D (drag to turn it) with its own axes x, y, z,
optionally its principal axes, the vectors L, ω and τ and the trail of ω's direction, and a readout below.
Right: the *Inertia* tab (the tensor about G in the body's axes, its principal moments and axes), *Components*
(ω, L, τ or I against time in space's, the body's or the principal axes, with the equations at one instant
written out below: `equations_snapshot`), *Orientation* (the start pose, and the orientation four ways), *Drift*
(the start stepped torque-free as a matrix, as Euler angles and as a quaternion: `rigid_body.compare`), *Check* (the energy and L against what the torque put in) and the
Theory. The model is core/rigid_body.py (shared maths in core/rotation.py), stepped on the page's frame loop.

Its `PROBLEMS` (the Practice window) ask for a box's moments, a point mass's shift of G and product of inertia,
whether a spin flips and how fast, an Euler angle's rate and a quaternion's turn, set up through `load_setup`.

Working on it: `tests/test_rigid_body.py` checks the model, `tests/test_gui_smoke.py`'s
`test_rigid_body_page` and `test_rigid_body_orientation` the page, `tests/test_practice.py` the problems, and
`tests/test_tutorial_numbers.py` the notes' numbers (the worked example's among them).
"""

from __future__ import annotations

import enum
import itertools
import math
import time
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..core import rotation
from ..core.practice import Prediction, Problem, Question, Requirement, at_most, kept, number, pick, rounded
from ..core.rigid_body import SHAPES, TORQUES, Body, Drift, Shape, Torque, Tumbler, compare
from ..core.rotation import Turning
from ..core.stepping import FRAME_BUDGET, Budget
from ..view_models.components import SHOW_I, SHOW_L, SHOW_W, series
from ..view_models.text import number as sig
from . import coalesce, rigid_body_figures
from .cms_time import SampleBuffer
from .components_tab import ComponentsTab
from .equations_snapshot import bracket, equation, snapshot_html
from .frame_loop import FrameLoop
from .layouts import ColumnScroll
from .lesson import (
    Dot,
    Form,
    RunButton,
    TrialBox,
    column,
    combo,
    connect_all,
    note_label,
    plot,
    readout_label,
    set_quietly,
    stack,
    theory,
    three_panes,
    trials_html,
)
from .links import load_notes, page_link
from .live_label import LiveLabel
from .orbit_view import OrbitView
from .style import colors
from .theming import add_legend
from .transport import Transport

TAB_INERTIA, TAB_COMPONENTS, TAB_ORIENTATION, TAB_DRIFT, TAB_CHECK = range(5)
PATHS = ("In space", "On the body", "Both", "None")  # where the view traces ω's direction
PATH_SPACE, PATH_BODY, PATH_BOTH, PATH_NONE = range(4)
POLHODE = 20.0  # s of ω's path on the body drawn
DRIFT_STEPS = (("1/20 of a turn", 20), ("1/50 of a turn", 50))
DRIFT_SPANS = (5.0, 10.0, 20.0)  # s
PLAYBACK = (0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
REAL_TIME = PLAYBACK.index(1.0)
HISTORY = 200_000  # samples kept
PROBE_STEPS = 200  # steps in a frame while the cost of a step isn't known yet
CHECK_POINTS = 2000  # the plots' samples, at most
TRAIL = 3.0  # s of ω's direction drawn
SNAPSHOT_FRAMES = 10  # while it runs the snapshot is rewritten every this many frames
SPIN_AXES = ("Principal axis 1 (least I)", "Principal axis 2 (middle I)", "Principal axis 3 (greatest I)")
STRIPS = 24  # a cylinder's side, in flat faces
BODY_AXES = 6  # the body's axes' colour, colors.mass[k], as on the Gyroscope page


class Axes(enum.Enum):
    SPACE = "Space (fixed)"
    BODY = "Body (its own x, y, z)"
    PRINCIPAL = "Principal axes (1, 2, 3)"


FRAMES = tuple(Axes)
AXIS_NAMES = {Axes.SPACE: ("X", "Y", "Z"), Axes.BODY: ("x", "y", "z"), Axes.PRINCIPAL: ("1", "2", "3")}
WHERE = {
    Axes.SPACE: "in fixed axes (X, Y, Z)",
    Axes.BODY: "in the body's own axes (x, y, z: the shape's)",
    Axes.PRINCIPAL: "in the body's principal axes (1, 2, 3)",
}


@dataclass(frozen=True)
class RigidTrial:
    name: str
    note: str
    shape: Shape = Shape.CUBOID
    mass: float = 1.0
    a: float = 0.30
    b: float = 0.20
    c: float = 0.10
    point: float = 0.0
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    spin_axis: int = 1  # principal axis, 0 to 2
    rate: float = 6.0  # rad/s
    nudge: float = 1.0  # degrees off the axis
    held: Torque = Torque.NONE
    torque: tuple[float, float, float] = (0.0, 0.0, 0.0)
    duration: float = 1.0  # s the torque acts from the start
    playback: int = REAL_TIME
    tab: int = TAB_COMPONENTS
    show: int = SHOW_W
    frame: Axes = Axes.PRINCIPAL
    principal: bool = False  # draw the principal axes
    pose: tuple[float, float, float] = (0.0, 0.0, 0.0)  # the start's Euler angles φ, θ, ψ, degrees
    path: int = PATH_SPACE  # where the view traces ω
    predict: Prediction | None = None  # asked before it runs (Things to try)

    @property
    def body(self) -> Body:
        return Body(self.shape, self.mass, self.a, self.b, self.c, self.point, self.offset)


def start_rate(body: Body, axis: int, rate: float, nudge_deg: float) -> np.ndarray:
    """ω in the body's axes at the start: `rate` about principal axis `axis`, tipped `nudge_deg` toward the next."""
    _, P = body.principal()
    d = math.radians(nudge_deg)
    return rate * (math.cos(d) * P[:, axis] + math.sin(d) * P[:, (axis + 1) % 3])


TRIALS = (
    RigidTrial(
        "Spun about the middle axis",
        "A 30 × 20 × 10 cm box spun at 6 rad/s about its middle axis (2), nudged 1° off it. The nudge grows, and "
        "the box turns over and back again and again: the intermediate-axis (tennis-racket) effect. Watch "
        "<m>ω</m>'s components in the principal axes, and L staying put in space.",
        spin_axis=1,
        nudge=1.0,
        predict=Prediction(
            'A 30 × 20 × 10 cm box spun about its middle axis, nudged 1° off it. What does it do?',
            (
                "It keeps spinning about that axis, wobbling a little",
                "It turns over and back, again and again",
            ),
            1,
            why=(
                'Spin about the least or the greatest axis is stable, but this is the middle one: a nudge grows (the tennis-racket effect).',
                '',
            ),
        ),
    ),
    RigidTrial(
        "Spun about the greatest axis",
        "The same box spun about its greatest axis (3), nudged 5° off it: the nudge only circles, and the box "
        "wobbles but never turns over. Try axis 1 too.",
        spin_axis=2,
        nudge=5.0,
        predict=Prediction(
            'The same box spun about its greatest axis, nudged 5° off it. What does it do?',
            (
                "It keeps spinning about that axis, wobbling a little",
                "It turns over and back, again and again",
            ),
            0,
            why=(
                '',
                "It flips only about the middle axis. About the greatest, as about the least, a nudge only circles.",
            ),
        ),
    ),
    RigidTrial(
        "ω and L point apart",
        "Spun 30° off its least axis, the box's <m>ω</m> isn't along L = I<m>ω</m>: L stays fixed in space (no "
        "torque) while <m>ω</m> wanders round it, its trail a loop about L.",
        spin_axis=0,
        nudge=30.0,
        tab=TAB_COMPONENTS,
        show=SHOW_L,
        frame=Axes.SPACE,
    ),
    RigidTrial(
        "An off-centre mass tilts the principal axes",
        "A 300 g point mass 10 cm along x and 8 cm along y: G moves toward it, the tensor in the box's own axes "
        "gets products of inertia, and the principal axes (dashed) turn away from x, y, z.",
        point=0.3,
        offset=(0.10, 0.08, 0.0),
        spin_axis=2,
        nudge=0.0,
        tab=TAB_INERTIA,
        principal=True,
    ),
    RigidTrial(
        "I is constant in the body, not in space",
        "The tensor's six numbers in space as the box tumbles: they change all the time, products of inertia "
        "and all. Switch to the body's axes and they stand still.",
        spin_axis=0,
        nudge=30.0,
        show=SHOW_I,
        frame=Axes.SPACE,
    ),
    RigidTrial(
        "A push held in space",
        "From rest, a 0.02 N·m torque about Z held in space for 1 s: L grows along Z by 0.02 N·m·s and then "
        "stays. The box spins up about Z, which isn't a principal axis of this tilted point-mass box, so it "
        "wobbles once the push ends.",
        point=0.3,
        offset=(0.10, 0.0, 0.05),
        rate=0.0,
        nudge=0.0,
        held=Torque.SPACE,
        torque=(0.0, 0.0, 0.02),
        duration=1.0,
        show=SHOW_L,
        frame=Axes.SPACE,
    ),
    RigidTrial(
        "A torque held in the body",
        "Spinning at 6 rad/s about its greatest axis, the box gets a 0.01 N·m torque about its own x for 3 s. "
        "Held in the body it turns with the spin, so L only wobbles round instead of growing one way: the "
        "trick behind a spinning spacecraft's thrusters firing in pulses.",
        spin_axis=2,
        nudge=0.0,
        held=Torque.BODY,
        torque=(0.01, 0.0, 0.0),
        duration=3.0,
        show=SHOW_L,
        frame=Axes.SPACE,
    ),
    RigidTrial(
        "A sphere: every axis is principal",
        "A uniform sphere's moments are all the same, so I = I·1 in any axes: L is along <m>ω</m> whichever way "
        "it spins, and it never wobbles.",
        shape=Shape.SPHERE,
        mass=2.0,
        a=0.12,
        spin_axis=0,
        nudge=30.0,
        show=SHOW_W,
        frame=Axes.SPACE,
    ),
    RigidTrial(
        "Gimbal lock",
        "The box at rest, upright: Euler's θ = 0. Its first and third turns, φ and ψ, are then both about Z, so in "
        "the Orientation tab changing either turns it the same way: only φ + ψ counts, and one of the three angles "
        "is lost. B and q have no such place.",
        rate=0.0,
        nudge=0.0,
        pose=(40.0, 0.0, 0.0),
        tab=TAB_ORIENTATION,
    ),
    RigidTrial(
        "Euler's rates blow up at the lock",
        "The box turning steadily at 2 rad/s about its x, which carries its z within 1.7° of Z, at 0.1× "
        "playback. Its ω is steady, but φ̇ = (ω₁ sin ψ + ω₂ cos ψ)/sin θ climbs to about 67 rad/s, 33 times ω, "
        "as θ passes its least, and φ and ψ swing through half a turn in a moment; through Z itself they'd be "
        "unbounded. The quaternion's rates, ½q ⊗ ω, stay as steady as ω.",
        spin_axis=0,
        rate=2.0,
        nudge=0.0,
        pose=(0.0, 20.0, 175.0),
        playback=PLAYBACK.index(0.1),
        tab=TAB_ORIENTATION,
    ),
    RigidTrial(
        "ω's path on the body",
        "Spun 15° off its greatest axis, the box's <m>ω</m> traces a closed loop round that axis in the body's own "
        "axes (drawn turning with it: Poinsot's polhode), while in space it wanders round L. Try the middle "
        "axis: its path runs from one end of the axis to the other.",
        spin_axis=2,
        nudge=15.0,
        path=PATH_BOTH,
        tab=TAB_COMPONENTS,
        show=SHOW_W,
        frame=Axes.PRINCIPAL,
    ),
    RigidTrial(
        "Step it three ways",
        "The box tipped 50° and spun 20° off its least axis, stepped torque-free at a coarse 1/20 of a turn as a "
        "rotation matrix, as Euler angles and as a quaternion, for 10 s. The quaternion ends about 0.05° off, the "
        "matrix about 0.3° and no longer quite a rotation, and the Euler angles about 13°: the nearer they pass to "
        "the lock, the worse they step. Then try the rates trial's start, which passes it within 1.7°.",
        spin_axis=0,
        nudge=20.0,
        pose=(30.0, 50.0, 20.0),
        tab=TAB_DRIFT,
    ),
)

TIPS = {
    "shape": "<p>A uniform box (sides a, b, c along its x, y, z), a cylinder (radius a, length c along z) or a "
    "sphere (radius a).</p>",
    "mass": "<p>The shape's mass, spread evenly through it.</p>",
    "a": "<p>A box's side along x; a cylinder's or a sphere's radius.</p>",
    "b": "<p>A box's side along y.</p>",
    "c": "<p>A box's side along z; a cylinder's length.</p>",
    "point": "<p>A point mass fixed to the body: 0 for none. Off the body's axes it adds products of inertia and "
    "tilts the principal axes.</p>",
    "offset": "<p>Where the point mass sits from the shape's centre, along the body's x, y, z.</p>",
    "spin_axis": "<p>Which principal axis it spins about at the start: 1 has the least moment, 3 the greatest.</p>",
    "rate": "<p>How fast it spins at the start, rad/s.</p>",
    "nudge": "<p>How far ω starts off the axis, toward the next principal axis. 0 spins exactly about it.</p>",
    "held": "<p>The torque's axes. <b>Body</b>: its components are along the body's x, y, z, so it turns with "
    "the body (a thruster). <b>Space</b>: along X, Y, Z, a push that keeps its direction.</p>",
    "torque": "<p>The torque's components in the axes it's held in, N·m.</p>",
    "duration": "<p>How long the torque acts from the start, or from <i>Kick</i>.</p>",
    "kick": "<p>Apply the torque again, from now, for its time.</p>",
    "playback": "<p>Simulated time per real second.</p>",
    "comp_show": "<p>What to plot as it runs: ω, L = Iω (about G), the torque τ, or the six numbers of the "
    "inertia tensor I.</p>",
    "comp_frame": "<p>Whose axes the components are in. <b>Space</b>: fixed. <b>Body</b>: the shape's own x, "
    "y, z, turning with it. <b>Principal</b>: the body's principal axes, turning with it too, in which I is "
    "diagonal.</p>",
    "comp_span": "<p>How much of the run to plot.</p>",
    "path": "<p>Where the view traces ω's direction. <b>In space</b>: where it has pointed, the last few seconds. "
    "<b>On the body</b>: where it has pointed in the body's own axes, drawn turning with it: torque-free, a closed "
    "loop round the axis it spins about (Poinsot's polhode), or, spun about the middle axis, a path from that "
    "axis's one end to the other.</p>",
    "drift_step": "<p>The step the three are stepped with, as a fraction of a turn at the body's starting rate: a "
    "coarse step, so the drift shows within the run.</p>",
    "drift_span": "<p>How long to step them for.</p>",
    "pose": "<p>How the body is turned at the start, as Euler's z-x-z angles: φ about Z, then θ about the line of "
    "nodes (where x has got to), then ψ about the body's own z. B = R<sub>z</sub>(φ) R<sub>x</sub>(θ) "
    "R<sub>z</sub>(ψ).</p>",
}
SIDES = (100.0, 150.0, 200.0, 250.0, 300.0, 400.0)  # mm, a problem's box's


def _box(rng: np.random.Generator) -> tuple[float, float, float, float]:
    """A problem's box: its mass (kg) and three different sides (mm), largest first."""
    a, b, c = sorted(rng.choice(SIDES, 3, replace=False), reverse=True)
    return pick(rng, (0.5, 1.0, 2.0, 4.0)), float(a), float(b), float(c)


def _box_moment(rng: np.random.Generator) -> Question:
    m, a, b, c = _box(rng)
    k = int(rng.integers(3))
    axis = "xyz"[k]
    others = [s for j, s in enumerate((a, b, c)) if j != k]
    body = Body(Shape.CUBOID, m, a / 1000, b / 1000, c / 1000)
    answer = float(body.inertia()[k, k])
    return Question(
        f"A uniform {sig(m, 6)} kg box, {sig(a, 6)} × {sig(b, 6)} × {sig(c, 6)} mm along its x, y, z. What is its moment of inertia about "
        f"its {axis} axis, through its centre?",
        answer,
        f"I<sub>{axis}{axis}</sub> = m(s₁² + s₂²)/12 with the two sides across that axis: {sig(m, 6)} × ({sig(others[0] / 1000, 6)}² + "
        f"{sig(others[1] / 1000, 6)}²)/12 = {number(answer)} kg·m².",
        unit="kg·m²",
        setup=dict(mass=m, a=a / 1000, b=b / 1000, c=c / 1000, rate=0.0, nudge=0.0, tab=TAB_INERTIA),
    )


def _point_centre(rng: np.random.Generator) -> Question:
    m, a, b, c = _box(rng)
    mp = pick(rng, (0.1, 0.2, 0.5, 1.0))
    x = pick(rng, (50.0, 80.0, 100.0, 120.0))
    G = Body(Shape.CUBOID, m, a / 1000, b / 1000, c / 1000, mp, (x / 1000, 0.0, 0.0)).centre()[0] * 1000
    return Question(
        f"A {sig(m, 6)} kg box carries a {sig(mp, 6)} kg point mass {sig(x, 6)} mm along x from its centre. How far from the box's "
        "centre is the centre of mass G?",
        G,
        f"G is the masses' average place: {sig(mp, 6)} × {sig(x, 6)}/({sig(m, 6)} + {sig(mp, 6)}) = {number(G)} mm along x.",
        unit="mm",
        setup=dict(
            mass=m,
            a=a / 1000,
            b=b / 1000,
            c=c / 1000,
            point=mp,
            offset=(x / 1000, 0.0, 0.0),
            rate=0.0,
            nudge=0.0,
            tab=TAB_INERTIA,
        ),
    )


def _point_product(rng: np.random.Generator) -> Question:
    m, a, b, c = _box(rng)
    mp = pick(rng, (0.1, 0.2, 0.3, 0.5))
    x, y = pick(rng, (50.0, 80.0, 100.0)), pick(rng, (40.0, 60.0, 80.0))
    body = Body(Shape.CUBOID, m, a / 1000, b / 1000, c / 1000, mp, (x / 1000, y / 1000, 0.0))
    answer = float(body.inertia()[0, 1]) * 1e4  # kg·cm²: a few, where kg·m² would be 10⁻⁴
    reduced = m * mp / (m + mp)
    return Question(
        f"A {sig(m, 6)} kg box carries a {sig(mp, 6)} kg point mass at ({sig(x, 6)}, {sig(y, 6)}, 0) mm from its centre. How big is the "
        "product of inertia I<sub>xy</sub> about the centre of mass, in the box's axes, in kg·cm²? (It is "
        "negative: give its size.)",
        -answer,
        f"I<sub>xy</sub> = −Σmxy about G. The box alone has none; moved to G and with the point mass it comes to "
        f"−(m m<sub>p</sub>/(m + m<sub>p</sub>))xy = −{number(reduced)} × {sig(x / 10, 6)} × {sig(y / 10, 6)} = "
        f"{number(answer)} kg·cm² (lengths in cm), of size {number(-answer)}.",
        unit="kg·cm²",
        setup=dict(
            mass=m,
            a=a / 1000,
            b=b / 1000,
            c=c / 1000,
            point=mp,
            offset=(x / 1000, y / 1000, 0.0),
            rate=0.0,
            nudge=0.0,
            tab=TAB_INERTIA,
        ),
    )


def _flip_predict(rng: np.random.Generator) -> Question:
    m, a, b, c = _box(rng)
    k = int(rng.integers(3))  # principal axis k: the longest side's (least I), the middle's, the shortest's
    side = ("longest", "middle", "shortest")[k]
    flips, keeps = "It turns over and back, again and again", "It keeps spinning about that axis, wobbling a little"
    return Question(
        f"A box {sig(a, 6)} × {sig(b, 6)} × {sig(c, 6)} mm, free in space, is spun about the axis along its {side} side, a little off "
        "it. What does it do?",
        0,
        f"The moment about an axis is m(s₁² + s₂²)/12 over the two sides across it, so the {side} side's axis has the "
        + ("least" if k == 0 else "middle" if k == 1 else "greatest")
        + " moment. "
        + (
            "Spin about the middle axis is unstable: a nudge grows, and it flips (the tennis-racket effect)."
            if k == 1
            else "Spin about the least or the greatest is stable: a nudge only circles."
        ),
        choices=(flips, keeps) if k == 1 else (keeps, flips),
        why=(
            "",
            (
                "Spin about the least or the greatest axis is stable, but this is the middle one: a nudge grows "
                "(the tennis-racket effect)."
                if k == 1
                else f"It flips only about the middle axis; the {side} side's has the "
                + ("least" if k == 0 else "greatest")
                + " moment, where a nudge only circles."
            ),
        ),
        setup=dict(mass=m, a=a / 1000, b=b / 1000, c=c / 1000, spin_axis=k, rate=6.0, nudge=2.0, show=SHOW_W),
        predict=True,
    )


def _flip_rate(rng: np.random.Generator) -> Question:
    m, a, b, c = _box(rng)
    rate = pick(rng, (2.0, 4.0, 6.0, 10.0))
    I1, I2, I3 = (rounded(v, 4) for v in Body(Shape.CUBOID, m, a / 1000, b / 1000, c / 1000).principal()[0])
    lam = rate * math.sqrt((I2 - I1) * (I3 - I2) / (I1 * I3))
    return Question(
        f"A box with principal moments {number(I1 * 1e3, 4)}, {number(I2 * 1e3, 4)} and {number(I3 * 1e3, 4)} "
        f"×10⁻³ kg·m² is spun at {sig(rate, 6)} rad/s about its middle axis. At what rate λ does a small nudge grow?",
        lam,
        f"λ = ω₀√((I₂ − I₁)(I₃ − I₂)/(I₁I₃)) = {sig(rate, 6)} × √(({number(I2 * 1e3, 4)} − {number(I1 * 1e3, 4)})"
        f"({number(I3 * 1e3, 4)} − {number(I2 * 1e3, 4)})/({number(I1 * 1e3, 4)} × {number(I3 * 1e3, 4)})) = "
        f"{number(lam)} /s: it grows e-fold every {number(1 / lam)} s.",
        unit="1/s",
        setup=dict(mass=m, a=a / 1000, b=b / 1000, c=c / 1000, spin_axis=1, rate=rate, nudge=1.0, show=SHOW_W),
    )


def _euler_rate(rng: np.random.Generator) -> Question:
    theta = pick(rng, (2.0, 5.0, 10.0, 30.0, 60.0))
    psi = pick(rng, (0.0, 30.0, 45.0, 90.0))
    w1, w2 = pick(rng, (0.5, 1.0, 2.0)), pick(rng, (0.5, 1.0, 3.0))  # both > 0: φ̇ > 0
    answer = rotation.euler_rates(math.radians(theta), math.radians(psi), np.array([w1, w2, 0.0]))[0]
    return Question(
        f"A body's Euler angles (z-x-z) are θ = {sig(theta, 6)}° and ψ = {sig(psi, 6)}°, and it turns at ω₁ = {sig(w1, 6)}, ω₂ = "
        f"{sig(w2, 6)} rad/s in its own axes. How fast is φ changing?",
        answer,
        f"φ̇ = (ω₁ sin ψ + ω₂ cos ψ)/sin θ = ({sig(w1, 6)} sin {sig(psi, 6)}° + {sig(w2, 6)} cos {sig(psi, 6)}°)/sin {sig(theta, 6)}° = "
        f"{number(answer)} rad/s" + (": the smaller θ, the faster, whatever ω does." if theta <= 5 else "."),
        unit="rad/s",
    )


def _quaternion_angle(rng: np.random.Generator) -> Question:
    q0 = pick(rng, (0.966, 0.866, 0.707, 0.5, 0.259))
    answer = math.degrees(2.0 * math.acos(q0))
    return Question(
        f"An orientation's unit quaternion has q₀ = {sig(q0, 6)}. Through what angle χ does it turn the body, about its "
        "axis?",
        answer,
        f"q₀ = cos(χ/2), so χ = 2 cos⁻¹ {sig(q0, 6)} = {number(answer)}°. The other three components are sin(χ/2) times "
        "the axis.",
        unit="°",
    )


def _cylinder_length(rng: np.random.Generator) -> Question:
    r = pick(rng, (20.0, 30.0, 50.0, 80.0))
    m = pick(rng, (0.5, 1.0, 2.0))
    length = math.sqrt(3.0) * r
    return Question(
        f"A uniform {sig(m, 6)} kg cylinder is {sig(r, 6)} mm in radius. How long must it be for its moment across (about a "
        "diameter, through its centre) to equal its moment along its axis, in mm?",
        length,
        f"Across, m(3r² + L²)/12; along, ½mr². Equal when 3r² + L² = 6r², L = √3 r = 1.732 × {sig(r, 6)} = "
        f"{number(length)} mm. Then every axis through G is principal, as for a sphere: no axis is the middle one.",
        unit="mm",
        setup=dict(
            shape=Shape.CYLINDER, mass=m, a=r / 1000, c=round(length) / 1000, rate=0.0, nudge=0.0, tab=TAB_INERTIA
        ),
    )


def _momentum_angle(rng: np.random.Generator) -> Question:
    m, a, b, c = _box(rng)
    I1, I2, _ = (rounded(v, 3) for v in Body(Shape.CUBOID, m, a / 1000, b / 1000, c / 1000).principal()[0])
    w1, w2 = pick(rng, (1.0, 2.0, 3.0)), pick(rng, (1.0, 2.0, 3.0))
    L1, L2 = I1 * w1, I2 * w2
    angle = math.degrees(math.atan2(L2, L1) - math.atan2(w2, w1))
    return Question(
        f"A box with principal moments I₁ = {number(I1 * 1e3, 3)} and I₂ = {number(I2 * 1e3, 3)} ×10⁻³ kg·m² turns "
        f"at ω = ({sig(w1, 6)}, {sig(w2, 6)}, 0) rad/s in its principal axes. By how many degrees does its angular momentum L "
        "lean away from ω?",
        abs(angle),
        f"L = Iω = (I₁ω₁, I₂ω₂, 0) = ({number(L1 * 1e3, 3)}, {number(L2 * 1e3, 3)}, 0) ×10⁻³: at "
        f"atan({number(L2 * 1e3, 3)}/{number(L1 * 1e3, 3)}) = {number(math.degrees(math.atan2(L2, L1)))}° from "
        f"axis 1, against ω's atan({sig(w2, 6)}/{sig(w1, 6)}) = {number(math.degrees(math.atan2(w2, w1)))}°: "
        f"{number(abs(angle))}° apart. Only about a principal axis are L and ω parallel.",
        unit="°",
    )


def _flip_rate_of(t: RigidTrial) -> float:
    """How fast a nudge grows about the middle axis, for `t`'s body at its rate (1/s)."""
    I1, I2, I3 = t.body.principal()[0]
    return t.rate * math.sqrt(max(0.0, (I2 - I1) * (I3 - I2)) / (I1 * I3))


def _slow_flip_spec(rng: np.random.Generator) -> Question:
    rate = pick(rng, (4.0, 6.0))
    most = pick(rng, (0.5, 0.6))

    def goal(t: RigidTrial):
        return (
            Requirement("A box", t.shape.value, t.shape is Shape.CUBOID),
            Requirement("Spun about the middle axis", "yes" if t.spin_axis == 1 else "no", t.spin_axis == 1),
            kept("The rate", t.rate, rate, "rad/s"),
            Requirement("No point mass", f"{sig(t.point, 6)} kg", t.point == 0.0),
            at_most("The nudge's growth rate λ", _flip_rate_of(t), most, "1/s"),
        )

    return Question(
        f"The box spins at {sig(rate, 6)} rad/s about its middle axis, and a nudge flips it. Change its sides (no point "
        f"mass, the same rate) until a nudge grows no faster than λ = {sig(most, 6)} /s.",
        0,
        "λ = ω₀√((I₂ − I₁)(I₃ − I₂)/(I₁I₃)) is small when the middle moment nearly equals one of the others: make "
        "two sides nearly equal. 300 × 200 × 198 mm gives λ = 0.373 /s at 6 rad/s; with two sides equal the body is "
        "symmetric and its spin about the odd axis is steady.",
        setup=dict(rate=rate, nudge=1.0, spin_axis=1, show=SHOW_W),
        goal=goal,
        design=dict(rate=rate, nudge=1.0, spin_axis=1, c=0.198, show=SHOW_W),
    )


PROBLEMS = tuple(
    Problem(key, title, "rigid_body_page", f"See {page_link('rigid_body_page', anchor, section)}.", make)
    for key, title, anchor, section, make in (
        ("box-moment", "A box's moment of inertia", "inertia-tensor", "The inertia tensor", _box_moment),
        ("point-centre", "A point mass moves G", "inertia-tensor", "The inertia tensor", _point_centre),
        ("point-product", "A product of inertia", "inertia-tensor", "The inertia tensor", _point_product),
        ("flip-predict", "Predict: does it flip?", "intermediate-axis", "The intermediate axis", _flip_predict),
        ("flip-rate", "How fast a nudge grows", "intermediate-axis", "The intermediate axis", _flip_rate),
        ("euler-rate", "Euler's rates", "orientation", "Writing down how it's turned", _euler_rate),
        ("quaternion-angle", "A quaternion's turn", "orientation", "Writing down how it's turned", _quaternion_angle),
        ("cylinder-length", "A cylinder with no middle axis", "principal-axes", "Principal axes", _cylinder_length),
        ("momentum-angle", "L leans away from ω", "inertia-tensor", "The inertia tensor", _momentum_angle),
        ("slow-flip-spec", "Spec: a slow flip", "intermediate-axis", "The intermediate axis", _slow_flip_spec),
    )
)

VIEW_TIP = (
    "<p>The body in 3D, about its centre of mass G: drag to turn the view, double-click to reset it. The coloured "
    "lines are its own axes x, y, z (dashed: its principal axes 1, 2, 3, or the one it was spun about when the "
    "<i>Nudge</i> tipped ω off it); the dot, the point mass. With "
    "<i>Vectors</i> ticked: L, ω and the torque τ, each scaled to fit, and the thin line where ω has pointed.</p>"
)
DRIFT_TIP = (
    "<p>The body's start (its shape, spin and pose; no torque) stepped by RK4 three ways at one coarse step: its "
    "rotation matrix, never corrected; its Euler angles; and its quaternion, made unit each step. Top: how far "
    "each orientation is from the true one (a quaternion stepped 8 times finer), in degrees. Below: how far the "
    "matrix has strayed from being a rotation, and how far a quaternion's length does if it isn't made unit.</p>"
)
SNAPSHOT_TIP = (
    "<p>The equations of motion at one instant, in the axes picked, with the numbers. While it runs it is the "
    "latest instant; pause it and drag the line across the plot to see any other.</p>"
)
CHECK_TIP = (
    "<p>How far the simulation's energy and angular momentum have strayed from what the torque put in "
    "(∫τ·ω dt and ∫τ dt), over their sizes: RK4's error, falling as the step's fourth power.</p>"
)

NOTES_HTML = load_notes("rigid_body", figures=rigid_body_figures.FIGURES, trials=trials_html(TRIALS))


def _shade(base: QtGui.QColor, lit: float) -> QtGui.QColor:
    """`base` lit from a side `lit` (the cosine of its angle to the light: 1 full on, 0 or less in shadow): toward
    white in the light and black in the shade, a little see-through so the axes inside still show."""
    k = max(0.0, min(1.0, lit))
    if k > 0.6:
        mix, toward = (k - 0.6) / 0.4 * 0.5, 1.0
    else:
        mix, toward = (0.6 - k) / 0.6 * 0.6, 0.0
    r, g, b = (base.redF(), base.greenF(), base.blueF())
    out = QtGui.QColor.fromRgbF(*(v + (toward - v) * mix for v in (r, g, b)))
    out.setAlphaF(0.82)
    return out


def _matrices(q: np.ndarray) -> np.ndarray:
    """The body's axes in space as rows (n, 3, 3), for quaternions q (n, 4)."""
    return np.transpose(rotation.matrix(q), (0, 2, 1))


def frame_rows(frame: Axes, body_rows: np.ndarray, P: np.ndarray) -> np.ndarray:
    """A frame's axes in space as rows (n, 3, 3): space's, the body's, or its principal axes (P's columns)."""
    if frame is Axes.SPACE:
        return np.broadcast_to(np.eye(3), body_rows.shape).copy()
    if frame is Axes.BODY:
        return body_rows
    return np.einsum("ji,njk->nik", P, body_rows)


class BodyView(OrbitView, pg.PlotWidget):
    """The body in 3D, orthographically projected about its centre of mass; drag to turn it."""

    YAW, PITCH = math.radians(30.0), math.radians(20.0)
    RIM = 48

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAccessibleName("The body in 3D")
        self.setToolTip(VIEW_TIP)
        self.setAspectLocked(True)
        for side in ("left", "bottom"):
            self.hideAxis(side)
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.hideButtons()
        self.yaw, self.pitch = self.YAW, self.PITCH
        self._drag: QtCore.QPointF | None = None
        self.body = Body()
        self.reach = 0.3
        self.show_vectors = True
        self.show_principal = False
        self.nudged: int | None = None  # the principal axis ω was nudged off at the start: drawn (dashed) even
        # with the principal axes off, so the small angle to L and ω reads as the nudge (#409 N-5)
        self._last: tuple | None = None
        plot = self.getPlotItem()
        self.space = pg.PlotCurveItem(connect="finite")  # X, Y, Z, dotted
        self.space_labels = [pg.TextItem(t, anchor=(0.5, 0.5)) for t in "XYZ"]
        self.outline = pg.PlotCurveItem(connect="finite")  # lines on the body, its visible side only
        self.far = pg.PlotCurveItem(connect="finite")  # a sphere's screen outline
        self.faces = [QtWidgets.QGraphicsPathItem() for _ in range(STRIPS + 2)]  # those facing the viewer, shaded
        self.ball = QtWidgets.QGraphicsPathItem()  # a sphere's disc, shaded by a gradient
        self.axes = [pg.PlotCurveItem() for _ in range(3)]
        self.axis_labels = [pg.TextItem(t, anchor=(0.5, 0.5)) for t in "xyz"]
        self.principal = [pg.PlotCurveItem() for _ in range(3)]
        self.principal_labels = [pg.TextItem(t, anchor=(0.5, 0.5)) for t in "123"]
        self.trail = pg.PlotCurveItem()  # ω's direction in space (Poinsot's herpolhode's direction)
        self.polhode = pg.PlotCurveItem()  # ω's direction on the body, turning with it (the polhode's)
        self.path_mode = PATH_SPACE
        self.vectors = [pg.PlotCurveItem() for _ in range(3)]  # L, ω, τ
        self.labels = [pg.TextItem(t, anchor=(0.5, 0.5)) for t in ("L", "ω", "τ")]
        self.mass_dot = Dot(10)
        self.centre = Dot(5)
        bigger = QtGui.QFont()
        bigger.setPointSizeF(bigger.pointSizeF() * 1.15)
        for label in (*self.axis_labels, *self.principal_labels, *self.labels):
            label.setFont(bigger)
        for face in (self.ball, *self.faces):
            face.setZValue(-1)  # under the lines: the axes and vectors show over the body
        self.space.setZValue(-2)  # space's axes, behind it
        for item in (
            self.space,
            *self.space_labels,
            self.ball,
            *self.faces,
            self.far,
            self.outline,
            self.trail,
            self.polhode,
            *self.axes,
            *self.principal,
            *self.vectors,
            *self.axis_labels,
            *self.principal_labels,
            *self.labels,
        ):
            plot.addItem(item)
        plot.addItem(self.mass_dot, ignoreBounds=True)
        plot.addItem(self.centre, ignoreBounds=True)
        self.apply_theme()

    def set_body(self, body: Body) -> None:
        self.body = body
        half = 0.5 * np.array([body.a, body.b, body.c]) if body.shape is Shape.CUBOID else None
        size = float(np.linalg.norm(half)) if half is not None else math.hypot(body.a, 0.5 * body.c)
        if body.shape is Shape.SPHERE:
            size = body.a
        if body.point > 0:
            size = max(size, float(np.linalg.norm(np.asarray(body.offset) - body.centre())))
        self.reach = 1.25 * size
        span = 1.5 * self.reach
        self.setRange(xRange=(-span, span), yRange=(-span, span), padding=0)
        self._place_static()

    def _place_static(self) -> None:
        R = self.reach
        ends = np.array([[1.2 * R, 0, 0], [0, 1.2 * R, 0], [0, 0, 1.2 * R]])
        line = []
        for e in ends:
            line += [np.zeros(3), e, np.full(3, np.nan)]
        self.space.setData(*self.project(np.array(line)))
        for label, e in zip(self.space_labels, ends):
            x, y = self.project(1.1 * e)
            label.setPos(float(x), float(y))
        if self._last is not None:
            self.draw(*self._last)

    def toward(self) -> np.ndarray:
        """The unit vector from the body toward the viewer, in space."""
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([-sy * math.cos(self.pitch), -cy * math.cos(self.pitch), math.sin(self.pitch)])

    def light(self) -> np.ndarray:
        """Where the light comes from: toward the viewer, from the upper left of the screen."""
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        across = np.array([cy, -sy, 0.0])  # the screen's x, in space
        up = np.cross(self.toward(), across)  # its y
        v = self.toward() + 0.6 * up - 0.4 * across
        return v / np.linalg.norm(v)

    def _faces(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """The body's flat faces in its own axes from G, each (corners (k, 3), outward normal): a box's six, a
        cylinder's STRIPS around its side and its two ends; none for a sphere."""
        body = self.body
        if body.shape is Shape.CUBOID:
            c = body.corners()
            sides = (
                ((4, 5, 7, 6), (1, 0, 0)),
                ((0, 2, 3, 1), (-1, 0, 0)),
                ((2, 6, 7, 3), (0, 1, 0)),
                ((0, 1, 5, 4), (0, -1, 0)),
                ((1, 3, 7, 5), (0, 0, 1)),
                ((0, 4, 6, 2), (0, 0, -1)),
            )
            return [(c[list(k)], np.array(n, dtype=float)) for k, n in sides]
        if body.shape is Shape.CYLINDER:
            a, h, G = body.a, 0.5 * body.c, body.centre()
            out = []
            edges = np.linspace(0.0, 2.0 * math.pi, STRIPS + 1)
            for t0, t1 in itertools.pairwise(edges):
                p0, p1 = (a * math.cos(t0), a * math.sin(t0)), (a * math.cos(t1), a * math.sin(t1))
                quad = np.array([[*p0, -h], [*p1, -h], [*p1, h], [*p0, h]]) - G
                mid = 0.5 * (t0 + t1)
                out.append((quad, np.array([math.cos(mid), math.sin(mid), 0.0])))
            ring = np.linspace(0.0, 2.0 * math.pi, self.RIM, endpoint=False)
            for z in (h, -h):
                cap = np.column_stack([a * np.cos(ring), a * np.sin(ring), np.full_like(ring, z)]) - G
                out.append((cap, np.array([0.0, 0.0, math.copysign(1.0, z)])))
            return out
        return []

    def _shape_lines(self) -> tuple[np.ndarray, np.ndarray]:
        """Lines fixed on the body, so its turning shows, in its own axes from G, as one polyline with NaN breaks,
        and each point's outward normal (drawn only where it faces the viewer): a cylinder's rims, four lines
        along its side and a mark on its end, a sphere's three great circles. A box's edges are its faces'."""
        body, nan = self.body, np.full((1, 3), np.nan)
        G = body.centre()
        ring = np.linspace(0.0, 2.0 * math.pi, self.RIM + 1)
        points: list[np.ndarray] = []
        normals: list[np.ndarray] = []
        if body.shape is Shape.CUBOID:
            return np.zeros((0, 3)), np.zeros((0, 3))
        if body.shape is Shape.CYLINDER:
            a, h = body.a, 0.5 * body.c
            radial = np.column_stack([np.cos(ring), np.sin(ring), np.zeros_like(ring)])
            for z in (-h, h):  # a rim shows if its side or its end faces the viewer
                points += [np.column_stack([a * np.cos(ring), a * np.sin(ring), np.full_like(ring, z)]), nan]
                normals += [radial + [0.0, 0.0, math.copysign(1.0, z)], nan]
            for k in range(4):
                c, s_ = math.cos(k * math.pi / 2), math.sin(k * math.pi / 2)
                points += [np.array([[a * c, a * s_, -h], [a * c, a * s_, h]]), nan]
                normals += [np.array([[c, s_, 0.0], [c, s_, 0.0]]), nan]
            points += [np.array([[0.0, 0.0, h], [a, 0.0, h]]), nan]
            normals += [np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]), nan]
            return np.vstack(points) - G, np.vstack(normals)
        a = body.a
        for i, j in ((0, 1), (1, 2), (2, 0)):  # three great circles, fixed in it
            pts = np.zeros((len(ring), 3))
            pts[:, i], pts[:, j] = np.cos(ring), np.sin(ring)
            points += [a * pts, nan]
            normals += [pts, nan]
        return np.vstack(points) - G, np.vstack(normals)

    def _draw_body(self, B: np.ndarray) -> None:
        """The faces toward the viewer, filled and lit (a convex body hides the rest), and the lines on its
        visible side."""
        c = colors
        toward, light = self.toward(), self.light()
        base = QtGui.QColor(c.mass[0])
        edge = (
            pg.mkPen(c.mass[0], width=1.5) if self.body.shape is Shape.CUBOID else QtGui.QPen(QtCore.Qt.PenStyle.NoPen)
        )
        faces = self._faces()
        for k, item in enumerate(self.faces):
            if k >= len(faces):
                item.hide()
                continue
            corners, normal = faces[k]
            n = B @ normal
            if float(n @ toward) <= 0.0:
                item.hide()
                continue
            x, y = self.project(corners @ B.T)
            path = QtGui.QPainterPath()
            path.moveTo(float(x[0]), float(y[0]))
            for u, v in zip(x[1:], y[1:]):
                path.lineTo(float(u), float(v))
            path.closeSubpath()
            item.setPath(path)
            item.setBrush(QtGui.QBrush(_shade(base, float(n @ light))))
            item.setPen(edge)
            item.show()
        points, normals = self._shape_lines()
        if len(points):
            facing = (normals @ B.T) @ toward
            shown = np.where((facing > 0.0)[:, None], points, np.nan)
            self.outline.setData(*self.project(shown @ B.T))
        else:
            self.outline.setData([], [])
        if self.body.shape is Shape.SPHERE:
            a = self.body.a
            sx, sy = self.project(B @ -self.body.centre())
            cx, cy = float(sx), float(sy)
            path = QtGui.QPainterPath()
            path.addEllipse(QtCore.QPointF(cx, cy), a, a)
            self.ball.setPath(path)
            lx, ly = self.project(light)  # the bright spot, toward the light on the screen
            glow = QtGui.QRadialGradient(QtCore.QPointF(cx + 0.45 * a * float(lx), cy + 0.45 * a * float(ly)), 1.4 * a)
            glow.setColorAt(0.0, _shade(base, 1.0))
            glow.setColorAt(1.0, _shade(base, 0.0))
            self.ball.setBrush(QtGui.QBrush(glow))
            self.ball.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
            self.ball.show()
            ring = np.linspace(0.0, 2.0 * math.pi, self.RIM + 1)
            self.far.setData(cx + a * np.cos(ring), cy + a * np.sin(ring))
        else:
            self.ball.hide()
            self.far.setData([], [])

    @coalesce.coalesced(leaf=True)
    def draw(
        self, q: np.ndarray, momentum: np.ndarray, torque: np.ndarray, trail: np.ndarray, on_body: np.ndarray
    ) -> None:
        """The body turned by the quaternion q, with L and τ (space); `trail` where ω has pointed in space (n, 3)
        and `on_body` where it has pointed in the body's axes (m, 3), drawn turning with it."""
        self._last = (q, momentum, torque, trail, on_body)
        body, R = self.body, self.reach
        B = rotation.matrix(np.asarray(q)[None])[0]
        self._draw_body(B)
        cx, cy = self.project(np.zeros((1, 3)))
        self.centre.setData(cx, cy)
        if body.point > 0:
            px, py = self.project((B @ (np.asarray(body.offset) - body.centre()))[None])
            self.mass_dot.setData(px, py)
        else:
            self.mass_dot.setData([], [])
        _, P = body.principal()
        for k in range(3):
            self._line(self.axes[k], self.axis_labels[k], B[:, k], 0.8 * R, True)
            shown = self.show_principal or k == self.nudged
            self._line(self.principal[k], self.principal_labels[k], B @ P[:, k], 0.95 * R, shown)
        I_inv = np.linalg.inv(body.inertia())
        omega = B @ (I_inv @ (B.T @ momentum))
        for item, label, v, length in zip(self.vectors, self.labels, (momentum, omega, torque), (1.2, 1.05, 0.7)):
            if not self.show_vectors or float(np.linalg.norm(v)) < 1e-12:
                item.setData([], [])
                label.hide()
                continue
            self._line(item, label, v / float(np.linalg.norm(v)), length * R, True)
        if self.path_mode in (PATH_SPACE, PATH_BOTH) and trail.size:
            self.trail.setData(*self.project(1.05 * R * trail))
        else:
            self.trail.setData([], [])
        if self.path_mode in (PATH_BODY, PATH_BOTH) and on_body.size:
            self.polhode.setData(*self.project(1.05 * R * on_body @ B.T))
        else:
            self.polhode.setData([], [])

    def _line(self, item: pg.PlotCurveItem, label: pg.TextItem, v: np.ndarray, length: float, on: bool) -> None:
        if not on:
            item.setData([], [])
            label.hide()
            return
        x, y = self.project(np.array([np.zeros(3), length * v]))
        item.setData(x, y)
        lx, ly = self.project(1.12 * length * v)
        label.setPos(float(lx), float(ly))
        label.show()

    def apply_theme(self) -> None:
        c = colors
        self.space.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DotLine))
        for label in self.space_labels:
            label.setColor(c.grey)
        self.outline.setPen(pg.mkPen(c.mass[0], width=2))
        self.far.setPen(pg.mkPen(c.mass[0], width=1))
        for item, label in zip(self.axes, self.axis_labels):
            item.setPen(pg.mkPen(c.mass[BODY_AXES], width=2))
            label.setColor(c.mass[BODY_AXES])
        for item, label in zip(self.principal, self.principal_labels):
            item.setPen(pg.mkPen(c.strong, width=2, style=QtCore.Qt.PenStyle.DashLine))
            label.setColor(c.strong)
        self.trail.setPen(pg.mkPen(c.mass[2], width=1))
        self.polhode.setPen(pg.mkPen(c.mass[1], width=1.5))
        for item, label, color in zip(self.vectors, self.labels, (c.force, c.mass[2], c.mass[3])):
            item.setPen(pg.mkPen(color, width=2.5))
            label.setColor(color)
        self.mass_dot.setBrush(pg.mkBrush(c.strong))
        self.mass_dot.setPen(pg.mkPen(None))
        self.centre.setBrush(pg.mkBrush(c.grey))
        self.centre.setPen(pg.mkPen(None))
        if self._last is not None:
            self.draw(*self._last)


class RigidBodyPage(QtWidgets.QWidget):
    """Controls on the left, the body in the centre, its inertia, components and check on the right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def transport(self) -> Transport:
        """The run bar drives the body's tumble and its playback speed."""
        return Transport(self.pause_button, self.restart_button, self.playback)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.body = Body()
        self.sim = Tumbler(self.body, (0.0, 0.0, 0.0))
        self.history = SampleBuffer(10, HISTORY)  # q, L, τ
        self.running = False  # held at the start until started
        self._frames = 0
        self._snap_t: float | None = None  # the instant dragged to; None: the latest
        self._budget = Budget()  # s of computing per step, measured
        self._pace = 1.0  # the share of the time asked for that the frames step, smoothed (below 1: slowed)
        self._nudge_note = ""  # the readout's word on the start's nudge (`restart`)

        self.trials = TrialBox(
            [t.name for t in TRIALS],
            "<p>Each sets the body up and lets it go, to show one idea from the Theory.</p>",
            predictions=[t.predict for t in TRIALS],
        )
        self.trials.picked.connect(self.load_trial)

        form = Form("Body")
        self.shape = combo([s.value for s in SHAPES], TIPS["shape"])
        form.add("Shape:", self.shape, TIPS["shape"])
        self.mass = form.number("Mass:", 0.1, 50.0, 1.0, 2, " kg", TIPS["mass"])
        self.side_a = form.number("Side a (x):", 10.0, 1000.0, 300.0, 0, " mm", TIPS["a"])
        self.side_b = form.number("Side b (y):", 10.0, 1000.0, 200.0, 0, " mm", TIPS["b"])
        self.side_c = form.number("Side c (z):", 10.0, 1000.0, 100.0, 0, " mm", TIPS["c"])
        extra = Form("Point mass")
        self.point = extra.number("Mass:", 0.0, 10.0, 0.0, 2, " kg", TIPS["point"])
        self.offset = [
            extra.number(f"At {axis}:", -500.0, 500.0, 0.0, 0, " mm", TIPS["offset"]) for axis in ("x", "y", "z")
        ]
        spin = Form("Spin")
        self.spin_axis = combo(SPIN_AXES, TIPS["spin_axis"])
        spin.add("About:", self.spin_axis, TIPS["spin_axis"])
        self.rate = spin.number("Rate:", 0.0, 30.0, 6.0, 2, " rad/s", TIPS["rate"])
        self.nudge = spin.number("Nudge:", 0.0, 90.0, 1.0, 1, "°", TIPS["nudge"])
        push = Form("Torque")
        self.held = combo([t.value for t in TORQUES], TIPS["held"])
        push.add("Held:", self.held, TIPS["held"])
        self.torque = [push.number(f"τ{k}:", -1.0, 1.0, 0.0, 3, " N·m", TIPS["torque"]) for k in "₁₂₃"]
        self.duration = push.number("For:", 0.0, 60.0, 1.0, 2, " s", TIPS["duration"])
        self.kick_button = QtWidgets.QPushButton("Kick")
        self.kick_button.setToolTip(TIPS["kick"])
        self.kick_button.clicked.connect(self.kick)
        push.row(self.kick_button)
        play = Form("Playback")
        self.playback = combo([f"{sig(p, 6)}×" for p in PLAYBACK], TIPS["playback"])
        self.playback.setCurrentIndex(REAL_TIME)
        play.add("Playback:", self.playback, TIPS["playback"])
        self.pause_button = RunButton()
        self.pause_button.toggled.connect(self._on_pause)
        self.restart_button = QtWidgets.QPushButton("Reset")
        self.restart_button.setToolTip("<p>Back to the start, held until started.</p>")
        self.restart_button.clicked.connect(self.reset)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.pause_button)
        buttons.addWidget(self.restart_button)
        play.row(buttons)
        self.vectors = QtWidgets.QCheckBox("Vectors L, ω, τ")
        self.vectors.setChecked(True)
        self.vectors.toggled.connect(self._on_view_options)
        play.row(self.vectors)
        self.principal_box = QtWidgets.QCheckBox("Principal axes")
        self.principal_box.toggled.connect(self._on_view_options)
        play.row(self.principal_box)
        self.path_box = combo(PATHS, TIPS["path"])
        self.path_box.currentIndexChanged.connect(self._on_view_options)
        play.add("ω's path:", self.path_box, TIPS["path"])
        connect_all(
            (self.shape, self.mass, self.side_a, self.side_b, self.side_c, self.point, *self.offset),
            self._on_inputs,
        )
        connect_all((self.spin_axis, self.rate, self.nudge), self.restart)
        connect_all((self.held, *self.torque), self._on_torque)

        self.view = BodyView()
        self.readout = LiveLabel(wrap=True)  # rewritten every frame (#409 P-2)

        # Inertia tab
        self.inertia_text = readout_label()
        # Components tab
        self.components = ComponentsTab(
            [f.value for f in FRAMES],
            {"show": TIPS["comp_show"], "frame": TIPS["comp_frame"], "span": TIPS["comp_span"]},
            SNAPSHOT_TIP,
        )
        self.components.changed.connect(self._draw_components)
        self.components.dragged.connect(self._on_snapshot_dragged)
        # Check tab
        self.check_plot = plot("Strayed", None, "Time", "s", log_y=True, height=220)
        self.check_plot.setToolTip(CHECK_TIP)
        add_legend(self.check_plot.getPlotItem(), offset=(10, 10))
        self.energy_item = pg.PlotDataItem(name="energy, |ΔE − W|/E")
        self.momentum_item = pg.PlotDataItem(name="momentum, |ΔL − ∫τ|/|L|")
        for item in (self.energy_item, self.momentum_item):
            self.check_plot.addItem(item)
        self.check_note = note_label()
        # Orientation tab
        pose = Form("Start pose (Euler's angles, z-x-z)")
        self.pose = [
            pose.number("φ, about Z:", -180.0, 180.0, 0.0, 1, "°", TIPS["pose"]),
            pose.number("θ, then about x:", 0.0, 180.0, 0.0, 1, "°", TIPS["pose"]),
            pose.number("ψ, then about z:", -180.0, 180.0, 0.0, 1, "°", TIPS["pose"]),
        ]
        for turn in self.pose[0::2]:
            turn.setWrapping(True)
        pose.changed.connect(self.restart)
        self.orient_text = readout_label()
        # Drift tab
        drift = Form("Step it three ways (torque-free)")
        self.drift_step = combo([name for name, _ in DRIFT_STEPS], TIPS["drift_step"])
        drift.add("Step:", self.drift_step, TIPS["drift_step"])
        self.drift_span = combo([f"{sig(v, 6)} s" for v in DRIFT_SPANS], TIPS["drift_span"])
        self.drift_span.setCurrentIndex(1)
        drift.add("For:", self.drift_span, TIPS["drift_span"])
        self.drift_button = QtWidgets.QPushButton("Step them")
        self.drift_button.setToolTip(DRIFT_TIP)
        self.drift_button.clicked.connect(self.step_three)
        drift.row(self.drift_button)
        self.drift_plot = plot("Off by", "°", "Time", "s", log_y=True, height=200)
        self.drift_plot.setToolTip(DRIFT_TIP)
        add_legend(self.drift_plot.getPlotItem(), offset=(-10, -10))  # bottom right: the errors grow upward
        self.drift_items = [pg.PlotDataItem(name=n) for n in ("rotation matrix", "Euler angles", "quaternion")]
        for item in self.drift_items:
            self.drift_plot.addItem(item)
        self.strayed_plot = plot("Not a rotation by", None, "Time", "s", log_y=True, height=160)
        self.strayed_plot.setToolTip(DRIFT_TIP)
        add_legend(self.strayed_plot.getPlotItem(), offset=(-10, -10))
        self.strayed_items = [
            pg.PlotDataItem(name=n) for n in ("matrix: ‖BᵀB − 1‖", "quaternion left alone: ||q| − 1|")
        ]
        for item in self.strayed_items:
            self.strayed_plot.addItem(item)
        self.drift_note = note_label("Press <i>Step them</i> to step the body's start three ways.")
        self.drift: Drift | None = None

        self.notes = theory(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(ColumnScroll(column(self.inertia_text)), "Inertia")
        self.tabs.addTab(self.components, "Components")
        self.tabs.addTab(ColumnScroll(column(pose, self.orient_text)), "Orientation")
        self.tabs.addTab(ColumnScroll(column(drift, self.drift_plot, self.strayed_plot, self.drift_note)), "Drift")
        self.tabs.addTab(column(self.check_plot, self.check_note, stretch=False), "Check")
        self.tabs.addTab(self.notes, "Theory")
        self.tabs.currentChanged.connect(self._draw_tab)

        three_panes(
            self,
            column(self.trials, form, extra, spin, push, play),
            stack((self.view, 5), (self.readout, 0)),
            self.tabs,
        )
        self.loop = FrameLoop(self, "3D rigid body", self._tick, self._runs, self._sim_t)
        self.tabs.setCurrentIndex(TAB_INERTIA)
        self._on_inputs()
        self.apply_theme()

    # ------------------------------------------------------------- inputs
    def load_trial(self, index: int) -> None:
        self.apply_trial(TRIALS[index])

    def load_setup(self, setup: dict) -> None:
        """Set the page up as a practice problem's `setup` says (a trial's fields, with no note)."""
        self.apply_trial(RigidTrial("", "", **setup))

    def apply_trial(self, t: RigidTrial) -> None:
        for box, value in (
            (self.shape, SHAPES.index(t.shape)),
            (self.mass, t.mass),
            (self.side_a, t.a * 1000),
            (self.side_b, t.b * 1000),
            (self.side_c, t.c * 1000),
            (self.point, t.point),
            *((box, v * 1000) for box, v in zip(self.offset, t.offset)),
            (self.spin_axis, t.spin_axis),
            (self.rate, t.rate),
            (self.nudge, t.nudge),
            (self.held, TORQUES.index(t.held)),
            *((box, v) for box, v in zip(self.torque, t.torque)),
            (self.duration, t.duration),
            *((box, v) for box, v in zip(self.pose, t.pose)),
            (self.playback, t.playback),
            (self.components.show_box, t.show),
            (self.components.frame_box, FRAMES.index(t.frame)),
            (self.principal_box, t.principal),
            (self.path_box, t.path),
        ):
            set_quietly(box, value)
        self.trials.show_note(t.note)
        self.tabs.setCurrentIndex(t.tab)
        self.running = True  # a trial lets it go
        self.pause_button.set_running(True)
        self._on_view_options()
        self._on_inputs()
        if t.tab == TAB_DRIFT:
            self.step_three()

    def practice_state(self) -> RigidTrial:
        """The page's settings, as a trial (what a practice problem reads)."""
        return RigidTrial(
            "",
            "",
            shape=SHAPES[self.shape.currentIndex()],
            mass=self.mass.value(),
            a=self.side_a.value() / 1000,
            b=self.side_b.value() / 1000,
            c=self.side_c.value() / 1000,
            point=self.point.value(),
            offset=tuple(box.value() / 1000 for box in self.offset),  # type: ignore[arg-type]
            spin_axis=self.spin_axis.currentIndex(),
            rate=self.rate.value(),
            nudge=self.nudge.value(),
            held=TORQUES[self.held.currentIndex()],
            torque=tuple(box.value() for box in self.torque),  # type: ignore[arg-type]
            duration=self.duration.value(),
            pose=tuple(box.value() for box in self.pose),  # type: ignore[arg-type]
            playback=self.playback.currentIndex(),
            tab=self.tabs.currentIndex(),
        )

    def _on_inputs(self) -> None:
        shape = SHAPES[self.shape.currentIndex()]
        names = {
            Shape.CUBOID: ("Side a (x):", "Side c (z):"),
            Shape.CYLINDER: ("Radius a:", "Length c (z):"),
            Shape.SPHERE: ("Radius a:", "Length c (z):"),
        }[shape]
        form = self.side_a.parentWidget()
        assert isinstance(form, Form), "the Body form holds the sizes"
        form.label(self.side_a).setText(names[0])
        form.label(self.side_c).setText(names[1])
        for box, on in ((self.side_b, shape is Shape.CUBOID), (self.side_c, shape is not Shape.SPHERE)):
            box.setVisible(on)
            form.label(box).setVisible(on)
        self.body = self.practice_state().body
        self.view.set_body(self.body)
        self._draw_inertia()
        self.restart()

    def _on_torque(self) -> None:
        """A new torque acts at once, for what is left of its time (a run in progress isn't let go again)."""
        s = self.practice_state()
        self.sim.held, self.sim.torque = s.held, s.torque
        self._draw_now()

    def _on_view_options(self) -> None:
        self.view.show_vectors = self.vectors.isChecked()
        self.view.show_principal = self.principal_box.isChecked()
        self.view.path_mode = self.path_box.currentIndex()
        self._draw_now()

    def kick(self) -> None:
        """Apply the torque again from now (a held-still body is let go)."""
        self._on_torque()
        self.sim.kick(self.duration.value())
        if not self.running:
            self.running = True
            self.pause_button.set_running(True)
            self.loop.start()

    def restart(self) -> None:
        """Let the body go again from the start (if it is running; held there if not)."""
        self._clear_drift()
        s = self.practice_state()
        self._snap_t = None
        omega = start_rate(self.body, s.spin_axis, s.rate, s.nudge)
        nudged = s.rate != 0.0 and s.nudge != 0.0
        self.view.nudged = s.spin_axis if nudged else None
        self._nudge_note = (
            f" ω started {sig(s.nudge, 6)}° off principal axis {s.spin_axis + 1} (dashed), toward axis "
            f"{(s.spin_axis + 1) % 3 + 1}: the Nudge."
            if nudged
            else ""
        )
        start = rotation.quaternion(rotation.from_euler(*np.radians(s.pose)))
        self.sim = Tumbler(
            self.body, (float(omega[0]), float(omega[1]), float(omega[2])), s.torque, s.held, q=list(start)
        )
        self.sim.kick(s.duration)
        self._pace = 1.0
        self.history.clear()
        self.history.extend(np.array([0.0]), np.concatenate([self.sim.q, self.sim.L, self._torque_now()])[None])
        self._draw_tab()
        self._draw_now()
        self.pause_button.set_running(self.running)
        self.loop.start()

    def _torque_now(self) -> np.ndarray:
        sim = self.sim
        if sim.held is Torque.NONE or sim.t >= sim.torque_until:
            return np.zeros(3)
        tau = np.asarray(sim.torque, dtype=float)
        return rotation.matrix(np.asarray(sim.q)[None])[0] @ tau if sim.held is Torque.BODY else tau

    def _on_pause(self, on: bool) -> None:
        self.running = on
        if not on and self.tabs.currentIndex() == TAB_COMPONENTS:
            self._draw_snapshot()
        if not on and self.tabs.currentIndex() == TAB_ORIENTATION:
            self._draw_orientation()
        self.loop.start()

    def reset(self) -> None:
        """Back to the start, held there until started."""
        self.running = False
        self.pause_button.rest()
        self.restart()

    # ------------------------------------------------------------- the loop
    def _runs(self) -> bool:
        return self.running

    def _sim_t(self) -> float:
        return self.sim.t

    def _tick(self, dt: float) -> None:
        self.step(dt * PLAYBACK[self.playback.currentIndex()], FRAME_BUDGET)

    def frame_context(self) -> str:
        return f"{self.shape.currentText()}, {self.spin_axis.currentText()}, {sig(self.rate.value(), 6)} rad/s"

    def step(self, duration: float, budget: float | None = None) -> None:
        """Run the body on by `duration` s of simulated time and redraw.

        With a `budget` (s of computing; the frame loop's), no more steps than it allows at the cost seen so far:
        past it the rest is dropped, so a fast spin plays slower than asked but every frame still ends (and Pause
        answers). None: all of it.
        """
        if budget is None:
            steps = 10**9  # all of it (MAX_STEPS_PER_ADVANCE still caps a call)
        else:
            steps = max(1, int(self._budget.allows(PROBE_STEPS, budget)))
        began = time.perf_counter()
        s = self.sim.advance(duration, max_steps=steps, thin=True)
        if self.sim.last_steps:
            self._budget.spent(time.perf_counter() - began, self.sim.last_steps)
        if duration > 0:  # the share of what was asked that ran, smoothed for the readout
            self._pace = 0.8 * self._pace + 0.2 * (1.0 - self.sim.dropped / duration)
        if s.t.size:
            self.history.extend(s.t, np.column_stack([s.q, s.momentum, s.torque]))
        self._draw_now()
        self._draw_live_tab()

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        self.view.apply_theme()
        self.components.apply_theme()
        for item, color in zip(self.drift_items, (c.mass[0], c.mass[1], c.mass[2])):
            item.setPen(pg.mkPen(color, width=1.5))
        for item, color in zip(self.strayed_items, (c.mass[0], c.mass[2])):
            item.setPen(pg.mkPen(color, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        self.energy_item.setPen(pg.mkPen(c.mass[0], width=1))
        self.momentum_item.setPen(pg.mkPen(c.mass[1], width=1))
        self._draw_inertia()
        self._draw_tab()
        self._draw_now()

    @coalesce.coalesced
    def _draw_now(self) -> None:
        n = self.history.size
        if n == 0:
            return
        y = self.history.y[n - 1]
        q, L, tau = y[0:4], y[4:7], y[7:10]
        t, ys = self.history.window(TRAIL)
        k = slice(None, None, max(1, len(t) // 600))
        B = rotation.matrix(ys[k, 0:4])
        I_inv = np.linalg.inv(self.body.inertia())
        omega = np.einsum("nij,nj->ni", B, np.einsum("ij,nj->ni", I_inv, np.einsum("nji,nj->ni", B, ys[k, 4:7])))
        size = np.linalg.norm(omega, axis=1)
        trail = omega[size > 1e-12] / size[size > 1e-12, None]
        _, ys = self.history.window(POLHODE)  # on the body: ω = I⁻¹BᵀL in its own axes
        k = slice(None, None, max(1, len(ys) // 1200))
        w_body = np.einsum("ij,nj->ni", I_inv, np.einsum("nji,nj->ni", rotation.matrix(ys[k, 0:4]), ys[k, 4:7]))
        size = np.linalg.norm(w_body, axis=1)
        on_body = w_body[size > 1e-12] / size[size > 1e-12, None]
        self.view.draw(q, L, tau, trail, on_body)
        B0 = rotation.matrix(q[None])[0]
        w = B0 @ (I_inv @ (B0.T @ L))
        energy = 0.5 * float(w @ L)
        Ln, wn = float(np.linalg.norm(L)), float(np.linalg.norm(w))
        apart = math.degrees(math.acos(max(-1.0, min(1.0, float(w @ L) / (wn * Ln))))) if wn * Ln > 0 else 0.0
        self.readout.setText(
            f"t = {self.sim.t:.2f} s: |ω| = {sig(wn)} rad/s, |L| = {sig(Ln, 4)} N·m·s, energy ½ω·L = {sig(energy, 4)} J; "
            f"ω is {apart:.1f}° from L."
            + self._nudge_note
            + (
                f" Playing at {self._pace:.0%} of the speed asked: this fast a spin needs more steps a second "
                "than a frame has time for."
                if self._pace < 0.95
                else ""
            )
        )

    @coalesce.coalesced
    def _draw_tab(self) -> None:
        i = self.tabs.currentIndex()
        if i == TAB_COMPONENTS:
            self._draw_components()
        elif i == TAB_ORIENTATION:
            self._draw_orientation()
        elif i == TAB_CHECK:
            self._draw_check()

    def _draw_live_tab(self) -> None:
        self._frames += 1
        i = self.tabs.currentIndex()
        if i == TAB_COMPONENTS:
            self._draw_components()
        elif i == TAB_ORIENTATION and (not self.running or self._frames % SNAPSHOT_FRAMES == 0):
            self._draw_orientation()
        elif i == TAB_CHECK and self._frames % 30 == 0:
            self._draw_check()

    def _draw_inertia(self) -> None:
        body = self.body
        I = body.inertia()
        moments, P = body.principal()
        G = body.centre()
        products = abs(I[0, 1]) + abs(I[0, 2]) + abs(I[1, 2]) > 1e-9 * float(np.abs(I).max())
        out = [
            "<p><b>The inertia tensor about G</b>, kg·m², in the body's own axes x, y, z (products of inertia in "
            "bold): its diagonal the moments about each axis, Σm(y² + z²) and so on, and off it the products, "
            "−Σmxy and so on.</p>",
            equation("I =", bracket(I, bold_off_diagonal=True)),
        ]
        if body.point > 0:
            out.append(
                f"<p>G is {1000 * G[0]:.1f}, {1000 * G[1]:.1f}, {1000 * G[2]:.1f} mm from the shape's centre: the "
                "shape's tensor moved there by parallel axes, plus the point mass's own m<sub>p</sub>(|p|²1 − "
                "ppᵀ).</p>".replace("-", "−")
            )
        out += [
            "<p><b>Principal axes</b>: the tensor's eigenvectors, the columns of P (in x, y, z), and its "
            "eigenvalues the principal moments. In them I is diagonal: I = P diag(I₁, I₂, I₃) Pᵀ.</p>",
            equation("P =", bracket(P), "I₁, I₂, I₃ =", bracket(moments)),
            "<p>"
            + (
                "The point mass's products of inertia tip the principal axes away from x, y, z."
                if products
                else "No products of inertia: x, y, z are principal axes already."
            )
            + " "
            + (
                "All three moments are equal (a sphere's): every axis is principal, and L is always along ω."
                if np.ptp(moments) < 1e-9 * moments.max()
                else "Spun about axis 1 or 3 it is stable; about 2, the middle one, a nudge grows and it turns over."
            )
            + "</p>",
        ]
        self.inertia_text.setText("".join(out))

    def _samples(self):
        """(t, body rows, L, τ) over the span picked, thinned to CHECK_POINTS; None before a run."""
        n = self.history.size
        if n < 2:
            return None
        span = self.components.span()
        t, y = self.history.t[:n], self.history.y[:n]
        first = 0 if span is None else int(np.searchsorted(t, t[-1] - span))
        k = slice(first, n, max(1, -(-(n - first) // CHECK_POINTS)))
        return t[k], _matrices(y[k, 0:4]), y[k, 4:7], y[k, 7:10]

    def _draw_components(self) -> None:
        tab = self.components
        show = tab.show_box.currentIndex()
        frame = FRAMES[tab.frame_box.currentIndex()]
        data = self._samples()
        if data is None:
            tab.clear("Start it to see its components.")
            return
        t, rows, momentum, torque = data
        I_body = self.body.inertia()
        _, P = self.body.principal()
        axes = frame_rows(frame, rows, P)
        if show == SHOW_I:
            turn = np.einsum("nij,nkj->nik", axes, rows)  # the body's axes in the frame's: columns
            in_axes = np.einsum("nik,kl,njl->nij", turn, I_body, turn)
        else:
            if show == SHOW_W:
                body_L = np.einsum("nij,nj->ni", rows, momentum)
                vector = np.einsum("nji,nj->ni", rows, np.linalg.solve(I_body, body_L.T).T)
            else:
                vector = momentum if show == SHOW_L else torque
            in_axes = np.einsum("nij,nj->ni", axes, vector)
        tab.draw(t, series(show, AXIS_NAMES[frame], in_axes), self._components_note(show, frame))
        if not self.running or self._frames % SNAPSHOT_FRAMES == 0:
            self._draw_snapshot()

    def _components_note(self, show: int, frame: Axes) -> str:
        torque = self.sim.held is not Torque.NONE and float(np.linalg.norm(self.sim.torque)) > 0
        if show == SHOW_I:
            return {
                Axes.SPACE: "In fixed axes I = B I<sub>body</sub> B<sup>T</sup> changes as the body turns, products "
                "of inertia and all.",
                Axes.BODY: "Constant: the mass doesn't move in the body's own axes. With the point mass off its "
                "axes, there are products of inertia.",
                Axes.PRINCIPAL: "Constant and diagonal: the principal axes are the ones without products of inertia.",
            }[frame]
        if show == SHOW_L:
            return (
                "dL/dt = τ: L changes only while the torque acts."
                if torque
                else "No torque: L is fixed in space; only its components in turning axes change."
            ) + (" In fixed axes." if frame is Axes.SPACE else " In axes that turn with the body.")
        if show == SHOW_W:
            return (
                "ω = I<sup>−1</sup>L: along L only if it spins about a principal axis (or the body is a sphere). "
                "In the principal axes ω's components are what Euler's equations move."
            )
        return "The torque, in the axes it's held in or turned into these." if torque else "No torque."

    def _instant(self) -> int:
        """The sample the snapshot and the Orientation tab show: the one dragged to (paused), else the latest with
        two samples after it."""
        n = self.history.size
        if self._snap_t is None or self.running:
            return max(n - 3, 0)
        return min(int(np.searchsorted(self.history.t[:n], self._snap_t)), n - 1)

    def _draw_orientation(self) -> None:
        """The body's orientation at that instant written four ways: B, Euler's angles (and their rates, with how
        near the lock), the quaternion (and its axis and angle), and the homogeneous transform with the point
        mass carried through it."""
        n = self.history.size
        if n == 0:
            self.orient_text.setText("")
            return
        k = self._instant()
        t, y = float(self.history.t[k]), self.history.y[k]
        q = y[0:4] / np.linalg.norm(y[0:4])
        q = -q if q[0] < 0 else q
        B = rotation.matrix(q)
        phi, theta, psi, locked = rotation.to_euler(B)
        w_body = np.linalg.solve(self.body.inertia(), B.T @ y[4:7])
        rates = rotation.euler_rates(theta, psi, w_body)
        chi, axis = rotation.axis_angle(q)
        g = self.body.centre()
        T = rotation.homogeneous(B, -B @ g)  # the shape's frame (origin at its centre) to space's (origin at G)
        deg = math.degrees
        sin = math.sin(theta)

        def rate(v: float) -> str:
            if not math.isfinite(v):
                return "∞"
            return "0" if abs(v) < 1e-9 else f"{sig(v)}".replace("-", "−")

        out = [
            f"<p><b>At t = {t:.3f} s</b>, the body's orientation four ways.</p>",
            "<p><b>1. Rotation matrix</b> B: its columns are the body's axes x, y, z in space. Nine numbers, held "
            "to three by BᵀB = 1; turns compose by multiplying, and a vector's components go from the body's axes "
            "to space's as v = B v<sub>b</sub>.</p>",
            equation("B =", bracket(B)),
            "<p><b>2. Euler's angles</b>, z-x-z: turn φ about Z, then θ about the line of nodes (where x has got "
            "to), then ψ about the body's z, so B = R<sub>z</sub>(φ) R<sub>x</sub>(θ) R<sub>z</sub>(ψ). Three "
            "numbers, as few as there can be, and easy to picture.</p>",
            f"<p>φ = {deg(phi):.1f}°, θ = {deg(theta):.1f}°, ψ = {deg(psi):.1f}°.".replace("-", "−")
            + (
                " <b>Locked:</b> sin θ = 0, so φ and ψ turn about the same axis and only their "
                + ("sum" if theta < 1.0 else "difference")
                + " is set (ψ is taken as 0)."
                if locked
                else ""
            )
            + "</p>",
            f"<p>Their rates from ω: φ̇ = (ω₁ sin ψ + ω₂ cos ψ)/sin θ = {rate(rates[0])}, θ̇ = {rate(rates[1])}, "
            f"ψ̇ = ω₃ − φ̇ cos θ = {rate(rates[2])} rad/s; 1/sin θ = "
            + ("∞" if sin < rotation.LOCK else f"{sig(1 / sin)}")
            + ". Near θ = 0 or 180° φ̇ and ψ̇ grow without bound, whatever ω is: a simulation stepping the angles "
            "fails there.</p>",
            "<p><b>3. Quaternion</b> q = (q₀, q₁, q₂, q₃) = (cos χ/2, sin χ/2 n̂): Euler's rotation theorem says "
            "every orientation is one turn χ about one axis n̂. Four numbers with |q| = 1, and q and −q are the "
            "same turn. q̇ = ½ q ⊗ (0, ω) has no singular place, which is why the page steps it.</p>",
            equation("q =", bracket(q), "n̂ =", bracket(axis)),
            f"<p>A turn of χ = {deg(chi):.1f}° about n̂.</p>",
            "<p><b>4. Homogeneous transform</b> T, 4 × 4: B with the shape's centre's place in space as a fourth "
            "column, Bd (d from G to the shape's centre, in the body's axes), so a point of the body p (from the "
            "shape's centre, with a 1 as fourth component) is at T(p, 1) in space, and a direction (0) only "
            "turns. A robot arm's links chain these, one per joint.</p>",
            equation("T =", bracket(T)),
        ]
        if self.body.point > 0:
            p = np.append(np.asarray(self.body.offset, dtype=float), 1.0)
            out += [
                "<p>The point mass, at p in the body, is at T(p, 1) in space (mm):</p>",
                equation("T", bracket(1000 * p[:3]), "→", bracket(1000 * (T @ p)[:3])),
            ]
        else:
            out.append(
                "<p>With no point mass G is the shape's centre, d = 0, and T's fourth column is (0, 0, 0, 1): add one "
                "to see the translation.</p>"
            )
        self.orient_text.setText("".join(out))

    def snapshot_now(self) -> rotation.Snapshot | None:
        """The `Snapshot` the Components tab shows: at the instant dragged to (paused), else the latest with two
        samples after it; None before a run."""
        n = self.history.size
        if n < 2:
            return None
        t = self.history.t[:n]
        k = self._instant()
        near = slice(max(k - 2, 0), min(k + 3, n))
        y = self.history.y[near]
        rows = _matrices(y[:, 0:4])
        frame = FRAMES[self.components.frame_box.currentIndex()]
        _, P = self.body.principal()
        turning = Turning.FIXED if frame is Axes.SPACE else Turning.BODY
        return rotation.snapshot(
            t[near],
            rows,
            frame_rows(frame, rows, P),
            y[:, 4:7],
            y[:, 7:10],
            self.body.inertia(),
            turning,
            k - near.start,
            kind=frame,
        )

    def _draw_snapshot(self) -> None:
        s = self.snapshot_now()
        if s is None:
            self.components.set_snapshot(None, "")
            return
        self.components.set_snapshot(
            s.t,
            snapshot_html(
                s,
                WHERE[FRAMES[self.components.frame_box.currentIndex()]],
                frame_note="AᵀB = Pᵀ, the body's axes in its principal ones: constant, as both turn with it.",
            ),
        )

    def _on_snapshot_dragged(self, t: float) -> None:
        self._snap_t = t
        self._draw_snapshot()

    def _clear_drift(self) -> None:
        self.drift = None
        for item in (*self.drift_items, *self.strayed_items):
            item.setData([], [])
        self.drift_note.setText("Press <i>Step them</i> to step the body's start three ways.")

    def step_three(self) -> None:
        """Step the body's start (no torque) three ways at the step picked, and plot how each drifts."""
        s = self.practice_state()
        omega = start_rate(self.body, s.spin_axis, s.rate, s.nudge)
        rate = float(np.linalg.norm(omega))
        if rate == 0.0:
            self._clear_drift()
            self.drift_note.setText("It isn't turning: give it a rate in <i>Spin</i> to step.")
            return
        h = 2.0 * math.pi / rate / DRIFT_STEPS[self.drift_step.currentIndex()][1]
        q0 = rotation.quaternion(rotation.from_euler(*np.radians(s.pose)))
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            d = compare(self.body, omega, q0, DRIFT_SPANS[self.drift_span.currentIndex()], h)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.drift = d
        floor = 1e-12
        for item, values in zip(self.drift_items, (d.matrix, d.euler, d.quaternion)):
            ok = np.isfinite(values)
            item.setData(d.t[ok], np.maximum(values[ok], floor))
        for item, values in zip(self.strayed_items, (d.orthogonality, d.length)):
            item.setData(d.t, np.maximum(values, floor))
        failed = np.flatnonzero(~np.isfinite(d.euler))
        euler = (
            f"The Euler angles met the lock at t = {d.t[failed[0]]:.2f} s (a rate of 1/sin θ = ∞) and stop there."
            if failed.size
            else f"The Euler angles end {sig(d.euler[-1], 2)}° off; their 1/sin θ reached {sig(np.max(d.lock))}, and the "
            "nearer the lock, the worse they step."
        )
        self.drift_note.setText(
            f"At a step of {sig(h * 1000)} ms, after {sig(d.t[-1])} s: the matrix is {sig(d.matrix[-1], 2)}° off and "
            f"‖BᵀB − 1‖ = {sig(d.orthogonality[-1], 2)}, no longer quite a rotation (left alone, it keeps growing); "
            f"the quaternion is {sig(d.quaternion[-1], 2)}° off and exactly a rotation (made unit each step; left alone "
            f"its length would be out by {sig(d.length[-1], 2)}). {euler} RK4's error falls as the step's fourth power: "
            "try the finer step."
        )

    def _draw_check(self) -> None:
        n = self.history.size
        if n < 3:
            self.energy_item.setData([], [])
            self.momentum_item.setData([], [])
            self.check_note.setText("Start it to check the simulation.")
            return
        t, y = self.history.t[:n], self.history.y[:n]
        q, L, tau = y[:, 0:4], y[:, 4:7], y[:, 7:10]
        B = rotation.matrix(q)
        I = self.body.inertia()
        body_L = np.einsum("nji,nj->ni", B, L)
        w_body = np.linalg.solve(I, body_L.T).T
        energy = 0.5 * np.einsum("ni,ni->n", w_body, body_L)
        omega = np.einsum("nij,nj->ni", B, w_body)
        dt = np.diff(t)
        # a sample's torque is what acted over the step ending at it: the work and impulse step by step
        on = tau[1:]
        work = np.concatenate([[0.0], np.cumsum(np.einsum("ni,ni->n", on, 0.5 * (omega[1:] + omega[:-1])) * dt)])
        impulse = np.vstack([np.zeros(3), np.cumsum(on * dt[:, None], axis=0)])
        scale_E = max(float(np.abs(energy).max()), 1e-300)
        scale_L = max(float(np.linalg.norm(L, axis=1).max()), 1e-300)
        e_err = np.abs(energy - energy[0] - work) / scale_E
        l_err = np.linalg.norm(L - L[0] - impulse, axis=1) / scale_L
        k = slice(1, n, max(1, n // CHECK_POINTS))
        floor = 1e-16
        self.energy_item.setData(t[k], np.maximum(e_err[k], floor))
        self.momentum_item.setData(t[k], np.maximum(l_err[k], floor))
        torque = bool(np.abs(tau).max() > 0)
        self.check_note.setText(
            f"Energy: {sig(e_err.max(), 2)} of its size at most; momentum: {sig(l_err.max(), 2)}. "
            + (
                "With a torque the energy changes by its work ∫τ·ω dt and L by its impulse ∫τ dt; what's plotted "
                "is what's left after those (the work's estimate is good to the step's square, so it sits above "
                "RK4's own error)."
                if torque
                else "Torque-free, L is fixed exactly (its rate is the torque, 0) and the energy drifts only by "
                "RK4's error, which falls as the step's fourth power."
            )
        )
