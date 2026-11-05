"""The Deriving the equations page's double pendulum: its two derivations written out, its drawing and its
practice problems (the model is core/double_pendulum.py; the page, its trials and its tabs are in deriving.py)."""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
import scipy.optimize

from ..core.double_pendulum import Angles, DoublePendulum, matrices, mix, modes, newton, slipped
from ..core.practice import Problem, Question, Requirement, at_most, kept, number, pick
from .deriving_text import (
    PENDULUM,
    SHOW_MOTION,
    TAB_CHAOS,
    TAB_PMODES,
    TAB_SWING,
    entries,
    frequencies_html,
    matrices_html,
)
from .lesson import Dot
from .links import page_link
from .style import colors

TWO_PI = 2.0 * math.pi
ANGLES = tuple(Angles)
M1, M2, L1, L2 = "m<sub>1</sub>", "m<sub>2</sub>", "L<sub>1</sub>", "L<sub>2</sub>"
TH1, TH2 = "θ<sub>1</sub>", "θ<sub>2</sub>"
W1, W2 = "θ̇<sub>1</sub>", "θ̇<sub>2</sub>"
A1, A2 = "θ̈<sub>1</sub>", "θ̈<sub>2</sub>"
SHOWN = 25.0  # degrees: a mode's upper rod swings this far, drawn


def q_names(angles: Angles) -> tuple[str, str]:
    return (TH1, TH2) if angles is Angles.ABSOLUTE else (TH1, "φ")


def mass_symbols(angles: Angles) -> tuple[str, str, str]:
    if angles is Angles.ABSOLUTE:
        return f"({M1} + {M2}){L1}²", f"{M2}{L1}{L2}", f"{M2}{L2}²"
    return f"({M1} + {M2}){L1}² + 2{M2}{L1}{L2} + {M2}{L2}²", f"{M2}{L1}{L2} + {M2}{L2}²", f"{M2}{L2}²"


def stiffness_symbols(angles: Angles) -> tuple[str, str, str]:
    if angles is Angles.ABSOLUTE:
        return f"({M1} + {M2})g{L1}", "0", f"{M2}g{L2}"
    return f"({M1} + {M2})g{L1} + {M2}g{L2}", f"{M2}g{L2}", f"{M2}g{L2}"


def _coordinates(angles: Angles) -> str:
    text = (
        f"<p>Each rod's angle from the vertical, {TH1} and {TH2}. The bobs sit at x<sub>1</sub> = {L1} sin {TH1}, "
        f"y<sub>1</sub> = −{L1} cos {TH1} and x<sub>2</sub> = x<sub>1</sub> + {L2} sin {TH2}, y<sub>2</sub> = "
        f"y<sub>1</sub> − {L2} cos {TH2} (y up from the pivot)."
    )
    if angles is Angles.RELATIVE:
        text += f" The coordinates are {TH1} and the elbow's bend φ = {TH2} − {TH1}, so {TH2} = {TH1} + φ."
    return text + "</p>"


def _shapes(p: DoublePendulum) -> str:
    slow, fast = modes(p)
    return (
        f"<p>The slower mode swings the rods together, {TH2} = {slow.ratio:.3g} {TH1}; the faster against each "
        f"other, {TH2} = {fast.ratio:.3g} {TH1}.</p>"
    ).replace("-", "−")


def lagrange_html(p: DoublePendulum, angles: Angles) -> str:
    """Lagrange's derivation: whole, then for small swings, with the pendulum's numbers."""
    m = matrices(p, angles)
    q1, q2 = q_names(angles)
    html = [
        "<h3>1. The coordinates</h3>",
        _coordinates(angles),
        "<h3>2. The kinetic and potential energies</h3>",
        f"<p>Differentiating the positions, v<sub>1</sub>² = {L1}²{W1}² and v<sub>2</sub>² = {L1}²{W1}² + "
        f"2{L1}{L2} cos({TH1} − {TH2}){W1}{W2} + {L2}²{W2}², so</p>"
        f"<p align='center'><b>T = ½({M1} + {M2}){L1}²{W1}² + {M2}{L1}{L2} cos({TH1} − {TH2}) {W1}{W2} + "
        f"½{M2}{L2}²{W2}²</b>,</p>"
        f"<p align='center'><b>V = −({M1} + {M2})g{L1} cos {TH1} − {M2}g{L2} cos {TH2}</b>.</p>"
        "<p>Unlike the car's, T depends on the angles too: its cross term carries the cosine of the rods' angle "
        "apart.</p>",
        "<h3>3. Lagrange's equations</h3>",
        f"<p>∂T/∂{W1} = ({M1} + {M2}){L1}²{W1} + {M2}{L1}{L2} cos Δ {W2} (Δ = {TH1} − {TH2}), whose time derivative "
        f"brings in −{M2}{L1}{L2} sin Δ (Δ̇){W2}; and now ∂T/∂{TH1} = −{M2}{L1}{L2} sin Δ {W1}{W2} too. The "
        f"{W1}{W2} terms cancel, and with ∂V/∂{TH1} = ({M1} + {M2})g{L1} sin {TH1}:</p><ul>"
        f"<li><b>({M1} + {M2}){L1}²{A1} + {M2}{L1}{L2}(cos Δ {A2} + sin Δ {W2}²) + ({M1} + {M2})g{L1} sin {TH1} = "
        "0</b></li>"
        f"<li><b>{M2}{L2}²{A2} + {M2}{L1}{L2}(cos Δ {A1} − sin Δ {W1}²) + {M2}g{L2} sin {TH2} = 0</b></li></ul>"
        "<p>No rod's pull appears: the coordinates already keep the rods' lengths, so the forces that do it do no "
        "work. The squared rates are the bobs' centripetal accelerations seen from the other rod.</p>",
    ]
    if angles is Angles.RELATIVE:
        html.append(
            f"<p>In {TH1} and φ the same steps (or these equations with {TH2} = {TH1} + φ, mixed by Tᵀ) give the "
            "whole equations in the bend.</p>"
        )
    html += [
        "<h3>4. Small swings</h3>",
        f"<p>Near the bottom cos Δ ≈ 1, sin θ ≈ θ, and the squared rates are second order, so they drop: M is T's "
        f"Hessian at rest and K is V's. In ({q1}, {q2}): {entries('M', mass_symbols(angles), m.M)}; "
        f"{entries('K', stiffness_symbols(angles), m.K)}.</p>",
        matrices_html(m),
        "<p>Symmetric. "
        + (
            "M couples the angles (dynamic coupling), K doesn't.</p>"
            if angles is Angles.ABSOLUTE
            else "Here both M and K couple them.</p>"
        ),
        "<h3>5. The frequencies</h3>",
        frequencies_html(m),
        _shapes(p),
    ]
    return "".join(html)


def fbd_html(p: DoublePendulum, angles: Angles) -> str:
    """The free-body diagram's derivation, the rods' pulls found and eliminated, with the pendulum's numbers."""
    rows = newton(p, angles)
    html = [
        "<h3>1. Cut each bob free</h3>",
        _coordinates(angles),
        f"<p>Bob 2 feels its weight {M2}g and rod 2's pull P<sub>2</sub>, toward bob 1. Bob 1 feels its weight, "
        "rod 1's pull P<sub>1</sub> toward the pivot, and rod 2's pull P<sub>2</sub> toward bob 2. Four unknowns: "
        f"{A1}, {A2}, P<sub>1</sub> and P<sub>2</sub>.</p>",
        "<h3>2. The accelerations</h3>",
        f"<p>Bob 1 moves on a circle: {L1}{A1} across rod 1 and {L1}{W1}² toward the pivot. Bob 2's acceleration is "
        f"bob 1's plus its own circle about bob 1: {L2}{A2} across rod 2 and {L2}{W2}² along it.</p>",
        "<h3>3. Newton, along the rods' normals</h3>",
        f"<p>Bob 2 across its rod, where P<sub>2</sub> has no part (Δ = {TH1} − {TH2}):<br><b>{M2}({L2}{A2} + "
        f"{L1}{A1} cos Δ − {L1}{W1}² sin Δ) = −{M2}g sin {TH2}</b>.</p>"
        f"<p>Bob 1 across rod 1, where P<sub>2</sub> does pull: <b>{M1}{L1}{A1} = −{M1}g sin {TH1} − P<sub>2</sub> "
        f"sin Δ</b>, and P<sub>2</sub> comes from bob 2 along its rod: P<sub>2</sub> = {M2}(g cos {TH2} + "
        f"{L2}{W2}² + {L1}{W1}² cos Δ + {L1}{A1} sin Δ). P<sub>1</sub> is left in bob 1's equation along rod 1, "
        "which isn't needed.</p>",
        "<h3>4. Small swings</h3>",
        f"<p>P<sub>2</sub> ≈ {M2}g and sin Δ ≈ {TH1} − {TH2}, so the two rows (per metre: forces) are "
        f"{M1}{L1}{A1} + ({M1} + {M2})g{TH1} − {M2}g{TH2} = 0 and {M2}{L1}{A1} + {M2}{L2}{A2} + {M2}g{TH2} = 0"
        + (", with " + TH2 + " = " + TH1 + " + φ put in" if angles is Angles.RELATIVE else "")
        + ":</p>",
        matrices_html(rows),
        "<h3>5. Check</h3>",
        f"<p>Right, but not symmetric. {L1} × (the sum of the rows) is both bobs' moments about the pivot, and "
        f"{L2} × the second is bob 2's about the elbow: those"
        + (", mixed again by Tᵀ for the bend," if angles is Angles.RELATIVE else "")
        + " are Lagrange's rows:</p>",
        matrices_html(type(rows)(*(mix(p, angles) @ A for A in (rows.M, rows.K)))),
    ]
    wrong = slipped(p)
    w = np.sort(np.sqrt(np.abs(np.linalg.eigvals(np.linalg.solve(wrong.M, wrong.K)).real))) / TWO_PI
    right = [mode.frequency for mode in modes(p)]
    html.append(
        "<p><b>The usual slip</b> is to treat bob 2 as a pendulum hanging from a fixed elbow, leaving out "
        f"{M2}{L1}{A1}, the elbow's own acceleration: the frequencies come out <b>{w[0]:.3g} and {w[1]:.3g} Hz</b> "
        f"instead of {right[0]:.3g} and {right[1]:.3g}.</p>"
    )
    html += [frequencies_html(matrices(p, angles)), _shapes(p)]
    return "".join(html)


NOTES = """
<h3><a name="double-pendulum"></a>A double pendulum</h3>
<p>Two bobs on light rods, the second hung from the first. Its free-body diagram needs the rods' pulls: bob 2
across its rod gives one equation, but bob 1's needs rod 2's pull, found from bob 2 along its rod, and both
bobs' accelerations have centripetal parts. Lagrange's equations need none of that. In the rods' angles</p>
<p align="center">T = ½(m<sub>1</sub> + m<sub>2</sub>)L<sub>1</sub>²θ̇<sub>1</sub>² + m<sub>2</sub>L<sub>1</sub>L<sub>2</sub>
cos(θ<sub>1</sub> − θ<sub>2</sub>) θ̇<sub>1</sub>θ̇<sub>2</sub> + ½m<sub>2</sub>L<sub>2</sub>²θ̇<sub>2</sub>², &nbsp;
V = −(m<sub>1</sub> + m<sub>2</sub>)gL<sub>1</sub> cos θ<sub>1</sub> − m<sub>2</sub>gL<sub>2</sub> cos θ<sub>2</sub>,</p>
<p>and two differentiations give the whole nonlinear equations; the rods' pulls never appear, because the
coordinates keep the rods' lengths and those forces do no work. For small swings M and K are T's and V's
Hessians at rest. With equal bobs and rods, ω² = (2 ∓ √2)g/L: the rods swing together (θ<sub>2</sub> =
√2 θ<sub>1</sub>) or against each other (−√2 θ<sub>1</sub>).</p>
<p>Large swings are another matter. Released in a mode's shape at 40°, its period is 4 % longer and the shape
doesn't hold. Released high, the motion is <b>chaotic</b>: two releases a millionth of a radian apart drift
apart exponentially, doubling their difference about every half second, until nothing about one predicts the
other. The Swing tab sets the whole equations against the linear ones, and the Chaos tab follows the twins.</p>
"""


# ------------------------------------------------------------------ the drawing
class PendulumView(pg.PlotWidget):
    """The pendulum from the side: the pivot, the rods, the bobs and the lower bob's trail."""

    def __init__(self) -> None:
        super().__init__()
        self.setMenuEnabled(False)
        self.hideButtons()
        self.setAspectLocked(True)
        self.setMouseEnabled(False, False)
        for side in ("left", "bottom"):
            self.hideAxis(side)
        self.trail = pg.PlotDataItem()
        self.rods = pg.PlotDataItem()
        self.pivot = Dot(8)
        self.bobs = [Dot(16), Dot(16)]
        for item in (self.trail, self.rods):
            self.addItem(item)
        for dot in (self.pivot, *self.bobs):
            self.addItem(dot, ignoreBounds=True)
        self.caption = pg.TextItem(anchor=(0, 0))
        self.addItem(self.caption, ignoreBounds=True)
        self.getViewBox().sigRangeChanged.connect(self._place_caption)
        self._reach = 0.0

    def _place_caption(self) -> None:
        (x0, _), (_, y1) = self.getViewBox().viewRange()
        self.caption.setPos(x0, y1)

    def apply_theme(self) -> None:
        c = colors
        self.rods.setPen(pg.mkPen(c.structure, width=2.5))
        self.trail.setPen(pg.mkPen(c.mass[1], width=1))
        self.pivot.setPen(pg.mkPen(c.structure, width=1.5))
        self.pivot.setBrush(pg.mkBrush(c.background))
        for dot, color in zip(self.bobs, (c.mass[0], c.mass[1])):
            dot.setPen(pg.mkPen(color, width=1.5))
            dot.setBrush(pg.mkBrush(color))
        self.caption.setColor(c.muted)

    def draw(self, p: DoublePendulum, th1: float, th2: float, trail: tuple[np.ndarray, np.ndarray] | None) -> None:
        reach = 1.1 * (p.upper_length + p.lower_length)
        if reach != self._reach:
            self._reach = reach
            self.setXRange(-reach, reach, padding=0)
            self.setYRange(-reach, reach, padding=0)
            heavier = max(p.upper, p.lower)
            for dot, m in zip(self.bobs, (p.upper, p.lower)):
                size = 8 + 14 * (m / heavier) ** (1 / 3)
                dot.setRect(-0.5 * size, -0.5 * size, size, size)
        x1, y1, x2, y2 = (float(v) for v in p.positions(th1, th2))
        self.rods.setData([0.0, x1, x2], [0.0, y1, y2])
        self.pivot.setData([0.0], [0.0])
        self.bobs[0].setData([x1], [y1])
        self.bobs[1].setData([x2], [y2])
        if trail is None:
            self.trail.setData([], [])
        else:
            self.trail.setData(*trail)


# ------------------------------------------------------------------ practice problems
# Each question's numbers are printed round, and its answer is worked from them as printed.
P = "deriving_page"


def _equal_frequency(rng: np.random.Generator) -> Question:
    m, L = pick(rng, (0.5, 1, 2)), pick(rng, (0.25, 0.5, 1.0))
    slow = modes(DoublePendulum(m, m, L, L))[0].frequency
    return Question(
        f"A double pendulum has two {m:g} kg bobs on light rods {L:g} m long. At what frequency is its slower "
        "small-swing mode?",
        slow,
        f"M = mL²[[2, 1], [1, 1]] and K = mgL diag(2, 1), so det(K − ω²M) = 0 gives ω² = (2 ∓ √2)g/L. The slower: "
        f"ω = √({number(2 - math.sqrt(2))} × 9.81/{L:g}) = {number(TWO_PI * slow)} rad/s, f = {number(slow)} Hz.",
        unit="Hz",
        setup=dict(system=PENDULUM, upper=m, lower=m, upper_length=L, lower_length=L),
    )


def _cross_mass(rng: np.random.Generator) -> Question:
    m2, a, b = pick(rng, (0.5, 1, 2, 3)), pick(rng, (0.3, 0.5, 0.8)), pick(rng, (0.2, 0.4, 0.6))
    entry = m2 * a * b
    return Question(
        f"A double pendulum's lower bob is {m2:g} kg, its rods {a:g} m (upper) and {b:g} m (lower). In the rods' "
        "angles from the vertical, what is M₁₂, the mass matrix's coupling term?",
        entry,
        f"T's cross term is m<sub>2</sub>L<sub>1</sub>L<sub>2</sub> cos(θ<sub>1</sub> − θ<sub>2</sub>) θ̇<sub>1</sub>"
        f"θ̇<sub>2</sub>, cos → 1 at rest: M₁₂ = {m2:g} × {a:g} × {b:g} = {number(entry)} kg·m².",
        unit="kg·m²",
        setup=dict(system=PENDULUM, lower=m2, upper_length=a, lower_length=b),
    )


def _shape(rng: np.random.Generator) -> Question:
    m1, m2 = pick(rng, (1, 2, 3)), pick(rng, (0.5, 1, 2))
    L = pick(rng, (0.4, 0.5, 0.8))
    slow = modes(DoublePendulum(m1, m2, L, L))[0]
    mu = m2 / m1
    return Question(
        f"A double pendulum with equal rods, {m1:g} kg above and {m2:g} kg below, swings in its slower mode. What "
        "is the ratio of the lower rod's angle to the upper's?",
        slow.ratio,
        f"With equal rods and μ = m<sub>2</sub>/m<sub>1</sub> = {number(mu)}, write λ = ω²L/g. The second row of "
        "(K − ω²M)φ = 0 gives θ<sub>2</sub>/θ<sub>1</sub> = λ/(1 − λ), and the determinant (1 + μ)(1 − λ)² = μλ², so "
        f"in the slower mode θ<sub>2</sub>/θ<sub>1</sub> = √((1 + μ)/μ) = {number(slow.ratio)}.",
        tolerance=0.01,
        setup=dict(system=PENDULUM, upper=m1, lower=m2, upper_length=L, lower_length=L),
    )


def _small_release(rng: np.random.Generator) -> Question:
    return Question(
        "The double pendulum (equal bobs and rods) is released from rest with the upper rod at 10° and the lower at "
        "14.1°, √2 times as far. What does it do?",
        0,
        "It swings in its slower mode, both rods together and the shape held, at the linear model's 0.540 Hz: the "
        "release is the mode's own shape, and at 10° the whole equations barely differ from the linear ones.",
        choices=(
            "Swings in its slower mode, the shape held",
            "Swings in its faster mode",
            "Mixes both modes irregularly",
            "Tumbles over the top",
        ),
        setup=dict(system=PENDULUM, theta1=10.0, theta2=14.1, show=SHOW_MOTION, tab=TAB_SWING),
        predict=True,
    )


def _twins(rng: np.random.Generator) -> Question:
    return Question(
        "Two double pendulums are released at 120° and 120°, the second's lower rod a millionth of a radian further "
        "over. Ten seconds later, how do their motions compare?",
        0,
        "They're wholly different: at that height the motion is chaotic, and the difference doubles about every "
        "0.49 s, past a radian within ten seconds.",
        choices=(
            "Wholly different",
            "Still a millionth of a radian apart",
            "About a thousandth of a radian apart",
            "Identical: the equations are deterministic",
        ),
        setup=dict(system=PENDULUM, theta1=120.0, theta2=120.0, show=SHOW_MOTION, tab=TAB_CHAOS),
        predict=True,
    )


def _ratio(p: DoublePendulum) -> float:
    slow, fast = modes(p)
    return fast.frequency / slow.frequency


def _close_spec(rng: np.random.Generator) -> Question:
    m1, L = pick(rng, (1, 2, 4)), pick(rng, (0.4, 0.5, 0.6))
    limit = pick(rng, (1.4, 1.6, 1.8))
    mu = scipy.optimize.brentq(lambda u: _ratio(DoublePendulum(1.0, u, L, L)) - 0.97 * limit, 1e-4, 1.0)
    m2 = math.floor(100 * mu * m1) / 100

    def goal(t):
        p = t.pendulum
        return (
            Requirement("The double pendulum", "on" if t.system == PENDULUM else "off", t.system == PENDULUM),
            kept("The upper bob", t.upper, m1, "kg"),
            kept("The upper rod", t.upper_length, L, "m"),
            kept("The lower rod", t.lower_length, L, "m"),
            at_most("The faster mode's frequency over the slower's", _ratio(p), limit),
        )

    return Question(
        f"A double pendulum with a {m1:g} kg upper bob and two {L:g} m rods. Choose the lower bob so that its two "
        f"modes' frequencies are within a factor of {limit:g} of each other.",
        0,
        f"A lighter lower bob brings the modes together (with equal bobs they're 2.41 apart). {m2:g} kg gives "
        f"{number(_ratio(DoublePendulum(m1, m2, L, L)))}.",
        setup=dict(system=PENDULUM, upper=m1, lower=m1, upper_length=L, lower_length=L, tab=TAB_PMODES),
        goal=goal,
        design=dict(system=PENDULUM, upper=m1, lower=m2, upper_length=L, lower_length=L, tab=TAB_PMODES),
    )


PROBLEMS = tuple(
    Problem(key, title, P, f"See {page_link(P, 'double-pendulum', 'A double pendulum')}.", make)
    for key, title, make in (
        ("pendulum-frequency", "A double pendulum's slower mode", _equal_frequency),
        ("pendulum-mass", "M₁₂ of a double pendulum", _cross_mass),
        ("pendulum-shape", "The slower mode's shape", _shape),
        ("pendulum-release", "Predict: released in a mode's shape", _small_release),
        ("pendulum-twins", "Predict: two releases at 120°", _twins),
        ("pendulum-spec", "Spec: modes close together", _close_spec),
    )
)
