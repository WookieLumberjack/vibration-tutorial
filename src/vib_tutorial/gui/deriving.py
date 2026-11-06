"""Deriving the equations page: a car's body in bounce and pitch, and a double pendulum, by free-body diagrams
and by Lagrange.

Left: the system, then its controls (the car's body, suspension, coordinates and a bump; the pendulum's bobs,
angles and release). Centre: the system, swinging in a mode or released, and its matrices in the coordinates
chosen. Right: the two derivations step by step with the page's numbers in them, then the system's own tabs
(the car's modes against the rear spring and its ride; the pendulum's modes against its lower bob, its swing
against the linear one, and two releases drifting apart), and the Theory. The page has its own models
(core/bounce_pitch.py, core/double_pendulum.py); the pendulum's derivations, drawing and problems are in
deriving_pendulum.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.bounce_pitch import (
    TWO_PI,
    Bump,
    Car,
    Coordinates,
    matrices,
    modes,
    newton,
    ride,
    slipped,
    sweep,
)
from ..core.double_pendulum import Angles, DoublePendulum, divergence, swing
from ..core.double_pendulum import matrices as pendulum_matrices
from ..core.double_pendulum import modes as pendulum_modes
from ..core.double_pendulum import sweep as pendulum_sweep
from ..core.pendulum import measured_period
from ..core.practice import Problem, Question, Requirement, at_most, kept, number, pick
from . import deriving_pendulum as dp
from . import frame_stats, settle_plots
from .deriving_text import (
    CAR,
    PENDULUM,
    SHOW_MODE1,
    SHOW_MOTION,
    TAB_BUMP,
    TAB_CHAOS,
    TAB_FBD,
    TAB_LAGRANGE,
    TAB_MODES,
    TAB_PMODES,
    TAB_SWING,
    TAB_THEORY,
    dotted,
    entries,
    frequencies_html,
    matrices_html,
    num,
)
from .lesson import (
    Dot,
    Form,
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
from .style import colors
from .theming import add_legend

SYSTEMS = ("A car in bounce and pitch", "A double pendulum")
SYSTEM_TABS = {  # the tabs each system shows
    CAR: (TAB_FBD, TAB_LAGRANGE, TAB_MODES, TAB_BUMP, TAB_THEORY),
    PENDULUM: (TAB_FBD, TAB_LAGRANGE, TAB_PMODES, TAB_SWING, TAB_CHAOS, TAB_THEORY),
}
MOTIONS = ("Ride over the bump", "Swing, released")  # each system's last Show item
COORDS = tuple(Coordinates)
ANGLES = tuple(Angles)
MASS_RATIOS = np.geomspace(0.01, 100.0, 161)  # the pendulum's Modes tab: the lower bob, × the upper
SWING_S = 20.0  # s of the pendulum's swing
CHAOS_S = 15.0  # s the twins are followed
TRAIL_S = 2.0  # s of the lower bob's trail
RATIOS = np.geomspace(0.25, 4.0, 161)  # the Modes tab's rear springs, × the front's
FRAME_MS = 16
SHOW_HZ = 0.6  # the modes swing at this pace, whatever their frequency
SWING = 0.25  # m: a mode's larger end swings this far, drawn
RIDE_SCALE = 3.0  # the ride's road and motion drawn this many times their size
OVERHANG = 0.85  # m: the body past each axle
WHEEL = 0.3  # m: a wheel's radius
FLOOR, ROOF, CABIN = 0.85, 1.3, 1.75  # m: the body's underside, waist and roof at rest
G_HEIGHT = 1.1  # m: the centre of mass's height
VIEW_X = 3.6  # m either side of G


@dataclass(frozen=True)
class DerivingTrial:
    """The page's settings: the system shown, and both systems' controls."""

    name: str
    note: str
    system: int = CAR
    mass: float = 1200.0  # kg
    gyration: float = 1.3  # m
    front: float = 1.2  # m
    rear: float = 1.5  # m
    k_front: float = 40.0  # kN/m
    k_rear: float = 40.0  # kN/m
    c_front: float = 2500.0  # N·s/m
    c_rear: float = 2500.0  # N·s/m
    coords: Coordinates = Coordinates.CENTRE
    offset: float = 0.0  # m: P ahead of G
    speed_kmh: float = 30.0
    height: float = 75.0  # mm
    length: float = 3.7  # m
    upper: float = 1.0  # kg
    lower: float = 1.0  # kg
    upper_length: float = 0.5  # m
    lower_length: float = 0.5  # m
    angles: Angles = Angles.ABSOLUTE
    theta1: float = 10.0  # degrees, released from
    theta2: float = 14.1  # degrees
    show: int = SHOW_MODE1
    tab: int = TAB_LAGRANGE

    @property
    def car(self) -> Car:
        return Car(
            self.mass,
            self.gyration,
            self.front,
            self.rear,
            1e3 * self.k_front,
            1e3 * self.k_rear,
            self.c_front,
            self.c_rear,
        )

    @property
    def bump(self) -> Bump:
        return Bump(self.speed_kmh / 3.6, self.height / 1e3, self.length)

    @property
    def pendulum(self) -> DoublePendulum:
        return DoublePendulum(self.upper, self.lower, self.upper_length, self.lower_length)

    @property
    def release(self) -> tuple[float, float]:
        return math.radians(self.theta1), math.radians(self.theta2)


TRIALS = (
    DerivingTrial(
        "A car in bounce and pitch",
        "A 1200 kg body (r = 1.3 m) on 40 kN/m at each axle, 1.2 m ahead of G and 1.5 m behind it. Its slower "
        "mode (1.25 Hz) turns about a node 1.92 m behind G, outside the wheelbase: mostly bounce. The faster "
        "(1.41 Hz) turns about a node 0.88 m ahead of G, between the axles: mostly pitch. Neither is pure, "
        "because k<sub>f</sub>a ≠ k<sub>r</sub>b: the springs couple the two (static coupling).",
    ),
    DerivingTrial(
        "No static coupling",
        "The rear spring at 32 kN/m, so k<sub>f</sub>a = k<sub>r</sub>b = 48 kN: K is diagonal in (x<sub>G</sub>, "
        "θ), and the modes are pure bounce (1.23 Hz, no node) and pure pitch (1.27 Hz, about G). The Modes "
        "tab's nodes run off to infinity there.",
        k_rear=32.0,
        tab=TAB_MODES,
    ),
    DerivingTrial(
        "Coordinates at the elastic centre",
        "x<sub>P</sub> and θ at the elastic centre, e* = (k<sub>f</sub>a − k<sub>r</sub>b)/(k<sub>f</sub> + "
        "k<sub>r</sub>) = −0.15 m (behind G): a force there pushes the body straight, so K is diagonal. The "
        "coupling has moved into M (−m e = 180 kg·m): dynamic coupling. The frequencies are the same 1.25 and "
        "1.41 Hz: coupling belongs to the coordinates, not the car.",
        coords=Coordinates.POINT,
        offset=-0.15,
    ),
    DerivingTrial(
        "Moments about the front axle",
        "x<sub>P</sub> and θ with P above the front axle (e = 1.2 m), by the free-body diagram. Taking moments "
        "about P keeps the moment of G's inertia force, −m e ẍ<sub>G</sub>: the moment balance is (J + m e²)θ̈ "
        "− m e ẍ<sub>P</sub> = ΣM<sub>P</sub>. Leave that term out, as J<sub>P</sub>θ̈ = ΣM<sub>P</sub>, and "
        "the frequencies come out 0.93 and 1.38 Hz instead of 1.25 and 1.41.",
        coords=Coordinates.POINT,
        offset=1.2,
        tab=TAB_FBD,
    ),
    DerivingTrial(
        "The two ends as coordinates",
        "x<sub>f</sub> and x<sub>r</sub>, the body above each axle. Each spring now sees one coordinate, so K "
        "is diagonal, and M couples by (m a b − J)/L² = 18.1 kg. The free-body diagram's force balance gives "
        "M the row (667, 533) kg and its moment balance (751, −751) kg·m: right, but not symmetric. Moments "
        "about each axle, which are Lagrange's rows, make them so.",
        coords=Coordinates.ENDS,
        tab=TAB_FBD,
    ),
    DerivingTrial(
        "Dynamic index 1: the ends uncouple",
        "The radius of gyration at √(ab) = 1.34 m, so J = m a b and the dynamic index r²/(ab) is 1. In the "
        "ends' coordinates M is diagonal too, and each end bounces on its own spring: 1.23 Hz at the front "
        "(667 kg on 40 kN/m), 1.38 Hz at the rear (533 kg), the nodes on the axles. A bump at the front then "
        "barely moves the rear: most cars are near 1.",
        gyration=1.342,
        coords=Coordinates.ENDS,
        tab=TAB_MODES,
    ),
    DerivingTrial(
        "Over a hump, front stiffer than rear",
        "The front on 48 kN/m and the rear on 30: the front bounces at 1.35 Hz and the rear at 1.19. Over a "
        "75 mm hump at 30 km/h the rear wheel arrives 0.32 s after the front and the slower rear falls further "
        "behind: once both wheels are past, the body pitches up to 1.95°.",
        k_front=48.0,
        k_rear=30.0,
        show=SHOW_MOTION,
        tab=TAB_BUMP,
    ),
    DerivingTrial(
        "Olley's flat ride",
        "The springs swapped: 30 kN/m at the front (1.07 Hz) and 48 at the rear (1.51 Hz). Now the rear, hit "
        "later, catches the front up, and once past the hump the body pitches only 1.52°. Olley's rule: the "
        "front's ride frequency below the rear's (here by 29 %).",
        k_front=30.0,
        k_rear=48.0,
        show=SHOW_MOTION,
        tab=TAB_BUMP,
    ),
    DerivingTrial(
        "A double pendulum, small swings",
        "Two 1 kg bobs on 0.5 m rods. Lagrange's T has a cross term m<sub>2</sub>L<sub>1</sub>L<sub>2</sub> "
        "cos(θ<sub>1</sub> − θ<sub>2</sub>)θ̇<sub>1</sub>θ̇<sub>2</sub>, so for small swings M couples the angles "
        "(0.25 kg·m²) and K doesn't. The slower mode (0.540 Hz) swings the rods together, θ<sub>2</sub> = 1.41 "
        "θ<sub>1</sub>; the faster (1.30 Hz) against each other, −1.41 θ<sub>1</sub>. Each rod alone would swing "
        "at 0.705 Hz.",
        system=PENDULUM,
    ),
    DerivingTrial(
        "The double pendulum's free-body diagram",
        "Each bob across its rod: bob 2 needs no rod's pull there, but bob 1 needs rod 2's, found from bob 2 along "
        "its rod. In M the rows come out (0.5, 0) and (0.5, 0.5) kg·m: right, but not symmetric; L<sub>1</sub> × "
        "their sum and L<sub>2</sub> × the second are Lagrange's. Hanging bob 2 from a fixed elbow (the elbow's "
        "acceleration left out) would make both modes 0.705 Hz, a simple pendulum's each.",
        system=PENDULUM,
        tab=TAB_FBD,
    ),
    DerivingTrial(
        "The elbow's bend as a coordinate",
        "θ<sub>1</sub> and φ = θ<sub>2</sub> − θ<sub>1</sub>: now K couples too (4.91 N·m), and M is fuller "
        "(M₁₁ = 1.25 kg·m²). The frequencies are the same 0.540 and 1.30 Hz; the slower mode bends the elbow by "
        "0.41 θ<sub>1</sub>.",
        system=PENDULUM,
        angles=Angles.RELATIVE,
    ),
    DerivingTrial(
        "Released in its slower mode",
        "Released from rest at 10° and 14.1° (√2 times as far), the mode's own shape: it swings in that mode, "
        "with a period of 1.858 s against the linear 1.853, and after 20 s the upper rod is within 1.8° of the "
        "linear model's.",
        system=PENDULUM,
        show=SHOW_MOTION,
        tab=TAB_SWING,
    ),
    DerivingTrial(
        "A larger swing in the same shape",
        "Released at 40° and 56.6°: the period is 1.924 s, 3.8 % longer than the linear 1.853, and the shape "
        "doesn't hold, since the whole equations couple the modes. By 20 s the linear model is 77° out.",
        system=PENDULUM,
        theta1=40.0,
        theta2=56.6,
        show=SHOW_MOTION,
        tab=TAB_SWING,
    ),
    DerivingTrial(
        "Chaos: released at 120° and 120°",
        "High up the motion is chaotic. A twin released 10⁻⁶ rad further over drifts away, the difference doubling "
        "every 0.49 s, and by 10 s nothing about one predicts the other. The equations are exact and the "
        "integration tight (the energy holds to 10⁻¹⁰): the sensitivity is the pendulum's own.",
        system=PENDULUM,
        theta1=120.0,
        theta2=120.0,
        show=SHOW_MOTION,
        tab=TAB_CHAOS,
    ),
)


# ------------------------------------------------------------------ the derivations, written out
def q_names(coords: Coordinates) -> tuple[str, str]:
    return {
        Coordinates.CENTRE: ("x<sub>G</sub>", "θ"),
        Coordinates.POINT: ("x<sub>P</sub>", "θ"),
        Coordinates.ENDS: ("x<sub>f</sub>", "x<sub>r</sub>"),
    }[coords]


def deflections(coords: Coordinates) -> tuple[str, str]:
    """δ_f and δ_r in the coordinates."""
    return {
        Coordinates.CENTRE: ("x<sub>G</sub> + aθ", "x<sub>G</sub> − bθ"),
        Coordinates.POINT: ("x<sub>P</sub> + (a − e)θ", "x<sub>P</sub> − (b + e)θ"),
        Coordinates.ENDS: ("x<sub>f</sub>", "x<sub>r</sub>"),
    }[coords]


def mass_symbols(coords: Coordinates) -> tuple[str, str, str]:
    """M₁₁, M₁₂, M₂₂ in symbols."""
    return {
        Coordinates.CENTRE: ("m", "0", "J"),
        Coordinates.POINT: ("m", "−m e", "J + m e²"),
        Coordinates.ENDS: ("(m b² + J)/L²", "(m a b − J)/L²", "(m a² + J)/L²"),
    }[coords]


def spring_symbols(coords: Coordinates, k: str) -> tuple[str, str, str]:
    """K₁₁, K₁₂, K₂₂ (or C's, with k = "c") in symbols."""
    f, r = f"{k}<sub>f</sub>", f"{k}<sub>r</sub>"
    return {
        Coordinates.CENTRE: (f"{f} + {r}", f"{f}a − {r}b", f"{f}a² + {r}b²"),
        Coordinates.POINT: (f"{f} + {r}", f"{f}(a − e) − {r}(b + e)", f"{f}(a − e)² + {r}(b + e)²"),
        Coordinates.ENDS: (f, "0", r),
    }[coords]


def coordinates_html(car: Car, coords: Coordinates, e: float) -> str:
    df, dr = deflections(coords)
    if coords is Coordinates.CENTRE:
        where = "q = (x<sub>G</sub>, θ): G's rise and the pitch, nose up."
    elif coords is Coordinates.POINT:
        where = f"q = (x<sub>P</sub>, θ), P e = {e:.3g} m ahead of G: x<sub>G</sub> = x<sub>P</sub> − eθ (for small θ)."
    else:
        where = (
            f"q = (x<sub>f</sub>, x<sub>r</sub>), the body above each axle (L = {car.wheelbase:.3g} m): "
            "x<sub>G</sub> = (b x<sub>f</sub> + a x<sub>r</sub>)/L, θ = (x<sub>f</sub> − x<sub>r</sub>)/L."
        )
    return (
        f"<p>{where} The springs stretch δ<sub>f</sub> = {df} and δ<sub>r</sub> = {dr}, measured from where the "
        "car sits at rest, so the weight and the springs' preload cancel and gravity drops out.</p>"
    )


def fbd_html(car: Car, coords: Coordinates, e: float) -> str:
    """The free-body diagram's derivation in the coordinates, with the car's numbers."""
    q1, q2 = q_names(coords)
    html = [
        "<h3>1. Cut the body free</h3>",
        coordinates_html(car, coords, e),
        "<p>The springs and dampers push on the body with F<sub>f</sub> = −k<sub>f</sub>δ<sub>f</sub> − "
        "c<sub>f</sub>δ̇<sub>f</sub> at a ahead of G, and F<sub>r</sub> = −k<sub>r</sub>δ<sub>r</sub> − "
        "c<sub>r</sub>δ̇<sub>r</sub> at b behind it.</p>",
        "<h3>2. Newton and Euler</h3>",
    ]
    if coords is Coordinates.CENTRE:
        html.append(
            "<p>Forces: <b>m ẍ<sub>G</sub> = F<sub>f</sub> + F<sub>r</sub></b>. Moments about G: <b>J θ̈ = a "
            "F<sub>f</sub> − b F<sub>r</sub></b>. Both are already in q.</p>"
        )
        rows = matrices(car, coords)
    elif coords is Coordinates.POINT:
        html.append(
            "<p>Forces: m ẍ<sub>G</sub> = F<sub>f</sub> + F<sub>r</sub>, and ẍ<sub>G</sub> = ẍ<sub>P</sub> − eθ̈, so "
            "<b>m ẍ<sub>P</sub> − m e θ̈ = F<sub>f</sub> + F<sub>r</sub></b>.</p>"
            "<p>Moments about P, which isn't G: the forces' moment about P is J θ̈ plus the moment of G's "
            "inertia force m ẍ<sub>G</sub>, which acts e behind P: ΣM<sub>P</sub> = (a − e)F<sub>f</sub> − (b + e)"
            "F<sub>r</sub> = J θ̈ − m e ẍ<sub>G</sub>, so <b>(J + m e²)θ̈ − m e ẍ<sub>P</sub> = (a − e)F<sub>f</sub> − "
            "(b + e)F<sub>r</sub></b>.</p>"
        )
        rows = matrices(car, coords, e)
    else:
        html.append(
            "<p>Forces: <b>m ẍ<sub>G</sub> = F<sub>f</sub> + F<sub>r</sub></b>, moments about G: <b>J θ̈ = a "
            "F<sub>f</sub> − b F<sub>r</sub></b>, with ẍ<sub>G</sub> = (b ẍ<sub>f</sub> + a ẍ<sub>r</sub>)/L and θ̈ = "
            "(ẍ<sub>f</sub> − ẍ<sub>r</sub>)/L put in.</p>"
        )
        rows = newton(car, coords)
    html += [
        "<h3>3. In matrix form</h3>",
        f"<p>Moving the springs' and dampers' terms across, the two balances are M q̈ + C q̇ + K q = 0 with q = "
        f"({q1}, {q2}):</p>",
        matrices_html(rows),
        "<h3>4. Check</h3>",
    ]
    lagrange = matrices(car, coords, e)
    if coords is Coordinates.CENTRE:
        html.append("<p>Symmetric, and the same as Lagrange's. About G the free-body diagram is the quickest way.</p>")
    elif coords is Coordinates.POINT:
        wrong = slipped(car, e)
        w = np.sort(np.sqrt(np.abs(np.real(np.linalg.eigvals(np.linalg.solve(wrong.M, wrong.K)))))) / TWO_PI
        right = [m.frequency for m in modes(car, coords, e)]
        if abs(e) < 1e-9:
            html.append("<p>At G (e = 0) this is the moment balance about G itself.</p>")
        else:
            html.append(
                "<p>Symmetric, and the same as Lagrange's. <b>The usual slip</b> is to write the moment balance "
                "about P as J<sub>P</sub>θ̈ = ΣM<sub>P</sub>, J<sub>P</sub> = J + m e², leaving out −m e ẍ<sub>P</sub>: "
                f"M's lower left becomes 0 instead of {num(-car.mass * e).replace('-', '−')}, the matrices stop "
                f"being symmetric, and the frequencies come out <b>{w[0]:.3g} and {w[1]:.3g} Hz</b> instead of "
                f"{right[0]:.3g} and {right[1]:.3g}. Moments about a point that isn't G need G's inertia force "
                "too.</p>"
            )
    else:
        html.append(
            "<p>Right, but not symmetric: the rows are a force and a moment, each mixing both coordinates. Mix "
            "them again: b/L × the force balance + 1/L × the moment balance is the moment balance about the "
            "rear axle, over L; a/L × the force balance − 1/L × the moment balance is the one about the front. "
            "That is Tᵀ times the rows, T the matrix taking q to (x<sub>G</sub>, θ), and it gives Lagrange's "
            "symmetric matrices:</p>" + matrices_html(lagrange)
        )
    html.append(frequencies_html(lagrange))
    return "".join(html)


VELOCITIES = {  # ẋ_G and θ̇ in q̇, as T needs them
    Coordinates.CENTRE: "",
    Coordinates.POINT: ", with ẋ<sub>G</sub> = ẋ<sub>P</sub> − e θ̇",
    Coordinates.ENDS: ", with ẋ<sub>G</sub> = (b ẋ<sub>f</sub> + a ẋ<sub>r</sub>)/L and θ̇ = (ẋ<sub>f</sub> − "
    "ẋ<sub>r</sub>)/L",
}


def lagrange_html(car: Car, coords: Coordinates, e: float) -> str:
    """Lagrange's derivation in the coordinates, with the car's numbers."""
    m = matrices(car, coords, e)
    q1, q2 = q_names(coords)
    v1, v2 = dotted(q1, 1), dotted(q2, 1)
    html = [
        "<h3>1. The coordinates</h3>",
        coordinates_html(car, coords, e),
        "<h3>2. The kinetic energy</h3>",
        f"<p>T = ½ m ẋ<sub>G</sub>² + ½ J θ̇²{VELOCITIES[coords]}, is a quadratic in the velocities:</p>"
        f"<p><b>T = ½(M₁₁{v1}² + 2M₁₂{v1}{v2} + M₂₂{v2}²)</b>, {entries('M', mass_symbols(coords), m.M)}.</p>",
        "<h3>3. The potential energy and the dissipation</h3>",
        f"<p><b>V = ½ k<sub>f</sub>δ<sub>f</sub>² + ½ k<sub>r</sub>δ<sub>r</sub>² = ½(K₁₁{q1}² + 2K₁₂{q1}{q2} + "
        f"K₂₂{q2}²)</b>, {entries('K', spring_symbols(coords, 'k'), m.K)}.</p>"
        f"<p>Rayleigh's dissipation function <b>D = ½ c<sub>f</sub>δ̇<sub>f</sub>² + ½ c<sub>r</sub>δ̇<sub>r</sub>²</b> "
        f"in the same way: {entries('C', spring_symbols(coords, 'c'), m.C)}.</p>",
        "<h3>4. Lagrange's equations</h3>",
        "<p>For each coordinate q<sub>i</sub>: <b>d/dt(∂T/∂q̇<sub>i</sub>) − ∂T/∂q<sub>i</sub> + ∂D/∂q̇<sub>i</sub> + "
        "∂V/∂q<sub>i</sub> = Q<sub>i</sub></b>, Q<sub>i</sub> = 0 for the free car. T depends on the velocities "
        "alone, so ∂T/∂q<sub>i</sub> = 0, and each derivative of a quadratic is a row of its matrix:</p><ul>",
    ]
    for i, (qi, vi) in enumerate(((q1, v1), (q2, v2))):
        terms = []
        for A, order in ((m.M, 2), (m.C, 1), (m.K, 0)):
            for j, qj in enumerate((q1, q2)):
                if abs(A[i, j]) > 1e-9:
                    sign = "−" if A[i, j] < 0 else "+"
                    name = dotted(qj, order) if order else qj
                    terms.append(f"{sign} {num(abs(A[i, j]))} {name}")
        eq = " ".join(terms).removeprefix("+ ")
        if eq.startswith("− "):
            eq = "−" + eq[2:]
        html.append(
            f"<li>{qi}: ∂T/∂{vi} = M<sub>{i + 1}1</sub>{v1} + M<sub>{i + 1}2</sub>{v2}, and so <b>{eq} = 0</b></li>"
        )
    html += [
        "</ul><h3>5. The matrices</h3>",
        matrices_html(m),
        "<p>Symmetric in any coordinates: each is the matrix of a quadratic form. An off-diagonal term (bold) in "
        "M is <b>dynamic coupling</b>, in K <b>static coupling</b>.",
    ]
    couplings = [name for name, A in (("M", m.M), ("K", m.K)) if abs(A[0, 1]) > 1e-9 * max(abs(A[0, 0]), 1.0)]
    html.append(
        f" Here {' and '.join(couplings)} {'couple' if len(couplings) > 1 else 'couples'} the coordinates.</p>"
        if couplings
        else " Here neither couples them: each coordinate is a mode.</p>"
    )
    html += ["<h3>6. The frequencies</h3>", frequencies_html(m), "<p>The same in every set of coordinates.</p>"]
    return "".join(html)


# ------------------------------------------------------------------ practice problems
# Each question's numbers are printed round, and its answer is worked from them as printed.


def _bounce(rng: np.random.Generator) -> Question:
    m, kf, kr = pick(rng, (900, 1200, 1500, 1800)), pick(rng, (30, 40, 50)), pick(rng, (30, 40, 50, 60))
    car = Car(mass=m, k_front=1e3 * kf, k_rear=1e3 * kr)
    f = car.uncoupled_frequencies()[0]
    return Question(
        f"A {m:g} kg car body sits on {kf:g} kN/m at the front axle and {kr:g} kN/m at the rear. If it bounced "
        "without pitching, at what frequency would it?",
        f,
        f"Straight up and down both springs stretch alike: f = √((k<sub>f</sub> + k<sub>r</sub>)/m)/2π = "
        f"√({number(1e3 * (kf + kr))}/{m:g})/2π = {number(f)} Hz. It only bounces purely if k<sub>f</sub>a = "
        "k<sub>r</sub>b.",
        unit="Hz",
        setup=dict(mass=m, k_front=kf, k_rear=kr, tab=TAB_MODES),
    )


def _elastic_centre(rng: np.random.Generator) -> Question:
    a, b = pick(rng, (1.0, 1.1, 1.2)), pick(rng, (1.4, 1.5, 1.6))
    kf, kr = pick(rng, (30, 35, 40)), pick(rng, (40, 45, 50))
    e = Car(front=a, rear=b, k_front=kf, k_rear=kr).elastic_centre()
    return Question(
        f"A car's front axle is {a:g} m ahead of its centre of mass and the rear {b:g} m behind, on {kf:g} and "
        f"{kr:g} kN/m. How far behind G is its elastic centre?",
        -e,
        f"Where a force pushes the body straight, the springs' moments about it cancel: e* = (k<sub>f</sub>a − "
        f"k<sub>r</sub>b)/(k<sub>f</sub> + k<sub>r</sub>) = ({kf:g} × {a:g} − {kr:g} × {b:g})/{kf + kr:g} = "
        f"{number(e)} m, behind G since the rear's k<sub>r</sub>b is the larger. In coordinates there, K is "
        "diagonal.",
        unit="m",
        tolerance=0.01,
        setup=dict(front=a, rear=b, k_front=kf, k_rear=kr, coords=Coordinates.POINT, offset=round(e, 3)),
    )


def _end_frequency(rng: np.random.Generator) -> Question:
    m, a, b = pick(rng, (1000, 1200, 1500)), pick(rng, (1.0, 1.2, 1.4)), pick(rng, (1.3, 1.5, 1.7))
    kf = pick(rng, (30, 35, 40, 45))
    mf = m * b / (a + b)
    f = math.sqrt(1e3 * kf / mf) / TWO_PI
    return Question(
        f"A {m:g} kg car has its centre of mass {a:g} m behind the front axle and {b:g} m ahead of the rear. "
        f"The front springs give {kf:g} kN/m. What is the front's ride frequency, the front end bouncing on its "
        "own spring?",
        f,
        f"The front axle carries m b/L = {m:g} × {b:g}/{number(a + b)} = {number(mf)} kg, so f = √({number(1e3 * kf)}/"
        f"{number(mf)})/2π = {number(f)} Hz. It's exactly a mode's frequency when the dynamic index r²/(ab) is 1.",
        unit="Hz",
        setup=dict(mass=m, front=a, rear=b, k_front=kf, coords=Coordinates.ENDS, tab=TAB_MODES),
    )


def _index_one(rng: np.random.Generator) -> Question:
    a, b = pick(rng, (1.0, 1.2, 1.4)), pick(rng, (1.3, 1.5, 1.7))
    r = math.sqrt(a * b)
    return Question(
        f"A car's axles are {a:g} m ahead of and {b:g} m behind its centre of mass. What radius of gyration in "
        "pitch makes its ends bounce independently?",
        r,
        f"In the ends' coordinates M couples by (m a b − J)/L², nil when J = m r² = m a b: r = √(ab) = √({number(a * b)}) "
        f"= {number(r)} m, a dynamic index r²/(ab) of 1.",
        unit="m",
        tolerance=0.01,
        setup=dict(front=a, rear=b, gyration=round(r, 3), coords=Coordinates.ENDS, tab=TAB_MODES),
    )


def _point_mass(rng: np.random.Generator) -> Question:
    m, r, e = pick(rng, (1000, 1200, 1500)), pick(rng, (1.1, 1.2, 1.3, 1.4)), pick(rng, (0.5, 0.8, 1.0, 1.2))
    J = m * r * r
    entry = J + m * e * e
    return Question(
        f"A {m:g} kg body with a {r:g} m radius of gyration is described by x<sub>P</sub> and θ at a point P {e:g} m "
        "ahead of its centre of mass. What is M₂₂, the entry multiplying θ̈ in the moment equation?",
        entry,
        f"x<sub>G</sub> = x<sub>P</sub> − eθ, so T = ½m(ẋ<sub>P</sub> − eθ̇)² + ½Jθ̇² and M₂₂ = J + m e² = {number(J)} + "
        f"{m:g} × {e:g}² = {number(entry)} kg·m² (the parallel-axis theorem). M₁₂ = −m e = {number(-m * e)} kg·m couples it.",
        unit="kg·m²",
        setup=dict(mass=m, gyration=r, coords=Coordinates.POINT, offset=e),
    )


def _ends_mass(rng: np.random.Generator) -> Question:
    m, r = pick(rng, (1000, 1200, 1500)), pick(rng, (0.9, 1.0, 1.1))  # r² < ab: M₁₂ > 0
    a, b = pick(rng, (1.0, 1.2, 1.4)), pick(rng, (1.3, 1.5, 1.7))
    L = a + b
    entry = (m * a * b - m * r * r) / L**2
    return Question(
        f"A {m:g} kg body (radius of gyration {r:g} m) with its axles {a:g} m ahead of and {b:g} m behind G, "
        "described by x<sub>f</sub> and x<sub>r</sub>, the body above each axle. What is M₁₂, its mass matrix's "
        "coupling term?",
        entry,
        f"x<sub>G</sub> = (b x<sub>f</sub> + a x<sub>r</sub>)/L and θ = (x<sub>f</sub> − x<sub>r</sub>)/L, so T's "
        f"cross term gives M₁₂ = (m a b − J)/L² = ({m:g} × {a:g} × {b:g} − {m:g} × {r:g}²)/{number(L)}² = "
        f"{number(entry)} kg.",
        unit="kg",
        setup=dict(mass=m, gyration=r, front=a, rear=b, coords=Coordinates.ENDS, tab=TAB_LAGRANGE),
    )


def _same_frequencies(rng: np.random.Generator) -> Question:
    return Question(
        "The car's equations are derived again in x<sub>P</sub> and θ at the front axle instead of x<sub>G</sub> "
        "and θ. M and K change: what happens to the natural frequencies?",
        0,
        "Nothing: the coordinates describe the same motion, so the frequencies (1.25 and 1.41 Hz) don't change. "
        "Only where the coupling shows does: the mass matrix now couples them too (−m e).",
        choices=("They stay the same", "They rise", "They fall", "One rises and one falls"),
        setup=dict(coords=Coordinates.POINT, offset=1.2),
        predict=True,
    )


def _pure_modes(rng: np.random.Generator) -> Question:
    return Question(
        "The rear spring is softened until k<sub>f</sub>a = k<sub>r</sub>b. What do the two modes become?",
        0,
        "Pure bounce and pure pitch: K is diagonal in (x<sub>G</sub>, θ), and so is M, so each is a mode. The "
        "bounce's node goes off to infinity and the pitch's sits on G.",
        choices=(
            "Pure bounce and pure pitch about G",
            "Each end bouncing on its own",
            "Both pitch about the axles",
            "They stay mixed as before",
        ),
        setup=dict(k_rear=32.0, tab=TAB_MODES),
        predict=True,
    )


def _uncouple_spec(rng: np.random.Generator) -> Question:
    a, b = pick(rng, (1.0, 1.1, 1.2, 1.3)), pick(rng, (1.4, 1.5, 1.6))
    kf = pick(rng, (35, 40, 45))
    kr = kf * a / b
    base = DerivingTrial("", "")

    def goal(t: DerivingTrial):
        car = t.car
        return (
            kept("The front axle", t.front, a, "m"),
            kept("The rear axle", t.rear, b, "m"),
            kept("The front spring", t.k_front, kf, "kN/m"),
            kept("The mass", t.mass, base.mass, "kg"),
            kept("The radius of gyration", t.gyration, base.gyration, "m"),
            Requirement("The equations in x<sub>G</sub> and θ", t.coords.value, t.coords is Coordinates.CENTRE),
            at_most("The elastic centre's distance from G", 100 * abs(car.elastic_centre()), 0.5, "cm"),
        )

    return Question(
        f"The car with its axles {a:g} m ahead of and {b:g} m behind G and {kf:g} kN/m at the front. Choose the "
        "rear spring so that, in x<sub>G</sub> and θ, bounce and pitch are its modes: the elastic centre within "
        "5 mm of G.",
        0,
        f"Static coupling goes when k<sub>f</sub>a = k<sub>r</sub>b: k<sub>r</sub> = {kf:g} × {a:g}/{b:g} = "
        f"{number(kr)} kN/m.",
        setup=dict(front=a, rear=b, k_front=kf, k_rear=60.0, tab=TAB_LAGRANGE),
        goal=goal,
        design=dict(front=a, rear=b, k_front=kf, k_rear=round(kr, 1), tab=TAB_LAGRANGE),
    )


PROBLEMS = (
    tuple(
        Problem(key, title, "deriving_page", f"See {page_link('deriving_page', anchor, section)}.", make)
        for key, title, anchor, section, make in (
            ("bounce", "Pure bounce", "bounce-pitch", "A car in bounce and pitch", _bounce),
            ("elastic-centre", "The elastic centre", "coupling", "Coordinates and coupling", _elastic_centre),
            ("end-frequency", "A ride frequency", "coupling", "Coordinates and coupling", _end_frequency),
            ("index-one", "Dynamic index 1", "coupling", "Coordinates and coupling", _index_one),
            ("point-mass", "M about a point", "lagrange", "Lagrange's equations", _point_mass),
            ("ends-mass", "M in the ends' coordinates", "lagrange", "Lagrange's equations", _ends_mass),
            (
                "same-frequencies",
                "Predict: other coordinates",
                "coupling",
                "Coordinates and coupling",
                _same_frequencies,
            ),
            ("pure-modes", "Predict: k_f a = k_r b", "coupling", "Coordinates and coupling", _pure_modes),
            (
                "uncouple-spec",
                "Spec: uncouple bounce and pitch",
                "coupling",
                "Coordinates and coupling",
                _uncouple_spec,
            ),
        )
    )
    + dp.PROBLEMS
)

TIPS = {
    "mass": "<p>The body's mass (the wheels left out: the Quarter-car page has them).</p>",
    "gyration": "<p>The body's radius of gyration in pitch about G: J = m r².</p>",
    "front": "<p>a: from G forward to the front axle.</p>",
    "rear": "<p>b: from G back to the rear axle.</p>",
    "k_front": "<p>The front suspension's stiffness, both wheels together, at the axle.</p>",
    "k_rear": "<p>The rear suspension's stiffness, both wheels together, at the axle.</p>",
    "c_front": "<p>The front dampers, both together, at the axle.</p>",
    "c_rear": "<p>The rear dampers, both together, at the axle.</p>",
    "coords": "<p>The two coordinates the equations are written in. The motion and its frequencies are the "
    "same in each; M, C and K are not.</p>",
    "offset": "<p>e: how far P is ahead of G (negative behind).</p>",
    "speed": "<p>The car's speed over the bump.</p>",
    "height": "<p>The bump's height (a 1 − cos shape).</p>",
    "length": "<p>The bump's length along the road.</p>",
    "show": "<p>A mode swinging (at a slow pace of its own, not its frequency); or the car's ride over the bump "
    "in real time, its heights drawn three times their size; or the pendulum released, by the whole equations.</p>",
    "system": "<p>The car's body in bounce and pitch (linear from the start), or a double pendulum (nonlinear: "
    "its small swings linearized, its large ones not).</p>",
    "upper": "<p>m₁, the bob on the upper rod.</p>",
    "lower": "<p>m₂, the bob hung below it.</p>",
    "upper_length": "<p>L₁, from the pivot to the upper bob.</p>",
    "lower_length": "<p>L₂, from the upper bob to the lower.</p>",
    "angles": "<p>Each rod's angle from the vertical, or the upper rod's and the elbow's bend. The motion and "
    "its frequencies are the same in each; M and K are not.</p>",
    "theta": "<p>Where the rod is released from rest (from the vertical; 180° is straight up).</p>",
}
MODES_TIP = (
    "<p>The two modes as the rear spring changes (the front's held): their frequencies (top) and where each "
    "turns the body (bottom), from G, forward. Grey: each end on its own spring. Red: this car's rear spring; grey "
    "dotted: where k<sub>f</sub>a = k<sub>r</sub>b.</p>"
)
BUMP_TIP = "<p>The body above each axle, and its pitch, as the car rides over the bump.</p>"
PMODES_TIP = (
    "<p>The pendulum's two small-swing modes as the lower bob changes (the upper held): their frequencies (top) "
    "and shapes (bottom, the lower rod's angle over the upper's). Grey: each rod as a pendulum alone. Red: this "
    "pendulum.</p>"
)
SWING_TIP = "<p>Released from rest: the whole equations (solid) against the linear ones (dashed).</p>"
CHAOS_TIP = (
    "<p>Two releases, the second's lower rod 10⁻⁶ rad further over: how far apart they swing (log scale). A "
    "straight rise is exponential growth: chaos.</p>"
)

NOTES_HTML = load_notes("deriving", pendulum=dp.NOTES, trials=trials_html(TRIALS))


# ------------------------------------------------------------------ the car, drawn
def _body_outline(car: Car) -> np.ndarray:
    """The body's side, (x from G, z from the road at rest), closed."""
    a, b = car.front, car.rear
    x0, x1 = -b - OVERHANG, a + OVERHANG
    return np.array(
        [
            (x0, FLOOR),
            (x1, FLOOR),
            (x1, ROOF - 0.1),
            (a - 0.1, ROOF),
            (a - 0.7, CABIN),
            (-b + 0.2, CABIN),
            (-b - 0.4, ROOF),
            (x0, ROOF - 0.05),
            (x0, FLOOR),
        ]
    )


def _spring(x: float, bottom: float, top: float, turns: int = 6, width: float = 0.09) -> np.ndarray:
    """A zigzag spring from (x, bottom) up to (x, top)."""
    z = np.linspace(bottom, top, 2 * turns + 3)
    dx = np.zeros_like(z)
    dx[1:-1:2], dx[2:-1:2] = width, -width
    dx[-2] = 0.0
    return np.column_stack([x + dx, z])


def _circle(x: float, z: float, r: float, n: int = 40) -> np.ndarray:
    t = np.linspace(0, TWO_PI, n)
    return np.column_stack([x + r * np.cos(t), z + r * np.sin(t)])


class CarView(pg.PlotWidget):
    """The car side on: its body, springs and wheels, the road, G, the coordinates' points and a mode's node."""

    def __init__(self) -> None:
        super().__init__()
        self.setMenuEnabled(False)
        self.hideButtons()
        self.setAspectLocked(True)
        self.setMouseEnabled(False, False)
        for side in ("left", "bottom"):
            self.hideAxis(side)
        self.setXRange(-VIEW_X, VIEW_X, padding=0)
        self.setYRange(-0.25, 2.1, padding=0)
        self.road = pg.PlotDataItem()
        self.body = pg.PlotDataItem()
        self.springs = pg.PlotDataItem(connect="finite")
        self.wheels = pg.PlotDataItem(connect="finite")
        self.node = pg.InfiniteLine(angle=90)
        self.centre = Dot(10)
        self.points = [Dot(9), Dot(9)]
        for item in (self.road, self.springs, self.wheels, self.body):
            self.addItem(item)
        self.addItem(self.node, ignoreBounds=True)
        for dot in (self.centre, *self.points):
            self.addItem(dot, ignoreBounds=True)
        self.caption = pg.TextItem(anchor=(0, 0))
        self.addItem(self.caption, ignoreBounds=True)
        self.getViewBox().sigRangeChanged.connect(self._place_caption)

    def _place_caption(self) -> None:
        """Keep the caption in the top left corner, however tall the view's aspect makes it."""
        (x0, _), (_, y1) = self.getViewBox().viewRange()
        self.caption.setPos(x0 + 0.05, y1 - 0.05)

    def apply_theme(self) -> None:
        c = colors
        self.road.setPen(pg.mkPen(c.structure, width=2))
        self.body.setPen(pg.mkPen(c.mass[0], width=2.5))
        self.body.setFillLevel(None)
        self.springs.setPen(pg.mkPen(c.structure, width=1.5))
        self.wheels.setPen(pg.mkPen(c.structure, width=2))
        self.node.setPen(pg.mkPen(c.fit, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        self.centre.setPen(pg.mkPen(c.structure, width=1.5))
        self.centre.setBrush(pg.mkBrush(c.background))
        for dot in self.points:
            dot.setPen(pg.mkPen(c.force, width=1.5))
            dot.setBrush(pg.mkBrush(c.force))
        self.caption.setColor(c.muted)

    def draw(
        self,
        car: Car,
        rise: float,
        pitch: float,
        road: tuple[np.ndarray, np.ndarray] | None,
        wheels: tuple[float, float],
        points: list[float],
        node: float | None,
    ) -> None:
        """The body risen `rise` (m, at G) and pitched `pitch` (rad, nose up), drawn as given; the road's
        (x, z), the road under each wheel, the coordinates' points (x from G) and the node (x, or None)."""
        cs, sn = math.cos(pitch), math.sin(pitch)

        def place(p: np.ndarray) -> np.ndarray:
            x, z = p[:, 0], p[:, 1] - G_HEIGHT
            return np.column_stack([x * cs - z * sn, G_HEIGHT + rise + x * sn + z * cs])

        outline = place(_body_outline(car))
        self.body.setData(outline[:, 0], outline[:, 1])
        if road is None:
            road = (np.array([-VIEW_X, VIEW_X]), np.zeros(2))
        self.road.setData(*road)
        parts, rims = [], []
        for x, y in zip((car.front, -car.rear), wheels):
            top = place(np.array([[x, FLOOR]]))[0]
            parts += [_spring(top[0], WHEEL + y, top[1]), np.full((1, 2), np.nan)]
            rims += [_circle(x, WHEEL + y, WHEEL), np.full((1, 2), np.nan)]
        s, w = np.vstack(parts), np.vstack(rims)
        self.springs.setData(s[:, 0], s[:, 1])
        self.wheels.setData(w[:, 0], w[:, 1])
        g = place(np.array([[0.0, G_HEIGHT]]))[0]
        self.centre.setData([g[0]], [g[1]])
        for dot, x in zip(self.points, points + [None] * 2):
            if x is None:
                dot.setData([], [])
            else:
                p = place(np.array([[x, G_HEIGHT]]))[0]
                dot.setData([p[0]], [p[1]])
        if node is not None and abs(node) < VIEW_X:
            self.node.setPos(node)
            self.node.show()
        else:
            self.node.hide()


class DerivingPage(QtWidgets.QWidget):
    """Controls on the left, the system and its matrices in the centre, the derivations and the system's own
    tabs on the right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.trials = TrialBox(
            [t.name for t in TRIALS],
            "<p>Each sets up the car or the pendulum to show one idea from the Theory tab.</p>",
        )
        self.trials.picked.connect(self.load_trial)

        chosen = Form("System")
        self.system = combo(SYSTEMS, TIPS["system"])
        chosen.add("Derive:", self.system, TIPS["system"])

        # --- the car's controls
        body = Form("Body")
        self.mass = body.number("Mass m:", 100.0, 5000.0, 1200.0, 0, " kg", TIPS["mass"])
        self.gyration = body.number("Radius of gyration r:", 0.3, 3.0, 1.3, 3, " m", TIPS["gyration"])
        self.front = body.number("G to front axle a:", 0.3, 3.0, 1.2, 2, " m", TIPS["front"])
        self.rear = body.number("G to rear axle b:", 0.3, 3.0, 1.5, 2, " m", TIPS["rear"])
        self.body_info = note_label()
        body.row(self.body_info)

        suspension = Form("Suspension (per axle)")
        self.k_front = suspension.number("Front spring k_f:", 1.0, 500.0, 40.0, 1, " kN/m", TIPS["k_front"])
        self.k_rear = suspension.number("Rear spring k_r:", 1.0, 500.0, 40.0, 1, " kN/m", TIPS["k_rear"])
        self.c_front = suspension.number("Front damper c_f:", 0.0, 50000.0, 2500.0, 0, " N·s/m", TIPS["c_front"])
        self.c_rear = suspension.number("Rear damper c_r:", 0.0, 50000.0, 2500.0, 0, " N·s/m", TIPS["c_rear"])

        coordinates = Form("Coordinates")
        self.coords = combo([c.value for c in COORDS], TIPS["coords"])
        coordinates.add("Write the equations in:", self.coords, TIPS["coords"])
        self.offset = coordinates.number("P ahead of G, e:", -3.0, 3.0, 0.0, 3, " m", TIPS["offset"])
        self.coords_form = coordinates

        bump = Form("Bump")
        self.speed = bump.number("Speed:", 5.0, 150.0, 30.0, 0, " km/h", TIPS["speed"])
        self.height = bump.number("Height:", 5.0, 200.0, 75.0, 0, " mm", TIPS["height"])
        self.length = bump.number("Length:", 0.1, 10.0, 3.7, 2, " m", TIPS["length"])
        self.car_forms = (body, suspension, coordinates, bump)

        # --- the pendulum's controls
        bobs = Form("Bobs and rods")
        self.upper = bobs.number("Upper bob m₁:", 0.01, 100.0, 1.0, 3, " kg", TIPS["upper"])
        self.lower = bobs.number("Lower bob m₂:", 0.01, 100.0, 1.0, 3, " kg", TIPS["lower"])
        self.upper_length = bobs.number("Upper rod L₁:", 0.05, 5.0, 0.5, 2, " m", TIPS["upper_length"])
        self.lower_length = bobs.number("Lower rod L₂:", 0.05, 5.0, 0.5, 2, " m", TIPS["lower_length"])
        angles = Form("Coordinates")
        self.angles = combo([a.value for a in ANGLES], TIPS["angles"])
        angles.add("Write the equations in:", self.angles, TIPS["angles"])
        release = Form("Release (from rest)")
        self.theta1 = release.number("Upper rod θ₁:", -180.0, 180.0, 10.0, 1, "°", TIPS["theta"])
        self.theta2 = release.number("Lower rod θ₂:", -180.0, 180.0, 14.1, 1, "°", TIPS["theta"])
        self.pendulum_forms = (bobs, angles, release)

        show = Form("Show")
        self.show_box = combo(("Mode 1", "Mode 2", MOTIONS[CAR]), TIPS["show"])
        show.add("Show:", self.show_box, TIPS["show"])
        self.animate = QtWidgets.QCheckBox("Animate")
        self.animate.setChecked(True)
        show.row(self.animate)

        connect_all(
            (
                self.mass,
                self.gyration,
                self.front,
                self.rear,
                self.k_front,
                self.k_rear,
                self.c_front,
                self.c_rear,
                self.coords,
                self.offset,
                self.speed,
                self.height,
                self.length,
                self.upper,
                self.lower,
                self.upper_length,
                self.lower_length,
                self.angles,
                self.theta1,
                self.theta2,
            ),
            self.redraw,
        )
        self.system.currentIndexChanged.connect(lambda _: self._on_system())
        self.show_box.currentIndexChanged.connect(lambda _: self._on_show())
        self.animate.toggled.connect(lambda _: self._on_show())

        # --- centre
        self.view = CarView()
        self.pview = dp.PendulumView()
        self.views = QtWidgets.QStackedWidget()
        self.views.addWidget(self.view)
        self.views.addWidget(self.pview)
        self.matrix_label = readout_label()
        self.readout = readout_label()

        # --- right: the derivations
        self.fbd = theory("")
        self.lagrange = theory("")

        # the car's modes and ride
        self.freq_plot = plot("Frequency", "Hz", "Rear spring / front spring", None, log_x=True, height=170)
        self.node_plot = plot("Node, ahead of G", "m", "Rear spring / front spring", None, log_x=True, height=170)
        self.node_plot.setXLink(self.freq_plot)
        for p in (self.freq_plot, self.node_plot):
            p.setToolTip(MODES_TIP)
        add_legend(self.freq_plot.getPlotItem(), offset=(10, 10))
        self.freq_items = [pg.PlotDataItem(name=f"mode {i + 1}") for i in range(2)]
        self.end_items = [pg.PlotDataItem(name=n) for n in ("front end alone", "rear end alone")]
        self.node_items = [pg.PlotDataItem(connect="finite") for _ in range(2)]
        self.axle_lines = [pg.InfiniteLine(angle=0), pg.InfiniteLine(angle=0)]
        self.now_lines = [pg.InfiniteLine(angle=90), pg.InfiniteLine(angle=90)]
        self.balance_lines = [pg.InfiniteLine(angle=90), pg.InfiniteLine(angle=90)]
        for item in (*self.end_items, *self.freq_items, self.now_lines[0], self.balance_lines[0]):
            self.freq_plot.addItem(item)
        for item in (*self.axle_lines, *self.node_items, self.now_lines[1], self.balance_lines[1]):
            self.node_plot.addItem(item)
        self.modes_note = note_label()

        self.ends_plot = plot("Body above the axles", "m", "Time", "s", height=170)
        self.pitch_plot = plot("Pitch", "°", "Time", "s", height=150)
        self.pitch_plot.setXLink(self.ends_plot)
        for p in (self.ends_plot, self.pitch_plot):
            p.setToolTip(BUMP_TIP)
        add_legend(self.ends_plot.getPlotItem(), offset=(-10, 10))
        self.end_curves = [pg.PlotDataItem(name=n) for n in ("front", "rear")]
        self.road_curve = pg.PlotDataItem(name="road, front")
        self.pitch_curve = pg.PlotDataItem()
        self.leave_lines = [pg.InfiniteLine(angle=90), pg.InfiniteLine(angle=90)]
        self.time_lines = [pg.InfiniteLine(angle=90), pg.InfiniteLine(angle=90)]
        for item in (self.road_curve, *self.end_curves, self.leave_lines[0], self.time_lines[0]):
            self.ends_plot.addItem(item)
        for item in (self.pitch_curve, self.leave_lines[1], self.time_lines[1]):
            self.pitch_plot.addItem(item)
        self.bump_note = note_label()

        # the pendulum's modes, swing and twins
        self.pfreq_plot = plot("Frequency", "Hz", "Lower bob / upper bob", None, log_x=True, height=170)
        self.shape_plot = plot("θ₂/θ₁ in the mode", None, "Lower bob / upper bob", None, log_x=True, height=170)
        self.shape_plot.setXLink(self.pfreq_plot)
        for p in (self.pfreq_plot, self.shape_plot):
            p.setToolTip(PMODES_TIP)
        add_legend(self.pfreq_plot.getPlotItem(), offset=(10, 10))
        self.pfreq_items = [pg.PlotDataItem(name=f"mode {i + 1}") for i in range(2)]
        self.rod_items = [pg.PlotDataItem(name=n) for n in ("upper rod alone", "lower rod alone")]
        self.shape_items = [pg.PlotDataItem() for _ in range(2)]
        self.pnow_lines = [pg.InfiniteLine(angle=90), pg.InfiniteLine(angle=90)]
        for item in (*self.rod_items, *self.pfreq_items, self.pnow_lines[0]):
            self.pfreq_plot.addItem(item)
        for item in (*self.shape_items, self.pnow_lines[1]):
            self.shape_plot.addItem(item)
        self.pmodes_note = note_label()

        self.th1_plot = plot("Upper rod θ₁", "°", "Time", "s", height=150)
        self.th2_plot = plot("Lower rod θ₂", "°", "Time", "s", height=150)
        self.th2_plot.setXLink(self.th1_plot)
        for p in (self.th1_plot, self.th2_plot):
            p.setToolTip(SWING_TIP)
        add_legend(self.th1_plot.getPlotItem(), offset=(-10, 10))
        self.whole_items = [pg.PlotDataItem(name="whole equations"), pg.PlotDataItem()]
        self.linear_items = [pg.PlotDataItem(name="linear"), pg.PlotDataItem()]
        self.swing_lines = [pg.InfiniteLine(angle=90), pg.InfiniteLine(angle=90)]
        for p, whole, linear, line in zip(
            (self.th1_plot, self.th2_plot), self.whole_items, self.linear_items, self.swing_lines
        ):
            for item in (linear, whole, line):
                p.addItem(item)
        self.swing_note = note_label()

        self.chaos_plot = plot("Twins apart, |Δθ₁| + |Δθ₂|", "rad", "Time", "s", log_y=True, height=200)
        self.chaos_plot.setToolTip(CHAOS_TIP)
        self.apart_item = pg.PlotDataItem()
        self.chaos_line = pg.InfiniteLine(angle=90)
        for item in (self.apart_item, self.chaos_line):
            self.chaos_plot.addItem(item)
        self.chaos_note = note_label()

        self.notes = theory(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.fbd, "Free body")
        self.tabs.addTab(self.lagrange, "Lagrange")
        self.tabs.addTab(column(self.freq_plot, self.node_plot, self.modes_note, stretch=False), "Modes")
        self.tabs.addTab(column(self.ends_plot, self.pitch_plot, self.bump_note, stretch=False), "Bump")
        self.tabs.addTab(column(self.pfreq_plot, self.shape_plot, self.pmodes_note, stretch=False), "Modes")
        self.tabs.addTab(column(self.th1_plot, self.th2_plot, self.swing_note, stretch=False), "Swing")
        self.tabs.addTab(column(self.chaos_plot, self.chaos_note, stretch=False), "Chaos")
        self.tabs.addTab(self.notes, "Theory")
        self.tabs.setCurrentIndex(TAB_LAGRANGE)
        self.tabs.currentChanged.connect(lambda _: self._draw_tab())

        three_panes(
            self,
            column(self.trials, chosen, *self.car_forms, *self.pendulum_forms, show),
            stack((self.views, 4), (self.matrix_label, 0), (self.readout, 0)),
            self.tabs,
        )

        self._phase = 0.0
        self._time = 0.0
        self._clock = QtCore.QElapsedTimer()
        self._last = 0.0
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._frame)
        self.ride = None
        self.swing = None
        self.twins = None
        self._apply_system()
        self.apply_theme()

    # ------------------------------------------------------------- inputs
    def load_trial(self, index: int) -> None:
        self.apply_trial(TRIALS[index])

    def load_setup(self, setup: dict) -> None:
        """Set the page up as a practice problem's `setup` says (a trial's fields, with no note)."""
        self.apply_trial(DerivingTrial("", "", **setup))

    def practice_state(self) -> DerivingTrial:
        """The page's settings, as a trial: what a practice spec checks."""
        return DerivingTrial(
            "",
            "",
            system=self.system.currentIndex(),
            mass=self.mass.value(),
            gyration=self.gyration.value(),
            front=self.front.value(),
            rear=self.rear.value(),
            k_front=self.k_front.value(),
            k_rear=self.k_rear.value(),
            c_front=self.c_front.value(),
            c_rear=self.c_rear.value(),
            coords=COORDS[self.coords.currentIndex()],
            offset=self.offset.value(),
            speed_kmh=self.speed.value(),
            height=self.height.value(),
            length=self.length.value(),
            upper=self.upper.value(),
            lower=self.lower.value(),
            upper_length=self.upper_length.value(),
            lower_length=self.lower_length.value(),
            angles=ANGLES[self.angles.currentIndex()],
            theta1=self.theta1.value(),
            theta2=self.theta2.value(),
            show=self.show_box.currentIndex(),
            tab=self.tabs.currentIndex(),
        )

    def apply_trial(self, t: DerivingTrial) -> None:
        for box, value in (
            (self.system, t.system),
            (self.mass, t.mass),
            (self.gyration, t.gyration),
            (self.front, t.front),
            (self.rear, t.rear),
            (self.k_front, t.k_front),
            (self.k_rear, t.k_rear),
            (self.c_front, t.c_front),
            (self.c_rear, t.c_rear),
            (self.coords, COORDS.index(t.coords)),
            (self.offset, t.offset),
            (self.speed, t.speed_kmh),
            (self.height, t.height),
            (self.length, t.length),
            (self.upper, t.upper),
            (self.lower, t.lower),
            (self.upper_length, t.upper_length),
            (self.lower_length, t.lower_length),
            (self.angles, ANGLES.index(t.angles)),
            (self.theta1, t.theta1),
            (self.theta2, t.theta2),
            (self.show_box, t.show),
        ):
            set_quietly(box, value)
        if t.note:
            self.trials.show_note(t.note)
        self.tabs.blockSignals(True)
        self._apply_system()
        self.tabs.setCurrentIndex(t.tab)
        self.tabs.blockSignals(False)
        self.redraw()

    def _on_system(self) -> None:
        self.tabs.blockSignals(True)
        self._apply_system()
        self.tabs.blockSignals(False)
        self.redraw()

    def _apply_system(self) -> None:
        """Show the system's controls, view, tabs and motion."""
        system = self.system.currentIndex()
        for form in self.car_forms:
            form.setVisible(system == CAR)
        for form in self.pendulum_forms:
            form.setVisible(system == PENDULUM)
        self.views.setCurrentIndex(system)
        self.show_box.setItemText(SHOW_MOTION, MOTIONS[system])
        shown = SYSTEM_TABS[system]
        for i in range(self.tabs.count()):
            self.tabs.setTabVisible(i, i in shown)
        if self.tabs.currentIndex() not in shown:
            self.tabs.setCurrentIndex(TAB_LAGRANGE)

    @property
    def is_car(self) -> bool:
        return self.system.currentIndex() == CAR

    @property
    def car(self) -> Car:
        return self.practice_state().car

    @property
    def pendulum(self) -> DoublePendulum:
        return self.practice_state().pendulum

    @property
    def coordinates(self) -> Coordinates:
        return COORDS[self.coords.currentIndex()]

    @property
    def angle_set(self) -> Angles:
        return ANGLES[self.angles.currentIndex()]

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        dash, dot = QtCore.Qt.PenStyle.DashLine, QtCore.Qt.PenStyle.DotLine
        self.view.apply_theme()
        self.pview.apply_theme()
        for items in (self.freq_items, self.node_items, self.pfreq_items, self.shape_items):
            for item, color in zip(items, (c.mode[0], c.mode[1])):
                item.setPen(pg.mkPen(color, width=2))
        for items in (self.end_items, self.rod_items):
            for item, style in zip(items, (dash, dot)):
                item.setPen(pg.mkPen(c.grey, width=1.5, style=style))
        for line in self.axle_lines:
            line.setPen(pg.mkPen(c.grey, width=1, style=dash))
        for line in (*self.now_lines, *self.pnow_lines):
            line.setPen(pg.mkPen(c.force, width=1.5, style=dot))
        for line in self.balance_lines:
            line.setPen(pg.mkPen(c.grey, width=1, style=dot))
        for item, color in zip(self.end_curves, (c.mass[0], c.mass[1])):
            item.setPen(pg.mkPen(color, width=2))
        self.road_curve.setPen(pg.mkPen(c.grey, width=1.5, style=dash))
        self.pitch_curve.setPen(pg.mkPen(c.mode[1], width=2))
        for line in self.leave_lines:
            line.setPen(pg.mkPen(c.grey, width=1, style=dot))
        for item, color in zip(self.whole_items, (c.mass[0], c.mass[1])):
            item.setPen(pg.mkPen(color, width=2))
        for item in self.linear_items:
            item.setPen(pg.mkPen(c.grey, width=1.5, style=dash))
        self.apart_item.setPen(pg.mkPen(c.force, width=2))
        for line in (*self.time_lines, *self.swing_lines, self.chaos_line):
            line.setPen(pg.mkPen(c.force, width=1))
        self.redraw()

    def redraw(self) -> None:
        self.ride = self.swing = self.twins = None
        if self.is_car:
            self._readout_car()
        else:
            self._readout_pendulum()
        self._draw_tab()
        self._on_show()

    def _readout_car(self) -> None:
        car, coords, e = self.car, self.coordinates, self.offset.value()
        self.coords_form.form.setRowVisible(self.offset, coords is Coordinates.POINT)
        self.modes = modes(car, coords, e)
        self.body_info.setText(
            f"J = {car.inertia:.4g} kg·m², L = {car.wheelbase:.3g} m, dynamic index r²/(ab) = {car.dynamic_index:.3g}."
        )
        q1, q2 = q_names(coords)
        self.matrix_label.setText(
            f"<b>M q̈ + C q̇ + K q = 0</b>, q = ({q1}, {q2}), from Lagrange's equations (bold: coupling):"
            + matrices_html(matrices(car, coords, e))
        )
        parts = []
        for i, m in enumerate(self.modes):
            node = m.node
            if not math.isfinite(node):
                where = "no node: pure bounce"
            else:
                side = "ahead of" if node >= 0 else "behind"
                kind = "between the axles: mostly pitch" if m.inside(car) else "outside the wheelbase: mostly bounce"
                where = f"node {abs(node):.3g} m {side} G, {kind}"
                if abs(node) < 1e-3:
                    where = "node at G: pure pitch"
            parts.append(f"Mode {i + 1}: <b>{m.frequency:.3g} Hz</b>, {where}")
        fb, fp = car.uncoupled_frequencies()
        ff, fr = car.end_frequencies()
        e_star = car.elastic_centre()
        self.readout.setText(
            ". ".join(parts)
            + f". Uncoupled: bounce {fb:.3g} Hz, pitch {fp:.3g} Hz; each end alone: front {ff:.3g}, rear {fr:.3g} Hz. "
            f"Elastic centre {abs(e_star):.3g} m {'ahead of' if e_star >= 0 else 'behind'} G."
        )

    def _readout_pendulum(self) -> None:
        p, angles = self.pendulum, self.angle_set
        self.pmodes = pendulum_modes(p)
        q1, q2 = dp.q_names(angles)
        self.matrix_label.setText(
            f"<b>M q̈ + K q = 0</b> for small swings, q = ({q1}, {q2}), from Lagrange's equations (bold: coupling):"
            + matrices_html(pendulum_matrices(p, angles))
        )
        parts = [
            f"Mode {i + 1}: <b>{m.frequency:.3g} Hz</b>, θ₂ = {m.ratio:.3g} θ₁ ".replace("-", "−")
            + ("(the rods together)" if m.in_phase else "(the rods against each other)")
            for i, m in enumerate(self.pmodes)
        ]
        fu, fl = p.separate_frequencies()
        self.readout.setText(". ".join(parts) + f". Each rod alone: upper {fu:.3g} Hz, lower {fl:.3g} Hz.")

    def _draw_tab(self) -> None:
        i = self.tabs.currentIndex()
        if self.is_car:
            car, coords, e = self.car, self.coordinates, self.offset.value()
            if i == TAB_FBD:
                self._set_html(self.fbd, fbd_html(car, coords, e))
            elif i == TAB_LAGRANGE:
                self._set_html(self.lagrange, lagrange_html(car, coords, e))
            elif i == TAB_MODES:
                self._draw_modes()
            elif i == TAB_BUMP:
                self._draw_bump()
        else:
            p, angles = self.pendulum, self.angle_set
            if i == TAB_FBD:
                self._set_html(self.fbd, dp.fbd_html(p, angles))
            elif i == TAB_LAGRANGE:
                self._set_html(self.lagrange, dp.lagrange_html(p, angles))
            elif i == TAB_PMODES:
                self._draw_pendulum_modes()
            elif i == TAB_SWING:
                self._draw_swing()
            elif i == TAB_CHAOS:
                self._draw_chaos()
        settle_plots(self)

    @staticmethod
    def _set_html(browser: QtWidgets.QTextBrowser, html: str) -> None:
        bar = browser.verticalScrollBar()
        at = bar.value()
        browser.setHtml(html)
        bar.setValue(at)

    def _draw_modes(self) -> None:
        car = self.car
        f, nodes = sweep(car, RATIOS)
        mf, mr = car.end_masses
        for item, values in zip(self.freq_items, f.T):
            item.setData(RATIOS, values)
        self.end_items[0].setData(RATIOS, np.full(RATIOS.size, math.sqrt(car.k_front / mf) / TWO_PI))
        self.end_items[1].setData(RATIOS, np.sqrt(RATIOS * car.k_front / mr) / TWO_PI)
        reach = 3.0 * car.wheelbase
        for item, values in zip(self.node_items, nodes.T):
            shown = np.where(np.abs(values) < reach, values, np.nan)
            item.setData(RATIOS, shown)
        self.axle_lines[0].setPos(car.front)
        self.axle_lines[1].setPos(-car.rear)
        ratio = car.k_rear / car.k_front
        balance = car.front / car.rear  # k_f a = k_r b
        for line in self.now_lines:
            line.setPos(math.log10(ratio))
        for line in self.balance_lines:
            line.setPos(math.log10(balance))
        self.freq_plot.setXRange(math.log10(RATIOS[0]), math.log10(RATIOS[-1]), padding=0)
        self.freq_plot.setYRange(0.0, 1.15 * float(np.nanmax(f)), padding=0)
        self.node_plot.setYRange(-reach, reach, padding=0)
        self.modes_note.setText(
            f"At k<sub>r</sub>/k<sub>f</sub> = {balance:.3g} (k<sub>f</sub>a = k<sub>r</sub>b) the modes are pure "
            "bounce and pitch: one node runs off to infinity and the other sits on G. Elsewhere one node lies "
            "outside the wheelbase (a mostly bouncing mode) and one inside (mostly pitch). Dashed lines: the "
            "axles. With a dynamic index of 1 the modes are the ends' own, and their nodes sit on the axles."
        )

    def _ensure_ride(self) -> None:
        if self.ride is None:
            t = self.practice_state()
            self.ride = ride(t.car, t.bump)

    def _draw_bump(self) -> None:
        self._ensure_ride()
        r = self.ride
        self.road_curve.setData(r.t, r.road[:, 0])
        for item, values in zip(self.end_curves, r.ends.T):
            item.setData(r.t, values)
        self.pitch_curve.setData(r.t, np.degrees(r.pitch))
        for line in self.leave_lines:
            line.setPos(r.leaves)
        self.ends_plot.setXRange(0.0, float(r.t[-1]), padding=0)
        after = r.after()
        peak, settled = np.degrees(np.abs(r.pitch).max()), np.degrees(np.abs(r.pitch[after]).max())
        lag = self.car.wheelbase / (self.speed.value() / 3.6)
        self.bump_note.setText(
            f"The rear wheel meets the bump {lag:.2g} s after the front. The body pitches up to {peak:.3g}° in all, "
            f"and <b>{settled:.3g}°</b> once both wheels are past it (dotted), where the car's own modes are left: "
            f"the ends rise {1e3 * np.abs(r.ends).max(axis=0)[0]:.3g} mm at the front and "
            f"{1e3 * np.abs(r.ends).max(axis=0)[1]:.3g} mm at the rear."
        )

    def _draw_pendulum_modes(self) -> None:
        p = self.pendulum
        f, shape = pendulum_sweep(p, MASS_RATIOS)
        for item, values in zip(self.pfreq_items, f.T):
            item.setData(MASS_RATIOS, values)
        for item, rod in zip(self.rod_items, p.separate_frequencies()):
            item.setData(MASS_RATIOS, np.full(MASS_RATIOS.size, rod))
        for item, values in zip(self.shape_items, shape.T):
            item.setData(MASS_RATIOS, np.clip(values, -12.0, 12.0))
        ratio = p.lower / p.upper
        for line in self.pnow_lines:
            line.setPos(math.log10(ratio))
        self.pfreq_plot.setXRange(math.log10(MASS_RATIOS[0]), math.log10(MASS_RATIOS[-1]), padding=0)
        self.pfreq_plot.setYRange(0.0, 1.1 * float(f.max()), padding=0)
        self.shape_plot.setYRange(-12.0, 12.0, padding=0)
        slow, fast = self.pmodes
        self.pmodes_note.setText(
            f"This pendulum (red): {slow.frequency:.3g} and {fast.frequency:.3g} Hz, {fast.frequency / slow.frequency:.3g} "
            "apart. A light lower bob barely loads the upper rod: the modes close in on the rods' own frequencies, "
            "one swinging the lower rod far more than the upper. A heavy one makes the upper rod's mode stiff and "
            "fast, and the slower mode a single pendulum of length L₁ + L₂ (θ₂ → θ₁)."
        )

    def _ensure_swing(self) -> None:
        if self.swing is None:
            t = self.practice_state()
            self.swing = swing(t.pendulum, t.release, SWING_S)

    def _draw_swing(self) -> None:
        self._ensure_swing()
        s = self.swing
        for i in range(2):
            self.whole_items[i].setData(s.t, np.degrees(s.whole[:, i]))
            self.linear_items[i].setData(s.t, np.degrees(s.linear[:, i]))
        self.th1_plot.setXRange(0.0, SWING_S, padding=0)
        whole, linear = measured_period(s.t, s.whole[:, 0]), measured_period(s.t, s.linear[:, 0])
        apart = np.degrees(np.abs(s.whole[:, 0] - s.linear[:, 0]).max())
        period = (
            f"The upper rod's period is {whole:.4g} s, against the linear {linear:.4g} s "
            f"({100 * (whole / linear - 1):+.2g} %). "
            if math.isfinite(whole) and math.isfinite(linear)
            else ""
        )
        self.swing_note.setText(
            period + f"Over {SWING_S:.0f} s the linear model strays up to {apart:.3g}° from the whole motion. The "
            f"whole equations' energy holds to {s.drift:.1g} of the swing's."
        )

    def _draw_chaos(self) -> None:
        if self.twins is None:
            t = self.practice_state()
            self.twins = divergence(t.pendulum, t.release, CHAOS_S)
        d = self.twins
        self.apart_item.setData(d.t, np.maximum(d.apart, 1e-12))
        self.chaos_plot.setXRange(0.0, CHAOS_S, padding=0)
        self.chaos_plot.setYRange(-8.0, 1.5, padding=0)
        doubling = d.doubling_time()
        self.chaos_note.setText(
            f"A twin released 10⁻⁶ rad further over (the lower rod). After {CHAOS_S:.0f} s they're "
            f"{d.apart[-1]:.2g} rad apart"
            + (
                f": the difference doubles every <b>{doubling:.2g} s</b>, exponentially. That's chaos: the motion is "
                "exact, but no measurement could be fine enough to predict it for long."
                if math.isfinite(doubling)
                else ": the difference grows, but slowly, not exponentially. The motion may look irregular, but "
                "it's predictable."
            )
        )

    # ------------------------------------------------------------- the animation
    def _on_show(self) -> None:
        self._time = 0.0
        self._draw_system()
        self._start_timer()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._start_timer()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._timer.stop()

    @property
    def running(self) -> bool:
        return self.animate.isChecked()

    def _start_timer(self) -> None:
        if self.isVisible() and self.running and not self._timer.isActive():
            self._clock.restart()
            self._last = 0.0
            self._timer.start(FRAME_MS)

    def _frame(self) -> None:
        began = self._clock.elapsed()
        now = began / 1000.0
        dt = min(now - self._last, 0.1)
        self._last = now
        with frame_stats.frame("Deriving the equations", lambda: self._time, FRAME_MS):
            self.step(dt)
            settle_plots(self)
        if self.isVisible() and self.running:  # the next frame FRAME_MS after this one began, not after it ended
            self._timer.start(max(1, FRAME_MS - (self._clock.elapsed() - began)))

    def frame_context(self) -> str:
        return f"{SYSTEMS[self.system.currentIndex()]}: {self.show_box.currentText()}"

    def step(self, dt: float) -> None:
        """Move the animation on by dt seconds."""
        self._phase = (self._phase + TWO_PI * SHOW_HZ * dt) % TWO_PI
        self._time += dt
        self._draw_system()

    def _draw_system(self) -> None:
        if self.is_car:
            self._draw_car()
        else:
            self._draw_pendulum()

    def _coordinate_points(self) -> list[float]:
        car, coords = self.car, self.coordinates
        if coords is Coordinates.CENTRE:
            return [0.0]
        if coords is Coordinates.POINT:
            return [self.offset.value()]
        return [car.front, -car.rear]

    def _draw_car(self) -> None:
        car, points = self.car, self._coordinate_points()
        show = self.show_box.currentIndex()
        if show == SHOW_MOTION:
            self._ensure_ride()
            r = self.ride
            k = min(int(np.searchsorted(r.t, self._time % r.t[-1])) if self.running else 0, r.t.size - 1)
            xs = np.linspace(-VIEW_X, VIEW_X, 241)
            bump = self.practice_state().bump
            road = RIDE_SCALE * bump.profile(bump.speed * (r.t[k] - r.lead) - (car.front - xs))
            xg, th = r.centre[k]
            self.view.draw(
                car, RIDE_SCALE * xg, RIDE_SCALE * th, (xs, road), tuple(RIDE_SCALE * r.road[k]), points, None
            )
            self.view.caption.setText(f"t = {r.t[k]:.2f} s, heights × {RIDE_SCALE:g}")
            for line in self.time_lines:
                line.setPos(r.t[k])
            return
        mode = self.modes[show]
        xg, th = mode.centre * SWING
        swing_now = math.sin(self._phase) if self.running else 1.0
        node = mode.node if math.isfinite(mode.node) else None
        self.view.draw(car, swing_now * xg, swing_now * th, None, (0.0, 0.0), points, node)
        self.view.caption.setText(f"Mode {show + 1}, {mode.frequency:.3g} Hz (swung slowly)")

    def _draw_pendulum(self) -> None:
        p = self.pendulum
        show = self.show_box.currentIndex()
        if show == SHOW_MOTION:
            self._ensure_swing()
            s = self.swing
            k = min(int(np.searchsorted(s.t, self._time % s.t[-1])) if self.running else 0, s.t.size - 1)
            th1, th2 = s.whole[k, 0], s.whole[k, 1]
            back = max(0, k - int(TRAIL_S / (s.t[1] - s.t[0])))
            _, _, x2, y2 = p.positions(s.whole[back : k + 1, 0], s.whole[back : k + 1, 1])
            self.pview.draw(p, th1, th2, (x2, y2))
            self.pview.caption.setText(f"t = {s.t[k]:.2f} s")
            for line in (*self.swing_lines, self.chaos_line):
                line.setPos(s.t[k])
            return
        mode = self.pmodes[show]
        scale = math.radians(dp.SHOWN) / max(1.0, abs(mode.ratio))
        swing_now = math.sin(self._phase) if self.running else 1.0
        self.pview.draw(p, scale * swing_now, scale * mode.ratio * swing_now, None)
        self.pview.caption.setText(f"Mode {show + 1}, {mode.frequency:.3g} Hz (swung slowly)")
