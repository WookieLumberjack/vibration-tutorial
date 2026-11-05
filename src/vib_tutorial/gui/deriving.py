"""Deriving the equations page: a car's body in bounce and pitch, by free-body diagrams and by Lagrange.

Left: the body, its suspension, the coordinates and a bump. Centre: the car, swinging in a mode or riding the
bump, and its M, C and K in the coordinates chosen. Right: the two derivations step by step with the page's
numbers in them, the modes against the rear spring, the ride, and the Theory. The page has its own model
(core/bounce_pitch.py).
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
    Matrices,
    matrices,
    modes,
    newton,
    ride,
    slipped,
    sweep,
)
from ..core.practice import Problem, Question, Requirement, at_most, kept, number, pick
from . import frame_stats, settle_plots
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
from .links import before, page_link, takeaways
from .style import colors
from .theming import add_legend

TAB_FBD, TAB_LAGRANGE, TAB_MODES, TAB_BUMP = range(4)
SHOW_MODE1, SHOW_MODE2, SHOW_RIDE = range(3)
SHOWS = ("Mode 1", "Mode 2", "Ride over the bump")
COORDS = tuple(Coordinates)
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
class PitchTrial:
    name: str
    note: str
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


TRIALS = (
    PitchTrial(
        "A car in bounce and pitch",
        "A 1200 kg body (r = 1.3 m) on 40 kN/m at each axle, 1.2 m ahead of G and 1.5 m behind it. Its slower "
        "mode (1.25 Hz) turns about a node 1.92 m behind G, outside the wheelbase: mostly bounce. The faster "
        "(1.41 Hz) turns about a node 0.88 m ahead of G, between the axles: mostly pitch. Neither is pure, "
        "because k<sub>f</sub>a ≠ k<sub>r</sub>b: the springs couple the two (static coupling).",
    ),
    PitchTrial(
        "No static coupling",
        "The rear spring at 32 kN/m, so k<sub>f</sub>a = k<sub>r</sub>b = 48 kN: K is diagonal in (x<sub>G</sub>, "
        "θ), and the modes are pure bounce (1.23 Hz, no node) and pure pitch (1.27 Hz, about G). The Modes "
        "tab's nodes run off to infinity there.",
        k_rear=32.0,
        tab=TAB_MODES,
    ),
    PitchTrial(
        "Coordinates at the elastic centre",
        "x<sub>P</sub> and θ at the elastic centre, e* = (k<sub>f</sub>a − k<sub>r</sub>b)/(k<sub>f</sub> + "
        "k<sub>r</sub>) = −0.15 m (behind G): a force there pushes the body straight, so K is diagonal. The "
        "coupling has moved into M (−m e = 180 kg·m): dynamic coupling. The frequencies are the same 1.25 and "
        "1.41 Hz: coupling belongs to the coordinates, not the car.",
        coords=Coordinates.POINT,
        offset=-0.15,
    ),
    PitchTrial(
        "Moments about the front axle",
        "x<sub>P</sub> and θ with P above the front axle (e = 1.2 m), by the free-body diagram. Taking moments "
        "about P keeps the moment of G's inertia force, −m e ẍ<sub>G</sub>: the moment balance is (J + m e²)θ̈ "
        "− m e ẍ<sub>P</sub> = ΣM<sub>P</sub>. Leave that term out, as J<sub>P</sub>θ̈ = ΣM<sub>P</sub>, and "
        "the frequencies come out 0.93 and 1.38 Hz instead of 1.25 and 1.41.",
        coords=Coordinates.POINT,
        offset=1.2,
        tab=TAB_FBD,
    ),
    PitchTrial(
        "The two ends as coordinates",
        "x<sub>f</sub> and x<sub>r</sub>, the body above each axle. Each spring now sees one coordinate, so K "
        "is diagonal, and M couples by (m a b − J)/L² = 18.1 kg. The free-body diagram's force balance gives "
        "M the row (667, 533) kg and its moment balance (751, −751) kg·m: right, but not symmetric. Moments "
        "about each axle, which are Lagrange's rows, make them so.",
        coords=Coordinates.ENDS,
        tab=TAB_FBD,
    ),
    PitchTrial(
        "Dynamic index 1: the ends uncouple",
        "The radius of gyration at √(ab) = 1.34 m, so J = m a b and the dynamic index r²/(ab) is 1. In the "
        "ends' coordinates M is diagonal too, and each end bounces on its own spring: 1.23 Hz at the front "
        "(667 kg on 40 kN/m), 1.38 Hz at the rear (533 kg), the nodes on the axles. A bump at the front then "
        "barely moves the rear: most cars are near 1.",
        gyration=1.342,
        coords=Coordinates.ENDS,
        tab=TAB_MODES,
    ),
    PitchTrial(
        "Over a hump, front stiffer than rear",
        "The front on 48 kN/m and the rear on 30: the front bounces at 1.35 Hz and the rear at 1.19. Over a "
        "75 mm hump at 30 km/h the rear wheel arrives 0.32 s after the front and the slower rear falls further "
        "behind: once both wheels are past, the body pitches up to 1.95°.",
        k_front=48.0,
        k_rear=30.0,
        show=SHOW_RIDE,
        tab=TAB_BUMP,
    ),
    PitchTrial(
        "Olley's flat ride",
        "The springs swapped: 30 kN/m at the front (1.07 Hz) and 48 at the rear (1.51 Hz). Now the rear, hit "
        "later, catches the front up, and once past the hump the body pitches only 1.52°. Olley's rule: the "
        "front's ride frequency below the rear's (here by 29 %).",
        k_front=30.0,
        k_rear=48.0,
        show=SHOW_RIDE,
        tab=TAB_BUMP,
    ),
)


# ------------------------------------------------------------------ the derivations, written out
def num(value: float) -> str:
    """A matrix entry: whole numbers past 100, four figures below."""
    value = float(value)
    if abs(value) < 1e-9:
        return "0"
    return f"{value:.0f}" if abs(value) >= 100 else f"{value:.4g}"


def matrix_html(name: str, A: np.ndarray) -> str:
    """A 2 × 2 matrix as a bracketed table, its coupling (off-diagonal) entries in bold."""

    def cell(i: int, j: int) -> str:
        text = num(A[i, j]).replace("-", "−")
        return f"<b>{text}</b>" if i != j and text != "0" else text

    bracket = "<td rowspan='2' valign='middle'><span style='font-size:22pt; font-weight:200'>{}</span></td>"
    rows = [
        f"<td align='right'>{cell(i, 0)}</td><td>&nbsp;&nbsp;</td><td align='right'>{cell(i, 1)}</td>" for i in range(2)
    ]
    return (
        f"<table cellspacing='0' cellpadding='1'><tr><td rowspan='2' valign='middle'>{name} =&nbsp;</td>"
        f"{bracket.format('[')}{rows[0]}{bracket.format(']')}</tr><tr>{rows[1]}</tr></table>"
    )


def matrices_html(m: Matrices) -> str:
    return (
        "<table><tr>"
        + "".join(
            f"<td valign='middle'>{matrix_html(n, A)}</td><td>&nbsp;&nbsp;</td>" for n, A in zip("MCK", (m.M, m.C, m.K))
        )
        + "</tr></table>"
    )


def q_names(coords: Coordinates) -> tuple[str, str]:
    return {
        Coordinates.CENTRE: ("x<sub>G</sub>", "θ"),
        Coordinates.POINT: ("x<sub>P</sub>", "θ"),
        Coordinates.ENDS: ("x<sub>f</sub>", "x<sub>r</sub>"),
    }[coords]


def dotted(name: str, n: int) -> str:
    """ẋ or ẍ for an x, θ̇ or θ̈ for a θ."""
    if name.startswith("x"):
        return "ẋẍ"[n - 1] + name[1:]
    return name + "̇̈"[n - 1]


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


def _entries(letter: str, symbols: tuple[str, str, str], A: np.ndarray) -> str:
    """M₁₁ = m = 1200, M₁₂ = …: each entry in symbols and in numbers."""
    return ", ".join(
        f"{letter}{ij} = {s} = {num(v)}".replace("-", "−")
        for ij, s, v in zip(("₁₁", "₁₂", "₂₂"), symbols, (A[0, 0], A[0, 1], A[1, 1]))
    )


def frequencies_html(m: Matrices) -> str:
    """det(K − ω²M) = 0 as a quadratic in ω², with the page's numbers, and its roots."""
    M, K = m.M, m.K
    a = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    b = M[0, 0] * K[1, 1] + M[1, 1] * K[0, 0] - M[0, 1] * K[1, 0] - M[1, 0] * K[0, 1]
    c = K[0, 0] * K[1, 1] - K[0, 1] * K[1, 0]
    disc = b * b - 4 * a * c
    roots = sorted(((b - math.sqrt(max(disc, 0.0))) / (2 * a), (b + math.sqrt(max(disc, 0.0))) / (2 * a)))
    f = [math.sqrt(max(w2, 0.0)) / TWO_PI for w2 in roots]
    return (
        "<p>det(K − ω²M) = 0 is a quadratic in ω²: "
        f"<b>{number(a, 4)} ω⁴ − {number(b, 4)} ω² + {number(c, 4)} = 0</b>, so ω² = {number(roots[0], 4)} and "
        f"{number(roots[1], 4)} (rad/s)², "
        f"<b>f = {f[0]:.3g} and {f[1]:.3g} Hz</b>.</p>"
    )


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
        f"<p><b>T = ½(M₁₁{v1}² + 2M₁₂{v1}{v2} + M₂₂{v2}²)</b>, {_entries('M', mass_symbols(coords), m.M)}.</p>",
        "<h3>3. The potential energy and the dissipation</h3>",
        f"<p><b>V = ½ k<sub>f</sub>δ<sub>f</sub>² + ½ k<sub>r</sub>δ<sub>r</sub>² = ½(K₁₁{q1}² + 2K₁₂{q1}{q2} + "
        f"K₂₂{q2}²)</b>, {_entries('K', spring_symbols(coords, 'k'), m.K)}.</p>"
        f"<p>Rayleigh's dissipation function <b>D = ½ c<sub>f</sub>δ̇<sub>f</sub>² + ½ c<sub>r</sub>δ̇<sub>r</sub>²</b> "
        f"in the same way: {_entries('C', spring_symbols(coords, 'c'), m.C)}.</p>",
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
    base = PitchTrial("", "")

    def goal(t: PitchTrial):
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


PROBLEMS = tuple(
    Problem(key, title, "deriving_page", f"See {page_link('deriving_page', anchor, section)}.", make)
    for key, title, anchor, section, make in (
        ("bounce", "Pure bounce", "bounce-pitch", "A car in bounce and pitch", _bounce),
        ("elastic-centre", "The elastic centre", "coupling", "Coordinates and coupling", _elastic_centre),
        ("end-frequency", "A ride frequency", "coupling", "Coordinates and coupling", _end_frequency),
        ("index-one", "Dynamic index 1", "coupling", "Coordinates and coupling", _index_one),
        ("point-mass", "M about a point", "lagrange", "Lagrange's equations", _point_mass),
        ("ends-mass", "M in the ends' coordinates", "lagrange", "Lagrange's equations", _ends_mass),
        ("same-frequencies", "Predict: other coordinates", "coupling", "Coordinates and coupling", _same_frequencies),
        ("pure-modes", "Predict: k_f a = k_r b", "coupling", "Coordinates and coupling", _pure_modes),
        ("uncouple-spec", "Spec: uncouple bounce and pitch", "coupling", "Coordinates and coupling", _uncouple_spec),
    )
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
    "show": "<p>What the car shows: a mode swinging (at a slow pace of its own, not its frequency), or the "
    "ride over the bump in real time, its heights drawn three times their size.</p>",
}
MODES_TIP = (
    "<p>The two modes as the rear spring changes (the front's held): their frequencies (top) and where each "
    "turns the body (bottom), from G, forward. Grey: each end on its own spring. Red: this car's rear spring; grey "
    "dotted: where k<sub>f</sub>a = k<sub>r</sub>b.</p>"
)
BUMP_TIP = "<p>The body above each axle, and its pitch, as the car rides over the bump.</p>"

NOTES_HTML = """
<!--BEFORE-->
<h3><a name="two-ways"></a>Two ways to the equations</h3>
<p>Every page from here on starts from <b>M q̈ + C q̇ + K q = f</b>. For one mass on a spring the equation can
be written down by eye; for anything bigger it has to be derived, and there are two ways.</p>
<ul>
<li><b>Free-body diagrams</b> (Newton and Euler): cut each body free, draw every force on it, and write
m a<sub>G</sub> = ΣF and, about G, J θ̈ = ΣM<sub>G</sub>. Direct, but it takes vectors, forces of constraint and
care over which point the moments are taken about.</li>
<li><b>Lagrange's equations</b>: write the kinetic energy T, the potential energy V and the dissipation D in any
coordinates q that describe the motion, and differentiate. Scalars only, no forces of constraint, and the
matrices come out symmetric whatever the coordinates.</li>
</ul>
<p>Both give the same motion. The page derives a car's body in bounce and pitch both ways, step by step, with its
own numbers (the Free-body diagram and Lagrange tabs follow the controls).</p>

<h3><a name="free-body"></a>Free-body diagrams</h3>
<p>For a rigid body in a plane: <b>m ẍ<sub>G</sub> = ΣF</b> and <b>J<sub>G</sub> θ̈ = ΣM<sub>G</sub></b>, the
moments about the centre of mass G. Moments may be taken about another point P instead, often to drop a force
that passes through it, but then the inertia force m a<sub>G</sub>, acting at G, has a moment about P too:
ΣM<sub>P</sub> = J<sub>G</sub> θ̈ + (r<sub>G</sub> − r<sub>P</sub>) × m a<sub>G</sub>. Writing ΣM<sub>P</sub> =
J<sub>P</sub> θ̈ alone (true only when P is fixed, or is G) is the classic slip, and it gives wrong frequencies
(the trial <i>Moments about the front axle</i>). The rows that come out are a force and a moment, so in other
coordinates M, C and K need not be symmetric, though they're right.</p>

<h3><a name="lagrange"></a>Lagrange's equations</h3>
<p>With T(q, q̇), V(q) and Rayleigh's dissipation function D(q̇) = ½ Σ c<sub>i</sub>(stretch rate)², each
coordinate q<sub>i</sub> has</p>
<p align="center"><b>d/dt(∂T/∂q̇<sub>i</sub>) − ∂T/∂q<sub>i</sub> + ∂D/∂q̇<sub>i</sub> + ∂V/∂q<sub>i</sub> =
Q<sub>i</sub></b>,</p>
<p>where Q<sub>i</sub> is the generalized force: the work the applied forces do in a small δq<sub>i</sub>, over
δq<sub>i</sub>. For small motion about equilibrium T = ½ q̇ᵀMq̇, V = ½ qᵀKq and D = ½ q̇ᵀCq̇, and the equations
are M q̈ + C q̇ + K q = Q: <b>M, K and C are the Hessians of T, V and D</b>, so they're symmetric. Changing to new
coordinates q = T q′ makes them TᵀMT, TᵀCT and TᵀKT. The free-body diagram's rows, mixed by Tᵀ, are the same
equations. Measuring each spring from where it sits at rest drops gravity: its work cancels the springs'
preload. (The Pendulums page's gravity stiffness is where it doesn't: there the weight's own potential
curves.)</p>

<h3><a name="bounce-pitch"></a>A car in bounce and pitch</h3>
<p>A car's body on its springs, the wheels and tyres left out: a rigid body of mass m and pitch inertia J =
m r² about G, on k<sub>f</sub> and c<sub>f</sub> at a ahead of G and k<sub>r</sub> and c<sub>r</sub> at b behind
(L = a + b). Two coordinates describe its motion, x<sub>G</sub> and θ for instance, and</p>
<p align="center">T = ½ m ẋ<sub>G</sub>² + ½ J θ̇², &nbsp; V = ½ k<sub>f</sub>(x<sub>G</sub> + aθ)² + ½
k<sub>r</sub>(x<sub>G</sub> − bθ)²,</p>
<p>so M = diag(m, J) and K = [[k<sub>f</sub> + k<sub>r</sub>, k<sub>f</sub>a − k<sub>r</sub>b], [k<sub>f</sub>a −
k<sub>r</sub>b, k<sub>f</sub>a² + k<sub>r</sub>b²]]. Pure bounce would be at √((k<sub>f</sub> + k<sub>r</sub>)/m)
and pure pitch at √((k<sub>f</sub>a² + k<sub>r</sub>b²)/J), but unless k<sub>f</sub>a = k<sub>r</sub>b the two
couple and each mode is a turn about a <b>node</b>: outside the wheelbase for the mostly bouncing one, inside it
for the mostly pitching one.</p>

<h3><a name="coupling"></a>Coordinates and coupling</h3>
<p>Coupling belongs to the coordinates, not to the car. In (x<sub>G</sub>, θ) only K couples (<b>static
coupling</b>). At the <b>elastic centre</b> e* = (k<sub>f</sub>a − k<sub>r</sub>b)/(k<sub>f</sub> +
k<sub>r</sub>), where a force pushes the body straight, K is diagonal, and M couples through −m e (<b>dynamic
coupling</b>). In the ends' coordinates (x<sub>f</sub>, x<sub>r</sub>), each spring sees one coordinate and K is
diagonal again, while M couples by (m a b − J)/L². When J = m a b, a <b>dynamic index</b> r²/(ab) of 1, that
vanishes too, and each end bounces on its own spring at √(k/m<sub>end</sub>), m<sub>f</sub> = m b/L and
m<sub>r</sub> = m a/L: a bump under the front wheel leaves the rear alone. Most cars sit near 1. In every set of
coordinates the frequencies are the same.</p>

<h3><a name="olley"></a>Flat ride</h3>
<p>Over a bump the rear wheel arrives L/v after the front. Maurice Olley's rule from the 1930s: make the front's
ride frequency lower than the rear's (about 20 % lower), so that the rear, hit later but quicker, catches the
front up and the body bounces rather than pitches. The Bump tab rides a speed hump either way round.</p>

<h3><a name="try-it"></a>Things to try</h3>
<p>Each of these sets up the car (Things to try, top left).</p>
<!--TRIALS-->
<!--TAKEAWAYS-->
"""
NOTES_HTML = (
    NOTES_HTML.replace(
        "<!--BEFORE-->",
        before(
            ("sdof_page", "equation", "The equation of motion"),
            ("pendulum_page", "rotation", "The rotational form"),
        ),
    )
    .replace("<!--TRIALS-->", trials_html(TRIALS))
    .replace(
        "<!--TAKEAWAYS-->",
        takeaways(
            "Free-body diagrams: m a<sub>G</sub> = ΣF and J θ̈ = ΣM<sub>G</sub>; moments about another point need G's "
            "inertia force too.",
            "Lagrange: write T, V and D in any coordinates; M, K and C are their Hessians, so always symmetric.",
            "Coupling depends on the coordinates (static in K, dynamic in M); the frequencies don't.",
            "A car's bounce and pitch couple unless k<sub>f</sub>a = k<sub>r</sub>b; at a dynamic index of 1 its ends "
            "bounce apart.",
        ),
    )
)


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
    """Controls on the left, the car and its matrices in the centre, the derivations, modes and ride on the
    right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.trials = TrialBox(
            [t.name for t in TRIALS], "<p>Each sets up the car to show one idea from the Theory tab.</p>"
        )
        self.trials.picked.connect(self.load_trial)

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

        show = Form("Show")
        self.show = combo(SHOWS, TIPS["show"])
        show.add("The car:", self.show, TIPS["show"])
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
            ),
            self.redraw,
        )
        self.show.currentIndexChanged.connect(lambda _: self._on_show())
        self.animate.toggled.connect(lambda _: self._on_show())

        # --- centre
        self.view = CarView()
        self.matrix_label = readout_label()
        self.readout = readout_label()

        # --- right
        self.fbd = theory("")
        self.lagrange = theory("")
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

        self.notes = theory(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.fbd, "Free-body diagram")
        self.tabs.addTab(self.lagrange, "Lagrange")
        self.tabs.addTab(column(self.freq_plot, self.node_plot, self.modes_note, stretch=False), "Modes")
        self.tabs.addTab(column(self.ends_plot, self.pitch_plot, self.bump_note, stretch=False), "Bump")
        self.tabs.addTab(self.notes, "Theory")
        self.tabs.setCurrentIndex(TAB_LAGRANGE)
        self.tabs.currentChanged.connect(lambda _: self._draw_tab())

        three_panes(
            self,
            column(self.trials, body, suspension, coordinates, bump, show),
            stack((self.view, 4), (self.matrix_label, 0), (self.readout, 0)),
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
        self.apply_theme()

    # ------------------------------------------------------------- inputs
    def load_trial(self, index: int) -> None:
        self.apply_trial(TRIALS[index])

    def load_setup(self, setup: dict) -> None:
        """Set the page up as a practice problem's `setup` says (a trial's fields, with no note)."""
        self.apply_trial(PitchTrial("", "", **setup))

    def practice_state(self) -> PitchTrial:
        """The page's settings, as a trial: what a practice spec checks."""
        return PitchTrial(
            "",
            "",
            self.mass.value(),
            self.gyration.value(),
            self.front.value(),
            self.rear.value(),
            self.k_front.value(),
            self.k_rear.value(),
            self.c_front.value(),
            self.c_rear.value(),
            COORDS[self.coords.currentIndex()],
            self.offset.value(),
            self.speed.value(),
            self.height.value(),
            self.length.value(),
            self.show.currentIndex(),
            self.tabs.currentIndex(),
        )

    def apply_trial(self, t: PitchTrial) -> None:
        for box, value in (
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
            (self.show, t.show),
        ):
            set_quietly(box, value)
        if t.note:
            self.trials.show_note(t.note)
        self.tabs.blockSignals(True)
        self.tabs.setCurrentIndex(t.tab)
        self.tabs.blockSignals(False)
        self.redraw()

    @property
    def car(self) -> Car:
        return self.practice_state().car

    @property
    def coordinates(self) -> Coordinates:
        return COORDS[self.coords.currentIndex()]

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        self.view.apply_theme()
        for item, color in zip(self.freq_items, (c.mode[0], c.mode[1])):
            item.setPen(pg.mkPen(color, width=2))
        for item, color in zip(self.node_items, (c.mode[0], c.mode[1])):
            item.setPen(pg.mkPen(color, width=2))
        for item, style in zip(self.end_items, (QtCore.Qt.PenStyle.DashLine, QtCore.Qt.PenStyle.DotLine)):
            item.setPen(pg.mkPen(c.grey, width=1.5, style=style))
        for line in self.axle_lines:
            line.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DashLine))
        for line in self.now_lines:
            line.setPen(pg.mkPen(c.force, width=1.5, style=QtCore.Qt.PenStyle.DotLine))
        for line in self.balance_lines:
            line.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DotLine))
        for item, color in zip(self.end_curves, (c.mass[0], c.mass[1])):
            item.setPen(pg.mkPen(color, width=2))
        self.road_curve.setPen(pg.mkPen(c.grey, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        self.pitch_curve.setPen(pg.mkPen(c.mode[1], width=2))
        for line in self.leave_lines:
            line.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DotLine))
        for line in self.time_lines:
            line.setPen(pg.mkPen(c.force, width=1))
        self.redraw()

    def redraw(self) -> None:
        car, coords, e = self.car, self.coordinates, self.offset.value()
        self.coords_form.form.setRowVisible(self.offset, coords is Coordinates.POINT)
        self.modes = modes(car, coords, e)
        self.ride = None
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
        self._draw_tab()
        self._on_show()

    def _draw_tab(self) -> None:
        i = self.tabs.currentIndex()
        car, coords, e = self.car, self.coordinates, self.offset.value()
        if i == TAB_FBD:
            self._set_html(self.fbd, fbd_html(car, coords, e))
        elif i == TAB_LAGRANGE:
            self._set_html(self.lagrange, lagrange_html(car, coords, e))
        elif i == TAB_MODES:
            self._draw_modes()
        elif i == TAB_BUMP:
            self._draw_bump()
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

    # ------------------------------------------------------------- the animation
    def _on_show(self) -> None:
        self._time = 0.0
        self._draw_car()
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
        return f"{SHOWS[self.show.currentIndex()]}, {self.coordinates.name.lower()}"

    def step(self, dt: float) -> None:
        """Move the animation on by dt seconds."""
        self._phase = (self._phase + TWO_PI * SHOW_HZ * dt) % TWO_PI
        self._time += dt
        self._draw_car()

    def _coordinate_points(self) -> list[float]:
        car, coords = self.car, self.coordinates
        if coords is Coordinates.CENTRE:
            return [0.0]
        if coords is Coordinates.POINT:
            return [self.offset.value()]
        return [car.front, -car.rear]

    def _draw_car(self) -> None:
        car, points = self.car, self._coordinate_points()
        show = self.show.currentIndex()
        if show == SHOW_RIDE:
            self._ensure_ride()
            r = self.ride
            k = int(np.searchsorted(r.t, self._time % r.t[-1])) if self.running else 0
            k = min(k, r.t.size - 1)
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
        swing = math.sin(self._phase) if self.running else 1.0
        node = mode.node if math.isfinite(mode.node) else None
        self.view.draw(car, swing * xg, swing * th, None, (0.0, 0.0), points, node)
        self.view.caption.setText(f"Mode {show + 1}, {mode.frequency:.3g} Hz (swung slowly)")
