"""Modal coupling page: two subsystems, one mode each, joined as a 2-DOF system.

The chain from the Simulation page is split into A (grounded, masses 1..j) and
B (the rest, its first spring and damper tied to ground). One mode of each
becomes an equivalent oscillator, and the two oscillators are joined B's on
A's. The page compares the 2-DOF model's modes with the chain's, and shows how
the two modes veer apart as B's frequency or mass is swept: the mode splitting
of a vibration absorber, which happens whether or not it was intended.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core import ChainSystem, ModalResult, frf, modal_analysis
from ..core.modal_coupling import (
    A_MASS_NAMES,
    A_MASSES,
    FRF_POINTS,
    SWEEPS,
    CoupledComparison,
    CoupledModel,
    Subsystem,
    Sweep,
    alone_frf,
    chain_dof,
    compare_coupled,
    coupled_frf,
    coupled_model,
    subsystems,
    sweep,
)
from ..core.practice import Problem, Question, Requirement, at_least, at_most, kept, number, pick, rounded
from . import weak
from .animation import MASS_WIDTH, spring_path
from .axes import log_axes
from .coupling_notes import THEORY_HTML, TRIALS, CouplingTrial, matrices_html
from .layouts import ColumnScroll, form_layout, header_tip
from .lesson import TrialCombo
from .links import page_link
from .style import MAX_DOF, colors, text_on
from .substructuring import error_color, zeta_text
from .theming import add_legend, mute

DASH = QtCore.Qt.PenStyle.DashLine
DOT = QtCore.Qt.PenStyle.DotLine
SPLIT_TIP = (
    "<p>Split the chain after this mass. <b>A</b> is the masses up to it, on the ground as "
    "before. <b>B</b> is the rest; the spring and damper that joined it to A are tied to the "
    "ground instead.</p>"
    "<p>Each is solved on its own, one mode of each becomes a single-DOF oscillator, and the two "
    "oscillators are joined again, B's on A's.</p>"
)
A_MASS_TIP = (
    "<p>The mass of A's oscillator.</p>"
    "<p><b>Modal mass at the interface</b>, 1/φ<sub>tip</sub>² (φ mass-normalized): the mass of "
    "A's mode as its tip feels it. B is joined to A at A's tip, so this is the right one.</p>"
    "<p><b>Effective mass (base)</b>, Γ² with Γ = φᵀM1: the mass of A's mode as a shaking ground "
    "feels it. Right for B, which is joined at its base, but not for A. Choose it to see the "
    "difference.</p>"
)
RESIDUAL_TIP = (
    "<p>B's mass that is not in its chosen mode, m<sub>B</sub> − m<sub>eff</sub>, belongs to its "
    "other modes. Well below their frequencies they move rigidly with B's base, which is A's "
    "tip, so this mass rides on A's oscillator.</p>"
    "<p>With it (and the modal mass at the interface), the 2-DOF model is a Rayleigh–Ritz model "
    "of the chain, so its frequencies can only be too high.</p>"
)
SWEEP_TIP = (
    "<p><b>B's frequency:</b> B's springs are scaled so that f<sub>B</sub>/f<sub>A</sub> runs "
    "across the plot while the mass ratio μ stays as it is.</p>"
    "<p><b>Mass ratio:</b> all of B (masses and springs) is scaled so that μ runs across the plot "
    "while f<sub>B</sub> stays as it is.</p>"
    "<p>B's dampers are scaled with it, so its damping ratios stay the same.</p>"
)
SWEEP_NAMES = {"frequency": "B's frequency (f_B / f_A)", "mass": "Mass ratio (μ = m_b / m_a)"}
FRF_POINT_TIP = (
    "<p>Where the force is applied and the response read (a drive-point FRF).</p>"
    "<p><b>A's tip (the interface):</b> the 2-DOF model's u<sub>a</sub>.</p>"
    "<p><b>B's tip (the free end):</b> where a force is usually applied. The 2-DOF model gives it as "
    "u<sub>a</sub> + Γφ<sub>tip</sub>(u<sub>b</sub> − u<sub>a</sub>): B moving with its base plus its "
    "mode relative to it (the shapes on the Mode shapes tab).</p>"
    "<p>Both views also show A and B on their own, each at its own tip: A with B removed, B with A "
    "removed and its base on the ground. Their peaks are the modes the oscillators stand for.</p>"
)
FRF_POINT_NAMES = {"interface": "A's tip (the interface)", "free": "B's tip (the free end)"}


# ------------------------------------------------------------------ practice problems
TAB_VEERING, TAB_SHAPES, TAB_FRF, TAB_MATRICES = range(4)
TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class CouplingSetup:
    """A practice problem's setup: a chain (for the Simulation page) and how this page splits and couples it."""

    masses: tuple[float, ...] = (1.0, 1.0, 0.1, 0.1)
    springs: tuple[float, ...] = (400.0, 400.0, 40.0, 40.0)
    dampers: tuple[float, ...] = (2.0, 2.0, 0.2, 0.2)
    split: int = 2  # A is masses 1..split
    mode_a: int = 0  # 0-based
    mode_b: int = 0
    a_mass: str = "tip"
    residual: bool = True
    tab: int = TAB_VEERING

    def system(self) -> ChainSystem:
        return ChainSystem(np.array(self.masses), np.array(self.springs), np.array(self.dampers))


PRACTICE_SETUP = CouplingSetup  # what a practice problem's setup and the page's practice_state are


def _chain(system: ChainSystem) -> dict:
    """A chain as a setup's elements."""
    return dict(
        masses=tuple(float(v) for v in system.masses),
        springs=tuple(float(v) for v in system.stiffness),
        dampers=tuple(float(v) for v in system.damping),
    )


# Each question's numbers are printed round, and its answer is worked from them as printed.


def _tip_mass(rng: np.random.Generator) -> Question:
    m1, m2 = pick(rng, (1, 2, 5)), pick(rng, (0.5, 1, 2))
    k1, k2 = pick(rng, (400, 1000)), pick(rng, (200, 400, 1000))
    chain: dict[str, Any] = dict(masses=(m1, m2, 0.1, 0.1), springs=(k1, k2, 40.0, 40.0), dampers=(0.0,) * 4)
    A, _ = subsystems(CouplingSetup(**chain).system(), 2)
    psi = rounded(A.Phi[0, 0] / A.Phi[1, 0], 3)
    m_tip = m1 * psi * psi + m2
    return Question(
        f"A is two masses, m1 = {m1:g} kg and m2 = {m2:g} kg (its tip). Its mode 1, scaled to 1 at the tip, is "
        f"ψ = ({number(psi)}, 1). What is its modal mass at the interface, the mass of A's oscillator?",
        m_tip,
        f"m<sub>tip</sub> = 1/φ<sub>tip</sub>² = ψ<sup>T</sup>Mψ with ψ scaled to 1 at the tip: {m1:g} × "
        f"{number(psi)}² + {m2:g} × 1² = {number(m_tip)} kg, less than A's {m1 + m2:g} kg: per unit of tip motion the "
        "mode looks light.",
        unit="kg",
        setup=dict(**chain, tab=TAB_MATRICES),
    )


def _effective_mass(rng: np.random.Generator) -> Question:
    m3, m4 = pick(rng, (0.1, 0.2, 0.5)), pick(rng, (0.1, 0.2, 0.5))
    k3, k4 = pick(rng, (40, 100)), pick(rng, (20, 40, 100))
    chain: dict[str, Any] = dict(masses=(1.0, 1.0, m3, m4), springs=(400.0, 400.0, k3, k4), dampers=(0.0,) * 4)
    _, B = subsystems(CouplingSetup(**chain).system(), 2)
    phi = B.Phi[:, 0]
    psi = rounded(phi[0] / phi[1], 3)
    gamma2 = (m3 * psi + m4) ** 2 / (m3 * psi * psi + m4)
    return Question(
        f"B is two masses on a held base, m3 = {m3:g} kg and m4 = {m4:g} kg. Its mode 1, scaled to 1 at m4, is "
        f"ψ = ({number(psi)}, 1). What is its effective mass, the mass of B's oscillator?",
        gamma2,
        f"For any scaling, Γ² = (ψ<sup>T</sup>M<b>1</b>)²/(ψ<sup>T</sup>Mψ) = ({m3:g} × {number(psi)} + {m4:g})² / "
        f"({m3:g} × {number(psi)}² + {m4:g}) = {number((m3 * psi + m4) ** 2)}/{number(m3 * psi * psi + m4)} = "
        f"{number(gamma2)} kg. The rest of B's {m3 + m4:g} kg, {number(m3 + m4 - gamma2)} kg, is residual mass.",
        unit="kg",
        setup=dict(**chain, tab=TAB_MATRICES),
    )


def _oscillator_stiffness(rng: np.random.Generator) -> Question:
    m, f = pick(rng, (0.05, 0.1, 0.2, 0.5, 1.4)), pick(rng, (2, 5, 10, 20))
    k = m * (TWO_PI * f) ** 2
    return Question(
        f"A mode at {f:g} Hz has an effective mass of {m:g} kg. What stiffness does its equivalent oscillator get?",
        k,
        f"The oscillator keeps the mode's frequency: k = mω² = {m:g} × (2π × {f:g})² = {number(k)} N/m.",
        unit="N/m",
    )


def _tuned_gap(rng: np.random.Generator) -> Question:
    fa, mu = pick(rng, (2, 5, 10, 50)), pick(rng, (0.01, 0.02, 0.05, 0.1, 0.2))
    gap = math.sqrt(mu) * fa
    return Question(
        f"Two oscillators are tuned to the same {fa:g} Hz and joined, B's on A's, with a mass ratio μ = {mu:g}. How "
        "far apart are the two modes they split into?",
        gap,
        f"Tuned, the roots' product is ω<sub>a</sub>⁴ and their sum (2 + μ)ω<sub>a</sub>², so (ω₂ − ω₁)² = "
        f"μω<sub>a</sub>²: Δf = √μ f<sub>a</sub> = √{mu:g} × {fa:g} = {number(gap)} Hz, whatever μ is.",
        unit="Hz",
        tolerance=0.01,
    )


def _tuned_lower(rng: np.random.Generator) -> Question:
    fa, mu = pick(rng, (2, 5, 10, 50)), pick(rng, (0.01, 0.05, 0.1, 0.2, 0.5))
    lower = fa * math.sqrt(1 + mu / 2 - math.sqrt(mu + mu * mu / 4))
    return Question(
        f"Two oscillators are tuned to {fa:g} Hz and joined, B's on A's, with μ = {mu:g}. Where is the lower of the "
        "two modes?",
        lower,
        f"Tuned, ω²/ω<sub>a</sub>² = 1 + μ/2 ± √(μ + μ²/4): with the minus, "
        f"{number(1 + mu / 2 - math.sqrt(mu + mu * mu / 4), 4)}, so f₁ = {fa:g} × √that = {number(lower)} Hz.",
        unit="Hz",
        tolerance=0.01,
    )


LIGHT = dict(masses=(1.0, 1.0, 0.1, 0.1), springs=(400.0, 400.0, 40.0, 40.0), dampers=(2.0, 2.0, 0.2, 0.2))


def _wrong_mass(rng: np.random.Generator) -> Question:
    return Question(
        "A's oscillator is given its effective mass Γ² instead of its modal mass at the interface. What happens to the "
        "2-DOF model's frequencies against the chain's?",
        0,
        "They're no longer an upper bound: here mode 1 comes out 2.9 % high and mode 2 2.2 % low, against 0.2 % and "
        "0.3 % high with the modal mass at the interface. Only that mass (with the residual mass) makes the model "
        "Rayleigh–Ritz.",
        choices=(
            "Their errors grow, and can now have either sign",
            "They stay upper bounds, just less close",
            "Nothing: either mass will do",
        ),
        setup=dict(**LIGHT, a_mass="effective"),
        predict=True,
    )


def _no_residual(rng: np.random.Generator) -> Question:
    return Question(
        "B's residual mass is left off A's oscillator. Which way do the 2-DOF model's frequencies move?",
        0,
        "Up: the model loses mass that does move with A's tip (B's other modes, riding rigidly on it), so both "
        "frequencies rise, here from 0.2 % to 0.35 % high and from 0.33 % to 0.55 % high.",
        choices=("Both up", "Both down", "They don't move"),
        setup=dict(**LIGHT, residual=False),
        predict=True,
    )


def _heavier_b(rng: np.random.Generator) -> Question:
    return Question(
        "B stays tuned to A, but is made four times as heavy (its masses and springs both ×4). What happens to the two "
        "coupled modes?",
        0,
        "They split further apart: at tuning they're √μ f<sub>A</sub> apart, and μ grew four times, so the gap "
        "about doubles (the chain's 1.63 and 2.35 Hz become 1.35 and 2.75 Hz).",
        choices=("They split about twice as far apart", "They split four times as far", "They move closer", "Nothing"),
        setup=dict(masses=(1.0, 1.0, 0.4, 0.4), springs=(400.0, 400.0, 160.0, 160.0), dampers=(2.0, 2.0, 0.8, 0.8)),
        predict=True,
    )


def _detuned(rng: np.random.Generator) -> Question:
    return Question(
        "B is retuned to twice A's frequency (its springs ×4). What does coupling do to the two modes, against A's "
        "and B's own frequencies?",
        0,
        "It pushes them apart: the lower mode, mostly A, comes out below A's 1.97 Hz (1.81 Hz); the upper, mostly B, "
        "above B's 3.93 Hz (4.10 Hz). Far apart they don't split, each keeps its mode, shifted outwards.",
        choices=(
            "The lower moves below A's, the upper above B's",
            "Both stay where they were",
            "Both move up",
            "They meet in the middle",
        ),
        setup=dict(masses=(1.0, 1.0, 0.1, 0.1), springs=(400.0, 400.0, 160.0, 160.0), dampers=(2.0, 2.0, 0.8, 0.8)),
        predict=True,
    )


def _shared_damping(rng: np.random.Generator) -> Question:
    absorber = _chain(CouplingTrial("", "", "absorber", 1).system())
    return Question(
        "A machine with ζ = 0.025 carries an undamped absorber tuned to it. What damping do the two coupled modes get?",
        0,
        "About half each, the average: at exact tuning each coupled mode is half machine and half absorber, and here "
        "both come out at ζ = 0.012. That's how a tuned mass damper shares damping.",
        choices=("About half the machine's each", "The machine's 0.025 and 0", "None", "0.025 each"),
        setup=dict(**absorber, split=1),
        predict=True,
    )


def _split_spec(rng: np.random.Generator) -> Question:
    base = ChainSystem.uniform(3)
    A, _ = subsystems(base, 2)
    fa = float(A.fn_hz[0])
    md = pick(rng, (0.05, 0.08, 0.1, 0.15, 0.2))

    def chain(m3: float, ratio: float) -> dict:
        k3 = round(m3 * (TWO_PI * fa * ratio) ** 2, 2)
        return dict(masses=(1.0, 1.0, m3), springs=(400.0, 400.0, k3), dampers=(2.0, 2.0, 0.0))

    def gap(t: CouplingSetup) -> float:
        f = sorted(m.fn_hz for m in modal_analysis(t.system()).modes)
        return (f[1] - f[0]) / fa

    wanted = rounded(100 * gap(CouplingSetup(**chain(md, 1.0))) / 1.05, 2)
    heaviest = rounded(1.25 * md, 2)

    def goal(t: CouplingSetup):
        three = len(t.masses) == 3
        tuned = math.sqrt(t.springs[2] / t.masses[2]) / TWO_PI / fa if three else math.nan
        return (
            Requirement("Three masses", f"{len(t.masses)}", three),
            *(
                kept(what, now, given, unit)
                for what, now, given, unit in (
                    ("m1", t.masses[0], 1.0, "kg"),
                    ("m2", t.masses[1] if three else math.nan, 1.0, "kg"),
                    ("k1", t.springs[0], 400.0, "N/m"),
                    ("k2", t.springs[1] if three else math.nan, 400.0, "N/m"),
                )
            ),
            Requirement(
                f"B tuned to A: f<sub>B</sub> within 2 % of {number(fa)} Hz",
                f"f<sub>B</sub>/f<sub>A</sub> = {number(tuned, 4)}",
                abs(tuned - 1) <= 0.02,
            ),
            at_least(
                "The chain's modes 1 and 2 apart, over f<sub>A</sub>", 100 * gap(t) if three else 0.0, wanted, "%"
            ),
            at_most("B's mass m3", t.masses[2] if three else math.inf, heaviest, "kg"),
        )

    return Question(
        f"A is the uniform chain's m1 and m2 (1 kg, 400 N/m each; its mode 1 at {number(fa)} Hz). On the Simulation "
        f"page, choose B, a single m3 on k3, so that B is tuned to A and the chain's first two modes split at least "
        f"{number(wanted, 2)} % of f<sub>A</sub> apart, with m3 no more than {number(heaviest, 2)} kg.",
        0,
        f"Tuned, the modes are √μ f<sub>A</sub> apart, μ = m3/m<sub>tip</sub> with A's modal mass at the interface "
        f"m<sub>tip</sub> = {number(float(A.tip_masses()[0]))} kg (B has no residual mass: it is one mass). So m3 ≥ "
        f"({number(wanted / 100, 3)})² × {number(float(A.tip_masses()[0]))} ≈ "
        f"{number((wanted / 100) ** 2 * float(A.tip_masses()[0]))} kg, and k3 = m3 (2π × {number(fa)})² keeps it "
        f"tuned: m3 = {md:g} kg on k3 = {number(chain(md, 1.0)['springs'][2])} N/m splits them "
        f"{number(100 * gap(CouplingSetup(**chain(md, 1.0))))} %.",
        setup=dict(**chain(md, 1.5), split=2),
        goal=goal,
        design=dict(**chain(md, 1.0), split=2),
    )


PROBLEMS = tuple(
    Problem(key, title, "coupling_page", f"See {page_link('coupling_page', anchor, section)}.", make)
    for key, title, anchor, section, make in (
        (
            "tip-mass",
            "A's modal mass at the interface",
            "a-modal-mass",
            "A: its modal mass at the interface",
            _tip_mass,
        ),
        ("effective-mass", "B's effective mass", "effective-mass", "B: its effective mass", _effective_mass),
        ("oscillator-stiffness", "An oscillator's spring", "masses", "Which mass?", _oscillator_stiffness),
        ("tuned-gap", "Split by tuning", "splitting", "The 2-DOF system and mode splitting", _tuned_gap),
        ("tuned-lower", "The lower mode", "splitting", "The 2-DOF system and mode splitting", _tuned_lower),
        ("wrong-mass", "Predict: the wrong mass for A", "rayleigh-ritz", "Why this works", _wrong_mass),
        ("no-residual", "Predict: no residual mass", "effective-mass", "B: its effective mass", _no_residual),
        ("heavier-b", "Predict: a heavier B", "splitting", "The 2-DOF system and mode splitting", _heavier_b),
        ("detuned", "Predict: B far off", "splitting", "The 2-DOF system and mode splitting", _detuned),
        ("shared-damping", "Predict: shared damping", "damping", "Damping", _shared_damping),
        ("split-spec", "Spec: a split to order", "splitting", "The 2-DOF system and mode splitting", _split_spec),
    )
)


def _same_chain(a: ChainSystem, b: ChainSystem) -> bool:
    return a.n == b.n and all(
        np.allclose(x, y) for x, y in ((a.masses, b.masses), (a.stiffness, b.stiffness), (a.damping, b.damping))
    )


def sub_color(name: str) -> str:
    return colors.sub["AB".index(name)]


def _fit_table(table: QtWidgets.QTableWidget) -> None:
    # The header's size hint, not its height: a hidden table has not laid its header out yet.
    header = table.horizontalHeader()
    height = max(header.height(), header.sizeHint().height()) + 2 * table.frameWidth()
    height += sum(table.rowHeight(r) for r in range(table.rowCount()))
    table.setFixedHeight(height)


def _table(headers: list[str], tips: list[str], selectable: bool = False) -> QtWidgets.QTableWidget:
    table = QtWidgets.QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    for col, tip in enumerate(tips):
        header_tip(table, col, tip)
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    if selectable:
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
    else:
        table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
    table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
    return table


def _set_row(table: QtWidgets.QTableWidget, r: int, cells: list[tuple[str, str | None]]) -> None:
    for col, (text, color) in enumerate(cells):
        item = QtWidgets.QTableWidgetItem(text)
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        if color:
            item.setForeground(pg.mkColor(color))
        table.setItem(r, col, item)


def _g(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}g}"


class CouplingSchematic(pg.PlotWidget):
    """A and B on their own (top), and the equivalent 2-DOF system they become (bottom)."""

    GAP = 0.6  # extra room before B, for its own ground

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        self.hideAxis("left")
        self.hideAxis("bottom")
        self.setAspectLocked(True)
        self.setFixedHeight(250)
        self._items: list = []

    def _add(self, item):
        self.addItem(item)
        self._items.append(item)
        return item

    def _wall(self, x: float, y: float) -> None:
        pen = pg.mkPen(colors.structure, width=2)
        self._add(pg.PlotDataItem([x, x], [y - 0.3, y + 0.3], pen=pg.mkPen(colors.structure, width=4)))
        hx, hy = [], []
        for yy in np.linspace(y - 0.25, y + 0.3, 6):
            hx += [x, x - 0.08, np.nan]
            hy += [yy, yy - 0.08, np.nan]
        self._add(pg.PlotDataItem(hx, hy, pen=pen, connect="finite"))

    def _mass(self, xc: float, y: float, fill: str, text: str, outline: str | None = None, w: float = MASS_WIDTH):
        rect = QtWidgets.QGraphicsRectItem(QtCore.QRectF(xc - w / 2, y - 0.2, w, 0.4))
        rect.setBrush(pg.mkBrush(fill))
        rect.setPen(pg.mkPen(outline or colors.structure, width=2 if outline else 1))
        rect.setZValue(5)
        self._add(rect)
        label = pg.TextItem(text, color=text_on(fill), anchor=(0.5, 0.5))
        label.setPos(xc, y)
        label.setZValue(6)
        self._add(label)

    def _spring(self, x0: float, x1: float, y: float, color: str) -> None:
        self._add(pg.PlotDataItem(*spring_path(x0, x1, y), pen=pg.mkPen(color, width=2)))

    def _text(self, text: str, x: float, y: float, color: str, anchor=(0.5, 0.5)) -> None:
        item = pg.TextItem(text, color=color, anchor=anchor)
        item.setPos(x, y)
        self._add(item)

    def set_model(self, n: int, split: int, model: CoupledModel | None) -> None:
        for item in self._items:
            self.removeItem(item)
        self._items = []
        half = MASS_WIDTH / 2
        # --- top: A and B on their own
        y = 0.75
        self._text("on their own", -0.25, y + 0.5, colors.muted, anchor=(0.0, 0.5))
        self._wall(0.0, y)
        right = 0.0
        for i in range(n):
            b = i >= split
            name = "B" if b else "A"
            xc = i + 1.0 + (self.GAP if b else 0.0)
            if i == split:  # B's own ground, where A's tip was
                right = split + 0.35 + self.GAP / 2
                self._wall(right, y)
            self._spring(right, xc - half, y, sub_color(name))
            self._mass(xc, y, colors.mass[i], f"m{i + 1}")
            right = xc + half
        self._text("A", split / 2 + 0.5, y - 0.4, sub_color("A"))
        self._text("B", split + 1 + self.GAP + (n - split - 1) / 2, y - 0.4, sub_color("B"))
        # --- bottom: the equivalent 2-DOF system
        y = -0.85
        self._text("equivalent 2-DOF", -0.25, y + 0.55, colors.muted, anchor=(0.0, 0.5))
        self._wall(0.0, y)
        xa, xb = 1.5, 3.2
        w = 0.6
        if model is None:
            self._text("—", 1.5, y, colors.muted)
        else:
            self._spring(0.0, xa - w / 2, y, sub_color("A"))
            self._spring(xa + w / 2, xb - w / 2, y, sub_color("B"))
            self._mass(xa, y, sub_color("A"), "m_a", w=w)
            self._mass(xb, y, sub_color("B"), "m_b", w=w)
            self._text(f"A{model.mode_a + 1}", xa / 2 - 0.1, y + 0.3, sub_color("A"))
            self._text(f"B{model.mode_b + 1}", (xa + xb) / 2, y + 0.3, sub_color("B"))
            res = f" + {model.m_residual:.3g}" if model.residual and model.m_residual > 1e-9 else ""
            self._text(f"m_a = {model.osc_a.m:.3g}{res} kg", xa, y - 0.25, colors.foreground, anchor=(0.5, 0.0))
            self._text(f"m_b = {model.osc_b.m:.3g} kg", xb + 0.2, y - 0.25, colors.foreground, anchor=(0.0, 0.0))
        self.setRange(QtCore.QRectF(-0.4, -1.45, n + 1.0 + self.GAP + 0.2, 2.55), padding=0)


class SubsystemTable(QtWidgets.QTableWidget):
    """The modes of A and of B on their own, with the masses a joint would feel. A click couples that mode."""

    HEADERS = [
        "Mode",
        "fₙ [Hz]",
        "ζ modal",
        "Effective\nmass [kg]",
        "Share of\nits mass",
        "Modal mass at\ninterface [kg]",
    ]
    TIPS = [
        "Mode r of A (grounded, tip free) or of B (its first spring and damper tied to ground). "
        "Click a row to couple that mode.",
        "Natural frequency of the subsystem on its own: Kφ = ω²Mφ",
        "Modal damping ratio φᵀCφ / (2ω), φ mass-normalized: the damping of the equivalent oscillator",
        "Γ² with Γ = φᵀM1: the mass of this mode that a motion of the subsystem's base (the ground) "
        "feels. They add up to the subsystem's mass. For B, whose base is A's tip, this is the mass "
        "of its oscillator.",
        "Effective mass as a fraction of the subsystem's total mass",
        "1/φ_tip² for A: the mass of this mode that A's tip feels, as for a vibration absorber's "
        "primary. B is joined to A at A's tip, so this is the mass of A's oscillator (by default).",
    ]

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(0, len(self.HEADERS), parent)
        self.setHorizontalHeaderLabels(self.HEADERS)
        for col, tip in enumerate(self.TIPS):
            header_tip(self, col, tip)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.rows: list[tuple[str, int]] = []  # (subsystem, 0-based mode) of each row

    def set_subsystems(self, A: Subsystem, B: Subsystem, mode_a: int, mode_b: int, a_mass: str) -> None:
        self.rows = [("A", r) for r in range(A.omegas.size)] + [("B", r) for r in range(B.omegas.size)]
        self.setRowCount(len(self.rows))
        for row, (name, r) in enumerate(self.rows):
            sub = A if name == "A" else B
            used = r == (mode_a if name == "A" else mode_b)
            meff = sub.effective_masses[r]
            rigid = sub.omegas[r] < 1e-9
            tip = sub.tip_masses()[r] if name == "A" else None
            # The mass the oscillator takes is shown bold.
            used_col = (5 if a_mass == "tip" else 3) if name == "A" else 3
            cells = [
                (f"{name}{r + 1}" + ("  ◀" if used else ""), sub_color(name)),
                ("0 (rigid)" if rigid else _g(sub.fn_hz[r]), None),
                ("—" if rigid else f"{sub.zetas[r]:.4f}", None),
                (_g(meff, 3), None),
                (f"{100 * meff / sub.total_mass:.1f}%", None),
                ("—" if tip is None else ("∞" if not np.isfinite(tip) else _g(tip, 3)), None),
            ]
            _set_row(self, row, cells)
            if used:
                for col in (0, used_col):
                    item = self.item(row, col)
                    assert item is not None, "_set_row filled the row"
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
        _fit_table(self)


class VeeringPlot(QtWidgets.QWidget):
    """Natural frequencies of the chain and of the 2-DOF model as B's frequency or mass is swept."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.kind = QtWidgets.QComboBox()
        for k in SWEEPS:
            self.kind.addItem(SWEEP_NAMES[k], k)
        self.kind.setToolTip(SWEEP_TIP)
        top = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel("Sweep:")
        label.setToolTip(SWEEP_TIP)
        top.addWidget(label)
        top.addWidget(self.kind)
        top.addStretch(1)
        self.plot = pg.PlotWidget(axisItems=log_axes())
        self.plot.setLogMode(x=True, y=True)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel("left", "Natural frequency", units="Hz")
        self.plot.setMouseEnabled(x=False, y=False)
        self.legend = add_legend(self.plot, offset=(5, 5))
        self.note = QtWidgets.QLabel()
        self.note.setWordWrap(True)
        mute(self.note)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.note)
        self.data: Sweep | None = None

    def apply_theme(self) -> None:
        self.legend.setBrush(colors.legend_brush())

    def clear(self, reason: str) -> None:
        self.data = None
        self.plot.clear()
        self.legend.clear()
        self.plot.setTitle("")
        self.note.setText(reason)

    def set_model(self, model: CoupledModel, comparisons: list[CoupledComparison]) -> None:
        self.plot.clear()
        self.legend.clear()
        kind = self.kind.currentData()
        try:
            self.data = s = sweep(model, kind)
        except ValueError as exc:
            self.clear(str(exc))
            return
        paired = {c.full_index - 1 for c in comparisons}
        for r in range(s.full.shape[1]):
            color = colors.mode[r % len(colors.mode)]
            width = 3 if r in paired else 1.5
            self.plot.plot(
                s.x, s.full[:, r], pen=pg.mkPen(color, width=width), name="chain (every mode)" if r == 0 else None
            )
        for r in range(2):
            self.plot.plot(
                s.x,
                s.two_dof[:, r],
                pen=pg.mkPen(colors.strong, width=2, style=DASH),
                name="2-DOF model" if r == 0 else None,
            )
        self.plot.plot(s.x, s.f_a, pen=pg.mkPen(sub_color("A"), width=2, style=DOT), name=f"A{model.mode_a + 1} alone")
        self.plot.plot(s.x, s.f_b, pen=pg.mkPen(sub_color("B"), width=2, style=DOT), name=f"B{model.mode_b + 1} alone")
        now = pg.InfiniteLine(
            pos=math.log10(s.x_now),
            angle=90,
            pen=pg.mkPen(colors.grey, width=1),
            label="now",
            labelOpts={"position": 0.95, "color": colors.muted},
        )
        self.plot.addItem(now)
        self.plot.plot(
            [s.x_now, s.x_now],
            list(model.fn_hz),
            pen=None,
            symbol="o",
            symbolSize=10,
            symbolBrush=None,
            symbolPen=pg.mkPen(colors.strong, width=2),
        )
        top = 1.4 * max(s.two_dof.max(), 1e-9)
        bottom = max(min(s.two_dof.min(), s.f_a.min()), 1e-9) / 1.4
        self.plot.setYRange(math.log10(bottom), math.log10(top), padding=0)
        self.plot.setXRange(math.log10(s.x[0]), math.log10(s.x[-1]), padding=0)
        if kind == "frequency":
            self.plot.setLabel("bottom", "f_B / f_A (B's springs scaled)")
            self.plot.setTitle(f"μ = {model.mass_ratio:.3g} fixed", size="10pt")
            gap = self._gap_at_tuning(s)
            self.note.setText(
                "Solid: the chain's natural frequencies (thick: the two the 2-DOF modes stand for). "
                "Dashed: the 2-DOF model. Dotted: A's and B's modes on their own. Far from f<sub>B</sub> = "
                "f<sub>A</sub> each coupled mode follows one of them. Near it the two modes <i>veer</i> "
                "apart instead of crossing, and trade shapes: the mode splitting of a vibration absorber. "
                + (
                    f"At f<sub>B</sub> = f<sub>A</sub> the 2-DOF modes are {100 * gap:.3g}% of f<sub>A</sub> "
                    f"apart; two oscillators tuned exactly are √μ = {100 * math.sqrt(model.mass_ratio):.3g}% "
                    "apart, whatever μ. "
                    if gap is not None
                    else ""
                )
                + "Circles: the current parameters."
            )
        else:
            self.plot.setLabel("bottom", "μ = m_b / m_a (all of B scaled)")
            self.plot.setTitle(f"f_B / f_A = {model.freq_ratio:.3g} fixed", size="10pt")
            self.note.setText(
                "Solid: the chain's natural frequencies (thick: the two the 2-DOF modes stand for). "
                "Dashed: the 2-DOF model. Dotted: A's and B's modes on their own, which do not change. "
                "A light B barely moves A's mode; the heavier B, the more the two modes are pushed apart. "
                "The closer f<sub>B</sub>/f<sub>A</sub> is to 1, the smaller the mass ratio it takes. "
                "Circles: the current parameters."
            )

    @staticmethod
    def _gap_at_tuning(s: Sweep) -> float | None:
        """(f2 - f1) / f_A where f_B = f_A, if the sweep reaches it."""
        if not (s.x[0] <= 1.0 <= s.x[-1]):
            return None
        f1 = np.interp(0.0, np.log(s.x), s.two_dof[:, 0])
        f2 = np.interp(0.0, np.log(s.x), s.two_dof[:, 1])
        return float((f2 - f1) / s.f_a[0])


class ShapePlots(pg.GraphicsLayoutWidget):
    """Each 2-DOF mode drawn on the chain, over the chain's own mode it stands for."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.plots = []
        self.legends = []
        for r in range(2):
            p = self.addPlot(row=r, col=0)
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setMouseEnabled(x=False, y=False)
            p.setYRange(-1.15, 1.6, padding=0)
            p.getAxis("left").setTicks([[(v, f"{v:g}") for v in (-1, -0.5, 0, 0.5, 1)]])
            p.setLabel("left", "Normalized amplitude")
            self.plots.append(p)
            self.legends.append(add_legend(p, offset=(5, 2), colCount=2))
        self.plots[1].setLabel("bottom", "Position along chain (0 = ground)")
        self.apply_theme()

    def apply_theme(self) -> None:
        for legend in self.legends:
            legend.setBrush(colors.legend_brush())

    def set_model(self, model: CoupledModel | None, full: ModalResult, comparisons: list[CoupledComparison]) -> None:
        for p, legend in zip(self.plots, self.legends):
            p.clear()
            legend.clear()
            p.setTitle("")
        if model is None:
            return
        n = model.system.n
        xs = np.arange(n + 1)
        for r, (p, c) in enumerate(zip(self.plots, comparisons)):
            line = pg.InfiniteLine(
                pos=model.split + 0.5,
                angle=90,
                pen=pg.mkPen(colors.grey, width=1, style=DASH),
                label="split",
                labelOpts={"position": 0.08, "color": colors.muted},
            )
            p.addItem(line)
            p.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(colors.faint, width=1)))
            true = full.modes[c.full_index - 1].shape
            red = model.shapes[:, r]
            if red @ true < 0:
                red = -red
            color = colors.mode[(c.full_index - 1) % len(colors.mode)]
            p.plot(
                xs,
                np.concatenate([[0.0], true]),
                pen=pg.mkPen(color, width=3),
                symbol="s",
                symbolSize=9,
                symbolPen=None,
                symbolBrush=color,
                name=f"chain mode {c.full_index}: {c.fn_full:.4g} Hz",
            )
            p.plot(
                xs,
                np.concatenate([[0.0], red]),
                pen=pg.mkPen(colors.strong, width=2, style=DASH),
                symbol="o",
                symbolSize=11,
                symbolBrush=None,
                symbolPen=pg.mkPen(colors.strong, width=2),
                name=f"2-DOF mode {r + 1}: {c.fn:.4g} Hz",
            )
            ua, ub = model.eta[:, r] / np.abs(model.eta[:, r]).max()
            phase = "in phase" if ua * ub > 0 else "in opposite phase"
            err = "—" if c.error is None else f"{100 * c.error:+.3g}%"
            p.setTitle(
                f"2-DOF mode {r + 1}: m_a and m_b {phase} (u_b/u_a = {ub / ua:.3g}) · error {err}, MAC {c.mac:.3f}"
                if abs(ua) > 1e-12
                else f"2-DOF mode {r + 1}: m_a still · error {err}, MAC {c.mac:.3f}",
                size="10pt",
            )
            p.setXRange(0, n, padding=0.05)
            p.getAxis("bottom").setTicks([[(0, "ground")] + [(i, f"m{i}") for i in range(1, n + 1)]])


class DrivePointFrf(QtWidgets.QWidget):
    """Receptance at A's tip (the interface) or B's tip (the free end) for a force there.

    The chain, the 2-DOF model, and the side that holds the point on its own.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.point = QtWidgets.QComboBox()
        for p in FRF_POINTS:
            self.point.addItem(FRF_POINT_NAMES[p], p)
        self.point.setToolTip(FRF_POINT_TIP)
        top = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel("Force and response at:")
        label.setToolTip(FRF_POINT_TIP)
        top.addWidget(label)
        top.addWidget(self.point)
        top.addStretch(1)
        self.plot = pg.PlotWidget(axisItems=log_axes())
        self.plot.setLogMode(x=True, y=True)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel("left", "|X / F|  [m/N]")
        self.plot.setLabel("bottom", "Frequency", units="Hz")
        self.legend = add_legend(self.plot, offset=(-5, 5))
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.plot, 1)
        self.point.currentIndexChanged.connect(self._redraw)
        self._args: tuple[CoupledModel | None, ChainSystem, ModalResult] | None = None

    def apply_theme(self) -> None:
        self.legend.setBrush(colors.legend_brush())

    def set_model(self, model: CoupledModel | None, system: ChainSystem, full: ModalResult) -> None:
        self._args = (model, system, full)
        self._redraw()

    def _redraw(self) -> None:
        p = self.plot
        p.clear()
        self.legend.clear()
        model, system, full = self._args if self._args is not None else (None, None, None)
        if model is None or system is None or full is None:
            p.setTitle("")
            return
        point = self.point.currentData()
        dof = chain_dof(model, point)
        fn = np.array([m.fn_hz for m in full.modes if m.fn_hz > 0])
        ref = np.concatenate([fn, model.fn_hz[model.fn_hz > 0]])
        lo = 0.2 * ref.min() if ref.size else 0.1
        hi = 1.5 * ref.max() if ref.size else 10.0
        f = np.unique(np.concatenate([np.geomspace(lo, hi, 2000), ref[(ref > lo) & (ref < hi)]]))
        x = f"x{dof + 1} / F{dof + 1}"
        p.plot(f, np.abs(frf(system, f, dof)[:, dof]), pen=pg.mkPen(colors.strong, width=2), name=f"chain: {x}")
        p.plot(
            f,
            np.abs(coupled_frf(model, f, point)),
            pen=pg.mkPen(colors.force, width=2, style=DASH),
            name=f"2-DOF: {x}" if point == "free" else "2-DOF: u_a / F",
        )
        n = system.n
        for side, name in (
            ("A", f"A alone at m{model.split} (B removed)"),
            ("B", f"B alone at m{n} (A removed, base on ground)"),
        ):
            p.plot(f, np.abs(alone_frf(model, f, side)), pen=pg.mkPen(sub_color(side), width=1.5, style=DOT), name=name)
        for fr in model.fn_hz:
            if lo < fr < hi:
                p.addItem(pg.InfiniteLine(pos=math.log10(fr), angle=90, pen=pg.mkPen(colors.grey, width=1, style=DOT)))
        where = "A's tip (the interface)" if point == "interface" else "B's tip (the free end)"
        p.setTitle(f"Force at m{dof + 1}, {where} · dotted lines: 2-DOF fₙ", size="10pt")
        p.setXRange(math.log10(lo), math.log10(hi), padding=0)


class ModalCouplingPage(QtWidgets.QWidget):
    """Controls, schematic and tables on the left; veering, shapes, FRF and theory on the right."""

    dof_requested = QtCore.Signal(int)  # the user changed N here; the main window owns the system
    edit_parameters = QtCore.Signal()  # the user wants the Simulation page's parameter panel
    system_requested = QtCore.Signal(object)  # a trial's chain, for the Simulation page

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.system: ChainSystem | None = None
        self.full: ModalResult | None = None
        self.model: CoupledModel | None = None
        self.comparisons: list[CoupledComparison] = []
        self._dirty = False
        self._split_wanted: int | None = None  # None: the middle of the chain
        self._modes_wanted = [0, 0]

        trials = QtWidgets.QGroupBox("Things to try")
        tv = QtWidgets.QVBoxLayout(trials)
        self.trial = TrialCombo(
            [t.name for t in TRIALS],
            "<p>Each loads a chain into the Simulation page and splits and couples it to "
            "show one idea from the Theory tab.</p>",
        )
        self.trial.picked.connect(self.load_trial)
        tv.addWidget(self.trial)
        self.trial_note = QtWidgets.QLabel()
        self.trial_note.setWordWrap(True)
        mute(self.trial_note)
        self.trial_note.setVisible(False)
        tv.addWidget(self.trial_note)

        box = QtWidgets.QGroupBox("Modal coupling")
        form = form_layout(box)
        self.dof = QtWidgets.QSpinBox()
        self.dof.setRange(1, MAX_DOF)
        self.dof.setToolTip("Same chain as the Simulation page; changing it here changes it there too.")
        self.dof.valueChanged.connect(self.dof_requested)
        form.addRow("Number of masses:", self.dof)
        splits = QtWidgets.QWidget()
        splits.setToolTip(SPLIT_TIP)
        self.split_row = QtWidgets.QHBoxLayout(splits)
        self.split_row.setContentsMargins(0, 0, 0, 0)
        self.split_row.setSpacing(3)
        self.split_group = QtWidgets.QButtonGroup(self)
        self.split_group.setExclusive(True)
        self.split_group.idClicked.connect(self._on_split)
        self.split_buttons: list[QtWidgets.QPushButton] = []
        split_label = QtWidgets.QLabel("Split after:")
        split_label.setToolTip(SPLIT_TIP)
        form.addRow(split_label, splits)
        self.mode_a = QtWidgets.QComboBox()
        self.mode_b = QtWidgets.QComboBox()
        for s, combo in enumerate((self.mode_a, self.mode_b)):
            combo.setToolTip(
                "The mode of this subsystem that becomes its oscillator. Or click its row in the table below."
            )
            combo.activated.connect(weak.slot(self, lambda self, i, s=s: self._on_mode(s, i)))
        self.mode_labels = [QtWidgets.QLabel(), QtWidgets.QLabel()]
        form.addRow(self.mode_labels[0], self.mode_a)
        form.addRow(self.mode_labels[1], self.mode_b)
        self.a_mass = QtWidgets.QComboBox()
        for key in A_MASSES:
            self.a_mass.addItem(A_MASS_NAMES[key][0].upper() + A_MASS_NAMES[key][1:], key)
        self.a_mass.setToolTip(A_MASS_TIP)
        self.a_mass.currentIndexChanged.connect(self.refresh)
        a_mass_label = QtWidgets.QLabel("Mass of A's oscillator:")
        a_mass_label.setToolTip(A_MASS_TIP)
        form.addRow(a_mass_label, self.a_mass)
        self.residual = QtWidgets.QCheckBox("Add B's residual mass to A")
        self.residual.setChecked(True)
        self.residual.setToolTip(RESIDUAL_TIP)
        self.residual.toggled.connect(self.refresh)
        form.addRow(self.residual)
        link = QtWidgets.QLabel("Masses, springs and dampers are edited on the <a href='#sim'>Simulation page</a>.")
        mute(link)
        link.linkActivated.connect(self.edit_parameters)
        form.addRow(link)

        self.schematic = CouplingSchematic()
        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        self.sub_title = QtWidgets.QLabel("<b>A and B on their own</b> (click a row to couple that mode)")
        self.sub_table = SubsystemTable()
        self.sub_table.cellClicked.connect(self._on_sub_row)
        self.osc_title = QtWidgets.QLabel("<b>Equivalent oscillators</b>")
        self.osc_table = _table(
            ["Oscillator", "m [kg]", "k [N/m]", "c [N·s/m]", "fₙ [Hz]", "ζ"],
            [
                "The mode each oscillator stands for",
                "m_a: A's modal mass at the interface (or its effective mass), plus B's residual mass if "
                "added. m_b: B's effective mass.",
                "k = m ω²: the oscillator keeps its mode's natural frequency",
                "c = m φᵀCφ = 2ζωm: the oscillator keeps its mode's modal damping ratio",
                "Natural frequency of the oscillator on its own (without the residual mass)",
                "Damping ratio of the oscillator on its own",
            ],
        )
        self.result_title = QtWidgets.QLabel("<b>Coupled: the 2-DOF model against the whole chain</b>")
        self.result_table = _table(
            ["2-DOF\nmode", "fₙ 2-DOF\n[Hz]", "ζ 2-DOF", "Chain\nmode", "fₙ chain\n[Hz]", "ζ chain", "Error", "MAC"],
            [
                "Mode of the 2-DOF model (ground – k_a – m_a – k_b – m_b)",
                "Natural frequency of the 2-DOF model",
                "Exact damping ratio of the 2-DOF model (from its damped eigenvalues)",
                "The chain's mode whose shape the 2-DOF mode matches best (each 2-DOF mode a different one)",
                "Natural frequency of that mode of the whole chain",
                "Exact damping ratio of that mode of the whole chain",
                "(f_2DOF − f_chain) / f_chain. With the modal mass at the interface and the residual mass "
                "the 2-DOF model is a Rayleigh–Ritz model, so its mode 1 is never below the chain's mode 1, "
                "nor its mode 2 below the chain's mode 2. Paired by shape with higher chain modes, the "
                "error can be negative.",
                "Modal assurance criterion between the chain's mode and the 2-DOF mode drawn on the chain "
                "(Mode shapes tab). Low: the two modes are too far apart for this pairing to mean much.",
            ],
        )
        self.result_note = QtWidgets.QLabel()
        self.result_note.setWordWrap(True)
        mute(self.result_note)
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.addWidget(trials)
        lv.addWidget(box)
        lv.addWidget(self.schematic)
        lv.addWidget(self.summary)
        lv.addSpacing(6)
        lv.addWidget(self.result_title)
        lv.addWidget(self.result_table)
        lv.addWidget(self.result_note)
        lv.addSpacing(6)
        lv.addWidget(self.osc_title)
        lv.addWidget(self.osc_table)
        lv.addSpacing(6)
        lv.addWidget(self.sub_title)
        lv.addWidget(self.sub_table)
        lv.addStretch(1)
        left_scroll = ColumnScroll(left)

        self.veering = VeeringPlot()
        self.veering.kind.currentIndexChanged.connect(self._on_sweep_kind)
        self.shapes = ShapePlots()
        self.frf = DrivePointFrf()
        self.matrices = QtWidgets.QTextBrowser()
        self.theory = QtWidgets.QTextBrowser()
        self.theory.setHtml(THEORY_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.veering, "Veering && splitting")
        self.tabs.addTab(self.shapes, "Mode shapes")
        self.tabs.addTab(self.frf, "Drive-point FRF")
        self.tabs.addTab(self.matrices, "Matrices (step by step)")
        self.tabs.addTab(self.theory, "Theory")

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(self.tabs)
        splitter.setSizes([620, 1080])
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)
        self.apply_theme()

    def apply_theme(self) -> None:
        """Recolour the labels now; everything drawn from the model is redrawn when the page shows."""
        for name, label in zip("AB", self.mode_labels):
            label.setText(f"<b style='color:{sub_color(name)}'>Mode of {name}:</b>")
        self.veering.apply_theme()
        self.shapes.apply_theme()
        self.frf.apply_theme()
        scroll = self.theory.verticalScrollBar().value()
        self.theory.setHtml(THEORY_HTML)  # its links, in the new link colour
        self.theory.verticalScrollBar().setValue(scroll)
        if self.system is not None:
            self._dirty = True
            if self.isVisible():
                self.refresh()

    # ------------------------------------------------------------ inputs
    @property
    def split(self) -> int:
        """Number of masses in A (split after mass `split`); 0 when the chain cannot be split."""
        return self.split_group.checkedId() + 1

    def set_split(self, split: int) -> None:
        """Split the chain after mass `split` (1-based: A is masses 1..split)."""
        if self.system is None or not 1 <= split <= self.system.n - 1:
            raise ValueError("the split must leave at least one mass on each side")
        self._on_split(split - 1)

    def set_modes(self, mode_a: int, mode_b: int) -> None:
        """Couple A's mode `mode_a` with B's mode `mode_b` (0-based; clipped to what there is)."""
        self._modes_wanted = [mode_a, mode_b]
        self.refresh()

    def set_system(self, system: ChainSystem, full: ModalResult | None = None) -> None:
        """New chain parameters. Recomputed now if visible, else when the page is shown."""
        self.system = system
        self.full = full
        self._dirty = True
        if self.isVisible():
            self.refresh()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._dirty:
            self.refresh()

    def load_trial(self, index: int) -> None:
        """Load TRIALS[index]: its chain (through the Simulation page), split, modes and A's mass."""
        trial = TRIALS[index]
        self.trial_note.setText(trial.note)
        self.trial_note.setVisible(True)
        system = trial.system()
        self.system_requested.emit(system)
        if self.system is None or self.system.n != system.n or not np.allclose(self.system.masses, system.masses):
            self.set_system(system)  # nobody passed it back: show it here alone
        self.a_mass.blockSignals(True)
        self.a_mass.setCurrentIndex(self.a_mass.findData(trial.a_mass))
        self.a_mass.blockSignals(False)
        self.residual.blockSignals(True)
        self.residual.setChecked(True)
        self.residual.blockSignals(False)
        self._split_wanted = trial.split
        self._modes_wanted = [trial.mode_a, trial.mode_b]
        self.refresh()
        self.tabs.setCurrentIndex(trial.tab)

    def load_setup(self, setup: dict) -> None:
        """Set the page up as a practice problem's `setup` says: its chain (through the Simulation page), then how
        it's split and coupled. Worked out now only if the page shows."""
        t = CouplingSetup(**setup)
        self.trial_note.setVisible(False)
        system = t.system()
        self.system_requested.emit(system)
        if self.system is None or not _same_chain(self.system, system):
            self.set_system(system)  # nobody passed it back: show it here alone
        self.a_mass.blockSignals(True)
        self.a_mass.setCurrentIndex(self.a_mass.findData(t.a_mass))
        self.a_mass.blockSignals(False)
        self.residual.blockSignals(True)
        self.residual.setChecked(t.residual)
        self.residual.blockSignals(False)
        self._split_wanted = t.split
        self._modes_wanted = [t.mode_a, t.mode_b]
        self.tabs.setCurrentIndex(t.tab)
        self._dirty = True
        if self.isVisible():
            self.refresh()

    def practice_state(self) -> CouplingSetup:
        """The chain and how it's split and coupled, as a setup: what a practice spec checks."""
        s = self.system
        n = s.n if s is not None else 0
        split = int(np.clip(self._split_wanted or n // 2, 1, max(n - 1, 1)))
        return CouplingSetup(
            **(_chain(s) if s is not None else dict(masses=(), springs=(), dampers=())),
            split=split,
            mode_a=int(np.clip(self._modes_wanted[0], 0, max(split - 1, 0))),
            mode_b=int(np.clip(self._modes_wanted[1], 0, max(n - split - 1, 0))),
            a_mass=self.a_mass.currentData(),
            residual=self.residual.isChecked(),
            tab=self.tabs.currentIndex(),
        )

    def _on_split(self, i: int) -> None:
        self._split_wanted = i + 1
        self.refresh()

    def _on_mode(self, s: int, index: int) -> None:
        self._modes_wanted[s] = index
        self.refresh()

    def _on_sub_row(self, row: int, _col: int) -> None:
        name, r = self.sub_table.rows[row]
        self._on_mode("AB".index(name), r)

    def _on_sweep_kind(self) -> None:
        if self.model is not None:
            self.veering.set_model(self.model, self.comparisons)

    def _build_split_buttons(self, n: int) -> None:
        for b in self.split_buttons:
            self.split_group.removeButton(b)
            self.split_row.removeWidget(b)
            b.deleteLater()
        self.split_buttons = []
        for i in range(n - 1):
            b = QtWidgets.QPushButton(f"m{i + 1}")
            b.setCheckable(True)
            b.setMinimumWidth(10)
            b.setToolTip(SPLIT_TIP)
            self.split_group.addButton(b, i)
            self.split_row.addWidget(b, 1)
            self.split_buttons.append(b)

    def _sync_controls(self, n: int) -> None:
        if len(self.split_buttons) != n - 1:
            self._build_split_buttons(n)
        if n < 2:
            return
        split = int(np.clip(self._split_wanted or n // 2, 1, n - 1))
        for i, b in enumerate(self.split_buttons):
            b.blockSignals(True)
            b.setChecked(i == split - 1)
            b.setText(f"m{i + 1} ✂" if i == split - 1 else f"m{i + 1}")
            font = b.font()
            font.setBold(i == split - 1)
            b.setFont(font)
            b.blockSignals(False)

    def _fill_mode_combos(self, A: Subsystem, B: Subsystem) -> list[int]:
        chosen = []
        for s, (combo, sub) in enumerate(zip((self.mode_a, self.mode_b), (A, B))):
            r = int(np.clip(self._modes_wanted[s], 0, sub.omegas.size - 1))
            combo.blockSignals(True)
            combo.clear()
            for k, f in enumerate(sub.fn_hz):
                combo.addItem(f"{sub.name}{k + 1}: {f:.4g} Hz")
            combo.setCurrentIndex(r)
            combo.blockSignals(False)
            chosen.append(r)
        return chosen

    # ----------------------------------------------------------- compute
    def refresh(self) -> None:
        if self.system is None:
            return
        self._dirty = False
        system = self.system
        n = system.n
        self.dof.blockSignals(True)
        self.dof.setValue(n)
        self.dof.blockSignals(False)
        self._sync_controls(n)
        full = self.full if self.full is not None else modal_analysis(system)
        self.full = full
        if n < 2:
            self._clear("Modal coupling needs two subsystems: use at least 2 masses.", n)
            return
        split = self.split
        A, B = subsystems(system, split)
        mode_a, mode_b = self._fill_mode_combos(A, B)
        a_mass = self.a_mass.currentData()
        self.sub_table.set_subsystems(A, B, mode_a, mode_b, a_mass)
        try:
            model = coupled_model(system, split, mode_a, mode_b, a_mass, self.residual.isChecked())
        except ValueError as exc:
            self._clear(str(exc), n, split)
            return
        self.model = model
        self.comparisons = comparisons = compare_coupled(model, full)
        self.schematic.set_model(n, split, model)
        self.summary.setText(self._summary(model))
        self._fill_oscillators(model)
        self.result_table.setRowCount(2)
        for r, c in enumerate(comparisons):
            color = colors.mode[(c.full_index - 1) % len(colors.mode)]
            _set_row(
                self.result_table,
                r,
                [
                    (str(c.index), None),
                    (_g(c.fn), None),
                    (zeta_text(c.zeta), None),
                    (f"mode {c.full_index}", color),
                    (_g(c.fn_full), None),
                    (zeta_text(c.zeta_full), None),
                    (
                        "—" if c.error is None else f"{100 * c.error:+.3g}%",
                        None if c.error is None else error_color(c.error),
                    ),
                    (f"{c.mac:.3f}", colors.poor if c.mac < 0.8 else None),
                ],
            )
        _fit_table(self.result_table)
        self.result_note.setText(self._result_note(model, full, comparisons))
        self.veering.set_model(model, comparisons)
        self.shapes.set_model(model, full, comparisons)
        self.frf.set_model(model, system, full)
        scroll = self.matrices.verticalScrollBar().value()
        self.matrices.setHtml(matrices_html(model, full))
        self.matrices.verticalScrollBar().setValue(scroll)

    def _clear(self, reason: str, n: int, split: int = 0) -> None:
        self.model, self.comparisons = None, []
        self.schematic.set_model(n, split if split else n, None)
        self.summary.setText(f"<b style='color:{colors.poor}'>{reason}</b>")
        self.osc_table.setRowCount(0)
        self.result_table.setRowCount(0)
        self.result_note.setText("")
        if not split:
            self.sub_table.setRowCount(0)
            for combo in (self.mode_a, self.mode_b):
                combo.clear()
        self.veering.clear(reason)
        self.matrices.setHtml(f"<p>{reason}</p>")
        assert self.system is not None and self.full is not None, "cleared while showing a chain"
        self.shapes.set_model(None, self.full, [])
        self.frf.set_model(None, self.system, self.full)

    def _fill_oscillators(self, model: CoupledModel) -> None:
        self.osc_table.setRowCount(2)
        a, b = model.osc_a, model.osc_b
        res = f" + {model.m_residual:.3g}" if model.residual and model.m_residual > 1e-9 else ""
        for r, (name, mode, osc, m_text) in enumerate(
            (("A", model.mode_a, a, _g(a.m, 3) + res), ("B", model.mode_b, b, _g(b.m, 3)))
        ):
            _set_row(
                self.osc_table,
                r,
                [
                    (f"{name}{mode + 1}", sub_color(name)),
                    (m_text, None),
                    (_g(osc.k, 4), None),
                    (_g(osc.c, 3), None),
                    (_g(osc.fn_hz), None),
                    (zeta_text(osc.zeta) if osc.k > 0 else "—", None),
                ],
            )
        _fit_table(self.osc_table)

    @staticmethod
    def _summary(model: CoupledModel) -> str:
        mu, ratio = model.mass_ratio, model.freq_ratio
        f1, f2 = model.fn_hz
        fa, fb = model.osc_a.fn_hz, model.osc_b.fn_hz
        tuned = ""
        if np.isfinite(ratio) and ratio > 0:
            detune = abs(math.log(ratio))
            if detune < math.sqrt(mu):
                tuned = (
                    "f<sub>B</sub>/f<sub>A</sub> is within about √μ of 1, so the modes are "
                    "<b>strongly coupled</b>: each coupled mode is a mix of A's and B's, and they split "
                    "apart either side."
                )
            else:
                tuned = (
                    "The two modes are far apart compared with √μ, so they are <b>weakly coupled</b>: "
                    "each coupled mode is mostly A's or B's, shifted a little."
                )
        return (
            f"A (m1–m{model.split}) mode {model.mode_a + 1} at {fa:.4g} Hz and B "
            f"(m{model.split + 1}–m{model.system.n}) mode {model.mode_b + 1} at {fb:.4g} Hz, "
            f"f<sub>B</sub>/f<sub>A</sub> = <b>{ratio:.3g}</b>, mass ratio "
            f"μ = m<sub>b</sub>/m<sub>a</sub> = <b>{mu:.3g}</b>. Joined, they become <b>{f1:.4g}</b> and "
            f"<b>{f2:.4g} Hz</b>. {tuned}"
        )

    @staticmethod
    def _result_note(model: CoupledModel, full: ModalResult, comparisons: list[CoupledComparison]) -> str:
        if model.rayleigh_ritz:
            note = (
                "With the modal mass at the interface and B's residual mass, the 2-DOF model is a "
                "Rayleigh–Ritz model of the chain (A's mode shape, and B's on a fixed base), so its "
                "frequencies are upper bounds of the chain's lowest two"
            )
            if model.a_mass == "effective":
                note += " (A is one mass, so its effective mass and its modal mass at the interface are the same)"
            if [c.full_index for c in comparisons] == [1, 2]:
                note += ". "
            else:
                note += (
                    f" ({model.fn_hz[0]:.4g} ≥ {full.modes[0].fn_hz:.4g} Hz and {model.fn_hz[1]:.4g} ≥ "
                    f"{full.modes[1].fn_hz:.4g} Hz), not of the modes they are paired with here. "
                )
        else:
            note = (
                "Without "
                + ("B's residual mass" if not model.residual else "A's modal mass at the interface")
                + " the 2-DOF model is no longer a Rayleigh–Ritz model, so "
                "its errors can have either sign. "
            )
        if min(c.mac for c in comparisons) < 0.8:
            note += (
                "A MAC well below 1: the other modes of A or B, which the 2-DOF model leaves out, "
                "take part in that chain mode. "
            )
        return note + "The chain's other modes come from the modes of A and B that are not coupled here."
