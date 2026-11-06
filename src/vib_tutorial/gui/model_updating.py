"""Model updating page: a model of the chain, with errors in it, correlated against a modal test of the chain.

Left: the test (the virtual modal test, or the exact modes with noise; which masses have sensors),
the model's errors (factors on each mass, spring and damper), and errors to try. Centre: the MAC
of every test shape with every model shape, and the pairs it makes, with their frequency errors.
Right: the frequencies against each other, a pair's shapes, the test's auto-MAC, the COMAC, notes.
Follows the chain from the Simulation page, which is the structure tested.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.complex_modes import complexity
from ..core.modal import ModalResult, modal_analysis
from ..core.model import ChainSystem
from ..core.model_updating import (
    Correlation,
    Errors,
    Kind,
    Residuals,
    Source,
    Test,
    Update,
    correlate,
    frequency_sensitivity,
    mac,
    measure,
    pair,
    parameters,
    update,
)
from ..core.practice import Problem, Question, Requirement, at_most, number, pick
from .layouts import ColumnScroll, FlowLayout, form_layout, group, plot_height
from .lesson import TrialCombo
from .links import load_notes, page_link
from .modal_extraction import MacPlot, colored
from .panels import spin
from .style import MAX_DOF, colors
from .theming import add_legend, mute

RECOMPUTE_MS = 150  # the model's errors settle this long before it is correlated again

TIPS = {
    "source": "<p><b>Virtual modal test</b>: a hammer on m1, every mass's displacement measured with a little "
    "noise, H1 averaged, LSCF's stable poles and LSFD's shapes, as on the <i>Virtual modal test</i> page: "
    "its modes carry its errors, and the weakest may be missed. <b>Exact modes + noise</b>: every mode, "
    "with a little noise on its frequency, damping and shape.</p>",
    "sensors": "<p>The masses with a sensor. The test's shapes, and the model's, are compared there only.</p>",
    "again": "<p>Measure again: the same structure, other noise.</p>",
    "errors": "<p>The model's errors, as factors on the structure's values: 1.3 under k3 makes the model's "
    "spring 3 30 % stiffer than the structure's. In practice they are unknown; here they are set, so "
    "the correlation's verdict can be checked.</p>",
}
MAC_TIP = (
    "<p>MAC of each test shape (rows) with each model shape (columns), at the sensors: "
    "|ψ<sup>H</sup>φ|²/(ψ<sup>H</sup>ψ φ<sup>T</sup>φ). The pairs are outlined.</p>"
)
FREQ_TIP = (
    "<p>Each pair's model frequency against its test frequency: on the diagonal, they agree. A model "
    "uniformly too stiff or too heavy shifts every point the same way; a local error moves some modes more "
    "than others.</p>"
)
SHAPE_TIP = (
    "<p>A pair's shapes at the sensors: the test's (rotated to its best-fit line and scaled onto the "
    "model's) and the model's.</p>"
)
AUTO_TIP = (
    "<p>The test's shapes against each other. Distinct modes should be orthogonal-looking (small "
    "off-diagonal MAC); with too few or badly placed sensors different modes look alike: spatial "
    "aliasing.</p>"
)
COMAC_TIP = (
    "<p>COMAC per sensor: how well the paired shapes agree there, over all the pairs, "
    "(Σ|ψ<sub>i</sub>φ<sub>i</sub>|)²/(Σ|ψ<sub>i</sub>|² Σφ<sub>i</sub>²), each test shape scaled onto its "
    "model shape first. Low where the shapes disagree.</p>"
)

UPDATE_TIP = (
    "<p>Sensitivity-based updating: the chosen parameters are changed, by Gauss–Newton steps on their logarithms, "
    "until the model's paired frequencies (and, if ticked, its shapes at the sensors and its damping ratios) "
    "match the test's as closely as they can. Each step's slopes come from the model's sensitivities. "
    "Regularization λ adds λ²‖ln p‖²: it keeps the model near where it started, and settles what the data "
    "can't.</p>"
)
FACTOR_TIP = (
    "<p>Each parameter's model value over the structure's, step by step: 1 is the truth, which only "
    "this page can know. Springs solid, masses dashed, dampers dotted, in their mass's colour.</p>"
)
SENSITIVITY_TIP = (
    "<p>∂ln f<sub>r</sub>/∂ln p: how much each model mode's frequency changes for a change in each parameter. A "
    "spring's is half its share of the mode's strain energy, a mass's minus half its share of the kinetic energy. "
    "A parameter no mode is sensitive to can't be updated from frequencies.</p>"
)
DEFAULT_UPDATE = {"kinds": (Kind.SPRING,), "shapes": True, "damping": False, "weight": 0.1, "reg": 0.0}

# Errors to try: (name, {"k": {element: factor}, "m": ..., "c": ..., "all_k": factor, "sensors": [...],
# "source": Source}, note). Elements count from 0; -1 is the last.
TRIALS: tuple[tuple[str, dict[str, Any], str], ...] = (
    (
        "One spring 30 % too stiff",
        {"k": {2: 1.3}},
        "The model's k3 is 1.3 times the structure's. Its frequencies are all too high, by 2.2, 4.1, 0.6 and 6.6 % "
        "(the hammer test finds the first three; the table compares the model with that test, whose own errors "
        "shift these a little, so mode 3 reads +0.3 % there): mode 3's least, since k3 stretches little in it. The shapes "
        "barely change (MACs of 0.99 and up), so they are little help in finding it.",
    ),
    (
        "One mass 20 % too heavy",
        {"m": {1: 1.2}},
        "The model's m2 is 1.2 times the structure's: its frequencies are too low, by 1.8, 3.2, 0.5 and 3.5 %. A "
        "mass error and a spring error both just move the frequencies; which one it is, and where, takes the "
        "pattern of the errors over the modes.",
    ),
    (
        "Every spring 10 % too soft",
        {"all_k": 0.9},
        "Every frequency is √0.9 − 1 = −5.1 % off, and every shape exactly right: a uniform stiffness (or mass) "
        "error scales the frequencies and leaves the shapes alone, so the MACs see nothing.",
    ),
    (
        "Dampers twice too strong",
        {"all_c": 2.0},
        "The model's dampers are twice the structure's: its damping ratios are double the test's, and its "
        "frequencies and shapes don't move. Damping can only be checked against the measured ζ.",
    ),
    (
        "Two sensors",
        {"k": {2: 1.3}, "sensors": [0, -1], "source": Source.EXACT},
        "The same spring error, every mode measured, but on m1 and the last mass only. Each shape is now two "
        "numbers, and four modes can't all look different in two: modes 2 and 4 have an auto-MAC of 0.91 without noise "
        "(spatial aliasing), and only their frequencies tell them apart.",
    ),
    (
        "No error: the test's own noise",
        {"source": Source.HAMMER},
        "The model is exactly the structure. Every difference left is the test's: its frequencies within "
        "0.3 %, and the highest mode it finds (mode 3) at a MAC of only 0.92, its curve fit's error. The model "
        "can't be correlated more finely than that.",
    ),
    (
        "Update: find the stiff spring",
        {"k": {2: 1.3}, "source": Source.HAMMER},
        "The springs updated against the hammer test's three frequencies and shapes (Updating tab): k3 comes down "
        "from 1.30 to 1.02 in a few steps, and the other springs stay within 3.5 % of the truth. What is left is "
        "the test's: its shapes' errors pull the springs a little. Apply it to the model and correlate again.",
    ),
    (
        "Update: masses too, not unique",
        {"k": {2: 1.3}, "source": Source.EXACT, "update": {"kinds": (Kind.SPRING, Kind.MASS)}},
        "Springs and masses updated together against every mode: the fit is near perfect, but every factor comes "
        "out 2 to 4 % high instead of k3 alone coming down. Scaling every mass and spring together changes no "
        "frequency and no shape: one singular value is 0, and the data can't fix that direction. Pin it with a "
        "known mass (untick masses) or regularization.",
    ),
    (
        "Update: regularized",
        {"k": {2: 1.3}, "source": Source.HAMMER, "update": {"reg": 1.0}},
        "The same springs, regularization λ = 1: the fit is held near where it started, and k3 only comes down to "
        "1.29. Regularization trades the fit for the smallest change; it settles what the data can't, but here "
        "it keeps the model's error. Lower it and watch k3 come down.",
    ),
    (
        "Update: the dampers",
        {"all_c": 2.0, "source": Source.HAMMER, "update": {"kinds": (Kind.DAMPER,), "shapes": False, "damping": True}},
        "The model's dampers are twice the structure's; the dampers updated against the measured damping ratios. "
        "They come back to within 15 % of the truth: the hammer test's ζ are its least accurate numbers.",
    ),
)

# ------------------------------------------------------------------ practice problems
# Each question's numbers are printed round, and its answer is worked from them as printed.
TAB_FREQUENCIES, TAB_SHAPES, TAB_AUTO, TAB_COMAC, TAB_UPDATING, TAB_SENSITIVITY = range(6)


@dataclass(frozen=True)
class UpdatingSetup:
    """A practice problem's setup: the structure (a chain; none: the one there is), the test, the sensors (0-based;
    none: every mass), the model's error factors (none: all 1) and the tab to show."""

    masses: tuple[float, ...] = ()
    springs: tuple[float, ...] = ()
    dampers: tuple[float, ...] = ()
    source: Source = Source.EXACT
    sensors: tuple[int, ...] = ()
    mass_factors: tuple[float, ...] = ()
    spring_factors: tuple[float, ...] = ()
    damper_factors: tuple[float, ...] = ()
    tab: int = TAB_FREQUENCIES

    def system(self) -> ChainSystem | None:
        if not self.masses:
            return None
        return ChainSystem(np.array(self.masses), np.array(self.springs), np.array(self.dampers))

    def errors(self, n: int) -> Errors:
        def factors(given: tuple[float, ...]) -> np.ndarray:
            return np.array(given, dtype=float) if given else np.ones(n)

        return Errors(factors(self.mass_factors), factors(self.spring_factors), factors(self.damper_factors))


PRACTICE_SETUP = UpdatingSetup  # what a practice problem's setup and the page's practice_state are


def _uniform(n: int) -> dict:
    """The uniform chain of n masses, as a setup's structure."""
    s = ChainSystem.uniform(n)
    return dict(masses=tuple(s.masses), springs=tuple(s.stiffness), dampers=tuple(s.damping))


def _one_factor(n: int, element: int, factor: float) -> tuple[float, ...]:
    f = [1.0] * n
    f[element] = factor
    return tuple(f)


def _mac(rng: np.random.Generator) -> Question:
    while True:
        a, b = pick(rng, (0.5, 0.8, 1.2, 1.5)), pick(rng, (-1, -0.5, 0.5, 1, 1.5))
        c, d = pick(rng, (0.4, 0.8, 1, 1.5)), pick(rng, (-1.2, -0.6, 0.3, 0.9, 2))
        psi, phi = np.array([1.0, a, b]), np.array([1.0, c, d])
        value = float(mac(psi[:, None], phi[:, None])[0, 0])
        if 0.2 < value < 0.95:
            break
    dot, pp, ff = float(psi @ phi), float(psi @ psi), float(phi @ phi)
    return Question(
        f"A test shape ψ = (1, {a:g}, {b:g}) and a model shape φ = (1, {c:g}, {d:g}), at three sensors. What is their "
        "MAC?",
        value,
        f"MAC = (ψᵀφ)²/(ψᵀψ φᵀφ) = {number(dot)}²/({number(pp)} × {number(ff)}) = {number(value)}.",
        tolerance=0.01,
    )


def _uniform_error(rng: np.random.Generator) -> Question:
    springs = rng.integers(2) == 0
    factor = pick(rng, (0.8, 0.9, 1.1, 1.2, 1.44) if springs else (0.8, 0.9, 1.1, 1.25))
    ratio = math.sqrt(factor) if springs else 1 / math.sqrt(factor)
    what = "spring" if springs else "mass"
    way = "high" if ratio > 1 else "low"
    return Question(
        f"Every {what} of a model is {factor:g} times the structure's. By how many percent are all its frequencies "
        f"too {way}?",
        100 * abs(ratio - 1),
        f"f ∝ √(k/m), and scaling every {what} scales every mode's frequency alike: "
        f"{'√' if springs else '1/√'}{factor:g} = {number(ratio, 4)}, {number(100 * abs(ratio - 1))} % {way}. The shapes "
        "don't change at all.",
        unit="%",
        tolerance=0.01,
        setup=dict(
            **_uniform(4),
            spring_factors=(factor,) * 4 if springs else (),
            mass_factors=() if springs else (factor,) * 4,
        ),
    )


def _spring_share(rng: np.random.Generator) -> Question:
    share, change = pick(rng, (10, 20, 30, 40, 60)), pick(rng, (5, 10, 20))
    mode, spring = int(pick(rng, (1, 2, 3))), int(pick(rng, (1, 2, 3, 4)))
    rise = share / 100 / 2 * change
    return Question(
        f"Spring k{spring} holds {share:g} % of mode {mode}'s strain energy. Made {change:g} % stiffer, by how many "
        f"percent does mode {mode}'s frequency rise, to first order?",
        rise,
        f"∂ln f/∂ln k is half the spring's share of the strain energy: {share / 200:g}. So Δf/f ≈ {share / 200:g} × "
        f"{change:g} % = {number(rise)} %.",
        unit="%",
        tolerance=0.01,
    )


def _mass_share(rng: np.random.Generator) -> Question:
    share, change = pick(rng, (10, 20, 30, 50)), pick(rng, (5, 10, 20))
    mode, mass_ = int(pick(rng, (1, 2, 3))), int(pick(rng, (1, 2, 3, 4)))
    fall = share / 100 / 2 * change
    return Question(
        f"Mass m{mass_} carries {share:g} % of mode {mode}'s kinetic energy. Made {change:g} % heavier, by how many "
        f"percent does mode {mode}'s frequency fall, to first order?",
        fall,
        f"∂ln f/∂ln m is minus half the mass's share of the kinetic energy: −{share / 200:g}. So Δf/f ≈ "
        f"−{share / 200:g} × {change:g} % = −{number(fall)} %.",
        unit="%",
        tolerance=0.01,
    )


def _gauss_newton(rng: np.random.Generator) -> Question:
    f_test = pick(rng, (5, 8, 10, 12))
    off = pick(rng, (-6, -4, -3, 3, 4, 6))
    f_model = round(f_test * (1 + off / 100), 2)
    S = pick(rng, (0.2, 0.3, 0.4, 0.5))
    residual = math.log(f_test / f_model)
    step = residual / S
    r, d = (number(v).replace("-", "−") for v in (residual, step))
    factor = math.exp(step)
    return Question(
        f"A model's mode is at {f_model:g} Hz, the test's at {f_test:g} Hz, and ∂ln f/∂ln k = {S:g} for the one spring "
        "being updated. By what factor does one Gauss–Newton step on ln k multiply that spring?",
        factor,
        f"The residual is ln(f<sub>test</sub>/f<sub>model</sub>) = ln({f_test:g}/{f_model:g}) = {r}; the step is that "
        f"over the slope, Δln k = {r}/{S:g} = {d}, so k is multiplied by e<sup>{d}</sup> = {number(factor, 4)}.",
        tolerance=0.005,
    )


def _pairing(rng: np.random.Generator) -> Question:
    """A 3 × 3 MAC table where one test mode's best match is taken by a better pair first."""
    perm = rng.permutation(3)
    M = rng.uniform(0.02, 0.3, (3, 3))
    for i in range(3):
        M[i, perm[i]] = rng.uniform(0.86, 0.99)
    t = int(rng.integers(3))  # the test mode asked about
    other = (t + 1 + int(rng.integers(2))) % 3  # a test mode that takes t's second choice first
    M[t, perm[other]] = M[other, perm[other]] - 0.03  # t matches it well too, just not as well
    M[t, perm[t]] = min(M[t, perm[t]], M[t, perm[other]] - 0.05)  # and its own a little less: the greedy trap
    M = np.round(M, 2)
    pairs = pair(M, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    found = {p.test: p.model for p in pairs}
    answer = found.get(t, 3)
    rows = "; ".join(f"test {i + 1}: " + ", ".join(f"{v:.2f}" for v in M[i]) for i in range(3))
    return Question(
        f"A MAC table, each test mode's row against model modes 1, 2 and 3: {rows}. Pairing best MAC first (none "
        f"under 0.5), which model mode does test mode {t + 1} pair with?",
        answer,
        f"The best MAC in the table, {M.max():.2f}, pairs first, then the best left, and so on. Test mode {t + 1}'s "
        f"highest MAC is with model mode {perm[other] + 1}, but test mode {other + 1} matches that one better and "
        f"takes it; test mode {t + 1} goes with "
        + (f"model mode {answer + 1}." if answer < 3 else "none: what's left is under 0.5."),
        choices=("Model mode 1", "Model mode 2", "Model mode 3", "None"),
    )


def _uniform_shapes(rng: np.random.Generator) -> Question:
    return Question(
        "Every spring of the model is 10 % too soft. What do the MACs between the test's shapes and the model's say?",
        0,
        "Nothing: they stay at 1 (the test's noise aside), since scaling every spring scales every frequency by "
        "√0.9 and leaves every shape exactly as it was. Only the frequencies, all 5.1 % low (the test's noise aside), "
        "show the error.",
        choices=(
            "Nothing: they stay at about 1",
            "They drop most for the highest modes",
            "They drop evenly to about 0.9",
        ),
        setup=dict(**_uniform(4), spring_factors=(0.9,) * 4),
        predict=True,
    )


def _dampers(rng: np.random.Generator) -> Question:
    return Question(
        "The model's dampers are twice the structure's. What happens to its frequencies against the test's?",
        0,
        "They don't move (the paired frequencies agree to the test's noise): the undamped frequencies don't depend on "
        "the dampers, and light damping hardly moves the damped ones. Only the damping ratios, doubled, show it.",
        choices=("They don't move", "They all fall", "They all rise", "The higher modes' fall most"),
        setup=dict(**_uniform(4), damper_factors=(2.0,) * 4),
        predict=True,
    )


def _two_sensors(rng: np.random.Generator) -> Question:
    return Question(
        "Four modes measured with sensors on m1 and m4 only. What does the test's auto-MAC show?",
        0,
        "Some modes look alike: two numbers per shape can't keep four shapes apart, and modes 2 and 4 have an "
        "auto-MAC of 0.91 (spatial aliasing). Only their frequencies tell them apart.",
        choices=(
            "Some modes look alike: high off-diagonal terms",
            "Every mode still looks distinct",
            "Every off-diagonal term is exactly 0",
        ),
        setup=dict(**_uniform(4), sensors=(0, 3), tab=TAB_AUTO),
        predict=True,
    )


def _least_moved(rng: np.random.Generator) -> Question:
    return Question(
        "The model's k3 is 30 % too stiff, on the uniform four-mass chain. Which mode's frequency is least affected?",
        2,
        "Mode 3, only 0.6 % high (the others 2.2, 4.1 and 6.6 %; the test's noise moves these a little): a spring "
        "moves a mode's frequency by half its share of the strain energy, and k3 hardly stretches in mode 3.",
        choices=("Mode 1", "Mode 2", "Mode 3", "Mode 4"),
        setup=dict(**_uniform(4), spring_factors=_one_factor(4, 2, 1.3), tab=TAB_SENSITIVITY),
        predict=True,
    )


LOOK_ALIKE = 0.1  # the spec's largest off-diagonal auto-MAC


def _auto_mac(system: ChainSystem, sensors: tuple[int, ...], modes: int) -> float:
    """The largest off-diagonal MAC among the structure's first `modes` undamped shapes, at the sensors."""
    result = modal_analysis(ChainSystem(system.masses, system.stiffness, system.damping))
    shapes = np.column_stack([m.shape for m in result.modes[:modes]])[list(sensors)]
    if not len(sensors) or shapes.shape[1] < modes:
        return 1.0
    M = mac(shapes, shapes)
    return float(np.max(M[~np.eye(modes, dtype=bool)]))


def _sensor_spec(rng: np.random.Generator) -> Question:
    count = int(pick(rng, (3, 4)))
    n = int(pick(rng, (5, 6, 7, 8) if count == 3 else (6, 7, 8)))
    chain = _uniform(n)
    system = ChainSystem.uniform(n)
    best = min(itertools.combinations(range(n), count), key=lambda sub: _auto_mac(system, sub, count))

    def goal(t: UpdatingSetup):
        s = t.system()
        uniform = (
            s is not None
            and s.n == n
            and all(
                np.allclose(a, b)
                for a, b in ((s.masses, system.masses), (s.stiffness, system.stiffness), (s.damping, system.damping))
            )
        )
        sensors = t.sensors or tuple(range(len(t.masses)))
        worst = _auto_mac(s, sensors, count) if s is not None else 1.0
        return (
            Requirement(f"The uniform chain of {n} masses", f"{len(t.masses)} masses", uniform),
            Requirement(f"No more than {count} sensors", f"{len(sensors)}", len(sensors) <= count),
            at_most(f"The largest auto-MAC between modes 1 to {count}", worst, LOOK_ALIKE),
        )

    names = ", ".join(f"m{i + 1}" for i in best)
    return Question(
        f"The uniform chain of {n} masses. Place no more than {count} sensors so that its first {count} modes all look "
        f"distinct: no off-diagonal auto-MAC between them above {LOOK_ALIKE:g} (noise-free; the page's Auto-MAC tab "
        "shows the test's, with its noise).",
        0,
        f"Bunched together, sensors see every low mode as nearly the same smooth curve. Spread out, and off the "
        f"modes' nodes, they keep them apart: {names} gives {number(_auto_mac(system, best, count))}.",
        setup=dict(**chain, sensors=tuple(range(count)), tab=TAB_AUTO),
        goal=goal,
        design=dict(**chain, sensors=best, tab=TAB_AUTO),
    )


PROBLEMS = tuple(
    Problem(key, title, "updating_page", f"See {page_link('updating_page', anchor, section)}.", make)
    for key, title, anchor, section, make in (
        ("mac", "A MAC", "mac", "The modal assurance criterion", _mac),
        ("uniform-error", "A uniform error", "shapes-blind", "What the shapes don't see", _uniform_error),
        ("spring-share", "A spring's sensitivity", "sensitivities", "Updating by sensitivities", _spring_share),
        ("mass-share", "A mass's sensitivity", "sensitivities", "Updating by sensitivities", _mass_share),
        ("gauss-newton", "A Gauss–Newton step", "sensitivities", "Updating by sensitivities", _gauss_newton),
        ("pairing", "Pairing by MAC", "mac", "The modal assurance criterion", _pairing),
        (
            "uniform-shapes",
            "Predict: a uniform error's MACs",
            "shapes-blind",
            "What the shapes don't see",
            _uniform_shapes,
        ),
        ("dampers", "Predict: dampers too strong", "shapes-blind", "What the shapes don't see", _dampers),
        ("two-sensors", "Predict: two sensors", "sensors", "Sensors", _two_sensors),
        ("least-moved", "Predict: the mode least moved", "sensitivities", "Updating by sensitivities", _least_moved),
        ("sensor-spec", "Spec: placing sensors", "sensors", "Sensors", _sensor_spec),
    )
)

NOTES_HTML = load_notes("model_updating")


def _plot(left: str, bottom: str) -> pg.PlotWidget:
    p = pg.PlotWidget()
    p.setMenuEnabled(False)
    p.hideButtons()
    p.showGrid(x=True, y=True, alpha=0.15)
    p.setLabel("left", left)
    p.setLabel("bottom", bottom)
    p.getAxis("left").enableAutoSIPrefix(False)
    p.getAxis("bottom").enableAutoSIPrefix(False)
    p.setMouseEnabled(x=False, y=False)
    return p


def _cell(text: str, colour: str | None = None) -> QtWidgets.QTableWidgetItem:
    item = QtWidgets.QTableWidgetItem(text)
    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
    if colour:
        item.setForeground(pg.mkColor(colour))
    return item


class ModelUpdatingPage(QtWidgets.QWidget):
    """The test and the model's errors on the left, the MAC and the pairs in the centre, the rest on the right."""

    dof_requested = QtCore.Signal(int)  # the user changed N here; the main window owns the system
    system_requested = QtCore.Signal(object)  # a practice problem's structure, for the Simulation page
    edit_parameters = QtCore.Signal()  # the user wants the Simulation page's parameter panel

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.system: ChainSystem | None = None
        self.result: ModalResult | None = None
        self.errors: Errors | None = None
        self.test: Test | None = None
        self.corr: Correlation | None = None
        self.seed = 1
        self._remeasure = True
        self._dirty = False
        self._loading = False
        self._recompute = QtCore.QTimer(self)
        self._recompute.setSingleShot(True)
        self._recompute.setInterval(RECOMPUTE_MS)
        self._recompute.timeout.connect(self.recompute)

        # --- left: the structure, the test, the model's errors, errors to try
        box = QtWidgets.QGroupBox("Test")
        form = form_layout(box)
        self.dof = QtWidgets.QSpinBox()
        self.dof.setRange(1, MAX_DOF)
        self.dof.setToolTip(
            "Same chain as the Simulation page, the structure tested; changing it here changes it there too."
        )
        self.dof.valueChanged.connect(self.dof_requested)
        form.addRow("Number of masses:", self.dof)
        self.source = QtWidgets.QComboBox()
        for s in Source:
            self.source.addItem(s.value, s)
        self.source.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.source.setToolTip(TIPS["source"])
        form.addRow("Modes from:", self.source)
        self.sensor_row = QtWidgets.QWidget()
        self.sensor_layout = FlowLayout(self.sensor_row)
        self.sensor_layout.setContentsMargins(0, 0, 0, 0)
        self.sensor_row.setToolTip(TIPS["sensors"])
        self.sensors: list[QtWidgets.QCheckBox] = []
        label = QtWidgets.QLabel("Sensors:")
        label.setToolTip(TIPS["sensors"])
        form.addRow(label, self.sensor_row)
        self.again = QtWidgets.QPushButton("Measure again")
        self.again.setToolTip(TIPS["again"])
        form.addRow(self.again)
        self.test_info = QtWidgets.QLabel()
        self.test_info.setWordWrap(True)
        mute(self.test_info)
        form.addRow(self.test_info)
        link = QtWidgets.QLabel("The structure is the chain on the <a href='#sim'>Simulation page</a>.")
        mute(link)
        link.setWordWrap(True)
        link.linkActivated.connect(lambda _: self.edit_parameters.emit())
        form.addRow(link)

        model_box = QtWidgets.QGroupBox("The model's errors")
        mv = QtWidgets.QVBoxLayout(model_box)
        self.error_grid = QtWidgets.QGridLayout()
        self.error_grid.setHorizontalSpacing(4)
        for col, head in enumerate(("", "× mass", "× spring", "× damper")):
            h = QtWidgets.QLabel(head)
            h.setToolTip(TIPS["errors"])
            self.error_grid.addWidget(h, 0, col)
        self.error_rows: list[
            tuple[QtWidgets.QLabel, QtWidgets.QDoubleSpinBox, QtWidgets.QDoubleSpinBox, QtWidgets.QDoubleSpinBox]
        ] = []
        mv.addLayout(self.error_grid)
        self.clear_errors = QtWidgets.QPushButton("No errors")
        self.clear_errors.setToolTip("<p>Set every factor to 1: the model is then the structure.</p>")
        mv.addWidget(self.clear_errors)

        trials = QtWidgets.QGroupBox("Things to try")
        tv = QtWidgets.QVBoxLayout(trials)
        self.trial = TrialCombo([name for name, _, _ in TRIALS], prompt=None)
        self.trial.setToolTip(
            "<p>Model errors (and tests) that show something about correlation, set on the chain as it is now.</p>"
        )
        tv.addWidget(self.trial)
        self.trial_note = QtWidgets.QLabel()
        self.trial_note.setWordWrap(True)
        mute(self.trial_note)
        tv.addWidget(self.trial_note)

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        for b in (box, trials, model_box):
            lv.addWidget(b)
        lv.addStretch(1)
        left_scroll = ColumnScroll(left)

        # --- centre: the MAC matrix and the pairs
        self.mac_plot = MacPlot()
        self.mac_plot.setLabel("bottom", "Model mode")
        self.mac_plot.setLabel("left", "Test mode")
        self.mac_plot.setTitle("MAC: test against model shapes", size="9pt")
        self.mac_plot.setToolTip(MAC_TIP)
        plot_height(self.mac_plot, 220)
        self.pair_boxes: list[pg.PlotDataItem] = []
        self.table = QtWidgets.QTableWidget()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.setToolTip(
            "<p>Each test mode, the model mode it pairs with, their frequencies and damping ratios "
            "and the MAC between their shapes. Select one to see its shapes.</p>"
        )
        self.readout = QtWidgets.QLabel()
        self.readout.setWordWrap(True)
        self.readout.setTextFormat(QtCore.Qt.TextFormat.RichText)
        center = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(center)
        cv.addWidget(self.mac_plot, 3)
        cv.addWidget(self.table, 2)
        cv.addWidget(self.readout)

        # --- right: frequencies, shapes, auto-MAC, COMAC, notes
        self.freq_plot = _plot("Model frequency (Hz)", "Test frequency (Hz)")
        self.freq_plot.setToolTip(FREQ_TIP)
        self.error_plot = _plot("Frequency error (%)", "Test mode")
        self.error_plot.setToolTip(FREQ_TIP)
        plot_height(self.freq_plot, 180)
        plot_height(self.error_plot, 130)
        freq_tab = QtWidgets.QWidget()
        fv = QtWidgets.QVBoxLayout(freq_tab)
        fv.addWidget(self.freq_plot, 3)
        fv.addWidget(self.error_plot, 2)

        self.shape_plot = _plot("Shape", "Mass")
        self.shape_plot.setToolTip(SHAPE_TIP)
        self.shape_legend = add_legend(self.shape_plot.getPlotItem(), offset=(-10, 10))
        plot_height(self.shape_plot, 200)
        self.shape_note = QtWidgets.QLabel()
        self.shape_note.setWordWrap(True)
        mute(self.shape_note)
        shape_tab = QtWidgets.QWidget()
        sv = QtWidgets.QVBoxLayout(shape_tab)
        sv.addWidget(self.shape_plot, 1)
        sv.addWidget(self.shape_note)

        self.auto_plot = MacPlot()
        self.auto_plot.setLabel("bottom", "Test mode")
        self.auto_plot.setLabel("left", "Test mode")
        self.auto_plot.setTitle("Auto-MAC: the test's shapes against each other", size="9pt")
        self.auto_plot.setToolTip(AUTO_TIP)
        plot_height(self.auto_plot, 200)
        self.auto_note = QtWidgets.QLabel()
        self.auto_note.setWordWrap(True)
        self.auto_note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        auto_tab = QtWidgets.QWidget()
        av = QtWidgets.QVBoxLayout(auto_tab)
        av.addWidget(self.auto_plot, 1)
        av.addWidget(self.auto_note)

        self.comac_plot = _plot("COMAC", "Sensor")
        self.comac_plot.setToolTip(COMAC_TIP)
        plot_height(self.comac_plot, 200)
        self.comac_note = QtWidgets.QLabel()
        self.comac_note.setWordWrap(True)
        self.comac_note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        comac_tab = QtWidgets.QWidget()
        ov = QtWidgets.QVBoxLayout(comac_tab)
        ov.addWidget(self.comac_plot, 1)
        ov.addWidget(self.comac_note)

        self.update_kind = {
            k: QtWidgets.QCheckBox(name)
            for k, name in ((Kind.SPRING, "springs"), (Kind.MASS, "masses"), (Kind.DAMPER, "dampers"))
        }
        self.fit_shapes = QtWidgets.QCheckBox("shapes, weight")
        self.shape_weight = spin(0.0, 10.0, 0.1, 2)
        self.fit_damping = QtWidgets.QCheckBox("damping ratios")
        self.regularization = spin(0.0, 100.0, 0.0, 3)
        for w in (
            *self.update_kind.values(),
            self.fit_shapes,
            self.shape_weight,
            self.fit_damping,
            self.regularization,
        ):
            w.setToolTip(UPDATE_TIP)
        self.apply_update = QtWidgets.QPushButton("Apply to the model")
        self.apply_update.setToolTip("<p>Set the model's errors to the updated ones, and correlate again.</p>")
        self.factor_plot = _plot("Model / structure", "Iteration")
        self.factor_plot.setToolTip(FACTOR_TIP)
        self.factor_legend = add_legend(self.factor_plot.getPlotItem(), offset=(-10, 10), colCount=2)
        self.cost_plot = _plot("‖residual‖", "Iteration")
        self.cost_plot.setLogMode(x=False, y=True)
        self.cost_plot.setToolTip(UPDATE_TIP)
        plot_height(self.factor_plot, 170)
        plot_height(self.cost_plot, 90)
        self.update_note = QtWidgets.QLabel()
        self.update_note.setWordWrap(True)
        self.update_note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        update_tab = QtWidgets.QWidget()
        uv = QtWidgets.QVBoxLayout(update_tab)
        row = FlowLayout()
        row.addWidget(group("Update:", *self.update_kind.values()))
        row.addWidget(group("Fit frequencies,", self.fit_shapes, self.shape_weight, self.fit_damping))
        row.addWidget(group("Regularization λ:", self.regularization))
        row.addWidget(self.apply_update)
        uv.addLayout(row)
        uv.addWidget(self.factor_plot, 3)
        uv.addWidget(self.cost_plot, 2)
        uv.addWidget(self.update_note)
        update_scroll = ColumnScroll(update_tab)

        self.sens_plot = MacPlot()
        self.sens_plot.setLabel("bottom", "Parameter")
        self.sens_plot.setLabel("left", "Model mode")
        self.sens_plot.setTitle("∂ln f / ∂ln p", size="9pt")
        self.sens_plot.setToolTip(SENSITIVITY_TIP)
        plot_height(self.sens_plot, 200)
        self.sens_note = QtWidgets.QLabel(
            "Colour: the size; the number: the value. A spring stiffens every mode it stretches in; a mass slows "
            "every mode it moves in."
        )
        self.sens_note.setWordWrap(True)
        mute(self.sens_note)
        sens_tab = QtWidgets.QWidget()
        nv = QtWidgets.QVBoxLayout(sens_tab)
        nv.addWidget(self.sens_plot, 1)
        nv.addWidget(self.sens_note)
        self.update_result: Update | None = None

        self.notes = QtWidgets.QTextBrowser()
        self.notes.setHtml(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.freq_tab, self.shape_tab, self.auto_tab, self.comac_tab = freq_tab, shape_tab, auto_tab, comac_tab
        self.tabs.addTab(freq_tab, "Frequencies")
        self.tabs.addTab(shape_tab, "Shapes")
        self.tabs.addTab(auto_tab, "Auto-MAC")
        self.tabs.addTab(comac_tab, "COMAC")
        self.update_tab, self.sens_tab = update_scroll, sens_tab
        self.tabs.addTab(update_scroll, "Updating")
        self.tabs.addTab(sens_tab, "Sensitivity")
        self.tabs.addTab(self.notes, "Theory")

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(center)
        splitter.addWidget(self.tabs)
        splitter.setSizes([360, 700, 640])
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        # --- wiring
        self.source.currentIndexChanged.connect(lambda _: self._schedule(measure=True))
        self.again.clicked.connect(self._on_again)
        self.clear_errors.clicked.connect(self._on_clear)
        self.trial.picked.connect(lambda _: self.apply_trial())
        self.table.itemSelectionChanged.connect(self._draw_shapes)
        for w in (*self.update_kind.values(), self.fit_shapes, self.fit_damping):
            w.toggled.connect(lambda _: self._on_update_settings())
        for w in (self.shape_weight, self.regularization):
            w.valueChanged.connect(lambda _: self._on_update_settings())
        self.apply_update.clicked.connect(self._on_apply_update)
        self.tabs.currentChanged.connect(lambda _: self._draw_update())
        self._set_update_settings(DEFAULT_UPDATE)

    # ------------------------------------------------------------- inputs
    def set_system(self, system: ChainSystem, result: ModalResult | None = None) -> None:
        """The structure changed: measured again (now if shown); the model's errors stay as they are."""
        n_changed = self.system is None or system.n != self.system.n
        self.system = system
        self.result = (
            modal_analysis(ChainSystem(system.masses, system.stiffness, system.damping))
            if result is None or system.friction.any()
            else result
        )
        self._loading = True
        self.dof.blockSignals(True)
        self.dof.setValue(system.n)
        self.dof.blockSignals(False)
        if n_changed:
            self._build_rows(system.n)
        self._loading = False
        if self.errors is None:
            self.apply_trial()
        else:
            self._schedule(measure=True)

    def _build_rows(self, n: int) -> None:
        kept = [box.isChecked() for box in self.sensors]
        for box in self.sensors:
            self.sensor_layout.removeWidget(box)
            box.deleteLater()
        self.sensors = []
        for i in range(n):
            box = QtWidgets.QCheckBox(f"m{i + 1}")
            box.setChecked(kept[i] if i < len(kept) else True)
            box.setToolTip(TIPS["sensors"])
            box.toggled.connect(lambda _: self._schedule(measure=False))
            self.sensor_layout.addWidget(box)
            self.sensors.append(box)
        values = [[s.value() for s in row[1:]] for row in self.error_rows]
        while self.error_rows:
            for w in self.error_rows.pop():
                self.error_grid.removeWidget(w)
                w.deleteLater()
        for i in range(n):
            label = QtWidgets.QLabel(f"{i + 1}")
            m, k, c = (spin(0.05, 20.0, values[i][col] if i < len(values) else 1.0, 3) for col in range(3))
            for factor in (m, k, c):
                factor.setToolTip(TIPS["errors"])
                factor.valueChanged.connect(self._on_errors)
            for col, cell in enumerate((label, m, k, c)):
                self.error_grid.addWidget(cell, i + 1, col)
            self.error_rows.append((label, m, k, c))
        self.errors = self._read_errors() if self.errors is not None else None

    def _read_errors(self) -> Errors:
        cols = np.array([[box.value() for box in row[1:]] for row in self.error_rows]).reshape(-1, 3)
        return Errors(cols[:, 0], cols[:, 1], cols[:, 2])

    def _set_errors(self, errors: Errors) -> None:
        self._loading = True
        for row, m, k, c in zip(self.error_rows, errors.mass, errors.stiffness, errors.damping):
            for box, v in zip(row[1:], (m, k, c)):
                box.setValue(float(v))
        self._loading = False
        self.errors = errors

    def apply_trial(self) -> None:
        """Set the errors (and the test) the chosen trial asks for, on the chain as it is, measured with the first
        seed: the notes quote that measurement."""
        if self.system is None:
            return
        _, changes, note = TRIALS[self.trial.currentIndex()]
        n = self.system.n
        e = Errors.none(n)
        for key, arr in (("m", e.mass), ("k", e.stiffness), ("c", e.damping)):
            for i, f in changes.get(key, {}).items():
                arr[min(i % n, n - 1)] = f
        e.stiffness[:] *= changes.get("all_k", 1.0)
        e.damping[:] *= changes.get("all_c", 1.0)
        self._set_errors(e)
        want = {i % n for i in changes.get("sensors", range(n))}
        self._loading = True
        for i, box in enumerate(self.sensors):
            box.setChecked(i in want)
        if "source" in changes:
            self.source.setCurrentIndex(self.source.findData(changes["source"]))
        self._set_update_settings({**DEFAULT_UPDATE, **changes.get("update", {})})
        self._loading = False
        self.trial_note.setText(note)
        again = self.test is None or self.test.source is not self.source.currentData() or self.seed != 1
        self.seed = 1  # the notes' numbers are this measurement's
        self._schedule(measure=again)

    def load_setup(self, setup: dict) -> None:
        """Set the page up as a practice problem's `setup` says: its structure (through the Simulation page), then
        the test, the sensors and the model's errors, measured with the first seed."""
        t = UpdatingSetup(**setup)
        system = t.system()
        if system is not None:
            self.system_requested.emit(system)
            if self.system is None or not _same_chain(self.system, system):
                self.set_system(system)  # nobody passed it back: show it here alone
        if self.system is None:
            return
        n = self.system.n
        self._set_errors(t.errors(n))
        want = set(t.sensors) if t.sensors else set(range(n))
        self._loading = True
        for i, box in enumerate(self.sensors):
            box.setChecked(i in want)
        self.source.setCurrentIndex(self.source.findData(t.source))
        self._set_update_settings(DEFAULT_UPDATE)
        self._loading = False
        self.trial_note.setText("")
        self.tabs.setCurrentIndex(t.tab)
        self.seed = 1
        self._schedule(measure=True)

    def practice_state(self) -> UpdatingSetup:
        """The structure, the test and the model's errors, as a setup: what a practice spec checks."""
        s, e = self.system, self.errors
        return UpdatingSetup(
            tuple(float(v) for v in s.masses) if s is not None else (),
            tuple(float(v) for v in s.stiffness) if s is not None else (),
            tuple(float(v) for v in s.damping) if s is not None else (),
            self.source.currentData(),
            tuple(i for i, box in enumerate(self.sensors) if box.isChecked()),
            tuple(float(v) for v in e.mass) if e is not None else (),
            tuple(float(v) for v in e.stiffness) if e is not None else (),
            tuple(float(v) for v in e.damping) if e is not None else (),
            self.tabs.currentIndex(),
        )

    def _set_update_settings(self, settings: dict) -> None:
        loading, self._loading = self._loading, True
        for k, box in self.update_kind.items():
            box.setChecked(k in settings["kinds"])
        self.fit_shapes.setChecked(settings["shapes"])
        self.fit_damping.setChecked(settings["damping"])
        self.shape_weight.setValue(settings["weight"])
        self.regularization.setValue(settings["reg"])
        self._loading = loading

    def _on_update_settings(self) -> None:
        self.shape_weight.setEnabled(self.fit_shapes.isChecked())
        if not self._loading:
            self.update_result = None
            self._draw_update()

    def _on_apply_update(self) -> None:
        if self.update_result is not None:
            self._set_errors(self.update_result.errors)
            self._schedule(measure=False, now=True)

    def _on_errors(self, *_args) -> None:
        if not self._loading:
            self.errors = self._read_errors()
            self._schedule(measure=False)

    def _on_clear(self) -> None:
        if self.system is not None:
            self._set_errors(Errors.none(self.system.n))
            self._schedule(measure=False)

    def _on_again(self) -> None:
        self.seed += 1
        self._schedule(measure=True, now=True)

    def sensor_mask(self) -> np.ndarray:
        return np.array([box.isChecked() for box in self.sensors], dtype=bool)

    def _schedule(self, measure: bool, now: bool = False) -> None:
        if self._loading:
            self._remeasure |= measure
            return
        self._remeasure |= measure
        if not self.isVisible():
            self._dirty = True
            return
        if now:
            self.recompute()
        else:
            self._recompute.start()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._dirty:
            self.recompute()

    # ------------------------------------------------------------- computing
    def recompute(self) -> None:
        """Measure again if asked (or the structure changed), then correlate and draw."""
        self._recompute.stop()
        self._dirty = False
        if self.system is None or self.errors is None:
            return
        mask = self.sensor_mask()
        if mask.sum() == 0:
            self.readout.setText("<b>No sensors</b>: tick at least one mass.")
            return
        if self._remeasure or self.test is None:
            self.test = measure(self.system, self.source.currentData(), mask, self.seed, self.result)
            self._remeasure = False
        self.corr = correlate(self.system, self.errors, self.test.with_sensors(mask))
        self.update_result = None
        self.redraw()

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        for p in (self.mac_plot, self.auto_plot):
            p.apply_theme()
        if self.corr is not None and not self._dirty:  # a new chain while hidden: redrawn when it shows
            self.redraw()

    def redraw(self) -> None:
        assert self.corr is not None, "drawn after recompute"
        c = self.corr
        test = c.test
        n_model = len(c.result.modes)
        found = len(test.modes)
        self.test_info.setText(
            f"The test found {found} of the structure's {n_model} modes, measured at {int(test.sensors.sum())} "
            f"sensor{'s' if test.sensors.sum() != 1 else ''}."
        )
        rows = [f"T{i + 1}" for i in range(found)]
        cols = [f"M{j + 1}" for j in range(n_model)]
        self.mac_plot.set_matrix(c.macs, rows, cols)
        for item in self.pair_boxes:
            self.mac_plot.removeItem(item)
        self.pair_boxes = []
        for p in c.pairs:
            x, y = p.model, p.test
            box = pg.PlotDataItem([x, x + 1, x + 1, x, x], [y, y, y + 1, y + 1, y], pen=pg.mkPen(colors.force, width=2))
            self.mac_plot.addItem(box)
            self.pair_boxes.append(box)
        self._fill_table()
        self._draw_frequencies()
        self._draw_shapes()
        self._draw_auto()
        self._draw_comac()
        self._draw_update()
        if c.pairs:
            worst = max(c.pairs, key=lambda p: abs(p.error))
            low = min(c.pairs, key=lambda p: p.mac)
            unpaired = found - len(c.pairs)
            self.readout.setText(
                f"{len(c.pairs)} pairs. Largest frequency error: <b>{100 * worst.error:+.2f} %</b> (test mode "
                f"{worst.test + 1}, model mode {worst.model + 1}); lowest MAC <b>{low.mac:.3f}</b> (test mode "
                f"{low.test + 1})."
                + (
                    f" {unpaired} test mode{'s' if unpaired > 1 else ''} unpaired (MAC under "
                    "0.5 with every model mode)."
                    if unpaired
                    else ""
                )
            )
        else:
            self.readout.setText("No pairs: no test shape matches a model shape with a MAC of 0.5 or more.")

    def _fill_table(self) -> None:
        assert self.corr is not None, "drawn after recompute"
        c = self.corr
        t = self.table
        current = t.currentRow()
        t.blockSignals(True)
        t.clear()
        heads = ["Test", "f test (Hz)", "ζ test", "Model", "f model (Hz)", "ζ model", "Δf", "MAC"]
        t.setColumnCount(len(heads))
        t.setHorizontalHeaderLabels(heads)
        t.setRowCount(len(c.test.modes))
        by_test = {p.test: p for p in c.pairs}
        for i, m in enumerate(c.test.modes):
            cells = [_cell(f"T{i + 1}"), _cell(f"{m.fn_hz:.4g}"), _cell(f"{m.zeta:.3g}")]
            p = by_test.get(i)
            if p is None:
                cells += [_cell("unpaired", colors.poor)] + [_cell("") for _ in range(4)]
            else:
                mode = c.result.modes[p.model]
                zeta = mode.damped.zeta if mode.damped is not None else float("nan")
                text, colour = colored(100 * p.error, 1.0, 3.0, f"{100 * p.error:+.2f} %")
                mtext, mcolour = colored(1 - p.mac, 0.02, 0.1, f"{p.mac:.3f}")
                cells += [
                    _cell(f"M{p.model + 1}", colors.mode[p.model % len(colors.mode)]),
                    _cell(f"{p.f_model:.4g}"),
                    _cell(f"{zeta:.3g}"),
                    _cell(text, colour),
                    _cell(mtext, mcolour),
                ]
            for j, item in enumerate(cells):
                t.setItem(i, j, item)
        t.resizeColumnsToContents()
        t.selectRow(current if 0 <= current < t.rowCount() else 0)
        t.blockSignals(False)

    def _draw_frequencies(self) -> None:
        assert self.corr is not None, "drawn after recompute"
        c = self.corr
        p, q = self.freq_plot, self.error_plot
        p.clear()
        q.clear()
        if not c.pairs:
            return
        ft = np.array([pr.f_test for pr in c.pairs])
        fm = np.array([pr.f_model for pr in c.pairs])
        top = 1.1 * max(ft.max(), fm.max())
        p.plot([0, top], [0, top], pen=pg.mkPen(colors.faint, width=1))
        brushes = [pg.mkBrush(colors.mode[pr.model % len(colors.mode)]) for pr in c.pairs]
        p.addItem(pg.ScatterPlotItem(ft, fm, size=10, pen=None, brush=brushes))
        p.setRange(xRange=(0, top), yRange=(0, top), padding=0)
        err = np.array([100 * pr.error for pr in c.pairs])
        x = np.array([pr.test + 1 for pr in c.pairs], dtype=float)
        q.addItem(pg.BarGraphItem(x=x, height=err, width=0.6, brushes=brushes, pen=None))
        q.addItem(pg.InfiniteLine(0.0, angle=0, pen=pg.mkPen(colors.faint, width=1)))
        span = max(1.0, 1.2 * float(np.abs(err).max()))
        q.setRange(xRange=(0.4, len(c.test.modes) + 0.6), yRange=(-span, span), padding=0)
        q.getAxis("bottom").setTicks([[(i + 1, f"T{i + 1}") for i in range(len(c.test.modes))]])

    def _draw_shapes(self) -> None:
        assert self.system is not None, "drawn after recompute"
        c = self.corr
        if c is None:
            return
        p = self.shape_plot
        p.clear()
        self.shape_legend.clear()
        row = self.table.currentRow()
        pr = next((x for x in c.pairs if x.test == row), None)
        sensors = np.flatnonzero(c.test.sensors)
        if pr is None:
            self.shape_note.setText("Select a paired test mode in the table.")
            return
        psi = c.test.modes[pr.test].shape[sensors]
        phi = c.result.modes[pr.model].shape[sensors]
        psi = psi * np.exp(-1j * complexity(psi).angle)  # onto its best-fit line
        real = psi.real
        scale = float(real @ phi / (real @ real)) if real @ real > 0 else 1.0
        n = self.system.n
        full_phi = np.concatenate([[0.0], c.result.modes[pr.model].shape])
        colour = colors.mode[pr.model % len(colors.mode)]
        model = p.plot(np.arange(n + 1), full_phi, pen=pg.mkPen(colour, width=2))
        test = p.plot(
            sensors + 1,
            scale * real,
            pen=None,
            symbol="o",
            symbolSize=9,
            symbolBrush=pg.mkBrush(colors.strong),
            symbolPen=None,
        )
        self.shape_legend.addItem(model, f"Model mode {pr.model + 1}")
        self.shape_legend.addItem(test, f"Test mode {pr.test + 1}")
        p.addItem(pg.InfiniteLine(0.0, angle=0, pen=pg.mkPen(colors.faint, width=1)))
        p.setXRange(-0.25, n + 0.25, padding=0)
        p.setYRange(-1.15, 1.15, padding=0)
        p.getAxis("bottom").setTicks([[(i + 1, f"m{i + 1}") for i in range(n)]])
        self.shape_note.setText(
            f"Test mode {pr.test + 1} ({pr.f_test:.4g} Hz, dots at the sensors) against model mode {pr.model + 1} "
            f"({pr.f_model:.4g} Hz, the line): MAC {pr.mac:.3f}."
        )

    def _draw_auto(self) -> None:
        assert self.corr is not None, "drawn after recompute"
        c = self.corr
        k = len(c.test.modes)
        names = [f"T{i + 1}" for i in range(k)]
        self.auto_plot.set_matrix(c.auto_mac, names, names)
        if k < 2:
            self.auto_note.setText("")
            return
        off = c.auto_mac - np.diag(np.diag(c.auto_mac))
        i, j = np.unravel_index(int(np.argmax(off)), off.shape)
        sensors = int(c.test.sensors.sum())
        text = f"Largest off-diagonal term: <b>{off[i, j]:.2f}</b>, test modes {i + 1} and {j + 1}."
        if off[i, j] > 0.5:
            text += (
                f" At {sensors} sensor{'s' if sensors > 1 else ''} these modes look alike: add a sensor or move one."
            )
        self.auto_note.setText(text)

    def _draw_comac(self) -> None:
        assert self.corr is not None and self.system is not None, "drawn after recompute"
        c = self.corr
        p = self.comac_plot
        p.clear()
        sensors = np.flatnonzero(c.test.sensors)
        values = np.nan_to_num(c.comac, nan=0.0)
        p.addItem(
            pg.BarGraphItem(
                x=sensors + 1, height=values, width=0.6, brushes=[pg.mkBrush(colors.mass[i]) for i in sensors], pen=None
            )
        )
        p.setRange(xRange=(0.4, self.system.n + 0.6), yRange=(min(0.8, float(values.min()) - 0.02), 1.005), padding=0)
        p.getAxis("bottom").setTicks([[(i + 1, f"m{i + 1}") for i in sensors]])
        if c.pairs and sensors.size:
            low = int(np.argmin(values))
            self.comac_note.setText(
                f"Lowest at m{sensors[low] + 1}: <b>{values[low]:.3f}</b>, over {len(c.pairs)} "
                "pairs. A weak locator: a stiffness error changes every shape a little."
            )
        else:
            self.comac_note.setText("")

    def _draw_update(self) -> None:
        """The Updating and Sensitivity tabs, when they show (updated once per correlation and setting)."""
        assert self.system is not None and self.errors is not None, "drawn after recompute"
        c = self.corr
        tab = self.tabs.currentWidget()
        if c is None or tab not in (self.update_tab, self.sens_tab):
            return
        kinds = [k for k, box in self.update_kind.items() if box.isChecked()]
        params = parameters(self.system.n, kinds)
        if tab is self.sens_tab:
            sens = frequency_sensitivity(c.model, c.result, params)
            top = float(np.max(np.abs(sens))) if sens.size else 1.0
            self.sens_plot.set_matrix(
                np.abs(sens) / (top or 1.0),
                [f"M{r + 1}" for r in range(sens.shape[0])],
                [p.name for p in params],
                text=sens,
                fmt="{:+.2f}",
            )
            return
        if self.update_result is None:
            what = Residuals(self.fit_shapes.isChecked(), self.fit_damping.isChecked(), self.shape_weight.value())
            self.update_result = update(self.system, self.errors, c.test, params, what, self.regularization.value())
        u = self.update_result
        p, q = self.factor_plot, self.cost_plot
        p.clear()
        q.clear()
        self.factor_legend.clear()
        steps = np.arange(len(u.history))
        styles = {
            Kind.SPRING: QtCore.Qt.PenStyle.SolidLine,
            Kind.MASS: QtCore.Qt.PenStyle.DashLine,
            Kind.DAMPER: QtCore.Qt.PenStyle.DotLine,
        }
        factors = np.array([u.factors(it.errors) for it in u.history]).reshape(len(u.history), len(u.params))
        for j, prm in enumerate(u.params):
            curve = p.plot(
                steps, factors[:, j], pen=pg.mkPen(colors.mass[prm.element], width=2, style=styles[prm.kind])
            )
            self.factor_legend.addItem(curve, prm.name)
        p.addItem(pg.InfiniteLine(1.0, angle=0, pen=pg.mkPen(colors.faint, width=1)))
        if factors.size:
            lo, hi = min(float(factors.min()), 1.0), max(float(factors.max()), 1.0)
            pad = 0.08 * (hi - lo) + 0.02
            p.setRange(xRange=(0, max(1, len(u.history) - 1)), yRange=(lo - pad, hi + pad), padding=0)
        costs = np.maximum([it.cost for it in u.history], 1e-12)
        q.plot(
            steps,
            costs,
            pen=pg.mkPen(colors.strong, width=2),
            symbol="o",
            symbolSize=5,
            symbolBrush=pg.mkBrush(colors.strong),
            symbolPen=None,
        )
        q.setXRange(0, max(1, len(u.history) - 1), padding=0)
        self.apply_update.setEnabled(len(u.history) > 1)
        if not u.params or not u.pairs:
            self.update_note.setText("Nothing to update: tick a kind of parameter, and pair some modes.")
            return
        final = factors[-1]
        worst = int(np.argmax(np.abs(np.log(final))))
        start = factors[0]
        moved = int(np.argmax(np.abs(np.log(final / start))))
        sv = u.history[1].singular if len(u.history) > 1 else np.zeros(0)
        text = (
            f"{len(u.history) - 1} steps; ‖residual‖ {u.history[0].cost:.3g} → <b>{u.history[-1].cost:.3g}</b>. "
            f"Largest change: {u.params[moved].name}, {start[moved]:.3f} → <b>{final[moved]:.3f}</b> of the "
            f"structure's. Furthest from the truth now: {u.params[worst].name} at {final[worst]:.3f}."
        )
        if sv.size:
            unseen = int(np.sum(sv < 1e-3 * sv[0]))
            rank_note = (
                f" {unseen} of the {len(u.params)} directions in the parameters "
                f"{'changes' if unseen == 1 else 'change'} nothing the test measures (a singular value of "
                f"0): the data can't settle {'it' if unseen == 1 else 'them'}."
                if unseen
                else ""
            )
            text += f" Singular values {sv.max():.3g} to {sv.min():.2g}.{rank_note}"
        self.update_note.setText(text)


def _same_chain(a: ChainSystem, b: ChainSystem) -> bool:
    return a.n == b.n and all(
        np.allclose(x, y) for x, y in ((a.masses, b.masses), (a.stiffness, b.stiffness), (a.damping, b.damping))
    )
