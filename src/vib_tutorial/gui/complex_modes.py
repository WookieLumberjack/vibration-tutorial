"""Complex modes page: when the chain's damping is not proportional, and what that does to its modes.

Left: the chain's damping (its coupling index, a button that makes it proportional), chains to
try, and the mode looked at. Centre: that mode's exact shape ψ in the complex plane with its
best-fit line, its snapshots over half a cycle, and every mode's exact pole and shape against
the proportional approximation's. Right: the FRF, exact and approximated; a free decay from an
undamped mode's shape, and what leaks to the others; each mode's phase scatter; the notes.
Follows the chain from the Simulation page (friction left out: modes are linear).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.complex_modes import (
    IdentificationResult,
    ModeComparison,
    Parameter,
    Sweep,
    compare,
    complexity,
    free_decay,
    frf_grid,
    frf_pair,
    identify,
    made_proportional,
    strobe,
    sweep,
)
from ..core.modal import TWO_PI, ModalResult, modal_analysis
from ..core.model import ChainSystem
from ..core.practice import Problem, Question, Requirement, at_least, number, pick, rounded
from ..core.presets import tuned_mass_damper
from .axes import log_axes
from .layouts import ColumnScroll, FlowLayout, group, plot_height
from .links import load_notes, page_link
from .panels import spin
from .style import MAX_DOF, colors
from .theming import add_legend, mute

SNAPSHOTS = 8  # over half a cycle, on the snapshots plot

TIPS = {
    "coupling": "<p>The largest off-diagonal term of Φ<sup>T</sup>CΦ against its diagonal's, "
    "|c<sub>rs</sub>|/√(c<sub>rr</sub>c<sub>ss</sub>): 0 when the damping is proportional, near 1 or "
    "more when it couples the modes strongly.</p>",
    "proportional": "<p>Set every damper in proportion to its spring, c<sub>i</sub> = βk<sub>i</sub>, so "
    "C = βK, which the undamped modes diagonalize. β keeps mode 1's damping ratio.</p>",
    "mode": "<p>The mode drawn in the complex plane and the snapshots, and released in the free decay.</p>",
    "frf": "<p>The receptance from a force on one mass to another's displacement.</p>",
}
ARGAND_TIP = (
    "<p>The exact mode shape ψ in the complex plane: one arrow per mass. A real mode's arrows lie on one line "
    "(the dashed best-fit line); a complex mode's spread from it, each mass with its own phase.</p>"
)
STROBE_TIP = (
    "<p>The mode's shape at eight instants over half a cycle, Re(ψe<sup>iθ</sup>): a real mode's are one shape, "
    "scaled, and its nodes stay put (a standing wave); a complex mode's change shape as they go, and the nodes "
    "move along the chain. Grey: ±|ψ|, the envelope.</p>"
)
TABLE_TIP = (
    "<p>Each mode: the exact pole's frequency |λ|/2π and damping ratio, against the proportional approximation's "
    "(ζ = φ<sup>T</sup>Cφ/2ω, the undamped frequency); the exact shape's MAC with the undamped one, its MPC "
    "(1 real, 0 spread evenly) and its mean phase deviation from the best-fit line.</p>"
)
FRF_TIP = (
    "<p>The receptance, exact (K − ω²M + iωC)<sup>−1</sup>, and with the proportional approximation's damping "
    "C<sub>p</sub> = MΦ diag(Φ<sup>T</sup>CΦ) Φ<sup>T</sup>M: the off-diagonal terms dropped.</p>"
)
DECAY_TIP = (
    "<p>The chain released at rest in an undamped mode's shape, its modal coordinates q<sub>s</sub> = "
    "φ<sub>s</sub><sup>T</sup>Mx over time, exact. The approximation keeps the released mode alone, decaying at "
    "its ζ. With proportional damping the others stay at 0; otherwise the damping passes motion to them.</p>"
)
SWEEP_TIP = (
    "<p>One damper (or spring) swept over a wide range, the rest of the chain as it is: each mode's exact pole "
    "(the root locus), its damping ratio, exact and approximated, and its MPD. The dotted line and the dots: the "
    "chain's own value.</p>"
)
LOCUS_TIP = (
    "<p>The root locus: each mode's exact pole λ as the element changes, Re λ across (the decay rate), "
    "Im λ/2π up (the damped frequency). Grey crosses: real roots, where a mode is overdamped.</p>"
)
IDENT_TIP = (
    "<p>The FRF column for a force on one mass, with a little noise, fitted by LSFD on the exact poles two ways: "
    "with complex modes (any complex residues) and with real modes (A<sub>r</sub>/(ω<sub>r</sub>² − ω² + "
    "2iζ<sub>r</sub>ω<sub>r</sub>ω), A<sub>r</sub> real), the model of a proportionally damped structure.</p>"
)
SCATTER_TIP = (
    "<p>Each mode's entries' angles from its best-fit line (one dot per mass, mass colours), and their mean, "
    "weighted by size (the MPD, a bar). All on 0 for real modes.</p>"
)


def _chain(damping: list[float]) -> ChainSystem:
    return ChainSystem(np.ones(4), np.full(4, 400.0), np.array(damping, dtype=float))


# Things to try: (name, chain, note)
TRIALS = (
    (
        "A damper per spring, in proportion",
        ChainSystem.uniform(4),
        "Four equal masses, springs and dampers: c<sub>i</sub> = k<sub>i</sub>/200, so C = K/200 and the damping is "
        "proportional. Every mode is real (MPC 1, MPD 0°), the approximation is exact, and a mode released "
        "stays alone.",
    ),
    (
        "One damper, at the base",
        _chain([20.0, 0, 0, 0]),
        "The same chain with a single 20 N·s/m damper, on spring 1. Mode 3 is the most complex (MPC 0.20, "
        "MPD 29°), and the approximation puts mode 4's ζ at 0.049, twice the exact 0.023. Released in mode 2's "
        "shape, the chain passes 45 % of it to mode 1.",
    ),
    (
        "One damper, at the top",
        _chain([0, 0, 0, 20.0]),
        "The damper between m3 and m4: mode 4's ζ is 0.017, not the approximation's 0.17, ten times less. "
        "The approximation counts the damper's stretch in the undamped mode 4, but the exact mode 4 shakes m1 "
        "and m2 and leaves m3 and m4 moving nearly together (under a third of the stretch), so the damper barely "
        "works.",
    ),
    (
        "A strong damper locks a joint",
        _chain([0, 200.0, 0, 0]),
        "200 N·s/m between m1 and m2: so stiff a damper that the two move together, and it hardly works. Mode 1's "
        "ζ is 0.050, not the approximation's 0.58; mode 3's 0.0018, not 2.6; mode 4 has become a pair of real "
        "roots. Mode 2, where m1 and m2 already move together, is untouched.",
    ),
    (
        "Tuned mass damper",
        tuned_mass_damper(ChainSystem.uniform(4), 0.05),
        "The default chain with a Den Hartog damper on m4 (m5). Modes 1 and 2 are the structure's mode 1 split in "
        "two, close together, and the damper couples them: the most complex (MPD 10° and 17°), while the "
        "modes above, far from them, stay nearly real.",
    ),
)

TAB_FRF, TAB_DECAY, TAB_PHASES, TAB_SWEEP, TAB_IDENTIFICATION = range(5)


@dataclass(frozen=True)
class ComplexSetup:
    """A practice problem's setup: a chain (the default 4-mass one, its dampers given) and the tab to show."""

    damping: tuple[float, ...] = (2.0, 2.0, 2.0, 2.0)
    masses: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    stiffness: tuple[float, ...] = (400.0, 400.0, 400.0, 400.0)
    tab: int = TAB_FRF

    def system(self) -> ChainSystem:
        return ChainSystem(np.array(self.masses), np.array(self.stiffness), np.array(self.damping))


PRACTICE_SETUP = ComplexSetup  # what a practice problem's setup and the page's practice_state are

# ------------------------------------------------------------------ practice problems
# Each question's numbers are printed round, and its answer is worked from them as printed.


def _mpc(rng: np.random.Generator) -> Question:
    re_ = pick(rng, (-0.8, -0.5, 0.5, 0.8))
    im = pick(rng, (0.1, 0.2, 0.3, 0.5))
    psi = np.array([1.0, complex(re_, im)])
    mpc = complexity(psi).mpc
    return Question(
        f"A mode shape of two entries, ψ = (1, {re_:g} + {im:g}i). What is its MPC?",
        mpc,
        f"s<sub>xx</sub> = Σ(Re ψ)² = {number(1 + re_**2)}, s<sub>yy</sub> = Σ(Im ψ)² = {number(im**2)}, "
        f"s<sub>xy</sub> = Σ Re ψ Im ψ = {number(re_ * im)}; MPC = [(s<sub>xx</sub> − s<sub>yy</sub>)² + "
        f"4s<sub>xy</sub>²]/(s<sub>xx</sub> + s<sub>yy</sub>)² = {number(mpc)}. (1 for a real mode.)",
        tolerance=0.01,
    )


def _approx_zeta(rng: np.random.Generator) -> Question:
    c = pick(rng, (5, 10, 20))
    r = int(pick(rng, (1, 2, 3, 4)))
    e = int(pick(rng, (1, 2, 3, 4)))  # the damper's element
    system = ChainSystem(np.ones(4), np.full(4, 400.0), np.zeros(4))
    mode = modal_analysis(system).modes[r - 1]
    phi = np.round(mode.shape_mass_normalized, 3)
    w = rounded(mode.omega_n)
    lower = phi[e - 2] if e > 1 else 0.0
    stretch = phi[e - 1] - lower
    zeta = c * stretch**2 / (2 * w)
    ends = (
        f"φ<sub>{e - 1}</sub> = {lower:g} and φ<sub>{e}</sub> = {phi[e - 1]:g}"
        if e > 1
        else f"φ<sub>1</sub> = {phi[0]:g} (the ground still)"
    )
    return (
        Question(
            f"The default chain with a single damper, c = {c:g} N·s/m on element {e}. Its undamped mode {r} (mass-normalized) "
            f"has {ends}, at ω = {number(w)} rad/s. What ζ does the proportional approximation give it?",
            zeta,
            f"ζ ≈ φᵀCφ/2ω, and a single damper's φᵀCφ is c times its stretch squared: {c:g} × ({number(stretch)})² / "
            f"(2 × {number(w)}) = {number(zeta)}. Whether the exact mode agrees depends on how much the damper "
            "changes the shape.",
            tolerance=0.03,
            setup=ComplexSetup(damping=tuple(c if i == e - 1 else 0.0 for i in range(4))).__dict__,
        )
        if abs(stretch) > 0.05
        else _approx_zeta(rng)
    )


def _drift_time(rng: np.random.Generator) -> Question:
    n = int(pick(rng, (2, 3, 4)))
    m = pick(rng, (0.5, 1, 2))
    c1 = pick(rng, (1, 2, 5, 10))
    tau = n * m / c1
    return Question(
        f"A chain of {n} masses of {m:g} kg has no spring to the ground, only a damper of {c1:g} N·s/m there. Pushed, "
        "it slides to a new position as its velocity dies away. With what time constant?",
        tau,
        f"Moving together, the chain is its whole mass on the ground damper: m<sub>total</sub>v' = −c<sub>1</sub>v, a "
        f"real root λ ≈ −c<sub>1</sub>/Σm, τ = Σm/c<sub>1</sub> = {number(n * m)}/{c1:g} = {number(tau)} s.",
        unit="s",
    )


def _mac(rng: np.random.Generator) -> Question:
    b = pick(rng, (0.5, 1, 1.5, 2))
    im = pick(rng, (0.5, 1, 1.5))
    phi = np.array([1.0, b])
    psi = np.array([1.0, complex(b, im)])
    mac = abs(phi @ psi) ** 2 / float(phi @ phi) / float(np.real(np.vdot(psi, psi)))
    return Question(
        f"An undamped shape φ = (1, {b:g}) against an exact one ψ = (1, {b:g} + {im:g}i). What is their MAC?",
        mac,
        f"MAC = |φᵀψ|²/(φᵀφ ψᴴψ): φᵀψ = {number(1 + b * b)} + {number(b * im)}i, |φᵀψ|² = "
        f"{number(abs(phi @ psi) ** 2)}; φᵀφ = {number(1 + b * b)}, ψᴴψ = {number(1 + b * b + im * im)}: "
        f"{number(mac)}.",
        tolerance=0.005,
    )


def _made_proportional(rng: np.random.Generator) -> Question:
    return Question(
        "A chain with a single damper has complex modes. Its dampers are replaced by ones proportional to its springs "
        "(c<sub>i</sub> = βk<sub>i</sub>). What becomes of every mode's MPC?",
        0,
        "Every one becomes 1: with C = βK, ΦᵀCΦ is diagonal, the modes don't couple, and the exact shapes are the "
        "real undamped ones.",
        choices=("They all become 1", "They stay as they were", "They all fall"),
        setup=ComplexSetup(tab=TAB_PHASES).__dict__,
        predict=True,
    )


def _damper_at_top(rng: np.random.Generator) -> Question:
    return Question(
        "A single 20 N·s/m damper between m3 and m4 of the default chain. How does the proportional approximation's ζ "
        "of mode 4 compare with the exact one?",
        0,
        "About ten times too high (0.17 against 0.017): the approximation counts the damper's stretch in the "
        "undamped mode 4, but in the exact mode m3 and m4 move nearly together, so the damper hardly works.",
        choices=("Far too high", "About right", "Far too low"),
        setup=ComplexSetup(damping=(0.0, 0.0, 0.0, 20.0)).__dict__,
        predict=True,
    )


def _stiffer_damper(rng: np.random.Generator) -> Question:
    return Question(
        "A single damper at the base of the chain is made stiffer and stiffer. What does mode 1's exact damping ratio do?",
        0,
        "It rises to a peak and falls again: a stiff damper holds its ends together and stops working, and the "
        "chain's modes become those with that joint locked. The approximation's ζ keeps rising.",
        choices=("It rises, then falls", "It keeps rising", "It rises, then levels off"),
        setup=ComplexSetup(damping=(20.0, 0.0, 0.0, 0.0), tab=TAB_SWEEP).__dict__,
        predict=True,
    )


def _best_damper_spec(rng: np.random.Generator) -> Question:
    least = pick(rng, (0.12, 0.13, 0.14))

    def goal(t: ComplexSetup):
        system = t.system()
        n = system.n
        same = (
            n == 4
            and bool(np.allclose(system.masses, 1.0, rtol=0.005))
            and bool(np.allclose(system.stiffness, 400.0, rtol=0.005))
        )
        others = float(np.sum(system.damping[1:])) if n > 1 else 0.0
        d = modal_analysis(system).modes[0].damped
        zeta = -d.eigenvalue.real / abs(d.eigenvalue) if d is not None else 0.0
        return (
            Requirement("The default chain's masses and springs", f"{n} masses", same),
            Requirement(
                "Only a damper at the base (c<sub>2</sub> … c<sub>4</sub> = 0)",
                f"{number(others)} N·s/m besides",
                others == 0,
            ),
            at_least("Mode 1's exact ζ", zeta, least),
        )

    return Question(
        f"The default chain with only one damper, at its base. Choose c<sub>1</sub> (on the Simulation page) so mode "
        f"1's exact damping ratio is at least {least:g}.",
        0,
        "Mode 1's ζ rises with c<sub>1</sub> to a peak of 0.147 at about 60 N·s/m, then falls as the damper locks m1 "
        "to the ground (the Sweep tab draws it); a stiffer damper is worse, not better. Anything near 60 N·s/m does.",
        setup=ComplexSetup(damping=(5.0, 0.0, 0.0, 0.0), tab=TAB_SWEEP).__dict__,
        goal=goal,
        design=ComplexSetup(damping=(60.0, 0.0, 0.0, 0.0), tab=TAB_SWEEP).__dict__,
    )


PROBLEMS = tuple(
    Problem(key, title, "complex_page", f"See {page_link('complex_page', anchor, section)}{more}.", make)
    for key, title, anchor, section, more, make in (
        ("mpc", "MPC of a shape", "how-complex", "How complex", "", _mpc),
        ("approx-zeta", "The approximation's ζ", "ignoring-it", "What ignoring it costs", "", _approx_zeta),
        ("drift-time", "A chain that slides", "state-space", "The state-space form", ": real eigenvalues", _drift_time),
        ("mac", "MAC of two shapes", "how-complex", "How complex", "", _mac),
        (
            "made-proportional",
            "Predict: make it proportional",
            "complex-shapes",
            "Complex modes",
            "",
            _made_proportional,
        ),
        ("damper-at-top", "Predict: a damper at the top", "ignoring-it", "What ignoring it costs", "", _damper_at_top),
        ("stiffer-damper", "Predict: a stiffer damper", "sweep", "Sweeping a damper", "", _stiffer_damper),
        (
            "best-damper-spec",
            "Spec: the best damper",
            "sweep",
            "Sweeping a damper",
            ": is the stiffest best?",
            _best_damper_spec,
        ),
    )
)


NOTES_HTML = load_notes(
    "complex_modes", trials="<ul>" + "".join(f"<li><b>{name}.</b> {note}</li>" for name, _, note in TRIALS) + "</ul>"
)


def _plot(left: str, left_units: str | None, bottom: str, bottom_units: str | None, log: bool = False) -> pg.PlotWidget:
    p = pg.PlotWidget(axisItems=log_axes()) if log else pg.PlotWidget()
    p.setMenuEnabled(False)
    p.hideButtons()
    p.showGrid(x=True, y=True, alpha=0.15)
    p.setLabel("left", left, units=left_units)
    p.setLabel("bottom", bottom, units=bottom_units)
    p.getAxis("left").enableAutoSIPrefix(False)
    return p


def _cell(text: str) -> QtWidgets.QTableWidgetItem:
    item = QtWidgets.QTableWidgetItem(text)
    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
    return item


def _percent(approx: float, exact: float) -> str:
    if not (exact and np.isfinite(exact) and np.isfinite(approx)):
        return "–"
    error = 100 * (approx / exact - 1)
    return "0 %" if abs(error) < 1e-9 else f"{error:+.3g} %"  # not round-off's ±1e-13 %


class ComplexModesPage(QtWidgets.QWidget):
    """The damping and the mode on the left, the mode's shape and the table in the centre, the rest on the right."""

    dof_requested = QtCore.Signal(int)  # the user changed N here; the main window owns the system
    edit_parameters = QtCore.Signal()  # the user wants the Simulation page's parameter panel
    system_requested = QtCore.Signal(object)  # a ChainSystem for the Simulation page to load

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.system: ChainSystem | None = None
        self.result: ModalResult | None = None
        self.rows: list[ModeComparison] = []
        self._dirty = False

        # --- left: the chain's damping, chains to try, the mode
        box = QtWidgets.QGroupBox("Damping")
        form = QtWidgets.QFormLayout(box)
        self.dof = QtWidgets.QSpinBox()
        self.dof.setRange(1, MAX_DOF)
        self.dof.setToolTip("Same chain as the Simulation page; changing it here changes it there too.")
        self.dof.valueChanged.connect(self.dof_requested)
        form.addRow("Number of masses:", self.dof)
        self.coupling = QtWidgets.QLabel()
        self.coupling.setWordWrap(True)
        self.coupling.setToolTip(TIPS["coupling"])
        form.addRow(self.coupling)
        self.make_proportional = QtWidgets.QPushButton("Make it proportional")
        self.make_proportional.setToolTip(TIPS["proportional"])
        form.addRow(self.make_proportional)
        link = QtWidgets.QLabel("Masses, springs and dampers are edited on the <a href='#sim'>Simulation page</a>.")
        mute(link)
        link.setWordWrap(True)
        link.linkActivated.connect(lambda _: self.edit_parameters.emit())
        form.addRow(link)

        trials = QtWidgets.QGroupBox("Things to try")
        tv = QtWidgets.QVBoxLayout(trials)
        self.trial = QtWidgets.QComboBox()
        self.trial.addItem("Load a chain…")
        self.trial.addItems([name for name, _, _ in TRIALS])
        self.trial.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.trial.setToolTip(
            "<p>Chains that show something about complex modes. Loading one replaces the Simulation page's chain.</p>"
        )
        tv.addWidget(self.trial)
        self.trial_note = QtWidgets.QLabel()
        self.trial_note.setWordWrap(True)
        self.trial_note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        mute(self.trial_note)
        self.trial_note.setVisible(False)
        tv.addWidget(self.trial_note)

        mode_box = QtWidgets.QGroupBox("Mode")
        form = QtWidgets.QFormLayout(mode_box)
        self.mode = QtWidgets.QComboBox()
        self.mode.setToolTip(TIPS["mode"])
        label = QtWidgets.QLabel("Show:")
        label.setToolTip(TIPS["mode"])
        form.addRow(label, self.mode)
        self.mode_info = QtWidgets.QLabel()
        self.mode_info.setWordWrap(True)
        self.mode_info.setTextFormat(QtCore.Qt.TextFormat.RichText)
        form.addRow(self.mode_info)

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        for b in (box, mode_box, trials):
            lv.addWidget(b)
        lv.addStretch(1)
        left_scroll = ColumnScroll(left)

        # --- centre: the mode in the complex plane, its snapshots, the table
        self.argand = _plot("Im ψ", None, "Re ψ", None)
        self.argand.setAspectLocked(True)
        self.argand.setMouseEnabled(x=False, y=False)
        self.argand.setToolTip(ARGAND_TIP)
        self.strobe = _plot("Re(ψ e<sup>iθ</sup>)", None, "Mass", None)
        self.strobe.setMouseEnabled(x=False, y=False)
        self.strobe.setToolTip(STROBE_TIP)
        for p in (self.argand, self.strobe):
            plot_height(p, 220)
        shapes = QtWidgets.QWidget()
        sh = QtWidgets.QHBoxLayout(shapes)
        sh.setContentsMargins(0, 0, 0, 0)
        sh.addWidget(self.argand, 1)
        sh.addWidget(self.strobe, 1)
        self.table = QtWidgets.QTableWidget()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.setToolTip(TABLE_TIP)
        self.readout = QtWidgets.QLabel()
        self.readout.setWordWrap(True)
        self.readout.setTextFormat(QtCore.Qt.TextFormat.RichText)
        center = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(center)
        cv.addWidget(shapes, 3)
        cv.addWidget(self.table, 2)
        cv.addWidget(self.readout)

        # --- right: the FRF, the free decay, the phase scatter, the notes
        self.frf_in = QtWidgets.QComboBox()
        self.frf_out = QtWidgets.QComboBox()
        for b in (self.frf_in, self.frf_out):
            b.setToolTip(TIPS["frf"])
        self.frf_mag = _plot("|x/F|", None, "Frequency", "Hz", log=True)
        self.frf_mag.setLogMode(x=True, y=True)
        self.frf_mag.getAxis("left").enableAutoSIPrefix(False)
        self.frf_mag.setToolTip(FRF_TIP)
        self.frf_legend = add_legend(self.frf_mag.getPlotItem(), offset=(-10, 10))
        self.frf_phase = _plot("Phase (°)", None, "Frequency", "Hz", log=True)
        self.frf_phase.setLogMode(x=True, y=False)
        self.frf_phase.getAxis("left").enableAutoSIPrefix(False)
        self.frf_phase.setXLink(self.frf_mag)
        self.frf_phase.setToolTip(FRF_TIP)
        plot_height(self.frf_mag, 180)
        plot_height(self.frf_phase, 110)
        self.frf_note = QtWidgets.QLabel()
        self.frf_note.setWordWrap(True)
        self.frf_note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        frf_tab = QtWidgets.QWidget()
        fv = QtWidgets.QVBoxLayout(frf_tab)
        row = QtWidgets.QFormLayout()
        row.addRow("Force on:", group(self.frf_in, "response of", self.frf_out))
        fv.addLayout(row)
        fv.addWidget(self.frf_mag, 3)
        fv.addWidget(self.frf_phase, 2)
        fv.addWidget(self.frf_note)

        self.decay_plot = _plot("Released mode's q", None, "Time", "s")
        self.decay_plot.setToolTip(DECAY_TIP)
        self.decay_legend = add_legend(self.decay_plot.getPlotItem(), offset=(-10, 10))
        self.leak_plot = _plot("The others' q", None, "Time", "s")
        self.leak_plot.setToolTip(DECAY_TIP)
        self.leak_legend = add_legend(self.leak_plot.getPlotItem(), offset=(-10, 10))
        self.leak_plot.setXLink(self.decay_plot)
        plot_height(self.decay_plot, 150)
        plot_height(self.leak_plot, 150)
        self.decay_note = QtWidgets.QLabel()
        self.decay_note.setWordWrap(True)
        self.decay_note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        decay_tab = QtWidgets.QWidget()
        dv = QtWidgets.QVBoxLayout(decay_tab)
        dv.addWidget(self.decay_plot, 1)
        dv.addWidget(self.leak_plot, 1)
        dv.addWidget(self.decay_note)

        self.scatter_plot = _plot("Angle from the best-fit line (°)", None, "Mode", None)
        self.scatter_plot.setMouseEnabled(x=False, y=False)
        self.scatter_plot.setToolTip(SCATTER_TIP)
        plot_height(self.scatter_plot, 200)
        self.scatter_note = QtWidgets.QLabel(
            "Each mass's phase from its mode's best-fit line; bars: the MPD. Real modes put every dot on 0."
        )
        self.scatter_note.setWordWrap(True)
        mute(self.scatter_note)
        scatter_tab = QtWidgets.QWidget()
        pv = QtWidgets.QVBoxLayout(scatter_tab)
        pv.addWidget(self.scatter_plot, 1)
        pv.addWidget(self.scatter_note)

        self.sweep_parameter = QtWidgets.QComboBox()
        for prm in Parameter:
            self.sweep_parameter.addItem(prm.value, prm)
        self.sweep_element = QtWidgets.QComboBox()
        for b in (self.sweep_parameter, self.sweep_element):
            b.setToolTip(SWEEP_TIP)
        self.locus = _plot("Im λ / 2π (Hz)", None, "Re λ (1/s)", None)
        self.locus.setToolTip(LOCUS_TIP)
        self.sweep_zeta = _plot("ζ", None, "", None, log=True)
        self.sweep_zeta.setLogMode(x=True, y=False)
        self.sweep_zeta.setToolTip(SWEEP_TIP)
        self.sweep_zeta_legend = add_legend(self.sweep_zeta.getPlotItem(), offset=(10, 10))
        self.sweep_mpd = _plot("MPD (°)", None, "", None, log=True)
        self.sweep_mpd.setLogMode(x=True, y=False)
        self.sweep_mpd.setXLink(self.sweep_zeta)
        self.sweep_mpd.setToolTip(SWEEP_TIP)
        for p in (self.sweep_zeta, self.sweep_mpd):
            p.getAxis("left").enableAutoSIPrefix(False)
        plot_height(self.locus, 150)
        plot_height(self.sweep_zeta, 120)
        plot_height(self.sweep_mpd, 100)
        self.sweep_note = QtWidgets.QLabel()
        self.sweep_note.setWordWrap(True)
        self.sweep_note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        sweep_tab = QtWidgets.QWidget()
        wv = QtWidgets.QVBoxLayout(sweep_tab)
        row = FlowLayout()
        row.addWidget(group("Sweep:", self.sweep_parameter, self.sweep_element))
        wv.addLayout(row)
        wv.addWidget(self.locus, 4)
        wv.addWidget(self.sweep_zeta, 3)
        wv.addWidget(self.sweep_mpd, 2)
        wv.addWidget(self.sweep_note)
        sweep_scroll = ColumnScroll(sweep_tab)

        self.ident_in = QtWidgets.QComboBox()
        self.ident_out = QtWidgets.QComboBox()
        self.ident_noise = spin(0.0, 20.0, 1.0, 1, " %")
        for b in (self.ident_in, self.ident_out, self.ident_noise):
            b.setToolTip(IDENT_TIP)
        self.ident_noise.setToolTip("<p>Noise on each response, its RMS this share of the response's peak.</p>")
        self.ident_plot = _plot("|x/F| (m/N)", None, "Frequency", "Hz", log=True)
        self.ident_plot.setLogMode(x=False, y=True)
        self.ident_plot.getAxis("left").enableAutoSIPrefix(False)
        self.ident_plot.setToolTip(IDENT_TIP)
        self.ident_legend = add_legend(self.ident_plot.getPlotItem(), offset=(-10, 10))
        plot_height(self.ident_plot, 170)
        self.ident_table = QtWidgets.QTableWidget()
        self.ident_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.ident_table.verticalHeader().setVisible(False)
        self.ident_table.setToolTip(
            "<p>Each mode's identified shape: its MAC with the exact complex shape ψ for both fits, the real "
            "fit's MAC with the undamped shape φ, and the complex fit's MPC against the exact one.</p>"
        )
        self.ident_note = QtWidgets.QLabel()
        self.ident_note.setWordWrap(True)
        self.ident_note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        ident_tab = QtWidgets.QWidget()
        iv = QtWidgets.QVBoxLayout(ident_tab)
        row = FlowLayout()
        row.addWidget(group("Force on:", self.ident_in))
        row.addWidget(group("plot:", self.ident_out))
        row.addWidget(group("Noise:", self.ident_noise))
        iv.addLayout(row)
        iv.addWidget(self.ident_plot, 3)
        iv.addWidget(self.ident_table, 2)
        iv.addWidget(self.ident_note)
        self.sweep_result: Sweep | None = None
        self.ident_result: IdentificationResult | None = None
        self._stale: set[QtWidgets.QWidget] = set()

        self.notes = QtWidgets.QTextBrowser()
        self.notes.setHtml(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.frf_tab, self.decay_tab, self.scatter_tab = frf_tab, decay_tab, scatter_tab
        self.sweep_tab, self.ident_tab = sweep_scroll, ident_tab
        self.tabs.addTab(frf_tab, "FRF")
        self.tabs.addTab(decay_tab, "Free decay")
        self.tabs.addTab(scatter_tab, "Phases")
        self.tabs.addTab(sweep_scroll, "Sweep")
        self.tabs.addTab(ident_tab, "Identification")
        self.tabs.addTab(self.notes, "Theory")

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(center)
        splitter.addWidget(self.tabs)
        splitter.setSizes([320, 760, 620])
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        # --- wiring
        self.make_proportional.clicked.connect(self._on_make_proportional)
        self.trial.currentIndexChanged.connect(self._on_trial)
        self.mode.currentIndexChanged.connect(self._on_mode)
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        self.frf_in.currentIndexChanged.connect(lambda _: self._draw_frf())
        self.frf_out.currentIndexChanged.connect(lambda _: self._draw_frf())
        self.sweep_parameter.currentIndexChanged.connect(self._on_sweep_parameter)
        self.sweep_element.currentIndexChanged.connect(lambda _: self._refresh(self.sweep_tab))
        self.ident_in.currentIndexChanged.connect(lambda _: self._refresh(self.ident_tab))
        self.ident_noise.valueChanged.connect(lambda _: self._refresh(self.ident_tab))
        self.ident_out.currentIndexChanged.connect(lambda _: self._draw_identification())
        self.tabs.currentChanged.connect(lambda _: self._catch_up())

    # ------------------------------------------------------------- inputs
    def set_system(self, system: ChainSystem, result: ModalResult | None = None) -> None:
        """New chain parameters: worked out again (now if shown)."""
        n_changed = self.system is None or system.n != self.system.n
        self.system = system
        self.result = result if result is not None else modal_analysis(system)
        self.dof.blockSignals(True)
        self.dof.setValue(system.n)
        self.dof.blockSignals(False)
        if n_changed:
            for box, default in (
                (self.frf_in, system.n - 1),
                (self.frf_out, 0),
                (self.ident_in, 0),
                (self.ident_out, system.n - 1),
            ):
                box.blockSignals(True)
                box.clear()
                box.addItems([f"m{i + 1}" for i in range(system.n)])
                box.setCurrentIndex(default)
                box.blockSignals(False)
            self.mode.blockSignals(True)
            current = self.mode.currentIndex()
            self.mode.clear()
            self.mode.addItems([f"Mode {i + 1}" for i in range(system.n)])
            self.mode.setCurrentIndex(current if 0 <= current < system.n else 0)
            self.mode.blockSignals(False)
            self._fill_elements()
        if (
            self.sweep_parameter.currentData() is Parameter.DAMPER
            and system.damping.any()
            and not (system.damping[self.sweep_element.currentIndex()])
        ):
            self.sweep_element.blockSignals(True)  # sweep a damper the chain has
            self.sweep_element.setCurrentIndex(int(np.argmax(system.damping)))
            self.sweep_element.blockSignals(False)
        if self.isVisible():
            self.recompute()
        else:
            self._dirty = True

    def load_trial(self, index: int) -> None:
        """Ask for chain `index` of TRIALS, and show its note."""
        _, system, note = TRIALS[index]
        self.trial_note.setText(note)
        self.trial_note.setVisible(True)
        self.system_requested.emit(system.copy())

    def load_setup(self, setup: dict) -> None:
        """Set the page up as a practice problem's `setup` says (a `ComplexSetup`'s fields): its chain through the
        Simulation page, and its tab."""
        t = ComplexSetup(**setup)
        system = t.system()
        self.trial_note.setVisible(False)
        self.system_requested.emit(system.copy())
        if self.system is None or self.system.n != system.n or not np.allclose(self.system.damping, system.damping):
            self.set_system(system)  # nobody passed it back: show it here alone
        self.tabs.setCurrentIndex(t.tab)

    def practice_state(self) -> ComplexSetup:
        """The chain the page follows and its tab: what a practice spec checks."""
        system = self.system
        return ComplexSetup(
            tuple(float(c) for c in system.damping),
            tuple(float(m) for m in system.masses),
            tuple(float(k) for k in system.stiffness),
            self.tabs.currentIndex(),
        )

    def _on_trial(self, index: int) -> None:
        if index <= 0:
            return
        self.trial.blockSignals(True)
        self.trial.setCurrentIndex(0)
        self.trial.blockSignals(False)
        self.load_trial(index - 1)

    def _on_make_proportional(self) -> None:
        if self.system is not None:
            self.trial_note.setVisible(False)
            self.system_requested.emit(made_proportional(self.system, self.result))

    def _on_mode(self, index: int) -> None:
        if index < 0 or not self.rows:
            return
        self.table.blockSignals(True)
        self.table.selectRow(index)
        self.table.blockSignals(False)
        self._draw_mode()
        self._draw_decay()

    def _on_table_selection(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows and rows[0].row() != self.mode.currentIndex():
            self.mode.setCurrentIndex(rows[0].row())

    def _fill_elements(self) -> None:
        n = self.system.n
        letter = "c" if self.sweep_parameter.currentData() is Parameter.DAMPER else "k"
        current = self.sweep_element.currentIndex()
        self.sweep_element.blockSignals(True)
        self.sweep_element.clear()
        self.sweep_element.addItems([f"{letter}{i + 1}" for i in range(n)])
        self.sweep_element.setCurrentIndex(current if 0 <= current < n else 0)
        self.sweep_element.blockSignals(False)

    def _on_sweep_parameter(self, _index: int) -> None:
        self._fill_elements()
        self._refresh(self.sweep_tab)

    def _refresh(self, tab: QtWidgets.QWidget) -> None:
        """Work out the sweep or the identification again, now if its tab shows, else when it does."""
        self._stale.add(tab)
        self._catch_up()

    def _catch_up(self) -> None:
        tab = self.tabs.currentWidget()
        if tab not in self._stale or self.system is None:
            return
        self._stale.discard(tab)
        if tab is self.sweep_tab:
            self.sweep_result = sweep(
                self.system, self.sweep_parameter.currentData(), max(self.sweep_element.currentIndex(), 0)
            )
            self._draw_sweep()
        elif tab is self.ident_tab:
            self.ident_result = identify(
                self.system, self.result, max(self.ident_in.currentIndex(), 0), self.ident_noise.value() / 100.0
            )
            self._draw_identification()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._dirty:
            self.recompute()

    # ------------------------------------------------------------- computing
    def recompute(self) -> None:
        self._dirty = False
        if self.system is None:
            return
        self.rows = compare(self.system, self.result)
        self._stale |= {self.sweep_tab, self.ident_tab}
        self.redraw()

    def selected(self) -> int:
        return min(max(self.mode.currentIndex(), 0), len(self.rows) - 1)

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        if self.rows and not self._dirty:  # a new chain while hidden: its rows are the old chain's until it shows
            self.redraw()

    def redraw(self) -> None:
        r = self.result
        if r.is_proportional:
            self.coupling.setText(
                "Coupling index <b>0</b>: the damping is <b>proportional</b>, and every mode real. "
                "Pick a chain under <i>Things to try</i> to see one that isn't."
            )
        else:
            self.coupling.setText(
                f"Coupling index <b>{r.coupling:.2f}</b>: the damping is <b>not proportional</b>, "
                "and the modes are complex."
            )
        self.make_proportional.setEnabled(not r.is_proportional)
        self._fill_table()
        self._draw_mode()
        self._draw_frf()
        self._draw_decay()
        self._draw_scatter()
        if self.sweep_result is not None:
            self._draw_sweep()
        if self.ident_result is not None:
            self._draw_identification()
        self._catch_up()
        worst = max((c for c in self.rows if c.measures is not None), key=lambda c: c.measures.mpd, default=None)
        over = [c.index for c in self.rows if c.exact is None and c.fn_hz > 0]
        text = ""
        if worst is not None and worst.measures.mpd > 0.05:
            text = f"The most complex: mode {worst.index}, MPC {worst.measures.mpc:.2f}, MPD {worst.measures.mpd:.3g}°."
        elif worst is not None:
            text = "Every mode is real: the approximation is exact."
        if over:
            text += (
                f" Overdamped (no oscillating pole): mode{'s' if len(over) > 1 else ''} "
                + ", ".join(map(str, over))
                + "."
            )
        self.readout.setText(text)

    def _fill_table(self) -> None:
        t = self.table
        t.blockSignals(True)
        t.clear()
        heads = ["Mode", "f exact (Hz)", "f approx", "ζ exact", "ζ approx", "ζ error", "MAC", "MPC", "MPD"]
        t.setColumnCount(len(heads))
        t.setHorizontalHeaderLabels(heads)
        t.setRowCount(len(self.rows))
        for i, c in enumerate(self.rows):
            cells = [f"{c.index}"]
            if c.exact is None:
                cells += ["overdamped", f"{c.fn_hz:.4g}", "–", f"{c.zeta_approx:.3g}", "–", "–", "–", "–"]
            else:
                m = c.measures
                cells += [
                    f"{c.fn_exact:.4g}",
                    f"{c.fn_hz:.4g}",
                    f"{c.zeta_exact:.3g}",
                    f"{c.zeta_approx:.3g}",
                    _percent(c.zeta_approx, c.zeta_exact),
                    f"{c.mac:.3f}",
                    f"{m.mpc:.3f}",
                    f"{m.mpd:.2f}°",
                ]
            for j, text in enumerate(cells):
                item = _cell(text)
                if j == 0:
                    item.setForeground(pg.mkColor(colors.mode[i % len(colors.mode)]))
                t.setItem(i, j, item)
        t.resizeColumnsToContents()
        t.selectRow(self.selected())
        t.blockSignals(False)

    def _draw_mode(self) -> None:
        if not self.rows:
            return
        c = self.rows[self.selected()]
        n = self.system.n
        a, s = self.argand, self.strobe
        a.clear()
        s.clear()
        if c.shape is None:
            self.mode_info.setText(
                f"Mode {c.index} has no oscillating pole: the damping has made it a pair of real roots."
            )
            return
        psi, m = c.shape, c.measures
        circle = np.exp(1j * np.linspace(0, 2 * np.pi, 97))
        a.plot(circle.real, circle.imag, pen=pg.mkPen(colors.faint, width=1, style=QtCore.Qt.PenStyle.DotLine))
        a.addItem(pg.InfiniteLine(0.0, angle=0, pen=pg.mkPen(colors.faint, width=1)))
        a.addItem(pg.InfiniteLine(0.0, angle=90, pen=pg.mkPen(colors.faint, width=1)))
        a.addItem(
            pg.InfiniteLine(
                (0.0, 0.0),
                angle=math.degrees(m.angle),
                pen=pg.mkPen(colors.grey, width=1.5, style=QtCore.Qt.PenStyle.DashLine),
            )
        )
        for i, z in enumerate(psi):
            a.plot([0.0, z.real], [0.0, z.imag], pen=pg.mkPen(colors.mass[i], width=2))
        a.addItem(
            pg.ScatterPlotItem(
                psi.real, psi.imag, size=8, pen=None, brush=[pg.mkBrush(colors.mass[i]) for i in range(n)]
            )
        )
        a.setRange(xRange=(-1.15, 1.15), yRange=(-1.15, 1.15), padding=0)
        a.setTitle(f"Mode {c.index}: MPC {m.mpc:.3f}, MPD {m.mpd:.2f}°", size="9pt")

        xs = np.arange(n + 1)
        env = np.concatenate([[0.0], np.abs(psi)])
        for sign in (1, -1):
            s.plot(xs, sign * env, pen=pg.mkPen(colors.grey, width=1, style=QtCore.Qt.PenStyle.DashLine))
        colour = colors.mode[(c.index - 1) % len(colors.mode)]
        for k, snap in enumerate(strobe(psi, SNAPSHOTS)):
            pen = pg.mkPen(_alpha(colour, 255 if k == 0 else 110), width=2 if k == 0 else 1)
            s.plot(xs, np.concatenate([[0.0], snap]), pen=pen)
        s.addItem(pg.InfiniteLine(0.0, angle=0, pen=pg.mkPen(colors.faint, width=1)))
        s.setXRange(-0.25, n + 0.25, padding=0)
        s.setYRange(-1.1, 1.1, padding=0)
        s.getAxis("bottom").setTicks([[(0, "ground")] + [(i + 1, f"m{i + 1}") for i in range(n)]])
        s.setTitle("Half a cycle, 8 snapshots", size="9pt")
        lam = c.exact
        self.mode_info.setText(
            f"Exact: {c.fn_exact:.4g} Hz, ζ = {c.zeta_exact:.3g} (λ = {lam.real:.3g} ± {lam.imag:.4g}i). "
            f"Approximation: {c.fn_hz:.4g} Hz, ζ = {c.zeta_approx:.3g}."
        )

    def _draw_frf(self) -> None:
        if not self.rows:
            return
        inp, out = max(self.frf_in.currentIndex(), 0), max(self.frf_out.currentIndex(), 0)
        f = frf_grid(self.result)
        exact, approx = frf_pair(self.system, self.result, f, inp, out)
        p, q = self.frf_mag, self.frf_phase
        p.clear()
        q.clear()
        self.frf_legend.clear()
        e = p.plot(f, np.maximum(np.abs(exact), 1e-300), pen=pg.mkPen(colors.strong, width=2))
        ap = p.plot(
            f, np.maximum(np.abs(approx), 1e-300), pen=pg.mkPen(colors.fit, width=2, style=QtCore.Qt.PenStyle.DashLine)
        )
        self.frf_legend.addItem(e, "Exact")
        self.frf_legend.addItem(ap, "Proportional approximation")
        q.plot(f, np.degrees(np.angle(exact)), pen=pg.mkPen(colors.strong, width=2))
        q.plot(f, np.degrees(np.angle(approx)), pen=pg.mkPen(colors.fit, width=2, style=QtCore.Qt.PenStyle.DashLine))
        q.setYRange(-185, 185, padding=0)
        p.setLabel("left", f"|x{out + 1}/F{inp + 1}| (m/N)")
        p.setXRange(math.log10(f[0]), math.log10(f[-1]), padding=0)
        peak = float(np.max(np.abs(exact)))
        p.setYRange(math.log10(peak) - 4, math.log10(peak) + 0.5, padding=0)
        err = np.abs(approx - exact) / np.max(np.abs(exact))
        self.frf_note.setText(
            f"Largest difference: <b>{100 * err.max():.3g} %</b> of the exact FRF's peak, at {f[np.argmax(err)]:.3g} Hz."
            if err.max() > 1e-9
            else "The two are the same: the damping is proportional."
        )

    def _draw_decay(self) -> None:
        if not self.rows:
            return
        k = self.selected()
        p, q = self.decay_plot, self.leak_plot
        p.clear()
        q.clear()
        self.decay_legend.clear()
        self.leak_legend.clear()
        try:
            run = free_decay(self.system, self.result, k)
        except ValueError as e:
            self.decay_note.setText(f"Mode {k + 1}: {e}.")
            return
        colour = colors.mode[k % len(colors.mode)]
        c = p.plot(run.t, run.q[:, k], pen=pg.mkPen(colour, width=2))
        a = p.plot(run.t, run.approx, pen=pg.mkPen(colors.strong, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        self.decay_legend.addItem(c, f"q{k + 1}, exact")
        self.decay_legend.addItem(a, "Approximation")
        p.setYRange(-1.05, 1.05, padding=0)
        p.setXRange(0, run.t[-1], padding=0)
        top = max(float(run.leak.max()), 1e-3)
        for s in range(self.system.n):
            if s != k:
                curve = q.plot(run.t, run.q[:, s], pen=pg.mkPen(colors.mode[s % len(colors.mode)], width=1.5))
                self.leak_legend.addItem(curve, f"q{s + 1}")
        q.setYRange(-1.1 * top, 1.1 * top, padding=0)
        s = int(np.argmax(run.leak))
        self.decay_note.setText(
            f"Released in mode {k + 1}'s undamped shape: <b>{100 * run.leak[s]:.3g} %</b> of it reaches mode {s + 1}."
            if run.leak[s] > 1e-9
            else f"Released in mode {k + 1}'s undamped shape, it stays there: no other mode moves."
        )

    def _draw_scatter(self) -> None:
        p = self.scatter_plot
        p.clear()
        n = self.system.n
        for c in self.rows:
            if c.measures is None:
                continue
            dev = c.measures.deviation
            p.addItem(
                pg.ScatterPlotItem(
                    np.full(n, float(c.index)),
                    dev,
                    size=8,
                    pen=None,
                    brush=[pg.mkBrush(colors.mass[i]) for i in range(n)],
                )
            )
            for sign in (1, -1):
                p.plot(
                    [c.index - 0.3, c.index + 0.3],
                    [sign * c.measures.mpd] * 2,
                    pen=pg.mkPen(colors.mode[(c.index - 1) % len(colors.mode)], width=2),
                )
        p.addItem(pg.InfiniteLine(0.0, angle=0, pen=pg.mkPen(colors.faint, width=1)))
        p.setXRange(0.5, n + 0.5, padding=0)
        p.setYRange(-92, 92, padding=0)
        p.getAxis("bottom").setTicks([[(i + 1, str(i + 1)) for i in range(n)]])

    def _draw_sweep(self) -> None:
        sw = self.sweep_result
        if sw is None:
            return
        n = sw.fn.shape[1]
        own = (self.system.damping if sw.parameter is Parameter.DAMPER else self.system.stiffness)[sw.element]
        at = int(np.argmin(np.abs(sw.values - own)))
        unit = "N·s/m" if sw.parameter is Parameter.DAMPER else "N/m"
        name = f"{'c' if sw.parameter is Parameter.DAMPER else 'k'}{sw.element + 1}"
        dot = QtCore.Qt.PenStyle.DotLine
        p = self.locus
        p.clear()
        for k in range(n):
            colour = colors.mode[k % len(colors.mode)]
            lam = sw.poles[:, k]
            p.plot(lam.real, lam.imag / TWO_PI, pen=pg.mkPen(colour, width=2), connect="finite")
            if np.isfinite(lam[at]):
                p.addItem(
                    pg.ScatterPlotItem(
                        [lam[at].real], [lam[at].imag / TWO_PI], size=9, pen=None, brush=pg.mkBrush(colour)
                    )
                )
        real = np.concatenate(sw.roots) if sw.roots else np.zeros(0)
        if real.size:
            p.addItem(
                pg.ScatterPlotItem(
                    real, np.zeros(real.size), size=4, symbol="x", pen=None, brush=pg.mkBrush(colors.grey)
                )
            )
        p.addItem(pg.InfiniteLine(0.0, angle=90, pen=pg.mkPen(colors.faint, width=1)))
        finite = sw.poles[np.isfinite(sw.poles)]
        if finite.size:
            left = min(float(finite.real.min()), 0.0)
            p.setXRange(max(left, -8 * float(finite.imag.max())), 0.0, padding=0.03)
            p.setYRange(0.0, 1.05 * float(finite.imag.max()) / TWO_PI, padding=0)
        p.setTitle(f"Root locus as {name} goes from {sw.values[0]:.3g} to {sw.values[-1]:.3g} {unit}", size="9pt")

        z, m = self.sweep_zeta, self.sweep_mpd
        z.clear()
        m.clear()
        self.sweep_zeta_legend.clear()
        for k in range(n):
            colour = colors.mode[k % len(colors.mode)]
            curve = z.plot(sw.values, sw.zeta_exact[:, k], pen=pg.mkPen(colour, width=2), connect="finite")
            z.plot(
                sw.values,
                np.minimum(sw.zeta_approx[:, k], 2.0),
                pen=pg.mkPen(colour, width=1, style=QtCore.Qt.PenStyle.DashLine),
                connect="finite",
            )
            self.sweep_zeta_legend.addItem(curve, f"Mode {k + 1}")
            m.plot(sw.values, sw.mpd[:, k], pen=pg.mkPen(colour, width=2), connect="finite")
        self.sweep_zeta_legend.addItem(
            pg.PlotDataItem(pen=pg.mkPen(colors.grey, width=1, style=QtCore.Qt.PenStyle.DashLine)), "Approx."
        )
        for p in (z, m):
            if own > 0:
                p.addItem(pg.InfiniteLine(math.log10(own), angle=90, pen=pg.mkPen(colors.muted, width=1, style=dot)))
            p.setXRange(math.log10(sw.values[0]), math.log10(sw.values[-1]), padding=0)
        z.setYRange(0.0, 1.05, padding=0)
        m.setYRange(0.0, max(5.0, 1.1 * float(np.nanmax(sw.mpd, initial=0.0))), padding=0)
        m.setLabel("bottom", f"{name} ({unit})")
        lines = []
        if sw.parameter is Parameter.DAMPER:
            for k in range(n):
                value, zeta = sw.peak(k)
                if np.isfinite(zeta):
                    lines.append(f"mode {k + 1} ζ = {zeta:.3g} at {value:.3g}")
            self.sweep_note.setText(
                f"Each mode's damping peaks, then falls as {name} locks its joint: "
                + "; ".join(lines)
                + f" N·s/m. The chain's {name} is {own:.3g} N·s/m. The approximation (dashed) rises for ever."
            )
        else:
            mpd = np.where(np.isfinite(sw.mpd), sw.mpd, -1.0)
            v, worst = np.unravel_index(int(np.argmax(mpd)), mpd.shape)
            self.sweep_note.setText(
                f"Mode {worst + 1} is most complex (MPD {mpd[v, worst]:.3g}°) at {name} = {sw.values[v]:.3g} N/m; "
                f"the chain's {name} is {own:.3g} N/m. The springs move the modes; the dampers decide how much "
                "they couple them."
            )

    def _draw_identification(self) -> None:
        res = self.ident_result
        if res is None:
            return
        out = min(max(self.ident_out.currentIndex(), 0), res.H.shape[1] - 1)
        p = self.ident_plot
        p.clear()
        self.ident_legend.clear()
        meas = p.plot(res.freqs, np.maximum(np.abs(res.H[:, out]), 1e-300), pen=pg.mkPen(colors.grey, width=1))
        cf = p.plot(res.freqs, np.maximum(np.abs(res.complex_fit.H[:, out]), 1e-300), pen=pg.mkPen(colors.fit, width=2))
        rf = p.plot(
            res.freqs,
            np.maximum(np.abs(res.real_fit.H[:, out]), 1e-300),
            pen=pg.mkPen(colors.force, width=2, style=QtCore.Qt.PenStyle.DashLine),
        )
        self.ident_legend.addItem(meas, "Measured")
        self.ident_legend.addItem(cf, "Complex modes")
        self.ident_legend.addItem(rf, "Real modes")
        p.setLabel("left", f"|x{out + 1}/F{max(self.ident_in.currentIndex(), 0) + 1}| (m/N)")
        peak = float(np.max(np.abs(res.exact[:, out])))
        p.setYRange(math.log10(peak) - 3.5, math.log10(peak) + 0.4, padding=0)
        p.setXRange(0.0, res.freqs[-1], padding=0)

        t = self.ident_table
        t.clear()
        heads = ["Mode", "MAC(ψ) complex", "MAC(ψ) real", "MAC(φ) real", "MPC complex", "MPC exact"]
        t.setColumnCount(len(heads))
        t.setHorizontalHeaderLabels(heads)
        t.setRowCount(len(res.modes))
        for i, k in enumerate(res.modes):
            exact_mpc = self.rows[k].measures.mpc if self.rows[k].measures is not None else float("nan")
            cells = [
                f"{k + 1}",
                f"{res.complex_fit.mac[i]:.3f}",
                f"{res.real_fit.mac[i]:.3f}",
                f"{res.real_fit.mac_undamped[i]:.3f}",
                f"{complexity(res.complex_fit.modes[i].shape).mpc:.3f}",
                f"{exact_mpc:.3f}",
            ]
            for j, text in enumerate(cells):
                item = _cell(text)
                if j == 0:
                    item.setForeground(pg.mkColor(colors.mode[k % len(colors.mode)]))
                t.setItem(i, j, item)
        t.resizeColumnsToContents()
        worst = int(np.argmin(res.real_fit.mac)) if res.modes else 0
        self.ident_note.setText(
            f"The fits' FRFs miss the exact one by <b>{_share(res.complex_fit.error)} %</b> (complex modes) and "
            f"<b>{_share(res.real_fit.error)} %</b> (real modes). "
            + (
                f"The real modes' worst shape: mode {res.modes[worst] + 1}, MAC {res.real_fit.mac[worst]:.3f} with ψ."
                if res.modes
                else ""
            )
        )


def _share(x: float) -> str:
    """A fraction as a percentage, round-off shown as 0."""
    return f"{100 * x:.2g}" if x > 1e-8 else "0"


def _alpha(color: str, alpha: int):
    c = pg.mkColor(color)
    c.setAlpha(alpha)
    return c
