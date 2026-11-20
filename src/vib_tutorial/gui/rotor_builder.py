"""Rotor builder page: build a rotor from shaft sections, discs, bearings and seals, and see it whirl.

Left: Things to try, the selected part's properties (a section's length, diameters, material and elements; a
disc from the library or by its mass and inertias; a bearing's springs and dampers, oil film or magnet; a seal's
stiffness, damping and swirl), the model's settings (beam theory, horizontal or vertical, the speed) and what to
show (the mode or the unbalance response, and the node the Unbalance tab plots). Centre: the tools, the side view
(click the shaft to split it or to put a part there, drag a part along it), bending in the mode picked, and the
modes at the speed with their damping and whirl. Right: the mode whirling in 3D; the sweeps to the top speed
(core/rotor_builder_sweep.py), worked out when their tab first shows after a change (on a `Recompute`'s worker
after a control's change, the tab saying so and keeping its curves; at once for a trial or a practice setup): the Campbell diagram with
its critical speeds, the unbalance response's Bode plot, and the stability map; the global matrices with the
picked part's share tinted, the part's own matrices, and the Theory. The page has its own model (core/rotor_builder.py) and swings the mode on its own frame loop while it
shows.

The trials and practice problems are gui/rotor_builder_trials.py's (re-exported here, where the practice window and
the Theory tab look for them); the side view and the 3D whirl are gui/rotor_builder_views.py's `RotorCanvas` and
`WhirlView`.
"""

from __future__ import annotations

import dataclasses
import functools
import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.fluid_bearing import JournalBearing
from ..core.frame import MATERIALS
from ..core.practice import number
from ..core.rotor_builder import (
    BEARING_KINDS,
    DISC_LIBRARY,
    GRAVITY,
    MAX_ELEMENTS,
    RPM,
    THEORIES,
    TWO_PI,
    Assembly,
    Bearing,
    BearingKind,
    Disc,
    Part,
    Rotor,
    RotorError,
    Seal,
    Section,
    Selection,
    Theory,
    WhirlModes,
    assemble,
    dof_label,
    film_state,
    modes,
)
from ..core.rotor_builder_sweep import N_BRANCHES, Sweep, lag, orbit, sweep, unbalance_response
from . import rotor_builder_figures, settle_plots, weak
from .frame_builder import matrix_html
from .frame_loop import SWING_MS, FrameLoop
from .layouts import FitColumns, FlowLayout, group
from .lesson import (
    TAB_ROOM,
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
from .links import load_notes
from .recompute import Recompute
from .rotor_builder_trials import (
    EMPTY,
    MATRICES,
    SWINGS,
    TAB_ASSEMBLY,
    TAB_CAMPBELL,
    TAB_ELEMENT,
    TAB_STABILITY,
    TAB_UNBALANCE,
    TAB_WHIRL,
    Matrix,
    RotorTrial,
    Swing,
)
from .rotor_builder_trials import PROBLEMS as PROBLEMS
from .rotor_builder_trials import TRIALS as TRIALS
from .rotor_builder_views import TOOL_HINTS, TOOLS, RotorCanvas, Tool, WhirlView
from .style import colors
from .theming import add_legend, mute
from .transport import Transport

SWEEP_TABS = (TAB_CAMPBELL, TAB_UNBALANCE, TAB_STABILITY)
RESPONSE_SPEEDS = 241  # the unbalance response's speeds, 0 to the top one
WORKING = "Sweeping to the top speed…"  # a sweep's tab while its result is worked out on the worker
API_LOG_DEC = 0.1  # API 684's least log decrement at the running speed
SHOW_HZ = 0.6  # the animation's own pace: every mode whirls at it
SWING = 0.15  # the largest lateral motion drawn, as a fraction of the side view's height
MODE_ROWS = 20  # modes listed at most
GRIDS = (0.005, 0.01, 0.025, 0.05, 0.1)  # m
MATRIX_LIMIT = 40  # DOFs: beyond this the Assembly tab shows the nodes round the picked part


TIPS = {
    "length": "<p>The section's length along the shaft. Parts past it move with its end.</p>",
    "outer": "<p>The section's outside diameter: its I grows as D⁴, its mass as D².</p>",
    "inner": "<p>The section's bore (0: solid). A hollow shaft is stiffer for its mass.</p>",
    "material": "<p>Steel (210 GPa, 7850 kg/m³), aluminium (70 GPa, 2700) or titanium (114 GPa, 4430).</p>",
    "elements": "<p>Beam elements in the section (more where a part cuts it): more find the higher modes better.</p>",
    "z": "<p>Where the part sits along the shaft, from its left end.</p>",
    "library": "<p>A disc from the library, by material and size (its mass and inertias worked out from them), or "
    "Custom, by its mass and inertias.</p>",
    "disc_mass": "<p>The disc's mass, on its x and y.</p>",
    "ip": "<p>The polar moment of inertia, about the spin axis: the gyroscopic moment I<sub>p</sub>Ω × the tilt "
    "rate.</p>",
    "id": "<p>The diametral moment of inertia, about a diameter: on the slopes θx and θy.</p>",
    "diameter": "<p>The disc's diameter (drawn; from the library, it sets the inertias).</p>",
    "thickness": "<p>The disc's thickness (drawn; from the library, it sets the inertias).</p>",
    "bearing_kind": "<p>Springs and dampers to the ground; a plain journal bearing's oil film, its stiffness and "
    "damping changing with the speed and its load; or an active magnetic bearing under PID control.</p>",
    "kxx": "<p>The bearing's stiffness along x (sideways).</p>",
    "kyy": "<p>The bearing's stiffness along y (up).</p>",
    "kxy": "<p>Cross-coupled stiffness: a push along x from a displacement in y, f<sub>x</sub> = −k<sub>xy</sub> y. "
    "k<sub>xy</sub> = −k<sub>yx</sub> feeds a forward whirl.</p>",
    "kyx": "<p>Cross-coupled stiffness: f<sub>y</sub> = −k<sub>yx</sub> x.</p>",
    "cxx": "<p>The bearing's damping along x.</p>",
    "cyy": "<p>The bearing's damping along y.</p>",
    "film_length": "<p>The oil film's axial length L<sub>b</sub>.</p>",
    "film_diameter": "<p>The journal's diameter D<sub>b</sub> (the shaft's, where it sits).</p>",
    "clearance": "<p>The radial clearance c<sub>r</sub> between the journal and its bearing.</p>",
    "viscosity": "<p>The oil's viscosity η.</p>",
    "gap": "<p>Each magnet's air gap with the journal centred.</p>",
    "bias": "<p>The bias current in every coil.</p>",
    "kp": "<p>The PID's proportional gain: k<sub>i</sub>k<sub>p</sub> has to beat the magnets' k<sub>m</sub>.</p>",
    "ki": "<p>The PID's integral gain (0: none).</p>",
    "kd": "<p>The PID's derivative gain: the bearing's damping.</p>",
    "amplifier": "<p>The amplifier's bandwidth: its lag turns the damping negative above it.</p>",
    "seal_k": "<p>The seal's direct stiffness, on x and y.</p>",
    "seal_c": "<p>The seal's direct damping.</p>",
    "swirl": "<p>The fluid in the seal swirls round at λ times the shaft's speed, which gives it a cross-coupled "
    "stiffness k<sub>xy</sub> = λΩc, growing with the speed.</p>",
    "seal_mass": "<p>The fluid's added mass, on x and y.</p>",
    "theory": "<p>Timoshenko's beams shear, and their sections' turning has inertia (and spins, with its "
    "gyroscopic moment); Euler–Bernoulli's here keep the turning inertia (a Rayleigh beam) but don't shear.</p>",
    "gravity": "<p>A horizontal rotor's weight loads its bearings (an oil film's stiffness depends on it); a "
    "vertical one's doesn't, and its films run centred.</p>",
    "rpm": "<p>The speed the modes are worked out at: the gyroscopic matrix, an oil film's coefficients and a "
    "seal's cross-coupling all change with it.</p>",
    "grid": "<p>The spacing parts and cuts snap to.</p>",
    "unbalance": "<p>The disc's unbalance m e: its mass times how far its centre of mass is off the shaft's "
    "(1 g·mm = 10⁻⁶ kg·m). A library disc's is 10 µm of its mass.</p>",
    "heavy": "<p>Where the heavy spot is, as an angle from the shaft's keyphasor mark in the direction of spin.</p>",
    "internal": "<p>The shaft's internal damping, as its loss time τ<sub>i</sub> (its material's, or a shrink fit's "
    "slip): τ<sub>i</sub>K<sub>shaft</sub> on the "
    "bending rate in the turning shaft. It damps at rest, but above a critical speed it drives the forward "
    "whirl.</p>",
    "top_rpm": "<p>The top of the speed range the Campbell diagram, the unbalance response and the stability map "
    "sweep.</p>",
    "swing": "<p>What the side view and the Whirl tab show: the mode picked in the table, or the steady response "
    "to the discs' unbalance at the speed.</p>",
    "probe": "<p>The node whose orbit the Unbalance tab plots against the speed.</p>",
    "matrix": "<p>The matrix the Assembly and Element tabs show, at the speed.</p>",
    "animate": "<p>Whirl the rotor in the mode picked (slowed to the same pace whatever its frequency, and "
    "exaggerated). Space.</p>",
}
TABLE_TIP = (
    "<p>The modes at the speed: each one's damped frequency, its damping ratio and log decrement (negative: it "
    "grows, the rotor is unstable), and whether it whirls with the spin (forward) or against it. Click a row to "
    "show it.</p>"
)
ASSEMBLY_TIP = (
    "<p>The global matrix, node by node [x, θx, y, θy]: tinted, the picked part's share. A dot is an exact zero.</p>"
)

NOTES_HTML = load_notes("rotor_builder", trials=trials_html(TRIALS), figures=rotor_builder_figures.FIGURES)


def _tagged(change: int, job):
    """`job()` with the change it was asked for, so a result for a rotor since changed is dropped."""
    return change, job()


def _swept(assembly: Assembly, top_rpm: float) -> tuple[Sweep | None, str]:
    """The sweep, or why there's none: a pure function, run on the worker."""
    try:
        return sweep(assembly, top_rpm), ""
    except np.linalg.LinAlgError as error:
        return None, str(error)


def _responded(assembly: Assembly, rpm: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray] | None, str]:
    """The unbalance response at `rpm`, or why there's none."""
    try:
        return (rpm, unbalance_response(assembly, rpm)), ""
    except np.linalg.LinAlgError as error:
        return None, str(error)


def _whirl_word(w: float) -> str:
    return "forward" if w > 0.5 else "backward" if w < -0.5 else "mixed" if abs(w) > 0.05 else "straight"


# ------------------------------------------------------------------ the part's matrices as text
def _matrix(part: Part, which: Matrix, omega: float) -> np.ndarray:
    if which is Matrix.K:
        return part.stiffness(omega)
    if which is Matrix.C:
        return part.damping(omega)
    return part.M if which is Matrix.M else part.G


def _global(assembly: Assembly, which: Matrix, omega: float) -> np.ndarray:
    if which is Matrix.K:
        return assembly.stiffness(omega)
    if which is Matrix.C:
        return assembly.damping(omega)
    return assembly.M if which is Matrix.M else assembly.G


def part_html(rotor: Rotor, assembly: Assembly, selection: Selection, which: Matrix, omega: float) -> str:
    """The picked part: what it is, and its own share of the matrix, with the global DOFs it lands on."""
    kind, i = selection
    owned = [p for p in assembly.parts if p.owner == selection]
    if not owned:
        return "<p>Pick a section or a part (the Select tool) to see its matrices here.</p>"
    part = owned[0]
    lines = []
    if kind == "section":
        s = rotor.sections[i]
        lines.append(
            f"<p><b>Section {i + 1}</b>: {fmt(s.length, 'm')} of {s.material.name.lower()}, D = "
            f"{fmt(s.outer, 'm')}"
            + (f", bore d = {fmt(s.inner, 'm')}" if s.inner else ", solid")
            + f": A<sub>s</sub> = {number(s.area * 1e6, 4)} mm², I = {number(s.inertia * 1e12, 4)} mm⁴, EI = "
            f"{fmt(s.ei, 'N·m²')}, {fmt(s.mass, 'kg')}. In {len(owned)} elements; the first, "
            f"{fmt(part.length, 'm')} long, is shown"
        )
        lines.append(
            f" (φ<sub>s</sub> = 12EI/κ<sub>s</sub>G<sub>m</sub>A<sub>s</sub>L<sub>e</sub>² = {part.phi:.3g}, κ<sub>s</sub> = {s.kappa:.3f})."
            if rotor.theory is Theory.TIMOSHENKO
            else "."
        )
        lines.append(
            " Its G is twice its rotary inertia, coupling the x plane's slopes to the y plane's.</p>"
            if which is Matrix.G
            else "</p>"
        )
    elif kind == "disc":
        d = rotor.discs[i]
        lines.append(
            f"<p><b>Disc {i + 1}</b> ({d.name}) at z = {fmt(d.z, 'm')}: m = {fmt(d.mass, 'kg')}, I<sub>p</sub> = "
            f"{d.ip:.4g} kg·m², I<sub>d</sub> = {d.id:.4g} kg·m² (I<sub>p</sub>/I<sub>d</sub> = "
            f"{d.ip / d.id if d.id else math.inf:.3g}). Rigid: it adds to M on its node, and I<sub>p</sub> to G "
            "between the slopes.</p>"
        )
    elif kind == "bearing":
        b = rotor.bearings[i]
        lines.append(f"<p><b>Bearing {i + 1}</b> at z = {fmt(b.z, 'm')}: {b.kind.value.lower()}")
        if b.kind is BearingKind.FILM:
            w = assembly.loads[i]
            st = film_state(b.film, w, omega)
            lines.append(
                f", L<sub>b</sub> = {fmt(b.film.length, 'm')}, D<sub>b</sub> = {fmt(b.film.diameter, 'm')}, c<sub>r</sub> = {fmt(b.film.clearance, 'm')}, "
                f"η = {fmt(b.film.viscosity, 'Pa·s')}, under {fmt(w, 'N')} of the rotor's weight. At "
                f"{omega / RPM:.4g} rpm its journal sits at ε = {st.eccentricity:.3f}, {math.degrees(st.attitude):.3g}° "
                "from the load: its K and C are full, and change with the speed.</p>"
            )
        elif b.kind is BearingKind.MAGNET:
            m = b.magnet
            lines.append(
                f": k<sub>m</sub> = {fmt(m.negative_stiffness, 'N/m')} (on K's diagonal, negative), k<sub>i</sub> = "
                f"{fmt(m.current_gain, 'N/A')}; its PID's {m.n_states} states a side follow q and q̇ in the state: "
                f"k<sub>i</sub>k<sub>p</sub> = {fmt(m.current_gain * m.kp, 'N/m')} holds it.</p>"
            )
        else:
            lines.append(".</p>")
    else:
        seal = rotor.seals[i]
        lines.append(
            f"<p><b>Seal {i + 1}</b> at z = {fmt(seal.z, 'm')}: k = {fmt(seal.k, 'N/m')}, c = {fmt(seal.c, 'N·s/m')}, "
            "swirl "
            f"λ = {seal.swirl:.3g}: at {omega / RPM:.4g} rpm, k<sub>xy</sub> = λΩc = {fmt(seal.swirl * omega * seal.c, 'N/m')} "
            "(and k<sub>yx</sub> = −k<sub>xy</sub>), in K(Ω).</p>"
        )
    own = _matrix(part, which, omega)
    labels = [dof_label(d) for d in part.dofs]
    sym = which.name
    lines.append(f"<p>Its share of {sym}, added at {', '.join(labels)} (tinted on the Assembly tab):</p>")
    lines.append(matrix_html(own, labels, labels, title=sym))
    return "".join(lines)


# ------------------------------------------------------------------ the properties
class Properties(QtWidgets.QGroupBox):
    """The selected item's settings: a section's, a disc's, a bearing's or a seal's (or nothing)."""

    def __init__(self) -> None:
        super().__init__("Selected")
        v = QtWidgets.QVBoxLayout(self)
        self.empty = note_label("Nothing picked: pick the Select tool and click a section or a part.")
        v.addWidget(self.empty)

        self.section = Form("")
        self.section.setFlat(True)
        s = self.section
        self.length = s.number("Length:", 1.0, 10000.0, 250.0, 1, " mm", TIPS["length"])
        self.outer = s.number("Diameter D:", 1.0, 2000.0, 50.0, 1, " mm", TIPS["outer"])
        self.inner = s.number("Bore d:", 0.0, 1999.0, 0.0, 1, " mm", TIPS["inner"])
        self.material = combo([m.name for m in MATERIALS], TIPS["material"])
        s.add("Material:", self.material, TIPS["material"])
        self.elements = QtWidgets.QSpinBox()
        self.elements.setRange(1, MAX_ELEMENTS)
        self.elements.setKeyboardTracking(False)
        s.add("Elements:", self.elements, TIPS["elements"])
        self.section_info = note_label()
        s.row(self.section_info)
        self.add_section = QtWidgets.QPushButton("Add a section after the shaft's end")
        self.add_section.setToolTip("<p>A copy of the last section, added at the right-hand end.</p>")
        s.row(self.add_section)
        v.addWidget(self.section)

        self.disc = Form("")
        self.disc.setFlat(True)
        d = self.disc
        self.disc_z = d.number("At z:", 0.0, 100.0, 0.0, 3, " m", TIPS["z"])
        self.library = combo(["Custom", *[t.name for t in DISC_LIBRARY]], TIPS["library"])
        self.library.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        d.add("Library:", self.library, TIPS["library"])
        self.disc_mass = d.number("Mass m:", 0.0, 1e5, 10.0, 3, " kg", TIPS["disc_mass"])
        self.ip = d.number("Polar Ip:", 0.0, 1e5, 0.05, 5, " kg·m²", TIPS["ip"])
        self.id = d.number("Diametral Id:", 0.0, 1e5, 0.025, 5, " kg·m²", TIPS["id"])
        self.diameter = d.number("Diameter:", 1.0, 5000.0, 200.0, 1, " mm", TIPS["diameter"])
        self.thickness = d.number("Thickness:", 1.0, 2000.0, 30.0, 1, " mm", TIPS["thickness"])
        self.unbalance = d.number("Unbalance:", 0.0, 1e6, 0.0, 1, " g·mm", TIPS["unbalance"])
        self.heavy = d.number("Heavy spot at:", -360.0, 360.0, 0.0, 0, "°", TIPS["heavy"])
        v.addWidget(self.disc)

        self.bearing = Form("")
        self.bearing.setFlat(True)
        b = self.bearing
        self.bearing_z = b.number("At z:", 0.0, 100.0, 0.0, 3, " m", TIPS["z"])
        self.bearing_kind = combo([k.value for k in BEARING_KINDS], TIPS["bearing_kind"])
        b.add("Kind:", self.bearing_kind, TIPS["bearing_kind"])
        self.kxx = b.number("kxx:", -1e6, 1e6, 50.0, 3, " MN/m", TIPS["kxx"])
        self.kyy = b.number("kyy:", -1e6, 1e6, 50.0, 3, " MN/m", TIPS["kyy"])
        self.kxy = b.number("kxy:", -1e6, 1e6, 0.0, 3, " MN/m", TIPS["kxy"])
        self.kyx = b.number("kyx:", -1e6, 1e6, 0.0, 3, " MN/m", TIPS["kyx"])
        self.cxx = b.number("cxx:", 0.0, 1e6, 2.0, 3, " kN·s/m", TIPS["cxx"])
        self.cyy = b.number("cyy:", 0.0, 1e6, 2.0, 3, " kN·s/m", TIPS["cyy"])
        self.film_length = b.number("Film length L:", 1.0, 1000.0, 25.0, 1, " mm", TIPS["film_length"])
        self.film_diameter = b.number("Journal D:", 1.0, 2000.0, 50.0, 1, " mm", TIPS["film_diameter"])
        self.clearance = b.number("Clearance c:", 1.0, 5000.0, 50.0, 1, " µm", TIPS["clearance"])
        self.viscosity = b.number("Viscosity η:", 0.1, 10000.0, 30.0, 1, " mPa·s", TIPS["viscosity"])
        self.gap = b.number("Air gap:", 0.05, 20.0, 1.0, 2, " mm", TIPS["gap"])
        self.bias = b.number("Bias current:", 0.01, 100.0, 1.0, 2, " A", TIPS["bias"])
        self.kp = b.number("kp:", 0.0, 1e7, 3500.0, 0, " A/m", TIPS["kp"])
        self.ki = b.number("ki:", 0.0, 1e9, 35000.0, 0, " A/(m·s)", TIPS["ki"])
        self.kd = b.number("kd:", 0.0, 1e5, 5.0, 2, " A·s/m", TIPS["kd"])
        self.amplifier = b.number("Amplifier:", 1.0, 1e5, 1000.0, 0, " Hz", TIPS["amplifier"])
        self.bearing_info = note_label()
        b.row(self.bearing_info)
        v.addWidget(self.bearing)
        self.spring_rows = (self.kxx, self.kyy, self.kxy, self.kyx, self.cxx, self.cyy)
        self.film_rows = (self.film_length, self.film_diameter, self.clearance, self.viscosity)
        self.magnet_rows = (self.gap, self.bias, self.kp, self.ki, self.kd, self.amplifier)

        self.seal = Form("")
        self.seal.setFlat(True)
        e = self.seal
        self.seal_z = e.number("At z:", 0.0, 100.0, 0.0, 3, " m", TIPS["z"])
        self.seal_k = e.number("Stiffness k:", 0.0, 1e6, 2.0, 3, " MN/m", TIPS["seal_k"])
        self.seal_c = e.number("Damping c:", 0.0, 1e6, 2.0, 3, " kN·s/m", TIPS["seal_c"])
        self.swirl = e.number("Swirl λ:", 0.0, 1.0, 0.5, 2, "", TIPS["swirl"])
        self.seal_mass = e.number("Added mass:", 0.0, 1e4, 0.0, 3, " kg", TIPS["seal_mass"])
        v.addWidget(self.seal)

        self.remove = QtWidgets.QPushButton("Delete")
        self.remove.setToolTip("<p>Remove the picked section or part. Del does it too.</p>")
        v.addWidget(self.remove)
        self.forms = (self.empty, self.section, self.disc, self.bearing, self.seal)
        self.show_form(self.empty)

    def show_form(self, form: QtWidgets.QWidget) -> None:
        for w in self.forms:
            w.setVisible(w is form)

    def current(self) -> QtWidgets.QWidget:
        return next(w for w in self.forms if not w.isHidden())

    def show_kind(self, kind: BearingKind) -> None:
        for rows, on in (
            (self.spring_rows, kind is BearingKind.SPRINGS),
            (self.film_rows, kind is BearingKind.FILM),
            (self.magnet_rows, kind is BearingKind.MAGNET),
        ):
            for box in rows:
                self.bearing.form.setRowVisible(box, on)


def _hz(f: float) -> str:
    """Four significant figures, trailing zeros kept (171.0, not 171)."""
    return f"{f:#.4g}".rstrip(".")


def _count(n: int, thing: str) -> str:
    return f"{n} {thing}" + ("" if n == 1 else "s")


def _percent(zeta: float) -> str:
    return f"{100 * zeta:.3g} %"


# ------------------------------------------------------------------ the page
class RotorBuilderPage(QtWidgets.QWidget):
    """Controls on the left, the side view and its modes in the centre, the 3D whirl and the matrices on the
    right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def transport(self) -> Transport:
        """The run bar drives the rotor's whirl."""
        return Transport(self.run_button, self.reset_button)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.rotor = EMPTY
        self.assembly: Assembly | None = None
        self.result: WhirlModes | None = None
        self.error = ""
        self.mode = 0
        self.picked: Selection | None = None  # the part on the Element tab
        self._probe: int | None = None  # the node the Unbalance tab plots
        self._steady: tuple[float, np.ndarray] | None = None  # (Ω, the unbalance response there)
        self.running = True  # a mode whirls on its own
        self._phase = 0.0
        self._history: list[Rotor] = []
        self._future: list[Rotor] = []

        self.trials = TrialBox(
            [t.name for t in TRIALS],
            "<p>Each loads a rotor, to show one idea from the Theory tab. It's then yours to change.</p>",
            predictions=[t.predict for t in TRIALS],
        )
        self.trials.picked.connect(self.load_trial)

        self.props = Properties()
        p = self.props
        for box in (p.length, p.outer, p.inner, p.elements):
            box.editingFinished.connect(self._on_section_box)
        p.material.currentIndexChanged.connect(self._on_section_box)
        p.add_section.clicked.connect(self._on_add_section)
        for box in (p.disc_z, p.disc_mass, p.ip, p.id, p.diameter, p.thickness, p.unbalance, p.heavy):
            box.editingFinished.connect(self._on_disc_box)
        p.library.currentIndexChanged.connect(self._on_library)
        for box in (p.bearing_z, *p.spring_rows, *p.film_rows, *p.magnet_rows):
            box.editingFinished.connect(self._on_bearing_box)
        p.bearing_kind.currentIndexChanged.connect(self._on_bearing_box)
        for box in (p.seal_z, p.seal_k, p.seal_c, p.swirl, p.seal_mass):
            box.editingFinished.connect(self._on_seal_box)
        p.remove.clicked.connect(weak.slot(self, lambda self: self.canvas.delete_selected()))

        model = Form("Model")
        self.theory = combo([t.value for t in THEORIES], TIPS["theory"])
        model.add("Beams:", self.theory, TIPS["theory"])
        self.gravity = combo(["Horizontal", "Vertical"], TIPS["gravity"])
        model.add("Rotor:", self.gravity, TIPS["gravity"])
        self.internal = model.number("Internal damping τ<sub>i</sub>:", 0.0, 1e4, 0.0, 2, " µs", TIPS["internal"])
        self.rpm = model.number("Speed:", 0.0, 100000.0, 0.0, 0, " rpm", TIPS["rpm"])
        self.top_rpm = model.number("Sweep to:", 100.0, 100000.0, 12000.0, 0, " rpm", TIPS["top_rpm"])
        self.grid = combo([f"{1000 * g:g} mm" for g in GRIDS], TIPS["grid"])
        model.add("Grid:", self.grid, TIPS["grid"])
        self.theory.currentIndexChanged.connect(self._on_model)
        self.gravity.currentIndexChanged.connect(self._on_model)
        self.internal.editingFinished.connect(self._on_model)
        self.top_rpm.valueChanged.connect(self._on_top)
        self.rpm.valueChanged.connect(self._on_speed)
        self.grid.currentIndexChanged.connect(self._on_grid)

        show = Form("Show")
        self.matrix = combo([m.value for m in MATRICES], TIPS["matrix"])
        show.add("Matrix:", self.matrix, TIPS["matrix"])
        self.swing = combo([s.value for s in SWINGS], TIPS["swing"])
        show.add("Whirl in:", self.swing, TIPS["swing"])
        self.probe = QtWidgets.QSpinBox()
        self.probe.setRange(1, 1)
        self.probe.setKeyboardTracking(False)
        show.add("Response at node:", self.probe, TIPS["probe"])
        self.swing.currentIndexChanged.connect(self._on_swing)
        self.probe.valueChanged.connect(self._on_probe)
        self.run_button = RunButton()
        self.run_button.set_running(True)
        self.run_button.setToolTip(TIPS["animate"])
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.setToolTip("<p>Back to the start of the whirl.</p>")
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.reset_button)
        show.row(buttons)
        self.matrix.currentIndexChanged.connect(self._draw_tabs)
        self.run_button.toggled.connect(self._on_animate)
        self.reset_button.clicked.connect(self.reset)

        # --- centre: the tools, the side view, the modes
        self.canvas: RotorCanvas = RotorCanvas()
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
        self.disc_kind = combo(["Custom", *[t.name for t in DISC_LIBRARY]], TIPS["library"])
        self.disc_kind.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.disc_kind.setCurrentIndex(1)
        self.disc_kind.currentIndexChanged.connect(self._on_disc_kind)
        self.bearing_tool_kind = combo([k.value for k in BEARING_KINDS], TIPS["bearing_kind"])
        self.bearing_tool_kind.currentIndexChanged.connect(self._on_bearing_tool_kind)
        self.undo_button = QtWidgets.QToolButton()
        self.undo_button.setText("Undo")
        self.undo_button.setToolTip("<p>Undo the last change (Ctrl+Z).</p>")
        self.redo_button = QtWidgets.QToolButton()
        self.redo_button.setText("Redo")
        self.redo_button.setToolTip("<p>Redo it (Ctrl+Shift+Z).</p>")
        self.clear_button = QtWidgets.QToolButton()
        self.clear_button.setText("Clear")
        self.clear_button.setToolTip("<p>Start again from a bare shaft (Undo brings it back).</p>")
        self.undo_button.clicked.connect(self.undo)
        self.redo_button.clicked.connect(self.redo)
        self.clear_button.clicked.connect(
            weak.slot(self, lambda self: self._set_rotor(Rotor(theory=self.rotor.theory, gravity=self.rotor.gravity)))
        )
        flow.add(group(self.disc_kind), group(self.bearing_tool_kind))
        flow.add(self.undo_button, self.redo_button, self.clear_button)
        for keys, slot in (("Ctrl+Z", self.undo), ("Ctrl+Shift+Z", self.redo), ("Ctrl+Y", self.redo)):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(keys), self)
            shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)
        self.hint = note_label(TOOL_HINTS[Tool.SELECT])

        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setToolTip(TABLE_TIP)
        self.table.setHorizontalHeaderLabels(["Mode", "f (Hz)", "ζ", "Log dec δ", "Whirl"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        FitColumns(self.table)
        self.table.itemSelectionChanged.connect(self._on_row)
        self.readout = readout_label()

        # --- right
        self.whirl = WhirlView()
        self.whirl_note = note_label()
        self.assembly_view = QtWidgets.QTextBrowser()
        self.assembly_view.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.NoWrap)
        self.assembly_view.setToolTip(ASSEMBLY_TIP)
        self.assembly_note = note_label()
        self.element_view = QtWidgets.QTextBrowser()
        self.notes = theory(NOTES_HTML)

        self.campbell_note = note_label()
        self.campbell = plot("Frequency", "Hz", "Speed (rpm)", None, height=260)
        self.stability_note = note_label()
        self.stability = plot("Log decrement δ", None, "Speed (rpm)", None, height=260)
        self.unbalance_note = note_label()
        self.bode = plot("Orbit", "µm", "Speed (rpm)", None, log_y=True)
        self.phase = plot("Lag", "°", "Speed (rpm)", None)
        legend = add_legend(self.campbell, offset=(10, 10))
        self.curves: dict[str, pg.PlotCurveItem] = {}
        for name, p in (
            ("forward", self.campbell),
            ("backward", self.campbell),
            ("one_x", self.campbell),
            ("d_forward", self.stability),
            ("d_backward", self.stability),
            ("orbit", self.bode),
            ("lag", self.phase),
        ):
            # a PlotDataItem on the log axis: a bare PlotCurveItem ignores the log mode
            self.curves[name] = pg.PlotDataItem() if name == "orbit" else pg.PlotCurveItem(connect="finite")
            p.addItem(self.curves[name])
        legend.addItem(self.curves["forward"], "Forward whirl")
        legend.addItem(self.curves["backward"], "Backward whirl")
        legend.addItem(self.curves["one_x"], "1X: the shaft's speed")
        self.criticals = pg.PlotDataItem(pen=None, symbol="o", symbolSize=10)
        self.damped_criticals = pg.PlotDataItem(pen=None, symbol="o", symbolSize=10)
        self.campbell.addItem(self.criticals)
        self.campbell.addItem(self.damped_criticals)
        self.speed_lines = [pg.InfiniteLine(angle=90) for _ in range(4)]
        for line, p in zip(self.speed_lines, (self.campbell, self.stability, self.bode, self.phase)):
            p.addItem(line, ignoreBounds=True)
        self.zero_line = pg.InfiniteLine(angle=0, pos=0.0)
        self.api_line = pg.InfiniteLine(angle=0, pos=API_LOG_DEC)
        self.onset_line = pg.InfiniteLine(angle=90)
        for line in (self.zero_line, self.api_line, self.onset_line):
            self.stability.addItem(line, ignoreBounds=True)
        self.critical_lines = [pg.InfiniteLine(angle=90) for _ in range(6)]
        for line in self.critical_lines:
            self.bode.addItem(line, ignoreBounds=True)
        self.sweep_result: Sweep | None = None  # worked out when a sweep's tab shows (None: not since a change)
        self.response: np.ndarray | None = None  # the unbalance response at RESPONSE_SPEEDS speeds
        self.response_rpm = np.zeros(0)
        # The sweeps on the worker after a control's change (~200 ms; #409 P-8), at once for a trial or a setup
        # (`_now`). `_change` counts the changes that make them stale: a result for an older one is dropped.
        self.sweeper = Recompute(self._on_sweep, self)
        self.responder = Recompute(self._on_responses, self)
        self._change = 0
        self._now = 0
        self._asked = {"sweep": -1, "response": -1}  # the change each was last asked for
        self._got = {"sweep": -1, "response": -1}  # and the change its last result was for

        self.tabs = QtWidgets.QTabWidget()
        whirl_tab = QtWidgets.QWidget()
        wl = QtWidgets.QVBoxLayout(whirl_tab)
        wl.addWidget(self.whirl_note)
        wl.addWidget(self.whirl, 1)
        self.tabs.addTab(whirl_tab, "Whirl")
        self.tabs.addTab(column(self.campbell_note, self.campbell, stretch=False), "Campbell")
        self.tabs.addTab(column(self.unbalance_note, self.bode, self.phase, stretch=False), "Unbalance")
        self.tabs.addTab(column(self.stability_note, self.stability, stretch=False), "Stability")
        self.tabs.addTab(column(self.assembly_note, self.assembly_view, stretch=False), "Assembly")
        self.tabs.addTab(self.element_view, "Element")
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
        three_panes(self, column(self.trials, self.props, model, show), centre, self.tabs, sizes=TAB_ROOM)

        self.loop = FrameLoop(self, "Rotor builder", self._tick, self._runs, self._sim_t, frame_ms=SWING_MS)
        self._on_tool(0)
        self._on_disc_kind(self.disc_kind.currentIndex())
        self.apply_trial(TRIALS[0], keep_history=False)
        self.apply_theme()

    # ------------------------------------------------------------- trials and practice
    def load_trial(self, index: int) -> None:
        self.apply_trial(TRIALS[index])

    def load_setup(self, setup: dict) -> None:
        """Set the page up as a practice problem's `setup` says (a trial's fields, with no note)."""
        self.apply_trial(RotorTrial("", "", **setup))

    def practice_state(self) -> RotorTrial:
        """The page's settings, as a trial: what a practice spec checks."""
        return RotorTrial(
            "",
            "",
            rotor=self.rotor,
            rpm=self.rpm.value(),
            mode=self.mode,
            tab=self.tabs.currentIndex(),
            picked=self.picked,
            matrix=MATRICES[self.matrix.currentIndex()],
            grid=self.canvas.grid,
            top_rpm=self.top_rpm.value(),
            probe=self.probe.value() - 1,
            swing=SWINGS[self.swing.currentIndex()],
        )

    def apply_trial(self, t: RotorTrial, keep_history: bool = True) -> None:
        self._now += 1  # a trial's sweeps at once: the reader waits for the whole setup (and tests read it)
        try:
            self._apply(t, keep_history)
        finally:
            self._now -= 1

    def _apply(self, t: RotorTrial, keep_history: bool) -> None:
        set_quietly(self.matrix, MATRICES.index(t.matrix))
        set_quietly(self.rpm, t.rpm)
        set_quietly(self.top_rpm, t.top_rpm)
        set_quietly(self.swing, SWINGS.index(t.swing))
        self._probe = t.probe
        self.canvas.grid = t.grid
        set_quietly(self.grid, int(np.argmin([abs(g - t.grid) for g in GRIDS])))
        self.mode = t.mode
        self.picked = t.picked
        self.trials.show_note(t.note)
        self._set_rotor(t.rotor, record=keep_history, selection=t.picked)
        self.tabs.setCurrentIndex(t.tab)
        self._phase = 0.0
        self.running = True  # a trial starts it
        self.run_button.set_running(True)
        self._show_mode()
        self.loop.start()

    # ------------------------------------------------------------- editing
    @property
    def omega(self) -> float:
        return self.rpm.value() * RPM

    def _set_rotor(
        self, rotor: Rotor, record: bool = True, selection: Selection | None = None, fit: bool = True
    ) -> None:
        if record and rotor != self.rotor:
            self._history.append(self.rotor)
            self._future.clear()
        self.rotor = rotor
        set_quietly(self.theory, THEORIES.index(rotor.theory))
        set_quietly(self.gravity, 0 if rotor.gravity else 1)
        set_quietly(self.internal, rotor.internal * 1e6)
        self.sweep_result, self.response, self._steady = None, None, None
        self._change += 1
        self._fit_probe()
        try:
            self.assembly = assemble(rotor)
            self.error = ""
        except RotorError as error:
            self.assembly, self.error = None, str(error)
        if selection is not None:
            kind, i = selection
            if i >= (len(rotor.sections) if kind == "section" else len(rotor.parts(kind))):
                selection = None
        self.canvas.selection = selection
        self.picked = selection if selection is not None else self.picked
        if self.picked is not None:
            kind, i = self.picked
            if i >= (len(rotor.sections) if kind == "section" else len(rotor.parts(kind))):
                self.picked = None
        self.canvas.set_rotor(rotor, fit=fit)
        self._fill_props()
        self._work()
        self.undo_button.setEnabled(bool(self._history))
        self.redo_button.setEnabled(bool(self._future))

    def _work(self) -> None:
        """The modes at the speed, and everything drawn from them."""
        self.result = None
        if self.assembly is not None:
            try:
                self.result = modes(self.assembly, self.omega)
            except (RotorError, np.linalg.LinAlgError) as error:
                self.error = str(error)
        count = self.result.hz.size if self.result is not None else 0
        self.mode = min(self.mode, max(count - 1, 0))
        self._fill_table()
        self._fill_readout()
        self._draw_tabs()
        self._show_mode()
        self.loop.start()

    def undo(self) -> None:
        if self._history:
            self._future.append(self.rotor)
            self._set_rotor(self._history.pop(), record=False)

    def redo(self) -> None:
        if self._future:
            self._history.append(self.rotor)
            self._set_rotor(self._future.pop(), record=False)

    def _on_edit(self, rotor: Rotor, selection) -> None:
        self._set_rotor(rotor, selection=selection, fit=rotor.length != self.rotor.length)

    def _on_selected(self, selection) -> None:
        if selection is not None:
            self.picked = selection
        self._fill_props()
        self._draw_tabs()

    def _on_tool(self, index: int) -> None:
        tool = TOOLS[index]
        self.canvas.set_tool(tool)
        self.disc_kind.setVisible(tool is Tool.DISC)
        self.bearing_tool_kind.setVisible(tool is Tool.BEARING)
        self.hint.setText(TOOL_HINTS[tool])

    def _on_disc_kind(self, index: int) -> None:
        if index > 0:
            self.canvas.template["disc"] = DISC_LIBRARY[index - 1].at(0.0)

    def _on_bearing_tool_kind(self, index: int) -> None:
        b = self.canvas.template["bearing"]
        assert isinstance(b, Bearing), "the bearing tool's template"
        self.canvas.template["bearing"] = dataclasses.replace(b, kind=BEARING_KINDS[index])

    def _on_grid(self, index: int) -> None:
        self.canvas.grid = GRIDS[index]
        self.canvas.draw()

    def _on_model(self) -> None:
        rotor = dataclasses.replace(
            self.rotor,
            theory=THEORIES[self.theory.currentIndex()],
            gravity=GRAVITY if self.gravity.currentIndex() == 0 else 0.0,
            internal=self.internal.value() * 1e-6,
        )
        if rotor == self.rotor:
            return
        self._set_rotor(rotor, selection=self.canvas.selection, fit=False)

    def _on_speed(self) -> None:
        self._work()

    def _on_top(self) -> None:
        self.sweep_result, self.response = None, None
        self._change += 1
        self._draw_tabs()

    def _fit_probe(self) -> None:
        """The probe's box sized to the nodes, on the node asked for (`_probe`), else the first disc's."""
        r = self.rotor
        nodes = max(r.nodes.size, 1)
        node = self._probe
        if node is None or node >= nodes:
            node = r.node_at(r.discs[0].z) if r.discs else nodes // 2
        self._probe = node
        self.probe.blockSignals(True)
        self.probe.setMaximum(nodes)
        self.probe.setValue(node + 1)
        self.probe.blockSignals(False)

    def _on_probe(self) -> None:
        self._probe = self.probe.value() - 1
        if self.tabs.currentIndex() == TAB_UNBALANCE:
            self._draw_unbalance()

    def _on_swing(self) -> None:
        self._show_mode()
        if self.tabs.currentIndex() == TAB_WHIRL:
            self._draw_whirl()

    def _selected(self, kind: str) -> int | None:
        sel = self.canvas.selection
        return sel[1] if sel is not None and sel[0] == kind else None

    def _on_section_box(self) -> None:
        k = self._selected("section")
        if k is None:
            return
        p = self.props
        outer = p.outer.value() / 1000
        s = Section(
            p.length.value() / 1000,
            outer,
            min(p.inner.value() / 1000, 0.95 * outer),
            MATERIALS[p.material.currentIndex()],
            p.elements.value(),
        )
        if s != self.rotor.sections[k]:
            new = self.rotor.replace_section(k, s)
            self._set_rotor(new, selection=("section", k), fit=new.length != self.rotor.length)

    def _on_add_section(self) -> None:
        self._set_rotor(self.rotor.with_section(), selection=("section", len(self.rotor.sections)))

    def _clamp(self, z: float) -> float:
        return float(np.clip(z, 0.0, self.rotor.length))

    def _on_disc_box(self) -> None:
        i = self._selected("disc")
        if i is None:
            return
        p = self.props
        d = Disc(
            self._clamp(p.disc_z.value()),
            p.disc_mass.value(),
            p.ip.value(),
            p.id.value(),
            p.diameter.value() / 1000,
            p.thickness.value() / 1000,
            unbalance=p.unbalance.value() * 1e-6,
            unbalance_angle=math.radians(p.heavy.value()),
        )
        old = self.rotor.discs[i]
        if (d.mass, d.ip, d.id, d.diameter, d.thickness) == (old.mass, old.ip, old.id, old.diameter, old.thickness):
            d = dataclasses.replace(d, name=old.name)
        if d != old:
            self.canvas.template["disc"] = d
            self._set_rotor(self.rotor.replace_part("disc", i, d), selection=("disc", i), fit=False)

    def _on_library(self, index: int) -> None:
        i = self._selected("disc")
        if i is None or index == 0:
            return
        old = self.rotor.discs[i]
        d = DISC_LIBRARY[index - 1].at(old.z)
        self.canvas.template["disc"] = d
        self._set_rotor(self.rotor.replace_part("disc", i, d), selection=("disc", i), fit=False)

    def _on_bearing_box(self) -> None:
        i = self._selected("bearing")
        if i is None:
            return
        p = self.props
        old = self.rotor.bearings[i]
        kind = BEARING_KINDS[p.bearing_kind.currentIndex()]
        z = self._clamp(p.bearing_z.value())
        film = JournalBearing(
            p.film_length.value() / 1000,
            p.film_diameter.value() / 1000,
            p.clearance.value() * 1e-6,
            p.viscosity.value() / 1000,
            0.0,
        )
        if kind is BearingKind.FILM and old.kind is not BearingKind.FILM:
            film = dataclasses.replace(film, diameter=self.rotor.outer_at(z))
        try:
            magnet = dataclasses.replace(
                old.magnet,
                gap=p.gap.value() / 1000,
                bias=p.bias.value(),
                kp=p.kp.value(),
                ki=p.ki.value(),
                kd=p.kd.value(),
                amplifier_hz=p.amplifier.value(),
            )
        except ValueError:
            magnet = old.magnet
        b = Bearing(
            z,
            kind,
            p.kxx.value() * 1e6,
            p.kyy.value() * 1e6,
            p.kxy.value() * 1e6,
            p.kyx.value() * 1e6,
            p.cxx.value() * 1e3,
            p.cyy.value() * 1e3,
            film,
            magnet,
        )
        if b != old:
            other = self.rotor.part_at("bearing", z)
            if other is not None and other != i:  # two bearings can't share a place
                b = dataclasses.replace(b, z=old.z)
            self.canvas.template["bearing"] = b
            self._set_rotor(self.rotor.replace_part("bearing", i, b), selection=("bearing", i), fit=False)

    def _on_seal_box(self) -> None:
        i = self._selected("seal")
        if i is None:
            return
        p = self.props
        s = Seal(
            self._clamp(p.seal_z.value()),
            p.seal_k.value() * 1e6,
            p.seal_c.value() * 1e3,
            p.swirl.value(),
            p.seal_mass.value(),
        )
        if s != self.rotor.seals[i]:
            self.canvas.template["seal"] = s
            self._set_rotor(self.rotor.replace_part("seal", i, s), selection=("seal", i), fit=False)

    # ------------------------------------------------------------- filling in
    def _fill_props(self) -> None:
        p = self.props
        sel = self.canvas.selection
        if sel is None:
            p.show_form(p.empty)
            p.remove.setEnabled(False)
            return
        kind, i = sel
        p.remove.setEnabled(kind != "section" or len(self.rotor.sections) > 1)
        top = self.rotor.length
        for at in (p.disc_z, p.bearing_z, p.seal_z):
            at.setMaximum(top)
        if kind == "section":
            s = self.rotor.sections[i]
            for box, value in (
                (p.length, s.length * 1000),
                (p.outer, s.outer * 1000),
                (p.inner, s.inner * 1000),
                (p.material, [m.name for m in MATERIALS].index(s.material.name)),
                (p.elements, s.elements),
            ):
                set_quietly(box, value)
            a = self.rotor.ends[i]
            p.section_info.setText(
                f"From z = {a:.4g} to {a + s.length:.4g} m: A = {number(s.area * 1e6, 4)} mm², I = "
                f"{number(s.inertia * 1e12, 4)} mm⁴, {s.mass:.4g} kg, L/D = {s.length / s.outer:.3g}."
            )
            p.section.setTitle(f"Section {i + 1}")
            p.show_form(p.section)
        elif kind == "disc":
            d = self.rotor.discs[i]
            names = [t.name for t in DISC_LIBRARY]
            for box, value in (
                (p.disc_z, d.z),
                (p.library, names.index(d.name) + 1 if d.name in names else 0),
                (p.disc_mass, d.mass),
                (p.ip, d.ip),
                (p.id, d.id),
                (p.diameter, d.diameter * 1000),
                (p.thickness, d.thickness * 1000),
                (p.unbalance, d.unbalance * 1e6),
                (p.heavy, math.degrees(d.unbalance_angle)),
            ):
                set_quietly(box, value)
            p.disc.setTitle(f"Disc {i + 1}")
            p.show_form(p.disc)
        elif kind == "bearing":
            b = self.rotor.bearings[i]
            for box, value in (
                (p.bearing_z, b.z),
                (p.bearing_kind, BEARING_KINDS.index(b.kind)),
                (p.kxx, b.kxx / 1e6),
                (p.kyy, b.kyy / 1e6),
                (p.kxy, b.kxy / 1e6),
                (p.kyx, b.kyx / 1e6),
                (p.cxx, b.cxx / 1e3),
                (p.cyy, b.cyy / 1e3),
                (p.film_length, b.film.length * 1000),
                (p.film_diameter, b.film.diameter * 1000),
                (p.clearance, b.film.clearance * 1e6),
                (p.viscosity, b.film.viscosity * 1000),
                (p.gap, b.magnet.gap * 1000),
                (p.bias, b.magnet.bias),
                (p.kp, b.magnet.kp),
                (p.ki, b.magnet.ki),
                (p.kd, b.magnet.kd),
                (p.amplifier, b.magnet.amplifier_hz),
            ):
                set_quietly(box, value)
            p.show_kind(b.kind)
            p.bearing_info.setText(self._bearing_info(i))
            p.bearing.setTitle(f"Bearing {i + 1}")
            p.show_form(p.bearing)
        else:
            seal = self.rotor.seals[i]
            for box, value in (
                (p.seal_z, seal.z),
                (p.seal_k, seal.k / 1e6),
                (p.seal_c, seal.c / 1e3),
                (p.swirl, seal.swirl),
                (p.seal_mass, seal.mass),
            ):
                set_quietly(box, value)
            p.seal.setTitle(f"Seal {i + 1}")
            p.show_form(p.seal)

    def _bearing_info(self, i: int) -> str:
        b = self.rotor.bearings[i]
        if b.kind is BearingKind.FILM:
            if self.assembly is None:
                return ""
            w = self.assembly.loads[i]
            st = film_state(b.film, w, self.omega)
            return f"Load {w:.4g} N; at the speed, ε = {st.eccentricity:.3f}."
        if b.kind is BearingKind.MAGNET:
            m = b.magnet
            margin = m.current_gain * m.kp - m.negative_stiffness
            return (
                f"k<sub>m</sub> = {fmt(m.negative_stiffness, 'N/m')}, k<sub>i</sub>k<sub>p</sub> − k<sub>m</sub> = "
                f"{fmt(margin, 'N/m')}" + (" (too weak: it can't hold the journal)." if margin <= 0 else ".")
            )
        return ""

    def _fill_table(self) -> None:
        self.table.blockSignals(True)
        r = self.result
        count = min(r.hz.size, MODE_ROWS) if r is not None else 0
        self.table.setRowCount(count)
        for i in range(count):
            assert r is not None
            texts = [str(i + 1), _hz(r.hz[i]), _percent(r.zeta[i]), f"{r.log_dec[i]:.3g}", _whirl_word(r.whirl[i])]
            for col, text in enumerate(texts):
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                if col in (2, 3) and r.zeta[i] < 0:
                    item.setForeground(QtGui.QColor(colors.force))
                self.table.setItem(i, col, item)
        if count:
            self.table.selectRow(min(self.mode, count - 1))
        self.table.blockSignals(False)
        rows = sum(self.table.rowHeight(i) for i in range(min(count, 5))) if count else 0
        self.table.setFixedHeight(self.table.horizontalHeader().height() + rows + 2 * self.table.frameWidth() + 2)

    def _fill_readout(self) -> None:
        r = self.rotor
        text = (
            f"{_count(len(r.sections), 'section')}, {_count(len(r.discs), 'disc')}, "
            f"{_count(len(r.bearings), 'bearing')}, {_count(len(r.seals), 'seal')}: {r.nodes.size} nodes, "
            f"{r.n_dof} DOFs, {r.mass:.4g} kg"
        )
        a = self.assembly
        if a is not None and a.n_control:
            text += f" and {a.n_control} controller states"
        m = self.result
        if m is not None:
            text += f". At {self.rpm.value():.5g} rpm: {m.hz.size} whirl modes"
            if m.rigid:
                text += f", <b>{m.rigid} roots at 0</b> (it isn't held: a rigid body)"
            if m.roots.size:
                text += f", {m.roots.size} real roots"
            unstable = int((m.zeta < 0).sum())
            if unstable:
                text += f". <span style='color:{colors.force}'><b>{unstable} growing: unstable.</b></span>"
            else:
                text += "."
        elif self.error:
            text += f". <span style='color:{colors.force}'>{self.error}</span>"
        self.readout.setText(text)

    def _draw_tabs(self) -> None:
        i = self.tabs.currentIndex()
        if i == TAB_WHIRL:
            self._draw_whirl()
        elif i == TAB_CAMPBELL:
            self._draw_campbell()
        elif i == TAB_UNBALANCE:
            self._draw_unbalance()
        elif i == TAB_STABILITY:
            self._draw_stability()
        elif i == TAB_ASSEMBLY:
            self._draw_assembly()
        elif i == TAB_ELEMENT:
            self._draw_element()

    def _draw_whirl(self) -> None:
        if SWINGS[self.swing.currentIndex()] is Swing.UNBALANCE:
            self._draw_steady()
            return
        m = self.result
        if m is None or self.mode >= m.hz.size:
            self.whirl.set_mode(self.rotor, None)
            self.whirl_note.setText(self.error or "No whirl modes.")
            return
        w = float(m.whirl[self.mode])
        self.whirl.set_mode(self.rotor, m.shapes[:, self.mode])
        self.whirl.show_phase(self._phase)
        stable = "decays" if m.zeta[self.mode] >= 0 else "<b>grows</b>"
        self.whirl_note.setText(
            f"Mode {self.mode + 1} at {self.rpm.value():.5g} rpm: {m.hz[self.mode]:.4g} Hz, ζ = "
            f"{_percent(m.zeta[self.mode])} (it {stable}), whirling {_whirl_word(w)} (whirl ratio {w:+.2f}: +1 "
            "a circle with the spin, −1 against it, 0 a straight line)."
        )

    def _draw_steady(self) -> None:
        Q = self.steady()
        if Q is None or not np.abs(Q).any():
            self.whirl.set_mode(self.rotor, None)
            self.whirl_note.setText(
                self.error
                or (
                    "Nothing turns at 0 rpm: set a speed."
                    if self.omega == 0
                    else "No disc has an unbalance: select one and give it some (Unbalance, on the left)."
                )
            )
            return
        size = orbit(Q, np.arange(self.rotor.nodes.size))
        k = int(np.argmax(size))
        self.whirl.set_mode(self.rotor, Q / size[k])
        self.whirl.show_phase(self._phase)
        self.whirl_note.setText(
            f"The steady response to the discs' unbalance at {self.rpm.value():.5g} rpm (once a revolution, so at "
            f"{self.rpm.value() / 60:.4g} Hz): the largest orbit is {fmt(size[k], 'm')} across its major semi-axis, "
            f"at node {k + 1} (z = {self.rotor.nodes[k]:.3g} m)."
        )

    # ------------------------------------------------------------- the sweeps
    def _sweep(self) -> Sweep | None:
        """The Campbell diagram and the stability map: worked out if they aren't since the last change, on the
        worker unless a trial is being set up (None meanwhile: `_waiting`)."""
        if self.sweep_result is None and self.assembly is not None:
            self._ask("sweep", self.sweeper, functools.partial(_swept, self.assembly, self.top_rpm.value()))
        return self.sweep_result

    def _responses(self) -> np.ndarray | None:
        """The unbalance response from 0 to the top speed, worked out if it isn't since the last change (as
        `_sweep`)."""
        if self.response is None and self.assembly is not None:
            rpm = np.linspace(0.0, self.top_rpm.value(), RESPONSE_SPEEDS)
            self._ask("response", self.responder, functools.partial(_responded, self.assembly, rpm))
        return self.response

    def _ask(self, what: str, worker: Recompute, job) -> None:
        change = self._change
        if self._now:
            self._asked[what] = change
            worker.now(functools.partial(_tagged, change, job))
        elif self._asked[what] != change:  # once a change: a redraw meanwhile doesn't start it again
            self._asked[what] = change
            worker.request(functools.partial(_tagged, change, job))

    def _waiting(self, what: str) -> bool:
        """`what`'s result is being worked out on the worker: its tab keeps what it last drew."""
        return self.assembly is not None and self._asked[what] == self._change != self._got[what]

    def _on_sweep(self, tagged: tuple[int, tuple[Sweep | None, str]]) -> None:
        change, (result, error) = tagged
        if change != self._change:
            return  # for a rotor since changed
        self.sweep_result = result
        self.error = error or self.error
        self._got["sweep"] = change
        if not self._now:
            self._draw_tabs()
            settle_plots(self)

    def _on_responses(self, tagged: tuple[int, tuple[tuple[np.ndarray, np.ndarray] | None, str]]) -> None:
        change, (result, error) = tagged
        if change != self._change:
            return
        if result is not None:
            self.response_rpm, self.response = result
        self.error = error or self.error
        self._got["response"] = change
        if not self._now:
            self._draw_tabs()
            settle_plots(self)

    def steady(self) -> np.ndarray | None:
        """The unbalance response at the speed (complex, n_dof), kept until the rotor or the speed changes."""
        if self.assembly is None:
            return None
        if self._steady is None or self._steady[0] != self.omega:
            try:
                Q = unbalance_response(self.assembly, np.array([self.rpm.value()]))[0]
            except np.linalg.LinAlgError:
                return None
            self._steady = (self.omega, Q)
        return self._steady[1]

    def _place_speed(self) -> None:
        for line in self.speed_lines:
            line.setPos(self.rpm.value())

    def _branches(self, s: Sweep, values: np.ndarray, forward: bool) -> tuple[np.ndarray, np.ndarray]:
        """Every branch's line, NaN where it whirls the other way and between branches."""
        keep = np.where((s.whirl > 0) == forward, values, np.nan)
        gap = np.full((1, keep.shape[1]), np.nan)
        y = np.vstack([keep, gap]).T.ravel()
        x = np.tile(np.r_[s.rpm, np.nan], keep.shape[1])
        return x, y

    def _clear(self, *names: str) -> None:
        for name in names:
            self.curves[name].setData([], [])

    def _draw_campbell(self) -> None:
        self._place_speed()
        s = self._sweep()
        if s is None and self._waiting("sweep"):
            self.campbell_note.setText(WORKING)
            return
        if s is None:
            self._clear("forward", "backward", "one_x")
            for item in (self.criticals, self.damped_criticals):
                item.setData([], [])
            self.campbell_note.setText(self.error)
            return
        top = float(s.rpm[-1])
        self.curves["forward"].setData(*self._branches(s, s.hz, True))
        self.curves["backward"].setData(*self._branches(s, s.hz, False))
        self.curves["one_x"].setData([0.0, top], [0.0, top / 60.0])
        sharp = [c for c in s.criticals if not c.damped]
        soft = [c for c in s.criticals if c.damped]
        self.criticals.setData([c.rpm for c in sharp], [c.hz for c in sharp])
        self.damped_criticals.setData([c.rpm for c in soft], [c.hz for c in soft])
        low = np.nan_to_num(s.hz[:, :4], nan=0.0).max(initial=0.0)
        self.campbell.setXRange(0.0, top, padding=0)
        self.campbell.setYRange(0.0, 1.05 * max(1.4 * top / 60.0, low), padding=0)
        if sharp:
            listed = ", ".join(
                f"{c.rpm:.0f} rpm {'forward' if c.forward else 'backward'} (AF {c.amplification:.3g})" for c in sharp
            )
            text = f"Critical speeds, where a branch meets 1X (circles): {listed}."
        else:
            text = f"No branch meets 1X below {top:.0f} rpm: no critical speed in the range."
        if soft:
            text += (
                f" Hollow: {len(soft)} crossings critically damped (ζ ≥ 20 %: an amplification factor 1/2ζ under "
                "2.5, which API 684 doesn't count)."
            )
        rpm = self.rpm.value()
        if sharp and rpm > 0:
            near = min(sharp, key=lambda c: abs(c.rpm - rpm))
            text += (
                f" The speed, {rpm:.0f} rpm, is {100 * abs(near.rpm - rpm) / rpm:.3g} % from the nearest, "
                f"{near.rpm:.0f} rpm (its separation margin)."
            )
        self.campbell_note.setText(text)

    def _draw_stability(self) -> None:
        self._place_speed()
        s = self._sweep()
        if s is None and self._waiting("sweep"):
            self.stability_note.setText(WORKING)
            return
        if s is None:
            self._clear("d_forward", "d_backward")
            self.onset_line.hide()
            self.stability_note.setText(self.error)
            return
        delta = s.log_dec
        self.curves["d_forward"].setData(*self._branches(s, delta, True))
        self.curves["d_backward"].setData(*self._branches(s, delta, False))
        top = float(s.rpm[-1])
        self.stability.setXRange(0.0, top, padding=0)
        finite = delta[np.isfinite(delta)]
        lo = float(min(finite.min(initial=0.0), -0.1))
        self.stability.setYRange(max(lo, -1.0) * 1.1, min(max(float(finite.max(initial=0.5)), 0.5), 2.0), padding=0)
        self.onset_line.setVisible(s.onset is not None)
        if s.onset is not None:
            self.onset_line.setPos(s.onset)
            text = (
                f"Unstable from <b>{s.onset:.0f} rpm</b> (the red line): there the least damped mode's δ crosses 0 "
                "and it grows by itself."
            )
        else:
            text = f"Stable to {top:.0f} rpm: every mode's δ stays above 0."
        m = self.result
        if m is not None and m.zeta.size:
            i = int(np.argmin(m.zeta[:N_BRANCHES]))  # of the low modes the map shows, not the mesh's highest
            ok = "meets" if m.log_dec[i] >= API_LOG_DEC else "<b>misses</b>"
            text += (
                f" At {self.rpm.value():.0f} rpm the least damped mode, {m.hz[i]:.4g} Hz {_whirl_word(m.whirl[i])}, "
                f"has δ = {m.log_dec[i]:.3g}: it {ok} API 684's δ ≥ {API_LOG_DEC:g} (dashed)."
            )
        self.stability_note.setText(text)

    def _draw_unbalance(self) -> None:
        self._place_speed()
        Q = self._responses()
        s = self._sweep()
        for line in self.critical_lines:
            line.hide()
        if Q is None and self._waiting("response"):
            self.unbalance_note.setText(WORKING)
            return
        if Q is None:
            self._clear("orbit", "lag")
            self.unbalance_note.setText(self.error)
            return
        rotor = self.rotor
        node = min(self._probe or 0, rotor.nodes.size - 1)
        rpm = self.response_rpm[1:]
        size = orbit(Q[1:], node)
        if not size.any():
            self._clear("orbit", "lag")
            self.unbalance_note.setText("No disc has an unbalance: select one and give it some (on the left).")
            return
        self.curves["orbit"].setData(rpm, np.maximum(size * 1e6, 1e-6))
        self.curves["lag"].setData(rpm, lag(Q[1:], node, rotor))
        self.bode.setLabel("left", "Orbit", units="µm")
        self.bode.getAxis("left").enableAutoSIPrefix(False)
        for p in (self.bode, self.phase):
            p.setXRange(0.0, float(self.response_rpm[-1]), padding=0)
        self.phase.setYRange(0.0, 360.0, padding=0)
        peak = float(size.max()) * 1e6
        self.bode.setYRange(math.log10(peak) - 3, math.log10(peak) + 0.3, padding=0)
        if s is not None:
            sharp = [c for c in s.criticals if not c.damped]
            for line, c in zip(self.critical_lines, sharp):
                line.setPos(c.rpm)
                line.setPen(
                    pg.mkPen(colors.force if c.forward else colors.grey, width=1, style=QtCore.Qt.PenStyle.DotLine)
                )
                line.show()
        k = int(np.argmax(size))
        steady = self.steady()
        here = float(orbit(steady, node)) if steady is not None and self.omega > 0 else 0.0
        unbalanced = [d for d in rotor.discs if d.unbalance]
        listed = ", ".join(
            f"disc {rotor.discs.index(d) + 1} {d.unbalance * 1e6:.4g} g·mm at {math.degrees(d.unbalance_angle):.0f}°"
            for d in unbalanced
        )
        self.unbalance_note.setText(
            f"The steady orbit at node {node + 1} (z = {rotor.nodes[node]:.3g} m; its major semi-axis) against the "
            f"speed, from the discs' unbalance ({listed}), and how far its x lags behind the heavy spot. It peaks at "
            f"<b>{size[k] * 1e6:.4g} µm at {rpm[k]:.0f} rpm</b>; at {self.rpm.value():.0f} rpm, {here * 1e6:.3g} µm. "
            "The dotted lines are the critical speeds, red forward and grey backward: an unbalance turns forward, "
            "and through a forward critical the lag swings by about 180°."
        )

    def _draw_assembly(self) -> None:
        a = self.assembly
        if a is None:
            self.assembly_view.setHtml(f"<p>{self.error}</p>")
            self.assembly_note.setText("")
            return
        which = MATRICES[self.matrix.currentIndex()]
        A = _global(a, which, self.omega)
        n = A.shape[0]
        tint = np.zeros((n, n), dtype=bool)
        nodes: set[int] = set()
        if self.picked is not None:
            for part in a.parts:
                if part.owner == self.picked:
                    own = _matrix(part, which, self.omega)
                    dofs = list(part.dofs)
                    tint[np.ix_(dofs, dofs)] |= own != 0
                    nodes |= {d // 4 for d in dofs}
        keep = np.arange(n)
        if n > MATRIX_LIMIT:
            around = sorted(nodes) if nodes else [0]
            lo, hi = max(min(around) - 1, 0), min(max(around) + 1, n // 4 - 1)
            hi = max(hi, min(lo + MATRIX_LIMIT // 4 - 1, n // 4 - 1)) if hi - lo < 2 else hi
            keep = np.arange(4 * lo, 4 * hi + 4)
        labels = [dof_label(int(d)) for d in keep]
        idx = np.ix_(keep, keep)
        self.assembly_view.setHtml(matrix_html(A[idx], labels, labels, tint[idx], title=which.name))
        picked = " Pick a section or a part to tint its share."
        if self.picked is not None:
            picked = f" Tinted: {self.picked[0]} {self.picked[1] + 1}'s share."
        unit = {
            Matrix.K: "N/m and N/rad",
            Matrix.M: "kg and kg·m²",
            Matrix.C: "N·s/m",
            Matrix.G: "kg·m² (Ω G in the equations)",
        }[which]
        span = f"nodes {keep[0] // 4 + 1}–{keep[-1] // 4 + 1} of {n // 4}" if keep.size < n else f"all {n} DOFs"
        speed = f" at {self.rpm.value():.5g} rpm" if which in (Matrix.K, Matrix.C) else ""
        self.assembly_note.setText(f"{which.value}{speed} over {span} ({unit}).{picked}")

    def _draw_element(self) -> None:
        if self.assembly is None:
            self.element_view.setHtml(f"<p>{self.error}</p>")
            return
        if self.picked is None:
            self.element_view.setHtml("<p>Pick a section or a part (the Select tool) to see its matrices here.</p>")
            return
        which = MATRICES[self.matrix.currentIndex()]
        self.element_view.setHtml(part_html(self.rotor, self.assembly, self.picked, which, self.omega))

    # ------------------------------------------------------------- the modes
    def _on_row(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.mode = rows[0].row()
            self._show_mode()
            if self.tabs.currentIndex() == TAB_WHIRL:
                self._draw_whirl()

    def _on_animate(self, on: bool) -> None:
        self.running = on
        self.loop.start()

    def reset(self) -> None:
        """The whirl back at its start, still whirling."""
        self._phase = 0.0
        self.running = True
        self.run_button.set_running(True)
        self._show_mode()
        self.loop.start()

    def mode_shape(self) -> np.ndarray | None:
        """The mode picked, or the unbalance response at the speed (complex, n_dof), scaled so its largest
        translation is SWING of the side view."""
        if SWINGS[self.swing.currentIndex()] is Swing.UNBALANCE:
            Q = self.steady()
            if Q is None:
                return None
            size = float(orbit(Q, np.arange(self.rotor.nodes.size)).max())
            return Q * (SWING * self.canvas.height_m / size) if size > 0 else None
        m = self.result
        if m is None or self.mode >= m.hz.size:
            return None
        return m.shapes[:, self.mode] * (SWING * self.canvas.height_m)

    def _show_mode(self) -> None:
        shape = self.mode_shape()
        self.canvas.q = None if shape is None else (shape * np.exp(1j * self._phase)).real
        self.canvas.draw()
        if self.tabs.currentIndex() == TAB_WHIRL:
            self.whirl.show_phase(self._phase)

    def _runs(self) -> bool:
        return self.running and self.result is not None

    def _sim_t(self) -> float:
        return self._phase

    def _tick(self, dt: float) -> None:
        self.step(dt)

    def frame_context(self) -> str:
        r = self.rotor
        return (
            f"{len(r.sections)} sections, {len(r.discs)} discs, {len(r.bearings)} bearings, {r.nodes.size} nodes, "
            f"mode {self.mode + 1} at {self.rpm.value():g} rpm"
        )

    def step(self, dt: float) -> None:
        """Whirl the mode on by dt seconds at the animation's own pace (not while a part is dragged)."""
        if self.canvas._drag is not None:
            return
        self._phase = (self._phase + TWO_PI * SHOW_HZ * dt) % TWO_PI
        self._show_mode()

    def apply_theme(self) -> None:
        self.canvas.apply_theme()
        self.whirl.apply_theme()
        mute(self.hint)
        dash, dot = QtCore.Qt.PenStyle.DashLine, QtCore.Qt.PenStyle.DotLine
        for name, color in (("forward", colors.mode[0]), ("d_forward", colors.mode[0])):
            self.curves[name].setPen(pg.mkPen(color, width=2))
        for name in ("backward", "d_backward"):
            self.curves[name].setPen(pg.mkPen(colors.mode[1], width=2, style=dash))
        self.curves["one_x"].setPen(pg.mkPen(colors.force, width=1.5))
        self.curves["orbit"].setPen(pg.mkPen(colors.strong, width=2))
        self.curves["lag"].setPen(pg.mkPen(colors.strong, width=2))
        self.criticals.setSymbolBrush(pg.mkBrush(colors.force))
        self.criticals.setSymbolPen(None)
        self.damped_criticals.setSymbolBrush(None)
        self.damped_criticals.setSymbolPen(pg.mkPen(colors.force, width=1.5))
        for line in self.speed_lines:
            line.setPen(pg.mkPen(colors.grey, width=1.5, style=dash))
        for line in self.critical_lines:
            line.setPen(pg.mkPen(colors.force, width=1, style=dot))
        self.zero_line.setPen(pg.mkPen(colors.muted, width=1))
        self.api_line.setPen(pg.mkPen(colors.muted, width=1, style=dash))
        self.onset_line.setPen(pg.mkPen(colors.force, width=1.5))
        self._fill_table()
        self._draw_tabs()
