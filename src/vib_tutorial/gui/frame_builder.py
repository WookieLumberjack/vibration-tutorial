"""Frame builder page: build a plane frame or truss on a grid, watch its matrices assemble, see its modes.

Left: Things to try, the selected item's properties (a node's position, support, point mass and torsion spring;
a member's kind, material and section; a spring's or damper's value), the model's settings (consistent or lumped
mass, Euler–Bernoulli or Timoshenko beams) and what to show. Centre: the tools, the canvas (click to place nodes,
click two nodes to join them, drag a node to move it), swinging in the mode picked, and the modes' table. Right:
the global matrix with the picked element's entries tinted and the struck-out rows and columns greyed, the
element's own matrices and its transformation, and the Theory. The page has its own model (core/frame.py) and
animates the mode on its own frame loop while it shows.

The trials and practice problems are gui/frame_builder_trials.py's (re-exported here, where the practice window and
the Theory tab look for them), and the canvas is gui/frame_builder_canvas.py's `FrameCanvas`. `matrix_html` stays
here: the Rotor builder's Assembly tab draws its matrices with it too.
"""

from __future__ import annotations

import dataclasses
import math

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
    Theory,
    assemble,
    dof_label,
    modes,
)
from ..core.frame_response import LOAD_KINDS, Formulations, Load, LoadKind, Responses, formulations, responses
from ..core.practice import number
from . import frame_figures, plot_patches, weak
from .challenge import ChallengeBox
from .frame_builder_canvas import NAN2, TOOL_HINTS, TOOLS, FrameCanvas, Tool, _segments
from .frame_builder_trials import (
    MATRICES,
    SWINGS,
    TAB_ASSEMBLY,
    TAB_ELEMENT,
    TAB_FORMULATIONS,
    TAB_RESPONSE,
    FrameTrial,
    Matrix,
    Swing,
)
from .frame_builder_trials import PROBLEMS as PROBLEMS
from .frame_builder_trials import TRIALS as TRIALS
from .frame_loop import FRAME_MS, SWING_MS, FrameLoop
from .layouts import FitColumns, FlowLayout, group
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
from .links import load_notes
from .style import colors, rgb
from .theming import add_legend, mute
from .transport import Transport

SHOW_HZ = 0.6  # the animation's own pace: every mode swings at it
SWING = 0.08  # the largest displacement drawn, as a fraction of the frame's size
MODE_ROWS = 20  # modes listed at most
GRIDS = (0.05, 0.1, 0.125, 0.25, 0.5, 1.0)  # m
MATRIX_LIMIT = 90  # DOFs: beyond this the Assembly tab lists the free ones only
PLAY_S = 8.0  # s: a response's record plays over this long on the canvas
DOF_CHOICES = ("u, along x", "v, along y", "θ, a turn")


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
        self.parts.setAccessibleName("Divide into parts")
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

    def transport(self) -> Transport:
        """The run bar drives the frame's swing."""
        return Transport(self.run_button, self.reset_button)

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
            predictions=[t.predict for t in TRIALS],
        )
        self.trials.picked.connect(self.load_trial)
        self.challenge = ChallengeBox(self, PROBLEMS)

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
        FitColumns(self.table)
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
        plot_patches.hold_still(self.time_plot)  # the playhead moves over still curves (16 ms a frame → 0.5)
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
        FitColumns(self.forms_table)
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
        three_panes(self, column(self.trials, self.challenge, self.props, model, loading, show), centre, self.tabs)

        self.loop = FrameLoop(self, "Frame builder", self._tick, self._runs, self._sim_t)
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
        self.loop.start()

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
        self.loop.start()

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
        self.loop.start()

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
        self.loop.start()

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

    def _runs(self) -> bool:
        return self.running and self.result is not None

    def _sim_t(self) -> float:
        return self._phase

    def _tick(self, dt: float) -> None:
        self.step(dt)

    def frame_context(self) -> str:
        swing = SWINGS[self.swing.currentIndex()].name.lower()
        return f"{len(self.frame.nodes)} nodes, {len(self.frame.elements)} elements, mode {self.mode + 1}, {swing}"

    def step(self, dt: float) -> None:
        """Swing the mode on by dt seconds at the animation's own pace (not while a node is dragged)."""
        if self.canvas._drag is not None:
            return
        playing = SWINGS[self.swing.currentIndex()] is Swing.RESPONSE and self.runs is not None
        self.loop.frame_ms = FRAME_MS if playing else SWING_MS  # a response played back smoothly, a mode at 30
        if playing:
            assert self.runs is not None, "playing only with runs"
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
