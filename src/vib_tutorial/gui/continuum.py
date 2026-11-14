"""Continuous systems page: a string, a bar or a beam against the meshes of masses and springs that approximate it.

Left: the preset, the medium (its kind, size, material and ends), the mesh (how many
elements, which ways of sharing out the mass to show, the mode to look at) and the waves
(the pulse, the mesh's mass, a dashpot on the far end, a step in the section). Centre,
three tabs: the dispersion curves, the continuum's and each mesh's, with the modes on them;
the frequency errors, against the mode number and, for the chosen mode, against the number
of elements; and a pulse running along the continuum and the mesh. Right: the chosen mode's
shape, the frequencies in a table, and the notes. The page has its own model
(core/continuum.py and core/continuum_waves.py), not the chain.

Working on it: the Waves tab, its controls and its run are `continuum_waves_view.WavesView`, which the page
gives each medium and mesh (`set_medium`); the page keeps `running`, `play_button` and `loop` for the main
window and the tests. The Errors tab is drawn only while it shows (`LazyTabs`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.continuum import (
    CONVERGENCE_COUNTS,
    ENDS,
    MAX_ELEMENTS,
    PRESETS,
    SCHEMES,
    TWO_PI,
    Continuum,
    End,
    FEModel,
    Kind,
    Scheme,
    convergence,
    dispersion,
    errors,
    flexible_hz,
    order,
    string_dispersion,
)
from ..core.continuum_waves import (
    WAVE_PRESETS,
    WavePreset,
    Waves,
)
from ..core.practice import Problem, Question, Requirement, at_most, number, pick
from . import continuum_figures, maths
from .continuum_waves_view import WavesView
from .frame_loop import FrameLoop
from .layouts import ColumnScroll, FlowLayout, form_layout, form_row, group, plot_height
from .lesson import RunButton, plot, trials_html
from .links import load_notes, page_link
from .notes_view import NotesBrowser
from .panels import spin
from .style import colors
from .tab_groups import LazyTabs
from .theming import add_legend, mute

CUSTOM = "Custom"
ALL_PRESETS = (*PRESETS, *WAVE_PRESETS)
KAPPA_POINTS = 241  # samples of each mesh's dispersion curve
SHAPE_POINTS = 801
PAST_CUTOFF = 1.25  # the dispersion plot runs this far past the mesh's cut-off, π/h

TIPS = {
    "kind": "<p><b>String</b>: transverse waves under tension, c = √(T/μ). <b>Bar</b>: axial (lengthwise) "
    "waves, c = √(E/ρ). Both obey the wave equation, so every wave travels at c. <b>Beam</b>: bending "
    "(Euler–Bernoulli), ω = √(EI/ρA) κ²: short waves travel faster than long ones.</p>",
    "length": "<p>The length between the ends.</p>",
    "tension": "<p>The string's tension T.</p>",
    "line_density": "<p>The string's mass per unit length μ.</p>",
    "modulus": "<p>Young's modulus E: steel 210 GPa, aluminium 69, wood about 10 along the grain.</p>",
    "density": "<p>The material's density ρ: steel 7850 kg/m³, aluminium 2700.</p>",
    "section": "<p>The rectangular section, width × depth. A beam bends across its depth (I = b d³/12), so its "
    "frequencies go as the depth and don't depend on the width; nor do a bar's on either.</p>",
    "ends": "<p>How each end is held. String and bar: <b>fixed</b> (held still) or <b>free</b> (a string's end "
    "on a ring that slides on a rod). Beam: <b>clamped</b> (no deflection, no slope), <b>pinned</b> "
    "(no deflection, free to turn) or <b>free</b>.</p>",
    "elements": "<p>The number of equal elements the length is cut into. Lumped, a string is then beads on a "
    "massless string, and a bar the chain of the first page.</p>",
    "schemes": "<p>How each element's mass is shared out. <b>Lumped</b>: half to each end node (a beam's "
    "slopes get none). <b>Consistent</b>: from the same shape functions as the stiffness: an "
    "upper bound on every frequency. <b>Mixed</b>: their average.</p>",
    "mode": "<p>The flexible mode to look at (the rigid ones, at 0 Hz, are left out): its shape, where it sits on "
    "the dispersion curves, and how its error falls with the number of elements. Or click a mode on "
    "the error plot.</p>",
}
DISPERSION_TIP = (
    "<p>Frequency against wavenumber κ = 2π/λ. The continuum's (black) is a straight line for a string or "
    "bar (every wave at the same speed) and a parabola for a beam. Each mesh's bends away from it and stops at "
    "κ = π/h, two elements a wavelength: the shortest wave the nodes can carry. The dots are the mesh's "
    "modes, each at its exact mode's wavenumber. A beam's consistent or mixed mesh has a second branch "
    "(dashed), where the slopes swing against the deflections, drawn from π/h on to 2π/h: the mesh's modes past "
    "π/h are on it, its nodes seeing a shorter wave's deflections as a longer one's.</p>"
)
ERROR_TIP = (
    "<p>Each mode's frequency on the mesh against the exact one, (f<sub>mesh</sub> − f) / f. The low modes "
    "span many elements and come out nearly right; the high ones are wrong by tens of per cent. Click a "
    "mode to choose it.</p>"
)
CONVERGENCE_TIP = (
    "<p>The chosen mode's error against the number of elements, both logarithmic: a straight line of slope "
    "−p is an error going as N<sup>−p</sup>. The dotted line marks the mesh shown. Points stop where the "
    "error reaches the eigensolver's round-off.</p>"
)
SHAPE_TIP = (
    "<p>The chosen mode's shape: the exact one (black) and each mesh's, the nodes as dots and the elements' own "
    "shape functions between them (straight for a string or bar, cubic for a beam), largest +1.</p>"
)

VIEW_DISPERSION, VIEW_ERRORS, VIEW_WAVES = range(3)


@dataclass(frozen=True)
class ContinuumSetup:
    """A practice problem's setup (and the page's `practice_state`): the medium, the mesh, the view and the mode."""

    continuum: Continuum = field(default_factory=lambda: PRESETS[0].continuum)
    elements: int = 12
    view: int = VIEW_DISPERSION
    mode: int = 1


PRACTICE_SETUP = ContinuumSetup  # what a practice problem's setup and the page's practice_state are


def _string(length: float = 1.0, tension: float = 100.0, line_density: float = 0.01) -> Continuum:
    return Continuum(Kind.STRING, length, End.FIXED, End.FIXED, tension=tension, line_density=line_density)


# ------------------------------------------------------------------ practice problems
# Each question's numbers are printed round, and its answer is worked from them as printed.


def _string_fundamental(rng: np.random.Generator) -> Question:
    length = pick(rng, (0.33, 0.5, 0.65, 1))
    tension = pick(rng, (50, 70, 100, 150))
    mu = pick(rng, (1, 2, 5, 10))  # g/m
    c = _string(length, tension, mu / 1e3)
    f = float(c.exact_hz(1)[0])
    return Question(
        f"A string {length:g} m long, fixed at both ends, under {tension:g} N, weighing {mu:g} g/m. What is its "
        "fundamental frequency?",
        f,
        f"c = √(T/μ) = √({tension:g}/{number(mu / 1e3)}) = {number(c.speed)} m/s, and half a wavelength fits: "
        f"f<sub>1</sub> = c/2L = {number(f)} Hz, the overtones whole multiples.",
        unit="Hz",
        setup=ContinuumSetup(c, 12).__dict__,
    )


def _bar_fundamental(rng: np.random.Generator) -> Question:
    length = pick(rng, (0.5, 1, 2, 3))
    c = Continuum(Kind.BAR, length, End.FIXED, End.FREE, modulus=210e9, density=7850.0, width=0.02, depth=0.02)
    f = float(c.exact_hz(1)[0])
    return Question(
        f"A steel bar (E = 210 GPa, ρ = 7850 kg/m³) {length:g} m long is fixed at one end and free at the other. "
        "What is its first axial natural frequency?",
        f,
        f"c = √(E/ρ) = {number(c.speed)} m/s; fixed–free, a quarter wavelength fits: f<sub>1</sub> = c/4L = "
        f"{number(f)} Hz (then 3, 5, 7 times it).",
        unit="Hz",
        setup=ContinuumSetup(c, 10).__dict__,
    )


def _cantilever_second(rng: np.random.Generator) -> Question:
    f1 = pick(rng, (5, 10, 12, 20, 50))
    beam = Continuum(Kind.BEAM, 1.0, End.CLAMPED, End.FREE)
    ratio = float(beam.exact_hz(2)[1] / beam.exact_hz(1)[0])
    return Question(
        f"A cantilever's first bending mode is at {f1:g} Hz. Where is its second?",
        f1 * ratio,
        f"cos βL cosh βL = −1 gives β<sub>1</sub>L = 1.875 and β<sub>2</sub>L = 4.694, and ω ∝ β²: f<sub>2</sub> = "
        f"(4.694/1.875)² f<sub>1</sub> = {number(ratio)} × {f1:g} = {number(f1 * ratio)} Hz. A beam's overtones are "
        "no multiples.",
        unit="Hz",
        setup=ContinuumSetup(beam, 8, mode=2).__dict__,
    )


def _lumped_mode(rng: np.random.Generator) -> Question:
    n_el = int(pick(rng, (4, 6, 8, 12)))
    n = int(pick(rng, (1, 2, 3)))
    c = _string()
    f = float(flexible_hz(FEModel(c, n_el, Scheme.LUMPED))[n - 1])
    exact = float(c.exact_hz(n)[-1])
    return Question(
        f"A string fixed at both ends (c = 100 m/s, 1 m long) is modelled as {n_el} lumped elements (beads on "
        f"springs). Where does the model put mode {n}?",
        f,
        f"The mesh's modes sit on its dispersion curve at the exact κ<sub>n</sub> = nπ/L: ω = (2c/h) sin(κh/2) with "
        f"h = {number(1 / n_el)} m, so f = (c/πh) sin(nπ/2N) = {number(f)} Hz, below the exact {number(exact)} Hz "
        f"by {number(100 * (1 - f / exact))} %.",
        unit="Hz",
        tolerance=0.005,
        setup=ContinuumSetup(c, n_el, mode=n).__dict__,
    )


def _cut_off(rng: np.random.Generator) -> Question:
    n_el = int(pick(rng, (5, 10, 20, 50)))
    speed = pick(rng, (100, 200, 343))
    h = 1.0 / n_el
    f = speed / (math.pi * h)
    return Question(
        f"A 1 m string with a wave speed of {speed:g} m/s is cut into {n_el} lumped elements. What is the highest "
        "frequency a wave on the endless chain of such elements can have (its cut-off)?",
        f,
        f"ω = (2c/h) sin(κh/2) tops out at κh = π: ω = 2c/h, f = c/πh = {speed:g}/(π × {number(h)}) = {number(f)} Hz. "
        "Nothing faster travels on the mesh.",
        unit="Hz",
        setup=ContinuumSetup(_string(), n_el).__dict__,
    )


def _matched_dashpot(rng: np.random.Generator) -> Question:
    tension = pick(rng, (50, 100, 200))
    mu = pick(rng, (1, 5, 10))
    z = math.sqrt(tension * mu / 1e3)
    return Question(
        f"A string under {tension:g} N weighing {mu:g} g/m ends on a dashpot. What dashpot coefficient lets a wave "
        "arrive without reflecting at all?",
        z,
        f"An end reflects (Z − c<sub>d</sub>)/(Z + c<sub>d</sub>) of a wave, so none when c<sub>d</sub> = Z, the "
        f"string's impedance √(Tμ) = √({tension:g} × {number(mu / 1e3)}) = {number(z)} N·s/m: it takes the wave as "
        "an endless string would.",
        unit="N·s/m",
    )


def _consistent_side(rng: np.random.Generator) -> Question:
    return Question(
        "A bar is meshed with consistent masses (the finite element method's). Where do its frequencies lie?",
        0,
        "Above the exact ones, every one: the consistent mesh is a Rayleigh–Ritz model, and Rayleigh–Ritz frequencies "
        "are upper bounds. (Lumped masses come out below.)",
        choices=("Above the exact ones", "Below the exact ones", "Some above, some below"),
        setup=ContinuumSetup(PRESETS[1].continuum, 10, VIEW_ERRORS).__dict__,
        predict=True,
    )


def _halved_elements(rng: np.random.Generator) -> Question:
    return Question(
        "A string's lumped mesh is refined to elements half as long. What happens to its low modes' frequency errors?",
        0,
        "They fall to a quarter: the lumped error is −(κh)²/24 for long waves, second order in h. (The mixed mass's "
        "falls 16 times.)",
        choices=("They fall to a quarter", "They halve", "They fall 16 times"),
        setup=ContinuumSetup(_string(), 24, VIEW_ERRORS).__dict__,
        predict=True,
    )


def _beam_overtones(rng: np.random.Generator) -> Question:
    return Question(
        "A guitar string's overtones are whole multiples of its fundamental. Are a cantilever beam's?",
        0,
        "No: 1 : 6.27 : 17.5. Bending waves are dispersive, ω = aκ², and the clamped and free ends give a "
        "transcendental equation for κ, not a whole number of half wavelengths.",
        choices=("No: 1 : 6.27 : 17.5", "Yes: 1 : 2 : 3", "Yes, the odd ones: 1 : 3 : 5"),
        setup=ContinuumSetup(Continuum(Kind.BEAM, 1.0, End.CLAMPED, End.FREE), 8).__dict__,
        predict=True,
    )


def _mesh_spec(rng: np.random.Generator) -> Question:
    n = int(pick(rng, (2, 3, 5)))
    most = pick(rng, (0.5, 1, 2))
    c = _string()
    fewest = next(N for N in range(n + 1, 201) if abs(errors(FEModel(c, N, Scheme.LUMPED))[1][n - 1]) * 100 <= most)

    def goal(t: ContinuumSetup):
        m = t.continuum
        string = m.kind is Kind.STRING and m.left is End.FIXED and m.right is End.FIXED
        found = errors(FEModel(m, t.elements, Scheme.LUMPED))[1]
        err = abs(found[n - 1]) * 100 if found.size >= n else 100.0
        return (
            Requirement("A string fixed at both ends", f"{m.kind.value}, {m.left.value}–{m.right.value}", string),
            at_most(f"The lumped mesh's error in mode {n}", err, most, "%"),
            at_most("Elements", t.elements, fewest),
        )

    return Question(
        f"Mesh a string fixed at both ends with lumped masses: the fewest elements that get mode {n} within {most:g} %.",
        0,
        f"The lumped error is about −(κh)²/24 with κ = nπ/L and h = L/N: (nπ/N)²/24 ≤ {most / 100:g} needs N ≥ "
        f"nπ/√(24 × {most / 100:g}) ≈ {number(n * math.pi / math.sqrt(24 * most / 100))}, so {fewest} elements "
        "(the Frequencies tab lists the errors).",
        setup=ContinuumSetup(c, n + 1).__dict__,
        goal=goal,
        design=ContinuumSetup(c, fewest).__dict__,
    )


PROBLEMS = tuple(
    Problem(key, title, "continuum_page", f"See {page_link('continuum_page', anchor, section)}{more}.", make)
    for key, title, anchor, section, more, make in (
        (
            "string-fundamental",
            "A string's fundamental",
            "continuum-waves",
            "The continuum's waves",
            "",
            _string_fundamental,
        ),
        ("bar-fundamental", "A bar's fundamental", "continuum-waves", "The continuum's waves", "", _bar_fundamental),
        (
            "cantilever-second",
            "A cantilever's second mode",
            "continuum-waves",
            "The continuum's waves",
            "",
            _cantilever_second,
        ),
        ("lumped-mode", "A lumped mesh's mode", "dispersion", "The chain's waves", "", _lumped_mode),
        ("cut-off", "The mesh's cut-off", "dispersion", "The chain's waves", "", _cut_off),
        ("matched-dashpot", "A matched end", "reflections", "Waves and reflections", "", _matched_dashpot),
        (
            "consistent-side",
            "Predict: consistent masses",
            "mass-schemes",
            "Lumped and consistent masses",
            "",
            _consistent_side,
        ),
        (
            "halved-elements",
            "Predict: half the element length",
            "mass-schemes",
            "Lumped and consistent masses",
            "",
            _halved_elements,
        ),
        (
            "beam-overtones",
            "Predict: a beam's overtones",
            "continuum-waves",
            "The continuum's waves",
            "",
            _beam_overtones,
        ),
        ("mesh-spec", "Spec: just enough elements", "dispersion", "The chain's waves", "", _mesh_spec),
    )
)


NOTES_HTML = load_notes("continuum", figures=continuum_figures.FIGURES, trials=trials_html(ALL_PRESETS))


def fmt_hz(f: float) -> str:
    return f"{f / 1e3:.4g} kHz" if f >= 1e4 else f"{f:.4g} Hz"


def scheme_color(scheme: Scheme) -> str:
    return {Scheme.LUMPED: colors.mode[0], Scheme.CONSISTENT: colors.mode[1], Scheme.MIXED: colors.mode[3]}[scheme]


class ContinuumPage(QtWidgets.QWidget):
    """Controls on the left, dispersion and errors in the centre, the mode's shape, the table and the notes on the right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.continuum: Continuum = PRESETS[0].continuum
        self.models: dict[Scheme, FEModel] = {}
        self._loading = False

        # --- left: the preset, the medium, the mesh
        self.preset = QtWidgets.QComboBox()
        self.preset.addItems([p.name for p in ALL_PRESETS] + [CUSTOM])
        self.preset.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.preset_note = QtWidgets.QLabel()
        self.preset_note.setWordWrap(True)
        self.preset_note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        mute(self.preset_note)

        medium = QtWidgets.QGroupBox("Medium")
        mf = form_layout(medium)
        self.kind = QtWidgets.QComboBox()
        self.kind.addItems([k.value for k in Kind])
        self.length = spin(0.01, 100.0, 1.0, 3, " m")
        form_row(mf, "Kind:", self.kind, TIPS["kind"])
        form_row(mf, "Length:", self.length, TIPS["length"])
        self.material = QtWidgets.QStackedWidget()
        string_page = QtWidgets.QWidget()
        form = form_layout(string_page)
        form.setContentsMargins(0, 0, 0, 0)
        self.tension = spin(0.01, 1e6, 100.0, 2, " N")
        self.line_density = spin(0.001, 1e5, 10.0, 3, " g/m")
        form_row(form, "Tension T:", self.tension, TIPS["tension"])
        form_row(form, "Mass per length μ:", self.line_density, TIPS["line_density"])
        self.material.addWidget(string_page)
        solid_page = QtWidgets.QWidget()
        form = form_layout(solid_page)
        form.setContentsMargins(0, 0, 0, 0)
        self.modulus = spin(0.001, 1000.0, 210.0, 3, " GPa")
        self.density = spin(1.0, 30000.0, 7850.0, 0, " kg/m³")
        self.section_width = spin(0.01, 1000.0, 20.0, 2, " mm")
        self.section_depth = spin(0.01, 1000.0, 10.0, 2, " mm")
        form_row(form, "Modulus E:", self.modulus, TIPS["modulus"])
        form_row(form, "Density ρ:", self.density, TIPS["density"])
        form_row(form, "Section:", group(self.section_width, "×", self.section_depth), TIPS["section"])
        self.material.addWidget(solid_page)
        mf.addRow(self.material)
        self.left_end = QtWidgets.QComboBox()
        self.right_end = QtWidgets.QComboBox()
        form_row(mf, "Ends:", group(self.left_end, "–", self.right_end), TIPS["ends"])
        self.medium_info = QtWidgets.QLabel()
        self.medium_info.setWordWrap(True)
        mute(self.medium_info)
        mf.addRow(self.medium_info)

        mesh = QtWidgets.QGroupBox("Mesh")
        sf = form_layout(mesh)
        self.elements = QtWidgets.QSpinBox()
        self.elements.setRange(1, MAX_ELEMENTS)
        self.elements.setKeyboardTracking(False)
        form_row(sf, "Elements N:", self.elements, TIPS["elements"])
        self.shown: dict[Scheme, QtWidgets.QCheckBox] = {}
        row = FlowLayout()
        for scheme in SCHEMES:
            check = QtWidgets.QCheckBox(scheme.value)
            check.setChecked(True)
            check.setToolTip(TIPS["schemes"])
            self.shown[scheme] = check
            row.add(check)
        label = QtWidgets.QLabel("Mass:")
        label.setToolTip(TIPS["schemes"])
        sf.addRow(label, _holder(row))
        self.mode = QtWidgets.QSpinBox()
        self.mode.setRange(1, 1)
        self.mode.setKeyboardTracking(False)
        form_row(sf, "Mode:", self.mode, TIPS["mode"])
        self.mesh_info = QtWidgets.QLabel()
        self.mesh_info.setWordWrap(True)
        mute(self.mesh_info)
        sf.addRow(self.mesh_info)

        self.waves_view = WavesView()  # the Waves pane (its controls) and tab

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        presets = QtWidgets.QGroupBox("Things to try")
        preset_layout = QtWidgets.QVBoxLayout(presets)
        preset_layout.addWidget(self.preset)
        preset_layout.addWidget(self.preset_note)
        lv.addWidget(presets)
        lv.addWidget(medium)
        lv.addWidget(mesh)
        lv.addWidget(self.waves_view.controls)
        lv.addStretch(1)
        left_scroll = ColumnScroll(left)

        # --- centre: dispersion, errors
        self.dispersion = plot("Frequency", "Hz", "Wavenumber κ", "rad/m")
        self.dispersion.setToolTip(DISPERSION_TIP)
        plot_height(self.dispersion, 260)
        self.dispersion_legend = add_legend(self.dispersion.getPlotItem(), offset=(10, 10))  # top left
        self.readout = QtWidgets.QLabel()
        self.readout.setWordWrap(True)
        self.readout.setTextFormat(QtCore.Qt.TextFormat.RichText)
        dispersion_tab = QtWidgets.QWidget()
        dv = QtWidgets.QVBoxLayout(dispersion_tab)
        dv.addWidget(self.dispersion, 1)
        dv.addWidget(self.readout)

        self.errors = plot("Frequency error", "%", "Mode", None)
        self.errors.setToolTip(ERROR_TIP)
        plot_height(self.errors, 180)
        self.errors_legend = add_legend(self.errors.getPlotItem(), offset=(10, 10))  # top left
        self.errors.scene().sigMouseClicked.connect(self._on_errors_click)
        self.convergence = plot("|Frequency error|", None, "Elements N", None, log_ready=True)
        self.convergence.setLogMode(x=True, y=True)
        self.convergence.setToolTip(CONVERGENCE_TIP)
        plot_height(self.convergence, 180)
        self.convergence_legend = add_legend(self.convergence.getPlotItem(), offset=(-10, 10))  # top right
        self.convergence_note = QtWidgets.QLabel()
        self.convergence_note.setWordWrap(True)
        self.convergence_note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        errors_tab = QtWidgets.QWidget()
        ev = QtWidgets.QVBoxLayout(errors_tab)
        ev.addWidget(self.errors, 1)
        ev.addWidget(self.convergence, 1)
        ev.addWidget(self.convergence_note)
        self.views = QtWidgets.QTabWidget()
        self.dispersion_tab = dispersion_tab
        self.errors_tab = errors_tab
        self.views.addTab(dispersion_tab, "Dispersion")
        self.views.addTab(errors_tab, "Errors")

        self.waves_tab = self.waves_view
        self.views.addTab(self.waves_view, "Waves")
        # The Errors tab is drawn only while it shows: the convergence study is ~2 s of eigensolves for a new kind
        # or set of ends (cached after), which made the page's first opening hang.
        self.lazy = LazyTabs(self.views, {self.errors_tab: self._draw_errors})

        # --- right: the mode's shape, the table, the notes
        self.shape = plot("Deflection", None, "Position x", "m")
        self.shape.setMouseEnabled(x=False, y=False)
        self.shape.setToolTip(SHAPE_TIP)
        plot_height(self.shape, 200)
        self.shape_legend = add_legend(self.shape.getPlotItem(), offset=(10, 5), colCount=2)  # along the top
        self.shape_note = QtWidgets.QLabel()
        self.shape_note.setWordWrap(True)
        self.shape_note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        shape_tab = QtWidgets.QWidget()
        sv = QtWidgets.QVBoxLayout(shape_tab)
        sv.addWidget(self.shape, 1)
        sv.addWidget(self.shape_note)

        self.table = QtWidgets.QTableWidget()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.setToolTip(
            "<p>Each flexible mode's exact frequency and each mesh's, with its error. "
            "Pick a row to choose the mode.</p>"
        )
        self.table.itemSelectionChanged.connect(self._on_table_selection)

        self.notes = NotesBrowser()
        self.notes.setHtml(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.shape_tab = shape_tab
        self.tabs.addTab(shape_tab, "Mode shape")
        self.tabs.addTab(self.table, "Frequencies")
        self.tabs.addTab(self.notes, "Theory")

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(self.views)
        splitter.addWidget(self.tabs)
        splitter.setSizes([340, 810, 550])
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        # --- wiring
        self.preset.currentIndexChanged.connect(self._on_preset)
        self.kind.currentIndexChanged.connect(self._on_kind)
        for box in (
            self.length,
            self.tension,
            self.line_density,
            self.modulus,
            self.density,
            self.section_width,
            self.section_depth,
        ):
            box.valueChanged.connect(self._on_edit)
        self.left_end.currentIndexChanged.connect(self._on_edit)
        self.right_end.currentIndexChanged.connect(self._on_edit)
        self.elements.valueChanged.connect(self._on_edit)
        for check in self.shown.values():
            check.toggled.connect(self.redraw)
        self.mode.valueChanged.connect(self._on_mode)

        self.load_preset(PRESETS[0].name)

    # ------------------------------------------------------------- inputs
    def load_preset(self, name: str) -> None:
        """A preset's medium and mesh; a wave preset's pulse too, on the Waves tab, started (as a trial is)."""
        p = next(p for p in ALL_PRESETS if p.name == name)
        self._load(p.continuum, p.elements, name, p.note, p.waves if isinstance(p, WavePreset) else Waves())
        if isinstance(p, WavePreset):
            self.waves_view.play()
            self.views.setCurrentWidget(self.waves_view)

    def load_setup(self, setup: dict) -> None:
        """Set the page up as a practice problem's `setup` says (a `ContinuumSetup`'s fields)."""
        t = ContinuumSetup(**setup)
        self._load(t.continuum, t.elements, CUSTOM, "", Waves())
        self.views.setCurrentIndex(t.view)
        self.mode.setValue(t.mode)

    def practice_state(self) -> ContinuumSetup:
        """The medium, the mesh, the view and the mode: what a practice spec checks."""
        return ContinuumSetup(self.continuum, self.elements.value(), self.views.currentIndex(), self.mode.value())

    def _load(self, c: Continuum, elements: int, name: str, note: str, waves: Waves) -> None:
        self._loading = True
        self.preset.setCurrentText(name)
        self.preset_note.setText(maths.markup(note))
        self.kind.setCurrentText(c.kind.value)
        self._fill_ends(c.kind)
        self.length.setValue(c.length)
        self.tension.setValue(c.tension)
        self.line_density.setValue(c.line_density * 1e3)
        self.modulus.setValue(c.modulus / 1e9)
        self.density.setValue(c.density)
        self.section_width.setValue(c.width * 1e3)
        self.section_depth.setValue(c.depth * 1e3)
        self.left_end.setCurrentText(c.left.value)
        self.right_end.setCurrentText(c.right.value)
        self.elements.setValue(elements)
        self.waves_view.load(waves, c.kind)
        self._loading = False
        self.mode.blockSignals(True)
        self.mode.setValue(1)
        self.mode.blockSignals(False)
        self.set_continuum(c, elements)

    def frame_context(self) -> str:
        """For the frame log (frame_stats): the preset showing."""
        return self.preset.currentText()

    def _on_preset(self, index: int) -> None:
        if not self._loading and index < len(ALL_PRESETS):
            self.load_preset(ALL_PRESETS[index].name)

    def _fill_ends(self, kind: Kind) -> None:
        for box in (self.left_end, self.right_end):
            old = box.currentText()
            box.blockSignals(True)
            box.clear()
            box.addItems([e.value for e in ENDS[kind]])
            if old in [e.value for e in ENDS[kind]]:
                box.setCurrentText(old)
            box.blockSignals(False)
        self.material.setCurrentIndex(0 if kind is Kind.STRING else 1)
        for i in range(self.material.count()):  # only the page showing sets the height
            page = self.material.widget(i)
            assert page is not None, "i < count"
            policy = (
                QtWidgets.QSizePolicy.Policy.Preferred
                if i == self.material.currentIndex()
                else QtWidgets.QSizePolicy.Policy.Ignored
            )
            page.setSizePolicy(policy, policy)

    def _on_kind(self, _index: int) -> None:
        if self._loading:
            return
        kind = Kind(self.kind.currentText())
        self._fill_ends(kind)
        self._loading = True
        self.waves_view.load(self.waves_view.waves(kind), kind)  # a beam's pulse starts at rest, on its own ends
        self._loading = False
        self._on_edit()

    def _read(self) -> Continuum:
        return Continuum(
            Kind(self.kind.currentText()),
            self.length.value(),
            End(self.left_end.currentText()),
            End(self.right_end.currentText()),
            tension=self.tension.value(),
            line_density=self.line_density.value() / 1e3,
            modulus=self.modulus.value() * 1e9,
            density=self.density.value(),
            width=self.section_width.value() / 1e3,
            depth=self.section_depth.value() / 1e3,
        )

    def _on_edit(self, *_args) -> None:
        if self._loading:
            return
        self.preset.blockSignals(True)
        self.preset.setCurrentText(CUSTOM)
        self.preset.blockSignals(False)
        self.preset_note.setText("")
        self.set_continuum(self._read(), self.elements.value())

    def set_continuum(self, c: Continuum, elements: int) -> None:
        """Mesh the medium every way and redraw."""
        self.continuum = c
        self.models = {s: FEModel(c, elements, s) for s in SCHEMES}
        count = max(flexible_hz(m).size for m in self.models.values())
        self._exact_kappa = c.exact_wavenumbers(count) if count else np.zeros(0)
        self.mode.blockSignals(True)
        self.mode.setRange(1, max(count, 1))
        self.mode.setEnabled(count > 0)
        self.mode.blockSignals(False)
        self.shape.setLabel("left", "Displacement (along the bar)" if c.kind is Kind.BAR else "Deflection")
        self._show_info()
        self._fill_table()
        self.redraw()
        self.waves_view.set_medium(c, elements)

    # ------------------------------------------------------------- the waves' run
    @property
    def running(self) -> bool:
        """The pulse playing (the page's one motion)."""
        return self.waves_view.running

    @property
    def play_button(self) -> RunButton:
        return self.waves_view.play_button

    @property
    def loop(self) -> FrameLoop:
        return self.waves_view.loop

    def _shown(self) -> list[Scheme]:
        return [s for s in SCHEMES if self.shown[s].isChecked()]

    def _show_info(self) -> None:
        c = self.continuum
        h = c.length / self.elements.value()
        if c.kind is Kind.BEAM:
            speed = f"a = √(EI/ρA) = {c.speed:.4g} m²/s (ω = aκ²)"
        else:
            speed = f"wave speed c = {c.speed:.4g} m/s"
        rigid = {0: "", 1: "; one rigid-body mode", 2: "; two rigid-body modes"}[c.rigid_modes]
        self.medium_info.setText(f"{speed}; mass {c.mass_per_length * c.length:.4g} kg{rigid}.")
        cut = math.pi / h
        tops = []
        for s in SCHEMES:
            top = self._mesh_omega(s, np.array([cut]))[0, 0] / TWO_PI
            tops.append(f"{s.value.lower()} {fmt_hz(top)}")
        self.mesh_info.setText(
            f"h = {h * 1e3:.4g} mm. Cut-off at κ = π/h = {cut:.4g} rad/m (λ = 2h), where the "
            f"continuum is at {fmt_hz(float(c.omega(cut)) / TWO_PI)}; the meshes top out "
            f"at {', '.join(tops)}."
        )

    def _mesh_omega(self, scheme: Scheme, kappa: np.ndarray) -> np.ndarray:
        """A mesh's dispersion (rad/s), a column per branch: the string's and bar's in closed form."""
        c, h = self.continuum, self.continuum.length / self.elements.value()
        if c.kind is Kind.BEAM:
            return dispersion(c, h, scheme, kappa)
        return string_dispersion(c, h, scheme, kappa)[:, None]

    def _fill_table(self) -> None:
        exact_count = self._exact_kappa.size
        exact = self.continuum.omega(self._exact_kappa) / TWO_PI
        self.table.blockSignals(True)
        self.table.clear()
        self.table.setColumnCount(2 + 2 * len(SCHEMES))
        self.table.setHorizontalHeaderLabels(["Mode", "Exact"] + [x for s in SCHEMES for x in (s.value, "error")])
        self.table.setRowCount(exact_count)
        for i in range(exact_count):
            self.table.setItem(i, 0, _cell(f"{i + 1}"))
            self.table.setItem(i, 1, _cell(fmt_hz(exact[i])))
        for j, s in enumerate(SCHEMES):
            f = flexible_hz(self.models[s])
            for i, fi in enumerate(f):
                self.table.setItem(i, 2 + 2 * j, _cell(fmt_hz(fi)))
                self.table.setItem(i, 3 + 2 * j, _cell(f"{100 * (fi / exact[i] - 1):+.3g} %"))
        self.table.resizeColumnsToContents()
        self.table.blockSignals(False)

    def _on_mode(self, n: int) -> None:
        self.redraw()
        self.table.blockSignals(True)
        self.table.selectRow(n - 1)
        self.table.blockSignals(False)

    def _on_table_selection(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.mode.setValue(rows[0].row() + 1)

    def _on_errors_click(self, event) -> None:
        vb = self.errors.getPlotItem().vb
        if not self.mode.isEnabled() or not vb.sceneBoundingRect().contains(event.scenePos()):
            return
        x = vb.mapSceneToView(event.scenePos()).x()
        self.mode.setValue(int(np.clip(round(x), 1, self.mode.maximum())))

    # ------------------------------------------------------------- waves
    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        self.waves_view.apply_theme()
        self.redraw()

    def redraw(self) -> None:
        self._draw_dispersion()
        self.lazy.stale(self.errors_tab)
        self._draw_mode()

    def _draw_dispersion(self) -> None:
        p, c = self.dispersion, self.continuum
        p.clear()
        self.dispersion_legend.clear()
        h = c.length / self.elements.value()
        cut = math.pi / h
        kappa = np.linspace(0.0, cut, KAPPA_POINTS)
        shown = self._shown()
        count = max((flexible_hz(self.models[s]).size for s in shown), default=0)
        right = max(PAST_CUTOFF * cut, 1.02 * self._exact_kappa[count - 1] if count else 0.0)
        far = np.linspace(0.0, right, KAPPA_POINTS)
        exact = p.plot(far, c.omega(far) / TWO_PI, pen=pg.mkPen(colors.strong, width=2))
        self.dispersion_legend.addItem(exact, "Continuum")
        top = float(c.omega(cut)) / TWO_PI
        for s in shown:
            color = scheme_color(s)
            w = self._mesh_omega(s, kappa) / TWO_PI
            curve = p.plot(kappa, w[:, 0], pen=pg.mkPen(color, width=2))
            self.dispersion_legend.addItem(curve, s.value)
            for b in range(1, w.shape[1]):  # the slopes' branch, in the next zone: from π/h on to 2π/h
                p.plot(2 * cut - kappa, w[:, b], pen=pg.mkPen(color, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
            f = flexible_hz(self.models[s])
            if f.size:
                p.plot(
                    self._exact_kappa[: f.size],
                    f,
                    pen=None,
                    symbol="o",
                    symbolSize=6,
                    symbolBrush=color,
                    symbolPen=pg.mkPen(colors.background),
                )
            top = max(top, float(w.max()))
        line = pg.InfiniteLine(
            cut,
            angle=90,
            pen=pg.mkPen(colors.muted, width=1, style=QtCore.Qt.PenStyle.DashLine),
            label="π/h",
            labelOpts={"position": 0.95, "color": colors.muted},
        )
        p.addItem(line)
        n = self.mode.value() if self.mode.isEnabled() else 0
        if n:
            k = self._exact_kappa[n - 1]
            p.addItem(
                pg.InfiniteLine(k, angle=90, pen=pg.mkPen(colors.force, width=1, style=QtCore.Qt.PenStyle.DotLine))
            )
        p.setXRange(0.0, right, padding=0.02)
        p.setYRange(0.0, 1.1 * top, padding=0.0)
        self._show_readout()

    def _show_readout(self) -> None:
        c = self.continuum
        if not self.mode.isEnabled():
            self.readout.setText("No flexible mode: one element between two held ends has nothing free to move.")
            return
        n = self.mode.value()
        h = c.length / self.elements.value()
        k = self._exact_kappa[n - 1]
        per = 2 * math.pi / (k * h)
        exact = float(c.omega(k)) / TWO_PI
        parts = []
        for s in self._shown():
            f = flexible_hz(self.models[s])
            if f.size >= n:
                parts.append(f"{s.value.lower()} {fmt_hz(f[n - 1])} ({100 * (f[n - 1] / exact - 1):+.3g} %)")
        found = "; ".join(parts) if parts else "no mesh shown"
        self.readout.setText(
            f"<b>Mode {n}</b>: κ = {k:.4g} rad/m, a wavelength of {2 * math.pi / k:.4g} m, "
            f"<b>{per:.3g} elements</b> (κh = {k * h:.3g}). Exact {fmt_hz(exact)}; {found}."
        )

    def _draw_errors(self) -> None:
        p = self.errors
        p.clear()
        self.errors_legend.clear()
        p.addItem(pg.InfiniteLine(0.0, angle=0, pen=pg.mkPen(colors.faint, width=1)))
        exact = self.continuum.omega(self._exact_kappa) / TWO_PI
        for s in self._shown():
            f = flexible_hz(self.models[s])
            if not f.size:
                continue
            err = 100 * (f / exact[: f.size] - 1)
            curve = p.plot(
                np.arange(1, f.size + 1),
                err,
                pen=pg.mkPen(scheme_color(s), width=1.5),
                symbol="o",
                symbolSize=5,
                symbolBrush=scheme_color(s),
                symbolPen=None,
            )
            self.errors_legend.addItem(curve, s.value)
        n = self.mode.value() if self.mode.isEnabled() else 0
        if n:
            p.addItem(
                pg.InfiniteLine(n, angle=90, pen=pg.mkPen(colors.force, width=1, style=QtCore.Qt.PenStyle.DotLine))
            )
        self._draw_convergence()

    def _draw_convergence(self) -> None:
        p = self.convergence
        p.clear()
        self.convergence_legend.clear()
        if not self.mode.isEnabled():
            self.convergence_note.setText("")
            return
        n = self.mode.value()
        rates = []
        for s in self._shown():
            counts, errs = convergence(self.continuum, s, n)
            if not counts.size:
                continue
            curve = p.plot(
                counts,
                np.abs(errs),
                pen=pg.mkPen(scheme_color(s), width=1.5),
                symbol="o",
                symbolSize=5,
                symbolBrush=scheme_color(s),
                symbolPen=None,
            )
            self.convergence_legend.addItem(curve, s.value)
            rate = order(counts, errs)
            if rate is not None:
                rates.append(f"{s.value.lower()} N<sup>−{rate:.2g}</sup>")
        p.addItem(
            pg.InfiniteLine(
                math.log10(self.elements.value()),
                angle=90,
                pen=pg.mkPen(colors.muted, width=1, style=QtCore.Qt.PenStyle.DotLine),
            )
        )
        p.setXRange(0.0, math.log10(CONVERGENCE_COUNTS[-1]), padding=0.03)
        self.convergence_note.setText(
            f"Mode {n}'s error falls as " + ", ".join(rates) + " (the three finest meshes' slopes)."
            if rates
            else f"No mesh shown has a mode {n} yet."
        )

    def _draw_mode(self) -> None:
        p, c = self.shape, self.continuum
        p.clear()
        self.shape_legend.clear()
        if not self.mode.isEnabled():
            self.shape_note.setText("")
            return
        n = self.mode.value()
        x = np.linspace(0.0, c.length, SHAPE_POINTS)
        k = self._exact_kappa[n - 1]
        exact = c.exact_shape(k, x)
        p.addItem(pg.InfiniteLine(0.0, angle=0, pen=pg.mkPen(colors.faint, width=1)))
        curve = p.plot(x, exact, pen=pg.mkPen(colors.strong, width=2))
        self.shape_legend.addItem(curve, "Exact")
        notes = []
        reach = 1.0
        for s in self._shown():
            model = self.models[s]
            hz, shapes = model.modes()
            i = n - 1 + c.rigid_modes
            if i >= hz.size:
                continue
            shape = np.array(shapes[:, i])
            nodes = model.deflections(shape)
            sign = np.sign(nodes @ c.exact_shape(k, model.nodes)) or 1.0
            color = scheme_color(s)
            field = sign * model.field(shape, x)
            reach = max(reach, float(np.max(np.abs(field))))
            line = p.plot(x, field, pen=pg.mkPen(color, width=1.5))
            p.plot(
                model.nodes,
                sign * nodes,
                pen=None,
                symbol="o",
                symbolSize=7,
                symbolBrush=color,
                symbolPen=pg.mkPen(colors.background),
            )
            self.shape_legend.addItem(line, s.value)
            at = c.exact_shape(k, model.nodes)
            mac = (nodes @ at) ** 2 / ((nodes @ nodes) * (at @ at))
            notes.append(f"{s.value.lower()} {mac:.6f}")
        p.setXRange(0.0, c.length, padding=0.02)
        p.setYRange(-1.1 * reach, 1.65 * reach, padding=0.0)  # room for the legend at the top
        per = 2 * math.pi / (k * c.length / self.elements.value())
        self.shape_note.setText(
            f"Mode {n}, {per:.3g} elements a wavelength. The MAC of each mesh's nodes against the exact shape "
            "there (1: the same shape): " + ", ".join(notes) + "."
            if notes
            else f"Mode {n}: no mesh shown has it."
        )


def _cell(text: str) -> QtWidgets.QTableWidgetItem:
    item = QtWidgets.QTableWidgetItem(text)
    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
    return item


def _holder(layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
    w = QtWidgets.QWidget()
    w.setLayout(layout)
    return w
