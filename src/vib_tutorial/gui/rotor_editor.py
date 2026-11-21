"""The Jeffcott rotor page's parameters: the *Shaft and disc* and *Bearing supports* boxes, and the rotor they
describe (#218 C2: split from `rotor.py`).

`RotorEditor` owns the boxes and the ties between them (isotropic supports, B the same as A, a film's or a
magnet's journal holding only its mass), and reads and writes a `RotorSystem`: `system()` builds one from the
boxes and the film, magnet and rub panels (the page's tabs, given to it), `load(system)` fills them without
`changed`. It keeps what the boxes don't show: the balance weights (`set_weights`) and an unbalance a balancing
exercise hides (`hide_unbalance`, `reveal_unbalance`). `changed` fires on any box's change; the page then
works the rotor out again (`RotorPage._on_params`).

Working on it: the rotor page's tests in `tests/test_gui_smoke.py` drive these boxes (`page.editor.<box>`).
"""

from __future__ import annotations

import math

from PySide6 import QtCore, QtWidgets

from ..core.fluid_bearing import JournalBearing
from ..core.magnetic_bearing import MagneticBearing
from ..core.rotor import CRACK_DEPTH_MAX, STEEL, Bearing, Crack, PlaneWeight, RotorSystem
from . import accessible
from .layouts import form_layout, wrapped
from .panels import NumberBox, spin
from .rotor_bearing import BearingPanel
from .rotor_magnet import MagnetPanel
from .rotor_rub import RubPanel
from .rotor_tips import (
    ASYMMETRY_TIP,
    BOW_TIP,
    CRACK_TIP,
    ELEMENTS_TIP,
    FILM_TIP,
    GRAVITY_TIP,
    INTERNAL_TIP,
    KXY_TIP,
    MAGNET_TIP,
    NONLINEAR_FILM_TIP,
    SHAFT_MASS_TIP,
    SKEW_TIP,
    SWIRL_TIP,
)

SUPPORTS = ("Springs and dampers", "Oil-film journal bearings", "Magnetic bearings")
SPRINGS, FILMS, MAGNETS = range(len(SUPPORTS))
BEARING_WIDTHS = (56, 56, 50, 50, 46, 56, 40)  # px, the bearing table's columns: kx, ky, cx, cy, m, kxy, λ
BEARING_GAP = 0.02  # of the span: the closest the disc may be put to a bearing
GRAVITY = 9.81  # m/s²


# Every angle on the rotor counts the same way, said on each angle box.
ANGLE_NOTE = (
    "<p><i>Angles</i> on this page count from the keyphasor mark, counter-clockwise seen from beyond bearing B: "
    "the way the shaft turns. A phase counts the same way, as a lead, so a lag is negative (field instruments "
    "mostly count lag: see the notes' <i>Which way phase is counted</i>).</p>"
)


class RotorEditor(QtWidgets.QWidget):
    """The rotor's boxes: Shaft and disc, then Bearing supports (see the module's docstring)."""

    changed = QtCore.Signal()

    def __init__(
        self,
        bearing_panel: BearingPanel,
        magnet_panel: MagnetPanel,
        rub_panel: RubPanel,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.bearing_panel, self.magnet_panel, self.rub_panel = bearing_panel, magnet_panel, rub_panel
        self.weights: tuple[complex, ...] = ()  # balance weights on the disc (kg·m)
        self.plane_weights: tuple[PlaneWeight, ...] = ()  # weights in balancing planes
        self.hidden: dict[str, float] | None = None  # RotorSystem fields kept out of the boxes (SI)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._rotor_box())
        layout.addWidget(self._bearing_box())

    def _emit(self, *_) -> None:
        self.changed.emit()

    def _rotor_box(self) -> QtWidgets.QGroupBox:
        s = RotorSystem()
        box = QtWidgets.QGroupBox("Shaft and disc")
        form = form_layout(box)
        self.length = spin(0.05, 5.0, s.length, 3, " m")
        self.length.setToolTip("Bearing span L")
        self.diameter = spin(1.0, 500.0, s.diameter * 1e3, 1, " mm")
        self.diameter.setToolTip("Shaft diameter d; EI = E π d⁴/64")
        self.youngs = spin(1.0, 1000.0, s.youngs / 1e9, 1, " GPa")
        self.youngs.setToolTip("Young's modulus E of the shaft (steel ≈ 200 GPa)")
        self.position = PositionBox()
        self.position.setValue(s.position)
        self.position.setToolTip(
            "<p>Where the disc sits, a/L from bearing A. At 0.5 (midspan) with identical bearings the "
            "disc's translation and tilt are uncoupled, so the spin's gyroscopic moment never shows in "
            "the translational whirl: the classic Jeffcott rotor. Move it off centre and the disc tilts "
            "as it whirls.</p><p>Above 1 the disc overhangs bearing B (below 0, bearing A): an "
            "<i>overhung</i> rotor, like a fan or a pump impeller. The disc then tilts as much as it "
            "moves, and the gyroscopic moment moves even the first critical speed.</p>"
        )
        self.disc_mass = spin(0.01, 1000.0, s.disc_mass, 3, " kg")
        self.ip = spin(1e-6, 100.0, s.ip, 6, " kg·m²")
        self.ip.setToolTip(
            "Polar moment of inertia I<sub>p</sub>, about the spin axis (a thin disc of radius r<sub>d</sub>: m r<sub>d</sub>²/2)"
        )
        self.id = spin(1e-6, 100.0, s.id, 6, " kg·m²")
        self.id.setToolTip(
            "Diametral moment of inertia I<sub>d</sub>, about a diameter (a thin disc: m r<sub>d</sub>²/4 = I<sub>p</sub>/2)"
        )
        self.unbalance = spin(0.0, 1e6, s.unbalance * 1e6, 1, " g·mm")
        self.unbalance.setToolTip(
            "<p>Mass unbalance U = m e: the disc's mass times the distance e of its centre of mass from "
            "the shaft's axis. 20 g·mm on a 1 kg disc is e = 20 µm, about balance grade G6.3 at "
            "3000 rpm (e Ω = 6.3 mm/s).</p>"
        )
        form.addRow("Span L:", self.length)
        form.addRow("Shaft diameter d:", self.diameter)
        form.addRow("Young's modulus:", self.youngs)
        self.shaft_mass = QtWidgets.QCheckBox("Shaft mass, ρ")
        self.shaft_mass.setToolTip(SHAFT_MASS_TIP)
        self.density = spin(100.0, 30000.0, STEEL, 0, " kg/m³")
        self.density.setToolTip("The shaft's density (steel ≈ 7850 kg/m³, aluminium ≈ 2700)")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.shaft_mass)
        row.addWidget(self.density, 1)
        form.addRow("Shaft:", row)
        self.density.setAccessibleName("Shaft density")
        self.elements = QtWidgets.QSpinBox()
        self.elements.setRange(2, 60)
        self.elements.setValue(s.elements)
        self.elements.setKeyboardTracking(False)
        self.mass_matrix = QtWidgets.QComboBox()
        self.mass_matrix.addItems(["Consistent", "Lumped"])
        for w in (self.elements, self.mass_matrix):
            w.setToolTip(ELEMENTS_TIP)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.elements, 1)
        row.addWidget(self.mass_matrix, 1)
        form.addRow("Beam elements:", row)
        self.elements.setAccessibleName("Beam elements")
        self.mass_matrix.setAccessibleName("Mass matrix")
        self.shaft_mass.toggled.connect(self._emit)
        self.mass_matrix.currentIndexChanged.connect(self._emit)
        form.addRow("Disc position a/L:", self.position)
        form.addRow("Disc mass m:", self.disc_mass)
        form.addRow("Polar inertia I<sub>p</sub>:", self.ip)
        form.addRow("Diametral inertia I<sub>d</sub>:", self.id)
        self.unbalance_angle = spin(-180.0, 180.0, math.degrees(s.unbalance_angle), 0, "°")
        self.unbalance_angle.setToolTip(
            "<p>Where the heavy spot is: its angle from the keyphasor mark on the shaft, in the direction "
            "of spin. The probes' phase is measured from the mark, as on a real machine.</p>" + ANGLE_NOTE
        )
        self.bow = spin(0.0, 5000.0, s.bow * 1e6, 1, " µm")
        self.bow_angle = spin(-180.0, 180.0, math.degrees(s.bow_angle), 0, "°")
        self.skew = spin(0.0, 10.0, math.degrees(s.skew), 3, "°")
        self.skew_angle = spin(-180.0, 180.0, math.degrees(s.skew_angle), 0, "°")
        self.asymmetry = spin(0.0, 90.0, s.asymmetry * 100, 1, " %")
        self.asymmetry_angle = spin(-180.0, 180.0, math.degrees(s.asymmetry_angle), 0, "°")
        crack = Crack()
        self.crack_depth = spin(0.0, CRACK_DEPTH_MAX * 100, 0.0, 1, " %")
        self.crack_depth.setSingleStep(5.0)
        self.crack_angle = spin(-180.0, 180.0, math.degrees(crack.angle), 0, "°")
        for size, angle, tip, label in (
            (self.unbalance, self.unbalance_angle, None, "Unbalance m·e:"),
            (self.bow, self.bow_angle, BOW_TIP, "Shaft bow δ<sub>b</sub>:"),
            (self.skew, self.skew_angle, SKEW_TIP, "Disc skew τ:"),
            (self.asymmetry, self.asymmetry_angle, ASYMMETRY_TIP, "Asymmetry ±Δ:"),
            (self.crack_depth, self.crack_angle, CRACK_TIP, "Crack depth a<sub>c</sub>/d:"),
        ):
            angle.setWrapping(True)
            angle.setStepType(QtWidgets.QAbstractSpinBox.StepType.DefaultStepType)
            angle.setSingleStep(15.0)
            row = QtWidgets.QHBoxLayout()
            row.addWidget(size, 3)
            row.addWidget(QtWidgets.QLabel("at"))
            row.addWidget(angle, 2)
            if tip:
                size.setToolTip(tip)
                angle.setToolTip(tip + ANGLE_NOTE)
            form.addRow(label, row)
            size.setAccessibleName(accessible.plain(label))
            angle.setAccessibleName(f"{accessible.plain(label)}, at")
        self.crack_position = spin(-1.0, 2.0, crack.position, 3, "")
        self.crack_position.setSingleStep(0.05)
        self.crack_breathing = QtWidgets.QCheckBox("Breathing")
        self.crack_breathing.setChecked(crack.breathing)
        self.crack_breathing.toggled.connect(self._emit)
        self.crack_position.setToolTip(CRACK_TIP)
        self.crack_breathing.setToolTip(CRACK_TIP)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.crack_position, 3)
        row.addWidget(self.crack_breathing, 2)
        form.addRow("Crack at z/L:", row)
        self.crack_position.setAccessibleName("Crack at z/L")
        self.internal_damping = spin(0.0, 1e5, s.internal_damping, 1, " N·s/m")
        self.internal_damping.setToolTip(INTERNAL_TIP)
        form.addRow("Internal damping c<sub>i</sub>:", self.internal_damping)
        self.gravity = QtWidgets.QCheckBox(f"Horizontal rotor (g = {GRAVITY} m/s²)")
        self.gravity.setToolTip(GRAVITY_TIP)
        self.gravity.toggled.connect(self._emit)
        form.addRow("Gravity:", self.gravity)
        self.shaft_note = wrapped(muted=True)
        form.addRow(self.shaft_note)
        for box_ in (
            self.length,
            self.diameter,
            self.youngs,
            self.position,
            self.disc_mass,
            self.ip,
            self.id,
            self.unbalance,
            self.unbalance_angle,
            self.bow,
            self.bow_angle,
            self.skew,
            self.skew_angle,
            self.asymmetry,
            self.asymmetry_angle,
            self.crack_depth,
            self.crack_angle,
            self.crack_position,
            self.internal_damping,
            self.density,
            self.elements,
        ):
            box_.valueChanged.connect(self._emit)
        return box

    def _bearing_box(self) -> QtWidgets.QGroupBox:
        b = Bearing()
        box = QtWidgets.QGroupBox("Bearing supports")
        grid = QtWidgets.QGridLayout(box)
        grid.setHorizontalSpacing(4)
        # Units on a second line, so that the headers are no wider than the boxes below them.
        heads = (
            ("k<sub>x</sub><br>[N/m]", "Horizontal support stiffness"),
            ("k<sub>y</sub><br>[N/m]", "Vertical support stiffness"),
            ("c<sub>x</sub><br>[N·s/m]", "Horizontal support damping"),
            ("c<sub>y</sub><br>[N·s/m]", "Vertical support damping"),
            ("m<br>[kg]", "Mass of the journal and whatever moves with it on the support"),
            ("k<sub>xy</sub><br>[N/m]", KXY_TIP),
            ("λ<br>&nbsp;", SWIRL_TIP),
        )
        for c, width in enumerate(BEARING_WIDTHS):
            grid.setColumnStretch(c + 1, width)
        for c, (text, tip) in enumerate(heads):
            label = QtWidgets.QLabel(text)
            label.setToolTip(tip)
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(label, 0, c + 1)
        self.bearing_rows: list[list[QtWidgets.QDoubleSpinBox]] = []
        BEARING_NAMES = ("kx", "ky", "cx", "cy", "mass", "kxy", "swirl λ")
        for r, name in enumerate("AB"):
            grid.addWidget(QtWidgets.QLabel(f"<b>{name}</b>"), r + 1, 0)
            row = [
                spin(1e2, 1e10, b.kx, 0),
                spin(1e2, 1e10, b.ky, 0),
                spin(0.0, 1e6, b.cx, 1),
                spin(0.0, 1e6, b.cy, 1),
                spin(0.001, 100.0, b.mass, 3),
                spin(-1e9, 1e9, b.kxy, 0),
                spin(0.0, 1.0, b.swirl, 2),
            ]
            row[5].setToolTip(KXY_TIP)
            row[6].setToolTip(SWIRL_TIP)
            row[6].setSingleStep(0.05)
            for c, w in enumerate(row):
                # No arrows (the wheel and the arrow keys still step): seven columns leave no room for them.
                w.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
                w.setMinimumWidth(BEARING_WIDTHS[c])
                w.valueChanged.connect(self._emit)
                grid.addWidget(w, r + 1, c + 1)
                w.setAccessibleName(f"Bearing {name}, {BEARING_NAMES[c]}")
            self.bearing_rows.append(row)
        self.isotropic = QtWidgets.QCheckBox("Isotropic (ky = kx, cy = cx)")
        self.isotropic.setChecked(True)
        self.isotropic.setToolTip(
            "<p>Untick to give the supports a different stiffness or damping vertically. The single "
            "critical speed then splits in two, the orbits become ellipses, and between the two "
            "criticals the shaft whirls backward, against its spin.</p>"
        )
        self.same_bearings = QtWidgets.QCheckBox("B same as A")
        self.same_bearings.setChecked(True)
        self.same_bearings.setToolTip(
            "<p>Untick to give the two bearings different supports. Like an off-centre disc, unequal "
            "supports couple the disc's translation and tilt, so the gyroscopic moment shows.</p>"
        )
        self.supports = QtWidgets.QComboBox()
        self.supports.setAccessibleName("Supports")
        self.supports.addItems(SUPPORTS)
        self.supports.setToolTip(
            "<p>What holds the journals: the springs and dampers in the table, or these bearings at both ends.</p>"
        )
        self.supports.setItemData(FILMS, FILM_TIP, QtCore.Qt.ItemDataRole.ToolTipRole)
        self.supports.setItemData(MAGNETS, MAGNET_TIP, QtCore.Qt.ItemDataRole.ToolTipRole)
        self.supports.currentIndexChanged.connect(self._emit)
        self.nonlinear_film = QtWidgets.QCheckBox("Nonlinear film")
        self.nonlinear_film.setChecked(True)
        self.nonlinear_film.setToolTip(NONLINEAR_FILM_TIP)
        for check in (self.isotropic, self.same_bearings, self.nonlinear_film):
            check.toggled.connect(self._emit)
        grid.addWidget(self.isotropic, 3, 0, 1, 8)
        grid.addWidget(self.same_bearings, 4, 0, 1, 8)
        grid.addWidget(self.supports, 5, 0, 1, 5)
        grid.addWidget(self.nonlinear_film, 5, 5, 1, 3)
        return box

    def system(self) -> RotorSystem:
        """The rotor described by the panels, after applying the isotropic and same-as-A ties."""
        iso, same = self.isotropic.isChecked(), self.same_bearings.isChecked()
        kind = self.supports.currentIndex()
        film = kind != SPRINGS  # a film or a magnet holds the journal
        for r, row in enumerate(self.bearing_rows):
            src = self.bearing_rows[0] if same and r == 1 else row
            for c, w in enumerate(row):
                value = src[c].value()
                if iso and c in (1, 3):
                    value = src[c - 1].value()
                if w.value() != value:
                    w.blockSignals(True)
                    w.setValue(value)
                    w.blockSignals(False)
                # A film or a magnet holds the journal in place of the springs: only its mass is left to set.
                w.setEnabled(not (same and r == 1) and not (iso and c in (1, 3)) and not (film and c != 4))
        widget: QtWidgets.QWidget
        for widget in (self.isotropic, self.same_bearings):
            widget.setEnabled(not film)
        # A turning shaft's split time step keeps the films linearized, and has no stator ring.
        round_shaft = not self.asymmetry.value() and not self.crack_depth.value()
        self.nonlinear_film.setEnabled(kind == FILMS and round_shaft)
        self.rub_panel.set_allowed(round_shaft and not (kind == FILMS and self.nonlinear_film.isChecked()))
        self.nonlinear_film.setVisible(kind == FILMS)
        oil = self.bearing_panel.bearing(rotor=False) if kind == FILMS else None
        magnet = self.magnet_panel.bearing() if kind == MAGNETS else None
        bearings = [_bearing(row, oil, magnet) for row in self.bearing_rows]
        for widget in (self.density, self.elements, self.mass_matrix):
            widget.setEnabled(self.shaft_mass.isChecked())
        cracked = self.crack_depth.value() > 0.0
        for widget in (self.crack_angle, self.crack_position, self.crack_breathing):
            widget.setEnabled(cracked)
        return RotorSystem(
            length=self.length.value(),
            diameter=self.diameter.value() / 1e3,
            youngs=self.youngs.value() * 1e9,
            position=_off_bearing(self.position.value()),
            disc_mass=self.disc_mass.value(),
            ip=self.ip.value(),
            id=self.id.value(),
            bearing_a=bearings[0],
            bearing_b=bearings[1],
            unbalance=self.unbalance.value() / 1e6,
            unbalance_angle=math.radians(self.unbalance_angle.value()),
            weights=self.weights,
            plane_weights=self.plane_weights,
            gravity=GRAVITY if self.gravity.isChecked() else 0.0,
            bow=self.bow.value() / 1e6,
            bow_angle=math.radians(self.bow_angle.value()),
            skew=math.radians(self.skew.value()),
            skew_angle=math.radians(self.skew_angle.value()),
            asymmetry=self.asymmetry.value() / 100,
            asymmetry_angle=math.radians(self.asymmetry_angle.value()),
            crack=self._crack() if cracked else None,
            internal_damping=self.internal_damping.value(),
            density=self.density.value() if self.shaft_mass.isChecked() else 0.0,
            elements=self.elements.value(),
            lumped=self.mass_matrix.currentIndex() == 1,
            nonlinear_films=kind == FILMS and self.nonlinear_film.isChecked(),
            stator=self.rub_panel.stator(),
        ).with_(**(self.hidden or {}))

    def _crack(self) -> Crack:
        """The crack the panel describes, kept inside the shaft (and off its ends)."""
        lo, hi = (
            x / self.length.value()
            for x in (
                min(0.0, self.position.value() * self.length.value()),
                max(self.length.value(), self.position.value() * self.length.value()),
            )
        )
        margin = 1e-3 * (hi - lo)
        return Crack(
            position=min(max(self.crack_position.value(), lo + margin), hi - margin),
            depth=self.crack_depth.value() / 100,
            angle=math.radians(self.crack_angle.value()),
            breathing=self.crack_breathing.isChecked(),
        )

    def load(self, system: RotorSystem) -> None:
        """Fill the boxes (and the film, magnet and rub panels) from `system`, without `changed`."""
        boxes = (
            self.length,
            self.diameter,
            self.youngs,
            self.position,
            self.disc_mass,
            self.ip,
            self.id,
            self.unbalance,
            self.unbalance_angle,
            self.bow,
            self.bow_angle,
            self.skew,
            self.skew_angle,
            self.asymmetry,
            self.asymmetry_angle,
            self.crack_depth,
            self.crack_angle,
            self.crack_position,
            self.crack_breathing,
            self.internal_damping,
            self.gravity,
            self.isotropic,
            self.same_bearings,
            self.supports,
            self.nonlinear_film,
            self.shaft_mass,
            self.density,
            self.elements,
            self.mass_matrix,
            *(w for row in self.bearing_rows for w in row),
        )
        for w in boxes:
            w.blockSignals(True)
        self.length.setValue(system.length)
        self.diameter.setValue(system.diameter * 1e3)
        self.youngs.setValue(system.youngs / 1e9)
        self.position.setValue(system.position)
        self.disc_mass.setValue(system.disc_mass)
        self.ip.setValue(system.ip)
        self.id.setValue(system.id)
        self._clear_hidden()
        self.unbalance.setValue(system.unbalance * 1e6)
        self.unbalance_angle.setValue(math.degrees(system.unbalance_angle))
        self.weights = system.weights
        self.plane_weights = system.plane_weights
        self.gravity.setChecked(system.gravity > 0.0)
        self.bow.setValue(system.bow * 1e6)
        self.bow_angle.setValue(math.degrees(system.bow_angle))
        self.skew.setValue(math.degrees(system.skew))
        self.skew_angle.setValue(math.degrees(system.skew_angle))
        self.asymmetry.setValue(system.asymmetry * 100)
        self.asymmetry_angle.setValue(math.degrees(system.asymmetry_angle))
        crack = system.crack
        self.crack_depth.setValue(0.0 if crack is None else crack.depth * 100)
        if crack is not None:
            self.crack_angle.setValue(math.degrees(crack.angle))
            self.crack_position.setValue(crack.position)
            self.crack_breathing.setChecked(crack.breathing)
        self.internal_damping.setValue(system.internal_damping)
        self.shaft_mass.setChecked(system.density > 0.0)
        if system.density > 0.0:
            self.density.setValue(system.density)
        self.elements.setValue(system.elements)
        self.mass_matrix.setCurrentIndex(1 if system.lumped else 0)
        A, B = system.bearing_a, system.bearing_b
        self.isotropic.setChecked(all(b.kx == b.ky and b.cx == b.cy for b in (A, B)))
        self.same_bearings.setChecked(A == B)
        for row, b in zip(self.bearing_rows, (A, B)):
            for w, v in zip(row, (b.kx, b.ky, b.cx, b.cy, b.mass, b.kxy, b.swirl)):
                w.setValue(v)
        self.supports.setCurrentIndex(FILMS if A.film is not None else MAGNETS if A.magnet is not None else SPRINGS)
        self.nonlinear_film.setChecked(system.nonlinear_films or A.film is None)
        if A.film is not None:
            self.bearing_panel.set_bearing(A.film, notify=False)
        if A.magnet is not None:
            self.magnet_panel.set_bearing(A.magnet, notify=False)
        self.rub_panel.set_stator(system.stator)
        for w in boxes:
            w.blockSignals(False)

    def set_weights(self, weights: tuple[complex, ...], plane_weights: tuple[PlaneWeight, ...] = ()) -> None:
        """Put these balance weights on the disc (kg·m each) and in these planes, in place of any there."""
        self.weights = tuple(weights)
        self.plane_weights = tuple(plane_weights)

    def _hideable(self) -> dict[str, tuple[QtWidgets.QDoubleSpinBox, float]]:
        """The boxes a balancing exercise can hide, by RotorSystem field, with the factor from SI to the box."""
        deg = 180.0 / math.pi
        return {
            "unbalance": (self.unbalance, 1e6),
            "unbalance_angle": (self.unbalance_angle, deg),
            "skew": (self.skew, deg),
            "skew_angle": (self.skew_angle, deg),
        }

    def hide_unbalance(self, unbalance: float, angle: float, skew: tuple[float, float] | None = None) -> None:
        """Give the disc this unbalance (kg·m at `angle` rad) without showing it: a balancing exercise.

        With `skew` (τ, its angle in rad), hide a couple unbalance too; otherwise the skew stays as set.
        """
        self._clear_hidden()
        self.hidden = {"unbalance": unbalance, "unbalance_angle": angle}
        if skew is not None:
            self.hidden.update(skew=skew[0], skew_angle=skew[1])
        boxes = self._hideable()
        for name in self.hidden:
            box = boxes[name][0]
            box.blockSignals(True)
            box.setSpecialValueText("hidden")
            box.setValue(box.minimum())
            box.setEnabled(False)
            box.blockSignals(False)

    def reveal_unbalance(self) -> bool:
        """Show the hidden unbalance (and couple) in the panel again; False if none was hidden."""
        if self.hidden is None:
            return False
        hidden, boxes = self.hidden, self._hideable()
        self._clear_hidden()
        for name, value in hidden.items():
            box, scale = boxes[name]
            box.blockSignals(True)
            box.setValue(value * scale)
            box.blockSignals(False)
        return True

    def _clear_hidden(self) -> None:
        if self.hidden is not None:
            for name in self.hidden:
                box = self._hideable()[name][0]
                box.setSpecialValueText("")
                box.setEnabled(True)
        self.hidden = None


def _bearing(
    row: list[QtWidgets.QDoubleSpinBox], film: JournalBearing | None, magnet: MagneticBearing | None
) -> Bearing:
    """A bearing from its row of boxes (kx, ky, cx, cy, mass, kxy, swirl), on a film or a magnet if given."""
    kx, ky, cx, cy, mass, kxy, swirl = (w.value() for w in row)
    return Bearing(kx, ky, cx, cy, mass, kxy, swirl, film=film, magnet=magnet)


class PositionBox(NumberBox):
    """The disc's position a/L: between the bearings or overhung past either, stepping over the bearings."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setRange(-1.0, 2.0)
        self.setDecimals(3)
        self.setSingleStep(0.05)
        self.setKeyboardTracking(False)

    def stepBy(self, steps: int) -> None:
        super().stepBy(steps)
        if _off_bearing(self.value()) != self.value():
            super().stepBy(1 if steps > 0 else -1)


def _off_bearing(position: float) -> float:
    """a/L moved off a bearing: the disc cannot sit on one (it would have no shaft on one side)."""
    for bearing in (0.0, 1.0):
        if abs(position - bearing) < BEARING_GAP:
            inward = 0.5 - bearing  # toward midspan, if it is right on the bearing
            return bearing + math.copysign(BEARING_GAP, position - bearing if position != bearing else inward)
    return position
