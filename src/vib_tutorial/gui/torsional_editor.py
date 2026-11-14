"""The Torsional page's drive-train editor: pick a disc or a link and edit it, add and remove discs, make a shaft a
gear mesh and back, and the excitation's pane (`excitation_box`, beside it in the page's left column).

It holds the train it shows (`show_train`, from the page) and builds a new one from each edit, which it hands
the page by `built` (the page then shows it everywhere, this editor too); an edit that can't be built is said
in its info line, and the page keeps the last train. `picked` says which element is chosen (the page's drawing
marks it); `select` chooses one from the page (a click on the drawing).

Working on it: `test_gui_smoke.py`'s `test_torsional_page` edits through it.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ..core.torsional import PRESETS, Disc, DriveTrain, Excitation, Mesh, Shaft
from .layouts import FlowLayout, form_layout, form_row, group, show_in_stack
from .lesson import connect_all
from .panels import spin
from .theming import mute

MAX_DISCS = 12
TIPS = {
    "inertia": "<p>The disc's polar moment of inertia J about its shaft.</p>",
    "disc_damping": "<p>A damper from the disc to ground: a fan's or pump's load, which rises with speed "
    "(its slope dT/dΩ), or windage. It damps every mode the disc swings in.</p>",
    "stiffness": "<p>The shaft's (or coupling's) torsional stiffness: torque per radian of twist, GJ/L for a "
    "plain shaft.</p>",
    "shaft_damping": "<p>The shaft's damping: torque per rad/s of twist rate. A rubber coupling has a lot; "
    "steel very little.</p>",
    "teeth": "<p>Teeth on the gear on the motor's side (in) and on the far side (out). The far shaft turns "
    "at z<sub>in</sub>/z<sub>out</sub> times the near one's speed: more teeth out slows it down.</p>",
    "base_radius": "<p>The base radius of the gear on the motor's side; the far gear's is r z<sub>out</sub> / "
    "z<sub>in</sub>.</p>",
    "mesh_stiffness": "<p>The mesh's stiffness along the line of action: the teeth bending and the contact "
    "flattening, about 14 N/µm per mm of face width for steel spur gears.</p>",
    "mesh_damping": "<p>The mesh's damping along the line of action.</p>",
    "error": "<p>The transmission error: how far the far gear lags or leads where perfect involutes would put "
    "it, here one harmonic at the tooth-pass frequency (teeth × speed). It drives the mesh as an "
    "imposed stretch.</p>",
    "excite": "<p>A pulsating torque on one disc, each harmonic an order of that disc's own speed: an "
    "engine's firing (a four-stroke six's majors are 3× and 6×), a reciprocating compressor's "
    "strokes, a misaligned coupling's 2×.</p>",
}


class TrainEditor(QtWidgets.QGroupBox):
    """The Drive train pane: the element chosen and its editor, add and remove; and the Excitation pane."""

    built = QtCore.Signal(object)  # a DriveTrain, edited
    picked = QtCore.Signal(int)  # the element chosen in the list: 2i a disc, 2i + 1 the link after it

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Drive train", parent)
        self.train: DriveTrain = PRESETS[0].train
        self.selected = 0  # into elements: 2i a disc, 2i + 1 the link after it
        self._loading = False

        tv = QtWidgets.QVBoxLayout(self)
        self.element = QtWidgets.QComboBox()
        self.element.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.element.setToolTip("<p>The disc or link to edit: or click it in the <i>Drive train</i> drawing.</p>")
        tv.addWidget(self.element)
        self.editor = QtWidgets.QStackedWidget()
        tv.addWidget(self.editor)

        disc_page = QtWidgets.QWidget()
        form = form_layout(disc_page)
        form.setContentsMargins(0, 0, 0, 0)
        self.disc_name = QtWidgets.QLineEdit()
        self.inertia = spin(1e-5, 1e5, 1.0, 4, " kg·m²")
        self.disc_damping = spin(0.0, 1e6, 0.0, 2, " N·m·s/rad")
        form_row(form, "Name:", self.disc_name, "<p>What the disc is called.</p>")
        form_row(form, "Inertia J:", self.inertia, TIPS["inertia"])
        form_row(form, "Damping to ground:", self.disc_damping, TIPS["disc_damping"])
        self.editor.addWidget(disc_page)

        shaft_page = QtWidgets.QWidget()
        form = form_layout(shaft_page)
        form.setContentsMargins(0, 0, 0, 0)
        self.shaft_name = QtWidgets.QLineEdit()
        self.shaft_stiffness = spin(1e-3, 1e7, 100.0, 3, " kN·m/rad")
        self.shaft_damping = spin(0.0, 1e6, 0.0, 2, " N·m·s/rad")
        form_row(form, "Name:", self.shaft_name, "<p>What the shaft or coupling is called.</p>")
        form_row(form, "Stiffness k:", self.shaft_stiffness, TIPS["stiffness"])
        form_row(form, "Damping c:", self.shaft_damping, TIPS["shaft_damping"])
        self.to_mesh = QtWidgets.QPushButton("Make it a gear mesh")
        self.to_mesh.setToolTip("<p>Replace this shaft by a gear pair: the discs either side become the gears.</p>")
        form.addRow(self.to_mesh)
        self.editor.addWidget(shaft_page)

        mesh_page = QtWidgets.QWidget()
        form = form_layout(mesh_page)
        form.setContentsMargins(0, 0, 0, 0)
        self.mesh_name = QtWidgets.QLineEdit()
        self.teeth_in = QtWidgets.QSpinBox()
        self.teeth_out = QtWidgets.QSpinBox()
        for box in (self.teeth_in, self.teeth_out):
            box.setRange(5, 1000)
            box.setKeyboardTracking(False)
        self.base_radius = spin(1.0, 5000.0, 100.0, 1, " mm")
        self.mesh_stiffness = spin(1.0, 1e5, 1000.0, 1, " N/µm")
        self.mesh_damping = spin(0.0, 1e7, 0.0, 0, " N·s/m")
        self.error = spin(0.0, 1000.0, 0.0, 2, " µm")
        form_row(form, "Name:", self.mesh_name, "<p>What the mesh is called.</p>")
        form_row(form, "Teeth:", group(self.teeth_in, "in,", self.teeth_out, "out"), TIPS["teeth"])
        form_row(form, "Base radius (in):", self.base_radius, TIPS["base_radius"])
        form_row(form, "Mesh stiffness:", self.mesh_stiffness, TIPS["mesh_stiffness"])
        form_row(form, "Mesh damping:", self.mesh_damping, TIPS["mesh_damping"])
        form_row(form, "Transmission error:", self.error, TIPS["error"])
        self.to_shaft = QtWidgets.QPushButton("Make it a shaft")
        self.to_shaft.setToolTip(
            "<p>Replace the gear pair by a shaft of the mesh's referred stiffness "
            "(the far side then turns at the near side's speed).</p>"
        )
        form.addRow(self.to_shaft)
        self.editor.addWidget(mesh_page)

        row = FlowLayout()
        self.add_button = QtWidgets.QPushButton("Add a disc after")
        self.add_button.setToolTip(
            "<p>A copy of the chosen disc (or of the disc before the chosen link) after it, on a copy of its shaft.</p>"
        )
        self.remove_button = QtWidgets.QPushButton("Remove disc")
        self.remove_button.setToolTip("<p>Take the chosen disc off the train, with the link after it.</p>")
        row.add(self.add_button, self.remove_button)
        tv.addLayout(row)
        self.train_info = QtWidgets.QLabel()
        self.train_info.setWordWrap(True)
        mute(self.train_info)
        tv.addWidget(self.train_info)

        self.excitation_box = excite = QtWidgets.QGroupBox("Excitation")
        excite.setToolTip(TIPS["excite"])
        ef = form_layout(excite)
        self.excite_disc = QtWidgets.QComboBox()
        self.excite_disc.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.excite_name = QtWidgets.QLineEdit()
        self.orders = [spin(0.0, 500.0, 0.0, 2, "×") for _ in range(2)]
        self.amplitudes = [spin(0.0, 1e7, 0.0, 1, " N·m") for _ in range(2)]
        form_row(ef, "Torque on:", self.excite_disc, TIPS["excite"])
        form_row(ef, "Called:", self.excite_name, "<p>What the excitation is (its orders' label).</p>")
        for k in range(2):
            form_row(
                ef,
                f"Harmonic {k + 1}:",
                group(self.orders[k], self.amplitudes[k]),
                "<p>An order of the disc's own speed and its torque's amplitude (0: off).</p>",
            )

        self.element.currentIndexChanged.connect(self._on_element)
        connect_all(
            (
                self.inertia,
                self.disc_damping,
                self.shaft_stiffness,
                self.shaft_damping,
                self.base_radius,
                self.mesh_stiffness,
                self.mesh_damping,
                self.error,
                self.teeth_in,
                self.teeth_out,
            ),
            self._on_edit,
        )
        for line in (self.disc_name, self.shaft_name, self.mesh_name):
            line.editingFinished.connect(self._on_edit)
        self.to_mesh.clicked.connect(self._make_mesh)
        self.to_shaft.clicked.connect(self._make_shaft)
        self.add_button.clicked.connect(self._add_disc)
        self.remove_button.clicked.connect(self._remove_disc)
        self.excite_disc.currentIndexChanged.connect(self._on_excitation)
        self.excite_name.editingFinished.connect(self._on_excitation)
        connect_all((*self.orders, *self.amplitudes), self._on_excitation)

    def show_train(self, train: DriveTrain) -> None:
        """Show `train` (the page's, after an edit or a load): the lists, the editor and the info line."""
        self.train = train
        self.selected = min(self.selected, 2 * train.n - 2)
        self._fill_lists()
        self._fill_editor()
        self._show_info()

    def _try(self, make) -> None:
        """Apply the train `make()` builds, or say why it can't be built and keep the last one."""
        try:
            train = make()
        except ValueError as e:
            self.train_info.setText(f"<b>{e}</b>: the page keeps the last train.")
            return
        self.built.emit(train)

    def _fill_lists(self) -> None:
        self._loading = True
        t = self.train
        self.element.clear()
        for i, d in enumerate(t.discs):
            self.element.addItem(f"{i + 1}. {d.name}")
            if i < len(t.links):
                link = t.links[i]
                kind = f"mesh {link.teeth_in}:{link.teeth_out}" if isinstance(link, Mesh) else "shaft"
                self.element.addItem(f"    {link.name} ({kind})")
        self.element.setCurrentIndex(self.selected)
        self.excite_disc.clear()
        self.excite_disc.addItem("None")
        self.excite_disc.addItems([d.name for d in t.discs])
        exc = t.excitation
        self.excite_disc.setCurrentIndex(0 if exc is None else exc.disc + 1)
        harmonics = list(exc.harmonics) if exc is not None else []
        harmonics += [(0.0, 0.0)] * (2 - len(harmonics))
        for k in range(2):
            self.orders[k].setValue(harmonics[k][0])
            self.amplitudes[k].setValue(harmonics[k][1])
        self.excite_name.setText(exc.name if exc is not None else "Torque")
        for w in (self.excite_name, *self.orders, *self.amplitudes):
            w.setEnabled(exc is not None)
        self.add_button.setEnabled(t.n < MAX_DISCS)
        self.remove_button.setEnabled(t.n > 2)
        self._loading = False

    def _fill_editor(self) -> None:
        self._loading = True
        t = self.train
        i, is_link = divmod(self.selected, 2)
        page = 0 if not is_link else 1 if isinstance(t.links[i], Shaft) else 2
        show_in_stack(self.editor, page)  # the stack as tall as the page showing, not the tallest
        if not is_link:
            d = t.discs[i]
            self.disc_name.setText(d.name)
            self.inertia.setValue(d.inertia)
            self.disc_damping.setValue(d.damping)
        elif isinstance(t.links[i], Shaft):
            s = t.links[i]
            self.shaft_name.setText(s.name)
            self.shaft_stiffness.setValue(s.stiffness / 1e3)
            self.shaft_damping.setValue(s.damping)
        else:
            m = t.links[i]
            assert isinstance(m, Mesh), "a link is a shaft or a mesh"
            self.mesh_name.setText(m.name)
            self.teeth_in.setValue(m.teeth_in)
            self.teeth_out.setValue(m.teeth_out)
            self.base_radius.setValue(m.base_radius * 1e3)
            self.mesh_stiffness.setValue(m.stiffness / 1e6)
            self.mesh_damping.setValue(m.damping)
            self.error.setValue(m.error * 1e6)
        self._loading = False

    def select(self, element: int) -> None:
        """Edit element `element`: 2i the i-th disc, 2i + 1 the link after it."""
        self.selected = element
        self.element.blockSignals(True)
        self.element.setCurrentIndex(element)
        self.element.blockSignals(False)
        self._fill_editor()

    def _on_element(self, index: int) -> None:
        if not self._loading and index >= 0:
            self.select(index)
            self.picked.emit(index)

    def _on_edit(self) -> None:
        if self._loading:
            return
        t = self.train
        i, is_link = divmod(self.selected, 2)
        if not is_link:
            disc = Disc(
                self.disc_name.text().strip() or t.discs[i].name, self.inertia.value(), self.disc_damping.value()
            )
            self._try(lambda: t.with_disc(i, disc))
        elif isinstance(t.links[i], Shaft):
            shaft = Shaft(
                self.shaft_stiffness.value() * 1e3,
                self.shaft_damping.value(),
                self.shaft_name.text().strip() or t.links[i].name,
            )
            self._try(lambda: t.with_link(i, shaft))
        else:
            mesh = Mesh(
                self.teeth_in.value(),
                self.teeth_out.value(),
                self.base_radius.value() / 1e3,
                self.mesh_stiffness.value() * 1e6,
                self.mesh_damping.value(),
                self.error.value() / 1e6,
                self.mesh_name.text().strip() or t.links[i].name,
            )
            self._try(lambda: t.with_link(i, mesh))

    def _make_mesh(self) -> None:
        i = self.selected // 2
        shaft = self.train.links[i]
        # a 1:1 pair of 100 mm base radius whose referred stiffness is the shaft's
        r = 0.1
        mesh = Mesh(40, 40, r, shaft.stiffness / (r * r), shaft.damping / (r * r), 0.0, "Gear mesh")
        self._try(lambda: self.train.with_link(i, mesh))

    def _make_shaft(self) -> None:
        i = self.selected // 2
        mesh = self.train.links[i]
        assert isinstance(mesh, Mesh), "offered for a mesh"
        r = mesh.base_radius
        self._try(lambda: self.train.with_link(i, Shaft(mesh.stiffness * r * r, mesh.damping * r * r, "Shaft")))

    def _add_disc(self) -> None:
        i = self.selected // 2
        self.selected = 2 * (i + 1)
        self._try(lambda: self.train.inserted(i))

    def _remove_disc(self) -> None:
        i = self.selected // 2
        self.selected = 2 * max(i - 1, 0)
        self._try(lambda: self.train.removed(i))

    def _on_excitation(self) -> None:
        if self._loading:
            return
        disc = self.excite_disc.currentIndex() - 1
        if disc < 0:
            self._try(lambda: self.train.with_(excitation=None))
            return
        harmonics = tuple((self.orders[k].value(), self.amplitudes[k].value()) for k in range(2))
        if self.train.excitation is None and not any(o > 0 and a > 0 for o, a in harmonics):
            harmonics = ((1.0, 100.0),)  # a fresh excitation: something to see
        name = self.excite_name.text().strip() or "Torque"
        self._try(lambda: self.train.with_(excitation=Excitation(disc, harmonics, name)))

    def _show_info(self) -> None:
        t = self.train
        ref = t.referred()
        j = sum(d.inertia for d in ref.discs)
        self.train_info.setText(
            f"{t.n} discs, {len(t.meshes)} gear mesh{'es' if len(t.meshes) != 1 else ''}; "
            f"{j:.4g} kg·m² referred to the motor's shaft."
        )
