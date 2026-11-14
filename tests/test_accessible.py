"""`gui/accessible.py`: every control has a name a screen reader can read (#218 U5)."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from vib_tutorial.gui import accessible  # noqa: E402
from vib_tutorial.gui.layouts import form_layout, group  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def _free_windows(app, free_windows):
    """Each test's windows torn down and freed after it (conftest's `free_windows`)."""


def test_a_control_is_named_after_the_label_beside_it(app):
    panel = QtWidgets.QGroupBox("Panel")
    form = form_layout(panel)
    own = QtWidgets.QSpinBox()
    form.addRow("Masses:", own)  # Qt's own: the label's buddy
    width, at = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()
    form.addRow("Pulse:", group(width, "wide at", at))  # a row of widgets as the field
    step = QtWidgets.QCheckBox("Step at")
    ratio = QtWidgets.QDoubleSpinBox()
    form.addRow(group(step, ratio))  # a row with no form label
    grid = QtWidgets.QGridLayout()
    grid.addWidget(QtWidgets.QLabel("<i>k</i><sub>xx</sub>"), 0, 0)
    kxx = QtWidgets.QDoubleSpinBox()
    grid.addWidget(kxx, 0, 1)
    form.addRow(grid)
    bare = QtWidgets.QDoubleSpinBox()
    form.addRow(bare)  # nothing beside it: left unnamed
    accessible.name_controls(panel)
    names = {box: box.accessibleName() for box in (own, width, at, ratio, kxx, bare)}
    assert names == {own: "", width: "Pulse", at: "Pulse, wide at", ratio: "Step at", kxx: "kxx", bare: ""}
    assert accessible.name(own) == "Masses:" and accessible.labelled(own), "named by Qt already: left alone"
    assert not accessible.labelled(bare), "Qt reads the group box's title for it: the panel's name, not its own"
    panel.close()


def test_every_pages_controls_have_names(main_window, app):
    w = main_window
    for i in range(w.pages.count()):
        w.pages.setCurrentIndex(i)  # builds each page, which names its controls
    app.processEvents()
    assert accessible.labelled(w.theme_combo) and accessible.labelled(w.size_combo), "the top bar's"
    unnamed = [
        f"{w.pages.tabText(i)}: {type(c).__name__} {c.toolTip()[:60]!r}"
        for i in range(w.pages.count())
        for c in accessible.controls(w.pages.widget(i))
        if not accessible.labelled(c)
    ]
    assert not unnamed, "\n".join(unnamed)
