"""`gui/accessible.py`: every control has a name a screen reader can read (#218 U5)."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from vib_tutorial.gui import accessible  # noqa: E402
from vib_tutorial.gui.layouts import form_layout, form_row, group  # noqa: E402


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
    form_row(form, "Masses:", own, "")  # Qt's own: the label's buddy
    width, at = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()
    form_row(form, "Pulse:", group(width, "wide at", at), "")  # a row of widgets as the field
    step = QtWidgets.QCheckBox("Step at")
    ratio = QtWidgets.QDoubleSpinBox()
    form.addRow(group(step, ratio))  # a check box before it
    bare = QtWidgets.QDoubleSpinBox()
    form.addRow(bare)  # nothing beside it: left unnamed
    names = {box: box.accessibleName() for box in (own, width, at, ratio, bare)}
    assert names == {own: "", width: "Pulse", at: "Pulse, wide at", ratio: "Step at", bare: ""}
    assert accessible.name(own) == "Masses:" and accessible.labelled(own), "named by Qt already: left alone"
    assert not accessible.labelled(bare), "Qt reads the group box's title for it: the panel's name, not its own"
    panel.close()


# Controls not named yet, by page: their rows are built by hand (a sub-layout in a form row, a grid), outside the
# helpers that name them. Each count may only go down: name a page's controls and drop its entry.
GAPS = {
    "rotor_page": 63,
    "sim_page": 26,
    "updating_page": 12,
    "test_page": 11,
    "duffing_page": 11,
    "oma_page": 10,
    "random_page": 8,
    "shock_page": 5,
    "coupling_page": 5,
    "complex_page": 2,
    "frf_page": 1,
    "frame_page": 1,
    "signals_page": 1,
}


def test_every_pages_controls_have_names(main_window, app):
    w = main_window
    for i in range(w.pages.count()):
        w.pages.setCurrentIndex(i)  # builds each page
    app.processEvents()
    assert accessible.labelled(w.theme_combo) and accessible.labelled(w.size_combo), "the top bar's"
    over = []
    for i in range(w.pages.count()):
        page = w.pages.widget(i)
        name = w.page_name(page)
        unnamed = [c for c in accessible.controls(page) if not accessible.labelled(c)]
        if len(unnamed) > GAPS.get(name, 0):
            tips = "; ".join(f"{type(c).__name__} {c.toolTip()[:50]!r}" for c in unnamed[:5])
            over.append(f"{name}: {len(unnamed)} unnamed, {GAPS.get(name, 0)} allowed ({tips})")
        elif len(unnamed) < GAPS.get(name, 0):
            over.append(f"{name}: only {len(unnamed)} unnamed now: lower its GAPS entry")
    assert not over, "\n".join(over)
