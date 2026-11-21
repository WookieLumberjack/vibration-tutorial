"""`gui/accessible.py`: every control has a name a screen reader can read (#218 U5)."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from vib_tutorial.gui import accessible  # noqa: E402
from vib_tutorial.gui.layouts import form_layout, form_row, group  # noqa: E402


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


def test_every_pages_controls_and_plots_have_names(main_window, app):
    import pyqtgraph as pg

    w = main_window
    for i in range(w.pages.count()):
        w.pages.setCurrentIndex(i)  # builds each page
    app.processEvents()
    assert accessible.labelled(w.theme_combo) and accessible.labelled(w.size_combo), "the top bar's"
    missing = []
    for i in range(w.pages.count()):
        page = w.pages.widget(i)
        name = w.page_name(page)
        for c in accessible.controls(page):
            if not accessible.labelled(c):
                missing.append(f"{name}: {type(c).__name__} {c.toolTip()[:50]!r}")
        for view in page.findChildren(pg.GraphicsView):
            if not view.accessibleName():
                missing.append(f"{name}: a {type(view).__name__} with no title or axes: name it where it's made")
    assert not missing, "\n".join(missing)


def test_a_plot_is_named_after_its_title_or_axes(app):
    from vib_tutorial.gui.lesson import plot

    p = plot("Amplitude", "m", "Frequency", "Hz")
    titled = plot("x", "m", "t", "s")
    titled.setTitle("The <b>response</b>")
    named = plot("x", "m", "t", "s")
    named.setAccessibleName("Its own")
    box = QtWidgets.QWidget()
    for view in (p, titled, named):
        view.setParent(box)
    accessible.name_plots(box)
    assert [v.accessibleName() for v in (p, titled, named)] == [
        "Amplitude against Frequency",
        "The response",
        "Its own",
    ]
    box.close()
