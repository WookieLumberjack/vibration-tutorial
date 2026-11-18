"""What a page shows as it opens, and where a control's change shows (#409 U-21, I-5)."""

import pytest
from PySide6 import QtWidgets

pytestmark = pytest.mark.gui


def test_no_page_opens_with_no_sensors(main_window):
    """Nonlinearity & forces built its sensor boxes unticked while its state said m1: it opened on "No sensors" and
    an empty force panel, and Defaults put that back."""
    from vib_tutorial.gui import links, theming

    app = theming.application()
    w = main_window
    for attr in links.order():
        page = getattr(w, attr)
        w.pages.setCurrentWidget(page)
        tabs = page.findChild(QtWidgets.QTabWidget)
        for k in range(tabs.count() if tabs is not None else 1):
            if tabs is not None:
                tabs.setCurrentIndex(k)
            app.processEvents()
            said = [lab.text() for lab in page.findChildren(QtWidgets.QLabel) if "No sensors" in lab.text()]
            assert not said, f"{attr} opens saying {said[0]!r}"
    p = w.nonlinear_page
    assert p._sensors() == (0,) and p.practice_state().sensors == (0,)
    w.pages.setCurrentWidget(p)
    w.restore_defaults()
    assert p._sensors() == (0,)


def test_a_ring_control_turns_to_the_tab_it_drives(main_window, monkeypatch):
    """Variability's ring controls changed nothing on its opening tab, Frequencies: each turns to the tab it drives,
    and draws it once."""
    from vib_tutorial.gui import variability as v

    w = main_window
    p = w.variability_page
    w.pages.setCurrentWidget(p)
    drawn = []
    draw = v.VariabilityPage._draw_tab
    monkeypatch.setattr(
        v.VariabilityPage, "_draw_tab", lambda self: (drawn.append(self.tabs.currentIndex()), draw(self))
    )
    assert p.tabs.currentIndex() == v.TAB_HISTOGRAM
    p.coupling.setValue(10.0)
    assert p.tabs.currentIndex() == v.TAB_MODES and drawn == [v.TAB_MODES]
    p.eo.setValue(2)
    assert p.tabs.currentIndex() == v.TAB_FORCED and drawn[-1] == v.TAB_FORCED and len(drawn) == 2
    p.blades.setValue(10)  # shows on the forced ring too: it stays
    assert p.tabs.currentIndex() == v.TAB_FORCED and len(drawn) == 3
    p.mode.setValue(3)
    assert p.tabs.currentIndex() == v.TAB_MODES and len(drawn) == 4
    p.blades.setValue(2)  # the mode clamps to 2 quietly, drawn with the rest
    assert p.mode.value() == 2 and p.tabs.currentIndex() == v.TAB_MODES and len(drawn) == 5
    p.tabs.setCurrentIndex(v.TAB_HISTOGRAM)
    p.cov_k.setValue(8.0)  # a scatter control: its tab is this one
    assert p.tabs.currentIndex() == v.TAB_HISTOGRAM
