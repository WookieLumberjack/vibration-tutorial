"""`gui/present.py`: F5 folds the pages' side panes away for presenting, and the window's minimum with them."""

from __future__ import annotations

import pytest


@pytest.fixture
def window(main_window):
    """The main window, put back at all panes after the test (the level is kept in the worker's settings)."""
    yield main_window
    main_window.set_present(0)


def _panes(page):
    from vib_tutorial.gui import present

    s = present.splitter(page)
    assert s is not None
    return [s.widget(i) for i in range(s.count())]


def test_f5_steps_every_page_through_the_levels_and_back(window):
    from PySide6 import QtWidgets

    from vib_tutorial.gui import present

    w = window
    w.show()
    pages = (w.sim_page, w.sdof_page, w.signals_page, w.rotor_page)  # the Simulation page is its own splitter
    QtWidgets.QApplication.processEvents()
    wide = w.minimumSizeHint().width()
    for level in (1, 2, 0):
        w.step_present()
        assert w.present_level == level and present.saved() == level
        for page in pages:
            panes = _panes(page)
            assert panes[0].isHidden() == (level >= 1)
            assert panes[-1].isHidden() == (level == 2 and len(panes) >= 3)
            assert all(not p.isHidden() for p in panes[1:-1])  # the centre always shows
        if level == 2:
            w.pages.setCurrentWidget(w.sdof_page)
            QtWidgets.QApplication.processEvents()
            assert w.minimumSizeHint().width() < 0.75 * wide  # the plots alone: a larger Size fits
    assert w.minimumSizeHint().width() == wide

    w.set_present(1)
    page = w.duffing_page  # built now: it takes the level
    assert _panes(page)[0].isHidden() and not _panes(page)[-1].isHidden()


def test_every_page_has_panes_to_fold(window):
    """Each page's outermost splitter is found, so F5 reaches it."""
    from vib_tutorial.gui import present

    w = window
    for i in range(w.pages.count()):
        w.pages.setCurrentIndex(i)  # builds it
        page = w.pages.widget(i)
        assert page is not None and present.splitter(page) is not None, w.pages.tabText(i)
