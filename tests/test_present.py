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
    from vib_tutorial.gui.main_window import PAGE_GROUPS

    w = window
    for _, pages in PAGE_GROUPS:
        for name, _ in pages:
            page = getattr(w, name)  # built now
            assert present.splitter(page) is not None, name


def test_ink_draws_on_the_page_the_pointer_follows_and_esc_clears_them(window):
    from PySide6 import QtCore, QtGui, QtTest, QtWidgets

    from vib_tutorial.gui.ink import Mode

    w = window
    w.show()
    QtWidgets.QApplication.processEvents()
    ink = w.ink
    assert not ink.isVisible() and not w.ink_escape.isEnabled()

    w.toggle_ink()  # F7: the overlay takes the mouse
    assert ink.mode is Mode.INK and ink.isVisible() and w.ink_escape.isEnabled()
    assert ink.parentWidget() is w.pages and ink.geometry() == w.pages.rect()
    assert not ink.testAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    QtTest.QTest.mousePress(ink, QtCore.Qt.MouseButton.LeftButton, pos=QtCore.QPoint(100, 100))
    move = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseMove,
        QtCore.QPointF(160, 130),
        QtCore.QPointF(ink.mapToGlobal(QtCore.QPoint(160, 130))),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    QtWidgets.QApplication.sendEvent(ink, move)
    QtTest.QTest.mouseRelease(ink, QtCore.Qt.MouseButton.LeftButton, pos=QtCore.QPoint(160, 130))
    assert len(ink.strokes) == 1 and ink.strokes[0].size() == 2
    ink.grab()  # paints

    w.toggle_ink()  # F7 again: the page has the mouse back, the ink stays
    assert ink.mode is Mode.OFF and ink.isVisible() and len(ink.strokes) == 1
    assert ink.testAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    w.pages.setCurrentWidget(w.sdof_page)  # another page: its ink goes
    assert not ink.strokes and not ink.isVisible() and not w.ink_escape.isEnabled()

    w.toggle_pointer()  # F6
    assert ink.mode is Mode.POINTER and ink.isVisible()
    assert ink.timer.isActive()
    QtGui.QCursor.setPos(ink.mapToGlobal(QtCore.QPoint(50, 60)))
    ink._tick()
    ink.grab()
    w.ink_escape.activated.emit()  # Esc
    assert ink.mode is Mode.OFF and ink.pointer is None and not ink.isVisible() and not ink.timer.isActive()
