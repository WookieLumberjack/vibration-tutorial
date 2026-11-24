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


def test_stepping_trials_while_presenting_shows_the_note_large(window):
    """B15: with the controls folded away, Alt+PgDown shows the step's note over the page; Esc hides it."""
    from PySide6 import QtWidgets

    from vib_tutorial.gui import sim_trials

    w = window
    w.show()
    overlay = w.trial_overlay
    w.next_trial()  # all panes: the box's own note shows, not the overlay
    assert not overlay.isVisible()
    w.set_present(1)
    w.next_trial()
    QtWidgets.QApplication.processEvents()
    first, second = sim_trials.TRIALS[:2]
    assert overlay.isVisible() and f"2 / {len(sim_trials.TRIALS)} · {second.name}" == overlay.title.text()
    box = w.sim_page.trials  # what its box shows: the question, if it asks one first
    assert overlay.body.text() == (box.question if box.asking is not None else box.note).text() != ""
    assert overlay.geometry().right() <= w.pages.width() and overlay.height() > 0
    assert w.ink_escape.isEnabled()
    w.previous_trial()
    assert first.name in overlay.title.text()
    w.clear_ink()  # Esc
    assert not overlay.isVisible() and not w.ink_escape.isEnabled()
    w.next_trial()
    w.set_present(0)  # back to all panes: the box shows its note again
    assert not overlay.isVisible()


def test_a_prediction_is_answered_from_the_large_note(window):
    """A step that asks a prediction shows its question and choices; a choice answers it and the note follows."""
    from PySide6 import QtWidgets

    from vib_tutorial.gui import transient

    w = window
    w.show()
    w.show_page("transient_page")
    w.set_present(1)
    w.next_trial()  # the first trial asks first
    overlay, box = w.trial_overlay, w.transient_page.trials
    assert box.asking == 0 and overlay.isVisible()
    assert overlay.body.text() == box.question.text()
    buttons = [overlay.choices.itemAt(k).widget() for k in range(overlay.choices.count())]
    p = transient.TRIALS[0].predict
    order = p.shown_order()  # as the box shows them
    assert [b.text() for b in buttons] == [*(p.choices[k] for k in order), "Just run it"]
    buttons[order.index(p.answer)].click()
    QtWidgets.QApplication.processEvents()
    assert box.asking is None and "Right" in overlay.body.text() and box.note.text() in overlay.body.text()
    assert overlay.choices.count() == 0
    w.show_page("sim_page")  # a new page hides it
    assert not overlay.isVisible()


def test_presenting_shows_a_way_back_in_the_corner(window):
    """The level outlives a launch (a new Size needs one), so while it's on the corner bar says so, and its button
    shows every pane again; at all panes it takes no room."""
    from vib_tutorial.gui import present

    w = window
    w.show()
    assert w.present_button.isHidden()
    w.set_present(2)
    assert not w.present_button.isHidden() and "centre only" in w.present_button.toolTip()
    w.present_button.click()
    assert w.present_level == 0 and present.saved() == 0 and w.present_button.isHidden()
    assert all(not p.isHidden() for p in _panes(w.sim_page))
