"""`gui/transport.py`: the window's run bar drives the motion of the page and tab showing (#316 B13)."""

from __future__ import annotations


def _settle(w) -> None:
    from PySide6 import QtWidgets

    QtWidgets.QApplication.processEvents()
    w.run_bar.sync()


def test_the_bar_drives_a_playback_page(main_window):
    w = main_window
    w.show_page("sdof_page")
    _settle(w)
    bar, pb = w.run_bar, w.sdof_page.playback
    assert bar.source is not None and bar.source.run is pb.run
    assert bar.run.isEnabled() and bar.run.text() == "Start" and not pb.run.running
    bar.run.click()
    _settle(w)
    assert pb.run.running and bar.run.isChecked() and bar.run.text() == pb.run.text()
    bar.speed.setCurrentIndex(0)
    bar.speed.activated.emit(0)
    assert pb.speed.currentIndex() == 0
    bar.reset.click()
    _settle(w)
    assert not pb.run.running and bar.run.text() == "Start" and pb.t == 0


def test_the_bar_follows_the_page_and_says_when_nothing_runs(main_window):
    w = main_window
    _settle(w)
    sim = w.sim_page
    assert w.run_bar.source is not None and w.run_bar.source.run is sim.controls.run
    w.run_bar.run.click()  # the simulation's Pause
    _settle(w)
    assert w.run_bar.run.isChecked() == sim.controls.run.isChecked()
    w.show_page("signals_page")  # spectra of a signal: nothing plays
    _settle(w)
    assert w.run_bar.source is None and not w.run_bar.run.isEnabled() and "Nothing" in w.run_bar.what.text()
    w.show_page("sim_page")
    _settle(w)
    assert w.run_bar.source is not None and w.run_bar.source.run is sim.controls.run


def test_every_page_with_a_start_gives_the_bar_its_motion(main_window):
    """On every page and tab, a Start button showing means the bar drives it: pressing the bar's starts it."""
    from PySide6 import QtWidgets

    from vib_tutorial.gui import links
    from vib_tutorial.gui.transport import transport_of

    w = main_window
    missed, driven = [], set()
    for name in links.order():
        w.show_page(name)
        page = w.pages.currentWidget()
        tab_widgets = page.findChildren(QtWidgets.QTabWidget) or [None]
        for tabs in tab_widgets:
            for i in range(tabs.count() if tabs is not None else 1):
                if tabs is not None:
                    tabs.setCurrentIndex(i)
                _settle(w)
                starts = [
                    b
                    for b in page.findChildren(QtWidgets.QPushButton)
                    if b.isVisible() and b.text() in ("Start", "Pause", "Resume")
                ]
                t = transport_of(page)
                if starts and t is None:
                    missed.append((name, tabs.tabText(i) if tabs is not None else ""))
                if t is not None and name not in driven and t.run.isEnabled() and not t.run.isChecked():
                    w.run_bar.run.click()
                    _settle(w)
                    assert t.run.isChecked() and w.run_bar.run.isChecked(), name
                    w.run_bar.run.click()  # and pauses it again
                    _settle(w)
                    assert not t.run.isChecked(), name
                    driven.add(name)
    assert not missed, f"a Start the bar doesn't drive: {missed}"
    assert {"duffing_page", "rotor_page", "gyroscope_page", "torsional_page", "continuum_page", "sdof_page"} <= driven
    # what ran outside the bar (#409 U-16): the rainflow count, building H, the modal test's measurement
    assert {"fatigue_page", "frf_page", "test_page"} <= driven


def test_resuming_torsionals_start_up_shows_its_tab(main_window):
    """The bar's Resume ran the start-up on a hidden tab; Start already showed it (#409 U-16)."""
    w = main_window
    w.show_page("torsional_page")
    page = w.torsional_page
    _settle(w)
    w.run_bar.run.click()  # Start: its tab shows
    _settle(w)
    assert page.views.currentWidget() is page.startup_view and page.startup_view.running
    w.run_bar.run.click()  # Pause, then look at the criticals
    page.views.setCurrentIndex(0)
    _settle(w)
    w.run_bar.run.click()  # Resume: the start-up's tab again
    _settle(w)
    assert page.views.currentWidget() is page.startup_view and page.startup_view.running


def test_space_goes_to_a_control_reached_by_the_keyboard(main_window):
    """With a check box tabbed to, Space ticks it and the run stays put; clicked (a click keeps a button's focus
    on Linux and Windows), Space runs the page as before. The same for a page's own Start (the Simulation's force
    button), not just the bar's."""
    from PySide6 import QtCore, QtTest

    w = main_window
    w.activateWindow()
    w.show_page("sdof_page")
    _settle(w)
    page, pb = w.sdof_page, w.sdof_page.playback
    box = page.compare
    ticked = box.isChecked()
    box.setFocus(QtCore.Qt.FocusReason.TabFocusReason)
    _settle(w)
    QtTest.QTest.keyClick(w, QtCore.Qt.Key.Key_Space)
    _settle(w)
    assert box.isChecked() is not ticked and not pb.run.running
    box.clearFocus()
    QtTest.QTest.mouseClick(box, QtCore.Qt.MouseButton.LeftButton)  # clicked: focused by the mouse
    _settle(w)
    assert box.hasFocus() and box.isChecked() is ticked
    QtTest.QTest.keyClick(w, QtCore.Qt.Key.Key_Space)
    _settle(w)
    assert box.isChecked() is ticked and pb.run.running
    w.show_page("sim_page")
    _settle(w)
    force = w.sim_page.force_panel.force
    on = force.on
    w.sim_page.force_panel.kind.setFocus(QtCore.Qt.FocusReason.TabFocusReason)
    _settle(w)
    QtTest.QTest.keyClick(w, QtCore.Qt.Key.Key_Space)  # the combo box opens; the force stays as it was
    _settle(w)
    assert force.on is on
    QtTest.QTest.keyClick(w, QtCore.Qt.Key.Key_Escape)
    _settle(w)


def test_the_pages_own_controls_hide_and_space_runs_through_the_bar(main_window):
    """The bar stands in for the page's Start, Reset, speed and clock: they're hidden, and Space presses the bar's
    Start, but on the Simulation page, whose Space is its force."""
    from PySide6 import QtCore, QtTest

    w = main_window
    w.activateWindow()
    w.show_page("sdof_page")
    _settle(w)
    pb = w.sdof_page.playback
    assert all(x.isHidden() for x in (pb.run, pb.reset, pb.speed, pb.clock))
    assert w.run_bar.space.isEnabled()
    QtTest.QTest.keyClick(w, QtCore.Qt.Key.Key_Space)
    _settle(w)
    assert pb.run.running and w.run_bar.run.isChecked()
    w.show_page("sim_page")
    _settle(w)
    assert w.sim_page.controls.run.isHidden() and not w.run_bar.space.isEnabled()  # Space stays the force's


def test_the_modal_tests_measurement_runs_from_the_bar(main_window):
    """The test measures on its own, and the bar shows it: Pause holds it, Resume carries on, and once it's done
    Start (or Reset) measures it again (#409 U-16: it ran while the bar said nothing did)."""
    import time

    w = main_window
    w.show_page("test_page")
    page = w.test_page
    page.speed.setCurrentIndex(1)  # paced, so it takes a while
    page.restart()
    _settle(w)
    assert w.run_bar.source is not None and w.run_bar.source.run is page.measure
    assert page.measuring and page.measure.isChecked() and w.run_bar.run.isChecked()
    w.run_bar.run.click()  # Pause
    _settle(w)
    assert page.measuring and not page.running and not page._timer.isActive() and page.measure.text() == "Resume"
    w.run_bar.run.click()  # Resume
    _settle(w)
    assert page.running and page._timer.isActive()
    page.speed.setCurrentIndex(page.speed.findText("Instant"))  # finish it now
    end = time.perf_counter() + 30.0
    while page.measuring and time.perf_counter() < end:
        _settle(w)
    _settle(w)
    assert not page.measuring and page.measure.text() == "Start" and not w.run_bar.run.isChecked()
    done = page.acq
    w.run_bar.reset.click()  # Measure again
    assert page.acq is not done


def test_the_bar_says_when_the_page_is_updating(main_window):
    """A page's slow work (a `Recompute`) busy past `BUSY_MS`: the bar says "Updating…", on a page with nothing to
    run too, and not on another page (#409 U-24)."""
    import time

    from vib_tutorial.gui import recompute

    w = main_window
    w.show_page("random_page")
    page = w.random_page
    _settle(w)
    assert w.run_bar.source is None and not w.run_bar.updating.isVisible()
    job = recompute.Recompute(lambda _result: None, page, delay_ms=0)
    job.request(lambda: time.sleep(0.4))
    time.sleep(recompute.BUSY_MS / 1000 + 0.02)
    _settle(w)
    assert w.run_bar.updating.isVisible()
    w.show_page("sim_page")
    _settle(w)
    assert not w.run_bar.updating.isVisible()  # another page's work
    w.show_page("random_page")
    job.wait()
    _settle(w)
    assert not w.run_bar.updating.isVisible()
