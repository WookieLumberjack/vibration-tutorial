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
