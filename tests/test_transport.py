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
