"""`gui/display_size.py`: the app's size (presenter mode), its setting, Qt's scale factor and the fit."""

from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from vib_tutorial.gui import display_size  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def _free_windows(app, free_windows):
    """Each test's windows torn down and freed after it (conftest's `free_windows`)."""


@pytest.fixture
def scale(monkeypatch, tmp_path):
    """Settings of the test's own, and the module's state put back after it."""
    monkeypatch.setenv("VIB_SETTINGS", str(tmp_path))
    monkeypatch.setattr(display_size, "_active", 1.0)
    monkeypatch.setattr(display_size, "_from_env", False)
    monkeypatch.delenv(display_size.ENV, raising=False)
    return display_size


def test_the_size_saved_is_qts_scale_factor_at_launch(scale, monkeypatch):
    assert scale.saved() == 1.0
    scale.apply_saved()
    assert display_size.ENV not in os.environ and scale.active() == 1.0, "100 %: Qt's own"
    scale.save(1.5)
    assert scale.saved() == 1.5
    scale.apply_saved()
    assert os.environ[display_size.ENV] == "1.5" and scale.active() == 1.5 and not scale.from_environment()
    scale.save(1.7)
    assert scale.saved() == 1.0, "a size not offered"


def test_a_scale_set_in_the_environment_wins(scale, monkeypatch):
    scale.save(1.5)
    monkeypatch.setenv(display_size.ENV, "2")
    scale.apply_saved()
    assert scale.active() == 2.0 and scale.from_environment()


def test_a_size_fits_if_the_window_does_at_it(scale, app):
    widget = QtWidgets.QWidget()
    space = widget.screen().availableGeometry().size()
    widget.setMinimumSize(space.width() // 2 + 10, space.height() // 4)
    assert scale.fits(widget, 1.0) and scale.fits(widget, 1.5) and not scale.fits(widget, 2.0)
    scale._active = 2.0  # drawn at 200 % already: the screen is twice what Qt reports
    assert scale.fits(widget, 2.0)
    assert scale.screen_width_needed(widget, 2.0) == 2 * widget.minimumWidth()


class FakeProcess:
    started: list[tuple[str, list[str], object]] = []

    def setProgram(self, program):
        self.program = program

    def setArguments(self, args):
        self.args = args

    def setProcessEnvironment(self, environment):
        self.environment = environment

    def startDetached(self):
        FakeProcess.started.append((self.program, self.args, self.environment))
        return True


def test_a_restart_launches_the_app_again_without_this_scale(scale, app, monkeypatch):
    FakeProcess.started = []
    monkeypatch.setattr(display_size.QtCore, "QProcess", FakeProcess)
    monkeypatch.setattr(app, "quit", lambda: None)
    monkeypatch.setattr(sys, "argv", ["vib-tutorial", "--flag"])
    monkeypatch.setenv(display_size.ENV, "1.5")
    scale.restart()
    [(program, args, environment)] = FakeProcess.started
    assert program == sys.executable and args == ["-m", "vib_tutorial", "--flag"]
    assert not environment.contains(display_size.ENV), "the new process reads the size saved"


def test_a_saved_size_too_large_for_the_screen_goes_back_to_100(scale, app, monkeypatch):
    restarts = []
    monkeypatch.setattr(scale, "restart", lambda: restarts.append(True))
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *args: None)
    widget = QtWidgets.QWidget()
    widget.setMinimumSize(widget.screen().availableGeometry().size())
    assert scale.check_fits(widget) and not restarts, "100 %"
    scale.save(1.5)
    scale._active = 1.5
    widget.setMinimumSize(widget.screen().availableGeometry().size() * 1.1)  # drawn at 150 %: too large for it
    assert scale.check_fits(widget) is False and restarts == [True] and scale.saved() == 1.0


def test_the_size_box(scale, app, monkeypatch):
    from vib_tutorial.gui.main_window import MainWindow

    restarts = []
    monkeypatch.setattr(scale, "restart", lambda: restarts.append(True))
    told = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *args: told.append(args[2]))
    answer = QtWidgets.QMessageBox.StandardButton.No
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *args: answer)
    w = MainWindow()
    assert w.size_combo.currentText() == "100 %" and w.size_combo.isEnabled()
    monkeypatch.setattr(scale, "fits", lambda window, s: s <= 1.5)
    w.size_combo.setCurrentIndex(3)
    w.size_combo.activated.emit(3)
    assert "200 %" in told[0] and w.size_combo.currentText() == "100 %" and scale.saved() == 1.0, "too large"
    w.size_combo.setCurrentIndex(2)
    w.size_combo.activated.emit(2)
    assert scale.saved() == 1.5 and not restarts, "saved, for the next launch"
    answer = QtWidgets.QMessageBox.StandardButton.Yes
    w.size_combo.activated.emit(1)
    assert scale.saved() == 1.25 and restarts == [True]
    w.close()
