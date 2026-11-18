"""`gui/errors.py`: an error nothing caught is logged and shown on the window, with its details to copy."""

from __future__ import annotations

import sys
import threading

import pytest


def _fail() -> None:
    raise ValueError("a slot's mistake\nsecond line")


def _forget_exception() -> None:
    """Raised on purpose: what Python keeps of it would fail the test (conftest's `free_windows`)."""
    for name in ("last_exc", "last_type", "last_value", "last_traceback"):
        if hasattr(sys, name):
            delattr(sys, name)


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")  # the thread's, on purpose
def test_a_slots_error_is_logged_and_shown(main_window, tmp_path, monkeypatch):
    from PySide6 import QtWidgets

    from vib_tutorial.gui import errors, theming

    monkeypatch.setenv(theming.SETTINGS_DIR, str(tmp_path))
    banner = main_window.error_banner
    assert banner.isHidden()
    uninstall = errors.install(banner)
    try:
        button = QtWidgets.QPushButton()
        button.clicked.connect(_fail)
        button.click()  # PySide hands a slot's exception to sys.excepthook
        QtWidgets.QApplication.processEvents()
        worker = threading.Thread(target=_fail)
        worker.start()
        worker.join()
        QtWidgets.QApplication.processEvents()  # a thread's error is shown from the GUI thread
    finally:
        uninstall()
        _forget_exception()
    assert sys.excepthook is sys.__excepthook__ or "errors" not in sys.excepthook.__module__
    log = (tmp_path / errors.LOG_NAME).read_text(encoding="utf-8")
    assert log.count("ValueError: a slot's mistake") == 2 and "Vibration Tutorial" in log
    assert not banner.isHidden() and "ValueError: a slot's mistake" in banner.label.text()
    assert "second line" not in banner.label.text() and "2 errors so far" in banner.label.text()
    banner.copy.click()
    copied = QtWidgets.QApplication.clipboard().text()
    assert "Traceback" in copied and "_fail" in copied and str(tmp_path) in copied
    banner.close_button.click()
    assert banner.isHidden() and banner.count == 0


def test_the_log_is_kept_under_its_limit(tmp_path, monkeypatch):
    from vib_tutorial.gui import errors

    monkeypatch.setattr(errors, "LOG_LIMIT", 1000)
    for k in range(60):
        errors.write_log(f"error {k:03d} " + "x" * 40, tmp_path)
    text = (tmp_path / errors.LOG_NAME).read_text(encoding="utf-8")
    assert len(text) < 1200 and "error 059" in text and "error 000" not in text
