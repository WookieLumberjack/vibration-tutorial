"""What the app does with an error nothing caught: log it, and say so on the window (#409 U-23).

The release builds have no console (`console=False` in the spec), so an exception a slot raised went nowhere:
the page stayed half-updated and nobody knew why. `install(banner)`, from `gui.run()` only, sets `sys.excepthook`
(PySide passes a slot's exception there) and `threading.excepthook`: each error's traceback, with the version
and platform, is appended to `errors.log` in `log_folder()` (the settings' folder: $VIB_SETTINGS, or the
platform's app-data folder; kept under `LOG_LIMIT`), still printed to stderr as before, and shown in the main
window's `ErrorBanner` along its top: what went wrong in a line, *Copy details* (the traceback, for a bug
report) and a close button. A worker thread's error is logged there and shown from the GUI thread.

The tests don't install it: conftest fails a test in which a slot raised, reading what Python keeps
(`sys.last_exc`), and the hook leaves that as it was.

Working on it: `tests/test_errors.py` raises in a slot with the hook installed and checks the log, the banner
and the copy.
"""

from __future__ import annotations

import os
import platform
import sys
import threading
import time
import traceback
import weakref
from collections.abc import Callable
from pathlib import Path
from types import TracebackType

from PySide6 import QtCore, QtGui, QtWidgets
from shiboken6 import isValid

from . import theming
from .layouts import wrapped

LOG_NAME = "errors.log"
LOG_LIMIT = 1_000_000  # bytes: past it the oldest half goes


def log_folder() -> Path:
    """Where the log goes: the tests' and scripts' $VIB_SETTINGS, else the platform's app-data folder (the settings
    themselves are in the registry on Windows, which has no folder)."""
    folder = os.environ.get(theming.SETTINGS_DIR)
    if folder:
        return Path(folder)
    where = QtCore.QStandardPaths.StandardLocation.AppLocalDataLocation
    return Path(QtCore.QStandardPaths.writableLocation(where) or Path.home() / ".vib-tutorial")


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("vib-tutorial")
    except PackageNotFoundError:
        return "unknown"


def report(kind: type[BaseException], error: BaseException, tb: TracebackType | None) -> str:
    """The error as the log and *Copy details* give it: when, the version and platform, and the traceback."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    where = f"Vibration Tutorial {_version()}, Python {platform.python_version()}, {platform.platform()}"
    return f"--- {stamp}: {where}\n" + "".join(traceback.format_exception(kind, error, tb))


def write_log(text: str, folder: Path | None = None) -> Path | None:
    """Append `text` to the log, trimming it first if it's grown past LOG_LIMIT; its path (None if it can't be
    written: the error is still shown)."""
    try:
        folder = folder or log_folder()
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / LOG_NAME
        if path.exists() and path.stat().st_size > LOG_LIMIT:
            old = path.read_text(encoding="utf-8", errors="replace")
            path.write_text(old[len(old) // 2 :], encoding="utf-8")
        with path.open("a", encoding="utf-8") as f:
            f.write(text + "\n")
        return path
    except OSError:
        return None


class ErrorBanner(QtWidgets.QFrame):
    """A line along the window's top saying something went wrong, with *Copy details* and a close button. Hidden
    until `show_error`; a second error before it's closed replaces the first and counts them."""

    raised = QtCore.Signal(str, str)  # the line, the details: from any thread, shown on the GUI thread

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setAutoFillBackground(True)
        self.setBackgroundRole(QtGui.QPalette.ColorRole.ToolTipBase)
        self.setForegroundRole(QtGui.QPalette.ColorRole.ToolTipText)
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(8, 3, 4, 3)
        self.label = wrapped()
        self.label.setForegroundRole(QtGui.QPalette.ColorRole.ToolTipText)
        self.copy = QtWidgets.QToolButton()
        self.copy.setText("Copy details")
        self.copy.setToolTip("Copy the error's details (for a bug report) to the clipboard.")
        self.copy.clicked.connect(self._copy)
        self.close_button = QtWidgets.QToolButton()
        self.close_button.setText("✕")
        self.close_button.setAccessibleName("Close the error message")
        self.close_button.clicked.connect(self._dismiss)
        row.addWidget(self.label, 1)
        row.addWidget(self.copy)
        row.addWidget(self.close_button)
        self.details = ""
        self.count = 0
        self.raised.connect(self.show_error, QtCore.Qt.ConnectionType.QueuedConnection)
        self.hide()

    def show_error(self, line: str, details: str) -> None:
        self.count += 1
        self.details = details
        more = f" ({self.count} errors so far; the latest shown.)" if self.count > 1 else ""
        self.label.setText(
            f"<b>Something went wrong:</b> {line}. The page may be half-updated: Defaults (Ctrl+R) usually puts it "
            f"right. The details are in the log, or copy them for a bug report.{more}"
        )
        self.label.setToolTip(details)
        self.show()

    def _copy(self) -> None:
        theming.application().clipboard().setText(self.details)

    def _dismiss(self) -> None:
        self.count = 0
        self.hide()


def _line(kind: type[BaseException], error: BaseException) -> str:
    text = str(error).splitlines()[0] if str(error) else ""
    line = f"{kind.__name__}: {text}" if text else kind.__name__
    return line if len(line) <= 160 else line[:157] + "…"


_handling = threading.local()


def install(banner: ErrorBanner) -> Callable[[], None]:
    """Log and show every uncaught error from now on (`gui.run()`); the previous hooks still run first. Holds the
    banner weakly (once its window is gone, errors are only logged); returns what puts the old hooks back."""
    previous, previous_thread = sys.excepthook, threading.excepthook
    ref = weakref.ref(banner)

    def handle(kind: type[BaseException], error: BaseException, tb: TracebackType | None) -> None:
        if getattr(_handling, "busy", False):
            return  # an error while handling one: the first is enough
        _handling.busy = True
        try:
            details = report(kind, error, tb)
            path = write_log(details)
            if path is not None:
                details += f"\n(logged in {path})"
            shown = ref()
            if shown is not None and isValid(shown):
                shown.raised.emit(_line(kind, error), details)
        except Exception:
            pass  # never raise from the hook: Python would print that and lose the first
        finally:
            _handling.busy = False

    def hook(kind: type[BaseException], error: BaseException, tb: TracebackType | None) -> None:
        previous(kind, error, tb)  # to stderr, as before
        if not issubclass(kind, KeyboardInterrupt):
            handle(kind, error, tb)

    def thread_hook(args: threading.ExceptHookArgs) -> None:
        previous_thread(args)
        if args.exc_value is not None and not issubclass(args.exc_type, SystemExit):
            handle(args.exc_type, args.exc_value, args.exc_traceback)

    def uninstall() -> None:
        sys.excepthook, threading.excepthook = previous, previous_thread

    sys.excepthook = hook
    threading.excepthook = thread_hook
    return uninstall
