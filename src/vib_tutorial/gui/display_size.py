"""The app's size: everything larger for a projector or a TV across the room (the presenter mode, #218 U4).

*Size* (top right, beside *Theme*) scales the whole app, its text, lines, markers, plots and figures alike,
through Qt's own scale factor (`QT_SCALE_FACTOR`), as a high-DPI screen does: nothing is laid out again or
drawn coarser, so every page stays as it is, only larger, and a 1 px curve becomes 1.5 px at 150 %. Qt reads
the factor once, when the QApplication is created, so `apply_saved` sets it from the settings before
`make_app` creates it, and a new size needs a restart (`restart`: the app launched again, which opens on the
page last shown). A factor set in the environment already wins, and isn't saved.

At 150 % the window needs 1.5 times the screen it needs at 100 %: `fits` compares the window's minimum size at
the new scale with its screen's, so a size the screen can't hold is refused when picked, and a saved size that
the screen it opens on can't hold (a projector unplugged) goes back to 100 % (`check_fits`).

Working on it: `tests/test_display_size.py` checks the setting, the environment and the fit.
"""

from __future__ import annotations

import os
import sys

from PySide6 import QtCore, QtWidgets

from . import theming

SIZES = (1.0, 1.25, 1.5, 2.0)
KEY = "display/scale"
ENV = "QT_SCALE_FACTOR"

_active = 1.0  # the scale this process's QApplication was created at
_from_env = False  # set in the environment, not by the settings: the Size box can't change it


def label(scale: float) -> str:
    return f"{round(scale * 100)} %"


def saved() -> float:
    try:
        value = float(str(theming.settings().value(KEY, 1.0)))
    except (TypeError, ValueError):
        return 1.0
    return value if value in SIZES else 1.0


def save(scale: float) -> None:
    theming.settings().setValue(KEY, scale)


def apply_saved() -> None:
    """Before the QApplication exists: Qt's scale factor from the settings (unless the environment sets one)."""
    global _active, _from_env
    _from_env = ENV in os.environ
    if _from_env:
        try:
            _active = float(os.environ[ENV])
        except ValueError:
            _active = 1.0
        return
    _active = saved()
    if _active != 1.0:
        os.environ[ENV] = f"{_active:g}"


def active() -> float:
    """The scale the app is drawn at now (a new one shows after a restart)."""
    return _active


def from_environment() -> bool:
    """Whether the environment set the scale (`QT_SCALE_FACTOR`), which the Size box then leaves alone."""
    return _from_env


def _smallest(window: QtWidgets.QWidget) -> QtCore.QSize:
    """The smallest the window can be: its layout's minimum, or the minimum set on it."""
    return window.minimumSizeHint().expandedTo(window.minimumSize())


def fits(window: QtWidgets.QWidget, scale: float) -> bool:
    """Whether `window`, at its minimum size, fits its screen at `scale`."""
    screen = window.screen()
    if screen is None:
        return True
    space = screen.availableGeometry().size() * _active  # the screen in pixels at 100 %
    need = _smallest(window) * scale
    return need.width() <= space.width() and need.height() <= space.height()


def screen_width_needed(window: QtWidgets.QWidget, scale: float) -> int:
    return round(_smallest(window).width() * scale)


def restart() -> None:
    """Launch the app again (its new process reads the size saved) and quit this one."""
    if getattr(sys, "frozen", False):  # a packaged app: its own executable
        program, args = sys.executable, sys.argv[1:]
    else:
        program, args = sys.executable, ["-m", "vib_tutorial", *sys.argv[1:]]
    environment = QtCore.QProcessEnvironment.systemEnvironment()
    environment.remove(ENV)  # set from the settings, by apply_saved: the new process reads the size saved
    process = QtCore.QProcess()
    process.setProgram(program)
    process.setArguments(args)
    process.setProcessEnvironment(environment)
    process.startDetached()
    theming.application().closeAllWindows()
    theming.application().quit()


def check_fits(window: QtWidgets.QWidget) -> bool:
    """At launch: if the saved size doesn't fit this screen, say so, save 100 % and restart. False if it restarts."""
    if _active == 1.0 or _from_env or fits(window, _active):
        return True
    QtWidgets.QMessageBox.information(
        window,
        "Size",
        f"At {label(_active)} the window is larger than this screen, so the app goes back to 100 %.",
    )
    save(1.0)
    restart()
    return False
