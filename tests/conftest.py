"""Test-wide setup, run before any test module imports NumPy."""

import gc
import os
import re
import sys
import tempfile
import traceback
import weakref

import pytest

from vib_tutorial import ONE_THREAD

# BLAS on one thread, as the app runs it (vib_tutorial.main): the matrices are small, and
# OpenBLAS's pool makes their solves several times slower (worst on Windows CI).
for name in ONE_THREAD:
    os.environ.setdefault(name, "1")

# The app's settings (theme, playback speed, solved problems) in a directory of this process's own, not the user's:
# set in every process that imports this, so each xdist worker has its own (on Windows the app's native store is
# the registry, which all the workers shared, and QSettings.setPath doesn't reach it).
os.environ["VIB_SETTINGS"] = tempfile.mkdtemp(prefix="vib-settings-")

WARN_SHARE = 0.8  # warn when a worker's peak address space passes this share of `ulimit -v`


@pytest.fixture
def main_window(request):
    """A main window, shown, its frame timer stopped, for the tests that only build, read or set up pages: each
    page is built as the test asks for it. Freed after the test, as every GUI test's windows are (`free_windows`)."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    w._timer.stop()
    w.show()
    yield w
    w.close()
    del w
    request.node.funcargs.pop("main_window", None)  # pytest keeps it there until after `free_windows` checks


@pytest.fixture
def free_windows(request):
    """Tear down the windows a test built and check they're freed: the GUI modules' autouse fixtures use it.

    `close_graphics` clears their plots and blocks their signals, and once the test has dropped them `gc` frees
    them (their pages' own cycles need it). One still alive after that is held by something that lives on: a
    lambda or closure over `self` connected to a signal of the window's own (`gui/weak.py`). Not checked after
    a failed test, whose traceback still holds its window.

    It fails a test in which a slot raised, too: Qt only prints the exception, and `sys.last_exc` keeps it,
    with the frames that hold the window."""
    from PySide6 import QtCore, QtWidgets

    _forget_exception()
    before = set(QtWidgets.QApplication.topLevelWidgets())
    yield
    raised = _forget_exception()  # as text: the exception itself holds the frames, and so the window
    refs = _close_new(before)
    del before
    gc.collect()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    gc.collect()
    report = getattr(request.node, "rep_call", None)
    if report is not None and report.passed:
        assert raised is None, f"a slot raised during the test:\n{raised}"
        alive = [name for name, ref in refs if ref() is not None]
        assert not alive, f"windows still alive after the test: {alive} (see gui/weak.py)"


def _forget_exception() -> str | None:
    """Clear the exception a slot last raised (Qt prints it, and Python keeps it); its traceback, if one did."""
    exc = getattr(sys, "last_exc", None)
    text = "".join(traceback.format_exception(exc)) if exc is not None else None
    for name in ("last_exc", "last_type", "last_value", "last_traceback"):
        if hasattr(sys, name):
            delattr(sys, name)
    return text


def _close_new(before) -> list[tuple[str, weakref.ref]]:
    """`close_graphics` on each top-level widget not in `before`; weak references to the app's own (not
    pyqtgraph's menus, or a test's plain widgets). A function of its own, so no local holds one on."""
    from PySide6 import QtWidgets
    from shiboken6 import isValid

    from vib_tutorial.gui import close_graphics

    refs = []
    for w in QtWidgets.QApplication.topLevelWidgets():
        if w not in before and isValid(w):
            close_graphics(w)
            if type(w).__module__.startswith("vib_tutorial."):
                refs.append((type(w).__name__, weakref.ref(w)))
    return refs


# ------------------------------------------------------------------ core and GUI tests
# A test that builds a QApplication is marked `gui`, by the fixtures that make one or by hand: the CI runs
# `-m "not gui"` without Qt's system libraries and a display, and `-m gui` beside it (`.github/workflows/tests.yml`).
GUI_FIXTURES = {"app", "main_window", "free_windows"}
_RAN_GUI = False  # once a GUI test has run in this process, its QApplication stays


def pytest_collection_modifyitems(items):
    for item in items:
        if GUI_FIXTURES & set(getattr(item, "fixturenames", ())):
            item.add_marker(pytest.mark.gui)


def pytest_runtest_setup(item):
    global _RAN_GUI
    _RAN_GUI = _RAN_GUI or item.get_closest_marker("gui") is not None


def pytest_runtest_teardown(item):
    """Fail a test not marked `gui` that built a QApplication: it would fail in the CI's core job."""
    if _RAN_GUI or "PySide6.QtCore" not in sys.modules:
        return
    from PySide6 import QtCore

    if QtCore.QCoreApplication.instance() is not None:
        pytest.fail(f"{item.nodeid} built a QApplication: mark it `gui` or use a GUI fixture", pytrace=False)


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    report = yield
    setattr(item, f"rep_{report.when}", report)  # for free_windows: was the test itself passed?
    return report


# ------------------------------------------------------------------ the margin
# Each worker's peak address space, in the run's summary: a worker that reaches `ulimit -v` fails with a
# MemoryError or a segfault rather than a message, and windows that stop being freed would add up.


def _peak_mb() -> float | None:
    """This process's peak address space (VmPeak), MB; None where there's no /proc."""
    try:
        with open("/proc/self/status", encoding="ascii") as f:
            found = re.search(r"VmPeak:\s+(\d+) kB", f.read())
    except OSError:
        return None
    return int(found.group(1)) / 1024 if found else None


def _limit_mb() -> float | None:
    """`ulimit -v` in MB, or None (unlimited, or Windows: no `resource` module)."""
    try:
        import resource
    except ImportError:
        return None
    soft = resource.getrlimit(resource.RLIMIT_AS)[0]
    return None if soft == resource.RLIM_INFINITY else soft / 2**20


_PEAKS: dict[str, float] = {}


def pytest_sessionfinish(session):
    output = getattr(session.config, "workeroutput", None)
    if output is not None:  # a worker: its peak goes back to the controller
        output["peak_mb"] = _peak_mb()


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error):
    peak = getattr(node, "workeroutput", {}).get("peak_mb")
    if peak:
        _PEAKS[node.gateway.id] = peak


def pytest_terminal_summary(terminalreporter, config):
    if hasattr(config, "workerinput"):
        return
    peaks = dict(_PEAKS)
    if not peaks and (own := _peak_mb()):  # no workers: this process ran them all
        peaks["main"] = own
    if not peaks:
        return
    limit = _limit_mb()
    text = ", ".join(f"{name} {mb / 1024:.1f} GB" for name, mb in sorted(peaks.items()))
    terminalreporter.write_sep("-", "peak address space per worker")
    terminalreporter.write_line(text + (f" (ulimit -v {limit / 1024:.1f} GB)" if limit else ""))
    if limit:
        for name, mb in sorted(peaks.items()):
            if mb > WARN_SHARE * limit:
                terminalreporter.write_line(
                    f"WARNING: {name} reached {100 * mb / limit:.0f} % of ulimit -v: are the GUI tests' windows "
                    "still freed? (see CLAUDE.md)",
                    yellow=True,
                    bold=True,
                )
