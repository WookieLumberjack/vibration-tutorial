"""Test-wide setup, run before any test module imports NumPy."""

import os
import re

import pytest

from vib_tutorial import ONE_THREAD

# BLAS on one thread, as the app runs it (vib_tutorial.main): the matrices are small, and
# OpenBLAS's pool makes their solves several times slower (worst on Windows CI).
for name in ONE_THREAD:
    os.environ.setdefault(name, "1")

# Two malloc arenas, not glibc's eight per core: the xdist workers inherit it, and their arenas' reserved address
# space (not memory used) put the worker running the GUI smoke tests over an 8 GB `ulimit -v`.
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

WARN_SHARE = 0.8  # warn when a worker's peak address space passes this share of `ulimit -v`


@pytest.fixture(scope="session")
def shared_window():
    """One main window, its frame timer stopped, for the tests that only build, read or set up pages; each page is
    built as a test asks for it. No window a GUI test builds is ever freed (PySide's signal connections hold them),
    and each test that built every page in a window of its own kept another 300–600 MB. The tests that use it are
    marked `xdist_group("shared_window")`, so `--dist loadgroup` (pyproject.toml) runs them on one worker and only
    that worker builds it."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from vib_tutorial.gui import close_graphics
    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    w._timer.stop()
    w.show()
    yield w
    w.close()
    close_graphics(w)


# ------------------------------------------------------------------ the margin
# Each worker's peak address space, in the run's summary: the GUI tests' windows add up, and a worker that
# reaches `ulimit -v` fails with a MemoryError or a segfault rather than a message.


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
                    f"WARNING: {name} reached {100 * mb / limit:.0f} % of ulimit -v: the GUI tests' windows are "
                    "close to running it out (see CLAUDE.md).",
                    yellow=True,
                    bold=True,
                )
