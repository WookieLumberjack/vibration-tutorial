"""`gui/recompute.py`: slow work off the GUI thread, debounced, the latest result only."""

from __future__ import annotations

import os
import threading
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6 import QtCore  # noqa: E402

from vib_tutorial.gui.recompute import Recompute  # noqa: E402


@pytest.fixture(autouse=True)
def _free_windows(app, free_windows):
    """Each test's windows torn down and freed after it (conftest's `free_windows`)."""


class Page(QtCore.QObject):
    def __init__(self, delay_ms: int = 30) -> None:
        super().__init__()
        self.got: list = []
        self.threads: list[str] = []
        self.r = Recompute(self.done, self, delay_ms)

    def done(self, result) -> None:
        self.got.append(result)
        self.threads.append(threading.current_thread().name)


def slow(value, seconds: float = 0.05, ran: list | None = None):
    def job():
        if ran is not None:
            ran.append((value, threading.current_thread().name))
        time.sleep(seconds)
        return value

    return job


def test_a_burst_of_requests_runs_once_off_the_gui_thread_and_delivers_on_it(app):
    p = Page()
    ran: list = []
    for v in range(5):  # a spin box turned five steps quickly
        p.r.request(slow(v, ran=ran))
    p.r.wait()
    assert ran and ran[-1][0] == 4 and len(ran) == 1  # debounced: only the last request ran
    assert ran[0][1].startswith("recompute")  # on the worker
    assert p.got == [4] and p.threads == [threading.main_thread().name]


def test_a_request_while_one_runs_wins_and_the_stale_result_is_dropped(app):
    p = Page(delay_ms=0)
    ran: list = []
    p.r.request(slow("old", 0.2, ran))
    end = time.monotonic() + 5
    while not ran and time.monotonic() < end:  # until the first is running
        app.processEvents()
        time.sleep(0.002)
    p.r.request(slow("new", 0.01, ran))
    p.r.wait()
    assert [v for v, _ in ran] == ["old", "new"] and p.got == ["new"]


def test_now_runs_at_once_and_drops_what_was_pending(app):
    p = Page()
    p.r.request(slow("pending"))
    p.r.now(slow("now", 0.0))
    assert p.got == ["now"]
    p.r.wait()
    for _ in range(10):
        app.processEvents()
    assert p.got == ["now"]


def test_a_failing_job_raises_on_the_gui_thread(app, monkeypatch):
    p = Page(delay_ms=0)

    def bad():
        raise ValueError("no")

    with pytest.raises(ValueError, match="no"):
        p.r.now(bad)
    # from the worker: raised again in the GUI thread's slot (where conftest reports a slot that raised)
    raised: list = []
    real = Recompute._deliver

    def deliver(self, result):
        try:
            real(self, result)
        except ValueError as error:
            raised.append((error, threading.current_thread().name))

    monkeypatch.setattr(Recompute, "_deliver", deliver)
    p.r.request(bad)
    p.r.wait()
    assert len(raised) == 1 and str(raised[0][0]) == "no" and raised[0][1] == threading.main_thread().name


def test_busy_a_while_it_says_its_updating(app):
    """`updating` once busy `BUSY_MS` (the run bar's "Updating…"), not for a quick job, and not once it's done
    (#409 U-24)."""
    from vib_tutorial.gui import recompute

    page, other = Page(delay_ms=10), Page()
    page.r.request(slow(1, seconds=0.2))
    assert page.r.busy and not page.r.updating  # not yet: a quick job never says so
    end = time.monotonic() + recompute.BUSY_MS / 1000 + 0.02
    while time.monotonic() < end:
        QtWidgets.QApplication.processEvents()
        time.sleep(0.005)
    assert page.r.updating and recompute.updating_under(page) and not recompute.updating_under(other)
    page.r.wait()
    assert page.got == [1] and not page.r.updating and not recompute.updating_under(page)
    page.r.now(slow(2, seconds=0.0))
    assert not page.r.updating
