"""Work too slow for a control's change, done off the GUI thread: debounced, and only the latest result drawn.

A page whose control change works out something slow (a stepped sine by harmonic balance, a long record) holds
the window still while it does: the spin box doesn't step, the window doesn't repaint. A `Recompute` instead
runs it on a worker thread:

- `request(job)` waits `delay_ms` for the reader to stop turning the box (each new request restarts the wait),
  then runs `job()` on the worker. `job` is built on the GUI thread from the controls' values and the page's
  plain data (a frozen `ChainSystem`, arrays it no longer changes), and touches no widget: it's a pure function,
  the same work a view model does.
- When it's done, `done(result)` runs on the GUI thread, unless a newer request came in meanwhile: then that
  one runs, and the stale result is dropped (latest wins), so a page never draws an answer to old settings.
- `now(job)` runs it at once, on the GUI thread, and drops anything pending: for a trial, Defaults or a new
  chain, where the reader asked for a whole new setup and waits for it (and tests read the page right after).
- `wait()` processes events until nothing is pending (tests).

An exception in `job` is raised again in `done`'s place on the GUI thread, where the tests see it.

Working on it: the page keeps the drawing; a `Recompute` only moves a pure function off the thread. The job
holds the GIL while it runs Python, so the window stays responsive but not idle: keep jobs to a second or so.
`tests/test_recompute.py` checks the debounce, latest-wins and the exception; the Nonlinearity page's sweep is
the first user (#316 D2). Use one for a control's work over ~100 ms, not a page's own QTimer, which only
debounces and still holds the window while the work runs (#409 P-14: Shock and Model updating moved off theirs).
Users: Nonlinearity's sweeps, Random vibration, OMA, Seismic's records, Vibro-acoustics' slow tabs, Variability,
the Rotor builder's sweeps, Fluid–structure's plate, Shock, Model updating.
"""

from __future__ import annotations

import time
import weakref
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from PySide6 import QtCore, QtWidgets

_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="recompute")


class _Failed:
    def __init__(self, error: BaseException) -> None:
        self.error = error


class Recompute(QtCore.QObject):
    _finished = QtCore.Signal(object)  # (generation, result), from the worker: queued to the GUI thread

    def __init__(self, done: Callable[[Any], None], parent: QtCore.QObject, delay_ms: int = 80) -> None:
        super().__init__(parent)
        # Not held strongly: `done` is the page's method, and the page owns this object.
        self._done = weakref.WeakMethod(done) if hasattr(done, "__self__") else (lambda: done)
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(delay_ms)
        self._timer.timeout.connect(self._start)
        self._finished.connect(self._on_finished)
        self._job: Callable[[], Any] | None = None
        self._generation = 0
        self._running = False

    def request(self, job: Callable[[], Any]) -> None:
        self._job = job
        self._generation += 1
        self._timer.start()

    def now(self, job: Callable[[], Any]) -> None:
        self._timer.stop()
        self._job = None
        self._generation += 1  # whatever is running is stale now
        self._deliver(job())

    @property
    def busy(self) -> bool:
        return self._running or self._job is not None

    def wait(self, timeout: float = 30.0) -> None:
        end = time.monotonic() + timeout
        while self.busy and time.monotonic() < end:
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 20)
            time.sleep(0.002)
        if self.busy:
            raise TimeoutError("a Recompute job didn't finish")

    def _start(self) -> None:
        if self._running or self._job is None:
            return  # the job in hand starts when the running one finishes
        job, self._job = self._job, None
        self._running = True
        _POOL.submit(self._run, job, self._generation)

    def _run(self, job: Callable[[], Any], generation: int) -> None:
        try:
            result: Any = job()
        except BaseException as error:  # handed to the GUI thread to raise there
            result = _Failed(error)
        try:
            self._finished.emit((generation, result))
        except RuntimeError:  # the page went while the job ran
            pass

    def _on_finished(self, payload: tuple[int, Any]) -> None:
        self._running = False
        generation, result = payload
        if generation == self._generation and self._job is None:
            self._deliver(result)
        elif self._job is not None and not self._timer.isActive():
            self._start()  # a newer request waited for this one

    def _deliver(self, result: Any) -> None:
        if isinstance(result, _Failed):
            raise result.error
        if (done := self._done()) is not None:
            done(result)
