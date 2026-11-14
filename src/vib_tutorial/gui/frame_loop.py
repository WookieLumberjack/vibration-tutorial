"""A page's frame loop: the timer that steps its motion about 60 times a second while it runs and shows.

A page that animates holds one `FrameLoop`, given its own methods:

- `tick(dt)`: the frame's work, `dt` the wall-clock seconds since the last frame (at most `MAX_DT`, so a
  stall doesn't jump the motion). The page scales it by its playback speed and steps and draws.
- `runs()`: whether it should be running now (usually `self.running`; a builder also needs its result,
  Continuum its Waves tab showing). Read before each frame and before starting.
- `sim_t()`: the simulated time for the frame log (`frame_stats`), or None.
- `restart()` (optional): called as the loop starts or resumes, with the clock just reset (a page that
  keeps a target time sets it to where the simulation is).

The loop starts itself when the page is shown and stops when it's hidden (an event filter on the page),
and the page calls `start()` when it starts or resumes (a Start, a trial) and `stop()` if it must halt
at once; a frame queued before a Pause or Reset checks `runs()` and does nothing. Each frame runs inside
`frame_stats.frame` and ends with `settle_plots(page)`, and the next is queued FRAME_MS after this one
*began*, not after it ended, so a frame's work comes out of the wait instead of adding to it.

It holds the page and its methods weakly (`weakref.WeakMethod`), and is a QObject child of the page:
a bound method held strongly would keep the window alive through C++ (see `weak`).

Working on it: `tests/test_frame_loop.py` checks the timing, the show/hide and the weak references;
`test_gui_smoke.py`'s frame test runs it on the pages that use it.
"""

from __future__ import annotations

import inspect
import weakref
from collections.abc import Callable
from typing import Any

from PySide6 import QtCore, QtWidgets

from . import frame_stats, settle_plots

FRAME_MS = 16  # ~60 fps
MAX_DT = 0.1  # s: the longest step a frame takes, however long since the last


def _weak(method: Callable[..., Any] | None) -> weakref.WeakMethod | None:
    if method is None:
        return None
    if not inspect.ismethod(method):  # a lambda or closure over the page would keep it alive
        raise TypeError(f"FrameLoop takes the page's bound methods, not {method!r}")
    return weakref.WeakMethod(method)


class FrameLoop(QtCore.QObject):
    def __init__(
        self,
        page: QtWidgets.QWidget,
        name: str,
        tick: Callable[[float], None],
        runs: Callable[[], bool],
        sim_t: Callable[[], float | None],
        restart: Callable[[], None] | None = None,
        frame_ms: int = FRAME_MS,
    ) -> None:
        super().__init__(page)
        self.name = name
        self.frame_ms = frame_ms
        self._page = weakref.ref(page)
        self._tick = _weak(tick)
        self._runs = _weak(runs)
        self._sim_t = _weak(sim_t)
        self._restart = _weak(restart)
        self._clock = QtCore.QElapsedTimer()
        self._last = 0.0
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._frame)
        page.installEventFilter(self)

    def _call(self, ref: weakref.WeakMethod | None, *args: Any) -> Any:
        method = ref() if ref is not None else None
        return None if method is None else method(*args)

    def _should_run(self) -> bool:
        page = self._page()
        return page is not None and page.isVisible() and bool(self._call(self._runs))

    def start(self) -> None:
        """Run, if the page shows and wants to (`runs`) and the loop isn't already: the next frame in FRAME_MS."""
        if self._should_run() and not self.timer.isActive():
            self._clock.restart()
            self._last = 0.0
            self._call(self._restart)
            self.timer.start(self.frame_ms)

    def stop(self) -> None:
        self.timer.stop()

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        page = getattr(self, "_page", None)  # None while the page is torn down: Qt still calls the filter
        if page is not None and watched is page():
            if event.type() == QtCore.QEvent.Type.Show:
                self.start()
            elif event.type() == QtCore.QEvent.Type.Hide:
                self.stop()
        return False

    def _frame(self) -> None:
        page = self._page()
        if page is None or not self._call(self._runs):  # paused or reset since this frame was queued
            return
        began = self._clock.elapsed()
        now = began / 1000.0
        dt = min(now - self._last, MAX_DT)
        self._last = now
        sim_t = self._sim_t() if self._sim_t is not None else None
        with frame_stats.frame(self.name, sim_t, self.frame_ms):
            self._call(self._tick, dt)
            settle_plots(page)
        if self._should_run():  # the next frame FRAME_MS after this one began, not after it ended
            self.timer.start(max(1, self.frame_ms - (self._clock.elapsed() - began)))
