"""`gui/frame_loop.py`: a page's frame loop, on a bare widget standing in for a page."""

from __future__ import annotations

import gc
import os
import time
import weakref

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from vib_tutorial.gui import frame_loop  # noqa: E402
from vib_tutorial.gui.frame_loop import FrameLoop  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def _free_windows(app, free_windows):
    """Each test's windows torn down and freed after it (conftest's `free_windows`)."""


class Page(QtWidgets.QWidget):
    def __init__(self, work_s: float = 0.0) -> None:
        super().__init__()
        self.running = False
        self.work_s = work_s
        self.ticks: list[float] = []
        self.restarts = 0
        self.t = 0.0
        self.loop = FrameLoop(self, "Test page", self._tick, self._runs, self._sim_t, self._restart)

    def _tick(self, dt: float) -> None:
        self.ticks.append(dt)
        self.t += dt
        time.sleep(self.work_s)

    def _runs(self) -> bool:
        return self.running

    def _sim_t(self) -> float:
        return self.t

    def _restart(self) -> None:
        self.restarts += 1


def test_it_runs_while_the_page_shows_and_wants_to(app):
    page = Page()
    page.loop.start()
    assert not page.loop.timer.isActive(), "not shown yet"
    page.show()
    assert not page.loop.timer.isActive(), "shown, but at rest"
    page.running = True
    page.loop.start()
    assert page.loop.timer.isActive() and page.restarts == 1
    page.loop.start()
    assert page.restarts == 1, "already running: the clock isn't reset"
    page.hide()
    assert not page.loop.timer.isActive(), "hidden: stopped"
    page.show()
    assert page.loop.timer.isActive() and page.restarts == 2, "shown again: on again, from a fresh clock"
    page.running = False  # a Pause: the frame already queued does nothing, and queues no other
    page.loop.stop()
    page.loop._frame()
    assert page.ticks == [] and not page.loop.timer.isActive()
    page.close()


def test_the_next_frame_is_frame_ms_after_this_one_began(app):
    """A frame's work comes out of the wait, not on top of it (16 ms + its work held the rotor at 47 fps)."""
    page = Page(work_s=0.008)
    page.running = True
    page.show()
    page.loop._frame()
    assert len(page.ticks) == 1 and page.loop.timer.isActive()
    assert page.loop.timer.remainingTime() <= frame_loop.FRAME_MS - 7
    page.close()


def test_a_stall_steps_at_most_max_dt(app):
    page = Page()
    page.running = True
    page.show()
    page.loop.start()
    time.sleep(0.15)
    page.loop._frame()
    assert page.ticks == [frame_loop.MAX_DT]
    page.close()


def test_it_holds_the_page_weakly(app):
    """A loop holding its page's methods strongly would keep the page alive through Qt (`gui/weak.py`)."""
    page = Page()
    with pytest.raises(TypeError):
        FrameLoop(page, "Test page", lambda dt: None, page._runs, page._sim_t)
    page.running = True
    page.show()
    page.loop.start()
    ref = weakref.ref(page)
    page.close()
    del page
    gc.collect()
    assert ref() is None
