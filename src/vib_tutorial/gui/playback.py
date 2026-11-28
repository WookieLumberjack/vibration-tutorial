"""A response played back in time: Start/Pause, Reset, a speed, a clock, and a time cursor on its plots.

The small pages (the One mass group) work a response out in closed form for a whole span at once and draw
it as a curve. A `Playback` walks a time `t` along that span so the page can move a body with it:

- `run` (a `lesson.RunButton`, Space), `reset` (back to t = 0, waiting for Start), `speed` (slow motion:
  a 3 Hz swing is a blur at 1×) and `clock` (the time), laid out by `bar()`.
- `add_cursor(plot)`: a `drag.DragLine` at `t` on a time plot. Dragging it scrubs (pausing the motion); it
  follows `t` as it plays.
- `moved(t)` whenever `t` changes (a frame, a scrub, a rewind, a new span): the page places its body there.
- `set_span(span)` when the page redraws its curves; `t` is kept, clamped into it. At the span's end the
  motion pauses (Resume plays it again from the start).
- `play()` from the start: a trial's or a pluck's release. A control change neither starts nor stops it.

Its `FrameLoop` runs only while the page shows and the motion runs (CLAUDE.md's run state). The page calls
`apply_theme()` from its own.

Working on it: `tests/test_playback.py` checks the clock, the end, the scrub and the rewind, and each page's body
following it (the Quarter-car's; the One oscillator pages that played back are live now, #478); those pages are
in `test_lesson_pages_trials_and_tabs` and the run-state tests.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from . import coalesce, plot_patches
from .drag import DragLine
from .frame_loop import FrameLoop
from .layouts import FlowLayout, ResetButton, group
from .lesson import LiveLabel, RunButton, combo
from .transport import speed_text

SPEEDS = tuple((speed_text(s), s) for s in (1.0, 0.5, 0.25, 0.1))
SPEED_TIP = (
    "<p>How fast the motion plays: 0.25× shows a 3 Hz swing at less than 1 Hz. The plots and readouts are in real "
    "time whatever the speed.</p>"
)
CURSOR_TIP = "<p><b>Drag</b> the time cursor to scrub through the motion (it pauses).</p>"


class Playback(QtCore.QObject):
    moved = QtCore.Signal(float)  # t (s)

    def __init__(self, page: QtWidgets.QWidget, name: str, speed: float = 0.25) -> None:
        super().__init__(page)
        self.t = 0.0
        self.span = 1.0
        self.run = RunButton()
        self.run.toggled.connect(self._on_run)
        self.reset = ResetButton()
        self.reset.setToolTip("<p>Back to t = 0, at the start, waiting for Start.</p>")
        self.reset.clicked.connect(self.rewind)
        self.speed = combo([label for label, _ in SPEEDS], SPEED_TIP)
        self.speed.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)  # short items
        self.speed.setCurrentIndex(next(i for i, (_, s) in enumerate(SPEEDS) if s == speed))
        self.speed.setAccessibleName("Playback speed")
        self.clock = LiveLabel()  # rewritten every frame: no layout pass each time
        self.clock.setAccessibleName("Time")
        self.cursors: list[DragLine] = []
        self.loop = FrameLoop(page, name, self._tick, self._runs, self._sim_t)
        self._show_clock()

    def bar(self, *extra: QtWidgets.QWidget) -> QtWidgets.QWidget:
        """The controls in a row that wraps: Start, Reset, the speed, the clock, and the page's `extra`."""
        bar = QtWidgets.QWidget()
        row = FlowLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.run)
        row.addWidget(self.reset)
        row.addWidget(group("Speed:", self.speed))
        row.addWidget(self.clock)
        for widget in extra:
            row.addWidget(widget)
        return bar

    def add_cursor(self, plot) -> DragLine:
        """A time cursor on `plot` that follows `t`, and scrubs it when dragged."""
        line = DragLine(CURSOR_TIP, bounds=(0.0, self.span), role="strong")
        plot.addItem(line, ignoreBounds=True)
        plot_patches.hold_still(plot)  # the cursor moves over still curves: each one drawn from its pixmap
        line.dragged.connect(self.seek)
        line.set_value(self.t)
        self.cursors.append(line)
        return line

    @property
    def running(self) -> bool:
        return self.run.running

    @property
    def rate(self) -> float:
        return SPEEDS[self.speed.currentIndex()][1]

    def set_span(self, span: float) -> None:
        """The span the page's curves cover: `t` stays where it is, if it still falls inside."""
        self.span = max(float(span), 1e-9)
        for line in self.cursors:
            line.setBounds([0.0, self.span])
        self._move(min(self.t, self.span))

    def play(self) -> None:
        """From the start: a trial, or a body let go."""
        self._move(0.0)
        self.run.set_running(True)
        self.loop.start()

    def pause(self) -> None:
        if self.running:
            self.run.set_running(False)

    def rewind(self) -> None:
        """Reset: back to t = 0, waiting for Start."""
        self.run.rest()
        self._move(0.0)

    def seek(self, t: float) -> None:
        """Scrub to `t` (the cursor dragged): the motion pauses there."""
        self.pause()
        self._move(min(max(float(t), 0.0), self.span))

    def apply_theme(self) -> None:
        for line in self.cursors:
            line.apply_theme()

    # ------------------------------------------------------------------
    def _on_run(self, on: bool) -> None:
        if on:
            if self.t >= self.span:  # Resume at the end: play it again
                self._move(0.0)
            self.loop.start()

    def _runs(self) -> bool:
        return self.running

    def _sim_t(self) -> float | None:
        return self.t

    def _tick(self, dt: float) -> None:
        t = self.t + dt * self.rate
        if t >= self.span:
            t = self.span
            self.run.set_running(False)  # paused at the end: Resume plays it again
        self._move(t)

    def _move(self, t: float) -> None:
        self.t = t
        self._place()

    @coalesce.coalesced
    def _place(self) -> None:
        """The cursors, the clock and the page's body at `t`: once a trial, at its end (`coalesce`)."""
        for line in self.cursors:
            if not line.moving:
                line.set_value(self.t)
        self._show_clock()
        self.moved.emit(self.t)

    def _show_clock(self) -> None:
        self.clock.setText(f"t = {self.t:.2f} s")
