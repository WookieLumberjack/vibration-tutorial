"""Scrubbing back through a running page's record while it's paused.

A page that simulates as it runs (the Simulation page, Duffing, Torsional's start-up, the rotor) keeps what it
has recorded. Paused, a `Scrub` shows a cursor on each of its time plots at the latest instant; dragging one
moves them all and emits `scrubbed(t)`, and the page draws its body as it was then. Resumed, the cursors go and
the run carries on from where it was paused (the past is only looked at, not rewound to).

- `Scrub(*plots)`: the cursors (`drag.DragLine`s, in the theme's strong color), hidden.
- `show(latest, earliest)`: paused: the cursors at `latest`, draggable back to `earliest`.
- `hide()`: running again (or reset): the cursors go, and `t` is None.
- `t`: the instant scrubbed to, or None while the page shows its latest state.

The page calls `apply_theme()` from its own.

Working on it: `tests/test_scrub.py` scrubs each page that has one.
"""

from __future__ import annotations

from PySide6 import QtCore

from .drag import DragLine

TIP = (
    "<p><b>Drag</b> the cursor back along the record to see the motion as it was then (the run is paused). "
    "Start again and it carries on from where it was paused.</p>"
)


class Scrub(QtCore.QObject):
    scrubbed = QtCore.Signal(float)  # t (s)

    def __init__(self, *plots, tip: str = TIP) -> None:
        super().__init__()
        self.t: float | None = None
        self.lines: list[DragLine] = []
        for plot in plots:
            line = DragLine(tip, role="strong")
            line.hide()
            plot.addItem(line, ignoreBounds=True)
            line.dragged.connect(self._on_dragged)
            self.lines.append(line)

    @property
    def shown(self) -> bool:
        return self.lines[0].isVisible() if self.lines else False

    def show(self, latest: float, earliest: float = 0.0) -> None:
        for line in self.lines:
            line.set_scale(False, (earliest, latest))
            line.set_value(latest if self.t is None else min(max(self.t, earliest), latest))
            line.show()

    def hide(self) -> None:
        self.t = None
        for line in self.lines:
            line.hide()

    def apply_theme(self) -> None:
        for line in self.lines:
            line.apply_theme()

    def _on_dragged(self, t: float) -> None:
        self.t = t
        for line in self.lines:
            if not line.moving:
                line.set_value(t)
        self.scrubbed.emit(t)
