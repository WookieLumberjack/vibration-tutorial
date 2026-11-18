"""The Components tab the 3D rigid body and the Gyroscope share (#409 P-4, S-6).

Show (ω, L, τ or the inertia tensor), the axes to show it in and how much of the run, a plot of the components
against time with a cursor at the instant the snapshot under it is taken, and a note. The page works out the
curves (`view_models.components.series`) and the snapshot from its own record, and hands them over: `draw`,
`clear`, `set_snapshot`. A box changed emits `changed`; the cursor dragged, `dragged(t)`.

It draws every frame while its page runs, so: the legend is rebuilt only when its entries change (a show or an
axes picked), not every frame (that, and the y axis autoranged every frame, held the 3D rigid body to 38 fps and
the Gyroscope to 44); the y range is held while the curves fit it (`SteadyRange`); the note and the snapshot are
LiveLabels.

Working on it: `tests/test_gui_smoke.py`'s Components tests of both pages; `tests/test_live_labels.py`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..view_models.components import SHOWS, SPANS, Series
from .drag import DragLine
from .layouts import ColumnScroll, form_layout, form_row
from .lesson import column, combo, plot, relabel
from .live_label import LiveLabel
from .plots import SteadyRange
from .style import colors
from .theming import add_legend, mute

CURVES = 6  # at most: the tensor's six entries


class ComponentsTab(ColumnScroll):
    changed = QtCore.Signal()
    dragged = QtCore.Signal(float)

    def __init__(
        self,
        frames: Sequence[str],
        tips: dict[str, str],
        snapshot_tip: str,
        plot_tip: str = "",
    ) -> None:
        super().__init__(QtWidgets.QWidget())  # its column set below, once built
        self.show_box = combo(SHOWS, tips["show"])
        self.frame_box = combo(frames, tips["frame"])
        self.span_box = combo([name for name, _ in SPANS], tips["span"])
        self.span_box.setCurrentIndex(1)
        self.plot = plot("ω", "rad/s", "Time", "s", height=240)
        if plot_tip:
            self.plot.setToolTip(plot_tip)
        self.legend = add_legend(self.plot.getPlotItem(), offset=(10, 10), colCount=4)
        self.items = [pg.PlotDataItem() for _ in range(CURVES)]
        self.size_item = pg.PlotDataItem()  # the vector's magnitude
        for item in (*self.items, self.size_item):
            self.plot.addItem(item)
        self.y_range = SteadyRange(self.plot, "y")
        self.time_line = DragLine(snapshot_tip, role="strong")  # the snapshot's instant
        self.time_line.setToolTip(snapshot_tip)
        self.time_line.sigDragged.connect(self._on_dragged)
        self.plot.addItem(self.time_line, ignoreBounds=True)
        self.note = LiveLabel(wrap=True)
        mute(self.note)
        self.snapshot = LiveLabel(wrap=True)
        self.snapshot.setToolTip(snapshot_tip)
        self._labels: tuple[str, ...] = ()  # the legend's entries now
        self._axis: tuple[str, str] | None = None  # the left axis's name and unit now
        boxes = QtWidgets.QWidget()
        form = form_layout(boxes)
        form.setContentsMargins(0, 0, 0, 0)
        for text, box in (("Show:", self.show_box), ("In axes:", self.frame_box), ("Over:", self.span_box)):
            form_row(form, text, box, box.toolTip())
            box.currentIndexChanged.connect(self._on_box)
        self.setWidget(column(boxes, self.plot, self.note, self.snapshot))
        self.apply_theme()

    def span(self) -> float | None:
        """How much of the run to plot, s; None: all of it."""
        return SPANS[self.span_box.currentIndex()][1]

    def _on_box(self) -> None:
        self.y_range.reset()
        self.changed.emit()

    def _on_dragged(self) -> None:
        self.dragged.emit(float(self.time_line.value()))

    def clear(self, note: str) -> None:
        """Nothing run yet: no curves, `note` saying so."""
        for item in (*self.items, self.size_item):
            item.setData([], [])
        self._set_legend((), [])
        self.note.setText(note)
        self.snapshot.setText("")
        self.time_line.hide()

    def draw(self, t: np.ndarray, s: Series, note: str) -> None:
        if self._axis != (s.name, s.unit):
            self._axis = (s.name, s.unit)
            relabel(self.plot, "left", s.name, s.unit)
        for k, item in enumerate(self.items):
            if k < len(s.curves):
                item.setData(t, s.curves[k][1])
            else:
                item.setData([], [])
        if s.size is None:
            self.size_item.setData([], [])
        else:
            self.size_item.setData(t, s.size)
        if s.labels != self._labels:
            entries = [(item, label) for item, (label, _) in zip(self.items, s.curves, strict=False)]
            self._set_legend(s.labels, entries + ([(self.size_item, s.size_label)] if s.size is not None else []))
        shown = [values for _, values in s.curves] + ([s.size] if s.size is not None else [])
        if t.size:
            self.y_range.fit(float(min(v.min() for v in shown)), float(max(v.max() for v in shown)))
        self.note.setText(note)
        self.time_line.show()

    def _set_legend(self, labels: tuple[str, ...], entries: list[tuple[pg.PlotDataItem, str]]) -> None:
        self._labels = labels
        self.legend.clear()
        for item, label in entries:
            self.legend.addItem(item, label)

    def set_snapshot(self, t: float | None, html: str) -> None:
        """The snapshot's text, its cursor at `t` (None: no snapshot)."""
        if t is not None:
            self.time_line.setValue(t)
        self.snapshot.setText(html)

    def apply_theme(self) -> None:
        c = colors
        for k, item in enumerate(self.items):  # 1 px: the body's components swing at the spin
            item.setPen(pg.mkPen(c.mass[k % len(c.mass)], width=1))
        self.size_item.setPen(pg.mkPen(c.strong, width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.time_line.apply_theme()
