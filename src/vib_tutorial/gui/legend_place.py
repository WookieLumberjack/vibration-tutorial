"""A plot's legend kept off its data: moved to the corner (or the middle of the top or bottom edge) with the
least drawn under it, whenever its curves or range change.

`follow(legend)` (`theming.add_legend` calls it for every legend it makes; a legend made by hand over a plot
calls it itself) gives the legend a `Placer`. It rasterizes what the plot draws into cells of `CELL` px (the
lines between points too, not only the points: a steep line crosses a corner with no point in it), counts
the cells under the legend at each of `SPOTS`, and moves it where fewest are covered (`best`): it stays
while its spot covers at most `KEEP` cells, so a live plot's legend doesn't hop from frame to frame, and
where it's put first (the offset it was made with) wins ties. Where every spot covers more than `SLIVER`,
it goes above the plot (`OUTSIDE`), in as few rows as fit, in the row a PlotItem keeps for its top axis
(the ViewBox clips what it holds); it comes back in once a spot is clear.

It works at most every `THROTTLE_MS` (a frame loop's `setData` only starts the timer), doubling up to
`SLOWEST_MS` while its plot keeps changing and the legend stays put (back to `THROTTLE_MS` after a second's
quiet, or when it moves), on the displayed
data (decimated, in log units on a log axis), and only while the plot has a size: a placement costs well
under a millisecond (`tests/test_legend_place.py`). Dragging the legend by hand turns it off for that legend.

Working on it: `coverage` and `best` are pure (cells and rectangles in px); `tests/test_legend_place.py`
checks them and a plot's legend moving off a curve, and `test_no_legend_hides_data` in `test_gui_smoke`
walks every page's tabs at 1280×800 for a legend over more than a sliver of its plot's data.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pyqtgraph as pg
import shiboken6
from PySide6 import QtCore

CELL = 4.0  # px a side
KEEP = 2  # cells a legend may cover before it looks for a better spot
SLIVER = 8  # cells a legend may cover inside its plot, where no spot is clearer, before it goes above the plot
OUTSIDE = -1  # the spot above the plot, the legend in a row
THROTTLE_MS = 300
SLOWEST_MS = 2400  # the throttle, doubled after each placement that left it put while its plot keeps changing
MAX_SAMPLES = 200_000  # samples along a plot's lines, at most: coarser beyond
# (where on the legend, where on the plot) as fractions of their widths and heights
SPOTS = (
    ((0.0, 0.0), (0.0, 0.0)),
    ((1.0, 0.0), (1.0, 0.0)),
    ((0.0, 1.0), (0.0, 1.0)),
    ((1.0, 1.0), (1.0, 1.0)),
    ((0.5, 0.0), (0.5, 0.0)),
    ((0.5, 1.0), (0.5, 1.0)),
)


def cells(xs: list[np.ndarray], ys: list[np.ndarray], lines: list[bool], size: tuple[float, float]) -> np.ndarray:
    """A boolean grid (rows down, `CELL` px each) of the cells the points, and the lines between them where
    `lines`, pass through, in a plot `size` px wide and high; x and y in px from its top left."""
    nx, ny = max(math.ceil(size[0] / CELL), 1), max(math.ceil(size[1] / CELL), 1)
    grid = np.zeros((ny, nx), bool)
    for x, y, joined in zip(xs, ys, lines, strict=True):
        if x.size == 0:
            continue
        if joined and x.size > 1:
            sx, sy = _along(x, y, nx * CELL, ny * CELL)
            x, y = np.concatenate([x, sx]), np.concatenate([y, sy])
        ok = np.isfinite(x) & np.isfinite(y)
        col, row = np.floor(x[ok] / CELL).astype(np.int64), np.floor(y[ok] / CELL).astype(np.int64)
        inside = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
        grid[row[inside], col[inside]] = True
    return grid


def _along(x: np.ndarray, y: np.ndarray, width: float, height: float) -> tuple[np.ndarray, np.ndarray]:
    """Points every cell or so along the lines between consecutive points, each line clipped to the plot
    (Liang–Barsky) first: a line to a point far off the plot still crosses it."""
    x0, y0, dx, dy = x[:-1], y[:-1], np.diff(x), np.diff(y)
    ok = np.isfinite(x0) & np.isfinite(y0) & np.isfinite(dx) & np.isfinite(dy)
    t0, t1 = np.zeros(dx.size), np.ones(dx.size)
    with np.errstate(divide="ignore", invalid="ignore"):
        for p, q in ((-dx, x0), (dx, width - x0), (-dy, y0), (dy, height - y0)):
            r = q / p
            t0 = np.where(p < 0, np.maximum(t0, r), t0)
            t1 = np.where(p > 0, np.minimum(t1, r), t1)
            ok &= ~((p == 0) & (q < 0))
    ok &= t0 <= t1
    x0, y0, dx, dy, t0, t1 = x0[ok], y0[ok], dx[ok], dy[ok], t0[ok], t1[ok]
    span = (t1 - t0) * np.maximum(np.abs(dx), np.abs(dy)) / CELL
    steps = np.ceil(span).astype(np.int64) + 1
    total = int(steps.sum())
    if total > MAX_SAMPLES:
        steps = np.maximum(np.ceil(steps * (MAX_SAMPLES / total)).astype(np.int64), 1)
        total = int(steps.sum())
    if total == 0:
        return np.empty(0), np.empty(0)
    seg = np.repeat(np.arange(steps.size), steps)
    k = np.arange(total) - np.repeat(np.cumsum(steps) - steps, steps)
    denom = np.maximum(np.repeat(steps, steps) - 1, 1)
    t = t0[seg] + (t1[seg] - t0[seg]) * k / denom
    return x0[seg] + t * dx[seg], y0[seg] + t * dy[seg]


def coverage(grid: np.ndarray, rect: tuple[float, float, float, float]) -> int:
    """How many of `grid`'s filled cells a rectangle (left, top, width, height in px) covers."""
    left, top, width, height = rect
    c0, r0 = max(int(left // CELL), 0), max(int(top // CELL), 0)
    c1, r1 = math.ceil((left + width) / CELL), math.ceil((top + height) / CELL)
    return int(grid[r0:r1, c0:c1].sum())


def spot_rect(spot: int, legend: tuple[float, float], plot: tuple[float, float], margin: float):
    """Where the legend (its width, height) sits in the plot at `SPOTS[spot]`, `margin` px in from the edges."""
    (ix, iy), (px, py) = SPOTS[spot]
    left = px * plot[0] - ix * legend[0] + margin * (1 - 2 * ix if ix != 0.5 else 0)
    top = py * plot[1] - iy * legend[1] + margin * (1 - 2 * iy if iy != 0.5 else 0)
    return left, top, legend[0], legend[1]


def best(
    grid: np.ndarray,
    legend: tuple[float, float],
    plot: tuple[float, float],
    margin: float,
    now: int,
    outside: bool = True,
) -> int:
    """The spot to use, from the one in use, `now` (`OUTSIDE` above the plot, where `outside` allows it).

    A spot covering at most `KEEP` cells is clear: the one in use stays while it's clear, else the clearest
    is taken (`now` first, then `SPOTS`' order). Where none is clear, the clearest inside will do if it
    covers at most `SLIVER`; failing that the legend goes above the plot, or, where it can't, moves only
    to a spot covering half as many as its own."""
    scores = [coverage(grid, spot_rect(i, legend, plot, margin)) for i in range(len(SPOTS))]
    inside = now if now != OUTSIDE else 0
    if now != OUTSIDE and scores[now] <= KEEP:
        return now
    order = [inside] + [i for i in range(len(SPOTS)) if i != inside]
    pick = min(order, key=lambda i: scores[i])
    if scores[pick] <= KEEP:
        return pick
    if now == OUTSIDE:
        return OUTSIDE  # back in only where it's clear
    if scores[pick] <= SLIVER:
        return pick
    if outside:
        return OUTSIDE
    return pick if scores[pick] <= scores[now] / 2 else now


class Placer(QtCore.QObject):
    """Keeps one legend, over its ViewBox, where it hides least (the module's docstring)."""

    def __init__(self, legend: pg.LegendItem) -> None:
        super().__init__(legend)
        self.legend = legend
        self.vb = legend.parentItem()
        plot = self.vb.parentItem()
        # Above the plot only where there's a row for it: a PlotItem's, its top axis hidden.
        self.plot = plot if isinstance(plot, pg.PlotItem) and not plot.getAxis("top").isVisible() else None
        self.columns = legend.columnCount
        self.inside_size: tuple[float, float] | None = None  # the legend's size inside, while it's outside
        self.manual = False
        offset = legend.opts.get("offset") or (30, 30)
        self.margin = float(max(abs(offset[0]), abs(offset[1]), 2))
        right = offset[0] < 0
        bottom = offset[1] < 0
        self.spot = (1 if right else 0) + (2 if bottom else 0)
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(THROTTLE_MS)
        self.timer.timeout.connect(self.place)
        self._last = 0.0  # time.monotonic() of the last placement
        self.vb.sigRangeChanged.connect(self.soon)
        self.vb.sigResized.connect(self.soon)
        self.vb.childGroup.itemsChangedListeners.append(self)
        self._watched: set[int] = set()
        legend.mouseDragEvent = self._dragged  # pyqtgraph's scene calls it from Python
        self._drag = pg.LegendItem.mouseDragEvent
        self.soon()

    def itemsChanged(self) -> None:  # the ViewBox's ChildGroup: an item added or taken away
        self.soon()

    def soon(self, *_) -> None:
        if self.manual or self.timer.isActive():
            return
        if time.monotonic() - self._last > (self.timer.interval() + 1000) / 1000:
            self.timer.setInterval(THROTTLE_MS)  # quiet a while: a control's change, placed promptly
        self.timer.start()

    def _dragged(self, event) -> None:
        self.manual = True
        self._drag(self.legend, event)

    def _items(self) -> list:
        items = [
            i for i in self.vb.addedItems if isinstance(i, (pg.PlotDataItem, pg.PlotCurveItem, pg.ScatterPlotItem))
        ]
        for item in items:
            if id(item) not in self._watched and hasattr(item, "sigPlotChanged"):
                item.sigPlotChanged.connect(self.soon)
                self._watched.add(id(item))
        return [i for i in items if i.isVisible()]

    def grid(self) -> np.ndarray | None:
        """The cells the plot's data fills, or None while there's nothing to place against."""
        legend = self.legend
        width, height = self.vb.width(), self.vb.height()
        if width < 10 or height < 10 or not legend.isVisible() or not legend.items:
            return None
        (x0, x1), (y0, y1) = self.vb.viewRange()
        if x1 <= x0 or y1 <= y0:
            return None
        inverted_x, inverted_y = self.vb.state["xInverted"], self.vb.state["yInverted"]
        xs, ys, lines = [], [], []
        for item in self._items():
            x, y = item.getData()
            if x is None or y is None:
                continue
            x, y = np.asarray(x, float), np.asarray(y, float)
            n = min(x.size, y.size)
            px = (x[:n] - x0) / (x1 - x0) * width
            py = (y[:n] - y0) / (y1 - y0) * height
            xs.append(width - px if inverted_x else px)
            ys.append(py if inverted_y else height - py)
            lines.append(_joined(item))
        return cells(xs, ys, lines, (width, height))

    def covered(self) -> tuple[int, int]:
        """The cells of data under the legend where it is now, and the data's cells in all (for the tests)."""
        grid = self.grid()
        if grid is None:
            return 0, 0
        if self.spot == OUTSIDE:
            return 0, int(grid.sum())
        rect = self.legend.boundingRect()
        size = (rect.width(), rect.height())
        return coverage(grid, spot_rect(self.spot, size, (self.vb.width(), self.vb.height()), self.margin)), int(
            grid.sum()
        )

    def place(self) -> None:
        """Move the legend to its best spot now (the timer calls it)."""
        if self.manual or not all(shiboken6.isValid(o) for o in (self.legend, self.vb, self.plot) if o is not None):
            return  # its plot torn down while the timer ran
        self._last = time.monotonic()
        grid = self.grid()
        if grid is None:
            return
        before = self.spot
        rect = self.legend.boundingRect()
        size = self.inside_size or (rect.width(), rect.height())
        plot = (self.vb.width(), self.vb.height())
        spot = best(grid, size, plot, self.margin, self.spot, outside=self.plot is not None)
        if spot == OUTSIDE:
            self._above(size)
        elif spot != self.spot:
            if self.spot == OUTSIDE:
                self._inside()
            self.spot = spot
            (ix, iy), (px, py) = SPOTS[spot]
            m = self.margin
            self.legend.anchor(
                (ix, iy), (px, py), (m * (1 - 2 * ix) if ix != 0.5 else 0, m * (1 - 2 * iy) if iy != 0.5 else 0)
            )
        # A live plot's legend that stays put is placed ever less often, down to every SLOWEST_MS.
        moved = self.spot != before
        self.timer.setInterval(THROTTLE_MS if moved else min(2 * self.timer.interval(), SLOWEST_MS))

    def _above(self, inside: tuple[float, float]) -> None:
        """Above the plot, in as few rows as fit its width, the plot's top-axis row made room for it."""
        assert self.plot is not None, "best() goes outside only where there's a PlotItem"
        legend, plot = self.legend, self.plot
        if self.spot != OUTSIDE:
            self.spot, self.inside_size = OUTSIDE, inside
            legend.setParentItem(plot)
            for n in range(len(legend.items), 0, -1):
                legend.setColumnCount(n)
                if legend.boundingRect().width() <= self.vb.width() or n == 1:
                    break
        rect, box = legend.boundingRect(), self.vb.geometry()
        plot.layout.setRowMinimumHeight(1, rect.height() + 2)
        legend.anchor(
            (0, 0), (0, 0), (box.left() + max((box.width() - rect.width()) / 2, 0), box.top() - rect.height() - 1)
        )

    def _inside(self) -> None:
        """Back over the plot, its own columns, the row given back."""
        assert self.plot is not None, "it went outside over a PlotItem"
        self.plot.layout.setRowMinimumHeight(1, 0)
        self.legend.setParentItem(self.vb)
        self.legend.setColumnCount(self.columns)
        self.inside_size = None


def _joined(item) -> bool:
    """Whether the item draws lines between its points (a scatter, or a curve with no pen, doesn't)."""
    if isinstance(item, pg.ScatterPlotItem):
        return False
    pen = item.opts.get("pen")
    connect = item.opts.get("connect", "all")
    return pen is not None and not (isinstance(connect, str) and connect == "pairs")


def follow(legend: pg.LegendItem) -> Placer:
    """Keep `legend` (over its plot's ViewBox) off the data; its Placer (the one it has, if it has one)."""
    return legend.findChild(Placer) or Placer(legend)
