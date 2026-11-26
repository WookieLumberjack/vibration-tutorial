"""A mode shape swinging on a plot: Re(ψ e^{iθ}) drawn as θ turns, its nodes marked as they move.

A real mode is a standing wave: every point reaches its peak together and its nodes stay put. A complex mode
(non-proportional damping, a gyroscopic term) is partly traveling: each point peaks at its own phase, so
the shape never passes through zero all at once and its nodes slide along the chain. `ModeSwing` shows which.

- `ModeSwing(plot, name, host=None, loop=None)`: its items go on `plot` (`attach()` again after the page clears
  the plot), and its `FrameLoop` runs while `host` (a widget; the plot when it is one) shows: a mode shape swings
  on its own (CLAUDE.md's run state). Swings that show together share one `loop` (`SwingLoop`), which steps
  them all in a frame: a loop each painted the page once each (Modal coupling's two: 121 frames a second).
- `set_shape(x, psi, colour)`: the points' positions and complex amplitudes (the first a fixed end at 0 when
  `fixed_first`, drawn without a dot); one cycle takes `SECONDS_PER_CYCLE`.
- `nodes()`: where the shape crosses zero now (interpolated between points).

Used by Complex modes (the mode looked at, on its snapshots) and Modal coupling (the chain's two modes).

Working on it: `tests/test_complex_modes.py` checks a proportional mode's node stays put and a complex one's
travels; `test_gui_smoke.py`'s frame test runs its loop.
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore

from . import coalesce
from .frame_loop import SWING_MS, FrameLoop
from .style import colors

SECONDS_PER_CYCLE = 2.0


class ModeSwing(QtCore.QObject):
    def __init__(self, plot, name: str, host=None, loop: SwingLoop | None = None) -> None:
        super().__init__(host or plot)
        self.plot = plot
        self.theta = 0.0
        self.x = np.zeros(0)
        self.psi = np.zeros(0, complex)
        self.fixed_first = True
        self.line = pg.PlotDataItem()
        self.dots = pg.PlotDataItem(pen=None, symbol="o", symbolSize=10)
        self.node_marks = pg.PlotDataItem(pen=None, symbol="d", symbolSize=13)
        for item in (self.line, self.dots, self.node_marks):
            item.setZValue(10)
        self.colour = colors.mode[0]
        self.attach()
        self.apply_theme()
        if loop is None:
            self.loop = FrameLoop(host or plot, name, self._tick, self._runs, self._sim_t, frame_ms=SWING_MS)
        else:
            loop.swings.append(self)
            self.loop = loop.loop

    def attach(self) -> None:
        for item in (self.line, self.dots, self.node_marks):
            if item.scene() is None:
                self.plot.addItem(item)

    def apply_theme(self) -> None:
        self.line.setPen(pg.mkPen(self.colour, width=3))
        self.dots.setSymbolBrush(pg.mkBrush(self.colour))
        self.dots.setSymbolPen(pg.mkPen(colors.structure, width=1))
        self.node_marks.setSymbolBrush(pg.mkBrush(colors.force))
        self.node_marks.setSymbolPen(None)
        self._draw()

    def set_shape(self, x: np.ndarray, psi: np.ndarray, colour=None, fixed_first: bool = True) -> None:
        self.x = np.asarray(x, dtype=float)
        self.psi = np.asarray(psi, dtype=complex)
        self.fixed_first = fixed_first
        if colour is not None:
            self.colour = colour
        self.attach()
        self.apply_theme()
        self.loop.start()

    def shape_now(self) -> np.ndarray:
        return (self.psi * np.exp(1j * self.theta)).real

    def nodes(self) -> np.ndarray:
        """Where the shape crosses zero now, between points (a fixed end isn't one)."""
        y = self.shape_now()
        if not y.size or float(np.max(np.abs(y))) < 1e-9:
            return np.zeros(0)
        scale = float(np.max(np.abs(self.psi)))
        out = []
        tiny = 1e-9 * scale
        for i in range(y.size):
            if abs(y[i]) <= tiny:  # on the axis (a fixed end isn't a node)
                if not (self.fixed_first and i == 0):
                    out.append(self.x[i])
            elif i + 1 < y.size and abs(y[i + 1]) > tiny and y[i] * y[i + 1] < 0:
                out.append(self.x[i] + (self.x[i + 1] - self.x[i]) * y[i] / (y[i] - y[i + 1]))
        return np.array(out)

    @coalesce.coalesced(leaf=True)
    def _draw(self) -> None:
        if not self.x.size:
            for item in (self.line, self.dots, self.node_marks):
                item.setData([], [])
            return
        y = self.shape_now()
        self.line.setData(self.x, y)
        start = 1 if self.fixed_first else 0
        self.dots.setData(self.x[start:], y[start:])
        nodes = self.nodes()
        self.node_marks.setData(nodes, np.zeros_like(nodes))

    # ------------------------------------------------------------------ the loop
    def _runs(self) -> bool:
        return bool(self.x.size)

    def _sim_t(self) -> float | None:
        return None

    def _tick(self, dt: float) -> None:
        self.theta = (self.theta + 2 * math.pi * dt / SECONDS_PER_CYCLE) % (2 * math.pi)
        self._draw()


class SwingLoop(QtCore.QObject):
    """One frame loop for several `ModeSwing`s shown together (pass it as their `loop`): each frame steps them all,
    and the page paints once."""

    def __init__(self, host, name: str) -> None:
        super().__init__(host)
        self.swings: list[ModeSwing] = []
        self.loop = FrameLoop(host, name, self._tick, self._runs, self._sim_t, frame_ms=SWING_MS)

    def _runs(self) -> bool:
        return any(s._runs() for s in self.swings)

    def _sim_t(self) -> float | None:
        return None

    def _tick(self, dt: float) -> None:
        for s in self.swings:
            if s._runs():
                s._tick(dt)
