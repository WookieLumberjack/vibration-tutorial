"""The Qt parts #478's live pages share (the Single oscillator): drawn from a `LiveMass`'s record.

- `EnergyTab`: the energy balance over the last minute (put in, taken out, the live changes' terms, stored) and the
  ledger now, by `live_mass.energy_view`, its rows in their theme roles' colours.
- `LoopTab`: the damping laws' Loop & loss tab: the hysteresis loop the run draws over its last cycle beside each
  law's (`damping.loop_view`), and their loss factors against frequency (`damping.loss_view`).
- `ChangeMarks`: the dashed lines on the time plots where m, k or c changed, touched only when they change.
- `run_row`: the speed, Start/Pause, Reset and clock a live page gives the run bar (`transport.Transport`).

Working on it: `tests/test_sdof_page.py` checks them through its page.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.damping_models import MODELS
from ..view_models.damping import ROLES, SHORT, LoopView, LossView
from ..view_models.live_mass import LiveMass, energy_view
from ..view_models.text import fmt
from .layouts import FlowLayout, ResetButton
from .lesson import LiveLabel, RunButton, plot
from .plots import decimate
from .style import colors, role
from .theming import add_legend, mute
from .transport import speed_text

ENERGY_SECONDS = 60.0  # the Energy tab's window (its ledger is the whole run's)
SPEEDS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
ENERGY_TIP = (
    "<p>Where the energy went: what the drive and your hand put in, what the damping took out, what the changes "
    "cost or added (changing a spring at a stretch, capturing or dropping mass), and what's stored now. Put in, "
    "less taken out, with the changes, is what's stored, at every instant.</p>"
)


class EnergyTab(QtWidgets.QWidget):
    """The energy balance and its ledger (see the module)."""

    CURVES = (("put_in", "put in"), ("taken_out", "taken out"), ("changes", "the changes"), ("stored", "stored"))

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        self.plot = plot("Energy", "J", "Time", "s", height=160)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.setToolTip(ENERGY_TIP)
        add_legend(self.plot.getPlotItem(), offset=(-10, 10))
        self.items = {name: pg.PlotDataItem(name=label) for name, label in self.CURVES}
        for item in self.items.values():
            self.plot.addItem(item)
        decimate(self.plot)
        self.ledger = LiveLabel(wrap=True)  # rewritten each frame: a QLabel laid the page out again each time
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.ledger)
        self.apply_theme()

    def apply_theme(self) -> None:
        c = colors
        for name, colour in (
            ("put_in", c.force),
            ("taken_out", c.energy["dissipated"]),
            ("changes", c.energy["added"]),
            ("stored", c.energy["stored"]),
        ):
            self.items[name].setPen(pg.mkPen(colour, width=2))

    def draw(self, live: LiveMass) -> None:
        t, y = live.record.window(ENERGY_SECONDS)
        e = energy_view(t, y)
        if e is None:
            for item in self.items.values():
                item.setData([], [])
            self.ledger.setText("Start it, or pluck the mass, to see where the energy goes.")
            return
        for name, _ in self.CURVES:
            self.items[name].setData(e.t, getattr(e, name))
        rows = "".join(
            f'<tr><td><span style="color:{role(r.role)}">■</span> {r.name}</td>'
            f'<td align="right">&nbsp;{fmt(r.value, "J", smallest="n")}</td></tr>'
            for r in e.rows
            if r.value != 0 or r.name.startswith("Stored")
        )
        self.ledger.setText(f"<table>{rows}</table><p>{e.text}</p>")


LOOP_TIP = (
    "<p>The damper's force against the mass's position over the last swing. The damper pushes against the motion "
    "both ways, so going out and coming back trace different paths and close into a loop: the area inside is the "
    "energy the damper took out that swing. <b>This run</b> is the bold line; the faint ones are the loop each "
    "damper type would draw at the same swing (viscous an ellipse, friction a rectangle).</p>"
)
LOSS_TIP = (
    "<p>How much each damper type loses a swing, against how fast it swings: the loss factor η, the energy lost a "
    "cycle over 2π times the most stored. They're matched to lose the same at f<sub>n</sub> (η = 2ζ), so they "
    "cross there: viscous grows with frequency, hysteretic stays flat (friction, at the swing it's matched at, lies "
    "on it), and Zener peaks where you set it. The vertical line is the drive's frequency.</p>"
)


class LoopTab(QtWidgets.QWidget):
    """The Loop & loss tab (see the module); the page binds `damping_only` and draws it from its view model."""

    def __init__(self, tip: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        self.damping_only = QtWidgets.QCheckBox("Damping force alone (untick to add the spring's kx)")
        self.damping_only.setToolTip(tip)
        layout.addWidget(self.damping_only)
        self.loop_plot = plot("Force", "N", "Displacement x", "m", height=170)
        self.loop_plot.setMouseEnabled(x=False, y=False)
        self.loop_plot.setToolTip(LOOP_TIP)
        self.legend = add_legend(self.loop_plot.getPlotItem(), offset=(-10, 10))
        self.loop_item = pg.PlotDataItem(name="This run")
        self.theory_loops = {m: pg.PlotDataItem(name=SHORT[m]) for m in MODELS}
        self.spring_line = pg.PlotDataItem(name="Spring kx")
        for item in (*self.theory_loops.values(), self.spring_line, self.loop_item):
            self.loop_plot.addItem(item)
        self.legend.removeItem(self.spring_line)  # in the legend only while it's drawn
        self.note = LiveLabel(wrap=True)  # rewritten each frame, as the ledger
        mute(self.note)
        self.loss_plot = plot("Loss factor η", None, "Frequency / fₙ", None, log_x=True, log_y=True, height=120)
        self.loss_plot.setMouseEnabled(x=False, y=False)
        self.loss_plot.setToolTip(LOSS_TIP)
        add_legend(self.loss_plot.getPlotItem(), offset=(-10, 10))
        self.loss_items = {m: pg.PlotDataItem(name=SHORT[m]) for m in MODELS}
        for item in self.loss_items.values():
            self.loss_plot.addItem(item)
        self.loss_now = pg.InfiniteLine(angle=90)
        self.loss_plot.addItem(self.loss_now, ignoreBounds=True)
        self._spring_listed = False
        self._axis = "Force"
        layout.addWidget(self.loop_plot, 3)
        layout.addWidget(self.note)
        layout.addWidget(self.loss_plot, 2)
        self.apply_theme()

    def apply_theme(self) -> None:
        c = colors
        for m, item in self.theory_loops.items():
            item.setPen(pg.mkPen(role(ROLES[m]), width=1))
        for m, item in self.loss_items.items():
            item.setPen(pg.mkPen(role(ROLES[m]), width=2))
        self.loss_now.setPen(pg.mkPen(c.force, width=1))
        self.spring_line.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.loop_item.setPen(pg.mkPen(c.mass[0], width=2.5))

    def draw_loss(self, view: LossView) -> None:
        for m, curve in view.loss:
            self.loss_items[m].setData(*curve)
        self.loss_now.setVisible(view.now is not None)
        if view.now is not None:
            self.loss_now.setValue(np.log10(view.now))

    def draw_loop(self, view: LoopView) -> None:
        self.loop_item.setData(*view.live)
        drawn = dict(view.theory)
        for m, item in self.theory_loops.items():
            curve = drawn.get(m)
            item.setData(*(curve if curve is not None else ([], [])))
        spring = len(view.spring.x) > 0
        if spring != self._spring_listed:  # touched only when it changes: a legend change repaints the plot
            self._spring_listed = spring
            if spring:
                self.legend.addItem(self.spring_line, "Spring kx")
            else:
                self.legend.removeItem(self.spring_line)
        self.spring_line.setData(*view.spring)
        if view.axis != self._axis:
            self._axis = view.axis
            self.loop_plot.setLabel("left", view.axis, units="N")
        self.note.setText(view.note)


class ChangeMarks:
    """Dashed lines at the changes' times on `plots`, within the window shown."""

    def __init__(self, *plots: pg.PlotWidget) -> None:
        self.plots = plots
        self.lines: list[pg.InfiniteLine] = []
        self.marks: list[float] = []

    def draw(self, edits: list[float], start: float) -> None:
        marks = [t for t in edits if t > start]
        if marks == self.marks:
            return
        self.marks = marks
        n = len(self.plots)
        while len(self.lines) < n * len(marks):
            line = pg.InfiniteLine(angle=90, pen=self._pen())
            self.plots[len(self.lines) % n].addItem(line, ignoreBounds=True)
            self.lines.append(line)
        for i, line in enumerate(self.lines):
            on = i // n < len(marks)
            line.setVisible(on)
            if on:
                line.setValue(marks[i // n])

    @staticmethod
    def _pen():
        return pg.mkPen(colors.faint, width=1, style=QtCore.Qt.PenStyle.DashLine)

    def apply_theme(self) -> None:
        for line in self.lines:
            line.setPen(self._pen())


def run_row(page: QtWidgets.QWidget) -> tuple[FlowLayout, QtWidgets.QComboBox, RunButton, ResetButton, LiveLabel]:
    """The run's controls: (the row, speed, Start/Pause, Reset, clock). The page connects them."""
    speed = QtWidgets.QComboBox(page)
    speed.setAccessibleName("Playback speed")
    for s in SPEEDS:
        speed.addItem(speed_text(s), s)
    speed.setCurrentIndex(SPEEDS.index(1.0))
    speed.setToolTip("<p>Simulated seconds per second.</p>")
    run = RunButton()
    reset = ResetButton()
    clock = LiveLabel("t = 0 s")
    row = FlowLayout()
    row.add(speed, run, reset, clock)
    return row, speed, run, reset, clock
