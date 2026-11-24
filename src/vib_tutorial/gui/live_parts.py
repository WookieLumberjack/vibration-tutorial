"""The Qt parts #478's live pages share (the Single oscillator, Damping models): drawn from a `LiveMass`'s record.

- `EnergyTab`: the energy balance over the last minute (put in, taken out, the live changes' terms, stored) and the
  ledger now, by `live_mass.energy_view`, its rows in their theme roles' colours.
- `ChangeMarks`: the dashed lines on the time plots where m, k or c changed, touched only when they change.
- `run_row`: the speed, Start/Pause, Reset and clock a live page gives the run bar (`transport.Transport`).

Working on it: `tests/test_sdof_page.py` and `tests/test_damping_page.py` check them through their pages.
"""

from __future__ import annotations

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..view_models.live_mass import LiveMass, energy_view
from ..view_models.text import fmt
from .layouts import FlowLayout, ResetButton
from .lesson import LiveLabel, RunButton, plot
from .plots import decimate
from .style import colors, role
from .theming import add_legend
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
        self.ledger = QtWidgets.QLabel()
        self.ledger.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.ledger.setWordWrap(True)
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
