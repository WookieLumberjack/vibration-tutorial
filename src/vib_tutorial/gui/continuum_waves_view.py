"""The Continuous systems page's Waves tab: a pulse running along the continuum and along the mesh that
approximates it, with a dashpot on the far end or a step in the section to reflect it. Its controls
(`controls`, the Waves pane in the page's left column) and the tab (this widget).

The page gives it each medium and mesh (`set_medium`), which starts the pulse afresh: at once if the tab shows,
else when it next does (`showEvent`), so a change elsewhere builds no run nobody sees. `load` fills the
controls (a preset's pulse, or the same pulse on a new kind of medium), `play` starts it (a wave preset), and
its `FrameLoop` runs only while the tab shows.

Working on it: `test_gui_smoke.py`'s Continuum tests run the pulse through it (`page.waves_view`).
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.continuum import PRESETS, SCHEMES, Continuum, Kind, Scheme
from ..core.continuum_waves import PLAYBACK, WaveRun, Waves, end_reflection, step_coefficients
from .frame_loop import FrameLoop
from .layouts import FlowLayout, form_layout, form_row, group, plot_height
from .lesson import RunButton, plot
from .panels import spin
from .style import colors
from .theming import add_legend, mute

NODES_DRAWN = 80  # the mesh's nodes are dotted up to this many elements
SHAPE_POINTS = 801
WAVES_TIP = (
    "<p>The pulse on the continuum (black) and on the mesh: its nodes, and the elements' own shape functions "
    "between them. The continuum's is exact: d'Alembert's two waves, carried along and scattered at the ends "
    "and the step in closed form (a string or bar), or the sum of its exact modes (a beam).</p>"
)
TIPS = {
    "pulse": "<p>The pulse: a raised cosine, its full width and its centre as fractions of the length. The fewer "
    "elements across it, the more of it is short waves the mesh gets wrong.</p>",
    "moving": "<p>Start the pulse travelling right (d'Alembert's right-going wave alone), or at rest, when it "
    "splits into two halves going opposite ways. A beam's starts at rest.</p>",
    "wave_scheme": "<p>How the mesh the pulse runs on shares out its mass.</p>",
    "dashpot": "<p>Put a dashpot on the far end in place of the medium's own end, as a multiple of the "
    "impedance Z = √(Tμ) (string) or ρcA (bar). At 1 it is matched: a wave runs into it and is "
    "gone, as if the medium went on for ever. Reflected: (1 − r)/(1 + r) of the wave.</p>",
    "step": "<p>A step in the medium at this point: past it the impedance is Z₂/Z₁ times as much (a string "
    "that much heavier squared, so its waves go Z₁/Z₂ as fast; a bar of that much more section, at the "
    "same speed). A wave reflects (Z₁ − Z₂)/(Z₁ + Z₂) of itself and passes 2Z₁/(Z₁ + Z₂). On the "
    "mesh, at the nearest node.</p>",
    "playback": "<p>How fast the pulse plays: at 1× it crosses the length in about four seconds.</p>",
}


class WavesView(QtWidgets.QWidget):
    """The Waves tab, its controls (`controls`) and its run."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.continuum: Continuum = PRESETS[0].continuum
        self.elements = PRESETS[0].elements
        self._kind = self.continuum.kind

        self.controls = QtWidgets.QGroupBox("Waves")
        wf = form_layout(self.controls)
        self.pulse_width = spin(1.0, 100.0, 12.5, 1, " %")
        self.pulse_at = spin(0.0, 100.0, 25.0, 1, " %")
        form_row(wf, "Pulse:", group(self.pulse_width, "wide at", self.pulse_at), TIPS["pulse"])
        self.moving = QtWidgets.QCheckBox("Travelling right")
        self.moving.setChecked(True)
        self.moving.setToolTip(TIPS["moving"])
        wf.addRow(self.moving)
        self.wave_scheme = QtWidgets.QComboBox()
        self.wave_scheme.addItems([sc.value for sc in SCHEMES])
        form_row(wf, "Mesh's mass:", self.wave_scheme, TIPS["wave_scheme"])
        self.dashpot_on = QtWidgets.QCheckBox("Dashpot on the far end")
        self.dashpot_on.setToolTip(TIPS["dashpot"])
        self.dashpot = spin(0.0, 100.0, 1.0, 2, " Z")
        wf.addRow(group(self.dashpot_on, self.dashpot))
        self.dashpot.setToolTip(TIPS["dashpot"])
        self.step_on = QtWidgets.QCheckBox("Step at")
        self.step_on.setToolTip(TIPS["step"])
        self.step_at = spin(1.0, 99.0, 50.0, 1, " %")
        self.step_ratio = spin(0.05, 20.0, 2.0, 2, "")
        self.step_ratio.setPrefix("Z₂/Z₁ ")
        for box in (self.step_at, self.step_ratio):
            box.setToolTip(TIPS["step"])
        wf.addRow(group(self.step_on, self.step_at, self.step_ratio))
        self.wave_info = QtWidgets.QLabel()
        self.wave_info.setWordWrap(True)
        mute(self.wave_info)
        wf.addRow(self.wave_info)

        # the pulse on the continuum and on the mesh
        self.waves_plot = plot("Displacement", None, "Position x", "m")
        self.waves_plot.setMouseEnabled(x=False, y=False)
        self.waves_plot.setToolTip(WAVES_TIP)
        plot_height(self.waves_plot, 240)
        self.waves_legend = add_legend(self.waves_plot.getPlotItem(), offset=(-10, 10))
        self.waves_zero = pg.InfiniteLine(0.0, angle=0)
        self.waves_plot.addItem(self.waves_zero)
        self.step_region = pg.LinearRegionItem(movable=False)
        self.step_region.setZValue(-10)
        self.waves_plot.addItem(self.step_region)
        self.end_line = pg.InfiniteLine(0.0, angle=90, label="dashpot", labelOpts={"position": 0.9})
        self.waves_plot.addItem(self.end_line)
        self.exact_curve = self.waves_plot.plot([], [])
        self.mesh_curve = self.waves_plot.plot([], [])
        self.mesh_nodes = self.waves_plot.plot([], [], pen=None, symbol="o", symbolSize=6)
        self.waves_legend.addItem(self.exact_curve, "Continuum")
        self.waves_legend.addItem(self.mesh_curve, "Mesh")
        self.apply_theme()
        self.play_button = RunButton()
        self.restart_button = QtWidgets.QPushButton("Reset")
        self.restart_button.setToolTip("<p>The pulse back where it began, waiting for Start.</p>")
        self.playback = QtWidgets.QComboBox()
        for x in PLAYBACK:
            self.playback.addItem(f"{x:g}×", x)
        self.playback.setCurrentIndex(PLAYBACK.index(1.0))
        self.playback.setToolTip(TIPS["playback"])
        self.wave_status = QtWidgets.QLabel()
        self.wave_status.setWordWrap(True)
        self.wave_status.setTextFormat(QtCore.Qt.TextFormat.RichText)
        wv = QtWidgets.QVBoxLayout(self)
        wv.addWidget(self.waves_plot, 1)
        row = FlowLayout()
        row.add(self.play_button, self.restart_button, group("Speed:", self.playback))
        wv.addLayout(row)
        wv.addWidget(self.wave_status)

        self.run: WaveRun | None = None  # built when the tab first shows after a change
        self.running = False  # the pulse waits for Start
        self._reach = 1.0  # the plot's half height: the largest displacement seen yet

        for box in (self.pulse_width, self.pulse_at, self.dashpot, self.step_at, self.step_ratio):
            box.valueChanged.connect(self._on_waves)
        for check in (self.moving, self.dashpot_on, self.step_on):
            check.toggled.connect(self._on_waves)
        self.wave_scheme.currentIndexChanged.connect(self._on_waves)
        self.play_button.toggled.connect(self._on_play)
        self.restart_button.clicked.connect(self.reset)
        self.playback.currentIndexChanged.connect(self._on_playback)
        self.loop = FrameLoop(self, "Continuous systems", self._tick, self._runs, self._sim_t)

    def set_medium(self, c: Continuum, elements: int) -> None:
        """A new medium or mesh: the pulse starts afresh on it."""
        self.continuum, self.elements = c, elements
        self.waves_plot.setLabel("left", "Displacement (along the bar)" if c.kind is Kind.BAR else "Deflection")
        self._new_run()

    def play(self) -> None:
        """Start the pulse (a wave preset), from where it is."""
        self.running = True
        self.play_button.set_running(True)
        self.loop.start()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self.run is None:  # changed while the tab was hidden
            self.restart()

    def load(self, waves: Waves, kind: Kind) -> None:
        """Fill the wave controls (signals held), a beam's without the dashpot, the step or a moving start."""
        self._kind = kind
        beam = kind is Kind.BEAM
        boxes = (
            self.pulse_width,
            self.pulse_at,
            self.moving,
            self.wave_scheme,
            self.dashpot_on,
            self.dashpot,
            self.step_on,
            self.step_at,
            self.step_ratio,
        )
        for box in boxes:
            box.blockSignals(True)
        self.pulse_width.setValue(100 * waves.width)
        self.pulse_at.setValue(100 * waves.at)
        self.moving.setChecked(waves.moving and not beam)
        self.wave_scheme.setCurrentText(waves.scheme.value)
        self.dashpot_on.setChecked(waves.dashpot is not None and not beam)
        if waves.dashpot is not None:
            self.dashpot.setValue(waves.dashpot)
        self.step_on.setChecked(waves.step_at is not None and not beam)
        if waves.step_at is not None:
            self.step_at.setValue(100 * waves.step_at)
        self.step_ratio.setValue(waves.step_ratio)
        for box in boxes:
            box.blockSignals(False)
        for box in (self.moving, self.dashpot_on, self.step_on):
            box.setEnabled(not beam)
        self._enable_waves()

    def _enable_waves(self) -> None:
        self.dashpot.setEnabled(self.dashpot_on.isChecked() and self.dashpot_on.isEnabled())
        for box in (self.step_at, self.step_ratio):
            box.setEnabled(self.step_on.isChecked() and self.step_on.isEnabled())

    def waves(self, kind: Kind | None = None) -> Waves:
        """The pulse the controls set up, for the medium's kind (`kind`: a new one, before it's meshed)."""
        beam = (kind or self._kind) is Kind.BEAM
        return Waves(
            width=self.pulse_width.value() / 100,
            at=self.pulse_at.value() / 100,
            moving=self.moving.isChecked() and not beam,
            scheme=Scheme(self.wave_scheme.currentText()),
            dashpot=self.dashpot.value() if self.dashpot_on.isChecked() and not beam else None,
            step_at=self.step_at.value() / 100 if self.step_on.isChecked() and not beam else None,
            step_ratio=self.step_ratio.value(),
        )

    def _on_waves(self, *_args) -> None:
        self._enable_waves()
        self._new_run()

    def _new_run(self) -> None:
        """Start the pulse afresh: now if the Waves tab shows, else when it next does (`showEvent`)."""
        self.run = None
        self._show_info()
        if self.isVisible():
            self.restart()

    def restart(self) -> None:
        self.run = WaveRun(self.continuum, self.elements, self.waves(), float(self.playback.currentData()))
        self._reach = 1.0
        c = self.continuum
        self.waves_plot.setXRange(0.0, c.length, padding=0.02)
        self.waves_plot.setYRange(-1.15, 1.15, padding=0.0)
        step = self.run.step_x
        self.step_region.setVisible(step is not None)
        if step is not None:
            self.step_region.setRegion((step, c.length))
        self.end_line.setVisible(self.run.waves.dashpot is not None)
        self.end_line.setPos(c.length)
        self._draw_waves()
        self._update_play()
        self.loop.start()

    def _show_info(self) -> None:
        c, w = self.continuum, self.waves()
        h = c.length / self.elements
        parts = [f"{w.width * c.length / h:.3g} elements across the pulse."]
        if c.kind is not Kind.BEAM:
            z = math.sqrt(c.rigidity * c.mass_per_length)
            parts.append(f"Z = {z:.4g} N·s/m.")
            if w.step_at is not None:
                r, t = step_coefficients(w.step_ratio)
                parts.append(
                    f"The step reflects {r:+.3g} and passes {t:.3g} of a wave from the left "
                    f"(energy {r * r:.3g} : {t * t * w.step_ratio:.3g})."
                )
            if w.dashpot is not None:
                parts.append(
                    f"The dashpot ({w.dashpot:.3g} Z = {w.dashpot * z:.4g} N·s/m) reflects "
                    f"{end_reflection(c.right, w.dashpot):+.3g} of a wave."
                )
        self.wave_info.setText(" ".join(parts))

    def reset(self) -> None:
        """The pulse back where it began, waiting for Start."""
        self.running = False
        self.play_button.rest()
        self.restart()

    def _on_play(self, on: bool) -> None:
        self.running = on
        if self.running:
            self.loop.start()
        else:
            self.loop.stop()

    def _update_play(self) -> None:
        self.play_button.set_running(self.running)

    def _on_playback(self, _index: int) -> None:
        if self.run is not None:
            self.run.set_playback(float(self.playback.currentData()))

    def _runs(self) -> bool:
        """Playing (the loop runs only while the Waves tab shows: the other tabs don't move)."""
        return self.running and self.run is not None

    def _tick(self, _dt: float) -> None:
        self.step()  # a frame's worth of the pulse, at its own playback rate

    def _sim_t(self) -> float | None:
        return self.run.t if self.run is not None else None

    def step(self, frames: int = 1) -> None:
        """Run the pulse on by `frames` frames and redraw."""
        if self.run is None:
            self.restart()
        assert self.run is not None, "restart made it"
        self.run.advance(frames)
        self._draw_waves()

    def _draw_waves(self) -> None:
        run = self.run
        assert run is not None, "drawn once restart has made the run"
        x, u = run.exact()
        self.exact_curve.setData(x, u)
        dense = np.linspace(0.0, self.continuum.length, SHAPE_POINTS)
        shape = run.mesh_shape()
        field = run.model.field(shape, dense)
        self.mesh_curve.setData(dense, field)
        nodes, w = run.mesh()
        if self.elements <= NODES_DRAWN:
            self.mesh_nodes.setData(nodes, w)
        else:
            self.mesh_nodes.setData([], [])
        reach = max(float(np.max(np.abs(u))), float(np.max(np.abs(field))))
        if reach > self._reach:  # grow, never shrink, till the pulse starts again
            self._reach = reach
            self.waves_plot.setYRange(-1.15 * self._reach, 1.15 * self._reach, padding=0.0)
        crossings = run.t / run.transit
        self.wave_status.setText(
            f"t = {crossings:.2f} crossings ({_fmt_time(run.t)}; one crossing {_fmt_time(run.transit)}). "
            f"The mesh ({run.waves.scheme.value.lower()}) keeps <b>{100 * run.energy_left:.3g} %</b> of its "
            "energy."
        )

    def apply_theme(self) -> None:
        self.waves_zero.setPen(pg.mkPen(colors.faint, width=1))
        self.exact_curve.setPen(pg.mkPen(colors.strong, width=2))
        mesh = colors.mode[0]
        self.mesh_curve.setPen(pg.mkPen(mesh, width=1.5))
        self.mesh_nodes.setSymbolBrush(mesh)
        self.mesh_nodes.setSymbolPen(pg.mkPen(colors.background))
        shade = pg.mkColor(colors.faint)
        shade.setAlpha(70)
        self.step_region.setBrush(pg.mkBrush(shade))
        for line in self.step_region.lines:
            line.setPen(pg.mkPen(colors.muted, width=1))
        self.end_line.setPen(pg.mkPen(colors.force, width=3))
        self.end_line.label.setColor(colors.force)


def _fmt_time(t: float) -> str:
    return f"{t * 1e3:.3g} ms" if t < 1.0 else f"{t:.3g} s"
