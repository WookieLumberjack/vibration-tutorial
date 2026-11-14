"""The virtual modal test's two plots: the latest recorded block and the force's spectrum (`SignalView`), and the
estimated FRF against the exact one with its coherence (`FrfView`). The page (`modal_test.py`) feeds them each
estimate; they hold no measurement of their own.

Working on it: `test_gui_smoke.py`'s modal test tests read them through `page.signals` and `page.frf_view`.
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core import (
    ChainSystem,
    Estimate,
    Excitation,
    MeasurementSettings,
    ModalResult,
    Processing,
    Response,
    Window,
    frf,
    friction_frf,
    from_receptance,
    impact_spectrum,
    transfer,
)
from ..core.measurement import SINE_AMPLITUDE
from .axes import log_axes
from .style import colors
from .theming import add_legend


def phase_deg(h: np.ndarray) -> np.ndarray:
    """Phase in degrees, wrapped to (-190, 170] so a phase near -180 does not flicker to +180."""
    return (np.degrees(np.angle(h)) + 190.0) % 360.0 - 190.0


class SignalView(pg.GraphicsLayoutWidget):
    """The latest block: force and one response against time (with the windows), and the force spectrum."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.force = self.addPlot(row=0, col=0)
        self.force.setLabel("left", "Force", units="N")
        self.spectrum = self.addPlot(row=0, col=1, rowspan=2)
        self.spectrum.setLabel("left", "Force spectrum", units="dB")
        self.spectrum.setLabel("bottom", "Frequency", units="Hz")
        self.spectrum.showGrid(x=True, y=True, alpha=0.3)
        self.response = self.addPlot(row=1, col=0)
        self.response.setLabel("bottom", "Time in block", units="s")
        self.response.setXLink(self.force)
        self.zero_lines = [pg.InfiniteLine(pos=0, angle=0) for _ in range(2)]
        for p, line in zip((self.force, self.response), self.zero_lines):
            p.addItem(line)
        self.ci.layout.setColumnStretchFactor(0, 3)
        self.ci.layout.setColumnStretchFactor(1, 2)
        self.f_curve = self.force.plot()
        self.f_window = self.force.plot()
        self.f_fit = self.force.plot()
        self.x_curve = self.response.plot()
        self.x_window = self.response.plot()
        self.x_fit = self.response.plot()
        self.s_curve = self.spectrum.plot()
        self.s_tip = self.spectrum.plot()
        self.band = pg.LinearRegionItem(movable=False, pen=pg.mkPen(None))
        self.spectrum.addItem(self.band)
        self.apply_theme()

    def apply_theme(self) -> None:
        """Every curve but the response and the measured force, which set_data styles."""
        dash = QtCore.Qt.PenStyle.DashLine
        for line in self.zero_lines:
            line.setPen(pg.mkPen(colors.faint, width=1))
        self.f_fit.setPen(pg.mkPen(colors.force, width=1))
        self.s_curve.setPen(pg.mkPen(colors.force, width=1))
        for curve in (self.f_window, self.x_window):
            curve.setPen(pg.mkPen(colors.grey, width=1, style=dash))
        self.s_tip.setPen(pg.mkPen(colors.strong, width=1, style=dash))
        self.band.setBrush(pg.mkBrush(*colors.band_shade))

    def clear_measurement(self) -> None:
        for curve in (self.f_curve, self.f_window, self.f_fit, self.x_curve, self.x_window, self.x_fit, self.s_curve):
            curve.clear()

    def set_data(self, est: Estimate, j: int, settings: MeasurementSettings, stepped: bool) -> None:
        self.set_time(est, j, stepped)
        self.set_spectrum(est, settings, stepped)

    def set_time(self, est: Estimate, j: int, stepped: bool, shown: float | None = None) -> None:
        """The latest record against time; while it is being recorded, only its first `shown` s."""
        n = est.t.size if shown is None else int(np.searchsorted(est.t, shown, side="right"))
        # Block excitations: the samples joined by lines. Stepped sine: the samples as dots
        # (as few as 2.5 per cycle at the band edge) and the fitted sine through them.
        for plot, curve, fit, y, y_fit, color in (
            (self.force, self.f_curve, self.f_fit, est.f, None if est.fit is None else est.fit[1], colors.force),
            (
                self.response,
                self.x_curve,
                self.x_fit,
                est.x[:, j],
                None if est.fit is None else est.fit[2][:, j],
                colors.mass[j],
            ),
        ):
            if y_fit is None:
                curve.setData(est.t[:n], y[:n], pen=pg.mkPen(color, width=1), symbol=None)
                fit.setData([], [])
            else:
                curve.setData(est.t[:n], y[:n], pen=None, symbol="o", symbolSize=4, symbolPen=None, symbolBrush=color)
                assert est.fit is not None, "a fitted curve comes from the fit"
                m = est.fit[0].size if shown is None else int(np.searchsorted(est.fit[0], shown, side="right"))
                fit.setData(est.fit[0][:m], y_fit[:m])
            # Time spans the record. (Auto-ranging it padded by the widget's size when the range was
            # worked out, which could be before the layout settled.)
            plot.setXRange(0.0, float(est.t[-1]), padding=0.02)
            if shown is None:
                plot.enableAutoRange(axis="y")
            else:  # the whole record's range, so the axes hold still while it is drawn
                plot.setYRange(float(np.min(y)), float(np.max(y)))
        self.x_fit.setPen(pg.mkPen(colors.mass[j], width=1))
        self.response.setLabel("left", f"{est.response.symbol}{j + 1}", units=est.response.unit)
        for curve, window, signal in (
            (self.f_window, est.force_window, est.f),
            (self.x_window, est.response_window, est.x[:, j]),
        ):
            if window is None:
                curve.setData([], [])
            else:
                curve.setData(est.t, window * float(np.abs(signal).max() or 1.0))
        what = (f"{est.freqs[-1]:.3g} Hz" if stepped else f"Block {est.count}") + (
            " (recording…)" if shown is not None else ""
        )
        self.force.setTitle(
            what
            + (
                ": dots as measured · line: fitted sine"
                if stepped
                else ": signals as measured · dashed: window (scaled)"
            ),
            size="9pt",
        )

    def set_spectrum(self, est: Estimate, settings: MeasurementSettings, stepped: bool) -> None:
        spec = est.force_spectrum
        with np.errstate(divide="ignore"):
            db = 10.0 * np.log10(spec / spec.max()) if spec.size and spec.max() > 0 else spec
        self.s_curve.setData(
            est.freqs,
            db,
            connect="finite",
            symbol="o" if stepped else None,
            symbolSize=4,
            symbolPen=None,
            symbolBrush=colors.force,
        )
        nyq = settings.fs / 2
        if settings.excitation is Excitation.IMPACT:
            f = np.linspace(0.0, nyq, 400)
            with np.errstate(divide="ignore"):
                self.s_tip.setData(f, 20.0 * np.log10(impact_spectrum(settings.tip_width, f)), connect="finite")
        else:
            self.s_tip.setData([], [])
        self.band.setRegion((settings.band, nyq))
        self.spectrum.setXRange(0.0, nyq, padding=0)
        self.spectrum.setYRange(-60.0, 5.0, padding=0)
        self.spectrum.setTitle(
            "Averaged |F|² (dB re peak)"
            + (" · dashed: ideal hammer pulse" if settings.excitation is Excitation.IMPACT else ""),
            size="9pt",
        )


class FrfView(pg.GraphicsLayoutWidget):
    """Estimated receptance of one response against the exact one: magnitude, phase, coherence.

    A draggable band on the magnitude plot sets the frequencies the modal fit uses.
    """

    fit_band_changed = QtCore.Signal(float, float)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.mag = self.addPlot(row=0, col=0, axisItems=log_axes())
        self.mag.setLogMode(x=False, y=True)
        self.mag.setLabel("left", "|H|  [m/N]")
        add_legend(self.mag, offset=(-5, 5))
        self.phase = self.addPlot(row=1, col=0)
        self.phase.setLabel("left", "Phase", units="deg")
        self.phase.getAxis("left").setTickSpacing(90.0, 45.0)
        self.coh = self.addPlot(row=2, col=0)
        self.coh.setLabel("left", "Coherence γ²")
        self.coh.setLabel("bottom", "Frequency", units="Hz")
        self.coh.setYRange(0.0, 1.05, padding=0)
        for p in (self.mag, self.phase, self.coh):
            p.showGrid(x=True, y=True, alpha=0.3)
            if p is not self.mag:
                p.setXLink(self.mag)
        self.phase.setYRange(-195.0, 175.0, padding=0)
        self.ci.layout.setRowStretchFactor(0, 3)
        self.ci.layout.setRowStretchFactor(1, 2)
        self.ci.layout.setRowStretchFactor(2, 1)
        self._key: tuple | None = None
        self.measured: list[pg.PlotDataItem] = []  # magnitude, phase, coherence
        self.fit_curves: list[pg.PlotDataItem] = []  # magnitude, phase
        self.held_curves: list[pg.PlotDataItem] = []  # magnitude, phase
        self.fit_region = pg.LinearRegionItem()
        self.fit_region.setZValue(-5)
        self.fit_region.sigRegionChangeFinished.connect(self._on_region)
        self.apply_theme()

    def apply_theme(self) -> None:
        """The fit band now; the curves are drawn again with the next data."""
        shade = pg.mkColor(colors.fit)
        shade.setAlpha(26 if colors.dark else 28)
        self.fit_region.setBrush(pg.mkBrush(shade))
        for line in self.fit_region.lines:
            line.setPen(pg.mkPen(colors.fit, width=1))
        self._key = None

    def _on_region(self) -> None:
        lo, hi = self.fit_region.getRegion()
        self.fit_band_changed.emit(max(0.0, float(lo)), float(hi))

    def set_fit_band(self, lo: float, hi: float) -> None:
        self.fit_region.blockSignals(True)
        self.fit_region.setRegion((lo, hi))
        self.fit_region.blockSignals(False)

    def clear_measurement(self) -> None:
        """Hide the measurement (a new one has started); the exact curves stay."""
        for curve in self.measured:
            curve.clear()

    def set_fit(self, freqs: np.ndarray | None, h: np.ndarray | None) -> None:
        """The FRF rebuilt from the identified modes (None: hide it)."""
        if not self.fit_curves:
            return
        mag_curve, phase_curve = self.fit_curves
        legend = self.mag.legend
        shown = legend.getLabel(mag_curve) is not None
        if h is None:
            mag_curve.setData([], [])
            phase_curve.setData([], [])
            if shown:
                legend.removeItem(mag_curve)
            return
        with np.errstate(invalid="ignore"):
            mag_curve.setData(freqs, np.where(np.abs(h) > 0, np.abs(h), np.nan), connect="finite")
        phase_curve.setData(freqs, phase_deg(h), connect="finite")
        if not shown:
            legend.addItem(mag_curve, "Fitted modal model")

    def set_held(self, freqs: np.ndarray | None, h: np.ndarray | None, label: str = "") -> None:
        """An FRF held for comparison (None: hide it)."""
        if not self.held_curves:
            return
        mag_curve, phase_curve = self.held_curves
        legend = self.mag.legend
        if legend.getLabel(mag_curve) is not None:
            legend.removeItem(mag_curve)
        if h is None or freqs is None:
            mag_curve.setData([], [])
            phase_curve.setData([], [])
            return
        keep = freqs > 0
        with np.errstate(invalid="ignore"):
            mag_curve.setData(freqs[keep], np.where(np.abs(h[keep]) > 0, np.abs(h[keep]), np.nan), connect="finite")
        phase_curve.setData(freqs[keep], phase_deg(h[keep]), connect="finite")
        legend.addItem(mag_curve, label)

    def set_data(
        self,
        est: Estimate,
        j: int,
        system: ChainSystem,
        result: ModalResult,
        settings: MeasurementSettings,
        processing: Processing,
        stepped: bool,
    ) -> None:
        # The exact curves, mode lines and bands change only with the test; while it runs,
        # only the measured curves are updated.
        key = (
            id(system),
            id(result),
            settings.fs,
            settings.input_dof,
            settings.response,
            j,
            stepped,
            processing.window,
            processing.exp_end,
            processing.estimator,
            settings.force_level,
        )
        if key != self._key:
            self._key = key
            self._draw_reference(j, system, result, settings, processing, stepped)
        keep = est.freqs > 0
        fm, Hm = est.freqs[keep], est.H[keep, j]
        with np.errstate(invalid="ignore"):
            mag = np.where(np.abs(Hm) > 0, np.abs(Hm), np.nan)
        mag_curve, phase_curve, coh_curve = self.measured
        mag_curve.setData(fm, mag, connect="finite")
        phase_curve.setData(fm, phase_deg(Hm), connect="finite")
        if est.coherence is not None:
            coh_curve.setData(fm, est.coherence[keep, j], connect="finite")
        noun = "frequencies" if stepped else "averages"
        self.mag.legend.getLabel(mag_curve).setText(f"Measured {processing.estimator.name} ({est.count} {noun})")

    def _draw_reference(
        self,
        j: int,
        system: ChainSystem,
        result: ModalResult,
        settings: MeasurementSettings,
        processing: Processing,
        stepped: bool,
    ) -> None:
        for p in (self.mag, self.phase, self.coh):
            p.clear()
        self.mag.legend.clear()
        nyq = settings.fs / 2
        f = np.linspace(nyq / 2000, nyq, 2000)
        peaks = [m.damped.fd_hz for m in result.modes if m.damped is not None and m.damped.fd_hz < nyq]
        f = np.unique(np.concatenate([f, peaks]))
        response = settings.response
        exact = from_receptance(f, frf(system, f, settings.input_dof)[:, j], response)
        dash = QtCore.Qt.PenStyle.DashLine
        exact_name = "Exact, without friction" if np.any(system.friction > 0) else "Exact"
        self.mag.plot(f, np.abs(exact), pen=pg.mkPen(colors.strong, width=1.5), name=exact_name)
        self.phase.plot(f, phase_deg(exact), pen=pg.mkPen(colors.strong, width=1.5))
        if processing.window is Window.FORCE_EXPONENTIAL and processing.exp_end < 1.0 and not stepped:
            s = 1j * 2 * np.pi * f + 1.0 / processing.exp_tau(settings)
            damped = s**response.power * transfer(system, s, settings.input_dof)[:, j]
            pen = pg.mkPen(colors.grey, width=1, style=dash)
            self.mag.plot(f, np.abs(damped), pen=pen, name="Exact + window damping")
            self.phase.plot(f, phase_deg(damped), pen=pen)
        if stepped and np.any(system.friction > 0):
            # A stepped sine reads the first harmonic at each frequency, which is what the
            # describing function approximates. A mass held by friction leaves a gap.
            amplitude = settings.force_level * SINE_AMPLITUDE
            fd = np.unique(np.concatenate([np.linspace(nyq / 600, nyq, 600), peaks]))
            rf = friction_frf(system, fd, settings.input_dof, amplitude)
            h = rf.X[:, j] / amplitude
            h[rf.stuck[:, j]] = np.nan
            described = from_receptance(fd, h, response)
            pen = pg.mkPen(colors.strong, width=1.5, style=QtCore.Qt.PenStyle.DashDotLine)
            self.mag.plot(
                fd, np.abs(described), pen=pen, connect="finite", name=f"Describing function at {amplitude:.3g} N"
            )
            self.phase.plot(fd, phase_deg(described), pen=pen, connect="finite")

        held_pen = pg.mkPen(colors.grey, width=1.5)
        self.held_curves = [self.mag.plot(pen=held_pen), self.phase.plot(pen=held_pen)]
        color = colors.mass[j]
        symbol = dict(symbol="o", symbolSize=5, symbolPen=None, symbolBrush=color) if stepped else {}
        self.measured = [
            self.mag.plot(pen=pg.mkPen(color, width=1.5), name="Measured", **symbol),
            self.phase.plot(pen=pg.mkPen(color, width=1), **symbol),
            self.coh.plot(pen=pg.mkPen(color, width=1)),
        ]
        fit_pen = pg.mkPen(colors.fit, width=2, style=QtCore.Qt.PenStyle.DashLine)
        self.fit_curves = [self.mag.plot(pen=fit_pen), self.phase.plot(pen=fit_pen)]
        self.mag.addItem(self.fit_region)
        if stepped:
            label = pg.TextItem("A stepped sine has no coherence (one reading per frequency)", color=colors.muted)
            label.setPos(0.02 * nyq, 0.6)
            self.coh.addItem(label)

        for r, m in enumerate(result.modes):
            if m.damped is None or m.damped.fd_hz >= nyq:
                continue
            pen = pg.mkPen(colors.mode[r % len(colors.mode)], width=1, style=QtCore.Qt.PenStyle.DotLine)
            for p in (self.mag, self.phase, self.coh):
                p.addItem(pg.InfiniteLine(pos=m.damped.fd_hz, angle=90, pen=pen))
        for p in (self.mag, self.phase, self.coh):
            band = pg.LinearRegionItem(
                (settings.band, nyq), movable=False, brush=pg.mkBrush(*colors.band_shade), pen=pg.mkPen(None)
            )
            band.setZValue(-10)
            p.addItem(band)
        # An accelerance falls as ω² towards 0 Hz: range it on the band, not on the first lines.
        shown = np.isfinite(exact) & ((f >= 0.02 * nyq) if response is Response.ACCELERATION else True)
        finite = np.abs(exact[shown])
        if finite.size:
            lo, hi = math.log10(finite.min()), math.log10(finite.max())
            self.mag.setYRange(lo - 1.0, hi + 0.5, padding=0)
        self.mag.setXRange(0.0, nyq, padding=0)
        drive = "driving-point" if j == settings.input_dof else "transfer"
        name, unit = ("H", "m/N") if response is Response.DISPLACEMENT else ("A", "(m/s²)/N")
        self.mag.setLabel("left", f"|{name}|  [{unit}]")
        self.mag.setTitle(
            f"{name}<sub>{j + 1}{settings.input_dof + 1}</sub> = {response.symbol}{j + 1} / F "
            f"at m{settings.input_dof + 1} ({drive} {response.frf_name}) · dotted: f<sub>d</sub> of "
            "each mode · grey: above the passband · green: fit band (drag it)",
            size="9pt",
        )
