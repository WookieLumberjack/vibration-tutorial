"""The virtual modal test's settings: the Test, Acquisition, Measurement noise and Processing panes, which the
page shows in its left column, and what is read from them: the test (`settings`), its processing
(`processing`), the line under the acquisition (`show_info`) and the check of the settings against the chain's
modes (`setup_check`, the page's Setup tab).

A change that needs new data (the excitation, where the force is, the sensor, the block, the sample rate, the
force level...) says `remeasure`; one that only processes the same data again (the window, the estimator, the
noise) says `reprocess`; the number of averages says `averages_changed`, since the record carries on. A
trial's settings are set by the page's `apply_trial`, its signals held, without any of them.

Working on it: `test_gui_smoke.py`'s modal test tests set the controls through `page.test_setup`.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6 import QtCore, QtWidgets

from ..core import (
    Acquisition,
    ChainSystem,
    Estimator,
    Excitation,
    MeasurementSettings,
    ModalResult,
    Processing,
    Response,
    Window,
    impact_spectrum,
)
from ..core.measurement import AA_CUTOFF, FORCE_RMS, IMPACT_PEAK, SINE_AMPLITUDE
from .layouts import form_layout
from .panels import spin
from .style import MAX_DOF, colors
from .theming import mute

BLOCK_SIZES = [256, 512, 1024, 2048, 4096, 8192]
OVERLAPS = [0.0, 0.5, 0.75]

# The window each excitation is normally measured with; picked when the excitation changes.
RECOMMENDED_WINDOW = {
    Excitation.IMPACT: Window.FORCE_EXPONENTIAL,
    Excitation.RANDOM: Window.HANN,
    Excitation.BURST_RANDOM: Window.RECTANGULAR,
    Excitation.PERIODIC_RANDOM: Window.RECTANGULAR,
    Excitation.CHIRP: Window.RECTANGULAR,
    Excitation.STEPPED_SINE: Window.RECTANGULAR,
}

EXCITATION_TIPS = {
    Excitation.IMPACT: "A hammer hit: a short half-sine pulse at the start of each block. Quick, and "
    "no shaker needed. The tip sets how short the pulse is, and so how high it excites.",
    Excitation.RANDOM: "A shaker driven by continuous band-limited noise. Never periodic in the "
    "block, so it leaks without a window (Hann). Averaging reduces noise, and random excitation averages any non-linearity into a best "
    "linear estimate.",
    Excitation.BURST_RANDOM: "Noise for the first part of each block, then silence while the "
    "response dies away. If it dies away inside the block, no window is needed.",
    Excitation.PERIODIC_RANDOM: "A random signal exactly one block long, repeated until the "
    "response is steady, so both signals are periodic in the block: no leakage.",
    Excitation.CHIRP: "A sine sweeping from 0 to the band edge in one block, repeated. Periodic "
    "in the block, like periodic random.",
    Excitation.STEPPED_SINE: "One frequency at a time: wait for steady state, then fit a sine to "
    "force and response. Slow, but the most accurate, with no windows or leakage.",
}

WINDOW_TIPS = {
    Window.RECTANGULAR: "No window. Right when the signals are periodic in the block or have died "
    "away by its end; otherwise it leaks.",
    Window.HANN: "Tapers both ends to zero, which cuts leakage for random signals. It also blurs "
    "sharp peaks a little, and it wipes out a hammer hit at the start of the block.",
    Window.FLAT_TOP: "Very wide main lobe: accurate amplitudes of pure tones, but blurs resonances.",
    Window.FORCE_EXPONENTIAL: "For impact tests. Force: keeps the hit and zeroes the noise after "
    "it. Exponential on both channels: makes the response die away inside the block, which adds "
    "known extra damping to every mode.",
}


FORCE_LEVEL_TIP = (
    f"<p>Scales the force. At 100% the hammer hits with a {IMPACT_PEAK:g} N peak, the shaker drives "
    f"{FORCE_RMS:g} N RMS and the stepped sine has a {SINE_AMPLITUDE:g} N amplitude.</p>"
    "<p>A linear chain's FRF does not depend on it: the response scales with the force, and so "
    "does the noise, which is sized to each channel's range. With <b>friction</b> on a mass "
    "(F<sub>f</sub> on the Simulation page) it does. A lab checks linearity this way: measure at "
    "two force levels and overlay the FRFs (<i>Hold for comparison</i>).</p>"
    "<p>With friction and a stepped sine, the plot also shows the describing-function FRF at "
    "this level (dash-dot): the frequency response worked out with each friction force "
    "replaced by its first harmonic, the same approximation as the dash-dot curves on the "
    "Simulation page's Frequency response tab. A stepped sine reads that first harmonic, so "
    "the two should agree while every mass slides; a random or impact test has no single "
    "amplitude to compare with.</p>"
)


def friction_masses(system: ChainSystem) -> str:
    """'m1, m3' for the masses with friction ('' if none)."""
    return ", ".join(f"m{i + 1}" for i in np.flatnonzero(system.friction > 0))


def suggested_fs(result: ModalResult) -> float:
    """A sample rate whose passband (AA_CUTOFF x Nyquist) clears the highest mode by 25%."""
    f_top = max((m.fn_hz for m in result.complex_modes if m.is_oscillatory), default=1.0)
    need = 1.25 * f_top * 2.0 / AA_CUTOFF
    scale = 10.0 ** (math.floor(math.log10(need)) - 1)
    return math.ceil(need / scale) * scale


def percent(fraction: float) -> str:
    return "&lt; 0.01%" if fraction < 1e-4 else f"{100 * fraction:.2g}%"


class TestSetup(QtWidgets.QWidget):
    """The test's four panes, one above the other."""

    remeasure = QtCore.Signal()  # a setting the data depend on changed: measure again
    reprocess = QtCore.Signal()  # only the processing (or the noise added) changed: the same data again
    averages_changed = QtCore.Signal()
    dof_requested = QtCore.Signal(int)  # the user changed N here; the main window owns the system
    edit_parameters = QtCore.Signal()  # the user wants the Simulation page's parameter panel

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        setup = QtWidgets.QGroupBox("Test")
        form = form_layout(setup)
        self.dof = QtWidgets.QSpinBox()
        self.dof.setRange(1, MAX_DOF)
        self.dof.setToolTip("Same chain as the Simulation page; changing it here changes it there too.")
        self.dof.valueChanged.connect(self.dof_requested)
        form.addRow("Number of masses:", self.dof)
        self.excitation = QtWidgets.QComboBox()
        for i, e in enumerate(Excitation):
            self.excitation.addItem(e.value, e)
            self.excitation.setItemData(i, EXCITATION_TIPS[e], QtCore.Qt.ItemDataRole.ToolTipRole)
        form.addRow("Excitation:", self.excitation)
        self.input = QtWidgets.QComboBox()
        self.input.setToolTip("Where the force is applied. Every mass's response is measured.")
        form.addRow("Force at:", self.input)
        self.response = QtWidgets.QComboBox()
        for r in Response:
            self.response.addItem(f"{r.value} ({r.frf_name})", r)
        self.response.setToolTip(
            "<p>What the sensor on each mass measures. <b>Displacement</b> gives the receptance "
            "H = X/F directly. <b>Acceleration</b>, as with the accelerometers of a real test, gives "
            "the accelerance A = −ω²H.</p><p>Modes are extracted from the receptance, so an "
            "accelerance is divided by −ω² first; that amplifies the low-frequency noise.</p>"
        )
        form.addRow("Response sensor:", self.response)
        level_row = QtWidgets.QHBoxLayout()
        self.force_level = spin(1.0, 10000.0, 100.0, 0, " %")
        self.force_level.setToolTip(FORCE_LEVEL_TIP)
        self.force_level_info = QtWidgets.QLabel()
        mute(self.force_level_info)
        level_row.addWidget(self.force_level, 1)
        level_row.addWidget(self.force_level_info)
        form.addRow("Force level:", level_row)
        self.tip = spin(1.0, 2000.0, 50.0, 1, " ms")
        self.tip.setToolTip(
            "<p>Duration of the hammer's half-sine pulse. A hard (metal) tip gives a short pulse, "
            "a soft (rubber) tip a long one.</p><p>The pulse's spectrum is flat at low frequency, "
            "rolls off above about 1/τ<sub>p</sub> and first reaches zero at 1.5/τ<sub>p</sub>. Modes above that are "
            "barely excited, so noise swamps them.</p>"
        )
        self.tip_label = QtWidgets.QLabel("Hammer pulse τₚ:")
        form.addRow(self.tip_label, self.tip)
        self.burst = spin(5.0, 100.0, 50.0, 0, " %")
        self.burst.setToolTip("How much of each block the shaker is on. The rest lets the response die away.")
        self.burst_label = QtWidgets.QLabel("Burst length:")
        form.addRow(self.burst_label, self.burst)
        self.overlap = QtWidgets.QComboBox()
        for o in OVERLAPS:
            self.overlap.addItem(f"{o:.0%}", o)
        self.overlap.setCurrentIndex(1)
        self.overlap.setToolTip(
            "Consecutive blocks share this much data. With a Hann window, 50% "
            "overlap uses the data the window tapers away and gives more "
            "averages from the same measuring time."
        )
        self.overlap_label = QtWidgets.QLabel("Block overlap:")
        form.addRow(self.overlap_label, self.overlap)
        self.points = QtWidgets.QSpinBox()
        self.points.setRange(10, 400)
        self.points.setValue(100)
        self.points.setKeyboardTracking(False)
        self.points.setToolTip("Frequencies measured, evenly spaced up to the band edge.")
        self.points_label = QtWidgets.QLabel("Frequencies:")
        form.addRow(self.points_label, self.points)
        link = QtWidgets.QLabel("Masses, springs and dampers are edited on the <a href='#sim'>Simulation page</a>.")
        mute(link)
        link.setWordWrap(True)
        link.linkActivated.connect(self.edit_parameters)
        form.addRow(link)

        # --- acquisition
        daq = QtWidgets.QGroupBox("Acquisition")
        form = form_layout(daq)
        fs_row = QtWidgets.QHBoxLayout()
        self.fs = spin(0.5, 10000.0, 20.0, 2, " Hz")
        self.fs.setToolTip(
            "Samples per second on every channel. Nyquist frequency fs/2 is the "
            "highest frequency the samples can represent."
        )
        self.fs_auto = QtWidgets.QCheckBox("Auto")
        self.fs_auto.setChecked(True)
        self.fs_auto.setToolTip(
            "Follow the chain: choose fs so the highest mode is well inside "
            f"the anti-alias passband ({AA_CUTOFF:.0%} of Nyquist)."
        )
        fs_row.addWidget(self.fs, 1)
        fs_row.addWidget(self.fs_auto)
        form.addRow("Sample rate fs:", fs_row)
        self.block = QtWidgets.QComboBox()
        for b in BLOCK_SIZES:
            self.block.addItem(str(b), b)
        self.block.setCurrentIndex(BLOCK_SIZES.index(1024))
        self.block.setToolTip(
            "Samples per block Nb. The block lasts T = Nb/fs and the FRF lines "
            "are Δf = 1/T apart: longer blocks resolve narrower peaks."
        )
        self.block_label = QtWidgets.QLabel("Block size Nb:")
        form.addRow(self.block_label, self.block)
        self.averages = QtWidgets.QSpinBox()
        self.averages.setRange(1, 100)
        self.averages.setValue(10)
        self.averages.setKeyboardTracking(False)
        self.averages.setToolTip("Blocks averaged. Noise in the estimate falls as 1/√(averages).")
        self.averages_label = QtWidgets.QLabel("Averages:")
        form.addRow(self.averages_label, self.averages)
        self.anti_alias = QtWidgets.QCheckBox("Anti-alias filter")
        self.anti_alias.setChecked(True)
        self.anti_alias.setToolTip(
            "<p>A steep low-pass on every channel before sampling, with its passband edge at "
            f"{AA_CUTOFF:.0%} of Nyquist (the shaded band on the plots is above it).</p>"
            "<p>Without it, response above fs/2 is not lost but folds back: a mode at f "
            "appears at |f − k·fs|, mixed in with the real response.</p>"
        )
        form.addRow(self.anti_alias)
        self.daq_info = QtWidgets.QLabel()
        self.daq_info.setWordWrap(True)
        mute(self.daq_info)
        form.addRow(self.daq_info)

        # --- noise
        noise = QtWidgets.QGroupBox("Measurement noise")
        form = form_layout(noise)
        noise_tip = (
            "Random noise added to the channel, as a percentage of the channel's peak in "
            "each block (the range an analyzer would set). Changing it reuses the same "
            "measurement, so you can compare like with like."
        )
        self.force_noise = spin(0.0, 100.0, 0.0, 2, " %")
        self.force_noise.setToolTip(noise_tip + " Noise on the force biases H1 low.")
        form.addRow("Force channel:", self.force_noise)
        self.response_noise = spin(0.0, 100.0, 0.0, 2, " %")
        self.response_noise.setToolTip(noise_tip + " Noise on the response biases H2 high.")
        form.addRow("Response channels:", self.response_noise)

        # --- processing
        proc = QtWidgets.QGroupBox("Processing")
        form = form_layout(proc)
        self.taper = QtWidgets.QComboBox()
        for i, kind in enumerate(Window):
            self.taper.addItem(kind.value, kind)
            self.taper.setItemData(i, WINDOW_TIPS[kind], QtCore.Qt.ItemDataRole.ToolTipRole)
        self.taper.setToolTip(
            "Multiplies each block before the FFT. Choosing an excitation picks its "
            "usual window; change it to see what the others do to the same data."
        )
        form.addRow("Window:", self.taper)
        self.exp_end = spin(0.1, 100.0, 100.0, 1, " %")
        self.exp_end.setToolTip(
            "<p>The exponential window's value at the end of the block; 100% = no exponential "
            "(force window only).</p><p>e<sup>−t/τ<sub>w</sub></sup> multiplies the response, so every "
            "pole moves 1/τ<sub>w</sub> to the left: each mode looks more damped by Δζ = 1/(τ<sub>w</sub>ω). The dashed "
            "grey curve is the exact FRF with that extra damping.</p>"
        )
        self.exp_label = QtWidgets.QLabel("Exponential at block end:")
        form.addRow(self.exp_label, self.exp_end)
        self.estimator_combo = QtWidgets.QComboBox()
        for estimator in Estimator:
            self.estimator_combo.addItem(estimator.value, estimator)
        self.estimator_combo.setToolTip(
            "<p><b>H1</b> = G<sub>xf</sub>/G<sub>ff</sub>: noise on the response averages out; "
            "noise on the force inflates G<sub>ff</sub> and biases H1 low.</p>"
            "<p><b>H2</b> = G<sub>xx</sub>/G<sub>fx</sub>: noise on the force averages out; "
            "noise on the response inflates G<sub>xx</sub> and biases H2 high.</p>"
            "<p>When uncorrelated noise is the only error, the true FRF lies between them; leakage or "
            "aliasing can push both the same way. Where coherence is 1 they agree.</p>"
        )
        form.addRow("Estimator:", self.estimator_combo)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        for pane in (setup, daq, noise, proc):
            layout.addWidget(pane)

        self.excitation.currentIndexChanged.connect(self._on_excitation)
        for choice in (self.input, self.response, self.block, self.overlap):
            choice.currentIndexChanged.connect(self.remeasure)
        for field in (self.tip, self.burst, self.fs, self.force_level):
            field.valueChanged.connect(self.remeasure)
        self.points.valueChanged.connect(self.remeasure)
        self.anti_alias.toggled.connect(self.remeasure)
        self.fs_auto.toggled.connect(self._on_fs_auto)
        self.averages.valueChanged.connect(self.averages_changed)
        for choice in (self.taper, self.estimator_combo):
            choice.currentIndexChanged.connect(self._on_processing)
        for field in (self.exp_end, self.force_noise, self.response_noise):
            field.valueChanged.connect(self._on_processing)
        self._select_window(Excitation.IMPACT)
        self._show_rows()

    def settings(self, n: int, seed: int | None) -> MeasurementSettings:
        """The test as the controls set it up, on a chain of `n` masses (`seed`: fixed random signals, or None)."""
        return MeasurementSettings(
            excitation=self.excitation.currentData(),
            input_dof=min(max(0, self.input.currentIndex()), n - 1),
            response=self.response.currentData(),
            fs=self.fs.value(),
            block=self.block.currentData(),
            averages=self.averages.value(),
            overlap=self.overlap.currentData(),
            anti_alias=self.anti_alias.isChecked(),
            tip_width=self.tip.value() * 1e-3,
            burst=self.burst.value() / 100.0,
            sine_points=self.points.value(),
            force_level=self.force_level.value() / 100.0,
            seed=seed,
        )

    def processing(self) -> Processing:
        return Processing(
            window=self.taper.currentData(),
            exp_end=self.exp_end.value() / 100.0,
            estimator=self.estimator_combo.currentData(),
            force_noise=self.force_noise.value() / 100.0,
            response_noise=self.response_noise.value() / 100.0,
        )

    def fill_inputs(self, n: int) -> None:
        """One entry per mass in the force's combo (kept if it already has them)."""
        if self.input.count() == n:
            return
        self.input.blockSignals(True)
        self.input.clear()
        self.input.addItems([f"m{j + 1}" for j in range(n)])
        self.input.setCurrentIndex(n - 1)
        self.input.blockSignals(False)

    def follow(self, n: int, result: ModalResult) -> None:
        """A new test of a chain of `n` masses: its size shown, and fs from its modes if Auto."""
        self.dof.blockSignals(True)
        self.dof.setValue(n)
        self.dof.blockSignals(False)
        self.fs.setEnabled(not self.fs_auto.isChecked())
        if self.fs_auto.isChecked():
            self.fs.blockSignals(True)
            self.fs.setValue(suggested_fs(result))
            self.fs.blockSignals(False)

    def show_level(self, s: MeasurementSettings) -> None:
        """The force level in newtons, beside its box."""
        self.force_level_info.setText(self._level_text(s))

    def _level_text(self, s: MeasurementSettings) -> str:
        e = s.excitation
        if e is Excitation.IMPACT:
            return f"{s.force_level * IMPACT_PEAK:.4g} N peak"
        if e is Excitation.STEPPED_SINE:
            return f"{s.force_level * SINE_AMPLITUDE:.4g} N amplitude"
        return f"{s.force_level * FORCE_RMS:.4g} N RMS"

    def _on_excitation(self) -> None:
        self._select_window(self.excitation.currentData())
        self._show_rows()
        self.remeasure.emit()

    def _on_fs_auto(self, on: bool) -> None:
        self.fs.setEnabled(not on)
        if on:
            self.remeasure.emit()

    def _on_processing(self) -> None:
        self._show_rows()
        self.reprocess.emit()

    def _select_window(self, excitation: Excitation) -> None:
        self.taper.blockSignals(True)
        self.taper.setCurrentIndex(list(Window).index(RECOMMENDED_WINDOW[excitation]))
        self.taper.blockSignals(False)

    def _show_rows(self) -> None:
        e = self.excitation.currentData()
        rows = {
            (self.tip_label, self.tip): e is Excitation.IMPACT,
            (self.burst_label, self.burst): e is Excitation.BURST_RANDOM,
            (self.overlap_label, self.overlap): e is Excitation.RANDOM,
            (self.points_label, self.points): e is Excitation.STEPPED_SINE,
            (self.averages_label, self.averages): e is not Excitation.STEPPED_SINE,
            (self.block_label, self.block): e is not Excitation.STEPPED_SINE,
            (self.exp_label, self.exp_end): self.taper.currentData() is Window.FORCE_EXPONENTIAL,
        }
        for widgets, on in rows.items():
            for w in widgets:
                w.setVisible(on)
        self.taper.setEnabled(e is not Excitation.STEPPED_SINE)

    def show_info(self, a: Acquisition) -> None:
        """The acquisition's numbers under its pane: block length, line spacing, Nyquist, waits."""
        s = a.settings
        if a.stepped:
            text = f"{s.sine_points} frequencies, {s.band / s.sine_points:.4g} Hz apart<br>"
        else:
            text = f"T = Nb/fs = {s.duration:.4g} s · Δf = 1/T = {s.resolution:.4g} Hz<br>"
        text += f"Nyquist fs/2 = {s.fs / 2:.4g} Hz · passband and excitation to {s.band:.4g} Hz"
        if s.excitation.settles:
            text += f"<br>Waits {a.settle:.3g} s for steady state (transients down to 0.1%)"
        elif s.excitation.waits:
            text += f"<br>Waits {a.settle:.3g} s between hits for the response to die away (to 0.1%)"
        self.daq_info.setText(text)


def setup_check(a: Acquisition, w: Window) -> str:
    """The settings against the chain's modes: what each mode will suffer from (`w`: the window chosen)."""
    s = a.settings
    T, nyq = s.duration, s.fs / 2
    df = s.band / s.sine_points if a.stepped else s.resolution
    rows = []
    warnings = []
    columns = 5 + (0 if a.stepped else 1) + (1 if s.excitation is Excitation.IMPACT else 0)
    for m in a.result.modes:
        d = m.damped
        if d is None:
            continue
        color = colors.mode[(m.index - 1) % len(colors.mode)]
        bw = 2.0 * d.zeta * d.fn_hz
        lines = bw / df
        left = math.exp(d.eigenvalue.real * T)
        notes = []
        if d.fd_hz > nyq:
            alias = abs(d.fd_hz - s.fs * round(d.fd_hz / s.fs))
            notes.append(
                "above Nyquist: " + ("removed by the filter" if s.anti_alias else f"<b>aliases to {alias:.3g} Hz</b>")
            )
        elif d.fd_hz > s.band:
            notes.append(
                "in the filter's roll-off" if s.anti_alias else f"above the usable band ({AA_CUTOFF:.0%} of Nyquist)"
            )
        if lines < 2.0 and d.fd_hz <= nyq:
            notes.append(
                "<b>under-resolved</b>: "
                + ("the peak falls between frequencies" if a.stepped else "peak too low, looks too damped")
            )
        if s.excitation in (Excitation.IMPACT, Excitation.BURST_RANDOM) and left > 0.01:
            notes.append("<b>still ringing at block end</b>: leakage unless a window is used")
        tip = ""
        if s.excitation is Excitation.IMPACT:
            level = 20.0 * math.log10(max(float(impact_spectrum(s.tip_width, np.array([d.fd_hz]))[0]), 1e-12))
            tip = f"{round(level) + 0:.0f} dB"
            if level < -20.0:
                notes.append("<b>hardly excited by this tip</b>: noise will dominate")
        rows.append(
            f"<tr><td style='white-space:nowrap; color:{color}'><b>{m.index}</b></td><td style='white-space:nowrap'>{d.fd_hz:.4g}</td>"
            f"<td style='white-space:nowrap'>{d.zeta:.4f}</td><td style='white-space:nowrap'>{bw:.3g}</td><td style='white-space:nowrap'>{lines:.3g}</td>"
            + ("" if a.stepped else f"<td style='white-space:nowrap'>{percent(left)}</td>")
            + (f"<td style='white-space:nowrap'>{tip}</td>" if s.excitation is Excitation.IMPACT else "")
            + "</tr>"
            + (f"<tr><td></td><td colspan='{columns - 1}'>{'; '.join(notes)}</td></tr>" if notes else "")
        )
    # Numbers and headers don't wrap (in a narrow pane they broke mid-number); a mode's notes, if
    # it has any, take a row of their own under it, across the table.
    head = (
        "<tr><th style='white-space:nowrap'>Mode</th><th style='white-space:nowrap'>f<sub>d</sub><br>[Hz]</th><th style='white-space:nowrap'>ζ</th>"
        "<th style='white-space:nowrap'>2ζf<sub>n</sub><br>[Hz]</th>"
        + (
            "<th style='white-space:nowrap'>Points<br>in it</th>"
            if a.stepped
            else "<th style='white-space:nowrap'>Lines<br>in it</th><th style='white-space:nowrap'>Left<br>at T</th>"
        )
        + ("<th style='white-space:nowrap'>Hammer<br>level</th>" if s.excitation is Excitation.IMPACT else "")
        + "</tr>"
    )
    e = s.excitation
    if e is Excitation.IMPACT and not s.anti_alias and s.tip_width < 2.0 / s.fs:
        warnings.append(
            "The hit lasts less than 2 samples. Without the anti-alias filter to spread it "
            "out, the samples can miss most of it, and the measured force is wrong."
        )
    if e is Excitation.IMPACT and w is Window.HANN:
        warnings.append("A Hann window is zero at the start of the block, where the hit is: it throws the force away.")
    if e is Excitation.RANDOM and w is Window.RECTANGULAR:
        warnings.append("Continuous random is not periodic in the block: without a window it leaks.")
    if e in (Excitation.PERIODIC_RANDOM, Excitation.CHIRP) and w is not Window.RECTANGULAR:
        warnings.append("The signals are periodic in the block, so no window is needed; one only blurs the peaks.")
    rubbing = friction_masses(a.system)
    if rubbing:
        warnings.append(
            f"Friction on {rubbing}: the chain is nonlinear, so what is measured is a linear "
            "approximation that depends on the force level and the excitation. <i>Exact</i> is the "
            "chain without friction. Friction damps small motions heavily, so a light force gives low, "
            "flat peaks (or none, if the masses hardly slide); a strong one approaches the exact FRF. "
            "Measure at two force levels with <i>Hold for comparison</i>, and watch the coherence. "
            "The response is simulated step by step, so measuring takes longer."
        )
    warn = "".join(f"<p style='color:{colors.fair}'>{t}</p>" for t in warnings)
    return (
        f"<p><b>{e.value}</b>, force at m{s.input_dof + 1}, {s.response.value.lower()} measured "
        f"({s.response.frf_name}). {EXCITATION_TIPS[e]}</p>{warn}"
        f"<table border='1' cellspacing='0' cellpadding='3' width='100%'>{head}{''.join(rows)}</table>"
        f"<p style='color:{colors.muted}'>2ζf<sub>n</sub> is the half-power bandwidth of each peak; "
        "with fewer than about 2 lines (Δf apart, or stepped-sine frequencies) across it, the peak is missed. "
        "<i>Left at T</i> is how much of a free decay is left at the end of a block "
        "(e<sup>σT</sup>): a hit or burst that has not died away is cut off, which leaks. A problem with a "
        "mode is noted on a row under it; a mode with nothing under it should measure well.</p>"
    )
