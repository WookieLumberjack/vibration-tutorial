"""Pieces the One mass pages and the Signals & spectra page share: plots, the Things to try box, numbers.

Each of those pages has its own small model (core/sdof.py and the modules built on it), works its plots
out in closed form or by the exact step when a control changes, and has no frame loop of its own.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from . import weak
from .axes import log_axes
from .layouts import form_layout, plot_height
from .panels import space_bar, spin
from .theming import mute


def plot(
    left: str,
    left_units: str | None,
    bottom: str,
    bottom_units: str | None,
    log_x: bool = False,
    log_y: bool = False,
    height: int = 160,
) -> pg.PlotWidget:
    """A plot with a faint grid, no menu and no auto-range button; log axes if asked."""
    p = pg.PlotWidget(axisItems=log_axes() if (log_x or log_y) else None)
    p.setMenuEnabled(False)
    p.hideButtons()
    p.showGrid(x=True, y=True, alpha=0.15)
    p.setLabel("left", left, units=left_units)
    p.setLabel("bottom", bottom, units=bottom_units)
    for side, units in (("left", left_units), ("bottom", bottom_units)):
        if units is None:  # a pure number: no "(x0.001)" prefix
            p.getAxis(side).enableAutoSIPrefix(False)
    if log_x or log_y:
        p.setLogMode(x=log_x, y=log_y)
    plot_height(p, height)
    return p


class Dot(QtWidgets.QGraphicsEllipseItem):
    """A marker `size` px across, moved by setData(x, y) to the last point given (hidden for none), as a
    one-point ScatterPlotItem would be: that rebuilds its spot table on every setData (~0.2 ms), and the
    rotor and gyroscope pages move them each frame. Add it to a plot with ignoreBounds=True."""

    def __init__(self, size: float) -> None:
        super().__init__(-0.5 * size, -0.5 * size, size, size)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.setZValue(10)  # over the trails

    def setData(self, x, y) -> None:
        if len(x):
            self.setPos(float(x[-1]), float(y[-1]))
            self.show()
        else:
            self.hide()


def note_label(text: str = "") -> QtWidgets.QLabel:
    """A muted, wrapping rich-text label for a readout or a caption."""
    label = QtWidgets.QLabel(text)
    label.setWordWrap(True)
    label.setTextFormat(QtCore.Qt.TextFormat.RichText)
    mute(label)
    return label


def readout_label() -> QtWidgets.QLabel:
    """A wrapping rich-text label for the numbers a plot shows (full contrast, not muted)."""
    label = QtWidgets.QLabel()
    label.setWordWrap(True)
    label.setTextFormat(QtCore.Qt.TextFormat.RichText)
    return label


class TrialCombo(QtWidgets.QComboBox):
    """The Things to try, numbered in the order the notes take them, after a prompt ("Pick one…"; none on a
    page that always shows one).

    `picked(i)` fires with the trial's index, and the combo keeps showing it. Picking the shown one again sets
    it up again: Qt only says so through `activated`, which also follows a change, so a change marks the
    `activated` that comes with it (in the same call) as handled. Setting the index from code picks too.
    """

    picked = QtCore.Signal(int)

    def __init__(
        self,
        names: Sequence[str],
        tip: str = "",
        prompt: str | None = "Pick one…",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.names = list(names)
        self.first = 0 if prompt is None else 1  # the first trial's row: a page that always shows one has no prompt
        if prompt is not None:
            self.addItem(prompt)
        self.addItems([f"{i}. {name}" for i, name in enumerate(self.names, 1)])
        if tip:
            self.setToolTip(tip)
        self.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self._changed = False
        self.currentIndexChanged.connect(self._on_change)
        self.activated.connect(self._on_activated)

    def pick(self, name: str) -> None:
        """Pick the trial called `name`, as the reader would (again, if it's the one shown)."""
        index = self.names.index(name) + self.first
        if index == self.currentIndex():
            self.picked.emit(index - self.first)
        else:
            self.setCurrentIndex(index)

    def _on_change(self, index: int) -> None:
        if index < self.first:
            return
        self._changed = True
        QtCore.QTimer.singleShot(0, self, weak.call(self._settle))  # a change from code has no `activated` to follow it
        self.picked.emit(index - self.first)

    def _settle(self) -> None:
        self._changed = False

    def _on_activated(self, index: int) -> None:
        if self._changed:
            self._changed = False
        elif index >= self.first:
            self.picked.emit(index - self.first)


class RunButton(QtWidgets.QPushButton):
    """Every page's one control for its motion: Start at rest, then Pause / Resume, and Space while its page
    shows. A page that simulates a response opens at rest and waits for it (a mode shape moves on its own);
    a trial starts it, a control doesn't. `toggled(on)` when the reader presses it; `set_running` and `rest`
    (back to Start, with the page's Reset) from code, without the signal."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Start", parent)
        self.setCheckable(True)
        space_bar(self)
        self.setToolTip("<p>Start, pause or resume the motion (Space).</p>")
        self._started = False
        self.toggled.connect(self._label)

    @property
    def running(self) -> bool:
        return self.isChecked()

    @property
    def started(self) -> bool:
        """Started since it was last at rest (Pause and Resume keep it)."""
        return self._started

    def set_running(self, on: bool) -> None:
        self.blockSignals(True)
        self.setChecked(on)
        self.blockSignals(False)
        self._label(on)

    def rest(self) -> None:
        """Back to Start: the page is at rest at t = 0."""
        self._started = False
        self.set_running(False)

    def _label(self, on: bool) -> None:
        self._started |= on
        self.setText("Pause" if on else "Resume" if self._started else "Start")


class TrialBox(QtWidgets.QGroupBox):
    """Things to try: a `TrialCombo` and the picked one's note below it; `picked(i)` with the trial's index."""

    picked = QtCore.Signal(int)

    def __init__(self, names: Sequence[str], tip: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Things to try", parent)
        layout = QtWidgets.QVBoxLayout(self)
        self.combo = TrialCombo(names, tip)
        self.combo.picked.connect(self.picked)
        layout.addWidget(self.combo)
        self.note = note_label()
        self.note.setVisible(False)
        layout.addWidget(self.note)

    def show_note(self, text: str) -> None:
        self.note.setText(text)
        self.note.setVisible(bool(text))


def trials_html(trials) -> str:
    """The trials as the Theory tab lists them."""
    return "<ol>" + "".join(f"<li><b>{t.name}.</b> {t.note}</li>" for t in trials) + "</ol>"


class Form(QtWidgets.QGroupBox):
    """A titled form of spin boxes, each with a tooltip on its label too; `changed` when any is edited."""

    changed = QtCore.Signal()

    def __init__(self, title: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(title, parent)
        self.form = form_layout(self)

    def number(
        self, label: str, lo: float, hi: float, value: float, decimals: int, suffix: str = "", tip: str = ""
    ) -> QtWidgets.QDoubleSpinBox:
        box = spin(lo, hi, value, decimals, suffix)
        self.add(label, box, tip)
        box.valueChanged.connect(self.changed)
        return box

    def add(self, label: str, widget: QtWidgets.QWidget, tip: str = "") -> None:
        text = QtWidgets.QLabel(label)
        if tip:
            text.setToolTip(tip)
            widget.setToolTip(tip)
        self.form.addRow(text, widget)

    def label(self, widget: QtWidgets.QWidget) -> QtWidgets.QLabel:
        """The label `add` gave `widget`."""
        text = self.form.labelForField(widget)
        assert isinstance(text, QtWidgets.QLabel), "added with a label"
        return text

    def row(self, widget: QtWidgets.QWidget | QtWidgets.QLayout) -> None:
        if isinstance(widget, QtWidgets.QLayout):
            self.form.addRow(widget)
        else:
            self.form.addRow(widget)


def set_quietly(box: QtWidgets.QWidget, value) -> None:
    """Set a spin box's or slider's value, a check box's state or a combo's index without its signals."""
    box.blockSignals(True)
    if isinstance(box, QtWidgets.QAbstractButton):
        box.setChecked(bool(value))
    elif isinstance(box, QtWidgets.QComboBox):
        box.setCurrentIndex(int(value))
    elif isinstance(box, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox, QtWidgets.QAbstractSlider)):
        box.setValue(value)
    box.blockSignals(False)


def fmt(value: float, unit: str, digits: int = 3) -> str:
    """A number with an SI prefix (µ to k), e.g. fmt(0.0251, "m") is "25.1 mm"."""
    if value == 0 or not math.isfinite(value):
        return f"{value:.{digits}g} {unit}" if math.isfinite(value) else f"∞ {unit}"
    for scale, prefix in ((1e3, "k"), (1.0, ""), (1e-3, "m"), (1e-6, "µ"), (1e-9, "n")):
        if abs(value) >= scale * 0.9995:
            return f"{value / scale:.{digits}g} {prefix}{unit}"
    return f"{value:.{digits}g} {unit}"


def db(ratio: float) -> float:
    """20 log10 of an amplitude ratio."""
    return 20.0 * math.log10(ratio) if ratio > 0 else -math.inf


def connect_all(boxes: Sequence[QtWidgets.QWidget], slot: Callable[[], None]) -> None:
    """Call `slot` when any of the spin boxes, check boxes or combos changes."""
    call = weak.call(slot)
    for box in boxes:
        if isinstance(box, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
            box.valueChanged.connect(call)
        elif isinstance(box, QtWidgets.QAbstractButton):
            box.toggled.connect(call)
        elif isinstance(box, QtWidgets.QComboBox):
            box.currentIndexChanged.connect(call)


def marker(size: int = 11, symbol: str = "o", name: str | None = None) -> pg.PlotDataItem:
    """Points without a line. A PlotDataItem, not a ScatterPlotItem: only it follows a plot's log axes."""
    return pg.PlotDataItem(pen=None, symbol=symbol, symbolSize=size, name=name)


def mark(item: pg.PlotDataItem, color: str, filled: bool = False, width: float = 2.0) -> None:
    """Colour a `marker`: an outline, or a filled dot."""
    item.setSymbolPen(pg.mkPen(color, width=width) if not filled else None)
    item.setSymbolBrush(pg.mkBrush(color) if filled else None)


def combo(items: Sequence[str], tip: str = "") -> QtWidgets.QComboBox:
    """A combo box that shrinks to its pane (long items are elided)."""
    box = QtWidgets.QComboBox()
    box.addItems(list(items))
    box.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    if tip:
        box.setToolTip(tip)
    return box


def column(*widgets: QtWidgets.QWidget, stretch: bool = True) -> QtWidgets.QWidget:
    """Widgets stacked top to bottom (a page's left pane, or a tab)."""
    w = QtWidgets.QWidget()
    v = QtWidgets.QVBoxLayout(w)
    for item in widgets:
        v.addWidget(item)
    if stretch:
        v.addStretch(1)
    return w


def stack(*items: tuple[QtWidgets.QWidget, int]) -> QtWidgets.QWidget:
    """Widgets stacked with stretch factors, no margins (a page's centre)."""
    w = QtWidgets.QWidget()
    v = QtWidgets.QVBoxLayout(w)
    v.setContentsMargins(0, 0, 0, 0)
    for item, factor in items:
        v.addWidget(item, factor)
    return w


def three_panes(
    page: QtWidgets.QWidget,
    left: QtWidgets.QWidget,
    centre: QtWidgets.QWidget,
    right: QtWidgets.QWidget,
    sizes: Sequence[int] = (320, 760, 560),
) -> QtWidgets.QSplitter:
    """The lesson pages' layout: controls (scrolling) | the main plots | the tabs."""
    from .layouts import ColumnScroll

    splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
    splitter.addWidget(ColumnScroll(left))
    splitter.addWidget(centre)
    splitter.addWidget(right)
    splitter.setSizes(list(sizes))
    layout = QtWidgets.QHBoxLayout(page)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(splitter)
    return splitter


def theory(html: str) -> QtWidgets.QTextBrowser:
    browser = QtWidgets.QTextBrowser()
    browser.setHtml(html)
    return browser


def relabel(p: pg.PlotWidget, side: str, text: str, units: str | None = None) -> None:
    """Change an axis's label; a pure number (no units) gets no SI prefix."""
    axis = p.getAxis(side)
    axis.enableAutoSIPrefix(units is not None)
    if units is None:
        axis.autoSIPrefixScale = 1.0  # disabling keeps the last prefix's scale
    p.setLabel(side, text, units=units)
