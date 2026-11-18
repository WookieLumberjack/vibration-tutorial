"""Pieces the One mass pages and the Signals & spectra page share: plots, the Things to try box, numbers.

Each of those pages has its own small model (core/sdof.py and the modules built on it), works its plots
out in closed form or by the exact step when a control changes, and has no frame loop of its own.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.practice import Prediction
from ..view_models.text import db as db  # re-exported for the pages
from ..view_models.text import fmt as fmt
from . import accessible, weak
from .axes import log_axes
from .layouts import form_layout, plot_height
from .maths import markup
from .notes_view import NotesBrowser
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
    log_ready: bool = False,
) -> pg.PlotWidget:
    """A plot with a faint grid, no menu and no auto-range button; log axes if asked (`log_ready`: axes that
    label a log scale well, for a plot switched to one later, and label only as many ticks as fit)."""
    p = pg.PlotWidget(axisItems=log_axes() if (log_x or log_y or log_ready) else None)
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


class LiveLabel(QtWidgets.QWidget):
    """A line or a paragraph of rich text a frame loop rewrites every frame: a clock, a run's status.

    A QLabel's `setText` asks for a layout pass each time (`updateGeometry`), and the window then lays the page
    out again, every frame: the Waves tab's status took 6.5 ms a frame that way, the Playback clock ~2.7. This
    draws its own text (a QTextDocument) and asks for a layout only when the text needs more room: a `wrap`ped
    paragraph another line (or one fewer), a line a wider box than it has. Otherwise a new text repaints the
    label alone. `text()` is the rich text last set."""

    def __init__(self, text: str = "", wrap: bool = False, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.wrap = wrap
        self._doc = QtGui.QTextDocument(self)
        self._doc.setDocumentMargin(0)
        self._text = ""
        self._height = -1
        policy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Preferred if wrap else QtWidgets.QSizePolicy.Policy.Fixed,
        )
        policy.setHeightForWidth(wrap)
        self.setSizePolicy(policy)
        self.setText(text)

    def text(self) -> str:
        return self._text

    def setText(self, text: str) -> None:
        if text == self._text:
            return
        self._text = text
        self._doc.setDefaultFont(self.font())
        self._doc.setHtml(text)
        if self.wrap:
            height = self.heightForWidth(max(self.width(), 1))
            grow = height != self._height
            self._height = height
        else:
            grow = self._line().width() > self.width() or self._line().height() != self.height()
        if grow:
            self.updateGeometry()
        self.setAccessibleDescription(self._doc.toPlainText())
        self.update()

    def _line(self) -> QtCore.QSize:
        self._doc.setTextWidth(-1)
        return QtCore.QSize(math.ceil(self._doc.idealWidth()), math.ceil(self._doc.size().height()))

    def hasHeightForWidth(self) -> bool:
        return self.wrap

    def heightForWidth(self, width: int) -> int:
        if not self.wrap:
            return self._line().height()
        self._doc.setTextWidth(width)
        return math.ceil(self._doc.size().height())

    def sizeHint(self) -> QtCore.QSize:
        line = self._line()
        return QtCore.QSize(line.width(), self.heightForWidth(line.width()))

    def minimumSizeHint(self) -> QtCore.QSize:
        if not self.wrap:
            return self._line()
        return QtCore.QSize(0, self.heightForWidth(max(self.width(), 1)))

    def changeEvent(self, event: QtCore.QEvent) -> None:
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.Type.FontChange:
            self._doc.setDefaultFont(self.font())
            self._height = -1
            self.updateGeometry()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self.wrap:
            self._height = self.heightForWidth(max(self.width(), 1))

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        self._doc.setTextWidth(self.width() if self.wrap else -1)
        context = QtGui.QAbstractTextDocumentLayout.PaintContext()
        context.palette = self.palette()
        context.palette.setColor(QtGui.QPalette.ColorRole.Text, self.palette().color(self.foregroundRole()))
        self._doc.documentLayout().draw(painter, context)
        painter.end()


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
    """The Things to try, numbered in the order the notes take them, after a prompt ("Pick one…"): `chosen()`
    is the trial shown, or None at the prompt (a page that works its settings out from the chain sets up its
    own opening ones there, and a trial's when one shows).

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
        self.setAccessibleName("Things to try")
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

    def chosen(self) -> int | None:
        """The trial shown, or None at the prompt."""
        index = self.currentIndex() - self.first
        return index if index >= 0 else None

    def step(self, by: int) -> None:
        """The next (`by` = 1) or previous (−1) trial, as if picked: from the prompt, the first. It stops at the
        ends."""
        now = self.chosen()
        to = 0 if now is None else min(max(now + by, 0), len(self.names) - 1)
        if (now is None and by < 0) or to == now:
            return
        self.setCurrentIndex(to + self.first)

    def next(self) -> None:
        self.step(1)

    def previous(self) -> None:
        self.step(-1)

    def unpick(self) -> None:
        """Back to the prompt, picking nothing (a page put back as it opened)."""
        if self.first:
            self.blockSignals(True)
            self.setCurrentIndex(0)
            self.blockSignals(False)

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


class TrialSteps(QtWidgets.QWidget):
    """A `TrialCombo` with ◂ and ▸ beside it: the previous and the next Thing to try, in the notes' order
    (Alt+PgUp, Alt+PgDown on the page showing: `MainWindow.step_trial`)."""

    def __init__(self, combo: TrialCombo, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.combo = combo
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        row.addWidget(combo, 1)
        self.back = QtWidgets.QToolButton()
        self.back.setText("◂")
        self.back.setToolTip("<p>The Thing to try before this one (Alt+PgUp).</p>")
        self.back.setAccessibleName("Previous thing to try")
        self.back.clicked.connect(combo.previous)
        self.on = QtWidgets.QToolButton()
        self.on.setText("▸")
        self.on.setToolTip("<p>The next Thing to try: they build on each other in this order (Alt+PgDown).</p>")
        self.on.setAccessibleName("Next thing to try")
        self.on.clicked.connect(combo.next)
        row.addWidget(self.back)
        row.addWidget(self.on)
        combo.currentIndexChanged.connect(self._update)
        self._update()

    def _update(self, *_) -> None:
        now = self.combo.chosen()
        self.back.setEnabled(now is not None and now > 0)
        self.on.setEnabled(now is None or now < len(self.combo.names) - 1)


class TrialBox(QtWidgets.QGroupBox):
    """Things to try: a `TrialCombo` and the picked one's note below it; `picked(i)` with the trial's index.

    A trial with a `Prediction` (`predictions[i]`, from its `predict` field) asks it first: its question and a
    button per choice (and *Just run it*) take the note's place, and the page isn't touched until one is pressed.
    Then the verdict shows, `picked(i)` sets the trial up, and its note (the explanation) shows below. Calling
    the page's `load_trial` directly (tests, the Practice window) skips the question.
    """

    picked = QtCore.Signal(int)

    def __init__(
        self,
        names: Sequence[str],
        tip: str,
        parent: QtWidgets.QWidget | None = None,
        predictions: Sequence[Prediction | None] = (),
        prompt: str | None = "Pick one…",
    ) -> None:
        super().__init__("Things to try", parent)
        layout = QtWidgets.QVBoxLayout(self)
        self.predictions = list(predictions) + [None] * (len(names) - len(predictions))
        self.combo = TrialCombo(names, tip, prompt)
        self.combo.picked.connect(self._on_pick)
        layout.addWidget(TrialSteps(self.combo))
        self.asking: int | None = None  # the trial whose prediction is being asked
        self.ask = QtWidgets.QWidget()  # the question, its choices and Just run it
        ask = QtWidgets.QVBoxLayout(self.ask)
        ask.setContentsMargins(0, 0, 0, 0)
        self.question = QtWidgets.QLabel()
        self.question.setWordWrap(True)
        self.question.setTextFormat(QtCore.Qt.TextFormat.RichText)
        ask.addWidget(self.question)
        self.choices = QtWidgets.QVBoxLayout()
        ask.addLayout(self.choices)
        self.skip = QtWidgets.QPushButton("Just run it")
        self.skip.setFlat(True)
        self.skip.setToolTip("<p>Set the trial up without predicting.</p>")
        self.skip.clicked.connect(self._on_skip)
        ask.addWidget(self.skip)
        self.ask.setVisible(False)
        layout.addWidget(self.ask)
        self.verdict = QtWidgets.QLabel()
        self.verdict.setWordWrap(True)
        self.verdict.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.verdict.setVisible(False)
        layout.addWidget(self.verdict)
        self.note = note_label()
        self.note.setVisible(False)
        layout.addWidget(self.note)

    def show_note(self, text: str) -> None:
        """The trial's note; none puts the box back as it opened (no question, no verdict)."""
        self.note.setText(markup(text))
        self.note.setVisible(bool(text))
        if not text:
            self.asking = None
            self.ask.setVisible(False)
            self.verdict.setVisible(False)

    def answer(self, choice: int) -> None:
        """Answer the prediction being asked with choice `choice` (a button's click): the verdict, then the trial."""
        i = self.asking
        assert i is not None, "answered only while a prediction is asked"
        p = self.predictions[i]
        assert p is not None, "asked only for a trial with a prediction"
        right = choice == p.answer
        self.verdict.setText(
            markup(
                "<b>✓ Right.</b> Here it is:"
                if right
                else f"<b>✗ Not quite:</b> it's <i>{p.choices[p.answer]}</i>. See why:"
            )
        )
        self._run(i)

    def _on_pick(self, i: int) -> None:
        p = self.predictions[i] if i < len(self.predictions) else None
        self.verdict.setVisible(False)
        if p is None:
            self.asking = None
            self.ask.setVisible(False)
            self.picked.emit(i)
            return
        self.asking = i
        self.note.setVisible(False)
        self.question.setText(markup(f"<b>Predict first:</b> {p.prompt}"))
        while self.choices.count():
            item = self.choices.takeAt(0)
            if item is not None and (widget := item.widget()) is not None:
                widget.hide()  # out of the layout, it would show where it was until it's deleted
                widget.deleteLater()
        for k in p.shown_order():  # shuffled, the same each time (a scale stays a scale)
            button = QtWidgets.QPushButton(p.choices[k])  # plain text: a button shows no rich text
            button.setProperty("choice", k)
            button.clicked.connect(self._on_choice)
            self.choices.addWidget(button)
        self.ask.setVisible(True)

    def _on_choice(self) -> None:
        button = self.sender()
        if isinstance(button, QtWidgets.QPushButton):
            self.answer(int(button.property("choice")))

    def _on_skip(self) -> None:
        assert self.asking is not None, "Just run it shows only while a prediction is asked"
        self.verdict.setText("")
        self._run(self.asking)

    def _run(self, i: int) -> None:
        self.asking = None
        self.ask.setVisible(False)
        self.verdict.setVisible(bool(self.verdict.text()))
        self.picked.emit(i)


def trials_html(trials) -> str:
    """The trials as the Theory tab lists them."""

    def item(t) -> str:
        p = getattr(t, "predict", None)
        ask = f"<i>Predict first: {markup(p.prompt)}</i> " if p is not None else ""
        return f"<li><b>{t.name}.</b> {ask}{markup(t.note)}</li>"

    return "<ol>" + "".join(item(t) for t in trials) + "</ol>"


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
        accessible.name_field(text, widget)

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


TAB_ROOM = (320, 560, 760)  # `three_panes` sizes for a page whose tabs need the room at 1280 px (no scroll arrows)


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


class TabRoom(QtCore.QObject):
    """Widens a splitter's last pane, taking from the panes before it (the nearest first, each down to its
    minimum), until every tab bar in it shows all its tabs' names (no scroll arrows), as it's shown or resized,
    until the reader moves the splitter themselves; and its notes (a Theory tab) until they need no sideways
    scroll. `MainWindow._page` gives every page's panes one.

    `sizes` and `TAB_ROOM` lay the panes out for this machine's font; another's (macOS's 13 pt system font,
    Ubuntu's DejaVu Sans) is wider, and the tabs' names with it."""

    def __init__(self, splitter: QtWidgets.QSplitter) -> None:
        super().__init__(splitter)
        self._splitter = splitter
        self._moved = False
        self._pending = False
        splitter.splitterMoved.connect(self._on_moved)
        last = splitter.widget(splitter.count() - 1)
        assert last is not None, "a splitter with panes"
        last.installEventFilter(self)
        for tabs in last.findChildren(QtWidgets.QTabWidget):
            tabs.installEventFilter(self)  # a tab page's own tabs show when it's picked
        for notes in last.findChildren(QtWidgets.QTextBrowser):
            bar = notes.horizontalScrollBar()
            assert bar is not None, "a text browser's scroll bar"
            bar.rangeChanged.connect(self._on_range)  # the notes laid out wider than their pane

    def _on_moved(self, _pos: int, _index: int) -> None:
        self._moved = True

    def _on_range(self, _lo: int, _hi: int) -> None:
        self._schedule()

    def _schedule(self) -> None:
        # after the splitter's own layout pass, which would undo sizes set in the middle of it
        if not self._moved and not self._pending:
            self._pending = True
            QtCore.QTimer.singleShot(0, self, self.fit)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() in (QtCore.QEvent.Type.Resize, QtCore.QEvent.Type.Show):
            self._schedule()
        return False

    def short(self) -> int:
        """How many pixels the last pane is short of: for its visible tab bars' names, and its visible notes'
        widest table or figure."""
        last = self._splitter.widget(self._splitter.count() - 1)
        if last is None or not last.isVisible() or last.width() == 0:
            return 0
        short = 0
        for tabs in [last, *last.findChildren(QtWidgets.QTabWidget)]:
            if isinstance(tabs, QtWidgets.QTabWidget) and tabs.isVisible():
                bar = tabs.tabBar()
                need = sum(bar.tabSizeHint(i).width() for i in range(bar.count()) if bar.isTabVisible(i))
                short = max(short, need - tabs.width())
        for notes in last.findChildren(QtWidgets.QTextBrowser):
            scroll = notes.horizontalScrollBar()
            if notes.isVisible() and scroll is not None:
                short = max(short, scroll.maximum())
        return short

    def fit(self) -> None:
        self._pending = False
        short = self.short()
        n = self._splitter.count()
        if short <= 0 or n < 2:
            return
        sizes = self._splitter.sizes()
        want = short + 2
        for i in range(n - 2, -1, -1):
            pane = self._splitter.widget(i)
            if pane is None or not pane.isVisible():
                continue
            give = min(want, sizes[i] - pane.minimumSizeHint().width())
            if give > 0:
                sizes[i] -= give
                sizes[n - 1] += give
                want -= give
            if want <= 0:
                break
        if sizes != self._splitter.sizes():
            self._splitter.setSizes(sizes)


def theory(html: str) -> NotesBrowser:
    """A Theory tab showing the notes `html` (their equations and figures drawn)."""
    return NotesBrowser(html)


def relabel(p: pg.PlotWidget, side: str, text: str, units: str | None = None) -> None:
    """Change an axis's label; a pure number (no units) gets no SI prefix."""
    axis = p.getAxis(side)
    axis.enableAutoSIPrefix(units is not None)
    if units is None:
        axis.autoSIPrefixScale = 1.0  # disabling keeps the last prefix's scale
    p.setLabel(side, text, units=units)
