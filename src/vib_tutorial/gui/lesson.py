"""Pieces every lesson page builds from (#409 S-15: it began as the One mass pages' and grew to all of them).

- Plots and markers: `plot` (a faint grid, no menu, log axes if asked, `height`), `relabel`, `marker`/`mark`,
  `Dot` (a marker moved every frame), `cell` (a table's number), `note_label`/`readout_label`, `StartHint`.
- The Things to try: `TrialCombo`, `TrialSteps` (◂ ▸), `TrialBox` (the combo, its note and its prediction);
  their Theory list is `trials_text.trials_html`, without Qt.
- Controls: `Form` (a titled form whose boxes name themselves), `combo`, `set_quietly`, `connect_all`, and
  `RunButton` (Start / Pause / Resume; a motion's Reset is `layouts.ResetButton`), `ActionToggle` (an action
  that stays on, a drive or a sweep: lit in the force's colour while it is, as the Simulation page's force).
- Layout: `three_panes` (the page's controls, plots and tabs, folded by Present mode), `column`, `stack`,
  `TabRoom` (room for the right-hand tabs and the notes), `theory` (the Theory tab's `NotesBrowser`).
- Re-exported for the pages: `fmt` and `db` (`view_models/text.py`), `LiveLabel`.

Working on it: no module of its own; `tests/test_gui_smoke.py` (`test_lesson_pages_trials_and_tabs`, the
TrialCombo and RunButton tests), `test_drag.py` and `test_plots.py` check its pieces through the pages.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.practice import Placement, Prediction
from ..view_models.text import db as db  # re-exported for the pages
from ..view_models.text import fmt as fmt
from . import accessible, coalesce, drag, progress, weak
from .axes import log_axes
from .layouts import form_layout, plot_height
from .live_label import LiveLabel as LiveLabel  # re-exported for the pages
from .maths import markup
from .notes_view import NotesBrowser
from .panels import space_bar, spin
from .style import colors, text_on
from .theming import mute


def plot(
    left: str,
    left_units: str | None,
    bottom: str,
    bottom_units: str | None,
    log_x: bool = False,
    log_y: bool = False,
    height: int | None = 160,
    log_ready: bool = False,
) -> pg.PlotWidget:
    """A plot with a faint grid, no menu and no auto-range button; log axes if asked (`log_ready`: axes that
    label a log scale well, for a plot switched to one later, and label only as many ticks as fit). `height`:
    as short as it may be (`plot_height`), or None to leave it to the caller."""
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
    if height is not None:
        plot_height(p, height)
    return p


def cell(text: str, colour: str | None = None) -> QtWidgets.QTableWidgetItem:
    """A table's number, right-aligned, in `colour` if given."""
    item = QtWidgets.QTableWidgetItem(text)
    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
    if colour:
        item.setForeground(pg.mkColor(colour))
    return item


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


def readout_label(text: str = "") -> QtWidgets.QLabel:
    """A wrapping rich-text label for the numbers a plot shows (full contrast, not muted)."""
    label = QtWidgets.QLabel(text)
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

    def show_tried(self, numbers: set[int]) -> None:
        """Tick the trials tried (`numbers` from 1, as listed: #409 I-9)."""
        for i, name in enumerate(self.names, 1):
            self.setItemText(i - 1 + self.first, f"{'✓ ' if i in numbers else ''}{i}. {name}")

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


class ActionToggle(QtWidgets.QPushButton):
    """An action that stays on until pressed again (a drive, a sweep): `off` names it, in bold in the force's
    colour (set apart from a plain button: Clear points), and on it says so and how to stop, filled in the
    force's colour, as the Simulation page's *Apply force*. Checkable, so it binds
    as a checkbox does; `set_texts` when what it does changes (a force or an unbalance); `apply_theme` from the
    page's."""

    def __init__(self, off: str, on: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(off, parent)
        self.setCheckable(True)
        self.setMinimumHeight(32)
        self._texts = (off, on)

    def checkStateSet(self) -> None:  # every change of state, a quiet one (`set_quietly`) too: not `toggled`
        super().checkStateSet()
        self._show(self.isChecked())

    def set_texts(self, off: str, on: str) -> None:
        self._texts = (off, on)
        self._show(self.isChecked())

    def apply_theme(self) -> None:
        self._show(self.isChecked())

    def _show(self, on: bool) -> None:
        self.setText(self._texts[on])
        self.setStyleSheet(  # off, its text in the force's colour: an action to take, not a plain button
            f"background-color: {colors.force}; color: {text_on(colors.force)}; font-weight: bold;"
            if on
            else f"color: {colors.force}; font-weight: bold;"
        )


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

    A `Placement` is asked on a plot instead: the page's plot it names (an attribute of a widget above the box)
    hides its curves, shows the span and a line to drag, and *That's my guess* runs the trial with the guess
    left on the plot, dashed, until the next pick or Defaults (`guessed`).

    `ask_after_play` (#478 item 4, a page rebuilt to its standard): predictions come after play. Until the page
    calls `set_played()` (the reader has handled the system), a pick runs at once; then a trial not yet tried asks
    its prediction first (a case the reader hasn't seen), and a line invites the next such question. It's an
    invitation, never a gate: *Just run it* is always there.
    """

    picked = QtCore.Signal(int)

    def __init__(
        self,
        names: Sequence[str],
        tip: str,
        parent: QtWidgets.QWidget | None = None,
        predictions: Sequence[Prediction | Placement | None] = (),
        prompt: str | None = "Pick one…",
        ask_after_play: bool = False,
    ) -> None:
        super().__init__("Things to try", parent)
        layout = QtWidgets.QVBoxLayout(self)
        self.names = list(names)
        self.predictions = list(predictions) + [None] * (len(names) - len(predictions))
        self.after_play = ask_after_play
        self.played = not ask_after_play  # predictions are asked once the reader has handled the system
        self.combo = TrialCombo(names, tip, prompt)
        self.combo.picked.connect(self._on_pick)
        layout.addWidget(TrialSteps(self.combo))
        self.invite = note_label()  # after play: the next question worth predicting
        self.invite.setVisible(False)
        layout.addWidget(self.invite)
        self.asking: int | None = None  # the trial whose prediction is being asked
        self.page_name = ""  # the page's attribute, given by MainWindow as it builds the page: what's tried is kept
        self.ask = QtWidgets.QWidget()  # the question, its choices and Just run it
        ask = QtWidgets.QVBoxLayout(self.ask)
        ask.setContentsMargins(0, 0, 0, 0)
        self.question = QtWidgets.QLabel()
        self.question.setWordWrap(True)
        self.question.setTextFormat(QtCore.Qt.TextFormat.RichText)
        ask.addWidget(self.question)
        self.choices = QtWidgets.QVBoxLayout()
        ask.addLayout(self.choices)
        self.place = QtWidgets.QPushButton("That's my guess")
        self.place.setToolTip("<p>Run the trial with the line where you've put it.</p>")
        self.place.clicked.connect(self._on_place)
        ask.addWidget(self.place)
        self.line: drag.DragLine | None = None  # the line dragged to a Placement's guess, while it's asked
        self.guessed: pg.InfiniteLine | None = None  # the guess, left on the plot after the trial runs
        self._placing: tuple[Any, list, Any, Any] | None = None  # the plot, its hidden curves, its ranges then
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
            self._stop_placing()
            self._clear_guess()
            self.asking = None
            self.ask.setVisible(False)
            self.verdict.setVisible(False)

    def answer(self, choice: int) -> None:
        """Answer the prediction being asked with choice `choice` (a button's click): the verdict, then the trial."""
        i = self.asking
        assert i is not None, "answered only while a prediction is asked"
        p = self.predictions[i]
        assert isinstance(p, Prediction), "a choice answers a Prediction"
        if choice == p.answer:
            text = "<b>✓ Right.</b> Here it is:"
        elif p.why:  # why the one picked is wrong, then the trial's note says why the answer is right
            text = f"<b>✗ Not quite:</b> it's <i>{p.choices[p.answer]}</i>. {p.why[choice]} Here it is:"
        else:
            text = f"<b>✗ Not quite:</b> it's <i>{p.choices[p.answer]}</i>. See why:"
        self.verdict.setText(markup(text))
        self._run(i, p.prompt)

    def show_tried(self) -> None:
        """The combo's ticks, from what's been tried (`progress`)."""
        if self.page_name:
            self.combo.show_tried(progress.tried_numbers(self.page_name))
        self._show_invite()

    def _tried(self) -> set[int]:
        return progress.tried_numbers(self.page_name) if self.page_name else set()

    def asks(self, i: int) -> bool:
        """Whether picking trial `i` asks its prediction first (see the class)."""
        if self.predictions[i] is None:
            return False
        return not self.after_play or (self.played and i + 1 not in self._tried())

    def set_played(self, played: bool = True) -> None:
        """The reader has handled the system (or, False, the page is back as it opened): predictions on (off)."""
        if not self.after_play or played == self.played:
            return
        self.played = played
        self._show_invite()

    def _show_invite(self) -> None:
        if not self.after_play:
            return
        nxt = next((i for i in range(len(self.names)) if self.asks(i)), None)
        self.invite.setVisible(nxt is not None)
        if nxt is not None:
            self.invite.setText(
                markup(f"<b>Now predict:</b> <i>{self.names[nxt]}</i> asks what you'll see before it runs.")
            )

    def _record(self, i: int, prompt: str | None = None) -> None:
        """Trial `i` run, after its prediction (`prompt`) was answered if it had one: kept, and ticked."""
        if not self.page_name:  # a page built outside the window (a test's)
            return
        if prompt is not None:
            progress.mark_answered(prompt)
        progress.mark_tried(self.page_name, i + 1)
        self.show_tried()

    def place_guess(self, value: float) -> None:
        """Answer the Placement being asked with `value` (data units), as dragging the line there and pressing
        *That's my guess* would."""
        assert self.line is not None, "placed only while a Placement is asked"
        self.line.set_value(value)
        self._on_place()

    def _on_place(self) -> None:
        i = self.asking
        assert i is not None and self.line is not None, "placed only while a Placement is asked"
        p = self.predictions[i]
        assert isinstance(p, Placement), "a guess answers a Placement"
        guess = self.line.data_value()
        self.verdict.setText(markup(p.judge(guess)[1]))
        self._stop_placing()
        self._run(i, p.prompt)
        self._show_guess(p, guess)  # after the trial's redraw, which sets the plot's ranges

    def _plot_named(self, name: str):
        """The page's plot `name`: an attribute of the nearest widget above the box that has one."""
        w = self.parentWidget()
        while w is not None and not hasattr(w, name):
            w = w.parentWidget()
        assert w is not None, f"no plot {name} above the Things to try"
        found = getattr(w, name)
        return found.getPlotItem() if hasattr(found, "getPlotItem") else found

    def _start_placing(self, p: Placement) -> None:
        item = self._plot_named(p.plot)
        box = item.getViewBox()
        hidden = [c for c in item.items if c.isVisible()]
        self._placing = (item, hidden, box.viewRange(), box.autoRangeEnabled())
        for c in hidden:  # the last trial's curves and markers would give the answer away, or mislead
            c.setVisible(False)
        lo, hi = (math.log10(v) for v in p.span) if p.log else p.span
        if p.axis == "y":
            item.setYRange(lo, hi, padding=0)
        else:
            item.setXRange(lo, hi, padding=0)
        self.line = drag.DragLine(
            "<p>Drag to your guess, then press <i>That's my guess</i>.</p>",
            angle=0 if p.axis == "y" else 90,
            log=p.log,
            bounds=p.span,
            role="strong",
        )
        item.addItem(self.line, ignoreBounds=True)
        self.line.set_value(p.start())

    def _stop_placing(self) -> None:
        if self._placing is None:
            return
        item, hidden, ranges, auto = self._placing
        self._placing = None
        if self.line is not None:
            item.removeItem(self.line)
            self.line = None
        for c in hidden:
            c.setVisible(True)
        box = item.getViewBox()
        box.setRange(xRange=ranges[0], yRange=ranges[1], padding=0)
        box.enableAutoRange(x=auto[0], y=auto[1])

    def _show_guess(self, p: Placement, guess: float) -> None:
        item = self._plot_named(p.plot)
        line = pg.InfiniteLine(
            math.log10(guess) if p.log else guess,
            angle=0 if p.axis == "y" else 90,
            pen=pg.mkPen(colors.strong, width=2, style=QtCore.Qt.PenStyle.DashLine),
            label="your guess",
            labelOpts={"position": 0.08 if p.axis == "y" else 0.92, "color": colors.strong},
        )
        line.setZValue(15)
        item.addItem(line, ignoreBounds=True)
        self.guessed = line

    def _clear_guess(self) -> None:
        if self.guessed is not None:
            scene_item = self.guessed.getViewBox()
            if scene_item is not None:
                scene_item.removeItem(self.guessed)
            self.guessed = None

    def _on_pick(self, i: int) -> None:
        p = self.predictions[i] if i < len(self.predictions) and self.asks(i) else None
        self._stop_placing()
        self._clear_guess()
        self.verdict.setVisible(False)
        if p is None:
            self.asking = None
            self.ask.setVisible(False)
            with coalesce.batch():  # the page's redraws once, with the whole trial loaded
                self.picked.emit(i)
            self._record(i)
            return
        self.asking = i
        self.note.setVisible(False)
        while self.choices.count():
            item = self.choices.takeAt(0)
            if item is not None and (widget := item.widget()) is not None:
                widget.hide()  # out of the layout, it would show where it was until it's deleted
                widget.deleteLater()
        self.place.setVisible(isinstance(p, Placement))
        if isinstance(p, Placement):
            self.question.setText(markup(f"<b>Predict first:</b> {p.prompt} Drag the line on the plot to your guess."))
            self._start_placing(p)
            self.ask.setVisible(True)
            return
        self.question.setText(markup(f"<b>Predict first:</b> {p.prompt}"))
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
        self._stop_placing()
        self.verdict.setText("")
        self._run(self.asking)

    def _run(self, i: int, answered: str | None = None) -> None:
        """Set trial `i` up, its prediction answered (its prompt) or skipped (None)."""
        self.asking = None
        self.ask.setVisible(False)
        self.verdict.setVisible(bool(self.verdict.text()))
        with coalesce.batch():  # the page's redraws once, with the whole trial loaded
            self.picked.emit(i)
        self._record(i, answered)


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


class StartHint(QtWidgets.QLabel):
    """A quiet line centred over a plot that waits for Start ("Start (Space) runs it from rest"), so its empty axes
    read as waiting rather than broken (#409 U-22). The page shows it while it has nothing to draw: `waiting(bool)`.
    A plot under ~60 px (the rotor's strip chart at 1280 × 800) has no room for one: its ranges at rest say enough."""

    def __init__(self, plot: QtWidgets.QWidget, text: str = "Start (Space) runs it from rest") -> None:
        super().__init__(text, plot)
        mute(self)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        plot.installEventFilter(self)
        self.resize(plot.size())

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.Resize and isinstance(watched, QtWidgets.QWidget):
            self.resize(watched.size())
        return False

    def waiting(self, on: bool) -> None:
        if on != self.isVisible():
            self.setVisible(on)


class TabRoom(QtCore.QObject):
    """Widens a splitter's last pane, taking from the panes before it (the nearest first, each down to its
    minimum), until every tab bar in it shows all its tabs' names (no scroll arrows), as it's shown or resized,
    until the reader moves the splitter themselves; and while its notes (a Theory tab's `NotesBrowser`) show,
    until they need no sideways scroll, giving that room back to the panes it came from once they hide (it took
    120 px from Signals' plots for good, #409 U-13). Not a page's other text views (the Frame builder's K matrix,
    ~770 px): they scroll sideways in their pane, which would otherwise take the room from the page's main view
    (#409 U-11). `MainWindow._page` gives every page's panes one.

    `sizes` and `TAB_ROOM` lay the panes out for this machine's font; another's (macOS's 13 pt system font,
    Ubuntu's DejaVu Sans) is wider, and the tabs' names with it."""

    def __init__(self, splitter: QtWidgets.QSplitter) -> None:
        super().__init__(splitter)
        self._splitter = splitter
        self._moved = False
        self._pending = False
        self._lent: dict[int, int] = {}  # px taken from each pane (by index) for the notes, given back as they hide
        splitter.splitterMoved.connect(self._on_moved)
        last = splitter.widget(splitter.count() - 1)
        assert last is not None, "a splitter with panes"
        last.installEventFilter(self)
        for tabs in last.findChildren(QtWidgets.QTabWidget):
            tabs.installEventFilter(self)  # a tab page's own tabs show when it's picked
        for notes in last.findChildren(NotesBrowser):
            notes.installEventFilter(self)  # hidden: their room goes back
            bar = notes.horizontalScrollBar()
            assert bar is not None, "a text browser's scroll bar"
            bar.rangeChanged.connect(self._on_range)  # the notes laid out wider than their pane

    def _on_moved(self, _pos: int, _index: int) -> None:
        self._moved = True
        self._lent.clear()  # the reader's sizes now: nothing to give back

    def _on_range(self, _lo: int, _hi: int) -> None:
        self._schedule()

    def _schedule(self) -> None:
        # after the splitter's own layout pass, which would undo sizes set in the middle of it
        if not self._moved and not self._pending:
            self._pending = True
            QtCore.QTimer.singleShot(0, self, self.fit)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() in (QtCore.QEvent.Type.Resize, QtCore.QEvent.Type.Show, QtCore.QEvent.Type.Hide):
            self._schedule()
        return False

    def _last(self) -> QtWidgets.QWidget | None:
        last = self._splitter.widget(self._splitter.count() - 1)
        return last if last is not None and last.isVisible() and last.width() > 0 else None

    def tabs_short(self) -> int:
        """How many pixels the last pane is short of for its visible tab bars' names."""
        last = self._last()
        short = 0
        for tabs in [last, *last.findChildren(QtWidgets.QTabWidget)] if last is not None else []:
            if isinstance(tabs, QtWidgets.QTabWidget) and tabs.isVisible():
                bar = tabs.tabBar()
                need = sum(bar.tabSizeHint(i).width() for i in range(bar.count()) if bar.isTabVisible(i))
                short = max(short, need - tabs.width())
        return short

    def notes_short(self) -> int | None:
        """How many pixels the last pane is short of for its visible notes' widest table or figure; None when no
        notes show."""
        last = self._last()
        widths = [
            scroll.maximum()
            for notes in (last.findChildren(NotesBrowser) if last is not None else [])
            if notes.isVisible() and (scroll := notes.horizontalScrollBar()) is not None
        ]
        return max(widths) if widths else None

    def fit(self) -> None:
        self._pending = False
        n = self._splitter.count()
        if n < 2 or self._last() is None:
            return
        sizes = self._splitter.sizes()
        tabs = self.tabs_short()
        notes = self.notes_short()
        if notes is None and self._lent:  # the notes hid: their room back, as far as the tab bars allow
            spare = sizes[n - 1] - max(0, tabs) - self._minimum(n - 1)
            for i, px in sorted(self._lent.items(), reverse=True):
                give = min(px, spare)
                sizes[i] += give
                sizes[n - 1] -= give
                spare -= give
            self._lent.clear()
        if tabs > 0:
            self._take(sizes, tabs + 2, lend=False)
        if notes is not None and notes > 0:
            self._take(sizes, notes + 2, lend=True)
        if sizes != self._splitter.sizes():
            self._splitter.setSizes(sizes)

    def _minimum(self, i: int) -> int:
        pane = self._splitter.widget(i)
        return pane.minimumSizeHint().width() if pane is not None else 0

    def _take(self, sizes: list[int], want: int, lend: bool) -> None:
        """`want` px for the last pane from the panes before it, the nearest first; kept in `_lent` if `lend`."""
        n = len(sizes)
        for i in range(n - 2, -1, -1):
            pane = self._splitter.widget(i)
            if pane is None or not pane.isVisible():
                continue
            give = min(want, sizes[i] - self._minimum(i))
            if give > 0:
                sizes[i] -= give
                sizes[n - 1] += give
                want -= give
                if lend:
                    self._lent[i] = self._lent.get(i, 0) + give
            if want <= 0:
                break


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
