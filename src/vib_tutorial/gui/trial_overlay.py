"""The Thing to try's note, large over the page, while presenting (#316 B15).

In Present mode (F5) the left pane, and the Things to try's note with it, is folded away. Stepping through the
trials then (Alt+PgDown / Alt+PgUp, `MainWindow.step_trial`) shows the step's name and note in this panel at the
top of the page, in large type, so the room reads what the step is about. A trial that asks a prediction first
shows its question and a button per choice (and *Just run it*): pressing one answers it as the hidden box's
buttons would, and the panel then shows the verdict and the note. The × button, Esc, a new page or leaving
Present mode hide it.

The text is read from the page's own `TrialBox` after the step, so it is what the box shows: its note (or
question).

Working on it: `tests/test_present.py` steps trials with the panes folded and answers a prediction from here.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from .lesson import TrialBox, TrialCombo

SCALE = 1.5  # the text's size over the app's
WIDTH = 0.7  # of the page's width, at most MAX_WIDTH px
MAX_WIDTH = 1000


def _box(combo: TrialCombo) -> TrialBox | None:
    parent = combo.parentWidget()
    while parent is not None and not isinstance(parent, TrialBox):
        parent = parent.parentWidget()
    return parent


class TrialOverlay(QtWidgets.QFrame):
    """The panel (see the module's docstring)."""

    shown = QtCore.Signal(bool)

    def __init__(self, pages: QtWidgets.QWidget) -> None:
        super().__init__(pages)
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setAutoFillBackground(True)
        self.setAccessibleName("Thing to try, large")
        self.box: TrialBox | None = None
        layout = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        self.title = QtWidgets.QLabel()
        self.title.setWordWrap(True)
        self.close_button = QtWidgets.QToolButton()
        self.close_button.setText("×")
        self.close_button.setAutoRaise(True)
        self.close_button.setToolTip("<p>Hide the note (Esc).</p>")
        self.close_button.setAccessibleName("Hide the note")
        self.close_button.clicked.connect(self.dismiss)
        top.addWidget(self.title, 1)
        top.addWidget(self.close_button, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        layout.addLayout(top)
        self.body = QtWidgets.QLabel()
        self.body.setWordWrap(True)
        self.body.setTextFormat(QtCore.Qt.TextFormat.RichText)
        layout.addWidget(self.body)
        self.choices = QtWidgets.QHBoxLayout()
        layout.addLayout(self.choices)
        for label in (self.title, self.body):
            font = QtGui.QFont(label.font())
            font.setPointSizeF(font.pointSizeF() * SCALE)
            font.setBold(label is self.title)
            label.setFont(font)
        pages.installEventFilter(self)
        self.hide()

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if watched is self.parentWidget() and event.type() == QtCore.QEvent.Type.Resize and self.isVisible():
            self.fit()
        return False

    def fit(self) -> None:
        """At the top of the page, centred, as tall as its text."""
        parent = self.parentWidget()
        if parent is None:
            return
        page = parent.currentWidget() if hasattr(parent, "currentWidget") else None
        top = (page.mapTo(parent, QtCore.QPoint(0, 0)).y() if page is not None else 0) + 8  # below the tabs
        width = min(int(parent.width() * WIDTH), MAX_WIDTH)
        self.setFixedWidth(width)
        layout = self.layout()
        assert layout is not None, "laid out in __init__"
        self.setFixedHeight(max(min(layout.totalHeightForWidth(width), parent.height() - top - 8), 0))
        self.move((parent.width() - width) // 2, top)

    def show_for(self, page: QtWidgets.QWidget, combo: TrialCombo) -> None:
        """The note of the trial `combo` has just picked on `page`."""
        i = combo.chosen()
        if i is None:
            self.dismiss()
            return
        self.box = _box(combo)
        self.title.setText(f"{i + 1} / {len(combo.names)} · {combo.names[i]}")
        self._clear_choices()
        box = self.box
        if box is not None and box.asking is not None:
            prediction = box.predictions[box.asking]
            assert prediction is not None, "asking only a trial with a prediction"
            self.body.setText(box.question.text())
            for k, text in enumerate(prediction.choices):
                button = QtWidgets.QPushButton(text)
                button.setProperty("choice", k)
                button.clicked.connect(self._on_choice)
                self.choices.addWidget(button)
            skip = QtWidgets.QPushButton("Just run it")
            skip.setFlat(True)
            skip.clicked.connect(self._on_skip)
            self.choices.addWidget(skip)
        elif box is not None:
            verdict = box.verdict.text() if box.verdict.isVisibleTo(box) else ""
            self.body.setText(" ".join(t for t in (verdict, box.note.text()) if t))
        else:
            self.body.setText("")
        self.fit()
        self.show()
        self.raise_()
        self.shown.emit(True)

    def _clear_choices(self) -> None:
        while self.choices.count():
            item = self.choices.takeAt(0)
            if item is not None and (widget := item.widget()) is not None:
                widget.hide()
                widget.deleteLater()

    def _on_choice(self) -> None:
        button = self.sender()
        if isinstance(button, QtWidgets.QPushButton) and self.box is not None:
            self._answered(lambda box: box.answer(int(button.property("choice"))))

    def _on_skip(self) -> None:
        self._answered(lambda box: box.skip.click())

    def _answered(self, answer) -> None:
        box = self.box
        assert box is not None and box.asking is not None, "choices show only while a prediction is asked"
        answer(box)
        page = self.parentWidget()
        assert page is not None, "a child of the pages"
        self.show_for(page, box.combo)

    def dismiss(self) -> None:
        if self.isVisible():
            self.hide()
            self._clear_choices()
            self.shown.emit(False)
