"""`LiveLabel`: text a frame loop rewrites every frame, without laying the page out again each time.

Working on it: `tests/test_playback.py` checks it asks for a layout only when it needs more room, and
`tests/test_live_labels.py` that no page's frame loop rewrites a QLabel every frame (#409 P-2).
"""

from __future__ import annotations

import math

from PySide6 import QtCore, QtGui, QtWidgets


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
