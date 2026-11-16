"""A pointer and ink over the window, for presenting (#316 B8): point at a curve, or draw on the page.

An `InkOverlay` lies over the main window's pages, transparent, and paints only what it's given:

- **F6**, the pointer: a large dot in the force colour follows the mouse (read by an event filter on the app, so
  the overlay lets every click through to the page under it).
- **F7**, ink: the overlay takes the mouse, and dragging draws (each stroke in the force colour, `INK_WIDTH`
  px). F7 again gives the mouse back to the page, the ink staying on the screen.
- **Esc** clears the ink and turns both off. A page change clears the ink too: it was drawn on the old page.

Working on it: `tests/test_present.py` draws a stroke, points, and clears them.
"""

from __future__ import annotations

import enum

from PySide6 import QtCore, QtGui, QtWidgets

from .style import colors

INK_WIDTH = 4.0
POINTER_RADIUS = 9.0


class Mode(enum.Enum):
    OFF = "off"
    POINTER = "pointer"
    INK = "ink"


class InkOverlay(QtWidgets.QWidget):
    changed = QtCore.Signal()  # the mode or the ink

    def __init__(self, window: QtWidgets.QMainWindow) -> None:
        super().__init__(window)
        self.main = window
        self.mode = Mode.OFF
        self.strokes: list[QtGui.QPolygonF] = []
        self.pointer: QtCore.QPointF | None = None
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAccessibleName("Pointer and ink")
        window.installEventFilter(self)
        self._fit()
        self.hide()

    # ------------------------------------------------------------------ modes
    def set_mode(self, mode: Mode) -> None:
        self.mode = mode
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, mode is not Mode.INK)
        app = QtWidgets.QApplication.instance()
        if app is not None:
            if mode is Mode.POINTER:
                app.installEventFilter(self)
            else:
                app.removeEventFilter(self)
                self.pointer = None
        self.setCursor(QtCore.Qt.CursorShape.CrossCursor if mode is Mode.INK else QtCore.Qt.CursorShape.ArrowCursor)
        self._show()
        self.changed.emit()

    def toggle(self, mode: Mode) -> None:
        """F6 / F7: that mode, or off if it's on."""
        self.set_mode(Mode.OFF if self.mode is mode else mode)

    def clear(self) -> None:
        """Esc: the ink gone, the pointer and ink off."""
        self.strokes = []
        self.set_mode(Mode.OFF)

    def clear_ink(self) -> None:
        if self.strokes:
            self.strokes = []
            self._show()
            self.changed.emit()

    @property
    def active(self) -> bool:
        return self.mode is not Mode.OFF or bool(self.strokes)

    def _show(self) -> None:
        self.setVisible(self.active)
        if self.active:
            self._fit()
            self.raise_()
        self.update()

    def _fit(self) -> None:
        central = self.main.centralWidget()
        if central is not None:
            self.setGeometry(central.geometry())

    # ------------------------------------------------------------------ events
    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if not hasattr(self, "main"):  # torn down (a test's window freed): its attributes go first
            return False
        if watched is self.main and event.type() == QtCore.QEvent.Type.Resize:
            self._fit()
        elif self.mode is Mode.POINTER and event.type() == QtCore.QEvent.Type.MouseMove:
            at = self.mapFromGlobal(QtGui.QCursor.pos())
            self.pointer = QtCore.QPointF(at) if self.rect().contains(at) else None
            self.update()
        return False

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.mode is Mode.INK and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.strokes.append(QtGui.QPolygonF([event.position()]))
            self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.mode is Mode.INK and self.strokes and event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            self.strokes[-1].append(event.position())
            self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.mode is Mode.INK:
            self.changed.emit()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        colour = QtGui.QColor(colors.force)
        pen = QtGui.QPen(colour, INK_WIDTH)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        for stroke in self.strokes:
            if stroke.size() == 1:
                p.drawPoint(stroke.at(0))
            else:
                p.drawPolyline(stroke)
        if self.mode is Mode.POINTER and self.pointer is not None:
            halo = QtGui.QColor(colour)
            halo.setAlpha(70)
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(halo)
            p.drawEllipse(self.pointer, 2 * POINTER_RADIUS, 2 * POINTER_RADIUS)
            p.setBrush(colour)
            p.drawEllipse(self.pointer, POINTER_RADIUS, POINTER_RADIUS)
        p.end()
