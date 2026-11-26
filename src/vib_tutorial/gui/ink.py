"""A pointer and ink over the window, for presenting (#316 B8): point at a curve, or draw on the page.

An `InkOverlay` lies over the main window's pages (a child of them), transparent, and paints only what it's
given. While it shows, a timer (`TICK_MS`) keeps it over the pages as the window resizes and moves the pointer.
Keep it holding no Python reference to the window, filtering no events of the window's, and the window
overriding none for it: each was called into, or torn down, as `gc` freed a window half gone, and segfaulted
the tests (two windows built in a row).

- **F6**, the pointer: a large dot in the force color follows the mouse (read from the cursor's position each
  tick, so the overlay lets every click through to the page under it).
- **F7**, ink: the overlay takes the mouse, and dragging draws (each stroke in the force color, `INK_WIDTH`
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
TICK_MS = 16


class Mode(enum.Enum):
    OFF = "off"
    POINTER = "pointer"
    INK = "ink"


class InkOverlay(QtWidgets.QWidget):
    changed = QtCore.Signal()  # the mode or the ink

    def __init__(self, pages: QtWidgets.QWidget) -> None:
        super().__init__(pages)  # holding no Python reference to the window: freed with it, not in a cycle
        self.mode = Mode.OFF
        self.strokes: list[QtGui.QPolygonF] = []
        self.pointer: QtCore.QPointF | None = None
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAccessibleName("Pointer and ink")
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(TICK_MS)
        self.timer.timeout.connect(self._tick)
        self.fit()
        self.hide()

    # ------------------------------------------------------------------ modes
    def set_mode(self, mode: Mode) -> None:
        self.mode = mode
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, mode is not Mode.INK)
        if mode is not Mode.POINTER:
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
            self.fit()
            self.raise_()
            self.timer.start()
        else:
            self.timer.stop()
        self.update()

    def fit(self) -> None:
        parent = self.parentWidget()
        if parent is not None:
            self.setGeometry(parent.rect())

    # ------------------------------------------------------------------ events
    def _tick(self) -> None:
        """Over the pages still, and the pointer at the mouse while it's over them."""
        parent = self.parentWidget()
        if parent is not None and parent.rect() != self.geometry():
            self.fit()
        if self.mode is not Mode.POINTER:
            return
        at = self.mapFromGlobal(QtGui.QCursor.pos())
        pointer = QtCore.QPointF(at) if self.rect().contains(at) else None
        if pointer != self.pointer:
            self.pointer = pointer
            self.update()

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
