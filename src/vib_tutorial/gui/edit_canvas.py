"""The builders' editing canvas (#409 S-12): the frame builder's and the rotor builder's had the same events.

`EditCanvas` goes before `pg.PlotWidget` in a canvas's bases. A press (taking the keyboard focus) is the canvas's
`press(point, right=)`, a move `drag_to(point)`, a release `release()`, in view coordinates; a double-click is
eaten (no pyqtgraph auto-range) and the wheel left to the page, which scrolls (the view fits what's drawn).
Delete or Backspace is `delete_selected()`, Esc `cancel()` (the frame's half-drawn element; nothing by default).
pyqtgraph's own right-click menu is the canvas's right press (`plot_export` leaves a view that handles presses
itself alone).

Working on it: `tests/test_frame_builder.py` and `tests/test_rotor_builder.py` click and drag on the canvases.
"""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore


class EditCanvas:
    """The editing events (see the module); the canvas has `press`, `drag_to`, `release`, `delete_selected`."""

    def _view_point(self, event) -> np.ndarray:
        scene = self.mapToScene(event.position().toPoint())  # type: ignore[attr-defined]
        p = self.getPlotItem().getViewBox().mapSceneToView(scene)  # type: ignore[attr-defined]
        return np.array([p.x(), p.y()])

    def press(self, point: np.ndarray, right: bool = False) -> None:
        raise NotImplementedError

    def drag_to(self, point: np.ndarray) -> None:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError

    def delete_selected(self) -> None:
        raise NotImplementedError

    def cancel(self) -> None:
        """Esc: drop what's half done (nothing, unless the canvas says)."""

    def mousePressEvent(self, event) -> None:
        self.setFocus()  # type: ignore[attr-defined]
        self.press(self._view_point(event), right=event.button() == QtCore.Qt.MouseButton.RightButton)
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        self.drag_to(self._view_point(event))
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self.release()
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        event.accept()

    def wheelEvent(self, event) -> None:
        event.ignore()  # the page scrolls; the view fits what's drawn

    def keyPressEvent(self, event) -> None:
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.cancel()
        elif event.key() in (QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace):
            self.delete_selected()
        else:
            super().keyPressEvent(event)  # type: ignore[misc]
