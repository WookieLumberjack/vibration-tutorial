"""A 3D view the reader turns by dragging (#409 S-5): the top, the box on its mounts, the rigid body and the
rotor were four copies of the same mouse handlers.

`OrbitView` goes before `pg.PlotWidget` in a view's bases. Dragging with the left button turns it (the side
nearest you follows the cursor: yaw from the drag's x, pitch from its y, held to ±90°), a double-click puts it
back at the class's `YAW` and `PITCH`, and each turn calls the view's `_place_static()` to redraw what doesn't
move. `project` is `core/rotation.py`'s `project_z_up`, or `project_y_up` with `UP = "y"` (a shaft along z).

Working on it: `tests/test_gyroscope.py`, `test_rigid_body.py`, `test_rigid_mounts.py` and the rotor's tests
drag their views (N-4's yaw sign).
"""

from __future__ import annotations

import math

import numpy as np
from PySide6 import QtCore

from ..core.rotation import project_y_up, project_z_up

TURN = 0.01  # rad a pixel of drag


class OrbitView:
    """The drag-to-turn mixin (see the module); the view sets `yaw`, `pitch` and has `_place_static`."""

    YAW: float
    PITCH: float
    UP = "z"
    yaw: float
    pitch: float
    _drag: QtCore.QPointF | None = None

    def _place_static(self) -> None:
        raise NotImplementedError

    def project(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Screen (x, y) of points p (..., 3) at the view's turn (see the module)."""
        return (project_y_up if self.UP == "y" else project_z_up)(p, self.yaw, self.pitch)

    def mousePressEvent(self, event) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag = event.position()
            event.accept()
        else:
            super().mousePressEvent(event)  # type: ignore[misc]

    def mouseMoveEvent(self, event) -> None:
        if self._drag is None:
            super().mouseMoveEvent(event)  # type: ignore[misc]
            return
        d = event.position() - self._drag
        self._drag = event.position()
        self.yaw = (self.yaw + TURN * d.x()) % (2 * math.pi)  # the side nearest you follows the cursor
        self.pitch = float(np.clip(self.pitch + TURN * d.y(), -math.pi / 2, math.pi / 2))
        self._place_static()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self._drag = None
        super().mouseReleaseEvent(event)  # type: ignore[misc]

    def mouseDoubleClickEvent(self, event) -> None:
        self.yaw, self.pitch = self.YAW, self.PITCH
        self._place_static()
        event.accept()
