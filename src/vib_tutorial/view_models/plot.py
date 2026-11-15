"""What every view model's plots are made of."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Curve:
    """A line's or a marker's points; `item.setData(*curve)` draws it (an empty one hides a marker)."""

    x: np.ndarray
    y: np.ndarray

    def __iter__(self):
        return iter((self.x, self.y))

    @staticmethod
    def point(x: float, y: float) -> Curve:
        return Curve(np.array([x]), np.array([y]))

    @staticmethod
    def empty() -> Curve:
        return Curve(np.empty(0), np.empty(0))
