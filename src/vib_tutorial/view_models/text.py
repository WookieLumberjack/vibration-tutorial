"""Numbers as the pages print them (`gui/lesson.py` re-exports these for the pages without a view model)."""

from __future__ import annotations

import math


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
