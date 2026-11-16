"""Numbers as the pages print them, in one place (#316 D8; `gui/lesson.py` re-exports `fmt` and `db` for the pages
without a view model).

One rule for trailing zeros: a number is printed to its significant figures with the trailing zeros dropped
(`f"{v:.3g}"`: 2.5, not 2.50), unless `zeros=True` keeps them where the notes quote a figure's precision
(Duffing's 24.0 mm, 8.80 Hz). No "−0": a value that rounds to zero prints as 0.

- `number(v, digits, zeros)`: the figures alone.
- `fmt(v, unit, digits, largest, smallest, zeros)`: with an SI prefix between `largest` and `smallest` ("k" to "n"
  unless asked): fmt(0.0251, "m") is "25.1 mm". A value past either end keeps that end's prefix.
- `plain(v, unit, digits)`: no prefix (a unit that takes none: g, dB, %).
- `fixed(v, decimals)`: to a set number of decimals, trailing zeros dropped (a table's column).

Working on it: `tests/test_view_models.py` checks them; a page's own formatter (a length, an energy, a torque)
is a call to `fmt` with its prefixes.
"""

from __future__ import annotations

import math

PREFIXES = (("G", 1e9), ("M", 1e6), ("k", 1e3), ("", 1.0), ("m", 1e-3), ("µ", 1e-6), ("n", 1e-9), ("p", 1e-12))
_NAMES = [p for p, _ in PREFIXES]


def _signless_zero(text: str) -> str:
    return text[1:] if text.startswith("-") and float(text) == 0 else text


def number(value: float, digits: int = 3, zeros: bool = False) -> str:
    """`value` to `digits` significant figures; trailing zeros dropped unless `zeros` (see the module's docstring)."""
    if value == 0 or not math.isfinite(value):
        return f"{value:g}" if math.isfinite(value) else ("∞" if value > 0 else "−∞" if value < 0 else "–")
    if not zeros:
        return _signless_zero(f"{value:.{digits}g}")
    shown = float(f"{value:.{digits}g}")
    decimals = max(0, digits - 1 - math.floor(math.log10(abs(shown))))
    return _signless_zero(f"{value:.{decimals}f}")


def fmt(value: float, unit: str, digits: int = 3, largest: str = "k", smallest: str = "n", zeros: bool = False) -> str:
    """A number with an SI prefix from `largest` down to `smallest`, e.g. fmt(0.0251, "m") is "25.1 mm"."""
    if not math.isfinite(value):
        return f"{number(value)} {unit}"
    if value == 0:
        return f"0 {unit}"
    scales = PREFIXES[_NAMES.index(largest) : _NAMES.index(smallest) + 1]
    near = 1.0 - 0.5 * 10.0**-digits  # 999.6 µm rounds to 1.00 mm
    prefix, scale = next(((p, s) for p, s in scales if abs(value) >= s * near), scales[-1])
    return f"{number(value / scale, digits, zeros)} {prefix}{unit}"


def plain(value: float, unit: str, digits: int = 3) -> str:
    """A number and its unit, no prefix."""
    return f"{number(value, digits)} {unit}".strip()


def fixed(value: float, decimals: int) -> str:
    """`value` to `decimals` places, trailing zeros and a bare point dropped, no −0."""
    text = f"{value:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def db(ratio: float) -> float:
    """20 log10 of an amplitude ratio."""
    return 20.0 * math.log10(ratio) if ratio > 0 else -math.inf
