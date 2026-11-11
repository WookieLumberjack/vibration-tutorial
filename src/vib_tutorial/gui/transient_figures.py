"""The Transient response Theory's figures (see `figures`): the page's inputs, and a rectangle as two steps."""

from __future__ import annotations

import math

from . import figures as fig
from .style import colors

WIDE, HIGH = 92.0, 42.0  # a panel's time axis and its force's height, px


def _polyline(points: list[tuple[float, float]], color: str, width: float = 2.0) -> str:
    return (
        "<polyline points='" + " ".join(f"{x:.1f},{y:.1f}" for x, y in points) + f"' fill='none' stroke='{color}' "
        f"stroke-width='{width}' stroke-linejoin='round'/>"
    )


def _axes(x0: float, y0: float, name: str) -> str:
    """A panel's time axis from (x0, y0), its force axis up from it, and its name below."""
    muted = colors.muted
    return "".join(
        [
            fig.arrow(x0, y0, x0 + WIDE + 8, y0, muted, 1.0),
            fig.arrow(x0, y0, x0, y0 - HIGH - 14, muted, 1.0),
            fig.math_label(x0 + WIDE + 4, y0 + 15, "t", muted),
            fig.text(x0 + WIDE / 2, y0 + 30, name, muted),
        ]
    )


def _level(x0: float, y0: float) -> str:
    """The force's level F, marked on its axis."""
    return fig.line(x0 - 4, y0 - HIGH, x0, y0 - HIGH, colors.muted, 1.0) + fig.math_label(
        x0 - 7, y0 - HIGH + 5, "F", colors.foreground, "end"
    )


def _length(x0: float, y0: float, at: float, label: str = "T") -> str:
    """The length T, marked on the time axis at `at` px along."""
    return fig.line(x0 + at, y0, x0 + at, y0 + 4, colors.muted, 1.0) + fig.math_label(
        x0 + at, y0 + 16, label, colors.foreground
    )


def pulses_svg() -> str:
    """The page's five inputs: an impulse I, a step F, a ramp to F over T, a rectangle and a half-sine of length T."""
    force = colors.force
    parts = []
    top, bottom = 72.0, 172.0  # the two rows' time axes
    # an impulse: an arrow at t = 0, its area I
    x0 = 24.0
    parts += [
        _axes(x0, top, "impulse"),
        fig.arrow(x0 + 3, top, x0 + 3, top - HIGH, force, 2.5),
        fig.math_label(x0 + 12, top - HIGH + 10, "I", colors.foreground, "start"),
    ]
    # a step: F from t = 0 on
    x0 = 148.0
    parts += [
        _axes(x0, top, "step"),
        _level(x0, top),
        _polyline([(x0, top), (x0, top - HIGH), (x0 + WIDE, top - HIGH)], force),
    ]
    # a ramp to F over T, then held
    x0 = 272.0
    parts += [
        _axes(x0, top, "ramp, then hold"),
        _level(x0, top),
        _polyline([(x0, top), (x0 + 50, top - HIGH), (x0 + WIDE, top - HIGH)], force),
        _length(x0, top, 50),
    ]
    # a rectangle of length T
    x0 = 86.0
    parts += [
        _axes(x0, bottom, "rectangular pulse"),
        _level(x0, bottom),
        _polyline([(x0, bottom), (x0, bottom - HIGH), (x0 + 50, bottom - HIGH), (x0 + 50, bottom)], force),
        _polyline([(x0 + 50, bottom), (x0 + WIDE, bottom)], force),
        _length(x0, bottom, 50),
    ]
    # a half-sine F sin(πt/T) of length T
    x0 = 228.0
    arch = [(x0 + 50 * i / 40, bottom - HIGH * math.sin(math.pi * i / 40)) for i in range(41)]
    parts += [
        _axes(x0, bottom, "half-sine pulse"),
        _level(x0, bottom),
        _polyline([*arch, (x0 + WIDE, bottom)], force),
        _length(x0, bottom, 50),
    ]
    return fig.svg(380, 215, "".join(parts))


def steps_svg() -> str:
    """A rectangular pulse is a step F at 0 less a step F at T: its response is two step responses' difference."""
    force, fg, muted = colors.force, colors.foreground, colors.muted
    y0 = 70.0
    parts = []
    # the pulse
    x0 = 20.0
    parts += [
        _axes(x0, y0, ""),
        _level(x0, y0),
        _polyline([(x0, y0), (x0, y0 - HIGH), (x0 + 50, y0 - HIGH), (x0 + 50, y0), (x0 + WIDE, y0)], force),
        _length(x0, y0, 50),
    ]
    parts.append(fig.text(140, y0 - 16, "=", fg, size=fig.FONT + 6))
    # a step at 0
    x0 = 156.0
    w = 84.0
    parts += [
        fig.arrow(x0, y0, x0 + w, y0, muted, 1.0),
        fig.arrow(x0, y0, x0, y0 - HIGH - 14, muted, 1.0),
        _level(x0, y0),
        _polyline([(x0, y0), (x0, y0 - HIGH), (x0 + w - 4, y0 - HIGH)], force),
    ]
    parts.append(fig.text(258, y0 - 16, "−", fg, size=fig.FONT + 6))
    # a step at T
    x0 = 278.0
    parts += [
        fig.arrow(x0, y0, x0 + w, y0, muted, 1.0),
        fig.arrow(x0, y0, x0, y0 - HIGH - 14, muted, 1.0),
        _polyline([(x0, y0), (x0 + 40, y0), (x0 + 40, y0 - HIGH), (x0 + w - 4, y0 - HIGH)], force),
        _length(x0, y0, 40),
        fig.math_label(x0 + w - 2, y0 + 15, "t", muted),
    ]
    return fig.svg(380, 100, "".join(parts))


FIGURES = {"fig:pulses": pulses_svg, "fig:steps": steps_svg}
