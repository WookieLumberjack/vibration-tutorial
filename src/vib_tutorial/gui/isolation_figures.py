"""The Vibration isolation Theory's figures (see `figures`): force in and motion in, and the three mounts."""

from __future__ import annotations

from collections.abc import Callable

from . import figures as fig
from .style import colors


def _spring(color: str, coils: int = 4) -> Callable[[float, float, float], str]:
    return lambda a, b, y: fig.spring(a, b, y, color, coils=coils, amplitude=7.0)


def _damper(color: str) -> Callable[[float, float, float], str]:
    return lambda a, b, y: fig.damper(a, b, y, color, half=8.0)


def _block(cx: float, top: float, w: float, h: float, label: str) -> str:
    mass, fg = colors.mass[0], colors.foreground
    return fig.rect(cx - w / 2, top, w, h, mass, mass, 2.0, opacity=0.2) + fig.math_label(
        cx, top + h / 2 + 6, label, fg
    )


def _mount(cx: float, top: float, bottom: float, spread: float = 22.0) -> str:
    """A spring and a damper side by side, from `top` down to `bottom`."""
    fg = colors.foreground
    return fig.upright(cx - spread, top, bottom, _spring(fg)) + fig.upright(cx + spread, top, bottom, _damper(fg))


def two_ways_svg() -> str:
    """Left: a force F on the machine, F_floor into a fixed floor. Right: the floor moves y, the machine x."""
    fg, muted, force = colors.foreground, colors.muted, colors.force
    top, base, floor = 40.0, 85.0, 150.0
    parts = []
    # force in
    cx = 95.0
    parts += [
        _block(cx, top, 90, base - top, "M"),
        fig.arrow(cx, 8, cx, top - 3, force, 2.0),
        fig.math_label(cx + 10, 22, "F", fg, "start"),
        _mount(cx, base, floor),
        fig.math_label(cx - 42, 122, "k", fg),
        fig.math_label(cx + 42, 122, "c", fg),
        fig.floor(cx - 70, cx + 70, floor, muted),
        fig.arrow(cx + 60, floor - 30, cx + 60, floor - 2, force, 2.0),
        fig.math_label(cx + 66, floor - 16, "F<sub>" + fig.UPRIGHT.format("floor") + "</sub>", fg, "start"),
    ]
    # motion in
    cx = 270.0
    parts += [
        _block(cx, top, 90, base - top, "M"),
        fig.arrow(cx + 58, top + 25, cx + 58, top - 5, fg, 1.5),
        fig.math_label(cx + 64, top + 8, "x", fg, "start"),
        _mount(cx, base, floor),
        fig.floor(cx - 70, cx + 70, floor, muted),
        fig.arrow(cx + 75, floor + 20, cx + 75, floor - 8, fg, 1.5),
        fig.math_label(cx + 81, floor + 8, "y", fg, "start"),
    ]
    parts += [
        fig.text(95, 182, "force in", muted),
        fig.text(270, 182, "motion in", muted),
    ]
    return fig.svg(360, 190, "".join(parts))


def mounts_svg() -> str:
    """The three mounts: a spring and damper; the damper on a spring Nk (relaxation); two stages."""
    fg, muted = colors.foreground, colors.muted
    top, base, floor = 22.0, 52.0, 170.0
    parts = []
    # a spring and a damper
    cx = 60.0
    parts += [
        _block(cx, top, 70, base - top, "M"),
        _mount(cx, base, floor),
        fig.math_label(cx - 40, 115, "k", fg),
        fig.math_label(cx + 42, 115, "c", fg),
        fig.floor(cx - 45, cx + 45, floor, muted),
    ]
    # relaxation: k beside Nk in series with c
    cx = 180.0
    joint = 110.0
    parts += [
        _block(cx, top, 70, base - top, "M"),
        fig.upright(cx - 22, base, floor, _spring(fg)),
        fig.math_label(cx - 40, 115, "k", fg),
        fig.upright(cx + 22, base, joint, _spring(fg, 3)),
        fig.math_label(cx + 46, 84, "Nk", fg, "start"),
        fig.circle(cx + 22, joint, 2.5, fill=fg),
        fig.upright(cx + 22, joint, floor, _damper(fg)),
        fig.math_label(cx + 44, 145, "c", fg, "start"),
        fig.floor(cx - 45, cx + 45, floor, muted),
    ]
    # two stages
    cx = 300.0
    mid = 92.0
    parts += [
        _block(cx, top, 60, base - top, "M"),
        _mount(cx, base, mid, 16.0),
        _block(cx, mid, 80, 22, "m<sub>2</sub>"),
        _mount(cx, mid + 22, floor, 16.0),
        fig.floor(cx - 45, cx + 45, floor, muted),
    ]
    parts += [
        fig.text(60, 192, "spring and damper", muted),
        fig.text(180, 192, "relaxation", muted),
        fig.text(300, 192, "two stages", muted),
    ]
    return fig.svg(360, 200, "".join(parts))


FIGURES = {"fig:two-ways": two_ways_svg, "fig:mounts": mounts_svg}
