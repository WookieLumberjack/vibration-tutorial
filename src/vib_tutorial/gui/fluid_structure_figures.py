"""The Fluid–structure page's Theory figures (see `figures`): a plate mode's cells, the fluid flowing from the
ones pushing out to the ones pulling back, no further out than about a cell; and a sloshing tank beside its
spring–mass analogue, the rigid mass m₀ fixed to the tank and the odd modes' masses on springs, drawn to the
default tank's masses (`sloshing.slosh_mass`)."""

from __future__ import annotations

import math

import numpy as np

from ..core.sloshing import Tank, rigid_mass, slosh_mass
from . import figures as fig
from .style import colors

TANK = Tank(1.0, 0.5, 1.0)  # the page's default: 1 m long, filled to 0.5 m


def cells_svg() -> str:
    """A plate in section bending in three half waves, its rest line dashed; above it the fluid's flow, arcs from
    each cell pushing out to its neighbours pulling back, reaching about a cell's width out."""
    fg, muted, mass, force = colors.foreground, colors.muted, colors.mass[0], colors.force
    x0, w, y0, amp = 30.0, 250.0, 118.0, 14.0
    cells = 3
    xs = np.linspace(0.0, 1.0, 121)
    shape = [(x0 + w * float(u), y0 - amp * math.sin(cells * math.pi * float(u))) for u in xs]
    cell = w / cells
    parts = [
        fig.line(x0 - 10, y0, x0 + w + 10, y0, muted, 1.0, "3,3"),
        fig.curve(shape, mass, 3.0),
        fig.rect(x0 - 14, y0 - 4, 4, 30, fg, fg, 1.0),
        fig.rect(x0 + w + 10, y0 - 4, 4, 30, fg, fg, 1.0),
        fig.text(x0 + w / 2, y0 + 30, "plate (a mode with three cells)", mass),
        fig.text(x0 + w + 12, 22, "fluid", muted, "end"),
    ]
    for j in range(cells - 1):  # over each node: from the cell pushing out (up) to the one pulling back
        xa, xb = x0 + cell * (j + 0.5), x0 + cell * (j + 1.5)
        up = j % 2 == 0  # cell j bulges up (into the fluid), cell j + 1 down
        start, end = (xa, xb) if up else (xb, xa)
        rx, ry = cell / 2, cell * 0.42
        cy = y0 - 2 * amp
        sweep = 1 if start < end else 0
        parts.append(
            f"<path d='M {start:.1f} {cy:.1f} A {rx:.1f} {ry:.1f} 0 0 {sweep} {end:.1f} {cy:.1f}' fill='none' "
            f"stroke='{force}' stroke-width='1.6'/>"
        )
        parts.append(fig.head(end, cy, math.pi / 2, force))
    top = y0 - 2 * amp - cell * 0.42
    parts += [
        fig.line(x0 + w + 2, y0 - 2 * amp, x0 + w + 2, top, fg, 1.0),
        fig.text(x0 + w + 6, top - 6, "≈ a cell", fg, "end"),
    ]
    return fig.svg(310, 160, "".join(parts))


def analogue_svg() -> str:
    """Left, the default tank (L = 1 m, d = 0.5 m) sloshing in its first mode, the surface tilted; right, the
    analogue: m₀ fixed to the tank, m₁ and m₃ on springs to its walls, each block's area in proportion to its mass."""
    fg, muted, mass, force = colors.foreground, colors.muted, colors.mass[0], colors.force
    water = colors.mass[1]
    # the tank
    x0, y0, L, d = 20.0, 140.0, 110.0, 55.0
    tilt = 16.0
    surface = [(x0, y0 - d - tilt / 2), (x0 + L, y0 - d + tilt / 2), (x0 + L, y0), (x0, y0)]
    parts = [
        fig.area(surface, water, 0.3),
        fig.line(x0, y0 - d - tilt / 2, x0 + L, y0 - d + tilt / 2, water, 2.0),
        fig.line(x0, y0 - d, x0 + L, y0 - d, muted, 1.0, "3,3"),
        fig.curve([(x0, y0 - 100), (x0, y0), (x0 + L, y0), (x0 + L, y0 - 100)], fg, 2.0),
        fig.math_label(x0 + L / 2, y0 + 18, "L", fg),
        fig.math_label(x0 + L + 10, y0 - d / 2 + 5, "d", fg, "start"),
        fig.text(154, y0 - 46, "=", fg, size=20),
    ]
    # the analogue
    ax, w = 178.0, 116.0
    m0, m1, m3 = rigid_mass(TANK), float(slosh_mass(TANK, 1)), float(slosh_mass(TANK, 3))
    side = 34.0 / math.sqrt(max(m0, m1))  # px per √kg, so a block's area goes as its mass

    def block(cx: float, cy: float, m: float, label: str) -> list[str]:
        s = max(side * math.sqrt(m), 9.0)
        return [
            fig.rect(cx - s / 2, cy - s / 2, s, s, mass, mass, 1.5, opacity=0.35),
            fig.math_label(cx, cy + 5 if s > 20 else cy - s / 2 - 5, label, fg),
        ]

    s0 = side * math.sqrt(m0)
    parts += [fig.curve([(ax, y0 - 112), (ax, y0), (ax + w, y0), (ax + w, y0 - 112)], fg, 2.0)]
    parts += block(ax + w / 2, y0 - s0 / 2, m0, "m<sub>0</sub>")
    parts += [fig.line(ax + w / 2 - s0 / 2 - 6, y0 - 1, ax + w / 2 + s0 / 2 + 6, y0 - 1, fg, 2.0)]
    for m, y, label, spring in (
        (m1, y0 - s0 - 30, "m<sub>1</sub>", "k<sub>1</sub>"),
        (m3, y0 - s0 - 66, "m<sub>3</sub>", "k<sub>3</sub>"),
    ):
        s = max(side * math.sqrt(m), 9.0)
        cx = ax + w / 2
        parts += [
            fig.spring(ax, cx - s / 2, y, force, coils=4, amplitude=4.0),
            fig.spring(cx + s / 2, ax + w, y, force, coils=4, amplitude=4.0),
        ]
        parts += block(cx, y, m, label)
        parts.append(fig.math_label(ax + 14, y - 8, spring, force))
    return fig.svg(310, 180, "".join(parts))


FIGURES = {"fig:cells": cells_svg, "fig:analogue": analogue_svg}
