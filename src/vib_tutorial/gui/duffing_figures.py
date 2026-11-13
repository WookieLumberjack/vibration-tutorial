"""The Duffing & Mathieu page's Theory figures (see `figures`): the three springs' forces and energies, hardening,
linear and softening, with the softening one's peak force and hilltop (its escape amplitude); and a pumped
spring's timing, the swing at half the pump's frequency against the stiffness's pulse, from the page's own
`Duffing`."""

from __future__ import annotations

import math

import numpy as np

from ..core.duffing import Duffing
from . import figures as fig
from .style import colors

K = 400.0
K3 = 2e4  # N/m³: the softening preset's |k₃|, so its escape amplitude is 141 mm
SPRINGS = (Duffing(cubic=K3), Duffing(cubic=0.0), Duffing(cubic=-K3))
PUMP_LAG = -math.pi / 4  # the growing swing's lag at the tongue's centre, x = a cos(Ωt/2 + π/4)


def springs_svg() -> str:
    """Left, the force kx + k₃x³ of the three springs against x; right, their energy kx²/2 + k₃x⁴/4. The
    softening spring's force peaks at x_e/√3 and is nil again at the escape amplitude x_e, its energy's hilltop."""
    fg, muted = colors.foreground, colors.muted
    hues = (colors.mass[0], muted, colors.mass[1])
    names = ("hardening", "linear", "softening")
    soft = SPRINGS[2]
    xe = soft.escape
    xs = np.linspace(0.0, 1.06 * xe, 121)
    parts = []
    w, h, y0 = 116.0, 104.0, 128.0
    panels = (
        (30.0, "force", lambda o: o.restoring(xs), float(soft.restoring(xe / math.sqrt(3))) * 2.4),
        (178.0, "energy", lambda o: o.potential(xs), float(soft.potential(xe)) * 2.4),
    )
    for x0, title, value, top in panels:

        def at(x: float, v: float, x0: float = x0, top: float = top) -> tuple[float, float]:
            return x0 + w * x / xs[-1], y0 - h * max(min(v / top, 1.05), -0.06)

        parts += [
            fig.arrow(x0, y0, x0 + w + 10, y0, muted, 1.0),
            fig.arrow(x0, y0 + 10, x0, y0 - h - 12, muted, 1.0),
            fig.math_label(x0 + w + 10, y0 + 16, "x", fg, "end"),
            fig.text(x0 + 6, y0 - h - 6, title, fg, "start"),
        ]
        for osc, hue in zip(SPRINGS, hues, strict=True):
            vals = value(osc)
            keep = vals <= 1.05 * top
            pts = [at(float(x), float(v)) for x, v in zip(xs[keep], vals[keep], strict=True)]
            parts.append(fig.curve(pts, hue, 2.0, "4,3" if osc.cubic == 0 else ""))
        ex = at(xe, 0.0)[0]
        parts += [fig.line(ex, y0 - 4, ex, y0 + 4, fg, 1.2), fig.math_label(ex, y0 + 18, "x<sub>e</sub>", fg)]
    # the softening force's peak, and the hilltop
    px, py = (
        panels[0][0] + w / math.sqrt(3) * xe / xs[-1],
        y0 - h * float(soft.restoring(xe / math.sqrt(3))) / panels[0][3],
    )
    hx, hy = panels[1][0] + w * xe / xs[-1], y0 - h * float(soft.potential(xe)) / panels[1][3]
    parts += [
        fig.circle(px, py, 3.0, hues[2], hues[2]),
        fig.line(px, py, px, y0, hues[2], 1.0, "2,2"),
        fig.math_label(px, y0 + 18, "x<sub>e</sub>/√3", fg),
        fig.circle(hx, hy, 3.0, hues[2], hues[2]),
    ]
    for i, (name, hue) in enumerate(zip(names, hues, strict=True)):
        parts.append(fig.text((30, 118, 186)[i], 172, name, hue, "start"))
    return fig.svg(320, 180, "".join(parts))


def pump_svg() -> str:
    """One of the swing's cycles, x = a cos(Ωt/2 + π/4), below the stiffness k(1 + μ cos Ωt), the stiff
    halves shaded: stiff as the mass swings back toward the middle, soft as it swings out."""
    fg, muted, mass, force = colors.foreground, colors.muted, colors.mass[0], colors.force
    x0, w = 26.0, 268.0
    t = np.linspace(0.0, 4.0 * math.pi, 241)  # Ωt over one of the swing's cycles (two of the pump's)
    ks, kh = 40.0, 16.0  # the stiffness's axis and swing
    xs, xh = 112.0, 40.0  # the displacement's

    def px(u: float) -> float:
        return x0 + w * u / t[-1]

    parts = []
    for j in range(3):  # cos Ωt > 0: within a quarter of the pump's cycle of each peak
        lo, hi = max(2 * math.pi * j - math.pi / 2, 0.0), min(2 * math.pi * j + math.pi / 2, t[-1])
        parts.append(fig.rect(px(lo), 18, px(hi) - px(lo), 140, force, force, 0.0, opacity=0.12))
    parts += [
        fig.line(x0, ks, x0 + w, ks, muted, 1.0, "3,3"),
        fig.curve([(px(u), ks - kh * math.cos(u)) for u in t], force, 2.0),
        fig.line(x0, xs, x0 + w + 8, xs, muted, 1.0),
        fig.curve([(px(u), xs - xh * math.cos(u / 2 - PUMP_LAG)) for u in t], mass, 2.0),
        fig.math_label(x0 - 6, ks + 5, "k", force, "end"),
        fig.math_label(x0 - 6, xs + 5, "x", mass, "end"),
        fig.math_label(x0 + w + 8, xs + 16, "t", fg, "end"),
        fig.text(px(2 * math.pi), 170, "stiff, swinging back", force, "middle"),
    ]
    return fig.svg(310, 176, "".join(parts))


FIGURES = {"fig:cubic": springs_svg, "fig:pump": pump_svg}
