"""The Variability & localization page's Theory figures (see `figures`): the ring of blades (each on its spring to
the disc, neighbours joined by weak springs, with three of them unrolled), and two blades sharing a mode, its
amplitudes cos θ and sin θ as the mistuning grows past the coupling."""

from __future__ import annotations

import math

import numpy as np

from ..core.variability import Ring, participation, ring_modes
from . import figures as fig
from .style import colors


def _along(x1: float, y1: float, x2: float, y2: float, color: str, coils: int, amplitude: float) -> str:
    """A spring from (x1, y1) to (x2, y2), at any angle."""
    length = math.hypot(x2 - x1, y2 - y1)
    deg = math.degrees(math.atan2(y2 - y1, x2 - x1))
    body = fig.spring(0.0, length, 0.0, color, coils=coils, amplitude=amplitude)
    return f"<g transform='translate({x1:.1f} {y1:.1f}) rotate({deg:.1f})'>{body}</g>"


def ring_svg() -> str:
    """Left: twelve blades round a disc held still, each on a spring k to it, neighbours joined by springs k_c
    (the ring closes). Right: three neighbours unrolled, blade j between j − 1 and j + 1."""
    fg, muted, mass = colors.foreground, colors.muted, colors.mass[0]
    cx, cy, hub, rb, half, n = 78.0, 86.0, 18.0, 62.0, 6.0, 12
    parts = [fig.circle(cx, cy, hub, muted, colors.background, 1.5)]
    for j in range(n):
        a = 90.0 - 360.0 * j / n
        x0, y0 = fig.point(cx, cy, hub, a)
        x1, y1 = fig.point(cx, cy, rb - half, a)
        parts.append(_along(x0, y0, x1, y1, fg, 3, 2.5))
        b = a - 360.0 / n
        p0 = fig.point(cx, cy, rb, a - 9)
        p1 = fig.point(cx, cy, rb, b + 9)
        parts.append(_along(*p0, *p1, muted, 2, 2.0))
    for j in range(n):
        bx, by = fig.point(cx, cy, rb, 90.0 - 360.0 * j / n)
        parts.append(fig.rect(bx - half, by - half, 2 * half, 2 * half, mass, mass, 1.5, opacity=0.35))
    parts += [
        fig.text(cx, cy + 4, "disc", fg, size=10),
        fig.math_label(cx + 12, cy - 38, "k", fg),
        fig.math_label(cx + 46, cy - 62, "k<sub>c</sub>", muted),
    ]

    floor_y, top, w, h, gap = 136.0, 58.0, 20.0, 18.0, 19.0
    x = 162.0
    parts.append(fig.floor(x - 6, x + 3 * w + 4 * gap - 2, floor_y, muted))
    parts.append(fig.text(x + 1.5 * w + 2 * gap - 8, floor_y + 22, "the disc, held", muted))
    lefts = [x + gap + i * (w + gap) for i in range(3)]
    parts.append(_along(x, top + h / 2, lefts[0], top + h / 2, muted, 2, 3.0))
    for i, left in enumerate(lefts):
        mid = left + w / 2
        parts += [
            fig.upright(mid, top + h, floor_y, lambda a, b, y: fig.spring(a, b, y, fg, coils=3, amplitude=4.0)),
            fig.rect(left, top, w, h, mass, mass, 2.0, opacity=0.25),
            fig.math_label(mid, top - 8, ("j − 1", "j", "j + 1")[i], fg, size=12),
        ]
        right = lefts[i + 1] if i < 2 else left + w + 0.7 * gap
        parts.append(_along(left + w, top + h / 2, right, top + h / 2, muted, 2, 3.0))
    parts += [
        fig.math_label(lefts[1] + w / 2 + 9, 112, "k", fg, "start"),
        fig.math_label(lefts[1] + w + gap / 2, top + h + 14, "k<sub>c</sub>", muted),
        fig.math_label(lefts[1] + w / 2, top + h / 2 + 5, "m", fg, size=12),
    ]
    return fig.svg(302, 170, "".join(parts))


RATIOS = (0.0, 1.0, 5.0)  # kΔ/k_c, the two blades' difference over their coupling


def pair_modes() -> list[tuple[float, float, float]]:
    """For each ratio, the lower mode of two blades coupled by k_c = 1 % of k, their stiffness factors differing
    by Δ: (amplitude on the softer blade, on the stiffer, participation number), the shape scaled to unit
    length."""
    out = []
    ring = Ring(blades=2, coupling=0.01)
    for r in RATIOS:
        d = r * ring.coupling
        _, phi = ring_modes(ring, np.array([-d / 2, d / 2]))
        v = phi[:, 0] / math.hypot(*phi[:, 0])
        v = v if v[0] > 0 else -v
        out.append((float(v[0]), float(v[1]), float(participation(phi)[0])))
    return out


def pair_svg() -> str:
    """The lower mode of two coupled blades, as bars, for kΔ/k_c = 0, 1 and 5: shared alike, then mostly on the
    softer blade (θ = 31.7°), then almost all of it (θ = 10.9°)."""
    fg, muted, mass = colors.foreground, colors.muted, colors.mass[0]
    base, scale, bar = 104.0, 70.0, 22.0
    parts = []
    for k, ((a, b, p), r) in enumerate(zip(pair_modes(), RATIOS, strict=True)):
        x0 = 22.0 + 96.0 * k
        parts.append(fig.line(x0 - 6, base, x0 + 2 * bar + 18, base, muted, 1.0))
        for i, v in enumerate((a, b)):
            x = x0 + i * (bar + 12)
            parts.append(fig.rect(x, base - scale * v, bar, scale * v, mass, mass, 1.5, opacity=0.35))
        parts += [
            fig.math_label(x0 + bar + 8, 24, "kΔ/k<sub>c</sub>", fg, "end", size=12),
            fig.text(x0 + bar + 10, 24, f"= {r:g}", fg, "start"),
            fig.text(x0 + bar / 2, base + 16, "soft", muted, size=10),
            fig.text(x0 + 1.5 * bar + 12, base + 16, "stiff", muted, size=10),
            fig.math_label(x0 + bar - 2, base + 34, "P", fg, "end", size=12),
            fig.text(x0 + bar, base + 34, f"= {p:.2f}", fg, "start"),
        ]
    return fig.svg(302, 148, "".join(parts))


FIGURES = {"fig:ring": ring_svg, "fig:pair": pair_svg}
