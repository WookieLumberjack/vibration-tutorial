"""The Active control page's Theory figures (see `figures`): the uniform chain with a sensor on m1 and an actuator
on m4, and its first two modes' shapes there (`undamped_modes`), which agree in sign in mode 1 and disagree in
mode 2 (spillover); and a mass on a mount over a moving floor with its damper to the floor (passive) and to the
sky (skyhook)."""

from __future__ import annotations

from ..core.active_control import undamped_modes
from ..core.model import ChainSystem
from . import figures as fig
from .style import colors


def spill_svg() -> str:
    """Top: the chain, wall on the left, a velocity sensor on m1 and a force actuator on m4. Below: modes 1 and 2
    as bars at each mass, the sensor's and actuator's masses outlined."""
    fg, muted, force, strong = colors.foreground, colors.muted, colors.force, colors.strong
    hues = (colors.mass[0], colors.mass[1])
    _, phi = undamped_modes(ChainSystem.uniform(4))
    x0, step, cy, size = 30.0, 66.0, 40.0, 26.0
    xs = [x0 + 40 + step * j for j in range(4)]
    parts = [fig.ground(x0, cy - 22, cy + 22, fg)]
    prev = x0
    for j, x in enumerate(xs):
        parts.append(fig.spring(prev, x - size / 2, cy, muted, coils=3, amplitude=5.0))
        parts.append(fig.rect(x - size / 2, cy - size / 2, size, size, fg, "none", 1.5))
        parts.append(fig.math_label(x, cy + 5, f"m<sub>{j + 1}</sub>", fg, size=fig.FONT))
        prev = x + size / 2
    parts += [
        fig.circle(xs[0], cy - size / 2 - 9, 6.0, strong, "none", 1.8),
        fig.math_label(xs[0] + 10, cy - size / 2 - 10, "s", strong, "start"),
        fig.arrow(xs[3] + 34, cy, xs[3] + size / 2 + 2, cy, force, 2.0),
        fig.math_label(xs[3] + 26, cy - 8, "a", force),
    ]
    for r, (base, hue) in enumerate(zip((112.0, 168.0), hues, strict=True)):
        scale = 32.0 / 0.7
        parts.append(fig.line(xs[0] - 16, base, xs[3] + 16, base, muted, 1.0))
        parts.append(fig.text(xs[3] + 24, base + 4, f"mode {r + 1}", hue, "start"))
        for j, x in enumerate(xs):
            h = float(phi[j, r]) * scale * (1 if phi[3, 0] > 0 else -1)
            top, height = (base - h, h) if h > 0 else (base, -h)
            width = 1.8 if j in (0, 3) else 1.0
            parts.append(fig.rect(x - 8, top, 16, height, hue, hue, width, opacity=0.5))
    return fig.svg(340, 200, "".join(parts))


def skyhook_svg() -> str:
    """Left, passive: the mass on a spring and a damper, both to the moving floor. Right, skyhook: the spring to the
    floor, the damper up to a fixed point in space."""
    fg, muted, mass, force = colors.foreground, colors.muted, colors.mass[0], colors.force
    parts = []
    for cx, label, sky in ((82.0, "passive", False), (236.0, "skyhook", True)):
        top, w, h = 70.0, 64.0, 32.0
        floor_y = 150.0
        parts += [
            fig.rect(cx - w / 2, top, w, h, mass, mass, 1.5, opacity=0.35),
            fig.math_label(cx, top + h / 2 + 5, "M", fg),
            fig.upright(cx - 16, top + h, floor_y, lambda a, b, y: fig.spring(a, b, y, fg, coils=4, amplitude=5.0)),
            fig.line(cx - 40, floor_y, cx + 40, floor_y, fg, 2.0),
            fig.arrow(cx + 46, floor_y + 6, cx + 46, floor_y - 14, muted, 1.5),
            fig.math_label(cx + 52, floor_y + 2, "y", muted, "start"),
            fig.text(cx, 172, label, fg),
        ]
        if sky:
            parts += [
                fig.upright(cx + 16, 22, top, lambda a, b, y: fig.damper(a, b, y, force, half=7.0)),
                fig.line(cx - 4, 22, cx + 36, 22, fg, 2.0),
                fig.text(cx + 40, 26, "sky", muted, "start"),
            ]
        else:
            parts.append(fig.upright(cx + 16, top + h, floor_y, lambda a, b, y: fig.damper(a, b, y, force, half=7.0)))
    return fig.svg(320, 180, "".join(parts))


FIGURES = {"fig:spill": spill_svg, "fig:skyhook": skyhook_svg}
