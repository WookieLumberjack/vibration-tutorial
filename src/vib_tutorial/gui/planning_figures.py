"""The Test planning page's Theory figures (see `figures`): the noise each placement leaves on the fitted modal
coordinates, as ellipses (smaller where the Fisher information's determinant is larger), and an ODS measured
with a reference sensor that stays put while another roves."""

from __future__ import annotations

import numpy as np

from ..core.model import ChainSystem
from ..core.test_planning import undamped_modes
from . import figures as fig
from .style import colors

PLACEMENTS = (("every mass", (0, 1, 2, 3)), ("EfI: m1, m3, m4", (0, 2, 3)), ("bunched: m1–m3", (0, 1, 2)))


def ellipse_axes() -> list[np.ndarray]:
    """For each placement on the uniform 4-mass chain (its first three modes the targets), the 2 × 2 covariance
    of the fitted η̂1, η̂2 per unit sensor noise: the top-left of (Φ_sᵀΦ_s)⁻¹."""
    _, phi = undamped_modes(ChainSystem.uniform(4))
    T = phi[:, :3]
    return [np.linalg.inv(T[list(s)].T @ T[list(s)])[:2, :2] for _, s in PLACEMENTS]


def ellipse_svg() -> str:
    """The three placements' one-σ noise ellipses for modes 1 and 2, centred on the true coordinates: a circle with
    a sensor on every mass, a little larger and tilted with EfI's three, long and thin with three bunched at the
    ground, where modes 1 and 2 look alike."""
    fg, muted = colors.foreground, colors.muted
    cx, cy, unit = 86.0, 80.0, 15.0
    shades = (fg, colors.mass[0], colors.force)
    parts = [
        fig.arrow(cx - 80, cy, cx + 82, cy, muted, 1.0),
        fig.arrow(cx, cy + 66, cx, cy - 68, muted, 1.0),
        fig.math_label(cx + 74, cy + 20, "η̂<sub>1</sub>", fg, size=13),
        fig.math_label(cx + 14, cy - 58, "η̂<sub>2</sub>", fg, size=13),
    ]
    t = np.linspace(0.0, 2.0 * np.pi, 73)
    for k, (cov, colour) in enumerate(zip(ellipse_axes(), shades, strict=True)):
        L = np.linalg.cholesky(cov)
        xy = L @ np.vstack([np.cos(t), np.sin(t)])
        pts = [(cx + unit * a, cy - unit * b) for a, b in xy.T]
        parts.append(fig.curve(pts, colour, 2.0 if k else 1.4))
        name = PLACEMENTS[k][0]
        parts.append(fig.text(178, 50 + 22 * k, name, colour, "start"))
    parts.append(fig.text(178, 128, "one σ of noise", muted, "start"))
    return fig.svg(302, 150, "".join(parts))


def roving_svg() -> str:
    """The uniform 4-mass chain running under a force nobody measures on m4; a reference accelerometer on m4 that
    stays, and a roving one moved from mass to mass (dashed where it goes next)."""
    fg, muted, mass, force, strong = colors.foreground, colors.muted, colors.mass[0], colors.force, colors.strong
    y, wall, gap, w, h = 62.0, 10.0, 30.0, 32.0, 28.0
    parts = [fig.ground(wall, y - 22, y + 22, muted)]
    x = wall
    mids = []
    for i in range(4):
        left = x + gap
        mid = left + w / 2
        mids.append(mid)
        parts += [
            fig.spring(x, left, y, fg, coils=3, amplitude=5.0),
            fig.rect(left, y - h / 2, w, h, mass, mass, 2.0, opacity=0.2),
            fig.math_label(mid, y + 6, f"m<sub>{i + 1}</sub>", fg),
        ]
        x = left + w
    top = y - h / 2
    parts += [
        fig.arrow(mids[3] - 14, top - 12, mids[3] + 16, top - 12, force, 1.8),
        fig.text(mids[3] - 20, top - 8, "running force", force, "end"),
        fig.rect(mids[3] - 5, y + h / 2, 10, 9, strong, strong, 1.0),
        fig.text(mids[3], y + h / 2 + 26, "reference", strong),
        fig.text(mids[3], y + h / 2 + 40, "stays", muted),
        fig.rect(mids[0] - 5, y + h / 2, 10, 9, fg, fg, 1.0),
    ]
    for m in mids[1:3]:
        parts.append(fig.rect(m - 5, y + h / 2, 10, 9, muted, muted, 1.0, opacity=0.0))
    parts += [
        fig.arrow(mids[0] + 4, y + h / 2 + 22, mids[2] - 4, y + h / 2 + 22, muted, 1.2, dash="3,3"),
        fig.text((mids[0] + mids[2]) / 2, y + h / 2 + 40, "roving", fg),
    ]
    return fig.svg(302, 136, "".join(parts))


FIGURES = {"fig:noise": ellipse_svg, "fig:roving": roving_svg}
