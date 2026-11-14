"""The Rotor builder page's Theory figure (see `figures`): a stepped shaft cut into sections and elements, its
nodes, a disc of radii R and r and thickness t_d, the bearings as springs and dampers to the ground, and a
seal."""

from __future__ import annotations

from . import figures as fig
from .style import colors


def model_svg() -> str:
    """A rotor as the page builds it: three sections (the middle one bored, D and d), each cut into elements
    between nodes; a disc at a node; a bearing at each end; a seal."""
    fg, muted, faint = colors.foreground, colors.muted, colors.faint
    shaft, disc, spring = colors.mass[0], colors.mass[1], colors.force
    cy = 96.0
    # (z0, z1, half-height) of each section, and its elements
    sections = ((34.0, 110.0, 12.0, 2), (110.0, 230.0, 20.0, 3), (230.0, 306.0, 12.0, 2))
    parts = [fig.arrow(14, cy, 330, cy, faint, 1.0), fig.text(326, cy + 16, "z", fg, italic=True)]
    nodes: set[float] = set()
    for z0, z1, h, n in sections:
        parts.append(fig.rect(z0, cy - h, z1 - z0, 2 * h, shaft, shaft, 1.5, 0.25))
        for i in range(n + 1):
            nodes.add(z0 + i * (z1 - z0) / n)
    parts.append(fig.line(110, cy - 8, 230, cy - 8, shaft, 1.0, "4,3"))  # the bore
    parts.append(fig.line(110, cy + 8, 230, cy + 8, shaft, 1.0, "4,3"))
    for z in sorted(nodes):
        parts.append(fig.circle(z, cy, 2.8, fg, fg))
    # the disc at z = 190
    parts += [
        fig.rect(184, cy - 58, 12, 116, disc, disc, 1.5, 0.35),
        fig.text(190, cy - 64, "disc", fg),
        fig.line(202, cy, 202, cy - 58, muted, 1.0),
        fig.math_label(206, cy - 34, "R", fg, "start"),
    ]
    # bearings: a spring and a damper from each end node down to the ground
    for z in (34.0, 306.0):
        parts += [
            fig.upright(z - 8, cy + 12, cy + 62, lambda a, b, y: fig.spring(a, b, y, fg, 4, 5.0)),
            fig.upright(z + 9, cy + 12, cy + 62, lambda a, b, y: fig.damper(a, b, y, fg, 6.0)),
            fig.floor(z - 22, z + 24, cy + 62, fg),
        ]
    parts.append(fig.text(34, cy + 84, "bearing", fg))
    parts.append(fig.text(306, cy + 84, "bearing", fg))
    # a seal round the first section
    parts += [
        fig.rect(64, cy - 26, 20, 9, spring, spring, 1.0, 0.4),
        fig.rect(64, cy + 17, 20, 9, spring, spring, 1.0, 0.4),
        fig.text(74, cy - 32, "seal", fg),
    ]
    # labels: a section's D and d, an element, a node
    parts += [
        fig.math_label(130, cy - 26, "D", fg),
        fig.math_label(130, cy + 5, "d", fg),
        fig.text(268, cy - 22, "section", fg),
        fig.line(110, cy + 30, 150, cy + 30, muted, 1.0),
        fig.line(110, cy + 25, 110, cy + 35, muted, 1.0),
        fig.line(150, cy + 25, 150, cy + 35, muted, 1.0),
        fig.text(130, cy + 48, "element", fg),
        fig.text(258, cy + 30, "node", fg, "start"),
        fig.line(256, cy + 26, 232, cy + 3, muted, 1.0),
    ]
    return fig.svg(340, 196, "".join(parts))


FIGURES = {"fig:rotor-model": model_svg}
