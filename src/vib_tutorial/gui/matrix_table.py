"""A matrix as the notes show it: a rich-text table with labeled rows and columns, its cells tinted by block
(#409 S-13: the CMS, coupling and rotor notes drew their own).

A cell's tint is its pair of groups' (`colors.block_tints`, keyed by the sorted pair: interior "i", boundary "b",
kept modes "q"); an entry is `text.entry` (four figures, 0 where it's negligible beside the matrix's largest),
grayed when it's 0 (`table_html` takes them written already, and marks a coupling term in bold). `dim_cols` grays the columns from it on (discarded modes); `nowrap` keeps a wide matrix's
numbers whole, so it scrolls sideways rather than breaking them (the rotor's). Qt-free: the course's data.

Working on it: `tests/test_view_models.py` checks the entries; the CMS, coupling and rotor tests draw them.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..view_models.text import entry
from .style import colors


def matrix_html(
    A: np.ndarray,
    rows: list[str],
    cols: list[str],
    row_groups: list[str],
    col_groups: list[str],
    title: str = "",
    dim_cols: int | None = None,
    nowrap: bool = False,
) -> str:
    """An HTML table of A with labeled rows and columns, its cells tinted by block (see the module's docstring)."""
    scale = float(np.abs(A).max()) if A.size else 1.0
    texts = [[entry(float(A[r, c]), scale) for c in range(len(cols))] for r in range(len(rows))]
    return table_html(texts, rows, cols, row_groups, col_groups, title, dim_cols, nowrap)


def table_html(
    texts: Sequence[Sequence[str]],
    rows: Sequence[str],
    cols: Sequence[str],
    row_groups: Sequence[str],
    col_groups: Sequence[str],
    title: str = "",
    dim_cols: int | None = None,
    nowrap: bool = False,
    marked: Sequence[Sequence[bool]] | None = None,
    lit: Sequence[int] = (),
    lit_color: str = "",
) -> str:
    """`matrix_html` with its entries already written (`texts`): a page that rounds its own (State-space modes,
    three figures, so two 4×4s fit side by side). `marked` entries are bold in `colors.force` (a coupling term); the
    `lit` columns' heads are in `lit_color` and their entries bold (the pair whose shape is being dragged)."""
    wrap = "white-space:nowrap" if nowrap else ""
    th = f"<th style='{wrap}'>" if nowrap else "<th>"
    head = "".join(
        f"<th style='{wrap}; color:{lit_color}'>{c}</th>" if k in lit else f"{th}{c}</th>" for k, c in enumerate(cols)
    )
    out = [f"<table border='1' cellspacing='0' cellpadding='3'><tr>{th}{title}</th>{head}</tr>"]
    for r, (label, rg) in enumerate(zip(rows, row_groups)):
        cells = []
        for c, cg in enumerate(col_groups):
            dim = dim_cols is not None and c >= dim_cols
            tint = colors.dim_tint if dim else colors.block_tints[tuple(sorted((rg, cg)))]
            text = texts[r][c]
            hot = marked is not None and marked[r][c]
            colour = (
                f"color:{colors.force}; font-weight:bold"
                if hot
                else (f"color:{colors.grey}" if text == "0" or dim else "")
            )
            bold = "font-weight:bold" if c in lit and not hot else ""
            styles = [s for s in (wrap, colour, bold) if s]
            style = f" style='{'; '.join(styles)}'" if styles else ""
            cells.append(f"<td align='right' bgcolor='{tint}'{style}>{text}</td>")
        out.append(f"<tr>{th}{label}</th>{''.join(cells)}</tr>")
    out.append("</table>")
    return "".join(out)


def side_by_side(*tables: str) -> str:
    """Tables in a row, top-aligned."""
    cells = "".join(f"<td valign='top' style='padding-right:14px'>{t}</td>" for t in tables)
    return f"<table cellspacing='0' cellpadding='0'><tr>{cells}</tr></table>"
