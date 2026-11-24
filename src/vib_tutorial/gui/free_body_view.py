"""The live free-body diagram (#478 item 2) in a pane of its own, beside the moving mass: the mass held still, its
forces as arrows out of it, a row each, so it reads while the body swings.

Each force keeps its row, and its row's words keep their places: the symbol in a column on the left (`kx`, `cẋ`,
`f`, …), the size in a column on the right, and between them the mass, the arrow leaving the side the force pushes
towards. Only the arrows move: they turn round as a force changes direction, and their lengths follow the forces
on `free_body.ArrowScale`'s steady scale. The net force, m·ẍ, is the last row, dashed, under a rule (what the
forces above sum to). The sizes are rewritten at most `TEXT_EVERY` s apart, so they can be read while it runs (the
last always shows within that).

Drawn on the body itself, the arrows and their labels moved with the mass and swapped sides as each force turned
(the check-in's first finding): this pane replaced them.

Working on it: `tests/test_sdof_page.py` reads its rows. Colours are theme roles' (`style.role`), so `apply_theme`
redraws them; the arrows' room is in the view's own units, the words' columns pinned to its edges.
"""

from __future__ import annotations

import time

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..view_models.free_body import Arrow, FreeBody
from ..view_models.text import fmt
from .style import colors, role

HALF = 0.45  # the mass's half-width
ROOM = 0.85  # the largest arrow's length, either side of the mass
EDGE = HALF + ROOM + 0.08  # where the words' columns start
SPAN = 2.5  # the view's half-width: the columns get SPAN − EDGE each
GAP = 0.42  # between rows
TEXT_EVERY = 0.2  # s between rewrites of the sizes
TIP = (
    "<p>The forces on the mass now, a row each: the arrow leaves the side the force pushes it towards, its length "
    "the force's size (the scale steady while they change). The dashed row is their sum, the mass times its "
    "acceleration, m·ẍ.</p>"
)


class FreeBodyView(pg.PlotWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAccessibleName("The forces on the mass")
        self.setToolTip(TIP)
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        self.hideAxis("left")
        self.hideAxis("bottom")
        self.setAspectLocked(True)
        self.setMinimumWidth(260)
        self.block = QtWidgets.QGraphicsRectItem()
        self.block.setZValue(2)
        self.addItem(self.block)
        self.rule = pg.PlotDataItem()
        self.addItem(self.rule)
        self.shafts: list[pg.PlotDataItem] = []
        self.heads: list[pg.ArrowItem] = []
        self.symbols: list[pg.TextItem] = []
        self.sizes: list[pg.TextItem] = []
        self._arrows: tuple[Arrow, ...] = ()
        self._written = -1.0  # time.monotonic() the sizes were last written
        self._later = QtCore.QTimer(self)  # writes the last sizes when a rewrite was held back
        self._later.setSingleShot(True)
        self._later.timeout.connect(self._write_sizes)
        self.apply_theme()

    def apply_theme(self) -> None:
        self.block.setBrush(pg.mkBrush(colors.mass[0]))
        self.block.setPen(pg.mkPen(colors.structure, width=1.5))
        self.rule.setPen(pg.mkPen(colors.faint, width=1))
        if self._arrows:
            self._build(self._arrows)

    def rows(self) -> list[tuple[str, str]]:
        """Each row's (symbol, size) as shown (tests read them)."""
        return [(s.toPlainText(), z.toPlainText()) for s, z in zip(self.symbols, self.sizes)]

    def show(self, body: FreeBody) -> None:
        arrows = (*body.arrows, body.net)
        if [(a.name, a.role) for a in arrows] != [(a.name, a.role) for a in self._arrows]:
            self._build(arrows)
        n = len(arrows)
        for i, (arrow, shaft, head) in enumerate(zip(arrows, self.shafts, self.heads)):
            y = self._row_y(i, n)
            length = float(body.length(arrow)) * ROOM
            start = HALF if length >= 0 else -HALF
            tip = start + length
            shaft.setData([start, tip], [y, y])
            shown = abs(length) > 0.02
            head.setVisible(shown)
            if shown:
                head.setPos(tip, y)
                head.setStyle(angle=180 if length > 0 else 0)
        self._arrows = arrows
        wait = TEXT_EVERY - (time.monotonic() - self._written)
        if wait <= 0:
            self._write_sizes()
        elif not self._later.isActive():
            self._later.start(int(1000 * wait) + 1)

    def _write_sizes(self) -> None:
        self._written = time.monotonic()
        for arrow, text in zip(self._arrows, self.sizes):
            text.setText(fmt(abs(arrow.value), "N"))

    @staticmethod
    def _row_y(i: int, n: int) -> float:
        """Row i of n from the top, the net's (the last) a half row further down, under the rule."""
        return -GAP * (i + (0.5 if i == n - 1 else 0.0))

    def _build(self, arrows: tuple[Arrow, ...]) -> None:
        for item in (*self.shafts, *self.heads, *self.symbols, *self.sizes):
            self.removeItem(item)
        self.shafts, self.heads, self.symbols, self.sizes = [], [], [], []
        n = len(arrows)
        for i, arrow in enumerate(arrows):
            colour = role(arrow.role)
            dash = QtCore.Qt.PenStyle.DashLine if arrow.name == "net" else QtCore.Qt.PenStyle.SolidLine
            y = self._row_y(i, n)
            shaft = pg.PlotDataItem(pen=pg.mkPen(colour, width=3, style=dash))
            head = pg.ArrowItem(angle=180, headLen=12, tipAngle=40, pen=None, brush=colour)
            symbol = pg.TextItem(arrow.symbol, color=colour, anchor=(1.0, 0.5))
            symbol.setPos(-EDGE, y)
            size = pg.TextItem(fmt(abs(arrow.value), "N"), color=colour, anchor=(0.0, 0.5))
            size.setPos(EDGE, y)
            for item in (shaft, head, symbol, size):
                item.setZValue(4)
                self.addItem(item)
            self.shafts.append(shaft)
            self.heads.append(head)
            self.symbols.append(symbol)
            self.sizes.append(size)
        last = self._row_y(n - 1, n)
        self.block.setRect(QtCore.QRectF(-HALF, last - 0.3, 2 * HALF, 0.6 - last))
        rule_y = 0.5 * (self._row_y(n - 2, n) + last) if n > 1 else last + GAP
        self.rule.setData([-SPAN + 0.1, SPAN - 0.1], [rule_y, rule_y])
        self.setRange(QtCore.QRectF(-SPAN, last - 0.5, 2 * SPAN, 1.0 - last), padding=0)
        self._written = -1.0
