"""A page's matrices drawn as `matrix_table`'s labeled tables, in a tab of their own (Coordinates & bases, State-space
modes).

`MatricesLabel` is a `LiveLabel` given two versions of its rich text, a wide one (tables side by side) and a tall
one (the widest of them one above the other): it shows the wide one where it fits the label's width and the tall
one where it doesn't, picked again as it's resized, so no table runs off the pane's edge at 1280 px.
`scrolling(widget)` puts a tab's contents in a scroll area that scrolls up and down, never sideways.

Working on it: `tests/test_state_space_page.py` and `tests/test_bases_page.py` resize it and check the pick.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from .live_label import LiveLabel


class MatricesLabel(LiveLabel):
    """See the module: `set_variants(wide, tall)`."""

    def __init__(self) -> None:
        super().__init__(wrap=True)
        self._variants = ("", "")
        self._measure = QtGui.QTextDocument(self)
        self._measure.setDocumentMargin(0)

    def set_variants(self, wide: str, tall: str) -> None:
        self._variants = (wide, tall)
        self._pick()

    def _pick(self) -> None:
        wide, tall = self._variants
        self._measure.setDefaultFont(self.font())
        self._measure.setTextWidth(max(self.width(), 1))  # its text wraps there; only a table can stick out
        self._measure.setHtml(wide)
        text = wide if self._measure.idealWidth() <= self.width() + 0.5 else tall
        if text != self.text():
            self.setText(text)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._pick()


def scrolling(widget: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
    """`widget` in a frameless scroll area, up and down only: a tall step goes below, not off the edge."""
    scroll = QtWidgets.QScrollArea()
    scroll.setWidget(widget)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    return scroll
