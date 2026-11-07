"""Tabs in groups: a row of group tabs, each with its own row of tabs below.

One row of ten tabs needs ~970 px, far more than a side pane has, so the rotor page
groups its views. Reach a page through `show_page` and `current_page`, which work
whichever group it's in, and see through a scroll area put round a page; `changed`
fires whenever the page showing changes.

Drawing that waits while a tab hides it: `Deferred` (a panel, or one tab), `LazyTabs` (a tab widget's tabs).

Working on it
-------------
The rotor page's right pane is `TabGroups` (`gui/tab_groups.py`): five group tabs, each with its own row of
tabs (ten in one row didn't fit). Reach a view with `tabs.show_page(widget)` and `tabs.current_page()`, not
`setCurrentWidget`/`currentWidget`: they work across the groups and see through the `ColumnScroll` round the
Balancing and bearing panels. Those panels draw only while their tab shows, through a `Deferred`
(`tab_groups.py`): `request()` draws now or marks it stale (`tucked_away`, which checks the tabs above them
too: a panel that's its group's current tab isn't `isHidden()` while another group shows), and it draws on the
panel's Show event or `catch_up()` (the page calls it on a tab switch, since a window never shown gets no show
events). A tab widget's tabs drawn only while they show are a `LazyTabs` (a `Deferred` per tab; `stale(*tabs)`
after a change): use it, not a hand-kept stale set. They're still built with the page, so the right pane keeps
their minimum widths. What they keep as state (the bearing panel's load box, the balancing buttons) stays
outside the deferred part; a test that reads their drawing shows the tab first.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeGuard

from PySide6 import QtCore, QtWidgets


class TabGroups(QtWidgets.QTabWidget):
    changed = QtCore.Signal()  # the page showing changed (either row)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.currentChanged.connect(self.changed)

    def add_group(self, title: str, *pages: tuple[QtWidgets.QWidget, str]) -> None:
        """A group tab: one page goes in as it is, several get their own row of tabs."""
        if len(pages) == 1:
            self.addTab(pages[0][0], title)
            return
        inner = QtWidgets.QTabWidget()
        inner.setDocumentMode(True)  # no second frame inside the first
        for page, name in pages:
            inner.addTab(page, name)
        inner.currentChanged.connect(self.changed)
        self.addTab(inner, title)

    def show_page(self, page: QtWidgets.QWidget) -> None:
        """Bring `page` to the front, in its group."""
        while page is not self:
            stack = page.parentWidget()  # a tab's page sits in its tab widget's stack
            assert stack is not None, "the page is in these tabs"
            tabs = stack.parentWidget()
            if isinstance(tabs, QtWidgets.QTabWidget) and tabs.indexOf(page) >= 0:
                tabs.setCurrentWidget(page)
            page = stack

    def current_page(self) -> QtWidgets.QWidget:
        page: QtWidgets.QWidget | None = self.currentWidget()
        if self._is_group(page):
            page = page.currentWidget()
        if isinstance(page, QtWidgets.QScrollArea):
            page = page.widget()
        assert page is not None, "every group has pages"
        return page

    def _is_group(self, widget: QtWidgets.QWidget | None) -> TypeGuard[QtWidgets.QTabWidget]:
        stack = widget.parentWidget() if isinstance(widget, QtWidgets.QTabWidget) else None
        return stack is not None and stack.parentWidget() is self


def tucked_away(widget: QtWidgets.QWidget | None) -> bool:
    """True if `widget` is hidden, or a tab (at any level) hides it, whether or not its window is shown yet."""
    while widget is not None:
        if widget.isHidden():
            return True
        widget = widget.parentWidget()
        if widget is not None and widget.isWindow():
            return False  # a window never shown is hidden, but its tabs say what would show
    return False


class Deferred(QtCore.QObject):
    """Drawing (or working out) that waits while a tab hides `widget`. `request()` draws now if it shows, else
    marks it stale; a stale one is drawn when the widget next shows (its Show event), or on `catch_up()` (a window
    never shown gets no Show events, so a tab switch there calls it)."""

    def __init__(self, widget: QtWidgets.QWidget, draw: Callable[[], object]) -> None:
        super().__init__(widget)
        self.widget, self.draw = widget, draw
        self.stale = False
        widget.installEventFilter(self)

    def request(self) -> None:
        self.stale = True
        self.catch_up()

    def catch_up(self) -> None:
        if self.stale and not tucked_away(self.widget):
            self.stale = False
            self.draw()

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.Show:
            self.catch_up()
        return False


class LazyTabs:
    """A tab widget's tabs, each drawn by its function only while it shows, and once until marked stale again
    (`stale`). Tabs without a function draw themselves."""

    def __init__(self, tabs: QtWidgets.QTabWidget, draws: Mapping[QtWidgets.QWidget, Callable[[], object]]) -> None:
        self.tabs = {tab: Deferred(tab, draw) for tab, draw in draws.items()}
        tabs.currentChanged.connect(self.catch_up)

    def stale(self, *tabs: QtWidgets.QWidget) -> None:
        """`tabs` (every one, if none given) out of date: the one showing is drawn now, the rest when they show."""
        for tab in tabs or self.tabs:
            self.tabs[tab].request()

    def catch_up(self) -> None:
        for deferred in self.tabs.values():
            deferred.catch_up()

    def is_stale(self, tab: QtWidgets.QWidget) -> bool:
        return self.tabs[tab].stale


class PageGroups(QtWidgets.QWidget):
    """The window's pages, in a row of group tabs, each group with its own row of pages.

    Fourteen pages in one row need ~1,800 px, more than most screens, so they are grouped
    (Foundations, Responses, …). The pages are also numbered in one flat order, the order
    they are meant to be read in, and the methods below mirror QTabWidget's over that
    order, so a caller can count, find and show a page without knowing its group.
    """

    currentChanged = QtCore.Signal(int)  # noqa: N815 (QTabWidget's name) the flat index of the page now showing

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.groups = QtWidgets.QTabWidget()
        self.groups.setDocumentMode(True)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.groups)
        self._rows: list[QtWidgets.QTabWidget] = []
        self._current = -1
        self.groups.currentChanged.connect(self._changed)

    def add_group(self, title: str, *pages: tuple[QtWidgets.QWidget, str]) -> None:
        row = QtWidgets.QTabWidget()
        row.setDocumentMode(True)
        for page, name in pages:
            row.addTab(page, name)
        row.currentChanged.connect(self._changed)
        self._rows.append(row)
        self.groups.addTab(row, title)

    def setCornerWidget(self, widget: QtWidgets.QWidget, corner=QtCore.Qt.Corner.TopRightCorner) -> None:
        self.groups.setCornerWidget(widget, corner)

    # --- QTabWidget's API over the flat order
    def count(self) -> int:
        return sum(row.count() for row in self._rows)

    def _locate(self, index: int) -> tuple[QtWidgets.QTabWidget, int]:
        for row in self._rows:
            if index < row.count():
                return row, index
            index -= row.count()
        raise IndexError(index)

    def widget(self, index: int) -> QtWidgets.QWidget | None:
        try:
            row, i = self._locate(index)
        except IndexError:
            return None
        return row.widget(i)

    def tabText(self, index: int) -> str:
        row, i = self._locate(index)
        return row.tabText(i)

    def group_title(self, index: int) -> str:
        """The title of the group the page at `index` is in."""
        row, _ = self._locate(index)
        return self.groups.tabText(self._rows.index(row))

    def indexOf(self, widget: QtWidgets.QWidget) -> int:
        start = 0
        for row in self._rows:
            i = row.indexOf(widget)
            if i >= 0:
                return start + i
            start += row.count()
        return -1

    def currentIndex(self) -> int:
        row = self.groups.currentWidget()
        if row not in self._rows:
            return -1
        return sum(r.count() for r in self._rows[: self._rows.index(row)]) + row.currentIndex()

    def currentWidget(self) -> QtWidgets.QWidget | None:
        row = self.groups.currentWidget()
        return row.currentWidget() if isinstance(row, QtWidgets.QTabWidget) else None

    def setCurrentIndex(self, index: int) -> None:
        row, i = self._locate(index)
        row.setCurrentIndex(i)
        self.groups.setCurrentWidget(row)

    def setCurrentWidget(self, widget: QtWidgets.QWidget) -> None:
        index = self.indexOf(widget)
        if index >= 0:
            self.setCurrentIndex(index)

    def rewind(self) -> None:
        """Every group but the one showing back to its first page, as a group opens before it's been read."""
        for row in self._rows:
            if row is not self.groups.currentWidget():
                row.setCurrentIndex(0)

    def replace(self, old: QtWidgets.QWidget, new: QtWidgets.QWidget) -> None:
        """Put `new` in `old`'s tab, with its title, showing if `old` was; `old` is taken out (not deleted)."""
        row, i = self._locate(self.indexOf(old))
        showing = row.currentIndex() == i
        row.blockSignals(True)  # the page showing doesn't change, only what's in its tab
        row.insertTab(i, new, row.tabText(i))
        if showing:
            row.setCurrentIndex(i)
        row.removeTab(i + 1)
        row.blockSignals(False)

    def _changed(self, _: int) -> None:
        index = self.currentIndex()
        if index != self._current:
            self._current = index
            self.currentChanged.emit(index)
