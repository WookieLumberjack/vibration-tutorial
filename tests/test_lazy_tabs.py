"""tab_groups' Deferred and LazyTabs: a tab is drawn only while it shows, once per change, whether or not its
window has been shown."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from vib_tutorial.gui.tab_groups import Deferred, LazyTabs  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.mark.parametrize("shown", [True, False])
def test_tabs_draw_only_while_they_show(app, shown):
    tabs = QtWidgets.QTabWidget()
    a, b, plain = QtWidgets.QWidget(), QtWidgets.QWidget(), QtWidgets.QWidget()
    for w, name in ((a, "A"), (b, "B"), (plain, "Plain")):
        tabs.addTab(w, name)
    if shown:
        tabs.show()
        app.processEvents()
    drawn = []
    lazy = LazyTabs(tabs, {a: lambda: drawn.append("a"), b: lambda: drawn.append("b")})
    lazy.stale()
    assert drawn == ["a"] and lazy.is_stale(b) and not lazy.is_stale(a)
    lazy.stale(a)
    assert drawn == ["a", "a"]
    tabs.setCurrentWidget(b)
    app.processEvents()
    assert drawn == ["a", "a", "b"]  # drawn once it shows, once
    tabs.setCurrentWidget(a)
    tabs.setCurrentWidget(b)
    app.processEvents()
    assert drawn == ["a", "a", "b"]  # nothing changed: nothing drawn
    tabs.setCurrentWidget(plain)
    lazy.stale()
    assert drawn == ["a", "a", "b"] and lazy.is_stale(a) and lazy.is_stale(b)
    tabs.setCurrentWidget(a)
    assert drawn == ["a", "a", "b", "a"]
    tabs.close()


def test_a_deferred_panel_draws_when_it_shows(app):
    window = QtWidgets.QWidget()
    panel = QtWidgets.QWidget(window)
    QtWidgets.QVBoxLayout(window).addWidget(panel)
    drawn = []
    deferred = Deferred(panel, lambda: drawn.append(1))
    window.show()
    app.processEvents()
    panel.hide()
    deferred.request()
    assert not drawn and deferred.stale
    panel.show()  # its Show event
    assert drawn == [1] and not deferred.stale
    deferred.request()
    assert drawn == [1, 1]
    window.close()


def test_a_tab_shown_mid_trial_draws_once_at_its_end(app):
    from vib_tutorial.gui import coalesce

    tabs = QtWidgets.QTabWidget()
    first, second = QtWidgets.QWidget(), QtWidgets.QWidget()
    tabs.addTab(first, "1")
    tabs.addTab(second, "2")
    drawn: list[str] = []
    lazy = LazyTabs(tabs, {first: lambda: drawn.append("1"), second: lambda: drawn.append("2")})
    tabs.show()
    lazy.stale()
    assert drawn == ["1"]  # the one showing, at once
    with coalesce.batch():
        tabs.setCurrentWidget(second)  # a trial's tab, before its state
        lazy.stale(second)  # its state loaded
        assert drawn == ["1"]
    assert drawn == ["1", "2"]  # once, after both
    tabs.close()
