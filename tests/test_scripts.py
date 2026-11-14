"""The scripts in scripts/: they drive the GUI's widgets directly (private attributes too), so a refactor can break
them, and nothing else runs them (#278 and #279 broke both unnoticed). Each README shot runs here, its waits and
its PNG write stubbed out, and each of mac_timings' scenarios sets up its page."""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6 import QtCore  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
SLOW_SHOTS = {"jeffcott_cracked"}  # ~50 s: a cracked shaft's Hill's-method modes at each speed


def _shot_names() -> list[str]:
    """The SHOTS list's names, read from the script's source: collecting the tests builds no app."""
    tree = ast.parse((SCRIPTS / "readme_screenshots.py").read_text(encoding="utf-8"))
    shots = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "SHOTS" for t in node.targets)
    )
    assert isinstance(shots, ast.List), "SHOTS is a list of the shot functions"
    return [elt.id for elt in shots.elts if isinstance(elt, ast.Name)]


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def _free_windows(app, free_windows):
    """Each test's windows torn down and freed after it (conftest's `free_windows`)."""


def _import(name: str):
    """A script as a module (it sets XDG_CONFIG_HOME for its own runs: put back)."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    sys.path.insert(0, str(SCRIPTS))
    try:
        return __import__(name)
    finally:
        sys.path.remove(str(SCRIPTS))
        if xdg is None:
            os.environ.pop("XDG_CONFIG_HOME", None)
        else:
            os.environ["XDG_CONFIG_HOME"] = xdg


@pytest.fixture(scope="module")
def screenshots(app):
    return _import("readme_screenshots")


def _pump(*_args, **_kwargs) -> None:
    """What a wait does for the shots, at once: the events, and the windows closed since freed."""
    app = QtWidgets.QApplication.instance()
    assert app is not None, "the module's app"
    app.processEvents()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


@pytest.mark.parametrize(
    "name", [pytest.param(n, marks=pytest.mark.slow) if n in SLOW_SHOTS else n for n in _shot_names()]
)
def test_readme_shot_runs(screenshots, name, monkeypatch, tmp_path):
    saved = []
    monkeypatch.setattr(screenshots, "OUT", tmp_path)
    monkeypatch.setattr(screenshots, "settle", _pump)
    monkeypatch.setattr(screenshots.QTest, "qWait", staticmethod(_pump))
    monkeypatch.setattr(screenshots, "write_png", lambda grab, path: saved.append((path.name, grab.size())))
    getattr(screenshots, name)()
    _pump()
    assert saved and all(not size.isEmpty() for _, size in saved), saved  # a window (or a panel of one) grabbed


def test_mac_timings_scenarios_set_up(main_window):
    timings = _import("mac_timings")
    w = main_window
    for scenario in timings.SCENARIOS:
        page, speed = scenario.setup(w)
        assert page is w.pages.currentWidget() and speed > 0, scenario.label
        for loop in (getattr(page, "loop", None), w.loop):
            if loop is not None:
                loop.enabled = False  # set up, not run
        _pump()
