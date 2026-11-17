"""*Defaults* (Ctrl+R): every page's controls back as they first showed, its motion as it opened (#218 U3)."""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from vib_tutorial.gui.lesson import RunButton, TrialCombo  # noqa: E402
from vib_tutorial.gui.main_window import LAZY_PAGES  # noqa: E402

pytestmark = pytest.mark.slow  # builds every page


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def _free_windows(app, free_windows):
    """Each test's windows torn down and freed after it (conftest's `free_windows`)."""


def _fields(state) -> dict[str, str]:
    return {
        f.name: repr(getattr(state, f.name)) for f in dataclasses.fields(state) if f.name not in ("name", "note", "tab")
    }  # the reader stays on their tab


def _runs(page) -> list[tuple[bool, bool]]:
    return [(r.isChecked(), r.started) for r in page.findChildren(RunButton)]


@pytest.mark.parametrize("name", [name for name, _ in LAZY_PAGES])
def test_defaults_puts_a_page_back_as_it_opened(main_window, app, name):
    w = main_window
    w.show_page(name)
    page = w.pages.currentWidget()
    app.processEvents()
    opening, runs, chain = _fields(page.practice_state()), _runs(page), w.sim_page.sim.system.copy()
    combos = page.findChildren(TrialCombo)
    pick = combos[0] if combos else page.preset  # its Things to try, or its presets
    pick.setCurrentIndex(pick.count() - 1)  # the last: other settings, often another chain
    app.processEvents()
    w.restore_defaults()
    app.processEvents()
    now = _fields(page.practice_state())
    assert {k: v for k, v in now.items() if v != opening[k]} == {}, "every setting as it opened"
    assert _runs(page) == runs, "the motion as it opened: at rest, or a mode swinging"
    assert all(c.currentIndex() == 0 for c in combos if c.first), "Things to try back at its prompt"
    assert w.sim_page.sim.system.n == chain.n and np.allclose(w.sim_page.sim.system.stiffness, chain.stiffness), (
        "the chain it opened with"
    )


def test_defaults_on_the_simulation_page(main_window, app):
    w = main_window
    w.pages.setCurrentWidget(w.sim_page)
    opening = w.sim_page.practice_state()
    w.sim_page.trials.combo.setCurrentIndex(w.sim_page.trials.combo.count() - 1)
    app.processEvents()
    assert w.sim_page.practice_state() != opening
    w.restore_defaults()
    app.processEvents()
    assert w.sim_page.practice_state() == opening and not w.sim_page.force.on


@pytest.mark.parametrize("name", ["random_page", "oma_page", "updating_page", "shock_page"])
def test_a_page_set_up_from_the_chain_opens_at_its_prompt(main_window, app, name):
    """Random vibration, OMA, Model updating and Shock work their settings out from the chain: they open at
    "Pick one…" like every page (#316 B13), set up as their first trial would, its note left out."""
    w = main_window
    w.show_page(name)
    page = w.pages.currentWidget()
    app.processEvents()
    assert page.trials.combo.chosen() is None and page.trials.note.text() == ""
    opening = _fields(page.practice_state())
    page.trials.combo.pick(page.trials.combo.names[0])
    if page.trials.asking is not None:  # it asks a prediction first (#316 B3): answered, it runs
        page.trials.answer(0)
    app.processEvents()
    assert page.trials.combo.chosen() == 0 and page.trials.note.text()
    assert _fields(page.practice_state()) == opening, "the first trial is the opening setup, with its note"
    page.trials.combo.pick(page.trials.combo.names[-1])
    app.processEvents()
    page.load_setup({})
    assert page.trials.combo.chosen() is None, (
        "a practice setup is no trial: a new chain won't set the last one up again"
    )
