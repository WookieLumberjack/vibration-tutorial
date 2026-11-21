"""`gui/challenge.py`: a page's specs as challenges on the page, checked live as its controls are tuned."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

PAGES = ("isolation_page", "quarter_car_page", "active_control_page", "frame_page", "rotor_page", "seismic_page")


@pytest.fixture(autouse=True)
def _free_windows(app, free_windows):
    """Each test's windows torn down and freed after it (conftest's `free_windows`)."""


@pytest.fixture(autouse=True)
def _forget_progress():
    """Solving a challenge writes the worker's settings: forget it, or the course map shows it to later tests."""
    yield
    from vib_tutorial.gui import practice

    practice.reset_progress()


@pytest.mark.parametrize("name", PAGES)
def test_each_challenge_starts_unmet_and_is_met_by_its_design(main_window, app, name):
    """(conftest points the settings at the worker's own file: solving one writes them.)"""
    from vib_tutorial.gui import practice
    from vib_tutorial.gui.challenge import ChallengeBox, challenges

    w = main_window
    page = getattr(w, name)
    w.pages.setCurrentWidget(page)
    box = page.findChild(ChallengeBox)
    assert box is not None and box.problems == challenges(practice.problems(name)) and box.problems
    solved: list[str] = []
    box.solved.connect(solved.append)
    for problem in box.problems:
        box.pick(problem.key, seed=3)
        q = box.question
        assert q is not None and q.is_spec and box.prompt.text() == q.prompt and box.timer.isActive()
        box.poll(force=True)
        assert not box.met and "Not yet" in box.status.text()  # the start doesn't meet it
        page.load_setup(dict(q.design))
        box.poll()
        assert box.met and "It meets the spec" in box.status.text()
        assert f"{name}/{problem.key}" in practice._solved()
        box.start_over()  # back where it starts; solved stays solved
        assert "Not yet" in box.status.text() and box.met
    assert solved == [p.key for p in box.problems]


def test_a_given_number_is_listed_only_once_it_moves_and_defaults_clears_the_box(main_window, app):
    from vib_tutorial.gui.challenge import PROMPT

    w = main_window
    page = w.quarter_car_page
    w.pages.setCurrentWidget(page)
    box = page.challenge
    box.pick(box.problems[0].key, seed=3)
    box.poll(force=True)
    text = box.status.text()
    assert "kept at" not in text and "given numbers kept as they are" in text
    page.speed.setValue(page.speed.value() + 10.0)  # move a number the spec gives
    box.poll()
    assert "✗ The speed kept at" in box.status.text()

    box.new_numbers()  # the same challenge, set up afresh
    assert box.problem is box.problems[0] and "The speed kept at" not in box.status.text()
    w.restore_defaults()
    assert box.combo.currentText() == PROMPT and box.question is None and not box.timer.isActive()
    assert not box.prompt.isVisible() and not box.restart.isVisible()
