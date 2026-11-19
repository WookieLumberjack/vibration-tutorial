"""What the reader has done kept and shown (#409 I-9): trials tried, predictions answered, goals ticked."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6 import QtWidgets

from vib_tutorial.core.practice import Prediction


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def _free_windows(app, free_windows):
    """Each test's windows torn down and freed after it (conftest's `free_windows`)."""


@pytest.fixture
def record(app):
    from vib_tutorial.gui import progress

    progress.reset()  # conftest points the settings at the worker's own file: start from nothing
    yield progress
    progress.reset()


def test_a_goal_ticks_its_trial_and_problem_and_itself_once_both_are_done(record):
    from vib_tutorial.gui import links

    html = "<ul><li>an earlier list</li></ul>" + links.goals(
        "sdof_page", ("1", "log-decrement", "measure the damping;")
    )
    assert record.marks(html) == html  # nothing done yet
    before = record.version()
    record.mark_tried("sdof_page", 1)
    assert record.version() > before and record.tried_numbers("sdof_page") == {1}
    once = record.marks(html)
    assert "Try it</a> ✓" in once and "Practise it</a> ✓" not in once and "<li>✓" not in once
    record.mark_solved("sdof_page", "log-decrement")
    both = record.marks(html)
    assert "<li>✓ measure the damping;" in both and "Practise it</a> ✓" in both
    record.reset()
    assert record.marks(html) == html and not record.tried() and not record.solved()


def test_an_answered_predictions_note_shows_in_the_theorys_list(record):
    from vib_tutorial.gui import lesson

    class Trial:
        name = "A trial"
        note = "Because of the damping."
        predict = Prediction("Which way?", ("up", "down"), 0, why=("", "Not down."))

    html = lesson.trials_html([Trial()])
    assert "Because of the damping" not in record.marks(html) and "Pick it to answer" in record.marks(html)
    record.mark_answered("Which way?")
    shown = record.marks(html)
    assert "Because of the damping" in shown and "Pick it to answer" not in shown


def test_counts_of_trials_tried(record):
    record.mark_tried("sdof_page", 1)
    record.mark_tried("sdof_page", 3)
    record.mark_tried("sdof_page", 99)  # since removed: not counted
    assert record.tried_counts({"sdof_page": 8, "pendulum_page": 7}) == {"sdof_page": (2, 8)}
    from vib_tutorial.gui.links import course_map

    line = course_map({"sdof_page": (1, 9)}, {"sdof_page": (2, 8), "pendulum_page": (7, 7)})
    assert "(1 of 9 solved · 2 of 8 tried)" in line and "(✓ all 7 tried)" in line


def test_the_page_ticks_what_the_reader_tries_and_the_notes_follow(main_window, record):
    from vib_tutorial.gui import sdof

    w = main_window
    w.show_page("sdof_page")
    p = w.sdof_page
    box = p.trials
    assert box.page_name == "sdof_page"
    plain = next(k for k, t in enumerate(sdof.TRIALS) if t.predict is None)
    box.combo.pick(sdof.TRIALS[plain].name)
    assert record.tried_numbers("sdof_page") == {plain + 1}
    assert box.combo.itemText(plain + box.combo.first).startswith("✓ ")
    asked = next(k for k, t in enumerate(sdof.TRIALS) if isinstance(t.predict, Prediction))
    prediction = sdof.TRIALS[asked].predict
    w.show_theory()
    waiting = p.notes.toPlainText().count("Pick it to answer and see")
    p.tabs.setCurrentIndex(0)
    box.combo.pick(sdof.TRIALS[asked].name)
    assert record.tried_numbers("sdof_page") == {plain + 1}  # not until it's answered
    box.answer(prediction.answer)
    assert record.tried_numbers("sdof_page") == {plain + 1, asked + 1}
    assert record.prompt_key(prediction.prompt) in record.answered()
    w.show_theory()  # its notes drawn again as they show: the note's back in the list, the map counts
    text = p.notes.toPlainText()
    assert text.count("Pick it to answer and see") == waiting - 1
    assert "2 of" in text and "tried" in text
