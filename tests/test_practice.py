"""Practice problems: marking, every problem's own answer over many seeds, and the Practice window."""

import math
import os
import re
from functools import partial

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6 import QtCore  # noqa: E402

from vib_tutorial.core.practice import Prediction, Question, check, number, parse_number, rounded  # noqa: E402
from vib_tutorial.gui.links import COURSE  # noqa: E402
from vib_tutorial.gui.practice import pages, problems  # noqa: E402

SEEDS = range(50)
ALL = [p for page in pages() for p in problems(page)]


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def _free_windows(app, free_windows):
    """Each test's windows torn down and freed after it (conftest's `free_windows`)."""


# ------------------------------------------------------------------ marking
def test_parse_number():
    assert parse_number(" 2.5 ") == 2.5
    assert parse_number("2,5") == 2.5
    assert parse_number("−3") == -3.0
    assert parse_number("1.2×10^-3") == pytest.approx(1.2e-3)
    assert parse_number("4e2") == 400.0
    assert parse_number("abc") is None and parse_number("inf") is None


def test_check_a_number_and_its_slips():
    q = Question("f?", 3.183, "", unit="Hz")
    assert check(q, "3.18").right and check(q, "3.2").right  # within 2 %
    assert not check(q, "3.3").right
    assert "2π" in check(q, str(3.183 * 2 * math.pi)).message  # rad/s for Hz
    assert "1000" in check(q, "3183").message
    assert "factor of 2" in check(q, str(3.183 / 2)).message
    assert "sign" in check(q, "-3.18").message
    assert check(q, "x").message == "That isn't a number."


def test_number_prints_as_rounded_keeps():
    for value in (0.012345, 3.14159, 62.66, 1234.5, 2.5e-5, 4.2e7):
        assert float(re.sub(r"×10<sup>(-?\d+)</sup>", r"e\1", number(value))) == rounded(value)


# ------------------------------------------------------------------ the problems
def test_the_pages_with_problems_are_listed():
    from vib_tutorial.gui.practice import problems as found

    course = [name for _, group in COURSE for name, _, _ in group]
    assert list(pages()) == [name for name in course if found(name)] and "sim_page" in pages()
    keys = [(p.page, p.key) for p in ALL]
    assert len(keys) == len(set(keys))
    assert all(p.page in pages() for p in ALL)


def _state(page: str, setup: dict):
    """What the page's `practice_state` would give once set up as `setup` says: its trial (or, where its trials
    aren't a class, its setup class) with those fields."""

    if page == "sim_page":
        from vib_tutorial.gui.sim_trials import SimTrial

        return SimTrial("", "", **setup)
    from vib_tutorial.gui.links import page_module

    module = page_module(page)
    setup_class = getattr(module, "PRACTICE_SETUP", None)  # where the trials aren't a class to set up
    return setup_class(**setup) if setup_class else type(module.TRIALS[0])("", "", **setup)


@pytest.mark.parametrize("problem", ALL, ids=lambda p: f"{p.page}/{p.key}")
def test_every_question_marks_its_own_answer(problem):
    for seed in SEEDS:
        q = problem.question(seed)
        assert q.prompt and q.solution
        assert problem.question(seed) == q  # the seed sets the numbers
        if q.is_spec:
            assert q.setup is not None and q.design is not None and not q.is_choice
            # The design meets the spec and the start doesn't (the page's own round trip is checked below).
            assert check(q, _state(problem.page, q.design)).right
            start = check(q, _state(problem.page, q.setup))
            assert not start.right and "✗" in start.details
        elif q.is_choice:
            assert 0 <= q.answer < len(q.choices) and len(set(q.choices)) == len(q.choices)
            assert check(q, q.answer).right
            assert not any(check(q, i).right for i in range(len(q.choices)) if i != q.answer)
            # A reason for each wrong choice, said when it's picked (#316 B14).
            assert q.why, "a choice says why each wrong one is wrong (Question.why)"
            assert q.why[q.answer] == "" and all(q.why[i] for i in range(len(q.choices)) if i != q.answer)
            assert all(q.why[i] in check(q, i).details for i in range(len(q.choices)) if i != q.answer)
        else:
            assert math.isfinite(q.answer) and q.answer > 0
            assert check(q, repr(q.answer)).right and check(q, number(q.answer)).right
            assert not check(q, repr(q.answer * (1 + 2 * q.tolerance))).right
        assert not q.predict or (q.is_choice and q.setup is not None)


def test_check_a_spec():
    from vib_tutorial.core.practice import Requirement, at_least, at_most, kept

    q = Question("Tune it.", 0, "", goal=lambda x: (at_most("x", x, 2.0, "mm"), kept("y", 5.0, 5.01, "kg")))
    assert q.is_spec and not q.is_choice
    v = check(q, 1.5)
    assert v.right and v.message == "It meets the spec." and v.details.count("✓") == 2
    v = check(q, 2.5)
    assert not v.right and "1 of 2" in v.message and "x ≤ 2.00 mm" in v.details and "now 2.50 mm" in v.details
    assert not kept("y", 5.0, 5.1).met and at_least("z", 3.0, 2.0).met
    assert isinstance(at_most("x", 1, 2), Requirement)


def test_a_wrong_choice_says_why_and_the_reasons_shuffle_with_the_choices():
    q = Question("Which?", 1, "", choices=("a", "b", "c"), why=("not a", "", "not c"))
    v = check(q, 2)
    assert not v.right and v.message == "Not that one." and "not c" in v.details
    assert check(q, 1).right and check(Question("Which?", 0, "", choices=("a", "b")), 1).details == ""
    with pytest.raises(ValueError):
        Question("Which?", 0, "", choices=("a", "b"), why=("",))
    from vib_tutorial.core.practice import Problem

    shuffled = Problem("k", "t", "sdof_page", "", lambda rng: q)
    for seed in range(10):
        s = shuffled.question(seed)
        assert s.why[s.choices.index("a")] == "not a" and s.why[s.answer] == ""


def test_every_problems_figure_is_drawn():
    from vib_tutorial.gui import figures

    with_figure = [p for p in ALL if p.figure]
    assert len(with_figure) >= 4
    for p in with_figure:
        assert figures.build(p.figure) is not None, (p.page, p.key, p.figure)


def test_choices_are_shuffled():
    for problem in ALL:
        if problem.question(0).is_choice:
            assert len({problem.question(seed).answer for seed in SEEDS}) > 1, problem.key


def test_the_short_cuts_the_solutions_offer_pass():
    """The log decrement's δ/2π and the half-power Δf/2f_n, worked from the printed numbers, are marked right."""
    from vib_tutorial.gui import sdof

    by_key = {p.key: p for p in sdof.PROBLEMS}
    for seed in SEEDS:
        q = by_key["log-decrement"].question(seed)
        x0, xn = (float(v) for v in re.findall(r"([\d.]+) mm", q.prompt))
        n = int(re.search(r"peak (\d+) cycles", q.prompt).group(1)) if "cycles" in q.prompt else 1
        assert check(q, str(math.log(x0 / xn) / n / (2 * math.pi))).right

        q = by_key["half-power"].question(seed)
        lo, hi = (float(v) for v in re.findall(r"([\d.]+) Hz", q.prompt))
        assert check(q, str((hi - lo) / (hi + lo))).right


def test_the_isolation_numbers_follow_the_rule_of_thumb():
    """15.76/√δ (δ in mm), as the solutions write it, is marked right."""
    from vib_tutorial.gui import isolation

    problem = next(p for p in isolation.PROBLEMS if p.key == "mount-frequency")
    for seed in SEEDS:
        q = problem.question(seed)
        sag = float(re.search(r"sags ([\d.]+) mm", q.prompt).group(1))
        assert check(q, str(15.76 / math.sqrt(sag))).right


def test_hints_link_to_sections_that_exist():
    from test_links import notes

    html = notes()
    for problem in ALL:
        found = re.findall(r'href="page:([a-z_]+)#([^"]+)"', problem.hint)
        assert found, problem.key
        for page, anchor in found:
            assert f'<a name="{anchor}">' in html[page], f"{problem.key} links to {page}#{anchor}"


# ------------------------------------------------------------------ the window
def test_every_setup_loads_on_its_page(app):
    from vib_tutorial.gui import close_graphics
    from vib_tutorial.gui.links import page_class
    from vib_tutorial.gui.main_window import MainWindow

    for name in pages():
        if name == "sim_page":  # the main window's own page: set up and read through the window
            window = MainWindow()
            load, state = partial(window.set_up_page, "sim_page"), window.sim_page.practice_state
        else:
            window = page = page_class(name)()
            load, state = page.load_setup, page.practice_state
        for problem in problems(name):
            for seed in range(10):
                q = problem.question(seed)
                if q.setup is not None:
                    load(q.setup)
                if q.is_spec:  # through the page's own boxes: the design still meets it, the start doesn't
                    assert not check(q, state()).right, (problem.key, seed)
                    load(q.design)
                    assert check(q, state()).right, (problem.key, seed)
        window.close()
        close_graphics(window)  # before the next page drops it: freed with its plots in their scenes segfaults


def test_practice_window(app):
    from PySide6 import QtGui

    from vib_tutorial.gui.main_window import MainWindow
    from vib_tutorial.gui.practice import TICK

    w = MainWindow()
    w.show()
    shortcuts = {s.key().toString(): s for s in w.findChildren(QtGui.QShortcut)}
    shortcuts["Ctrl+F2"].activated.emit()
    p = w.practice_window
    assert p.isVisible() and p.problem.page == "sim_page"  # the page showing's first problem
    assert list(w._built) == ["sim_page"] and p.check_button.isEnabled()

    # A number: wrong, then right, and the problem is ticked.
    p.show_problem("sdof_page", "natural-frequency", seed=3)
    assert list(w._built) == ["sim_page"]  # asking builds no page
    q = p.question
    p.number.setText(str(q.answer * 2 * math.pi))
    p.check()
    assert "2π" in p.text.toPlainText() and not p._problem_items["sdof_page", "natural-frequency"].text(0).startswith(
        TICK
    )
    p.number.setText(number(q.answer))
    p.number.returnPressed.emit()
    assert "Right" in p.text.toPlainText()
    assert p._problem_items["sdof_page", "natural-frequency"].text(0).startswith(TICK)
    assert "(1/" in p._page_items["sdof_page"].text(0)
    p.page_button.click()  # a number's setup is there from the start
    assert w.pages.currentWidget() is w.sdof_page and w.sdof_page.stiffness.value() == q.setup["stiffness"]

    # A hint's link opens the Theory window at its section.
    p.show_hint()
    assert "Free vibration" in p.text.toPlainText()
    p.text.anchorClicked.emit(QtCore.QUrl("page:sdof_page#free-vibration"))
    assert w.theory_window.isVisible() and w.theory_window.page == "sdof_page"

    # A prediction: the page is offered only once answered, and checking it runs it there.
    p.show_problem("isolation_page", "damping-above", seed=1)
    assert not p.page_button.isEnabled() and "isolation_page" not in w._built
    p.check()
    assert "Pick one first" in p.text.toPlainText()
    p._choices[p.question.answer].setChecked(True)
    p.check()
    assert w.pages.currentWidget() is w.isolation_page
    assert w.isolation_page.zeta.value() == pytest.approx(p.question.setup["zeta"])
    assert p.page_button.isEnabled() and "Solution" in p.text.toPlainText()

    # A spec: Check reads the page, the solution offers a design that meets it.
    p.show_problem("isolation_page", "isolate-spec", seed=2)
    assert p.page_button.isEnabled() and not p.answer_area.isVisible()
    p.page_button.click()
    assert w.isolation_page.rpm.value() == p.question.setup["rpm"]
    p.check()
    assert "Not yet" in p.text.toPlainText() and "✗" in p.text.toPlainText()
    p.show_solution()
    assert p.page_button.text() == "Show a design on the page"
    p.page_button.click()
    assert w.isolation_page.deflection.value() == pytest.approx(p.question.design["deflection_mm"])
    p.check()
    assert "It meets the spec" in p.text.toPlainText()
    assert not p._problem_items["isolation_page", "isolate-spec"].text(0).startswith(TICK)  # shown, not solved
    p.show_problem("isolation_page", "isolate-spec", seed=2)
    p.page_button.click()
    w.isolation_page.deflection.setValue(p.question.design["deflection_mm"])  # tuned by hand
    p.check()
    assert p._problem_items["isolation_page", "isolate-spec"].text(0).startswith(TICK)

    # A wrong choice says why; a figure shows above its question.
    p.show_problem("sdof_page", "heavier", seed=0)
    q = p.question
    wrong = next(i for i in range(len(q.choices)) if i != q.answer)
    p._choices[wrong].setChecked(True)
    p.check()
    assert "Not that one" in p.text.toPlainText() and q.why[wrong][:30] in p.text.toPlainText()
    p.show_problem("sdof_page", "natural-frequency", seed=0)
    assert 'src="fig:oscillator"' in p.text.toHtml()
    assert all(b.isHidden() for b in p.findChildren(QtWidgets.QRadioButton) if b not in p._choices)

    # The filter keeps the pages and problems that match; Next unsolved skips the solved ones.
    p.filter.setText("wheel hop")
    shown = [k for k, item in p._problem_items.items() if not item.isHidden()]
    assert shown == [("quarter_car_page", "wheel-hop")]
    assert not p._page_items["quarter_car_page"].isHidden() and p._page_items["sdof_page"].isHidden()
    p.filter.setText("single oscillator")  # a page's name: all its problems
    assert all(not p._problem_items["sdof_page", x.key].isHidden() for x in problems("sdof_page"))
    p.filter.clear()
    assert not any(item.isHidden() for item in p._problem_items.values())
    p.show_problem("sdof_page", "log-decrement", seed=0)
    p.next_unsolved()  # natural-frequency, before it, is solved; the next one after it isn't
    after = [x.key for x in problems("sdof_page")]
    assert p.problem.key == after[after.index("log-decrement") + 1]
    p.show_problem("isolation_page", "isolate-spec", seed=2)
    w.pages.setCurrentWidget(w.isolation_page)  # (checking the prediction above ran it on the sdof page)

    # New numbers ask again; following the page turns to its problems.
    p.new_numbers()
    assert p.problem.key == "isolate-spec" and "Solution" not in p.text.toPlainText()
    p.follow.setChecked(True)
    w.pages.setCurrentWidget(w.sdof_page)
    assert p.problem.page == "sdof_page" and p.problem.key != "natural-frequency"  # the first one unsolved
    p.follow.setChecked(False)

    w.set_theme("Dark")
    assert p.question.prompt[:20] in p.text.toHtml()
    w.set_theme("Light")
    w.close()
    assert not p.isVisible()


def test_course_map_counts_the_problems_solved(app, main_window):
    """The first page's course map counts each page's problems solved (none: the notes as built), in its Theory
    tab and the Theory window; Reset progress forgets them."""
    from vib_tutorial.gui import practice, sdof
    from vib_tutorial.gui.theory_window import page_html

    practice.reset_progress()
    assert sdof.notes_html() is sdof.NOTES_HTML
    first, second = practice.problems("transient_page")[:2]
    total = len(practice.problems("transient_page"))
    practice.mark_solved(first)
    assert practice.progress() == {"transient_page": (1, total)}
    assert f"(1 of {total} solved)" in page_html("sdof_page")
    main_window.show_page("sdof_page")
    page = main_window.pages.currentWidget()
    main_window.show_theory()
    assert f"(1 of {total} solved)" in page.notes.toPlainText()
    for p in practice.problems("transient_page"):
        practice.mark_solved(p)
    page.tabs.setCurrentIndex(0)
    main_window.show_theory()  # the map is redrawn as the notes show
    assert f"all {total} problems solved" in page.notes.toPlainText()
    window = main_window.open_practice_window()
    assert f"({total}/{total})" in window._page_items["transient_page"].text(0)
    window.reset()
    assert practice.progress() == {} and "(0/" in window._page_items["transient_page"].text(0)
    assert second.key  # two problems at least, so "1 of" was a part


def _trial_predictions():
    """Every trial's Prediction in the course, once each."""
    from vib_tutorial.gui import links, sim_trials

    found = {}
    for attr in links.order():
        trials = sim_trials.TRIALS if attr == "sim_page" else getattr(links.page_module(attr), "TRIALS", ())
        for t in trials:
            p = getattr(t, "predict", None)
            if p is not None:
                found[p.prompt] = p
    return list(found.values())


def test_the_right_prediction_isnt_where_the_buttons_give_it_away():
    """The trials' choices are shuffled as they show (#409 C-7: the right one was written last in 98 of 157), the
    same each time, and an ordered scale only reversed or not."""
    predictions = _trial_predictions()
    assert len(predictions) >= 150
    shown = [p.shown_order() for p in predictions]
    for p, order in zip(predictions, shown):
        assert sorted(order) == list(range(len(p.choices))) and order == p.shown_order()  # the same each time
        if p.ordered:
            assert order in (tuple(range(len(order))), tuple(reversed(range(len(order))))), p.prompt
    at = [order.index(p.answer) for p, order in zip(predictions, shown)]
    last = sum(a == len(o) - 1 for a, o in zip(at, shown)) / len(at)
    first = sum(a == 0 for a in at) / len(at)
    assert last < 0.4 and first < 0.4 and 1 - last - first > 0.25, (first, last)
    assert sum(p.ordered for p in predictions) >= 20  # the scales are marked


def test_every_wrong_prediction_says_why():
    """A trial's Prediction gives a reason for each wrong choice, shown when it's picked (#409 I-7), as a Practice
    choice does; the right one's is empty."""
    from vib_tutorial.gui import links, sim_trials

    missing = []
    for _, members in links.COURSE:
        for attr, _, _ in members:
            trials = sim_trials.TRIALS if attr == "sim_page" else getattr(links.page_module(attr), "TRIALS", ())
            for t in trials:
                p = getattr(t, "predict", None)
                if p is None:
                    continue
                if p.why:
                    assert len(p.why) == len(p.choices) and not p.why[p.answer], t.name
                    assert all(w for k, w in enumerate(p.why) if k != p.answer), t.name
                else:
                    missing.append(f"{attr}: {t.name}")
    assert not missing, missing
    with pytest.raises(ValueError):
        Prediction("?", ("a", "b"), 0, why=("right", "wrong"))  # the right one has no reason
    with pytest.raises(ValueError):
        Prediction("?", ("a", "b"), 0, why=("",))
