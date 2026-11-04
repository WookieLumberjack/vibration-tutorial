"""Practice problems: marking, every problem's own answer over many seeds, and the Practice window."""

import math
import os
import re
import tempfile

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6 import QtCore  # noqa: E402

# Keep the problems solved here out of the user's own settings.
QtCore.QSettings.setPath(QtCore.QSettings.Format.NativeFormat, QtCore.QSettings.Scope.UserScope, tempfile.mkdtemp())

from vib_tutorial.core.practice import Question, check, number, parse_number, rounded  # noqa: E402
from vib_tutorial.gui.links import COURSE  # noqa: E402
from vib_tutorial.gui.practice import PAGES, problems  # noqa: E402

SEEDS = range(50)
ALL = [p for page in PAGES for p in problems(page)]


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def _close_graphics(app):
    from shiboken6 import isValid

    from vib_tutorial.gui import close_graphics

    before = set(QtWidgets.QApplication.topLevelWidgets())
    yield
    for w in QtWidgets.QApplication.topLevelWidgets():
        if w not in before and isValid(w):
            close_graphics(w)


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
    from vib_tutorial.gui.main_window import _page_class  # noqa: F401 (the pages' modules)
    from vib_tutorial.gui.practice import problems as found

    course = [name for _, pages in COURSE for name, _, _ in pages if name != "sim_page"]
    assert sorted(PAGES) == sorted(name for name in course if found(name))
    keys = [(p.page, p.key) for p in ALL]
    assert len(keys) == len(set(keys))
    assert all(p.page in PAGES for p in ALL)


def _trial_class(page: str):
    """The page's trial class, which its setups (and its `practice_state`) are made of."""
    import sys

    from vib_tutorial.gui.main_window import _page_class

    module = sys.modules[_page_class(page).__module__]
    return type(module.TRIALS[0])


@pytest.mark.parametrize("problem", ALL, ids=lambda p: f"{p.page}/{p.key}")
def test_every_question_marks_its_own_answer(problem):
    for seed in SEEDS:
        q = problem.question(seed)
        assert q.prompt and q.solution
        assert problem.question(seed) == q  # the seed sets the numbers
        if q.is_spec:
            assert q.setup is not None and q.design is not None and not q.is_choice
            # The design meets the spec and the start doesn't (the page's own round trip is checked below).
            trial = _trial_class(problem.page)
            assert check(q, trial("", "", **q.design)).right
            start = check(q, trial("", "", **q.setup))
            assert not start.right and "✗" in start.details
        elif q.is_choice:
            assert 0 <= q.answer < len(q.choices) and len(set(q.choices)) == len(q.choices)
            assert check(q, q.answer).right
            assert not any(check(q, i).right for i in range(len(q.choices)) if i != q.answer)
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
    from vib_tutorial.gui.main_window import _page_class

    for name in PAGES:
        page = _page_class(name)()
        for problem in problems(name):
            for seed in range(10):
                q = problem.question(seed)
                if q.setup is not None:
                    page.load_setup(q.setup)
                if q.is_spec:  # through the page's own boxes: the design still meets it, the start doesn't
                    assert not check(q, page.practice_state()).right, (problem.key, seed)
                    page.load_setup(q.design)
                    assert check(q, page.practice_state()).right, (problem.key, seed)
        page.close()


def test_practice_window(app):
    from PySide6 import QtGui

    from vib_tutorial.gui.main_window import MainWindow
    from vib_tutorial.gui.practice import TICK

    w = MainWindow()
    w.show()
    shortcuts = {s.key().toString(): s for s in w.findChildren(QtGui.QShortcut)}
    shortcuts["Ctrl+F2"].activated.emit()
    p = w.practice_window
    assert p.isVisible() and p.problem is None  # the Simulation page has no problems yet
    assert not w._built and not p.check_button.isEnabled()

    # A number: wrong, then right, and the problem is ticked.
    p.show_problem("sdof_page", "natural-frequency", seed=3)
    assert not w._built  # asking builds no page
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
