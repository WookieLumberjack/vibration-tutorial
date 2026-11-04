"""The Practice window: each page's practice problems, with new numbers every time, beside the main window.

A tree of the pages that have problems (in the course's order) and their problems, ticked once solved. A
question is a number typed in its unit, or a choice. *Check* marks it (and says if a number is out by one of
the usual slips), *Hint* points into the page's notes (its links open the Theory window there), *Solution* shows
the working, *New numbers* asks it again. *Show on the page* sets the page up as the question describes; for a
prediction it's offered only once answered, and checking a prediction runs it on the page: predict, then see.
A spec is answered on the page itself: tune its controls, and *Check* reads them (the page's `practice_state`)
and lists each requirement, met or not; once the solution shows, *Show on the page* loads a design that meets it.

The problems are each page module's `PROBLEMS` (core.practice.Problem), so a page needn't be built to practise;
`PAGES` lists the pages that have them. The problems solved are kept in the settings.
"""

from __future__ import annotations

import random
import sys
from collections.abc import Callable
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from ..core.practice import Problem, Question, check
from . import links
from .style import colors
from .theming import SETTINGS

PAGES = (
    "sim_page",
    "energy_page",
    "sdof_page",
    "pendulum_page",
    "transient_page",
    "periodic_page",
    "damping_page",
    "isolation_page",
    "signals_page",
)  # the pages with PROBLEMS, any order (the tree follows the course)
PROBLEM_ROLE = QtCore.Qt.ItemDataRole.UserRole  # (page, problem key) of a problem's item, (page, "") a page's
TICK = "✓ "


def problems(name: str) -> tuple[Problem, ...]:
    """Page `name`'s practice problems, from its module (without building the page)."""
    if name == "sim_page":
        from .sim_trials import PROBLEMS

        return PROBLEMS
    from .main_window import _page_class

    return getattr(sys.modules[_page_class(name).__module__], "PROBLEMS", ())


def _solved() -> set[str]:
    stored = QtCore.QSettings(*SETTINGS).value("practice/solved", "")
    return set(filter(None, str(stored).split(",")))


def _mark_solved(problem: Problem) -> None:
    solved = _solved() | {f"{problem.page}/{problem.key}"}
    QtCore.QSettings(*SETTINGS).setValue("practice/solved", ",".join(sorted(solved)))


class PracticeWindow(QtWidgets.QWidget):
    """Practice problems in a window of their own.

    `set_up(page, setup)` shows a page in the main window set up as a question says; `read(page, anchor)` opens
    the Theory window at a hint's link; `state(page)` is the page's settings, which a spec checks.
    """

    def __init__(
        self,
        set_up: Callable[[str, dict], None],
        read: Callable[[str, str], None],
        state: Callable[[str], Any],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent, QtCore.Qt.WindowType.Window)
        self.setWindowTitle("Practice")
        self._set_up = set_up
        self._state = state
        self.problem: Problem | None = None
        self.question: Question | None = None
        self.seed = 0
        self._verdict = self._details = self._hint = self._solution = ""
        self._answered = False

        self.follow = QtWidgets.QCheckBox("Follow the page")
        self.follow.setToolTip("<p>Turn to the problems of each page the main window turns to.</p>")
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(self.follow)
        bar.addStretch(1)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemClicked.connect(self._on_item)
        self.tree.itemActivated.connect(self._on_item)
        self._problem_items: dict[tuple[str, str], QtWidgets.QTreeWidgetItem] = {}
        self._page_items: dict[str, QtWidgets.QTreeWidgetItem] = {}
        number = 1
        for group, pages in links.COURSE:
            top = None
            for name, title, about in pages:
                if name in PAGES:
                    if top is None:
                        top = QtWidgets.QTreeWidgetItem([group])
                        font = top.font(0)
                        font.setBold(True)
                        top.setFont(0, font)
                        top.setFlags(top.flags() & ~QtCore.Qt.ItemFlag.ItemIsSelectable)
                        self.tree.addTopLevelItem(top)
                    page = QtWidgets.QTreeWidgetItem([f"{number}. {title.replace('&&', '&')}"])
                    page.setData(0, PROBLEM_ROLE, (name, ""))
                    page.setToolTip(0, about)
                    top.addChild(page)
                    self._page_items[name] = page
                    for p in problems(name):
                        item = QtWidgets.QTreeWidgetItem([p.title])
                        item.setData(0, PROBLEM_ROLE, (name, p.key))
                        item.setToolTip(0, p.title)
                        page.addChild(item)
                        self._problem_items[name, p.key] = item
                    page.setExpanded(True)
                number += 1
            if top is not None:
                top.setExpanded(True)
        self._show_ticks()

        self.title = QtWidgets.QLabel()
        font = self.title.font()
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() * 1.15)
        self.title.setFont(font)
        self.text = QtWidgets.QTextBrowser()
        links.wire(self.text, read)

        self.number = QtWidgets.QLineEdit()
        self.number.setPlaceholderText("Your answer")
        self.number.returnPressed.connect(self.check)
        self.unit = QtWidgets.QLabel()
        number_row = QtWidgets.QWidget()
        nl = QtWidgets.QHBoxLayout(number_row)
        nl.setContentsMargins(0, 0, 0, 0)
        nl.addWidget(self.number, 1)
        nl.addWidget(self.unit)
        self.choice_box = QtWidgets.QWidget()
        self._choice_layout = QtWidgets.QVBoxLayout(self.choice_box)
        self._choice_layout.setContentsMargins(0, 0, 0, 0)
        self._choices: list[QtWidgets.QRadioButton] = []
        self._choice_group = QtWidgets.QButtonGroup(self)
        self.answer_area = QtWidgets.QStackedWidget()
        self.answer_area.addWidget(number_row)
        self.answer_area.addWidget(self.choice_box)
        self.answer_area.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum)

        self.check_button = self._button("Check", "Mark your answer.", self.check)
        self.hint_button = self._button(
            "Hint", "Where the page's notes explain it (the links open the Theory window).", self.show_hint
        )
        self.solution_button = self._button("Solution", "The working and the answer.", self.show_solution)
        self.new_button = self._button("New numbers", "The same problem with new numbers.", self.new_numbers)
        self.page_button = self._button(
            "Show on the page", "Set the page up as the question describes, to see it for yourself.", self.show_on_page
        )
        buttons = QtWidgets.QHBoxLayout()
        for b in (self.check_button, self.hint_button, self.solution_button, self.new_button):
            buttons.addWidget(b)
        buttons.addStretch(1)
        buttons.addWidget(self.page_button)

        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.title)
        rl.addWidget(self.text, 1)
        rl.addWidget(self.answer_area)
        rl.addLayout(buttons)

        split = QtWidgets.QSplitter()
        split.addWidget(self.tree)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([260, 560])
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(bar)
        layout.addWidget(split)
        self.resize(850, 560)
        geometry = QtCore.QSettings(*SETTINGS).value("practice_window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        self.follow.setChecked(QtCore.QSettings(*SETTINGS).value("practice_window/follow", "false") == "true")
        self.follow.toggled.connect(
            lambda on: QtCore.QSettings(*SETTINGS).setValue("practice_window/follow", "true" if on else "false")
        )
        self._render()

    def _button(self, text: str, tip: str, slot: Callable[[], None]) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(text)
        button.setToolTip(f"<p>{tip}</p>")
        button.clicked.connect(slot)
        return button

    # --------------------------------------------------------------- choosing
    def show_problem(self, page: str, key: str, seed: int | None = None) -> None:
        """Ask page `page`'s problem `key`, with the numbers from `seed` (new ones if None)."""
        self.problem = next(p for p in problems(page) if p.key == key)
        self.seed = random.randrange(2**32) if seed is None else seed
        self.question = self.problem.question(self.seed)
        self._verdict = self._details = self._hint = self._solution = ""
        self._answered = False
        self.number.clear()
        for b in self._choices:
            self._choice_group.removeButton(b)
            self._choice_layout.removeWidget(b)
            b.deleteLater()
        self._choices = []
        for i, text in enumerate(self.question.choices):
            b = QtWidgets.QRadioButton(text)
            self._choice_group.addButton(b, i)
            self._choice_layout.addWidget(b)
            self._choices.append(b)
        self.answer_area.setCurrentIndex(1 if self.question.is_choice else 0)
        self.unit.setText(self.question.unit)
        item = self._problem_items[page, key]
        self.tree.setCurrentItem(item)
        self.tree.scrollToItem(item)
        self._render()
        if not self.question.is_choice:
            self.number.setFocus()

    def show_page(self, name: str) -> None:
        """Turn to page `name`'s problems (its first unsolved one), if it has any."""
        if name not in PAGES or (self.problem is not None and self.problem.page == name):
            return
        solved = _solved()
        found = problems(name)
        first = next((p for p in found if f"{name}/{p.key}" not in solved), found[0])
        self.show_problem(name, first.key)

    def follow_page(self, name: str) -> None:
        """The main window turned to page `name`: show its problems if following."""
        if self.follow.isChecked():
            self.show_page(name)

    def new_numbers(self) -> None:
        if self.problem is not None:
            self.show_problem(self.problem.page, self.problem.key)

    def _on_item(self, item: QtWidgets.QTreeWidgetItem) -> None:
        target = item.data(0, PROBLEM_ROLE)
        if not target:
            return
        if target[1]:
            self.show_problem(*target)
        else:
            self.show_page(target[0])

    # --------------------------------------------------------------- answering
    def check(self) -> None:
        q = self.question
        if q is None:
            return
        if q.is_spec:
            verdict = check(q, self._state(self.problem.page))
        elif q.is_choice:
            picked = self._choice_group.checkedId()
            if picked < 0:
                self._verdict = "<p><i>Pick one first.</i></p>"
                self._render()
                return
            verdict = check(q, picked)
        else:
            verdict = check(q, self.number.text())
        colour = "good" if verdict.right else "bad"
        self._verdict = f'<p><b class="{colour}">{verdict.message}</b></p>'
        self._details = verdict.details
        if verdict.right and not self._solution:
            _mark_solved(self.problem)
            self._show_ticks()
        if q.is_choice:  # a choice, right or wrong, is answered: the working and the run follow
            self._answered = True
            self._solution = q.solution
            if q.predict and q.setup is not None:
                self._set_up(self.problem.page, q.setup)
        self._render()

    def show_hint(self) -> None:
        if self.problem is not None:
            self._hint = self.problem.hint
            self._render()

    def show_solution(self) -> None:
        if self.question is not None:
            self._solution = self.question.solution
            self._answered = True
            self._render()

    def show_on_page(self) -> None:
        """The question's setup, or a spec's design once its solution shows."""
        q = self.question
        if q is None:
            return
        setup = q.design if self._shows_design() else q.setup
        if setup is not None:
            self._set_up(self.problem.page, setup)

    def _shows_design(self) -> bool:
        return self.question is not None and self.question.design is not None and bool(self._solution)

    # --------------------------------------------------------------- drawing
    def _render(self) -> None:
        q = self.question
        if q is None:
            self.title.setText("Practice")
            self.text.setHtml(
                "<p>Pick a problem on the left. Each asks with new numbers every time; "
                "<i>New numbers</i> asks again.</p>"
            )
        else:
            self.title.setText(self.problem.title)
            parts = [f"<p>{q.prompt}</p>"]
            if q.predict:
                parts.append("<p><i>Predict it first: once you've answered, the page shows it.</i></p>")
            if q.is_spec:
                parts.append(
                    "<p><i>Start from <b>Show on the page</b>, tune the page's controls until it meets the spec, "
                    "then <b>Check</b>: it reads the page as it stands.</i></p>"
                )
            parts.append(self._verdict)
            parts.append(self._details)
            if self._hint:
                parts.append(f"<p><i>Hint:</i> {self._hint}</p>")
            if self._solution:
                parts.append(f"<p><i>Solution:</i> {self._solution}</p>")
            style = f"<style>.good {{ color: {colors.good}; }} .bad {{ color: {colors.poor}; }}</style>"
            self.text.setHtml(style + "".join(parts))
        asked = q is not None
        for b in (self.check_button, self.hint_button, self.solution_button, self.new_button):
            b.setEnabled(asked)
        self.page_button.setEnabled(asked and q.setup is not None and (not q.predict or self._answered))
        self.page_button.setText("Show a design on the page" if self._shows_design() else "Show on the page")
        self.answer_area.setVisible(asked and not q.is_spec)

    def _show_ticks(self) -> None:
        solved = _solved()
        for (page, key), item in self._problem_items.items():
            title = item.toolTip(0)
            item.setText(0, (TICK if f"{page}/{key}" in solved else "") + title)
        for page, item in self._page_items.items():
            found = problems(page)
            done = sum(f"{page}/{p.key}" in solved for p in found)
            name = item.text(0).split("  (")[0]
            item.setText(0, f"{name}  ({done}/{len(found)})")

    # --------------------------------------------------------------- window
    def apply_theme(self) -> None:
        self._render()  # the verdict's colour

    def save_geometry(self) -> None:
        QtCore.QSettings(*SETTINGS).setValue("practice_window/geometry", self.saveGeometry())

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.save_geometry()
        super().closeEvent(event)
