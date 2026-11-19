"""A page's spec problems as challenges on the page itself, checked live while the reader tunes its controls.

A *spec* (`core.practice`: a `Question` with a `goal`) asks for settings that meet some requirements: an
isolation at a speed, a ride within a travel. The Practice window asks them too, but there the reader checks by
pressing *Check*. A `ChallengeBox`, under a page's Things to try, lists the page's specs (`challenges`):

- picking one makes its question (new numbers each time; *New numbers* again), shows it, and sets the page up
  where the tuning starts (its `setup`, through the page's `load_setup`), as a trial would;
- while the box shows, it reads the page's settings (`practice_state()`) a few times a second, and when they've
  changed lists each requirement, met (✓) or not (✗), with what the page has now (a number the spec gives,
  `kept`, only once it's been moved: the rest are counted);
- once every one is met it says so and counts the problem solved, as the Practice window does
  (`practice.mark_solved`), on the course map too.

Reading the settings is cheap, and the goal is worked out only when they change, so no page has to say when
its controls move. *Defaults* (`MainWindow.restore_defaults`) puts the box back at its prompt (`clear`).

Working on it: `tests/test_challenge.py` tunes a page to meet a spec through its boxes and checks the box
follows, and that every page with a box lists only specs that set it up.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
from PySide6 import QtCore, QtWidgets

from ..core.practice import Problem, Question
from . import coalesce
from .lesson import note_label

POLL_MS = 250  # how often the box reads the page's settings while it shows
PROMPT = "Pick a challenge…"


class Tunable(Protocol):
    """A page with specs: its settings as its trial dataclass, and setting it up from a problem's `setup`."""

    def practice_state(self) -> Any: ...

    def load_setup(self, setup: dict) -> None: ...


def challenges(problems: Sequence[Problem]) -> tuple[Problem, ...]:
    """The specs among a page's problems, in their order."""
    return tuple(p for p in problems if p.question(0).is_spec)


def title(problem: Problem) -> str:
    """A spec's title without its "Spec: " (the box says what they are)."""
    name = problem.title.removeprefix("Spec: ")
    return name[:1].upper() + name[1:]


class ChallengeBox(QtWidgets.QGroupBox):
    solved = QtCore.Signal(str)  # the problem's key, the first time its spec is met

    def __init__(self, page: Tunable, problems: Sequence[Problem], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Challenge", parent)
        self.page = page
        self.problems = challenges(problems)
        self.problem: Problem | None = None
        self.question: Question | None = None
        self.met = False
        self._last: Any = None  # the settings last checked
        self._rng = np.random.default_rng()
        self._seed: int | None = None  # the next pick's numbers (tests), or new ones
        layout = QtWidgets.QVBoxLayout(self)
        self.combo = QtWidgets.QComboBox()
        self.combo.addItem(PROMPT)
        self.combo.addItems([title(p) for p in self.problems])
        self.combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo.setAccessibleName("Challenge")
        self.combo.setToolTip(
            "<p>A spec to meet by tuning the page's controls: pick one and the page sets up where to start; each "
            "requirement is ticked as you meet it.</p>"
        )
        self.combo.currentIndexChanged.connect(self._on_pick)
        layout.addWidget(self.combo)
        self.prompt = QtWidgets.QLabel()
        self.prompt.setWordWrap(True)
        self.prompt.setTextFormat(QtCore.Qt.TextFormat.RichText)
        layout.addWidget(self.prompt)
        self.status = note_label()
        self.status.setTextFormat(QtCore.Qt.TextFormat.RichText)
        layout.addWidget(self.status)
        row = QtWidgets.QHBoxLayout()
        self.restart = QtWidgets.QPushButton("Start over")
        self.restart.setToolTip("<p>Set the page up where this challenge starts again.</p>")
        self.restart.clicked.connect(self.start_over)
        self.again = QtWidgets.QPushButton("New numbers")
        self.again.setToolTip("<p>The same challenge with new numbers.</p>")
        self.again.clicked.connect(self.new_numbers)
        row.addWidget(self.restart)
        row.addWidget(self.again)
        layout.addLayout(row)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(POLL_MS)
        self.timer.timeout.connect(self.poll)
        self._show_parts(False)

    # ---------------------------------------------------------------- picking
    def pick(self, key: str, seed: int | None = None) -> None:
        """Pick the challenge `key` (a problem's), as the reader would; `seed` fixes its numbers (tests)."""
        i = next(i for i, p in enumerate(self.problems) if p.key == key)
        self._seed = seed
        if self.combo.currentIndex() == i + 1:
            self._on_pick(i + 1)
        else:
            self.combo.setCurrentIndex(i + 1)

    def clear(self) -> None:
        """Back to the prompt, checking nothing (the page put back as it opened)."""
        self.combo.blockSignals(True)
        self.combo.setCurrentIndex(0)
        self.combo.blockSignals(False)
        self._on_pick(0)

    def new_numbers(self) -> None:
        if self.problem is not None:
            self._ask(self.problem, None)

    def start_over(self) -> None:
        q = self.question
        if q is not None and q.setup is not None:
            with coalesce.batch():  # the page's redraws once, with the whole setup loaded
                self.page.load_setup(dict(q.setup))
        self.poll(force=True)

    def _on_pick(self, index: int) -> None:
        seed, self._seed = self._seed, None
        if index < 1:
            self.problem = self.question = None
            self.timer.stop()
            self._show_parts(False)
            return
        self._ask(self.problems[index - 1], seed)

    def _ask(self, problem: Problem, seed: int | None) -> None:
        self.problem = problem
        self.question = problem.question(int(self._rng.integers(2**31)) if seed is None else seed)
        self.met = False
        self.prompt.setText(self.question.prompt)
        self._show_parts(True)
        self.start_over()
        self.timer.start()

    def _show_parts(self, on: bool) -> None:
        for widget in (self.prompt, self.status, self.restart, self.again):
            widget.setVisible(on)

    # --------------------------------------------------------------- checking
    def poll(self, force: bool = False) -> None:
        """Check the page's settings against the spec, if they've changed (or `force`)."""
        q, p = self.question, self.problem
        if q is None or p is None or (not force and not self.isVisible()):
            return
        state = self.page.practice_state()
        try:
            same = bool(state == self._last)
        except ValueError:  # (settings holding arrays)
            same = False
        if same and not force:
            return
        self._last = state
        assert q.goal is not None, "only specs are challenges"
        found = q.goal(state)
        right = all(r.met for r in found)
        if right and not self.met:
            self.met = True
            from .practice import mark_solved  # (the Practice window's module; built only when it opens)

            mark_solved(p)
            self.solved.emit(p.key)
        missed = sum(not r.met for r in found)
        head = "<b>✓ It meets the spec.</b> Solved." if right else f"<b>Not yet: {missed} of {len(found)} not met.</b>"
        rows = "".join(  # a number the spec gives is listed only once it's been moved
            f"<p>{'✓' if r.met else '✗'} {r.wanted} (now {r.now})</p>" for r in found if not (r.given and r.met)
        )
        kept = sum(r.given and r.met for r in found)
        tail = (
            f"<p><i>{kept} given number{'s' if kept != 1 else ''} kept as {'they are' if kept != 1 else 'it is'}.</i></p>"
            if kept
            else ""
        )
        self.status.setText(f"<p>{head}</p>{rows}{tail}")
