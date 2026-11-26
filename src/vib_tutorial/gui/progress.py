"""What the reader has done beyond the problems solved (#409 I-9): the Things to try tried and the predictions answered.

They're kept in the settings beside the problems solved (`SOLVED`, which `practice` keeps): a trial as
"page#n" (its number in the page's list, as a goal's link names it), a prediction by its prompt's key
(`prompt_key`). They show in three places:

- each page's Things to try ticks the trials tried (`lesson.TrialBox.show_tried`);
- the notes (`marks`, which `notes_view.NotesBrowser` applies to whatever it shows, so every page's Theory tab and
  the Theory window): a goal (*On this page you'll*) ticks its trial once tried and its problem once solved, and
  is ticked itself when both are; and a predicted trial, which the Theory lists by its question alone (its note
  gives the answer away, #409 C-9), shows its note again once its prediction has been answered;
- the first page's course map counts each page's trials tried beside its problems solved (`tried_counts`).

A browser draws its notes again when it shows if anything has been marked since (`version`).

Working on it: `tests/test_progress.py`. Reset progress (the Practice window's) forgets these too (`reset`).
"""

from __future__ import annotations

import re

from .theming import settings
from .trials_text import ANSWER as ANSWER  # re-exported: the trials' Theory list and `marks`
from .trials_text import NOTES as NOTES
from .trials_text import prompt_key as prompt_key

SOLVED = "practice/solved"  # "page/key" of each problem solved
TRIED = "progress/tried"  # "page#n" of each trial tried
ANSWERED = "progress/answered"  # the prompt key of each prediction answered

_version = 0


def _read(key: str) -> set[str]:
    return set(filter(None, str(settings().value(key, "")).split(",")))


def _add(key: str, item: str) -> None:
    global _version
    now = _read(key)
    if item not in now:
        settings().setValue(key, ",".join(sorted(now | {item})))
        _version += 1


def version() -> int:
    """A number that changes whenever something is marked (or forgotten)."""
    return _version


def solved() -> set[str]:
    return _read(SOLVED)


def tried() -> set[str]:
    return _read(TRIED)


def answered() -> set[str]:
    return _read(ANSWERED)


def mark_solved(page: str, key: str) -> None:
    _add(SOLVED, f"{page}/{key}")


def mark_tried(page: str, number: int) -> None:
    """Trial `number` (from 1, as the page lists them) of page `page` tried."""
    _add(TRIED, f"{page}#{number}")


def mark_answered(prompt: str) -> None:
    _add(ANSWERED, prompt_key(prompt))


def reset() -> None:
    """Forget it all: problems solved, trials tried and predictions answered."""
    global _version
    for key in (SOLVED, TRIED, ANSWERED):
        settings().remove(key)
    _version += 1


def tried_numbers(page: str) -> set[int]:
    """The numbers (from 1) of page `page`'s trials tried."""
    return {int(item.partition("#")[2]) for item in tried() if item.partition("#")[0] == page}


def tried_counts(totals: dict[str, int]) -> dict[str, tuple[int, int]]:
    """(tried, trials) of each page in `totals` (its number of trials) with a trial tried."""
    out = {}
    for page, total in totals.items():
        n = sum(1 for k in tried_numbers(page) if k <= total)
        if n:
            out[page] = (n, total)
    return out


GOAL = re.compile(
    r'<li>((?:(?!<li>).)*?) <a href="trial:(\w+)#(\d+)">Try it</a> · <a href="practice:(\w+)#([\w-]+)">Practice it</a></li>',
    re.S,  # a goal's text runs over lines, but never across another item
)
REVEAL = re.compile(r"Pick it to answer and see\.<!--ANSWER:(\w+)-->")


def marks(html: str) -> str:
    """`html` (a page's notes) with the reader's progress marked: its goals ticked, its answered predictions'
    notes shown (see the module's docstring)."""
    if "trial:" not in html and "<!--ANSWER:" not in html:
        return html
    done_trials, done_problems, done_answers = tried(), solved(), answered()

    def goal(m: re.Match) -> str:
        text, page, n, practice_page, key = m.groups()
        t, p = f"{page}#{n}" in done_trials, f"{practice_page}/{key}" in done_problems
        tick = "✓ " if t and p else ""
        return (
            f'<li>{tick}{text} <a href="trial:{page}#{n}">Try it</a>{" ✓" if t else ""} · '
            f'<a href="practice:{practice_page}#{key}">Practice it</a>{" ✓" if p else ""}</li>'
        )

    def reveal(m: re.Match) -> str:
        key = m.group(1)
        return NOTES[key] if key in done_answers and key in NOTES else m.group(0)

    return REVEAL.sub(reveal, GOAL.sub(goal, html))
