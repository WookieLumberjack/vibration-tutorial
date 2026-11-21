"""The Things to try as the Theory lists them, without Qt (#409 S-3's first step: a notes module that lists its
page's trials no longer loads PySide6 for it).

`trials_html(trials)` is the list a page's notes take as their `trials` part: a trial with a prediction by its
question alone, as its note gives the answer away (#409 C-9), its note put back once the reader has answered
(`progress.marks` fills `ANSWER` from `NOTES`, keyed by `prompt_key`).

Working on it: `tests/test_qt_free.py` imports it alone; `tests/test_practice.py` checks the predictions' notes.
"""

from __future__ import annotations

import zlib

from .maths import markup

NOTES: dict[str, str] = {}  # a predicted trial's note (rich text), by its prompt's key: `trials_html` fills it
ANSWER = "<!--ANSWER:{}-->"  # where a predicted trial's note goes in the Theory's list once it's answered


def prompt_key(prompt: str) -> str:
    """A short key for a prediction's prompt (the same text, the same key)."""
    return format(zlib.crc32(prompt.encode()), "08x")


def trials_html(trials) -> str:
    """The trials as the Theory tab lists them: a trial with a prediction by its question alone, as its note
    gives the answer away (#409 C-9), and the notes are where a page first opens (`MainWindow.introduce_pages`)."""

    def item(t) -> str:
        p = getattr(t, "predict", None)
        if p is not None:  # its note once its prediction has been answered (progress.marks)
            key = prompt_key(p.prompt)
            NOTES[key] = markup(t.note)
            ask = f"<i>Predict first: {markup(p.prompt)}</i>"
            return f"<li><b>{t.name}.</b> {ask} Pick it to answer and see.{ANSWER.format(key)}</li>"
        return f"<li><b>{t.name}.</b> {markup(t.note)}</li>"

    return "<ol>" + "".join(item(t) for t in trials) + "</ol>"
