"""Practice problems: questions made afresh from a seed, answered from the same models the pages draw.

A `Problem` belongs to a page and makes a `Question` from a random generator, so each attempt has new numbers.
Its answer is worked out by the page's own `core` function from the numbers as the question prints them, so the
key can't drift from what the page shows. A question is either a number, typed in its `unit` and right within a
relative `tolerance`, or a choice among `choices`. A *prediction* is a choice whose `setup` (the page's settings
that show the answer, as its trials take them) is held back until it's answered: predict, then run it.

`check` marks an answer and, when a number is wrong, says if it's off by one of the usual slips (2π: rad/s for
Hz; 1000: a unit's prefix; 2: amplitude for peak to peak). No Qt here: the Practice window shows them.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np

TOLERANCE = 0.02  # a number's relative tolerance, unless the question sets its own
SLIPS = (  # (factor, what it suggests) for a number that's out by a factor, either way
    (2.0 * math.pi, "a factor of 2π: rad/s where the question wants Hz, or the other way?"),
    (1000.0, "a factor of 1000: check the unit's prefix (mm against m, g against kg)."),
    (2.0, "a factor of 2: an amplitude against a peak to peak, or a missing 2 in 2ζ?"),
    (math.sqrt(2.0), "a factor of √2: the half-power points are where |H| falls to 1/√2 of the peak."),
)
SLIP_BAND = 0.03  # how close the ratio has to come to a slip's factor for the hint


@dataclass(frozen=True)
class Question:
    """One problem's question, with its numbers filled in.

    `answer` is the number (in `unit`) or the index into `choices`. `setup` is the page's settings that show the
    situation (keyword arguments for its trials), or None; with `predict` it's offered only once answered.
    """

    prompt: str  # rich text
    answer: float | int
    solution: str  # rich text: the working
    unit: str = ""
    choices: tuple[str, ...] = ()
    tolerance: float = TOLERANCE
    setup: dict | None = None
    predict: bool = False

    @property
    def is_choice(self) -> bool:
        return bool(self.choices)


@dataclass(frozen=True)
class Problem:
    """A kind of question on a page: `make(rng)` gives a fresh one. `key` is stable (progress is kept by it)."""

    key: str
    title: str
    page: str  # the page's attribute, as in links.COURSE
    hint: str  # rich text; `page:` links to the notes that explain it
    make: Callable[[np.random.Generator], Question] = field(repr=False)

    def question(self, seed: int | None = None) -> Question:
        """A fresh question; a choice's options are shuffled (`make` may list the right one anywhere)."""
        rng = np.random.default_rng(seed)
        q = self.make(rng)
        if not q.is_choice:
            return q
        order = rng.permutation(len(q.choices))
        return replace(q, choices=tuple(q.choices[i] for i in order), answer=int(np.flatnonzero(order == q.answer)[0]))


@dataclass(frozen=True)
class Verdict:
    right: bool
    message: str  # rich text


def parse_number(text: str) -> float | None:
    """A typed number: a comma for the decimal point, a Unicode minus and ×10^ are taken too. None if not one."""
    s = text.strip().replace("−", "-").replace(",", ".").replace(" ", "")
    s = s.replace("×10^", "e").replace("x10^", "e").replace("*10^", "e")
    try:
        value = float(s)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def is_close(value: float, answer: float, tolerance: float) -> bool:
    return abs(value - answer) <= tolerance * abs(answer)


def check(question: Question, response: str | int) -> Verdict:
    """Mark `response`: the typed text for a number, the choice's index for a choice."""
    if question.is_choice:
        right = int(response) == question.answer
        return Verdict(right, "Right." if right else "Not that one.")
    value = parse_number(str(response))
    if value is None:
        return Verdict(False, "That isn't a number.")
    if is_close(value, question.answer, question.tolerance):
        return Verdict(True, f"Right: {number(question.answer)} {question.unit}".strip() + ".")
    message = "Not quite."
    if value != 0 and question.answer != 0:
        ratio = abs(value / question.answer)
        for factor, slip in SLIPS:
            if any(abs(ratio * f - 1.0) < SLIP_BAND for f in (factor, 1.0 / factor)):
                message = f"Not quite: you're out by {slip}"
                break
        else:
            if value * question.answer < 0 and is_close(-value, question.answer, question.tolerance):
                message = "Not quite: check the sign."
    return Verdict(False, message)


def number(value: float, digits: int = 3) -> str:
    """`value` to `digits` significant figures, as a question prints it (no exponent between 10⁻³ and 10⁶)."""
    if value == 0 or not math.isfinite(value):
        return f"{value:g}"
    exponent = math.floor(math.log10(abs(value)))
    if -3 <= exponent < 6:
        decimals = max(digits - 1 - exponent, 0)
        return f"{rounded(value, digits):.{decimals}f}"
    mantissa = f"{value / 10 ** exponent:.{digits - 1}f}"
    return f"{mantissa}×10<sup>{exponent}</sup>"


def rounded(value: float, digits: int = 3) -> float:
    """`value` as `number` prints it, so an answer worked from the printed numbers matches what the reader has."""
    if value == 0 or not math.isfinite(value):
        return value
    return round(value, digits - 1 - math.floor(math.log10(abs(value))))


def pick(rng: np.random.Generator, values) -> float:
    """One of `values`, at random (a plain float, not a NumPy scalar)."""
    return float(values[int(rng.integers(len(values)))])

