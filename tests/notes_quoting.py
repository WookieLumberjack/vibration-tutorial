"""`quoted`: a number as the notes print it, for a test to check against what the code works out.

`quoted("<notes file>", "1.7 %")` fails unless `theory/<notes file>.html` shows that number (outside its `<eq>`s)
and gives its value (1.7). Any test module may use it (`from notes_quoting import quoted`, with two string
literals); `test_tutorial_numbers.py`'s `test_every_quoted_number_is_checked` reads every module's calls, and fails on
a decimal or percentage in the notes that none ties and `notes_unchecked_numbers.txt` doesn't list.
"""

import functools
import html
import re
from pathlib import Path

import vib_tutorial

THEORY = Path(vib_tutorial.__file__).parent / "theory"
NUMBER = re.compile(r"(?<![\w.])\d+\.\d+(?:\s*%)?|(?<![\w.])\d+\s*%")


@functools.cache
def notes_numbers(notes: str) -> frozenset[str]:
    """The decimals and percentages `theory/<notes>.html` shows outside its `<eq>`s ("0.19 %", "2.010")."""
    text = (THEORY / f"{notes}.html").read_text(encoding="utf-8")
    text = re.sub(r"<eq>.*?</eq>|<!--.*?-->", " ", text, flags=re.S)
    # table cells, blocks and exponents stay apart ("0.878<sup>2</sup>" isn't 0.8782)
    text = re.sub(r"</?(?:td|th|tr|li|p|br|div|table|h\d|sup|sub)\b[^>]*>", " ", text)
    text = html.unescape(re.sub(r"<[^>]+>", "", text))
    return frozenset(re.sub(r"\s*%", " %", n) for n in NUMBER.findall(text))


def quoted(notes: str, number: str) -> float:
    """`number`, as `theory/<notes>.html` shows it ("1.7 %" gives 1.7): fails if the notes don't."""
    assert number in notes_numbers(notes), f"{notes}.html doesn't quote {number}"
    return float(number.removesuffix(" %"))
