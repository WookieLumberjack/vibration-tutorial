"""Drawing once per change: a page's redraws asked for while a trial is set up run once, at its end (#409 P-7).

Setting a page up for a trial is several changes in a row: its tab, its chain (which the Simulation page passes
back through `set_system`), then its boxes. Each one used to redraw what it touched, so a trial drew its slow
view two to seven times, each with a state only partly loaded. Now:

- `batch()` is the trial's setup: a `TrialBox` pick, a challenge's or a practice problem's setup, Defaults.
  Batches nest; the outermost one's end runs what was asked for.
- `@coalesced` on a page's redraw: called inside a batch, it's only noted (once per page and method, with the
  last call's arguments, after anything noted before it) and runs when the batch ends; outside one, at once.
- `tab_groups.Deferred` (and so `LazyTabs`) catches up the same way: a tab shown mid-trial draws once, with the
  trial's state, not first with the old one.

Working on it: a method marked `@coalesced` mustn't be one whose results the code after its call reads in the
same setup (that code would read the old ones); mark the page's outermost redraw, not a step that computes what
the rest uses. Calls straight to a page's `load_trial` (the tests, the Practice window's checks) aren't in a
batch, so they draw as before and read their plots at once. `tests/test_trial_redraws.py` counts each page's
redraws per trial against its budget; `tests/test_coalesce.py` checks the mechanism.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, cast

_depth = 0
_pending: dict[tuple[int, str], Callable[[], object]] = {}


def active() -> bool:
    """Inside a batch: a coalesced redraw waits for its end."""
    return _depth > 0


def defer(owner: object, name: str, call: Callable[[], object]) -> None:
    """`call` at the batch's end, once for `owner`'s `name` (a later one replaces it, and moves to the end)."""
    key = (id(owner), name)
    _pending.pop(key, None)
    _pending[key] = call


@contextmanager
def batch() -> Iterator[None]:
    """Coalesced redraws asked for inside run once, when the outermost batch ends."""
    global _depth
    _depth += 1
    try:
        yield
    finally:
        _depth -= 1
        if _depth == 0:
            flush()


def flush() -> None:
    """Run what was deferred, in the order it was last asked for; what that asks for in turn runs too."""
    while _pending:
        key = next(iter(_pending))
        _pending.pop(key)()


def coalesced[F: Callable[..., Any]](method: F) -> F:
    """A page's redraw that runs once per batch, at its end (see the module's docstring); returns None when
    deferred, so mark only a method whose callers don't use what it returns."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if _depth == 0:
            return method(self, *args, **kwargs)
        defer(self, method.__name__, functools.partial(method, self, *args, **kwargs))
        return None

    return cast(F, wrapper)
