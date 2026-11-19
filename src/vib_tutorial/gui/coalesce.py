"""Drawing once per change: a page's redraws asked for while a trial is set up run once, at its end (#409 P-7).

Setting a page up for a trial is several changes in a row: its tab, its chain (which the Simulation page passes
back through `set_system`), then its boxes. Each one used to redraw what it touched, so a trial drew its slow
view two to seven times, each with a state only partly loaded. Now:

- `batch()` is the trial's setup: a `TrialBox` pick, a challenge's or a practice problem's setup, Defaults.
  Batches nest; the outermost one's end runs what was asked for.
- `@coalesced` on a page's redraw: called inside a batch, it's only noted (once per page and method, with the
  last call's arguments, after anything noted before it) and runs when the batch ends; outside one, at once.
  `@coalesced(covers=("_draw_tab",))` on a redraw that draws the tab itself: a trial that switches the tab
  (asking for `_draw_tab`) and then redraws runs `redraw` alone, `_draw_tab` once inside it.
  `@coalesced(leaf=True)` on a view's own draw (`DampingView._draw`), which several of its page's setters ask
  for: it waits until the page's redraws have all run. While the batch's end runs them it stays open, so what
  they ask for merges with what's waiting.
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
import inspect
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, cast

MOST_CALLS = 10_000  # a batch's end runs at most this many: past it, redraws are asking for each other in a loop
_depth = 0
_pending: dict[tuple[int, str], Callable[[], object]] = {}
_leaves: dict[tuple[int, str], Callable[[], object]] = {}  # views' draws: after every page-level redraw
_covers: dict[tuple[int, str], tuple[str, ...]] = {}  # what each waiting redraw runs itself


def active() -> bool:
    """Inside a batch: a coalesced redraw waits for its end."""
    return _depth > 0


def defer(
    owner: object, name: str, call: Callable[[], object], leaf: bool = False, covers: tuple[str, ...] = ()
) -> None:
    """`call` at the batch's end, once for `owner`'s `name` (a later one replaces it, and moves to the end);
    a `leaf` (a view's own draw) after everything else. `covers`: `owner`'s methods this one runs itself, so
    waiting ones are dropped; and this one is dropped while one that covers it waits."""
    queue = _leaves if leaf else _pending
    key = (id(owner), name)
    if any(name in _covers.get(k, ()) for k in _pending if k[0] == key[0] and k != key):
        return  # a waiting redraw of the same owner draws it
    for part in covers:
        _pending.pop((key[0], part), None)
    _covers[key] = covers
    queue.pop(key, None)
    queue[key] = call


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
    """Run what was deferred, in the order it was last asked for, the leaves last. The batch stays open while it
    runs, so what those ask for in turn is deferred too, and merges with what's already waiting: a view asked to
    draw by its page's redraw and again by its playback draws once."""
    global _depth
    _depth += 1
    try:
        for _ in range(MOST_CALLS):
            if not (_pending or _leaves):
                break
            queue = _pending or _leaves
            key = next(iter(queue))
            call = queue.pop(key)
            _covers.pop(key, None)
            call()
        else:
            _pending.clear()
            _leaves.clear()
            _covers.clear()
            raise RuntimeError("coalesce.flush: redraws kept asking for each other")
    finally:
        _depth -= 1


def coalesced[F: Callable[..., Any]](
    method: F | None = None, *, leaf: bool = False, covers: tuple[str, ...] = ()
) -> Any:
    """A page's redraw that runs once per batch, at its end (see the module's docstring); returns None when
    deferred, so mark only a method whose callers don't use what it returns. `leaf=True` for a view's own draw,
    which its page's redraws feed: it runs after them. `covers`: the page's other coalesced methods this one
    calls itself (`redraw` covers `_draw_tab`), so asking for both runs this one alone."""

    def mark(method: F) -> F:
        # A signal passes what it carries (a tab's index) to a slot that takes it: PySide reads a method's
        # signature for that, and this wrapper's would take anything, so it drops what the method can't take.
        params = list(inspect.signature(method).parameters.values())[1:]
        takes = (
            None
            if any(p.kind is p.VAR_POSITIONAL for p in params)
            else sum(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in params)
        )

        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            args = args if takes is None else args[:takes]
            if _depth == 0:
                return method(self, *args, **kwargs)
            defer(self, method.__name__, functools.partial(method, self, *args, **kwargs), leaf, covers)
            return None

        return cast(F, wrapper)

    return mark(method) if method is not None else mark
