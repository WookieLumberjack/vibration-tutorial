"""Slots that hold their widget weakly, so a window can be freed once nothing uses it.

PySide keeps a lambda, closure or `functools.partial` connected to a signal for as long as the
sender lives, and the sender is a child the window owns on the C++ side. One that mentions `self`
(the page, the window) makes a cycle through C++, where `gc` can't see it, and the window lives
until the process ends: each GUI test's windows stayed, 20 MB and more of them. A bound method
connected directly is fine (PySide ties it to its object's lifetime), so connect `self.m` where it
takes no parameters or just the signal's, a signal (`self.edit_parameters`) to pass one on,
`call(self.m)` for a method whose defaulted parameters the signal's arguments would fill, and
`slot(self, lambda self, ...: ...)` for the rest. A `QTimer.singleShot` given a context object
(`singleShot(0, widget, fn)`) is dropped with it.

Working on it: conftest's `free_windows` (the GUI tests' autouse fixtures) fails a test whose
windows outlive it, and `tests/test_weak.py` checks `slot` and `call` and that no `gui/` module
hands a signal a slot holding `self`.
"""

from __future__ import annotations

import inspect
import weakref
from collections.abc import Callable
from typing import Any


def slot(obj: object, fn: Callable[..., Any]) -> Callable[..., Any]:
    """A slot calling `fn(obj, *args)` while `obj` lives, holding `obj` weakly.

    It passes `fn` as many of the signal's arguments as `fn` requires after `obj` (all of them
    if it takes `*args`), so a parameter with a default keeps it: `lambda self, _=False, m=mode:`
    on a `clicked(bool)` gets `m`, not the bool. Bind other widgets of the window the same way
    (as defaults, or through `self`): a closure over them can hold the window too."""
    ref = weakref.ref(obj)
    params = list(inspect.signature(fn).parameters.values())[1:]
    take = (
        None
        if any(p.kind is p.VAR_POSITIONAL for p in params)
        else sum(p.default is p.empty and p.kind is not p.KEYWORD_ONLY for p in params)
    )

    def call(*args: Any) -> Any:
        if (alive := ref()) is not None:
            return fn(alive, *(args if take is None else args[:take]))
        return None

    return call


def call(method: Callable[[], Any]) -> Callable[..., Any]:
    """A slot calling the bound `method()` with none of the signal's arguments, holding its object weakly
    (PySide would pass a method with defaulted parameters the signal's)."""
    func = getattr(method, "__func__", None)
    if func is None:  # not a bound method: nothing to hold weakly
        return lambda *_: method()
    return slot(method.__self__, lambda obj, *_: func(obj))  # type: ignore[attr-defined]
