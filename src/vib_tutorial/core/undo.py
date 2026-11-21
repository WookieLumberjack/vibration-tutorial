"""An undo stack of whole states (#409 S-12): the frame builder's and the rotor builder's edits.

A state is an immutable value (a `Frame`, a `Rotor`), so undoing is putting the one before back. `record(now)`
keeps `now` before an edit replaces it (and forgets what was undone); `undo(now)` and `redo(now)` give the state
to show, keeping `now` for the way back, or None with nothing to go to.

Working on it: `tests/test_undo.py`; the builders' tests undo and redo through their buttons.
"""

from __future__ import annotations


class UndoStack[T]:
    def __init__(self) -> None:
        self._back: list[T] = []
        self._forward: list[T] = []

    def record(self, now: T) -> None:
        """`now` is about to be replaced by an edit: keep it, and forget what was undone."""
        self._back.append(now)
        self._forward.clear()

    def undo(self, now: T) -> T | None:
        if not self._back:
            return None
        self._forward.append(now)
        return self._back.pop()

    def redo(self, now: T) -> T | None:
        if not self._forward:
            return None
        self._back.append(now)
        return self._forward.pop()

    @property
    def can_undo(self) -> bool:
        return bool(self._back)

    @property
    def can_redo(self) -> bool:
        return bool(self._forward)
