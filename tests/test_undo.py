"""`core/undo.py`: the builders' undo stack of whole states (#409 S-12)."""

from vib_tutorial.core.undo import UndoStack


def test_undo_and_redo_walk_the_states_and_a_new_edit_forgets_the_undone():
    edits: UndoStack[str] = UndoStack()
    assert not edits.can_undo and edits.undo("a") is None and edits.redo("a") is None
    edits.record("a")  # a → b
    edits.record("b")  # b → c
    assert edits.undo("c") == "b" and edits.undo("b") == "a" and not edits.can_undo
    assert edits.redo("a") == "b" and edits.can_redo
    edits.record("b")  # b → d: the undone c is gone
    assert not edits.can_redo and edits.undo("d") == "b"
