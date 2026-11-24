"""`gui/bindings.py`: a state dataclass's fields bound to their controls, read and written as one (#478 item 7)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")


@dataclass(frozen=True)
class State:
    name: str
    size: float = 1.0
    on: bool = False
    kind: int = 0
    gap_m: float = 0.002


def test_read_write_and_changed(app):
    from vib_tutorial.gui.bindings import Bindings

    holder = QtWidgets.QWidget()
    size, on, kind, gap = (
        QtWidgets.QDoubleSpinBox(holder),
        QtWidgets.QCheckBox(holder),
        QtWidgets.QComboBox(holder),
        (QtWidgets.QDoubleSpinBox(holder)),
    )
    kind.addItems(["a", "b"])
    gap.setRange(0, 100)
    b = Bindings(State, holder)
    b.bind("size", size)
    b.bind("on", on)
    b.bind("kind", kind)
    b.bind("gap_m", gap, scale=1e-3)  # mm in the box
    seen: list[str] = []
    b.changed.connect(seen.append)
    state = State("x", 3.5, True, 1, 0.007)
    b.write(state)
    assert seen == []  # written quietly
    assert (size.value(), on.isChecked(), kind.currentIndex(), gap.value()) == (3.5, True, 1, 7.0)
    assert b.read(State("x")) == state  # round-trips by construction
    kind.setCurrentIndex(0)
    on.setChecked(False)
    assert seen == ["kind", "on"]
    with pytest.raises(KeyError):
        b.bind("colour", size)
    with pytest.raises(KeyError):
        b.bind("size", size)
    holder.deleteLater()
