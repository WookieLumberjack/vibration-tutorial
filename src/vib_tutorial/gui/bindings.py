"""Controls bound to a page's state (#478 item 7, #409 S-2): each field of the state's dataclass to its widget, once.

A page lists its bindings, `field → widget`, and gets from them:

- `read(base)`: the state the controls show, `base` with the bound fields replaced (`practice_state`);
- `write(state)`: the controls set to a state, quietly (no signal: the page redraws once, after);
- `changed(field)`: the reader changed a control (spin box, check box, combo box, tab widget).

So Defaults round-trips by construction: `write(read(base))` leaves every control as it was, and a field added to
the state and bound here is read, written and watched with nothing else to keep in step. A widget's value can be
scaled into the field (`scale`: mm in the box, m in the state), and a combo's index pick one of `choices` (an
enum's members: Damping models' model).

Working on it: `tests/test_bindings.py`; `tests/test_defaults.py` checks every page round-trips.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

from PySide6 import QtCore, QtWidgets

from . import weak


@dataclass(frozen=True)
class Binding:
    field: str
    widget: QtWidgets.QWidget
    scale: float = 1.0  # the field's value is the widget's times this
    choices: tuple | None = None  # a combo's field is one of these (an enum's members), by its index


def _get(w: QtWidgets.QWidget) -> Any:
    if isinstance(w, QtWidgets.QAbstractButton):
        return w.isChecked()
    if isinstance(w, (QtWidgets.QComboBox, QtWidgets.QTabWidget)):
        return w.currentIndex()
    if isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
        return w.value()
    raise TypeError(f"no binding for a {type(w).__name__}")


def _set(w: QtWidgets.QWidget, value: Any) -> None:
    w.blockSignals(True)
    try:
        if isinstance(w, QtWidgets.QAbstractButton):
            w.setChecked(bool(value))
        elif isinstance(w, (QtWidgets.QComboBox, QtWidgets.QTabWidget)):
            w.setCurrentIndex(int(value))
        elif isinstance(w, QtWidgets.QSpinBox):
            w.setValue(int(value))
        elif isinstance(w, QtWidgets.QDoubleSpinBox):
            w.setValue(float(value))
        else:
            raise TypeError(f"no binding for a {type(w).__name__}")
    finally:
        w.blockSignals(False)


def _signal(w: QtWidgets.QWidget):
    if isinstance(w, QtWidgets.QAbstractButton):
        return w.toggled
    if isinstance(w, (QtWidgets.QComboBox, QtWidgets.QTabWidget)):
        return w.currentChanged if isinstance(w, QtWidgets.QTabWidget) else w.currentIndexChanged
    if isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
        return w.valueChanged
    raise TypeError(f"no binding for a {type(w).__name__}")


class Bindings(QtCore.QObject):
    """A state dataclass's fields bound to their controls (see the module)."""

    changed = QtCore.Signal(str)  # the field whose control the reader changed

    def __init__(self, state_type: type, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.state_type = state_type
        self._names = {f.name for f in fields(state_type)}
        self.bindings: list[Binding] = []

    def bind(
        self, field: str, widget: QtWidgets.QWidget, scale: float = 1.0, choices: tuple | None = None
    ) -> QtWidgets.QWidget:
        """Bind `field` to `widget` (whose value times `scale` is the field's; a combo's index picks from `choices`,
        an enum's members, where the field is one): the widget, for chaining."""
        if field not in self._names:
            raise KeyError(f"{self.state_type.__name__} has no field {field!r}")
        if any(b.field == field for b in self.bindings):
            raise KeyError(f"{field!r} is bound already")
        self.bindings.append(Binding(field, widget, scale, choices))
        _signal(widget).connect(weak.slot(self, lambda self, *_, f=field: self.changed.emit(f)))
        return widget

    def widget(self, field: str) -> QtWidgets.QWidget:
        return next(b.widget for b in self.bindings if b.field == field)

    def read(self, base: Any) -> Any:
        """`base` with every bound field as its control shows it."""
        values = {}
        for b in self.bindings:
            value = _get(b.widget)
            if b.choices is not None:
                value = b.choices[value]
            values[b.field] = value * b.scale if b.scale != 1.0 else value
        return replace(base, **values)

    def write(self, state: Any) -> None:
        """Every bound control set to `state`'s value, without its signal."""
        for b in self.bindings:
            value = getattr(state, b.field)
            if b.choices is not None:
                value = b.choices.index(value)
            _set(b.widget, value / b.scale if b.scale != 1.0 else value)
