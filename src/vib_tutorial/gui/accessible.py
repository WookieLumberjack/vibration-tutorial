"""Names for screen readers: each control named after the label it sits beside (#218 U5).

Qt names a control after its *buddy* label, which `QFormLayout.addRow(label, field)` sets when the field is the
control itself. A control that isn't a label's buddy has no name, and a screen reader reads out only "spin box".
Two cases leave controls unnamed:

- a field that is a row of widgets, as in a builder's *Selected* form, where the label's buddy is the row;
- a label and its control in a row of their own (`layouts.group`, a QHBoxLayout), with no form.

`name_controls(root)` names every unnamed control under `root` after the text a sighted reader would take as
its label: the form row's label, or the nearest label before it in its own row. A control with text of its own
(a button, a check box) or a name already set is left as it is. The main window calls it on each page as the
page is built.

Working on it: `tests/test_accessible.py` checks every page's controls have a name.
"""

from __future__ import annotations

import re
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

CONTROLS = (
    QtWidgets.QAbstractSpinBox,
    QtWidgets.QComboBox,
    QtWidgets.QAbstractSlider,
    QtWidgets.QLineEdit,
    QtWidgets.QAbstractButton,
)
TAG = re.compile(r"<[^>]+>")


def name(widget: QtWidgets.QWidget) -> str:
    """What a screen reader reads as `widget`'s name."""
    face = QtGui.QAccessible.queryAccessibleInterface(widget)
    return face.text(QtGui.QAccessible.Text.Name).strip() if face is not None else ""


def labelled(widget: QtWidgets.QWidget) -> bool:
    """Whether `widget` has a name of its own: one set, a button's text, or a label whose buddy it is. (Qt
    otherwise reads out the title of the group box it's in, which names the panel, not the control.)"""
    if widget.accessibleName():
        return True
    if isinstance(widget, QtWidgets.QAbstractButton) and widget.text().strip():
        return True
    parent = widget.parentWidget()
    if parent is None:
        return False
    direct = QtCore.Qt.FindChildOption.FindDirectChildrenOnly
    return any(label.buddy() is widget for label in parent.findChildren(QtWidgets.QLabel, options=direct))


def controls(root: QtWidgets.QWidget) -> list[QtWidgets.QWidget]:
    """The controls a reader can use under `root`: not a spin box's or combo's own parts, nor scroll bars."""
    found: list[QtWidgets.QWidget] = []
    for widget in root.findChildren(QtWidgets.QWidget):
        if not isinstance(widget, CONTROLS) or isinstance(widget, QtWidgets.QScrollBar):
            continue
        if isinstance(widget.parentWidget(), (QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox)):
            continue
        if isinstance(widget.parentWidget(), QtWidgets.QAbstractItemView):  # a table's corner button
            continue
        if widget.isHidden():  # hidden on purpose (a second plane's step while balancing in one)
            continue
        found.append(widget)
    return found


def _text(label: QtWidgets.QWidget | None) -> str:
    """A label's text, or a check box's (a check box before a spin box labels it: "Step at [50 %]")."""
    if not isinstance(label, QtWidgets.QLabel | QtWidgets.QCheckBox):
        return ""
    text = TAG.sub("", label.text()).replace("&nbsp;", " ").strip()
    return text.rstrip(":").strip()


def _path(layout: QtWidgets.QLayout, target: object) -> list[tuple[QtWidgets.QLayout, int]] | None:
    """The layouts from `layout` down to the one holding `target` (a widget or a layout), with its index in each."""
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item is None:
            continue
        if item.widget() is target or item.layout() is target:
            return [(layout, i)]
        inner = item.layout()
        if inner is not None and (found := _path(inner, target)) is not None:
            return [(layout, i), *found]
    return None


def _label_in(layout: QtWidgets.QLayout, index: int) -> str:
    """The label for item `index` of `layout`: its form row's, the one left of it in a grid, or the last before it
    in a row."""
    if isinstance(layout, QtWidgets.QFormLayout):
        position: Any = layout.getItemPosition(index)  # (row, role)
        row, role = position
        if role == QtWidgets.QFormLayout.ItemRole.FieldRole:
            label = layout.itemAt(row, QtWidgets.QFormLayout.ItemRole.LabelRole)
            return _text(label.widget()) if label is not None else ""
        return ""
    if isinstance(layout, QtWidgets.QGridLayout):
        cell: Any = layout.getItemPosition(index)  # (row, column, row span, column span)
        row, column, _, _ = cell
        for c in range(column - 1, -1, -1):
            item = layout.itemAtPosition(row, c)
            if item is not None and (text := _text(item.widget())):
                return text
        return ""
    if isinstance(layout, QtWidgets.QBoxLayout) and layout.direction() in (
        QtWidgets.QBoxLayout.Direction.LeftToRight,
        QtWidgets.QBoxLayout.Direction.RightToLeft,
    ):
        before = ""
        for i in range(index):
            item = layout.itemAt(i)
            if item is not None and (text := _text(item.widget())):
                before = text
        return before
    return ""


def _label(widget: QtWidgets.QWidget) -> str:
    """The label beside `widget`: in the layouts holding it, innermost first, then those holding the row it's in,
    up to the panel (a group box, a tab, a scroll area). A row's label inside a form row gets the form's in front
    ("Pulse, wide at")."""
    found: list[str] = []
    target: QtWidgets.QWidget = widget
    while True:
        parent = target.parentWidget()
        layout = parent.layout() if parent is not None else None
        for holder, index in reversed((_path(layout, target) if layout is not None else None) or []):
            if text := _label_in(holder, index):
                found.insert(0, text)
                if isinstance(holder, QtWidgets.QFormLayout | QtWidgets.QGridLayout):
                    return ", ".join(dict.fromkeys(found))
        if parent is None or isinstance(
            parent, QtWidgets.QGroupBox | QtWidgets.QTabWidget | QtWidgets.QAbstractScrollArea
        ):
            return ", ".join(dict.fromkeys(found))
        target = parent


def name_controls(root: QtWidgets.QWidget) -> None:
    """Name each unnamed control under `root` after its label (see the module's docstring)."""
    for widget in controls(root):
        if labelled(widget):
            continue
        if text := _label(widget):
            widget.setAccessibleName(text)
