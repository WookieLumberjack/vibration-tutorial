"""Names for screen readers: each control named after the label it sits beside (#218 U5).

Qt names a control after its *buddy* label, which `QFormLayout.addRow(label, field)` sets when the field is the
control itself. A control that isn't a label's buddy has no name, and a screen reader reads out only "spin box"
(or the title of the group box it's in, which names the panel, not the control). So the helpers that put a
label beside a control name it as they build the row:

- `layouts.group("Band", box, "to", box2)`: each control after the label (or check box) before it;
- `layouts.form_row` and `lesson.Form.add`: a control as the field becomes the label's buddy (`addRow` makes it
  one only for a label it makes from a string), and a row of widgets has its controls named after the label, in
  front of their own ("Pulse, wide at").

A control with no label beside it (a page's preset, a playback speed) gets `setAccessibleName` where it's made.
A plot is named when its page is built (`name_plots`, from `MainWindow._page`) after its title or its axes
("Amplitude against Frequency"), unless it has a name already; Qt reads its tooltip as its description.
This names nothing by walking the layouts after the fact: reading a form's item positions while a page is built
changed how Qt laid it out.

Working on it: `tests/test_accessible.py` checks every page's controls and plots have a name of their own.
"""

from __future__ import annotations

import re

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


def plain(text: str) -> str:
    """A label's text as a name: no tags, no closing colon."""
    return TAG.sub("", text).replace("&nbsp;", " ").strip().rstrip(":").strip()


def labelled(widget: QtWidgets.QWidget) -> bool:
    """Whether `widget` has a name of its own: one set, a button's text, or a label whose buddy it is."""
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
    """The controls a reader can use under `root`: not a spin box's or combo's own parts, a table's corner button,
    scroll bars, or one hidden on purpose."""
    found: list[QtWidgets.QWidget] = []
    for widget in root.findChildren(QtWidgets.QWidget):
        if not isinstance(widget, CONTROLS) or isinstance(widget, QtWidgets.QScrollBar):
            continue
        if isinstance(widget.parentWidget(), QtWidgets.QAbstractSpinBox | QtWidgets.QComboBox):
            continue
        if isinstance(widget.parentWidget(), QtWidgets.QAbstractItemView):
            continue
        if widget.isHidden():  # a second plane's step while balancing in one
            continue
        found.append(widget)
    return found


def name_row(items: list[QtWidgets.QWidget]) -> None:
    """A row's controls, each named after the label or check box before it (`layouts.group`)."""
    before = ""
    for item in items:
        if isinstance(item, QtWidgets.QLabel | QtWidgets.QCheckBox):
            before = plain(item.text())
        elif isinstance(item, CONTROLS) and before and not labelled(item):
            item.setAccessibleName(before)


def name_field(label: QtWidgets.QLabel, field: QtWidgets.QWidget) -> None:
    """A form row's names: a field that is a control becomes its label's buddy (`QFormLayout.addRow` only does that
    for a label it makes from a string); a row of widgets has its controls named after the label, in front of their
    own ("Pulse, wide at")."""
    if isinstance(field, CONTROLS):
        label.setBuddy(field)
        return
    if not (text := plain(label.text())):
        return
    for control in controls(field):
        if isinstance(control, QtWidgets.QAbstractButton) and control.text().strip():
            continue
        own = control.accessibleName()
        control.setAccessibleName(f"{text}, {own}" if own and own != text else text)


def plot_name(view: QtWidgets.QWidget) -> str:
    """A pyqtgraph view's name: its plots' titles, or their axes ("Amplitude against Frequency")."""
    import pyqtgraph as pg

    items = [view.getPlotItem()] if isinstance(view, pg.PlotWidget) else []
    if isinstance(view, pg.GraphicsLayoutWidget):
        items = [i for i in view.ci.items if isinstance(i, pg.PlotItem)]
    names = []
    for item in items:
        title = plain(item.titleLabel.text or "")
        if title:
            names.append(title)
            continue
        left, bottom = (plain(item.getAxis(side).labelText or "") for side in ("left", "bottom"))
        if left and bottom:
            names.append(f"{left} against {bottom}")
        elif left or bottom:
            names.append(left or bottom)
    return "; ".join(dict.fromkeys(names))


def name_plots(root: QtWidgets.QWidget) -> None:
    """Name each pyqtgraph view under `root` that has no name yet after its plots (`plot_name`)."""
    import pyqtgraph as pg

    for view in root.findChildren(pg.GraphicsView):
        if not view.accessibleName() and (text := plot_name(view)):
            view.setAccessibleName(text)
