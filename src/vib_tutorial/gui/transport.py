"""The window's run bar: one place for the motion's controls on every page (#316 B13).

A page's motion has its own controls (a `lesson.RunButton`, a Reset, a speed, a clock), laid out where the page
put them, so Start sat somewhere different on each page. The `TransportBar` along the foot of the window drives
whichever is runnable on the page and tab showing: its Start/Pause, Reset and speed press the page's own and its
clock and labels follow theirs (read back several times a second while it shows: a page changes its buttons in
ways no signal reports, `RunButton.rest` among them). On a page with nothing to run it says so, greyed out.

A page says what's runnable with `transport()`, a `Transport` or None (it may depend on the tab showing: the
Torsional page's Start runs its Start-up tab); a page with one `playback.Playback` and no `transport()` gets the
Playback's. The window asks again on a new page or tab.

Working on it: `tests/test_transport.py` drives every page's motion from the bar.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6 import QtCore, QtGui, QtWidgets
from shiboken6 import isValid

from . import theming

POLL_MS = 150  # how often the bar reads the page's controls back


# Controls that do something with Space themselves: a button presses, a check box toggles, a combo box opens, a
# list picks; a field types it (those already take it before a shortcut does: forwarding them is harmless).
SPACE_TAKERS = (
    QtWidgets.QAbstractButton,
    QtWidgets.QComboBox,
    QtWidgets.QAbstractItemView,
    QtWidgets.QAbstractSpinBox,
    QtWidgets.QLineEdit,
)


_clicked: list[QtWidgets.QWidget | None] = [None]  # the control the mouse last focused (a weak hold isn't needed:
# it's replaced at the next focus change, and only compared, never called)
_watching: list[bool] = [False]


def _on_focus_changed(_old: QtWidgets.QWidget | None, now: QtWidgets.QWidget | None) -> None:
    clicked = QtGui.QGuiApplication.mouseButtons() != QtCore.Qt.MouseButton.NoButton
    _clicked[0] = now if clicked else None


def watch_focus() -> None:
    """Note which control a click focuses (`space_to_focus` leaves those be); once per application."""
    if not _watching[0]:
        theming.application().focusChanged.connect(_on_focus_changed)
        _watching[0] = True


def space_to_focus() -> bool:
    """Give Space to the control the keyboard moved to (Tab) if it uses Space itself (`SPACE_TAKERS`), and say
    so. A window-wide Space shortcut calls it first: otherwise the shortcut always wins, and a keyboard user can't
    press a focused button or tick a focused box. A control focused by a click is left out: on Linux and Windows a
    clicked button keeps the focus, and Space after clicking Reset should still start the run."""
    w = QtWidgets.QApplication.focusWidget()
    if not isinstance(w, SPACE_TAKERS) or not w.isEnabled() or w is _clicked[0]:
        return False
    # Its own handlers, not sendEvent: an event sent goes through the shortcut map again, back to this.
    space, none = QtCore.Qt.Key.Key_Space, QtCore.Qt.KeyboardModifier.NoModifier
    w.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, space, none, " "))
    w.keyReleaseEvent(QtGui.QKeyEvent(QtCore.QEvent.Type.KeyRelease, space, none, " "))
    return True


@dataclass
class Transport:
    """A page's motion controls: `run` checkable (checked while it runs), the rest optional."""

    run: QtWidgets.QAbstractButton
    reset: QtWidgets.QAbstractButton | None = None
    speed: QtWidgets.QComboBox | None = None
    clock: QtWidgets.QWidget | None = None  # anything with text(): a QLabel, a LiveLabel
    label: str = ""  # what it runs, where a page has more than one motion ("Start-up")
    space: bool = True  # Space runs it (not the Simulation page's, whose Space is its force)

    def widgets(self) -> list[QtWidgets.QWidget]:
        return [w for w in (self.run, self.reset, self.speed, self.clock) if w is not None]


def hide_own(t: Transport) -> None:
    """Hide the page's own controls that the bar stands in for, with what belongs to them alone: a form's row, a
    `layouts.group` holder with its label, a label naming one of them."""
    targets = t.widgets()
    for w in targets:
        w.hide()
        parent = w.parentWidget()
        if parent is None:
            continue
        layout = parent.layout()
        if isinstance(layout, QtWidgets.QFormLayout):
            for row in range(layout.rowCount()):
                if _row_only(layout, row, targets):
                    layout.setRowVisible(row, False)
        elif parent.property("transport_holder") is None and _holder_only(parent, targets):
            parent.hide()
        top = w.window()
        for label in top.findChildren(QtWidgets.QLabel):
            if label.buddy() is w:
                label.hide()


def _row_only(layout: QtWidgets.QFormLayout, row: int, targets: list[QtWidgets.QWidget]) -> bool:
    """Whether a form's row holds nothing but `targets` (its label aside)."""
    field = layout.itemAt(row, QtWidgets.QFormLayout.ItemRole.FieldRole) or layout.itemAt(
        row, QtWidgets.QFormLayout.ItemRole.SpanningRole
    )
    if field is None:
        return False
    found = _widgets_in(field)
    return bool(found) and all(w in targets for w in found)


def _widgets_in(item: QtWidgets.QLayoutItem) -> list[QtWidgets.QWidget]:
    if (w := item.widget()) is not None:
        return [w]
    layout = item.layout()
    if layout is None:
        return []
    out: list[QtWidgets.QWidget] = []
    for i in range(layout.count()):
        sub = layout.itemAt(i)
        if sub is not None:
            out += _widgets_in(sub)
    return out


def _holder_only(holder: QtWidgets.QWidget, targets: list[QtWidgets.QWidget]) -> bool:
    """Whether `holder` is a small row of `targets` and their labels (`layouts.group`), nothing else."""
    layout = holder.layout()
    if not isinstance(layout, QtWidgets.QHBoxLayout):
        return False
    found = [w for w in _widgets_in(layout)]
    others = [w for w in found if w not in targets]
    return bool(found) and len(found) <= 4 and all(isinstance(w, QtWidgets.QLabel) for w in others)


def transport_of(page: QtWidgets.QWidget | None) -> Transport | None:
    """What's runnable on `page` now: its `transport()`, else its one Playback's, else None."""
    if page is None:
        return None
    ask = getattr(page, "transport", None)
    if callable(ask):
        found = ask()
        return found if isinstance(found, Transport) else None
    from .playback import Playback

    playbacks = page.findChildren(Playback)
    if len(playbacks) == 1:
        p = playbacks[0]
        return Transport(p.run, p.reset, p.speed, p.clock)
    return None


def _text(widget: QtWidgets.QWidget | None) -> str:
    text = getattr(widget, "text", None)
    return str(text()) if callable(text) else ""


class TransportBar(QtWidgets.QFrame):
    """The bar (see the module's docstring)."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setAccessibleName("Run bar")
        self.source: Transport | None = None
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(6, 1, 6, 1)  # as short as a button: every pixel here comes off the pages (720 px tall)
        row.setSpacing(4)
        self.run = QtWidgets.QPushButton("Start")
        self.run.setCheckable(True)
        self.run.setToolTip("<p>Start, pause or resume the page's motion (Space).</p>")
        self.run.setAccessibleName("Start or pause")
        self.run.clicked.connect(self._on_run)
        self.reset = QtWidgets.QPushButton("Reset")
        self.reset.setToolTip("<p>Back to t = 0, at rest, waiting for Start.</p>")
        self.reset.clicked.connect(self._on_reset)
        self.speed_label = QtWidgets.QLabel("Speed:")
        self.speed = QtWidgets.QComboBox()
        self.speed.setAccessibleName("Speed")
        self.speed.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.speed.activated.connect(self._on_speed)
        self.speed_label.setBuddy(self.speed)
        self.clock = QtWidgets.QLabel()
        self.clock.setAccessibleName("Time")
        self.what = QtWidgets.QLabel()
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        for w in (self.run, self.reset):
            row.addWidget(w)
        row.addSpacing(12)
        row.addWidget(self.speed_label)
        row.addWidget(self.speed)
        row.addSpacing(12)
        row.addWidget(self.clock)
        row.addStretch(1)
        row.addWidget(self.what)
        self.space = QtGui.QShortcut(QtGui.QKeySequence("Space"), self)  # where the page's own Start is hidden
        self.space.setContext(QtCore.Qt.ShortcutContext.WindowShortcut)
        self.space.activated.connect(self._on_space)
        watch_focus()
        self.space.setEnabled(False)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(POLL_MS)
        self.timer.timeout.connect(self.sync)
        self.timer.start()
        self.follow(None)

    def follow(self, transport: Transport | None) -> None:
        """Drive `transport` (None: nothing to run here)."""
        self.source = transport
        self._speeds: list[str] = []
        if transport is not None:
            hide_own(transport)
        self.space.setEnabled(transport is not None and transport.space)
        self.sync()

    def sync(self) -> None:
        """Read the page's controls back into the bar."""
        if not self.isVisible() and self.source is not None:
            return
        t = self.source
        alive = t is not None and isValid(t.run)
        self.run.setEnabled(alive and t.run.isEnabled())  # type: ignore[union-attr]
        if not alive:
            self.run.setChecked(False)
            self.run.setText("Start")
            for w in (self.reset, self.speed, self.speed_label):
                w.setEnabled(False)
            self.clock.setText("")
            self.what.setText("Nothing on this page runs in time")
            return
        assert t is not None, "alive"
        if self.run.isChecked() != t.run.isChecked():
            self.run.setChecked(t.run.isChecked())
        if self.run.text() != t.run.text():
            self.run.setText(t.run.text())
        self.reset.setEnabled(t.reset is not None and t.reset.isEnabled())
        has_speed = t.speed is not None
        self.speed.setEnabled(has_speed)
        self.speed_label.setEnabled(has_speed)
        if t.speed is not None:
            items = [t.speed.itemText(i) for i in range(t.speed.count())]
            if items != self._speeds:
                self._speeds = items
                self.speed.clear()
                self.speed.addItems(items)
            if self.speed.currentIndex() != t.speed.currentIndex():
                self.speed.setCurrentIndex(t.speed.currentIndex())
        elif self._speeds:
            self._speeds = []
            self.speed.clear()
        clock = _text(t.clock)
        if self.clock.text() != clock:
            self.clock.setText(clock)
        if self.what.text() != t.label:
            self.what.setText(t.label)

    def _on_run(self) -> None:
        t = self.source
        if t is not None and isValid(t.run) and t.run.isEnabled():
            t.run.click()
        self.sync()

    def _on_space(self) -> None:
        if not space_to_focus() and self.run.isEnabled():
            self.run.click()

    def _on_reset(self) -> None:
        t = self.source
        if t is not None and t.reset is not None and isValid(t.reset):
            t.reset.click()
        self.sync()

    def _on_speed(self, index: int) -> None:
        t = self.source
        if t is not None and t.speed is not None and isValid(t.speed):
            t.speed.setCurrentIndex(index)
            t.speed.activated.emit(index)  # as a reader's pick: the pages listen to either
        self.sync()
