"""The weak slots (`gui/weak.py`), and that no GUI module connects a slot that holds its window."""

import ast
import gc
import os
import pathlib
import weakref

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from vib_tutorial.gui import weak  # noqa: E402

GUI = pathlib.Path(__file__).parent.parent / "src" / "vib_tutorial" / "gui"


class Owner:
    def __init__(self):
        self.calls = []

    def note(self, *args):
        self.calls.append(args)

    def plain(self, keep=True):
        self.calls.append(keep)


def test_slot_passes_what_its_function_requires_and_holds_its_object_weakly():
    o = Owner()
    one = weak.slot(o, lambda o, v: o.note(v))
    none = weak.slot(o, lambda o, _=False, m="mode": o.note(m))  # a clicked(bool) leaves the default be
    every = weak.slot(o, lambda o, *args: o.note(*args))
    one(1, 2)
    none(True)
    every(3, 4)
    assert o.calls == [(1,), ("mode",), (3, 4)]
    ref = weakref.ref(o)
    del o
    gc.collect()
    assert ref() is None
    assert one(5) is None  # the object's gone: nothing called


def test_call_drops_the_signal_s_arguments():
    o = Owner()
    call = weak.call(o.plain)
    call(False)  # a toggled(bool): not passed on as `keep`
    assert o.calls == [True]
    ref = weakref.ref(o)
    del o, call
    gc.collect()
    assert ref() is None


def test_a_page_s_window_is_freed_once_dropped():
    from vib_tutorial.gui import close_graphics
    from vib_tutorial.gui.fatigue import FatiguePage

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    page = FatiguePage()
    page.show()
    app.processEvents()
    close_graphics(page)
    ref = weakref.ref(page)
    del page
    gc.collect()
    assert ref() is None


# ------------------------------------------------------------------ the GUI's own slots
HOLDERS = ("connect", "QShortcut", "singleShot")


def _uses_self(node: ast.AST) -> bool:
    """`node` mentions a `self` it doesn't take as a parameter of its own."""
    if isinstance(node, (ast.Lambda, ast.FunctionDef)):
        args = node.args
        own = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
        if "self" in own:
            return False
    return any(isinstance(n, ast.Name) and n.id == "self" for n in ast.walk(node))


def _held(path: pathlib.Path) -> list[str]:
    """Slots in `path` that hold `self` strongly: a lambda, `partial` or local function over it, handed to
    a signal or a shortcut (a single shot with a context object is let go when that goes)."""
    source = path.read_text()
    tree = ast.parse(source)
    local = {}
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef):
            for inner in fn.body:
                if isinstance(inner, ast.FunctionDef) and _uses_self(inner):
                    local[inner.name] = inner
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        if name not in HOLDERS or (name == "singleShot" and len(node.args) == 3):
            continue
        for arg in node.args:
            partial = isinstance(arg, ast.Call) and getattr(arg.func, "attr", getattr(arg.func, "id", "")) == "partial"
            if ((isinstance(arg, ast.Lambda) or partial) and _uses_self(arg)) or getattr(arg, "id", None) in local:
                found.append(f"{path.name}:{node.lineno}: {ast.get_source_segment(source, arg)}")
    return found


def test_no_gui_slot_holds_its_window():
    """PySide holds a connected lambda as long as its sender, which the window owns, so one over `self` keeps
    the window alive: connect a bound method, or `weak.slot(self, lambda self, ...: ...)`."""
    found = [line for path in sorted(GUI.glob("*.py")) for line in _held(path)]
    assert not found, "slots holding their window (see gui/weak.py):\n" + "\n".join(found)


def test_the_check_finds_them(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text(
        "class P:\n"
        "    def __init__(self):\n"
        "        self.b.clicked.connect(lambda: self.go())\n"
        "        self.b.clicked.connect(partial(self.go, 1))\n"
        "        QShortcut(key, self, lambda i=1: self.go(i))\n"
        "        QTimer.singleShot(0, lambda: self.go())\n"
        "        def follow():\n"
        "            self.go()\n"
        "        self.b.clicked.connect(follow)\n"
        "        self.b.clicked.connect(weak.slot(self, lambda self: self.go()))\n"  # fine
        "        QTimer.singleShot(0, self.b, lambda: self.go())\n"  # fine: let go with self.b
        "        self.b.clicked.connect(lambda on: settings().setValue('k', on))\n"  # fine: no self
    )
    assert [line.split(":")[1] for line in _held(bad)] == ["3", "4", "5", "6", "9"]
