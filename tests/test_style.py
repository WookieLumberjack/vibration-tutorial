"""The themes' colors: legible on their backgrounds, and the masses told apart by color-blind eyes too."""

import ast
import itertools
import pathlib
import re

import numpy as np
import pytest

from vib_tutorial.gui import style

# Machado, Oliveira & Fernandes (2009), severity 1: linear RGB -> what a dichromat sees.
DEUTERANOPIA = np.array(
    [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]]
)
PROTANOPIA = np.array(
    [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]]
)
SIGHT = {"normal": np.eye(3), "deuteranopia": DEUTERANOPIA, "protanopia": PROTANOPIA}
MIN_DELTA_E = 10.0  # CIE76; the tab10 green and red of masses 3 and 4 were 7 apart under deuteranopia
MASSES = 4  # the default chain's: the ones a student meets first


def _linear(color: str) -> np.ndarray:
    c = np.array(style.rgb(color)) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lab(linear: np.ndarray) -> np.ndarray:
    xyz = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]) @ np.clip(
        linear, 0, 1
    )
    t = xyz / np.array([0.9505, 1.0, 1.089])
    f = np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


@pytest.mark.parametrize("theme", style.THEMES.values(), ids=lambda t: t.name)
@pytest.mark.parametrize("sight", SIGHT)
def test_the_first_masses_stay_apart(theme, sight):
    colours = theme.mass[:MASSES]
    seen = [_lab(SIGHT[sight] @ _linear(c)) for c in colours]
    for (i, a), (j, b) in itertools.combinations(enumerate(seen), 2):
        assert np.linalg.norm(a - b) >= MIN_DELTA_E, (theme.name, sight, f"m{i + 1}", f"m{j + 1}")


@pytest.mark.parametrize("theme", style.THEMES.values(), ids=lambda t: t.name)
def test_curves_read_on_the_background(theme):
    for c in (*theme.mass[:MASSES], *theme.mode[:MASSES]):
        assert style.contrast(c, theme.background) >= 3.0, (theme.name, c)


@pytest.mark.parametrize("theme", style.THEMES.values(), ids=lambda t: t.name)
def test_mode_1_is_not_the_reference_curve(theme):
    """Mode 1's term and the full solution (strong) are drawn together, both 1 px. Light's old slate
    mode 1 (#264653) was 31 from its black and read as black; Nord's mauve on white is 38."""
    assert np.linalg.norm(_lab(_linear(theme.mode[0])) - _lab(_linear(theme.strong))) >= 35.0


def test_the_projector_theme_scales_every_pen_and_marker(main_window):
    """Projector: a pen twice as wide (a copy of a pen not again, the axes not at all) and markers 40 % larger;
    Light leaves them as they are (`plot_patches`, installed by `make_app`)."""
    import pyqtgraph as pg

    from vib_tutorial.gui import plot_patches

    plot_patches.install()  # as make_app does
    plot = main_window.findChildren(pg.PlotWidget)[0]  # items on a plot of the window, freed with it
    before = style.current()
    try:
        style.set_theme(style.THEMES["Projector"])
        pen = pg.mkPen("#000000", width=1.5)
        assert pen.widthF() == pytest.approx(3.0) and pg.mkPen(pen).widthF() == pytest.approx(3.0)
        assert pg.mkPen("#000000").widthF() == pytest.approx(2.0)  # a cosmetic pen counts as 1 px
        assert plot.getAxis("left").pen().widthF() <= 1.0
        item = pg.PlotDataItem([0, 1], [0, 1], symbol="o", symbolSize=10)
        plot.addItem(item)
        assert item.scatter.opts["size"] == pytest.approx(14.0)
        style.set_theme(style.THEMES["Light"])
        assert pg.mkPen("#000000", width=1.5).widthF() == pytest.approx(1.5)
        item.setData([0, 1], [1, 0])
        assert item.scatter.opts["size"] == pytest.approx(10.0)
        plot.removeItem(item)
    finally:
        style.set_theme(before)


def test_a_wide_solid_curve_is_drawn_in_segments(main_window):
    """A solid, opaque pen wider than 1 px from `mkPen` (flat caps) is drawn as segments, fast even antialiased;
    a dashed, translucent, thin or square-capped one strokes its polyline (`plot_patches`' *Wide lines*)."""
    import pyqtgraph as pg
    from PySide6 import QtCore, QtGui

    curve = pg.PlotCurveItem([0, 1, 2], [0, 1, 0], antialias=True)
    segmented = curve._shouldUseDrawLineSegments
    assert segmented(pg.mkPen("#000000", width=2))
    assert not segmented(pg.mkPen("#000000", width=2, style=QtCore.Qt.PenStyle.DashLine))
    assert not segmented(pg.mkPen("#00000080", width=2))
    assert not segmented(pg.mkPen("#000000", width=1))
    assert not segmented(QtGui.QPen(QtGui.QColor("#000000"), 2))  # square caps: they'd overlap at every joint
    assert pg.mkPen(pg.mkPen("#000000", width=2)).capStyle() == QtCore.Qt.PenCapStyle.FlatCap  # copies keep it


@pytest.mark.gui
@pytest.mark.parametrize("theme", style.THEMES.values(), ids=lambda t: t.name)
def test_the_palettes_secondary_text_and_links_read(theme):
    """The palette the app gets (after another theme: Light's is the platform's but for these), its muted text
    (`theming.mute`'s) and links at least 4.5:1 on the window and on text fields, and on the window macOS gives
    the Light theme (#ececec), where the platform's gray was 3.8:1 (#409 U-2)."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtGui, QtWidgets

    from vib_tutorial.gui import theming

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app = theming.application()
    role = QtGui.QPalette.ColorRole
    try:
        theming.apply("Dark")
        theming.apply(theme.name)
        pal = app.palette()
        grounds = {pal.color(r).name() for r in (role.Window, role.Base)} | (
            {"#ececec"} if theme.window is None else set()
        )
        for r in (role.PlaceholderText, role.Link, role.Text, role.WindowText):
            for ground in grounds:
                assert style.contrast(pal.color(r).name(), ground) >= 4.5, (theme.name, r.name, ground)
        if theme.window is None:  # the rest the platform's own
            assert pal.color(role.Window) == QtGui.QPalette().color(role.Window) != QtGui.QColor(style.DARK.window)
    finally:
        theming.apply("Light")


UNDERSCORED = re.compile(r"(?<![\w./{\[#'\"=])([A-Za-zωζθφλσμη]{1,2}_[A-Za-z0-9]{1,5})(?![\w(\[=])")
NOT_SHOWN = {"notation.py", "frame_stats.py"}  # the symbols' keys; the timing overlay's log


def test_no_underscored_symbols_in_what_the_reader_sees():
    """A symbol in a label, a readout, a title or a tooltip has a proper subscript (k<sub>f</sub>, fₙ), not
    an underscore (k_f) beside the real ones; plain text (a combo's items, a table's headers) says it in words
    (#409 U-20). Docstrings, and strings with no space or colon (keys, names), are left out."""
    root = pathlib.Path(style.__file__).parent
    found = []
    for path in sorted(root.glob("*.py")):
        if path.name in NOT_SHOWN:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docs = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
                if " " in node.value or ":" in node.value:
                    found += [f"{path.name}:{node.lineno}: {m.group(1)}" for m in UNDERSCORED.finditer(node.value)]
    assert not found, found
