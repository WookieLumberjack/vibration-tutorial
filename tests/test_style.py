"""The themes' colours: legible on their backgrounds, and the masses told apart by colour-blind eyes too."""

import itertools

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
