"""The Jeffcott rotor's angles counted one way, and shown so: the polar grid, the Balancing tab, the Bode tab (#409 N-1).

Every angle on the page counts counter-clockwise, the way the shaft turns, from the keyphasor mark at +x: a
reading's phase as a lead (arg V), a weight's and the heavy spot's from the mark.
"""

import math

import numpy as np
import pytest

from vib_tutorial.core.rotor import RotorSystem
from vib_tutorial.core.rotor_balance import weight
from vib_tutorial.core.rotor_simulator import RPM
from vib_tutorial.core.rotor_sweep import lag_degrees, phase_degrees, steady_vectors

pytestmark = pytest.mark.gui

HEAVY = 60.0  # degrees: the notes' example in *Which way phase is counted*
DISC = 1


def _angle(x: float, y: float) -> float:
    return math.degrees(math.atan2(y, x)) % 360.0


def test_phase_is_minus_the_lag_and_unwraps_falling():
    v = np.exp(1j * np.radians([-10.0, -170.0, -190.0, -350.0]))
    assert phase_degrees(v) == pytest.approx(-lag_degrees(v))
    assert phase_degrees(v) == pytest.approx([-10.0, -170.0, -190.0, -350.0])  # a growing lag keeps falling
    assert phase_degrees(np.exp(1j * np.radians(10.0)), near=np.array(-355.0)) == pytest.approx(-350.0)


def test_the_grid_counts_counter_clockwise_from_the_mark(main_window):
    from vib_tutorial.gui.polar_grid import PolarGrid, nice_ring

    for r in (3e-6, 26.6e-6, 1e-4, 4.4e-4, 2e-3):
        step, count = nice_ring(r)
        assert step * (count - 1) < r <= step * count * (1 + 1e-9) and 2 <= count <= 4
    p = main_window.rotor_page
    main_window.pages.setCurrentWidget(p)
    grid: PolarGrid = p.balancing.grid
    grid.fit(30e-6)
    R = grid.radius
    assert R == pytest.approx(30e-6)
    for deg, label in grid.angle_labels:
        assert label.toPlainText() == f"{deg}°"
        assert _angle(label.pos().x(), label.pos().y()) == pytest.approx(deg, abs=1e-6)  # 90° is up: counter-clockwise
    x, y = grid.ring_point(math.radians(90.0))
    assert (x, y) == pytest.approx((0.0, R), abs=1e-12)
    assert grid.texts["phase"].toPlainText() == "+ phase" and "keyphasor" in grid.texts["mark"].toPlainText()
    sx, sy = grid.key.getData()  # the arrows: the positive phase's runs counter-clockwise from 0°
    first = np.flatnonzero(np.isnan(sx))
    spin, phase = slice(0, first[0]), slice(first[1] + 1, None)
    for part in (spin, phase):
        a = np.unwrap(np.arctan2(sy[part], sx[part]))
        a = a[np.isfinite(a)]
        assert a[len(a) // 2] > a[0], "the arcs run counter-clockwise"


def test_the_balancing_tab_draws_the_correction_opposite_the_reading(main_window):
    """Below the critical, read at the x probe beside the keyphasor: V₀ near the heavy spot, W opposite."""
    p = main_window.rotor_page
    main_window.pages.setCurrentWidget(p)
    p.loop.enabled = False
    system = RotorSystem(unbalance_angle=math.radians(HEAVY))
    p.set_system(system)
    b = p.balancing
    p.tabs.show_page(b)
    b.station.setCurrentIndex(DISC)
    b.subtract.setChecked(False)
    b.trial_angle.setValue(30.0)
    omega = b.speed.value() / RPM
    trial = weight(b.trial_mass.value() / 1e3, b.trial_radius.value() / 1e3, math.radians(30.0))
    b.found_system = system
    b.trial = trial
    b.vectors["found"] = steady_vectors(system, np.array([omega]))[0]
    b.vectors["trial"] = steady_vectors(system.with_(weights=(trial,)), np.array([omega]))[0]
    b._draw()
    v0 = b.reading("found")
    assert v0 is not None
    assert math.degrees(np.angle(v0)) == pytest.approx(51.0, abs=0.5)  # 9° of lag behind the heavy spot
    text = b.results.text()
    assert "26.6 µm ∠ 51°" in text and "1 g at 20 mm, 240°" in text and "lead" in text
    angles = {name: [math.degrees(a) % 360.0 for a in v] for name, v in b.weight_angles().items()}
    assert angles["trial"] == pytest.approx([30.0]) and angles["heavy"] == pytest.approx([HEAVY])
    assert angles["correction"] == pytest.approx([HEAVY + 180.0], abs=0.01)
    # Drawn where they are: V₀'s tip at its phase, the correction's square on the ring opposite.
    tip = b.tips["found", 0].getData()
    assert _angle(tip[0][0], tip[1][0]) == pytest.approx(math.degrees(np.angle(v0)) % 360.0)
    wx, wy = b.weight_marks["correction"].getData()
    assert _angle(wx[0], wy[0]) == pytest.approx(240.0, abs=0.01)
    assert math.hypot(wx[0], wy[0]) == pytest.approx(b.grid.radius)
    assert "counter-clockwise" in b.convention.text() and "lag = 360° − lead" in b.convention.text()
    assert "counter-clockwise" in b.trial_angle.toolTip() and "lead" in b.vector_plot.toolTip()
    # The Bode under it: the phase starts at the heavy spot's angle and falls through −90° from it at the critical.
    _rpm, phase = b.phase_found[0].getData()
    assert phase[0] == pytest.approx(HEAVY, abs=1.0) and phase[-1] == pytest.approx(HEAVY - 180.0, abs=10.0)
    assert "lead" in b.phase_plot.getAxis("left").labelText


def test_the_bode_tab_and_the_panel_say_how_they_count(main_window):
    p = main_window.rotor_page
    main_window.pages.setCurrentWidget(p)
    p.loop.enabled = False
    p.set_system(RotorSystem())
    p._refresh_analysis()
    p.tabs.show_page(p.bode.parentWidget())
    p._draw_tracking(full=True)
    assert "lead +, lag −" in p.bode.phase.getAxis("left").labelText
    _rpm, phase = p.bode.steady_phase.getData()
    assert phase[0] == pytest.approx(0.0, abs=2.0) and phase[-1] < -170.0  # a lag shows as a negative phase
    assert "lead" in p.polar.toolTip() and p.polar.grid.radius >= np.abs(p._steady_probe()[0]).max()
    for box in (p.editor.unbalance_angle, p.editor.bow_angle, p.editor.crack_angle):
        assert "counter-clockwise" in box.toolTip()
