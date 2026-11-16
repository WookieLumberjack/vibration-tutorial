"""The Fatigue page's model (core/fatigue.py): rainflow, the moments and the spectral damage estimates."""

import math

import numpy as np
import pytest

from vib_tutorial.core.fatigue import (
    Curve,
    Stress,
    dirlik_damage,
    dirlik_range_pdf,
    miner,
    narrow_band_damage,
    rainflow,
    rates,
    rayleigh_range_pdf,
    reversals,
    synthesize,
)


def test_rainflow_on_the_astm_example():
    # ASTM E1049-85 fig. 6: −2, 1, −3, 5, −1, 3, −4, 4, −2
    ranges, counts = rainflow(np.array([-2, 1, -3, 5, -1, 3, -4, 4, -2], dtype=float))
    got = sorted(zip(ranges, counts))
    want = sorted([(3, 0.5), (4, 0.5), (4, 1.0), (8, 0.5), (9, 0.5), (8, 0.5), (6, 0.5)])
    assert got == want


def test_a_sine_is_one_cycle_a_period():
    t = np.linspace(0, 10, 10001)
    ranges, counts = rainflow(3 * np.sin(2 * np.pi * t))
    assert counts[ranges > 5.99].sum() == pytest.approx(10, abs=0.5) and np.max(ranges) == pytest.approx(6, abs=1e-3)
    c = Curve()
    assert miner(ranges, counts, c) * c.cycles(np.array([6.0]))[0] == pytest.approx(10, abs=0.5)


def test_moments_and_record_agree():
    s = Stress(sigma2=10.0)
    r = rates(s)
    assert r.sigma == pytest.approx(math.sqrt(20**2 + 10**2), rel=1e-3)
    _t, x = synthesize(s, 600.0, 3)
    assert np.std(x) == pytest.approx(r.sigma, rel=0.01)
    ups = np.count_nonzero((x[:-1] < 0) & (x[1:] >= 0)) / 600.0
    assert ups == pytest.approx(r.nu0, rel=0.03)
    d = np.diff(x)
    assert np.count_nonzero((d[:-1] > 0) & (d[1:] <= 0)) / 600.0 == pytest.approx(r.nup, rel=0.03)


@pytest.mark.parametrize("stress", [Stress(), Stress(sigma2=10.0), Stress(zeta1=0.3)])
def test_the_range_densities_integrate_to_the_damage(stress):
    c = Curve(b=4.0)
    S = np.linspace(0, 20 * rates(stress).sigma, 200001)
    for pdf in (rayleigh_range_pdf, dirlik_range_pdf):
        assert np.trapezoid(pdf(stress, S), S) == pytest.approx(1.0, abs=1e-6)
    r = rates(stress)
    assert dirlik_damage(stress, c) == pytest.approx(
        r.nup * np.trapezoid(S**c.b * dirlik_range_pdf(stress, S), S) / c.K, rel=1e-6
    )
    assert narrow_band_damage(stress, c) == pytest.approx(
        r.nu0 * np.trapezoid(S**c.b * rayleigh_range_pdf(stress, S), S) / c.K, rel=1e-6
    )


def test_rainflow_steps_count_what_rainflow_counts():
    """A step for each reversal read and each cycle let go: their counts are `rainflow`'s, the stack after each
    step what's still open."""
    from vib_tutorial.core.fatigue import rainflow_steps

    rng = np.random.default_rng(4)
    for _ in range(100):
        x = rng.normal(size=int(rng.integers(3, 60))).cumsum()
        p = reversals(x)
        steps = rainflow_steps(p)
        ranges, counts = rainflow(x)
        got = sorted((s.range, s.count) for s in steps if s.count)
        assert np.allclose(got, sorted(zip(ranges, counts)))
        assert sum(s.read >= 0 for s in steps) == p.size and steps[-1].stack == (p.size - 1,)


def test_the_rainflow_tab_steps_through_the_count(main_window):
    from PySide6 import QtWidgets

    from vib_tutorial.gui.fatigue import RF_SHOW, TAB_RAINFLOW

    w = main_window
    page = w.fatigue_page
    w.pages.setCurrentWidget(page)
    page.tabs.setCurrentIndex(TAB_RAINFLOW)
    QtWidgets.QApplication.processEvents()
    page.load_trial(1)  # two modes: ripples on the slow swing
    page.tabs.setCurrentIndex(TAB_RAINFLOW)
    assert page.rf_k == 0 and page.rf_p.size == RF_SHOW and not page.rf_back.isEnabled()
    assert "Step 0 of" in page.rf_tally.text()
    n = len(page.rf_steps)
    for k in range(1, n + 1):
        page.rf_step.click()
        assert page.rf_k == k
    assert not page.rf_step.isEnabled() and "Done." in page.rf_note.text()
    ranges, counts = rainflow(page.rf_p)
    full, half = int(np.sum(counts == 1.0)), int(np.sum(counts == 0.5))
    assert f"Counted: {full} full and {half} half cycles" in page.rf_tally.text()
    xs = page.rf_full.getData()[0]
    assert xs is not None and np.sum(np.isfinite(xs)) == 2 * full  # a line per full cycle
    page.rf_back.click()
    assert page.rf_k == n - 1

    page.rf_reset.click()
    assert page.rf_k == 0
    page.rf_play.click()  # a step at once, then one a tick; hidden, it stops
    assert page.rf_k == 1
    assert page.rf_timer.isActive() and page.rf_play.text() == "Pause"
    w.pages.setCurrentWidget(w.sim_page)
    assert not page.rf_timer.isActive() and page.rf_play.text() == "Play"
