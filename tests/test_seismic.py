"""Seismic response spectrum: effective masses, the rules, CQC's correlation, and the records against the spectrum."""

import numpy as np
import pytest

from vib_tutorial.core.random_vibration import G
from vib_tutorial.core.seismic import (
    Design,
    Rule,
    accelerogram,
    building,
    combine,
    correlation,
    design_sa,
    modal,
    time_history,
)


def test_effective_masses_add_up_to_the_mass():
    m = modal(building(6), Design())
    assert m.effective_mass.sum() == pytest.approx(m.total_mass)
    assert np.sum(m.base_shear) == pytest.approx(np.sum(m.effective_mass * m.sa))


def test_spectrum_corners():
    d = Design()
    S, TB, TC, TD = d.corners
    plateau = 2.5 * 0.3 * G * S
    assert design_sa(d, np.array([0.0, TB, TC, 2 * TC, 2 * TD])) == pytest.approx(
        [0.3 * G * S, plateau, plateau, plateau / 2, plateau * TC * TD / (4 * TD * TD)]
    )


def test_rules_agree_for_one_mode_and_order():
    peaks = np.array([[3.0, -4.0, 1.0]])
    rho = correlation(np.array([1.0, 3.0, 9.0]), 0.05)
    assert combine(peaks, Rule.SRSS)[0] == pytest.approx(np.sqrt(26))
    assert combine(peaks, Rule.ABS)[0] == 8
    assert combine(peaks, Rule.NRL)[0] == pytest.approx(4 + np.sqrt(10))
    assert combine(peaks, Rule.CQC, rho)[0] == pytest.approx(np.sqrt(26), rel=0.01)
    assert combine(peaks, Rule.CQC, np.ones((3, 3)))[0] == pytest.approx(0.0)  # fully correlated, signs kept


def test_correlation_is_one_on_the_diagonal_and_symmetric_for_equal_damping():
    rho = correlation(np.array([1.0, 1.1, 2.0]), 0.05)
    assert np.diag(rho) == pytest.approx(1.0)
    assert rho == pytest.approx(rho.T)


def test_the_record_matches_the_spectrum_and_one_mode_is_exact():
    d = Design()
    g = accelerogram(d, 3)
    err = g.spectrum / design_sa(d, 1 / g.f) - 1
    assert np.median(np.abs(err)) < 0.05
    one = building(1)  # one storey: its peak displacement is the record's spectral displacement at T
    h = time_history(one, d, g)
    m = modal(one, d)
    from vib_tutorial.core.shock import srs

    sd = srs(g.record, m.f, d.zeta).displacement[0]
    assert h.peak_displacement[0] == pytest.approx(sd, rel=3e-3)  # sampled peaks; the SRS refines its own


def test_the_building_sways_under_the_record_its_drifting_storey_reddened(main_window):
    """The Building tab plays the first record: the floors placed from its history, the storey drifting most the
    force colour; storey 1's box edits the chain (the Simulation page's too)."""
    from PySide6 import QtGui, QtWidgets

    from vib_tutorial.gui.seismic import TAB_BUILDING, TRIALS
    from vib_tutorial.gui.style import colors

    w = main_window
    page = w.seismic_page
    w.pages.setCurrentWidget(page)
    QtWidgets.QApplication.processEvents()
    for name, storey in (("A soft storey", 0), ("Base isolation", 0), ("A five-storey building", None)):
        page.load_trial(next(i for i, t in enumerate(TRIALS) if t.name == name))
        page.tabs.setCurrentIndex(TAB_BUILDING)
        h = page.history
        assert h is not None and page.playback.span == pytest.approx(h.t[-1])
        drift = np.abs(np.diff(h.displacement, prepend=0.0, axis=1))
        k = int(np.argmax(np.max(drift, axis=1)))  # the instant of the record's largest drift
        page.playback.seek(float(h.t[k]))
        assert not page.playback.running  # a scrub pauses it
        assert np.allclose(page.building.u, h.displacement[k])
        share = page.building.drifts()
        top = int(np.argmax(share))
        assert share[top] == pytest.approx(1.0)
        if storey is not None:  # the soft storey and the bearings take the drift
            assert top == storey
        pen = page.building.columns[top].opts["pen"].color()
        assert pen.name() == QtGui.QColor(colors.force).name()
        assert "Base shear" in page.building_note.text()

    page.k1.setValue(2e6)
    assert w.sim_page.sim.system.stiffness[0] == pytest.approx(2e6) and page.system.stiffness[0] == pytest.approx(2e6)
    assert "T = 2π√(Σm/k₁) = 0.993 s" in page.k1_info.text()  # 2π √(5 × 10 t / 2 MN/m)


def test_the_records_are_made_at_one_g_and_scaled():
    """The synthesis is linear in a_g: the records kept at 1 g, scaled, are the ones made at the PGA itself."""
    from vib_tutorial.core.seismic import records

    d = Design(0.37)
    for kept, made in zip(records(d, 3), (accelerogram(d, seed) for seed in (1, 2, 3))):
        assert np.allclose(kept.a, made.a, rtol=1e-10, atol=1e-12 * np.max(np.abs(made.a)))
        assert np.allclose(kept.spectrum, made.spectrum, rtol=1e-10)


def test_a_new_spectrums_records_are_made_on_the_worker(main_window):
    """Damping ζ changed: the spectrum and the table follow at once, the Floors tab once its records are in (made
    off the GUI thread, 0.5–1 s); a tab picked meanwhile draws now."""
    from PySide6 import QtWidgets

    from vib_tutorial.gui.seismic import TAB_FLOORS, TAB_GROUND

    w = main_window
    page = w.seismic_page
    w.pages.setCurrentWidget(page)
    page.tabs.setCurrentIndex(TAB_FLOORS)
    QtWidgets.QApplication.processEvents()
    floors, table = page.floor_note.text(), page.table.text()
    page.zeta.setValue(7.3)  # a ζ no test has made records for
    assert page.table.text() != table and page.worker.busy and page.floor_note.text() == floors
    page.worker.wait()
    assert page.floor_note.text() != floors
    page.zeta.setValue(7.7)
    page.tabs.setCurrentIndex(TAB_GROUND)  # now: the reader waits for it
    assert not page.worker.busy and "synthesized records" in page.ground_note.text()
