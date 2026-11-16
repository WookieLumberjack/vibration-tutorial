"""The FRF matrix page's build-up: the first r modal terms summed, stepped from one to all."""

import numpy as np
import pytest


def test_the_build_up_sums_the_first_terms_and_steps_to_all(main_window):
    from PySide6 import QtWidgets

    w = main_window
    page = w.frf_page
    w.pages.setCurrentWidget(page)
    QtWidgets.QApplication.processEvents()
    page.load_trial(0)
    n = len(page.mode_checks)
    assert page.build.maximum() == n and page.build_count.text() == f"{n} of {n}"

    page.build.setValue(2)  # the first two terms only
    assert list(page.selected_terms()) == [True, True] + [False] * (n - 2)
    assert page.practice_state().unticked == tuple(range(3, n + 1))  # modes 3 … n left out
    page.mode_checks[0].setChecked(False)  # not the first ones now: a pick of its own
    assert page.build_count.text() == "a pick" and page.build.value() == 2

    page.build_play.click()  # from one term, a step at a time
    assert page.build_timer.isActive() and page.build.value() == 1 and page.build_play.text() == "Stop"
    for r in range(2, n + 1):
        page._build_step()
        assert page.build.value() == r and page.selected_terms().sum() == r
    assert not page.build_timer.isActive() and page.build_play.text() == "Build H mode by mode"
    j, k = page.output, sorted(page.inputs)[0]
    assert np.allclose(page.term_H[:, :, j, k].sum(axis=0), page.H[:, j, k], rtol=1e-6, atol=1e-12)

    page.build_play.click()
    w.pages.setCurrentWidget(w.sim_page)  # hidden: it stops
    assert not page.build_timer.isActive()
