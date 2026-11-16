"""The FRF matrix page: its build-up (the first r modal terms summed, stepped from one to all), and its grid
kept across a new n."""

import numpy as np


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


def test_a_new_n_keeps_the_grids_shared_cells_and_draws_them_all(main_window):
    from PySide6 import QtWidgets

    w = main_window
    page = w.frf_page
    w.pages.setCurrentWidget(page)
    QtWidgets.QApplication.processEvents()
    grid = page.grid
    page.dof.setValue(4)
    QtWidgets.QApplication.processEvents()
    kept = [row[:3] for row in grid.cells[:3]]
    for n in (6, 3):
        page.dof.setValue(n)
        QtWidgets.QApplication.processEvents()
        assert grid.n == n and len(grid.cells) == n and all(len(row) == n for row in grid.cells)
        assert len(grid.left_axes) == len(grid.bottom_axes) == len(grid.headers) == len(grid.row_labels) == n
        assert grid.footer is not None and grid.ci.items[grid.footer] == [(n + 2, c) for c in range(2, n + 2)]
        for k, axis in enumerate(grid.bottom_axes):
            assert grid.ci.items[axis] == [(n + 1, k + 2)], "the frequency axes below the last row"
        assert all(grid.ci.getItem(j + 1, k + 2) is grid.cells[j][k] for j in range(n) for k in range(n))
    assert [row[:3] for row in grid.cells] == kept, "the cells both sizes share are kept, not made again"
    assert len(grid.ci.items) == 2 + 3 * 3 + 3 * 4, "nothing left behind: corner, footer, cells, labels and axes"
    page.show_terms.setChecked(True)
    QtWidgets.QApplication.processEvents()
    shown = [c for c in grid.terms[2][1] if c.isVisible()]
    assert len(shown) == len(page.terms), "a curve per term, the spare ones hidden"
    x, _ = shown[0].getData()
    assert np.allclose(x, np.log10(page.freqs)), "log10 of the frequencies: the curves take the log themselves"
