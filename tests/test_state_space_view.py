"""State-space modes' view model (`view_models/state_space.py`): the planes' coordinates and labels, the matrices'
steps and their marked coupling, the Sizes tab's leak and clean decays, the Shapes tab's phasors, the starts, and
`LiveBases.hold_state`."""

from __future__ import annotations

import numpy as np
import pytest

from vib_tutorial.core.bases import Steps, TwoMass
from vib_tutorial.view_models import state_space as v
from vib_tutorial.view_models.live_bases import COL, LiveBases

DEFAULT = v.system(v.StateTrial("", ""))
SYMMETRIC = TwoMass(c=1.0)


def test_the_page_opens_on_dampers_the_undamped_modes_cant_uncouple():
    assert DEFAULT == TwoMass(m2=2.0, c=1.0)
    marked = {b: any(any(row) for row in v.matrices_view(DEFAULT, b).product.marked) for b in range(3)}
    assert marked == {v.MASS: True, v.UNDAMPED: True, v.STATE: False}
    m = v.matrices_view(DEFAULT, v.UNDAMPED)
    assert m.coupled and "dampers still do" in m.note
    coupled = [(i, j) for i in range(4) for j in range(4) if m.product.marked[i][j]]
    assert coupled == [(1, 3), (3, 1)]  # the damping's terms alone, in each pair's velocity row
    assert m.product.entries[1][3] == "0.41" and m.product.entries[3][1] == "0.333"
    assert m.modes == (
        "Mode 1: λ = -0.358 ± 15.21i /s: f = 2.421 Hz, ζ = 2.35 %",
        "Mode 2: λ = -1.14 ± 22.74i /s: f = 3.619 Hz, ζ = 5.02 %",
    )
    assert m.product.rows == ("q₁", "q̇₁/ω₁", "q₂", "q̇₂/ω₂") and m.state.cols == ("x₁", "x₂", "ẋ₁", "ẋ₂")
    assert m.product.row_groups == ("i", "i", "b", "b") and m.given[0].rows == ("x₁", "x₂")
    s = v.matrices_view(DEFAULT, v.STATE)
    assert s.product.entries[0] == ("-0.358", "15.2", "0", "0") and not s.coupled


def test_proportional_dampers_leave_the_undamped_modes_uncoupled():
    m = v.matrices_view(SYMMETRIC, v.UNDAMPED)
    assert not m.coupled and "spread like the masses and springs" in m.note
    assert "nothing passes between" in v.sizes_note(SYMMETRIC)
    assert "no dampers" in v.matrices_view(TwoMass(m2=2.0), v.UNDAMPED).note


def test_the_planes_coordinates_and_setting_a_pair():
    B = v.basis_of(DEFAULT, v.STATE)
    x, xd = v.start_state(DEFAULT, 3)  # state-space mode 2
    w = v.pairs(B, x, xd)[0]
    assert w[:2] == pytest.approx([0.0, 0.0], abs=1e-9) and w[2] > 0 and w[3] == pytest.approx(0.0, abs=1e-9)
    x2, v2 = v.place(B, w, 0, 5.0, -2.0)  # mode 1's pair to (5, −2) mm, mode 2's kept
    assert v.pairs(B, x2, v2)[0] == pytest.approx([5.0, -2.0, w[2], w[3]])
    assert [p.title for p in v.labels(DEFAULT, v.UNDAMPED)] == ["Mode 1 of M and K", "Mode 2 of M and K"]
    assert v.labels(DEFAULT, v.MASS)[1].y.startswith("ẋ₂/ω̂₂")
    assert v.labels(TwoMass(m2=2.0, c=20.0), v.STATE)[1].title == "State-space mode 2: overdamped"


def test_the_starts():
    x, xd = v.start_state(DEFAULT, 0)
    assert list(x) == [v.START_REACH, 0.0] and not xd.any()
    x, xd = v.start_state(DEFAULT, 1)
    assert np.abs(x).max() == pytest.approx(v.START_REACH) and not xd.any()
    for r in (0, 1):
        x, xd = v.start_state(DEFAULT, 2 + r)
        assert np.abs(x).max() == pytest.approx(v.START_REACH) and np.abs(xd).max() > 0  # a mode starts moving
        w = v.pairs(v.basis_of(DEFAULT, v.STATE), x, xd)[0]
        assert np.hypot(w[2 * (1 - r)], w[2 * (1 - r) + 1]) == pytest.approx(0.0, abs=1e-9)


def test_the_sizes_tab_shows_the_leak_and_the_clean_decay():
    """Started in mode 2 of M and K, mode 1's undamped pair climbs from zero; the state-space pairs each decay as
    e^(σt) and nothing more."""
    M, C, K = DEFAULT.matrices()
    x, xd = v.start_state(DEFAULT, 1)
    steps = Steps(M, C, K, np.concatenate([x, xd]))
    ts, zs, _ = steps.advance(3.0)
    sz = v.sizes(DEFAULT, zs[:, :2], zs[:, 2:])
    assert sz.undamped[0][0] == pytest.approx(v.FLOOR * v.MM) and sz.undamped[0].max() > 0.05  # mm: the leak
    one, two = v.modes(DEFAULT)
    for size, m in zip(sz.state, (one, two)):
        assert size == pytest.approx(size[0] * np.exp(m.sigma * (ts - ts[0])), rel=1e-6)
    assert "climbs from nothing" in sz.note and "σ = -0.358 /s" in sz.note
    empty = v.sizes(DEFAULT, np.empty((0, 2)), np.empty((0, 2)))
    assert empty.state[0].size == 0


def test_the_shapes_and_the_x_plane():
    one, two = v.shapes_view(DEFAULT)
    assert one.note.startswith("m₂ lags m₁ by 4.51°, not the 0° of moving in step")
    assert two.note.startswith("m₂ lags m₁ by 173°, not the 180° of moving opposed")
    assert one.undamped[0][1] == 0.0 and one.masses[0][1] != 0.0
    p = v.x_plane(DEFAULT)
    assert "6.7° out of step" in p.note and all(sh.swings for sh in p.shapes)
    ex, ey = p.shapes[0].ellipse
    assert ex.size == ey.size == 121 and np.ptp(ex) > 0
    real = v.x_plane(SYMMETRIC)
    assert "along a line" in real.note and real.shapes[0].im == pytest.approx((0.0, 0.0), abs=1e-9)
    for sh in v.shapes_view(SYMMETRIC):
        assert sh.note.startswith("The masses move exactly")
    heavy = v.shapes_view(TwoMass(m2=2.0, c=20.0))
    assert heavy[1].note.startswith("Overdamped")
    assert v.x_plane(TwoMass(m2=2.0, c=60.0)).note.startswith("Both modes overdamped")


def test_a_held_state_is_let_go_moving():
    live = LiveBases(DEFAULT)
    x, xd = v.start_state(DEFAULT, 3)
    live.hold_state(x, xd)
    assert live.advance(0.5) == 0  # held: the clock stops
    assert live.books["released"] == pytest.approx(live.stored(x, xd))
    live.let_go()
    assert live.advance(0.01) > 0
    M, C, K = DEFAULT.matrices()
    _, zs, _ = Steps(M, C, K, np.concatenate([x, xd])).advance(0.01)  # as if let go moving from there
    first = live.record.y[0]
    assert [first[COL[k]] for k in ("x1", "x2", "v1", "v2")] == pytest.approx(list(zs[0]))
    live.set_rail(np.array([1.0, 0.0]))
    with pytest.raises(ValueError):
        live.hold_state(x, xd)
