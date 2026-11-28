"""State-space modes' view model (`view_models/state_space.py`): the planes' coordinates and labels, the matrices'
steps and their marked coupling, the Sizes tab's coupled and clean decays, the modes' phases, the starts, and
`LiveBases.hold_state`."""

from __future__ import annotations

import re

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
    assert not m.coupled and "the damping is proportional" in m.note
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
    assert v.STARTS[v.BUTTONS :] == ("m₁ pulled",)  # a trial's own: the reader pulls m₁ by hand
    x, xd = v.start_state(DEFAULT, 4)
    assert list(x) == [v.START_REACH, 0.0] and not xd.any()
    for r in (0, 1):  # a mode of M and K: displaced along it, from rest, the other undamped pair at zero
        x, xd = v.start_state(DEFAULT, r)
        assert np.abs(x).max() == pytest.approx(v.START_REACH) and not xd.any()
        w = v.pairs(v.basis_of(DEFAULT, v.UNDAMPED), x, xd)[0]
        assert np.hypot(w[2 * (1 - r)], w[2 * (1 - r) + 1]) == pytest.approx(0.0, abs=1e-9)
    for r in (0, 1):
        x, xd = v.start_state(DEFAULT, 2 + r)
        assert np.abs(x).max() == pytest.approx(v.START_REACH) and np.abs(xd).max() > 0  # a mode starts moving
        w = v.pairs(v.basis_of(DEFAULT, v.STATE), x, xd)[0]
        assert np.hypot(w[2 * (1 - r)], w[2 * (1 - r) + 1]) == pytest.approx(0.0, abs=1e-9)


def test_the_sizes_tab_shows_the_coupling_and_the_clean_decay():
    """Started in mode 2 of M and K, mode 1's undamped pair climbs from zero; the state-space pairs each decay as
    e^(σt) and nothing more."""
    M, C, K = DEFAULT.matrices()
    x, xd = v.start_state(DEFAULT, 1)
    steps = Steps(M, C, K, np.concatenate([x, xd]))
    ts, zs, _ = steps.advance(3.0)
    sz = v.sizes(DEFAULT, zs[:, :2], zs[:, 2:])
    assert sz.undamped[0][0] == pytest.approx(v.FLOOR * v.MM) and sz.undamped[0].max() > 0.05  # mm: passed across
    one, two = v.modes(DEFAULT)
    for size, m in zip(sz.state, (one, two)):
        assert size == pytest.approx(size[0] * np.exp(m.sigma * (ts - ts[0])), rel=1e-6)
    assert "climbs from nothing" in sz.note and "σ = -0.358 /s" in sz.note
    empty = v.sizes(DEFAULT, np.empty((0, 2)), np.empty((0, 2)))
    assert empty.state[0].size == 0


def test_the_shapes_and_the_x_plane():
    one, two = (sh.note for sh in v.pair_shapes(DEFAULT, v.STATE))
    assert "Complex: m₂ lags m₁ by 4.51°, not the 0° of moving in step" in one
    assert "Complex: m₂ lags m₁ by 173°, not the 180° of moving opposed" in two
    p = v.x_plane(DEFAULT)
    assert "6.7° out of step" in p.note and all(sh.swings for sh in p.shapes)
    ex, ey = p.shapes[0].ellipse
    assert ex.size == ey.size == 121 and np.ptp(ex) > 0
    real = v.x_plane(SYMMETRIC)
    assert "along a line" in real.note and real.shapes[0].im == pytest.approx((0.0, 0.0), abs=1e-9)
    for sh in v.pair_shapes(SYMMETRIC, v.STATE):
        assert "Real: the masses move exactly" in sh.note
    heavy = v.pair_shapes(TwoMass(m2=2.0, c=20.0), v.STATE)
    assert "overdamped" in heavy[1].note
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


def test_a_starts_note_gives_its_initial_conditions_and_tells_it_from_its_neighbor():
    near = v.start_note(DEFAULT, 1, v.STATE)  # mode 2 of M and K, seen in A's modes: mode 1's pair isn't zero
    assert near == (
        "Mode 2 of M and K: x = (30, -5.58) mm, ẋ = (0, 0) mm/s: displaced along the mode's shape, from rest. "
        "On the planes: (0.0779, -0.99) and (30.5, 1.65) mm."
    )
    own = v.start_note(DEFAULT, 3, v.STATE)
    assert "ẋ = (-31.5, 21) mm/s: displaced along Re ψ and already moving" in own and "(0, 0) and (30.5, 0)" in own
    assert "the mode of M and K itself" in v.start_note(TwoMass(m2=2.0, c=0.0), 3, v.STATE)


def test_your_own_shapes_start_as_the_modes_of_M_and_K_and_snap_onto_As():
    s = TwoMass(m2=2.0, c=3.0)
    start = v.pair_shapes(s, v.OWN)
    assert [o.m2[1] for o in start] == [0.0, 0.0] and all(o.movable for o in start)
    assert [o.closeness for o in start] == [0.0, 0.0]
    assert [o.m2[0] for o in start] == pytest.approx([o.start for o in start])
    assert "passes motion to mode 2's" in start[0].note
    assert start[0].m1 == (1.0, 0.0) and not start[0].m1_cut
    assert start[1].m1[0] == pytest.approx(v.EDGE * start[1].span) and start[1].m1_cut  # cut at mode 2's edge
    assert v.basis_of(s, v.OWN) == pytest.approx(v.basis_of(s, v.OWN, v.undamped_ratios(s)))
    true = [v.ratio(m) for m in v.modes(s)]
    near = v.snap(s, 0, true[0] + 0.5 * v.SNAP * v.own_span(s, 0))
    assert near == true[0]  # close enough: onto A's own
    far = true[0] + 3 * v.SNAP * v.own_span(s, 0)
    assert v.snap(s, 0, far) == far
    assert v.snap(s, 1, complex(100.0, -100.0)) == complex(v.own_span(s, 1), -v.own_span(s, 1))  # kept on its plot
    mine = (true[0], v.undamped_ratios(s)[1])
    one, two = v.pair_shapes(s, v.OWN, mine)
    assert one.closeness == 1.0 and "A's own mode 1" in one.note and two.closeness == 0.0
    halfway = v.pair_shapes(s, v.OWN, (0.5 * (true[0] + v.undamped_ratios(s)[0]), mine[1]))[0].closeness
    assert halfway is not None and 0.0 < halfway < 1.0
    assert "each mode's pair is A's own" in v.matrices_view(s, v.OWN, mine).note
    assert v.matrices_view(s, v.OWN, tuple(true)).note.startswith("Your own shapes are A's own modes")
    marked = v.matrices_view(s, v.OWN, mine).product.marked
    assert not any(marked[i][j] for i in (2, 3) for j in (0, 1))  # mode 1's columns: no red
    assert any(marked[i][j] for i in (0, 1) for j in (2, 3))  # mode 2's still passes to mode 1
    assert v.labels(s, v.OWN)[0].title == "Your mode 1"
    lit = v.matrices_view(s, v.OWN, mine, lit=1)
    assert lit.basis.lit == lit.product.lit == 1 and v.matrices_view(s, v.OWN, mine).basis.lit is None


@pytest.mark.parametrize("which", range(4))
def test_every_bases_pairs_are_re_and_im_of_psi_and_lambda_psi(which):
    """Each pair's shape gives the λ, ψ and λψ whose Re and Im are its columns of T, row by row, in every basis."""
    s = TwoMass(m2=2.0, c=3.0)
    T = v.basis_of(s, which)
    for r, o in enumerate(v.pair_shapes(s, which)):
        found = re.search(
            r"λ = (.+) /s\. Rows x₁, x₂: ψ = \((.+), (.+)\)\. Rows ẋ₁, ẋ₂: λψ = \((.+), (.+)\)\.", o.columns
        )
        assert found, o.columns
        lam, psi1, psi2, v1, v2 = (complex(g.replace(" ", "").replace("i", "j")) for g in found.groups())
        assert [v1, v2] == pytest.approx([lam * psi1, lam * psi2], rel=0.02, abs=0.01 * abs(lam))
        y = np.array([psi1, psi2, v1, v2])  # T's rows x₁, x₂, ẋ₁, ẋ₂, as printed
        assert T[:, 2 * r] == pytest.approx(y.real, rel=0.02, abs=0.02 * np.abs(y).max())
        assert T[:, 2 * r + 1] == pytest.approx(y.imag, rel=0.02, abs=0.02 * np.abs(y).max())
    mass = v.pair_shapes(s, v.MASS)
    assert mass[0].m2 == (0.0, 0.0) and mass[1].m1 == (0.0, 0.0) and "m₂ alone" in mass[1].note
    assert all(o.closeness is None and not o.movable for b in range(3) for o in v.pair_shapes(s, b))


def test_your_own_shapes_survive_the_trials_fields():
    own = (complex(2.5, -0.5), complex(-0.2, -0.05))
    t = v.StateTrial("", "", **v.own_fields(own))
    assert v.own_of(t) == own and v.own_of(v.StateTrial("", "")) is None
    assert v.own_fields(None) == {"own1": None, "own2": None}


def test_an_overdamped_mode_keeps_As_own_pair_in_your_basis():
    s = TwoMass(m2=2.0, c=20.0)
    two = v.pair_shapes(s, v.OWN)[1]
    assert (
        not two.movable
        and "overdamped" in two.note
        and "real eigenvectors" in two.columns
        and "rows ẋ₁, ẋ₂ are λψ" in two.columns
    )
    assert v.basis_of(s, v.OWN)[:, 2:] == pytest.approx(v.basis_of(s, v.STATE)[:, 2:])


def test_with_no_dampers_the_gauge_is_full_at_the_start_and_empties_as_you_drag_away():
    s = TwoMass(m2=2.0, c=0.0)  # the modes of M and K are A's own: the start is already the answer
    start = v.undamped_ratios(s)
    assert [o.closeness for o in v.pair_shapes(s, v.OWN)] == [1.0, 1.0]
    away = v.pair_shapes(s, v.OWN, (start[0] + 1.5j, start[1]))
    assert away[0].closeness is not None and away[0].closeness < 0.5 and "passes motion" in away[0].note
    assert away[1].closeness == 1.0


def test_written_complex_the_matrices_are_y_and_a_diagonal_y_inverse_a_y():
    s = TwoMass(m2=2.0, c=3.0)
    own = v.matrices_view(s, v.STATE, written=v.COMPLEX)
    assert own.basis.symbol == "Y" and own.product.symbol == "Y⁻¹AY"
    assert own.basis.cols == ("η₁", "η̄₁", "η₂", "η̄₂") and own.product.rows == own.basis.cols
    assert all(own.product.entries[i][j] == "0" for i in range(4) for j in range(4) if i != j)  # λ, λ̄ alone
    lam = v.modes(s)[0].eigenvalues[0]
    assert own.product.entries[0][0] == v._complex(lam) and own.product.entries[1][1] == v._complex(lam.conjugate())
    assert "diagonal" in own.note and not any(any(row) for row in own.product.marked)
    undamped = v.matrices_view(s, v.UNDAMPED, written=v.COMPLEX)
    assert any(any(row) for row in undamped.product.marked)  # the dampers still couple, as in T
    real = v.matrices_view(s, v.STATE)
    assert real.basis.symbol == "T" and "2×2 block" in real.note
    over = v.matrices_view(TwoMass(m2=2.0, c=20.0), v.STATE, written=v.COMPLEX)
    assert over.basis.cols[2:] == v.names(TwoMass(m2=2.0, c=20.0), v.STATE)[2:]  # an overdamped mode's stay real
    shape = v.pair_shapes(s, v.STATE, written=v.COMPLEX)[0]
    assert shape.columns.startswith("Y's η₁ and η̄₁ are y = [ψ; λψ] and its conjugate ȳ")
