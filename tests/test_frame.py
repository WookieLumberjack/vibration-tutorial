"""The frame builder's model (core/frame.py) against closed forms, and its trials' numbers."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from vib_tutorial.core import beams_plates as bp
from vib_tutorial.core.frame import (
    STEEL,
    Element,
    Frame,
    FrameError,
    Kind,
    Node,
    Section,
    Shape,
    Support,
    Theory,
    assemble,
    beam_matrices,
    cantilever,
    local,
    member_curve,
    modes,
    portal,
    simply_supported,
    storeys,
    warren,
)

SEC = Section(depth=0.05, width=0.05)
EULER_ROOTS = (1.8751041, 4.6940911, 7.8547574)


def bending(m, count: int = 3) -> np.ndarray:
    """The first `count` modes with mostly bending strain energy."""
    return m.hz[m.energy[1] > 0.5][:count]


def euler_cantilever(section: Section = SEC, length: float = 1.0) -> np.ndarray:
    a = math.sqrt(STEEL.E * section.inertia / (STEEL.rho * section.area))
    return np.array([b * b / length**2 * a / (2 * math.pi) for b in EULER_ROOTS])


def test_cantilever_converges_from_above():
    exact = euler_cantilever()
    previous = None
    for parts in (2, 4, 8, 16):
        f = bending(modes(cantilever(1.0, parts)))
        assert np.all(f >= exact * (1 - 1e-9))  # consistent mass: Rayleigh–Ritz upper bounds
        if previous is not None:
            assert np.all(f <= previous)
        previous = f
    assert previous == pytest.approx(exact, rel=1e-4)


def test_timoshenko_against_the_closed_form():
    deep = Section(depth=0.2, width=0.05)
    beam = bp.Beam(length=1.0, depth=0.2, width=0.05)
    for theory, ref in ((Theory.EULER, bp.Theory.EULER), (Theory.TIMOSHENKO, bp.Theory.TIMOSHENKO)):
        frame = dataclasses.replace(simply_supported(1.0, 32, deep), theory=theory)
        assert bending(modes(frame)) == pytest.approx(bp.beam_frequencies(beam, 3, ref), rel=2e-3)


def test_timoshenko_element_is_exact_statically():
    """One element, a tip load: δ = PL³/3EI + PL/κGA, the tip's turn PL²/2EI."""
    deep = Section(depth=0.2, width=0.05)
    frame = dataclasses.replace(cantilever(1.0, 1, deep), theory=Theory.TIMOSHENKO)
    a = assemble(frame)
    f = a.free_dofs
    load = np.zeros(frame.n_dof)
    load[4] = 1000.0
    d = np.zeros(frame.n_dof)
    d[f] = np.linalg.solve(a.K[np.ix_(f, f)], load[f])
    EI, kGA = STEEL.E * deep.inertia, deep.kappa(STEEL.nu) * STEEL.G * deep.area
    assert d[4] == pytest.approx(1000 / (3 * EI) + 1000 / kGA, rel=1e-12)
    assert d[5] == pytest.approx(1000 / (2 * EI), rel=1e-12)


def test_euler_matrices_are_the_textbook_ones():
    L, rho, A = 0.7, 7850.0, 2e-3
    k, m, _, phi = beam_matrices(STEEL.E, STEEL.G, rho, A, 3e-7, 5 / 6, L, Theory.EULER, False)
    assert phi == 0
    vt = [1, 2, 4, 5]
    ref = (
        rho
        * A
        * L
        / 420
        * np.array(
            [
                [156, 22 * L, 54, -13 * L],
                [22 * L, 4 * L * L, 13 * L, -3 * L * L],
                [54, 13 * L, 156, -22 * L],
                [-13 * L, -3 * L * L, -22 * L, 4 * L * L],
            ]
        )
    )
    assert m[np.ix_(vt, vt)] == pytest.approx(ref, rel=1e-12)
    assert k[np.ix_(vt, vt)][0, 1] == pytest.approx(6 * STEEL.E * 3e-7 / L**2)
    # Timoshenko's at φ → 0 (a huge G) is Euler–Bernoulli's plus only the rotary inertia
    k2, m2, _, phi2 = beam_matrices(STEEL.E, 1e30, rho, A, 3e-7, 5 / 6, L, Theory.TIMOSHENKO, False)
    assert phi2 < 1e-15 and k2 == pytest.approx(k, rel=1e-12)
    rot = (
        rho
        * 3e-7
        / (30 * L)
        * np.array(
            [
                [36, 3 * L, -36, 3 * L],
                [3 * L, 4 * L * L, -3 * L, -L * L],
                [-36, -3 * L, 36, -3 * L],
                [3 * L, -L * L, -3 * L, 4 * L * L],
            ]
        )
    )
    assert (m2 - m)[np.ix_(vt, vt)] == pytest.approx(rot, rel=1e-9, abs=1e-15)


@pytest.mark.parametrize("theory", list(Theory))
@pytest.mark.parametrize("lumped", [False, True])
@pytest.mark.parametrize("kind", [Kind.BAR, Kind.BEAM])
def test_a_rigid_translation_carries_the_whole_mass(theory, lumped, kind):
    frame = Frame((Node(0.0, 0.0), Node(0.6, 0.8)), (Element(kind, 0, 1, section=SEC),), lumped, theory)
    a = assemble(frame)
    total = STEEL.rho * SEC.area * 1.0
    for direction in ((1.0, 0.0), (0.0, 1.0), (0.6, -0.8)):
        r = np.zeros(frame.n_dof)
        r[0::3], r[1::3] = direction
        assert r @ a.M @ r == pytest.approx(total, rel=1e-12)
        assert np.abs(a.K @ r).max() < 1e-6 * np.abs(a.K).max()  # and strains nothing


def test_a_bar_at_45_degrees():
    frame = Frame((Node(0.0, 0.0, Support.PINNED), Node(1.0, 1.0)), (Element(Kind.BAR, 0, 1, section=SEC),))
    part = local(frame, 0)
    ea_l = STEEL.E * SEC.area / math.sqrt(2)
    K = part.stiffness()
    assert K == pytest.approx(ea_l / 2 * np.array([[1, 1, -1, -1], [1, 1, -1, -1], [-1, -1, 1, 1], [-1, -1, 1, 1]]))
    a = assemble(frame)
    assert a.unused.tolist() == [False, False, True, False, False, True]  # the turns
    assert a.fixed.tolist() == [True, True, False, False, False, False]


def test_assembly_is_symmetric_and_supports_strike_out():
    frame = portal(2.0, 1.5, 2)
    frame = frame.with_element(Element(Kind.SPRING, 1, 2, value=1e5))
    frame = frame.with_element(Element(Kind.DAMPER, 2, None, value=50.0, anchor=(3.0, 1.5)))
    frame = frame.with_element(Element(Kind.TORSION_SPRING, 1, value=2e3))
    frame = frame.with_element(Element(Kind.MASS, 2, value=4.0, inertia=0.1))
    a = assemble(frame)
    for A in (a.K, a.M, a.C):
        assert A == pytest.approx(A.T)
    assert int(a.fixed.sum()) == 6 and int(a.unused.sum()) == 0
    assert a.C[6, 6] == pytest.approx(50.0) and a.M[8, 8] > 0.1  # the damper along x at node 3; J on θ3


def test_mechanisms():
    square = Frame(
        (Node(0.0, 0.0, Support.PINNED), Node(1.0, 0.0, Support.PINNED), Node(1.0, 1.0), Node(0.0, 1.0)),
        tuple(Element(Kind.BAR, i, j) for i, j in ((0, 1), (1, 2), (2, 3), (3, 0))),
    )
    assert modes(square).mechanisms == 1 and modes(square).hz[0] == 0
    braced = modes(square.with_element(Element(Kind.BAR, 0, 2)))
    assert braced.mechanisms == 0 and braced.hz[0] > 100
    free = Frame((Node(0.0, 0.0), Node(1.0, 0.0), Node(2.0, 0.0)), (Element(Kind.BEAM, 0, 1), Element(Kind.BEAM, 1, 2)))
    m = modes(free)
    assert m.mechanisms == 3 and np.all(m.hz[:3] == 0) and m.hz[3] > 10
    # a fine mesh's soft bending is far below its stiff stretch, and still not a mechanism
    assert modes(cantilever(1.0, 16)).mechanisms == 0


def test_massless_dofs_are_condensed_exactly():
    spring_mass = Frame(
        (Node(0.0, 0.0, Support.ROLLER_Y),),
        (Element(Kind.MASS, 0, value=2.0), Element(Kind.SPRING, 0, anchor=(0.0, -1.0), value=800.0)),
    )
    assert modes(spring_mass).hz == pytest.approx([math.sqrt(400) / (2 * math.pi)])
    # two springs in series to a mass: the node between has no mass
    series = Frame(
        (Node(0.0, 0.0, Support.ROLLER_Y), Node(0.0, 1.0, Support.ROLLER_Y)),
        (
            Element(Kind.SPRING, 0, anchor=(0.0, -1.0), value=1000.0),
            Element(Kind.SPRING, 0, 1, value=1000.0),
            Element(Kind.MASS, 1, value=5.0),
        ),
    )
    m = modes(series)
    assert m.condensed == 1 and m.hz == pytest.approx([math.sqrt(500 / 5) / (2 * math.pi)])
    lumped = modes(dataclasses.replace(cantilever(1.0, 4), lumped=True))
    assert lumped.condensed == 4 and lumped.hz.size == 8
    loose = Frame((Node(0.0, 0.0, Support.ROLLER_Y), Node(0.0, 1.0)), (Element(Kind.SPRING, 0, 1, value=1.0),))
    with pytest.raises(FrameError):
        modes(loose)


def test_strain_shares():
    m = modes(cantilever(1.0, 4))
    assert m.energy.sum(axis=0) == pytest.approx(np.ones(m.hz.size))
    assert m.energy[0, 3] == pytest.approx(1.0, abs=1e-9)  # the axial mode
    truss = modes(warren())
    assert truss.energy[0] == pytest.approx(np.ones(truss.hz.size))


def test_member_curve_is_the_static_cubic():
    """A tip load on a one-element cantilever: the Hermite curve is the exact w = Px²(3L − x)/6EI."""
    frame = Frame((Node(0.0, 0.0, Support.CLAMPED), Node(0.6, 0.8)), (Element(Kind.BEAM, 0, 1, section=SEC),))
    a = assemble(frame)
    f = a.free_dofs
    across = np.array([-0.8, 0.6])
    load = np.zeros(6)
    load[3:5] = 50.0 * across
    q = np.zeros(6)
    q[f] = np.linalg.solve(a.K[np.ix_(f, f)], load[f])
    x, y = member_curve(frame, 0, q, points=11)
    s = np.linspace(0.0, 1.0, 11)
    EI = STEEL.E * SEC.inertia
    w = 50.0 * s**2 * (3 - s) / (6 * EI)
    moved = np.column_stack([x, y]) - np.outer(s, [0.6, 0.8])
    assert moved @ across == pytest.approx(w, rel=1e-6)
    assert moved[-1] == pytest.approx(q[3:5])


def test_editing():
    frame = portal(2.0, 1.5, 1)
    assert frame.divide(1, 3).divide(0, 1) == frame.divide(1, 3)
    split = frame.divide(1, 3)
    assert len(split.nodes) == 6 and len(split.elements) == 5
    gone = split.without_node(1)
    assert len(gone.nodes) == 5 and all(e.i < 5 and (e.j is None or e.j < 5) for e in gone.elements)
    assert not any(e.kind is Kind.BEAM and {e.i, e.j} == {0, 1} for e in gone.elements)
    assert frame.node_at(2.0, 1.5) == 2
    tube = Section(Shape.TUBE, depth=0.1, wall=0.05)
    assert tube.inner == 0 and tube.area == pytest.approx(math.pi * 0.0025)
    assert tube.kappa(0.3) == pytest.approx(6 * 1.3 / (7 + 6 * 0.3))


# ------------------------------------------------------------------ the trials' numbers
def test_trials():
    from vib_tutorial.gui.frame_builder import SQUARE, TRIALS

    by = {t.name: t for t in TRIALS}
    m = modes(by["A cantilever in four beams"].frame)
    assert m.hz[:4] == pytest.approx([41.78, 262.1, 738.7, 1301], abs=0.6)
    assert euler_cantilever() == pytest.approx([41.78, 261.8, 733.1], abs=0.05)
    assert m.energy[0, 3] > 0.999
    one = modes(by["One beam element"].frame).hz
    exact = euler_cantilever()
    assert (one[0], one[1]) == pytest.approx((41.97, 413.6), abs=0.05)
    assert 100 * (one[0] / exact[0] - 1) == pytest.approx(0.5, abs=0.05)
    assert 100 * (one[1] / exact[1] - 1) == pytest.approx(58, abs=0.5)
    m = modes(by["Lumped mass"].frame)
    assert m.hz[:3] == pytest.approx([40.61, 238.7, 632.1], abs=0.05)
    assert m.condensed == 4 and m.hz.size == 8
    deep = by["A deep beam: Timoshenko"].frame
    m = modes(deep)
    assert bending(m) == pytest.approx([441.0, 1539, 2992], abs=0.6)
    assert m.hz[1] == pytest.approx(1295, abs=0.5) and m.energy[0, 1] > 0.999
    e = bending(modes(dataclasses.replace(deep, theory=Theory.EULER)))
    assert e == pytest.approx([469.1, 1877, 4227], abs=0.6)
    drops = 100 * (bending(m) / e - 1)
    assert drops == pytest.approx([-6, -18, -29], abs=0.5)
    beam = bp.Beam(length=1.0, depth=0.2, width=0.05)
    assert bp.beam_frequencies(beam, 3) == pytest.approx([440.8, 1529, 2921], abs=0.6)
    frame = by["A portal frame: sway"].frame
    assert modes(frame).hz[0] == pytest.approx(14.90, abs=0.005)
    pinned = Frame(
        tuple(
            dataclasses.replace(n, support=Support.PINNED) if n.support is Support.CLAMPED else n for n in frame.nodes
        ),
        frame.elements,
    )
    assert modes(pinned).hz[0] == pytest.approx(6.78, abs=0.005)
    truss = by["A Warren truss: bars only"].frame
    a = assemble(truss)
    assert (int(a.unused.sum()), int(a.fixed.sum()), int(a.free.sum())) == (9, 3, 15)
    assert modes(truss).hz[0] == pytest.approx(112.6, abs=0.05)
    m = modes(SQUARE)
    assert by["A square of bars: a mechanism"].frame == SQUARE and m.mechanisms == 1 and m.hz.size == 4
    braced = SQUARE.with_element(Element(Kind.BAR, 0, 2))  # node 1 to node 3
    assert modes(braced).hz[0] == pytest.approx(276.9, abs=0.05)
    two = by["Two storeys: the chain again"].frame
    assert two == storeys()
    assert modes(two).hz[:2] == pytest.approx([3.44, 9.14], abs=0.005)
    column, floor = Section(depth=0.02, width=0.05), Section(depth=0.1, width=0.02)
    k = 2 * 12 * STEEL.E * column.inertia
    girder, columns = STEEL.rho * floor.area * 1.5, STEEL.rho * column.area * 2
    assert (k / 1e3, girder, columns / 2) == pytest.approx((168, 23.6, 7.9), abs=0.051)
    mass = 100 + girder + columns / 2
    assert mass == pytest.approx(131.4, abs=0.05)
    chain = np.sqrt(k / mass * np.array([(3 - math.sqrt(5)) / 2, (3 + math.sqrt(5)) / 2])) / (2 * math.pi)
    assert chain == pytest.approx([3.52, 9.21], abs=0.005)


def test_the_presets_hold_still():
    for frame in (cantilever(), simply_supported(), portal(), warren(), storeys()):
        assert modes(frame).mechanisms == 0
