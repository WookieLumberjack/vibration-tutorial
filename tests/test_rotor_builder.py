"""The Rotor builder's model: closed forms, the Jeffcott rotor page's model, editing, and the trials' numbers."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import scipy.optimize

from vib_tutorial.core import rotor as jeffcott
from vib_tutorial.core.fluid_bearing import JournalBearing
from vib_tutorial.core.frame import STEEL, Material
from vib_tutorial.core.magnetic_bearing import MagneticBearing
from vib_tutorial.core.rotor_builder import (
    DISC_LIBRARY,
    GRAVITY,
    RPM,
    Bearing,
    BearingKind,
    Disc,
    Rotor,
    RotorError,
    Seal,
    Section,
    Theory,
    assemble,
    deflection,
    modes,
    pinned_ends,
    solid_disc,
    uniform,
)
from vib_tutorial.core.rotor_modes import whirl_modes
from vib_tutorial.core.rotor_presets import JOURNAL_BEARING, MAGNETIC_BEARING


def euler_pinned(length: float, diameter: float, n: int = 3) -> np.ndarray:
    s = Section(length, diameter)
    return (np.arange(1, n + 1) * math.pi) ** 2 / (2 * math.pi) * math.sqrt(s.ei / (STEEL.rho * s.area * length**4))


def hz(rotor: Rotor, rpm: float = 0.0) -> np.ndarray:
    return modes(assemble(rotor), rpm * RPM).hz


# ------------------------------------------------------------------ closed forms
def test_a_pinned_shaft_comes_in_pairs_near_euler():
    rayleigh = pinned_ends(Rotor(uniform(1.0, 0.05, 4, 3), theory=Theory.RAYLEIGH))
    f = hz(rayleigh)
    assert f[0::2][:3] == pytest.approx(f[1::2][:3], rel=1e-9)  # each twice
    assert f[0::2][:3] == pytest.approx(euler_pinned(1.0, 0.05), rel=0.01)
    timoshenko = hz(dataclasses.replace(rayleigh, theory=Theory.TIMOSHENKO))
    assert np.all(timoshenko[:6] < f[:6])  # shear softens it


def test_a_free_shaft_has_rigid_roots_and_spins_into_nutation():
    free = Rotor(uniform(1.0, 0.05, 4, 3))
    a = assemble(free)
    m = modes(a, 0.0)
    assert m.rigid == 8  # two translations and two tilts, each a double root
    s = Section(1.0, 0.05)
    beta = 4.730041  # a free-free beam's first root
    euler = beta**2 / (2 * math.pi) * math.sqrt(s.ei / (STEEL.rho * s.area))
    assert m.hz[0] == pytest.approx(euler, rel=0.01)
    spinning = modes(a, 3000 * RPM)
    assert spinning.rigid == 6
    assert spinning.whirl[0] > 0.99
    assert spinning.hz[0] == pytest.approx(1.5 * 0.05**2 * 50, rel=0.02)  # I_p Ω / I_d of a slender rod


def test_the_spin_splits_forward_up_and_backward_down():
    rotor = pinned_ends(Rotor(uniform(1.0, 0.05, 4, 3), discs=(DISC_LIBRARY[1].at(0.25),)))
    a = assemble(rotor)
    rest, fast = modes(a, 0.0), modes(a, 6000 * RPM)
    for k in range(3):
        back, fwd = fast.hz[2 * k], fast.hz[2 * k + 1]
        assert fast.whirl[2 * k] < -0.99 and fast.whirl[2 * k + 1] > 0.99
        assert back < rest.hz[2 * k] < fwd
    assert np.allclose(a.G, -a.G.T) and np.allclose(a.M, a.M.T) and np.allclose(a.K, a.K.T)
    assert np.all(np.linalg.eigvalsh(a.M) > 0)
    assert np.abs(fast.zeta).max() == 0  # undamped: round-off is not growth


def test_a_disc_from_the_library():
    d = solid_disc(0.0, STEEL, 0.3, 0.04)
    assert d.mass == pytest.approx(7850 * math.pi * 0.15**2 * 0.04)
    assert d.ip == pytest.approx(d.mass * 0.15**2 / 2)
    assert d.id == pytest.approx(d.mass * (3 * 0.15**2 + 0.04**2) / 12)
    ring = solid_disc(0.0, STEEL, 0.3, 0.04, bore=0.1)
    assert ring.ip == pytest.approx(ring.mass * (0.15**2 + 0.05**2) / 2)


# ------------------------------------------------------------------ against the Jeffcott rotor page's model
def _jeffcott(system: jeffcott.RotorSystem, bearing: Bearing) -> Rotor:
    """The rotor page's rotor rebuilt: its journals' masses as discs, its shaft in Rayleigh beams."""
    mat = Material("steel", system.youngs, 0.3, jeffcott.STEEL)
    a = system.position * system.length
    sections = (
        Section(a, system.diameter, 0.0, mat, 3),
        Section(system.length - a, system.diameter, 0.0, mat, 5),
    )
    j = system.bearing_a.mass
    discs = (Disc(a, system.disc_mass, system.ip, system.id), Disc(0.0, j, 0.0, 0.0), Disc(system.length, j, 0.0, 0.0))
    bearings = (bearing, dataclasses.replace(bearing, z=system.length))
    return Rotor(sections, discs, bearings, theory=Theory.RAYLEIGH, gravity=system.gravity)


@pytest.mark.parametrize("kind", ["springs", "film", "magnet"])
def test_it_matches_the_jeffcott_rotor_page(kind):
    if kind == "springs":
        brg = jeffcott.Bearing()
        ours = Bearing(0.0, kxx=brg.kx, kyy=brg.ky, cxx=brg.cx, cyy=brg.cy)
    elif kind == "film":
        brg = JOURNAL_BEARING
        ours = Bearing(0.0, BearingKind.FILM, film=brg.film or JournalBearing())
    else:
        brg = MAGNETIC_BEARING
        ours = Bearing(0.0, BearingKind.MAGNET, magnet=brg.magnet or MagneticBearing())
    system = jeffcott.RotorSystem(
        density=jeffcott.STEEL, elements=8, position=0.3, bearing_a=brg, bearing_b=brg, gravity=GRAVITY
    )
    rotor = _jeffcott(system, ours)
    a = assemble(rotor)
    if kind == "film":
        assert a.loads == pytest.approx(system.bearing_loads, rel=1e-9)
    for rpm in (1000.0, 6000.0):
        theirs = whirl_modes(system, rpm * RPM)
        m = modes(a, rpm * RPM)
        # theirs reduces the shaft (Craig–Bampton), so the higher modes differ a little
        assert m.hz[:4] == pytest.approx(theirs.freq_hz[:4], rel=1e-3)
        assert m.zeta[:4] == pytest.approx(theirs.zeta[:4], rel=2e-3, abs=1e-5)


def test_the_bearings_carry_the_weight():
    rotor = Rotor(
        uniform(1.2, 0.05, 4, 3),
        discs=(DISC_LIBRARY[0].at(0.3),),
        bearings=(Bearing(0.0, BearingKind.FILM), Bearing(0.6), Bearing(1.2, BearingKind.FILM)),
    )
    loads = assemble(rotor).loads
    assert sum(loads) == pytest.approx(rotor.mass * GRAVITY)
    assert assemble(dataclasses.replace(rotor, gravity=0.0)).loads == (0.0, 0.0, 0.0)
    with pytest.raises(RotorError):
        assemble(dataclasses.replace(rotor, bearings=rotor.bearings[:1]))


def test_a_seals_swirl_feeds_the_forward_whirl():
    still = Rotor(
        uniform(1.0, 0.05, 4, 3),
        discs=(DISC_LIBRARY[1].at(0.5),),
        bearings=(Bearing(0.0, kxx=2e7, kyy=2e7), Bearing(1.0, kxx=2e7, kyy=2e7)),
        seals=(Seal(0.25, swirl=0.0),),
    )
    swirling = dataclasses.replace(still, seals=(Seal(0.25, swirl=0.5),))
    a = assemble(swirling)
    assert np.allclose(a.stiffness(100.0) - a.K, 100.0 * a.H)
    assert np.allclose(a.H, -a.H.T)
    m0, m1 = modes(assemble(still), 6000 * RPM), modes(a, 6000 * RPM)
    forward = [m.zeta[m.whirl > 0][0] for m in (m0, m1)]
    backward = [m.zeta[m.whirl < 0][0] for m in (m0, m1)]
    assert forward[1] < forward[0] and backward[1] > backward[0]


# ------------------------------------------------------------------ editing
def test_the_mesh_puts_a_node_at_every_part():
    rotor = Rotor(uniform(1.0, 0.05, 2, 2), discs=(Disc(0.3),), bearings=(Bearing(0.05), Bearing(0.95)))
    z = rotor.nodes
    for p in (0.3, 0.05, 0.95, 0.5):
        assert np.min(np.abs(z - p)) < 1e-12
    assert rotor.element_sections.size == z.size - 1
    with pytest.raises(RotorError):
        assemble(rotor.with_part("disc", Disc(1.5)))


def test_editing_moves_the_parts_with_the_shaft():
    rotor = Rotor(uniform(1.0, 0.05, 4, 2), discs=(Disc(0.6),), bearings=(Bearing(0.0), Bearing(1.0)))
    split = rotor.split(1, 0.4)
    assert len(split.sections) == 5 and split.length == pytest.approx(1.0) and split.mass == pytest.approx(rotor.mass)
    longer = rotor.replace_section(1, dataclasses.replace(rotor.sections[1], length=0.5))
    assert longer.length == pytest.approx(1.25)
    assert [b.z for b in longer.bearings] == pytest.approx([0.0, 1.25]) and longer.discs[0].z == pytest.approx(0.85)
    shorter = rotor.without_section(2)  # the disc's (0.5 to 0.75): it goes too
    assert shorter.length == pytest.approx(0.75) and not shorter.discs
    assert [b.z for b in shorter.bearings] == pytest.approx([0.0, 0.75])
    assert len(rotor.without_section(3).bearings) == 1  # the far end's bearing goes with its section
    assert rotor.with_section().length == pytest.approx(1.25)
    assert rotor.section_at(0.6) == 2 and rotor.section_at(1.1) is None
    stepped = Rotor((Section(0.5, 0.05), Section(0.5, 0.08)))
    assert stepped.outer_at(0.5) == 0.08 and stepped.outer_at(0.2) == 0.05


def test_the_deflection_passes_through_the_nodes():
    rotor = pinned_ends(Rotor(uniform(1.0, 0.05, 4, 3)))
    m = modes(assemble(rotor), 0.0)
    q = m.shapes[:, 0].real
    z, x, y = deflection(rotor, q)
    assert np.interp(rotor.nodes, z, x) == pytest.approx(q[0::4], abs=1e-12)
    assert np.interp(rotor.nodes, z, y) == pytest.approx(q[2::4], abs=1e-12)


# ------------------------------------------------------------------ the trials' numbers
def test_trials():
    from vib_tutorial.gui.rotor_builder import TRIALS

    by = {t.name: t for t in TRIALS}

    def at(name: str, rpm: float | None = None, theory: Theory | None = None):
        rotor = by[name].rotor
        if theory is not None:
            rotor = dataclasses.replace(rotor, theory=theory)
        a = assemble(rotor)
        return a, modes(a, (by[name].rpm if rpm is None else rpm) * RPM)

    _, m = at("A uniform shaft, pinned")
    assert m.hz[0:6:2] == pytest.approx([101.25, 401.5, 891.3], abs=0.05)
    assert euler_pinned(1.0, 0.05) == pytest.approx([101.56, 406.2, 914.0], abs=0.05)
    _, m = at("A uniform shaft, pinned", theory=Theory.RAYLEIGH)
    assert m.hz[0:6:2] == pytest.approx([101.48, 405.0, 908.0], abs=0.05)

    _, m = at("A stubby shaft: Timoshenko")
    assert m.hz[0:6:2] == pytest.approx([776.3, 2784, 5506], abs=0.51)
    _, e = at("A stubby shaft: Timoshenko", theory=Theory.RAYLEIGH)
    assert e.hz[0:6:2] == pytest.approx([802.5, 3099, 6610], abs=0.5)
    euler = euler_pinned(0.5, 0.1)
    assert euler == pytest.approx([812.4, 3250, 7312], abs=0.5)
    assert m.hz[4] / euler[2] == pytest.approx(0.75, abs=0.01)

    name = "A disc off centre: the gyroscopic split"
    assert by[name].rotor.discs[0].mass == pytest.approx(22.2, abs=0.05)
    _, m = at(name, 0.0)
    assert m.hz[0:6:2] == pytest.approx([62.28, 265.2, 501.4], abs=0.05)
    _, m = at(name)
    assert m.hz[:6] == pytest.approx([60.83, 63.70, 261.2, 268.6, 468.2, 538.2], abs=0.05)
    assert list(np.sign(m.whirl[:6])) == [-1, 1] * 3

    name = "An overhung fan"
    assert by[name].rotor.discs[0].mass == pytest.approx(17.4, abs=0.05)
    assert by[name].rotor.discs[0].ip / by[name].rotor.discs[0].id == pytest.approx(2, abs=0.01)
    assert at(name, 0.0)[1].hz[0] == pytest.approx(79.06, abs=0.005)
    assert at(name)[1].hz[:2] == pytest.approx([71.13, 86.24], abs=0.005)
    assert at(name, 6000.0)[1].hz[:2] == pytest.approx([63.10, 92.30], abs=0.005)

    name = "Free: no bearings"
    _, m = at(name, 0.0)
    assert m.rigid == 8 and m.hz[0:4:2] == pytest.approx([228.7, 623.3], abs=0.05)
    s = Section(1.0, 0.05)
    assert 4.730041**2 / (2 * math.pi) * math.sqrt(s.ei / (STEEL.rho * s.area)) == pytest.approx(230.2, abs=0.05)
    _, m = at(name)
    assert m.hz[0] == pytest.approx(0.19, abs=0.005) and m.whirl[0] > 0.99
    assert 1.5 * 0.05**2 == pytest.approx(0.00375)

    name = "Soft bearings, stiffer across"
    _, m = at(name, 0.0)
    assert m.hz[:2] == pytest.approx([30.58, 45.44], abs=0.005) and np.abs(m.whirl[:2]).max() < 1e-6
    q = m.shapes[:, 0]
    assert np.abs(q[0::4]).max() < 1e-6 * np.abs(q[2::4]).max()  # 30.58 Hz: up and down only
    a, m = at(name)
    assert by[name].mode == 2
    assert m.hz[2:4] == pytest.approx([82.29, 149.1], abs=0.05)
    assert m.whirl[2:4] == pytest.approx([-0.20, 0.32], abs=0.005)

    name = "A stepped compressor rotor"
    discs = by[name].rotor.discs
    assert [d.mass for d in discs] == pytest.approx([15.6, 15.6, 7.1], abs=0.05)
    _, m = at(name)
    assert m.hz[:6] == pytest.approx([167.8, 171.0, 199.3, 207.4, 422.5, 442.9], abs=0.05)

    name = "Oil-film bearings"
    a, m = at(name)
    assert a.loads == pytest.approx([184.5, 184.5], abs=0.05)
    assert by[name].rotor.mass == pytest.approx(37.6, abs=0.05)
    assert m.hz[:4] == pytest.approx([25.3, 25.6, 50.9, 51.1], abs=0.05)
    assert 100 * m.zeta[0] == pytest.approx(24, abs=0.5) and 100 * m.zeta[1] == pytest.approx(22.5, abs=0.5)
    onset = scipy.optimize.brentq(lambda rpm: modes(a, rpm * RPM).zeta.min(), 3000.0, 6000.0)
    assert onset == pytest.approx(5466, abs=1)
    whip = modes(a, 6000 * RPM)
    assert 100 * whip.zeta.min() == pytest.approx(-3.0, abs=0.05) and whip.whirl[np.argmin(whip.zeta)] > 0.9

    name = "Magnetic bearings"
    a, m = at(name)
    magnet = by[name].rotor.bearings[0].magnet
    assert magnet.negative_stiffness / 1e6 == pytest.approx(0.543, abs=0.0005)
    assert magnet.current_gain * magnet.kp / 1e6 == pytest.approx(2.71, abs=0.005)
    assert m.hz[0:4:2] == pytest.approx([37.85, 143.0], abs=0.05)
    assert 100 * m.zeta[0] == pytest.approx(4.57, abs=0.005) and 100 * m.zeta[2] == pytest.approx(38.4, abs=0.05)
    assert a.n_control == 12
    slow = m.roots[np.abs(m.roots) < 100]
    assert slow.size == 4 and slow == pytest.approx(-12.7, abs=0.05)

    name = "A seal's swirl"
    seal = by[name].rotor.seals[0]
    assert seal.swirl * 6000 * RPM * seal.c / 1e6 == pytest.approx(0.63, abs=0.005)
    _, m = at(name, 0.0)
    assert 100 * m.zeta[:2] == pytest.approx([4.17, 4.17], abs=0.005)
    a, m = at(name)
    forward, backward = m.zeta[:2][m.whirl[:2] > 0][0], m.zeta[:2][m.whirl[:2] < 0][0]
    assert (100 * forward, 100 * backward) == pytest.approx((0.516, 7.74), abs=0.005)
    onset = scipy.optimize.brentq(lambda rpm: modes(a, rpm * RPM).zeta[:2].min(), 6000.0, 9000.0)
    assert onset == pytest.approx(6845, abs=1)
