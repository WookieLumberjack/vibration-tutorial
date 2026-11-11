"""Statistical energy analysis: the two-oscillator power flow (exact against the formula), the corner junction's
energy balance and reciprocity, and the network's power balance."""

import math

import numpy as np
import pytest

from vib_tutorial.core.beams_plates import TWO_PI, Plate
from vib_tutorial.core.radiation import radiation_loss_factor, third_octaves
from vib_tutorial.core.sea import (
    Network,
    Oscillators,
    coupling_beta,
    junction_transmission,
    junction_waves,
    loss_factors,
    network,
    oscillator_pair,
    plate_coupling,
    plate_modal_density,
    room_modal_density,
)

P1 = Plate()  # 600 × 400 × 3 mm steel
P2 = Plate(0.5, 0.4, 0.002)


@pytest.mark.parametrize(
    "osc",
    [
        Oscillators(),
        Oscillators(f2=100.0, zeta1=0.01, zeta2=0.03, coupling=0.1),
        Oscillators(f2=160.0, zeta1=0.05, zeta2=0.002, coupling=0.2, m2=0.5),
        Oscillators(f2=70.0, zeta1=0.001, zeta2=0.001, coupling=0.3, m2=3.0),
    ],
)
def test_two_oscillators_exchange_power_in_proportion_to_their_energies(osc):
    beta = coupling_beta(osc)
    for p1, p2 in ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.2, 3.0)):
        r = oscillator_pair(osc, p1, p2)
        assert r.flow == pytest.approx(beta * (r.E1 - r.E2), rel=1e-8)
        tol = 1e-9 * (p1 + p2)
        assert p1 - r.flow == pytest.approx(r.lost1, abs=tol)  # each oscillator's balance
        assert p2 + r.flow == pytest.approx(r.lost2, abs=tol)


def test_the_flow_runs_from_the_more_energetic_oscillator():
    osc = Oscillators()
    r = oscillator_pair(osc, 0.3, 1.0)
    assert (r.flow > 0) == (r.E1 > r.E2)
    # equal energies, no flow: drive in the ratio that makes them so
    a, b = oscillator_pair(osc, 1.0, 0.0), oscillator_pair(osc, 0.0, 1.0)
    s = (a.E1 - a.E2) / (b.E2 - b.E1)  # E1 − E2 is linear in the drives
    assert oscillator_pair(osc, 1.0, s).flow == pytest.approx(0.0, abs=1e-12)


def test_tuned_oscillators_couple_best():
    betas = [coupling_beta(Oscillators(f2=f)) for f in (80.0, 95.0, 100.0, 105.0, 120.0)]
    assert int(np.argmax(betas)) == 2


def test_a_corner_conserves_energy():
    for theta in np.linspace(0.0, 1.55, 9):
        for a, b in ((P1, P2), (P2, P1)):
            tau, refl = junction_waves(a, b, 1000.0, float(theta))
            assert tau + refl == pytest.approx(1.0, abs=1e-12)


def test_identical_plates_at_a_corner():
    """Half gets through at normal incidence; a third of a diffuse field."""
    same = Plate(0.5, 0.4, 0.003)
    assert junction_waves(P1, same, 500.0, 0.0)[0] == pytest.approx(0.5)
    assert junction_transmission(P1, same, 500.0) == pytest.approx(1 / 3, rel=1e-6)


def test_coupling_loss_factors_are_reciprocal():
    for f in (125.0, 1000.0, 4000.0):
        left = plate_modal_density(P1) * plate_coupling(P1, P2, 0.4, f)
        right = plate_modal_density(P2) * plate_coupling(P2, P1, 0.4, f)
        assert left == pytest.approx(right, rel=1e-6)


def test_modal_densities():
    assert plate_modal_density(P1) == pytest.approx(P1.area / 2 / P1.scale)
    assert float(room_modal_density(50.0, 1000.0)) == pytest.approx(4 * math.pi * 1e6 * 50 / 343**3)


def test_network_balances_power():
    net = Network(P1, P2, 0.01, 0.01, 0.4)
    centres = third_octaves(100, 8000)
    r = network(net, centres)
    w = TWO_PI * centres
    for i in range(3):
        lost = w * r.eta[:, i, i] * r.energy[:, i]
        into = r.input_power if i == 0 else 0.0
        assert np.allclose(into - lost - r.flow[:, i, :].sum(axis=1), 0.0, atol=1e-12 * r.input_power[0])
    # all the input is lost somewhere
    total = sum(w * r.eta[:, i, i] * r.energy[:, i] for i in range(3))
    assert total == pytest.approx(r.input_power, rel=1e-9)
    eta, n = loss_factors(net, 1000.0)
    assert eta[0, 2] == pytest.approx(radiation_loss_factor(P1, 1000.0))
    assert n[0] * eta[0, 2] == pytest.approx(n[2] * eta[2, 0])


def test_strong_coupling_shares_modal_energy():
    """With little damping on plate 2, the plates' energies per mode come close (equipartition); with a lot,
    plate 2 holds far less."""
    centres = np.array([1000.0])
    light = network(Network(P1, P2, 0.01, 1e-4, 0.4), centres)
    heavy = network(Network(P1, P2, 0.01, 0.1, 0.4), centres)
    share = [(r.energy[0, 1] / r.density[0, 1]) / (r.energy[0, 0] / r.density[0, 0]) for r in (light, heavy)]
    assert 0.9 < share[0] < 1.0
    assert share[1] < 0.1
