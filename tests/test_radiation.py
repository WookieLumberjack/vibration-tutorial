"""Radiation from a baffled panel: Wallace's integral and its limits, the band averages against Maidanik, the
point-driven panel's energy balance, and the transmission loss's mass law and coincidence dip."""

import math

import numpy as np
import pytest

from vib_tutorial.core.beams_plates import Material, Plate
from vib_tutorial.core.radiation import (
    AIR,
    _factor,
    band_edges,
    bending_wavenumber,
    classify,
    coincidence_dip,
    critical_frequency,
    low_frequency_efficiency,
    maidanik,
    mass_law,
    modal_efficiency,
    mode_set,
    point_drive,
    radiation_loss_factor,
    resonant_average,
    third_octaves,
    transmission_loss,
)

PLATE = Plate()  # 600 × 400 × 3 mm steel


def test_bending_wave_meets_sound_at_the_critical_frequency():
    fc = critical_frequency(PLATE)
    assert float(bending_wavenumber(PLATE, fc)) == pytest.approx(2 * math.pi * fc / AIR.c)
    assert critical_frequency(Plate(thickness=0.006)) == pytest.approx(fc / 2)  # ∝ 1/h


def test_wallace_low_frequency_limit():
    for m, n in ((1, 1), (3, 1), (1, 3)):
        sigma = modal_efficiency(PLATE, np.array([m]), np.array([n]), 5.0)[0, 0]
        assert sigma == pytest.approx(float(low_frequency_efficiency(PLATE, m, n, 5.0)), rel=2e-3)


def test_even_indices_cancel_to_higher_powers_of_k():
    """σ ∝ k² for odd–odd, k⁴ with one even index, k⁶ with two."""
    f = np.array([2.0, 4.0])
    for (m, n), power in (((1, 1), 2), ((2, 1), 4), ((1, 2), 4), ((2, 2), 6)):
        s = [modal_efficiency(PLATE, np.array([m]), np.array([n]), fi)[0, 0] for fi in f]
        assert math.log2(s[1] / s[0]) == pytest.approx(power, abs=0.02)


def test_well_above_coincidence_every_mode_radiates_fully():
    sigma = modal_efficiency(PLATE, np.arange(1, 4), np.arange(1, 3), 20e3)
    assert sigma == pytest.approx(np.ones_like(sigma), abs=0.01)


def test_the_integrand_is_continuous_through_its_removable_poles():
    for m in (1, 2, 3, 4):
        alpha = m * math.pi + np.array([-1e-6, 0.0, 1e-6])
        values = _factor(alpha, np.array([m]))[0]
        assert values == pytest.approx(np.full(3, values[1]), rel=1e-5)


def test_modes_sorted_by_wallaces_regions():
    assert classify(PLATE, 1, 1, 66.6).startswith("a corner")
    assert classify(PLATE, 1, 3, 435.4).startswith("an edge")
    assert classify(PLATE, 1, 1, 2000.0).startswith("a surface")


def test_resonant_average_follows_maidanik():
    """Where each third-octave band holds several modes, their mean σ is within 2.5 dB of Maidanik's."""
    fc = critical_frequency(PLATE)
    modes = mode_set(PLATE, 2.5 * fc)
    centres = third_octaves(500, 8000)
    avg = resonant_average(modes, centres)
    lo, hi = band_edges(centres)
    for c, a, l_, h_ in zip(centres, avg, lo, hi):
        count = int(np.count_nonzero((modes.freq >= l_) & (modes.freq < h_)))
        if count >= 4 and abs(math.log10(c / fc)) > 0.06:  # not the coincidence band, whose peak is sharp
            assert abs(10 * math.log10(a / float(maidanik(PLATE, c)))) < 2.5, c


def test_point_drive_balances_energy():
    """The power in equals each mode's hysteretic loss, ½ωη(M/4)ω_mn²|q|², summed."""
    f = np.geomspace(30, 3000, 200)
    modes = mode_set(PLATE, 4500)
    d = point_drive(PLATE, 0.02, (0.21, 0.13), f, modes=modes)
    phi = np.sin(modes.m * math.pi * 0.21 / PLATE.a) * np.sin(modes.n * math.pi * 0.13 / PLATE.b)
    mass = PLATE.surface_mass * PLATE.a * PLATE.b
    w, wn = 2 * math.pi * f[:, None], 2 * math.pi * modes.freq[None, :]
    q = phi / (0.25 * mass * (wn**2 * (1 + 0.02j) - w**2))
    lost = 0.5 * w[:, 0] * 0.02 * 0.25 * mass * np.sum(wn**2 * np.abs(q) ** 2, axis=1)
    assert d.input_power == pytest.approx(lost, rel=1e-9)


def test_band_averages_tend_to_the_infinite_plate():
    fc = critical_frequency(PLATE)
    f = np.geomspace(20, 2.5 * fc, 700)
    d = point_drive(PLATE, 0.01, (0.21, 0.13), f)
    centres = third_octaves(500, 8000)
    lo, hi = band_edges(centres)
    ratios = [d.input_power[(f >= a) & (f < b)].mean() / d.infinite_input for a, b in zip(lo, hi)]
    level = 10 * np.log10(ratios)  # dB from the infinite plate
    assert np.all(np.abs(level) < 4.0)  # a band with few modes scatters
    assert np.all(np.abs(level[centres >= 2000]) < 1.0)  # a band with many doesn't
    assert abs(float(np.mean(level))) < 0.5
    # and the radiated power sits below the input as the radiation loss factor says, within 1.5 dB
    sel = (f >= lo[3]) & (f < hi[3])  # the 1 kHz band
    gap = 10 * math.log10(d.input_power[sel].mean() / d.radiated_power[sel].mean())
    eta_rad = radiation_loss_factor(PLATE, 1000.0)
    assert gap == pytest.approx(10 * math.log10(1 + 0.01 / eta_rad), abs=1.5)


def test_mass_law():
    lo = transmission_loss(PLATE, 0.01, np.array([250.0, 500.0]))
    assert lo[1] - lo[0] == pytest.approx(6.0, abs=0.2)  # 6 dB an octave
    heavy = transmission_loss(Plate(material=Material(rho=2 * 7850.0)), 0.01, 250.0)[0]
    assert heavy - lo[0] == pytest.approx(6.0, abs=0.2)  # and per doubling of mass
    assert float(mass_law(PLATE, 500.0)) - lo[1] == pytest.approx(5.2, abs=0.3)  # field incidence: ~5 dB less


def test_coincidence_dip():
    fc = critical_frequency(PLATE)
    f_dip, tl_dip = coincidence_dip(PLATE, 0.01)
    assert fc < f_dip < 1.2 * fc
    around = transmission_loss(PLATE, 0.01, f_dip * np.array([0.97, 1.03]))
    assert np.all(around > tl_dip)
    # f_c m″ is the material's: the same depth for any thickness of glass
    glass = Material(70e9, 0.22, 2500.0)
    depths = [coincidence_dip(Plate(1.2, 1.0, h / 1000, glass), 0.01)[1] for h in (4, 10)]
    assert depths[0] == pytest.approx(depths[1], abs=0.01)
    # damping fills it in, and lifts the TL above it by 10 log of the ratio
    assert coincidence_dip(PLATE, 0.1)[1] > tl_dip + 10
    above = transmission_loss(PLATE, 0.01, 3 * fc)[0]
    damped = transmission_loss(PLATE, 0.1, 3 * fc)[0]
    assert damped - above == pytest.approx(10.0, abs=1.0)
