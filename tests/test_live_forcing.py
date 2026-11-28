"""One mass forced live (#478 Phase 4, Forcing an oscillator): each force against its closed form, superposition by
hand, a release, a live change, a pulse's peak against `transient.peak_ratio`, the harmonics of each wave against
`periodic.response_coefficients`, and Duhamel's integral against the record."""

from __future__ import annotations

import math

import numpy as np
import pytest
from notes_quoting import quoted

from vib_tutorial.core.periodic import Forcing, Wave, response_coefficients
from vib_tutorial.core.sdof import Oscillator, impulse_response, step_response
from vib_tutorial.core.transient import Pulse, PulseKind, peak_ratio, pulse_response
from vib_tutorial.view_models import live_forcing as lf
from vib_tutorial.view_models.live_forcing import COL, LiveForcing

FRAME = 1 / 60
OSC = Oscillator(1.0, 400.0, 2.0)  # the Single oscillator's: 3.18 Hz, ζ = 0.05


def run(live: LiveForcing, seconds: float) -> None:
    for _ in range(round(seconds / FRAME)):
        live.advance(FRAME)


def recorded(live: LiveForcing) -> tuple[np.ndarray, np.ndarray]:
    n = live.record.size
    return live.record.t[:n], live.record.y[:n, COL["x"]]


@pytest.mark.parametrize(
    ("kind", "length"),
    [
        (PulseKind.IMPULSE, 0.0),
        (PulseKind.STEP, 0.0),
        (PulseKind.RAMP, 0.5),
        (PulseKind.RECTANGLE, 0.05),
        (PulseKind.HALF_SINE, 0.157),
    ],
)
def test_each_force_applied_as_it_runs_is_its_closed_form(kind, length):
    live = LiveForcing(OSC)
    run(live, 0.1)  # at rest, running: applied from now
    push = live.apply(Pulse(kind, 1.0, length))
    run(live, 2.0)
    t, x = recorded(live)
    ref = pulse_response(OSC, Pulse(kind, 1.0, length), t - push.start)
    assert np.max(np.abs(x - ref)) <= quoted("forcing", "0.05 %") / 100 * np.max(np.abs(ref))  # as the notes say


def test_a_second_hit_lands_on_the_first_ones_motion():
    live = LiveForcing(OSC)
    first = live.apply(Pulse(PulseKind.IMPULSE, 0.02))
    run(live, 0.4)
    second = live.apply(Pulse(PulseKind.IMPULSE, -0.01))
    run(live, 1.0)
    t, x = recorded(live)
    both = 0.02 * impulse_response(OSC, t - first.start) - 0.01 * impulse_response(OSC, t - second.start)
    assert np.max(np.abs(x - both)) <= 1e-9 and len(live.hits) == 2
    assert not second.quiet  # it landed on a moving mass


def test_a_push_held_then_released_rings_back_to_zero():
    live = LiveForcing(OSC)
    push = live.apply(Pulse(PulseKind.STEP, 2.0))
    assert live.holding()
    run(live, 0.75)
    live.release()
    assert not live.holding() and push.end == live.t
    run(live, 2.0)
    t, x = recorded(live)
    ref = 2.0 * (step_response(OSC, t - push.start) - step_response(OSC, t - push.end))
    assert np.max(np.abs(x - ref)) <= 1e-9 + 5e-4 * np.max(np.abs(ref))


def test_a_live_change_keeps_place_and_momentum():
    live = LiveForcing(OSC)
    live.apply(Pulse(PulseKind.IMPULSE, 0.05))
    run(live, 0.1)
    x, v = live.x, live.v
    live.set_params(Oscillator(1.0, 900.0, 2.0))  # stiffer: x and v carry on
    assert (live.x, live.v) == (x, v) and live.since == live.t
    live.set_params(Oscillator(2.0, 900.0, 2.0))  # twice the mass, captured at rest
    assert live.v == pytest.approx(v / 2)


def test_held_the_clock_stops_and_let_go_it_rings():
    live = LiveForcing(OSC)
    live.hold(0.01)
    assert live.advance(1.0) == 0 and live.t == 0
    live.let_go()
    run(live, 1.0)
    _, x = recorded(live)
    assert x[0] == pytest.approx(0.01, rel=1e-3) and live.since == 0.0


@pytest.mark.parametrize("kind", [PulseKind.RECTANGLE, PulseKind.HALF_SINE, PulseKind.RAMP])
@pytest.mark.parametrize("ratio", [0.2, 0.5, 1.3])
def test_a_pulses_peak_is_the_pulse_spectrums(kind, ratio):
    live = LiveForcing(OSC)
    push = live.apply(Pulse(kind, 1.0, ratio * OSC.period))
    run(live, ratio * OSC.period + 0.5 * OSC.period)
    assert lf.pulse_peak(live, push) is None  # not yet a period after it ended
    run(live, 1.0 * OSC.period)
    peak = lf.pulse_peak(live, push)
    assert peak is not None and peak.ratio == pytest.approx(ratio) and peak.kind is kind
    assert peak.peak == pytest.approx(peak_ratio(OSC.zeta, kind, np.array([ratio]))[0], rel=2e-3)


def test_a_hit_and_a_step_land_their_peaks_too():
    live = LiveForcing(OSC)
    hit = live.apply(Pulse(PulseKind.IMPULSE, 0.01))
    run(live, 1.2 * OSC.period)
    peak = lf.pulse_peak(live, hit)
    z = OSC.zeta
    first = math.exp(-z / math.sqrt(1 - z * z) * math.atan(math.sqrt(1 - z * z) / z))  # h's first peak, kx/Iω_n
    assert peak is not None and peak.ratio == 0 and peak.peak == pytest.approx(first, rel=0.02)
    step = LiveForcing(OSC)
    push = step.apply(Pulse(PulseKind.STEP, 1.0))
    run(step, 1.2 * OSC.period)
    assert lf.pulse_peak(step, push).peak == pytest.approx(1 + math.exp(-z * math.pi / math.sqrt(1 - z * z)), rel=1e-3)


def test_a_pulse_on_a_moving_mass_or_beside_another_force_lands_nothing():
    live = LiveForcing(OSC)
    live.apply(Pulse(PulseKind.IMPULSE, 0.05))
    run(live, 0.1)
    moving = live.apply(Pulse(PulseKind.RECTANGLE, 1.0, 0.05))
    run(live, 1.0)
    assert not moving.quiet and lf.pulse_peak(live, moving) is None
    other = LiveForcing(OSC)
    first = other.apply(Pulse(PulseKind.RECTANGLE, 1.0, 0.2))
    other.apply(Pulse(PulseKind.RECTANGLE, 1.0, 0.05))  # while the first acts
    run(other, 1.0)
    assert not first.alone and lf.pulse_peak(other, first) is None


@pytest.mark.parametrize("wave", list(Wave))
def test_each_waves_harmonics_are_the_series(wave):
    forcing = Forcing(wave, 1.0, 1.2)
    live = LiveForcing(OSC)
    live.set_wave(forcing)
    run(live, 0.5)
    assert lf.harmonics(live, 7) is None or not lf.harmonics(live, 7).steady  # still ringing from the start
    run(live, 9.5)
    h = lf.harmonics(live, 7)
    X = response_coefficients(OSC, forcing, 7)
    assert h is not None and h.steady and h.freq_hz == 1.2
    assert np.max(np.abs(h.response - X)) <= 5e-4 * np.max(np.abs(X))
    assert np.max(np.abs(h.force - forcing.coefficients(7))) <= 5e-3 * np.max(np.abs(forcing.coefficients(7)))


def test_hammer_blows_land_once_a_period():
    live = LiveForcing(OSC)
    live.set_wave(Forcing(Wave.IMPULSES, 0.01, 2.0))
    run(live, 2.0)
    times = np.array([t for t, _ in live.hits])
    assert times[:5] == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0], abs=1e-9)  # each at its own instant, not a step's


def test_duhamels_integral_is_the_recorded_motion():
    live = LiveForcing(OSC)
    live.apply(Pulse(PulseKind.HALF_SINE, 2.0, 0.08))
    run(live, 0.5)
    live.apply(Pulse(PulseKind.IMPULSE, 0.05))
    run(live, 0.3)
    c = lf.convolution(live)
    assert c is not None and c.since == 0 and len(c.hits) == 1
    assert c.area == pytest.approx(c.x, rel=2e-3, abs=1e-7)
    assert np.allclose(c.product, c.force * c.h) and c.tau[-1] == pytest.approx(live.t)
    earlier = lf.convolution(live, 0.3)  # at an earlier instant: the record's motion then
    assert earlier is not None and earlier.area == pytest.approx(earlier.x, rel=2e-3, abs=1e-7)
    live.hold(0.003)
    live.let_go()  # plucked: the integral starts again from here
    run(live, 0.2)
    assert lf.convolution(live).since == pytest.approx(0.8, abs=0.02)
