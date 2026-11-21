"""`core/stepping.py`: whole steps with the part of one carried, and the frame's computing budget (#409 S-8)."""

import pytest

from vib_tutorial.core import ChainSystem, ForceController, ForceKind, ForceSettings, component_mode_synthesis
from vib_tutorial.core.cms_response import CMSResponse
from vib_tutorial.core.stepping import Budget, StepClock
from vib_tutorial.core.torsional import FAN
from vib_tutorial.core.torsional_startup import FAN_START, StartUpSimulator


def test_a_clock_carries_the_part_of_a_step_left_over():
    clock, h = StepClock(), 0.01
    taken = [clock.steps(0.3 * h, h, 100) for _ in range(10)]  # a slow playback's frames
    assert sum(taken) == 3 and taken[:3] == [0, 0, 0] and clock.owed == pytest.approx(0.0, abs=1e-12)
    assert clock.steps(5 * h, h, 100) == 5, "a whole number of steps asked for stays whole"
    assert clock.steps(500 * h, h, 100) == 100 and clock.dropped == pytest.approx(400 * h) and clock.owed == 0.0
    clock.steps(0.5 * h, h, 100)
    clock.reset()
    assert clock.owed == 0.0 and clock.steps(0.5 * h, h, 100) == 0


def test_a_budget_counts_a_dearer_cost_at_once_and_a_cheaper_one_gradually():
    b = Budget(0.012)
    assert b.allows(50) == 50, "a probe until it's measured"
    b.spent(0.012, 100)
    assert b.cost == pytest.approx(1.2e-4) and b.allows(50) == pytest.approx(100)
    b.spent(0.024, 100)
    assert b.cost == pytest.approx(2.4e-4), "dearer: at once"
    b.spent(0.0, 100)
    assert b.cost == pytest.approx(0.8 * 2.4e-4), "cheaper: a fifth of the way"
    assert b.allows(50, seconds=0.006) == pytest.approx(0.006 / b.cost)
    b.spent(1.0, 0)  # nothing done: nothing learnt
    b.forget()
    assert b.cost == 0.0


@pytest.mark.parametrize("simulator", ["torsional start-up", "CMS in time"])
def test_a_slow_playback_moves_the_simulators_that_rounded_each_frame(simulator):
    """At a third of a step a frame, rounding each call (or skipping a frame under half a step) stood both
    still; carried, they keep time."""
    if simulator == "torsional start-up":
        sim = StartUpSimulator(FAN, FAN_START)
        h = sim.h
    else:
        s = ChainSystem([1.0, 2.0, 1.5], [400.0, 300.0, 500.0], [2.0, 8.0, 1.0])
        force = ForceController(ForceSettings(kind=ForceKind.HARMONIC, freq_hz=2.0))
        sim = CMSResponse(component_mode_synthesis(s, [1], [1, 1], "craig-bampton"), force)
        force.switch_on()
        h = sim.step_size()
    for _ in range(30):
        sim.advance(h / 3)
    assert sim.t == pytest.approx(10 * h, abs=0.5 * h)


def test_every_simulator_is_a_stepper():
    """`stepping.Stepper` (#409 S-16): the clock and `advance` a page drives, the shape a Rust kernel or a
    WebAssembly front end would take; every simulator in core/ has it."""
    from vib_tutorial.core.duffing import Duffing, DuffingSimulator
    from vib_tutorial.core.gyroscope import TopSimulator
    from vib_tutorial.core.rigid_body import Body, Tumbler
    from vib_tutorial.core.rotor import RotorSystem
    from vib_tutorial.core.rotor_simulator import RotorSimulator
    from vib_tutorial.core.simulator import Simulator
    from vib_tutorial.core.sliding_top import SlidingSimulator
    from vib_tutorial.core.stepping import Stepper

    simulators = [
        Simulator(ChainSystem.uniform(2), ForceController(ForceSettings())),
        RotorSimulator(RotorSystem()),
        DuffingSimulator(Duffing()),
        StartUpSimulator(FAN, FAN_START),
        Tumbler(Body(), (0.0, 6.0, 0.06)),
        CMSResponse(component_mode_synthesis(ChainSystem.uniform(3), [1], [1, 1], "craig-bampton")),
    ]
    from vib_tutorial.core.gyroscope import Motion, Top
    from vib_tutorial.core.sliding_top import RoundTop

    simulators += [
        TopSimulator(Motion(Top(), 60.0, 0.3)),
        SlidingSimulator(RoundTop.tippe(0.02, 0.015, 0.0045, 0.8, 0.3), 60.0, 0.3),
    ]
    for sim in simulators:
        assert isinstance(sim, Stepper), type(sim).__name__
        t0 = sim.t
        sim.advance(0.01)
        assert sim.t > t0, type(sim).__name__
