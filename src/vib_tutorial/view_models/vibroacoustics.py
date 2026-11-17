"""Vibro-acoustics' view model, its slow part: what three of its tabs work out, from the panel's settings
(`gui/vibroacoustics.py` draws it).

A change to the panel (its size, thickness, material, loss factor) has the Radiation, Driven panel and SEA tabs work
out 0.15–0.3 s each, so the page runs these on a `Recompute` worker; each is a pure function of plain values:

- `radiation(plate, m, n, f_top, modes)`: the picked mode's σ against frequency, and every mode up to `f_top`
  with σ at its own resonance (`modes`, when the page already has them for this panel: they're kept, as the Driven
  panel needs them too).
- `driven(plate, loss, at, f_top, modes)`: the panel driven by 1 N at `at`, by its modes (no more than MAX_MODES:
  the top frequency comes down if there are more).
- `sea(net)`: SEA's network in third octaves, and the room's level from each plate alone.

Working on it: keep it free of Qt (see the package's docstring). The page's drawing from these is quick; the
tests drive it through the page (`tests/test_gui_smoke.py`'s Vibro-acoustics test).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.beams_plates import Plate
from ..core.radiation import ModeSet, PointDrive, modal_efficiency, mode_set, point_drive, third_octaves
from ..core.sea import Network, NetworkResponse, network
from .plot import Curve

F_LOW = 20.0  # Hz, the plots' lowest frequency
MAX_MODES = 600  # modes the driven panel sums (its top frequency comes down to keep it to these)
CURVE_POINTS = 120  # frequencies σ_mn(f) of one mode is worked out at
DRIVE_POINTS = 700  # frequencies the driven panel is worked out at
SEA_LOW, SEA_TOP = 63.0, 10e3  # the SEA tab's third-octave bands


def modes_for(plate: Plate, f_top: float, modes: ModeSet | None = None) -> ModeSet:
    """Every mode up to `f_top`, each with σ at its own resonance (`modes` as they are, if given)."""
    return modes if modes is not None else mode_set(plate, f_top, count=4 * MAX_MODES)


@dataclass(frozen=True)
class Radiation:
    plate: Plate  # the panel it's for: the page keeps `modes` only while it shows this one
    one: Curve  # the picked mode's σ against frequency
    modes: ModeSet


def radiation(plate: Plate, m: int, n: int, f_top: float, modes: ModeSet | None = None) -> Radiation:
    f = np.geomspace(F_LOW, f_top, CURVE_POINTS)
    one = np.array([modal_efficiency(plate, np.array([m]), np.array([n]), fi)[0, 0] for fi in f])
    return Radiation(plate, Curve(f, one), modes_for(plate, f_top, modes))


@dataclass(frozen=True)
class Driven:
    plate: Plate
    drive: PointDrive
    top: float  # Hz: the drive's top frequency, below f_top if the modes ran out
    modes: ModeSet


def driven(plate: Plate, loss: float, at: tuple[float, float], f_top: float, modes: ModeSet | None = None) -> Driven:
    modes = modes_for(plate, f_top, modes)
    top = f_top
    if modes.freq.size > MAX_MODES:
        top = min(top, float(modes.freq[MAX_MODES - 1]) / 1.5)
    sub = ModeSet(*(arr[modes.freq <= 1.5 * top] for arr in (modes.m, modes.n, modes.freq, modes.sigma)))
    f = np.geomspace(F_LOW, top, DRIVE_POINTS)
    return Driven(plate, point_drive(plate, loss, at, f, modes=sub), top, modes)


@dataclass(frozen=True)
class Sea:
    response: NetworkResponse
    alone: tuple[NetworkResponse, NetworkResponse]  # the plate alone radiating, then the second plate alone


def sea(net: Network) -> Sea:
    f = third_octaves(SEA_LOW, SEA_TOP)
    return Sea(network(net, f), (network(net, f, silent=1), network(net, f, silent=0)))
