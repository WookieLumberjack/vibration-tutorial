"""Forcing an oscillator's notes' figures: the pulses and the steps that build them (`transient_figures`), and a
square wave built from its harmonics and a coefficient found by an average (`periodic_figures`), the two halves
of the merged page (#478 Phase 4). Qt-free, as every figure module (`tests/test_qt_free.py`)."""

from __future__ import annotations

from . import periodic_figures, transient_figures

FIGURES = {**transient_figures.FIGURES, **periodic_figures.FIGURES}
