"""`gui/rotor_readouts.py`: the rotor page's readout texts, without the page."""

from __future__ import annotations

import math

from vib_tutorial.core.rotor import Bearing, Crack, RotorSystem
from vib_tutorial.core.rotor_stability import Onset
from vib_tutorial.gui.rotor_readouts import fmt_len, instability_causes, onset_text, shaft_note


def test_fmt_len_picks_its_unit():
    assert (fmt_len(0.0123), fmt_len(54e-6), fmt_len(3.2e-9)) == ("12.3 mm", "54 µm", "3.2 nm")


def test_shaft_note_says_how_the_stiffness_is_worked_out():
    s = RotorSystem()
    assert " = 48EI/L³" in shaft_note(s) and "Static sag" not in shaft_note(s)
    off = s.with_(position=0.3)
    assert "(48EI/L³ = " in shaft_note(off) and "at midspan)" in shaft_note(off)
    assert "overhung by c = " in shaft_note(s.with_(position=1.2))
    heavy = s.with_(gravity=9.81, crack=Crack(depth=0.4), asymmetry=0.1)
    note = shaft_note(heavy)
    assert "Static sag at rest: disc " in note and "Crack 40% deep" in note and "Asymmetric: EI = " in note


def test_what_could_make_it_unstable():
    assert instability_causes(RotorSystem()).startswith("No cross-coupling")
    s = RotorSystem(bearing_a=Bearing(kxy=2e4), internal_damping=5.0)
    text = instability_causes(s)
    assert text.startswith("With k<sub>xy</sub> = 20,000 N/m at A") and "internal damping c<sub>i</sub> = 5" in text


def test_onset_text():
    assert onset_text(None, 6000.0) == "stable up to 6,000 rpm."
    grows = Onset(omega=1000 * math.pi / 30, freq_hz=10.0, whirl=1.0)
    assert onset_text(grows, 6000.0).startswith("<b>unstable above 1,000 rpm</b>")


def test_a_whirl_of_2_percent_of_the_span_pauses_the_page():
    from notes_quoting import quoted

    from vib_tutorial.gui.rotor import RUB

    assert 100 * RUB == quoted("rotor_notes", "2 %")
