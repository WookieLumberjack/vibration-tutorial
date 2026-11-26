"""The Jeffcott rotor page's readout texts, plain functions of the rotor and its analysis: no widgets, so they can
be checked without building the page (#218 C2: split from `rotor.py`).

`shaft_note` is the note under the Shaft and disk box (EI, the shaft's stiffness at the disk and how it is
worked out, the asymmetry, the crack, the shaft's mass and the static sag); `instability_causes` and
`onset_text` make the Stability tab's note; `fmt_len` writes a length in mm, µm or nm.

Working on it: `tests/test_rotor_readouts.py` checks them; the rotor page's GUI tests check the notes as shown.
"""

from __future__ import annotations

import math

from ..core.rotor import MODE_LIMIT, XD, YA, YB, YD, RotorSystem
from ..core.rotor_simulator import RPM
from ..core.rotor_stability import Onset
from ..view_models.text import fmt
from ..view_models.text import number as sig
from .rotor_notes import whirl_name


def fmt_len(meters: float) -> str:
    """A length in mm, µm or nm, three figures."""
    return fmt(meters, "m", largest="m", smallest="n")


def shaft_note(system: RotorSystem) -> str:
    """The note under the Shaft and disk box."""
    ks = 48.0 * system.ei / system.length**3
    if system.overhung:
        formula = f" = 3EI/(c²(L + c)), overhung by c = {sig(system.overhang * 1e3, 4)} mm"
    elif abs(system.position - 0.5) < 1e-9:
        formula = " = 48EI/L³"
    else:
        formula = f" (48EI/L³ = {ks:,.0f} N/m at midspan)"
    return (
        f"EI = {sig(system.ei, 4)} N·m²; shaft stiffness at the disk (bearings rigid) "
        f"{system.disc_stiffness:,.0f} N/m"
        + formula
        + _asymmetry(system)
        + _crack(system)
        + _shaft_mass(system)
        + _sag(system)
    )


def _asymmetry(system: RotorSystem) -> str:
    if not system.asymmetry:
        return ""
    d = system.asymmetry
    return (
        f"<br>Asymmetric: EI = {sig(system.ei * (1 + d), 4)} N·m² bending toward the stiff axis "
        f"({math.degrees(system.asymmetry_angle):.0f}° from the mark), {sig(system.ei * (1 - d), 4)} across it"
    )


def _crack(system: RotorSystem) -> str:
    crack = system.crack
    if crack is None:
        return ""
    c_open, c_across = system.crack_compliances
    soft, stiff = (system.axis_system(k).stiffness(0.0) for k in (False, True))
    loss = [1.0 - k[XD, XD] / system.with_(crack=None).stiffness(0.0)[XD, XD] for k in (soft, stiff)]
    state = (
        "held open"
        if not crack.breathing
        else "breathing, opened by the weight"
        if system.crack_opening
        else "breathing, but shut: nothing opens it without gravity"
    )
    return (
        f"<br>Crack {crack.depth:.0%} deep at z = {sig(system.crack_z * 1e3, 4)} mm ({state}): compliance "
        f"{sig(c_open)} rad/(N·m) opening it, {sig(c_across)} across; on average over a turn it takes "
        f"{loss[0]:.2%} and {loss[1]:.2%} off the disk's stiffness along and across its mouth"
    )


def _shaft_mass(system: RotorSystem) -> str:
    if not system.density:
        return ""
    f = system.fixed_interface_freqs
    held = (
        f"its lowest {f.size} mode{'s' if f.size != 1 else ''} with the stations held "
        f"({', '.join(f'{x:,.0f}' for x in f)} Hz)"
        if f.size
        else f"no modes with the stations held below {MODE_LIMIT:,.0f} Hz"
    )
    return (
        f"<br>Shaft {sig(system.shaft_mass)} kg ({system.shaft_mass / system.disc_mass:.0%} of the disk), "
        f"{system.nodes.size - 1} elements, reduced to the stations and {held}"
    )


def _sag(system: RotorSystem) -> str:
    if not system.gravity:
        return ""
    q = system.static_deflection(0.0)
    return f"<br>Static sag at rest: disk {fmt_len(-q[YD])}, journals {fmt_len(-q[YA])} (A) and {fmt_len(-q[YB])} (B)"


def instability_causes(s: RotorSystem) -> str:
    """What could make this rotor unstable (the Stability note's first sentence), or that nothing could."""
    bearings = (("A", s.bearing_a), ("B", s.bearing_b))
    coupled = [f"k<sub>xy</sub> = {b.kxy:,.0f} N/m at {n}" for n, b in bearings if b.kxy and b.film is None]
    coupled += [f"swirl λ = {sig(b.swirl, 6)} at {n}" for n, b in bearings if b.swirling]
    if s.films:
        coupled.append("oil-film bearings (their oil swirling at half the speed)")
    if s.magnets:
        coupled.append("magnetic bearings (their controllers' lag)")
    if s.internal_damping:
        coupled.append(f"internal damping c<sub>i</sub> = {sig(s.internal_damping, 6)} N·s/m")
    if s.asymmetry:
        coupled.append(f"the shaft's asymmetry Δ = {sig(s.asymmetry)}")
    if s.crack_opening and s.crack is not None:
        coupled.append(f"a {'breathing' if s.crack.breathing else 'open'} crack {s.crack.depth:.0%} deep")
    return (
        "With " + " and ".join(coupled) + ": "
        if coupled
        else "No cross-coupling, internal damping, asymmetry or crack: every mode stays damped, however fast it "
        "spins. Set c<sub>i</sub>, k<sub>xy</sub>, λ or Δ to see one go unstable. "
    )


def onset_text(o: Onset | None, top_rpm: float) -> str:
    """The Stability readout's verdict: stable up to the top speed, or where the rotor goes unstable."""
    if o is None:
        return f"stable up to {top_rpm:,.0f} rpm."
    whirl = whirl_name(o.whirl)
    if o.omega == 0.0:
        return f"<b>unstable even at rest</b>: the {whirl} whirl at {sig(o.freq_hz)} Hz grows."
    if math.isfinite(o.end):
        return (
            f"<b>unstable from {o.omega * RPM:,.0f} to {o.end * RPM:,.0f} rpm</b>, where the {whirl} whirl "
            f"at {sig(o.freq_hz)} Hz ({o.order:.2f}×) grows; stable again above."
        )
    return (
        f"<b>unstable above {o.omega * RPM:,.0f} rpm</b>, where the {whirl} whirl at "
        f"{sig(o.freq_hz)} Hz ({o.order:.2f}×) starts to grow."
    )
