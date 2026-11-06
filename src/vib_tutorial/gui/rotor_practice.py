"""The rotor page's practice problems (core.practice), and the setup they load on the page.

The page's *Things to try* are presets, whole rotors, so a problem's setup is `RotorSetup`: a preset with the
few parameters it changes, the speeds, or (as the page reads itself back for a spec) the page's whole rotor.
Each question's numbers are printed round, and its answer is worked from them as printed.
"""

from __future__ import annotations

import cmath
import functools
import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ..core.practice import Problem, Question, Requirement, at_least, at_most, kept, number, pick, rounded
from ..core.rotor import Bearing, RotorSystem
from ..core.rotor_modes import whirl_modes
from ..core.rotor_presets import RotorPreset, rotor_presets
from ..core.rotor_simulator import RPM
from .links import page_link

CLASSIC = "Classic Jeffcott rotor"
E_STEEL = 200e9  # Pa, the page's default Young's modulus


@functools.cache
def _presets() -> dict[str, RotorPreset]:
    return {p.name: p for p in rotor_presets()}


@dataclass(frozen=True)
class RotorSetup:
    """A preset, and what a problem changes on it (None: the preset's own). The bearings' fields apply to both
    bearings (`ky` None: as `kx`, `cx` both ways). `rotor`, when given, is the whole rotor instead."""

    preset: str = CLASSIC
    length: float | None = None  # m
    diameter: float | None = None  # m
    position: float | None = None  # a/L
    disc_mass: float | None = None  # kg
    internal_damping: float | None = None  # N·s/m
    kx: float | None = None  # N/m
    ky: float | None = None  # N/m
    cx: float | None = None  # N·s/m
    target_rpm: float | None = None
    top_rpm: float | None = None
    rotor: RotorSystem | None = None

    @property
    def base(self) -> RotorPreset:
        return _presets().get(self.preset, _presets()[CLASSIC])

    def system(self) -> RotorSystem:
        if self.rotor is not None:
            return self.rotor
        s = self.base.system
        changes = {
            name: value
            for name in ("length", "diameter", "position", "disc_mass", "internal_damping")
            if (value := getattr(self, name)) is not None
        }
        if self.kx is not None or self.cx is not None:

            def bearing(b: Bearing) -> Bearing:
                kx = b.kx if self.kx is None else self.kx
                ky = (b.ky if self.kx is None else kx) if self.ky is None else self.ky
                cx = b.cx if self.cx is None else self.cx
                return replace(b, kx=kx, ky=ky, cx=cx, cy=b.cy if self.cx is None else cx)

            changes.update(bearing_a=bearing(s.bearing_a), bearing_b=bearing(s.bearing_b))
        return s.with_(**changes)


def critical_rpm(system: RotorSystem) -> float:
    """A disc at midspan on identical, isotropic bearings: its critical speed is its first natural frequency,
    whatever the speed (the gyroscopic moment never enters the first mode)."""
    return float(whirl_modes(system, 0.0).freq_hz[0]) * 60.0


# ------------------------------------------------------------------ numbers
def _critical(rng: np.random.Generator) -> Question:
    d, length = pick(rng, (8, 10, 12)), pick(rng, (0.4, 0.5, 0.6))
    kb, m = pick(rng, (5e4, 1e5, 2e5)), pick(rng, (1, 2, 5))
    ei = E_STEEL * math.pi * (d / 1000) ** 4 / 64
    ks = 48 * ei / length**3
    keq = 1 / (1 / ks + 1 / (2 * kb))
    rpm = math.sqrt(keq / m) * RPM
    setup: dict[str, Any] = dict(length=length, diameter=d / 1000, disc_mass=m, kx=kb)
    model = critical_rpm(RotorSetup(**setup).system())
    return Question(
        f"A {m:g} kg disc at the middle of a {length:g} m steel shaft (E = 200 GPa) {d:g} mm across, its bearings on "
        f"supports of {number(kb)} N/m each. The shaft's mass and the journals' aside, what is its critical speed?",
        rpm,
        f"EI = Eπd⁴/64 = {number(ei)} N·m², so the shaft at midspan is k<sub>s</sub> = 48EI/L³ = {number(ks)} N/m, in "
        f"series with the two supports side by side, 2k<sub>b</sub> = {number(2 * kb)} N/m: k<sub>eq</sub> = "
        f"{number(keq)} N/m. ω<sub>c</sub> = √(k<sub>eq</sub>/m) = {number(math.sqrt(keq / m))} rad/s, "
        f"{number(rpm)} rpm. (The page's model, with its 0.1 kg journals, puts it at {number(model)} rpm.)",
        unit="rpm",
        setup=setup,
    )


def _whirl_radius(rng: np.random.Generator) -> Question:
    e, r, zeta = pick(rng, (10, 20, 50)), pick(rng, (0.5, 0.8, 1.2, 2, 3)), pick(rng, (0.02, 0.05, 0.1))
    x = e * r * r / math.sqrt((1 - r * r) ** 2 + (2 * zeta * r) ** 2)
    return Question(
        f"A disc's centre of mass is {e:g} µm off the shaft's axis, its mode's ζ = {zeta:g}. Running at {r:g} times the "
        "critical speed, how far does the disc whirl?",
        x,
        f"|X| = e r²/√((1 − r²)² + (2ζr)²) = {e:g} × {number(r * r)}/√({number((1 - r * r) ** 2)} + "
        f"{number((2 * zeta * r) ** 2)}) = {number(x)} µm."
        + (" Well above the critical it tends to e: the disc spins about its own centre of mass." if r > 1.5 else ""),
        unit="µm",
    )


def _phase_lag(rng: np.random.Generator) -> Question:
    r, zeta = pick(rng, (0.5, 0.8, 0.9, 1.1, 1.5, 2)), pick(rng, (0.02, 0.05, 0.1, 0.2))
    lag = math.degrees(math.atan2(2 * zeta * r, 1 - r * r))
    return Question(
        f"A rotor runs at {r:g} times its critical speed, ζ = {zeta:g}. By how many degrees does the high spot lag the "
        "heavy spot?",
        lag,
        f"tan(lag) = 2ζr/(1 − r²) = {number(2 * zeta * r)}/{number(1 - r * r)}, so the lag is {number(lag)}°"
        + (": past 90°, since it runs above the critical." if r > 1 else "."),
        unit="°",
    )


def _conical(rng: np.random.Generator) -> Question:
    i_d, ratio, k_theta = pick(rng, (0.01, 0.02, 0.05)), pick(rng, (0.25, 0.5, 0.75)), pick(rng, (1000, 5000, 20000))
    i_p = rounded(ratio * i_d)
    omega = math.sqrt(k_theta / (i_d - i_p))
    return Question(
        f"A long rotor's tilt (I<sub>d</sub> = {i_d:g} kg·m², I<sub>p</sub> = {number(i_p)} kg·m²) sits on a "
        f"rotational stiffness of {k_theta:g} N·m/rad. At what speed does its forward conical whirl meet the 1X line?",
        omega * RPM,
        f"The tilt whirls at I<sub>d</sub>ω² − I<sub>p</sub>Ωω − k<sub>θ</sub> = 0 forward; on the 1X line ω = Ω, so "
        f"(I<sub>d</sub> − I<sub>p</sub>)Ω² = k<sub>θ</sub>: Ω = √({k_theta:g}/{number(i_d - i_p)}) = "
        f"{number(omega)} rad/s, {number(omega * RPM)} rpm. A disc (I<sub>p</sub> ≥ I<sub>d</sub>) never meets it.",
        unit="rpm",
    )


def _balance(rng: np.random.Generator) -> Question:
    v0 = cmath.rect(pick(rng, (40, 60, 100)), math.radians(pick(rng, (30, 60, 120, 200))))
    trial = pick(rng, (100, 200))  # g·mm at 0°
    alpha = cmath.rect(pick(rng, (0.2, 0.5, 1.0)), math.radians(pick(rng, (45, 90, 135))))
    v1 = v0 + alpha * trial
    v1_amp, v1_deg = rounded(abs(v1)), round(math.degrees(cmath.phase(v1))) % 360
    v1p = cmath.rect(v1_amp, math.radians(v1_deg))
    a = (v1p - v0) / trial
    w = -v0 / a
    radius = pick(rng, (20, 40, 50))
    grams = abs(w) / radius
    v0_deg = round(math.degrees(cmath.phase(v0))) % 360
    return Question(
        f"Balancing at one speed: the disc reads {number(abs(v0))} µm at {v0_deg}° as found, and {number(v1_amp)} µm "
        f"at {v1_deg}° with a {trial:g} g·mm trial weight at 0°. With the trial off, what mass of correction goes at "
        f"{radius:g} mm radius?",
        grams,
        f"The trial's effect is V<sub>1</sub> − V<sub>0</sub>, so the influence coefficient is α = (V<sub>1</sub> − "
        f"V<sub>0</sub>)/T = {number(abs(a))} µm per g·mm at {round(math.degrees(cmath.phase(a))) % 360}°. The "
        f"correction is W = −V<sub>0</sub>/α = {number(abs(w))} g·mm at {round(math.degrees(cmath.phase(w))) % 360}°: "
        f"{number(grams)} g at {radius:g} mm.",
        unit="g",
    )


def _oil_whip(rng: np.random.Generator) -> Question:
    lam, f = pick(rng, (0.42, 0.45, 0.48, 0.5)), pick(rng, (20, 26, 30, 40))
    rpm = 60 * f / lam
    return Question(
        f"A rotor's first forward whirl is at {f:g} Hz, and its bearings' oil swirls round at λ = {lam:g} of the "
        "shaft's speed. Above what speed does the whirl grow (oil whip)?",
        rpm,
        f"The swirl's cross-coupling λcΩ outruns the damping cω once λΩ > ω: Ω > ω/λ, {f:g}/{lam:g} = "
        f"{number(f / lam)} Hz, {number(rpm)} rpm. Whatever the damping: c is on both sides.",
        unit="rpm",
    )


def _internal_onset(rng: np.random.Generator) -> Question:
    fn, ce, ci = pick(rng, (20, 25, 30)), pick(rng, (10, 20, 50)), pick(rng, (10, 20, 40))
    rpm = 60 * fn * (1 + ce / ci)
    return Question(
        f"A Jeffcott rotor on rigid bearings, its natural frequency {fn:g} Hz, has c<sub>e</sub> = {ce:g} N·s/m of "
        f"damping at the disc and c<sub>i</sub> = {ci:g} N·s/m in the shaft. Above what speed does it whirl unstably?",
        rpm,
        f"Internal damping drives a forward whirl once the shaft turns faster than it whirls: the onset is "
        f"Ω = ω<sub>n</sub>(1 + c<sub>e</sub>/c<sub>i</sub>) = {fn:g} Hz × {number(1 + ce / ci)} = "
        f"{number(fn * (1 + ce / ci))} Hz, {number(rpm)} rpm: above the critical, however little c<sub>i</sub> is.",
        unit="rpm",
    )


# ------------------------------------------------------------------ predictions
def _high_spot(rng: np.random.Generator) -> Question:
    return Question(
        "The classic Jeffcott rotor runs at 3200 rpm, about twice its 1584 rpm critical speed. Where is the disc's "
        "high spot, against its heavy spot?",
        0,
        "Nearly opposite it: the disc lags 176° there. Above the critical the shaft deflects away from the heavy spot, "
        "and the disc tends to spin about its own centre of mass (self-centring).",
        choices=("Nearly opposite it", "At it", "About 90° behind it", "About 45° behind it"),
        setup=dict(target_rpm=3200.0),
        predict=True,
    )


def _overhung(rng: np.random.Generator) -> Question:
    return Question(
        "The overhung rotor's first mode is at 870 rpm at rest. Spinning, where does the unbalance find its critical "
        "speed?",
        0,
        "Above it, at 1284 rpm: the overhung disc tilts as it whirls, and the gyroscopic moment stiffens the forward "
        "whirl as the speed rises. The backward branch meets 1X lower, at 664 rpm, but the unbalance doesn't drive it.",
        choices=("Above 870 rpm", "At 870 rpm", "Below 870 rpm"),
        setup=dict(preset="Overhung rotor"),
        predict=True,
    )


def _anisotropic(rng: np.random.Generator) -> Question:
    return Question(
        "A rotor's supports are four times stiffer vertically than horizontally. How many critical speeds does the "
        "unbalance find in its first mode?",
        0,
        "Two, 1365 rpm (horizontal) and 1668 rpm (vertical), and between them the shaft whirls backward on an ellipse.",
        choices=("Two, with backward whirl between them", "One, between the two stiffnesses'", "None: it's damped out"),
        setup=dict(preset="Anisotropic supports"),
        predict=True,
    )


def _oil_damping(rng: np.random.Generator) -> Question:
    return Question(
        "With oil swirling at half the shaft's speed, the rotor goes unstable at 3136 rpm. The bearings' damping is "
        "doubled. Where does the onset go?",
        0,
        "Nowhere: it stays at 3136 rpm. The swirl's cross-coupling is λcΩ, so doubling c doubles the force driving the "
        "whirl as well as the one damping it: the whirl grows once λΩ > ω, whatever c is.",
        choices=("It stays put", "Up, about twice as high", "Up a little", "Down"),
        setup=dict(preset="Oil swirl λ = ½ (oil whip)", cx=200.0),
        predict=True,
    )


# ------------------------------------------------------------------ the spec
def _stiff_shaft(rng: np.random.Generator) -> Question:
    m, d, kb, length = pick(rng, (1, 2, 5)), pick(rng, (10, 12, 15)), 2e5, 0.5

    def rotor(diameter_mm: float) -> RotorSystem:
        return RotorSetup(length=length, diameter=diameter_mm / 1000, disc_mass=m, kx=kb).system()

    run = 10 * math.floor(critical_rpm(rotor(d)) / 1.25 / 10)
    lowest = rounded(1.15 * run)
    widest = rounded(1.1 * d, 2)
    lo, hi = 0.5 * d, float(d)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if critical_rpm(rotor(mid)) < lowest else (lo, mid)

    def goal(t: RotorSetup):
        s = t.system()
        a, b = s.bearing_a, s.bearing_b
        return (
            kept("Span L", s.length, length, "m"),
            kept("Disc mass", s.disc_mass, m, "kg"),
            kept("Disc position a/L", s.position, 0.5),
            kept("Young's modulus", s.youngs / 1e9, E_STEEL / 1e9, "GPa"),
            *(
                kept(f"Bearing {n}'s {k}", v, kb, "N/m")
                for n, k, v in (
                    ("A", "k<sub>x</sub>", a.kx),
                    ("A", "k<sub>y</sub>", a.ky),
                    ("B", "k<sub>x</sub>", b.kx),
                    ("B", "k<sub>y</sub>", b.ky),
                )
            ),
            Requirement("A massless shaft (the Jeffcott model)", f"ρ = {number(s.density)} kg/m³", s.density == 0),
            at_least("Its critical speed", critical_rpm(s), lowest, "rpm"),
            at_most("The shaft's diameter", s.diameter * 1000, widest, "mm"),
        )

    return Question(
        f"A {m:g} kg disc at the middle of a 0.5 m steel shaft, on supports of 200,000 N/m, is to run at {run:g} rpm "
        f"with its critical speed at least 15 % above that ({number(lowest)} rpm), a stiff-shaft design. Choose the "
        f"shaft's diameter, no more than {number(widest)} mm.",
        0,
        f"The critical is √(k<sub>eq</sub>/m), the shaft (48EI/L³, ∝ d⁴) in series with the supports (2k<sub>b</sub>), "
        f"so a thicker shaft helps less and less as it outgrows the supports. From about {number(hi)} mm it meets the "
        f"spec; {d:g} mm gives {number(critical_rpm(rotor(d)))} rpm.",
        setup=dict(length=length, diameter=0.8 * d / 1000, disc_mass=m, kx=kb, target_rpm=float(run)),
        goal=goal,
        design=dict(length=length, diameter=d / 1000, disc_mass=m, kx=kb, target_rpm=float(run)),
    )


PROBLEMS = tuple(
    Problem(key, title, "rotor_page", f"See {page_link('rotor_page', anchor, section)}.", make)
    for key, title, anchor, section, make in (
        ("critical", "A critical speed", "classic", "The classic case", _critical),
        ("whirl-radius", "The unbalance's whirl", "unbalance", "Unbalance response", _whirl_radius),
        ("phase-lag", "Heavy spot to high spot", "unbalance", "Unbalance response", _phase_lag),
        ("conical", "A forward conical critical", "gyroscopic", "The gyroscopic effect", _conical),
        ("balance", "A balance weight", "single-plane", "Single-plane balancing", _balance),
        ("oil-whip", "Oil whip's onset", "cross-coupling", "Cross-coupling and internal damping", _oil_whip),
        (
            "internal-onset",
            "Internal damping's onset",
            "cross-coupling",
            "Cross-coupling and internal damping",
            _internal_onset,
        ),
        ("high-spot", "Predict: above the critical", "unbalance", "Unbalance response", _high_spot),
        ("overhung", "Predict: an overhung disc", "overhung", "The overhung rotor", _overhung),
        ("anisotropic", "Predict: anisotropic supports", "anisotropic", "Anisotropic supports", _anisotropic),
        (
            "oil-damping",
            "Predict: more damping against oil whip",
            "cross-coupling",
            "Cross-coupling and internal damping",
            _oil_damping,
        ),
        ("stiff-shaft", "Spec: a stiff shaft", "classic", "The classic case", _stiff_shaft),
    )
)
