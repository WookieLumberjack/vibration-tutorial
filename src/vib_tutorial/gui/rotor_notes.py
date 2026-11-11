"""Text for the Jeffcott rotor page: the theory notes and the live equations and matrices."""

from __future__ import annotations

import cmath
import math
from collections.abc import Collection

import numpy as np

from ..core.rotor import DOF_NAMES, MODE_LIMIT, XD, YA, YB, YD, RotorSystem
from ..core.rotor_modes import periodic_steady, whirl_modes
from ..core.rotor_simulator import RPM
from . import rotor_figures
from .cms_notes import fmt
from .links import load_notes
from .style import colors

LABELS = [
    "x<sub>A</sub>",
    "x<sub>D</sub>",
    "θ<sub>x</sub>",
    "x<sub>B</sub>",
    "y<sub>A</sub>",
    "y<sub>D</sub>",
    "θ<sub>y</sub>",
    "y<sub>B</sub>",
]
assert len(LABELS) == len(DOF_NAMES)
GROUPS = ["i"] * 4 + ["b"] * 4  # x plane, y plane: tinted as the CMS page's interior and boundary blocks


def whirl_name(ratio: float) -> str:
    """Forward, backward or straight-line, from a whirl ratio (+1 circular forward, -1 backward)."""
    if ratio > 0.1:
        return "forward"
    if ratio < -0.1:
        return "backward"
    return "straight-line"


def _legend() -> str:
    tints = colors.block_tints
    return (
        "<p><small>Cell colours: "
        f"<span style='background:{tints['i', 'i']}'>&nbsp;x plane (horizontal)&nbsp;</span> "
        f"<span style='background:{tints['b', 'b']}'>&nbsp;y plane (vertical)&nbsp;</span> "
        f"<span style='background:{tints['b', 'i']}'>&nbsp;coupling between the planes&nbsp;</span>"
        "</small></p>"
    )


def _num(v: float, scale: float) -> str:
    """Four significant figures without exponents where they fit: 0.000625, 7363, 56,820."""
    if abs(v) <= 1e-10 * scale:
        return "0"
    if abs(v) >= 1e7 or abs(v) < 1e-4:
        return f"{v:.3e}"
    if abs(v) >= 1e4:
        return f"{float(f'{v:.4g}'):,.0f}"
    return f"{v:.4g}"


def _matrix(A: np.ndarray, labels: list[str], groups: list[str], title: str) -> str:
    """An HTML table of A with labelled rows and columns, its cells tinted by block (see _legend)."""
    scale = float(np.abs(A).max()) if A.size else 1.0
    # Unwrapped: a wide matrix (a shaft with mass) scrolls sideways rather than breaking its numbers.
    nowrap = "white-space:nowrap"
    head = "".join(f"<th style='{nowrap}'>{c}</th>" for c in labels)
    out = [f"<table border='1' cellspacing='0' cellpadding='3'><tr><th style='{nowrap}'>{title}</th>{head}</tr>"]
    for r, (label, rg) in enumerate(zip(labels, groups)):
        cells = []
        for c, cg in enumerate(groups):
            text = _num(A[r, c], scale)
            style = nowrap + (f"; color:{colors.grey}" if text == "0" else "")
            tint = colors.block_tints[tuple(sorted((rg, cg)))]
            cells.append(f"<td align='right' style='{style}' bgcolor='{tint}'>{text}</td>")
        out.append(f"<tr><th style='{nowrap}'>{label}</th>{''.join(cells)}</tr>")
    out.append("</table>")
    return "".join(out)


def _labels(system: RotorSystem) -> tuple[list[str], list[str]]:
    """q's labels and blocks: the stations, then each plane's shaft modal coordinates ξ."""
    r = range(1, system.modes_kept + 1)
    return (
        LABELS + [f"ξ<sub>x{k}</sub>" for k in r] + [f"ξ<sub>y{k}</sub>" for k in r],
        GROUPS + ["i"] * len(r) + ["b"] * len(r),
    )


def _coupling(s: RotorSystem) -> str:
    """Each bearing's constant k_xy, or that its coupling is an oil film's, or none (a magnet)."""
    return " and ".join(
        "the oil film's (below) at " + n
        if b.film is not None
        else "none (a magnetic bearing) at " + n
        if b.magnet is not None
        else f"{fmt(b.kxy, 1.0)} N/m at {n}"
        for n, b in (("A", s.bearing_a), ("B", s.bearing_b))
    )


def _films_html(s: RotorSystem, omega: float) -> str:
    """The oil films' paragraph: their loads, equilibria and coefficients at speed Ω (empty without films)."""
    if not s.films:
        return ""
    parts = []
    for name, _b, w, st in zip("AB", (s.bearing_a, s.bearing_b), s.bearing_loads, s.bearing_states(omega)):
        if st is None:
            continue
        k, c = st.k, st.c
        parts.append(
            f"at {name}, under {fmt(w, 1.0)} N, ε = {st.eccentricity:.3f} at {math.degrees(st.attitude):.1f}° from the "
            f"load line: k<sub>xx</sub>, k<sub>xy</sub>, k<sub>yx</sub>, k<sub>yy</sub> = {fmt(k[0, 0], 1.0)}, "
            f"{fmt(k[0, 1], 1.0)}, {fmt(k[1, 0], 1.0)}, {fmt(k[1, 1], 1.0)} N/m and c<sub>xx</sub>, "
            f"c<sub>xy</sub> = c<sub>yx</sub>, c<sub>yy</sub> = {fmt(c[0, 0], 1.0)}, {fmt(c[0, 1], 1.0)}, "
            f"{fmt(c[1, 1], 1.0)} N·s/m"
        )
    held = max(
        abs(omega),
        min(
            b.film.with_load(abs(w)).slowest()
            for b, w in zip((s.bearing_a, s.bearing_b), s.bearing_loads)
            if b.film is not None
        ),
    )
    slow = (
        f" Below {held * RPM:,.0f} rpm the journal is nearly on its bottom, and the film is taken as it is "
        "there (ε = 0.9)."
        if held > abs(omega)
        else ""
    )
    return (
        "<h4>The oil films</h4><p>The journals run in plain journal bearings (short-bearing theory; see the "
        "<i>Fluid-film bearing</i> tab) in place of the springs and dampers. Each is linearized about where its "
        "journal settles under its share of the weight, f = W − K<sub>f</sub>(q − q<sub>J</sub>) − "
        "C<sub>f</sub>q̇: K<sub>f</sub> and C<sub>f</sub> are in K and C below, and the constant "
        "p = W + K<sub>f</sub>q<sub>J</sub> joins f, so the rotor rests with its journals at q<sub>J</sub>. "
        "Both change with the speed. K<sub>f</sub> is not symmetric, and its skew part is the oil's swirl at "
        "half the speed, k<sub>xy</sub> − k<sub>yx</sub> = ½Ω(c<sub>xx</sub> + c<sub>yy</sub>). At this speed "
        + "; ".join(parts)
        + "."
        + slow
        + (
            " The time response takes the films' whole force instead, the remainder r<sub>f</sub> beyond this "
            "linearization stepped as a load on the journals (<i>Theory</i>, Numerics)."
            if s.whole_films
            else ""
        )
        + "</p>"
    )


def _um(x: float) -> str:
    """x (m) in µm to three figures, round-off (below a picometre) as 0."""
    return "0" if abs(x) < 1e-12 else f"{x * 1e6:.3g}"


def _state_space(s: RotorSystem) -> str:
    """The state-space form: Z = [q, q̇], or with magnetic bearings their controllers' states too."""
    if not s.magnets:
        return (
            "<p>With Z = [q, q̇] the equations become Z' = A(Ω) Z + B u, u = f the force (or moment) on each "
            "DOF, B = [0 ; M<sup>−1</sup>]:</p><p align='center'>A(Ω) = [ 0 &nbsp; I ; "
            "−M<sup>−1</sup>(K + ΩH) &nbsp; −M<sup>−1</sup>(C + ΩG) ]</p>"
        )
    return (
        "<p>With Z = [q, q̇, x<sub>c</sub>], the magnets' controllers' states last "
        f"({s.n_control} of them), the equations become Z' = A(Ω) Z + B u, u = f the force (or moment) on each "
        "DOF, B = [0 ; M<sup>−1</sup> ; 0]:</p><p align='center'>A(Ω) = [ 0 &nbsp; I &nbsp; 0 ; "
        "−M<sup>−1</sup>(K + ΩH) &nbsp; −M<sup>−1</sup>(C + ΩG) &nbsp; M<sup>−1</sup>F<sub>c</sub> ; "
        "B<sub>c</sub> &nbsp; 0 &nbsp; A<sub>c</sub> ]</p><p>B<sub>c</sub> gives each controller its journal's "
        "displacement, A<sub>c</sub> holds their equations (above) and F<sub>c</sub> = −k<sub>i</sub> puts each "
        "current back on its journal. The filters and amplifiers are fast and do not oscillate, so their roots "
        "are left out of the modes below.</p>"
    )


def _magnets_html(s: RotorSystem) -> str:
    """The magnetic bearings' paragraph: the actuator, the controller's states and the push at the modes."""
    if not s.magnets:
        return ""
    magnets = [(n, b.magnet) for n, b in (("A", s.bearing_a), ("B", s.bearing_b)) if b.magnet is not None]
    _, m = magnets[0]
    which = "both bearings" if len(magnets) == 2 and magnets[0][1] == magnets[1][1] else f"bearing {magnets[0][0]}"
    lo, hi = m.damped_band()
    band = (
        f"It damps from {lo:.3g} to {hi:,.4g} Hz; outside that band c<sub>b</sub> &lt; 0 and it feeds a mode."
        if math.isfinite(lo)
        else "It damps nowhere: c<sub>b</sub> &lt; 0 at every frequency."
    )
    states = ", ".join((["∫x"] if m.integral else []) + (["x<sub>d</sub>"] if m.kd > 0.0 else []) + ["i"])
    return (
        "<h4>The magnetic bearings</h4><p>The journals float in active magnetic bearings (the <i>Magnetic "
        "bearing</i> tab) in place of the springs and dampers. On each axis a pair of magnets pulls the journal "
        f"toward the nearer one with k<sub>m</sub>x, here (at {which}) k<sub>m</sub> = {fmt(m.negative_stiffness, 1.0)} "
        "N/m: that is the −k<sub>m</sub> on the journals' diagonal in K below, a negative stiffness. The rest of "
        f"the push is the control current's, −k<sub>i</sub>i (k<sub>i</sub> = {m.current_gain:.4g} N/A), set by a "
        f"PID behind the amplifier, i = G(s)x with G(s) = (k<sub>p</sub> + k<sub>I</sub>/s + "
        f"k<sub>d</sub>s/(1 + s/ω<sub>d</sub>)) / (1 + s/ω<sub>a</sub>): k<sub>p</sub> = {m.kp:,.4g} A/m, "
        f"k<sub>I</sub> = {m.ki:,.4g} A/(m·s), k<sub>d</sub> = {m.kd:.4g} A·s/m, f<sub>d</sub> = {m.filter_hz:,.4g} Hz, "
        f"f<sub>a</sub> = {m.amplifier_hz:,.4g} Hz. Each axis's controller has {m.n_states} state"
        f"{'s' if m.n_states > 1 else ''} of its own, x<sub>c</sub> = [{states}]:</p>"
        "<p align='center'>(∫x)' = x, &nbsp; x<sub>d</sub>' = ω<sub>d</sub>(x − x<sub>d</sub>), &nbsp; i' = "
        "ω<sub>a</sub>(k<sub>p</sub>x + k<sub>I</sub>∫x + k<sub>d</sub>ω<sub>d</sub>(x − x<sub>d</sub>) − i)</p>"
        "<p>At a frequency ω the bearing pushes back as a spring and a damper, K<sub>b</sub>(iω) = "
        "k<sub>i</sub>G(iω) − k<sub>m</sub> = k<sub>b</sub> + iωc<sub>b</sub>. " + band + " The steady states "
        "(Bode and polar plots, harmonic balance) take K<sub>b</sub> at each harmonic; with the integral it is "
        "infinite at 0 Hz, and a steady load cannot move the journals.</p>"
    )


def matrices_html(system: RotorSystem, omega: float, hidden: Collection[str] = ()) -> str:
    """The equations of motion, every matrix with the current numbers, and the modes at speed Ω.

    `hidden` names the RotorSystem fields a balancing exercise keeps out of sight: the
    unbalance's size and angle, and perhaps the skew's.
    """
    M, C, G, K = system.matrices()
    H = system.circulatory()
    s = system
    labels, groups = _labels(s)
    films = _films_html(s, omega) + _magnets_html(s)
    if s.films:
        Kf, Cf, _ = s._film_terms(omega)
        K, C = K + Kf, C + Cf

    def _mat(A: np.ndarray, title: str) -> str:
        return _matrix(A, labels, groups, title)

    ks = s.shaft_stiffness()
    kept = s.modes_kept
    one = ["x<sub>A</sub>", "x<sub>D</sub>", "θ<sub>x</sub>", "x<sub>B</sub>"]
    one += [f"ξ<sub>{k}</sub>" for k in range(1, kept + 1)]
    shaft = _matrix(ks, one, ["i"] * len(one), "K<sub>shaft</sub>")
    modes = whirl_modes(system, omega)
    rows = []
    for r, (f, z, w) in enumerate(zip(modes.freq_hz, modes.zeta, modes.whirl)):
        eig = complex(-z * 2 * math.pi * f / math.sqrt(max(1 - z * z, 1e-12)), 2 * math.pi * f)
        rows.append(
            f"<tr><td align='center'>{r + 1}</td><td align='right'>{eig.real:.4g} + {eig.imag:.5g}i</td>"
            f"<td align='right'>{f:.4g}</td><td align='right'>{f * 60:,.0f}</td><td align='right'>{z:.3g}</td>"
            f"<td align='right'>{w:+.2f}</td><td>{whirl_name(w)}</td></tr>"
        )
    rpm = omega * RPM
    if s.asymmetry:
        Kc, _ = s.deviatoric
        asymmetric = (
            f"<h4>The asymmetric shaft ΔK(φ)</h4><p>The shaft is stiffer bending toward its stiff axis, at "
            f"φ<sub>s</sub> = φ + β<sub>s</sub> from x (β<sub>s</sub> = {math.degrees(s.asymmetry_angle):.0f}° from the mark), than "
            f"across it: "
            f"EI(1 ± Δ) with Δ = {s.asymmetry:.3g}. Its stiffness less the mean (K<sub>shaft</sub>, in K) is, "
            "in each plane pair,</p><p align='center'>ΔK(φ) = Δ [cos 2φ<sub>s</sub> K<sub>shaft</sub> &nbsp; "
            "sin 2φ<sub>s</sub> K<sub>shaft</sub> ; sin 2φ<sub>s</sub> K<sub>shaft</sub> &nbsp; −cos 2φ<sub>s</sub> "
            "K<sub>shaft</sub>]</p><p>or, as f<sub>x</sub> + i f<sub>y</sub>, "
            "−ΔK<sub>shaft</sub>e<sup>2iφ<sub>s</sub></sup>(x − iy). It turns with the "
            "shaft, twice per revolution as seen from the bearings, so the equations have a periodic "
            "stiffness and no fixed modes. Here it is with the stiff axis along x (φ<sub>s</sub> = 0):</p>"
            + _matrix(Kc, labels, groups, "ΔK(φ<sub>s</sub> = 0)")
            + "<p>In coordinates turning with the shaft, q = R(φ)q<sub>r</sub>, it is constant, and so is "
            "everything else if the supports are isotropic: M q̈<sub>r</sub> + (C + ΩG + 2ΩMJ) q̇<sub>r</sub> + "
            "(K + ΩH + ΔK(0) − Ω²M + ΩCJ + Ω²GJ) q<sub>r</sub> = R<sup>T</sup>f. The −Ω²M is the centrifugal force: between the two critical speeds it "
            "outweighs the soft axis's stiffness but not the stiff one's, and the whirl grows, locked to 1X. "
            + (
                "Here the supports are isotropic, so the modes below come from that constant system."
                if s.isotropic_supports
                else "Here the supports are not isotropic, so the modes below come from Hill's method instead: the "
                "periodic system's Floquet exponents, from its harmonics up to ±3 × 2Ω."
            )
            + " The time stepping takes each step as half a step of the rest (the exact update), the kick "
            "of ΔK at mid-step on the velocities, and the other half step.</p>"
        )
    else:
        asymmetric = ""
    cracked = _crack_note(s) if s.crack is not None else ""
    if s.overhung:
        near, _far = ("B", "A") if s.a > s.length else ("A", "B")
        elements = (
            f"Two Euler–Bernoulli beam elements: the span between the bearings (L = {s.length:.4g} m) and "
            f"the overhang from bearing {near} to the disc (ℓ = {s.overhang:.4g} m), with EI = {s.ei:.4g} N·m². "
            f"The disc sits at a = {s.a:.4g} m from A, outside the bearings."
        )
        disc_k = f"3EI/(ℓ²(L + ℓ)) = {s.disc_stiffness:,.0f} N/m for the overhung disc"
    else:
        elements = (
            f"Two Euler–Bernoulli beam elements, A to the disc (a = {s.a:.4g} m) and the disc to B "
            f"(b = {s.b:.4g} m), with EI = {s.ei:.4g} N·m²."
        )
        disc_k = f"3EIL/(a²b²) = {s.disc_stiffness:,.0f} N/m"
    n_el = s.nodes.size - 1
    if s.density:
        freqs = ", ".join(f"{f:,.0f}" for f in s.fixed_interface_freqs) or "none"
        elements = (
            f"The shaft has mass, {s.shaft_mass:.3g} kg ({s.density:,.0f} kg/m³): it is {n_el} Euler–Bernoulli "
            f"beam elements, with {'lumped' if s.lumped else 'consistent'} mass and their sections' rotary "
            f"inertia, a displacement and a slope at every node. That model is reduced (Craig–Bampton): the "
            f"stations' DOFs are kept, and the rest of the shaft moves as the static shape they give it plus "
            f"its {kept} lowest mode{'s' if kept != 1 else ''} with the stations held, up to {MODE_LIMIT:,.0f} Hz "
            f"({freqs} Hz). Each mode's mass-normalized amplitude ξ is a DOF of its own, in both planes. "
            + elements.replace("Two Euler–Bernoulli beam elements", "Without the extra nodes, the stretches are")
        )
        mass_text = (
            "M is no longer diagonal: the shaft's mass couples the stations and its modes. The disc's "
            "mass m and I<sub>d</sub>, and the journals', are added on their own DOFs."
        )
        gyro_text = (
            f"G has ±I<sub>p</sub> = ±{s.ip:.4g} kg·m² on the disc's slopes and the shaft's own sections' "
            f"(twice their rotary inertia), joining each plane to the other."
        )
        condensed = (
            "The stations' block of K<sub>shaft</sub> is the shaft's exact static stiffness, as for the "
            "massless shaft; each mode adds its ω² alone on the diagonal (mass normalized)."
        )
    else:
        mass_text = (
            "M is diagonal: the journal masses, the disc's mass m on its displacements and its diametral "
            "inertia I<sub>d</sub> on its slopes."
        )
        gyro_text = (
            f"G has just two entries, ±I<sub>p</sub> = ±{s.ip:.4g} kg·m², joining the disc's two slopes, "
            f"one plane to the other."
        )
        condensed = (
            "The bearings carry no moment, so the slopes at A and B are condensed out (they follow the four DOFs kept)."
        )
    kxy_a, kxy_b = s.cross_coupling(omega)
    if s.bearing_a.swirl or s.bearing_b.swirl:
        swirl = (
            f" With fluid swirl the cross-coupling grows with the speed, k<sub>xy</sub> + λcΩ (c the mean of "
            f"c<sub>x</sub> and c<sub>y</sub>), and that part "
            f"sits in ΩH (below): at this speed k<sub>xy</sub> is {fmt(kxy_a, 1.0)} N/m at A and "
            f"{fmt(kxy_b, 1.0)} N/m at B in all."
        )
    else:
        swirl = ""
    fb = ks @ s.bow_shape() * s.bow
    bow_journals = f"{fmt(-fb[0], 1.0)} N at A and {fmt(-fb[3], 1.0)} N at B"
    if s.gravity and s.periodic:
        qs, qk = (x[0] for x in periodic_steady(s, np.array([omega]), gravity=True, rotating=False, top=3))
        breathing = s.spacing == 1
        gravity = (
            f"<h4>Gravity</h4><p>On a horizontal rotor f also holds the weights, constant: "
            f"−m<sub>A</sub>g, −mg and −m<sub>B</sub>g on y<sub>A</sub>, y<sub>D</sub> and y<sub>B</sub> "
            f"(here {fmt(s.disc_mass * s.gravity, 1.0)} N on the disc)"
            + (f", and the shaft's own weight, {fmt(s.shaft_mass * s.gravity, 1.0)} N" if s.density else "")
            + ". On the turning shaft a constant force meets a stiffness that turns, so the rotor never "
            "settles: its steady state is periodic, "
            + (
                "q = Q<sub>0</sub> + Re(Q<sub>1</sub>e<sup>iφ</sup> + Q<sub>2</sub>e<sup>2iφ</sup> + "
                "Q<sub>3</sub>e<sup>3iφ</sup> + ...), found by harmonic balance (each harmonic n of the response "
                "is tied to n ± 1, n ± 2 and n ± 3 by the breathing crack's ΔK, up to ±6X)"
                if breathing
                else "q = Q<sub>0</sub> + Re(Q<sub>2</sub>e<sup>2iφ</sup>)"
                + ("" if s.rotating_frame else " + (4X, ...)")
                + ", found by harmonic balance (each "
                "harmonic n of the response is tied to n ± 2 by ΔK)"
            )
            + f". The mean sag Q<sub>0</sub> puts the disc at ({qs[XD].real * 1e6:.4g}, {qs[YD].real * 1e6:.4g}) "
            "µm; about it the disc whirls at "
            + (
                ", ".join(
                    f"{k + 1}X {abs(qk[k, XD]) * 1e6:.3g} µm in x and {abs(qk[k, YD]) * 1e6:.3g} µm in y"
                    for k in range(3)
                )
                + " at this speed (on top of the unbalance's 1X), 2X most where 2Ω "
                "meets a natural frequency and 3X where 3Ω does."
                if breathing
                else f"2X, {abs(qk[1, XD]) * 1e6:.4g} µm in x and {abs(qk[1, YD]) * 1e6:.4g} µm in y at this speed, "
                "most where 2Ω meets a natural frequency: the gravity critical."
            )
            + "</p>"
        )
    elif s.gravity and s.magnets:
        qs = s.static_deflection(omega)
        gravity = (
            f"<h4>Gravity</h4><p>On a horizontal rotor f also holds the weights, constant: "
            f"−m<sub>A</sub>g, −mg and −m<sub>B</sub>g on y<sub>A</sub>, y<sub>D</sub> and y<sub>B</sub> "
            f"(here {fmt(s.disc_mass * s.gravity, 1.0)} N on the disc)"
            + (f", and the shaft's own weight, {fmt(s.shaft_mass * s.gravity, 1.0)} N" if s.density else "")
            + ". They hold the rotor still where A Z + B f<sub>g</sub> = 0, the controllers' states included"
            + (
                ": with the integral the magnets hold their journals at the centre, the current growing until "
                "they are back, so the shaft alone sags"
                if any(b.magnet is not None and b.magnet.integral for b in (s.bearing_a, s.bearing_b))
                else ""
            )
            + f". Here the disc is at ({qs[XD] * 1e6:.4g}, {qs[YD] * 1e6:.4g}) µm and the journals at "
            f"({_um(qs[YA])}, {_um(qs[YB])}) µm vertically; the rotor vibrates about that as without "
            "gravity.</p>"
        )
    elif s.gravity:
        qs = s.static_deflection(omega)
        gravity = (
            f"<h4>Gravity</h4><p>On a horizontal rotor f also holds the weights, constant: "
            f"−m<sub>A</sub>g, −mg and −m<sub>B</sub>g on y<sub>A</sub>, y<sub>D</sub> and y<sub>B</sub> "
            f"(here {fmt(s.disc_mass * s.gravity, 1.0)} N on the disc)"
            + (
                f", and the shaft's own weight, {fmt(s.shaft_mass * s.gravity, 1.0)} N, spread over the stations "
                f"and its modes"
                if s.density
                else ""
            )
            + ". They hold the rotor at the static "
            + (
                "deflection q<sub>s</sub> = (K + ΩH)<sup>−1</sup>(f<sub>g</sub> + p), p the films' load (above), "
                if s.films
                else "deflection q<sub>s</sub> = (K + ΩH)<sup>−1</sup>f<sub>g</sub>, "
            )
            + f"here the disc at "
            f"({qs[XD] * 1e6:.4g}, {qs[YD] * 1e6:.4g}) µm, about which it vibrates as without gravity.</p>"
        )
    else:
        gravity = ""
    modal = (
        f", then the shaft's modal coordinates ξ, {kept} in each plane (in √kg·m: the modes are mass "
        f"normalized, so a mode moving alone has kinetic energy ½ξ̇²)"
        if kept
        else ""
    )
    rigid = " (ξ = 0: a rigid motion is a static shape)" if kept else ""
    net = s.mass_unbalance
    if hidden:
        unbalance, force = "The unbalance (hidden: find it on the <i>Balancing</i> tab)", "hidden"
    else:
        unbalance = (
            f"The unbalance U = m e = {s.unbalance * 1e6:.4g} g·mm at ψ = {math.degrees(s.unbalance_angle):.0f}°"
        )
        on_disc = len(s.weights) + sum(s.on_disc(p) for p in s.plane_weights)
        if on_disc:
            unbalance += (
                f", with {on_disc} balance weight{'s' if on_disc > 1 else ''}"
                f" (net {abs(net) * 1e6:.4g} g·mm at {math.degrees(math.atan2(net.imag, net.real)):.0f}°),"
            )
        force = fmt(abs(net) * omega**2, 1.0)
    on_shaft = [p for p in s.plane_weights if not s.on_disc(p)]
    if on_shaft:
        listed = "; ".join(
            f"{abs(p.weight) * 1e6:.4g} g·mm at {math.degrees(cmath.phase(p.weight)) % 360:.0f}°, "
            f"z = {p.z * 1e3:.0f} mm"
            for p in on_shaft
        )
        shaft_weights = (
            f" Balance weights on the shaft ({listed}) are point forces of the same form, shared over "
            f"the DOFs as a load at z is, Ψ(z)<sup>T</sup>."
        )
    else:
        shaft_weights = ""
    if "skew" in hidden:
        skew = "The disc's skew τ (hidden)"
        moment = "hidden"
    else:
        skew = (
            f"The disc's skew τ = {math.degrees(s.skew):.4g}° (toward {math.degrees(s.skew_angle):.0f}° from the mark)"
        )
        moment = fmt(abs(s.couple_unbalance) * omega**2, 1.0) + " N·m"
    faces = (
        " A balance weight U<sub>w</sub> on one of the disc's faces, Δz from its centre plane, adds U<sub>w</sub>Δz to "
        "(I<sub>d</sub> − I<sub>p</sub>)τe<sup>iγ</sup>."
        if any(p.disc for p in s.plane_weights)
        else ""
    )
    return f"""
<h3>Equations of motion at {rpm:,.0f} rpm (Ω = {omega:.4g} rad/s)</h3>
<p style='font-size:large' align='center'>M q̈ + (C + Ω G) q̇ + (K + Ω H{" + ΔK(φ)" if s.periodic else ""}) q = f(t)</p>
<p>q = [x<sub>A</sub>, x<sub>D</sub>, θ<sub>x</sub>, x<sub>B</sub> | y<sub>A</sub>, y<sub>D</sub>,
θ<sub>y</sub>, y<sub>B</sub>]: the journals' displacements at A and B, the disc's displacement, and the
shaft's slope at the disc (θ<sub>x</sub> = dx/dz, θ<sub>y</sub> = dy/dz), first in the horizontal x–z
plane, then in the vertical y–z plane{modal}. Units: m and rad; forces in N, moments in N·m.</p>
{_legend()}

<h4>Mass M and stiffness K</h4>
<p>{mass_text} K is the shaft's bending stiffness plus each support spring on its
journal. Only the bearings' cross-coupled stiffness k<sub>xy</sub> joins the two planes, as
K[x, y] = k<sub>xy</sub> and K[y, x] = k<sub>yx</sub> (−k<sub>xy</sub> unless set otherwise) at each
journal (here {_coupling(s)}). That part is
skew-symmetric, so K is symmetric only without it; with isotropic supports and no cross-coupling K has
two identical blocks.{swirl}</p>
{films}
{_mat(M, "M")}<br>{_mat(K, "K")}

<h4>The shaft (one plane; y is the same)</h4>
<p>{elements} {condensed} Moving the whole shaft sideways, or tilting it rigidly,
takes no force: K<sub>shaft</sub>[1, 1, 0, 1]<sup>T</sup> = 0 and K<sub>shaft</sub>[0, a, 1, L]<sup>T</sup> = 0{rigid}.
The supports hold it.</p>
{shaft}

<h4>Damping C and the gyroscopic matrix Ω G</h4>
<p>C has the support dampers, and the shaft's internal damping η K<sub>shaft</sub> in each plane
(η = c<sub>i</sub>/k<sub>disc</sub> = {s.loss_time:.4g} s, so that it is c<sub>i</sub> =
{s.internal_damping:g} N·s/m on the disc's own stiffness k<sub>disc</sub> = {disc_k}). {gyro_text}
It is <b>skew-symmetric</b> (G<sup>T</sup> = −G), so
q̇<sup>T</sup>(ΩG)q̇ = 0: unlike C, it takes no power out. It only turns the motion from one plane into
the other, and it grows with the speed.</p>
{_mat(C, "C")}<br>{_mat(omega * G, "Ω G")}

<h4>The circulatory matrix Ω H</h4>
<p>The internal damping acts on the bending rate the spinning shaft sees, q̇ − ΩJq in fixed
coordinates (J turns x toward y). The −ΩJq part becomes a stiffness that grows with the speed,
ΩH = Ωη[0 K<sub>shaft</sub>; −K<sub>shaft</sub> 0], skew-symmetric like the cross-coupling in K. A
fluid-film bearing's swirl adds its λc at the journal, where K has k<sub>xy</sub> (here λ =
{s.bearing_a.swirl:g} at A and {s.bearing_b.swirl:g} at B). A skew stiffness pushes a forward whirl
along its orbit: per cycle it puts in energy where C takes it out. Damping falls off as the speed rises,
and past the onset the whirl grows.</p>
{_mat(omega * H, "Ω H")}
{asymmetric}
{cracked}

<h4>State space</h4>
{_state_space(s)}
<p>A is {s.n_states} × {s.n_states} and not symmetric, and its eigenvalues s = −ζω<sub>n</sub> ± iω<sub>d</sub> come in
conjugate pairs. The modes at this speed, lowest first:</p>
<table border='1' cellspacing='0' cellpadding='3'>
<tr><th>Mode</th><th>s [1/s]</th><th>f<sub>d</sub> [Hz]</th><th>f<sub>d</sub> [rpm]</th><th>ζ</th>
<th>Whirl ratio</th><th>Whirl</th></tr>
{''.join(rows)}
</table>
<p><small>Whirl ratio: +1 for a circular forward whirl (the same way as the spin, x toward y), −1 for
circular backward whirl, 0 for a straight line; in between, an ellipse. The stations are weighted by
their masses (and the slopes by I<sub>d</sub>). A negative ζ is a mode that grows: the rotor is unstable
at this speed.{" For the turning shaft these are Floquet modes, e<sup>st</sup> times a motion periodic in φ: each is listed by its harmonic with the most kinetic energy, and a mode locked to 1X sits at exactly the speed." if s.periodic else ""}</small></p>

<h4>The 1X forces</h4>
<p>{unbalance} turns with the shaft, the keyphasor mark at angle φ (φ' = Ω). For the heavy spot at the
mark, keeping the disc's centre of mass on its path takes</p>
<p align='center'>f<sub>x</sub> = U(Ω² cos φ + Ω' sin φ), &nbsp; f<sub>y</sub> = U(Ω² sin φ − Ω' cos φ)</p>
<p>on the disc's x<sub>D</sub> and y<sub>D</sub>, or in general f<sub>x</sub> + i f<sub>y</sub> =
Ue<sup>iψ</sup>(Ω² − iΩ')e<sup>iφ</sup>, with ψ the heavy spot's angle from the mark. Balance weights add
to Ue<sup>iψ</sup>. Here the force is {force} N turning once per revolution. Ω' is the ramp's angular
acceleration, while the speed changes.{shaft_weights}</p>
<p>{skew} gives the same form on its slopes, M<sub>x</sub> + i M<sub>y</sub> =
(I<sub>d</sub> − I<sub>p</sub>)τ(Ω² − iΩ')e<sup>i(φ+γ)</sup>: here {moment}.{faces} The shaft's bow δ<sub>b</sub> =
{s.bow * 1e6:.4g} µm (toward {math.degrees(s.bow_angle):.0f}°) gives K<sub>shaft</sub>q<sub>bow</sub>
turning with the shaft whatever the speed: {fmt(s.disc_stiffness * s.bow, 1.0)} N on the disc, and the
bearings' share of it back on the journals, {bow_journals}.</p>
{gravity}
"""


THEORY_HTML = load_notes("rotor_notes", figures=rotor_figures.FIGURES)


def _crack_note(s: RotorSystem) -> str:
    """The live notes' section on the crack: its compliance, its loss of stiffness and how it breathes."""
    crack = s.crack
    assert crack is not None, "only written for a cracked shaft"
    c_open, c_across = s.crack_compliances
    if not s.crack_opening:
        state = (
            "It is a breathing crack, but nothing opens it: with no gravity its faces stay pressed together, "
            "and the shaft is as if round. Untick <i>Breathing</i> to hold it open."
        )
    elif crack.breathing:
        down = s.crack_sign > 0
        state = (
            "It breathes: the shaft's weight opens it as its mouth turns "
            + (
                "down (the shaft sags there, stretching its bottom)"
                if down
                else "up (the shaft hogs there, stretching its top)"
            )
            + ", and shuts it half a turn later, open by σ(φ) = (1 "
            + ("−" if down else "+")
            + " sin(φ + β<sub>c</sub>))/2. The loss is σ(φ) times the open crack's, "
            "so it has the mean ½P in K and harmonics 1X, 2X and 3X: ΔK(φ) = Σ E<sub>n</sub>e<sup>inφ</sup> + "
            "conj. Even on isotropic supports it changes in axes turning with the shaft (the weight does "
            "not turn), so the modes below come from Hill's method, from harmonics up to ±3Ω."
        )
    else:
        state = (
            "It is held open: the loss is P + [the deviatoric part], an asymmetric shaft's, its soft axis along "
            "the mouth, turning with the shaft."
        )
    return (
        f"<h4>The crack</h4><p>A crack {crack.depth:.0%} of the shaft's diameter deep, at z = {s.crack_z * 1e3:.4g} mm, "
        f"its mouth at β<sub>c</sub> = {math.degrees(crack.angle):.0f}° from the mark. Open, it is a hinge whose "
        f"compliance is κ<sub>o</sub> = {c_open:.3g} rad/(N·m) to bending that opens it and κ<sub>a</sub> = "
        f"{c_across:.3g} across. Condensed onto the stations it takes D = b<sub>h</sub>b<sub>h</sub><sup>T</sup>κ/"
        "(1 + κk<sub>h</sub>) off one plane's stiffness, b<sub>h</sub> the bending moment at the crack per unit of "
        "each station's displacement and k<sub>h</sub> the "
        "shaft's stiffness against a jump in its slope there: D<sub>o</sub> along the mouth and D<sub>a</sub> "
        "across, so in fixed axes P = (D<sub>o</sub> + D<sub>a</sub>)/2 on both planes and the deviatoric part of "
        f"(D<sub>o</sub> − D<sub>a</sub>)/2 turning at 2X, as the asymmetric shaft's. {state}</p>"
    )
