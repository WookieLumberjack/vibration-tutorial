"""The symbols the notes use, in one place: each page's notation table and the glossary are made from it.

A `Symbol` is one row: what is written (HTML, set as inline maths), what it means, its units, and the page
that brings it in (with an anchor there, if the section matters). A notes file lists the symbols it uses with
``<notation>key key …</notation>``, which `links.load_notes` writes out as its table (`notation_html`); the
glossary (`theory/glossary.html`'s ``<!--GLOSSARY-->``) lists them all, a table per course group
(`glossary_html`). So a symbol's meaning is written once, and a page's table and the glossary can't disagree.

A letter that does two jobs gets a row for each, with its own key (``T_trans``, transmissibility; ``energies``,
kinetic and potential energy): a page lists the one it means.

Working on it: `tests/test_notation.py` checks every key a page lists is here, that no page lists a symbol
brought in on a later page, that no page's table gives one letter two meanings, and that each row's page is
in the course.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import maths


@dataclass(frozen=True)
class Symbol:
    key: str
    symbol: str  # HTML, set as inline maths
    meaning: str  # HTML
    units: str
    page: str  # the page that brings it in (an attribute in links.COURSE)
    anchor: str = ""  # the section there, if it matters


SYMBOLS = (
    Symbol("m", "m, k, c", "mass, spring stiffness, viscous damping", "kg, N/m, N·s/m", "sdof_page"),
    Symbol("x", "x, ẋ, ẍ", "displacement, velocity, acceleration", "m, m/s, m/s²", "sdof_page"),
    Symbol("f", "f(t), F", "applied force; its amplitude", "N", "sdof_page"),
    Symbol(
        "omega_n",
        "ω<sub>n</sub>, f<sub>n</sub>",
        "undamped natural frequency, √(k/m); f<sub>n</sub> = ω<sub>n</sub>/2π",
        "rad/s, Hz",
        "sdof_page",
    ),
    Symbol("zeta", "ζ", "damping ratio, c/c<sub>c</sub>", "–", "sdof_page"),
    Symbol("c_c", "c<sub>c</sub>", "critical damping, 2√(km) = 2mω<sub>n</sub>", "N·s/m", "sdof_page"),
    Symbol(
        "omega_d",
        "ω<sub>d</sub>, f<sub>d</sub>",
        "damped natural frequency, ω<sub>n</sub>√(1 − ζ²)",
        "rad/s, Hz",
        "sdof_page",
    ),
    Symbol("tau", "τ", "time constant of the decay, 1/ζω<sub>n</sub>", "s", "sdof_page"),
    Symbol("delta", "δ", "logarithmic decrement, ln(x<sub>i</sub>/x<sub>i+1</sub>) ≈ 2πζ", "–", "sdof_page"),
    Symbol("Q", "Q", "quality factor, 1/2ζ: the peak over the static response", "–", "sdof_page"),
    Symbol("Omega", "Ω, f", "forcing (drive) frequency", "rad/s, Hz", "sdof_page"),
    Symbol("r", "r", "frequency ratio, Ω/ω<sub>n</sub>", "–", "sdof_page"),
    Symbol(
        "H",
        "H(Ω)",
        "frequency response function: response per unit force (the receptance, for displacement)",
        "m/N",
        "sdof_page",
    ),
    Symbol("phi", "φ", "phase of the response behind the force (also a mode shape, below)", "rad, °", "sdof_page"),
    Symbol(
        "s",
        "s, s<sub>1</sub>, s<sub>2</sub>",
        "pole: a root of m s² + c s + k = 0, −ζω<sub>n</sub> ± iω<sub>d</sub> below critical damping",
        "1/s",
        "sdof_page",
        "poles",
    ),
    Symbol(
        "x0",
        "x<sub>0</sub>, v<sub>0</sub>",
        "the displacement and velocity it starts from",
        "m, m/s",
        "sdof_page",
        "free-vibration",
    ),
    Symbol("n_count", "n", "a number of cycles (a count, where a page says)", "–", "sdof_page", "log-decrement"),
    Symbol(
        "X",
        "X",
        "the steady response's complex amplitude: x = Re(X e<sup>iΩt</sup>)",
        "m",
        "sdof_page",
        "forced-response",
    ),
    Symbol(
        "half_power",
        "f<sub>1</sub>, f<sub>2</sub>",
        "the half-power points, where |X| is 1/√2 of its peak",
        "Hz",
        "sdof_page",
        "half-power",
    ),
    Symbol(
        "unbalance",
        "m<sub>e</sub>, e",
        "an unbalance: its mass, and its radius (eccentricity)",
        "kg, m",
        "sdof_page",
        "unbalance",
    ),
    Symbol(
        "b_h",
        "b, h",
        "a rectangular section's width, and its depth in the direction it bends",
        "m",
        "sdof_page",
        "where-from",
    ),
    Symbol("g", "g, L", "gravity; a length (a bar's, a beam's, a pendulum's)", "m/s², m", "sdof_page", "where-from"),
    Symbol(
        "section",
        "E, A, I",
        "Young's modulus; a cross-section's area and its second moment of area",
        "Pa, m², m⁴",
        "sdof_page",
        "where-from",
    ),
    Symbol(
        "theta",
        "θ, θ̇, θ̈",
        "angle turned; angular velocity, angular acceleration",
        "rad, rad/s, rad/s²",
        "pendulum_page",
    ),
    Symbol(
        "J",
        "J",
        "moment of inertia about the axis it turns on (a pendulum's pivot; a disc's polar inertia in a drive train)",
        "kg·m²",
        "pendulum_page",
        "rotation",
    ),
    Symbol(
        "J_G",
        "J<sub>G</sub>, k<sub>G</sub>",
        "moment of inertia about the centre of mass G; the radius of gyration √(J<sub>G</sub>/m)",
        "kg·m², m",
        "pendulum_page",
        "rotation",
    ),
    Symbol(
        "k_theta",
        "k<sub>θ</sub>, c<sub>θ</sub>",
        "rotational stiffness and damping: restoring moment per radian, per rad/s",
        "N·m/rad, N·m·s/rad",
        "pendulum_page",
        "rotation",
    ),
    Symbol("M_moment", "M", "a moment (torque) applied about the axis", "N·m", "pendulum_page", "rotation"),
    Symbol("k_t", "k<sub>t</sub>", "a torsion spring's stiffness (a wire's twist)", "N·m/rad", "pendulum_page"),
    Symbol("d_arm", "d", "distance from the pivot to the centre of mass G", "m", "pendulum_page", "gravity-stiffness"),
    Symbol(
        "L_eq",
        "L<sub>eq</sub>",
        "equivalent length: the simple pendulum with the same period, J/md",
        "m",
        "pendulum_page",
        "gravity-stiffness",
    ),
    Symbol("r_disc", "r", "a disc's radius", "m", "pendulum_page"),
    Symbol(
        "T_period",
        "T, T<sub>0</sub>, ω",
        "a swing's period and angular frequency at amplitude A; T<sub>0</sub> = 2π/ω<sub>n</sub>, the small-swing period",
        "s, rad/s",
        "pendulum_page",
        "large-swings",
    ),
    Symbol("A_swing", "A", "a swing's amplitude: it turns between ±A", "rad, °", "pendulum_page", "large-swings"),
    Symbol("V_theta", "V", "potential energy (V(θ): as a function of the angle)", "J", "pendulum_page", "large-swings"),
    Symbol("K_ell", "K", "the complete elliptic integral of the first kind", "–", "pendulum_page", "large-swings"),
    Symbol(
        "k_3",
        "k<sub>3</sub>",
        "cubic stiffness: the Duffing oscillator's spring kx + k<sub>3</sub>x³ (k<sub>θ</sub>θ + k<sub>3</sub>θ³ turning)",
        "N/m³, N·m/rad³",
        "pendulum_page",
        "large-swings",
    ),
    Symbol("eta", "η", "loss factor of structural (hysteretic) damping; ≈ 2ζ at resonance", "–", "damping_page"),
    Symbol(
        "T_trans",
        "T",
        "transmissibility: the force (or motion) passed through, over that applied (T is also kinetic energy, below)",
        "–",
        "isolation_page",
    ),
    Symbol(
        "matrices", "M, C, K", "mass, damping and stiffness matrices: M ẍ + C ẋ + K x = f", "kg, N·s/m, N/m", "sim_page"
    ),
    Symbol("q", "q", "generalized (or modal) coordinates", "varies", "deriving_page"),
    Symbol("energies", "T, ℒ", "kinetic energy; the Lagrangian T − V", "J", "deriving_page"),
    Symbol("phi_i", "φ<sub>i</sub>, Φ", "mode shape i; the matrix of mode shapes (one per column)", "–", "sim_page"),
    Symbol("omega_i", "ω<sub>i</sub>", "natural frequency of mode i: K φ = ω² M φ", "rad/s", "sim_page"),
    Symbol("state", "z, A, B", "state z = [x, ẋ]; the state-space form ż = A z + B u", "–", "sim_page"),
    Symbol("lambda", "λ", "eigenvalue of A, σ ± iω<sub>d</sub>: decay rate and damped frequency", "1/s", "sim_page"),
    Symbol(
        "foh",
        "Φ, Γ<sub>0</sub>, Γ<sub>1</sub>, h",
        "the exact first-order-hold step z<sub>k+1</sub> = Φ z<sub>k</sub> + Γ<sub>0</sub> u<sub>k</sub> + Γ<sub>1</sub> u<sub>k+1</sub>; the step h",
        "–, s",
        "sim_page",
    ),
    Symbol(
        "Gamma_r",
        "Γ<sub>r</sub>, Γ<sub>r</sub>²",
        "mode r's participation factor φ<sub>r</sub><sup>T</sup>M1: how strongly a base motion drives it; its effective mass (all the modes' add up to the total)",
        "√kg, kg",
        "sim_page",
        "sweeps-and-ground-motion",
    ),
    Symbol("R", "R", "Rayleigh's quotient, an estimate of ω² from an assumed shape", "(rad/s)²", "energy_page"),
    Symbol(
        "H_jk",
        "H<sub>jk</sub>, α<sub>jk</sub>",
        "FRF (receptance) matrix entry: response at j to a force at k",
        "m/N",
        "frf_page",
    ),
    Symbol(
        "rho",
        "ρ",
        "density",
        "kg/m³",
        "continuum_page",
    ),
    Symbol("wavenumber", "κ, c", "wavenumber; wave speed (c also damping above)", "rad/m, m/s", "continuum_page"),
    Symbol(
        "MAC",
        "MAC",
        "modal assurance criterion: how alike two mode shapes are, 0 to 1",
        "–",
        "complex_page",
        "how-complex",
    ),
    Symbol("psd", "G, S", "power spectral density (one-sided G)", "unit²/Hz", "random_page"),
    Symbol("sigma", "σ", "RMS (standard deviation) of a random response", "as the response", "random_page"),
    Symbol(
        "Delta_f",
        "Δf, f<sub>s</sub>, N",
        "frequency resolution, sampling rate, record length in samples",
        "Hz, Hz, –",
        "signals_page",
    ),
    Symbol(
        "H1",
        "H1, H2",
        "FRF estimators from averaged spectra: G<sub>xf</sub>/G<sub>ff</sub> (unbiased by response noise) and G<sub>xx</sub>/G<sub>fx</sub> (by force noise)",
        "m/N",
        "test_page",
        "estimators",
    ),
    Symbol(
        "gamma2",
        "γ²",
        "coherence, |G<sub>xf</sub>|²/(G<sub>ff</sub>G<sub>xx</sub>) = H1/H2: the share of the response the force explains, 0 to 1",
        "–",
        "test_page",
        "estimators",
    ),
    Symbol(
        "SRS",
        "SRS",
        "shock response spectrum: each oscillator's peak response to a shock, against its f<sub>n</sub>",
        "as the response",
        "shock_page",
    ),
    Symbol(
        "f_c",
        "f<sub>c</sub>",
        "critical (coincidence) frequency: a plate's bending wave travels at the speed of sound",
        "Hz",
        "acoustics_page",
        "coincidence",
    ),
    Symbol(
        "rho_c",
        "ρc, k",
        "the air's characteristic impedance (415 rayl), sound's wavenumber ω/c",
        "Pa·s/m, rad/m",
        "acoustics_page",
        "efficiency",
    ),
    Symbol(
        "sigma_mn",
        "σ<sub>mn</sub>, σ",
        "radiation efficiency: W/ρcS⟨v²⟩ (a mode's, or a band's average; σ is also an RMS)",
        "–",
        "acoustics_page",
        "efficiency",
    ),
    Symbol("eta_rad", "η<sub>rad</sub>", "radiation loss factor, ρcσ/ωm″", "–", "acoustics_page", "driven"),
    Symbol("m_area", "m″", "a panel's mass per area, ρh", "kg/m²", "acoustics_page", "transmission"),
    Symbol(
        "TL",
        "TL, τ",
        "transmission loss, 10 log 1/τ; transmission coefficient",
        "dB, –",
        "acoustics_page",
        "transmission",
    ),
    Symbol(
        "beta_sea",
        "β",
        "two oscillators' coupling coefficient: P₁₂ = β(E₁ − E₂)",
        "1/s",
        "acoustics_page",
        "oscillators",
    ),
    Symbol(
        "eta_ij",
        "η<sub>ij</sub>, n<sub>i</sub>",
        "coupling loss factor from subsystem i to j; modal density (n<sub>i</sub>η<sub>ij</sub> = n<sub>j</sub>η<sub>ji</sub>)",
        "–, 1/Hz",
        "acoustics_page",
        "sea",
    ),
    Symbol(
        "tau_bar",
        "τ̄, c<sub>g</sub>",
        "a corner's diffuse-field transmission coefficient; a bending wave's group speed",
        "–, m/s",
        "acoustics_page",
        "junction",
    ),
    Symbol("Omega_spin", "Ω", "spin speed (a drive's frequency on the other pages)", "rad/s, rpm", "rotor_page"),
    Symbol(
        "I_p", "I<sub>p</sub>, I<sub>d</sub>", "polar and diametral moments of inertia of a disc", "kg·m²", "rotor_page"
    ),
    Symbol(
        "G_gyro", "G(Ω)", "gyroscopic matrix: its Ω G q̇ term splits forward and backward whirl", "kg·m²", "rotor_page"
    ),
    Symbol(
        "S_sommerfeld",
        "S",
        "Sommerfeld number of a fluid-film bearing, (μNL<sub>b</sub>D<sub>b</sub>/W)(R<sub>b</sub>/c<sub>r</sub>)²: the larger, the nearer the journal to the centre",
        "–",
        "rotor_page",
        "cross-coupling",
    ),
    Symbol("epsilon", "ε", "a journal's eccentricity ratio in its bearing", "–", "rotor_page"),
    Symbol("one_x", "1X, 2X", "once and twice per revolution", "–", "rotor_page"),
    Symbol(
        "mu",
        "μ",
        "a mass ratio (a tuned absorber), a friction coefficient, or the Mathieu pump's depth, as the page says",
        "–",
        "duffing_page",
    ),
    Symbol(
        "m_a",
        "m<sub>a</sub>, C<sub>a</sub>",
        "added mass: the fluid's kinetic energy over ½U²; over the fluid displaced, ρ<sub>f</sub>V",
        "kg, –",
        "fluid_page",
        "added-mass",
    ),
    Symbol(
        "rho_f", "ρ<sub>f</sub>, ρ<sub>s</sub>", "the fluid's and the body's densities", "kg/m³", "fluid_page", "wet"
    ),
    Symbol("R_o", "R<sub>o</sub>", "a concentric wall's radius round a body", "m", "fluid_page", "confined"),
    Symbol(
        "beta_avmi",
        "β",
        "a plate mode's AVMI factor: its added mass over its modal mass (also SEA's coupling coefficient)",
        "–",
        "fluid_page",
        "plates",
    ),
    Symbol(
        "m_L",
        "m<sub>L</sub>, m<sub>0</sub>, m<sub>n</sub>",
        "a tank's liquid, and the spring–mass analogue's rigid mass and mode n's sloshing mass",
        "kg",
        "fluid_page",
        "analogue",
    ),
    Symbol("zeta_d", "ζ<sub>d</sub>", "a tuned liquid damper's sloshing damping ratio", "–", "fluid_page", "damper"),
    Symbol(
        "St",
        "St",
        "Strouhal number: vortices shed from a cylinder at f<sub>s</sub> = St U/D, St ≈ 0.2",
        "–",
        "self_excited_page",
        "viv",
    ),
    Symbol(
        "I_principal",
        "I₁, I₂, I₃",
        "a body's principal moments of inertia (a symmetric top's I₁ = I₂ across its axis, I₃ along it)",
        "kg·m²",
        "gyroscope_page",
    ),
    Symbol(
        "J_top",
        "J (top)",
        "Jellett's constant L·(a e₃ + R ẑ): what survives a sliding top's friction",
        "kg·m³/s",
        "gyroscope_page",
        "jellett",
    ),
    Symbol("slip", "s", "an induction motor's slip, 1 − Ω/Ω<sub>s</sub>", "–", "torsional_page"),
)
BY_KEY = {s.key: s for s in SYMBOLS}
NOTATION = re.compile(r"<notation>(.*?)</notation>", re.S)
HEAD = '<tr><th align="left">Symbol</th><th align="left">Meaning</th><th align="left">Units</th>{more}</tr>'


def letters(symbol: Symbol) -> list[str]:
    """The separate symbols a row writes (``m, k, c``: three), as plain text."""
    text = re.sub(r"<[^>]+>", "", symbol.symbol)
    return [part.strip() for part in text.split(",") if part.strip()]


def notation_html(keys: list[str]) -> str:
    """A page's notation table: the symbols `keys`, in that order."""
    rows = []
    for key in keys:
        s = BY_KEY[key]
        rows.append(f"<tr><td>{maths.inline(s.symbol)}</td><td>{s.meaning}</td><td>{s.units}</td></tr>")
    return '<table cellspacing="0" cellpadding="3">' + HEAD.format(more="") + "".join(rows) + "</table>"


def glossary_html() -> str:
    """Every symbol, a table per course group, each with a link to the page that brings it in."""
    from . import links

    order = {page: i for i, (page, _, _) in enumerate(p for _, pages in links.COURSE for p in pages)}
    parts = []
    for group, pages in links.COURSE:
        names = {page for page, _, _ in pages}
        found = sorted((s for s in SYMBOLS if s.page in names), key=lambda s: order[s.page])
        if not found:
            continue
        anchor = re.sub(r"[^a-z0-9]+", "-", group.lower()).strip("-")
        rows = []
        for s in found:
            target = f"{s.page}#{s.anchor}" if s.anchor else s.page
            title = links.PAGE_TITLES[s.page].removesuffix(" page")
            rows.append(
                f"<tr><td>{maths.inline(s.symbol)}</td><td>{s.meaning}</td><td>{s.units}</td>"
                f'<td><a href="page:{target}">{title}</a></td></tr>'
            )
        parts.append(
            f'<h3><a name="{anchor}"></a>{group}</h3><table cellspacing="0" cellpadding="3">'
            + HEAD.format(more='<th align="left">Brought in on</th>')
            + "".join(rows)
            + "</table>"
        )
    return "\n".join(parts)
