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
        "complex frequency (x = e<sup>st</sup>; the Laplace variable); the poles s<sub>1</sub>, s<sub>2</sub> are the "
        "roots of m s² + c s + k = 0, −ζω<sub>n</sub> ± iω<sub>d</sub> below critical damping",
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
    Symbol("impulse", "I", "an impulse, ∫f dt: a hammer blow's", "N·s", "transient_page", "impulse-response"),
    Symbol(
        "h_t",
        "h(t)",
        "impulse response: the displacement after a unit impulse on the mass at rest",
        "m/(N·s)",
        "transient_page",
        "impulse-response",
    ),
    Symbol(
        "tau_conv",
        "τ",
        "the time of an earlier slice of force, in Duhamel's integral (τ is also a time constant, above)",
        "s",
        "transient_page",
        "convolution",
    ),
    Symbol(
        "laplace",
        "X(s), F(s)",
        "the Laplace transforms of x(t) and f(t); on s = iΩ, their Fourier transforms X(Ω), F(Ω)",
        "m·s, N·s",
        "transient_page",
        "laplace",
    ),
    Symbol(
        "H_s",
        "H(s)",
        "transfer function, X(s)/F(s) = 1/(m s² + c s + k): the Laplace transform of h(t)",
        "m/N",
        "transient_page",
        "transfer-function",
    ),
    Symbol(
        "step_metrics",
        "M<sub>p</sub>, t<sub>p</sub>, t<sub>s</sub>",
        "a step response's overshoot (a fraction of the static), its peak time and its settling time",
        "–, s, s",
        "transient_page",
        "step-response",
    ),
    Symbol(
        "T_pulse",
        "T, T<sub>n</sub>",
        "a pulse's length (a ramp's rise time); the natural period, 2π/ω<sub>n</sub>",
        "s",
        "transient_page",
        "pulses",
    ),
    Symbol(
        "r_pulse",
        "r",
        "a pulse's length in natural periods, T/T<sub>n</sub> (r is also a frequency ratio, above)",
        "–",
        "transient_page",
        "pulses",
    ),
    Symbol(
        "period",
        "T, f<sub>0</sub>, ω<sub>0</sub>",
        "a periodic force's period; its fundamental frequency, 1/T and 2π/T",
        "s, Hz, rad/s",
        "periodic_page",
        "fourier-series",
    ),
    Symbol(
        "j_harm", "j", "a harmonic's number: harmonic j is at jf<sub>0</sub>", "–", "periodic_page", "fourier-series"
    ),
    Symbol(
        "fourier",
        "a<sub>0</sub>, a<sub>j</sub>, b<sub>j</sub>",
        "a Fourier series' mean, and its cosine and sine coefficients",
        "as f",
        "periodic_page",
        "coefficients",
    ),
    Symbol(
        "C_j",
        "C<sub>j</sub>, X<sub>j</sub>",
        "harmonic j's complex amplitude: the force's, a<sub>j</sub> − ib<sub>j</sub>, and the response's, "
        "H(jω<sub>0</sub>)C<sub>j</sub>",
        "N, m",
        "periodic_page",
        "complex-form",
    ),
    Symbol("N_kept", "N", "the number of harmonics a series keeps", "–", "periodic_page", "gibbs"),
    Symbol("duty", "d", "the fraction of each period a rectangular pulse is on", "–", "periodic_page", "shapes"),
    Symbol(
        "state_period",
        "z, z<sub>0</sub>, z<sub>p</sub>, Φ<sub>T</sub>",
        "the state [x, ẋ]; where a steady period starts; where one period of the force leaves it from rest; "
        "what one period of free motion does to a state",
        "m and m/s, –",
        "periodic_page",
        "exact",
    ),
    Symbol(
        "W_d",
        "W<sub>d</sub>",
        "the energy a damper takes out in one cycle: the area of its loop",
        "J",
        "damping_page",
        "hysteresis-loop",
    ),
    Symbol("U_peak", "U", "the peak energy a spring stores in a cycle, ½K′X²", "J", "damping_page", "hysteresis-loop"),
    Symbol(
        "K_star",
        "K*, K′, K″",
        "complex (dynamic) stiffness, force over displacement in a harmonic cycle; its real part (storage) and "
        "imaginary part (loss)",
        "N/m",
        "damping_page",
        "viscous",
    ),
    Symbol(
        "eta",
        "η",
        "loss factor, K″/K′ = W<sub>d</sub>/2πU: structural (hysteretic) damping's; ≈ 2ζ at resonance",
        "–",
        "damping_page",
        "hysteretic",
    ),
    Symbol(
        "friction",
        "F<sub>f</sub>, c<sub>eq</sub>",
        "a dry friction force; the viscous damper that takes the same energy per cycle",
        "N, N·s/m",
        "damping_page",
        "coulomb",
    ),
    Symbol(
        "zener",
        "k<sub>0</sub>, k<sub>1</sub>, c<sub>1</sub>",
        "a Zener element's springs, relaxed and in its Maxwell arm, and that arm's dashpot",
        "N/m, N/m, N·s/m",
        "damping_page",
        "viscoelastic",
    ),
    Symbol(
        "tau_relax",
        "τ",
        "a viscoelastic element's relaxation time, c<sub>1</sub>/k<sub>1</sub> (τ is also a time constant, above)",
        "s",
        "damping_page",
        "viscoelastic",
    ),
    Symbol(
        "drag", "a", "a velocity-squared drag's coefficient: F = aẋ|ẋ|", "N·s²/m²", "damping_page", "equivalent-viscous"
    ),
    Symbol("P_loss", "P", "the mean power a vibration loses, ΩηU", "W", "damping_page", "measuring"),
    Symbol(
        "T_trans",
        "T",
        "transmissibility: the force (or motion) passed through, over that applied (T is also a pulse's length, above, and kinetic energy, below)",
        "–",
        "isolation_page",
    ),
    Symbol(
        "M_machine",
        "M, m<sub>2</sub>",
        "the mass on the mounts (a machine, with any block bolted to it); an intermediate mass between two stages",
        "kg",
        "isolation_page",
    ),
    Symbol(
        "delta_sag",
        "δ",
        "a mount's static deflection under its load (δ is also the log decrement, above)",
        "m",
        "isolation_page",
        "static-deflection",
    ),
    Symbol("y_floor", "y, Y", "the floor's (base's) motion; its amplitude", "m", "isolation_page", "transmissibility"),
    Symbol(
        "N_relax",
        "N",
        "a relaxation mount's series spring over its main spring, Nk (N is also a number of harmonics, above)",
        "–",
        "isolation_page",
        "relaxation",
    ),
    Symbol(
        "phi_turn",
        "φ, Ω̇",
        "the angle an unbalance has turned, ∫Ω dt; the rate its speed climbs (φ is also a phase, above)",
        "rad, rad/s²",
        "isolation_page",
        "start-up",
    ),
    Symbol(
        "matrices",
        "M, C, K",
        "mass, damping and stiffness matrices: M q̈ + C q̇ + K q = Q (M ẍ + C ẋ + K x = f for a chain)",
        "kg, N·s/m, N/m",
        "deriving_page",
        "lagrange",
    ),
    Symbol("q", "q", "generalized (or modal) coordinates", "varies", "deriving_page"),
    Symbol(
        "Q_gen",
        "Q<sub>i</sub>, δq<sub>i</sub>",
        "generalized force on q<sub>i</sub>: the work the applied forces do in a small change δq<sub>i</sub>, over "
        "δq<sub>i</sub>",
        "N or N·m",
        "deriving_page",
        "lagrange",
    ),
    Symbol("energies", "T, ℒ", "kinetic energy; the Lagrangian T − V", "J", "deriving_page"),
    Symbol(
        "D_diss",
        "D",
        "Rayleigh's dissipation function: half the sum of each damper's c × (its stretch rate)²",
        "W",
        "deriving_page",
        "lagrange",
    ),
    Symbol(
        "S_coords",
        "S, q′",
        "a change of coordinates q = S q′: new coordinates q′, the old ones S times them",
        "varies",
        "deriving_page",
        "lagrange",
    ),
    Symbol("G_P", "G, P", "a body's centre of mass; another point on it", "–", "deriving_page", "free-body"),
    Symbol(
        "x_G_P",
        "x<sub>G</sub>, x<sub>P</sub>, x<sub>f</sub>, x<sub>r</sub>",
        "the rise of G, of a point P, and of the body above the front and rear axles",
        "m",
        "deriving_page",
        "free-body",
    ),
    Symbol("F_sum", "ΣF", "the sum of the forces on a body", "N", "deriving_page", "free-body"),
    Symbol(
        "M_sum",
        "ΣM<sub>G</sub>, ΣM<sub>P</sub>, J<sub>P</sub>",
        "the sum of the moments about G, about P; the inertia about P, J<sub>G</sub> + me²",
        "N·m, kg·m²",
        "deriving_page",
        "free-body",
    ),
    Symbol("e_offset", "e, e*", "how far P is ahead of G; the elastic centre's e", "m", "deriving_page", "coupling"),
    Symbol(
        "car_axles",
        "a, b",
        "a car's front axle's distance ahead of G, its rear's behind",
        "m",
        "deriving_page",
        "bounce-pitch",
    ),
    Symbol(
        "k_fr",
        "k<sub>f</sub>, k<sub>r</sub>, c<sub>f</sub>, c<sub>r</sub>",
        "the front and rear suspension's stiffness and damping, per axle",
        "N/m, N·s/m",
        "deriving_page",
        "bounce-pitch",
    ),
    Symbol(
        "omega_i",
        "ω<sub>r</sub>, f<sub>r</sub>",
        "natural frequency of mode r: a root of det(K − ω²M) = 0; f<sub>r</sub> = ω<sub>r</sub>/2π",
        "rad/s, Hz",
        "deriving_page",
        "bounce-pitch",
    ),
    Symbol(
        "ell_node", "ℓ", "a mode's node: the point that stays still, ℓ ahead of G", "m", "deriving_page", "bounce-pitch"
    ),
    Symbol(
        "road",
        "y<sub>f</sub>, y<sub>r</sub>, v",
        "the road's rise under the front and rear wheels; the car's speed",
        "m, m/s",
        "deriving_page",
        "road",
    ),
    Symbol(
        "m_ends",
        "m<sub>f</sub>, m<sub>r</sub>",
        "the body's mass shared to each end, mb/L and ma/L",
        "kg",
        "deriving_page",
        "coupling",
    ),
    Symbol(
        "pendulum2",
        "m<sub>1</sub>, m<sub>2</sub>, L<sub>1</sub>, L<sub>2</sub>, θ<sub>1</sub>, θ<sub>2</sub>",
        "a double pendulum's bobs, its rods, and each rod's angle from the vertical",
        "kg, m, rad",
        "deriving_page",
        "double-pendulum",
    ),
    Symbol(
        "phi_bend",
        "φ",
        "the double pendulum's elbow bend, θ<sub>2</sub> − θ<sub>1</sub> (a phase on other pages)",
        "rad",
        "deriving_page",
        "double-pendulum",
    ),
    Symbol(
        "chain_i",
        "m<sub>i</sub>, k<sub>i</sub>, c<sub>i</sub>, x<sub>i</sub>",
        "the chain's mass i; the spring and damper between it and the mass before (the first's, the wall); its "
        "displacement",
        "kg, N/m, N·s/m, m",
        "sim_page",
        "equations-of-motion",
    ),
    Symbol("N_dof", "N", "the number of masses in the chain (its degrees of freedom)", "–", "sim_page"),
    Symbol("identity", "I", "the identity matrix", "–", "sim_page"),
    Symbol(
        "phi_i",
        "φ<sub>r</sub>, Φ",
        "mode shape r, mass-normalized (φ<sub>r</sub>ᵀMφ<sub>r</sub> = 1); the matrix of mode shapes (one per column)",
        "1/√kg",
        "sim_page",
        "undamped-normal-modes-real",
    ),
    Symbol("C_m", "C<sub>m</sub>", "the modal damping matrix, ΦᵀCΦ", "1/s", "sim_page", "undamped-normal-modes-real"),
    Symbol(
        "gamma_step",
        "γ<sub>r</sub>",
        "a uniform chain's mode r: the phase step from one mass to the next, (2r − 1)π/(2N + 1)",
        "rad",
        "sim_page",
        "uniform-chain",
    ),
    Symbol(
        "ritz",
        "W, a",
        "Rayleigh–Ritz: the guessed shapes, one per column; their weights, x = Wa",
        "–",
        "sim_page",
        "rayleigh-quotient",
    ),
    Symbol(
        "alpha_beta",
        "α, β",
        "Rayleigh damping's coefficients, C = αM + βK",
        "1/s, s",
        "sim_page",
        "proportional-damping",
    ),
    Symbol(
        "x_g",
        "x<sub>g</sub>, X<sub>g</sub>, u",
        "the ground's (wall's) motion; its amplitude; the masses' motion relative to it, x − x<sub>g</sub>1",
        "m",
        "sim_page",
        "sweeps-and-ground-motion",
    ),
    Symbol(
        "ones",
        "1, e<sub>1</sub>",
        "a vector of ones (every mass moving together); one with a 1 for mass 1 and 0 for the rest",
        "–",
        "sim_page",
        "sweeps-and-ground-motion",
    ),
    Symbol(
        "N_1",
        "N<sub>1</sub>",
        "the tension in element 1, its spring and damper together",
        "N",
        "sim_page",
        "sweeps-and-ground-motion",
    ),
    Symbol(
        "absorber",
        "m<sub>a</sub>, k<sub>a</sub>, c<sub>a</sub>, ζ<sub>a</sub>, ω<sub>a</sub>, ω<sub>1</sub>",
        "a vibration absorber's mass, spring and damper; its damping ratio c<sub>a</sub>/2m<sub>a</sub>ω<sub>1</sub>; "
        "its own frequency; the primary's alone, √(k<sub>1</sub>/m<sub>1</sub>)",
        "kg, N/m, N·s/m, –, rad/s, rad/s",
        "sim_page",
        "vibration-absorbers-and-tuned-mass-dampers",
    ),
    Symbol(
        "F_s",
        "F<sub>s</sub>",
        "static friction: the most a stuck mass's friction can hold",
        "N",
        "sim_page",
        "friction",
    ),
    Symbol(
        "R",
        "R, v",
        "Rayleigh's quotient vᵀKv/vᵀMv of a guessed shape v: an estimate of ω², never below ω<sub>1</sub>²",
        "(rad/s)²",
        "sim_page",
        "rayleigh-quotient",
    ),
    Symbol(
        "mu",
        "μ",
        "a mass ratio (a tuned absorber's, m<sub>a</sub>/m<sub>1</sub>), a friction coefficient, or the Mathieu "
        "pump's depth, as the page says",
        "–",
        "sim_page",
        "vibration-absorbers-and-tuned-mass-dampers",
    ),
    Symbol(
        "state", "A, B", "the state-space form ż = A z + B u, for the state z = [x, ẋ] of every mass", "–", "sim_page"
    ),
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
    Symbol(
        "eps_r",
        "ε<sub>r</sub>",
        "how much of mode r a guessed shape holds, against mode 1's: v = φ<sub>1</sub> + Σ ε<sub>r</sub>φ<sub>r</sub>",
        "–",
        "energy_page",
        "rayleigh",
    ),
    Symbol(
        "flex",
        "K<sup>−1</sup>, (K<sup>−1</sup>)<sub>ii</sub>",
        "flexibility matrix: column i is the deflection under 1 N on mass i; its diagonal entry is how far "
        "mass i itself moves",
        "m/N",
        "energy_page",
        "dunkerley",
    ),
    Symbol(
        "omega_ii",
        "ω<sub>ii</sub>",
        "mass i alone on the massless chain: its natural frequency, 1/√(m<sub>i</sub>(K<sup>−1</sup>)<sub>ii</sub>)",
        "rad/s",
        "energy_page",
        "dunkerley",
    ),
    Symbol(
        "holzer",
        "N<sub>i</sub>",
        "Holzer's table: the tension in spring i at a trial ω (x<sub>1</sub> = 1); N<sub>N+1</sub>, past the last "
        "mass, is the residual, zero at a natural frequency",
        "N",
        "energy_page",
        "holzer",
    ),
    Symbol(
        "H_jk",
        "H<sub>jk</sub>, α<sub>jk</sub>",
        "FRF (receptance) matrix entry: response at j to a force at k",
        "m/N",
        "frf_page",
    ),
    Symbol(
        "modal_const",
        "φ<sub>jr</sub>, φ<sub>jr</sub>φ<sub>kr</sub>",
        "mode r's entry at mass j; the modal constant, how much mode r is excited at k times how much it shows at j",
        "1/√kg, 1/kg",
        "frf_page",
        "sum-of-modes",
    ),
    Symbol(
        "residuals",
        "U, L",
        "residual flexibility (the modes above a band, as springs) and residual mass term (the modes below, −L/Ω²)",
        "m/N, 1/kg",
        "frf_page",
        "truncation",
    ),
    Symbol(
        "residue",
        "R<sub>r</sub>",
        "complex mode r's residue matrix: its term in H is R<sub>r</sub>/(iΩ − λ<sub>r</sub>) plus its conjugate's",
        "1/kg·s",
        "frf_page",
        "non-proportional-damping",
    ),
    Symbol(
        "a_sr",
        "a<sub>sr</sub>",
        "how much of undamped mode s a complex mode r holds, to first order in the damping",
        "–",
        "complex_page",
        "complex-shapes",
    ),
    Symbol(
        "psi",
        "ψ<sub>r</sub>",
        "complex mode r's exact shape: the displacements of the state matrix's eigenvector [ψ; λψ]",
        "–",
        "complex_page",
        "complex-shapes",
    ),
    Symbol(
        "eta_state",
        "η, Y, y<sub>r</sub>",
        "the state's modal coordinates, Y<sup>−1</sup>z; Y the eigenvectors y<sub>r</sub> = [ψ<sub>r</sub>; "
        "λ<sub>r</sub>ψ<sub>r</sub>] as columns (η is also a loss factor)",
        "–",
        "complex_page",
        "state-space-coordinates",
    ),
    Symbol(
        "mpc",
        "MPC, MPD, s<sub>xx</sub>, θ<sub>j</sub>",
        "modal phase collinearity (1 real, 0 spread evenly) from the sums s<sub>xx</sub>, s<sub>yy</sub>, "
        "s<sub>xy</sub>; mean phase deviation, the entries' angles θ<sub>j</sub> from the best-fit line",
        "–, °",
        "complex_page",
        "how-complex",
    ),
    Symbol(
        "C_p",
        "C<sub>p</sub>",
        "the proportional approximation's damping matrix, MΦ diag(C<sub>m</sub>) ΦᵀM",
        "N·s/m",
        "complex_page",
        "ignoring-it",
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
