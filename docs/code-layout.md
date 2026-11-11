[Contents](../README.md#contents)

# Code layout

- `core/model.py`: assembles $M$, $C$, $K$ and the state-space matrices, and the force in each element.
- `core/modal.py`: classical modes (`scipy.linalg.eigh`), all 2N state-space eigenpairs
  (`scipy.linalg.eig`), the MAC pairing, the modal-coordinate map, and the frequency response.
- `core/frf_matrix.py`: the full receptance matrix and its modal (pole–residue) terms.
- `core/describing.py`: the describing-function (harmonic balance) response with Coulomb
  friction: each friction force's first harmonic, solved per frequency for the sliding and
  stuck masses.
- `core/simulator.py`: the exact first-order-hold time stepper (piecewise, with stick-slip friction) and its exact energy ledger.
- `core/presets.py`: the preset chains, vibration absorbers and Den Hartog's tuning.
- `core/energy.py`: kinetic, potential and stored energy, and the energy in each mode.
- `core/measurement.py`: the virtual modal test: excitation signals, a fast exact response
  (the FOH update diagonalized into one first-order filter per eigenvalue, or the simulator's
  stick-slip stepping with friction), anti-alias
  filtering and sampling, noise, windows, and the H1/H2 and coherence estimates.
- `core/oma.py`: operational modal analysis: the ambient record (random forces at every mass,
  the exact response, anti-alias filter and sensor noise), the cross-spectral matrix, frequency
  domain decomposition and enhanced FDD's bell and autocorrelation fit, covariance-driven
  stochastic subspace identification, the harmonic test (filtered kurtosis), and a hammer test
  through the virtual modal test's chain for comparison.
- `core/random_vibration.py`: random vibration: band-limited force or base-acceleration PSDs,
  the response's PSD H G Hᴴ and cumulative RMS, Miles' equation mode by mode, a record synthesized
  to the PSD and stepped by the exact FOH update, and Welch's estimate.
- `core/complex_modes.py`: non-proportional damping: the complex modes' MPC and MPD, the
  proportional approximation's damping, poles and FRF against the exact ones, the free decay
  from an undamped mode's shape, a damper's or spring's root locus, and LSFD with real and with
  complex modes.
- `core/model_updating.py`: model updating and correlation: the model as the structure with errors,
  the test (the virtual modal test's or the exact modes with noise) at chosen sensors, the MAC and
  its pairs, frequency errors, the auto-MAC and COMAC; frequency sensitivities in closed form, and
  Gauss–Newton updating of masses, springs and dampers with regularization.
- `core/nonlinear_test.py`: stepped sine up and down by one-harmonic harmonic balance (describing functions
  of a cubic spring, a gap and Coulomb friction; jumps where a branch ends), homogeneity, and the steady
  response in time (the exact FOH update with the nonlinear forces by Newton) and its harmonics.
- `core/force_id.py`: force identification: G from the FOH update's Markov parameters (displacement or
  acceleration sensors), its SVD, the plain inverse, truncated SVD and Tikhonov, the L-curve's corner and the
  discrepancy principle.
- `core/test_planning.py`: sensor placement by effective independence (and weighted by the driving-point
  residue), its Fisher information, noise and auto-MAC; operational deflection shapes and each mode's
  share of them.
- `core/shock.py`: shock response spectra: the classical pulses, measured records and syntheses to a
  target (damped sines or wavelets), the oscillators stepped by the exact first-order-hold update
  (primary, residual, positive and negative peaks), and the chain on a shaken base, its peaks from the
  SRS mode by mode (ABS, SRSS, NRL) against the exact response.
- `core/identification.py`: modal parameter extraction from a measured FRF column: peak
  picking, circle fit, LSCF with its stabilization diagram, LSFD, and matching to the exact
  modes (MAC).
- `core/substructure.py`: component mode synthesis (Craig–Bampton, Rubin, MacNeal), mode
  comparison (frequency error, MAC) and the reduced-model FRF.
- `core/modal_coupling.py`: two subsystems reduced to one oscillator each and joined as a 2-DOF
  model (effective, residual and tip modal masses), its comparison with the chain, the
  drive-point FRFs, and the frequency and mass-ratio sweeps.
- `core/cms_response.py`: the full chain and a reduced model stepped side by side under the
  same force (MacNeal's massless boundary condensed statically).
- `core/back_expansion.py`: recovery of the interior DOFs from the coupled solution (by T,
  from the boundary alone, and by re-solving each interior with the boundary motion imposed,
  stepped in the same exact update) and of the spring forces.
- `core/frame.py`: a plane frame or truss of bars, Euler–Bernoulli or Timoshenko beams, springs,
  dampers and point masses: each element's local matrices and T, assembly, supports, unused and massless
  DOFs (condensed), mechanisms, the modes and their strain-energy split, and presets.
- `core/frame_response.py`: the frame's forced response two ways, the conventional real modes (ΦᵀCΦ's
  diagonal) and the state space (a damper on a massless DOF kept as a first-order state): Rayleigh damping,
  the complex modes paired with the undamped ones, FRFs, and the exact response to a harmonic force, an
  impulse or a step from rest.
- `core/rotor.py`: the Jeffcott rotor's model: shaft (beam elements: massless and condensed to the
  stations, or with mass and reduced to the stations and its fixed-interface modes), disc,
  bearings and the gyroscopic matrix, the 1X loads (unbalance, the disc's skew, the shaft's bow), an
  asymmetric shaft's turning stiffness, a cracked shaft (`Crack`: a hinge at the crack, breathing open
  under the weight), oil films linearized about each journal's equilibrium, and magnetic bearings with
  their controllers' states in the state vector.
- `core/rotor_modes.py`: its whirl: the modes at a speed (an asymmetric or cracked shaft's Floquet modes,
  in axes turning with the shaft or by Hill's method), the Campbell diagram and critical speeds, and the
  steady states: the 1X response, harmonic balance (`harmonic_response`, `gravity_response`), and every
  harmonic together (`steady_harmonics`, 1X–3X).
- `core/rotor_simulator.py`: its time response: an exact FOH stepper that follows a speed ramp, with the
  turning stiffness as a mid-step kick, the films stepped with their whole force (`nonlinear_films`), the
  stator ring's rub and the thermal bow.
- `core/fluid_bearing.py`: the plain journal bearing by short-bearing (Ocvirk) theory: its
  equilibrium, the eight stiffness and damping coefficients in closed form, and the film's whole
  force, in closed form (`force`) and by quadrature (`film_force`) to check it.
- `core/magnetic_bearing.py`: the active magnetic bearing: the magnets' negative stiffness and
  current gain, the PID with a filtered derivative behind a first-order amplifier, its transfer
  function (`gain`) and its state space.
- `core/rotor_sweep.py`: run-up and coast-down: the steady-state 1X (and gravity's 2X) vectors at each station,
  once-per-revolution tracking of the 1X and 2X vectors and peak (`RevolutionTracker`), and
  `RunUpCoastDown`, which drives the stepper through a sweep and records each leg.
- `core/rotor_balance.py`: single- and two-plane balancing: influence coefficient and matrix,
  correction weights, and `SteadyMeasurement`, which runs the stepper at a speed, lets it settle and averages the
  1X vector over some revolutions.
- `core/rotor_analysis.py`: what the rotor page works out when its parameters change (the
  Campbell diagram, the stability onset, the steady states: `analyse`), and `AnalysisWorker`,
  which runs it in a process of its own so the animation keeps going meanwhile.
- `core/rotor_presets.py`: the rotor page's presets, each a `RotorSystem` with a speed range
  and a note.
- `core/rotor_stability.py`: the least damped mode, the onset speed of instability (swept,
  then bisected, with the end of the band if it ends), the log decrement, the full spectrum of
  an orbit, and `AsymmetryChart`, the growth rate against asymmetry and speed, filled in a few
  cells at a time.
- `core/rotor_waterfall.py`: `Waterfall`, full spectra of the orbits taken block by block as
  samples are fed in, each with the free whirl frequencies at its speed, and the latest run-up
  or coast-down among them.
- `core/duffing.py`: the Duffing oscillator: harmonic balance's response curve (traced by
  amplitude, unstable branch included) with its folds and backbone, the pumped spring's
  half-harmonic response and first tongue, the Floquet growth rate over the Ince–Strutt chart, the
  swept drive, an exact FOH stepper with the cubic's and the pump's forces found by Newton each
  step, each drive cycle's first and half harmonics, and the page's presets.
- `core/torsional.py`: torsional drive trains: discs, shafts and gear meshes (with transmission
  error), the modes and their twist, the train referred to one shaft and with rigid gears, the
  orders and torsional critical speeds, the steady forced response, and the page's presets.
- `core/torsional_startup.py`: a motor starting the train: the induction motor's curve and
  switch-on transient, the synchronous motor's 2× slip pulsation and field, a load rising as speed
  squared, the exact FOH stepper with the two torques found by Newton each step, the steady running
  torques, and the start-up presets.
- `core/continuum.py`: a string, a bar or an Euler–Bernoulli beam: the exact modes (the beam's from
  its ends' frequency equation), meshes of N elements with lumped, consistent or mixed mass, their
  dispersion (in closed form, or a node's Bloch wave), each mode's error and its convergence with
  N, and the page's presets.
- `core/continuum_waves.py`: a pulse on the continuum (a digital waveguide carrying d'Alembert's
  waves, with the ends, a dashpot and a step scattering them in closed form, or a beam's exact
  modes) and on the mesh (stepped by its exact matrix exponential), and the wave presets.
- `core/pendulum.py`: pendulums (simple, a rod, a torsion disc, inverted): the exact period by
  quadrature, Duffing's estimate, the swing by `solve_ivp`, the phase plane's orbits, and a rod's
  best pivot.
- `core/periodic.py`: periodic forces as Fourier series in closed form, each harmonic's response, and
  the exact periodic steady state from one FOH-stepped period.
- `core/bounce_pitch.py`: a car's body in bounce and pitch: Lagrange's M, C and K in three sets of
  coordinates (T = the transformation to G's), the free-body diagram's rows and the slip about P, the
  modes and their nodes, and a ride over a bump stepped exactly.
- `core/double_pendulum.py`: a double pendulum: its whole equations (Lagrange's), small-swing M and K in
  the rods' angles or the elbow's bend, the free-body rows and how to mix them, the modes, a release
  stepped whole beside the linear model, and two releases drifting apart.
- `core/energy_methods.py`: Rayleigh's quotient, inverse iteration, Dunkerley's sum, Rayleigh–Ritz and
  Holzer's table on the chain.
- `core/beams_plates.py`: a simply supported beam by Euler–Bernoulli, Rayleigh, shear and Timoshenko
  theory, wave speeds, rectangular and circular plates' modes and shapes, and mode counts.
- `core/integration.py`: central difference, Newmark, HHT-α, RK4, explicit Euler and the exact update
  on one or two masses, and each scheme's amplification matrix (spectral radius, period error,
  numerical damping).
- `core/fatigue.py`: a stress PSD of one or two peaks, its spectral moments, rainflow counting (ASTM
  E1049), Basquin and Miner, and the narrow-band and Dirlik estimates.
- `core/quarter_car.py`: a quarter-car on an ISO 8608 road: the transfer functions, the outputs' PSDs
  and RMS, the comfort–grip trade-off, and a drive stepped exactly.
- `core/rigid_mounts.py`: a box on mounts with six degrees of freedom: K = Σ Tᵀ K<sub>m</sub> T, the
  modes and their kinetic-energy split, the elastic centre (overall and each plane's), the focal tilt, and
  the forced response to an unbalance or a torque (each mount's force, modal forces, the torque roll axis).
- `core/seismic.py`: a design spectrum, the building's modal masses and peaks, ABS/SRSS/NRL/CQC, and
  synthesized records stepped mode by mode.
- `core/radiation.py`: a baffled panel's sound: the critical frequency, each mode's radiation efficiency
  (Wallace's integral), their band average against Maidanik's, the point-driven panel's input and radiated
  power, and the infinite plate's transmission loss (mass law, coincidence dip).
- `core/variability.py`: Monte Carlo on the chain with lognormal scatter and the first-order estimate,
  and a mistuned bladed ring's modes, localization and forced response.
- `core/self_excited.py`: van der Pol, galloping, a wake oscillator and two-mode flutter: their
  stability, limit cycles by averaging, and time responses.
- `core/active_control.py`: velocity feedback on the chain (the closed loop with lags), the modal
  estimate, tracked damping, a tuned mass damper to compare, and skyhook isolation.
- `core/diagnostics.py`: a gearbox's synthesized vibration (mesh, gear and bearing faults), a run-up
  with order tracking by angular resampling, envelope analysis and the cepstrum.
- `gui/`: the PySide6 and pyqtgraph interface. It runs on a ~60 fps timer that advances
  the simulator by wall-clock time × speed. `gui/style.py` holds the colour themes; every
  widget reads its colours from the current one (`colors.mass[i]`, `colors.force`) and
  redraws in a new one through its `apply_theme()`. Log axes use `gui/axes.py`, which labels
  only as many ticks as fit.

To measure the frame rate on your own display, start the app with `VIB_FRAME_STATS=1`
(e.g. `VIB_FRAME_STATS=1 uv run vib-tutorial`): an overlay in the corner shows the running
page's frames per second, the time each frame spends computing and painting (median and 95th
percentile) against the display's refresh, the share of late frames, and the simulation's speed
against real time. `VIB_FRAME_STATS=frames.csv` also logs every frame to that file, with
what was on screen (page, sub-tab, rotor preset, window size, display scale and refresh rate)
and what you clicked, typed or picked in between. `uv run python scripts/frame_report.py
frames.csv` then summarizes it by view and input: frames per second, computing and drawing
time for each.

`scripts/readme_screenshots.py` regenerates the images in `docs/images` (offscreen, each in
the state its caption describes): `uv run python scripts/readme_screenshots.py`, or name the
shots to retake, e.g. `... chirp pluck`.

The `core` package has no Qt dependency, so you can use it from scripts or notebooks:

```python
from vib_tutorial.core import ChainSystem, modal_analysis

chain = ChainSystem([1.0] * 4, [400.0] * 4, [15.0, 2.0, 2.0, 2.0])
res = modal_analysis(chain)
[(m.fn_hz, m.zeta_modal, m.damped.zeta) for m in res.modes]  # classical, N modes
[(m.eigenvalue, m.shape) for m in res.complex_modes]  # state-space, 2N modes

import numpy as np
from vib_tutorial.core import Simulator, modal_coordinate_map

sim = Simulator(chain)
sim.set_displacement(0.01 * res.modes[2].shape)  # release classical mode 3
t, x, v, f = sim.advance(2.0)  # per-step samples
z = np.hstack([x, v])
q = z @ modal_coordinate_map(chain, res)  # classical, (steps, N), m
eta = z @ modal_coordinate_map(chain, res, complex_modes=True)  # one per pair, m

from vib_tutorial.core import modal_energies

E = modal_energies(chain, res, x, v)  # (steps, N), J; rows sum to T + V
sim.energy_added + sim.work - sim.dissipated - sim.stored_energy  # ~1e-15 J: exact balance
added, work, dissipated, friction = sim.ledger.T  # the same ledger at each step of the last advance

from vib_tutorial.core import ForceController, ForceKind, ForceSettings, element_forces, pluck_shape, transmissibility

spring, damper = element_forces(chain, x, v)  # (steps, N) tension, N
sim.set_displacement(pluck_shape(chain, 1, 0.02))  # hold m2 at 20 mm, then let go

shake = ForceController(
    ForceSettings(kind=ForceKind.CHIRP, sweep_start_hz=0.5, sweep_end_hz=8.0, base=True, base_amplitude=0.005)
)  # 5 mm ground sweep
sim = Simulator(chain, shake)
shake.switch_on()
t, x, v, xg = sim.advance(2.0)  # xg: the ground motion, m
transmissibility(chain, np.array([0.5, 2.0]))  # X / X_g, (freqs, N)

from vib_tutorial.core.presets import den_hartog, tuned_mass_damper, with_absorber

tmd = tuned_mass_damper(ChainSystem.uniform(4), mu=0.05)  # 5-mass chain, damper on m4 tuned to mode 1
den_hartog(0.05)  # (f_a / f_1, zeta_a on f_1) = (0.952, 0.127)

from vib_tutorial.core import compare_modes, craig_bampton

s = ChainSystem.uniform(8)
cb = craig_bampton(s, interfaces=[3], n_kept=[1, 1])  # cut at m4, 1 mode each
[(c.fn_true, c.fn_red, c.mac) for c in compare_modes(cb, modal_analysis(s))]

from vib_tutorial.core import compare_coupled, coupled_model
from vib_tutorial.core.modal_coupling import coupled_frf, sweep

pair = coupled_model(chain, split=2)  # A = m1-m2, B = m3-m4, mode 1 of each
[(c.fn, c.fn_full, c.error) for c in compare_coupled(pair, res)]  # 2-DOF vs chain
coupled_frf(pair, np.array([1.0, 2.0]), "free")  # x4 / F4 of the 2-DOF model
sweep(pair, "mass").two_dof  # (points, 2) Hz as mu is swept

from vib_tutorial.core import free_interface
from vib_tutorial.core.cms_response import CMSResponse

rubin = free_interface(s, interfaces=[3], n_kept=[1, 1])  # B keeps its rigid-body mode
macneal = free_interface(s, [3], [1, 1], residual_mass=False)  # massless residual: 2 modes

step = ForceController(ForceSettings(kind=ForceKind.STEP, amplitude=10.0))
both = CMSResponse(cb, step)  # full and reduced, same force at the tip
step.switch_on()
t, x_full, x_cb, f = both.advance(2.0)  # (steps, N) each; x_cb = T r

from vib_tutorial.core import Acquisition, Excitation, FrfEstimator, MeasurementSettings, Processing, Window

acq = Acquisition(
    chain, MeasurementSettings(excitation=Excitation.RANDOM, input_dof=3, fs=20.0, block=1024, averages=20, overlap=0.5)
)
while not acq.done:
    acq.step()  # one block per step
est = FrfEstimator(acq, Processing(window=Window.HANN, response_noise=0.02)).estimate()
est.freqs, est.H, est.coherence  # (F,), (F, N), (F, N)

from vib_tutorial.core.identification import auto_select, lscf, lsfd, match_modes

band = (0.0, acq.settings.band)
stab = lscf(est.freqs, est.H, band, max_order=30)  # poles at every order
ident = lsfd(est.freqs, est.H, band, [stab.pole(o, i) for o, i in auto_select(stab)])
[(r.mode, r.identified and r.identified.fn_hz, r.mac) for r in match_modes(ident.modes, res, 8.0)]

from vib_tutorial.core.rotor import STEEL, Bearing, RotorSystem
from vib_tutorial.core.rotor_modes import campbell, synchronous_response
from vib_tutorial.core.rotor_simulator import RPM, RotorSimulator

rotor = RotorSystem(position=0.3, bearing_a=Bearing(kx=2e4, ky=8e4))  # disc off centre, A anisotropic
M, C, G, K = rotor.matrices()  # 8x8; M q'' + (C + ΩG) q' + (K + ΩH) q = f
[(w * RPM, whirl) for w, whirl, zeta in campbell(rotor, 1000.0).criticals]  # rpm, +1 forward
Q = synchronous_response(rotor, np.array([150.0]))  # steady whirl at 150 rad/s, complex (1, 8)
bent = rotor.with_(bow=10e-6, bow_angle=np.pi, skew=np.radians(0.1))  # bow against the heavy spot, disc skewed
weighted = rotor.with_(weights=(2e-5 * np.exp(1j * np.pi),))  # a 20 g·mm balance weight at 180° from the mark
rotor.with_(gravity=9.81).static_deflection()  # sag under its own weight, (8,)
heavy = rotor.with_(density=STEEL, elements=8)  # the shaft's own mass: beam elements
heavy.n_dof, heavy.fixed_interface_freqs  # (12, ...): q = 8 stations + 2 shaft modes per plane
spin = RotorSimulator(rotor)
spin.target = 3000 / RPM  # ramps at spin.accel (rad/s²)
r = spin.advance(0.25)  # r.t, r.q (steps, 8), r.omega, r.phase

from vib_tutorial.core.rotor_sweep import RunUpCoastDown, probe, steady_vectors

V = steady_vectors(rotor, np.linspace(10.0, 300.0, 500))  # 1X vectors (500, 4 stations, x/y), complex
spin.accel = 1000 / RPM  # 1000 rpm/s
up, down = RunUpCoastDown(spin, 800 / RPM, 2400 / RPM).run()  # each leg tracked once per revolution
w, v, peak = up.trace(1, "x")  # disc, x probe: speed, 1X vector, max |x|
up.peak(1, "orbit")  # (largest 1X major axis, speed it came at)

from vib_tutorial.core.rotor_modes import gravity_response
from vib_tutorial.core.rotor_stability import full_spectrum, stability_onset

unstable = RotorSystem(internal_damping=20.0, bearing_a=Bearing(kxy=5e3))  # H = unstable.circulatory()
onset = stability_onset(unstable, 1000.0)  # Onset(omega, freq_hz, whirl) or None
whip = RotorSystem(bearing_a=Bearing(swirl=0.5), bearing_b=Bearing(swirl=0.5))  # k_xy grows as cΩ/2
stability_onset(whip, 1000.0).order  # 0.5: the whirl at half the speed at onset
overhung = RotorSystem(position=1.2)  # the disc 0.2 L past bearing B
f, amp = full_spectrum(r.t, r.q[:, [1, 5]])  # disc orbit: +f forward, −f backward
two_pole = RotorSystem(asymmetry=0.15, gravity=9.81)  # EI(1 ± 0.15), stiff axis at the mark
band = stability_onset(two_pole, 300.0)  # unstable from band.omega to band.end (rad/s)
q0, q2 = gravity_response(two_pole, np.array([82.2]))  # mean sag, and the 2X whirl near half the critical

from vib_tutorial.core.fluid_bearing import JournalBearing

oil = Bearing(film=JournalBearing())  # a plain journal bearing (short-bearing theory)
upright = RotorSystem(bearing_a=oil, bearing_b=oil, nonlinear_films=True)  # stepped with the films' whole force
JournalBearing().force(5e-6, -8e-6, 0.0, 1e-3, 300.0)  # the whole film's force (N) at any position, velocity

from vib_tutorial.core.magnetic_bearing import MagneticBearing
from vib_tutorial.core.rotor import Crack
from vib_tutorial.core.rotor_modes import steady_harmonics

floating = RotorSystem(bearing_a=Bearing(magnet=MagneticBearing()), bearing_b=Bearing(magnet=MagneticBearing()))
floating.n_states  # 2 n_dof + the controllers' states
MagneticBearing().dynamic_stiffness(2j * np.pi * 50.0)  # k_i G(s) - k_m at 50 Hz (N/m, complex)
cracked = RotorSystem(crack=Crack(position=0.5, depth=0.4), gravity=9.81, unbalance=5e-6)
q0, v = steady_harmonics(cracked, np.array([400 / RPM]))  # mean sag, and the 1X, 2X, 3X vectors

from vib_tutorial.core.rotor_waterfall import Waterfall

wf = Waterfall(0.5, 100.0, rotor)  # 0.5 s blocks, |f| up to 100 Hz
wf.feed(r.t, r.q[:, 1] + 1j * r.q[:, 5], r.omega)  # z = x + iy, (k, stations); returns new blocks
[(b.omega * RPM, b.natural) for b in wf.latest_ramp(up=True)]  # each block's speed, free whirl lines (Hz)

from vib_tutorial.core.rotor_balance import SteadyMeasurement, correction, influence, weight

v0 = probe(SteadyMeasurement(spin, 1200 / RPM).run()[1], "x")  # disc, x probe: settled 1X reading
trial = weight(0.5e-3, 0.02, 0.0)  # 0.5 g at 20 mm, at the mark (kg·m)
spin.set_system(rotor.with_(weights=(trial,)))
v1 = probe(SteadyMeasurement(spin, 1200 / RPM).run()[1], "x")
W = correction(v0, influence(v0, v1, trial))  # the weight that cancels v0 (kg·m)

# Two planes on the disc's faces (PlaneWeight(z, w, disc=True)), probes at A and B (x).
from vib_tutorial.core.rotor import PlaneWeight
from vib_tutorial.core.rotor_balance import corrections, influence_matrix

skewed = rotor.with_(position=0.3, skew=np.radians(0.5))
planes = [PlaneWeight(skewed.a - 0.02, 0j, disc=True), PlaneWeight(skewed.a + 0.02, 0j, disc=True)]


def read(s):  # steady 1X readings at the two bearings
    v = steady_vectors(s, np.array([1200 / RPM]))[0]
    return np.array([probe(v[0], "x"), probe(v[3], "x")])


v0 = read(skewed)
runs = [read(skewed.with_(plane_weights=(p._replace(weight=trial),))) for p in planes]
alpha = influence_matrix(v0, runs, [trial, trial])  # 2 x 2: a column per plane
W = corrections(v0, alpha)  # [7.32 g, 6.32 g] at 20 mm, 180° and 0°
balanced = skewed.with_(plane_weights=tuple(p._replace(weight=w) for p, w in zip(planes, W)))
```

---

[Contents](../README.md#contents)
