# Vibration Tutorial

An interactive, real-time teaching tool for multi-degree-of-freedom mechanical vibration.
It simulates a chain of lumped masses connected by springs and dampers, animates the
motion, and shows the modal-analysis solution for reference, all updating live while
you change parameters.

> **Disclaimer:** This entire project was created with [Claude Code](https://claude.com/claude-code).
> It is for educational use only, with no warranty, and its results are not guaranteed to be
> technically correct (see [Disclaimer and license](#disclaimer-and-license)).

```
ground ──[k1,c1]── m1 ──[k2,c2]── m2 ──[k3,c3]── m3 ──[k4,c4]── m4
```

![Main window: a harmonic force tuned to mode 2 drives the chain; the modal table and mode shapes are on the right](docs/images/main_window.png)

## Who it's for

Engineering students from about the third year of a degree, and engineers who want a refresher
or a place to try things out. The course assumes calculus and linear differential equations,
complex numbers (e<sup>iωt</sup>, a response's size and phase as one complex number), and
matrices with their eigenvalues and eigenvectors. The Laplace and Fourier transforms, Lagrange's
equations and the statistics of random signals are introduced on the pages that first use them.
Criteria quoted from standards (API, ISO, ASTM, the Eurocodes) come with their edition and are
for teaching, not design.

## Quick start

Standalone builds for Windows, macOS (Apple Silicon) and Linux, which need no Python
install, are on the [Releases](https://github.com/WookieLumberjack/vibration-tutorial/releases)
page. They aren't code-signed, so the system warns the first time you open one:

- **Windows:** unzip and run `vib-tutorial\vib-tutorial.exe`. If SmartScreen says "Windows
  protected your PC", click **More info → Run anyway**.
- **macOS:** unzip to get **Vibration Tutorial.app**. If macOS says the app "is damaged" or
  "can't be opened", run `xattr -dr com.apple.quarantine "Vibration Tutorial.app"` in
  Terminal once, then open it.
- **Linux:** unpack the `.tar.gz` and run `vib-tutorial/vib-tutorial` (needs an X11 or
  Wayland desktop).

To run from source instead, you need [uv](https://docs.astral.sh/uv/) and a
[Rust toolchain](https://rustup.rs/) (uv builds the app's Rust kernels, in `native/`, on
the first sync).

```bash
uv sync               # create .venv and install dependencies
uv run vib-tutorial   # launch the app (or: uv run python -m vib_tutorial)
uv run pytest         # run the tests
uv run ruff check     # lint (and `uv run ruff format` to format)
uv run mypy           # type-check
uvx pre-commit install  # run ruff and mypy before each commit
```

## Contents

The course runs through 11 groups of pages, in the order the app's tabs take them: each page uses only ideas from the pages before it.

**[One oscillator](docs/one-oscillator.md)**

- [Single oscillator](docs/one-oscillator.md#single-oscillator): one mass on a spring and a damper: free decay, damping, resonance, half-power, beats, poles
- [Pendulums](docs/one-oscillator.md#pendulums): gravity as the spring, large swings that soften, and the rotational form of every one-mass formula
- [Damping models](docs/one-oscillator.md#damping-models): viscous, hysteretic, friction and viscoelastic damping: their loops, losses and decays
- [Vibration isolation](docs/one-oscillator.md#vibration-isolation): a machine on mounts: transmissibility, static deflection, damping, two stages, start-up
- [Transient response](docs/one-oscillator.md#transient-response): an impulse, a step or a pulse: the impulse response, convolution, and how long a pulse has to be
- [Periodic forcing](docs/one-oscillator.md#periodic-forcing): a repeating force as a Fourier series: harmonics, Gibbs, resonance through a harmonic

**[Equations](docs/equations.md)**

- [Deriving the equations](docs/equations.md#deriving-the-equations): free-body diagrams against Lagrange's equations: a car in bounce and pitch, and a double pendulum
- [3D rigid body](docs/equations.md#3d-rigid-body): a body tumbling in space: its inertia tensor, principal axes, Euler's equations and the intermediate-axis flip

**[Modes](docs/modes.md)**

- [Simulation & modal analysis](docs/modes.md#simulation--modal-analysis): one mass, then a chain of them: modes, damping, energy, forcing, absorbers, friction
- [Energy methods](docs/modes.md#energy-methods): Rayleigh, Dunkerley, Rayleigh–Ritz and Holzer: frequencies from the energies, and how good they are
- [FRF matrix](docs/modes.md#frf-matrix): every receptance H<sub>jk</sub>, and how the modes add up to it
- [Complex modes](docs/modes.md#complex-modes): when damping makes the modes complex, and what the usual approximation misses
- [Rigid body on mounts](docs/modes.md#rigid-body-on-mounts): a machine on four mounts in 3D: six coupled modes, the elastic centre, and decoupling by tilting them

**[Continua and models](docs/continua-and-models.md)**

- [Continuous systems](docs/continua-and-models.md#continuous-systems): a string, a bar or a beam cut into a chain: which of its modes to trust
- [Beams & plates](docs/continua-and-models.md#beams--plates): Timoshenko's thick beam, a plate's modes and nodal lines, and modal density
- [Modal coupling](docs/continua-and-models.md#modal-coupling): two parts joined, one mode each: when their modes split
- [Substructuring (CMS)](docs/continua-and-models.md#substructuring-component-mode-synthesis): a chain reduced to a few modes per part, and back
- [Numerical integration](docs/continua-and-models.md#numerical-integration): central difference, Newmark, HHT-α and Runge–Kutta against the exact update: stability, period error, numerical damping
- [Frame builder](docs/continua-and-models.md#frame-builder): build a frame or truss from beams, bars and springs: assembly, supports, modes

**[Signals and random](docs/signals-and-random.md)**

- [Signals & spectra](docs/signals-and-random.md#signals--spectra): sampling, aliasing, leakage, windows and averaging, and the transducers that make the signals
- [Random vibration](docs/signals-and-random.md#random-vibration): a random force or base: the response's spectrum, RMS and peaks

**[Testing](docs/testing.md)**

- [Virtual modal test](docs/testing.md#virtual-modal-test): measuring the FRF from force and response signals, and fitting modes
- [Model updating](docs/testing.md#model-updating): comparing a model with a test, and correcting it
- [Operational modal analysis](docs/testing.md#operational-modal-analysis): the modes from the responses alone, the forces unknown

**[Environments](docs/environments.md)**

- [Shock spectrum](docs/environments.md#shock-spectrum): a short pulse on a bank of oscillators, and on the chain
- [Quarter-car](docs/environments.md#quarter-car): a suspension on an ISO 8608 road: comfort, grip and travel from the road's PSD
- [Fatigue](docs/environments.md#fatigue): a random stress counted by rainflow: S–N curves, Miner's rule, narrow-band and Dirlik estimates
- [Vibro-acoustics & SEA](docs/environments.md#vibro-acoustics--sea): a panel's sound: radiation efficiency, coincidence, the mass law, and energy flow between parts

**[Nonlinear and flow](docs/nonlinear-and-flow.md)**

- [Duffing & Mathieu](docs/nonlinear-and-flow.md#duffing--mathieu): a nonlinear spring: jumps, chaos, and parametric resonance
- [Fluid–structure](docs/nonlinear-and-flow.md#fluidstructure): a body in water: added mass of spheres, cylinders, tubes in pipes and plates; sloshing in a tank and a tuned liquid damper
- [Self-excited vibration](docs/nonlinear-and-flow.md#self-excited-vibration): negative damping and limit cycles: van der Pol, galloping, vortex-induced vibration and flutter

**[Rotating machines](docs/rotating-machines.md)**

- [Gyroscope](docs/rotating-machines.md#gyroscope): a spinning top: Euler's equations, steady precession, nutation and the sleeping top
- [Torsional vibration](docs/rotating-machines.md#torsional-vibration): a drive train of discs, shafts and gears, and its start-up
- [Jeffcott rotor](docs/rotating-machines.md#jeffcott-rotor): a spinning shaft: critical speeds, balancing, bearings, stability, rub
- [Rotor builder](docs/rotating-machines.md#rotor-builder): build a rotor from shafts, discs, bearings and seals: its whirl modes, criticals and stability
- [Machinery diagnostics](docs/rotating-machines.md#machinery-diagnostics): order tracking under a run-up, gear-mesh sidebands, bearing defects by the envelope, and the cepstrum

**[Control](docs/control.md)**

- [Active control](docs/control.md#active-control): velocity feedback, collocation, spillover and actuator lag, against a tuned mass damper

**[Appendix](docs/appendix.md)**

- [Seismic response](docs/appendix.md#seismic-response): the chain as a building: a design spectrum, effective modal mass, and ABS, SRSS and CQC against records
- [Test planning](docs/appendix.md#test-planning): where to put the sensors, and what the structure does as it runs
- [Nonlinearity & forces](docs/appendix.md#nonlinearity--forces): is the structure linear (levels, sweeps, harmonics), and which force made a response
- [Variability & localization](docs/appendix.md#variability--localization): scatter in the chain's properties by Monte Carlo, and modes localized in a mistuned ring of blades

**Reference**

- [The window](docs/the-window.md): the panes, the tabs, the Theory and Practice windows, and the themes
- [The math, briefly](docs/the-math.md): the equations behind the Simulation page
- [Code layout](docs/code-layout.md): the modules, the Rust kernels, and scripting the core
- [Contributing](CONTRIBUTING.md): setting up, the checks before a pull request, and adding a page
- [Changelog](CHANGELOG.md): what changed in each release

## Disclaimer and license

This software is provided for **educational purposes only**. It was written with an AI coding
assistant, [Claude Code](https://claude.com/claude-code), and although it has an automated test
suite, its results are **not guaranteed to be technically correct**: the simulations, modal
analysis, substructuring, identification and theory notes may contain errors. Validate any
calculation independently (by hand, with established references or with qualified software)
before relying on it, and do not use it for design, certification, safety-related or other
engineering decisions.

The software is provided **"as is", without warranty of any kind**, express or implied,
including the warranties of merchantability, fitness for a particular purpose and
non-infringement. In no event shall the authors or copyright holders be liable for any claim,
damages or other liability arising from the software or its use.

Released under the [MIT License](LICENSE). Copyright (c) 2026 WookieLumberjack.
