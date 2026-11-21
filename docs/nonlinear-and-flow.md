← [Environments](environments.md) · [Contents](../README.md#contents) · [Control](control.md) →

# Nonlinear and flow

Nonlinear springs, a structure in a fluid, and vibration that feeds itself.

The pages in the order the app's tabs take them.

- [Duffing & Mathieu](#duffing--mathieu): a nonlinear spring: jumps, chaos, and parametric resonance
- [Fluid–structure](#fluidstructure): a body in water: added mass of spheres, cylinders, tubes in pipes and plates, wet against dry
- [Self-excited vibration](#self-excited-vibration): negative damping and limit cycles: van der Pol, galloping, vortex-induced vibration and flutter

## Duffing & Mathieu

The **Duffing & Mathieu** page drives one mass on a spring with a cubic term,
m x″ + c x′ + k(1 + μ cos φ) x + k<sub>3</sub> x³ = F cos φ, that hardens (k<sub>3</sub> > 0) or
softens (k<sub>3</sub> < 0) as it stretches, and sweeps the drive's frequency slowly up and down.
The pump μ makes the stiffness itself pulse at that frequency (parametric excitation, the Mathieu
equation), and a small random force gives an unstable rest something to grow from.

- **Harmonic balance**: the first harmonic's steady amplitude against frequency, solid where
  stable and dashed where not, with the backbone (the free oscillation's frequency at each
  amplitude) and the linear spring's curve for comparison. Between the two folds there are three
  amplitudes, the middle one unstable, and the arrows mark the jumps from each fold.
- **Sweeps**: each drive cycle's measured first harmonic, coloured by the sweep's direction, on
  top of the curve, with the measured jump frequencies beside the folds'. Hold the frequency
  inside the band and *Start on a branch* puts the mass on the lower, middle or upper steady
  state.
- **Parametric resonance**: pumped, harmonic balance's half-harmonic response (rest, unstable
  inside the tongue, and a cubic spring's swinging branches, in closed form), with each cycle's
  measured ½X amplitude (both harmonics are taken over pairs of cycles, so a swing at Ω/2 doesn't
  leak into the first).
- **Ince–Strutt chart**: the pumped linear spring's growth rate against the pump's frequency and
  depth at the page's damping, from the Floquet multipliers of every cell at once; the tongues
  reach down to Ω = 2ω<sub>n</sub>/n, with harmonic balance's first tongue dashed over them.
  Click to pump there.
- **Phase plane**: the last few cycles' trajectory and, while the frequency is held, the
  Poincaré section (the state once per drive cycle).
- **Presets**: hardening and softening jumps, a linear spring for comparison, two stable states
  at one frequency, a softening spring thrown over its potential's hilltop (the page pauses when
  it escapes), parametric resonance on a linear spring (it grows until the page pauses) and on a
  hardening one (it settles), a parametric sweep with its jump, and Ueda's chaotic oscillator.
- **Notes**: harmonic balance, the backbone, the jump and its stability, sweep rate and
  critical slowing down, superharmonics, subharmonics and chaos, parametric excitation,
  Floquet theory and the Ince–Strutt chart.

![Duffing & Mathieu page, hardening spring (k = 400 N/m, k3 = 10^5 N/m³, c = 1 N·s/m, F = 2 N) swept from 2 to 6 Hz and back at 120 s a leg: harmonic balance gives two stable amplitudes between the folds at 3.64 and 4.45 Hz; the sweep up rides the upper branch to 71 mm and drops at 4.62 Hz, the sweep down follows the lower branch and jumps up at 3.56 Hz](images/duffing_jump.png)

A hardening spring's resonance leans toward higher frequencies, because a bigger swing is a
stiffer one. Sweeping up, the mass rides the upper branch past the linear resonance (3.18 Hz) to
its fold at 4.45 Hz, then drops to the lower branch. Sweeping down, it follows the lower branch to
the other fold, at 3.64 Hz, and jumps up. Near a fold the steady state is barely stable, so a
sweep overruns it before the mass leaves: here it drops at 4.62 Hz and jumps up at 3.56 Hz.

![Parametric sweep on the hardening spring, no force, the stiffness pulsing ±30 %, swept down from 8 to 5 Hz from rest and back up at 180 s a leg: rest is unstable between 5.89 and 6.80 Hz; sweeping down it stays at rest to 6.57 Hz, then jumps up to the branch; back up it grows at 6.11 Hz and rides the branch to 62 mm at 8 Hz, far past the tongue. Beside it the Ince–Strutt chart at ζ = 2.5 %, its first tongue's tip lifted to μ = 0.1](images/mathieu_sweep.png)

Pump the spring's stiffness instead of pushing the mass and rest itself turns unstable near twice
the natural frequency: the mass swings at half the pump's frequency. On a linear spring the swing
grows for ever (as e<sup>0.05ω<sub>n</sub>t</sup> with μ = 0.3 and ζ = 2.5 %, which the
simulation and the Floquet multipliers agree on), and damping needs a pump deeper than 4ζ to
start it. A hardening spring's swing detunes itself out of the tongue and settles, which makes a
jump of its own: sweeping down from rest, nothing happens until rest turns unstable at the
tongue's upper edge, and then the swing leaps up to a branch that the sweep back up rides far past
the tongue.

## Fluid–structure

What water does to a vibrating body. A body accelerating in a fluid has to push some of the fluid about,
and that flow's kinetic energy makes it behave as if it were heavier: its **added mass**. In open fluid a
sphere carries half the fluid it displaces and a cylinder all of it; the centre shows the potential flow
round the body, and how fast it falls off with distance. Inside a pipe, the fluid has to squeeze through
the gap, and the added mass grows without bound as the gap closes. The fluid adds no stiffness, so on its
springs the body's frequency falls by √(m/(m + m<sub>a</sub>)), set by the body's density against the
fluid's.

The *Plate* tab gives each mode of a baffled plate its own added mass, from the fluid's
pressure for each wavenumber in the mode's shape, and its AVMI factor β: a 3 mm steel panel in water
carries eight times its modal mass in its fundamental, and falls to a third of its frequency in air.
Higher modes carry less, as the fluid sloshes between their cells.

![Fluid–structure page: a 25 mm steel rod moving across its axis in water; centre, the potential flow round it (speed in colour, streamlines) and the fluid's speed ahead of it falling as R²/r²; right, the added-mass coefficient against a pipe's radius (1 in open water, growing as the gap closes) and the frequency's fall against the densities' ratio, 20 Hz to 18.8 Hz](images/fluid_structure.png)

A liquid with a free surface has modes of its own. The *Sloshing* tab puts it in a rectangular tank
(the centre then shows the tank: the surface at its extremes and the flow as it passes level). The
modes' frequencies follow ω² = gk tanh(kh), so a shallow tank's first mode runs at √(gh)/2L and a deep
one's stops depending on depth. Under a sideways motion the tank behaves as a **spring–mass analogue**:
a rigid mass and a mass on a spring for each odd mode, the first by far the largest. The *Damper* tab
uses that first mode as a **tuned liquid damper** on a structure. The water's depth tunes it, screens
damp it, and the peak against depth shows where Den Hartog's tuning lies.

![Fluid–structure page, a tuned liquid damper: a tank 2 m by 2 m filled to 0.4 m on a 20 t structure at 0.5 Hz; centre, the tank's first sloshing mode (0.466 Hz); right, the structure's response with the water sloshing (peak 5.58) and frozen (52.0), and the peak against the water's depth, lowest near 0.4 m](images/liquid_damper.png)

## Self-excited vibration

Motion that feeds itself, four ways: van der Pol's oscillator (negative damping that grows to a limit
cycle, and relaxation oscillations), galloping of a square section in a wind (Den Hartog's criterion and
the hysteresis between its branches), vortex-induced vibration with a wake oscillator (lock-in), and
two-mode flutter of a wing section, its root locus crossing into instability as the airspeed rises (in steady
lift by coalescence; in quasi-steady lift the pitch mode goes on its own). Drag a pole along the locus and
the speed follows, so you can walk it across Re λ = 0. The *Motion* tab plays the body itself (the mass,
the galloping square and the cylinder to scale, the wing plunging and pitching in its least stable mode), with
a slider for what drives it that moves the poles as you drag.

![Self-excited vibration page, flutter: a wing section with plunge at 3 Hz and pitch at 6 Hz; in steady lift; as the airspeed rises their frequencies close in and, nearly met, the plunge-like mode's damping crosses zero at 20.7 m/s, about half the 39 m/s divergence speed](images/self_excited.png)

---

← [Environments](environments.md) · [Contents](../README.md#contents) · [Control](control.md) →
