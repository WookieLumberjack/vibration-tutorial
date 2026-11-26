← [Testing](testing.md) · [Contents](../README.md#contents) · [Nonlinear and flow](nonlinear-and-flow.md) →

# Environments

What a structure meets in service: shocks, roads, fatigue and sound.

The pages in the order the app's tabs take them.

- [Shock spectrum](#shock-spectrum): a short pulse on a bank of oscillators, and on the chain
- [Quarter-car](#quarter-car): a suspension on an ISO 8608 road: comfort, grip and travel from the road's PSD
- [Fatigue](#fatigue): a random stress counted by rainflow: S–N curves, Miner's rule, narrow-band and Dirlik estimates
- [Vibro-acoustics & SEA](#vibro-acoustics--sea): a panel's sound: radiation efficiency, coincidence, the mass law, and energy flow between parts

## Shock spectrum

The **Shock spectrum** page puts a short pulse on the base of a bank of single-degree-of-freedom
oscillators, Q = 10 by custom, and plots each one's largest absolute acceleration against its natural
frequency: the shock response spectrum (SRS), what a component with that natural frequency mounted on
the base would see. The pulse's duration is set from the chain on the Simulation page, and the chain's
modes are marked on the spectrum.

- **The pulses**: the classical ones (half-sine, haversine, terminal- and initial-peak sawtooth,
  trapezoid, square), and two measured records with noise and an offset if you like. One is the
  acceleration of a mass on the chain with its base given a half-sine, the shock a component mounted
  there sees. The other is a pyrotechnic-like transient, a burst of damped sines that ends at rest.
  And two records synthesized to a target spectrum (a plateau above a knee, rising 6 dB an octave below
  it, at Q = 10), the way a shaker test is set up: damped sines, which leave the table moving, or
  Smallwood's wavelets, which start and end at rest. Their amplitudes are scaled round by round until
  the spectrum meets the target.
- **The spectrum**: the maximax, and on request the primary (while the pulse lasts) and residual (after
  it) spectra, and the positive and negative ones. Dashed lines mark its two asymptotes: the pulse's
  peak, which stiff oscillators follow, and 2π f Δv, the ringing of soft oscillators that take the pulse
  as an impulse.
- **The oscillator** (*Oscillator* tab): pick a frequency on the spectrum (drag the line, or click), and
  see that oscillator's acceleration against the pulse, its largest value marked, and its spring's
  stretch.
- **The chain's peak** (*Chain* tab): each of the chain's modes is an oscillator at its own frequency
  and damping, driven by its participation factor Γ times the base, so the SRS gives each mode's peak.
  Those peaks are combined by ABS, SRSS and NRL and set against the chain's exact response, for each
  mass's acceleration or each spring's force, with the picked output's history split mode by mode.
- **Four-way paper** (*Tripartite* tab): the pseudo-velocity, with diagonal lines of constant
  displacement and acceleration, and the pulse's Δv and peak acceleration dashed.
- **Pulse shapes** (*Pulse shapes* tab): the six classical pulses' spectra side by side, at the same peak
  or the same velocity change.
- **Things to try**:
  - a half-sine, a square, a terminal-peak sawtooth and a haversine;
  - light damping (Q = 50: the spectrum hardly changes);
  - a component on the chain's last mass;
  - a pyrotechnic transient;
  - the same transient with an accelerometer's zero shift;
  - the chain's modes combined;
  - four-way paper;
  - the pulse shapes compared;
  - damped sines, then wavelets, synthesized to a spectrum.

![Shock spectrum page: a 10 g, 100 ms half-sine on the base. Its velocity rises to 6.24 m/s; the Q = 10 spectrum peaks at 16.5 g at 8.3 Hz (f T = 0.83), its primary part rising to it from the left and its residual part falling away to the right, between the dashed 2π f Δv line and the pulse's 10 g; the chain's modes are dotted at 1.1 to 6.0 Hz. The 8.3 Hz oscillator's acceleration peaks at 16.5 g near the end of the pulse, then rings at 11 to 13 g](images/shock_spectrum.png)

A 10 g half-sine lasting 100 ms, 0.6 of the default chain's fastest period, puts its modes on the
rising side of the spectrum: a component tuned to mode 1 would see 4.1 g, one tuned to mode 4 15.7 g.
The spectrum peaks at 1.65 × the pulse, for an oscillator whose period is a little longer than the
pulse.

![Shock spectrum page, Chain tab: the 10 g, 100 ms half-sine on the default chain. On m4 the exact peak is 9.14 g; from the SRS at each mode, ABS gives 11 g (+20 %), SRSS 6.62 g (−28 %) and NRL 9.27 g (+1 %). Below, m4's exact response and each mode's part of it, the dashes each part's peak from the SRS; every part peaks within the first 0.3 s](images/shock_chain.png)

The SRS gives each mode's peak but not when it comes. A shock sets every mode off at once, so their
peaks come early and nearly together: the SRSS, which assumes they come at unrelated times, falls 28 %
short at m4, while NRL lands within 1 %.

![Shock spectrum page: 28 wavelets synthesized to a target of 10 g above a 5 Hz knee, rising 6 dB an octave below it. The Q = 10 spectrum stays within 1.9 dB of the target across the band, though the record itself peaks at only 5.19 g, and it ends at rest. On the Tripartite tab the target's rising slope is a constant pseudo-velocity and its plateau a constant acceleration](images/shock_synthesis.png)

## Quarter-car

A suspension on an ISO 8608 road: the road's PSD in space becomes a PSD in time at the car's speed, and
the body's acceleration (comfort), the tire's dynamic load (grip) and the suspension's travel follow as
|H|² G<sub>z</sub>. The conflict diagram over springs and dampers, each output against the damper, and a
drive over a synthesized road stepped exactly, as the check, with the car drawn driving it (the wheel on its
tire, the body on its suspension) and played back against a time cursor.

![Quarter-car page: a quarter of a family car at 72 km/h on a class C road; body bounce at 1.24 Hz and wheel hop at 11.8 Hz, 1.48 m/s² RMS at the body, 10.7 mm of travel, the tire's load swinging 24 % of its static 3.33 kN; on the trade-off, comfort is best at 819 N·s/m and grip at 2394](images/quarter_car.png)

## Fatigue

A random stress of one or two resonant peaks: a record synthesized from its PSD and counted by rainflow,
Basquin's S–N curve and Miner's sum, against the spectral estimates from its moments, the narrow-band
(Rayleigh) formula and Dirlik's. Damage grows as the b-th power of the stress, and a short record's
estimate scatters. The *Rainflow* tab steps the three-point count through the record's first
reversals (Step, Back, and Start on the run bar), the stack ringed and each cycle drawn as it's counted.

![Fatigue page: a stress of two modes (σ = 22.4 MPa, α₂ = 0.46); over 600 s rainflow counts 9057 cycles and a life of 8.29 days, the narrow-band formula 5.09 days (1.63 times the damage) and Dirlik 9.02 days](images/fatigue.png)

## Vibro-acoustics & SEA

How a vibrating panel makes sound. The panel is Beams & plates' simply supported rectangle, set in a
baffle with air on one side. Its bending waves are slower than sound at low frequency and faster at high:
they meet at the **critical frequency** f<sub>c</sub> = (c²/2π)√(ρh/D), 3.99 kHz for 3 mm of steel. Below
it, neighboring half waves cancel each other's sound, and only a mode's corners or edges radiate. The
*Radiation* tab gives each mode's **radiation efficiency** σ<sub>mn</sub> from Wallace's integral, and every
mode's at its own resonance against their third-octave average and Maidanik's formula for it.

The *Driven panel* tab drives the panel with a point force, mode by mode. Its band-averaged input power
settles to an infinite plate's F²/16√(Dρh), and the sound it radiates is set by the radiation loss factor
ρcσ/ωm″ against its own damping. *Transmission loss* puts the panel between two rooms: the mass law, 6 dB
per doubling of mass or frequency, until the coincidence dip at f<sub>c</sub>, where only damping holds the
sound back. The notes also cover decibels and third-octave bands, double walls and a room's reverberation
time.

*Two oscillators* is statistical energy analysis's basis: two oscillators joined by a spring, each driven by
its own noise, exchange power in exact proportion to the difference of their energies, P₁₂ = β(E₁ − E₂),
from the one with more to the one with less. *SEA* sums that over the modes of two plates meeting at a corner
and a room they both radiate into: the corner's coupling loss factor from the bending waves it transmits,
the plates' from their radiation, reciprocity n<sub>i</sub>η<sub>ij</sub> = n<sub>j</sub>η<sub>ji</sub>, and
each band's energies from the power balance.

![Vibro-acoustics & SEA page: a 3 mm steel panel's mode (1, 3), an edge mode at 435 Hz with σ = 0.059; center, the bending wavenumber crosses sound's at the critical frequency, 3.99 kHz; right, every mode's radiation efficiency at its own resonance, in three bands by the parity of m and n, with their third-octave average following Maidanik's formula up to the coincidence peak](images/vibroacoustics.png)

![Vibro-acoustics & SEA page, SEA tab: the 3 mm panel meets a 2 mm plate at a corner, both radiating into a 50 m³ room; the coupling loss factors in each third octave, the two plates' mean-square velocities against the panel's alone, and the room's sound pressure level with each plate's share](images/vibroacoustics_sea.png)

---

← [Testing](testing.md) · [Contents](../README.md#contents) · [Nonlinear and flow](nonlinear-and-flow.md) →
