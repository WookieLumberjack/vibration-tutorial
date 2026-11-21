← [Continuous systems](continuous-systems.md) · [Contents](../README.md#contents) · [Rotating machines](rotating-machines.md) →

# Signals and random

Sampling and spectra, then random vibration: the power spectral density, RMS and peaks.

The pages in the order the app's tabs take them.

- [Signals & spectra](#signals--spectra): sampling, aliasing, leakage, windows and averaging, and the transducers that make the signals
- [Random vibration](#random-vibration): a random force or base: the response's spectrum, RMS and peaks

## Signals & spectra

Before the measuring pages, the signal processing they use. An analog signal (two tones, a chirp or a
struck mode, plus noise) is sampled, with or without an anti-alias filter, cut into blocks, windowed and
averaged, as a peak amplitude spectrum or a PSD, with each tone's true amplitude marked where it should
show. Its tabs compare the windows' spectra (rectangular, Hann, flat top: main lobe, sidelobes,
scalloping loss, noise bandwidth), draw a spectrogram, and show the transducers as base-excited
oscillators: an accelerometer below its resonance, a seismometer above it, and a sensor's mass lowering
the mode it measures.

![Signals & spectra page: a tone of 1 m/s² at 50.5 Hz, half-way between lines, and one of 0.001 m/s² at 70 Hz, through a Hann window: the big tone reads 0.849 (−1.42 dB, the scalloping loss) and its skirts fall fast enough that the small one reads 0.0010; beside it the three windows' spectra about a line](images/signals_spectra.png)

## Random vibration

The **Random vibration** page drives the chain with a stationary random input given by its power
spectral density, flat over a band: a force on one mass or on every mass (N²/Hz), or the base's
acceleration, as on a shaker table (g²/Hz). It follows the chain on the Simulation page, without its
friction.

- **The response's PSD**: exact, G<sub>y</sub> = H G<sub>u</sub> H<sup>H</sup>, for a mass's
  displacement (relative to the base), its acceleration (absolute, in g) or a spring's force, with
  the record's Welch estimate on top.
- **The RMS**: the cumulative RMS √∫G df climbs a step at each mode, showing each one's share; its
  end is the RMS, against the record's standard deviation and its expected scatter.
- **Miles' equation**: each mode as a single oscillator under white input at its frequency, its RMS
  √((π/2) f<sub>n</sub> Q G) through the mode shape, and the modes combined by SRSS, in a table beside
  the exact RMS. It misses what isn't resonant, such as a forced mass's acceleration above the modes
  (the mass line).
- **The record**: the input synthesized to the PSD asked for, the chain stepped by the exact FOH
  update, sized from the narrowest peak (four Welch lines across its half-power width, sixteen blocks),
  with the block length and a new record at a click.
- **Statistics** (*Statistics* tab): the record's values against the Gaussian (the fraction beyond
  ±1, 2 and 3σ, skewness and kurtosis), and its peaks against Rice's distribution for the response's
  irregularity α = ν₀/ν<sub>p</sub>. Rice's rates from the PSD's spectral moments, ν₀ = √(m₂/m₀) upward
  zero crossings and ν<sub>p</sub> = √(m₄/m₂) peaks a second, sit beside the counts. Then the
  expected largest peak over a stretch, Davenport's (independent crossings) and Vanmarcke's (crossings
  clumped on a narrow band's envelope), against the record's stretches' largest values.
- **Things to try**: a base shake, a force on the last mass, forces on every mass, the base spring's
  force, a short record, coarse lines (the peak comes out low but the RMS keeps its area), and a
  narrow band on mode 2.

![Random vibration page: the default chain's base shaken flat at 0.01 g²/Hz to 10 Hz. The input's PSD from the record lies on the level; the last mass's acceleration PSD peaks at the four modes, the 2621 s record's Welch estimate following the exact H G Hᴴ; the cumulative RMS climbs mostly at mode 1 to 0.9109 g, the record's 0.9108 g and Miles' 0.9129 g beside it; the Miles table gives mode 1 92 % of the mean square](images/random_vibration.png)

Shaken at 0.01 g²/Hz, 0.32 g RMS, the default chain's last mass sees 0.91 g RMS, 92 % of its mean
square from mode 1 (Q = 29). Miles' equation, mode by mode, is 0.2 % off; the simulated record's RMS
is within its ±2.6 % scatter.

![Random vibration page, the Statistics tab for the base spring's force: the record's values follow the Gaussian (31.7 % beyond ±1σ, 32.2 % measured; kurtosis 2.91), its peaks Rice's distribution for α = 0.66; Rice's 1.14 upward zero crossings and 1.74 peaks a second against the record's 1.14 and 1.73; over 60 s Davenport expects a largest peak of 3.11σ and Vanmarcke 2.84σ, and the record's 43 stretches of 60 s average 2.79σ](images/random_statistics.png)

The force in the spring that carries the chain off the base crosses zero 1.14 times a second, as Rice
says from its spectrum. Over a minute Davenport expects its largest peak at 3.11σ; but most of it is
mode 1, a narrow band whose peaks come in clumps, and Vanmarcke's 2.84σ is closer to the record's
2.79σ.

---

← [Continuous systems](continuous-systems.md) · [Contents](../README.md#contents) · [Rotating machines](rotating-machines.md) →
