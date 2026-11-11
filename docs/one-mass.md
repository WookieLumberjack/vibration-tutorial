[Contents](../README.md#contents) · [Foundations](foundations.md) →

# One mass

The first group works on a single mass on a spring and a damper, each page with its own small model,
worked out in closed form or by the same exact first-order-hold step as the chain, as soon as a control
changes.

The pages in the order the app's tabs take them.

- [Single oscillator](#single-oscillator): one mass on a spring and a damper: free decay, damping, resonance, half-power, beats, poles
- [Pendulums](#pendulums): gravity as the spring, large swings that soften, and the rotational form of every one-mass formula
- [Transient response](#transient-response): an impulse, a step or a pulse: the impulse response, convolution, and how long a pulse has to be
- [Periodic forcing](#periodic-forcing): a repeating force as a Fourier series: harmonics, Gibbs, resonance through a harmonic
- [Damping models](#damping-models): viscous, hysteretic, friction and viscoelastic damping: their loops, losses and decays
- [Vibration isolation](#vibration-isolation): a machine on mounts: transmissibility, static deflection, damping, two stages, start-up

## Single oscillator

m ẍ + c ẋ + k x = f, from every side. The free vibration from any start, with its envelope, its peaks
and the log decrement measured from them (or ζ = 0.1, 1 and 2 side by side); a harmonic force or a
rotating unbalance switched on from rest, beside its steady state; and four views of the receptance:
amplification and lag with the half-power band marked, the Nyquist circle, the phase plane, and the
poles in the s-plane, where a click puts them and the page works out k and c. Nine trials include
critical damping, beats, the undamped resonance's straight-line growth and the unbalance's 180° turn.

![Single oscillator page, resonance from rest: 1 N at f<sub>n</sub> = 3.18 Hz on m = 1 kg, k = 400 N/m, ζ = 0.05; the free decay's peaks give δ = 0.3146 (each 1.37 times the next), the forced swing grows over a few time constants to the steady 25 mm, ten times the static 2.5 mm, lagging 90°; the half-power points are 3.012 and 3.331 Hz](images/single_oscillator.png)

## Pendulums

Gravity as the spring. A simple pendulum, a rod swinging about any pivot (with its centre of percussion
and the pivot that swings fastest), a disc on a torsion wire, and an inverted pendulum held up by a
spring, each linearized into the single oscillator's formula with J for m and the moment's stiffness for
k. Kept whole, sin θ softens the hanging pendulum: the exact period by quadrature beside Duffing's
estimate, the phase plane's orbits out to the separatrix, and an inverted pendulum that leans over and
buckles once gravity's negative stiffness beats the spring.

![Pendulums page, large swings: a 1 m pendulum (T₀ = 2.006 s) let go from 90° swings with a period of 2.368 s, 18 % longer (Duffing's estimate +20.2 %), and the linearized swing drifts ahead of it](images/pendulums.png)

## Transient response

An impulse, a step, a ramp, a rectangular or a half-sine pulse on the oscillator. The response is the
convolution of the force with the impulse response h(t), and the page draws the integral itself at any
instant (the force, h flipped and slid, and their product, whose area is the response there). Its tabs
set h(t) beside its Fourier transform, the receptance; measure a step's overshoot and settling time; and
plot each pulse's peak response against its length over the natural period, the shock spectrum's
single-oscillator form. The Theory brings in the Laplace transform: the transfer function, its poles, and
partial fractions back to time.

![Transient response page, a 0.2 s rectangular pulse of 1 N: at t = 0.3 s the area under f(τ) h(t − τ) is −2.58 mm, the exact response there; the peak, 4.64 mm, is 1.85 times static at T/T<sub>n</sub> = 0.637, on the pulse spectrum's rectangle curve](images/transient_response.png)

## Periodic forcing

A force that repeats (a square, sawtooth or triangle wave, a rectified sine, a pulse or impulse train) as
its Fourier series, each harmonic through H(kω₀), against the exact steady state (one period stepped
from the state that repeats). The harmonics' bars, the Gibbs overshoot, how fast smooth and sharp waves
converge, and resonance through a harmonic: a slow drive whose third harmonic sits on f<sub>n</sub>.

![Periodic forcing page: a square wave at f<sub>n</sub>/3 = 1.06 Hz on the 3.18 Hz oscillator; its third harmonic swings the mass 10.6 mm, 89.7 % of the response's power, three times the fundamental's 3.58 mm, and seven harmonics come within 0.67 % of the exact steady state](images/periodic_forcing.png)

## Damping models

Viscous, hysteretic (structural), Coulomb friction and viscoelastic (Zener) damping, each matched to the
same oscillator so it loses the same energy per cycle at f<sub>n</sub>. The hysteresis loops (damping
force alone, or with the spring as a test machine draws them), the energy per cycle, storage stiffness
and loss factor against frequency, the forced response (friction's by its describing function, which
can't bound a resonance) and the free decays (friction's a straight line that stops off centre).

![Damping models page: the four models matched to ζ = 0.05 at f<sub>n</sub> = 3.18 Hz; cycled through ±10 mm there each loop encloses 12.6 mJ, three ellipses and friction's ±0.31 N rectangle; the Zener element relaxes from 444 to 364 N/m with its loss peak of 0.1 at f<sub>n</sub>](images/damping_models.png)

## Vibration isolation

A machine with a rotating unbalance on mounts sized by their static deflection (f<sub>n</sub> =
15.76/√δ<sub>mm</sub> Hz). The transmissibility against frequency over f<sub>n</sub> (crossing 1 at √2
whatever the damping), the force on the floor against speed, a run-up through the mounts' resonance
stepped exactly, and a design chart of the sag each isolation needs. Mounts can be a spring and damper, a
relaxation mount (the damper on a spring) or two stages, and an inertia block can be added.

![Vibration isolation page: a 100 kg machine with a 0.01 kg·m unbalance on mounts that sag 5 mm (f<sub>n</sub> = 7.05 Hz, 423 rpm), run up to 1500 rpm in 2 s; through resonance the floor force peaks at 111 N, and running, 22.6 N of the unbalance's 247 N reaches the floor, 90.8 % isolation](images/vibration_isolation.png)

---

[Contents](../README.md#contents) · [Foundations](foundations.md) →
