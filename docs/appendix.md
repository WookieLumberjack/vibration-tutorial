← [Control](control.md) · [Contents](../README.md#contents)

# Appendix

Pages to be folded into others later. Until then they stay here, read after the rest.

The pages in the order the app's tabs take them.

- [Seismic response](#seismic-response): the chain as a building: a design spectrum, effective modal mass, and ABS, SRSS and CQC against records
- [Test planning](#test-planning): where to put the sensors, and what the structure does as it runs
- [Nonlinearity & forces](#nonlinearity--forces): is the structure linear (levels, sweeps, harmonics), and which force made a response
- [Variability & localization](#variability--localization): scatter in the chain's properties by Monte Carlo, and modes localized in a mistuned ring of blades

## Seismic response

The chain as a shear building on the ground: a design spectrum in the European code's shape, each
mode's participation and effective mass, peak floor displacements, drifts and storey shears from the
spectrum, combined by ABS, SRSS, NRL or CQC, against seven records synthesized to match the spectrum and
stepped exactly. A roof tank tuned to mode 1 shows why close modes need CQC; a soft storey and base
isolation are trials too, and the notes cover drift limits, ductility and the behaviour factor. The
*Building* tab plays the building swaying under the first record, scrubbed with a time cursor, each storey's
columns reddening as its drift nears the record's largest: the soft storey and the bearings light up. Storey
1's stiffness has a box of its own, with the period of the building as a rigid block on it, and a challenge
asks for the bearings that bring the shear above them under a limit without letting them move too far.

![Seismic response page: a roof tank tuned to mode 1 of the five-storey building splits it into modes at 0.96 and 1.06 Hz; SRSS gives a base shear of 165 kN, 21 % below the mean of seven matched records (209 kN), and the tank's swing 54 % high](images/seismic_response.png)

## Test planning

Before a test: where should the sensors go, and what does the structure do while it runs? For the chain's
target modes, sensors are placed by **effective independence** (from every mass, drop one at a time the one
with the smallest share of the Fisher information det(ΦᵀΦ)), by EfI weighted by the driving-point residue,
evenly, or by hand. The *Placement* tab shows EfI's eliminations step by step; *Noise & aliasing* sets the
placement beside the methods': the information, each mode's noise against a sensor on every mass, and the
auto-MAC at the sensors, whose off-diagonal shows spatial aliasing (three sensors bunched at the ground make
modes 1 and 2 look alike, MAC 0.976, and mode 1's noise 240 times larger). Below, the **operational
deflection shape** at a running frequency under a force on one mass, beside the mode most like it: the mode
itself at a lightly damped peak, a mix between peaks, and different when the force moves. The *ODS* tab
splits it into each mode's share, and the readout warns when the reference sensor sits where the ODS hardly
moves.

![Test planning page: three sensors on the eight-mass chain placed by effective independence at masses 2, 5 and 8, on the first three modes; the ODS at 1.842 Hz under a force on mass 4, beside mode 1 (MAC 0.650); the ODS tab shows every mass's response and each mode's share of the ODS](images/test_planning.png)

## Nonlinearity & forces

Two checks a test needs. **Is it linear?** The chain, with its own friction and a cubic spring or a gap the
page adds alongside one spring, is driven by a stepped sine at three levels, swept up and then down (harmonic
balance at each step, from the last step's state, with each element's describing function): a linear chain's
six FRFs lie on one curve; a hardening spring's peak leans and jumps at different frequencies each way, a gap
does nothing until it closes, and friction's FRF rises with the level. The *Distortion* tab steps the chain in
time at one point (the exact FOH update, the nonlinear forces found by Newton) and shows the odd harmonics a
symmetric nonlinearity makes. **Which force made this response?** An impact, two, or a random burst on one
mass is measured with noise at chosen sensors (displacement or acceleration); y = Gf, G from the FOH
update's Markov parameters, and f is found plainly (useless: G's singular values span nearly 8 decades, beyond two that are zero to round-off), by
truncated SVD, or by Tikhonov with λ from the L-curve's corner or the noise. The *Singular values* tab is the
Picard plot. An accelerometer on the force's own mass brings G down to 3 decades and the error to 6 %.

![Nonlinearity & forces page: a hardening spring on the chain swept up and down at 0.05, 0.5 and 1 N, the 1 N peak leaning to 2.43 Hz and jumping at 2.45 Hz up and 2.42 Hz down; below, an impact identified from a displacement at mass 1 by Tikhonov at the L-curve's corner, within 15 %; the L-curve on the right](images/nonlinear_forces.png)

## Variability & localization

Scatter in what is built. The chain with every mass, spring and damper scattered (lognormal) by Monte
Carlo: the frequencies' histograms against the first-order estimate from each mode's energies, and the
band the receptance falls in. Then a bladed disc, a ring of identical blades weakly coupled: tuned, its
modes are waves round the ring; mistuned by about the coupling, they localize on a few blades, and an
engine-order force drives the worst blade harder than any tuned one, up to Whitehead's bound.

![Variability & localization page, a mistuned bladed disc: twelve blades coupled by 2 % springs, driven at engine order 3; this 1 % pattern's worst blade peaks 1.06 times the tuned ring, and over 12 patterns per level the worst reaches 1.42 at 2 % mistuning, under Whitehead's bound of 2.23](images/variability.png)

---

← [Control](control.md) · [Contents](../README.md#contents)
