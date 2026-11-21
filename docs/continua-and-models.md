← [Modes](modes.md) · [Contents](../README.md#contents) · [Signals and random](signals-and-random.md) →

# Continua and models

Continuous structures cut into elements, models assembled and reduced, and stepping them in time.

The pages in the order the app's tabs take them.

- [Continuous systems](#continuous-systems): a string, a bar or a beam cut into a chain: which of its modes to trust
- [Beams & plates](#beams--plates): Timoshenko's thick beam, a plate's modes and nodal lines, and modal density
- [Modal coupling](#modal-coupling): two parts joined, one mode each: when their modes split
- [Substructuring (CMS)](#substructuring-component-mode-synthesis): a chain reduced to a few modes per part, and back
- [Numerical integration](#numerical-integration): central difference, Newmark, HHT-α and Runge–Kutta against the exact update: stability, period error, numerical damping
- [Frame builder](#frame-builder): build a frame or truss from beams, bars and springs: assembly, supports, modes

## Continuous systems

The **Continuous systems** page goes the other way from the chain: a string, a bar or a beam is a
continuum with infinitely many modes, and the chain is what you get by cutting it into N equal
elements. Its waves say which of the chain's modes to trust.

- **The medium**: a string (tension, mass per length), a bar (axial waves) or an Euler–Bernoulli
  beam (bending), its length, material and section, and each end fixed or free (a beam's clamped,
  pinned or free). The exact modes come in closed form, a beam's from its ends' frequency equation.
- **The mesh**: 1 to 200 elements, linear for the string and bar and Hermite cubics for the beam,
  with three ways to share out the mass, shown together: *lumped* (half to each node: beads on a
  string, or the first page's chain), *consistent* (from the stiffness's own shape functions, an
  upper bound on every frequency) and *mixed* (their average).
- **Dispersion**: frequency against wavenumber, the continuum's (ω = cκ, or aκ² for a beam) with each
  mesh's, which bends away and stops at κ = π/h, two elements a wavelength. The mesh's modes are dots
  on it: a string's or bar's lie exactly on the mesh's curve, since its modes are the endless mesh's
  waves sampled at the nodes. A beam's consistent mass adds a second branch, the slopes against the
  deflections, drawn past π/h, where its extra modes land.
- **Errors**: each mode's frequency error against its number, and the chosen mode's against the
  number of elements on log scales, with the slope: N<sup>−2</sup> lumped or consistent, N<sup>−4</sup>
  mixed for a string or bar; N<sup>−2</sup> lumped and N<sup>−4</sup> consistent for a beam.
- **Mode shape and frequencies**: the chosen mode, exact and on each mesh (the nodes and the
  elements' own interpolation between them), and a table of every mode's frequency and error.
- **Waves** (centre, *Waves* tab): a pulse runs along the continuum and the mesh side by side,
  travelling right or released at rest. The continuum's is exact: a string's or bar's two
  d'Alembert waves carried along, scattered at the ends in closed form, or a beam's exact modes
  summed. The mesh is stepped by its exact matrix exponential. On it the pulse sheds ripples
  (behind it with lumped mass, ahead with consistent), since its short waves travel at the wrong
  speed and none shorter than two elements travels at all. The pulse comes back from a fixed end
  upside down and from a free end upright. A *dashpot* on the far end reflects
  (1 − r)/(1 + r) of the wave, r its damping over the impedance Z (√(Tμ) or ρcA): matched (r = 1)
  it absorbs the continuum's pulse, but not the mesh's short waves. A *step* in the string's mass or
  the bar's section reflects (Z₁ − Z₂)/(Z₁ + Z₂) of a wave and passes 2Z₁/(Z₁ + Z₂). The mesh's
  energy is shown as it goes.
- **Presets**: a guitar string as beads, a steel bar held at one end, a steel ruler over a desk's
  edge, a vibraphone bar (free–free), and a pinned beam whose modes lie on the mesh's curve; and
  four wave presets: a pulse on 40 beads, a matched end, a light string tied to a heavy one, and a
  bending pulse on a beam.

![Continuous systems page, the Guitar string preset: on the dispersion plot the continuum's straight line ω = cκ (c = 106.8 m/s) and three 12-element meshes' curves bending away from it to the cut-off at κ = π/h = 58.2 rad/m, lumped low (topping out at 630 Hz where the continuum is at 989 Hz), consistent high (1090 Hz) and mixed close to the line, each mesh's modes dots on its curve; mode 6, four elements a wavelength, is marked: exact 494.4 Hz, lumped 445.2 Hz (−9.97 %), consistent 545.2 Hz (+10.3 %), mixed 487.6 Hz (−1.38 %). On the right its shape: each mesh's nodes are exactly on the sine](images/continuum_string.png)

A low E string as 12 beads: the 6th harmonic, four beads a wavelength, is 10 % flat, though its
shape at the beads is the sine's exactly (MAC 1). The frequency error is all in the dispersion: the
beaded string is slower for short waves, and carries none shorter than two beads.

![Continuous systems page, the Steel ruler preset's Errors tab: the frequency error of each of 8 Hermite elements' modes against the mode number, lumped falling to −24 %, consistent within 1.2 % for the first six, mixed between them, and above mode 8 the consistent and mixed meshes' slope modes far off; below, mode 3's error against the number of elements on log scales, falling as N⁻² lumped and mixed and N⁻⁴ consistent; on the right mode 3's shape, the meshes' nodes on the exact one (MAC 0.992 lumped, 1.000 consistent)](images/continuum_beam.png)

A clamped steel ruler's third mode: eight elements are already within 0.06 % with consistent
masses, which converge as N<sup>−4</sup>; lumped masses, leaving the slopes without inertia, only
as N<sup>−2</sup>.

![Continuous systems page, the Waves tab with a light string tied to a heavy one: a pulse launched rightward at 100 m/s has met the step halfway along (shaded), where the string is four times heavier. On the continuum (black) a third of it has come back upside down (−0.33) and two thirds have gone on (0.67) at half the speed, half as long; on the 100-element lumped mesh the transmitted pulse, five elements wide, reaches only 0.49 and trails ripples, and the reflected one lags the continuum's](images/continuum_waves.png)

Halfway along this string the mass per length is four times as much: twice the impedance and half the
speed. A third of the pulse comes back upside down and two thirds go on, squeezed to half the length.
On 100 elements the squeezed pulse is only five elements wide, and the mesh carries 0.49 of it where
the string carries 0.67, shedding ripples behind.

## Beams & plates

Beyond the Continuous systems page's slender beam: a simply supported beam by Euler–Bernoulli, Rayleigh
(rotary inertia), shear alone and Timoshenko (both, with its thickness-shear branch), and their wave
speeds; a simply supported rectangular plate's and a clamped circular plate's modes and nodal lines
(a square plate's twin modes mixed into Chladni's figures); and the modes per hertz, ∝ √f for a beam and
constant for a plate.

![Beams & plates page: a 400 mm square steel plate 3 mm thick, simply supported; its (1, 3) and (3, 1) modes share 461 Hz, and their sum has the curved nodal line of a Chladni figure](images/beams_plates.png)

## Modal coupling

Two systems that are each fine on their own can misbehave once joined: when a mode of one is
close in frequency to a mode of the other, and the masses are not too different, the two
modes **split**. This is how a vibration absorber works, but it usually happens when nobody
meant it to. The **Modal coupling** page shows it on the chain. *Split after* cuts the chain into
**A** (the masses up to the cut, on the ground as before) and **B** (the rest, with the spring
and damper that joined it to A tied to the ground instead). Each is solved on its own, one mode of
each is replaced by a single-DOF oscillator that keeps its frequency and modal damping ratio, and
the two oscillators are joined again, B's on A's:

```
ground ──[k_a,c_a]── m_a ──[k_b,c_b]── m_b
```

The oscillator masses are the masses the joint feels:

- **B**: its **effective mass** along x, Γ² with Γ = φ<sup>T</sup>M**1** (φ mass-normalized),
  since B is joined at its base. The rest of B's mass (its other modes) moves rigidly with the
  base, so it is added to m<sub>a</sub> as B's **residual mass**.
- **A**: its **modal mass at the interface**, 1/φ<sub>tip</sub>², since A is joined at its tip.
  A's effective mass can be chosen instead, to see how much worse it does.

With these choices the 2-DOF model is a Rayleigh–Ritz model of the chain (B's part is
Craig–Bampton with one mode kept; see the next section), so its two frequencies are upper bounds of the chain's lowest
two. The page compares them with the chain's (frequency, exact damping ratio, MAC) and draws the
2-DOF modes on the chain. Any mode of A can be coupled with any mode of B. The *Drive-point FRF*
tab compares the receptance of the chain and of the 2-DOF model at the interface (A's tip) or at
the free end (B's tip, where a force is usually applied), with A alone (B removed) and B alone
(A removed, its base on the ground) for reference. *Matrices (step by step)* works both masses through with the current
numbers: the generalized mass and mass normalization (why 1/φ<sub>tip</sub>² is a mass), the
influence vector **r** solved from a unit base motion, each mode projected on how the joint loads it
(a point load **e**<sub>tip</sub> on A, the inertia M**r** on B), the 2-DOF matrices, and the
Rayleigh–Ritz check, with a notation table.

The **Veering & splitting** tab sweeps B's frequency (or the mass ratio μ = m<sub>b</sub>/m<sub>a</sub>)
and plots the chain's and the 2-DOF model's frequencies. Near f<sub>B</sub> = f<sub>A</sub> the two
modes veer apart instead of crossing; two oscillators tuned exactly end up √μ·f<sub>a</sub> apart,
whatever μ.

The notes work the default split through by hand (the masses, the tuned-oscillator split, the
model's errors) and close with frequency-based substructuring: joining two parts by their FRFs. The *Mode shapes* tab swings the chain's two modes, standing waves whose nodes stay put.

![Modal coupling: m3–m4 made light (0.1 kg) and tuned to m1–m2; the 2-DOF model (dashed) follows the chain's two lowest modes through the veering, within 0.33%](images/modal_coupling.png)

## Substructuring (component mode synthesis)

The *Substructuring* page starts with the whole chain as one substructure, whose only
boundary DOF is the tip: the single-component case, to learn the fundamentals on (Guyan then
reduces it to that one DOF, the largest error the page can show). Click masses under
*Interfaces at* to cut the chain there into substructures A (grounded), B, C, ... up to the
free end, and click them again to remove the cuts. The boundary (master) DOFs are the interfaces and the last mass, which is where
the force is applied, so the loaded DOF stays physical. Each substructure keeps its own
chosen number of modes plus one static shape per boundary DOF, and the reduced
substructures are assembled on the shared interface DOFs. The
*Method* selector chooses the component modes:

- **Craig–Bampton** (fixed interface): modes with the boundary held, plus constraint modes.
- **Rubin** (free interface): modes with the boundary free, as a modal test would measure
  them, plus the *residual flexibility* of the discarded modes. B floats on its own, so its
  rigid-body mode is always kept.
- **MacNeal** (free interface): as Rubin, but the residual flexibility is massless, so the
  boundary DOFs carry no inertia and the model has only as many modes as modes kept.

Everything on the page follows the selector:

- **Comparison table**: true vs reduced natural frequencies, the error (never negative for
  Craig–Bampton and Rubin, whose frequencies are upper bounds), and the MAC of the recovered
  shape.
- **Substructures on their own**: each substructure's own modes (natural frequency and
  exact damping ratio with its boundary held, or free), whether each is kept, and the
  coupled mode it becomes (largest share of that mode's energy).
- **Component shape overlay**: click a row of that table to draw the clamped component
  shape over the coupled mode it becomes.
- **Basis (T)**: every column of T drawn as a shape along the chain, the set of shapes every
  reduced-model motion is built from.
- **Modes & FRF**: the selected mode shape, true vs reduced, and the tip and interface
  receptance of both models for the force at the tip.
- **Time response**: the reduced model simulated beside the full one, with the same exact
  stepper and the same force at the tip (a step, a harmonic, or a pulse). The two chains are
  animated one above the other at the same scale, over time histories of the tip and interface
  and of their difference. All three methods are exact statically, so the error is in the
  vibration: small after a step, large when driven at a mode the reduced model has shifted
  (below). *Tune to…* offers the true and the reduced natural frequencies.
- **Back expansion**: the coupled (reduced) model solves only for [q, x<sub>b</sub>]; this tab
  recovers the interior masses from them afterwards, as a system analyst does with a
  substructure delivered as reduced matrices. Three recoveries of one interior mass and one
  spring force are compared with the truth: from the coupled solution with the reduced
  model's own T (split into the part the boundary drags along and the part the kept modes
  add), from the boundary motion alone (static), and *enhanced*, by re-solving the
  substructure's interior with the coupled model's boundary motion imposed. The lower chain
  draws the recovered masses hollow.
- **Compare methods**: the same cut reduced by all three methods, with the frequency error of
  every mode, and how the selected mode's error falls as modes are added, one at a time.
- **Matrices (step by step)**: every stage with the current numbers: the full K and M,
  each substructure's partitioned matrices, its component modes (kept and discarded), the
  constraint modes Ψ or the residual flexibility G<sub>d</sub>, the transformation T, the
  reduced matrices, the assembled reduced model, and each substructure's recovery (OTM) and
  loads (LTM) transformation matrices.
- **Theory**: what a basis is (with a 2-mass worked example) and why Craig–Bampton chooses
  its shapes, original vs substructured formulation side by side, Hurty vs Craig–Bampton,
  Guyan reduction as the zero-mode case, why free-interface modes need residual flexibility,
  MacNeal vs Rubin, the three methods side by side, back expansion (data recovery), one-click "Things to try"
  walkthrough, and a notation
  table.
- **Presets**: *Guyan (0 modes)* (*Fewest modes* for the free-interface methods) and *All
  modes*.

With more cuts each substructure is smaller and has fewer modes to keep, but every interface
stays in the model as a physical DOF. For 8 masses and a 6-DOF model, one cut with 2 modes
each gets modes 1 to 4 within 0.7%; two cuts with 1 mode each only within 2.3%:

The notes work the default (the 4-mass chain, uncut, one mode kept) through by hand: the clamped
mode, the constraint mode, Guyan's 1.162 Hz and the 2 × 2 model's 1.108 Hz against the true 1.105 Hz.

![Three substructures: cuts at m3 and m6 with one clamped mode each; the six columns of T are three clamped modes and three constraint-mode "tents"](images/substructuring_three.png)

For each substructure, partitioned into interior $i$ and boundary $b$ DOFs,

$$\begin{bmatrix}x_i\\ x_b\end{bmatrix} = \underbrace{\begin{bmatrix}\Phi_k & \Psi\\ 0 & I\end{bmatrix}}_{T}\begin{bmatrix}q\\ x_b\end{bmatrix},
\qquad \Psi = -K_{ii}^{-1}K_{ib},\qquad \hat M = T^TMT,\ \hat K = T^TKT$$

where $\Phi_k$ are the $k$ lowest mass-normalized modes of $K_{ii}\phi = \omega^2 M_{ii}\phi$.
$\hat K$ is block diagonal, $\mathrm{diag}(\omega_k^2)$ and $K_{bb} - K_{bi}K_{ii}^{-1}K_{ib}$, while
$\hat M$ couples $q$ to $x_b$.

The free-interface methods use the lowest $k$ modes $\Phi_k$ of the whole substructure,
$K\phi = \omega^2 M\phi$, and keep the static response of the discarded ones through the
residual flexibility $G_d = \Phi_d\Lambda_d^{-1}\Phi_d^T$. Writing
$x = \Phi_k q + G_d E_b f_b$ and eliminating the interface forces $f_b$ with the boundary
rows gives a transformation of the same shape as Craig–Bampton's:

$$T = \begin{bmatrix}\Phi_{ik} - R\,\Phi_{bk} & R\\ 0 & I\end{bmatrix},\qquad R = G_{ib}G_{bb}^{-1}$$

Rubin's method projects with it, $\hat M = T^TMT$; MacNeal's drops the residual mass,
$\hat M = \mathrm{diag}(I, 0)$. Both have $\hat K = T^TKT$, which now couples $q$ to $x_b$.

![Time response: driven at the true mode 4 (3.84 Hz), the Craig–Bampton model with 1 mode per substructure resonates at 4.11 Hz instead; its tip moves less and lags](images/substructuring_time.png)

After the coupled solve each substructure recovers its interior from its own rows of T,
$x_i = \Psi x_b + \Phi_k q$ (Craig–Bampton). The *enhanced* recovery instead solves
$M_{ii}\ddot x_i + C_{ii}\dot x_i + K_{ii}x_i = -K_{ib}x_b - C_{ib}\dot x_b$ with the coupled
model's $x_b(t)$, so it keeps every interior mode and its only error is the error in $x_b$:

![Back expansion: driven at mode 4 with 2 modes kept per substructure, the coupled recovery of the force in k2 is off by nearly 40%; the enhanced recovery brings it to about 12%](images/substructuring_recovery.png)

![Compare methods: the same cut reduced three ways; Craig–Bampton and Rubin converge to the exact mode 3, MacNeal does not](images/substructuring_compare.png)

## Numerical integration

What the usual time-stepping schemes do to a vibration: central difference, Newmark (average and linear
acceleration), HHT-α, Runge–Kutta 4 and explicit Euler, against the exact first-order-hold update the app
uses. A run at any step per period beside the exact motion, each scheme's spectral radius, period error
and numerical damping against ωΔt, and a stiff second mode that breaks the explicit schemes.

![Numerical integration page at 3 steps a period (ωΔt = 2.09): central difference, past its limit ωΔt ≤ 2, blows up by 2.5 s, while Newmark's average acceleration stays bounded with its period 30 % long](images/numerical_integration.png)

## Frame builder

Build a plane frame or truss on a grid: click to place nodes, click two nodes to join them with a beam (it
bends and stretches), a bar (it only stretches) or a spring or damper (or anchor one to the ground), add
point masses and torsion springs, and give nodes supports (pinned, rollers, clamped, guided). Each node has
u, v and θ. Every change assembles the global M, C and K again: the *Assembly* tab shows the whole matrix,
with the picked element's share Tᵀk'T tinted and the DOFs a support holds or no element touches (a truss
node's turn) struck out; the *Element* tab shows its k' or m' in its own axes, its T, and the global DOFs
it lands on. Massless DOFs (a lumped beam's turns) are condensed out exactly, mechanisms show as modes at
0 Hz, and each mode swings on the frame with the beams bending along their cubics, its strain energy split
between stretching, bending and the springs. Consistent or lumped mass, and Euler–Bernoulli or Timoshenko
beams; the trials compare the mesh with Euler's cantilever and the Beams & plates page's thick beam.

![Frame builder page: a portal frame of 50 × 50 mm steel, clamped at its bases, swaying in its first mode at 14.90 Hz; the Assembly tab shows K with the picked column beam's share tinted and the bases' DOFs struck out](images/frame_builder.png)

Then load it: a harmonic force, an impulse or a step on any DOF (a moment on a turn), with Rayleigh damping
(ζ at the first two modes) and the dampers you've placed. The *Response* tab plots the receptance to any DOF
and the response from rest, worked out two ways: the **conventional** model (the undamped modes, each a
single oscillator with ζ = φᵀCφ/2ω, ΦᵀCΦ's off-diagonal terms dropped) and the **state space**
(A = [0 I; −M⁻¹K −M⁻¹C], nothing dropped), both stepped exactly, the load generated by states of its own. With
proportional damping they agree; a damper makes them part. The *Formulations* tab sets each undamped mode
beside the state space's, its f, ζ and MPC, with the complex shape in the complex plane, and the frame can
swing in it, its nodes out of step. A tuned mass damper is the worst case: its peaks come out a third too
low conventionally.

![Frame builder page: a cantilever with a tuned mass damper at its tip; the Response tab shows the tip's receptance with two peaks, the state space's (solid) a third higher than the conventional model's (dashed), and the response to a 40 Hz force from rest](images/frame_response.png)

---

← [Modes](modes.md) · [Contents](../README.md#contents) · [Signals and random](signals-and-random.md) →
