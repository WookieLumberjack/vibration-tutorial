"""The rotor page's tooltips: what each of its controls does."""

from __future__ import annotations

from ..core.rotor import (
    MODE_LIMIT,
)

ORDER_TIPS = {
    2.0: "<p>Twice the speed. A shaft stiffer one way than the other (a two-pole rotor's slots, a "
    "keyway, a crack) bends twice per turn under its weight, and misalignment pulls twice per turn "
    "too. Where a forward branch meets this line, at about half its critical speed, a horizontal "
    "rotor with such a shaft whirls at 2X: a <i>secondary critical</i> (set an asymmetry Δ or a "
    "crack, and tick <i>Horizontal rotor</i>).</p>",
    3.0: "<p>Three times the speed. A breathing crack, opened and closed by the shaft's weight once per "
    "turn, changes the stiffness with 3X in it too, so the weight drives a 3X whirl where a forward "
    "branch meets this line, at about a third of the critical speed. An asymmetric shaft's stiffness "
    "changes only at 2X: no 3X.</p>",
    0.5: "<p>Half the speed. Oil in a plain journal bearing swirls at just under half the shaft's speed, "
    "so its whirl starts near ½X (<i>oil whirl</i>) and, once ½X has passed the first natural "
    "frequency, locks onto it (<i>oil whip</i>): set the bearings' swirl ratio λ to 0.5. A rub "
    "or loose parts can also respond at ½X.</p>",
}


SPEED_TIP = (
    "<p>The speed the rotor ramps toward, at the ramp rate. Drag the red line on the Campbell diagram "
    "to set it too.</p><p>Ramp slowly through a critical speed and the whirl builds up to its "
    "steady-state peak; ramp quickly and it passes before the whirl can grow.</p>"
)


SWEEP_TIP = (
    "<p>Ramp to the first speed and let the start-up transient die away, then run up to the second "
    "speed and coast back down, at the ramp rate above. Once per revolution the 1X vector is measured "
    "at the chosen station, and drawn over the steady-state curve on the Bode and polar plots.</p>"
    "<p>Run it again at another ramp rate to compare: the faster the ramp, the lower the peak, and the "
    "further past the critical speed it comes.</p>"
)


HOLD_SWEEPS_TIP = (
    "<p>Keep the earlier sweeps on the plots when a new one starts, to compare ramp rates. Unticked, a "
    "new sweep replaces them. Changing the rotor clears them, since the steady-state curve changes.</p>"
)


TAP_TIP = (
    "<p>Hit the disc downward (an impulse of 5 mN·s). The tap sets every mode ringing at its own "
    "natural frequency: both the forward and the backward whirl of each mode. With the disc off "
    "centre, at speed, the gyroscopic effect pulls their frequencies apart and the orbit becomes a "
    "rosette (watch the Campbell diagram at the current speed).</p>"
)


REPEAT_TAP_TIP = (
    "<p>Tap the disc again and again as it runs, as in an impact test at speed. On the <i>Waterfall</i> "
    "tab each tap rings the free whirl frequencies at the speed it came at, so over a run-up the peaks "
    "trace the Campbell diagram's branches.</p>"
)


BLOCK_TIP = (
    "<p>How long each spectrum's block of signal is. The frequency resolution is 1 / block: 0.5 s "
    "resolves 2 Hz. But while the speed ramps, it changes over a block, smearing the 1X peak: at "
    "500 rpm/s, by 250 rpm (4 Hz) over 0.5 s. Each block overlaps the last by half.</p>"
)


INTERNAL_TIP = (
    "<p>Damping inside the rotating shaft (material hysteresis, slip in shrink fits and couplings), as "
    "the damping c<sub>i</sub> of a damper across the shaft at the disc. It resists the bending rate "
    "the spinning shaft sees, so below the critical speed it damps the forward whirl, and above it "
    "(the shaft then turns faster than it whirls) it drives it.</p>"
    "<p>Try 20 N·s/m: the rotor becomes unstable at 3,910 rpm and, above that, whirls at its first "
    "natural frequency, about 27 Hz, whatever the speed.</p>"
)


SHAFT_MASS_TIP = (
    "<p>The Jeffcott model's shaft has no mass: it is a spring holding the disc. Tick this to give it "
    "its mass (and its sections' rotary inertia and gyroscopic moment), spread along it: it then "
    "becomes a beam finite element model, and has bending modes of its own.</p>"
    "<p>A shaft light beside its disc changes little: the first critical speed drops a little, as if "
    "part of the shaft's mass sat on the disc. Make the shaft long and the disc light and the "
    "massless model fails: the shaft's own bending modes come down into the speed range.</p>"
)


ELEMENTS_TIP = (
    "<p>How many beam elements model the shaft, shared between its stretches by length. Each has a "
    "displacement and a slope at both ends, bending as a cubic in between. More elements, closer to "
    "the true beam's modes.</p>"
    "<p><i>Consistent</i> mass spreads each element's mass with the same cubic shapes as its "
    "stiffness. It is a Rayleigh–Ritz model, so at rest and undamped its frequencies are never too "
    "low, and come down toward the true ones as elements are added. <i>Lumped</i> puts half of each "
    "element's mass at each end. It is simpler, but it has no such bound: with few elements much of the "
    "mass sits on the bearings, where it hardly moves, and the frequencies come out high.</p>"
    "<p>The stations are kept as they are; the rest of the shaft follows them, plus its own modes "
    "with the stations held, up to " + f"{MODE_LIMIT:,.0f}" + " Hz (a Craig–Bampton reduction). The "
    "modes far above those would only slow the simulation down.</p>"
)


KXY_TIP = (
    "<p>Cross-coupled stiffness: a displacement in x pushes the journal in y, and one in y pushes it "
    "back in −x (k<sub>yx</sub> = −k<sub>xy</sub>). Fluid-film bearings, seals and impeller "
    "clearances do this, the fluid being dragged round by the shaft. Positive k<sub>xy</sub> pushes a "
    "forward whirl along its path: it lowers the forward modes' damping, and past the bearings' own "
    "damping, the rotor is unstable.</p><p>Here k<sub>xy</sub> is constant; in a real bearing it "
    "grows with speed, so read it as the value at the speed you look at. Try 15,000 N/m on both.</p>"
)


BOW_TIP = (
    "<p>A shaft bent for good (by a thermal bow, a sag left after standing, or a bent repair), by δ<sub>b</sub> at "
    "the disc, toward an angle measured from the keyphasor mark in the direction of spin. It turns with the "
    "shaft, so it drives the rotor once per revolution like an unbalance, but with a force kδ<sub>b</sub> that does "
    "not grow with speed.</p><p>Turning slowly, the probes read the bow itself (the <i>slow-roll "
    "runout</i>), and the polar plot starts from it rather than from the origin. Well above the critical "
    "speed the disc stays on the bearings' axis and the shaft bends round it. Put the bow at 180° to the "
    "heavy spot and the two cancel at one speed.</p>"
)


SKEW_TIP = (
    "<p>The disc mounted out of square: its own axis tilted by τ from the shaft's, toward an angle "
    "measured from the keyphasor mark in the direction of spin. Spinning, the tilted disc's inertia makes a "
    "moment (I<sub>d</sub> − I<sub>p</sub>) τ Ω² that turns with the shaft: a <i>couple unbalance</i>. "
    "It drives the disc's tilt directly, and pushes a thin disc (I<sub>p</sub> &gt; I<sub>d</sub>) toward "
    "square to the spin axis, bending the shaft at the disc by −τ at high speed.</p><p>At midspan on "
    "identical bearings it only tilts the disc; move the disc off centre and it moves it too.</p>"
)


ASYMMETRY_TIP = (
    "<p>A shaft stiffer about one axis than the other, as the slots of a two-pole generator's rotor make "
    "it (or a keyway, or a crack): bending toward its stiff axis it has EI(1 + Δ), across it EI(1 − Δ). "
    "The angle is the stiff axis's from the keyphasor mark, in the direction of spin.</p>"
    "<p>Its stiffness turns with it, so it has two critical speeds, one per axis, and between them it is "
    "unstable unless the damping holds it: the whirl grows locked to 1X. Laid horizontal, its weight acts "
    "on a stiffness that changes twice per turn and drives a 2X whirl, largest near half the critical "
    "speed (the <i>gravity critical</i>; plot 2X on the Bode plot). The 1X response depends on where the "
    "heavy spot sits from the stiff axis.</p>"
)


CRACK_TIP = (
    "<p>A transverse crack in the shaft, cut to a depth a<sub>c</sub> from the surface (a<sub>c</sub>/d of the "
    "shaft's diameter), its mouth at the angle "
    "given from the keyphasor mark. Open, it is a hinge: the shaft's slope jumps across it, much more "
    "under bending that pulls its mouth open than across (its compliance comes from fracture "
    "mechanics, for depths up to half the shaft).</p>"
    "<p><i>Breathing</i>, it is opened and closed by the shaft's weight as it turns (tick "
    "<i>Horizontal rotor</i>): fully open with its mouth on the side the sag stretches, shut half a turn "
    "later. That stiffness changes with 1X, 2X and 3X in it, so the weight drives 2X and 3X whirls, "
    "largest near a half and a third of the critical speed, and changes the 1X. Unticked, it is held "
    "open (a notch): then it is an asymmetric shaft, softened.</p>"
    "<p>The position is z/L from bearing A, as the disc's.</p>"
)


GRAVITY_TIP = (
    "<p>A horizontal rotor: gravity pulls the disc and the journals down. The shaft sags and the "
    "supports give, so the rotor whirls about a point below the bearings' centre. The model is linear, "
    "so the vibration about that point is just as without gravity: on these supports gravity alone "
    "makes no whirl. With cross-coupled bearings or internal damping the sag also goes sideways.</p>"
    "<p>An asymmetric or cracked shaft is the exception: its stiffness turns under its weight, so "
    "gravity drives it at 2X (and a breathing crack at 1X and 3X too). Without gravity nothing opens a "
    "breathing crack.</p>"
)


SAG_TIP = (
    "<p>Draw the motion with the static sag, as a DC-coupled probe reads the shaft's position in its "
    "bearing: the orbits sit below the centre. Untick to draw only the vibration about where gravity "
    "holds the rotor, as an AC-coupled probe does. The tracking, the spectrum and the phase are always "
    "of the vibration.</p>"
)


FILM_TIP = (
    "<p>Run both journals in plain journal bearings, oil films on rigid housings, in place of the springs "
    "and dampers: the bearing set on the <i>Fluid-film bearing</i> tab, each carrying its share of the "
    "rotor's weight (none on a vertical rotor). Their stiffness and damping change with the speed as the "
    "film lifts the journal, and their cross-coupling, the oil swirling at half the speed, makes oil whirl "
    "and whip from the bearing itself rather than from λ.</p><p>Laid horizontal, the journals sit off "
    "centre where the film holds them, and rise along the bearing's locus as the speed rises: tick "
    "<i>Static sag</i> to see them move. Only the journal's mass is left to set.</p><p>Upright, unloaded, "
    "a plain bearing has no direct stiffness, only the swirl's cross-coupling: a vertical rotor in such "
    "bearings whirls at any speed.</p>"
)


MAGNET_TIP = (
    "<p>Float both journals in active magnetic bearings, set on the <i>Magnetic bearing</i> tab: on each "
    "axis a pair of electromagnets, whose pull grows as the journal comes nearer (a negative stiffness), "
    "and a PID controller that sets their current from the journal's position. The bearing's stiffness "
    "and damping come from the gains, and change with the frequency: the controller's lag makes the "
    "damping negative at high frequencies, and a mode that gets there grows.</p><p>With the integral, "
    "gravity cannot move the journals: they stay centred, and the shaft alone sags. Only the journal's "
    "mass is left to set.</p>"
)


NONLINEAR_FILM_TIP = (
    "<p>Step the rotor with the films' whole force, not their linearization about where the journals rest. "
    "Off centre the film stiffens and damps ever harder toward the wall, so the journals stay inside their "
    "clearance: through a critical speed, and in oil whirl, which settles to a limit cycle at half the "
    "speed. In oil whip the shaft's whirl still grows, the films holding the journals but not the shaft's "
    "bending, until it would rub.</p><p>Only the time response: the Campbell diagram, stability, Bode "
    "and polar curves are always of the linearized films. Untick to step the linear films, as they "
    "predict. A shaft that turns its stiffness (asymmetric or cracked) keeps the films linear. The whole "
    "force is slow to step: past what a frame can compute, the page runs slower than the speed set.</p>"
)


SWIRL_TIP = (
    "<p>Fluid swirl ratio λ of a fluid-film bearing: the oil in the clearance is dragged round at λ times the "
    "shaft's speed: ½ in short-bearing theory, just under it (0.42 to 0.48) in real plain bearings. It makes a cross-coupled stiffness that grows "
    "with speed, k<sub>xy</sub> = λcΩ on top of the constant one (c the bearing's damping). Once the swirl "
    "λΩ outruns a forward whirl's frequency, the oil feeds it faster than c damps it: with λ = 0.5 the "
    "rotor goes unstable at twice its first natural frequency (<i>oil whip</i>).</p>"
)
