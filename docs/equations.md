← [One oscillator](one-oscillator.md) · [Contents](../README.md#contents) · [Modes](modes.md) →

# Equations

Where the equations come from: free-body diagrams and Lagrange's equations, then a rigid body in 3D.

The pages in the order the app's tabs take them.

- [Deriving the equations](#deriving-the-equations): free-body diagrams against Lagrange's equations: a car in bounce and pitch, and a double pendulum
- [3D rigid body](#3d-rigid-body): a body tumbling in space: its inertia tensor, principal axes, Euler's equations and the intermediate-axis flip

## Deriving the equations

A car's body in bounce and pitch, one rigid body on a front and a rear spring, derived two ways with the
page's own numbers written in: free-body diagrams (forces, and moments about G or another point) and
Lagrange's equations (T, V and Rayleigh's D, then their derivatives). The coordinates can be the centre
of mass, any point P on the body, or the two ends: M, C and K change (static coupling in K, dynamic in
M, the free-body diagram's rows not symmetric until mixed), the frequencies don't. The car swings in each
mode about its node, the modes are swept against the rear spring, and it rides a speed hump to show
Olley's flat ride.

The page's second system is a double pendulum. Its free-body diagram needs rod 2's pull, found and
eliminated; Lagrange's equations never see it. Its whole equations are nonlinear, and it is linearized only
after differentiating, in the rods' angles or the elbow's bend. Released, it swings by the whole equations
(DOP853, the energy held to 10⁻¹⁰) beside the linear model. Released high, a twin a millionth of a radian
further over drifts away exponentially: chaos.

![Deriving the equations page, moments about the front axle: in x<sub>P</sub> and θ at P = 1.2 m ahead of G both M and K couple, the free-body diagram's moment balance about P keeps G's inertia force, and leaving it out (J<sub>P</sub>θ̈ = ΣM<sub>P</sub>) would give 0.932 and 1.38 Hz instead of 1.25 and 1.41](images/deriving_equations.png)

![Deriving the equations page, the double pendulum released at 120° and 120°: two releases 10⁻⁶ rad apart drift apart exponentially, the difference doubling every 0.49 s, until by 8 s they are radians apart; the lower bob's trail loops over the top](images/deriving_pendulum.png)

## 3D rigid body

A body free to turn any way, with no gravity and no support: a box, a cylinder or a sphere, with a point
mass that can sit anywhere on it. Its inertia tensor about the centre of mass comes from the shape's own
moved there by parallel axes plus the point mass's, products of inertia and all, and its eigenvectors are
the principal axes (the *Inertia* tab writes both out). Spun about one of them, L is along ω; about any
other they point apart, L fixed in space and ω wandering round it. The notes derive L = Iω, the transport
theorem for vectors in turning axes and Euler's equations, and why a body spun about its middle axis turns
over and back (the tennis-racket effect: 1° off at 6 rad/s, the default box turns over in about 1.4 s).
A torque held in space or in the body for a set time (or a *Kick*) changes L by its impulse. The
*Components* tab plots ω, L, τ or the tensor in fixed, body or principal axes, with the equations written
out with numbers at any instant, and *Check* compares the energy and L with what the torque put in.

The *Orientation* tab writes how the body is turned four ways at once: the rotation matrix, Euler's z-x-z
angles (with their rates, which blow up near the gimbal lock at θ = 0), the quaternion with its axis and angle,
and the 4 × 4 homogeneous transform that carries the point mass to space. Its start pose is set there as
Euler angles: at θ = 0 changing φ or ψ turns the body the same way. The *Drift* tab steps the same torque-free run as a
rotation matrix, as Euler angles and as a quaternion at one coarse step and plots how far each drifts from the
true orientation: the quaternion least, the matrix more (and it stops being a rotation), the Euler angles most,
and from a start near the lock they fail. The view can trace ω's path in space or on the body itself, where,
torque-free, it runs round a closed loop (Poinsot's polhode). The body is drawn with shaded faces, lit
from the upper left, so its turning reads at a glance.

![3D rigid body page, the default 30 × 20 × 10 cm box spun at 6 rad/s 1° off its middle axis: 2.5 s on it has turned over, its middle-axis rate ω<sub>2</sub> gone from +6 to −5.8 rad/s while |L| holds at 0.05001 N·m·s, and the Components tab writes Euler's equations out with that instant's numbers](images/rigid_body.png)

---

← [One oscillator](one-oscillator.md) · [Contents](../README.md#contents) · [Modes](modes.md) →
