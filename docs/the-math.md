[Contents](../README.md#contents)

# The math, briefly

The *Theory* tab in the app explains this in more depth; this is the outline.

**Model.** Newton's second law for each mass gives

$$M\ddot{x} + C\dot{x} + Kx = f(t)$$

where $M$ is diagonal (the masses) and $C$, $K$ are tridiagonal, assembled element by
element: each spring or damper between two neighbours adds $\begin{bmatrix}v & -v\\ -v & v\end{bmatrix}$
to their rows and columns (element 1 connects to ground). Everything is SI.

**Classical modal analysis.** Ignoring damping, $x = \phi\cos\omega t$ gives the symmetric
generalized eigenproblem $K\phi = \omega^2 M\phi$, which has N real natural frequencies
$\omega_r$ and real mode shapes $\phi_r$. With mass-normalized shapes collected in $\Phi$,
$\Phi^T M\Phi = I$ and $\Phi^T K\Phi = \mathrm{diag}(\omega_r^2)$. Damping is estimated
from the diagonal of $\Phi^T C\Phi$:

$$\zeta_r = \frac{\phi_r^T C\,\phi_r}{2\omega_r}$$

This is exact only if $\Phi^T C\Phi$ is diagonal (*proportional* damping, e.g.
$C = \alpha M + \beta K$). The app reports a coupling index, the largest off-diagonal term
relative to the diagonal, to show how far from proportional the damping is.

**State-space (complex) modal analysis.** With $z = [x,\ \dot{x}]$ the N second-order
equations become 2N first-order ones:

$$\dot{z} = Az + Bf, \qquad A = \begin{bmatrix} 0 & I \\ -M^{-1}K & -M^{-1}C \end{bmatrix}$$

The eigenproblem $A\psi = \lambda\psi$ is exact for any damping. It has 2N eigenvalues,
$\lambda = -\zeta\omega_n \pm i\omega_d$ in conjugate pairs, and eigenvectors of the form
$[\psi_x,\ \lambda\psi_x]$. One pair together gives a real motion
$x(t) = 2\,\mathrm{Re}(\psi_x e^{\lambda t})$. With non-proportional damping $\psi_x$ is
complex, so each mass has its own phase. The two methods are linked by matching each
damped pair to the undamped mode it most resembles (the modal assurance criterion, MAC).

**Modal coordinates.** Each method gives a change of coordinates that the time histories
can plot. Classical: $x = \Phi q$, so $q = \Phi^T M x$, and the equations become

$$\ddot{q} + \Phi^T C\Phi\,\dot{q} + \mathrm{diag}(\omega_r^2)\,q = \Phi^T f$$

which separate into N single-DOF equations only if $\Phi^T C\Phi$ is diagonal. Otherwise
damping transfers motion between the $q_r$. State-space: $z = V\eta$, so $\eta = V^{-1}z$,
and $\dot{\eta} = \Lambda\eta + V^{-1}Bf$ is decoupled for any damping. A conjugate pair
contributes $2\,\mathrm{Re}(\psi_x\eta_r)$ to $x$. The app scales each coordinate to the
mode's displacement at the mass where its normalized shape is 1: $\phi_{r,\max}\,q_r$
and $2\,\mathrm{Re}(\eta_r)$.

**Ground motion.** When the wall moves as $x_g(t)$, element 1 stretches by $x_1 - x_g$,
so the ground enters as a force on mass 1:

$$M\ddot{x} + C\dot{x} + Kx = (k_1x_g + c_1\dot{x}_g)\,e_1, \qquad
\frac{X}{X_g} = H(\omega)\,e_1\,(k_1 + i\omega c_1)$$

The simulator steps this exactly for a piecewise-linear $x_g$ by adding $x_g$ and its
slope to the state of the matrix exponential, as for the energy integrals below.

**Frequency response.** The receptance is solved directly at each frequency:
$H(\omega) = (K - \omega^2 M + i\omega C)^{-1}$. It is also a sum of modal terms. With the
state-space eigenvectors $V$ (and $V^{-1}$), every eigenvalue contributes a residue matrix,

$$H(\omega) = \sum_{r=1}^{2N} \frac{R_r}{i\omega - \lambda_r}, \qquad R_r = (V)_{x,r}\,(V^{-1})_{r,\dot{x}}\,M^{-1}$$

which is exact for any damping. For proportional damping it reduces to the classical
$H = \sum_r \phi_r\phi_r^T / (\omega_r^2 - \omega^2 + 2i\zeta_r\omega_r\omega)$.

**Energy.** Multiplying the equations of motion by $\dot{x}^T$ gives the power balance

$$\frac{d}{dt}\Big(\tfrac12\dot{x}^TM\dot{x} + \tfrac12 x^TKx\Big) = f^T\dot{x} - \dot{x}^TC\dot{x}$$

so the energy given plus the work done by the force equals the energy stored plus the
energy dissipated. With $q = \Phi^T Mx$ the stored energy is
$\sum_r \tfrac12(\dot{q}_r^2 + \omega_r^2 q_r^2)$ for any damping. The off-diagonal terms of
$\Phi^T C\Phi$ move energy between these terms.

**Time simulation.** The simulator advances $\dot{z} = Az + Bf$ with the exact
discrete-time solution. Over a step $h$, with the force varying linearly across the step
(first-order hold),

$$z_{k+1} = e^{Ah}z_k + \Gamma_0 f_k + \Gamma_1 f_{k+1}$$

where $e^{Ah}$, $\Gamma_0$ and $\Gamma_1$ come from a single matrix exponential of an augmented
matrix. The free response is exact, so the scheme is unconditionally stable and adds no
numerical damping, however stiff the springs are made: any decay you see is physical. The
step size adapts to the fastest mode, the drive frequency and the pulse length.

The work and dissipation over a step are integrals of quadratic forms in the state and
force. Adding the force and its slope to the state makes the step a linear system
$\dot{w} = A_w w$. Van Loan's matrix exponential then gives each integral exactly as
$w_k^T W w_k$, so the energy balance closes to rounding error.

**Friction.** Coulomb friction makes the chain nonlinear, but only piecewise: while each
mass keeps sliding the same way or stays stuck, friction is a constant force on each
sliding mass, and a stuck mass's row of $A$ is zeroed. Adding the friction forces to the
state again gives a linear system, stepped exactly. Within each step the simulator finds
the first moment a sliding mass stops or a stuck one breaks free (the pull of its
springs, dampers and the force exceeds the static friction $F_s \ge F_f$), on a cubic
through the step's ends, steps exactly to it and updates which masses stick. A mass that
stops sticks if the pull on it is at most $F_s$ and slides back otherwise. Over each piece friction takes out exactly
$F_f$ times the distance slid, so the energy balance still closes.

---

[Contents](../README.md#contents)
