//! A linear system with nonlinear forces, stepped by the exact first-order hold with Newton at each step's
//! end: `DuffingSimulator._steps_reference`, `StartUpSimulator._steps_reference` and
//! `nonlinear_test._cycle_reference`, which this mirrors.
//!
//! The linear part is z' = A z + B u, stepped exactly with u linear over the step: z₁ = Φ z₀ + Γ₀ u₀ + Γ₁ u₁.
//! The inputs are an applied part, given for every step, plus the force laws' part, which depends on the
//! state: u₁ = a₁ + L(z₁, t₁). With z₁ = p + Γ₁ u₁ (p = Φ z₀ + Γ₀ u₀ known), Newton solves
//! r(u) = u − a₁ − L(p + Γ₁ u) = 0 for u₁, its Jacobian I − (∂L/∂z) Γ₁ a few inputs square. Second order,
//! as the trapezoidal rule is: the error quarters as the step halves.

use crate::Result;
use crate::linalg::{Mat, RowsMut, solve};

/// A force law: what it adds to the inputs at a state, and how that changes with the state.
#[derive(Clone, Debug)]
pub enum Law {
    /// A cubic spring with a pumped linear part on coordinate `i`: −(k₃ x² + k₁ c) x into input `col`,
    /// c the step's clock (a pump's cos φ).
    Cubic { col: usize, i: usize, k3: f64, k1: f64 },
    /// A drag rising as the square of a speed: k r |r| into input `col`, r = z_i / `reference`.
    Quadratic {
        col: usize,
        i: usize,
        k: f64,
        reference: f64,
    },
    /// A motor's air-gap torque into input `col` (`torsional_startup.Motor.torque`), the clock the time.
    Motor(Motor),
    /// A spring's force f(d) on the stretch d = z_i − z_j (z_i alone without j): −f into input `col`, +f into
    /// `back` (the mass at its other end, if not the ground).
    Spring {
        col: usize,
        back: Option<usize>,
        i: usize,
        j: Option<usize>,
        force: Spring,
    },
    /// Coulomb friction smoothed over a speed v_s: −F tanh(v / v_s) into input `col`, v = z_i.
    Friction {
        col: usize,
        i: usize,
        force: f64,
        smooth: f64,
    },
}

/// A spring's force on its stretch.
#[derive(Clone, Copy, Debug)]
pub enum Spring {
    /// k₃ d³.
    Cubic(f64),
    /// k (|d| − gap) sign d once the gap has closed, nothing before.
    Gap { gap: f64, k: f64 },
}

impl Spring {
    /// (f(d), f'(d)).
    fn at(self, d: f64) -> (f64, f64) {
        match self {
            Spring::Cubic(k3) => (k3 * d * d * d, 3.0 * k3 * d * d),
            Spring::Gap { gap, k } => {
                let over = d.abs() - gap;
                if over > 0.0 {
                    (k * over * 1.0_f64.copysign(d), k)
                } else {
                    (0.0, 0.0)
                }
            }
        }
    }
}

/// An induction motor started direct on line, or a synchronous one on its cage with a field that latches
/// on below `field_slip` (`torsional_startup.Motor`; its docstring has the model).
#[derive(Clone, Debug)]
pub struct Motor {
    pub col: usize,
    pub angle: usize, // the shaft's angle and speed in the state
    pub speed: usize,
    pub sync_speed: f64, // rad/s
    pub line: f64,       // ω_e, rad/s
    pub breakdown: f64,  // Kloss's T_b at the slip s_b
    pub breakdown_slip: f64,
    pub transient: f64, // induction: the switch-on torque, decaying over the time constant
    pub time_constant: f64,
    pub synchronous: bool,
    pub pole_pairs: f64,
    pub pulsating: f64, // synchronous: the poles' T_p, and the field's T_f once on
    pub field: f64,
    pub field_slip: f64,
    pub field_on: bool,
}

impl Motor {
    /// (T, dT/dθ, dT/dΩ) at time t, shaft angle and speed: `Motor.torque`, line for line.
    fn torque(&self, t: f64, angle: f64, speed: f64) -> (f64, f64, f64) {
        let ws = self.sync_speed;
        let (s, sb) = (1.0 - speed / ws, self.breakdown_slip);
        let d = s * s + sb * sb;
        let tc = 2.0 * self.breakdown * s * sb / d;
        let slope = 2.0 * self.breakdown * sb * (sb * sb - s * s) / (d * d);
        let d_speed = -slope / ws;
        if !self.synchronous {
            return (
                tc + self.transient * (-t / self.time_constant).exp() * (self.line * t).sin(),
                0.0,
                d_speed,
            );
        }
        let delta = self.line * t - self.pole_pairs * angle;
        let tf = if self.field_on { self.field } else { 0.0 };
        let (s2, c2) = (2.0 * delta).sin_cos();
        let poles = self.pulsating * s2 + tf * delta.sin();
        let d_angle = -self.pole_pairs * (2.0 * self.pulsating * c2 + tf * delta.cos());
        (tc + poles, d_angle, d_speed)
    }
}

impl Law {
    /// Add the law's inputs at z (clock c) into u, and their derivatives into du (inputs × states).
    fn add(&self, z: &[f64], c: f64, u: &mut [f64], du: &mut Mat) {
        match self {
            &Law::Cubic { col, i, k3, k1 } => {
                let x = z[i];
                u[col] -= (k3 * x * x + k1 * c) * x;
                du.set(col, i, du.at(col, i) - (3.0 * k3 * x * x + k1 * c));
            }
            &Law::Quadratic { col, i, k, reference } => {
                let r = z[i] / reference;
                u[col] += k * r * r.abs();
                du.set(col, i, du.at(col, i) + 2.0 * k * r.abs() / reference);
            }
            Law::Motor(m) => {
                let (t, d_angle, d_speed) = m.torque(c, z[m.angle], z[m.speed]);
                u[m.col] += t;
                du.set(m.col, m.angle, du.at(m.col, m.angle) + d_angle);
                du.set(m.col, m.speed, du.at(m.col, m.speed) + d_speed);
            }
            &Law::Spring { col, back, i, j, force } => {
                let (f, slope) = force.at(z[i] - j.map_or(0.0, |j| z[j]));
                // the stretch's derivative is +1 in z_i and −1 in z_j; the force goes −f on col, +f on back
                let mut put = |row: usize, sign: f64| {
                    u[row] += sign * f;
                    du.set(row, i, du.at(row, i) + sign * slope);
                    if let Some(j) = j {
                        du.set(row, j, du.at(row, j) - sign * slope);
                    }
                };
                put(col, -1.0);
                if let Some(back) = back {
                    put(back, 1.0);
                }
            }
            &Law::Friction { col, i, force, smooth } => {
                let t = (z[i] / smooth).tanh();
                u[col] -= force * t;
                du.set(col, i, du.at(col, i) - force * (1.0 - t * t) / smooth);
            }
        }
    }

    /// At a step's start: a synchronous motor's field latches on once the slip is below `field_slip`.
    fn start_step(&mut self, z: &[f64]) {
        if let Law::Motor(m) = self
            && m.synchronous
            && m.field > 0.0
            && !m.field_on
            && 1.0 - z[m.speed] / m.sync_speed < m.field_slip
        {
            m.field_on = true;
        }
    }

    fn check(&self, ns: usize, nu: usize) -> bool {
        match self {
            &Law::Cubic { col, i, .. } | &Law::Quadratic { col, i, .. } => col < nu && i < ns,
            Law::Motor(m) => m.col < nu && m.angle < ns && m.speed < ns,
            &Law::Spring { col, back, i, j, .. } => {
                col < nu && back.is_none_or(|b| b < nu) && i < ns && j.is_none_or(|j| j < ns)
            }
            &Law::Friction { col, i, .. } => col < nu && i < ns,
        }
    }
}

/// The steps of one FOH update (Φ, Γ₀, Γ₁) under the force laws.
pub struct NewtonSteps {
    phi: Mat,
    g0: Mat,
    g1: Mat,
    laws: Vec<Law>,
    tol: f64,
    iterations: usize,
    watch: Option<(usize, f64)>,
}

impl NewtonSteps {
    /// Newton stops when an update is within `tol` of the inputs' size (or 1, if they're smaller), after at
    /// most `iterations`. `watch` = (i, limit): stop after the step where |z_i| passes limit (or isn't finite).
    pub fn new(
        phi: Mat,
        g0: Mat,
        g1: Mat,
        laws: Vec<Law>,
        tol: f64,
        iterations: usize,
        watch: Option<(usize, f64)>,
    ) -> Result<Self> {
        let (ns, nu) = (phi.rows, g0.cols);
        phi.check("phi", ns, ns)?;
        g0.check("g0", ns, nu)?;
        g1.check("g1", ns, nu)?;
        if let Some(bad) = laws.iter().find(|law| !law.check(ns, nu)) {
            return Err(format!("{bad:?} reaches past the state ({ns}) or the inputs ({nu})"));
        }
        if watch.is_some_and(|(i, _)| i >= ns) {
            return Err("the watched coordinate isn't in the state".into());
        }
        Ok(NewtonSteps {
            phi,
            g0,
            g1,
            laws,
            tol,
            iterations,
            watch,
        })
    }

    /// Whether the motor's field is on (None without a motor).
    pub fn field_on(&self) -> Option<bool> {
        self.laws.iter().find_map(|law| match law {
            Law::Motor(m) => Some(m.field_on),
            _ => None,
        })
    }

    /// The laws' inputs at z (clock c), and their derivatives (inputs × states).
    fn laws(&self, z: &[f64], m: f64) -> (Vec<f64>, Mat) {
        let mut u = vec![0.0; self.g0.cols];
        let mut du = Mat::zeros(self.g0.cols, self.phi.rows);
        for law in &self.laws {
            law.add(z, m, &mut u, &mut du);
        }
        (u, du)
    }

    /// From the state z with the inputs u0, a step per row of `applied` (the inputs' applied part at each
    /// step's end; `clock` what the laws read that varies in time, there: a pump's cos φ, a motor's time):
    /// each step's state into `out` and its inputs into `inputs`. Returns how many steps it took: all of
    /// them, or up to the one where the watched coordinate passed its limit.
    pub fn run(
        &mut self,
        z: &[f64],
        u0: &[f64],
        applied: &Mat,
        clock: &[f64],
        mut out: RowsMut,
        mut inputs: RowsMut,
    ) -> Result<usize> {
        let (ns, nu, steps) = (self.phi.rows, self.g0.cols, applied.rows);
        if z.len() != ns
            || u0.len() != nu
            || applied.cols != nu
            || clock.len() != steps
            || out.dim() != (steps, ns)
            || inputs.dim() != (steps, nu)
        {
            return Err("the state, inputs or outputs don't match the update".into());
        }
        let mut z = z.to_vec();
        let mut u = u0.to_vec();
        let mut p = vec![0.0; ns];
        let mut zt = vec![0.0; ns];
        for (k, &m) in clock.iter().enumerate() {
            let a = applied.row(k);
            self.laws.iter_mut().for_each(|law| law.start_step(&z));
            p.fill(0.0);
            self.phi.mul_add(&z, &mut p);
            self.g0.mul_add(&u, &mut p);
            // Newton on u from the start's inputs
            for _ in 0..self.iterations {
                zt.copy_from_slice(&p);
                self.g1.mul_add(&u, &mut zt);
                let (l, dl) = self.laws(&zt, m);
                let r: Vec<f64> = (0..nu).map(|j| u[j] - a[j] - l[j]).collect();
                let jac = dl.matmul(&self.g1).combine(-1.0, &Mat::eye(nu), 1.0);
                let du = solve(jac, Mat::from_rows(nu, 1, r));
                let size = u.iter().fold(1.0_f64, |s, v| s.max(v.abs()));
                let step = du.data.iter().fold(0.0_f64, |s, v| s.max(v.abs()));
                u.iter_mut().zip(&du.data).for_each(|(v, d)| *v -= d);
                if step <= self.tol * size {
                    break;
                }
            }
            // the inputs the converged end state gives, and that state
            zt.copy_from_slice(&p);
            self.g1.mul_add(&u, &mut zt);
            let (l, _) = self.laws(&zt, m);
            (0..nu).for_each(|j| u[j] = a[j] + l[j]);
            z.copy_from_slice(&p);
            self.g1.mul_add(&u, &mut z);
            out.row(k).copy_from_slice(&z);
            inputs.row(k).copy_from_slice(&u);
            if let Some((i, limit)) = self.watch
                && (z[i].abs() > limit || !z[i].is_finite())
            {
                return Ok(k + 1);
            }
        }
        Ok(steps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_law_free_step_is_the_linear_update() {
        // Without laws the inputs are the applied ones and z₁ = Φ z₀ + Γ₀ u₀ + Γ₁ u₁.
        let phi = Mat::from_rows(2, 2, vec![0.9, 0.1, -0.1, 0.9]);
        let (g0, g1) = (
            Mat::from_rows(2, 1, vec![0.01, 0.1]),
            Mat::from_rows(2, 1, vec![0.02, 0.2]),
        );
        let mut steps = NewtonSteps::new(phi, g0, g1, vec![], 1e-14, 10, None).unwrap();
        let (mut out, mut inputs) = ([0.0; 2], [0.0; 1]);
        let n = steps
            .run(
                &[1.0, 0.0],
                &[1.0],
                &Mat::from_rows(1, 1, vec![2.0]),
                &[0.0],
                RowsMut {
                    data: &mut out,
                    rows: 1,
                    cols: 2,
                },
                RowsMut {
                    data: &mut inputs,
                    rows: 1,
                    cols: 1,
                },
            )
            .unwrap();
        assert_eq!(n, 1);
        assert!((out[0] - (0.9 + 0.01 + 0.04)).abs() < 1e-15 && (out[1] - (-0.1 + 0.1 + 0.4)).abs() < 1e-15);
        assert_eq!(inputs, [2.0]);
    }

    #[test]
    fn a_cubic_step_satisfies_its_law_and_stops_past_the_watch() {
        let phi = Mat::from_rows(2, 2, vec![0.9, 0.1, -0.1, 0.9]);
        let (g0, g1) = (
            Mat::from_rows(2, 1, vec![0.01, 0.1]),
            Mat::from_rows(2, 1, vec![0.02, 0.2]),
        );
        let law = Law::Cubic {
            col: 0,
            i: 0,
            k3: 50.0,
            k1: 3.0,
        };
        let mut steps = NewtonSteps::new(phi, g0, g1, vec![law], 1e-14, 20, Some((0, 0.5))).unwrap();
        let (mut out, mut inputs) = ([0.0; 4], [0.0; 2]);
        let n = steps
            .run(
                &[1.0, 0.0],
                &[0.0],
                &Mat::from_rows(2, 1, vec![1.0, 1.0]),
                &[0.5, 0.5],
                RowsMut {
                    data: &mut out,
                    rows: 2,
                    cols: 2,
                },
                RowsMut {
                    data: &mut inputs,
                    rows: 2,
                    cols: 1,
                },
            )
            .unwrap();
        let x = out[0];
        assert!((inputs[0] - (1.0 - (50.0 * x * x + 1.5) * x)).abs() < 1e-12);
        assert_eq!(n, 1, "x = {x} is past the watch's 0.5");
    }

    #[test]
    fn every_laws_derivatives_are_its_slopes() {
        // Each law's du/dz against central differences, at a state where every one is working (the gap closed,
        // the motor slipping, the field on).
        let motor = Motor {
            col: 2,
            angle: 0,
            speed: 3,
            sync_speed: 157.0,
            line: 314.0,
            breakdown: 1000.0,
            breakdown_slip: 0.2,
            transient: 300.0,
            time_constant: 0.05,
            synchronous: true,
            pole_pairs: 2.0,
            pulsating: 200.0,
            field: 500.0,
            field_slip: 0.05,
            field_on: true,
        };
        let laws = vec![
            Law::Cubic {
                col: 0,
                i: 0,
                k3: 40.0,
                k1: 3.0,
            },
            Law::Quadratic {
                col: 1,
                i: 3,
                k: 50.0,
                reference: 150.0,
            },
            Law::Motor(motor.clone()),
            Law::Motor(Motor {
                synchronous: false,
                ..motor
            }),
            Law::Spring {
                col: 1,
                back: Some(0),
                i: 1,
                j: Some(0),
                force: Spring::Cubic(7.0),
            },
            Law::Spring {
                col: 0,
                back: None,
                i: 0,
                j: None,
                force: Spring::Gap { gap: 0.1, k: 9.0 },
            },
            Law::Friction {
                col: 2,
                i: 2,
                force: 0.7,
                smooth: 0.3,
            },
        ];
        let ns = 4;
        let steps = NewtonSteps::new(
            Mat::eye(ns),
            Mat::zeros(ns, 3),
            Mat::zeros(ns, 3),
            laws,
            1e-12,
            10,
            None,
        )
        .unwrap();
        let z = [0.3, -0.4, 0.2, 120.0];
        let (_, du) = steps.laws(&z, 0.04);
        for j in 0..ns {
            let e = 1e-6 * z[j].abs().max(1.0);
            let (mut hi, mut lo) = (z, z);
            hi[j] += e;
            lo[j] -= e;
            let (uh, ul) = (steps.laws(&hi, 0.04).0, steps.laws(&lo, 0.04).0);
            for r in 0..3 {
                let fd = (uh[r] - ul[r]) / (2.0 * e);
                assert!(
                    (du.at(r, j) - fd).abs() <= 1e-6 * (1.0 + fd.abs()),
                    "du[{r},{j}] {} vs {fd}",
                    du.at(r, j)
                );
            }
        }
    }
}
