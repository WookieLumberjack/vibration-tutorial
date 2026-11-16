//! A linear system with nonlinear forces, stepped by the exact first-order hold with Newton at each step's
//! end: `DuffingSimulator._steps_reference`, which this mirrors (and, as more pages come over, the others that
//! step the same way).
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
    /// A cubic spring with a pumped linear part on coordinate `i`: −(k₃ x² + k₁ m) x into input `col`,
    /// m the step's modulation (`run`'s `modulation`: a pump's cos φ).
    Cubic { col: usize, i: usize, k3: f64, k1: f64 },
}

impl Law {
    /// Add the law's inputs at z (modulation m) into u, and their derivatives into du (inputs × states).
    fn add(&self, z: &[f64], m: f64, u: &mut [f64], du: &mut Mat) {
        match *self {
            Law::Cubic { col, i, k3, k1 } => {
                let x = z[i];
                u[col] -= (k3 * x * x + k1 * m) * x;
                let d = du.at(col, i) - (3.0 * k3 * x * x + k1 * m);
                du.set(col, i, d);
            }
        }
    }

    fn check(&self, ns: usize, nu: usize) -> bool {
        match *self {
            Law::Cubic { col, i, .. } => col < nu && i < ns,
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

    /// The laws' inputs at z (modulation m), and their derivatives (inputs × states).
    fn laws(&self, z: &[f64], m: f64) -> (Vec<f64>, Mat) {
        let mut u = vec![0.0; self.g0.cols];
        let mut du = Mat::zeros(self.g0.cols, self.phi.rows);
        for law in &self.laws {
            law.add(z, m, &mut u, &mut du);
        }
        (u, du)
    }

    /// From the state z with the inputs u0, a step per row of `applied` (the inputs' applied part at each
    /// step's end; `modulation` the laws' m there): each step's state into `out` and its inputs into
    /// `inputs`. Returns how many steps it took: all of them, or up to the one where the watched
    /// coordinate passed its limit.
    pub fn run(
        &self,
        z: &[f64],
        u0: &[f64],
        applied: &Mat,
        modulation: &[f64],
        mut out: RowsMut,
        mut inputs: RowsMut,
    ) -> Result<usize> {
        let (ns, nu, steps) = (self.phi.rows, self.g0.cols, applied.rows);
        if z.len() != ns
            || u0.len() != nu
            || applied.cols != nu
            || modulation.len() != steps
            || out.dim() != (steps, ns)
            || inputs.dim() != (steps, nu)
        {
            return Err("the state, inputs or outputs don't match the update".into());
        }
        let mut z = z.to_vec();
        let mut u = u0.to_vec();
        let mut p = vec![0.0; ns];
        let mut zt = vec![0.0; ns];
        for (k, &m) in modulation.iter().enumerate() {
            let a = applied.row(k);
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
        let steps = NewtonSteps::new(phi, g0, g1, vec![], 1e-14, 10, None).unwrap();
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
        let steps = NewtonSteps::new(phi, g0, g1, vec![law], 1e-14, 20, Some((0, 0.5))).unwrap();
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
}
