//! Vortex-induced vibration: a cylinder on a spring, y, coupled to a van der Pol wake oscillator, q
//! (`self_excited.Viv`, its `viv_steps_reference`), stepped by classical RK4 for many flow speeds side by side.
//!
//! For each case (a reduced velocity), with the per-case S (the lift per unit q), r (the damping, structural
//! and fluid) and Ω_f (the shedding frequency), and the shared mass m_t, stiffness k, coupling A/D and ε:
//!
//!   ÿ = (S q − r ẏ − k y) / m_t,   q̈ = (A/D) ÿ − ε Ω_f (q² − 1) q̇ − Ω_f² q.
//!
//! Working on it: keep the arithmetic `ode.rk4`'s, step for step (k1 … k4, then y + h/6 (k1 + 2k2 + 2k3 + k4)), so
//! the Python reference and this agree to rounding (`tests/test_native.py`).

/// The shared coefficients and each case's own.
pub struct Viv<'a> {
    pub mt: f64,
    pub k: f64,
    pub a_d: f64,
    pub eps: f64,
    pub s: &'a [f64],
    pub r: &'a [f64],
    pub wf: &'a [f64],
}

impl Viv<'_> {
    #[inline]
    fn rhs(&self, c: usize, z: [f64; 4]) -> [f64; 4] {
        let [y, v, q, p] = z;
        let wf = self.wf[c];
        let acc = (self.s[c] * q - self.r[c] * v - self.k * y) / self.mt;
        let dp = self.a_d * acc - self.eps * wf * (q * q - 1.0) * p - wf * wf * q;
        [v, acc, p, dp]
    }

    /// `steps` RK4 steps of h from `z0` (4 × cases, state-major), every `keep`-th state from the first:
    /// rows × (4 × cases), state-major in each row.
    pub fn run(&self, z0: &[f64], h: f64, steps: usize, keep: usize) -> crate::Result<Vec<f64>> {
        let n = self.s.len();
        if self.r.len() != n || self.wf.len() != n || z0.len() != 4 * n {
            return Err(format!(
                "viv_steps: {n} cases need r, Ω_f of {n} and a start of 4 × {n}, not {}, {} and {}",
                self.r.len(),
                self.wf.len(),
                z0.len()
            ));
        }
        if keep == 0 {
            return Err("viv_steps: keep must be at least 1".into());
        }
        let rows = steps / keep + 1;
        let mut out = vec![0.0; rows * 4 * n];
        let (half, sixth) = (h / 2.0, h / 6.0);
        for c in 0..n {
            let mut y = [z0[c], z0[n + c], z0[2 * n + c], z0[3 * n + c]];
            let mut put = |row: usize, y: &[f64; 4]| {
                for (s, v) in y.iter().enumerate() {
                    out[row * 4 * n + s * n + c] = *v;
                }
            };
            put(0, &y);
            for i in 1..=steps {
                let k1 = self.rhs(c, y);
                let k2 = self.rhs(c, std::array::from_fn(|s| y[s] + half * k1[s]));
                let k3 = self.rhs(c, std::array::from_fn(|s| y[s] + half * k2[s]));
                let k4 = self.rhs(c, std::array::from_fn(|s| y[s] + h * k3[s]));
                y = std::array::from_fn(|s| y[s] + sixth * (k1[s] + 2.0 * k2[s] + 2.0 * k3[s] + k4[s]));
                if i % keep == 0 {
                    put(i / keep, &y);
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_no_lift_or_wake_the_cylinder_rings_as_a_damped_spring() {
        // S = 0: y decouples; from y = 1 at rest, undamped, it swings as cos ωt.
        let (mt, k) = (2.0, 8.0); // ω = 2
        let viv = Viv {
            mt,
            k,
            a_d: 0.0,
            eps: 0.0,
            s: &[0.0],
            r: &[0.0],
            wf: &[0.0],
        };
        let (h, steps) = (1e-3, 1000);
        let z = viv.run(&[1.0, 0.0, 0.0, 0.0], h, steps, steps).unwrap();
        assert!((z[4] - (2.0_f64 * 1.0).cos()).abs() < 1e-9, "{}", z[4]);
    }

    #[test]
    fn a_shape_that_does_not_fit_is_an_error() {
        let viv = Viv {
            mt: 1.0,
            k: 1.0,
            a_d: 0.0,
            eps: 0.0,
            s: &[0.0, 1.0],
            r: &[0.0],
            wf: &[0.0],
        };
        assert!(viv.run(&[0.0; 8], 1e-3, 10, 1).is_err());
    }
}
