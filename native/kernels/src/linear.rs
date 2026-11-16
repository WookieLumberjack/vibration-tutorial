//! The chain's linear steps (no friction): `Simulator._advance_linear_reference`, which this mirrors.
//!
//! Each step is the exact first-order-hold update z₁ = Φ z₀ + g₀ u₀ + g₁ u₁ for a scalar input u
//! (a force on one mass, or the ground's motion) linear across the step. The Python works out Φ, g₀
//! and g₁ (`foh_input_step`, cached by step size and input) and the inputs at each step's end
//! (`ForceController.samples`); this runs the steps, a matrix-vector product each.

use crate::Result;
use crate::linalg::{Mat, RowsMut};

/// One step size's update for one input: z₁ = Φ z₀ + g₀ u₀ + g₁ u₁.
pub struct LinearSteps {
    phi: Mat,
    g0: Vec<f64>,
    g1: Vec<f64>,
}

impl LinearSteps {
    /// The steps of `foh_input_step`'s (Φ, g₀, g₁).
    pub fn new(phi: Mat, g0: Vec<f64>, g1: Vec<f64>) -> Result<Self> {
        let ns = phi.rows;
        phi.check("phi", ns, ns)?;
        if g0.len() != ns || g1.len() != ns {
            return Err("g0 and g1 must match phi".into());
        }
        Ok(LinearSteps { phi, g0, g1 })
    }

    /// From the state z with the input u0, through the inputs `u` (one per step, at its end): each step's
    /// state into `out` (steps × len(z)).
    pub fn advance(&self, z: &[f64], u0: f64, u: &[f64], out: RowsMut) -> Result<()> {
        let ns = self.phi.rows;
        if z.len() != ns || out.dim() != (u.len(), ns) {
            return Err("the state, inputs or output don't match the chain".into());
        }
        let mut a = z.to_vec();
        let mut b = vec![0.0; ns];
        let mut f0 = u0;
        for (row, &f1) in out.data.chunks_exact_mut(ns).zip(u) {
            for ((bi, g0), g1) in b.iter_mut().zip(&self.g0).zip(&self.g1) {
                *bi = g0 * f0 + g1 * f1;
            }
            self.phi.mul_add(&a, &mut b);
            row.copy_from_slice(&b);
            std::mem::swap(&mut a, &mut b);
            f0 = f1;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steps_the_update_and_refuses_a_wrong_shape() {
        // A 1×1 update, z₁ = ½ z₀ + u₀ + 2 u₁: from 1 under u = 0, 1, 1 it goes 0.5 + 0 + 2, then 1.25 + 1 + 2.
        let steps = LinearSteps::new(Mat::from_rows(1, 1, vec![0.5]), vec![1.0], vec![2.0]).unwrap();
        let mut out = [0.0; 2];
        let rows = RowsMut {
            data: &mut out,
            rows: 2,
            cols: 1,
        };
        steps.advance(&[1.0], 0.0, &[1.0, 1.0], rows).unwrap();
        assert_eq!(out, [2.5, 4.25]);
        let rows = RowsMut {
            data: &mut out,
            rows: 2,
            cols: 1,
        };
        assert!(steps.advance(&[1.0, 0.0], 0.0, &[1.0, 1.0], rows).is_err());
        assert!(LinearSteps::new(Mat::zeros(2, 2), vec![1.0], vec![1.0, 1.0]).is_err());
    }
}
