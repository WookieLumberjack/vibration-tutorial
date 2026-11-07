//! The chain's linear steps (no friction): `Simulator._advance_linear_reference`, which this mirrors.
//!
//! Each step is the exact first-order-hold update z₁ = Φ z₀ + g₀ u₀ + g₁ u₁ for a scalar input u
//! (a force on one mass, or the ground's motion) linear across the step. The Python works out Φ, g₀
//! and g₁ (`foh_input_step`, cached by step size and input) and the inputs at each step's end
//! (`ForceController.samples`); this runs the steps, a matrix-vector product each.

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::linalg::Mat;

/// One step size's update for one input: z₁ = Φ z₀ + g₀ u₀ + g₁ u₁.
#[pyclass(module = "vib_tutorial._native")]
pub struct LinearSteps {
    phi: Mat,
    g0: Vec<f64>,
    g1: Vec<f64>,
}

#[pymethods]
impl LinearSteps {
    /// The steps of `foh_input_step`'s (Φ, g₀, g₁).
    #[new]
    fn new(phi: PyReadonlyArray2<f64>, g0: PyReadonlyArray1<f64>, g1: PyReadonlyArray1<f64>) -> PyResult<Self> {
        let phi = Mat::from_numpy(&phi);
        let ns = phi.rows;
        phi.check("phi", ns, ns)?;
        let (g0, g1) = (g0.as_array().to_vec(), g1.as_array().to_vec());
        if g0.len() != ns || g1.len() != ns {
            return Err(PyValueError::new_err("g0 and g1 must match phi"));
        }
        Ok(LinearSteps { phi, g0, g1 })
    }

    /// From the state z with the input u0, through the inputs `u` (one per step, at its end): each step's
    /// state into `out` (steps × len(z)). Steps without the GIL.
    fn advance(
        &self,
        py: Python<'_>,
        z: PyReadonlyArray1<f64>,
        u0: f64,
        u: PyReadonlyArray1<f64>,
        mut out: PyReadwriteArray2<f64>,
    ) -> PyResult<()> {
        let ns = self.phi.rows;
        let (z, u) = (z.as_array(), u.as_array());
        let mut out = out.as_array_mut();
        if z.len() != ns || out.dim() != (u.len(), ns) {
            return Err(PyValueError::new_err(
                "the state, inputs or output don't match the chain",
            ));
        }
        let mut a = z.to_vec();
        let mut b = vec![0.0; ns];
        // A C-contiguous output (as `np.empty` makes) is written in place; any other through a copy.
        let mut copy = None;
        let rows = match out.as_slice_mut() {
            Some(rows) => rows,
            None => copy.insert(vec![0.0; u.len() * ns]).as_mut_slice(),
        };
        py.detach(|| {
            let mut f0 = u0;
            for (row, &f1) in rows.chunks_exact_mut(ns).zip(u.iter()) {
                for ((bi, g0), g1) in b.iter_mut().zip(&self.g0).zip(&self.g1) {
                    *bi = g0 * f0 + g1 * f1;
                }
                self.phi.mul_add(&a, &mut b);
                row.copy_from_slice(&b);
                std::mem::swap(&mut a, &mut b);
                f0 = f1;
            }
        });
        if let Some(rows) = copy {
            for (j, row) in rows.chunks_exact(ns).enumerate() {
                for i in 0..ns {
                    out[[j, i]] = row[i];
                }
            }
        }
        Ok(())
    }
}
