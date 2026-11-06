"""core.foh's FohSteps against stepping the FOH update directly: several inputs, across calls, diagonalized
and not (a free chain: its rigid-body λ = 0 is defective, so Φ's eigenvectors are near parallel, cond ~10⁶
and up)."""

import numpy as np
import pytest

from vib_tutorial.core.foh import FohSteps
from vib_tutorial.core.model import ChainSystem
from vib_tutorial.core.simulator import foh_discretize, state_space

CHAIN = ChainSystem([1.0, 2.0, 1.5], [400.0, 300.0, 500.0], [2.0, 1.0, 3.0])
FREE = ChainSystem([1.0] * 3, [0.0, 400.0, 400.0], [0.0, 2.0, 2.0])


@pytest.mark.parametrize(
    "system, h, modal",
    [(CHAIN, 1 / 160, True), (CHAIN, 1 / 2000, True), (FREE, 1 / 100, False), (FREE, 1 / 2000, False)],
)
def test_steps_match_the_update(system, h, modal):
    A, B = state_space(system)
    B = B[:, [0, 2]]
    f = np.random.default_rng(0).standard_normal((3000, 2))
    steps = FohSteps(A, B, h)
    assert steps.modal is modal
    z = np.vstack([steps.run(f[:1100]), steps.run(f[1100:1101]), steps.run(f[1101:])])  # carried across calls
    Phi, G0, G1 = foh_discretize(A, B, h)
    zk, f0, ref = np.zeros(A.shape[0]), np.zeros(2), []
    for f1 in f:
        zk = Phi @ zk + G0 @ f0 + G1 @ f1
        ref.append(zk)
        f0 = f1
    np.testing.assert_allclose(z, ref, atol=1e-10 * np.abs(ref).max())
    np.testing.assert_allclose(steps.state, ref[-1], atol=1e-10 * np.abs(ref).max())


@pytest.mark.parametrize("h", [1 / 160, 1 / 2000])
def test_a_state_set_is_where_it_carries_on(h):
    A, B = state_space(FREE)
    steps = FohSteps(A, B[:, :1], h)
    start = np.array([0.01, -0.02, 0.005, 0.0, 0.1, 0.0])
    steps.state = start
    np.testing.assert_allclose(steps.state, start, atol=1e-15)
    Phi, _, _ = foh_discretize(A, B[:, :1], h)
    np.testing.assert_allclose(steps.run(np.zeros(1))[0], Phi @ start, atol=1e-12)
