import unittest

import numpy as np


class BeamformingAlgebraTest(unittest.TestCase):
    def test_symmetric_f4_conjugated_basis_identity(self):
        axis = np.arange(4)
        f4 = np.exp(2j * np.pi * axis[:, None] * axis[None, :] / 4.0) / 2.0
        transform = np.kron(np.eye(2), f4)
        generator = np.random.default_rng(20260824)
        weights = generator.standard_normal((7, 8)) + 1j * generator.standard_normal((7, 8))
        snapshots = generator.standard_normal((8, 5)) + 1j * generator.standard_normal((8, 5))

        self.assertTrue(np.allclose(f4.T, f4, atol=1e-14, rtol=0.0))
        self.assertTrue(
            np.allclose(f4.conj().T @ f4, np.eye(4), atol=1e-14, rtol=0.0)
        )
        dense = weights.conj() @ snapshots
        transformed = (weights @ transform).conj() @ (transform @ snapshots)
        self.assertTrue(np.allclose(dense, transformed, atol=1e-12, rtol=1e-12))

    def test_conjugated_complex_decomposition(self):
        generator = np.random.default_rng(20260825)
        sparse = generator.standard_normal((3, 8)) + 1j * generator.standard_normal((3, 8))
        transformed = generator.standard_normal((8, 4)) + 1j * generator.standard_normal((8, 4))
        reference = sparse.conj() @ transformed
        real = sparse.real @ transformed.real + sparse.imag @ transformed.imag
        imag = sparse.real @ transformed.imag - sparse.imag @ transformed.real
        self.assertTrue(np.allclose(reference.real, real, atol=1e-12, rtol=1e-12))
        self.assertTrue(np.allclose(reference.imag, imag, atol=1e-12, rtol=1e-12))


if __name__ == "__main__":
    unittest.main()
