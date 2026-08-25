import unittest

import numpy as np

from src.quality.cusparselt_prune import (
    complex_matmul_fp16_output,
    prune_complex,
    prune_strip_real,
    prune_tile_real,
)


class CusparseLtPruneOracleTest(unittest.TestCase):
    def test_strip_keeps_two_per_group(self):
        values = np.arange(1.0, 17.0).reshape(2, 8)
        pruned = prune_strip_real(values)
        counts = np.count_nonzero(pruned.reshape(2, 2, 4), axis=-1)
        np.testing.assert_array_equal(counts, 2)
        np.testing.assert_array_equal(pruned[0], [0, 0, 3, 4, 0, 0, 7, 8])

    def test_tile_keeps_two_per_row_and_column(self):
        values = np.arange(1.0, 17.0).reshape(4, 4)
        pruned = prune_tile_real(values)
        np.testing.assert_array_equal(np.count_nonzero(pruned, axis=0), 2)
        np.testing.assert_array_equal(np.count_nonzero(pruned, axis=1), 2)

    def test_complex_prunes_components_independently(self):
        weights = np.asarray([[1 + 8j, 2 + 7j, 3 + 6j, 4 + 5j]], dtype=np.complex128)
        pruned = prune_complex(weights, "strip")
        np.testing.assert_array_equal(pruned.real, [[0, 0, 3, 4]])
        np.testing.assert_array_equal(pruned.imag, [[8, 7, 0, 0]])

    def test_complex_matmul_matches_four_call_rounding(self):
        weights = np.asarray([[1 + 2j, 3 + 4j]], dtype=np.complex128)
        snapshots = np.asarray([[5 + 6j], [7 + 8j]], dtype=np.complex128)
        output = complex_matmul_fp16_output(weights, snapshots)
        real_first = np.float16(1 * 5 + 3 * 7)
        expected_real = np.float16(np.float32(real_first) - 2 * 6 - 4 * 8)
        imag_first = np.float16(1 * 6 + 3 * 8)
        expected_imag = np.float16(np.float32(imag_first) + 2 * 5 + 4 * 7)
        self.assertEqual(output[0, 0], np.complex64(expected_real + 1j * expected_imag))


if __name__ == "__main__":
    unittest.main()
