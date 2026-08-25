import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class HierarchyContractTest(unittest.TestCase):
    def test_integer_candidate_formula_matches_quality_grid(self):
        full_beams = 1024
        coarse_beams = 128
        stride = round((full_beams - 1) / (coarse_beams - 1))
        self.assertEqual(stride, 8)
        for coarse_index in range(coarse_beams):
            integer_center = (
                coarse_index * (full_beams - 1) + (coarse_beams - 1) // 2
            ) // (coarse_beams - 1)
            quality_center = round(
                coarse_index * (full_beams - 1) / (coarse_beams - 1)
            )
            self.assertEqual(integer_center, quality_center)
            start = max(0, integer_center - stride // 2)
            stop = min(full_beams - 1, integer_center + stride // 2)
            self.assertLessEqual(stop - start + 1, 16)

    def test_primary_useful_work_ratio(self):
        hierarchical = 128 * 128 + 128 * 512
        exhaustive = 1024 * 512
        self.assertEqual(hierarchical / exhaustive, 0.15625)

    def test_small_workload_fallback_is_maintained(self):
        source = (ROOT / "src/cuda/beam24_hierarchical_system.cu").read_text()
        self.assertIn("HIERARCHY_MIN_BATCH_SNAPSHOTS = 3072", source)
        self.assertIn("static_cast<size_t>(batches) * snapshots", source)
        self.assertIn('"exhaustive_small_workload"', source)


if __name__ == "__main__":
    unittest.main()
