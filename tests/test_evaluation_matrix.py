import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class EvaluationMatrixTest(unittest.TestCase):
    def setUp(self):
        self.matrix = json.loads((ROOT / "baselines/evaluation_matrix.json").read_text())

    def test_internal_variants_are_not_main_rows(self):
        main_rows = set().union(*map(set, self.matrix["main_rows"].values()))
        for internal in self.matrix["internal_ablation_rows"]:
            self.assertNotIn(internal, main_rows)

    def test_every_dataset_has_a_performance_profile(self):
        profiles = set(self.matrix["operator_profiles"])
        for dataset in self.matrix["datasets"]:
            self.assertTrue(set(dataset["performance_profiles"]).issubset(profiles))


if __name__ == "__main__":
    unittest.main()
