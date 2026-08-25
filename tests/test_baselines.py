import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class BaselineInventoryTest(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((ROOT / "baselines/manifest.json").read_text())

    def test_required_operator_and_attribution_comparators_are_measured(self):
        by_id = {item["id"]: item for item in self.manifest["baselines"]}
        for baseline_id in (
            "CCGLIB_OPT_STATIC_A",
            "CCGLIB_HIERARCHICAL_TOP1",
            "DENSE_FUSED_TOP1_CONTROL",
        ):
            self.assertEqual(by_id[baseline_id]["status"], "measured")

    def test_required_baseline_suite_is_complete(self):
        incomplete = {
            item["id"]
            for item in self.manifest["baselines"]
            if item["required"] and item["status"] in {"pending", "partial"}
        }
        self.assertEqual(incomplete, set())
        self.assertEqual(
            next(
                item for item in self.manifest["baselines"]
                if item["id"] == "BEAM24_REP_CUSPARSELT"
            )["table_role"],
            "backend_ablation",
        )
        self.assertEqual(
            next(
                item for item in self.manifest["baselines"]
                if item["id"] == "CUFFT_UNIFORM_SPATIAL_FREQUENCY_LOWER_BOUND"
            )["table_role"],
            "lower_bound",
        )


if __name__ == "__main__":
    unittest.main()
