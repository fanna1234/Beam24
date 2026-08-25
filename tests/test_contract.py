import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TITLE = "Beam24: Exploiting Regular-Array Phase Structure for 2:4 Sparse Tensor-Core Beamforming"


class ContractTest(unittest.TestCase):
    def test_title_is_coherent(self):
        self.assertIn(TITLE, (ROOT / "README.md").read_text())

    def test_readme_math_is_not_setext_heading(self):
        readme = (ROOT / "README.md").read_text()
        self.assertNotRegex(readme, r"(?m)^=+$")
        self.assertIn(r"\overline{\mathbf{W}}\mathbf{X}", readme)

    def test_artifact_targets(self):
        manifest = json.loads((ROOT / "artifact/manifest.json").read_text())
        self.assertEqual(
            set(manifest["targets"]),
            {"smoke", "gpu-smoke", "quality", "robustness", "finite-gpu-quality", "fourier-control", "system", "hierarchy"},
        )

    def test_claim_anchors_and_roles(self):
        anchors = json.loads((ROOT / "artifact/expected/reference_anchors.json").read_text())
        self.assertEqual(anchors["operator"]["sm120_batch256_m1024_n1024_k512_speedup"], 1.47097)
        self.assertEqual(anchors["system"]["materialized_ccglib_pipeline_speedup"], 2.37665)
        self.assertEqual(anchors["system"]["same_output_dense_fused_speedup"], 1.50462)
        self.assertEqual(anchors["system"]["hierarchical_external_ccglib_speedup"], 10.83733)
        self.assertEqual(
            anchors["system"]["hierarchical_external_same_algorithm_speedup"],
            3.87988,
        )
        self.assertEqual(anchors["system"]["hierarchical_exhaustive_ablation_speedup"], 4.56517)
        self.assertEqual(anchors["system"]["hierarchical_dense_attribution_speedup"], 1.16019)
        self.assertEqual(
            anchors["claim_roles"]["exhaustive_external_system_control"],
            "system.materialized_ccglib_pipeline_speedup",
        )
        self.assertEqual(
            anchors["claim_roles"]["exhaustive_internal_attribution"],
            "system.same_output_dense_fused_speedup",
        )
        self.assertEqual(anchors["quality"]["spib48_vla_mean_map_correlation"], 0.99998)
        self.assertEqual(
            anchors["quality"]["regular_ula_robustness_local_top1_min"],
            0.998046875,
        )
        self.assertEqual(
            anchors["quality"]["regular_ula_robustness_hierarchy_top1_min"],
            1.0,
        )
        self.assertEqual(
            anchors["quality"]["finite_gpu_complete_dense_exact_top1"],
            0.9992897727272727,
        )
        self.assertEqual(
            anchors["quality"]["finite_gpu_hierarchy_local_exact_top1"],
            1.0,
        )
        self.assertEqual(anchors["system"]["streamed_cufft_fp32_ms"], 7.768946885)


if __name__ == "__main__":
    unittest.main()
