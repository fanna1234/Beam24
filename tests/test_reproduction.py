"""Check cold-start routing and failure cases without GPU access or real data."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from paired_records import validate_pairs
from check_gpu_idle import foreign_processes
import check_gpu_idle
from check_real_quality import validate as validate_quality
from check_spib_extraction import verify as verify_extraction
from run_external_hierarchy import validate_timing

spec = importlib.util.spec_from_file_location("baseline_fetch", ROOT / "artifact/scripts/get_baselines.py")
baseline_fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline_fetch)


class ReproductionTest(unittest.TestCase):
    def setUp(self):
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith("BEAM24_") and k not in ("PYTHON_BIN", "NVCC", "CUDACXX", "CUDA_HOME")}

    def command(self, *arguments, env=None):
        return subprocess.run(["bash", str(ROOT / "reproduce.sh"), *arguments],
                              env=env or self.env, text=True, capture_output=True)

    def test_help_and_bad_targets(self):
        self.assertIn("external-hierarchy", self.command("--help").stdout)
        self.assertEqual(self.command("unknown").returncode, 2)
        self.assertEqual(self.command("smoke", "hierarchy").returncode, 2)

    def test_all_targets_support_dry_run(self):
        targets = json.loads((ROOT / "artifact/manifest.json").read_text())["targets"]
        for target in (*targets, "all"):
            with self.subTest(target=target):
                result = self.command(target, "--dry-run")
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_interpreter_override_and_conflict(self):
        env = dict(self.env, BEAM24_PYTHON="/a path/python")
        result = self.command("external-hierarchy", "--dry-run", env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("/a\\ path/python", result.stdout)
        env["PYTHON_BIN"] = "/different/python"
        self.assertEqual(self.command("smoke", "--dry-run", env=env).returncode, 2)

    @unittest.skipUnless(shutil.which("cmake"), "CMake is unavailable")
    def test_requested_cuda_does_not_silently_configure_cpu_only(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(["cmake", "-S", str(ROOT), "-B", directory,
                                     "-DBEAM24_ENABLE_CUDA=ON", "-DCMAKE_CUDA_COMPILER=NOTFOUND"],
                                    text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("CUDA requested", result.stdout + result.stderr)

    def test_gpu_guard_preserves_other_work(self):
        self.assertEqual(foreign_processes(""), [])
        self.assertEqual(foreign_processes("1, Xorg, 1 MiB"), [])
        self.assertTrue(foreign_processes("99, training.py, 1024 MiB"))

    def test_gpu_guard_checks_only_the_selected_device(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-example"}), \
             patch("check_gpu_idle.subprocess.run", return_value=SimpleNamespace(stdout="")) as mocked:
            check_gpu_idle.assert_idle()
            self.assertIn("GPU-example", mocked.call_args.args[0])
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}):
            with self.assertRaises(RuntimeError):
                check_gpu_idle.assert_idle()

    def test_baseline_dependency_closure(self):
        sources = json.loads((ROOT / "baselines/manifest.json").read_text())["external_sources"]
        selected = baseline_fetch.resolve_sources(sources, "ccglib")
        self.assertEqual([source["id"] for source in selected], ["cudawrappers", "xtl", "xtensor", "ccglib"])
        with self.assertRaises(ValueError):
            baseline_fetch.resolve_sources([{"id": "a", "requires": ["a"]}], "a")

    @unittest.skipUnless(shutil.which("git"), "Git is unavailable")
    def test_pinned_but_dirty_baseline_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory) / "ccglib"
            subprocess.run(["git", "init", "-q", str(checkout)], check=True)
            (checkout / "source.txt").write_text("original\n")
            subprocess.run(["git", "-C", str(checkout), "add", "source.txt"], check=True)
            subprocess.run(["git", "-C", str(checkout), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                            "commit", "-qm", "fixture"], check=True)
            commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
            url = "https://example.invalid/ccglib.git"
            subprocess.run(["git", "-C", str(checkout), "remote", "add", "origin", url], check=True)
            (checkout / "source.txt").write_text("changed\n")
            with self.assertRaisesRegex(SystemExit, "dirty"):
                baseline_fetch.fetch({"id": "ccglib", "repository": url, "commit": commit}, Path(directory), False)


class ResultIntegrityTest(unittest.TestCase):
    def samples(self):
        return [dict(process=i, position=j, variant=v, milliseconds=4 if v == "E" else 1)
                for i in range(6) for j, v in enumerate(("E", "H") if i % 2 == 0 else ("H", "E"))]

    def test_complete_pairs_pass(self):
        self.assertEqual(len(validate_pairs(self.samples(), ("E", "H"))), 6)

    def test_duplicate_missing_wrong_order_and_nonfinite_pairs_fail(self):
        original = self.samples()
        cases = [original[:-1], original[:-1] + [original[0]]]
        for key, value in (("position", 1), ("milliseconds", float("nan")),
                           ("milliseconds", float("inf")), ("milliseconds", 0)):
            changed = copy.deepcopy(original)
            changed[0][key] = value
            cases.append(changed)
        for case in cases:
            with self.assertRaises(ValueError):
                validate_pairs(case, ("E", "H"))

    def test_external_timing_contract_rejects_static_or_graph_substitution(self):
        row = dict(batch=256, m=1024, n=1024, k=512, coarse_m=128, coarse_k=128,
                   fine_m=128, warmup=20, iterations=100, engine="ccglib_basic", launch_mode="direct")
        validate_timing(row, "E")
        with self.assertRaises(RuntimeError):
            validate_timing(dict(row, engine="ccglib_opt_static_a"), "E")
        with self.assertRaises(RuntimeError):
            validate_timing(dict(row, graph=True, dispatch="hierarchical"), "H")

    def quality(self):
        return {"contract": {"K": 48, "sample_rate": 1000, "frequency_hz": [160.0, 180.0]},
                "details": [dict(file=f"{i}.mat", variant="Q2_localf4_joint_top2",
                                 map_correlation=0.999975, peak_shift_deg=0) for i in range(10)],
                "summary": {"Q2_localf4_joint_top2": {"files": 10, "mean_map_correlation": 0.999975}}}

    def test_quality_does_not_use_latency_tolerance(self):
        anchors = {"spib48_vla_mean_map_correlation": 0.99998, "spib48_vla_max_peak_shift_deg": 0.25}
        validate_quality(self.quality(), anchors)
        changed = self.quality()
        for row in changed["details"]:
            row["map_correlation"] = 0.970747
        changed["summary"]["Q2_localf4_joint_top2"]["mean_map_correlation"] = 0.970747
        with self.assertRaises(ValueError):
            validate_quality(changed, anchors)
        changed = self.quality()
        changed["details"][1] = changed["details"][0]
        with self.assertRaises(ValueError):
            validate_quality(changed, anchors)

    def test_partial_or_changed_extraction_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            archive = base / "source.zip"
            unpacked = base / "unpacked"
            unpacked.mkdir()
            with zipfile.ZipFile(archive, "w") as output:
                for i in range(10):
                    name = f"sample-{i}.mat"
                    output.writestr(name, b"fixture")
                    (unpacked / name).write_bytes(b"fixture")
            verify_extraction(archive, unpacked)
            (unpacked / "sample-0.mat").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verify_extraction(archive, unpacked)


if __name__ == "__main__":
    unittest.main()
