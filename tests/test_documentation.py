"""Keep the delivered documentation and overview linked to real source files."""

import hashlib
import json
from pathlib import Path
import re
import unittest
from urllib.parse import unquote
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


class DocumentationTest(unittest.TestCase):
    def test_local_links_resolve(self):
        for document in [ROOT / "README.md", ROOT / "artifact/README.md", *sorted((ROOT / "docs").glob("*.md"))]:
            for target in re.findall(r"\]\(([^)]+)\)", document.read_text()):
                if "://" in target or target.startswith("mailto:"):
                    continue
                path, _, fragment = unquote(target).partition("#")
                resolved = (document.parent / path).resolve() if path else document
                with self.subTest(document=document.name, target=target):
                    self.assertTrue(resolved.exists(), str(resolved))
                    if fragment and resolved.suffix == ".md":
                        titles = re.findall(r"^#{1,6}\s+(.+)$", resolved.read_text(), re.M)
                        anchors = {re.sub(r"[^\w -]", "", title.lower()).replace(" ", "-") for title in titles}
                        self.assertIn(fragment, anchors)

    def test_overview_has_a_source_bound_four_mode_example(self):
        source = json.loads((ROOT / "docs/assets/beam24-overview.json").read_text())
        self.assertEqual(hashlib.sha256((ROOT / source["transform_source"]).read_bytes()).hexdigest(), source["transform_source_sha256"])
        self.assertEqual(len(set(source["retained_modes"])), 2)
        self.assertAlmostEqual(sum(source["mode_energy_fractions"]), 1)
        svg = ET.parse(ROOT / "docs/assets/beam24-overview.svg").getroot()
        self.assertEqual(svg.attrib["viewBox"], "0 0 1160 360")
        self.assertNotIn("script", {node.tag.rsplit("}", 1)[-1] for node in svg.iter()})


if __name__ == "__main__":
    unittest.main()
