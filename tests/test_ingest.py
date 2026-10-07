"""Fill the real review sheet the way a reviewer would, then check ingest_review.py reads it back."""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHEET = ROOT / "docs" / "review" / "AHUIKE_Igbo_Review_Sheet.docx"


@unittest.skipUnless(SHEET.exists(), "review sheet not generated")
class TestIngest(unittest.TestCase):
    def test_roundtrip(self):
        import docx

        doc = docx.Document(str(SHEET))
        tabs = {}
        for t in doc.tables:
            head = " | ".join(c.text for c in t.rows[0].cells).lower()
            if "english (what it must say)" in head:
                tabs["advice"] = t
            elif "facts to include" in head:
                tabs["native"] = t
            elif "meaning" in head and "english" in head:
                tabs["verify"] = t
        # Task 1: row 1 wrong + corrected; row 2 correct, no correction
        tabs["advice"].rows[1].cells[4].text = "W"
        tabs["advice"].rows[1].cells[5].text = "IHE IZIZI: Kpọga nwa ahụ ụlọ ọgwụ ugbu a. Mee ka ahụ ya kpoo ọkụ."
        tabs["advice"].rows[2].cells[4].text = "c"
        # Task 3: first card written in Igbo
        native_id = tabs["native"].rows[1].cells[1].text.strip()
        tabs["native"].rows[1].cells[4].text = "Amụrụ m nwa ụbọchị iri na atọ gara aga. Ike na-agwụ m mgbe niile."
        # Task 4: two verdicts
        v_ids = [tabs["verify"].rows[i].cells[1].text.strip() for i in (1, 2)]
        tabs["verify"].rows[1].cells[4].text = "SAME"
        tabs["verify"].rows[2].cells[4].text = "Wrong"
        tabs["verify"].rows[2].cells[5].text = "says hiccups instead of cough"

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            filled = d / "filled.docx"
            doc.save(str(filled))
            data = d / "data"
            data.mkdir()
            out = subprocess.run([sys.executable, str(ROOT / "scripts" / "ingest_review.py"), str(filled),
                                  "--lang", "ig", "--data", str(data)], capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(out.returncode, 0, out.stderr)
            advice = json.loads((data / "advice_i18n.json").read_text(encoding="utf-8"))["ig"]
            self.assertTrue(advice["child|EMERGENCY_REFER_NOW"].startswith("IHE IZIZI"))
            self.assertIn("child|CLINIC_WITHIN_24H", advice)          # "correct" keeps N-ATLAS wording
            self.assertNotIn("child|HOME_CARE", advice)               # untouched row stays unresolved
            native = [json.loads(l) for l in (data / "native_items.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([(n["case_id"], n["variant"]) for n in native], [(native_id, "native")])
            verified = json.loads((data / "verified_ig.json").read_text(encoding="utf-8"))
            self.assertEqual(verified[v_ids[0]]["verdict"], "same")
            self.assertEqual(verified[v_ids[1]]["verdict"], "wrong")


if __name__ == "__main__":
    unittest.main()
