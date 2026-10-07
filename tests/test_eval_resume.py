"""run_eval must reuse an answer only for the same prompt + text, and compute_metrics must
score only answers that match the current benchmark text."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ahuike.prompts import input_hash, target_json  # noqa: E402


class TestResume(unittest.TestCase):
    def test_input_hash_depends_on_text(self):
        self.assertNotEqual(input_hash("a"), input_hash("b"))
        self.assertEqual(input_hash("a"), input_hash("a"))

    def test_run_eval_writes_hash(self):
        src = (ROOT / "scripts" / "run_eval.py").read_text(encoding="utf-8")
        self.assertIn('"text_hash": input_hash(it["text"])', src)

    def test_metrics_ignore_stale_predictions(self):
        case = json.loads((ROOT / "data" / "cases_test.jsonl").read_text(encoding="utf-8").splitlines()[0])
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "cases_test.jsonl").write_text(json.dumps(case) + "\n", encoding="utf-8")
            item = {"case_id": case["id"], "lang": "en", "variant": "plain", "text": case["text_en"]}
            (d / "eval_items.jsonl").write_text(json.dumps(item) + "\n", encoding="utf-8")
            right = target_json(case["triage"], case["triggers"], "x")
            wrong = target_json("HOME_CARE" if case["triage"] != "HOME_CARE" else "EMERGENCY_REFER_NOW", [], "x")
            preds = [
                {"system": "base", **{k: item[k] for k in ("case_id", "lang", "variant")},
                 "text_hash": input_hash(item["text"]), "output": right},
                {"system": "base", **{k: item[k] for k in ("case_id", "lang", "variant")},
                 "text_hash": "stale", "output": wrong},  # answer to an OLD text: must be ignored
            ]
            (d / "preds.jsonl").write_text("\n".join(json.dumps(p) for p in preds) + "\n", encoding="utf-8")
            subprocess.run([sys.executable, str(ROOT / "scripts" / "compute_metrics.py"), "--preds", str(d / "preds.jsonl"),
                            "--cases", str(d / "cases_test.jsonl"), "--out", str(d), "--boot", "10"],
                           check=True, capture_output=True)
            m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(m["base"]["overall"]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
