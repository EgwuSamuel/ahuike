import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lafiya.metrics import clcc, detect_lang, mcnemar_exact, score_item, summarize  # noqa: E402
from lafiya.prompts import parse_output, target_json  # noqa: E402


def out(tri, signs=(), advice="Go now"):
    return target_json(tri, list(signs), advice)


GOLD = {
    "a": {"triage": "EMERGENCY_REFER_NOW", "triggers": ["convulsions"], "population": "child", "tags": []},
    "b": {"triage": "HOME_CARE", "triggers": [], "population": "pregnant", "tags": []},
}


class TestParse(unittest.TestCase):
    def test_valid_json(self):
        p = parse_output('Sure: {"triage": "HOME_CARE", "danger_signs": [], "advice": "x"}')
        self.assertTrue(p["valid_json"])
        self.assertEqual(p["triage"], "HOME_CARE")

    def test_fallback_and_garbage(self):
        self.assertEqual(parse_output("I think CLINIC_WITHIN_24H is right")["triage"], "CLINIC_WITHIN_24H")
        self.assertFalse(parse_output("I think CLINIC_WITHIN_24H")["valid_json"])
        self.assertIsNone(parse_output("no idea")["triage"])
        self.assertIsNone(parse_output("")["triage"])


class TestMetrics(unittest.TestCase):
    def items(self, preds):
        return [score_item({"case_id": cid, "lang": lang, "system": "s", "output": o}, GOLD[cid])
                for cid, lang, o in preds]

    def test_under_triage_counts_invalid(self):
        it = self.items([("a", "en", out("EMERGENCY_REFER_NOW", ["convulsions"])),
                         ("a", "ha", "garbage"),
                         ("b", "en", out("CLINIC_WITHIN_24H"))])
        s = summarize(it)
        self.assertAlmostEqual(s["under_triage_rate"], 0.5)
        self.assertAlmostEqual(s["over_triage_rate"], 1.0)
        self.assertAlmostEqual(s["accuracy"], 1 / 3)

    def test_clcc(self):
        langs = ("en", "ha", "yo", "ig")
        consistent = [("a", l, out("EMERGENCY_REFER_NOW")) for l in langs]
        inconsistent = [("b", l, out("HOME_CARE" if l != "ha" else "CLINIC_WITHIN_24H")) for l in langs]
        r = clcc(self.items(consistent + inconsistent))
        self.assertEqual(r["n_cases"], 2)
        self.assertAlmostEqual(r["clcc"], 0.5)
        self.assertEqual(r["worst_language"], "ha")
        self.assertAlmostEqual(r["worst_language_gap"], 0.5)

    def test_mcnemar(self):
        self.assertEqual(mcnemar_exact([True] * 5, [True] * 5)["p_value"], 1.0)
        r = mcnemar_exact([False] * 20, [True] * 20)
        self.assertLess(r["p_value"], 1e-5)

    def test_langid(self):
        self.assertEqual(detect_lang("Take the child to the health centre now."), "en")
        self.assertEqual(detect_lang("Ku kai yaro asibiti yanzu da gaggawa."), "ha")
        self.assertEqual(detect_lang("Ẹ gbé ọmọ náà lọ sí ilé ìwòsàn lẹ́sẹ̀kẹsẹ̀."), "yo")
        self.assertEqual(detect_lang("Kpọga nwa ahụ gaa ụlọ ọgwụ ozugbo."), "ig")

    def test_target_roundtrip(self):
        s = out("CLINIC_WITHIN_24H", ["fever"], "Je ka asibiti")
        self.assertEqual(json.loads(s)["danger_signs"], ["fever"])


if __name__ == "__main__":
    unittest.main()
