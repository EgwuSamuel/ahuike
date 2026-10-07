import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ahuike.backcheck import check, normalise_numbers  # noqa: E402
from translate_natlas import glossary_hint, split_sentences  # noqa: E402


def load(name):
    with open(ROOT / "data" / name, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]


class TestSplitAndGlossary(unittest.TestCase):
    def test_split_keeps_decimals(self):
        s = "CHEW note: 3 weeks old male infant. Findings: RR 44/min, temp 37.5°C. Please advise on triage."
        # Decimals never split; the short closing sentence is merged into its neighbour.
        self.assertEqual(split_sentences(s), [
            "CHEW note: 3 weeks old male infant.", "Findings: RR 44/min, temp 37.5°C. Please advise on triage."])

    def test_short_sentences_merged(self):
        self.assertEqual(split_sentences("Good evening. I am 9 months pregnant. I have leg cramps at night. What should I do?"),
                         ["Good evening. I am 9 months pregnant.", "I have leg cramps at night. What should I do?"])

    def test_split_roundtrip_all_cases(self):
        for c in load("cases_test.jsonl") + load("cases_train.jsonl")[:500]:
            self.assertEqual(" ".join(split_sentences(c["text_en"])), c["text_en"])

    def test_glossary_only_relevant_terms(self):
        self.assertIn("tari", glossary_hint("He has been coughing for 3 days.", "ha"))
        self.assertNotIn("gudawa", glossary_hint("He has been coughing for 3 days.", "ha"))
        self.assertIn("ìgbẹ́ gbuuru", glossary_hint("He has running stomach for 2 days.", "yo"))
        self.assertEqual(glossary_hint("What should I do?", "ig"), "")


class TestBackcheck(unittest.TestCase):
    def setUp(self):
        self.case = {"population": "child", "persona": "mother", "age_months": 31,
                     "findings": ["cough", "sunken_eyes", "fast_breathing", "diarrhoea"],
                     "durations": {"cough": 3, "diarrhoea": 6}, "vitals": {},
                     "triage": "CLINIC_WITHIN_24H", "triggers": ["fast_breathing"],
                     "text_en": "My daughter is 2 years 7 months old. She has had cough for 3 days now. "
                                "Her eyes look sunken. Her breathing is fast. She has been passing watery stool for 6 days."}

    def test_number_words(self):
        self.assertEqual(normalise_numbers("two years and seven months, twenty-four hours"),
                         "2 years and 7 months, 24 hours")

    def test_number_words_accepted(self):
        bt = ("My daughter is two years and seven months old. She has had a cough for three days. "
              "Her eyes are sunken. She breathes fast. She has had watery stool for six days.")
        ok, problems = check(self.case, "x" * 200, bt)
        self.assertTrue(ok, problems)

    def test_lost_label_relevant_finding_rejected(self):
        bt = ("My daughter is 2 years and 7 months old. She has had a cough for 3 days. "
              "Her eyes are sunken. She has had watery stool for 6 days.")  # fast breathing lost
        ok, problems = check(self.case, "x" * 200, bt)
        self.assertFalse(ok)
        self.assertIn("label_changed", problems)

    def test_lost_irrelevant_finding_tolerated(self):
        bt = ("My daughter is 2 years and 7 months old. She has had a cold for 3 days. "
              "Her eyes are sunken. She breathes fast. She has had watery stool for 6 days.")  # cough lost
        ok, problems = check(self.case, "x" * 200, bt)
        self.assertTrue(ok, problems)

    def test_spurious_red_flag_rejected(self):
        bt = ("My daughter is 2 years and 7 months old. She has had a cough for 3 days and had a convulsion. "
              "Her eyes are sunken. She breathes fast. She has had watery stool for 6 days.")
        self.assertFalse(check(self.case, "x" * 200, bt)[0])

    def test_refusal_rejected(self):
        bt = ("Sorry, but I can't fulfill this request. My daughter is 2 years and 7 months old. She has had a cough "
              "for 3 days. Her eyes are sunken. She breathes fast. She has had watery stool for 6 days.")
        ok, problems = check(self.case, "x" * 200, bt)
        self.assertFalse(ok)
        self.assertIn("meta_text", problems)

    def test_identity_passes_for_all_cases(self):
        for c in load("cases_test.jsonl") + load("cases_train.jsonl"):
            ok, problems = check(c, c["text_en"], c["text_en"])
            self.assertTrue(ok, (c["id"], problems))


if __name__ == "__main__":
    unittest.main()
