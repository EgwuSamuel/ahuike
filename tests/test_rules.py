import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ahuike.cases import sample_case  # noqa: E402
from ahuike.protocol import BY_ID, CLINIC, EMERGENCY, HOME, POPULATIONS, triage  # noqa: E402
from ahuike.protocol.rules import TRIGGERS, effective_findings, rr_threshold  # noqa: E402


def child(findings, months=24, durations=None, vitals=None):
    return {"population": "child", "age_months": months, "findings": findings,
            "durations": durations or {}, "vitals": vitals or {}}


def infant(findings, days=20, vitals=None):
    return {"population": "infant", "age_days": days, "findings": findings,
            "durations": {}, "vitals": vitals or {}}


def preg(findings, ga=30):
    return {"population": "pregnant", "ga_weeks": ga, "findings": findings, "durations": {}}


def pp(findings, days=5):
    return {"population": "postpartum", "pp_days": days, "findings": findings, "durations": {}}


class TestIMCIChild(unittest.TestCase):
    def test_general_danger_signs_are_emergencies(self):
        for s in ("unable_to_drink", "vomits_everything", "convulsions", "lethargic"):
            self.assertEqual(triage(child([s]))[0], EMERGENCY, s)

    def test_fast_breathing_thresholds_by_age(self):
        self.assertEqual(rr_threshold("child", 8), 50)
        self.assertEqual(rr_threshold("child", 14), 40)
        # Same RR, different ages -> different decisions.
        self.assertEqual(triage(child(["cough"], 8, {"cough": 3}, {"rr": 45}))[0], HOME)
        self.assertEqual(triage(child(["cough"], 14, {"cough": 3}, {"rr": 45})),
                         (CLINIC, ["fast_breathing"]))
        self.assertEqual(triage(child(["cough"], 14, {"cough": 3}, {"rr": 40}))[0], CLINIC)
        self.assertEqual(triage(child(["cough"], 14, {"cough": 3}, {"rr": 39}))[0], HOME)

    def test_chest_indrawing_beats_fast_breathing(self):
        self.assertEqual(triage(child(["cough", "chest_indrawing", "fast_breathing"])),
                         (EMERGENCY, ["chest_indrawing"]))

    def test_dehydration_needs_two_signs(self):
        self.assertEqual(triage(child(["diarrhoea", "sunken_eyes"], durations={"diarrhoea": 3}))[0], HOME)
        self.assertEqual(triage(child(["diarrhoea", "sunken_eyes", "drinks_eagerly"],
                                      durations={"diarrhoea": 3})), (CLINIC, ["some_dehydration"]))
        self.assertEqual(triage(child(["diarrhoea", "sunken_eyes", "skin_pinch_very_slow"],
                                      durations={"diarrhoea": 3})), (EMERGENCY, ["severe_dehydration"]))

    def test_persistent_diarrhoea(self):
        self.assertEqual(triage(child(["diarrhoea"], durations={"diarrhoea": 13}))[0], HOME)
        self.assertEqual(triage(child(["diarrhoea"], durations={"diarrhoea": 14})),
                         (CLINIC, ["persistent_diarrhoea"]))
        self.assertEqual(triage(child(["diarrhoea", "restless_irritable", "drinks_eagerly"],
                                      durations={"diarrhoea": 15}))[0], EMERGENCY)

    def test_chronic_cough_and_fever(self):
        self.assertEqual(triage(child(["cough"], durations={"cough": 14})), (CLINIC, ["chronic_cough"]))
        self.assertEqual(triage(child(["fever"], durations={"fever": 2})), (CLINIC, ["fever"]))
        self.assertEqual(triage(child(["fever", "stiff_neck"], durations={"fever": 2})),
                         (EMERGENCY, ["stiff_neck"]))

    def test_mild_only_is_home(self):
        self.assertEqual(triage(child(["runny_nose", "teething"])), (HOME, []))


class TestYoungInfant(unittest.TestCase):
    def test_psbi_signs(self):
        for s in ("not_feeding_well", "convulsions", "chest_indrawing", "fever",
                  "low_temperature", "movement_only_when_stimulated", "jaundice_palms_soles"):
            self.assertEqual(triage(infant([s]))[0], EMERGENCY, s)

    def test_vitals_thresholds(self):
        self.assertEqual(triage(infant([], vitals={"rr": 59, "temp": 37.4}))[0], HOME)
        self.assertEqual(triage(infant([], vitals={"rr": 60, "temp": 37.0})), (EMERGENCY, ["fast_breathing"]))
        self.assertEqual(triage(infant([], vitals={"rr": 40, "temp": 37.5})), (EMERGENCY, ["fever"]))
        self.assertEqual(triage(infant([], vitals={"rr": 40, "temp": 35.4})), (EMERGENCY, ["low_temperature"]))

    def test_local_infection(self):
        # Clinical review: a red / draining umbilicus is referred (sepsis risk); pustules stay at clinic.
        self.assertEqual(triage(infant(["umbilicus_pus"])), (EMERGENCY, ["umbilicus_pus"]))
        self.assertEqual(triage(infant(["skin_pustules"])), (CLINIC, ["skin_pustules"]))
        self.assertEqual(triage(infant(["milk_spit_up", "blocked_nose"])), (HOME, []))


class TestMaternal(unittest.TestCase):
    def test_pre_eclampsia_combination(self):
        # Clinical review: one sign is already an emergency (BP is not measured in the community).
        self.assertEqual(triage(preg(["severe_headache"])), (EMERGENCY, ["severe_headache"]))
        self.assertEqual(triage(pp(["blurred_vision"])), (EMERGENCY, ["blurred_vision"]))
        self.assertEqual(triage(preg(["severe_headache", "blurred_vision"])),
                         (EMERGENCY, ["pre_eclampsia_signs"]))
        self.assertEqual(triage(pp(["face_hand_swelling", "blurred_vision"])),
                         (EMERGENCY, ["pre_eclampsia_signs"]))

    def test_puerperal_sepsis_combination(self):
        self.assertEqual(triage(pp(["fever"]))[0], CLINIC)
        self.assertEqual(triage(pp(["foul_discharge"]))[0], CLINIC)
        self.assertEqual(triage(pp(["fever", "foul_discharge"])), (EMERGENCY, ["puerperal_sepsis_signs"]))

    def test_pregnancy_emergencies(self):
        for s in ("vaginal_bleeding", "convulsions", "water_broke", "no_fetal_movement",
                  "prolonged_labour", "difficulty_breathing"):
            self.assertEqual(triage(preg([s], ga=38))[0], EMERGENCY, s)
        self.assertEqual(triage(preg(["reduced_fetal_movement"], ga=32))[0], CLINIC)

    def test_common_complaints_are_home(self):
        self.assertEqual(triage(preg(["mild_nausea", "back_pain", "heartburn"])), (HOME, []))
        self.assertEqual(triage(pp(["sore_nipples", "normal_lochia", "after_pains"])), (HOME, []))


class TestGenerator(unittest.TestCase):
    def test_generated_cases_are_consistent(self):
        rng = random.Random(7)
        for _ in range(3000):
            pop = rng.choice(POPULATIONS)
            c = sample_case(rng, pop)
            self.assertEqual((c["triage"], c["triggers"]), triage(c))
            for t in c["triggers"]:
                self.assertIn(t, TRIGGERS)
            for fid in c["findings"]:
                f = BY_ID[fid]
                self.assertIn(pop, f.populations)
                for x in f.excludes:
                    self.assertNotIn(x, c["findings"], (fid, x))
                if f.requires:
                    self.assertTrue(any(r in c["findings"] for r in f.requires), fid)
            # CHEW vitals must agree with the sampled findings.
            if c["persona"] == "chew" and pop in ("child", "infant"):
                eff = effective_findings(c)
                self.assertEqual("fast_breathing" in eff, "fast_breathing" in c["findings"])
            self.assertNotIn("{", c["text_en"])

    def test_all_triggers_reachable(self):
        rng = random.Random(11)
        seen = set()
        for _ in range(30000):
            seen.update(sample_case(rng, rng.choice(POPULATIONS))["triggers"])
        self.assertEqual(set(TRIGGERS) - seen, set())


if __name__ == "__main__":
    unittest.main()
