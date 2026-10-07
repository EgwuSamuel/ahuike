"""LAFIYA protocol engine: the single source of truth for gold labels.

`triage(case)` maps a structured case (never its text) to a triage level and
the list of protocol triggers that justify it. Every gold label in
NaijaTriage-Bench is produced here, so labels are correct by construction
with respect to this (simplified) encoding of WHO IMCI and the Nigerian
CHEW Standing Orders maternal danger signs.

Case schema (dict):
  population: child | infant | pregnant | postpartum
  age_months (child), age_days (infant), ga_weeks (pregnant), pp_days (postpartum)
  findings:   list[str]  finding ids (see findings.py)
  durations:  dict[str, int]  days, for cough / diarrhoea / fever
  vitals:     dict  optional {"rr": breaths/min, "temp": deg C}
"""
from __future__ import annotations

from .findings import CHILD, INFANT, POSTPARTUM, PREGNANT

EMERGENCY = "EMERGENCY_REFER_NOW"
CLINIC = "CLINIC_WITHIN_24H"
HOME = "HOME_CARE"
LEVELS = (HOME, CLINIC, EMERGENCY)
SEVERITY = {HOME: 0, CLINIC: 1, EMERGENCY: 2}

INFANT_FEVER_C = 37.5
INFANT_HYPOTHERMIA_C = 35.5
CHRONIC_COUGH_DAYS = 14
PERSISTENT_DIARRHOEA_DAYS = 14

# Trigger vocabulary shown to the model (closed set, language independent).
TRIGGERS: dict[str, str] = {
    # child / infant
    "unable_to_drink": "unable to drink or breastfeed",
    "vomits_everything": "vomits everything",
    "convulsions": "convulsions / fits",
    "lethargic": "lethargic or unconscious",
    "chest_indrawing": "lower chest wall indrawing",
    "stridor": "stridor in a calm child",
    "stiff_neck": "stiff neck with fever",
    "severe_dehydration": "two or more signs of severe dehydration",
    "severe_persistent_diarrhoea": "diarrhoea >= 14 days with dehydration",
    "some_dehydration": "two or more signs of some dehydration",
    "persistent_diarrhoea": "diarrhoea for 14 days or more",
    "blood_in_stool": "blood in stool",
    "fast_breathing": "fast breathing for age",
    "chronic_cough": "cough for 14 days or more",
    "fever": "fever",
    "oedema_feet": "swelling of both feet",
    "not_feeding_well": "young infant not feeding well",
    "movement_only_when_stimulated": "young infant moves only when stimulated",
    "jaundice_palms_soles": "jaundice reaching palms and soles",
    "low_temperature": "young infant low body temperature",
    "umbilicus_pus": "red umbilicus or pus",
    "skin_pustules": "skin pustules",
    # maternal
    "vaginal_bleeding": "vaginal bleeding in pregnancy",
    "heavy_bleeding": "heavy bleeding after delivery",
    "pre_eclampsia_signs": "two or more of severe headache, blurred vision, face/hand swelling",
    "severe_abdominal_pain": "severe abdominal pain",
    "difficulty_breathing": "difficulty breathing",
    "water_broke": "waters broke before labour",
    "no_fetal_movement": "no fetal movement",
    "high_fever_very_weak": "high fever, too weak to get out of bed",
    "prolonged_labour": "labour longer than 12 hours",
    "puerperal_sepsis_signs": "fever with foul-smelling discharge after delivery",
    "reduced_fetal_movement": "reduced fetal movement",
    "vomiting_cannot_keep": "vomiting, cannot keep fluids down",
    "burning_urination": "burning urination",
    "severe_headache": "severe headache",
    "blurred_vision": "blurred vision",
    "face_hand_swelling": "swelling of face and hands",
    "foul_discharge": "foul-smelling discharge",
    "painful_red_breast": "painful red breast",
    "low_mood": "low mood, not coping",
}


def rr_threshold(population: str, age_months: int | None = None) -> int:
    """WHO IMCI fast-breathing cut-offs."""
    if population == INFANT:
        return 60
    if population == CHILD:
        return 50 if (age_months or 0) < 12 else 40
    raise ValueError(f"no RR threshold for {population}")


def effective_findings(case: dict) -> set[str]:
    """Findings stated in text plus findings implied by vitals and durations."""
    pop = case["population"]
    found = set(case.get("findings", ()))
    vitals = case.get("vitals") or {}
    rr = vitals.get("rr")
    if rr is not None and pop in (CHILD, INFANT):
        if rr >= rr_threshold(pop, case.get("age_months")):
            found.add("fast_breathing")
    temp = vitals.get("temp")
    if temp is not None and pop == INFANT:
        if temp >= INFANT_FEVER_C:
            found.add("fever")
        if temp < INFANT_HYPOTHERMIA_C:
            found.add("low_temperature")
    return found


def _child(case: dict, f: set[str]) -> list[tuple[str, str]]:
    out = []
    for sign in ("unable_to_drink", "vomits_everything", "convulsions", "lethargic",
                 "chest_indrawing", "stridor", "stiff_neck"):
        if sign in f:
            out.append((EMERGENCY, sign))

    dur = case.get("durations", {})
    if "diarrhoea" in f:
        severe = sum((
            "lethargic" in f,
            "sunken_eyes" in f,
            bool(f & {"unable_to_drink", "drinking_poorly"}),
            "skin_pinch_very_slow" in f,
        )) >= 2
        some = not severe and sum((
            "restless_irritable" in f,
            "sunken_eyes" in f,
            "drinks_eagerly" in f,
            bool(f & {"skin_pinch_slow", "skin_pinch_very_slow"}),
        )) >= 2
        if severe:
            out.append((EMERGENCY, "severe_dehydration"))
        elif some:
            out.append((CLINIC, "some_dehydration"))
        if dur.get("diarrhoea", 0) >= PERSISTENT_DIARRHOEA_DAYS:
            if severe or some:
                out.append((EMERGENCY, "severe_persistent_diarrhoea"))
            else:
                out.append((CLINIC, "persistent_diarrhoea"))
        if "blood_in_stool" in f:
            out.append((CLINIC, "blood_in_stool"))

    if "fast_breathing" in f:
        out.append((CLINIC, "fast_breathing"))
    if "cough" in f and dur.get("cough", 0) >= CHRONIC_COUGH_DAYS:
        out.append((CLINIC, "chronic_cough"))
    if "fever" in f:
        out.append((CLINIC, "fever"))  # all of Nigeria is malaria-endemic: test at PHC
    if "oedema_feet" in f:
        out.append((CLINIC, "oedema_feet"))
    return out


def _infant(case: dict, f: set[str]) -> list[tuple[str, str]]:
    out = []
    # Any sign of possible serious bacterial infection -> urgent referral.
    for sign in ("not_feeding_well", "convulsions", "chest_indrawing", "fast_breathing",
                 "fever", "low_temperature", "movement_only_when_stimulated",
                 "jaundice_palms_soles"):
        if sign in f:
            out.append((EMERGENCY, sign))
    for sign in ("umbilicus_pus", "skin_pustules"):
        if sign in f:
            out.append((CLINIC, sign))
    return out


def _pre_eclampsia(f: set[str]) -> bool:
    return len(f & {"severe_headache", "blurred_vision", "face_hand_swelling"}) >= 2


def _maternal(case: dict, f: set[str], postpartum: bool) -> list[tuple[str, str]]:
    out = []
    if postpartum:
        e_singles = ("heavy_bleeding", "convulsions", "severe_abdominal_pain",
                     "difficulty_breathing", "high_fever_very_weak")
        c_singles = ("fever", "foul_discharge", "painful_red_breast", "low_mood",
                     "burning_urination", "severe_headache", "blurred_vision",
                     "face_hand_swelling")
    else:
        e_singles = ("vaginal_bleeding", "convulsions", "severe_abdominal_pain",
                     "difficulty_breathing", "water_broke", "no_fetal_movement",
                     "high_fever_very_weak", "prolonged_labour")
        c_singles = ("fever", "reduced_fetal_movement", "vomiting_cannot_keep",
                     "burning_urination", "severe_headache", "blurred_vision",
                     "face_hand_swelling")
    for sign in e_singles:
        if sign in f:
            out.append((EMERGENCY, sign))
    if _pre_eclampsia(f):
        out.append((EMERGENCY, "pre_eclampsia_signs"))
    if postpartum and {"fever", "foul_discharge"} <= f:
        out.append((EMERGENCY, "puerperal_sepsis_signs"))
    for sign in c_singles:
        if sign in f:
            out.append((CLINIC, sign))
    return out


def fired_rules(case: dict) -> list[tuple[str, str]]:
    pop = case["population"]
    f = effective_findings(case)
    if pop == CHILD:
        return _child(case, f)
    if pop == INFANT:
        return _infant(case, f)
    if pop == PREGNANT:
        return _maternal(case, f, postpartum=False)
    if pop == POSTPARTUM:
        return _maternal(case, f, postpartum=True)
    raise ValueError(f"unknown population {pop!r}")


def triage(case: dict) -> tuple[str, list[str]]:
    """Return (triage level, sorted triggers at that level). HOME_CARE has no triggers."""
    fired = fired_rules(case)
    if not fired:
        return HOME, []
    top = max(SEVERITY[level] for level, _ in fired)
    level = LEVELS[top]
    return level, sorted({t for lv, t in fired if lv == level})


def reasoning_tags(case: dict) -> list[str]:
    """Tags marking cases whose label depends on a threshold or a combination rule."""
    tags = []
    pop = case["population"]
    vitals = case.get("vitals") or {}
    f = set(case.get("findings", ()))
    if vitals.get("rr") is not None and pop in (CHILD, INFANT):
        if abs(vitals["rr"] - rr_threshold(pop, case.get("age_months"))) <= 5:
            tags.append("rr_threshold")
    if vitals.get("temp") is not None and pop == INFANT:
        t = vitals["temp"]
        if abs(t - INFANT_FEVER_C) <= 0.4 or abs(t - INFANT_HYPOTHERMIA_C) <= 0.4:
            tags.append("temp_threshold")
    dur = case.get("durations", {})
    if pop == CHILD and any(7 <= dur.get(k, 0) <= 21 for k in ("cough", "diarrhoea")):
        tags.append("duration_threshold")
    pe = f & {"severe_headache", "blurred_vision", "face_hand_swelling"}
    dehyd = f & {"lethargic", "sunken_eyes", "unable_to_drink", "drinking_poorly",
                 "skin_pinch_very_slow", "restless_irritable", "drinks_eagerly",
                 "skin_pinch_slow"}
    if pe or (pop == POSTPARTUM and f & {"fever", "foul_discharge"}) or ("diarrhoea" in f and dehyd):
        tags.append("combination")
    return tags
