"""Back-translation fidelity check.

After N-ATLAS translates an English vignette into ha/yo/ig, it translates it back to
English. We keep the translation only if every clinical finding of the structured
case is still recognisable in the back-translation and every number survived.
Coarse by design (it cannot tell "slow" from "very slow"); see docs/DATASET_CARD.md.
"""
from __future__ import annotations

import re

from .protocol.findings import INFANT

STEMS: dict[str, tuple[str, ...]] = {
    "unable_to_drink": ("drink", "suck", "breast", "feed"),
    "drinking_poorly": ("drink", "sip", "water"),
    "vomits_everything": ("vomit", "throw"),
    "convulsions": ("convuls", "fit", "jerk", "seiz", "shak", "stiff", "spasm"),
    "lethargic": ("weak", "respond", "conscious", "wake", "react", "faint", "lifeless", "sleep"),
    "cough": ("cough",),
    "chest_indrawing": ("chest", "rib"),
    "stridor": ("noise", "sound", "harsh", "wheez", "stridor", "noisy"),
    "fast_breathing": ("breath",),
    "diarrhoea": ("stool", "diarr", "stomach", "purg", "toilet", "watery", "loose"),
    "sunken_eyes": ("eye",),
    "skin_pinch_very_slow": ("pinch", "skin"),
    "skin_pinch_slow": ("pinch", "skin"),
    "restless_irritable": ("restless", "irritab", "cry", "cries", "settle", "fuss"),
    "drinks_eagerly": ("thirst", "drink"),
    "blood_in_stool": ("blood",),
    "fever": ("fever", "hot", "temperature", "heat", "warm"),
    "stiff_neck": ("neck",),
    "oedema_feet": ("swell", "swollen", "feet", "foot", "leg"),
    "runny_nose": ("catarrh", "nose", "sneez", "cold", "mucus", "rhinorr"),
    "teething": ("teeth", "tooth", "drool"),
    "mild_rash": ("rash", "heat", "spot"),
    "not_feeding_well": ("feed", "breast", "suck"),
    "low_temperature": ("cold",),
    "movement_only_when_stimulated": ("mov", "touch", "shake", "stimul"),
    "jaundice_palms_soles": ("yellow", "jaundice"),
    "umbilicus_pus": ("navel", "cord", "umbilic", "belly button", "pus"),
    "skin_pustules": ("pus", "bump", "boil", "pimple", "spot", "blister"),
    "milk_spit_up": ("milk", "spit", "vomit", "bring"),
    "blocked_nose": ("nose", "sneez"),
    "vaginal_bleeding": ("bleed", "blood"),
    "heavy_bleeding": ("bleed", "blood"),
    "severe_headache": ("head",),
    "blurred_vision": ("eye", "see", "vision", "sight", "blur"),
    "face_hand_swelling": ("swell", "swollen", "puff"),
    "severe_abdominal_pain": ("stomach", "abdom", "belly"),
    "difficulty_breathing": ("breath",),
    "water_broke": ("water", "fluid", "leak"),
    "no_fetal_movement": ("mov", "kick"),
    "reduced_fetal_movement": ("mov", "kick"),
    "high_fever_very_weak": ("fever", "hot", "temperature"),
    "prolonged_labour": ("labour", "labor", "deliver", "contraction"),
    "vomiting_cannot_keep": ("vomit", "throw"),
    "burning_urination": ("urin", "pee", "burn", "dysuria"),
    "foul_discharge": ("discharge", "smell", "odor", "odour"),
    "painful_red_breast": ("breast",),
    "low_mood": ("sad", "cry", "cope", "coping", "depress", "unhappy", "sorrow", "mood"),
    "mild_nausea": ("vomit", "nause", "sick"),
    "back_pain": ("back", "waist"),
    "heartburn": ("heart", "chest", "burn"),
    "leg_cramps": ("leg", "cramp"),
    "feet_swelling_evening": ("feet", "foot", "leg", "swell"),
    "tiredness": ("tired", "weak", "fatigue", "exhaust"),
    "after_pains": ("cramp", "pain"),
    "sore_nipples": ("nipple", "breast"),
    "normal_lochia": ("bleed", "blood", "period", "lochia"),
}

_NUM = re.compile(r"\d+(?:\.\d+)?")


def numbers(text: str) -> set[str]:
    return set(_NUM.findall(text or ""))


def check(case: dict, translation: str, back_translation: str) -> tuple[bool, list[str]]:
    """Return (ok, problems)."""
    problems = []
    bt = (back_translation or "").lower()
    vitals = case.get("vitals") or {}
    for fid in case["findings"]:
        if case.get("persona") == "chew":
            if fid == "fast_breathing" and "rr" in vitals:
                continue
            if case["population"] == INFANT and fid in ("fever", "low_temperature") and "temp" in vitals:
                continue
        if not any(s in bt for s in STEMS[fid]):
            problems.append(f"missing:{fid}")
    have = numbers(translation) | numbers(back_translation)
    for n in numbers(case["text_en"]):
        if n not in have:
            problems.append(f"number:{n}")
    if not translation or len(translation) < 0.3 * len(case["text_en"]):
        problems.append("too_short")
    return (not problems, problems)
