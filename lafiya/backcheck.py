"""Back-translation fidelity check (label-preserving).

After N-ATLAS translates an English vignette into ha/yo/ig, it translates it back to
English. A translation is kept when the back-translation still supports the SAME gold
label:
  * every finding that the label depends on is still recognisable,
  * at most one non-critical finding is lost (and only in cases with 3+ findings),
  * removing the lost findings does not change the protocol label or triggers,
  * for child/infant cases every number (age, duration, RR, temperature) survives,
    with English number words normalised ("three" -> 3),
  * no new red-flag symptom (convulsions, bleeding, unconsciousness) appears.
Coarse by design: it cannot tell "slow" from "very slow". See docs/DATASET_CARD.md.
"""
from __future__ import annotations

import re

from .protocol.findings import CHILD, INFANT
from .protocol.rules import triage

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
    "diarrhoea": ("stool", "diarr", "stomach", "purg", "toilet", "watery", "loose", "poo"),
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

# Red flags that must not appear unless the case really contains them.
RED_FLAGS: dict[str, tuple[str, ...]] = {
    "convuls": ("convulsions",),
    "seizure": ("convulsions",),
    "unconscious": ("lethargic",),
    "bleeding": ("vaginal_bleeding", "heavy_bleeding", "normal_lochia", "blood_in_stool"),
}

# ----------------------------------------------------------------------------- numbers

_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
_NUMWORD = re.compile(
    r"\b(?:(" + "|".join(_TENS) + r")(?:[\s-](" + "|".join(k for k in _UNITS if _UNITS[k] < 10) + r"))?"
    r"|(" + "|".join(_UNITS) + r"))\b", re.IGNORECASE)
_NUM = re.compile(r"\d+(?:\.\d+)?")


def normalise_numbers(text: str) -> str:
    """'twenty-four hours' -> '24 hours'; 'three days' -> '3 days'."""
    def sub(m: re.Match) -> str:
        if m.group(3):
            return str(_UNITS[m.group(3).lower()])
        return str(_TENS[m.group(1).lower()] + (_UNITS[m.group(2).lower()] if m.group(2) else 0))
    return _NUMWORD.sub(sub, text or "")


def numbers(text: str) -> set[str]:
    return set(_NUM.findall(normalise_numbers(text)))


# ----------------------------------------------------------------------------- check

def _present(fid: str, bt: str) -> bool:
    return any(s in bt for s in STEMS[fid])


def check(case: dict, translation: str, back_translation: str) -> tuple[bool, list[str]]:
    """Return (ok, problems). Problems are reported even when ok (for analysis)."""
    problems: list[str] = []
    bt = (back_translation or "").lower()
    vitals = case.get("vitals") or {}
    checkable = []
    for fid in case["findings"]:
        if case.get("persona") == "chew":
            if fid == "fast_breathing" and "rr" in vitals:
                continue
            if case["population"] == INFANT and fid in ("fever", "low_temperature") and "temp" in vitals:
                continue
        checkable.append(fid)
    missing = [f for f in checkable if not _present(f, bt)]
    problems += [f"missing:{f}" for f in missing]

    fatal = False
    if missing:
        if len(missing) > 1 or len(case["findings"]) < 3:
            fatal = True
        else:
            reduced = dict(case, findings=[f for f in case["findings"] if f not in missing])
            if triage(reduced) != (case["triage"], case["triggers"]):
                fatal = True
                problems.append("label_changed")

    have = numbers(translation) | numbers(back_translation)
    lost = [n for n in numbers(case["text_en"]) if n not in have]
    problems += [f"number:{n}" for n in lost]
    if lost and case["population"] in (CHILD, INFANT):
        fatal = True

    for kw, owners in RED_FLAGS.items():
        if kw in bt and not any(o in case["findings"] for o in owners):
            problems.append(f"spurious:{kw}")
            fatal = True

    if not translation or len(translation) < 0.3 * len(case["text_en"]):
        problems.append("too_short")
        fatal = True
    return (not fatal, problems)
