"""Prompts, training targets and output parsing shared by data, training, eval and the app.

All systems (base N-ATLAS and AHỤIKE) receive the same system prompt, so any
difference in the benchmark comes from fine-tuning, not from prompting.
"""
from __future__ import annotations

import hashlib
import json
import re

from .protocol.findings import POSTPARTUM, PREGNANT
from .protocol.rules import CLINIC, EMERGENCY, HOME, LEVELS, REVIEW_EMERGENCY_SIGNS

LANGS = {"en": "English", "ha": "Hausa", "yo": "Yoruba", "ig": "Igbo"}

SYSTEM_PROMPT = """You are AHỤIKE, a maternal and child health triage assistant for Nigerian communities, built on N-ATLAS. A caregiver, a pregnant or postpartum woman, or a community health worker (CHEW) describes a case in English, Hausa, Yoruba or Igbo. Apply the protocol and reply with ONLY one JSON object:
{"triage": "EMERGENCY_REFER_NOW" | "CLINIC_WITHIN_24H" | "HOME_CARE", "patient": "child" | "pregnant" | "postpartum", "danger_signs": [protocol sign ids at the decided level]}
Use "patient": "child" for any baby or child under five, including newborns. Do not write advice: the app shows reviewed advice for the triage level.

PROTOCOL (WHO IMCI + Nigerian CHEW Standing Orders)
EMERGENCY_REFER_NOW if any of:
- Child 2-59 months: unable_to_drink, vomits_everything, convulsions, lethargic, chest_indrawing, stridor, stiff_neck; severe_dehydration (diarrhoea with 2+ of: lethargic, sunken eyes, unable to drink or drinking poorly, skin pinch very slow); severe_persistent_diarrhoea (diarrhoea 14+ days with some or severe dehydration).
- Young infant 0-59 days: not_feeding_well, convulsions, chest_indrawing, fast_breathing (RR 60+), fever (37.5C+ or hot body), low_temperature (below 35.5C or cold to touch), movement_only_when_stimulated, jaundice_palms_soles.
- Pregnancy: vaginal_bleeding, convulsions, severe_abdominal_pain, difficulty_breathing, water_broke, no_fetal_movement, high_fever_very_weak, prolonged_labour (over 12 h), pre_eclampsia_signs (2+ of severe headache, blurred vision, face/hand swelling).
- After delivery: heavy_bleeding, convulsions, severe_abdominal_pain, difficulty_breathing, high_fever_very_weak, pre_eclampsia_signs, puerperal_sepsis_signs (fever with foul-smelling discharge).
CLINIC_WITHIN_24H if no emergency sign and any of:
- Child: fast_breathing (RR 50+ at 2-11 months, 40+ at 12-59 months, or caregiver reports fast breathing), some_dehydration (diarrhoea with 2+ of: restless/irritable, sunken eyes, drinks eagerly, skin pinch slow), persistent_diarrhoea (14+ days), blood_in_stool, chronic_cough (14+ days), fever, oedema_feet.
- Young infant: umbilicus_pus, skin_pustules.
- Pregnancy: fever, reduced_fetal_movement, vomiting_cannot_keep, burning_urination, or only one of severe_headache / blurred_vision / face_hand_swelling.
- After delivery: fever, foul_discharge, painful_red_breast, low_mood, burning_urination, or only one of severe_headache / blurred_vision / face_hand_swelling.
HOME_CARE otherwise, with "danger_signs": [].
If unsure between two levels, choose the more urgent one."""

# English source advice. Translations live in data/advice_i18n.json (N-ATLAS translated,
# native-speaker reviewed) and are loaded with load_advice().
ADVICE_EN = {
    ("child", EMERGENCY): "DANGER SIGN. Take the child to the nearest hospital or health centre NOW. Do not wait and do not give agbo. Keep the child warm and keep breastfeeding or giving fluids on the way if the child can drink.",
    ("child", CLINIC): "Take the child to the health centre today, within 24 hours, to be checked and treated. Keep breastfeeding and giving fluids. If the child gets worse or shows any danger sign, go immediately.",
    ("child", HOME): "This can be cared for at home. Keep breastfeeding and give plenty of fluids and food. Go to the health centre if the child cannot drink, vomits everything, has convulsions, becomes very weak, breathes fast or gets worse.",
    ("pregnant", EMERGENCY): "DANGER SIGN. Go to the nearest hospital NOW with someone to accompany you. Do not wait and do not take herbs. Bring your antenatal card if you have one.",
    ("pregnant", CLINIC): "Go to the health centre today, within 24 hours, to be checked. If it gets worse, or you have bleeding, convulsions, severe headache with blurred vision or severe pain, go to the hospital immediately.",
    ("pregnant", HOME): "This is common in pregnancy and can be managed at home. Rest, eat well and drink enough water. Go to the health centre if you have bleeding, severe headache, blurred vision, fever, convulsions or the baby moves less.",
    ("postpartum", EMERGENCY): "DANGER SIGN. Go to the nearest hospital NOW with someone to accompany you. Do not wait and do not take herbs. Bring the baby if you are breastfeeding.",
    ("postpartum", CLINIC): "Go to the health centre today, within 24 hours, to be checked. If it gets worse, or you have heavy bleeding, convulsions, high fever or severe pain, go to the hospital immediately.",
    ("postpartum", HOME): "This is common after delivery and can be managed at home. Rest, eat well, drink enough water and keep breastfeeding. Go to the health centre if you have heavy bleeding, fever, bad-smelling discharge, severe headache or convulsions.",
}


def advice_group(population: str) -> str:
    if population in (PREGNANT, POSTPARTUM):
        return population
    return "child"  # child and young infant


def advice_key(population: str, level: str) -> str:
    return f"{advice_group(population)}|{level}"


def load_advice(path: str | None = None) -> dict[str, dict[str, str]]:
    """Return {lang: {"group|LEVEL": text}}. English always present."""
    table = {"en": {f"{g}|{lv}": t for (g, lv), t in ADVICE_EN.items()}}
    if path:
        with open(path, encoding="utf-8") as fh:
            table.update(json.load(fh))
    return table


def input_hash(text: str) -> str:
    """Identifies a model input: the system prompt AND the case text. A prediction is reused
    only when both are unchanged, so the baseline is re-run if the prompt ever changes."""
    return hashlib.sha1((SYSTEM_PROMPT + "\x00" + text).encode("utf-8")).hexdigest()[:12]


def user_message(text: str) -> str:
    return text.strip()


PATIENT_GROUPS = ("child", "pregnant", "postpartum")


def target_json(triage: str, triggers: list[str], patient: str) -> str:
    """The model's whole answer. Advice is NOT generated: it is looked up (advice_for) from
    messages reviewed by a clinician and native speakers, so it can never be hallucinated."""
    return json.dumps({"triage": triage, "patient": patient, "danger_signs": list(triggers)},
                      ensure_ascii=False)


def advice_for(parsed: dict, lang: str, table: dict[str, dict[str, str]]) -> str:
    """Reviewed advice for a parsed answer, in the user's language (English if not available)."""
    level = parsed.get("triage")
    if level is None:
        return ""
    group = parsed.get("patient") if parsed.get("patient") in PATIENT_GROUPS else "child"
    key = f"{group}|{level}"
    return table.get(lang, {}).get(key) or table["en"][key]


def build_messages(text: str, few_shot: list[tuple[str, str]] | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, a in few_shot or []:
        msgs.append({"role": "user", "content": user_message(u)})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": user_message(text)})
    return msgs


_LEVEL_RE = re.compile(r"EMERGENCY_REFER_NOW|CLINIC_WITHIN_24H|HOME_CARE")


def parse_output(text: str) -> dict:
    """Parse a model reply. Never raises.

    Returns {"triage": level|None, "danger_signs": [...], "advice": str, "valid_json": bool}.
    """
    out = {"triage": None, "patient": None, "danger_signs": [], "advice": "", "valid_json": False}
    if not text:
        return out
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, dict):
                tri = obj.get("triage")
                if tri in LEVELS:
                    out["triage"] = tri
                ds = obj.get("danger_signs") or []
                if isinstance(ds, list):
                    out["danger_signs"] = [str(x) for x in ds]
                if obj.get("patient") in PATIENT_GROUPS:
                    out["patient"] = obj["patient"]
                adv = obj.get("advice")
                out["advice"] = adv if isinstance(adv, str) else ""
                out["valid_json"] = out["triage"] is not None
        except json.JSONDecodeError:
            pass
    if out["triage"] is None:
        m = _LEVEL_RE.search(text)
        if m:
            out["triage"] = m.group(0)
    return out


def apply_review_guard(parsed: dict) -> dict:
    """Raise to EMERGENCY when the model names a sign the clinical review made an emergency.

    The model decides which signs are present; the reviewed protocol decides how urgent they are.
    Adds "raised_by_review": [signs] when it changes the answer. Never lowers urgency.
    """
    hit = sorted(set(parsed.get("danger_signs") or []) & REVIEW_EMERGENCY_SIGNS)
    if hit and parsed.get("triage") != EMERGENCY:
        return {**parsed, "triage": EMERGENCY, "raised_by_review": hit}
    return parsed
