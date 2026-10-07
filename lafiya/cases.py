"""Sample structured cases and render them as Nigerian-English vignettes.

The structured case is labelled by `protocol.rules.triage`; the text is only a
rendering of it. Context sentences (agbo, "I think it is malaria", distance to
the PHC) are deliberately label-neutral distractors.
"""
from __future__ import annotations

import random

from .protocol import findings as fx
from .protocol.findings import BY_ID, CHILD, INFANT, POSTPARTUM, PREGNANT
from .protocol.rules import (
    CHRONIC_COUGH_DAYS,
    PERSISTENT_DIARRHOEA_DAYS,
    effective_findings,
    reasoning_tags,
    rr_threshold,
    triage,
)

PERSONAS = {
    CHILD: ("mother", "father", "grandmother", "chew"),
    INFANT: ("mother", "father", "grandmother", "chew"),
    PREGNANT: ("self", "partner", "chew"),
    POSTPARTUM: ("self", "partner", "chew"),
}
PERSONA_WEIGHTS = {
    CHILD: (0.45, 0.15, 0.15, 0.25),
    INFANT: (0.5, 0.15, 0.15, 0.2),
    PREGNANT: (0.55, 0.2, 0.25),
    POSTPARTUM: (0.55, 0.2, 0.25),
}


# ----------------------------------------------------------------------------- sampling

def _eligible(population: str, case: dict) -> list:
    out = []
    for f in fx.for_population(population):
        if population == PREGNANT and case["ga_weeks"] < f.min_ga:
            continue
        out.append(f)
    return out


def _sample_duration(rng: random.Random, fid: str) -> int:
    if fid == "cough":
        if rng.random() < 0.35:
            return rng.randint(CHRONIC_COUGH_DAYS, 30)
        return rng.choice([rng.randint(1, 6), rng.randint(7, CHRONIC_COUGH_DAYS - 1)])
    if fid == "diarrhoea":
        if rng.random() < 0.25:
            return rng.randint(PERSISTENT_DIARRHOEA_DAYS, 21)
        return rng.choice([rng.randint(1, 4), rng.randint(5, PERSISTENT_DIARRHOEA_DAYS - 1)])
    return rng.randint(1, 6)  # fever


def sample_case(rng: random.Random, population: str) -> dict:
    case: dict = {"population": population, "findings": [], "durations": {}, "vitals": {}}
    if population == CHILD:
        case["age_months"] = rng.randint(2, 59)
        case["sex"] = rng.choice(("male", "female"))
    elif population == INFANT:
        case["age_days"] = rng.randint(1, 56)
        case["sex"] = rng.choice(("male", "female"))
    elif population == PREGNANT:
        case["ga_weeks"] = rng.randint(6, 41)
        case["woman_age"] = rng.randint(17, 42)
    else:
        case["pp_days"] = rng.randint(1, 42)
        case["woman_age"] = rng.randint(17, 42)
    case["persona"] = rng.choices(PERSONAS[population], PERSONA_WEIGHTS[population])[0]

    pool = _eligible(population, case)
    n = rng.choices((1, 2, 3, 4), (0.32, 0.36, 0.22, 0.10))[0]
    chosen: list[str] = []
    for _ in range(n * 3):  # bounded attempts
        if len(chosen) >= n:
            break
        cand = [f for f in pool if f.id not in chosen
                and not any(x in chosen for x in f.excludes)
                and not any(f.id in BY_ID[c].excludes for c in chosen)]
        if not cand:
            break
        f = rng.choices(cand, [c.weight for c in cand])[0]
        if f.requires and not any(r in chosen for r in f.requires):
            req = f.requires[0]
            if any(req in BY_ID[c].excludes for c in chosen):
                continue
            chosen.append(req)
        chosen.append(f.id)
    case["findings"] = chosen

    for fid in chosen:
        if BY_ID[fid].has_duration:
            case["durations"][fid] = _sample_duration(rng, fid)

    if case["persona"] == "chew" and population in (CHILD, INFANT):
        thr = rr_threshold(population, case.get("age_months"))
        near = rng.random() < 0.5
        if "fast_breathing" in chosen:
            case["vitals"]["rr"] = thr + (rng.randint(0, 4) if near else rng.randint(5, 18))
        else:
            case["vitals"]["rr"] = thr - (rng.randint(1, 5) if near else rng.randint(6, 18))
        if population == INFANT:
            if "fever" in chosen:
                t = rng.uniform(37.5, 37.9) if near else rng.uniform(38.0, 39.2)
            elif "low_temperature" in chosen:
                t = rng.uniform(35.1, 35.4) if near else rng.uniform(34.2, 35.0)
            else:
                t = rng.uniform(36.5, 37.4)
            case["vitals"]["temp"] = round(t, 1)
        elif "fever" in chosen:
            case["vitals"]["temp"] = round(rng.uniform(37.6, 39.6), 1)
        else:
            case["vitals"]["temp"] = round(rng.uniform(36.5, 37.3), 1)

    label, triggers = triage(case)
    case["triage"] = label
    case["triggers"] = triggers
    case["tags"] = reasoning_tags(case)
    case["combo_key"] = combo_key(case)
    case["context"] = _pick_context(rng, case)
    case["text_en"] = render(case, rng)
    return case


def combo_key(case: dict) -> str:
    """Clinical identity of a case: population + effective findings + duration flags.

    Train/test are split on this key so that no test clinical picture is seen in training.
    """
    eff = sorted(effective_findings(case))
    dur = case.get("durations", {})
    flags = []
    if dur.get("cough", 0) >= CHRONIC_COUGH_DAYS:
        flags.append("cough14")
    if dur.get("diarrhoea", 0) >= PERSISTENT_DIARRHOEA_DAYS:
        flags.append("diarrhoea14")
    return f"{case['population']}|{','.join(eff)}|{','.join(flags)}"


# ----------------------------------------------------------------------------- rendering

CONTEXT = {
    "child": (
        "We gave {him} agbo but nothing changed.",
        "I think it is malaria.",
        "We live far from the health centre.",
        "{His} elder brother had something like this last month.",
        "We bought paracetamol from the chemist.",
        "My mother-in-law says it will pass.",
        "{He} has taken all {his} immunisations.",
        "We are in the village now.",
    ),
    PREGNANT: (
        "This is my first pregnancy.",
        "I have not started antenatal clinic.",
        "My mother-in-law says it is normal.",
        "I took some herbs from the market.",
        "I live far from the hospital.",
        "I have three other children.",
    ),
    POSTPARTUM: (
        "This is my first baby.",
        "I delivered at home with a traditional birth attendant.",
        "My mother-in-law says it is normal.",
        "I took some herbs from the market.",
        "I live far from the hospital.",
        "I am breastfeeding the baby.",
    ),
}

CLOSING = {
    "child": ("What should I do?", "Should I take {him} to the hospital?", "Is this serious?",
              "Please, what should I do?", "Should I be worried?"),
    "self": ("What should I do?", "Should I go to the hospital?", "Is this normal?",
             "Please advise me.", "Should I be worried?"),
    "partner": ("What should we do?", "Should I take her to the hospital?", "Is this serious?"),
}
GREETING = ("", "", "Good morning. ", "Good evening. ", "Please help me. ", "Hello. ")

PRONOUNS = {
    "male": {"He": "He", "he": "he", "His": "His", "his": "his", "him": "him"},
    "female": {"He": "She", "he": "she", "His": "Her", "his": "her", "him": "her"},
}


def _pick_context(rng: random.Random, case: dict) -> list[int]:
    if case["persona"] == "chew":
        return []
    key = "child" if case["population"] in (CHILD, INFANT) else case["population"]
    k = rng.choices((0, 1, 2), (0.45, 0.4, 0.15))[0]
    return rng.sample(range(len(CONTEXT[key])), k)


def _dur(days: int, rng: random.Random) -> str:
    if days == 1:
        return rng.choice(("1 day", "one day", "since yesterday"))
    if days >= 14 and days % 7 == 0:
        w = days // 7
        return rng.choice((f"{days} days", f"{w} weeks"))
    if days > 14 and rng.random() < 0.5:
        return f"more than {days // 7} weeks" if days % 7 else f"{days // 7} weeks"
    return f"{days} days"


def _child_age(case: dict, rng: random.Random) -> str:
    if case["population"] == INFANT:
        d = case["age_days"]
        if d < 14 or d % 7:
            return f"{d} days old"
        return f"{d // 7} weeks old"
    m = case["age_months"]
    if m < 12 or (m < 24 and rng.random() < 0.5):
        return f"{m} months old"
    y, r = divmod(m, 12)
    ys = "1 year" if y == 1 else f"{y} years"
    return f"{ys} old" if r == 0 else f"{ys} {r} months old"


def _fill(s: str, pron: dict, case: dict, fid: str | None, rng: random.Random) -> str:
    vals = dict(pron)
    if fid and fid in case.get("durations", {}):
        vals["dur"] = _dur(case["durations"][fid], rng)
    return s.format(**vals)


def render(case: dict, rng: random.Random) -> str:
    pop = case["population"]
    persona = case["persona"]
    if persona == "chew":
        return _render_chew(case, rng)
    findings = list(case["findings"])
    rng.shuffle(findings)
    if pop in (CHILD, INFANT):
        pron = PRONOUNS[case["sex"]]
        son = "son" if case["sex"] == "male" else "daughter"
        rel = {"grandmother": "grand" + son}.get(persona, son)
        if pop == INFANT and rng.random() < 0.5 and persona != "grandmother":
            rel = "baby boy" if case["sex"] == "male" else "baby girl"
        parts = [f"{rng.choice(GREETING)}My {rel} is {_child_age(case, rng)}."]
        for fid in findings:
            parts.append(_fill(rng.choice(BY_ID[fid].child), pron, case, fid, rng))
        parts += [CONTEXT["child"][i].format(**pron) for i in case["context"]]
        parts.append(rng.choice(CLOSING["child"]).format(**pron))
        return " ".join(parts)

    voice = "self" if persona == "self" else "partner"
    if pop == PREGNANT:
        w = case["ga_weeks"]
        ga = f"{w} weeks" if rng.random() < 0.5 else f"{max(1, round(w / 4.3))} months"
        opener = f"I am {ga} pregnant." if voice == "self" else f"My wife is {ga} pregnant."
    else:
        d = case["pp_days"]
        pp = f"{d} days" if d < 14 or rng.random() < 0.5 else f"{d // 7} weeks"
        if d == 1:
            pp = "1 day"
        opener = (f"I delivered my baby {pp} ago." if voice == "self"
                  else f"My wife delivered {pp} ago.")
    parts = [f"{rng.choice(GREETING)}{opener}"]
    for fid in findings:
        texts = BY_ID[fid].self_ if voice == "self" else BY_ID[fid].partner
        parts.append(_fill(rng.choice(texts), {}, case, fid, rng))
    ctx = CONTEXT[pop]
    for i in case["context"]:
        s = ctx[i]
        if voice == "partner":
            s = (s.replace("I have", "She has").replace("I am", "She is").replace("I took", "She took")
                  .replace("I live", "We live").replace("I delivered", "She delivered")
                  .replace("my first", "her first").replace("My mother-in-law", "My mother"))
        parts.append(s)
    parts.append(rng.choice(CLOSING[voice]))
    return " ".join(parts)


def _render_chew(case: dict, rng: random.Random) -> str:
    pop = case["population"]
    findings = list(case["findings"])
    rng.shuffle(findings)
    vitals = case.get("vitals") or {}
    complaints = []
    for fid in findings:
        if pop in (CHILD, INFANT) and fid == "fast_breathing" and "rr" in vitals:
            continue  # conveyed only through the respiratory rate
        if pop == INFANT and fid in ("fever", "low_temperature") and "temp" in vitals:
            continue  # conveyed only through the temperature
        f = BY_ID[fid]
        txt = f.clinical
        if "{dur}" in txt:
            txt = txt.replace("{dur}", _dur(case["durations"][fid], rng))
        complaints.append(txt)
    if pop in (CHILD, INFANT):
        sex = "male" if case["sex"] == "male" else "female"
        age = _child_age(case, rng).replace(" old", "")
        head = f"CHEW note: {age} old {sex} {'infant' if pop == INFANT else 'child'}."
    elif pop == PREGNANT:
        head = f"CHEW note: {case['woman_age']}-year-old woman, {case['ga_weeks']} weeks pregnant."
    else:
        head = f"CHEW note: {case['woman_age']}-year-old woman, {case['pp_days']} days after delivery."
    parts = [head]
    if complaints:
        parts.append("Complaints: " + "; ".join(complaints) + ".")
    else:
        parts.append("Brought for review, no specific complaint.")
    if vitals:
        v = []
        if "rr" in vitals:
            v.append(f"RR {vitals['rr']}/min")
        if "temp" in vitals:
            v.append(f"temp {vitals['temp']}°C")
        parts.append("Findings: " + ", ".join(v) + ".")
    parts.append(rng.choice(("Please advise on triage.", "What is the triage decision?",
                             "Advise on next step.")))
    return " ".join(parts)
