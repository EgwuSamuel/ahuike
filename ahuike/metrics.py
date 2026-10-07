"""Benchmark metrics for NaijaTriage-Bench. Pure standard library.

Prediction record (one per model reply):
  {"case_id", "lang", "variant": "plain"|"cs", "system", "output": raw text}
Gold comes from data/cases_test.jsonl keyed by case id.

Key metrics:
  accuracy, macro_f1               - 3-way triage
  under_triage_rate                - gold EMERGENCY predicted as anything else (incl. unparseable)
  over_triage_rate                 - prediction more urgent than gold, among non-emergency gold
  danger_sign_f1                   - micro F1 of predicted vs protocol trigger ids
  CLCC                             - Cross-Lingual Clinical Consistency: share of cases whose
                                     triage is identical in en/ha/yo/ig
  worst_language_gap               - max - min per-language accuracy
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict

from .prompts import parse_output
from .protocol.rules import EMERGENCY, LEVELS, SEVERITY

CORE_LANGS = ("en", "ha", "yo", "ig")


# ----------------------------------------------------------------------------- language id

_STOP = {
    "en": {"the", "and", "to", "of", "is", "if", "or", "your", "you", "go", "health", "child",
           "now", "with", "this", "do", "not", "be", "at", "has"},
    "ha": {"da", "ba", "kuma", "shi", "ita", "yana", "tana", "zuwa", "domin", "ko", "amma",
           "wannan", "yaro", "asibiti", "nan", "sai", "za", "idan", "gaggawa", "yanzu", "cikin",
           "kada", "ki", "ku", "su", "mai", "ahuike"},
    "yo": {"ni", "ti", "si", "awọn", "ọmọ", "lati", "fun", "pẹlu", "tabi", "ṣe", "lọ", "yii",
           "kan", "nitori", "bi", "ko", "ile", "iwosan", "rẹ", "wa", "ma", "maa", "jẹ", "ọ̀rọ̀"},
    "ig": {"nke", "ka", "ga", "dị", "nwa", "ọ", "ma", "ebe", "ụlọ", "ọgwụ", "gị", "anyị", "mee",
           "bụ", "ya", "na", "ihe", "ozugbo", "dọkịta", "ahụ", "mgbe", "ndị"},
}
_CHARS = {"ha": "ƙɗɓƘƊƁ", "yo": "ẹṣẸṢ", "ig": "ịụṅỊỤṄ"}
_WORD = re.compile(r"[^\W\d_]+(?:[’'][^\W\d_]+)?", re.UNICODE)


def detect_lang(text: str) -> str:
    """Lightweight heuristic language id for en/ha/yo/ig. Returns 'unknown' if no signal."""
    if not text:
        return "unknown"
    words = [w.lower() for w in _WORD.findall(text)]
    scores = {lang: sum(w in stop for w in words) for lang, stop in _STOP.items()}
    for lang, chars in _CHARS.items():
        scores[lang] += 3 * sum(text.count(ch) for ch in chars)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"


# ----------------------------------------------------------------------------- scoring

def score_item(pred: dict, gold: dict) -> dict:
    p = parse_output(pred.get("output", ""))
    gt = gold["triage"]
    pt = p["triage"]
    gold_signs = set(gold["triggers"])
    pred_signs = set(p["danger_signs"]) if pt else set()
    return {
        "case_id": pred["case_id"],
        "lang": pred["lang"],
        "variant": pred.get("variant", "plain"),
        "system": pred["system"],
        "population": gold["population"],
        "tags": gold.get("tags", []),
        "gold": gt,
        "pred": pt,
        "valid_json": p["valid_json"],
        "correct": pt == gt,
        "under": gt == EMERGENCY and pt != EMERGENCY,
        "over": gt != EMERGENCY and pt is not None and SEVERITY[pt] > SEVERITY[gt],
        "tp": len(gold_signs & pred_signs),
        "fp": len(pred_signs - gold_signs),
        "fn": len(gold_signs - pred_signs),
        "advice_lang": detect_lang(p["advice"]) if p["advice"] else "unknown",
    }


def _safe(a: float, b: float) -> float:
    return a / b if b else float("nan")


def summarize(items: list[dict]) -> dict:
    n = len(items)
    if not n:
        return {"n": 0}
    f1s = []
    for lv in LEVELS:
        tp = sum(i["pred"] == lv and i["gold"] == lv for i in items)
        fp = sum(i["pred"] == lv and i["gold"] != lv for i in items)
        fn = sum(i["pred"] != lv and i["gold"] == lv for i in items)
        p, r = _safe(tp, tp + fp), _safe(tp, tp + fn)
        f1s.append(0.0 if (math.isnan(p) or math.isnan(r) or p + r == 0) else 2 * p * r / (p + r))
    n_em = sum(i["gold"] == EMERGENCY for i in items)
    tp = sum(i["tp"] for i in items)
    fp = sum(i["fp"] for i in items)
    fn = sum(i["fn"] for i in items)
    sp, sr = _safe(tp, tp + fp), _safe(tp, tp + fn)
    with_advice = [i for i in items if i["advice_lang"] != "unknown"]
    return {
        "n": n,
        "accuracy": sum(i["correct"] for i in items) / n,
        "macro_f1": sum(f1s) / len(f1s),
        "under_triage_rate": _safe(sum(i["under"] for i in items), n_em),
        "over_triage_rate": _safe(sum(i["over"] for i in items), n - n_em),
        "json_validity": sum(i["valid_json"] for i in items) / n,
        "danger_sign_precision": sp,
        "danger_sign_recall": sr,
        "danger_sign_f1": 0.0 if (math.isnan(sp) or math.isnan(sr) or sp + sr == 0) else 2 * sp * sr / (sp + sr),
        "advice_language_match": _safe(sum(i["advice_lang"] == i["lang"] for i in with_advice), len(with_advice)),
    }


def confusion(items: list[dict]) -> dict:
    m = {g: Counter() for g in LEVELS}
    for i in items:
        m[i["gold"]][i["pred"] or "INVALID"] += 1
    return {g: dict(c) for g, c in m.items()}


def clcc(items: list[dict], langs: tuple[str, ...] = CORE_LANGS) -> dict:
    """Cross-Lingual Clinical Consistency over cases present in every language."""
    by_case: dict[str, dict[str, dict]] = defaultdict(dict)
    for i in items:
        if i["variant"] == "plain" and i["lang"] in langs:
            by_case[i["case_id"]][i["lang"]] = i
    full = {cid: d for cid, d in by_case.items() if len(d) == len(langs)}
    if not full:
        return {"n_cases": 0}
    consistent = sum(len({d[l]["pred"] for l in langs}) == 1 and d[langs[0]]["pred"] is not None
                     for d in full.values())
    cons_correct = sum(all(d[l]["correct"] for l in langs) for d in full.values())
    em = [d for d in full.values() if d[langs[0]]["gold"] == EMERGENCY]
    any_under = sum(any(d[l]["under"] for l in langs) for d in em)
    per_lang_acc = {l: sum(d[l]["correct"] for d in full.values()) / len(full) for l in langs}
    per_lang_under = {l: _safe(sum(d[l]["under"] for d in em), len(em)) for l in langs}
    return {
        "n_cases": len(full),
        "clcc": consistent / len(full),
        "consistent_and_correct": cons_correct / len(full),
        "emergency_missed_in_any_language": _safe(any_under, len(em)),
        "per_language_accuracy": per_lang_acc,
        "per_language_under_triage": per_lang_under,
        "worst_language_gap": max(per_lang_acc.values()) - min(per_lang_acc.values()),
        "worst_language": min(per_lang_acc, key=per_lang_acc.get),
    }


def bootstrap_ci(items: list[dict], fn, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Case-clustered bootstrap CI for any metric fn(items) -> float."""
    by_case: dict[str, list[dict]] = defaultdict(list)
    for i in items:
        by_case[i["case_id"]].append(i)
    ids = list(by_case)
    if not ids:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample = [x for cid in rng.choices(ids, k=len(ids)) for x in by_case[cid]]
        v = fn(sample)
        if v is not None and not math.isnan(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    vals.sort()
    lo = vals[int((alpha / 2) * len(vals))]
    hi = vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))]
    return (lo, hi)


def mcnemar_exact(a_correct: list[bool], b_correct: list[bool]) -> dict:
    """Exact two-sided McNemar test on paired binary outcomes."""
    b = sum(x and not y for x, y in zip(a_correct, b_correct))  # A right, B wrong
    c = sum(y and not x for x, y in zip(a_correct, b_correct))  # B right, A wrong
    n = b + c
    if n == 0:
        return {"a_only": b, "b_only": c, "p_value": 1.0}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"a_only": b, "b_only": c, "p_value": min(1.0, 2 * tail)}


def paired(items_a: list[dict], items_b: list[dict], key: str = "correct",
           only_gold: str | None = None) -> dict:
    ka = {(i["case_id"], i["lang"], i["variant"]): i for i in items_a}
    kb = {(i["case_id"], i["lang"], i["variant"]): i for i in items_b}
    common = sorted(k for k in ka.keys() & kb.keys() if only_gold is None or ka[k]["gold"] == only_gold)
    return mcnemar_exact([bool(ka[k][key]) for k in common], [bool(kb[k][key]) for k in common])
