"""Assemble fine-tuning sets and the evaluation item list from cases + N-ATLAS translations.

Outputs (data/):
  sft_anchored.jsonl  - AHỤIKE: anchored cases in every language that passed + code-switched variants
  sft_ablation.jsonl  - ablation: same size and language mix, each case in ONE language only
  eval_items.jsonl    - every faithful test item {case_id, lang, variant, text} for run_eval.py
  benchmark_cases.json - the parallel core (test cases faithful in all four languages)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(ROOT))

from ahuike.backcheck import check  # noqa: E402
from ahuike.prompts import advice_key, build_messages, load_advice, target_json  # noqa: E402


def load_jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-anchored-examples", type=int, default=2000,
                    help="cap on AHỤIKE examples; the ablation needs as many UNIQUE cases")
    ap.add_argument("--allow-missing-advice", action="store_true",
                    help="fall back to English advice if advice_i18n.json is missing (debug only)")
    args = ap.parse_args()
    data = Path(args.data)

    train = load_jsonl(data / "cases_train.jsonl")
    test = load_jsonl(data / "cases_test.jsonl")
    all_cases = {c["id"]: c for c in train + test}
    trans = {}
    rejected = Counter()
    tpath = data / "translations.jsonl"
    if tpath.exists():
        for r in load_jsonl(tpath):
            c = all_cases.get(r["case_id"])
            if c is None:
                continue
            # Re-check with the CURRENT back-check so improved filters apply to existing rows.
            ok, problems = check(c, r["text"], r.get("back_translation", ""))
            if ok:
                trans[(r["case_id"], r["lang"], r["variant"])] = r["text"]
            elif r["bt_ok"]:
                rejected[next((p for p in problems if p in ("meta_text", "label_changed")), "other")] += 1

    apath = data / "advice_i18n.json"
    if not apath.exists() and not args.allow_missing_advice:
        sys.exit("data/advice_i18n.json missing: run translate_natlas.py first")
    advice = load_advice(str(apath) if apath.exists() else None)

    def text_for(case: dict, lang: str, variant: str = "plain") -> str | None:
        if lang == "en" and variant == "plain":
            return case["text_en"]
        return trans.get((case["id"], lang, variant))

    def example(case: dict, lang: str, text: str) -> dict:
        adv = advice.get(lang, advice["en"]).get(advice_key(case["population"], case["triage"]))
        msgs = build_messages(text)
        msgs.append({"role": "assistant", "content": target_json(case["triage"], case["triggers"], adv)})
        return {"case_id": case["id"], "lang": lang, "messages": msgs}

    LANGS4 = ("en", "ha", "yo", "ig")

    def available(c: dict) -> list[str]:
        return [lang for lang in LANGS4 if text_for(c, lang)]

    rng = random.Random(args.seed)

    # AHỤIKE set: anchored cases in EVERY language whose translation passed (en + at least one
    # Nigerian language), so the same clinical picture appears in parallel with one target.
    anchored, anchor_ids = [], []
    for c in train:
        if not c["anchored"] or len(available(c)) < 2:
            continue
        if len(anchored) >= args.max_anchored_examples:
            break
        anchor_ids.append(c["id"])
        anchored += [example(c, lang, text_for(c, lang)) for lang in available(c)]
    target = Counter(r["lang"] for r in anchored)

    # Ablation set: SAME size and SAME language mix, but every case appears in ONE language only
    # (no parallel anchoring). Scarcest language first so it gets the cases that have it.
    ablation, used = [], set()
    for lang in sorted(target, key=lambda l: sum(1 for c in train if text_for(c, l))):
        pool = [c for c in train if c["id"] not in used and text_for(c, lang)]
        rng.shuffle(pool)
        for c in pool[: target[lang]]:
            ablation.append(example(c, lang, text_for(c, lang)))
            used.add(c["id"])
    shortfall = {l: target[l] - sum(r["lang"] == l for r in ablation) for l in target}

    cs = []
    for c in train:
        if c.get("code_switch"):
            for lang in ("ha", "yo", "ig"):
                t = text_for(c, lang, "cs")
                if t:
                    cs.append(example(c, lang, t))

    sets = {"sft_anchored.jsonl": anchored + cs, "sft_ablation.jsonl": ablation + cs}
    for name, rows in sets.items():
        rng.shuffle(rows)
        with open(data / name, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Benchmark = every faithful test item. The PARALLEL CORE (cases faithful in all four
    # languages) is used for CLCC and for fair per-language comparison.
    core = [c for c in test if len(available(c)) == 4]
    (data / "benchmark_cases.json").write_text(json.dumps({
        "n_test_pool": len(test),
        "parallel_core_cases": len(core),
        "parallel_core_triage": Counter(c["triage"] for c in core),
        "parallel_core_case_ids": [c["id"] for c in core]}, indent=1), encoding="utf-8")
    items = []
    for c in test:
        for lang in LANGS4:
            t = text_for(c, lang)
            if t:
                items.append({"case_id": c["id"], "lang": lang, "variant": "plain", "text": t})
            if lang != "en" and c.get("code_switch"):
                t = text_for(c, lang, "cs")
                if t:
                    items.append({"case_id": c["id"], "lang": lang, "variant": "cs", "text": t})
    # Cases written directly by native speakers (scripts/ingest_review.py), benchmarked as variant "native".
    native_path = data / "native_items.jsonl"
    test_ids = {c["id"] for c in test}
    if native_path.exists():
        items += [r for r in load_jsonl(native_path) if r["case_id"] in test_ids and r["text"].strip()]
    with open(data / "eval_items.jsonl", "w", encoding="utf-8") as fh:
        for r in items:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "anchored_cases": len(anchor_ids),
        "sft_anchored": len(sets["sft_anchored.jsonl"]),
        "sft_ablation": len(sets["sft_ablation.jsonl"]),
        "code_switched_train": len(cs),
        "anchored_lang_counts": target,
        "ablation_lang_counts": Counter(r["lang"] for r in ablation),
        "ablation_shortfall": shortfall,
        "eval_items": len(items),
        "eval_by_lang_variant": Counter(f"{r['lang']}/{r['variant']}" for r in items),
        "parallel_core_cases": len(core),
        "previously_passed_now_rejected": rejected,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
