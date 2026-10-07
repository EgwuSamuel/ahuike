"""Assemble fine-tuning sets and the evaluation item list from cases + N-ATLAS translations.

Outputs (data/):
  sft_anchored.jsonl  - LAFIYA: anchored cases in all 4 languages + code-switched variants
  sft_ablation.jsonl  - ablation: same size, every case in ONE language (no parallel anchoring)
  eval_items.jsonl    - test items {case_id, lang, variant, text} for run_eval.py
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

from lafiya.prompts import advice_key, build_messages, load_advice, target_json  # noqa: E402


def load_jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-anchored", type=int, default=600, help="max anchored cases (x4 languages)")
    ap.add_argument("--max-test-cases", type=int, default=0, help="cap on benchmark cases (0 = all complete)")
    ap.add_argument("--allow-missing-advice", action="store_true",
                    help="fall back to English advice if advice_i18n.json is missing (debug only)")
    args = ap.parse_args()
    data = Path(args.data)

    train = load_jsonl(data / "cases_train.jsonl")
    test = load_jsonl(data / "cases_test.jsonl")
    trans = {}
    tpath = data / "translations.jsonl"
    if tpath.exists():
        for r in load_jsonl(tpath):
            if r["bt_ok"]:
                trans[(r["case_id"], r["lang"], r["variant"])] = r["text"]

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

    def complete(c: dict) -> bool:
        return all(text_for(c, lang) for lang in ("en", "ha", "yo", "ig"))

    anchored, ablation, cs = [], [], []
    missing = Counter()
    n_anchor_cases = 0
    for c in train:
        if c["anchored"]:
            # Parallel anchoring needs the SAME case in all four languages.
            if complete(c) and n_anchor_cases < args.max_anchored:
                n_anchor_cases += 1
                for lang in ("en", "ha", "yo", "ig"):
                    anchored.append(example(c, lang, text_for(c, lang)))
            elif not complete(c):
                for lang in ("ha", "yo", "ig"):
                    if not text_for(c, lang):
                        missing[f"anchored/{lang}"] += 1
        lang = c["ablation_lang"]
        t = text_for(c, lang)
        if t:
            ablation.append(example(c, lang, t))
        else:
            missing[f"ablation/{lang}"] += 1
        if c.get("code_switch"):
            for lang in ("ha", "yo", "ig"):
                t = text_for(c, lang, "cs")
                if t:
                    cs.append(example(c, lang, t))

    rng = random.Random(args.seed)
    # Equalise size: the ablation set is trimmed/kept to the anchored set's size.
    rng.shuffle(ablation)
    ablation = ablation[: len(anchored)]
    sets = {"sft_anchored.jsonl": anchored + cs, "sft_ablation.jsonl": ablation + cs}
    for name, rows in sets.items():
        rng.shuffle(rows)
        with open(data / name, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # The benchmark is the PARALLEL test set: cases whose translation passed in every language.
    bench = [c for c in test if complete(c)]
    if args.max_test_cases:
        bench = bench[: args.max_test_cases]
    (data / "benchmark_cases.json").write_text(json.dumps({
        "n_cases": len(bench), "n_test_pool": len(test),
        "triage": Counter(c["triage"] for c in bench),
        "population": Counter(c["population"] for c in bench),
        "case_ids": [c["id"] for c in bench]}, indent=1), encoding="utf-8")
    items = []
    for c in bench:
        for lang in ("en", "ha", "yo", "ig"):
            t = text_for(c, lang)
            if t:
                items.append({"case_id": c["id"], "lang": lang, "variant": "plain", "text": t})
            if lang != "en" and c.get("code_switch"):
                t = text_for(c, lang, "cs")
                if t:
                    items.append({"case_id": c["id"], "lang": lang, "variant": "cs", "text": t})
    with open(data / "eval_items.jsonl", "w", encoding="utf-8") as fh:
        for r in items:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "anchored_cases_complete": n_anchor_cases,
        "benchmark_cases": len(bench),
        "benchmark_triage": Counter(c["triage"] for c in bench),
        "sft_anchored": len(sets["sft_anchored.jsonl"]),
        "sft_ablation": len(sets["sft_ablation.jsonl"]),
        "code_switched_train": len(cs),
        "anchored_lang_counts": Counter(r["lang"] for r in anchored),
        "ablation_lang_counts": Counter(r["lang"] for r in ablation),
        "eval_items": len(items),
        "eval_by_lang_variant": Counter(f"{r['lang']}/{r['variant']}" for r in items),
        "missing_translations": missing,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
