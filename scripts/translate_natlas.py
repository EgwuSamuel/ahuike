"""Translate NaijaTriage cases into Hausa / Yoruba / Igbo with N-ATLAS, back-translate, and filter.

Resumable: finished jobs in --out are skipped, so a killed Kaggle session can just be re-run.

Jobs:
  test : every case x {ha, yo, ig} (plain) + code-switched variant for cases with code_switch
  train: anchored cases x {ha, yo, ig}; non-anchored cases only in their ablation_lang;
         code-switched variant for train cases with code_switch
  advice strings (9) x {ha, yo, ig} -> data/advice_i18n.json

Usage (Kaggle, 2xT4):
  python scripts/translate_natlas.py --backend vllm
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(ROOT))

from lafiya.backcheck import check  # noqa: E402
from lafiya.inference import ChatEngine  # noqa: E402
from lafiya.prompts import ADVICE_EN, LANGS  # noqa: E402

# Mild penalty against translation loops (the N-ATLaS card recommends 1.12 for free text).
REP_PENALTY = 1.1
TRANSLATOR_SYSTEM = "You are an expert translator for Nigerian languages. You translate faithfully and never add or remove information."


def to_lang_prompt(text: str, lang: str, variant: str) -> list[dict]:
    name = LANGS[lang]
    if variant == "cs":
        user = (f"Rewrite the following English text in {name} the way people in Nigeria speak casually, "
                f"naturally mixing in common English words (code-switching), for example words like "
                f"hospital, fever, drip, BP, convulsion. Keep every number written as digits and keep the "
                f"medical meaning exactly. Output only the rewritten text.\n\n{text}")
    else:
        user = (f"Translate the following text from English to {name}. Keep every number written as "
                f"digits and keep the medical meaning exactly. Output only the {name} translation.\n\n{text}")
    return [{"role": "system", "content": TRANSLATOR_SYSTEM}, {"role": "user", "content": user}]


def to_english_prompt(text: str, lang: str) -> list[dict]:
    name = LANGS[lang]
    user = (f"Translate the following {name} text to English. Keep every number written as digits. "
            f"Output only the English translation.\n\n{text}")
    return [{"role": "system", "content": TRANSLATOR_SYSTEM}, {"role": "user", "content": user}]


def clean(s: str) -> str:
    s = s.strip().strip('"').strip()
    for prefix in ("Translation:", "Here is the translation:", "English translation:"):
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix):].strip()
    return s


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def build_jobs(train: list[dict], test: list[dict]) -> list[tuple[str, str, str]]:
    jobs = []
    for c in test:
        for lang in ("ha", "yo", "ig"):
            jobs.append((c["id"], lang, "plain"))
            if c.get("code_switch"):
                jobs.append((c["id"], lang, "cs"))
    for c in train:
        for lang in ("ha", "yo", "ig"):
            if c["anchored"] or c["ablation_lang"] == lang:
                jobs.append((c["id"], lang, "plain"))
            if c.get("code_switch"):
                jobs.append((c["id"], lang, "cs"))
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="auto", choices=("auto", "vllm", "hf"))
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=None, help="default: <data>/translations.jsonl")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--retries", type=int, default=2, help="re-translate failed items with sampling")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N jobs")
    ap.add_argument("--test-only", action="store_true", help="translate only the test set first")
    args = ap.parse_args()

    data = Path(args.data)
    out_path = Path(args.out) if args.out else data / "translations.jsonl"
    train = load_jsonl(data / "cases_train.jsonl")
    test = load_jsonl(data / "cases_test.jsonl")
    cases = {c["id"]: c for c in train + test}

    jobs = build_jobs([] if args.test_only else train, test)
    if args.limit:
        jobs = jobs[: args.limit]

    engine = ChatEngine(backend=args.backend)

    # --- advice strings (tiny, done once)
    advice_path = data / "advice_i18n.json"
    if not advice_path.exists():
        keys = [f"{g}|{lv}" for (g, lv) in ADVICE_EN]
        convs = [to_lang_prompt(ADVICE_EN[tuple(k.split("|"))], lang, "plain")
                 for lang in ("ha", "yo", "ig") for k in keys]
        outs = engine.chat(convs, max_new_tokens=300, repetition_penalty=REP_PENALTY)
        table, i = {}, 0
        for lang in ("ha", "yo", "ig"):
            table[lang] = {}
            for k in keys:
                table[lang][k] = clean(outs[i])
                i += 1
        advice_path.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {advice_path} - send it to native speakers for review")

    # --- cases
    done = {}
    for r in load_jsonl(out_path):
        done[(r["case_id"], r["lang"], r["variant"])] = r
    print(f"{len(jobs)} jobs, {sum(j in done for j in jobs)} already done")

    for attempt_round in range(args.retries + 1):
        # Round r runs new jobs and retries failures that have not yet been tried in round r.
        pending = [j for j in jobs
                   if j not in done or (not done[j]["bt_ok"] and done[j]["attempt"] <= attempt_round)]
        if not pending:
            break
        temperature = 0.0 if attempt_round == 0 else 0.4
        for s in range(0, len(pending), args.chunk):
            chunk = pending[s:s + args.chunk]
            fwd = engine.chat([to_lang_prompt(cases[cid]["text_en"], lang, var) for cid, lang, var in chunk],
                              max_new_tokens=512, temperature=temperature,
                              repetition_penalty=REP_PENALTY)
            fwd = [clean(x) for x in fwd]
            back = engine.chat([to_english_prompt(t, lang) for t, (_, lang, _) in zip(fwd, chunk)],
                               max_new_tokens=512, repetition_penalty=REP_PENALTY)
            back = [clean(x) for x in back]
            with open(out_path, "a", encoding="utf-8") as fh:
                for (cid, lang, var), t, b in zip(chunk, fwd, back):
                    ok, problems = check(cases[cid], t, b)
                    row = {"case_id": cid, "lang": lang, "variant": var, "text": t, "back_translation": b,
                           "bt_ok": ok, "problems": problems, "attempt": attempt_round + 1}
                    prev = done.get((cid, lang, var))
                    if prev is None or not prev["bt_ok"]:
                        done[(cid, lang, var)] = row
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_ok = sum(done[j]["bt_ok"] for j in chunk)
            print(f"round {attempt_round} chunk {s // args.chunk}: {n_ok}/{len(chunk)} passed back-check")

    # Compact: keep only the best row per job.
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in done.values():
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {}
    for (cid, lang, var), r in done.items():
        k = f"{cases[cid]['split']}/{lang}/{var}"
        s = stats.setdefault(k, {"n": 0, "ok": 0})
        s["n"] += 1
        s["ok"] += r["bt_ok"]
    for s in stats.values():
        s["retention"] = round(s["ok"] / s["n"], 3)
    (data / "translation_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
