"""Translate NaijaTriage cases into Hausa / Yoruba / Igbo with N-ATLAS, back-translate, and filter.

Translation is done SENTENCE BY SENTENCE (long multi-symptom passages made N-ATLAS drop or
invent content), with a small symptom glossary hint for the terms a sentence contains.
Identical sentences are translated once and reused. Each case is then back-translated
sentence by sentence and checked with ahuike.backcheck (label-preserving filter).

Resumable: finished jobs in --out are skipped, rows are re-checked with the current
checker on load, and rows whose English source changed are discarded.

Jobs:
  test : every case x {ha, yo, ig} (plain) + code-switched variant for cases with code_switch
  train: anchored cases x {ha, yo, ig}; non-anchored cases only in their ablation_lang;
         code-switched variant for train cases with code_switch
  advice strings (9) x {ha, yo, ig} -> data/advice_i18n.json (MUST be native-speaker reviewed)

Usage (Kaggle, 2xT4):
  python scripts/translate_natlas.py --test-only
  python scripts/translate_natlas.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(ROOT))

from ahuike.backcheck import check  # noqa: E402
from ahuike.inference import ChatEngine  # noqa: E402
from ahuike.prompts import ADVICE_EN, LANGS  # noqa: E402

# Mild penalty against translation loops (the N-ATLaS card recommends 1.12 for free text).
REP_PENALTY = 1.1
TRANSLATOR_SYSTEM = ("You are an expert translator for Nigerian languages. You translate faithfully, "
                     "sentence for sentence, and never add or remove information.")

# Common symptom words, given as hints only when the English sentence contains the term.
# Native speakers should confirm these.
GLOSSARY: dict[str, list[tuple[tuple[str, ...], str, str]]] = {
    "ha": [(("cough",), "cough", "tari"),
           (("fever", "hot body", "body has been hot", "body is hot"), "fever", "zazzabi"),
           (("diarrhoea", "watery stool", "running stomach"), "diarrhoea", "gudawa"),
           (("vomit",), "vomit", "amai"),
           (("blood", "bleed"), "blood", "jini"),
           (("headache",), "headache", "ciwon kai"),
           (("breath",), "breathing", "numfashi")],
    "yo": [(("cough",), "cough", "ikọ́"),
           (("fever", "hot body", "body has been hot", "body is hot"), "fever", "ibà"),
           (("diarrhoea", "watery stool", "running stomach"), "diarrhoea", "ìgbẹ́ gbuuru"),
           (("vomit",), "vomit", "èébì"),
           (("blood", "bleed"), "blood", "ẹ̀jẹ̀"),
           (("headache",), "headache", "orí fífọ́"),
           (("breath",), "breathing", "mímí")],
    "ig": [(("cough",), "cough", "ụkwara"),
           (("fever", "hot body", "body has been hot", "body is hot"), "fever", "ahụ ọkụ"),
           (("diarrhoea", "watery stool", "running stomach"), "diarrhoea", "afọ ọsịsa"),
           (("vomit",), "vomit", "agbọ"),
           (("blood", "bleed"), "blood", "ọbara"),
           (("headache",), "headache", "isi ọwụwa"),
           (("breath",), "breathing", "iku ume")],
}

_SENT = re.compile(r"(?<=[.?!])\s+(?=[A-Z])")


def split_sentences(text: str, min_words: int = 5) -> list[str]:
    """Split into sentences, merging very short ones ("Good evening.", "What should I do?")
    into a neighbour: alone, N-ATLAS tends to comment on them instead of translating."""
    out: list[str] = []
    carry = ""
    for s in (x for x in _SENT.split(text.strip()) if x):
        s = f"{carry} {s}" if carry else s
        carry = ""
        if len(s.split()) < min_words:
            carry = s
        else:
            out.append(s)
    if carry:
        if out:
            out[-1] = f"{out[-1]} {carry}"
        else:
            out.append(carry)
    return out


def glossary_hint(sentence: str, lang: str) -> str:
    low = sentence.lower()
    pairs = [f"{en} = {tr}" for keys, en, tr in GLOSSARY.get(lang, []) if any(k in low for k in keys)]
    return f"Use these words: {'; '.join(pairs)}.\n" if pairs else ""


def to_lang_prompt(text: str, lang: str, variant: str) -> list[dict]:
    name = LANGS[lang]
    hint = glossary_hint(text, lang)
    if variant == "cs":
        user = (f"Rewrite the following English text in {name} the way people in Nigeria speak casually, "
                f"naturally mixing in a few common English words (code-switching), such as hospital, drip, BP. "
                f"Keep every number written as digits and keep the medical meaning exactly. {hint}"
                f"Output only the rewritten text.\n\n{text}")
    else:
        user = (f"Translate the following text from English to {name}. Keep every number written as "
                f"digits and keep the medical meaning exactly. {hint}"
                f"Output only the {name} translation.\n\n{text}")
    return [{"role": "system", "content": TRANSLATOR_SYSTEM}, {"role": "user", "content": user}]


def to_english_prompt(text: str, lang: str) -> list[dict]:
    name = LANGS[lang]
    user = (f"Translate the following {name} text to English. Keep every number written as digits. "
            f"Output only the English translation.\n\n{text}")
    return [{"role": "system", "content": TRANSLATOR_SYSTEM}, {"role": "user", "content": user}]


def clean(s: str) -> str:
    s = s.strip().strip('"').strip()
    for prefix in ("Translation:", "Here is the translation:", "English translation:", "Rewritten text:"):
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix):].strip()
    # One sentence in -> keep the first paragraph out (drops trailing commentary).
    return s.split("\n\n")[0].strip()


def src_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


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


def chat_dedup(engine: ChatEngine, convs: list[list[dict]], dedup: bool, **kw) -> list[str]:
    """Run chat, sending each distinct conversation once when dedup is on (greedy decoding)."""
    if not dedup:
        return [clean(x) for x in engine.chat(convs, **kw)]
    keys = [json.dumps(c, ensure_ascii=False) for c in convs]
    uniq = list(dict.fromkeys(keys))
    outs = dict(zip(uniq, (clean(x) for x in engine.chat([json.loads(k) for k in uniq], **kw))))
    return [outs[k] for k in keys]


def translate_cases(engine, jobs, cases, temperature):
    """Sentence-level forward + backward translation for a list of (case_id, lang, variant) jobs."""
    flat = []  # (job index, lang, variant, sentence)
    for j, (cid, lang, var) in enumerate(jobs):
        for s in split_sentences(cases[cid]["text_en"]):
            flat.append((j, lang, var, s))
    greedy = temperature == 0.0
    fwd = chat_dedup(engine, [to_lang_prompt(s, lang, var) for _, lang, var, s in flat], greedy,
                     max_new_tokens=160, temperature=temperature, repetition_penalty=REP_PENALTY)
    back = chat_dedup(engine, [to_english_prompt(t, lang) for t, (_, lang, _, _) in zip(fwd, flat)], True,
                      max_new_tokens=160, repetition_penalty=REP_PENALTY)
    tr = [[] for _ in jobs]
    bt = [[] for _ in jobs]
    for (j, _, _, _), t, b in zip(flat, fwd, back):
        tr[j].append(t)
        bt[j].append(b)
    return [" ".join(x) for x in tr], [" ".join(x) for x in bt]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="auto", choices=("auto", "vllm", "hf"))
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=None, help="default: <data>/translations.jsonl")
    ap.add_argument("--chunk", type=int, default=600, help="cases per batch")
    ap.add_argument("--retries", type=int, default=3, help="re-translate failed items with sampling")
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

    # --- load previous work; drop rows for changed cases; re-check with the current checker
    done = {}
    stale = 0
    for r in load_jsonl(out_path):
        c = cases.get(r["case_id"])
        if c is None or r.get("src") != src_hash(c["text_en"]):
            stale += 1
            continue
        r["bt_ok"], r["problems"] = check(c, r["text"], r["back_translation"])
        key = (r["case_id"], r["lang"], r["variant"])
        if key not in done or (r["bt_ok"] and not done[key]["bt_ok"]):
            done[key] = r
    print(f"{len(jobs)} jobs, {sum(j in done for j in jobs)} already done, {stale} stale rows dropped")

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
        print(f"wrote {advice_path} - MUST be reviewed by native speakers before training")

    # --- cases
    for attempt_round in range(args.retries + 1):
        pending = [j for j in jobs
                   if j not in done or (not done[j]["bt_ok"] and done[j]["attempt"] <= attempt_round)]
        if not pending:
            break
        temperature = 0.0 if attempt_round == 0 else 0.5
        for s in range(0, len(pending), args.chunk):
            chunk = pending[s:s + args.chunk]
            fwd, back = translate_cases(engine, chunk, cases, temperature)
            with open(out_path, "a", encoding="utf-8") as fh:
                for (cid, lang, var), t, b in zip(chunk, fwd, back):
                    ok, problems = check(cases[cid], t, b)
                    row = {"case_id": cid, "lang": lang, "variant": var, "text": t, "back_translation": b,
                           "bt_ok": ok, "problems": problems, "attempt": attempt_round + 1,
                           "src": src_hash(cases[cid]["text_en"])}
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
    for split in ("test", "train"):
        ok_langs: dict[str, set] = {}
        for (cid, lang, var), r in done.items():
            if cases[cid]["split"] == split and var == "plain" and r["bt_ok"]:
                ok_langs.setdefault(cid, set()).add(lang)
        stats[f"{split}/complete_all_4_languages"] = sum(len(v) == 3 for v in ok_langs.values())
    (data / "translation_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
