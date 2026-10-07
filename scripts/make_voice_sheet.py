"""Pick test items for voice recording and write a reading sheet + manifest template.

  python scripts/make_voice_sheet.py --per-lang 15 --langs ha yo ig en
Writes voice/recording_sheet.md (what each speaker reads) and voice/manifest.csv (fill in speaker).
Only caregiver/self personas are used (people don't speak CHEW notes aloud).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-lang", type=int, default=15)
    ap.add_argument("--langs", nargs="+", default=["ha", "yo", "ig", "en"])
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    cases = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data" / "cases_test.jsonl", encoding="utf-8")}
    items = [json.loads(l) for l in open(ROOT / "data" / "eval_items.jsonl", encoding="utf-8")]
    rng = random.Random(args.seed)
    out = Path(ROOT / "voice")
    out.mkdir(exist_ok=True)
    sheet = ["# LAFIYA voice recording sheet", "",
             "Read each passage naturally, as if you were calling a health worker. Record in a normal room "
             "(some background noise is fine). Save each recording with the file name shown. "
             "Every speaker signs the consent form first (docs/CONSENT_FORM.md).", ""]
    manifest = ["file,case_id,lang,speaker"]
    for lang in args.langs:
        pool = [it for it in items if it["lang"] == lang and it["variant"] == "plain"
                and cases[it["case_id"]]["persona"] != "chew"]
        by_level = {}
        for it in pool:
            by_level.setdefault(cases[it["case_id"]]["triage"], []).append(it)
        chosen = []
        for lv in sorted(by_level):
            rng.shuffle(by_level[lv])
        while len(chosen) < args.per_lang and any(by_level.values()):
            for lv in sorted(by_level):
                if by_level[lv] and len(chosen) < args.per_lang:
                    chosen.append(by_level[lv].pop())
        sheet += [f"## {lang.upper()}", ""]
        for k, it in enumerate(chosen, 1):
            fname = f"voice/{lang}_{it['case_id']}_SPEAKER.wav"
            sheet += [f"**{k}. `{fname}`**", "", it["text"], ""]
            manifest.append(f"{fname},{it['case_id']},{lang},SPEAKER")
    (out / "recording_sheet.md").write_text("\n".join(sheet), encoding="utf-8")
    (out / "manifest.csv").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"wrote voice/recording_sheet.md and voice/manifest.csv ({len(manifest) - 1} items)")


if __name__ == "__main__":
    main()
