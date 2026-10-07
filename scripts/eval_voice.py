"""Voice end-to-end evaluation: recorded test cases -> N-ATLAS ASR -> AHỤIKE -> triage.

Record speakers reading NaijaTriage test items aloud (the text from data/eval_items.jsonl),
then list them in voice/manifest.csv:
  file,case_id,lang,speaker
  voice/ha_te00012_s1.wav,te00012,ha,s1

  python scripts/eval_voice.py --backend hf --adapter outputs/ahuike-lora
Outputs results/voice_preds.jsonl and results/voice_eval.json (WER + end-to-end triage accuracy).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(ROOT))

from ahuike.pipeline import ASR, Ahuike, TriageLLM  # noqa: E402
from ahuike.protocol import EMERGENCY  # noqa: E402


def words(s: str) -> list[str]:
    return re.findall(r"[^\W_]+", s.lower())


def wer(ref: str, hyp: str) -> float:
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "voice" / "manifest.csv"))
    ap.add_argument("--backend", default="hf", choices=("gguf", "hf"))
    ap.add_argument("--gguf", default=None)
    ap.add_argument("--adapter", default=None)
    args = ap.parse_args()

    cases = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data" / "cases_test.jsonl", encoding="utf-8")}
    refs = {}
    for l in open(ROOT / "data" / "eval_items.jsonl", encoding="utf-8"):
        it = json.loads(l)
        if it["variant"] == "plain":
            refs[(it["case_id"], it["lang"])] = it["text"]

    app = Ahuike(TriageLLM(args.backend, gguf_path=args.gguf, adapter=args.adapter),
                 ASR(device=0 if args.backend == "hf" else None))
    rows = list(csv.DictReader(open(args.manifest, encoding="utf-8")))
    out_rows = []
    for r in rows:
        audio = str(ROOT / r["file"]) if not Path(r["file"]).is_absolute() else r["file"]
        res = app.triage_audio(audio, r["lang"])
        g = cases[r["case_id"]]
        ref = refs.get((r["case_id"], r["lang"]), "")
        # Also run the reference (typed) text, to separate ASR errors from triage errors.
        typed = app.triage_text(ref, r["lang"]) if ref else {"triage": None}
        out_rows.append({**r, "transcript": res["input"], "reference": ref,
                         "wer": wer(ref, res["input"]) if ref else None,
                         "gold": g["triage"], "pred_voice": res["triage"], "pred_typed": typed["triage"],
                         "asr_latency_s": res.get("asr_latency_s"), "llm_latency_s": res["latency_s"]})
        print(f"{r['file']}: gold={g['triage']} voice={res['triage']} wer={out_rows[-1]['wer']}")

    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "voice_preds.jsonl", "w", encoding="utf-8") as fh:
        for o in out_rows:
            fh.write(json.dumps(o, ensure_ascii=False) + "\n")
    by_lang = defaultdict(list)
    for o in out_rows:
        by_lang[o["lang"]].append(o)

    def agg(rs):
        em = [o for o in rs if o["gold"] == EMERGENCY]
        ws = [o["wer"] for o in rs if o["wer"] is not None]
        return {"n": len(rs),
                "voice_accuracy": sum(o["pred_voice"] == o["gold"] for o in rs) / len(rs),
                "typed_accuracy": sum(o["pred_typed"] == o["gold"] for o in rs) / len(rs),
                "voice_under_triage": (sum(o["pred_voice"] != EMERGENCY for o in em) / len(em)) if em else None,
                "mean_wer": sum(ws) / len(ws) if ws else None,
                "speakers": len({o["speaker"] for o in rs})}
    summary = {"overall": agg(out_rows), "by_language": {l: agg(rs) for l, rs in by_lang.items()}}
    (ROOT / "results" / "voice_eval.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
