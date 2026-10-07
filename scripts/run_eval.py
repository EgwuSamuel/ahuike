"""Run NaijaTriage-Bench through base N-ATLaS and AHỤIKE adapters. Resumable.

  python scripts/run_eval.py --systems base base_fewshot ahuike=outputs/ahuike-lora \
      ablation=outputs/ablation-lora --backend vllm

System specs:
  base            - N-ATLaS, protocol system prompt, zero-shot
  base_fewshot    - N-ATLaS, protocol system prompt + 3 English worked examples
  NAME=PATH       - N-ATLaS + LoRA adapter at PATH, zero-shot
Predictions are appended to results/preds.jsonl as
  {system, case_id, lang, variant, output}
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

from ahuike.inference import ChatEngine  # noqa: E402
from ahuike.prompts import advice_key, build_messages, input_hash, load_advice, target_json  # noqa: E402
from ahuike.protocol import CLINIC, EMERGENCY, HOME  # noqa: E402


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def few_shot_examples(train: list[dict]) -> list[tuple[str, str]]:
    """One English example per triage level, deterministic, from the training split."""
    adv = load_advice()["en"]
    out = []
    for level in (EMERGENCY, CLINIC, HOME):
        c = next(c for c in train if c["triage"] == level and c["persona"] != "chew")
        out.append((c["text_en"], target_json(c["triage"], c["triggers"],
                                              adv[advice_key(c["population"], c["triage"])])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--backend", default="auto", choices=("auto", "vllm", "hf"))
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "results" / "preds.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--batch-size", type=int, default=8, help="transformers backend only; lower it on OOM")
    args = ap.parse_args()

    data = Path(args.data)
    items = load_jsonl(data / "eval_items.jsonl")
    if args.limit:
        items = items[: args.limit]
    train = load_jsonl(data / "cases_train.jsonl")
    shots = few_shot_examples(train)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # An answer is reused only if it was produced for exactly the same input text.
    done = {(r["system"], r["case_id"], r["lang"], r["variant"], r.get("text_hash")) for r in load_jsonl(out)}

    specs = []
    for s in args.systems:
        name, _, path = s.partition("=")
        specs.append((name, path or None))
    engine = ChatEngine(backend=args.backend, enable_lora=any(p for _, p in specs))

    for name, adapter in specs:
        todo = [it for it in items
                if (name, it["case_id"], it["lang"], it["variant"], input_hash(it["text"])) not in done]
        print(f"[{name}] {len(todo)} items to run")
        if not todo:
            continue
        convs = [build_messages(it["text"], shots if name == "base_fewshot" else None) for it in todo]
        outs = engine.chat(convs, adapter=adapter, max_new_tokens=args.max_new_tokens,
                           batch_size=args.batch_size)
        with open(out, "a", encoding="utf-8") as fh:
            for it, o in zip(todo, outs):
                fh.write(json.dumps({"system": name, "case_id": it["case_id"], "lang": it["lang"],
                                     "variant": it["variant"], "text_hash": input_hash(it["text"]),
                                     "output": o}, ensure_ascii=False) + "\n")
        print(f"[{name}] done")


if __name__ == "__main__":
    main()
