"""Generate NaijaTriage-Bench English cases with protocol-derived labels.

Splits on clinical combination (combo_key) so no test clinical picture appears in training.

Outputs (data/):
  cases_train.jsonl  - train pool; first --anchor cases are the parallel-anchored set
  cases_test.jsonl   - held-out test cases
  generation_stats.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(ROOT))

from lafiya.cases import sample_case  # noqa: E402
from lafiya.protocol import CHILD, CLINIC, EMERGENCY, HOME, INFANT, POSTPARTUM, PREGNANT  # noqa: E402

POP_WEIGHTS = {CHILD: 0.40, INFANT: 0.15, PREGNANT: 0.30, POSTPARTUM: 0.15}
LEVEL_MIX = {EMERGENCY: 0.40, CLINIC: 0.35, HOME: 0.25}
NON_EN = ("ha", "yo", "ig")
ALL_LANGS = ("en", "ha", "yo", "ig")


def round_robin(combos: list[str], by_combo: dict[str, list[dict]], n: int, cap: int) -> list[dict]:
    """Take up to n cases cycling through combos (one per combo per round), max `cap` per combo."""
    out: list[dict] = []
    for r in range(cap):
        for k in combos:
            if len(out) >= n:
                return out
            if r < len(by_combo[k]):
                out.append(by_combo[k][r])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--pool", type=int, default=60000)
    ap.add_argument("--train", type=int, default=2400)
    ap.add_argument("--anchor", type=int, default=600, help="train cases translated into all 4 languages")
    ap.add_argument("--cs-train", type=int, default=100, help="anchored cases also given code-switched versions")
    ap.add_argument("--test", type=int, default=160)
    ap.add_argument("--cs-test", type=int, default=50)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--out", default=str(ROOT / "data"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pops, weights = zip(*POP_WEIGHTS.items())
    pool = [sample_case(rng, rng.choices(pops, weights)[0]) for _ in range(args.pool)]

    by_combo: dict[str, list[dict]] = defaultdict(list)
    for c in pool:
        by_combo[c["combo_key"]].append(c)
    # De-duplicate identical texts within a combo.
    for k, cs in by_combo.items():
        seen, uniq = set(), []
        for c in cs:
            if c["text_en"] not in seen:
                seen.add(c["text_en"])
                uniq.append(c)
        rng.shuffle(uniq)
        by_combo[k] = uniq

    combos_by_level: dict[str, list[str]] = defaultdict(list)
    for k, cs in by_combo.items():
        combos_by_level[cs[0]["triage"]].append(k)

    train, test = [], []
    for level, mix in LEVEL_MIX.items():
        ks = sorted(combos_by_level[level])
        rng.shuffle(ks)
        n_test_combos = max(1, round(len(ks) * args.test_frac))
        test_ks, train_ks = ks[:n_test_combos], ks[n_test_combos:]
        test += round_robin(test_ks, by_combo, round(args.test * mix), cap=2)
        train += round_robin(train_ks, by_combo, round(args.train * mix), cap=10)

    rng.shuffle(train)
    rng.shuffle(test)
    # Interleave so the anchored subset keeps the level mix.
    for i, c in enumerate(train):
        c["id"] = f"tr{i:05d}"
        c["split"] = "train"
        c["anchored"] = i < args.anchor
        c["code_switch"] = i < args.cs_train
        c["ablation_lang"] = ALL_LANGS[i % 4]
    for i, c in enumerate(test):
        c["id"] = f"te{i:05d}"
        c["split"] = "test"
        c["code_switch"] = i < args.cs_test

    train_keys = {c["combo_key"] for c in train}
    assert not train_keys & {c["combo_key"] for c in test}, "combo leakage between train and test"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("cases_train.jsonl", train), ("cases_test.jsonl", test)):
        with open(out / name, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    def summary(rows: list[dict]) -> dict:
        return {
            "n": len(rows),
            "unique_combos": len({r["combo_key"] for r in rows}),
            "triage": Counter(r["triage"] for r in rows),
            "population": Counter(r["population"] for r in rows),
            "persona": Counter(r["persona"] for r in rows),
            "reasoning_tags": Counter(t for r in rows for t in r["tags"]),
        }

    stats = {
        "seed": args.seed,
        "pool_size": len(pool),
        "pool_unique_combos": len(by_combo),
        "combos_per_level": {k: len(v) for k, v in combos_by_level.items()},
        "train": summary(train),
        "train_anchored": summary(train[: args.anchor]),
        "test": summary(test),
    }
    with open(out / "generation_stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
