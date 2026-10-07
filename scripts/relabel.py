"""Re-apply the protocol rules to the existing cases after a rule change (e.g. clinical review).

  python scripts/relabel.py            # updates data/cases_train.jsonl and data/cases_test.jsonl in place

Case texts, ids and the train/test split stay the same, so no translation is redone; only the
gold triage, triggers and reasoning tags are recomputed. Prints how many labels changed.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(ROOT))

from ahuike.protocol import reasoning_tags, triage  # noqa: E402


def main() -> None:
    for name in ("cases_train.jsonl", "cases_test.jsonl"):
        path = ROOT / "data" / name
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        changed = Counter()
        for c in rows:
            level, triggers = triage(c)
            if (level, triggers) != (c["triage"], c["triggers"]):
                changed[f"{c['triage']} -> {level}"] += 1
            c["triage"], c["triggers"], c["tags"] = level, triggers, reasoning_tags(c)
        path.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in rows), encoding="utf-8")
        print(f"{name}: {sum(changed.values())} labels changed {dict(changed)}")


if __name__ == "__main__":
    main()
