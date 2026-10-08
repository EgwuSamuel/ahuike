"""Score results/preds.jsonl against protocol gold labels and write the benchmark report.

  python scripts/compute_metrics.py            # -> results/metrics.json, results/REPORT.md
Runs on CPU in seconds; no GPU needed.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(ROOT))

from ahuike.metrics import (  # noqa: E402
    CORE_LANGS, bootstrap_ci, clcc, confusion, paired, paired_cases, score_item, summarize,
)
from ahuike.prompts import input_hash  # noqa: E402
from ahuike.protocol import EMERGENCY  # noqa: E402

ORDER = ["base", "base_fewshot", "ahuike_parallel", "ahuike"]
# ahuike          - the shipped model: diverse cases, each case in ONE language (sft_ablation.jsonl, outputs/ablation-lora)
# ahuike_parallel - the tested alternative: each case in all four languages (sft_anchored.jsonl, outputs/ahuike-lora)
SHIPPED, VARIANT = "ahuike", "ahuike_parallel"


def load_jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def pct(x) -> str:
    return "–" if x is None or x != x else f"{100 * x:.1f}"


def ci(lo_hi) -> str:
    lo, hi = lo_hi
    return "" if lo != lo else f" [{100 * lo:.1f}–{100 * hi:.1f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=str(ROOT / "results" / "preds.jsonl"))
    ap.add_argument("--cases", default=str(ROOT / "data" / "cases_test.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--reference", default="base", help="system to compare others against")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    gold = {c["id"]: c for c in load_jsonl(Path(args.cases))}
    preds = load_jsonl(Path(args.preds))
    # Last write wins if a system/item was re-run.
    # Score only answers given to the CURRENT benchmark text (translations can be redone).
    items_path = Path(args.cases).with_name("eval_items.jsonl")
    current = None
    if items_path.exists():
        current = {(i["case_id"], i["lang"], i["variant"]): input_hash(i["text"])
                   for i in load_jsonl(items_path)}
    latest = {}
    for p in preds:
        key = (p["case_id"], p["lang"], p.get("variant", "plain"))
        if current is not None and (key not in current or p.get("text_hash", current[key]) != current[key]):
            continue
        latest[(p["system"], *key)] = p
    by_sys: dict[str, list[dict]] = defaultdict(list)
    for p in latest.values():
        by_sys[p["system"]].append(score_item(p, gold[p["case_id"]]))
    systems = sorted(by_sys, key=lambda s: (ORDER.index(s) if s in ORDER else 99, s))

    report: dict = {}
    for s in systems:
        items = by_sys[s]
        plain = [i for i in items if i["variant"] == "plain"]
        csw = [i for i in items if i["variant"] == "cs"]
        em = [i for i in plain if i["gold"] == EMERGENCY]
        langs_by_case = defaultdict(set)
        for i in plain:
            langs_by_case[i["case_id"]].add(i["lang"])
        core_ids = {c for c, ls in langs_by_case.items() if set(CORE_LANGS) <= ls}
        core = [i for i in plain if i["case_id"] in core_ids]
        r = {
            "overall": summarize(plain),
            "overall_ci": {
                "accuracy": bootstrap_ci(plain, lambda x: summarize(x).get("accuracy", float("nan")), args.boot),
                "under_triage_rate": bootstrap_ci(em, lambda x: summarize(x).get("under_triage_rate", float("nan")), args.boot),
                "clcc": bootstrap_ci(plain, lambda x: clcc(x).get("clcc", float("nan")), args.boot),
            },
            "by_language": {l: summarize([i for i in plain if i["lang"] == l]) for l in CORE_LANGS},
            "parallel_core": summarize(core),
            "parallel_core_by_language": {l: summarize([i for i in core if i["lang"] == l]) for l in CORE_LANGS},
            "code_switched": {l: summarize([i for i in csw if i["lang"] == l]) for l in ("ha", "yo", "ig")},
            "code_switched_all": summarize(csw),
            "by_population": {p: summarize([i for i in plain if i["population"] == p])
                              for p in sorted({i["population"] for i in plain})},
            "by_reasoning_tag": {t: summarize([i for i in plain if t in i["tags"]])
                                 for t in sorted({t for i in plain for t in i["tags"]})},
            "clcc": clcc(plain),
            "confusion": confusion(plain),
        }
        # Human-validated subsets: native-written cases, and AI translations a native speaker marked SAME.
        r["native_written"] = {l: summarize([i for i in items if i["variant"] == "native" and i["lang"] == l])
                               for l in ("ha", "yo", "ig")}
        r["native_verified"] = {}
        for l in ("ha", "yo", "ig"):
            vpath = Path(args.cases).with_name(f"verified_{l}.json")
            if vpath.exists():
                same = {cid for cid, v in json.loads(vpath.read_text(encoding="utf-8")).items() if v["verdict"] == "same"}
                r["native_verified"][l] = summarize([i for i in plain if i["lang"] == l and i["case_id"] in same])
        if s != args.reference and args.reference in by_sys:
            ref = by_sys[args.reference]
            r[f"vs_{args.reference}"] = {
                "accuracy_mcnemar": paired(ref, items, "correct"),
                "emergency_detection_mcnemar": paired(ref, items, "correct", only_gold=EMERGENCY),
            }
        report[s] = r

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ markdown report
    L = ["# NaijaTriage-Bench results", "",
         "`ahuike` = the released AHỤIKE model. `ahuike_parallel` = the tested alternative (each training case in all four languages).", "",
         "Gold labels are produced by the AHỤIKE protocol engine (WHO IMCI + Nigerian CHEW Standing Orders).",
         "Brackets are 95% case-clustered bootstrap CIs. Under-triage = emergency cases not referred.", "",
         "## Headline: all faithful items (en/ha/yo/ig, plain text)", "",
         "| System | N | Accuracy | Macro-F1 | Under-triage ↓ | Over-triage | Danger-sign F1 | CLCC ↑ | Worst-lang gap ↓ | JSON valid |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for s in systems:
        o, c, k = report[s]["overall"], report[s]["overall_ci"], report[s]["clcc"]
        L.append(f"| {s} | {o['n']} | {pct(o['accuracy'])}{ci(c['accuracy'])} | {pct(o['macro_f1'])} | "
                 f"{pct(o['under_triage_rate'])}{ci(c['under_triage_rate'])} | {pct(o['over_triage_rate'])} | "
                 f"{pct(o['danger_sign_f1'])} | {pct(k.get('clcc'))}{ci(c['clcc'])} | "
                 f"{pct(k.get('worst_language_gap'))} | {pct(o['json_validity'])} |")
    L += ["", "## Accuracy / under-triage by language (all faithful items; case mix differs by language)", "",
          "| System | " + " | ".join(f"{l} acc | {l} under" for l in CORE_LANGS) + " |",
          "|---|" + "---|" * (2 * len(CORE_LANGS))]
    for s in systems:
        b = report[s]["by_language"]
        L.append(f"| {s} | " + " | ".join(f"{pct(b[l].get('accuracy'))} | {pct(b[l].get('under_triage_rate'))}"
                                          for l in CORE_LANGS) + " |")
    L += ["", "## Parallel core: the same cases in every language (fair language comparison)", "",
          "| System | Cases | " + " | ".join(f"{l} acc | {l} under" for l in CORE_LANGS) + " |",
          "|---|---|" + "---|" * (2 * len(CORE_LANGS))]
    for s in systems:
        b = report[s]["parallel_core_by_language"]
        L.append(f"| {s} | {report[s]['clcc'].get('n_cases', 0)} | " + " | ".join(
            f"{pct(b[l].get('accuracy'))} | {pct(b[l].get('under_triage_rate'))}" for l in CORE_LANGS) + " |")
    L += ["", "## Cross-lingual consistency", "",
          "| System | Cases (all 4 langs) | CLCC | Consistent & correct | Emergency missed in ≥1 language | Worst language |",
          "|---|---|---|---|---|---|"]
    for s in systems:
        k = report[s]["clcc"]
        L.append(f"| {s} | {k.get('n_cases', 0)} | {pct(k.get('clcc'))} | {pct(k.get('consistent_and_correct'))} | "
                 f"{pct(k.get('emergency_missed_in_any_language'))} | {k.get('worst_language', '–')} |")
    L += ["", "## Code-switched input (ha/yo/ig mixed with English)", "",
          "| System | N | Accuracy | Under-triage |", "|---|---|---|---|"]
    for s in systems:
        o = report[s]["code_switched_all"]
        L.append(f"| {s} | {o.get('n', 0)} | {pct(o.get('accuracy'))} | {pct(o.get('under_triage_rate'))} |")
    if any(report[s]["native_written"][l].get("n") or report[s]["native_verified"].get(l, {}).get("n")
           for s in systems for l in ("ha", "yo", "ig")):
        L += ["", "## Human-validated subsets", "",
              "| System | Subset | Language | N | Accuracy | Under-triage |", "|---|---|---|---|---|---|"]
        for s in systems:
            for l in ("ha", "yo", "ig"):
                for label, o in (("written by native speaker", report[s]["native_written"][l]),
                                 ("AI translation verified by native speaker", report[s]["native_verified"].get(l, {}))):
                    if o.get("n"):
                        L.append(f"| {s} | {label} | {l} | {o['n']} | {pct(o.get('accuracy'))} | {pct(o.get('under_triage_rate'))} |")
    tags = sorted({t for s in systems for t in report[s]["by_reasoning_tag"]})
    pops = sorted({p for s in systems for p in report[s]["by_population"]})
    L += ["", "## Accuracy by population and reasoning type", "",
          "| System | " + " | ".join(pops + tags) + " |", "|---|" + "---|" * (len(pops) + len(tags))]
    for s in systems:
        bp, bt = report[s]["by_population"], report[s]["by_reasoning_tag"]
        L.append(f"| {s} | " + " | ".join([pct(bp.get(p, {}).get("accuracy")) for p in pops]
                                          + [pct(bt.get(t, {}).get("accuracy")) for t in tags]) + " |")
    L += ["", f"## Paired significance vs `{args.reference}` (exact McNemar)", "",
          "| System | Items only ref right | Items only system right | p (all) | p (emergency cases) |",
          "|---|---|---|---|---|"]
    for s in systems:
        v = report[s].get(f"vs_{args.reference}")
        if v:
            a, e = v["accuracy_mcnemar"], v["emergency_detection_mcnemar"]
            L.append(f"| {s} | {a['a_only']} | {a['b_only']} | {a['p_value']:.2g} | {e['p_value']:.2g} |")
    if SHIPPED in by_sys and VARIANT in by_sys:
        a, b = by_sys[VARIANT], by_sys[SHIPPED]
        both = [dict(i, _s=0) for i in a] + [dict(i, _s=1) for i in b]

        def miss_gap(x):  # variant minus shipped under-triage, on emergency items
            em = [i for i in x if i["gold"] == EMERGENCY]
            r = [sum(i["under"] for i in em if i["_s"] == k) / max(1, sum(i["_s"] == k for i in em)) for k in (0, 1)]
            return r[0] - r[1]

        dd = {
            "accuracy_mcnemar": paired(a, b, "correct"),
            "emergency_mcnemar": paired(a, b, "correct", only_gold=EMERGENCY),
            "emergency_case_sign_test": paired_cases(a, b, "correct", only_gold=EMERGENCY),
            "under_triage_gap": miss_gap(both),
            "under_triage_gap_ci": bootstrap_ci(both, miss_gap, args.boot),
        }
        report["data_design"] = dd
        lo, hi = dd["under_triage_gap_ci"]
        L += ["", f"## Data design: `{SHIPPED}` (diverse cases) vs `{VARIANT}` (each case in all 4 languages)", "",
              "Same number of training examples and language mix. Paired on identical benchmark items.", "",
              "| Test | Only variant right | Only shipped right | p |", "|---|---|---|---|"]
        for label, k in (("All items (McNemar)", "accuracy_mcnemar"),
                         ("Emergency items (McNemar)", "emergency_mcnemar"),
                         ("Emergency cases, 4 languages = 1 unit (sign test)", "emergency_case_sign_test")):
            L.append(f"| {label} | {dd[k]['a_only']} | {dd[k]['b_only']} | {dd[k]['p_value']:.2g} |")
        L += ["", f"Missed-emergency rate, variant minus shipped: **{100 * dd['under_triage_gap']:.1f} points** "
                  f"(95% case-clustered CI {100 * lo:.1f} to {100 * hi:.1f})."]
        (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
