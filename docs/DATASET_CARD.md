---
license: other
license_name: n-atlas
language: [en, ha, yo, ig]
tags: [medical, triage, maternal-health, child-health, nigeria, benchmark, n-atlas]
pretty_name: NaijaTriage-Bench
---

# NaijaTriage-Bench

A **parallel** maternal and child danger-sign triage benchmark in **English, Hausa, Yorùbá and Igbo**, with
**protocol-derived gold labels**.

## Construction
1. **Rule engine** (`ahuike/protocol/rules.py`): a simplified encoding of WHO IMCI (children 2–59 months),
   WHO young-infant danger signs (0–59 days) and the Nigerian CHEW Standing Orders maternal danger signs (pregnancy, postpartum).
2. **Case sampler** (`ahuike/cases.py`): samples a population, age, 1–4 findings, durations and (for CHEW notes) respiratory
   rate and temperature. The rule engine labels the *structured* case.
3. **Renderer**: Nigerian-English vignettes from caregivers, women, partners and CHEW notes. Includes lay terms
   ("hot body", "running stomach", "catarrh") and label-neutral distractors (agbo, "I think it is malaria").
4. **Translation**: N-ATLaS translates into ha/yo/ig, plus code-switched variants. N-ATLaS back-translates,
   and items are kept only if every finding and number survives (`ahuike/backcheck.py`).
5. **Split by clinical combination** (`combo_key`): no test combination of findings appears in training.
6. **Clinical review** (8 Oct 2026): a medical doctor reviewed the rules; two were made stricter and every case was
   relabelled (`scripts/relabel.py`). 45 of 385 test cases moved from clinic to emergency.

## Fields (`cases_*.jsonl`)
`id, population, age_months|age_days|ga_weeks|pp_days, sex, persona, findings, durations, vitals, triage, triggers,
tags (reasoning type), combo_key, text_en, context, anchored, code_switch, ablation_lang`

## Size (seed 2026)
- Test: **385 held-out cases**. After the N-ATLaS translation quality gate, **1,091 plain-text items**
  (English 385, Hausa 341, Igbo 227, Yorùbá 138) plus **92 code-switched** items.
- **Parallel core:** 95 cases faithful in all four languages, for a fair language comparison.
- **Native-verified subset:** Igbo items a native speaker marked as the same meaning (`verified_ig.json`).
- Train: 2,400 cases (1,000 flagged for parallel anchoring), split from test by clinical combination.
- Reasoning tags: `combination` (e.g. two pre-eclampsia signs), `duration_threshold` (14-day cough/diarrhoea),
  `rr_threshold` (age-specific fast breathing), `temp_threshold` (young-infant fever/hypothermia)

## Files
| File | Contents |
|---|---|
| `data/cases_test.jsonl`, `data/cases_train.jsonl` | Structured cases, gold triage, triggers, English text |
| `data/eval_items.jsonl` | Every benchmark item `{case_id, lang, variant, text}` |
| `data/translation_stats.json` | N-ATLaS translation retention per language |
| `data/verified_ig.json` | Igbo native-speaker verdicts (same / small difference / wrong) |
| `data/advice_reviewed.json` | Reviewed Igbo advice messages |
| `results/preds.jsonl` | Raw outputs of base N-ATLaS, base few-shot, AHỤIKE and the parallel variant |
| `results/REPORT.md`, `results/metrics.json` | Scored results |

## Limitations
Synthetic, N-ATLAS-translated text. The back-translation filter is coarse: our Igbo reviewer judged 19 of 40 sampled
items that passed it to be wrong in meaning. See docs/ETHICS.md.
Native-speaker-verified subsets are flagged separately in the results.
