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

## Fields (`cases_*.jsonl`)
`id, population, age_months|age_days|ga_weeks|pp_days, sex, persona, findings, durations, vitals, triage, triggers,
tags (reasoning type), combo_key, text_en, context, anchored, code_switch, ablation_lang`

## Size (seed 2026)
- Test: 160 cases × 4 languages = **640 items**, plus up to 150 code-switched items
- Train: 2,400 cases (600 parallel-anchored)
- Label mix: about 40% emergency, 35% clinic, 25% home care
- Reasoning tags: `combination` (e.g. two pre-eclampsia signs), `duration_threshold` (14-day cough/diarrhoea),
  `rr_threshold` (age-specific fast breathing), `temp_threshold` (young-infant fever/hypothermia)

## Limitations
Synthetic, N-ATLAS-translated text. The back-translation filter is coarse. See docs/ETHICS.md.
Native-speaker-verified subsets are flagged separately in the results.
