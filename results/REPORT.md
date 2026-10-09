# NaijaTriage-Bench results

`ahuike` = the released AHỤIKE model. `ahuike_parallel` = the tested alternative (each training case in all four languages).
Gold labels follow the clinician-reviewed rules (8 Oct 2026). Fine-tuned systems are scored as deployed, after the review guard (`ahuike.prompts.apply_review_guard`).

Gold labels are produced by the AHỤIKE protocol engine (WHO IMCI + Nigerian CHEW Standing Orders).
Brackets are 95% case-clustered bootstrap CIs. Under-triage = emergency cases not referred.

## Headline: all faithful items (en/ha/yo/ig, plain text)

| System | N | Accuracy | Macro-F1 | Under-triage ↓ | Over-triage | Danger-sign F1 | CLCC ↑ | Worst-lang gap ↓ | JSON valid |
|---|---|---|---|---|---|---|---|---|---|
| base | 1091 | 25.6 [21.0–30.2] | 16.7 | 97.3 [95.7–98.6] | 46.7 | 23.1 | 88.4 [83.3–93.4] | 5.3 | 98.2 |
| base_fewshot | 1091 | 50.7 [46.7–54.6] | 45.0 | 40.9 [35.3–46.6] | 58.0 | 30.3 | 49.5 [41.3–57.6] | 10.5 | 96.7 |
| ahuike_parallel | 1091 | 91.8 [89.7–93.7] | 90.3 | 5.8 [3.7–8.3] | 4.3 | 76.6 | 77.9 [71.4–84.7] | 6.3 | 100.0 |
| ahuike | 1091 | 92.7 [90.3–94.8] | 90.9 | 3.3 [1.5–5.3] | 5.1 | 79.4 | 87.4 [82.7–92.6] | 3.2 | 100.0 |

## Accuracy / under-triage by language (all faithful items; case mix differs by language)

| System | en acc | en under | ha acc | ha under | yo acc | yo under | ig acc | ig under |
|---|---|---|---|---|---|---|---|---|
| base | 27.5 | 94.6 | 25.5 | 97.3 | 22.5 | 100.0 | 24.2 | 100.0 |
| base_fewshot | 56.4 | 37.6 | 51.0 | 29.5 | 42.0 | 56.8 | 45.8 | 54.1 |
| ahuike_parallel | 94.8 | 3.4 | 90.6 | 6.6 | 92.0 | 9.5 | 88.5 | 6.6 |
| ahuike | 94.5 | 2.0 | 92.1 | 4.4 | 91.3 | 5.4 | 91.2 | 2.5 |

## Parallel core: the same cases in every language (fair language comparison)

| System | Cases | en acc | en under | ha acc | ha under | yo acc | yo under | ig acc | ig under |
|---|---|---|---|---|---|---|---|---|---|
| base | 95 | 24.2 | 90.7 | 21.1 | 96.3 | 20.0 | 100.0 | 18.9 | 100.0 |
| base_fewshot | 95 | 45.3 | 50.0 | 47.4 | 33.3 | 36.8 | 66.7 | 43.2 | 57.4 |
| ahuike_parallel | 95 | 96.8 | 3.7 | 94.7 | 5.6 | 90.5 | 13.0 | 90.5 | 9.3 |
| ahuike | 95 | 95.8 | 1.9 | 93.7 | 3.7 | 92.6 | 7.4 | 93.7 | 3.7 |

## Cross-lingual consistency

| System | Cases (all 4 langs) | CLCC | Consistent & correct | Emergency missed in ≥1 language | Worst language |
|---|---|---|---|---|---|
| base | 95 | 88.4 | 17.9 | 100.0 | ig |
| base_fewshot | 95 | 49.5 | 20.0 | 75.9 | yo |
| ahuike_parallel | 95 | 77.9 | 77.9 | 24.1 | yo |
| ahuike | 95 | 87.4 | 87.4 | 9.3 | yo |

## Code-switched input (ha/yo/ig mixed with English)

| System | N | Accuracy | Under-triage |
|---|---|---|---|
| base | 92 | 18.5 | 98.2 |
| base_fewshot | 92 | 40.2 | 57.1 |
| ahuike_parallel | 92 | 89.1 | 5.4 |
| ahuike | 92 | 83.7 | 12.5 |

## Human-validated subsets

| System | Subset | Language | N | Accuracy | Under-triage |
|---|---|---|---|---|---|
| base | AI translation verified by native speaker | ig | 14 | 14.3 | 100.0 |
| base_fewshot | AI translation verified by native speaker | ig | 14 | 42.9 | 62.5 |
| ahuike_parallel | AI translation verified by native speaker | ig | 14 | 85.7 | 12.5 |
| ahuike | AI translation verified by native speaker | ig | 14 | 92.9 | 0.0 |

## Accuracy by population and reasoning type

| System | child | infant | postpartum | pregnant | combination | duration_threshold | rr_threshold | temp_threshold |
|---|---|---|---|---|---|---|---|---|
| base | 46.0 | 15.3 | 15.2 | 13.3 | 27.0 | 51.5 | 52.6 | 0.0 |
| base_fewshot | 51.4 | 72.9 | 57.6 | 41.5 | 53.4 | 53.2 | 61.4 | 20.0 |
| ahuike_parallel | 87.7 | 88.2 | 93.5 | 95.7 | 90.9 | 86.8 | 87.7 | 100.0 |
| ahuike | 86.4 | 89.4 | 93.5 | 99.0 | 90.2 | 84.7 | 80.7 | 80.0 |

## Paired significance vs `base` (exact McNemar)

| System | Items only ref right | Items only system right | p (all) | p (emergency cases) |
|---|---|---|---|---|
| base_fewshot | 105 | 399 | 2.3e-41 | 2.2e-106 |
| ahuike_parallel | 55 | 843 | 4e-182 | 7.9e-177 |
| ahuike | 63 | 855 | 2.5e-178 | 3.9e-180 |

## Data design: `ahuike` (diverse cases) vs `ahuike_parallel` (each case in all 4 languages)

Same number of training examples and language mix. Paired on identical benchmark items.

| Test | Only variant right | Only shipped right | p |
|---|---|---|---|
| All items (McNemar) | 35 | 39 | 0.73 |
| Emergency items (McNemar) | 9 | 20 | 0.061 |
| Emergency cases, 4 languages = 1 unit (sign test) | 6 | 16 | 0.052 |

Missed-emergency rate, variant minus shipped: **1.7 points** (95% case-clustered CI -0.5 to 3.8).
