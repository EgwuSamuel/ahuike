# NaijaTriage-Bench results

`ahuike` = the released AHỤIKE model. `ahuike_parallel` = the tested alternative (each training case in all four languages).

Gold labels are produced by the AHỤIKE protocol engine (WHO IMCI + Nigerian CHEW Standing Orders).
Brackets are 95% case-clustered bootstrap CIs. Under-triage = emergency cases not referred.

## Headline: all faithful items (en/ha/yo/ig, plain text)

| System | N | Accuracy | Macro-F1 | Under-triage ↓ | Over-triage | Danger-sign F1 | CLCC ↑ | Worst-lang gap ↓ | JSON valid |
|---|---|---|---|---|---|---|---|---|---|
| base | 1091 | 37.2 [31.9–42.1] | 21.9 | 96.5 [94.5–98.2] | 37.4 | 23.0 | 88.4 [83.3–93.4] | 5.3 | 98.2 |
| base_fewshot | 1091 | 53.8 [49.8–57.8] | 48.5 | 34.6 [28.8–40.7] | 53.6 | 30.1 | 49.5 [41.3–57.6] | 8.4 | 96.7 |
| ahuike_parallel | 1091 | 90.8 [88.4–92.9] | 90.7 | 9.4 [5.8–13.7] | 3.2 | 84.6 | 81.1 [75.0–87.5] | 6.3 | 100.0 |
| ahuike | 1091 | 90.9 [88.3–93.2] | 90.7 | 5.7 [2.8–9.2] | 5.7 | 87.1 | 81.1 [75.0–87.3] | 2.1 | 100.0 |

## Accuracy / under-triage by language (all faithful items; case mix differs by language)

| System | en acc | en under | ha acc | ha under | yo acc | yo under | ig acc | ig under |
|---|---|---|---|---|---|---|---|---|
| base | 39.2 | 93.1 | 37.8 | 96.5 | 34.8 | 100.0 | 34.4 | 100.0 |
| base_fewshot | 60.8 | 28.7 | 49.6 | 24.8 | 48.6 | 50.9 | 51.5 | 48.5 |
| ahuike_parallel | 94.3 | 6.9 | 89.7 | 10.6 | 92.0 | 10.5 | 85.9 | 11.1 |
| ahuike | 93.2 | 5.0 | 90.9 | 6.4 | 89.9 | 5.3 | 87.7 | 6.1 |

## Parallel core: the same cases in every language (fair language comparison)

| System | Cases | en acc | en under | ha acc | ha under | yo acc | yo under | ig acc | ig under |
|---|---|---|---|---|---|---|---|---|---|
| base | 95 | 36.8 | 88.1 | 33.7 | 95.2 | 32.6 | 100.0 | 31.6 | 100.0 |
| base_fewshot | 95 | 57.9 | 35.7 | 50.5 | 23.8 | 49.5 | 57.1 | 51.6 | 50.0 |
| ahuike_parallel | 95 | 94.7 | 9.5 | 93.7 | 9.5 | 90.5 | 14.3 | 88.4 | 14.3 |
| ahuike | 95 | 91.6 | 9.5 | 91.6 | 4.8 | 92.6 | 7.1 | 90.5 | 7.1 |

## Cross-lingual consistency

| System | Cases (all 4 langs) | CLCC | Consistent & correct | Emergency missed in ≥1 language | Worst language |
|---|---|---|---|---|---|
| base | 95 | 88.4 | 30.5 | 100.0 | ig |
| base_fewshot | 95 | 49.5 | 27.4 | 69.0 | yo |
| ahuike_parallel | 95 | 81.1 | 80.0 | 23.8 | ig |
| ahuike | 95 | 81.1 | 81.1 | 14.3 | ig |

## Code-switched input (ha/yo/ig mixed with English)

| System | N | Accuracy | Under-triage |
|---|---|---|---|
| base | 92 | 32.6 | 97.6 |
| base_fewshot | 92 | 47.8 | 50.0 |
| ahuike_parallel | 92 | 84.8 | 11.9 |
| ahuike | 92 | 81.5 | 4.8 |

## Human-validated subsets

| System | Subset | Language | N | Accuracy | Under-triage |
|---|---|---|---|---|---|
| base | AI translation verified by native speaker | ig | 14 | 21.4 | 100.0 |
| base_fewshot | AI translation verified by native speaker | ig | 14 | 50.0 | 57.1 |
| ahuike_parallel | AI translation verified by native speaker | ig | 14 | 85.7 | 0.0 |
| ahuike | AI translation verified by native speaker | ig | 14 | 92.9 | 0.0 |

## Accuracy by population and reasoning type

| System | child | infant | postpartum | pregnant | combination | duration_threshold | rr_threshold | temp_threshold |
|---|---|---|---|---|---|---|---|---|
| base | 46.0 | 15.3 | 36.9 | 33.4 | 50.5 | 51.5 | 52.6 | 0.0 |
| base_fewshot | 51.4 | 72.9 | 55.3 | 51.3 | 59.7 | 53.2 | 61.4 | 20.0 |
| ahuike_parallel | 87.7 | 88.2 | 88.5 | 95.7 | 88.9 | 86.8 | 87.7 | 100.0 |
| ahuike | 86.4 | 85.9 | 90.3 | 96.7 | 87.2 | 84.7 | 80.7 | 80.0 |

## Paired significance vs `base` (exact McNemar)

| System | Items only ref right | Items only system right | p (all) | p (emergency cases) |
|---|---|---|---|---|
| base_fewshot | 155 | 350 | 2.3e-18 | 1.2e-91 |
| ahuike_parallel | 59 | 692 | 5.9e-138 | 4.5e-131 |
| ahuike | 81 | 712 | 7.5e-127 | 4.3e-137 |

## Data design: `ahuike` (diverse cases) vs `ahuike_parallel` (each case in all 4 languages)

Same number of training examples and language mix. Paired on identical benchmark items.

| Test | Only variant right | Only shipped right | p |
|---|---|---|---|
| All items (McNemar) | 47 | 45 | 0.92 |
| Emergency items (McNemar) | 5 | 25 | 0.00032 |
| Emergency cases, 4 languages = 1 unit (sign test) | 4 | 15 | 0.019 |

Missed-emergency rate, variant minus shipped: **4.0 points** (95% case-clustered CI 1.4 to 7.1).
