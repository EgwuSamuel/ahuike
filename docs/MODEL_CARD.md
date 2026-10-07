---
license: other
license_name: n-atlas
base_model: NCAIR1/N-ATLaS
language: [en, ha, yo, ig]
tags: [medical, triage, maternal-health, child-health, nigeria, n-atlas, powered-by-awarri]
---

# AHUIKE-N-ATLaS-8B (Powered by Awarri)

Maternal and child **danger-sign triage** fine-tuned from **N-ATLaS-8B** for English, Hausa, Yorùbá and Igbo.
It returns one of `EMERGENCY_REFER_NOW`, `CLINIC_WITHIN_24H` or `HOME_CARE`, the protocol danger signs that triggered it,
and short advice in the user's language.

**Powered by Awarri.** This is a derivative of N-ATLaS and is distributed under the N-ATLaS licence (same terms).

*N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation and Digital Economy, and powered by Awarri Technologies.*

## Training
- QLoRA (r=16, α=32, all attention + MLP projections), 2 epochs, lr 2e-4, Unsloth, one T4 GPU.
- Data: ~2,700 parallel-anchored examples from NaijaTriage-Bench (train split). Each case appears in en/ha/yo/ig with an
  identical target, plus code-switched variants. Labels come from a rule engine encoding WHO IMCI and Nigerian CHEW Standing Orders.
- Hausa/Yorùbá/Igbo text translated by N-ATLaS and filtered by back-translation.

## Evaluation
See `results/REPORT.md` in the project repository: accuracy, under-triage rate, danger-sign F1,
Cross-Lingual Clinical Consistency (CLCC), compared with base N-ATLaS using paired McNemar tests.

## Prompt
Use the system prompt in `ahuike/prompts.py` (`SYSTEM_PROMPT`). The model replies with one JSON object.

## Intended use and limits
Decision support for community health workers and caregivers. **Not a diagnostic device.** It errs toward referral.
Trained on synthetic cases. See `docs/ETHICS.md` for protocol simplifications and limitations.
Prohibited: any use the N-ATLaS licence prohibits. The licence allows up to 1,000 active end-users without a separate agreement.
