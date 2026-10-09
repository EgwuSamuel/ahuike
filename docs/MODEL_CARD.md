---
license: other
license_name: n-atlas
base_model: NCAIR1/N-ATLaS
library_name: peft
language: [en, ha, yo, ig]
tags: [medical, triage, maternal-health, child-health, nigeria, n-atlas, lora, powered-by-awarri]
---

# AHUIKE-N-ATLaS-8B-LoRA (Powered by Awarri)

*Ahụike nne na nwa*: health for mother and child.

Maternal and child **danger-sign triage** fine-tuned from **N-ATLaS-8B** for English, Hausa, Yorùbá and Igbo.
It returns one of `EMERGENCY_REFER_NOW`, `CLINIC_WITHIN_24H` or `HOME_CARE`, the patient group (child, pregnant,
postpartum) and the protocol danger signs that triggered it. **It never writes advice**: the app shows one of nine
advice messages per language that a clinician and native speakers reviewed, selected by triage level and patient group.

**Powered by Awarri.** This is a derivative of N-ATLaS and is distributed under the N-ATLaS licence (same terms).
Code, benchmark and full results: https://github.com/EgwuSamuel/ahuike

*N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation and Digital Economy, and powered by Awarri Technologies.*

## Results (NaijaTriage-Bench, 1,091 plain-text items in 4 languages)

Every system gets the same protocol system prompt. Gold labels follow the clinician-reviewed rules; this model is scored
as deployed, with the review guard. Brackets are 95% case-clustered bootstrap CIs.

| System | Accuracy | Missed emergencies ↓ | Danger-sign F1 | Consistent & correct in all 4 languages |
|---|---|---|---|---|
| Base N-ATLaS | 25.6 [21.0–30.2] | 97.3 [95.7–98.6] | 23.1 | 17.9 |
| Base + 3 worked examples | 50.7 [46.7–54.6] | 40.9 [35.3–46.6] | 30.3 | 20.0 |
| **This model** | **92.7** [90.3–94.8] | **3.3** [1.5–5.3] | **79.4** | **87.4** |

- vs base N-ATLaS: exact McNemar p = 2.5×10⁻¹⁷⁸ (all items), 3.9×10⁻¹⁸⁰ (emergency items).
- Missed emergencies on the same 95 cases in every language: English 1.9, Hausa 3.7, Yorùbá 7.4, Igbo 3.7 (base: 91–100).
- Igbo items verified as faithful by a native speaker (n = 14): 92.9% accuracy, 0 missed emergencies.
- Code-switched input (92 items): 83.7% accuracy, 12.5% missed emergencies (weakest condition).
- Over-triage (non-emergencies sent up a level): 5.1%. The protocol errs toward referral by design.

## Clinical review and the review guard
A medical doctor (MBBS) on the team reviewed the triage rules after training and made two stricter: a red or draining
umbilicus in a young infant, and any single pre-eclampsia sign, are now emergencies. This adapter was trained before that
review, so apply `ahuike.prompts.apply_review_guard` to its parsed output: if it names one of those signs, the answer is
raised to `EMERGENCY_REFER_NOW`. The guard never lowers urgency. The next training round will use the relabelled data.

## Training
- QLoRA (r=16, α=32, all attention + MLP projections), 1 epoch, lr 2e-4, Unsloth, Kaggle T4 GPUs.
- Data: protocol-generated cases from the NaijaTriage-Bench train split. Each case appears **once, in one language**
  (English, Hausa, Yorùbá or Igbo), plus code-switched variants. Labels come from a rule engine encoding WHO IMCI
  and the Nigerian CHEW Standing Orders.
- Hausa/Yorùbá/Igbo text translated by N-ATLaS and filtered by back-translation.
- We also trained a *parallel-anchored* variant of the same size (each case repeated in every language its translation
  passed). It had similar accuracy but missed more emergencies (5.8% vs 3.3% with the reviewed rules; 9.4% vs 5.7% before
  the review, case-level sign test p = 0.019), so this diverse-cases model is the release.

## Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("NCAIR1/N-ATLaS")
base = AutoModelForCausalLM.from_pretrained("NCAIR1/N-ATLaS", device_map="auto", torch_dtype="auto")
model = PeftModel.from_pretrained(base, "SamEgwu/AHUIKE-N-ATLaS-8B-LoRA-Powered-by-Awarri")
```

Use the system prompt, parser and guard in `ahuike/prompts.py` (`SYSTEM_PROMPT`, `build_messages`, `parse_output`,
`apply_review_guard`).
The model replies with one JSON object, for example
`{"triage": "EMERGENCY_REFER_NOW", "patient": "pregnant", "danger_signs": ["vaginal_bleeding"]}`.

## Intended use and limits
Decision support for community health workers and caregivers. **Not a diagnostic device.** It errs toward referral.
Trained and tested on synthetic cases; real-world clinical validation is still needed. N-ATLaS translations carry errors
that back-translation does not always catch (our Igbo reviewer found meaning errors in 19 of 40 sampled items).
See `docs/ETHICS.md` for protocol simplifications and limitations.
Prohibited: any use the N-ATLaS licence prohibits. The licence allows up to 1,000 active end-users without a separate agreement.
