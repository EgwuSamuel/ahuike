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

Every system gets the same protocol system prompt. Brackets are 95% case-clustered bootstrap CIs.

| System | Accuracy | Missed emergencies ↓ | Danger-sign F1 | Consistent & correct in all 4 languages |
|---|---|---|---|---|
| Base N-ATLaS | 37.2 [31.9–42.1] | 96.5 [94.5–98.2] | 23.0 | 30.5 |
| Base + 3 worked examples | 53.8 [49.8–57.8] | 34.6 [28.8–40.7] | 30.1 | 27.4 |
| **This model** | **90.9** [88.3–93.2] | **5.7** [2.8–9.2] | **87.1** | **81.1** |

- vs base N-ATLaS: exact McNemar p = 7.5×10⁻¹²⁷ (all items), 4.3×10⁻¹³⁷ (emergency items).
- Missed emergencies on the same 95 cases in every language: English 9.5, Hausa 4.8, Yorùbá 7.1, Igbo 7.1 (base: 88–100).
- Igbo items verified as faithful by a native speaker (n = 14): 92.9% accuracy, 0 missed emergencies.
- Code-switched input (92 items): 81.5% accuracy, 4.8% missed emergencies.
- Over-triage (non-emergencies sent up a level): 5.7%. The protocol errs toward referral by design.

## Training
- QLoRA (r=16, α=32, all attention + MLP projections), 1 epoch, lr 2e-4, Unsloth, Kaggle T4 GPUs.
- Data: protocol-generated cases from the NaijaTriage-Bench train split. Each case appears **once, in one language**
  (English, Hausa, Yorùbá or Igbo), plus code-switched variants. Labels come from a rule engine encoding WHO IMCI
  and the Nigerian CHEW Standing Orders.
- Hausa/Yorùbá/Igbo text translated by N-ATLaS and filtered by back-translation.
- We also trained a *parallel-anchored* variant of the same size (each case repeated in every language its translation
  passed). Accuracy was the same, but it missed more emergencies (9.4% vs 5.7%; case-level sign test p = 0.019), so this
  diverse-cases model is the release.

## Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("NCAIR1/N-ATLaS")
base = AutoModelForCausalLM.from_pretrained("NCAIR1/N-ATLaS", device_map="auto", torch_dtype="auto")
model = PeftModel.from_pretrained(base, "SamEgwu/AHUIKE-N-ATLaS-8B-LoRA-Powered-by-Awarri")
```

Use the system prompt and output parser in `ahuike/prompts.py` (`SYSTEM_PROMPT`, `build_messages`, `parse_output`).
The model replies with one JSON object, for example
`{"triage": "EMERGENCY_REFER_NOW", "patient": "pregnant", "danger_signs": ["vaginal_bleeding"]}`.

## Intended use and limits
Decision support for community health workers and caregivers. **Not a diagnostic device.** It errs toward referral.
Trained and tested on synthetic cases; real-world clinical validation is still needed. N-ATLaS translations carry errors
that back-translation does not always catch (our Igbo reviewer found meaning errors in 19 of 40 sampled items).
See `docs/ETHICS.md` for protocol simplifications and limitations.
Prohibited: any use the N-ATLaS licence prohibits. The licence allows up to 1,000 active end-users without a separate agreement.
