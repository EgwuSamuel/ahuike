# Ethics, safety and limitations

## Intended use
Decision support for **community health extension workers (CHEWs), nurses and caregivers** to recognise
maternal and child danger signs and **refer early**. It is not a diagnostic device and does not prescribe medicine.

## Safety design
- **Errs toward referral.** The system prompt instructs the model to choose the more urgent level when unsure, and
  the headline metric is the **under-triage rate** (missed emergencies), not accuracy.
- **Advice is never generated.** The model outputs only a triage level, patient group and danger signs; the advice
  shown is one of nine fixed messages per language, reviewed by a clinician and native speakers.
- **Unparseable output fails safe.** The app shows "When in doubt, go to the health centre today."
- **Closed vocabulary.** Danger signs are protocol ids, so a health worker can audit why a case was escalated.
- **No real patient data.** All training and test cases are synthetic, generated from protocol rules.
  Voice recordings are made by consenting adult volunteers reading synthetic cases (docs/CONSENT_FORM.md).

## Clinical review (8 October 2026)
A medical doctor (MBBS) on our team reviewed every triage rule, six judgement calls and the nine advice messages, and
signed off the rules "as a reasonable simplified encoding … for use as decision support that errs toward referral".
- **Agreed:** 55 of 57 rules and all 9 advice messages.
- **Changed:** a red or draining umbilicus in a young infant is now **refer now** (sepsis risk), not clinic.
- **Changed:** a **single** pre-eclampsia sign (severe headache, blurred vision or face/hand swelling) is now
  **refer now**, because blood pressure cannot be measured in the community.
- **Kept, with a note:** skin pustules stay at clinic; the reviewer noted urgency depends on severity, which the
  cases do not yet encode (IMCI refers many or severe pustules).
- **Confirmed:** any fever in a child, reduced fetal movement, fever alone in pregnancy and postpartum low mood go
  to clinic within 24 h; bilateral foot swelling in a child goes to clinic.
- **Next version:** thoughts of self-harm or of harming the baby after delivery must be an emergency sign. The current
  protocol does not model them.

The released model was trained before this review. The app applies the two changes with a deterministic guard
(`ahuike.prompts.apply_review_guard`): if the model names one of those signs, the answer is raised to emergency.
The guard never lowers urgency. The benchmark is relabelled with the reviewed rules and scores the app as deployed.

## Protocol simplifications
The rule engine (`ahuike/protocol/rules.py`) is a **simplified triage encoding** of WHO IMCI (2014 chart booklet),
WHO young-infant (PSBI) signs and the maternal danger signs in Nigeria's National Standing Orders for CHEWs. Simplifications:
- Three triage levels only (refer now / clinic within 24 h / home care). Treatments are not modelled.
- Every child with fever goes to the clinic (Nigeria is malaria-endemic: test before treating).
- Malnutrition is reduced to bilateral oedema → clinic. MUAC and complicated SAM are not modelled.
- Fast-breathing thresholds follow IMCI: ≥60 (<2 months), ≥50 (2–11 months), ≥40 (12–59 months).
- Any one pre-eclampsia sign is an emergency (blood pressure isn't available in the community).

## Known limitations
- **Synthetic language.** Hausa/Yorùbá/Igbo text is N-ATLAS-translated, then filtered by back-translation.
  The filter is coarse (it can't tell "slow" from "very slow" skin pinch). A native-speaker-verified subset is reported separately.
- **Circularity.** N-ATLAS translates the cases it is later tested on. The split by clinical combination and the human-verified subset mitigate this.
- **Language ID** for advice is a heuristic (stop-words + diacritics).
- **ASR limits** (from the N-ATLaS model card): reduced accuracy for children's speech, noisy environments and code-switching.
- **Licence cap.** N-ATLaS allows up to 1,000 active end-users without a separate agreement.

## Prohibited uses
Surveillance, profiling, or any use the N-ATLaS licence prohibits. AHỤIKE must not replace clinical assessment.
