# Ethics, safety and limitations

## Intended use
Decision support for **community health extension workers (CHEWs), nurses and caregivers** to recognise
maternal and child danger signs and **refer early**. It is not a diagnostic device and does not prescribe medicine.

## Safety design
- **Errs toward referral.** The system prompt instructs the model to choose the more urgent level when unsure, and
  the headline metric is the **under-triage rate** (missed emergencies), not accuracy.
- **Unparseable output fails safe.** The app shows "When in doubt, go to the health centre today."
- **Closed vocabulary.** Danger signs are protocol ids, so a health worker can audit why a case was escalated.
- **No real patient data.** All training and test cases are synthetic, generated from protocol rules.
  Voice recordings are made by consenting adult volunteers reading synthetic cases (docs/CONSENT_FORM.md).

## Protocol simplifications (to be reviewed by the faculty supervisor / a clinician)
The rule engine (`ahuike/protocol/rules.py`) is a **simplified triage encoding** of WHO IMCI (2014 chart booklet),
WHO young-infant (PSBI) signs and the maternal danger signs in Nigeria's National Standing Orders for CHEWs. Simplifications:
- Three triage levels only (refer now / clinic within 24 h / home care). Treatments are not modelled.
- Every child with fever goes to the clinic (Nigeria is malaria-endemic: test before treating).
- Malnutrition is reduced to bilateral oedema → clinic. MUAC and complicated SAM are not modelled.
- Fast-breathing thresholds follow IMCI: ≥60 (<2 months), ≥50 (2–11 months), ≥40 (12–59 months).
- Pre-eclampsia is approximated as two or more of severe headache, blurred vision, face/hand swelling (blood pressure isn't available in the community).

## Known limitations
- **Synthetic language.** Hausa/Yorùbá/Igbo text is N-ATLAS-translated, then filtered by back-translation.
  The filter is coarse (it can't tell "slow" from "very slow" skin pinch). A native-speaker-verified subset is reported separately.
- **Circularity.** N-ATLAS translates the cases it is later tested on. The split by clinical combination and the human-verified subset mitigate this.
- **Language ID** for advice is a heuristic (stop-words + diacritics).
- **ASR limits** (from the N-ATLaS model card): reduced accuracy for children's speech, noisy environments and code-switching.
- **Licence cap.** N-ATLaS allows up to 1,000 active end-users without a separate agreement.

## Prohibited uses
Surveillance, profiling, or any use the N-ATLaS licence prohibits. AHỤIKE must not replace clinical assessment.
