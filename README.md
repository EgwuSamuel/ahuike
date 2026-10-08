# AHỤIKE — Same patient, four languages, one answer

*Ahụike nne na nwa*: health for mother and child. (*Ahụike*, pronounced ah-HOO-ee-keh, is Igbo for "health".)

**Cross-lingual maternal & child danger-sign triage on N-ATLAS, with Igbo as the in-depth human-validated case study.**
National AI Innovation Challenge 2026 · Academia & Research track · Problem Statement 03 (Sectoral Fine-Tuning: Health)

> *Your language should never change your diagnosis.*

A mother in Kano, Ibadan or Enugu should get the same advice for the same danger sign whether she
describes it in Hausa, Yorùbá, Igbo or Nigerian English. AHỤIKE measures whether N-ATLAS does this,
and fine-tunes it so it does.

**Why this matters now.** The N-ATLaS model card's own human evaluation shows uneven quality across languages
(average 4.21/5 in English, 3.98 in Hausa, 3.87 in Igbo, 2.69 in Yorùbá). For general chat that is a quality gap.
For a danger sign in pregnancy it is a safety gap. AHỤIKE turns that gap into a measured, clinical number
(CLCC, under-triage per language) and shows how much fine-tuning closes it.

**How AHỤIKE differs from a health chatbot on N-ATLaS.** It doesn't wrap the base model in a prompt. It
(1) fine-tunes N-ATLaS on protocol-grounded data, (2) proves the gain on a 1,183-item benchmark with
significance tests against base N-ATLaS, and (3) reports safety per language, not only on average.

**Headline.** With the full protocol in its prompt, base N-ATLaS sends only 3.5% of emergencies to hospital (it misses 96.5%).
AHỤIKE misses **5.7%**, with **90.9%** triage accuracy, and **0** missed emergencies on the Igbo items a native speaker verified.

| Links | |
|---|---|
| Model (LoRA on N-ATLaS-8B) | [SamEgwu/AHUIKE-N-ATLaS-8B-LoRA-Powered-by-Awarri](https://huggingface.co/SamEgwu/AHUIKE-N-ATLaS-8B-LoRA-Powered-by-Awarri) |
| Benchmark | [SamEgwu/NaijaTriage-Bench](https://huggingface.co/datasets/SamEgwu/NaijaTriage-Bench) |
| Full results | [`results/REPORT.md`](results/REPORT.md) · raw outputs [`results/preds.jsonl`](results/preds.jsonl) |

## What's novel

| Contribution | What it is |
|---|---|
| **Protocol-compiled data** | WHO IMCI and Nigerian CHEW Standing Orders encoded as a rule engine ([`ahuike/protocol/rules.py`](ahuike/protocol/rules.py)). Cases are *generated from* the rules, so every gold label is correct by construction and auditable. |
| **NaijaTriage-Bench** | 385 held-out clinical cases in English, translated by N-ATLaS and kept only where the meaning survived: 1,091 items (en 385, ha 341, ig 227, yo 138) + 92 code-switched. A 95-case *parallel core* exists in all four languages for fair language comparison. Train/test split by *clinical combination* so no test picture is seen in training. |
| **CLCC metric** | *Cross-Lingual Clinical Consistency*: share of cases triaged identically in all four languages, plus the worst-language gap. Linguistic equity as a number. |
| **Data-design experiment** | Two equal-size fine-tunes: each case repeated across languages (parallel anchoring) vs more distinct cases, each in one language. Diverse cases were significantly safer, so they became AHỤIKE. |
| **Native-speaker validation** | An Igbo speaker corrected the advice messages, confirmed the symptom glossary and judged 40 N-ATLaS translations, catching meaning errors the automatic back-check missed. |
| **Full N-ATLAS stack** | N-ATLAS LLM (translation, baseline, fine-tuned model) + all four N-ATLAS ASR models (voice input). |

## Results

All 1,091 faithful plain-text items; brackets are 95% case-clustered bootstrap CIs. Every system gets the same protocol system prompt.

| System | Accuracy | **Missed emergencies** ↓ | Over-triage | Danger-sign F1 | Consistent & correct in all 4 languages |
|---|---|---|---|---|---|
| Base N-ATLaS | 37.2 [31.9–42.1] | **96.5** [94.5–98.2] | 37.4 | 23.0 | 30.5 |
| Base + 3 worked examples | 53.8 [49.8–57.8] | **34.6** [28.8–40.7] | 53.6 | 30.1 | 27.4 |
| Parallel-anchored variant | 90.8 [88.4–92.9] | **9.4** [5.8–13.7] | 3.2 | 84.6 | 80.0 |
| **AHỤIKE** (released) | **90.9** [88.3–93.2] | **5.7** [2.8–9.2] | 5.7 | **87.1** | **81.1** |

AHỤIKE vs base N-ATLaS: exact McNemar p = 7.5×10⁻¹²⁷ (all items), 4.3×10⁻¹³⁷ (emergencies).

**Missed emergencies by language, same 95 cases in every language (parallel core)**

| System | English | Hausa | Yorùbá | Igbo |
|---|---|---|---|---|
| Base N-ATLaS | 88.1 | 95.2 | 100 | 100 |
| Base + 3 worked examples | 35.7 | 23.8 | **57.1** | 50.0 |
| **AHỤIKE** | 9.5 | 4.8 | **7.1** | 7.1 |

Even with worked examples, base N-ATLaS misses a Yorùbá-speaking mother's emergency more than twice as often as a Hausa speaker's.
After fine-tuning, the gap is gone: an emergency is missed in at least one language for 14.3% of cases, vs 100% for base N-ATLaS.

**Human-validated subset.** On the 14 Igbo benchmark items our native speaker marked as faithful translations, AHỤIKE is
92.9% accurate with **no missed emergencies** (base N-ATLaS: 21.4%, all emergencies missed).

**Code-switched input** (Hausa/Yorùbá/Igbo mixed with English, 92 items): AHỤIKE 81.5% accuracy, 4.8% missed emergencies.

### Finding: more distinct cases beat repeating each case across languages

We trained two models with the same number of examples and the same language mix. The *parallel-anchored* model sees each
case in every language its translation passed (English plus one to three Nigerian languages); AHỤIKE sees each case once,
in one language, so it learns from several times as many distinct clinical pictures. Accuracy is the same
(McNemar p = 0.92), but AHỤIKE misses fewer emergencies: 26 vs 43 of 457. That holds when each case's four language
versions count as one unit: AHỤIKE is safer on 15 emergency cases, the variant on 4 (exact sign test p = 0.019; gap
4.0 points, 95% CI 1.4–7.1). The cost is a little more over-referral (5.7% vs 3.2%), the right trade for a danger-sign tool.
For anyone fine-tuning N-ATLaS on a limited translation budget: **spend it on variety**. Cross-language consistency
(CLCC 81.1 for both) comes from multilingual protocol training, not from the parallel structure.

### Finding: back-translation checks are not enough
Our Igbo reviewer judged 40 N-ATLaS translations that had passed the automatic back-check: 17 same meaning, 4 small
differences, 19 wrong. Errors included *wife* → "husband", *grandson* → "son", a wrong word for *nipple*, and English words
left untranslated. Native review is needed for clinical text; the verified subset is reported separately above.

Full tables (per population, reasoning type, confusion matrices): [`results/REPORT.md`](results/REPORT.md).

## Pipeline

```
rules.py ─► generate_cases.py ─► English vignettes (Nigerian lay terms, distractors)
                                       │
         translate_natlas.py  (N-ATLAS → ha / yo / ig, + code-switched)
                                       │
         back-translation filter (N-ATLAS → en; keep only if every finding + number survives)
                                       │
         build_sft.py ─► finetune.py (Unsloth QLoRA on N-ATLaS-8B) ─► LoRA adapter ─► GGUF
                                       │
         run_eval.py (base, base few-shot, AHỤIKE, parallel variant) ─► compute_metrics.py ─► REPORT.md
                                       │
         app/app.py: mic ─► N-ATLAS ASR (ha/yo/ig/en) ─► AHỤIKE ─► triage card
```

## Quickstart

CPU-only parts (data, tests, metrics):

```bash
python -m unittest discover -s tests
python scripts/generate_cases.py
```

GPU parts run on Kaggle (free 2×T4). Build the notebooks with `python notebooks/build_notebooks.py`, then run, in order:

1. `notebooks/01_translate_and_baseline.ipynb`: N-ATLAS translation, back-translation filter, baseline benchmark
2. `notebooks/02_finetune_unsloth.ipynb`: AHỤIKE + parallel-variant fine-tunes, GGUF export
3. `notebooks/03_eval_and_demo.ipynb`: benchmark, live demo, voice evaluation
4. `notebooks/04_release_and_demo.ipynb`: publish the model and benchmark on Hugging Face, live demo link

In `results/preds.jsonl`, `ahuike` is the released model (adapter folder `outputs/ablation-lora`, trained on
`sft_ablation.jsonl`) and `ahuike_parallel` is the parallel-anchored variant (`outputs/ahuike-lora`). The folder
names date from before the results showed which design was safer.

You need: Hugging Face access to `NCAIR1/N-ATLaS` and the four `NCAIR1/*-ASR` models, and an `HF_TOKEN` secret.

## Repository map

```
ahuike/protocol/   findings catalogue + rule engine (gold labels)
ahuike/cases.py    case sampler + Nigerian-English renderer
ahuike/prompts.py  system prompt, targets, output parser (shared by train/eval/app)
ahuike/metrics.py  accuracy, under-triage, danger-sign F1, CLCC, bootstrap, McNemar
ahuike/backcheck.py back-translation fidelity filter
ahuike/inference.py vLLM / transformers chat engine (base + LoRA adapters)
ahuike/pipeline.py ASR → LLM end-to-end pipeline
scripts/           generate, translate, build_sft, finetune, run_eval, compute_metrics, eval_voice, export_gguf, release_hf
app/app.py         Gradio voice demo
docs/              architecture, N-ATLAS integration, model & dataset cards, ethics, consent form
```

## Safety

AHỤIKE is **decision support for community health workers and caregivers, not a diagnosis**.
The protocol always errs toward referral. No real patient data is used. See [`docs/ETHICS.md`](docs/ETHICS.md).
The model never writes advice: the app shows one of nine reviewed messages per language. Where a native speaker has not yet
approved a message (two of the nine Igbo messages), the app shows the English message rather than unreviewed Igbo.

## Licence & attribution

Code: Apache-2.0. Model derivatives of N-ATLaS are released under the N-ATLaS licence (v1.0, Sept 2025) and,
because they are renamed, carry the **"Powered by Awarri"** suffix, as that licence requires.

*N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation and Digital Economy, and powered by Awarri Technologies.*
