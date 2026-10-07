# N-ATLAS integration evidence

LAFIYA uses N-ATLAS at every stage. No other foundation model is called at runtime, for translation, or for evaluation.

| # | N-ATLAS component | How LAFIYA uses it | Code |
|---|---|---|---|
| 1 | **N-ATLaS-8B LLM** (`NCAIR1/N-ATLaS`) as **translator** | Translates every English vignette into Hausa, Yorùbá and Igbo, plus code-switched variants | `scripts/translate_natlas.py` → `to_lang_prompt()` |
| 2 | N-ATLaS-8B as **back-translator / quality gate** | Translates each output back to English; the item is kept only if every clinical finding and number survives | `scripts/translate_natlas.py` → `to_english_prompt()`, `lafiya/backcheck.py` |
| 3 | N-ATLaS-8B as **baseline** | Zero-shot and few-shot triage with the protocol system prompt: the reference that LAFIYA is measured against | `scripts/run_eval.py --systems base base_fewshot` |
| 4 | N-ATLaS-8B **fine-tuned** (QLoRA) | LAFIYA adapter trained on parallel-anchored protocol data | `scripts/finetune.py` |
| 5 | **N-ATLaS derivative release** | Merged Q4_K_M GGUF published as `LAFIYA-N-ATLaS-8B-GGUF-Powered-by-Awarri`, under the N-ATLaS licence | `scripts/export_gguf.py` |
| 6 | **N-ATLAS ASR** Hausa (`NCAIR1/Hausa-ASR`) | Voice input for Hausa speakers | `lafiya/pipeline.py` → `ASR_MODELS["ha"]` |
| 7 | **N-ATLAS ASR** Yorùbá (`NCAIR1/Yoruba-ASR`) | Voice input for Yorùbá speakers | `ASR_MODELS["yo"]` |
| 8 | **N-ATLAS ASR** Igbo (`NCAIR1/Igbo-ASR`) | Voice input for Igbo speakers | `ASR_MODELS["ig"]` |
| 9 | **N-ATLAS ASR** Nigerian-accented English (`NCAIR1/NigerianAccentedEnglish`) | Voice input in English | `ASR_MODELS["en"]` |

## Chat template

Confirmed against the N-ATLaS model card: Llama-3 headers with a system block that begins
`Cutting Knowledge Date: December 2023 / Today Date: <date_string>`.
All prompts are rendered with the N-ATLaS tokenizer's own chat template (`apply_chat_template(..., date_string=...)`),
with a fixed `date_string` so training and inference see identical headers (`lafiya/inference.py: DATE_STRING`).
Training computes loss only on assistant tokens (`train_on_responses_only`).

## What fine-tuning adds over the base model

The base model gets the full protocol in its system prompt. LAFIYA gets the *same* system prompt.
The benchmark difference therefore measures what fine-tuning taught the model, not prompt engineering.

## Evidence artefacts produced by the pipeline

- `data/translation_stats.json`: N-ATLAS translation retention rate per language (back-translation pass rate)
- `results/preds.jsonl`: every raw N-ATLaS / LAFIYA output
- `results/REPORT.md`, `results/metrics.json`: benchmark vs base N-ATLAS
- `outputs/*/train_stats.json`: training loss curve and runtime
- `results/voice_eval.json`: ASR word error rate + end-to-end voice triage accuracy
