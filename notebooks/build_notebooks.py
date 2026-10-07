"""Generate the three Kaggle notebooks (run once locally: python notebooks/build_notebooks.py).

Kaggle settings for every notebook: Accelerator = GPU T4 x2, Internet = On,
Add-ons > Secrets: HF_TOKEN (your Hugging Face read+write token, after N-ATLaS access is approved).
"""
import json
from pathlib import Path

REPO = "https://github.com/EgwuSamuel/lafiya.git"

SETUP = [
    ("md", "Kaggle settings: **GPU T4 x2**, **Internet on**, Secret **HF_TOKEN** attached."),
    ("code", "import os\nfrom kaggle_secrets import UserSecretsClient\n"
             "os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')\n"
             "!nvidia-smi --query-gpu=name,memory.total --format=csv"),
    ("code", f"!git clone -q {REPO} lafiya || (cd lafiya && git pull -q)\n%cd lafiya"),
    ("md", "Pre-flight check: token, gated access to N-ATLaS + 4 ASR models, GPUs, chat template. "
           "Everything should say PASS before you continue."),
    ("code", "!python scripts/check_setup.py"),
]

NOTEBOOKS = {
    "01_translate_and_baseline.ipynb": [
        ("md", "# LAFIYA 01 — N-ATLAS translation + baseline benchmark\n"
               "Translates NaijaTriage-Bench into Hausa/Yoruba/Igbo **with N-ATLAS**, back-translates and "
               "filters, builds the fine-tuning sets, and scores **base N-ATLAS** (zero-shot and few-shot). "
               "Every step is resumable: if the session dies, re-run the cells."),
        *SETUP,
        ("code", "!pip install -q vllm\n# --backend auto tries vLLM and falls back to transformers automatically.\n"
                 "# If the fallback runs out of GPU memory, restart the session and use --backend hf."),
        ("md", "## 1. Test set first (so the baseline can start early)"),
        ("code", "!python scripts/translate_natlas.py --backend auto --test-only"),
        ("md", "## 2. Training translations (anchored set, ablation set, code-switched)"),
        ("code", "!python scripts/translate_natlas.py --backend auto"),
        ("code", "!python scripts/build_sft.py\n!cat data/translation_stats.json | head -60"),
        ("md", "## 3. Baseline: base N-ATLAS zero-shot and few-shot"),
        ("code", "!python scripts/run_eval.py --backend auto --systems base base_fewshot"),
        ("code", "!python scripts/compute_metrics.py"),
        ("md", "## 4. Save everything for the fine-tuning notebook"),
        ("code", "!python scripts/hub_sync.py push data results"),
    ],
    "02_finetune_unsloth.ipynb": [
        ("md", "# LAFIYA 02 — QLoRA fine-tuning of N-ATLaS-8B (Unsloth)\n"
               "Trains **LAFIYA** (parallel-anchored) and the **ablation** (same size, no parallel anchoring). "
               "Each run takes about 1.5–3 h on one T4. Checkpoints every 50 steps: if the session dies, "
               "re-run with `--resume`."),
        *SETUP,
        ("code", "!pip install -q unsloth"),
        ("code", "!python scripts/hub_sync.py pull data"),
        ("md", "## 1. LAFIYA (parallel-anchored)"),
        ("code", "!python scripts/finetune.py --data data/sft_anchored.jsonl --out outputs/lafiya-lora --epochs 2"),
        ("code", "!python scripts/hub_sync.py push outputs/lafiya-lora"),
        ("md", "## 2. Ablation (no parallel anchoring) — run if time allows"),
        ("code", "!python scripts/finetune.py --data data/sft_ablation.jsonl --out outputs/ablation-lora --epochs 2"),
        ("code", "!python scripts/hub_sync.py push outputs/ablation-lora"),
        ("md", "## 3. Export merged Q4_K_M GGUF for the CPU demo (HF Space)\n"
               "Repo name must carry the **Powered-by-Awarri** suffix (N-ATLaS licence)."),
        ("code", "HF_USER = 'YOUR_HF_USERNAME'\n"
                 "!python scripts/export_gguf.py --adapter outputs/lafiya-lora --out /tmp/lafiya-gguf "
                 "--push {HF_USER}/LAFIYA-N-ATLaS-8B-GGUF-Powered-by-Awarri"),
    ],
    "03_eval_and_demo.ipynb": [
        ("md", "# LAFIYA 03 — Benchmark LAFIYA vs base N-ATLAS, then live demo"),
        *SETUP,
        ("code", "!pip install -q vllm"),
        ("code", "!python scripts/hub_sync.py pull data results outputs/lafiya-lora outputs/ablation-lora"),
        ("code", "!python scripts/run_eval.py --backend auto --systems lafiya=outputs/lafiya-lora "
                 "ablation=outputs/ablation-lora"),
        ("code", "!python scripts/compute_metrics.py\n!python scripts/hub_sync.py push results"),
        ("md", "## Live demo (restart the session first to free GPU memory from vLLM, re-run setup cells)\n"
               "Opens a public Gradio link: use it for the video and for live user sessions."),
        ("code", "!pip install -q gradio peft\n"
                 "!python scripts/hub_sync.py pull data outputs/lafiya-lora\n"
                 "!python app/app.py --backend hf --adapter outputs/lafiya-lora --share"),
        ("md", "## Voice evaluation (after uploading recordings to voice/ and filling voice/manifest.csv)"),
        ("code", "!python scripts/eval_voice.py --backend hf --adapter outputs/lafiya-lora"),
    ],
}


def cell(kind, src):
    lines = src.split("\n")
    source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": source}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": source}


def main():
    here = Path(__file__).parent
    for name, cells in NOTEBOOKS.items():
        nb = {"cells": [cell(k, s) for k, s in cells],
              "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                           "language_info": {"name": "python"}},
              "nbformat": 4, "nbformat_minor": 5}
        (here / name).write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print("wrote", name)


if __name__ == "__main__":
    main()
