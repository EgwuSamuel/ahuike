"""Generate the three Kaggle notebooks (run once locally: python notebooks/build_notebooks.py).

These are the exact cells validated on Kaggle (2x T4, October 2026). For every notebook:
Accelerator = GPU T4 x2, Internet = On, Add-ons > Secrets: HF_TOKEN ticked FOR THAT NOTEBOOK.
Run long notebooks with Save Version > "Save & Run All (Commit)" so a dropped connection cannot stop them.
Work moves between notebooks through a private HF dataset repo (scripts/hub_sync.py).
"""
import json
from pathlib import Path

REPO = "https://github.com/EgwuSamuel/ahuike.git"

TOKEN = ("code", "import os\nfrom kaggle_secrets import UserSecretsClient\n"
                 "os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')")
CLONE = f"!git clone -q {REPO} /kaggle/working/ahuike || (cd /kaggle/working/ahuike && git pull -q)\n%cd /kaggle/working/ahuike"

# vLLM installs its own PyTorch; Kaggle's preinstalled torchaudio then breaks transformers imports.
VLLM = ("code", CLONE + "\n!pip install -q vllm\n!pip uninstall -y -q torchaudio\n!python scripts/check_setup.py")
UNSLOTH = ("code", CLONE + "\n!pip install -q unsloth\n!pip uninstall -y -q torchaudio\n!python scripts/check_setup.py")
SETTINGS = ("md", "Kaggle settings: **GPU T4 x2**, **Internet on**, Secret **HF_TOKEN** ticked for this notebook. "
                  "Run with **Save Version > Save & Run All (Commit)**.")

NOTEBOOKS = {
    "01_translate_and_baseline.ipynb": [
        ("md", "# AHỤIKE 01: N-ATLAS translation + baseline benchmark (vLLM, ~1.5-2 h)\n"
               "Translates NaijaTriage-Bench into Hausa/Yoruba/Igbo **with N-ATLAS** sentence by sentence, "
               "back-translates and filters, builds the fine-tuning sets, and scores **base N-ATLAS** "
               "(zero-shot and few-shot). Resumable: re-running continues where it stopped."),
        SETTINGS, TOKEN, VLLM,
        ("md", "## 1. Recover earlier work, translate, save"),
        ("code", "!python scripts/hub_sync.py pull data\n!python scripts/translate_natlas.py\n"
                 "!python scripts/hub_sync.py push data"),
        ("md", "## 2. Fine-tuning sets + baseline benchmark"),
        ("code", "!python scripts/build_sft.py\n!python scripts/run_eval.py --systems base base_fewshot\n"
                 "!python scripts/compute_metrics.py\n!python scripts/hub_sync.py push data results\n"
                 "!cat data/translation_stats.json\n!cat results/REPORT.md"),
    ],
    "02_finetune_unsloth.ipynb": [
        ("md", "# AHỤIKE 02: QLoRA fine-tuning of N-ATLaS-8B (Unsloth, ~3.5 h)\n"
               "Trains **AHỤIKE** (parallel-anchored) and the **ablation** (same size and language mix, "
               "each case in one language only). 1 epoch each, ~1.5 h on Kaggle T4s. Each adapter is "
               "pushed to the private work repo as soon as it finishes."),
        SETTINGS, TOKEN,
        ("code", UNSLOTH[1] + "\n!python scripts/hub_sync.py pull data\n!python scripts/build_sft.py"),
        ("md", "## 1. AHỤIKE (parallel-anchored)"),
        ("code", "!python scripts/finetune.py --data data/sft_anchored.jsonl --out outputs/ahuike-lora\n"
                 "!python scripts/hub_sync.py push outputs/ahuike-lora"),
        ("md", "## 2. Ablation (no parallel anchoring)"),
        ("code", "!python scripts/finetune.py --data data/sft_ablation.jsonl --out outputs/ablation-lora\n"
                 "!python scripts/hub_sync.py push outputs/ablation-lora"),
        ("md", "## 3. (Optional) merged Q4_K_M GGUF for the CPU demo\n"
               "The repo name must carry the **Powered-by-Awarri** suffix (N-ATLaS licence)."),
        ("code", "HF_USER = 'SamEgwu'\n"
                 "!python scripts/export_gguf.py --adapter outputs/ahuike-lora --out /tmp/ahuike-gguf "
                 "--push {HF_USER}/AHUIKE-N-ATLaS-8B-GGUF-Powered-by-Awarri"),
    ],
    "03_eval_and_demo.ipynb": [
        ("md", "# AHỤIKE 03: benchmark AHỤIKE vs base N-ATLAS (vLLM, ~30-45 min), then live demo"),
        SETTINGS, TOKEN, VLLM,
        ("code", "!python scripts/hub_sync.py pull data results outputs/ahuike-lora outputs/ablation-lora\n"
                 "!python scripts/build_sft.py"),
        ("md", "## Benchmark (base is re-run automatically if the prompt changed)"),
        ("code", "!python scripts/run_eval.py --systems base base_fewshot ahuike=outputs/ahuike-lora "
                 "ablation=outputs/ablation-lora\n"
                 "!python scripts/compute_metrics.py\n!python scripts/hub_sync.py push results\n"
                 "!cat results/REPORT.md"),
        ("md", "## Live demo (restart the session first to free GPU memory, re-run the setup cells)\n"
               "Opens a public Gradio link: use it for the video and for live user sessions."),
        ("code", "!pip install -q gradio\n!python app/app.py --backend hf --adapter outputs/ahuike-lora --share"),
        ("md", "## Voice evaluation (after uploading recordings to voice/ and filling voice/manifest.csv)"),
        ("code", "!python scripts/eval_voice.py --backend hf --adapter outputs/ahuike-lora"),
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
        (here / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        print("wrote", name)


if __name__ == "__main__":
    main()
