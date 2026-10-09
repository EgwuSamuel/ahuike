"""Publish the AHỤIKE adapter and NaijaTriage-Bench as PUBLIC Hugging Face repos.

  python scripts/release_hf.py            # needs HF_TOKEN with write access

Pulls the released adapter (outputs/ablation-lora: diverse cases, one language per case) and the
translated benchmark from the private work repo, then uploads:
  <user>/AHUIKE-N-ATLaS-8B-LoRA-Powered-by-Awarri   model card = docs/MODEL_CARD.md
  <user>/NaijaTriage-Bench                          dataset card = docs/DATASET_CARD.md
The model name carries the "Powered by Awarri" suffix, as the N-ATLaS licence requires for renamed derivatives.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = "outputs/ablation-lora"
DATA_FILES = ["data/cases_test.jsonl", "data/cases_train.jsonl", "data/eval_items.jsonl",
              "data/translation_stats.json", "data/advice_reviewed.json", "data/verified_ig.json",
              "results/preds.jsonl", "results/REPORT.md", "results/metrics.json"]


def main() -> None:
    from huggingface_hub import HfApi, snapshot_download

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-repo", default=None)
    ap.add_argument("--dataset-repo", default=None)
    args = ap.parse_args()

    api = HfApi(token=os.environ["HF_TOKEN"])
    user = api.whoami()["name"]
    model_repo = args.model_repo or f"{user}/AHUIKE-N-ATLaS-8B-LoRA-Powered-by-Awarri"
    dataset_repo = args.dataset_repo or f"{user}/NaijaTriage-Bench"

    # Fetch only what git does not hold. Pulling the whole data/ folder would overwrite the
    # clinician-relabelled cases in git with the older labels in the work repo.
    if not (ROOT / ADAPTER / "adapter_config.json").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts/hub_sync.py"), "pull", ADAPTER], check=True)
    work_only = [f for f in ("data/eval_items.jsonl", "data/translation_stats.json") if not (ROOT / f).exists()]
    if work_only:
        snapshot_download(repo_id=f"{user}/lafiya-work", repo_type="dataset", local_dir=str(ROOT),
                          allow_patterns=work_only, token=os.environ["HF_TOKEN"])

    api.create_repo(model_repo, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(repo_id=model_repo, folder_path=str(ROOT / ADAPTER), commit_message="AHỤIKE LoRA adapter",
                      ignore_patterns=["checkpoint-*", "checkpoint-*/**"])
    api.upload_file(repo_id=model_repo, path_or_fileobj=str(ROOT / "docs/MODEL_CARD.md"), path_in_repo="README.md")
    print(f"model   -> https://huggingface.co/{model_repo}")

    with tempfile.TemporaryDirectory() as tmp:
        for f in DATA_FILES:
            if (ROOT / f).exists():
                dst = Path(tmp) / f
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(ROOT / f, dst)
            else:
                print(f"  (skipped missing {f})")
        shutil.copy(ROOT / "docs/DATASET_CARD.md", Path(tmp) / "README.md")
        api.create_repo(dataset_repo, repo_type="dataset", private=False, exist_ok=True)
        api.upload_folder(repo_id=dataset_repo, repo_type="dataset", folder_path=tmp,
                          commit_message="NaijaTriage-Bench")
    print(f"dataset -> https://huggingface.co/datasets/{dataset_repo}")


if __name__ == "__main__":
    main()
