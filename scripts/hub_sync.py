"""Move work files between Kaggle/Colab sessions through a PRIVATE Hugging Face dataset repo.

  python scripts/hub_sync.py push data results          # upload folders
  python scripts/hub_sync.py pull data results          # download folders
  python scripts/hub_sync.py push outputs/ahuike-lora   # adapters too

Repo defaults to <your-hf-username>/lafiya-work (private). Requires HF_TOKEN.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    from huggingface_hub import HfApi, snapshot_download

    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("push", "pull"))
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--repo", default=None)
    args = ap.parse_args()

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    repo = args.repo or f"{api.whoami()['name']}/lafiya-work"
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)

    for folder in args.folders:
        if args.action == "push":
            api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(ROOT / folder),
                              path_in_repo=folder, commit_message=f"sync {folder}",
                              ignore_patterns=["checkpoint-*", "checkpoint-*/**"])
            print(f"pushed {folder} -> {repo}")
        else:
            snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(ROOT),
                              allow_patterns=[f"{folder}/*", f"{folder}/**/*"], token=token)
            print(f"pulled {folder} <- {repo}")


if __name__ == "__main__":
    main()
