"""Create or update the permanent AHỤIKE demo: a free CPU Hugging Face Space.

  python scripts/deploy_space.py          # needs HF_TOKEN with write access

Uploads space/ (entry point, requirements, Space card), the ahuike package, app/app.py and the advice messages,
then points the Space at the GGUF made by scripts/export_gguf.py. The Space needs its own READ token as the secret
HF_TOKEN (gated N-ATLaS tokenizer and ASR models, private GGUF). Pass it as SPACE_HF_TOKEN, or add it by hand under
the Space's Settings > Variables and secrets.
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADVICE = ["data/advice_reviewed.json", "data/advice_i18n.json"]


def main() -> None:
    from huggingface_hub import HfApi, hf_hub_download

    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default=None, help="default <user>/AHUIKE")
    ap.add_argument("--gguf-repo", default=None, help="default <user>/AHUIKE-N-ATLaS-8B-GGUF-Powered-by-Awarri")
    args = ap.parse_args()

    api = HfApi(token=os.environ["HF_TOKEN"])
    user = api.whoami()["name"]
    space = args.space or f"{user}/AHUIKE"
    gguf_repo = args.gguf_repo or f"{user}/AHUIKE-N-ATLaS-8B-GGUF-Powered-by-Awarri"
    gguf = next(f for f in api.list_repo_files(gguf_repo) if f.endswith(".gguf"))

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copytree(ROOT / "space", tmp, dirs_exist_ok=True)
        shutil.copytree(ROOT / "ahuike", tmp / "ahuike", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (tmp / "app").mkdir()
        shutil.copy(ROOT / "app" / "app.py", tmp / "app" / "app.py")
        (tmp / "data").mkdir()
        for f in ADVICE:
            src = ROOT / f
            if not src.exists():  # the machine-translated advice lives in the private work repo
                src = Path(hf_hub_download(f"{user}/lafiya-work", f, repo_type="dataset",
                                           token=os.environ["HF_TOKEN"]))
            shutil.copy(src, tmp / f)
        api.create_repo(space, repo_type="space", space_sdk="gradio", private=False, exist_ok=True)
        api.upload_folder(repo_id=space, repo_type="space", folder_path=str(tmp),
                          commit_message="Deploy AHỤIKE demo")

    api.add_space_variable(space, "AHUIKE_GGUF", f"{gguf_repo}/{gguf}")
    if os.environ.get("SPACE_HF_TOKEN"):
        api.add_space_secret(space, "HF_TOKEN", os.environ["SPACE_HF_TOKEN"])
        print("Space secret HF_TOKEN set.")
    else:
        print(f"Add a READ token as secret HF_TOKEN at https://huggingface.co/spaces/{space}/settings")
    print(f"Space building at https://huggingface.co/spaces/{space} (first build ~15-20 min)")


if __name__ == "__main__":
    main()
