"""Create or update the permanent AHỤIKE demo and live API: a Hugging Face Space on ZeroGPU (free GPU).

  hf auth login                           # once, with a WRITE token
  SPACE_HF_TOKEN=<read token> python scripts/deploy_space.py

Gradio Spaces, including ZeroGPU, need a Hugging Face paid plan (PRO) as of October 2026. Uploads space/ (zerogpu_app.py, requirements, Space card), the ahuike
package, app/app.py and the advice messages. The Space needs its own READ token as the secret HF_TOKEN (gated N-ATLaS
model and ASR models): pass it as SPACE_HF_TOKEN, or add it under the Space's Settings > Variables and secrets.
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADVICE = ["data/advice_reviewed.json", "data/advice_i18n.json"]
ZEROGPU = "zero-a10g"  # huggingface_hub's flavour name for ZeroGPU hardware
CPU_ONLY = ("space_app.py",)  # llama.cpp entry point for self-hosting; not used on ZeroGPU


def main() -> None:
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError

    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default=None, help="default <user>/AHUIKE")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")  # falls back to the token saved by `hf auth login`
    api = HfApi(token=token)
    user = api.whoami()["name"]
    space = args.space or f"{user}/AHUIKE"

    try:
        api.create_repo(space, repo_type="space", space_sdk="gradio", space_hardware=ZEROGPU,
                        private=False, exist_ok=True)
    except HfHubHTTPError as e:
        raise SystemExit(
            f"Could not create {space} on ZeroGPU ({e.response.status_code if e.response is not None else e}).\n"
            "Create it in the browser instead: huggingface.co/new-space -> SDK Gradio -> Hardware 'ZeroGPU',\n"
            f"name it '{space.split('/')[-1]}', then run this script again to upload the code.\n"
            "Gradio/ZeroGPU Spaces need a Hugging Face paid plan (PRO).")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copytree(ROOT / "space", tmp, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*CPU_ONLY))
        shutil.copytree(ROOT / "ahuike", tmp / "ahuike", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (tmp / "app").mkdir()
        shutil.copy(ROOT / "app" / "app.py", tmp / "app" / "app.py")
        (tmp / "data").mkdir()
        for f in ADVICE:
            src = ROOT / f
            if not src.exists():  # the machine-translated advice lives in the private work repo
                src = Path(hf_hub_download(f"{user}/lafiya-work", f, repo_type="dataset", token=token))
            shutil.copy(src, tmp / f)
        api.upload_folder(repo_id=space, repo_type="space", folder_path=str(tmp),
                          commit_message="Deploy AHỤIKE demo and API (ZeroGPU)")

    try:
        api.request_space_hardware(space, ZEROGPU)
    except HfHubHTTPError as e:
        print(f"Could not switch hardware automatically ({e}); choose ZeroGPU under Settings > Hardware.")
    if os.environ.get("SPACE_HF_TOKEN"):
        api.add_space_secret(space, "HF_TOKEN", os.environ["SPACE_HF_TOKEN"])
        print("Space secret HF_TOKEN set.")
    else:
        print(f"Add a READ token as secret HF_TOKEN at https://huggingface.co/spaces/{space}/settings")
    print(f"Space building at https://huggingface.co/spaces/{space} (first start ~10-20 min: 16 GB model download)")
    print(f"API documentation: https://huggingface.co/spaces/{space} -> 'Use via API' at the bottom of the page")


if __name__ == "__main__":
    main()
