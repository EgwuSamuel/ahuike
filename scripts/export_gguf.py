"""Merge the AHỤIKE adapter into N-ATLaS and export a Q4_K_M GGUF for CPU inference (llama.cpp).

  python scripts/export_gguf.py --adapter outputs/ahuike-lora --out /tmp/ahuike-gguf \
      --push <user>/AHUIKE-N-ATLaS-8B-GGUF-Powered-by-Awarri

Licence note: N-ATLaS derivatives must keep the N-ATLaS licence and carry the
"Powered by Awarri" suffix. The model card in docs/MODEL_CARD.md is uploaded as README.md.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", default="/tmp/ahuike-gguf")
    ap.add_argument("--quant", default="q4_k_m")
    ap.add_argument("--push", default=None, help="HF repo id for the GGUF (public model repo)")
    args = ap.parse_args()

    from unsloth import FastLanguageModel

    token = os.environ.get("HF_TOKEN")
    model, tok = FastLanguageModel.from_pretrained(model_name=args.adapter, max_seq_length=4096,
                                                   load_in_4bit=False, token=token)
    model.save_pretrained_gguf(args.out, tok, quantization_method=args.quant)
    print(f"GGUF written under {args.out}")
    if args.push:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(args.push, repo_type="model", private=False, exist_ok=True)
        card = ROOT / "docs" / "MODEL_CARD.md"
        if card.exists():
            shutil.copy(card, Path(args.out) / "README.md")
        for f in Path(args.out).iterdir():
            if f.suffix == ".gguf" or f.name == "README.md":
                api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name, repo_id=args.push)
        print(f"pushed to https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
