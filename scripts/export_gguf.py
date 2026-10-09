"""Merge the AHỤIKE adapter into N-ATLaS and export a Q4_K_M GGUF for CPU inference (llama.cpp).

  python scripts/export_gguf.py --adapter outputs/ablation-lora \
      --push <user>/AHUIKE-N-ATLaS-8B-GGUF-Powered-by-Awarri

Runs on CPU in about 40 minutes (Kaggle: ~30 GB RAM, ~40 GB free disk under --work). No GPU needed:
the base model is merged in bf16 on CPU, converted with llama.cpp and quantised to Q4_K_M (~4.9 GB).

Licence note: N-ATLaS derivatives must keep the N-ATLaS licence and carry the "Powered by Awarri" suffix.
The base model is gated, so the merged GGUF is pushed PRIVATE by default (the demo Space reads it with a token).
"""
from __future__ import annotations

import argparse
import gc
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LLAMA_CPP = "https://github.com/ggml-org/llama.cpp"
GGUF_NAME = "AHUIKE-N-ATLaS-8B-Q4_K_M.gguf"


def run(cmd: list) -> None:
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--work", default="/tmp/ahuike-gguf")
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--push", default=None, help="HF model repo id for the GGUF")
    ap.add_argument("--public", action="store_true", help="make the GGUF repo public (default private)")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ahuike.inference import BASE_MODEL

    token = os.environ.get("HF_TOKEN")
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    print(f"free disk under {work}: {shutil.disk_usage(work).free / 1e9:.0f} GB", flush=True)
    merged, f16, out = work / "merged", work / "ahuike-f16.gguf", work / GGUF_NAME

    if not out.exists():
        # 1. Merge the LoRA adapter into the bf16 base model on CPU.
        if not f16.exists():
            base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cpu",
                                                        low_cpu_mem_usage=True, token=token)
            model = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
            model.save_pretrained(merged, safe_serialization=True, max_shard_size="4GB")
            AutoTokenizer.from_pretrained(BASE_MODEL, token=token).save_pretrained(merged)
            del model, base
            gc.collect()

        # 2. Convert to a 16-bit GGUF, then quantise with llama.cpp.
        lc = work / "llama.cpp"
        if not lc.exists():
            run(["git", "clone", "-q", "--depth", "1", LLAMA_CPP, lc])
        run([sys.executable, "-m", "pip", "install", "-q", lc / "gguf-py", "sentencepiece", "protobuf"])
        if not f16.exists():
            run([sys.executable, lc / "convert_hf_to_gguf.py", merged, "--outtype", "f16", "--outfile", f16])
        shutil.rmtree(merged, ignore_errors=True)
        run(["cmake", "-S", lc, "-B", lc / "build", "-DGGML_CUDA=OFF", "-DLLAMA_CURL=OFF"])
        run(["cmake", "--build", lc / "build", "--target", "llama-quantize", "-j", os.cpu_count() or 4])
        run([lc / "build" / "bin" / "llama-quantize", f16, out, args.quant])
        f16.unlink()
    print(f"GGUF: {out} ({out.stat().st_size / 1e9:.1f} GB)", flush=True)

    if args.push:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(args.push, repo_type="model", private=not args.public, exist_ok=True)
        api.upload_file(path_or_fileobj=str(out), path_in_repo=out.name, repo_id=args.push)
        api.upload_file(path_or_fileobj=str(ROOT / "docs" / "MODEL_CARD.md"), path_in_repo="README.md",
                        repo_id=args.push)
        print(f"pushed to https://huggingface.co/{args.push} ({'public' if args.public else 'private'})")


if __name__ == "__main__":
    main()
