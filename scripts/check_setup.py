"""Pre-flight check (about 1 minute, no model weights downloaded).

  python scripts/check_setup.py
Verifies: HF token, gated access to N-ATLaS and the four ASR models, GPUs, and that the
N-ATLaS chat template renders the Llama-3 headers that training and evaluation rely on.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(ROOT))

from ahuike.inference import BASE_MODEL, DATE_STRING  # noqa: E402
from ahuike.pipeline import ASR_MODELS  # noqa: E402
from ahuike.prompts import build_messages  # noqa: E402

ok = True


def report(name: str, passed: bool, detail: str = "") -> None:
    global ok
    ok &= passed
    print(f"[{'PASS' if passed else 'FAIL'}] {name}{' - ' + detail if detail else ''}")


def main() -> None:
    from huggingface_hub import HfApi, hf_hub_download

    token = os.environ.get("HF_TOKEN")
    report("HF_TOKEN set", bool(token), "add it under Kaggle Add-ons > Secrets" if not token else "")
    if token:
        try:
            who = HfApi(token=token).whoami()
            role = who.get("auth", {}).get("accessToken", {}).get("role", "?")
            report("token valid", True, f"user={who['name']} role={role}")
            report("token can write (needed for hub_sync / model upload)", role in ("write", "fineGrained"),
                   "create a WRITE token" if role not in ("write", "fineGrained") else "")
        except Exception as e:  # noqa: BLE001
            report("token valid", False, str(e)[:200])

    for repo in [BASE_MODEL, *ASR_MODELS.values()]:
        try:
            hf_hub_download(repo, "config.json", token=token)
            report(f"access {repo}", True)
        except Exception as e:  # noqa: BLE001
            report(f"access {repo}", False, type(e).__name__)

    try:
        import torch
        n = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(n)]
        report("GPU", n > 0, ", ".join(names) or "no GPU: set Accelerator to GPU T4 x2")
    except Exception as e:  # noqa: BLE001
        report("GPU", False, str(e)[:200])

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(BASE_MODEL, token=token)
        msgs = build_messages("My son is 10 months old. He is breathing very fast.")
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, date_string=DATE_STRING)
        has_headers = ("<|start_header_id|>assistant<|end_header_id|>" in text
                       and f"Today Date: {DATE_STRING}" in text)
        report("chat template has Llama-3 headers + date_string", has_headers)
        n_tokens = len(tok(text, add_special_tokens=False).input_ids)
        report("prompt length", n_tokens < 1400, f"{n_tokens} tokens (system prompt + example case)")
        print("\n--- rendered prompt (head) ---\n" + text[:400] + "\n...\n" + text[-200:])
    except Exception as e:  # noqa: BLE001
        report("tokenizer / chat template", False, str(e)[:300])

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED - fix them before the long runs")


if __name__ == "__main__":
    main()
