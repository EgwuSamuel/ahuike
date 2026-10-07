"""QLoRA fine-tune of N-ATLaS-8B with Unsloth (fits one Kaggle T4 16 GB).

  python scripts/finetune.py --data data/sft_anchored.jsonl --out outputs/ahuike-lora
  python scripts/finetune.py --data data/sft_ablation.jsonl --out outputs/ablation-lora

Loss is computed on assistant tokens only. Checkpoints every --save-steps so a killed
session can resume with --resume.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ahuike.inference import BASE_MODEL, DATE_STRING  # noqa: E402

USER_HEADER = "<|start_header_id|>user<|end_header_id|>\n\n"
ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=-1, help="stop after N steps (smoke test); -1 = full epochs")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--push", default=None, help="optional HF repo id to push the adapter to")
    args = ap.parse_args()

    from unsloth import FastLanguageModel, is_bfloat16_supported  # import first: patches transformers
    from unsloth.chat_templates import train_on_responses_only
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    token = os.environ.get("HF_TOKEN")
    model, tok = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=args.max_len, load_in_4bit=True, token=token)
    model = FastLanguageModel.get_peft_model(
        model, r=args.r, lora_alpha=2 * args.r, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=3407)

    with open(args.data, encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    texts = [tok.apply_chat_template(r["messages"], tokenize=False, date_string=DATE_STRING) for r in rows]
    if ASSISTANT_HEADER not in texts[0]:
        sys.exit("Chat template does not use Llama-3 headers; update USER_HEADER/ASSISTANT_HEADER.\n"
                 + texts[0][:800])
    lengths = sorted(len(tok(t).input_ids) for t in texts[:200])
    print(f"{len(texts)} examples; token length p50={lengths[len(lengths)//2]} max={lengths[-1]}")
    ds = Dataset.from_dict({"text": texts})

    cfg = SFTConfig(
        output_dir=args.out,
        dataset_text_field="text",
        max_seq_length=args.max_len,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        weight_decay=0.01,
        optim="adamw_8bit",
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        seed=3407,
    )
    try:
        trainer = SFTTrainer(model=model, processing_class=tok, train_dataset=ds, args=cfg)
    except TypeError:  # older TRL
        trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=ds, args=cfg)
    trainer = train_on_responses_only(trainer, instruction_part=USER_HEADER, response_part=ASSISTANT_HEADER)
    stats = trainer.train(resume_from_checkpoint=True if args.resume else None)

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    Path(args.out, "train_stats.json").write_text(json.dumps({
        "data": args.data, "examples": len(texts), "epochs": args.epochs, "lr": args.lr, "r": args.r,
        "train_runtime_s": stats.metrics.get("train_runtime"), "train_loss": stats.metrics.get("train_loss"),
        "log_history": trainer.state.log_history,
    }, indent=2))
    print(f"saved adapter to {args.out}")
    if args.push:
        model.push_to_hub(args.push, token=token)
        tok.push_to_hub(args.push, token=token)
        print(f"pushed to https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
