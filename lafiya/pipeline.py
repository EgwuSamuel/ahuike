"""End-to-end LAFIYA pipeline: speech (N-ATLAS ASR) -> text -> triage (LAFIYA / N-ATLaS LLM).

LLM backends:
  gguf - llama.cpp on CPU with the merged Q4_K_M GGUF (HF Space / laptop at a PHC)
  hf   - transformers + LoRA adapter on GPU (Kaggle / Colab demo)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .inference import BASE_MODEL, DATE_STRING
from .prompts import build_messages, parse_output

ASR_MODELS = {
    "en": "NCAIR1/NigerianAccentedEnglish",
    "ha": "NCAIR1/Hausa-ASR",
    "yo": "NCAIR1/Yoruba-ASR",
    "ig": "NCAIR1/Igbo-ASR",
}

# Fallback only: used if the N-ATLaS tokenizer (and its chat template) cannot be loaded.
_LLAMA3 = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
           "Cutting Knowledge Date: December 2023\nToday Date: {date}\n\n{system}<|eot_id|>"
           "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
           "<|start_header_id|>assistant<|end_header_id|>\n\n")


class ASR:
    """Lazy per-language N-ATLAS Whisper ASR pipelines."""

    def __init__(self, device: int | str | None = None):
        self._pipes = {}
        self.device = device

    def transcribe(self, audio_path: str, lang: str) -> str:
        if lang not in self._pipes:
            from transformers import pipeline
            kw = {"device": self.device} if self.device is not None else {}
            self._pipes[lang] = pipeline("automatic-speech-recognition", model=ASR_MODELS[lang],
                                         token=os.environ.get("HF_TOKEN"), chunk_length_s=30, **kw)
        return self._pipes[lang](audio_path)["text"].strip()


class TriageLLM:
    def __init__(self, backend: str = "gguf", gguf_path: str | None = None, adapter: str | None = None,
                 n_threads: int | None = None):
        self.backend = backend
        if backend == "gguf":
            from llama_cpp import Llama
            path = gguf_path or os.environ.get("LAFIYA_GGUF")
            if path and not Path(path).exists() and "/" in path:
                # "<repo_id>/<filename>" on the Hugging Face Hub
                from huggingface_hub import hf_hub_download
                repo, fname = path.rsplit("/", 1)
                path = hf_hub_download(repo, fname, token=os.environ.get("HF_TOKEN"))
            self.llm = Llama(model_path=path, n_ctx=4096, n_threads=n_threads or os.cpu_count(),
                             verbose=False)
            self.tok = None
            try:
                from transformers import AutoTokenizer
                self.tok = AutoTokenizer.from_pretrained(BASE_MODEL, token=os.environ.get("HF_TOKEN"))
            except Exception as e:  # noqa: BLE001 - tokenizer is optional
                print(f"N-ATLaS tokenizer unavailable ({e}); using built-in Llama-3 template")
        elif backend == "hf":
            from .inference import ChatEngine
            self.engine = ChatEngine(backend="hf")
            self.adapter = adapter
        else:
            raise ValueError(backend)

    def _prompt(self, text: str) -> str:
        msgs = build_messages(text)
        if self.tok is not None:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                date_string=DATE_STRING)
        return _LLAMA3.format(date=DATE_STRING, system=msgs[0]["content"], user=msgs[-1]["content"])

    def generate(self, text: str, max_new_tokens: int = 320) -> str:
        if self.backend == "gguf":
            out = self.llm(self._prompt(text), max_tokens=max_new_tokens, temperature=0.0,
                           stop=["<|eot_id|>", "<|end_of_text|>"])
            return out["choices"][0]["text"]
        return self.engine.chat([build_messages(text)], adapter=self.adapter,
                                max_new_tokens=max_new_tokens)[0]


class Lafiya:
    def __init__(self, llm: TriageLLM, asr: ASR | None = None, log_path: str | None = None):
        self.llm = llm
        self.asr = asr or ASR()
        self.log_path = log_path

    def triage_text(self, text: str, lang: str, source: str = "text") -> dict:
        t0 = time.time()
        raw = self.llm.generate(text)
        res = parse_output(raw)
        res.update({"input": text, "lang": lang, "raw": raw, "source": source,
                    "latency_s": round(time.time() - t0, 2)})
        self._log(res)
        return res

    def triage_audio(self, audio_path: str, lang: str) -> dict:
        t0 = time.time()
        transcript = self.asr.transcribe(audio_path, lang)
        res = self.triage_text(transcript, lang, source="voice")
        res["asr_latency_s"] = round(time.time() - t0 - res["latency_s"], 2)
        return res

    def _log(self, res: dict) -> None:
        if not self.log_path:
            return
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        row = {k: res.get(k) for k in ("lang", "source", "input", "triage", "danger_signs", "valid_json",
                                       "latency_s")}
        row["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
