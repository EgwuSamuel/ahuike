"""Batched chat inference against N-ATLaS, optionally through LoRA adapters.

Backends:
  vllm - fastest; serves the base model and any number of LoRA adapters from one engine.
  hf   - transformers + peft fallback; slower but works wherever a GPU does.

The HF token is read from the HF_TOKEN environment variable (set it via Kaggle/Colab Secrets).
"""
from __future__ import annotations

import os
from contextlib import nullcontext

BASE_MODEL = os.environ.get("NATLAS_MODEL", "NCAIR1/N-ATLaS")
# Fixed so that training and inference see exactly the same rendered system header.
DATE_STRING = "01 Oct 2026"


class ChatEngine:
    """backend: "vllm", "hf", or "auto" (try vLLM, fall back to transformers if it fails)."""

    def __init__(self, backend: str = "auto", enable_lora: bool = False,
                 max_model_len: int = 4096, base_model: str = BASE_MODEL):
        self.base_model = base_model
        self._lora_ids: dict[str, int] = {}
        if backend == "hf":
            # Less fragmentation on 16 GB T4s. Only for transformers: it breaks vLLM's multi-GPU
            # all-reduce and CUDA-graph capture, so it is never set when vLLM may run.
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if backend == "auto":
            try:
                self._init_vllm(enable_lora, max_model_len)
                backend = "vllm"
            except Exception as e:  # noqa: BLE001 - any vLLM failure means fall back
                print(f"[ChatEngine] vLLM unavailable ({type(e).__name__}: {e}); falling back to transformers")
                import gc
                gc.collect()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
                self._init_hf()
                backend = "hf"
        elif backend == "vllm":
            self._init_vllm(enable_lora, max_model_len)
        elif backend == "hf":
            self._init_hf()
        else:
            raise ValueError(backend)
        self.backend = backend
        print(f"[ChatEngine] backend = {backend}")

    def _init_vllm(self, enable_lora: bool, max_model_len: int) -> None:
        import torch
        from vllm import LLM
        self.llm = LLM(
            model=self.base_model,
            dtype="half",
            max_model_len=max_model_len,
            tensor_parallel_size=max(1, torch.cuda.device_count()),
            gpu_memory_utilization=0.90,
            enable_lora=enable_lora,
            max_lora_rank=64,
            max_loras=2,
            # T4s have no GPU peer-to-peer link; custom all-reduce failed during CUDA-graph
            # capture with LoRA enabled. Eager mode with LoRA trades a little speed for stability.
            disable_custom_all_reduce=True,
            enforce_eager=enable_lora,
        )

    def _init_hf(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        token = os.environ.get("HF_TOKEN")
        self.tok = AutoTokenizer.from_pretrained(self.base_model, token=token)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model, token=token, torch_dtype=torch.float16, device_map="auto").eval()
        self._peft = False

    # ------------------------------------------------------------------ public
    def chat(self, conversations: list[list[dict]], adapter: str | None = None,
             max_new_tokens: int = 256, temperature: float = 0.0, batch_size: int = 8,
             repetition_penalty: float = 1.0) -> list[str]:
        """repetition_penalty: the N-ATLaS card recommends 1.12 for free text; keep 1.0 for JSON."""
        if self.backend == "vllm":
            return self._vllm_chat(conversations, adapter, max_new_tokens, temperature, repetition_penalty)
        return self._hf_chat(conversations, adapter, max_new_tokens, temperature, batch_size,
                             repetition_penalty)

    # ------------------------------------------------------------------ vllm
    def _vllm_chat(self, conversations, adapter, max_new_tokens, temperature, repetition_penalty):
        from vllm import SamplingParams
        lora = None
        if adapter:
            from vllm.lora.request import LoRARequest
            if adapter not in self._lora_ids:
                self._lora_ids[adapter] = len(self._lora_ids) + 1
            lora = LoRARequest(os.path.basename(adapter.rstrip("/")) or "lora",
                               self._lora_ids[adapter], adapter)
        params = SamplingParams(temperature=temperature, max_tokens=max_new_tokens,
                                repetition_penalty=repetition_penalty)
        outs = self.llm.chat(conversations, params, lora_request=lora,
                             chat_template_kwargs={"date_string": DATE_STRING}, use_tqdm=True)
        return [o.outputs[0].text for o in outs]

    # ------------------------------------------------------------------ hf
    def _hf_set_adapter(self, adapter):
        if adapter is None:
            return self.model.disable_adapter() if self._peft else nullcontext()
        from peft import PeftModel
        name = f"a{abs(hash(adapter)) % 10**8}"
        if not self._peft:
            try:
                self.model = PeftModel.from_pretrained(self.model, adapter, adapter_name=name).eval()
            except ImportError as e:
                if "torchao" in str(e):
                    raise ImportError("An old torchao blocks peft. Run: pip uninstall -y torchao") from e
                raise
            self._peft = True
        elif name not in self.model.peft_config:
            self.model.load_adapter(adapter, adapter_name=name)
        self.model.set_adapter(name)
        return nullcontext()

    def _hf_chat(self, conversations, adapter, max_new_tokens, temperature, batch_size,
                 repetition_penalty):
        import torch
        from tqdm.auto import tqdm
        results = []
        with self._hf_set_adapter(adapter):
            for i in tqdm(range(0, len(conversations), batch_size)):
                batch = conversations[i:i + batch_size]
                prompts = [self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True,
                                                        date_string=DATE_STRING) for c in batch]
                enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
                enc = {k: v.to(self.model.device) for k, v in enc.items()}
                kw = {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}
                with torch.no_grad():
                    gen = self.model.generate(**enc, max_new_tokens=max_new_tokens,
                                              repetition_penalty=repetition_penalty,
                                              pad_token_id=self.tok.pad_token_id, **kw)
                results += self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                                 skip_special_tokens=True)
        return results
