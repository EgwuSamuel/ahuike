"""Hugging Face Space entry point on ZeroGPU: the AHỤIKE demo and a live API.

N-ATLaS-8B + the released AHỤIKE adapter (merged, bf16) and the four N-ATLAS ASR models are placed on CUDA at start-up;
ZeroGPU attaches a real GPU only while a @spaces.GPU function runs, and charges that time to the visitor's quota.

Space secret: HF_TOKEN = a READ token with access to NCAIR1/N-ATLaS and the four NCAIR1 ASR models.

API (also listed under "Use via API" at the bottom of the Space page):
  /triage        (text, lang)       -> JSON triage result
  /triage_voice  (audio file, lang) -> JSON triage result, with the transcript
"""
import importlib.util
import os
import sys
from pathlib import Path

import gradio as gr
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ahuike.inference import BASE_MODEL, DATE_STRING  # noqa: E402
from ahuike.pipeline import ASR, ASR_MODELS, Ahuike  # noqa: E402
from ahuike.prompts import build_messages  # noqa: E402

try:
    import spaces
    gpu = spaces.GPU
except ImportError:  # running outside a Space: the decorator does nothing
    def gpu(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda fn: fn

ADAPTER = os.environ.get("AHUIKE_ADAPTER", "SamEgwu/AHUIKE-N-ATLaS-8B-LoRA-Powered-by-Awarri")
TOKEN = os.environ.get("HF_TOKEN")
DEVICE = "cuda" if os.environ.get("AHUIKE_CPU") is None else "cpu"
API_FIELDS = ("triage", "patient", "danger_signs", "raised_by_review", "advice", "valid_json", "latency_s")

# ----------------------------------------------------------------------------- models (module level, as ZeroGPU requires)
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # noqa: E402

tok = AutoTokenizer.from_pretrained(BASE_MODEL, token=TOKEN)
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, token=TOKEN)
model = PeftModel.from_pretrained(base, ADAPTER, token=TOKEN).merge_and_unload().to(DEVICE).eval()
STOP_IDS = [i for i in {tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")} if isinstance(i, int) and i >= 0]

asr = ASR()
for lang, repo in ASR_MODELS.items():
    try:
        asr._pipes[lang] = pipeline("automatic-speech-recognition", model=repo, token=TOKEN,
                                    chunk_length_s=30, device=DEVICE)
    except Exception as e:  # noqa: BLE001 - voice for one language must not take the whole demo down
        print(f"[ASR] {lang} ({repo}) not preloaded: {e}")


class ZeroGPULLM:
    """Same interface as ahuike.pipeline.TriageLLM, on the merged transformers model."""

    @torch.inference_mode()
    def generate(self, text: str, max_new_tokens: int = 96) -> str:
        prompt = tok.apply_chat_template(build_messages(text), tokenize=False, add_generation_prompt=True,
                                         date_string=DATE_STRING)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             eos_token_id=STOP_IDS, pad_token_id=STOP_IDS[0])
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


ahuike = Ahuike(ZeroGPULLM(), asr, log_path=None)


# ----------------------------------------------------------------------------- GPU entry points
@gpu(duration=30)
def gpu_triage_text(text: str, lang: str) -> dict:
    return ahuike.triage_text(text, lang)


@gpu(duration=60)
def gpu_triage_audio(audio_path: str, lang: str) -> dict:
    return ahuike.triage_audio(audio_path, lang)


class SpaceApp:
    """What app/app.py's build() expects; every call goes through the GPU entry points above."""

    def triage_text(self, text, lang, source="text"):
        return gpu_triage_text(text, lang)

    def triage_audio(self, audio_path, lang):
        return gpu_triage_audio(audio_path, lang)


def _api_view(res: dict, transcript: bool = False) -> dict:
    out = {k: res.get(k) for k in API_FIELDS}
    out["danger_signs"] = out["danger_signs"] or []
    out["raised_by_review"] = out["raised_by_review"] or []
    out["advice"] = out["advice"] or "When in doubt, go to the health centre today."
    if transcript:
        out["transcript"] = res.get("input", "")
    return out


def api_triage(text: str, lang: str = "en") -> dict:
    """Triage one case. text: the case as a caregiver or CHEW would describe it. lang: en, ha, yo or ig."""
    lang = lang if lang in ASR_MODELS else "en"
    if not text or len(text.strip()) < 3:
        return {"error": "text must be at least 3 characters"}
    return _api_view(gpu_triage_text(text.strip()[:2000], lang))


def api_triage_voice(audio, lang: str = "en") -> dict:
    """Triage one spoken case: N-ATLAS speech recognition for `lang`, then the same triage."""
    lang = lang if lang in ASR_MODELS else "en"
    if not audio:
        return {"error": "audio file required"}
    return _api_view(gpu_triage_audio(audio, lang), transcript=True)


# ----------------------------------------------------------------------------- UI + API
spec = importlib.util.spec_from_file_location("ahuike_app", ROOT / "app" / "app.py")
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)

demo = ui.build(SpaceApp())
with demo:
    with gr.Row(visible=False):
        a_text = gr.Textbox(label="text")
        a_lang = gr.Textbox(label="lang", value="en")
        a_audio = gr.Audio(type="filepath", label="audio")
        a_out = gr.JSON(label="result")
        b_text = gr.Button("api triage")
        b_voice = gr.Button("api triage voice")
    b_text.click(api_triage, [a_text, a_lang], a_out, api_name="triage")
    b_voice.click(api_triage_voice, [a_audio, a_lang], a_out, api_name="triage_voice")

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch()
