"""AHỤIKE demo: speak or type a case in English, Hausa, Yoruba or Igbo -> triage card.

  # CPU / HF Space (merged GGUF):
  AHỤIKE_GGUF=<user>/AHUIKE-N-ATLaS-8B-GGUF-Powered-by-Awarri/<file>.gguf python app/app.py
  # GPU notebook (adapter):
  python app/app.py --backend hf --adapter outputs/ahuike-lora --share
"""
from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402

from ahuike.pipeline import ASR, Ahuike, TriageLLM  # noqa: E402
from ahuike.protocol import CLINIC, EMERGENCY, HOME, TRIGGERS  # noqa: E402

LANG_CHOICES = {"English": "en", "Hausa": "ha", "Yorùbá": "yo", "Igbo": "ig"}

CARD = {
    EMERGENCY: ("#b42318", "#fef3f2", "EMERGENCY — REFER NOW", "Go to the nearest hospital immediately."),
    CLINIC: ("#b54708", "#fffaeb", "CLINIC WITHIN 24 HOURS", "Visit the health centre today."),
    HOME: ("#067647", "#ecfdf3", "HOME CARE", "Care at home and watch for danger signs."),
    None: ("#475467", "#f2f4f7", "COULD NOT DECIDE", "When in doubt, go to the health centre today."),
}

EXAMPLES = [
    ["My son is 10 months old. He has had cough for 4 days. He is breathing very fast. What should I do?", "English"],
    ["I am 34 weeks pregnant. I have a very severe headache that will not go. My eyes are blurred, I cannot see clearly. Is this normal?", "English"],
    ["CHEW note: 1 year 2 months old female child. Complaints: cough x 3 days. Findings: RR 44/min, temp 37.0°C. Please advise on triage.", "English"],
    ["I delivered my baby 6 days ago. My nipples are sore. I feel tired most of the time. Should I be worried?", "English"],
]

ATTRIBUTION = ("N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation and Digital "
               "Economy, and powered by Awarri Technologies. AHỤIKE — Powered by Awarri.")
DISCLAIMER = ("AHỤIKE is decision support for community health workers and caregivers, not a diagnosis. "
              "It always errs toward referral. If you are worried, go to the health centre.")


def card_html(res: dict) -> str:
    fg, bg, title, sub = CARD.get(res.get("triage"), CARD[None])
    signs = res.get("danger_signs") or []
    sign_html = "".join(
        f"<li><b>{html.escape(s)}</b> — {html.escape(TRIGGERS.get(s, ''))}</li>" for s in signs)
    advice = html.escape(res.get("advice") or "")
    return f"""
<div style="border:2px solid {fg};background:{bg};border-radius:14px;padding:18px 20px;font-family:system-ui">
  <div style="color:{fg};font-size:22px;font-weight:800;letter-spacing:.3px">{title}</div>
  <div style="color:{fg};margin:2px 0 12px">{sub}</div>
  {f'<div style="font-weight:600;margin-bottom:4px">Danger signs found</div><ul style="margin:0 0 12px 18px">{sign_html}</ul>' if signs else ''}
  {f'<div style="font-size:17px;line-height:1.45">{advice}</div>' if advice else ''}
  <div style="color:#667085;font-size:12px;margin-top:12px">Response time {res.get('latency_s', '–')} s · Powered by N-ATLAS · Powered by Awarri</div>
</div>"""


def build(app: Ahuike) -> gr.Blocks:
    def run_text(text, lang_name):
        if not text or not text.strip():
            return "", "", {}
        res = app.triage_text(text, LANG_CHOICES[lang_name])
        return text, card_html(res), {k: res[k] for k in ("triage", "danger_signs", "advice", "raw")}

    def run_audio(audio, lang_name):
        if not audio:
            return "", "", {}
        res = app.triage_audio(audio, LANG_CHOICES[lang_name])
        return res["input"], card_html(res), {k: res[k] for k in ("triage", "danger_signs", "advice", "raw")}

    with gr.Blocks(title="AHỤIKE — maternal & child triage on N-ATLAS") as demo:
        gr.Markdown("# AHỤIKE\n**Same patient, four languages, one answer.** Maternal & child danger-sign "
                    "triage in English, Hausa, Yorùbá and Igbo, built on N-ATLAS (LLM + ASR).")
        with gr.Row():
            with gr.Column(scale=1):
                lang = gr.Radio(list(LANG_CHOICES), value="Hausa", label="Language")
                audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Speak the case")
                btn_audio = gr.Button("Triage from voice", variant="primary")
                text = gr.Textbox(lines=5, label="…or type / edit the case")
                btn_text = gr.Button("Triage from text")
            with gr.Column(scale=1):
                card = gr.HTML()
                with gr.Accordion("Model output (JSON)", open=False):
                    js = gr.JSON()
        gr.Examples(EXAMPLES, inputs=[text, lang])
        gr.Markdown(f"<small>{DISCLAIMER}<br>{ATTRIBUTION}</small>")
        btn_audio.click(run_audio, [audio, lang], [text, card, js])
        btn_text.click(run_text, [text, lang], [text, card, js])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="gguf", choices=("gguf", "hf"))
    ap.add_argument("--gguf", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--log", default=str(ROOT / "results" / "interactions.jsonl"))
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    llm = TriageLLM(backend=args.backend, gguf_path=args.gguf, adapter=args.adapter)
    asr = ASR(device=0 if args.backend == "hf" else None)
    build(Ahuike(llm, asr, log_path=args.log)).launch(share=args.share)


if __name__ == "__main__":
    main()
