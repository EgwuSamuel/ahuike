"""AHỤIKE HTTP API: maternal and child danger-sign triage on CPU (llama.cpp GGUF).

  uvicorn api.server:app --host 0.0.0.0 --port 8000

Environment:
  AHUIKE_GGUF  local .gguf path, or "<hf repo id>/<file>.gguf" (downloaded on first start)
  HF_TOKEN     Hugging Face read token (gated N-ATLaS tokenizer, private GGUF repo)
  AHUIKE_LOG   optional JSONL file for request logs

Interactive documentation is served at /docs. Requests are handled one at a time: llama.cpp runs a single
model instance, and keeping the system prompt in its cache makes each answer much faster.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ahuike.pipeline import Ahuike, TriageLLM  # noqa: E402

DESCRIPTION = """
Cross-lingual maternal and child danger-sign triage, fine-tuned from **N-ATLaS-8B** (Powered by Awarri).

Send a case in **English, Hausa, Yorùbá or Igbo** and receive the urgency level, the patient group, the protocol
danger signs behind the decision, and reviewed advice in the same language. The advice is never written by the
model: it is one of nine messages reviewed by a medical doctor and native speakers.

**Decision support only, not a diagnosis.** Answers take about 10–30 seconds on this CPU server.

Code and documentation: https://github.com/EgwuSamuel/ahuike
"""

EXAMPLES = {
    "pregnancy_emergency": {"summary": "English, pregnancy emergency", "value": {
        "text": "I am 32 weeks pregnant. I have a very severe headache that will not go and my eyes are blurred.",
        "lang": "en"}},
    "child_clinic": {"summary": "English, CHEW note (clinic)", "value": {
        "text": "CHEW note: 14-month-old girl, cough for 3 days, breathing 45 per minute, no chest indrawing, feeding well.",
        "lang": "en"}},
    "igbo_postpartum": {"summary": "Igbo, after delivery", "value": {
        "text": "Ụdị Ọgwụgwọ: Nwanyị dị afọ iri na asatọ, ụbọchị iri atọ na isii ka e mesịrị n'ịmụ nwa. "
                "Ihe mgbu: mgbu siri ike n'afọ. Biko gwa m banyere usoro mbụ.", "lang": "ig"}},
}


class TriageRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=2000, description="The case, as a caregiver or CHEW would describe it")
    lang: Literal["en", "ha", "yo", "ig"] = Field("en", description="Language of the case and of the advice")


class TriageResponse(BaseModel):
    triage: Optional[Literal["EMERGENCY_REFER_NOW", "CLINIC_WITHIN_24H", "HOME_CARE"]] = Field(
        None, description="Urgency level; null if the model reply could not be read (treat as: go to the health centre today)")
    patient: Optional[Literal["child", "pregnant", "postpartum"]] = None
    danger_signs: list[str] = Field(default_factory=list, description="Protocol trigger identifiers behind the decision")
    raised_by_review: list[str] = Field(default_factory=list,
                                        description="Signs for which the clinician-reviewed rule raised the level")
    advice: str = Field("", description="Reviewed advice for this patient group and level, in the requested language")
    valid_json: bool = Field(..., description="Whether the model's reply was a valid JSON object")
    lang: str
    latency_s: float


STATE: dict = {"ahuike": None, "loaded_at": None, "requests": 0}
LOCK = threading.Lock()
FALLBACK = "When in doubt, go to the health centre today."


def load() -> None:
    llm = TriageLLM(backend="gguf", gguf_path=os.environ.get("AHUIKE_GGUF"))
    STATE["ahuike"] = Ahuike(llm, log_path=os.environ.get("AHUIKE_LOG"))
    STATE["ahuike"].triage_text("Warm-up: my baby has a mild cough.", "en")  # caches the system prompt
    STATE["loaded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")


@asynccontextmanager
async def lifespan(_: FastAPI):
    load()
    yield


app = FastAPI(title="AHỤIKE API", version="1.0.0", description=DESCRIPTION, lifespan=lifespan,
              license_info={"name": "Code: Apache-2.0; model: N-ATLaS licence"})


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health", summary="Service status")
def health() -> dict:
    return {"status": "ok" if STATE["ahuike"] else "loading", "model_loaded_at": STATE["loaded_at"],
            "requests_served": STATE["requests"]}


@app.post("/triage", response_model=TriageResponse, summary="Triage one case",
          openapi_extra={"requestBody": {"content": {"application/json": {"examples": EXAMPLES}}}})
def triage(req: TriageRequest) -> TriageResponse:
    if STATE["ahuike"] is None:
        raise HTTPException(503, "Model is still loading; try again in a minute.")
    with LOCK:
        res = STATE["ahuike"].triage_text(req.text, req.lang)
        STATE["requests"] += 1
    return TriageResponse(
        triage=res.get("triage"), patient=res.get("patient"), danger_signs=res.get("danger_signs") or [],
        raised_by_review=res.get("raised_by_review") or [], advice=res.get("advice") or FALLBACK,
        valid_json=bool(res.get("valid_json")), lang=req.lang, latency_s=res.get("latency_s", 0.0))
