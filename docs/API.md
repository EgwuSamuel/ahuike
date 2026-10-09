# AHỤIKE API

AHỤIKE can be served in two ways: a **FastAPI server** that runs the model on CPU with llama.cpp (any Linux
machine), or a **Hugging Face Space** on ZeroGPU (requires a Hugging Face paid plan). Both run the same pipeline as the benchmark:
N-ATLaS-8B with the released adapter, the output parser, the clinician-reviewed guard and reviewed advice.

**Decision support only, not a diagnosis.**

## Hugging Face Space (GPU) option

`space/zerogpu_app.py` packages the same pipeline as a Hugging Face Space on ZeroGPU: a web demo plus two API
endpoints, documented automatically under "Use via API" on the Space page. Hosting Gradio or ZeroGPU Spaces requires a
Hugging Face paid plan (PRO), so no public instance is maintained; deploy your own with

```bash
hf auth login                                          # write token
SPACE_HF_TOKEN=<read token> python scripts/deploy_space.py
```

| Endpoint | Inputs | Output |
|---|---|---|
| `/triage` | `text` (string), `lang` (`en`, `ha`, `yo`, `ig`) | the JSON fields described below |
| `/triage_voice` | `audio` (wav/mp3 file), `lang` | the same fields plus `transcript` |

Python (`pip install gradio_client`), with `<user>/AHUIKE` your Space:

```python
from gradio_client import Client
client = Client("<user>/AHUIKE")
print(client.predict("I am 32 weeks pregnant. I have a very severe headache.", "en", api_name="/triage"))
```

## Self-hosted API (FastAPI, CPU)

The FastAPI server in `api/server.py` serves the same model on CPU. Its endpoints:

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/triage` | Triage one case |
| `GET` | `/health` | Service status and number of requests served |
| `GET` | `/docs` | Interactive documentation (OpenAPI / Swagger UI) |
| `GET` | `/openapi.json` | Machine-readable API description |

#### `POST /triage`

Request:

```json
{"text": "I am 32 weeks pregnant. I have a very severe headache that will not go.", "lang": "en"}
```

- `text`: the case as a caregiver or community health worker would describe it (3–2,000 characters).
- `lang`: `en`, `ha`, `yo` or `ig`. The advice is returned in this language.

Response:

```json
{
  "triage": "EMERGENCY_REFER_NOW",
  "patient": "pregnant",
  "danger_signs": ["severe_headache"],
  "raised_by_review": ["severe_headache"],
  "advice": "DANGER SIGN. Go to the nearest hospital NOW with someone to accompany you. ...",
  "valid_json": true,
  "lang": "en",
  "latency_s": 12.4
}
```

- `triage`: `EMERGENCY_REFER_NOW`, `CLINIC_WITHIN_24H` or `HOME_CARE`. `null` if the model's reply could not be read;
  the advice is then "When in doubt, go to the health centre today."
- `danger_signs`: protocol sign identifiers (see `ahuike/protocol/rules.py`, `TRIGGERS`).
- `raised_by_review`: present when a clinician-reviewed rule raised the level (single pre-eclampsia sign, red umbilicus).
- `advice`: one of nine reviewed messages, never written by the model.

## Examples

```bash
curl -X POST http://<server>:8000/triage -H "Content-Type: application/json" \
     -d '{"text": "CHEW note: 14-month-old girl, cough for 3 days, breathing 45 per minute.", "lang": "en"}'
```

```python
import requests
r = requests.post("http://<server>:8000/triage",
                  json={"text": "My baby is 3 weeks old and is not feeding well.", "lang": "en"}, timeout=120)
print(r.json()["triage"], r.json()["advice"])
```

## Running it yourself

Local (any Linux or macOS machine with 8 GB RAM):

```bash
pip install -r api/requirements.txt
export HF_TOKEN=<read token>     # gated N-ATLaS tokenizer and the GGUF repository
export AHUIKE_GGUF=/path/to/AHUIKE-N-ATLaS-8B-Q4_K_M.gguf
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Oracle Cloud Always Free (Ubuntu 22.04, VM.Standard.A1.Flex with 4 OCPUs and 24 GB RAM): open TCP port 8000 in the
VCN security list, then on the server run

```bash
curl -O https://raw.githubusercontent.com/EgwuSamuel/ahuike/main/deploy/oracle/setup.sh
HF_TOKEN=<read token> bash setup.sh
```

The script installs dependencies, compiles llama.cpp, downloads the model, opens the firewall port and starts the
API as a systemd service that restarts automatically.

## Design notes

- One model instance serves requests one at a time. llama.cpp keeps the system prompt in its cache, so after the
  first request only the case text is processed.
- The API uses exactly the same code path as the application (`ahuike.pipeline.Ahuike`): system prompt, output
  parser, review guard and reviewed advice. Tests: `tests/test_api.py`.
