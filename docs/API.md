# AHỤIKE HTTP API

A small REST API that runs the released AHỤIKE model (N-ATLaS-8B + LoRA, merged and quantised to GGUF Q4_K_M)
on CPU with llama.cpp. Interactive documentation, where every endpoint can be tried in the browser, is at `/docs`.

**Decision support only, not a diagnosis.** Answers take about 10–30 seconds on a 4-core ARM server.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/triage` | Triage one case |
| `GET` | `/health` | Service status and number of requests served |
| `GET` | `/docs` | Interactive documentation (OpenAPI / Swagger UI) |
| `GET` | `/openapi.json` | Machine-readable API description |

### `POST /triage`

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
