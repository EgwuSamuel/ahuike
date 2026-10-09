---
title: AHUIKE
emoji: 🩺
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.30.0
python_version: "3.10"
app_file: zerogpu_app.py
pinned: false
license: other
short_description: Maternal and child danger-sign triage on N-ATLAS
---

# AHỤIKE: same patient, four languages, one answer

*Ahụike nne na nwa*: health for mother and child. Type or speak a maternal or child health case in English, Hausa,
Yorùbá or Igbo and get a triage card: **emergency**, **clinic within 24 hours** or **home care**, with the danger signs
that triggered it. The advice shown is never AI-written: it is one of nine reviewed messages per language.

Runs N-ATLaS-8B with the released AHỤIKE adapter on ZeroGPU, a shared GPU that Hugging Face attaches for each request.
Answers take a few seconds; at busy times a request may wait briefly in the GPU queue. Use is free: Hugging Face gives every
visitor a small daily GPU allowance, so signing in to Hugging Face gives you more requests per day.

**Live API.** Two endpoints, documented under **"Use via API"** at the bottom of this page:
`/triage` (text, lang) and `/triage_voice` (audio, lang), where lang is `en`, `ha`, `yo` or `ig`.

```python
from gradio_client import Client
client = Client("SamEgwu/AHUIKE")
print(client.predict("I am 32 weeks pregnant and have a very severe headache.", "en", api_name="/triage"))
```

**Decision support, not a diagnosis.** If you are worried, go to the health centre.

Code, benchmark and results: https://github.com/EgwuSamuel/ahuike

*N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation and Digital Economy, and powered by
Awarri Technologies. AHỤIKE: Powered by Awarri.*
