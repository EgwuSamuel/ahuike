---
title: AHUIKE
emoji: 🩺
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.20.0
python_version: "3.11"
app_file: space_app.py
pinned: false
license: other
short_description: Maternal and child danger-sign triage on N-ATLAS
---

# AHỤIKE: same patient, four languages, one answer

*Ahụike nne na nwa*: health for mother and child. Type or speak a maternal or child health case in English, Hausa,
Yorùbá or Igbo and get a triage card: **emergency**, **clinic within 24 hours** or **home care**, with the danger signs
that triggered it. The advice shown is never AI-written: it is one of nine reviewed messages per language.

Runs N-ATLaS-8B fine-tuned for triage (Q4_K_M, llama.cpp) on a free CPU, so each answer takes about 20–60 seconds.
After a quiet period the Space sleeps; the first visit then takes a few minutes to wake it.

**Decision support, not a diagnosis.** If you are worried, go to the health centre.

Code, benchmark and results: https://github.com/EgwuSamuel/ahuike

*N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation and Digital Economy, and powered by
Awarri Technologies. AHỤIKE: Powered by Awarri.*
