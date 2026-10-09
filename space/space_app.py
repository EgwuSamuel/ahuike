"""Hugging Face Space entry point: the AHỤIKE demo on a free CPU (Q4_K_M GGUF through llama.cpp).

Space settings: variable AHUIKE_GGUF = <repo>/<file>.gguf, secret HF_TOKEN = a READ token with access to
NCAIR1/N-ATLaS, the four NCAIR1 ASR models and the (private) GGUF repo. scripts/deploy_space.py sets both.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ahuike.pipeline import ASR, Ahuike, TriageLLM  # noqa: E402

spec = importlib.util.spec_from_file_location("ahuike_app", ROOT / "app" / "app.py")
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)

llm = TriageLLM(backend="gguf")
# llama.cpp reuses a cached prompt prefix, so evaluating the long protocol prompt once at start-up
# means visitors only wait for their own message and the short answer.
llm.generate("warm-up")
demo = ui.build(Ahuike(llm, ASR(), log_path=None))
demo.queue(default_concurrency_limit=1).launch()
