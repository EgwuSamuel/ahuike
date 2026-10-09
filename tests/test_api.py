"""HTTP API with a stub model (skipped when FastAPI is not installed)."""
import importlib.util
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HAVE_API = all(importlib.util.find_spec(m) for m in ("fastapi", "httpx"))


class StubLLM:
    """Stands in for TriageLLM: answers like the fine-tuned model would."""

    def __init__(self, *a, **kw):
        self.prompts = []

    def generate(self, text, max_new_tokens=96):
        self.prompts.append(text)
        if "headache" in text:
            return json.dumps({"triage": "CLINIC_WITHIN_24H", "patient": "pregnant", "danger_signs": ["severe_headache"]})
        if "garbage" in text:
            return "sorry, I cannot help"
        return json.dumps({"triage": "HOME_CARE", "patient": "child", "danger_signs": []})


@unittest.skipUnless(HAVE_API, "fastapi/httpx not installed")
class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        import api.server as server
        server.TriageLLM = StubLLM
        cls.client = TestClient(server.app).__enter__()  # runs the lifespan (model load + warm-up)

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    def test_review_guard_applied(self):
        r = self.client.post("/triage", json={"text": "I am pregnant and have a severe headache.", "lang": "en"})
        body = r.json()
        self.assertEqual(r.status_code, 200)
        self.assertEqual(body["triage"], "EMERGENCY_REFER_NOW")  # single pre-eclampsia sign raised by the guard
        self.assertEqual(body["raised_by_review"], ["severe_headache"])
        self.assertTrue(body["advice"].startswith("DANGER SIGN"))

    def test_unreadable_reply_fails_safe(self):
        body = self.client.post("/triage", json={"text": "garbage input here", "lang": "ha"}).json()
        self.assertIsNone(body["triage"])
        self.assertFalse(body["valid_json"])
        self.assertIn("health centre", body["advice"])

    def test_validation(self):
        self.assertEqual(self.client.post("/triage", json={"text": "hi", "lang": "en"}).status_code, 422)
        self.assertEqual(self.client.post("/triage", json={"text": "my baby is hot", "lang": "fr"}).status_code, 422)

    def test_docs_served(self):
        self.assertEqual(self.client.get("/docs").status_code, 200)
        self.assertIn("/triage", self.client.get("/openapi.json").json()["paths"])


if __name__ == "__main__":
    unittest.main()
