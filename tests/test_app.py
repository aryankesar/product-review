"""Offline tests for the web UI backend (fake LLM and fake SVM, no API key needed)."""
import json

import pytest

import app as webapp
from src.llm_client import LLMClient, LLMError

ASPECTS = {"overall": "mixed", "one_line_summary": "Good battery, short cable.",
           "aspects": [{"aspect": "battery", "sentiment": "positive", "evidence": "battery lasts 12 hours"}]}


class FakeBackend:
    def generate(self, system, user):
        return json.dumps(ASPECTS)


class FakeSVM:
    def predict(self, texts):
        return ["Positive"]


@pytest.fixture
def client(tmp_path, monkeypatch):
    llm = LLMClient({"provider": "gemini", "model": "fake", "temperature": 0, "cache_dir": str(tmp_path)},
                    backend=FakeBackend())
    monkeypatch.setattr(webapp, "get_llm", lambda: llm)
    monkeypatch.setattr(webapp, "get_model", lambda: FakeSVM())
    return webapp.app.test_client()


def test_index_served(client):
    assert b"Review Desk" in client.get("/").data


def test_analyze_returns_svm_and_aspects(client):
    r = client.post("/api/analyze", json={"text": "Battery lasts 12 hours but the cable is short."}).get_json()
    assert r["svm"] == "Positive" and r["llm"]["overall"] == "mixed"
    assert r["llm"]["aspects"][0]["aspect"] == "battery"


def test_empty_review_rejected(client):
    assert client.post("/api/analyze", json={"text": "  "}).status_code == 400


def test_analyze_survives_missing_llm_key(client, monkeypatch):
    def no_key():
        raise LLMError("GEMINI_API_KEY is not set.")
    monkeypatch.setattr(webapp, "get_llm", no_key)
    r = client.post("/api/analyze", json={"text": "Nice product"}).get_json()
    assert r["svm"] == "Positive" and r["llm"] is None and "GEMINI_API_KEY" in r["llm_error"]
