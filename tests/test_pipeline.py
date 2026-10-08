"""Offline tests: a fake LLM backend stands in for the real API, so no key is needed."""
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from src.aspects import analyze_review
from src.classifier import build_pipeline, split_data
from src.compare import classify_batch, run_comparison, wilson_interval
from src.llm_client import (GeminiBackend, LLMAPIError, LLMClient, LLMError, extract_json,
                            find_api_key, key_env_name)
from src.preprocess import clean_text, load_dataset, rating_to_sentiment
from src.prompts import load_prompts, render
from src.summarize import find_product, sample_reviews, summarize_product

PROMPTS = load_prompts("prompts/prompts.yaml")


class FakeBackend:
    """Stands in for a provider backend; replies (or exceptions) are popped from a queue."""
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def generate(self, system, user):
        self.calls += 1
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_client(tmp_path, replies, **extra):
    cfg = {"provider": "gemini", "model": "fake", "max_tokens": 100, "temperature": 0.0,
           "parse_attempts": 2, "cache_dir": str(tmp_path), "retry_wait_seconds": 0, **extra}
    fake = FakeBackend(replies)
    return LLMClient(cfg, backend=fake), fake


def test_clean_text_and_labels():
    assert clean_text("Don't BUY!! 100%") == "don t buy"
    assert [rating_to_sentiment(r) for r in (1, 2, 3, 4, 5)] == \
        ["Negative", "Negative", "Neutral", "Positive", "Positive"]


def test_extract_json_handles_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! [{"id": 0}] hope it helps') == [{"id": 0}]
    with pytest.raises(LLMError):
        extract_json("no json here")


def test_prompt_rendering_keeps_json_braces():
    system, user = render(PROMPTS, "aspect_analysis", review="great phone")
    assert "great phone" in user and '"overall"' in system


def test_aspect_analysis_valid(tmp_path):
    reply = json.dumps({"overall": "MIXED", "aspects": [
        {"aspect": "Battery", "sentiment": "Positive", "evidence": "lasts 12 hours"}],
        "one_line_summary": "ok"})
    client, _ = make_client(tmp_path, [reply])
    out = analyze_review(client, PROMPTS, "Battery lasts 12 hours")
    assert out["overall"] == "mixed" and out["aspects"][0]["aspect"] == "battery"


def test_invalid_output_is_retried_then_succeeds(tmp_path):
    good = json.dumps({"overall": "positive", "aspects": []})
    client, fake = make_client(tmp_path, ['{"overall": "amazing"}', good])  # bad label, then good
    assert analyze_review(client, PROMPTS, "x")["overall"] == "positive"
    assert fake.calls == 2


def test_gives_up_after_max_attempts(tmp_path):
    client, _ = make_client(tmp_path, ["nonsense", "still nonsense"])
    with pytest.raises(LLMError):
        analyze_review(client, PROMPTS, "x")


def test_cache_avoids_second_api_call(tmp_path):
    reply = json.dumps({"overall": "neutral", "aspects": []})
    client, fake = make_client(tmp_path, [reply])
    analyze_review(client, PROMPTS, "same review")
    analyze_review(client, PROMPTS, "same review")
    assert fake.calls == 1 and client.cache_hits == 1


def test_batch_classify_maps_ids_in_order(tmp_path):
    reply = json.dumps([{"id": 1, "sentiment": "negative"}, {"id": 0, "sentiment": "POSITIVE"}])
    client, _ = make_client(tmp_path, [reply])
    assert classify_batch(client, PROMPTS, ["great", "awful"]) == ["Positive", "Negative"]


def test_batch_classify_rejects_missing_ids(tmp_path):
    bad = json.dumps([{"id": 0, "sentiment": "Positive"}])
    good = json.dumps([{"id": 0, "sentiment": "Positive"}, {"id": 1, "sentiment": "Negative"}])
    client, fake = make_client(tmp_path, [bad, good])
    assert classify_batch(client, PROMPTS, ["a", "b"]) == ["Positive", "Negative"]
    assert fake.calls == 2


def test_summarize_samples_all_classes_and_parses(tmp_path):
    df = pd.DataFrame({
        "ProductName": ["Kettle X"] * 30 + ["Other"],
        "Rate": [5.0] * 20 + [1.0] * 5 + [3.0] * 5 + [5.0],
        "Sentiment": ["Positive"] * 20 + ["Negative"] * 5 + ["Neutral"] * 5 + ["Positive"],
        "text": ["review text"] * 31,
    })
    assert find_product(df, "kettle") == "Kettle X"
    assert set(sample_reviews(df, "Kettle X", 9)["Sentiment"]) == {"Positive", "Negative", "Neutral"}
    reply = json.dumps({"verdict": "Good", "pros": ["fast"], "cons": ["leaks"], "recommendation": "Buy"})
    client, _ = make_client(tmp_path, [reply])
    product, n, out = summarize_product(client, PROMPTS, df, "kettle", 9, 100)
    assert product == "Kettle X" and n == 9 and out["cons"] == ["leaks"]


class RateLimit(Exception):
    code = 429


class NotFound(Exception):
    code = 404


def test_rate_limit_is_retried(tmp_path):
    good = json.dumps({"overall": "positive", "aspects": []})
    client, fake = make_client(tmp_path, [RateLimit("429 RESOURCE_EXHAUSTED"), good])
    assert analyze_review(client, PROMPTS, "x")["overall"] == "positive"
    assert fake.calls == 2


def test_persistent_api_error_is_not_retried_as_bad_json(tmp_path):
    client, fake = make_client(tmp_path, [PermissionError("invalid API key")])
    with pytest.raises(LLMAPIError):
        analyze_review(client, PROMPTS, "x")
    assert fake.calls == 1


def test_gemini_backend_falls_back_to_next_model_on_404():
    used = []

    def generate_content(model, contents, config):
        used.append(model)
        if model == "old-model":
            raise NotFound("404 NOT_FOUND")
        return SimpleNamespace(text='{"ok": true}')

    sdk = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    backend = GeminiBackend({"model": "old-model", "fallback_models": ["new-model"],
                             "temperature": 0, "max_tokens": 10}, "key", sdk_client=sdk)
    assert backend.generate("sys", "hi") == '{"ok": true}'
    assert used == ["old-model", "new-model"]
    backend.generate("sys", "again")
    assert used[-1] == "new-model"        # remembers the working model


def test_provider_key_lookup(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert find_api_key("gemini") is None
    monkeypatch.setenv("GOOGLE_API_KEY", "abc")
    assert find_api_key("gemini") == "abc"
    assert key_env_name("anthropic") == "ANTHROPIC_API_KEY"
    with pytest.raises(LLMError):
        key_env_name("openai")


def test_missing_key_gives_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        LLMClient({"provider": "gemini", "model": "m", "temperature": 0, "cache_dir": str(tmp_path)})


def test_load_dataset_removes_cross_product_duplicates(tmp_path):
    """The same review text under different products must collapse to one row (no train/test leak)."""
    csv = tmp_path / "reviews.csv"
    pd.DataFrame({
        "ProductName": ["A", "B", "C", "A"],
        "Price": [1, 2, 3, 4],
        "Rate": [5, 5, 1, 2],
        "Review": ["Nice", "Nice!", "Bad", "Poor"],
        "Summary": ["Good product", "good product.", "Not good", "Broke fast"],
    }).to_csv(csv, index=False)
    df = load_dataset(str(csv))
    assert df["clean_review"].is_unique
    assert len(df) == 3                       # "Nice. Good product" and "Nice!. good product." merge


def test_wilson_interval_behaviour():
    lo, hi = wilson_interval(90, 100)
    assert lo < 0.90 < hi and 0.82 < lo and hi < 0.96
    assert wilson_interval(0, 0) == (0.0, 0.0)
    narrow = wilson_interval(900, 1000)
    assert narrow[1] - narrow[0] < hi - lo    # more data -> tighter interval
    assert wilson_interval(10, 10)[1] == 1.0  # never above 1


def test_run_comparison_end_to_end_with_fake_llm(tmp_path):
    rows = []
    for label, rate, words in (("Negative", 1.0, "terrible broken waste"),
                               ("Neutral", 3.0, "average okay decent"),
                               ("Positive", 5.0, "excellent superb love")):
        for i in range(40):
            rows.append({"ProductName": "P", "Rate": rate, "Sentiment": label,
                         "text": f"{words} item {i}", "clean_review": f"{words} item {i}"})
    df = pd.DataFrame(rows)
    cfg = {"training": {"test_size": 0.25, "random_state": 42,
                        "tfidf": {"max_features": 500, "ngram_range": [1, 2], "sublinear_tf": True}},
           "compare": {"sample_size": 9, "batch_size": 10, "report_path": str(tmp_path / "report.txt")}}
    pipe = build_pipeline(cfg)
    train_df, _ = split_data(df, cfg)
    pipe.fit(train_df["clean_review"], train_df["Sentiment"])

    # sample order is Negative x3, Neutral x3, Positive x3; the fake LLM gets the last Positive wrong
    labels = ["Negative"] * 3 + ["Neutral"] * 3 + ["Positive"] * 2 + ["Neutral"]
    reply = json.dumps([{"id": i, "sentiment": l} for i, l in enumerate(labels)])
    client, _ = make_client(tmp_path, [reply])

    report, table = run_comparison(client, PROMPTS, pipe, df, cfg)
    assert len(table) == 9 and list(table["llm"]) == labels
    assert "95% CI" in report and "confusion matrix" in report
    assert (tmp_path / "report.txt").read_text(encoding="utf-8").startswith("Reviews compared")
