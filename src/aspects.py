"""LLM task 1: aspect-based sentiment analysis of a single review."""
from .llm_client import LLMClient
from .prompts import render

OVERALL = {"positive", "negative", "neutral", "mixed"}
ASPECT = {"positive", "negative", "neutral"}


def _validate(data: dict) -> dict:
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    overall = str(data["overall"]).lower()
    if overall not in OVERALL:
        raise ValueError(f"bad overall label: {overall}")
    aspects = []
    for a in data.get("aspects", []):
        sentiment = str(a["sentiment"]).lower()
        if sentiment not in ASPECT:
            raise ValueError(f"bad aspect sentiment: {sentiment}")
        aspects.append({
            "aspect": str(a["aspect"]).lower().strip(),
            "sentiment": sentiment,
            "evidence": str(a.get("evidence", "")).strip(),
        })
    return {"overall": overall, "aspects": aspects,
            "one_line_summary": str(data.get("one_line_summary", "")).strip()}


def analyze_review(client: LLMClient, prompts: dict, review: str) -> dict:
    system, user = render(prompts, "aspect_analysis", review=review.strip())
    return client.complete_json(system, user, validator=_validate)
