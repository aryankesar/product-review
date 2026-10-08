"""LLM task 2: turn many reviews of one product into a pros/cons summary."""
import pandas as pd

from .llm_client import LLMClient
from .prompts import render


def _validate(data: dict) -> dict:
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    return {
        "verdict": str(data["verdict"]).strip(),
        "pros": [str(x).strip() for x in data.get("pros", [])][:5],
        "cons": [str(x).strip() for x in data.get("cons", [])][:5],
        "recommendation": str(data["recommendation"]).strip(),
    }


def find_product(df: pd.DataFrame, query: str) -> str:
    """Return the best-matching product name (most reviewed among substring matches)."""
    mask = df["ProductName"].str.contains(query, case=False, regex=False, na=False)
    if not mask.any():
        raise ValueError(f"No product matches '{query}'.")
    return df[mask]["ProductName"].value_counts().idxmax()


def sample_reviews(df: pd.DataFrame, product: str, max_reviews: int, seed: int = 42) -> pd.DataFrame:
    """Sample across all sentiments so minority (negative) opinions are not drowned out."""
    sub = df[df["ProductName"] == product]
    per_class = -(-max_reviews // 3)  # ceil
    parts = [g.sample(min(len(g), per_class), random_state=seed) for _, g in sub.groupby("Sentiment")]
    return pd.concat(parts).head(max_reviews)


def summarize_product(client: LLMClient, prompts: dict, df: pd.DataFrame, query: str,
                      max_reviews: int, max_chars: int) -> tuple[str, int, dict]:
    product = find_product(df, query)
    sample = sample_reviews(df, product, max_reviews)
    lines = [f"[{int(r.Rate)} stars] {r.text[:max_chars]}" for r in sample.itertuples()]
    system, user = render(prompts, "product_summary",
                          product=product, count=len(lines), reviews="\n".join(lines))
    return product, len(lines), client.complete_json(system, user, validator=_validate)
