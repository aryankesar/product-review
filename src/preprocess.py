"""Data loading, cleaning and rating -> sentiment labelling."""
import re

import pandas as pd

LABELS = ["Negative", "Neutral", "Positive"]


def clean_text(text) -> str:
    """Lowercase, keep letters only, collapse whitespace."""
    text = re.sub(r"[^a-z]", " ", str(text).lower())
    return " ".join(text.split())


def rating_to_sentiment(rate: float) -> str:
    if rate >= 4:
        return "Positive"
    if rate == 3:
        return "Neutral"
    return "Negative"


def load_dataset(path: str) -> pd.DataFrame:
    """Load the Flipkart CSV and return a cleaned, labelled DataFrame.

    Adds: Sentiment (label from stars), text (raw Review + Summary, used for the
    LLM) and clean_review (normalised text, used for TF-IDF). Rows with identical
    cleaned text are collapsed to one so no review can appear in both train and test.
    """
    df = pd.read_csv(path, encoding="latin1", on_bad_lines="skip")
    df = df.dropna(subset=["Review", "Rate"])
    # latin1 decoding leaves artefacts in product names ("Â", "?ÿ"); tidy them for display
    df["ProductName"] = (df["ProductName"].astype(str)
                         .str.replace(r"[Â\xa0]+|(?:\?ÿ)+", " ", regex=True)
                         .str.replace(r"\s+", " ", regex=True).str.strip())
    df = df.drop_duplicates(subset=["ProductName", "Review", "Summary"])
    df["Rate"] = pd.to_numeric(df["Rate"], errors="coerce")
    df = df[df["Rate"].between(1, 5)].copy()
    df["Sentiment"] = df["Rate"].apply(rating_to_sentiment)
    df["text"] = df["Review"].astype(str) + ". " + df["Summary"].fillna("").astype(str)
    df["clean_review"] = df["text"].apply(clean_text)
    df = df[df["clean_review"] != ""]
    # The same review text is posted under many products (e.g. "Nice. Good product").
    # If identical text lands in both the train and test split, test accuracy is inflated,
    # so keep one copy of each distinct cleaned text. This is what prevents train/test leakage.
    df = df.drop_duplicates(subset=["clean_review"])
    return df.reset_index(drop=True)
