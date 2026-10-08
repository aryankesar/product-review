"""Classical ML baseline: TF-IDF (unigrams + bigrams) + LinearSVC."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .preprocess import clean_text


def split_data(df: pd.DataFrame, cfg: dict):
    """Deterministic stratified split, shared by training and comparison."""
    t = cfg["training"]
    return train_test_split(
        df, test_size=t["test_size"], random_state=t["random_state"], stratify=df["Sentiment"]
    )


def build_pipeline(cfg: dict) -> Pipeline:
    t = cfg["training"]["tfidf"]
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=t["max_features"],
            ngram_range=tuple(t["ngram_range"]),
            sublinear_tf=t["sublinear_tf"],
        )),
        ("svm", LinearSVC(class_weight="balanced")),
    ])


def train(df: pd.DataFrame, cfg: dict) -> str:
    """Fit on the train split only (vectorizer included), save, return a report."""
    train_df, test_df = split_data(df, cfg)
    pipe = build_pipeline(cfg)
    pipe.fit(train_df["clean_review"], train_df["Sentiment"])
    preds = pipe.predict(test_df["clean_review"])

    Path(cfg["model_path"]).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, cfg["model_path"])

    acc = accuracy_score(test_df["Sentiment"], preds)
    return (f"Train size: {len(train_df)}  Test size: {len(test_df)}\n"
            f"Accuracy: {acc:.4f}\n\n{classification_report(test_df['Sentiment'], preds)}")


def load_model(path: str) -> Pipeline:
    if not Path(path).exists():
        raise FileNotFoundError(f"No trained model at '{path}'. Run: python main.py train")
    return joblib.load(path)


def predict(pipe: Pipeline, text: str) -> str:
    return pipe.predict([clean_text(text)])[0]
