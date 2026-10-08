"""EDA, model comparison, product analysis and word clouds (everything from the report)."""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from wordcloud import STOPWORDS, WordCloud

from .classifier import split_data

FIG_DIR = Path("reports/figures")


def _finish(fig, name: str, show: bool):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_rating_distribution(df, show=True):
    fig, ax = plt.subplots()
    df["Rate"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set(title="Rating Distribution", xlabel="Rating", ylabel="Count")
    _finish(fig, "rating_distribution", show)


def plot_review_length(df, show=True):
    fig, ax = plt.subplots()
    df["Review"].astype(str).str.len().plot(kind="hist", bins=50, ax=ax)
    ax.set(title="Review Length Distribution", xlabel="Length of Review", ylabel="Frequency")
    _finish(fig, "review_length", show)


def plot_sentiment_distribution(df, show=True):
    fig, ax = plt.subplots()
    sns.countplot(x="Sentiment", data=df, order=["Positive", "Neutral", "Negative"], ax=ax)
    ax.set_title("Sentiment Distribution")
    _finish(fig, "sentiment_distribution", show)


def compare_models(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Train the four classic models on identical TF-IDF features.

    Returns accuracy and macro-F1. Macro-F1 averages the per-class F1 scores equally, so it
    is not dominated by the 77% Positive majority the way accuracy is.
    """
    train_df, test_df = split_data(df, cfg)
    t = cfg["training"]["tfidf"]
    vec = TfidfVectorizer(max_features=t["max_features"], ngram_range=tuple(t["ngram_range"]),
                          sublinear_tf=t["sublinear_tf"])
    X_train = vec.fit_transform(train_df["clean_review"])  # fit on train only: no leakage
    X_test = vec.transform(test_df["clean_review"])
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_jobs=-1, random_state=42),
        "SVM (LinearSVC)": LinearSVC(class_weight="balanced"),
    }
    rows = []
    for name, model in models.items():
        model.fit(X_train, train_df["Sentiment"])
        pred = model.predict(X_test)
        rows.append({"Model": name,
                     "Accuracy": round(accuracy_score(test_df["Sentiment"], pred) * 100, 2),
                     "Macro-F1": round(f1_score(test_df["Sentiment"], pred, average="macro") * 100, 2)})
    return pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)


def product_analysis(df: pd.DataFrame, min_reviews: int = 200, top: int = 10):
    """Rank products by share of positive reviews (fair to popular and niche products alike).

    Ranking by raw counts (as in the first notebook) makes heavily reviewed products
    appear in BOTH the best and worst lists.
    """
    g = df.groupby("ProductName")["Sentiment"]
    stats = pd.DataFrame({
        "reviews": g.size(),
        "positive_pct": g.apply(lambda s: (s == "Positive").mean() * 100).round(1),
        "negative_pct": g.apply(lambda s: (s == "Negative").mean() * 100).round(1),
    })
    stats = stats[stats["reviews"] >= min_reviews]
    return (stats.sort_values("positive_pct", ascending=False).head(top),
            stats.sort_values("negative_pct", ascending=False).head(top))


def _keep_negations(text: str) -> str:
    """'not good' -> 'not_good', "don t buy" -> 'not_buy' so negation survives stopword removal."""
    text = re.sub(r"\b(don|doesn|didn|isn|wasn|can|won|couldn|wouldn)\s+t\s+(\w+)", r"not_\2", text)
    return re.sub(r"\b(not|no|never)\s+(\w+)", r"not_\2", text)


def plot_wordcloud(df: pd.DataFrame, sentiment: str, show=True):
    text = _keep_negations(" ".join(df[df["Sentiment"] == sentiment]["clean_review"]))
    stop = set(STOPWORDS) | {"nan", "t", "s", "don", "u", "one", "product", "flipkart", "amazon"}
    if sentiment == "Positive":
        stop.add("good")  # appears everywhere and hides other words
    cloud = WordCloud(width=900, height=450, stopwords=stop, collocations=False,
                      background_color="black" if sentiment == "Negative" else "white").generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(cloud)
    ax.axis("off")
    ax.set_title(f"{sentiment} Review Word Cloud")
    _finish(fig, f"wordcloud_{sentiment.lower()}", show)
