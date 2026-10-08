"""LLM task 3: benchmark LLM vs SVM vs star-rating labels on held-out reviews."""
from math import sqrt
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .classifier import predict, split_data
from .llm_client import LLMClient
from .preprocess import LABELS
from .prompts import render


def _make_validator(expected_ids: set):
    def validate(data):
        if not isinstance(data, list):
            raise ValueError("expected a JSON array")
        out = {}
        for item in data:
            label = str(item["sentiment"]).strip().capitalize()
            if label not in LABELS:
                raise ValueError(f"bad label: {label}")
            out[int(item["id"])] = label
        if set(out) != expected_ids:
            raise ValueError("ids returned do not match ids sent")
        return out
    return validate


def classify_batch(client: LLMClient, prompts: dict, texts: list[str], max_chars: int = 400) -> list[str]:
    body = "\n".join(f"[{i}] {t[:max_chars]}" for i, t in enumerate(texts))
    system, user = render(prompts, "batch_classify", reviews=body)
    validate = _make_validator(set(range(len(texts))))
    result = client.complete_json(system, user, validator=lambda d: {str(k): v for k, v in validate(d).items()})
    return [result[str(i)] for i in range(len(texts))]


def wilson_interval(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson confidence interval for an accuracy measured on n reviews.

    With only ~100 test reviews a single accuracy number is noisy; the interval shows how
    much. Two models whose intervals overlap heavily cannot be called different.
    """
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def _accuracy_line(name: str, truth, pred) -> str:
    n = len(truth)
    correct = int((truth == pred).sum())
    lo, hi = wilson_interval(correct, n)
    return f"{name} accuracy : {correct / n:.3f}   (95% CI {lo:.3f} - {hi:.3f}, n={n})"


def _confusion(truth, pred) -> str:
    cm = pd.DataFrame(confusion_matrix(truth, pred, labels=LABELS),
                      index=[f"true {l}" for l in LABELS],
                      columns=[f"pred {l}" for l in LABELS])
    return cm.to_string()


def run_comparison(client, prompts, pipe, df, cfg) -> tuple[str, pd.DataFrame]:
    _, test_df = split_data(df, cfg)  # never sample from data the SVM trained on
    n_per_class = max(1, cfg["compare"]["sample_size"] // len(LABELS))
    sample = pd.concat([
        g.sample(min(len(g), n_per_class), random_state=cfg["training"]["random_state"])
        for _, g in test_df.groupby("Sentiment")
    ]).reset_index(drop=True)

    bs = cfg["compare"]["batch_size"]
    llm_preds = []
    for i in range(0, len(sample), bs):
        llm_preds += classify_batch(client, prompts, sample["text"].iloc[i:i + bs].tolist())

    sample["svm"] = [predict(pipe, t) for t in sample["text"]]
    sample["llm"] = llm_preds
    truth = sample["Sentiment"]

    lines = [
        f"Reviews compared (balanced, held-out): {len(sample)}",
        _accuracy_line("SVM", truth, sample["svm"]),
        _accuracy_line("LLM", truth, sample["llm"]),
        f"SVM/LLM agreement: {(sample['svm'] == sample['llm']).mean():.3f}",
        "", "--- LLM per-class report ---", classification_report(truth, sample["llm"], zero_division=0),
        "--- SVM per-class report ---", classification_report(truth, sample["svm"], zero_division=0),
        "--- LLM confusion matrix ---", _confusion(truth, sample["llm"]),
        "", "--- SVM confusion matrix ---", _confusion(truth, sample["svm"]),
        "",
        "Note: 'truth' is the star-rating label (4-5 Positive, 3 Neutral, 1-2 Negative). The SVM was",
        "trained on that labelling convention; the LLM judges the text, so disagreements are not",
        "automatically LLM mistakes (see reports/compare_results.csv).",
        f"LLM API calls: {client.api_calls}   cache hits: {client.cache_hits}",
    ]
    report = "\n".join(lines)

    report_path = cfg["compare"].get("report_path")
    if report_path:  # keep the numbers next to the code so the README can quote them
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(report + "\n", encoding="utf-8")
    return report, sample[["Rate", "Sentiment", "svm", "llm", "text"]]
