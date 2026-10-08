"""Command-line entry point.

  python main.py train
  python main.py eda
  python main.py predict "The product was not as expected"
  python main.py analyze "Battery lasts 12 hours but the cable is too short"
  python main.py summarize --product "electric kettle"
  python main.py compare
"""
import argparse
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src import classifier, eda
from src.aspects import analyze_review
from src.compare import run_comparison
from src.llm_client import LLMClient, LLMError
from src.preprocess import load_dataset
from src.prompts import load_prompts
from src.summarize import summarize_product


def load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_data(cfg: dict):
    path = cfg["data"]["full_path"]
    if not Path(path).exists():
        path = cfg["data"]["sample_path"]
        print(f"[note] Full dataset not found, using the sample: {path}\n")
    return load_dataset(path)


def make_llm(cfg: dict) -> LLMClient:
    return LLMClient(cfg["llm"])


def cmd_train(cfg, args):
    print(classifier.train(load_data(cfg), cfg))
    print(f"Model saved to {cfg['model_path']}")


def cmd_eda(cfg, args):
    import matplotlib
    matplotlib.use("Agg")  # save figures to reports/figures without opening windows
    df = load_data(cfg)
    eda.plot_rating_distribution(df, show=False)
    eda.plot_review_length(df, show=False)
    eda.plot_sentiment_distribution(df, show=False)
    for label in ("Positive", "Negative"):
        eda.plot_wordcloud(df, label, show=False)
    best, worst = eda.product_analysis(df)
    print("Most positively reviewed products (>=200 reviews):\n", best.to_string(), "\n")
    print("Most negatively reviewed products (>=200 reviews):\n", worst.to_string(), "\n")
    print("Comparing 4 models (Random Forest makes this take a few minutes)...")
    print(eda.compare_models(df, cfg).to_string(index=False))
    print(f"\nFigures saved to {eda.FIG_DIR}")


def cmd_predict(cfg, args):
    pipe = classifier.load_model(cfg["model_path"])
    print(f"SVM prediction: {classifier.predict(pipe, args.text)}")


def cmd_analyze(cfg, args):
    try:
        print(f"SVM (baseline) : {classifier.predict(classifier.load_model(cfg['model_path']), args.text)}")
    except FileNotFoundError:
        print("SVM (baseline) : skipped (run `python main.py train` first)")
    result = analyze_review(make_llm(cfg), load_prompts(cfg["prompts_path"]), args.text)
    print(f"LLM overall    : {result['overall']}")
    print(f"LLM summary    : {result['one_line_summary']}")
    print("LLM aspects    :")
    for a in result["aspects"] or []:
        print(f"  - {a['aspect']:<15} {a['sentiment']:<9} \"{a['evidence']}\"")
    if not result["aspects"]:
        print("  (none found)")


def cmd_summarize(cfg, args):
    df = load_data(cfg)
    s = cfg["summarize"]
    product, n, out = summarize_product(make_llm(cfg), load_prompts(cfg["prompts_path"]), df,
                                        args.product, s["max_reviews"], s["max_chars"])
    print(f"Product : {product}\nReviews analysed: {n}\n")
    print(f"Verdict : {out['verdict']}\n")
    print("Pros:");  [print(f"  + {p}") for p in out["pros"]]
    print("Cons:");  [print(f"  - {c}") for c in out["cons"]]
    print(f"\nRecommendation: {out['recommendation']}")


def cmd_compare(cfg, args):
    df = load_data(cfg)
    pipe = classifier.load_model(cfg["model_path"])
    report, table = run_comparison(make_llm(cfg), load_prompts(cfg["prompts_path"]), pipe, df, cfg)
    out = Path(cfg["compare"]["output_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(report, f"\nDetailed results saved to {out}")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Flipkart review sentiment: SVM baseline + LLM analysis")
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train", help="train and save the SVM baseline")
    sub.add_parser("eda", help="plots, word clouds, product ranking, 4-model comparison")
    p = sub.add_parser("predict", help="SVM sentiment for one review"); p.add_argument("text")
    p = sub.add_parser("analyze", help="LLM aspect-based analysis of one review"); p.add_argument("text")
    p = sub.add_parser("summarize", help="LLM pros/cons summary for a product"); p.add_argument("--product", required=True)
    sub.add_parser("compare", help="benchmark LLM vs SVM on held-out reviews")
    args = parser.parse_args()

    handlers = {"train": cmd_train, "eda": cmd_eda, "predict": cmd_predict, "analyze": cmd_analyze,
                "summarize": cmd_summarize, "compare": cmd_compare}
    try:
        handlers[args.command](load_config(args.config), args)
    except (LLMError, FileNotFoundError, ValueError) as err:
        sys.exit(f"Error: {err}")


if __name__ == "__main__":
    main()
