"""Web UI for the project:  python app.py  ->  http://127.0.0.1:5000

Thin Flask layer over src/: the SVM baseline, aspect analysis and product summaries.
The page works without an API key; only the LLM parts need one.
"""
import re
import threading
from pathlib import Path

import yaml
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory

from src import classifier
from src.aspects import analyze_review
from src.llm_client import LLMClient, LLMError
from src.preprocess import load_dataset
from src.prompts import load_prompts
from src.summarize import summarize_product

load_dotenv()
ROOT = Path(__file__).parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
app = Flask(__name__, static_folder=None)
_cache, _lock = {}, threading.Lock()


def _once(key, build):
    with _lock:
        if key not in _cache:
            _cache[key] = build()
        return _cache[key]


def get_data():
    path = CFG["data"]["full_path"]
    if not Path(path).exists():
        path = CFG["data"]["sample_path"]
    return _once("df", lambda: load_dataset(path))


def get_model():
    return _once("svm", lambda: classifier.load_model(CFG["model_path"]))


def get_llm():
    return _once("llm", lambda: LLMClient(CFG["llm"]))  # raises LLMError if no key


def get_prompts():
    return _once("prompts", lambda: load_prompts(CFG["prompts_path"]))


def _text():
    text = ((request.get_json(silent=True) or {}).get("text") or "").strip()
    if not text:
        raise ValueError("Write or paste a review first.")
    return text[:2000]


@app.errorhandler(LLMError)
def _llm_error(err):
    return jsonify(error=str(err)), 502


@app.errorhandler(ValueError)
def _bad_input(err):
    return jsonify(error=str(err)), 400


@app.errorhandler(FileNotFoundError)
def _missing(err):
    return jsonify(error=str(err)), 503


@app.get("/")
def index():
    return send_from_directory(ROOT / "web", "index.html")


@app.get("/figures/<path:name>")
def figures(name):
    return send_from_directory(ROOT / "reports" / "figures", name)


@app.get("/api/status")
def status():
    llm = CFG["llm"]
    try:
        get_llm()
        llm_ready = True
    except LLMError:
        llm_ready = False
    return jsonify(svm_ready=Path(CFG["model_path"]).exists(), llm_ready=llm_ready,
                   provider=llm["provider"], model=llm["model"])


@app.post("/api/analyze")
def analyze():
    text = _text()
    out = {"svm": None, "llm": None, "llm_error": None}
    try:
        out["svm"] = classifier.predict(get_model(), text)
    except FileNotFoundError:
        pass  # SVM is optional here; the UI shows it as "not trained"
    try:
        out["llm"] = analyze_review(get_llm(), get_prompts(), text)
    except LLMError as err:
        out["llm_error"] = str(err)
    return jsonify(out)


@app.get("/api/products")
def products():
    q = (request.args.get("q") or "").strip()
    df = get_data()
    if len(q) < 2:
        return jsonify([])
    hit = df[df["ProductName"].str.contains(q, case=False, regex=False, na=False)]
    counts = hit["ProductName"].value_counts().head(8)
    return jsonify([{"name": n, "reviews": int(c)} for n, c in counts.items()])


@app.post("/api/summarize")
def summarize():
    product = ((request.get_json(silent=True) or {}).get("product") or "").strip()
    if not product:
        raise ValueError("Type a product name first.")
    s = CFG["summarize"]
    try:
        name, n, result = summarize_product(get_llm(), get_prompts(), get_data(), product,
                                            s["max_reviews"], s["max_chars"])
    except ValueError as err:  # no product matched
        return jsonify(error=str(err)), 404
    return jsonify(product=name, reviews=n, **result)


@app.get("/api/report")
def report():
    df = get_data()
    counts = df["Sentiment"].value_counts()
    bench = {}
    path = Path(CFG["compare"]["report_path"])
    if path.exists():
        txt = path.read_text(encoding="utf-8")
        for m in re.finditer(r"(SVM|LLM) accuracy : ([\d.]+)\s+\(95% CI ([\d.]+) - ([\d.]+), n=(\d+)", txt):
            bench[m[1].lower()] = {"accuracy": float(m[2]), "low": float(m[3]), "high": float(m[4]), "n": int(m[5])}
    return jsonify(reviews=len(df), sentiment={k: int(counts.get(k, 0)) for k in ("Positive", "Neutral", "Negative")},
                   benchmark=bench)


if __name__ == "__main__":
    threading.Thread(target=get_data, daemon=True).start()  # warm the dataset while the page loads
    app.run(host="127.0.0.1", port=5000, debug=False)
