# Flipkart Review Sentiment Analysis with an LLM

An NLP project on Flipkart product reviews (189,874 raw rows, **103,568 unique reviews** after cleaning).
A classical ML model (TF-IDF + LinearSVC) provides a fast baseline, and an LLM (Google Gemini by default,
which has a free tier; Anthropic Claude is also supported) adds what the baseline cannot do:
**aspect-level understanding, product summaries, and an independent benchmark.**

## What the LLM does (and why it is not decoration)

| Command | LLM task | Why an LLM is needed |
|---|---|---|
| `analyze` | Aspect-based sentiment: which aspects (battery, price, ...) are praised or criticised, with quoted evidence | A one-label classifier cannot say *"great battery, bad cable"* |
| `summarize` | Reads a stratified sample of reviews of one product and writes verdict, pros, cons, recommendation | Requires reading and synthesising free text |
| `compare` | Classifies held-out reviews in batches and benchmarks LLM vs SVM against the star-rating labels | Gives an independent check of the baseline |

## Project structure

```
├── project.ipynb           MAIN NOTEBOOK: runs everything top to bottom (EDA, models, LLM)
├── main.py                 CLI entry point (same features as commands)
├── app.py + web/index.html Web UI (Flask): analyze, summarize, model report
├── config.yaml             all settings: paths, model name, temperature, batch sizes, ...
├── prompts/prompts.yaml    ALL prompts (aspect analysis, batch classification, product summary)
├── src/
│   ├── preprocess.py       loading, cleaning, de-duplication, rating -> label
│   ├── classifier.py       TF-IDF + LinearSVC baseline
│   ├── eda.py              plots, word clouds, product ranking, 4-model comparison
│   ├── llm_client.py       LLM client (Gemini / Claude): rate-limit retries, JSON validation, disk cache
│   ├── prompts.py          prompt loading/rendering
│   ├── aspects.py          LLM task 1
│   ├── summarize.py        LLM task 2
│   └── compare.py          LLM task 3 (accuracy with 95% confidence intervals, confusion matrices)
├── tests/test_pipeline.py  18 offline tests (fake LLM, no API key needed)
├── reports/                figures, and compare_report.txt written by the benchmark
├── data/sample_reviews.csv small sample so the repo runs out of the box
├── notebooks/              original EDA / baseline notebook
├── requirements.txt
└── .env.example            copy to .env and add your key
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                 # Windows: copy .env.example .env   then edit .env
```

**Free API key (no card):** open https://aistudio.google.com, sign in with a Google account, open the
**API keys** page (key icon in the sidebar), click **Create API key**, and paste it into `.env` as
`GEMINI_API_KEY=...`. Never commit `.env` (it is git-ignored).

**Using Claude instead:** in `config.yaml` set `provider: anthropic`, `model: claude-haiku-4-5-20251001`,
`min_seconds_between_calls: 0`, and put `ANTHROPIC_API_KEY` in `.env` (paid API).

**Dataset.** Download `flipkart_product.csv` (Kaggle "Flipkart Product Reviews") and place it at
`data/flipkart_product.csv`. If it is missing, the project automatically falls back to
`data/sample_reviews.csv` (about 5k reviews) so everything still runs.

## Run it as a notebook (easiest)

```bash
jupyter notebook project.ipynb        # or open the folder in VS Code and open project.ipynb
```
Choose **Run All**. The notebook reads the key from `.env` (or asks once; press Enter to skip the LLM
sections, everything else still runs). On Google Colab: upload the folder, `%cd flipkart-sentiment-llm`,
`!pip install -r requirements.txt`, then open `project.ipynb`.

## Web app

```bash
python app.py          # then open http://127.0.0.1:5000
```

A single-page UI over the same code as the CLI (`app.py` + `web/index.html`, Flask only, no build step):

- **Analyze a review**: SVM label and LLM overall label side by side, with each aspect's quoted evidence
  highlighted inside your review text.
- **Summarize a product**: type a product name (autocomplete from the dataset) for verdict, pros, cons.
- **Model report**: class balance, SVM vs LLM benchmark from `reports/compare_report.txt`, word clouds.

Without an API key the SVM still works and the page tells you the LLM is unavailable.

## Usage (command line)

```bash
python main.py eda                                    # plots, word clouds, product ranking, 4-model comparison (~4 min)
python main.py train                                  # train + save the SVM baseline
python main.py predict "The product was not as expected"

python main.py analyze "Battery lasts 12 hours and sound is crisp, but the cable is too short."
python main.py summarize --product "electric kettle"
python main.py compare                                # LLM vs SVM on held-out reviews (15 API calls)
python -m pytest -q                                   # run tests (no API key needed)
```

Figures are saved to `reports/figures/`. Run `train` before `predict`, `analyze` (baseline line) and `compare`.

## Configuration

Everything tunable lives in `config.yaml`: the LLM provider and model (`llm.provider`, `llm.model`,
`llm.fallback_models`), temperature, rate-limit settings, benchmark size, number of reviews per summary,
file paths and TF-IDF settings. If a model is unavailable to your key, the next one in `fallback_models`
is tried automatically.

## Design decisions

- **Prompts live in one file** with the same structure: role, closed label sets, JSON schema,
  a worked example including negation, grounding rules ("evidence must be quoted from the review"),
  and an instruction to treat review text as data (prompt-injection defence).
- **Structured, validated output.** Every reply must be JSON and is checked (labels, ids, fields).
  Invalid replies trigger a re-ask (`parse_attempts`), then a clear error.
- **Efficiency.** `compare` classifies 10 reviews per API call; identical requests are served
  from an on-disk cache; `temperature: 0` keeps outputs reproducible; a small, fast model is the default.
- **Free-tier friendly.** The client waits between calls (`min_seconds_between_calls`) and retries with
  growing waits on HTTP 429, so the whole notebook fits inside free-tier request limits (about 20 calls).
- **No train/test leakage.** Reviews are de-duplicated on their *cleaned text*, not just per product:
  the same text ("Nice. Good product") is posted under many products, and with per-product de-duplication
  41% of test reviews had an identical twin in the training set. The TF-IDF vectorizer is fitted on the
  training split only, and `compare` samples only from the held-out test split.
- **Balanced summaries.** Reviews sent for a product summary are sampled across sentiment classes
  so negative opinions are not drowned out by the 73% positive majority.
- **Honest benchmark.** Accuracy is reported with a 95% confidence interval and confusion matrices.

## Results

Baseline (TF-IDF + LinearSVC, stratified 80/20 split, 20,714 held-out test reviews). Macro-F1 weights the
three classes equally, so it is not inflated by the large Positive class:

| Model | Accuracy | Macro-F1 |
|---|---|---|
| **SVM (LinearSVC), used here** | **97.27%** | **94.15%** |
| Logistic Regression | 96.36% | 92.68% |
| Random Forest | 96.35% | 91.43% |
| Naive Bayes | 94.45% | 86.31% |

SVM per-class F1: Positive 0.99, Negative 0.97, Neutral 0.87.

*Why not 97.8%?* An earlier version reached 97.8%, but 41% of its test reviews also appeared verbatim in the
training data. After de-duplicating on cleaned text the score is 97.3%, which is the trustworthy figure.

**LLM vs SVM** (`python main.py compare`): the latest numbers, with confidence intervals and confusion matrices,
are written to `reports/compare_report.txt`. An earlier run (90 reviews, before de-duplication) gave
SVM 0.944 vs LLM 0.722 accuracy, with the LLM's Neutral recall at only 0.23. Read this carefully: the "truth"
is the star rating, which the SVM was trained on, while the LLM judges the text. For example the LLM calls
"Nice. Not bad" Positive, which is reasonable to a human but counts as an error against a 3-star label.
So the benchmark shows the LLM is *not a drop-in replacement* for the trained classifier, not that it
understands reviews worse. Where the LLM clearly adds value is aspect-level analysis and cases like sarcasm
(a sarcastic vacuum review was labelled Negative by the SVM but correctly understood as praise by the LLM).

## Limitations

- Labels come from star ratings, not human annotation, so "ground truth" is noisy
  (a 3-star review can read as positive, and vice versa).
- The Neutral class (about 9% of the data) is the weakest for both approaches.
- The LLM benchmark uses 150 reviews; its 95% confidence interval is roughly ±6-8 points.
- One marketplace, mostly English; Hinglish and emoji-heavy reviews are handled poorly (the CSV is not
  valid UTF-8, so emojis are lost when it is read as latin-1).
- LLM calls add latency (and cost money on paid plans; free tiers have rate limits and may use prompts for
  model training, which is fine for public product reviews); the SVM is orders of magnitude faster, which is
  why both are kept.
- Product-level word/ranking statistics depend on review volume, not only sentiment.

## Possible next steps

- Hand-label ~200 reviews to get a real ground truth and re-run the benchmark against it.
- Fine-tune a small transformer (e.g. DistilBERT) and compare with the SVM and the LLM.
- Add a batch-upload page (CSV of reviews in, labelled CSV out) to the web UI.
"# sentiment-analysis" 
