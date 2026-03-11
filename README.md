# Spam Classifier

A SMS spam detector built with **PyTorch** and **TF-IDF**, available in two modes.

## Approach 1 — Notebook (exploration)

Open `spam_classifier.ipynb` and run all cells. Everything is self-contained: data loading, training, evaluation, and inference in one place.

```bash
uv run jupyter notebook
```

## Approach 2 — Docker app (production)

One command: trains the model inside the image and serves a Streamlit web interface.

```bash
docker compose up --build
```

Then open [http://localhost:8501](http://localhost:8501).

**Files:**

```
model.py    # neural network definition
train.py    # trains and saves model.pth + vectorizer.pkl
main.py     # Streamlit web app
Dockerfile
```

## How the model works

1. **Data** — SMS Spam Collection dataset (~5 500 messages), labels mapped to 0 (ham) / 1 (spam)
2. **Vectorization** — TF-IDF converts each message into a numerical vector
3. **Model** — 3-layer feed-forward network (input → 64 → 32 → 1), BCEWithLogitsLoss, Adam, 100 epochs
4. **Inference** — sigmoid output thresholded at 0.5 → HAM or SPAM

## Dependencies


| Package        | Role                                |
| -------------- | ----------------------------------- |
| `torch`        | Neural network training & inference |
| `scikit-learn` | TF-IDF vectorizer, train/test split |
| `pandas`       | Dataset loading                     |
| `streamlit`    | Web interface (approach 2)          |
