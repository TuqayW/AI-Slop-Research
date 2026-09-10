import csv
import json
import joblib
import numpy as np

from collections import Counter
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


MODEL = "hc3_v2_model.joblib"
DATA = "dataset/human_detectors_grouped.csv"


def metric_rows(rows, model):
    texts = [r["text"] for r in rows]
    y = np.array([int(r["ai"]) for r in rows])

    word = model["word_vectorizer"]
    char = model["char_vectorizer"]

    from scipy.sparse import hstack

    X = hstack([
        word.transform(texts),
        char.transform(texts),
    ])

    scores = model["model"].predict_proba(X)[:, 1]
    threshold = float(model["threshold"])
    pred = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[0, 1],
    ).ravel()

    result = {
        "n": len(rows),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(
            y, pred, zero_division=0
        ),
        "recall": recall_score(
            y, pred, zero_division=0
        ),
        "f1": f1_score(
            y, pred, zero_division=0
        ),
        "fpr": fp / (fp + tn) if (fp + tn) else None,
        "auroc": roc_auc_score(y, scores),
    }

    return result


model = joblib.load(MODEL)

with open(DATA, encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

test = [
    r for r in rows
    if r["split"] == "test"
]

print("TEST TOTAL:", len(test))
print("THRESHOLD:", model["threshold"])

print("\nOVERALL")
print(json.dumps(metric_rows(test, model), indent=2))

models = sorted({
    r["generation_model"]
    for r in test
})

print("\nBY GENERATION CONDITION")

for name in models:
    subset = [r for r in test if r["generation_model"] == name]

    print("\n" + name)
    print(json.dumps(
        metric_rows(subset, model),
        indent=2
    ))

print("\nERRORS")

# Score every test example again so we can identify misses.
texts = [r["text"] for r in test]

from scipy.sparse import hstack

X = hstack([
    model["word_vectorizer"].transform(texts),
    model["char_vectorizer"].transform(texts),
])

scores = model["model"].predict_proba(X)[:, 1]
threshold = float(model["threshold"])

for row, score in zip(test, scores):
    actual = int(row["ai"])
    predicted = int(score >= threshold)

    if actual != predicted:
        print(
            "id =", row["id"],
            "| actual =", actual,
            "| score =", round(float(score), 6),
            "| model =", row["generation_model"],
            "| source =", row["source"],
            "| prompt =", row["prompt_id"]
        )
