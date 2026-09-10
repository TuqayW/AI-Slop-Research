import json
import re
import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


CSV_PATH = "dataset/human_detectors_grouped.csv"
MODEL_PATH = "hc3_v8_normalized_model.joblib"
RESULTS_PATH = "hc3_v8_normalized_results.json"


def normalize_text(text):
    text = str(text)

    # Remove HTML tags.
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove markdown emphasis markers.
    text = re.sub(r"\*\*", " ", text)
    text = re.sub(r"(?<!\w)\*(?!\w)", " ", text)

    # Normalize curly quotes to straight quotes.
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("‘", "'")
    text = text.replace("’", "'")

    # Normalize different dash characters.
    text = text.replace("—", "-")
    text = text.replace("–", "-")

    # Normalize repeated whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


df = pd.read_csv(CSV_PATH)

train = df[df["split"] == "train"].copy()
valid = df[df["split"] == "valid"].copy()
test = df[df["split"] == "test"].copy()

texts_train = [
    normalize_text(x)
    for x in train["text"]
]

texts_valid = [
    normalize_text(x)
    for x in valid["text"]
]

texts_test = [
    normalize_text(x)
    for x in test["text"]
]

y_train = train["ai"].astype(int).to_numpy()
y_valid = valid["ai"].astype(int).to_numpy()
y_test = test["ai"].astype(int).to_numpy()

print("Train:", len(train))
print("Valid:", len(valid))
print("Test :", len(test))

print("\nFitting normalized character TF-IDF...")

vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 6),
    min_df=1,
    max_features=60000,
    sublinear_tf=True,
)

X_train = vectorizer.fit_transform(texts_train)
X_valid = vectorizer.transform(texts_valid)
X_test = vectorizer.transform(texts_test)

print("Train matrix:", X_train.shape)
print("Valid matrix:", X_valid.shape)
print("Test matrix :", X_test.shape)

print("\nTraining LinearSVC C=10...")

model = LinearSVC(
    C=10.0,
)

model.fit(X_train, y_train)

train_scores = model.decision_function(X_train)
valid_scores = model.decision_function(X_valid)
test_scores = model.decision_function(X_test)


def choose_threshold(y_true, scores):
    negatives = scores[y_true == 0]
    positives = scores[y_true == 1]

    best = None

    for threshold in np.unique(scores):
        pred = (scores >= threshold).astype(int)

        fp = int(
            ((pred == 1) & (y_true == 0)).sum()
        )

        tp = int(
            ((pred == 1) & (y_true == 1)).sum()
        )

        fpr = fp / len(negatives)
        recall = tp / len(positives)

        if fpr <= 0.05 + 1e-12:
            candidate = (
                recall,
                -fpr,
                threshold,
            )

            if best is None or candidate > best[0]:
                best = (
                    candidate,
                    float(threshold),
                    float(fpr),
                    float(recall),
                    fp,
                    tp,
                )

    if best is None:
        return {
            "threshold": float(
                np.max(negatives) + 1e-12
            ),
            "fpr": 0.0,
            "recall": 0.0,
            "fp": 0,
            "tp": 0,
        }

    return {
        "threshold": best[1],
        "fpr": best[2],
        "recall": best[3],
        "fp": best[4],
        "tp": best[5],
    }


def metrics(y_true, scores, threshold):
    pred = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1],
    ).ravel()

    return {
        "n": int(len(y_true)),
        "accuracy": float(
            accuracy_score(y_true, pred)
        ),
        "precision": float(
            precision_score(
                y_true,
                pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                pred,
                zero_division=0,
            )
        ),
        "fpr": float(
            fp / (fp + tn)
        ) if (fp + tn) else 0.0,
        "auroc": float(
            roc_auc_score(y_true, scores)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


threshold_info = choose_threshold(
    y_valid,
    valid_scores,
)

threshold = threshold_info["threshold"]

print("\nValidation threshold selection")
print(json.dumps(threshold_info, indent=2))

train_metrics = metrics(
    y_train,
    train_scores,
    threshold,
)

valid_metrics = metrics(
    y_valid,
    valid_scores,
    threshold,
)

test_metrics = metrics(
    y_test,
    test_scores,
    threshold,
)

print("\nTRAIN")
print(json.dumps(train_metrics, indent=2))

print("\nVALID")
print(json.dumps(valid_metrics, indent=2))

print("\nCOMPARISON TEST")
print(json.dumps(test_metrics, indent=2))


print("\nTEST BY GENERATION MODEL")

by_model = {}

for generation_model in sorted(
    test["generation_model"].dropna().unique()
):
    mask = (
        test["generation_model"].astype(str)
        == str(generation_model)
    ).to_numpy()

    m = metrics(
        y_test[mask],
        test_scores[mask],
        threshold,
    )

    by_model[str(generation_model)] = m

    print(
        generation_model,
        "| N=",
        m["n"],
        "| ACC=",
        round(m["accuracy"], 4),
        "| RECALL=",
        round(m["recall"], 4),
        "| FPR=",
        round(m["fpr"], 4),
        "| AUROC=",
        round(m["auroc"], 4),
    )


print("\nEXTERNAL ARTICLES")

for path in [
    "article_human.txt",
    "article_ai.txt",
    "article_mixed.txt",
    "article.txt",
]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
    except FileNotFoundError:
        print(path, "| NOT FOUND")
        continue

    if not raw:
        print(path, "| EMPTY")
        continue

    text = normalize_text(raw)

    score = float(
        model.decision_function(
            vectorizer.transform([text])
        )[0]
    )

    print(
        path,
        "| words=",
        len(raw.split()),
        "| score=",
        round(score, 6),
        "| prediction=",
        "AI" if score >= threshold else "HUMAN",
    )


artifact = {
    "feature_set": "normalized_char_26",
    "analyzer": "char_wb",
    "ngram_range": [2, 6],
    "max_features": 60000,
    "sublinear_tf": True,
    "model": "LinearSVC",
    "C": 10.0,
    "normalization": [
        "remove HTML tags",
        "remove Markdown emphasis markers",
        "normalize curly quotes",
        "normalize dash variants",
        "normalize whitespace",
    ],
    "threshold_rule": (
        "max validation recall subject to "
        "validation FPR <= 5%"
    ),
    "threshold": float(threshold),
    "threshold_selection": threshold_info,
    "train_metrics": train_metrics,
    "valid_metrics": valid_metrics,
    "comparison_test_metrics": test_metrics,
    "test_by_generation_model": by_model,
}

joblib.dump(
    {
        "vectorizer": vectorizer,
        "model": model,
        "threshold": threshold,
        "normalize_text": normalize_text,
        "artifact": artifact,
    },
    MODEL_PATH,
)

with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        artifact,
        f,
        indent=2,
    )

print("\nSaved:", MODEL_PATH)
print("Saved:", RESULTS_PATH)
