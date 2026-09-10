import csv
import json
from pathlib import Path

import numpy as np
import joblib

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


DATA = "dataset/human_detectors_grouped.csv"
MODEL = "hc3_best_model.joblib"
REPORT = "hc3_best_results.json"

C = 10.0
MAX_FPR = 0.05


def metrics(y, scores, threshold):
    pred = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[0, 1],
    ).ravel()

    return {
        "n": int(len(y)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "accuracy": float(
            accuracy_score(y, pred)
        ),
        "precision": float(
            precision_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        "fpr": float(
            fp / (fp + tn)
        ) if (fp + tn) else None,
        "auroc": float(
            roc_auc_score(y, scores)
        ) if len(np.unique(y)) == 2 else None,
    }


def choose_threshold(y, scores):
    best = None

    for threshold in np.unique(scores):
        result = metrics(
            y,
            scores,
            float(threshold),
        )

        if result["fpr"] is None:
            continue

        if result["fpr"] > MAX_FPR:
            continue

        key = (
            result["recall"],
            result["f1"],
            result["auroc"],
            -result["fpr"],
            -float(threshold),
        )

        if best is None or key > best[0]:
            best = (
                key,
                float(threshold),
            )

    if best is None:
        return 1.0

    return best[1]


with open(DATA, encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))


def get(split):
    return [
        r for r in rows
        if r["split"] == split
    ]


train = get("train")
valid = get("valid")
test = get("test")


train_texts = [r["text"] for r in train]
valid_texts = [r["text"] for r in valid]
test_texts = [r["text"] for r in test]

y_train = np.array(
    [int(r["ai"]) for r in train]
)

y_valid = np.array(
    [int(r["ai"]) for r in valid]
)

y_test = np.array(
    [int(r["ai"]) for r in test]
)


word = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    lowercase=True,
    strip_accents="unicode",
    sublinear_tf=True,
    min_df=1,
    max_features=50000,
)

char = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    lowercase=True,
    sublinear_tf=True,
    min_df=1,
    max_features=50000,
)


X_train = hstack([
    word.fit_transform(train_texts),
    char.fit_transform(train_texts),
])

X_valid = hstack([
    word.transform(valid_texts),
    char.transform(valid_texts),
])

X_test = hstack([
    word.transform(test_texts),
    char.transform(test_texts),
])


model = LogisticRegression(
    C=C,
    solver="liblinear",
    max_iter=5000,
    class_weight="balanced",
    random_state=20260910,
)

model.fit(
    X_train,
    y_train,
)


train_scores = model.predict_proba(
    X_train
)[:, 1]

valid_scores = model.predict_proba(
    X_valid
)[:, 1]

test_scores = model.predict_proba(
    X_test
)[:, 1]


threshold = choose_threshold(
    y_valid,
    valid_scores,
)


report = {
    "configuration": {
        "feature_set": "word_char",
        "C": C,
        "word_ngrams": [1, 3],
        "character_ngrams": [3, 5],
        "character_analyzer": "char_wb",
        "validation_max_fpr": MAX_FPR,
    },
    "threshold": threshold,
    "train": metrics(
        y_train,
        train_scores,
        threshold,
    ),
    "valid": metrics(
        y_valid,
        valid_scores,
        threshold,
    ),
    "test": metrics(
        y_test,
        test_scores,
        threshold,
    ),
    "test_by_generation_model": {},
    "warning": (
        "Experimental research detector. "
        "These results are limited to this dataset and "
        "should not be interpreted as general web accuracy."
    ),
}


for generation_model in sorted({
    r["generation_model"]
    for r in test
}):
    indices = [
        i
        for i, r in enumerate(test)
        if r["generation_model"] == generation_model
    ]

    yy = y_test[indices]
    ss = test_scores[indices]

    report["test_by_generation_model"][
        generation_model
    ] = metrics(
        yy,
        ss,
        threshold,
    )


artifact = {
    "version": 3,
    "min_words": 80,
    "threshold": threshold,
    "word_vectorizer": word,
    "char_vectorizer": char,
    "model": model,
    "warning": report["warning"],
}


joblib.dump(
    artifact,
    MODEL,
)

Path(REPORT).write_text(
    json.dumps(
        report,
        indent=2,
    ),
    encoding="utf-8",
)


print(json.dumps(
    report,
    indent=2,
))
