import csv
import json
from pathlib import Path

import numpy as np

from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import GroupKFold


DATA = "dataset/human_detectors_grouped.csv"
OUTPUT = "hc3_optimization.json"

C_VALUES = [
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
    3.0,
    10.0,
]

FEATURE_SETS = [
    "word",
    "char",
    "word_char",
]


def load_data():
    with open(DATA, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    train = [r for r in rows if r["split"] == "train"]

    texts = [r["text"] for r in train]
    labels = np.array([int(r["ai"]) for r in train])
    groups = np.array([r["group"] for r in train])

    return texts, labels, groups


def make_vectorizers(feature_set):
    word = None
    char = None

    if feature_set in {"word", "word_char"}:
        word = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            lowercase=True,
            strip_accents="unicode",
            sublinear_tf=True,
            min_df=1,
            max_features=30000,
        )

    if feature_set in {"char", "word_char"}:
        char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=True,
            sublinear_tf=True,
            min_df=1,
            max_features=40000,
        )

    return word, char


def build_matrix(
    train_texts,
    test_texts,
    feature_set,
):
    word, char = make_vectorizers(feature_set)

    pieces_train = []
    pieces_test = []

    if word is not None:
        pieces_train.append(word.fit_transform(train_texts))
        pieces_test.append(word.transform(test_texts))

    if char is not None:
        pieces_train.append(char.fit_transform(train_texts))
        pieces_test.append(char.transform(test_texts))

    X_train = hstack(
        pieces_train,
        format="csr",
    )

    X_test = hstack(
        pieces_test,
        format="csr",
    )

    return X_train, X_test


def evaluate_cv(texts, labels, groups, feature_set, C):
    splitter = GroupKFold(n_splits=3)

    fold_scores = []

    for train_idx, valid_idx in splitter.split(
        texts,
        labels,
        groups,
    ):
        train_texts = [texts[i] for i in train_idx]
        valid_texts = [texts[i] for i in valid_idx]

        X_train, X_valid = build_matrix(
            train_texts,
            valid_texts,
            feature_set,
        )

        y_train = labels[train_idx]
        y_valid = labels[valid_idx]

        model = LogisticRegression(
            C=C,
            solver="liblinear",
            max_iter=5000,
            class_weight="balanced",
            random_state=20260910,
        )

        model.fit(X_train, y_train)

        scores = model.predict_proba(X_valid)[:, 1]
        predictions = (scores >= 0.5).astype(int)

        fold_scores.append({
            "accuracy": accuracy_score(
                y_valid,
                predictions,
            ),
            "f1": f1_score(
                y_valid,
                predictions,
                zero_division=0,
            ),
            "auroc": roc_auc_score(
                y_valid,
                scores,
            ),
        })

    return {
        "accuracy_mean": float(
            np.mean([x["accuracy"] for x in fold_scores])
        ),
        "f1_mean": float(
            np.mean([x["f1"] for x in fold_scores])
        ),
        "auroc_mean": float(
            np.mean([x["auroc"] for x in fold_scores])
        ),
        "folds": fold_scores,
    }


texts, labels, groups = load_data()

print("Training rows:", len(texts))
print("Groups:", len(set(groups)))

results = []

for feature_set in FEATURE_SETS:
    for C in C_VALUES:
        print(
            "Testing",
            feature_set,
            "C=",
            C,
        )

        scores = evaluate_cv(
            texts,
            labels,
            groups,
            feature_set,
            C,
        )

        results.append({
            "feature_set": feature_set,
            "C": C,
            **scores,
        })

results.sort(
    key=lambda x: (
        x["auroc_mean"],
        x["f1_mean"],
        x["accuracy_mean"],
    ),
    reverse=True,
)

output = {
    "data": DATA,
    "cv": "3-fold GroupKFold",
    "group_definition": "source + title + issue + generation_model",
    "results": results,
    "best": results[0],
}

Path(OUTPUT).write_text(
    json.dumps(output, indent=2),
    encoding="utf-8",
)

print("\nBEST MODEL")
print(json.dumps(results[0], indent=2))

print("\nTOP 10")

for item in results[:10]:
    print(
        item["feature_set"],
        "C=",
        item["C"],
        "AUROC=",
        round(item["auroc_mean"], 4),
        "F1=",
        round(item["f1_mean"], 4),
        "accuracy=",
        round(item["accuracy_mean"], 4),
    )
