import csv
import json
from pathlib import Path

import numpy as np

from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score


DATA = "dataset/human_detectors_grouped.csv"
OUTPUT = "hc3_v4_search.json"


C_VALUES = [
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
    "word_char_style",
]


MODELS = [
    "logistic",
    "linear_svm",
]


STYLE_NAMES = [
    "log_word_count",
    "type_token_ratio",
    "mean_word_length",
    "mean_sentence_length",
    "sentence_length_cv",
    "repeated_trigram_fraction",
    "repeated_sentence_fraction",
    "digit_token_fraction",
    "url_rate",
    "punctuation_rate",
]


def tokens(text):
    import re
    return re.findall(r"\b\w+(?:['’]\w+)?\b", text.lower())


def style_features(text):
    t = tokens(text)

    if not t:
        return np.zeros(len(STYLE_NAMES))

    import re

    sentences = [
        tokens(part)
        for part in re.split(r"[.!?]+", text)
    ]

    sentences = [
        s for s in sentences
        if s
    ]

    lengths = np.array(
        [len(s) for s in sentences],
        dtype=float,
    )

    trigrams = list(
        zip(t, t[1:], t[2:])
    )

    n = len(t)

    return np.array([
        np.log1p(n),
        len(set(t)) / n,
        np.mean([len(x) for x in t]),
        lengths.mean() if len(lengths) else 0.0,
        lengths.std() / max(lengths.mean(), 1.0)
            if len(lengths) else 0.0,
        1.0 - len(set(trigrams)) / len(trigrams)
            if trigrams else 0.0,
        1.0 - len(set(map(tuple, sentences))) / len(sentences)
            if sentences else 0.0,
        sum(
            any(c.isdigit() for c in x)
            for x in t
        ) / n,
        len(re.findall(r"https?://\S+", text)) / n,
        sum(
            c in ",;:!?—-()"
            for c in text
        ) / max(len(text), 1),
    ])


def load():
    with open(
        DATA,
        encoding="utf-8",
        newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    rows = [
        r for r in rows
        if r["split"] == "train"
    ]

    texts = [r["text"] for r in rows]
    y = np.array(
        [int(r["ai"]) for r in rows]
    )
    groups = np.array(
        [r["group"] for r in rows]
    )

    return texts, y, groups


def build_features(
    train_texts,
    test_texts,
    feature_set,
):
    pieces_train = []
    pieces_test = []

    if feature_set in {
        "word",
        "word_char",
        "word_char_style",
    }:
        word = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            lowercase=True,
            strip_accents="unicode",
            sublinear_tf=True,
            min_df=1,
            max_features=40000,
        )

        pieces_train.append(
            word.fit_transform(train_texts)
        )

        pieces_test.append(
            word.transform(test_texts)
        )

    if feature_set in {
        "char",
        "word_char",
        "word_char_style",
    }:
        char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=True,
            sublinear_tf=True,
            min_df=1,
            max_features=40000,
        )

        pieces_train.append(
            char.fit_transform(train_texts)
        )

        pieces_test.append(
            char.transform(test_texts)
        )

    if feature_set == "word_char_style":
        a = np.vstack([
            style_features(x)
            for x in train_texts
        ])

        b = np.vstack([
            style_features(x)
            for x in test_texts
        ])

        mean = a.mean(axis=0)
        scale = a.std(axis=0)
        scale[scale < 1e-12] = 1.0

        a = (a - mean) / scale
        b = (b - mean) / scale

        pieces_train.append(
            csr_matrix(a)
        )

        pieces_test.append(
            csr_matrix(b)
        )

    return (
        hstack(
            pieces_train,
            format="csr"
        ),
        hstack(
            pieces_test,
            format="csr"
        ),
    )


def run_model(model_name, C, X_train, y_train):
    if model_name == "logistic":
        model = LogisticRegression(
            C=C,
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            random_state=20260910,
        )

        model.fit(
            X_train,
            y_train,
        )

        return model.predict_proba(
            X_train
        )[:, 1], model

    model = LinearSVC(
        C=C,
        class_weight="balanced",
        max_iter=10000,
        random_state=20260910,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model.decision_function(
        X_train
    ), model


def score_model(model_name, model, X):
    if model_name == "logistic":
        return model.predict_proba(X)[:, 1]

    return model.decision_function(X)


def evaluate(
    texts,
    y,
    groups,
    feature_set,
    model_name,
    C,
):
    splitter = GroupKFold(
        n_splits=3
    )

    fold_auc = []

    for train_idx, valid_idx in splitter.split(
        texts,
        y,
        groups,
    ):
        train_texts = [
            texts[i]
            for i in train_idx
        ]

        valid_texts = [
            texts[i]
            for i in valid_idx
        ]

        X_train, X_valid = build_features(
            train_texts,
            valid_texts,
            feature_set,
        )

        y_train = y[train_idx]
        y_valid = y[valid_idx]

        if model_name == "logistic":
            model = LogisticRegression(
                C=C,
                solver="liblinear",
                class_weight="balanced",
                max_iter=5000,
                random_state=20260910,
            )
        else:
            model = LinearSVC(
                C=C,
                class_weight="balanced",
                max_iter=10000,
                random_state=20260910,
            )

        model.fit(
            X_train,
            y_train,
        )

        scores = score_model(
            model_name,
            model,
            X_valid,
        )

        fold_auc.append(
            roc_auc_score(
                y_valid,
                scores,
            )
        )

    return {
        "auroc_mean": float(
            np.mean(fold_auc)
        ),
        "auroc_std": float(
            np.std(fold_auc)
        ),
        "fold_auroc": [
            float(x)
            for x in fold_auc
        ],
    }


texts, y, groups = load()

print("Training rows:", len(texts))
print("Training groups:", len(set(groups)))

results = []

for feature_set in FEATURE_SETS:
    for model_name in MODELS:
        for C in C_VALUES:
            print(
                feature_set,
                model_name,
                "C=",
                C,
            )

            result = evaluate(
                texts,
                y,
                groups,
                feature_set,
                model_name,
                C,
            )

            results.append({
                "feature_set": feature_set,
                "model": model_name,
                "C": C,
                **result,
            })


results.sort(
    key=lambda x: (
        x["auroc_mean"],
        -x["auroc_std"],
    ),
    reverse=True,
)


output = {
    "data": DATA,
    "cv": "3-fold GroupKFold",
    "results": results,
    "best": results[0],
}

Path(OUTPUT).write_text(
    json.dumps(
        output,
        indent=2,
    ),
    encoding="utf-8",
)

print("\nBEST")
print(json.dumps(
    results[0],
    indent=2,
))

print("\nTOP 10")

for x in results[:10]:
    print(
        x["feature_set"],
        "|",
        x["model"],
        "| C=",
        x["C"],
        "| AUROC=",
        round(x["auroc_mean"], 5),
        "| SD=",
        round(x["auroc_std"], 5),
    )
