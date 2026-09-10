import argparse
import json
from pathlib import Path

import numpy as np
import joblib

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
from scipy.sparse import hstack


MIN_WORDS = 80


def words(text):
    import re
    return re.findall(r"\b\w+(?:['’]\w+)?\b", text.lower())


def load_data(path):
    import csv

    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    required = {"text", "group", "split", "ai"}

    if not rows:
        raise ValueError("CSV is empty.")

    if not required.issubset(rows[0].keys()):
        raise ValueError(f"Missing columns. Required: {sorted(required)}")

    for i, row in enumerate(rows, start=2):
        if row["ai"] not in {"0", "1"}:
            raise ValueError(f"Row {i}: ai must be 0 or 1.")

        if row["split"] not in {"train", "valid", "test"}:
            raise ValueError(f"Row {i}: invalid split.")

        if not row["group"].strip():
            raise ValueError(f"Row {i}: empty group.")

        if len(words(row["text"])) < MIN_WORDS:
            raise ValueError(
                f"Row {i}: requires at least {MIN_WORDS} words."
            )

    return rows


def make_features(train_texts):
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
        min_df=1,
        max_df=0.98,
        max_features=50000,
    )

    char = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        lowercase=True,
        sublinear_tf=True,
        min_df=1,
        max_features=50000,
    )

    word.fit(train_texts)
    char.fit(train_texts)

    return word, char


def transform(texts, word, char):
    X_word = word.transform(texts)
    X_char = char.transform(texts)
    return hstack([X_word, X_char], format="csr")


def get_metrics(y_true, scores, threshold):
    y_true = np.asarray(y_true)
    predicted = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predicted,
        labels=[0, 1],
    ).ravel()

    result = {
        "n": int(len(y_true)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(
            precision_score(y_true, predicted, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predicted, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, predicted, zero_division=0)
        ),
        "fpr": float(
            fp / (fp + tn)
        ) if (fp + tn) else None,
        "auroc": float(
            roc_auc_score(y_true, scores)
        ) if len(np.unique(y_true)) == 2 else None,
    }

    return result


def choose_threshold(y_true, scores, max_fpr):
    candidates = np.unique(scores)

    best = None

    for threshold in candidates:
        result = get_metrics(
            y_true,
            scores,
            float(threshold),
        )

        if result["fpr"] is None:
            continue

        if result["fpr"] > max_fpr:
            continue

        key = (
            result["recall"],
            result["f1"],
            result["auroc"] if result["auroc"] is not None else -1,
            -result["fpr"],
            -float(threshold),
        )

        if best is None or key > best[0]:
            best = (key, float(threshold))

    if best is None:
        return 1.0

    return best[1]


def train(data_path, model_path, report_path, C, max_fpr):
    rows = load_data(data_path)

    train_rows = [r for r in rows if r["split"] == "train"]
    valid_rows = [r for r in rows if r["split"] == "valid"]
    test_rows = [r for r in rows if r["split"] == "test"]

    train_texts = [r["text"] for r in train_rows]
    valid_texts = [r["text"] for r in valid_rows]
    test_texts = [r["text"] for r in test_rows]

    y_train = np.array([int(r["ai"]) for r in train_rows])
    y_valid = np.array([int(r["ai"]) for r in valid_rows])
    y_test = np.array([int(r["ai"]) for r in test_rows])

    word, char = make_features(train_texts)

    X_train = transform(train_texts, word, char)
    X_valid = transform(valid_texts, word, char)
    X_test = transform(test_texts, word, char)

    model = LogisticRegression(
        C=C,
        solver="liblinear",
        max_iter=5000,
        class_weight="balanced",
        random_state=20260910,
    )

    model.fit(X_train, y_train)

    train_scores = model.predict_proba(X_train)[:, 1]
    valid_scores = model.predict_proba(X_valid)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]

    threshold = choose_threshold(
        y_valid,
        valid_scores,
        max_fpr,
    )

    report = {
        "C": C,
        "max_fpr": max_fpr,
        "threshold": threshold,
        "features": {
            "word_ngrams": [1, 3],
            "char_ngrams": [3, 5],
        },
        "train": get_metrics(
            y_train,
            train_scores,
            threshold,
        ),
        "valid": get_metrics(
            y_valid,
            valid_scores,
            threshold,
        ),
        "test": get_metrics(
            y_test,
            test_scores,
            threshold,
        ),
        "warning": (
            "Experimental model. The supplied pilot has only 24 "
            "documents and does not represent the modern web."
        ),
    }

    artifact = {
        "version": 2,
        "min_words": MIN_WORDS,
        "threshold": threshold,
        "word_vectorizer": word,
        "char_vectorizer": char,
        "model": model,
        "warning": report["warning"],
    }

    joblib.dump(artifact, model_path)

    Path(report_path).write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))


def predict(model_path, text_path):
    artifact = joblib.load(model_path)

    text = Path(text_path).read_text(encoding="utf-8")

    count = len(words(text))

    if count < artifact["min_words"]:
        print(json.dumps({
            "status": "abstain",
            "reason": "too_short",
            "word_count": count,
        }, indent=2))
        return

    X = transform(
        [text],
        artifact["word_vectorizer"],
        artifact["char_vectorizer"],
    )

    score = float(
        artifact["model"].predict_proba(X)[0, 1]
    )

    threshold = float(artifact["threshold"])

    print(json.dumps({
        "status": "scored",
        "word_count": count,
        "ai_score": score,
        "threshold": threshold,
        "flag": bool(score >= threshold),
        "warning": artifact["warning"],
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    train_parser = sub.add_parser("train")
    train_parser.add_argument("data")
    train_parser.add_argument("model")
    train_parser.add_argument("report")
    train_parser.add_argument(
        "--C",
        type=float,
        default=1.0,
    )
    train_parser.add_argument(
        "--max-fpr",
        type=float,
        default=0.05,
    )

    predict_parser = sub.add_parser("predict")
    predict_parser.add_argument("model")
    predict_parser.add_argument("text")

    args = parser.parse_args()

    if args.command == "train":
        train(
            args.data,
            args.model,
            args.report,
            args.C,
            args.max_fpr,
        )

    elif args.command == "predict":
        predict(
            args.model,
            args.text,
        )


if __name__ == "__main__":
    main()
