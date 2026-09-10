import json
import re
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack
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


CSV_PATH = "dataset/combined_v11.csv"
MODEL_PATH = "hc3_v11_combined_model.joblib"
RESULTS_PATH = "hc3_v11_combined_results.json"


def normalize_text(text):
    text = str(text)

    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\*\*", " ", text)
    text = re.sub(r"(?<!\w)\*(?!\w)", " ", text)

    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("‘", "'")
    text = text.replace("’", "'")
    text = text.replace("—", "-")
    text = text.replace("–", "-")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


df = pd.read_csv(CSV_PATH)

train = df[df["split"] == "train"].copy()
valid = df[df["split"] == "valid"].copy()

texts_train = [
    normalize_text(x)
    for x in train["text"]
]

texts_valid = [
    normalize_text(x)
    for x in valid["text"]
]

y_train = train["ai"].astype(int).to_numpy()
y_valid = valid["ai"].astype(int).to_numpy()


print("Train:", len(train))
print("Valid:", len(valid))

print("\nTRAIN LABELS:")
print(train["ai"].value_counts())

print("\nVALID LABELS:")
print(valid["ai"].value_counts())


print("\nFitting WORD TF-IDF...")

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    min_df=2,
    max_features=50000,
    sublinear_tf=True,
)

X_train_word = word_vectorizer.fit_transform(
    texts_train
)

X_valid_word = word_vectorizer.transform(
    texts_valid
)


print("Word matrix:", X_train_word.shape)


print("\nFitting CHAR TF-IDF...")

char_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 6),
    min_df=2,
    max_features=80000,
    sublinear_tf=True,
)

X_train_char = char_vectorizer.fit_transform(
    texts_train
)

X_valid_char = char_vectorizer.transform(
    texts_valid
)

print("Char matrix:", X_train_char.shape)


X_train = hstack(
    [X_train_word, X_train_char],
    format="csr",
)

X_valid = hstack(
    [X_valid_word, X_valid_char],
    format="csr",
)

print("\nCombined matrix:", X_train.shape)


print("\nTraining LinearSVC C=3...")

model = LinearSVC(
    C=3.0,
)

model.fit(
    X_train,
    y_train,
)

train_scores = model.decision_function(
    X_train
)

valid_scores = model.decision_function(
    X_valid
)


def choose_threshold(y_true, scores):
    negatives = scores[y_true == 0]
    positives = scores[y_true == 1]

    best = None

    for threshold in np.unique(scores):

        pred = (
            scores >= threshold
        ).astype(int)

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


def metrics(
    y_true,
    scores,
    threshold,
):

    pred = (
        scores >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1],
    ).ravel()

    return {
        "n": int(len(y_true)),
        "accuracy": float(
            accuracy_score(
                y_true,
                pred,
            )
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
            roc_auc_score(
                y_true,
                scores,
            )
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

print("\nVALIDATION THRESHOLD")
print(
    json.dumps(
        threshold_info,
        indent=2,
    )
)


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


print("\nTRAIN")
print(
    json.dumps(
        train_metrics,
        indent=2,
    )
)

print("\nVALID")
print(
    json.dumps(
        valid_metrics,
        indent=2,
    )
)


print("\nVALID BY ORIGIN")

for origin in sorted(
    valid["origin"].unique()
):

    mask = (
        valid["origin"]
        .astype(str)
        == str(origin)
    ).to_numpy()

    m = metrics(
        y_valid[mask],
        valid_scores[mask],
        threshold,
    )

    print(
        origin,
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


print("\nVALID BY SOURCE")

for source in sorted(
    valid["source"].unique()
):

    mask = (
        valid["source"]
        .astype(str)
        == str(source)
    ).to_numpy()

    m = metrics(
        y_valid[mask],
        valid_scores[mask],
        threshold,
    )

    print(
        source,
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

external_files = [
    "article_human.txt",
    "article_ai.txt",
    "article_mixed.txt",
    "article.txt",
]

for path in external_files:

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:
            raw = f.read().strip()

    except FileNotFoundError:

        print(
            path,
            "| NOT FOUND",
        )

        continue

    if not raw:

        print(
            path,
            "| EMPTY",
        )

        continue

    text = normalize_text(raw)

    word_X = word_vectorizer.transform(
        [text]
    )

    char_X = char_vectorizer.transform(
        [text]
    )

    X = hstack(
        [word_X, char_X],
        format="csr",
    )

    score = float(
        model.decision_function(X)[0]
    )

    print(
        path,
        "| words=",
        len(raw.split()),
        "| score=",
        round(score, 6),
        "| prediction=",
        "AI"
        if score >= threshold
        else "HUMAN",
    )


artifact = {
    "feature_set": "normalized_word_char",
    "word_analyzer": "word",
    "word_ngram_range": [1, 3],
    "word_max_features": 50000,
    "char_analyzer": "char_wb",
    "char_ngram_range": [2, 6],
    "char_max_features": 80000,
    "model": "LinearSVC",
    "C": 3.0,
    "threshold_rule": (
        "maximum validation recall subject "
        "to validation FPR <= 5%"
    ),
    "threshold": float(threshold),
    "threshold_selection": threshold_info,
    "train_metrics": train_metrics,
    "valid_metrics": valid_metrics,
}


joblib.dump(
    {
        "word_vectorizer": word_vectorizer,
        "char_vectorizer": char_vectorizer,
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
