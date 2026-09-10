import json
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

TRAIN_PATH = "dataset/hc3_short_v6.csv"
FULL_PATH = "dataset/human_detectors_grouped.csv"

MODEL_PATH = "hc3_v6_short_model.joblib"
RESULTS_PATH = "hc3_v6_short_results.json"


def make_chunks(text, min_words=180, max_words=260):
    words = str(text).split()
    chunks = []
    start = 0

    while start < len(words):
        remaining = len(words) - start

        if remaining < min_words:
            break

        end = min(start + max_words, len(words))
        chunk = words[start:end]

        if len(chunk) >= min_words:
            chunks.append(" ".join(chunk))

        start += max_words

    return chunks


short_train = pd.read_csv(TRAIN_PATH)
full = pd.read_csv(FULL_PATH)

valid_full = full[full["split"] == "valid"].copy()
test_full = full[full["split"] == "test"].copy()

valid_rows = []

for _, row in valid_full.iterrows():
    chunks = make_chunks(row["text"])

    for i, chunk in enumerate(chunks):
        valid_rows.append({
            "original_id": row["id"],
            "prompt_id": row["prompt_id"],
            "source": row["source"],
            "generation_model": row["generation_model"],
            "ai": int(row["ai"]),
            "chunk_id": i,
            "text": chunk,
        })

valid = pd.DataFrame(valid_rows)

print("Short train:", len(short_train))
print("Short valid:", len(valid))
print("Full test :", len(test_full))

texts_train = short_train["text"].astype(str).tolist()
texts_valid = valid["text"].astype(str).tolist()
texts_test = test_full["text"].astype(str).tolist()

y_train = short_train["ai"].astype(int).to_numpy()
y_valid = valid["ai"].astype(int).to_numpy()
y_test = test_full["ai"].astype(int).to_numpy()

print("\nTraining label counts:")
print(short_train["ai"].value_counts())

print("\nValidation label counts:")
print(valid["ai"].value_counts())


print("\nFitting character TF-IDF...")

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

        fp = int(((pred == 1) & (y_true == 0)).sum())
        tp = int(((pred == 1) & (y_true == 1)).sum())

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
        threshold = float(np.max(negatives) + 1e-12)

        return {
            "threshold": threshold,
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


threshold_info = choose_threshold(
    y_valid,
    valid_scores,
)

threshold = threshold_info["threshold"]

print("\nValidation threshold selection")
print(json.dumps(threshold_info, indent=2))


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

print("\nSHORT VALID")
print(json.dumps(valid_metrics, indent=2))

print("\nFROZEN FULL TEST")
print(json.dumps(test_metrics, indent=2))


print("\nFULL TEST BY GENERATION MODEL")

for generation_model in sorted(
    test_full["generation_model"].dropna().unique()
):
    mask = (
        test_full["generation_model"].astype(str)
        == str(generation_model)
    ).to_numpy()

    m = metrics(
        y_test[mask],
        test_scores[mask],
        threshold,
    )

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

external_files = [
    "article_human.txt",
    "article_ai.txt",
    "article_mixed.txt",
    "article.txt",
]

for path in external_files:
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except FileNotFoundError:
        print(path, "| NOT FOUND")
        continue

    if not text:
        print(path, "| EMPTY")
        continue

    score = float(
        model.decision_function(
            vectorizer.transform([text])
        )[0]
    )

    print(
        path,
        "| words=",
        len(text.split()),
        "| score=",
        round(score, 6),
        "| prediction=",
        "AI" if score >= threshold else "HUMAN",
    )


artifact = {
    "feature_set": "char_26",
    "analyzer": "char_wb",
    "ngram_range": [2, 6],
    "max_features": 60000,
    "model": "LinearSVC",
    "C": 10.0,
    "training": "180 benchmark train documents chunked to 180-260 words",
    "threshold_rule": (
        "max validation recall subject to "
        "short-validation FPR <= 5%"
    ),
    "threshold": float(threshold),
    "threshold_selection": threshold_info,
    "train_metrics": train_metrics,
    "short_valid_metrics": valid_metrics,
    "frozen_full_test_metrics": test_metrics,
}

joblib.dump(
    {
        "vectorizer": vectorizer,
        "model": model,
        "threshold": threshold,
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
