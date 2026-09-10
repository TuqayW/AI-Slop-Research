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

CSV_PATH = "dataset/human_detectors_grouped.csv"
MODEL_PATH = "hc3_v5_best_model.joblib"
RESULTS_PATH = "hc3_v5_best_results.json"

df = pd.read_csv(CSV_PATH)

train = df[df["split"] == "train"].copy()
valid = df[df["split"] == "valid"].copy()
test = df[df["split"] == "test"].copy()

print("Train:", len(train))
print("Valid:", len(valid))
print("Test :", len(test))

texts_train = train["text"].astype(str).tolist()
texts_valid = valid["text"].astype(str).tolist()
texts_test = test["text"].astype(str).tolist()

y_train = pd.to_numeric(train["ai"]).astype(int).to_numpy()
y_valid = pd.to_numeric(valid["ai"]).astype(int).to_numpy()
y_test = pd.to_numeric(test["ai"]).astype(int).to_numpy()

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
    class_weight=None,
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

print("\nVALID")
print(json.dumps(valid_metrics, indent=2))

print("\nTEST")
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


print("\nTEST FALSE POSITIVES")

test_copy = test.copy()
test_copy["score"] = test_scores
test_copy["pred"] = (
    test_scores >= threshold
).astype(int)

fps = test_copy[
    (test_copy["ai"] == 0)
    & (test_copy["pred"] == 1)
]

print("False positives:", len(fps))

for _, row in fps.iterrows():
    print(
        "\nID:",
        row.get("id"),
        "\nSource:",
        row.get("source"),
        "\nTitle:",
        row.get("title"),
        "\nGeneration condition:",
        row.get("generation_model"),
        "\nScore:",
        row["score"],
    )


artifact = {
    "feature_set": "char_26",
    "analyzer": "char_wb",
    "ngram_range": [2, 6],
    "max_features": 60000,
    "sublinear_tf": True,
    "model": "LinearSVC",
    "C": 10.0,
    "class_weight": None,
    "threshold_rule": (
        "max validation recall subject to "
        "validation FPR <= 5%"
    ),
    "threshold": float(threshold),
    "threshold_selection": threshold_info,
    "train_metrics": train_metrics,
    "valid_metrics": valid_metrics,
    "test_metrics": test_metrics,
    "test_by_generation_model": by_model,
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
