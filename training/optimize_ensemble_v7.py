import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

CSV_PATH = "dataset/human_detectors_grouped.csv"

V5_MODEL = "hc3_v5_best_model.joblib"
V6_MODEL = "hc3_v6_short_model.joblib"

OUT_PATH = "hc3_v7_ensemble_search.json"


df = pd.read_csv(CSV_PATH)


def load_model(path):
    bundle = joblib.load(path)
    return (
        bundle["vectorizer"],
        bundle["model"],
        float(bundle["threshold"]),
    )


v5_vectorizer, v5_model, v5_threshold = load_model(V5_MODEL)
v6_vectorizer, v6_model, v6_threshold = load_model(V6_MODEL)


def get_scores(model_vectorizer, model, texts):
    X = model_vectorizer.transform(texts)
    return model.decision_function(X)


def prepare_short_scores(texts):
    """
    V6 was trained on 180-260 word chunks.
    For long documents we use the first 260 words.
    For short documents we use the complete document.
    """
    shortened = []

    for text in texts:
        words = str(text).split()

        if len(words) > 260:
            words = words[:260]

        shortened.append(" ".join(words))

    return np.asarray(
        get_scores(
            v6_vectorizer,
            v6_model,
            shortened,
        )
    )


def normalize_score(score, threshold):
    """
    Convert each model's score into distance from
    its own validation threshold.
    """
    return score - threshold


def metrics(y_true, scores, threshold):
    pred = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1],
    ).ravel()

    return {
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


train = df[df["split"] == "train"].copy()
valid = df[df["split"] == "valid"].copy()
test = df[df["split"] == "test"].copy()


def score_dataframe(data):
    texts = data["text"].astype(str).tolist()

    v5 = get_scores(
        v5_vectorizer,
        v5_model,
        texts,
    )

    v6 = prepare_short_scores(texts)

    v5n = normalize_score(
        v5,
        v5_threshold,
    )

    v6n = normalize_score(
        v6,
        v6_threshold,
    )

    return v5n, v6n


y_train = train["ai"].astype(int).to_numpy()
y_valid = valid["ai"].astype(int).to_numpy()
y_test = test["ai"].astype(int).to_numpy()


print("Scoring train...")
v5_train, v6_train = score_dataframe(train)

print("Scoring valid...")
v5_valid, v6_valid = score_dataframe(valid)

print("Scoring test...")
v5_test, v6_test = score_dataframe(test)


results = []


# Weight of V6 in the ensemble.
# V5 gets the remaining weight.
weights = np.arange(
    0.0,
    1.01,
    0.05,
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
                )

    if best is None:
        return {
            "threshold": float(
                np.max(negatives) + 1e-12
            ),
            "fpr": 0.0,
            "recall": 0.0,
        }

    return {
        "threshold": best[1],
        "fpr": best[2],
        "recall": best[3],
    }


for w6 in weights:

    w5 = 1.0 - w6

    train_score = (
        w5 * v5_train
        + w6 * v6_train
    )

    valid_score = (
        w5 * v5_valid
        + w6 * v6_valid
    )

    threshold_info = choose_threshold(
        y_valid,
        valid_score,
    )

    threshold = threshold_info["threshold"]

    m = metrics(
        y_valid,
        valid_score,
        threshold,
    )

    results.append({
        "v5_weight": float(w5),
        "v6_weight": float(w6),
        "threshold": float(threshold),
        "validation_accuracy": m["accuracy"],
        "validation_f1": m["f1"],
        "validation_recall": m["recall"],
        "validation_fpr": m["fpr"],
        "validation_auroc": m["auroc"],
    })


results.sort(
    key=lambda r: (
        r["validation_recall"],
        -r["validation_fpr"],
        r["validation_f1"],
        r["validation_auroc"],
    ),
    reverse=True,
)


print("\nTOP ENSEMBLES")

for r in results[:10]:
    print(
        "V5=",
        round(r["v5_weight"], 2),
        "| V6=",
        round(r["v6_weight"], 2),
        "| threshold=",
        round(r["threshold"], 4),
        "| recall=",
        round(r["validation_recall"], 4),
        "| FPR=",
        round(r["validation_fpr"], 4),
        "| F1=",
        round(r["validation_f1"], 4),
        "| AUROC=",
        round(r["validation_auroc"], 4),
    )


best = results[0]

w5 = best["v5_weight"]
w6 = best["v6_weight"]
threshold = best["threshold"]


train_ensemble = (
    w5 * v5_train
    + w6 * v6_train
)

valid_ensemble = (
    w5 * v5_valid
    + w6 * v6_valid
)

test_ensemble = (
    w5 * v5_test
    + w6 * v6_test
)


print("\nSELECTED ENSEMBLE")

print(
    "V5 weight:",
    w5,
)

print(
    "V6 weight:",
    w6,
)

print(
    "Threshold:",
    threshold,
)


print("\nVALIDATION")

print(
    json.dumps(
        metrics(
            y_valid,
            valid_ensemble,
            threshold,
        ),
        indent=2,
    )
)


print("\nFROZEN TEST")

test_metrics = metrics(
    y_test,
    test_ensemble,
    threshold,
)

print(
    json.dumps(
        test_metrics,
        indent=2,
    )
)


artifact = {
    "v5_weight": float(w5),
    "v6_weight": float(w6),
    "threshold": float(threshold),
    "threshold_rule": (
        "maximum validation recall at "
        "validation FPR <= 5%"
    ),
    "validation_metrics": metrics(
        y_valid,
        valid_ensemble,
        threshold,
    ),
    "frozen_test_metrics": test_metrics,
}

with open(
    OUT_PATH,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        {
            "search": results,
            "selected": artifact,
        },
        f,
        indent=2,
    )

print("\nSaved:", OUT_PATH)
