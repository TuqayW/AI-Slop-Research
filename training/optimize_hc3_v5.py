import json
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


CSV_PATH = "dataset/human_detectors_grouped.csv"
OUT_PATH = "hc3_v5_search.json"

df = pd.read_csv(CSV_PATH)
train = df[df["split"] == "train"].copy()

texts = train["text"].astype(str).to_numpy()
y = pd.to_numeric(train["ai"]).astype(int).to_numpy()
groups = train["prompt_id"].astype(str).to_numpy()

print("Training rows:", len(train))
print("Training groups:", len(np.unique(groups)))

feature_configs = {
    "char_35": {
        "analyzer": "char_wb",
        "ngram_range": (3, 5),
        "max_features": 40000,
    },
    "char_26": {
        "analyzer": "char_wb",
        "ngram_range": (2, 6),
        "max_features": 60000,
    },
    "char_36": {
        "analyzer": "char_wb",
        "ngram_range": (3, 6),
        "max_features": 60000,
    },
    "char_46": {
        "analyzer": "char_wb",
        "ngram_range": (4, 6),
        "max_features": 60000,
    },
    "word_char": None,
}

models = [
    ("linear_svm", 0.3, {0: 1.0, 1: 1.0}),
    ("linear_svm", 1.0, {0: 1.0, 1: 1.0}),
    ("linear_svm", 3.0, {0: 1.0, 1: 1.0}),
    ("linear_svm", 10.0, {0: 1.0, 1: 1.0}),
    ("linear_svm", 30.0, {0: 1.0, 1: 1.0}),

    ("linear_svm", 1.0, {0: 1.25, 1: 1.0}),
    ("linear_svm", 3.0, {0: 1.25, 1: 1.0}),
    ("linear_svm", 10.0, {0: 1.25, 1: 1.0}),

    ("linear_svm", 1.0, {0: 1.5, 1: 1.0}),
    ("linear_svm", 3.0, {0: 1.5, 1: 1.0}),
    ("linear_svm", 10.0, {0: 1.5, 1: 1.0}),

    ("linear_svm", 1.0, {0: 2.0, 1: 1.0}),
    ("linear_svm", 3.0, {0: 2.0, 1: 1.0}),
    ("linear_svm", 10.0, {0: 2.0, 1: 1.0}),

    ("linear_svm", 1.0, {0: 3.0, 1: 1.0}),
    ("linear_svm", 3.0, {0: 3.0, 1: 1.0}),
    ("linear_svm", 10.0, {0: 3.0, 1: 1.0}),

    ("logistic", 0.3, {0: 1.0, 1: 1.0}),
    ("logistic", 1.0, {0: 1.0, 1: 1.0}),
    ("logistic", 3.0, {0: 1.0, 1: 1.0}),
    ("logistic", 10.0, {0: 1.0, 1: 1.0}),

    ("logistic", 1.0, {0: 1.5, 1: 1.0}),
    ("logistic", 3.0, {0: 1.5, 1: 1.0}),
    ("logistic", 10.0, {0: 1.5, 1: 1.0}),

    ("logistic", 1.0, {0: 2.0, 1: 1.0}),
    ("logistic", 3.0, {0: 2.0, 1: 1.0}),
    ("logistic", 10.0, {0: 2.0, 1: 1.0}),
]

folds = list(
    GroupKFold(n_splits=3).split(
        texts,
        y,
        groups,
    )
)


def make_features(name):
    if name == "word_char":
        word = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            min_df=1,
            max_features=30000,
            sublinear_tf=True,
        )

        char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            max_features=40000,
            sublinear_tf=True,
        )

        return word, char

    cfg = feature_configs[name]

    return TfidfVectorizer(
        analyzer=cfg["analyzer"],
        ngram_range=cfg["ngram_range"],
        min_df=1,
        max_features=cfg["max_features"],
        sublinear_tf=True,
    )


def make_model(model_name, c, class_weight):
    if model_name == "linear_svm":
        return LinearSVC(
            C=c,
            class_weight=class_weight,
        )

    return LogisticRegression(
        C=c,
        max_iter=5000,
        class_weight=class_weight,
        solver="liblinear",
    )


def threshold_at_5pct_fpr(y_true, scores):
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
            current = (
                recall,
                -fpr,
                threshold,
            )

            if best is None or current > best[0]:
                best = (
                    current,
                    float(threshold),
                    float(fpr),
                    float(recall),
                )

    if best is None:
        return float(np.max(scores) + 1e-12), 0.0, 0.0

    return best[1], best[2], best[3]


results = []

for feature_name in [
    "char_35",
    "char_26",
    "char_36",
    "char_46",
    "word_char",
]:
    for model_name, c, class_weight in models:

        print(
            feature_name,
            model_name,
            "C=",
            c,
            "weight=",
            class_weight,
        )

        fold_auroc = []
        fold_acc = []
        fold_f1 = []
        fold_fpr = []
        fold_recall = []

        for fold_no, (tr_idx, va_idx) in enumerate(folds, start=1):

            x_tr_text = texts[tr_idx]
            x_va_text = texts[va_idx]

            if feature_name == "word_char":
                word_vec = make_features("word_char")[0]
                char_vec = make_features("word_char")[1]

                X_tr_word = word_vec.fit_transform(x_tr_text)
                X_va_word = word_vec.transform(x_va_text)

                X_tr_char = char_vec.fit_transform(x_tr_text)
                X_va_char = char_vec.transform(x_va_text)

                from scipy.sparse import hstack

                X_tr = hstack(
                    [X_tr_word, X_tr_char],
                    format="csr",
                )

                X_va = hstack(
                    [X_va_word, X_va_char],
                    format="csr",
                )
            else:
                vec = make_features(feature_name)

                X_tr = vec.fit_transform(x_tr_text)
                X_va = vec.transform(x_va_text)

            clf = make_model(
                model_name,
                c,
                class_weight,
            )

            clf.fit(
                X_tr,
                y[tr_idx],
            )

            if hasattr(clf, "decision_function"):
                scores = clf.decision_function(X_va)
            else:
                scores = clf.predict_proba(X_va)[:, 1]

            threshold, fpr, recall = threshold_at_5pct_fpr(
                y[va_idx],
                scores,
            )

            pred = (scores >= threshold).astype(int)

            fold_auroc.append(
                roc_auc_score(
                    y[va_idx],
                    scores,
                )
            )

            fold_acc.append(
                accuracy_score(
                    y[va_idx],
                    pred,
                )
            )

            fold_f1.append(
                f1_score(
                    y[va_idx],
                    pred,
                    zero_division=0,
                )
            )

            fold_fpr.append(fpr)
            fold_recall.append(recall)

        row = {
            "feature_set": feature_name,
            "model": model_name,
            "C": c,
            "class_weight_human": class_weight[0],
            "class_weight_ai": class_weight[1],
            "auroc_mean": float(np.mean(fold_auroc)),
            "auroc_std": float(np.std(fold_auroc)),
            "accuracy_mean": float(np.mean(fold_acc)),
            "f1_mean": float(np.mean(fold_f1)),
            "fpr_mean": float(np.mean(fold_fpr)),
            "recall_mean": float(np.mean(fold_recall)),
            "fold_auroc": fold_auroc,
            "fold_accuracy": fold_acc,
            "fold_f1": fold_f1,
            "fold_fpr": fold_fpr,
            "fold_recall": fold_recall,
        }

        results.append(row)


# Primary objective:
# maximize recall under a low average FPR,
# then AUROC, then F1.
results.sort(
    key=lambda r: (
        r["recall_mean"],
        -r["fpr_mean"],
        r["auroc_mean"],
        r["f1_mean"],
    ),
    reverse=True,
)

print("\nTOP 15")

for r in results[:15]:
    print(
        r["feature_set"],
        "|",
        r["model"],
        "| C=",
        r["C"],
        "| human_weight=",
        r["class_weight_human"],
        "| recall=",
        round(r["recall_mean"], 4),
        "| FPR=",
        round(r["fpr_mean"], 4),
        "| AUROC=",
        round(r["auroc_mean"], 5),
        "| F1=",
        round(r["f1_mean"], 4),
    )

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nSaved:", OUT_PATH)
