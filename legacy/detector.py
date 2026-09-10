"""Experimental English-text screening baseline, not proof of AI authorship."""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata


FEATURES = [
    "log_word_count", "type_token_ratio", "mean_word_length",
    "mean_sentence_length", "sentence_length_cv", "repeated_trigram_fraction",
    "repeated_sentence_fraction", "digit_token_fraction", "url_rate",
    "punctuation_rate",
]
MIN_WORDS = 80
TARGETS = ("ai", "low_quality")
SPLITS = ("train", "valid", "test")


def words(text):
    return re.findall(r"\b\w+(?:['’]\w+)?\b", text.lower())


def features(text):
    tokens = words(text)
    if not tokens:
        raise ValueError("Text must contain words.")
    sentences = [words(part) for part in re.split(r"[.!?]+", text)]
    sentences = [tuple(part) for part in sentences if part]
    lengths = np.array([len(part) for part in sentences], dtype=float)
    trigrams = list(zip(tokens, tokens[1:], tokens[2:]))
    count = len(tokens)
    return np.array([
        np.log1p(count), len(set(tokens)) / count,
        np.mean([len(token) for token in tokens]), lengths.mean(),
        lengths.std() / max(lengths.mean(), 1),
        1 - len(set(trigrams)) / len(trigrams) if trigrams else 0,
        1 - len(set(sentences)) / len(sentences),
        sum(any(char.isdigit() for char in token) for token in tokens) / count,
        len(re.findall(r"https?://\S+", text)) / count,
        sum(char in ",;:!?—-()" for char in text) / count,
    ], dtype=float)


def load_data(path, targets=TARGETS):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"text", "group", "split", *targets}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV requires columns: {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV has no observations.")
    groups = {}
    documents = set()
    for index, row in enumerate(rows, start=2):
        if row["split"] not in SPLITS or not row["group"].strip():
            raise ValueError(f"Row {index}: invalid split or empty group.")
        if any(row[target] not in {"0", "1"} for target in targets):
            raise ValueError(f"Row {index}: requested labels must be 0 or 1.")
        if len(words(row["text"])) < MIN_WORDS:
            raise ValueError(f"Row {index}: requires at least {MIN_WORDS} words.")
        previous = groups.setdefault(row["group"], row["split"])
        if previous != row["split"]:
            raise ValueError("A group crosses splits; rebuild the split manifest.")
        digest = hashlib.sha256(" ".join(words(row["text"])).encode()).hexdigest()
        if digest in documents:
            raise ValueError("Duplicate normalized text; deduplicate before training.")
        documents.add(digest)
    for split in SPLITS:
        for target in targets:
            labels = {row[target] for row in rows if row["split"] == split}
            if labels != {"0", "1"}:
                raise ValueError(f"{split}/{target} must contain both classes.")
    return rows


def fit_head(matrix, labels, regularization=0.1):
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("Regularization must be finite and positive.")
    design = np.column_stack([np.ones(len(matrix)), matrix])

    def objective(coefficients):
        logits = design @ coefficients
        loss = np.mean(np.logaddexp(0, logits) - labels * logits)
        loss += regularization * np.sum(coefficients[1:] ** 2) / 2
        gradient = design.T @ (expit(logits) - labels) / len(labels)
        gradient[1:] += regularization * coefficients[1:]
        return loss, gradient

    result = minimize(objective, np.zeros(design.shape[1]), jac=True,
                      method="L-BFGS-B", options={"maxiter": 1000})
    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")
    return result.x


def scores(matrix, coefficients):
    coefficients = np.asarray(coefficients)
    return expit(coefficients[0] + matrix @ coefficients[1:])


def metrics(labels, probabilities, threshold):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = probabilities >= threshold
    positive = labels == 1
    true_positive = int(np.sum(predicted & positive))
    false_positive = int(np.sum(predicted & ~positive))
    true_negative = int(np.sum(~predicted & ~positive))
    false_negative = int(np.sum(~predicted & positive))
    positives, negatives = int(positive.sum()), int((~positive).sum())
    precision = true_positive / (true_positive + false_positive) if predicted.any() else None
    recall = true_positive / positives if positives else None
    denominator = 2 * true_positive + false_positive + false_negative
    auc = None
    if positives and negatives:
        rank_sum = rankdata(probabilities)[positive].sum()
        auc = float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))
    return {
        "n": len(labels), "tp": true_positive, "fp": false_positive,
        "tn": true_negative, "fn": false_negative, "precision": precision,
        "recall": recall, "fpr": false_positive / negatives if negatives else None,
        "f1": 2 * true_positive / denominator if denominator else None,
        "accuracy": (true_positive + true_negative) / len(labels), "auroc": auc,
    }


def choose_threshold(labels, probabilities, target, max_fpr=0.05):
    candidates = np.append(np.unique(probabilities), np.nextafter(1.0, 2.0))
    ranked = []
    for threshold in candidates:
        result = metrics(labels, probabilities, float(threshold))
        if target == "ai":
            if result["fpr"] > max_fpr:
                continue
            key = (result["recall"], -result["fpr"], float(threshold))
        else:
            key = (result["f1"], -result["fpr"], float(threshold))
        ranked.append((key, float(threshold)))
    return max(ranked)[1]


def train(path, max_fpr=0.05):
    if not 0 <= max_fpr < 1:
        raise ValueError("max_fpr must lie in [0, 1).")
    rows = load_data(path)
    matrix = np.vstack([features(row["text"]) for row in rows])
    masks = {split: np.array([row["split"] == split for row in rows]) for split in SPLITS}
    mean = matrix[masks["train"]].mean(axis=0)
    scale = matrix[masks["train"]].std(axis=0)
    scale[scale < 1e-12] = 1
    standardized = (matrix - mean) / scale
    artifact = {
        "version": 1, "features": FEATURES, "min_words": MIN_WORDS,
        "mean": mean.tolist(), "scale": scale.tolist(), "heads": {},
        "regularization": 0.1, "validation_max_fpr": max_fpr,
        "dataset_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "warning": "Uncalibrated research scores; not proof of authorship or quality.",
    }
    report = {"dataset_sha256": artifact["dataset_sha256"], "targets": {}}
    predictions = {}
    for target in TARGETS:
        labels = np.array([int(row[target]) for row in rows])
        coefficients = fit_head(standardized[masks["train"]], labels[masks["train"]])
        probabilities = scores(standardized, coefficients)
        threshold = choose_threshold(labels[masks["valid"]], probabilities[masks["valid"]],
                                     target, max_fpr)
        artifact["heads"][target] = {"coefficients": coefficients.tolist(), "threshold": threshold}
        report["targets"][target] = {
            split: metrics(labels[mask], probabilities[mask], threshold)
            for split, mask in masks.items() if split != "train"
        }
        predictions[target] = probabilities >= threshold
    joint_labels = np.array([int(row["ai"]) * int(row["low_quality"]) for row in rows])
    joint_flags = (predictions["ai"] & predictions["low_quality"]).astype(float)
    report["joint_screen"] = {}
    for split in ("valid", "test"):
        mask = masks[split]
        result = metrics(joint_labels[mask], joint_flags[mask], 0.5)
        result.pop("auroc")
        report["joint_screen"][split] = result
    report["warning"] = "Validation metrics select thresholds; only test metrics are held out."
    return artifact, report


def predict(text, artifact):
    if artifact.get("version") != 1 or artifact.get("features") != FEATURES:
        raise ValueError("Incompatible model artifact.")
    count = len(words(text))
    if count < artifact["min_words"]:
        return {"status": "abstain", "reason": "too_short", "word_count": count}
    standardized = (features(text) - np.array(artifact["mean"])) / np.array(artifact["scale"])
    result = {"status": "scored", "word_count": count, "heads": {}, "warning": artifact["warning"]}
    if not artifact.get("heads") or not set(artifact["heads"]).issubset(TARGETS):
        raise ValueError("Model must contain recognized classifier heads.")
    for target in artifact["heads"]:
        head = artifact["heads"][target]
        coefficients = np.array(head["coefficients"])
        probability = float(scores(standardized, coefficients))
        contributions = standardized * coefficients[1:]
        order = np.argsort(-np.abs(contributions))[:3]
        result["heads"][target] = {
            "score": probability, "threshold": head["threshold"],
            "flag": bool(probability >= head["threshold"]),
            "top_logit_contributions": {FEATURES[index]: float(contributions[index]) for index in order},
        }
    result["review_candidate"] = (
        all(result["heads"][target]["flag"] for target in TARGETS)
        if set(result["heads"]) == set(TARGETS) else None
    )
    if "low_quality" not in result["heads"]:
        result["quality_status"] = "not_trained"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    training = subparsers.add_parser("train")
    training.add_argument("data", type=Path)
    training.add_argument("model", type=Path)
    training.add_argument("report", type=Path)
    training.add_argument("--max-fpr", type=float, default=0.05)
    inference = subparsers.add_parser("predict")
    inference.add_argument("model", type=Path)
    inference.add_argument("text", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "train":
            if len({args.data.resolve(), args.model.resolve(), args.report.resolve()}) != 3:
                raise ValueError("Input, model, and report must be different files.")
            artifact, report = train(args.data, args.max_fpr)
            args.model.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(json.dumps(report, indent=2, allow_nan=False))
        else:
            artifact = json.loads(args.model.read_text(encoding="utf-8"))
            print(json.dumps(predict(args.text.read_text(encoding="utf-8"), artifact),
                             indent=2, allow_nan=False))
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()