"""Exploratory analysis only: every original pilot article is now development data."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

import detector
from cycle2_stats import classification, cluster_intervals


ROOT = Path("pilot/artifacts")
CONFIGURATIONS = {
    "original": (0.1, list(range(10))),
    "pilot_selected": (10.0, [1, 2, 4, 5, 6, 7, 8, 9]),
    "length_only": (0.1, [0, 3]),
    "without_repetition": (0.1, [0, 1, 2, 3, 4, 7, 8, 9]),
}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fit_scores(matrix, labels, training, regularization, active):
    mean, scale = matrix[training].mean(axis=0), matrix[training].std(axis=0)
    scale[scale < 1e-12] = 1
    standardized = ((matrix - mean) / scale)[:, active]
    coefficients = detector.fit_head(standardized[training], labels[training], regularization)
    return detector.scores(standardized, coefficients)


def gap_thresholds(labels, scores):
    conservative = detector.choose_threshold(labels, scores, "ai")
    negative_max, positive_min = float(max(scores[labels == 0])), float(min(scores[labels == 1]))
    result = {"conservative": conservative}
    if negative_max < positive_min:
        result["midpoint_validation_gap"] = (negative_max + positive_min) / 2
        result["just_above_validation_negative_max"] = float(np.nextafter(negative_max, np.inf))
    return result


def fold_masks(groups, seed):
    unique = np.unique(groups)
    np.random.default_rng(seed).shuffle(unique)
    folds = []
    for fold_index, held_out in enumerate(unique):
        valid = {unique[(fold_index + offset) % len(unique)] for offset in (1, 2, 3)}
        testing = groups == held_out
        validation = np.isin(groups, list(valid))
        folds.append((~(testing | validation), validation, testing))
    return folds


def cross_validate(matrix, labels, groups, folds, configuration):
    regularization, active = CONFIGURATIONS[configuration]
    records, fold_records = [], []
    for fold_index, (training, validation, testing) in enumerate(folds):
        scores = fit_scores(matrix, labels, training, regularization, active)
        threshold = detector.choose_threshold(labels[validation], scores[validation], "ai")
        result = detector.metrics(labels[testing], scores[testing], threshold)
        fold_records.append({"fold": fold_index, "held_out_group": str(groups[testing][0]),
                             "threshold": threshold, "metrics": result})
        for index in np.flatnonzero(testing):
            records.append({"index": int(index), "group": str(groups[index]), "label": int(labels[index]),
                            "score": float(scores[index]), "threshold": threshold})
    return records, fold_records


def summarize_cv(records, folds, seed):
    labels = np.array([item["label"] for item in records])
    scores = np.array([item["score"] for item in records])
    thresholds = np.array([item["threshold"] for item in records])
    flags = (scores >= thresholds).astype(float)
    result = classification(labels, flags, 0.5)
    result.pop("auroc")
    result["mean_within_fold_auroc"] = float(np.mean([item["metrics"]["auroc"] for item in folds]))
    result["threshold_min_median_max"] = np.quantile([item["threshold"] for item in folds], [0, 0.5, 1]).tolist()
    result["topic_cluster_bootstrap"] = cluster_intervals(labels, scores, thresholds,
                                                         [item["group"] for item in records], seed)
    return result


def analyze(output, seed=20260910, permutations=99):
    if permutations < 1:
        raise ValueError("At least one permutation is required.")
    rows = detector.load_data(ROOT / "corpus.csv", targets=("ai",))
    matrix = np.vstack([detector.features(row["text"]) for row in rows])
    labels = np.array([int(row["ai"]) for row in rows])
    groups = np.array([row["group"] for row in rows])
    original_validation = np.array([row["split"] == "valid" for row in rows])
    original_test = np.array([row["split"] == "test" for row in rows])
    report = {"status": "exploratory_development_only", "seed": seed, "n": len(rows),
              "all_pilot_test_rows_reclassified_as_development": True,
              "input_sha256": hashlib.sha256((ROOT / "corpus.csv").read_bytes()).hexdigest(),
              "reproduction": {}, "fixed_model_threshold_comparison": {}, "formatting_controls": {},
              "source_disjoint_evaluation": {"status": "not_identifiable", "reason": "Historical origin and the single historical source coincide; generated origin and the single assistant coincide. Holding out a source removes an origin class."}}
    for name in ("baseline", "selected"):
        artifact = load_json(ROOT / f"{name}_model.json")
        scores = np.array([detector.predict(row["text"], artifact)["heads"]["ai"]["score"] for row in rows])
        threshold = artifact["heads"]["ai"]["threshold"]
        reproduction = classification(labels[original_test], scores[original_test], threshold)
        saved = load_json(ROOT / "test_results.json")["models"][name]
        for key in ("tp", "fp", "tn", "fn", "accuracy", "auroc", "recall"):
            if not np.isclose(reproduction[key], saved[key]):
                raise ValueError("Saved pilot result does not reproduce.")
        report["reproduction"][name] = reproduction
        report["fixed_model_threshold_comparison"][name] = {
            rule: {"threshold": candidate, "original_test_now_development": classification(labels[original_test], scores[original_test], candidate)}
            for rule, candidate in gap_thresholds(labels[original_validation], scores[original_validation]).items()
        }
        controls = {
            "uppercase": lambda text: text.upper(),
            "collapsed_whitespace": lambda text: " ".join(text.split()),
            "sentence_bullets": lambda text: "- " + "\n- ".join(re.split(r"(?<=[.!?])\s+", text)),
        }
        report["formatting_controls"][name] = {}
        for condition, transform in controls.items():
            changed = np.array([detector.predict(transform(row["text"]), artifact)["heads"]["ai"]["score"] for row in rows])
            report["formatting_controls"][name][condition] = {
                "documents": len(rows), "mean_absolute_score_change": float(np.mean(abs(changed - scores))),
                "maximum_absolute_score_change": float(max(abs(changed - scores))),
                "decision_flips": int(np.sum((changed >= threshold) != (scores >= threshold))),
                "original_test_now_development": classification(labels[original_test], changed[original_test], threshold),
                "caution": "Paired surface-format diagnostic, not an independent challenge corpus or comprehensive semantic-equivalence audit.",
            }
    folds = fold_masks(groups, seed)
    report["cross_validation"] = {}
    all_records = {}
    for configuration in CONFIGURATIONS:
        records, fold_records = cross_validate(matrix, labels, groups, folds, configuration)
        report["cross_validation"][configuration] = summarize_cv(records, fold_records, seed)
        all_records[configuration] = {"predictions": records, "folds": fold_records}
    rng = np.random.default_rng(seed)
    statistics = []
    for _ in range(permutations):
        permuted = labels.copy()
        for group in np.unique(groups):
            if rng.integers(0, 2):
                permuted[groups == group] = 1 - permuted[groups == group]
        _, fold_records = cross_validate(matrix, permuted, groups, folds, "original")
        statistics.append(float(np.mean([item["metrics"]["auroc"] for item in fold_records])))
    observed = report["cross_validation"]["original"]["mean_within_fold_auroc"]
    report["paired_label_permutation"] = {
        "permutations": permutations, "statistic": "mean within-fold AUROC after refitting and validation threshold selection",
        "observed": observed, "null_statistics": statistics,
        "monte_carlo_p": (1 + sum(value >= observed for value in statistics)) / (permutations + 1),
        "caution": "An association sanity check under within-pair exchangeability, not evidence that origin rather than era, source, or register caused the association.",
    }
    report["feature_shift_original_split"] = []
    train_mask = np.array([row["split"] == "train" for row in rows])
    scale = matrix[train_mask].std(axis=0)
    for origin in (0, 1):
        change = matrix[original_test & (labels == origin)].mean(axis=0) - matrix[original_validation & (labels == origin)].mean(axis=0)
        for index, feature in enumerate(detector.FEATURES):
            report["feature_shift_original_split"].append({"ai": origin, "feature": feature,
                "test_minus_validation_mean": float(change[index]),
                "change_in_training_sd": float(change[index] / scale[index]) if scale[index] > 1e-12 else None})
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "diagnostics.json", report)
    write_json(output / "fold_predictions.json", all_records)
    lines = [r"\begin{table}[htbp]", r"\centering\small", r"\begin{tabular}{lrrrr}",
             r"\toprule", r"Configuration & FPR & FNR & Accuracy & Mean fold AUROC \\", r"\midrule"]
    for name, result in report["cross_validation"].items():
        lines.append(f"{name.replace('_', ' ').capitalize()} & {result['fpr']:.3f} & {result['fnr']:.3f} & {result['accuracy']:.3f} & {result['mean_within_fold_auroc']:.3f} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\caption{Exploratory leave-one-topic-pair-out development results on all 24 original pilot articles. These are not new final-test estimates.}", r"\end{table}"])
    (output / "diagnostic_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": report["status"],
                      "cross_validation": {name: {key: result[key] for key in ('fpr', 'fnr', 'accuracy', 'threshold_min_median_max')}
                                           for name, result in report['cross_validation'].items()}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("cycle2/artifacts"))
    parser.add_argument("--permutations", type=int, default=99)
    args = parser.parse_args()
    analyze(args.output, permutations=args.permutations)