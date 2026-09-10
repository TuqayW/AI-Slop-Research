"""Statistical utilities shared by exploratory and gated evaluation commands."""

import numpy as np
from scipy.stats import binom, binomtest

import detector


def classification(labels, scores, threshold):
    if not len(labels):
        return {"n": 0, "status": "no_eligible_documents"}
    result = detector.metrics(labels, scores, threshold)
    result["fnr"] = 1 - result["recall"] if result["recall"] is not None else None
    for name, errors, total in (
        ("fpr", result["fp"], result["fp"] + result["tn"]),
        ("fnr", result["fn"], result["fn"] + result["tp"]),
    ):
        bounds = binomtest(errors, total).proportion_ci() if total else None
        result[f"{name}_iid_exact_95"] = [float(bounds.low), float(bounds.high)] if bounds else None
    result["interval_caution"] = "Exact binomial intervals assume independent observations; shared sources violate this assumption."
    return result


def cluster_intervals(labels, scores, thresholds, groups, seed=20260910, replicates=2000):
    labels, scores, thresholds, groups = map(np.asarray, (labels, scores, thresholds, groups))
    unique = np.unique(groups)
    if len(unique) < 2:
        return {"status": "insufficient_clusters", "clusters": len(unique)}
    flags = (scores >= thresholds).astype(float)
    indices = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in ("fpr", "fnr", "precision", "recall", "f1", "accuracy")}
    for _ in range(replicates):
        selected = np.concatenate([indices[index] for index in rng.integers(0, len(unique), len(unique))])
        result = detector.metrics(labels[selected], flags[selected], 0.5)
        result["fnr"] = 1 - result["recall"] if result["recall"] is not None else None
        for name in samples:
            if result[name] is not None:
                samples[name].append(result[name])
    return {
        "clusters": len(unique), "replicates": replicates,
        "percentile_95": {name: np.quantile(values, [0.025, 0.975]).tolist() if values else None
                          for name, values in samples.items()},
        "valid_replicates": {name: len(values) for name, values in samples.items()},
        "caution": "Conditional on fitted predictions; does not include training uncertainty. Boundary-degenerate intervals are not evidence of zero population risk.",
    }


def agreement(first, second, categories=(0, 1), quadratic=False):
    first, second = np.asarray(first), np.asarray(second)
    if len(first) != len(second):
        raise ValueError("Rater vectors must have equal length.")
    if not len(first):
        return {"n": 0, "raw_agreement": None, "kappa": None}
    if any(value not in categories for value in np.concatenate([first, second])):
        raise ValueError("Unexpected annotation category.")
    matrix = np.array([[np.sum((first == left) & (second == right)) for right in categories]
                       for left in categories], dtype=float)
    expected = np.outer(matrix.sum(axis=1), matrix.sum(axis=0)) / len(first)
    weights = np.array([[(left - right) ** 2 if quadratic else int(left != right)
                         for right in categories] for left in categories], dtype=float)
    observed_disagreement = float(np.sum(weights * matrix))
    expected_disagreement = float(np.sum(weights * expected))
    kappa = 1 - observed_disagreement / expected_disagreement if expected_disagreement else None
    return {"n": len(first), "raw_agreement": float(np.mean(first == second)), "kappa": kappa,
            "confusion_counts": matrix.astype(int).tolist(), "quadratic_weights": quadratic}


def np_threshold(negative_scores, alpha=0.05, delta=0.025):
    values = np.asarray(negative_scores, dtype=float)
    if not 0 < alpha < 1 or not 0 < delta < 1:
        raise ValueError("alpha and delta must be strictly between zero and one.")
    if values.ndim != 1 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Calibration scores must be a finite probability-score vector.")
    minimum = int(np.ceil(np.log(delta) / np.log1p(-alpha)))
    if len(values) < minimum:
        raise ValueError(f"At least {minimum} independent calibration negatives are required; received {len(values)}.")
    for rank in range(1, len(values) + 1):
        violation_bound = float(binom.sf(rank - 1, len(values), 1 - alpha))
        if violation_bound <= delta:
            threshold = float(np.nextafter(np.sort(values)[rank - 1], np.inf))
            return {"threshold": threshold, "rank": rank, "n": len(values),
                    "alpha": alpha, "delta": delta, "violation_bound": violation_bound,
                    "assumptions": "Independent exchangeable negative calibration draws and a scoring model fixed before calibration; no guarantee under distribution shift."}
    raise ValueError("No admissible order statistic.")