"""Exploratory pilot diagnostics and fail-closed readiness checks for cycle two."""

import argparse
import csv
import hashlib
import json
import platform
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import scipy

import detector


ROOT = Path("cycle2")
RESULTS = ROOT / "results"
DIMENSIONS = ("factual", "information", "relevance")
MISSING = {"", "unknown", "not_applicable"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized_hash(text):
    return hashlib.sha256(" ".join(detector.words(text)).encode()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def feature_mask(name):
    length = {"log_word_count", "mean_sentence_length"}
    repetition = {"repeated_trigram_fraction", "repeated_sentence_fraction"}
    included = {
        "all": set(detector.FEATURES), "without_length": set(detector.FEATURES) - length,
        "length": length, "repetition": repetition,
        "without_repetition": set(detector.FEATURES) - repetition,
    }
    return np.array([feature in included[name] for feature in detector.FEATURES])


def fit_scores(matrix, labels, train_indices, configuration):
    mean = matrix[train_indices].mean(axis=0)
    scale = matrix[train_indices].std(axis=0)
    scale[scale < 1e-12] = 1
    standardized = (matrix - mean) / scale
    active = feature_mask(configuration["features"])
    coefficients = detector.fit_head(standardized[train_indices][:, active], labels[train_indices],
                                     configuration["regularization"])
    return detector.scores(standardized[:, active], coefficients)


def descriptive_metrics(labels, scores, threshold):
    result = detector.metrics(labels, scores, threshold)
    result["fnr"] = 1 - result["recall"] if result["recall"] is not None else None
    result["coverage"] = 1.0
    return result


def diagnostic_threshold(labels, scores, policy):
    negatives = scores[labels == 0]
    positives = scores[labels == 1]
    if not len(negatives) or not len(positives):
        raise ValueError("Both calibration classes are required for pilot diagnostics.")
    if policy == "negative_max":
        return float(np.nextafter(negatives.max(), np.inf))
    if policy == "separating_midpoint" and negatives.max() < positives.min():
        return float((negatives.max() + positives.min()) / 2)
    return detector.choose_threshold(labels, scores, "ai", max_fpr=0.05)


def guarded_threshold(negative_scores, cluster_ids, target_fpr=0.05, alpha=0.025):
    scores = np.asarray(negative_scores, dtype=float)
    if not 0 < target_fpr < 1 or not 0 < alpha < 1:
        raise ValueError("FPR and alpha must lie strictly between zero and one.")
    if len(scores) != len(cluster_ids) or not len(scores) or len(set(cluster_ids)) != len(scores):
        raise ValueError("Require exactly one prespecified negative per independent cluster.")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Invalid calibration scores.")
    upper = float(1 - alpha ** (1 / len(scores)))
    if upper > target_fpr:
        raise ValueError(f"Insufficient calibration negatives: {len(scores)} gives upper bound {upper:.4f}.")
    return {"threshold": float(np.nextafter(scores.max(), np.inf)), "negative_clusters": len(scores),
            "one_sided_upper_bound": upper, "alpha": alpha,
            "assumptions": "Independent exchangeable cluster-selected negatives and a model fixed before calibration; not a guarantee under shift."}


def quantiles(values):
    values = np.asarray(values, dtype=float)
    return {"min": float(values.min()), "q025": float(np.quantile(values, 0.025)),
            "median": float(np.median(values)), "q975": float(np.quantile(values, 0.975)),
            "max": float(values.max()), "mean": float(values.mean())}


def diagnose():
    protocol = read_json(ROOT / "exploratory_plan.json")
    settings = protocol["exploratory"]
    rows = detector.load_data("pilot/artifacts/corpus.csv", targets=("ai",))
    matrix = np.vstack([detector.features(row["text"]) for row in rows])
    labels = np.array([int(row["ai"]) for row in rows])
    groups = sorted({row["group"] for row in rows})
    membership = np.array([row["group"] for row in rows])
    manifest = [{"id": row["id"], "cluster": row["group"], "original_split": row["split"],
                 "current_role": "exploratory_development_only", "normalized_sha256": normalized_hash(row["text"])}
                for row in rows]
    write_json(RESULTS / "development_manifest.json", manifest)
    reproduction = {}
    feature_shifts = []
    old_train = np.array([row["split"] == "train" for row in rows])
    old_valid = np.array([row["split"] == "valid" for row in rows])
    old_test = np.array([row["split"] == "test" for row in rows])
    for name in ("baseline", "selected"):
        artifact = read_json(f"pilot/artifacts/{name}_model.json")
        standardized = (matrix - np.array(artifact["mean"])) / np.array(artifact["scale"])
        scores = detector.scores(standardized, artifact["heads"]["ai"]["coefficients"])
        threshold = artifact["heads"]["ai"]["threshold"]
        metrics = descriptive_metrics(labels[old_test], scores[old_test], threshold)
        saved = read_json("pilot/artifacts/test_results.json")["models"][name]
        if any(metrics[key] != saved[key] for key in ("tp", "fp", "tn", "fn", "auroc")):
            raise ValueError("The recorded pilot result did not reproduce.")
        reproduction[name] = {
            "recorded_threshold": threshold, "former_test_metrics": metrics,
            "calibration_human_scores": scores[old_valid & (labels == 0)].tolist(),
            "calibration_ai_scores": scores[old_valid & (labels == 1)].tolist(),
            "former_test_human_scores": scores[old_test & (labels == 0)].tolist(),
            "former_test_ai_scores": scores[old_test & (labels == 1)].tolist(),
            "retrospective_development_counterfactuals": {
                policy: {"threshold": diagnostic_threshold(labels[old_valid], scores[old_valid], policy),
                         "metrics": descriptive_metrics(labels[old_test], scores[old_test],
                                     diagnostic_threshold(labels[old_valid], scores[old_valid], policy))}
                for policy in settings["threshold_policies"]
            },
        }
    scale = matrix[old_train].std(axis=0)
    scale[scale < 1e-12] = 1
    for feature_index, feature in enumerate(detector.FEATURES):
        for origin in (0, 1):
            valid_values = matrix[old_valid & (labels == origin), feature_index]
            test_values = matrix[old_test & (labels == origin), feature_index]
            feature_shifts.append({"feature": feature, "ai": origin,
                                   "calibration_mean": float(valid_values.mean()),
                                   "former_test_mean": float(test_values.mean()),
                                   "mean_difference_in_training_sd": float((test_values.mean() - valid_values.mean()) / scale[feature_index])})
    rng = np.random.default_rng(protocol["seed"])
    splits = []
    for repeat in range(settings["repetitions"]):
        shuffled = rng.permutation(groups)
        splits.append({"repeat": repeat, "train": shuffled[:6].tolist(),
                       "calibration": shuffled[6:9].tolist(), "assessment": shuffled[9:].tolist()})
    fold_rows = []
    for split in splits:
        train_indices = np.flatnonzero(np.isin(membership, split["train"]))
        calibration_indices = np.flatnonzero(np.isin(membership, split["calibration"]))
        assessment_indices = np.flatnonzero(np.isin(membership, split["assessment"]))
        for configuration in settings["configurations"]:
            probabilities = fit_scores(matrix, labels, train_indices, configuration)
            for policy in settings["threshold_policies"]:
                threshold = diagnostic_threshold(labels[calibration_indices], probabilities[calibration_indices], policy)
                metrics = descriptive_metrics(labels[assessment_indices], probabilities[assessment_indices], threshold)
                fold_rows.append({"repeat": split["repeat"], "configuration": configuration["name"],
                                  "policy": policy, "threshold": threshold, **metrics})
    summaries = []
    for configuration in settings["configurations"]:
        for policy in settings["threshold_policies"]:
            subset = [row for row in fold_rows if row["configuration"] == configuration["name"] and row["policy"] == policy]
            summaries.append({"configuration": configuration["name"], "policy": policy,
                              "repetitions": len(subset),
                              **{metric: quantiles([row[metric] for row in subset if row[metric] is not None])
                                 for metric in ("threshold", "fpr", "fnr", "recall", "auroc", "accuracy")}})
    null_rows = []
    for repeat in range(settings["repetitions"]):
        flipped = {group: int(rng.integers(0, 2)) for group in groups}
        permuted = np.array([int(label) ^ flipped[group] for label, group in zip(labels, membership)])
        scores = fit_scores(matrix, permuted, np.flatnonzero(old_train), settings["configurations"][0])
        threshold = diagnostic_threshold(permuted[old_valid], scores[old_valid], "legacy")
        null_rows.append({"repeat": repeat, **descriptive_metrics(permuted[old_test], scores[old_test], threshold)})
    null_aucs = [row["auroc"] for row in null_rows]
    artifact = read_json("pilot/artifacts/selected_model.json")
    probe_rows = []
    for row in rows:
        text = row["text"]
        probes = {
            "whitespace": "\n\t".join(text.split()), "uppercase": text.upper(),
            "sentence_bullets": "\n".join("- " + part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()),
            "digits_as_words": re.sub(r"\d+", "number", text),
        }
        before = detector.predict(text, artifact)["heads"]["ai"]
        for probe, altered in probes.items():
            after = detector.predict(altered, artifact)["heads"]["ai"]
            probe_rows.append({"id": row["id"], "ai": int(row["ai"]), "probe": probe,
                               "score_before": before["score"], "score_after": after["score"],
                               "absolute_score_change": abs(after["score"] - before["score"]),
                               "flag_changed": before["flag"] != after["flag"]})
    probe_summary = {
        name: {"n": len(rows), "flips": sum(row["flag_changed"] for row in probe_rows if row["probe"] == name),
               "max_absolute_score_change": max(row["absolute_score_change"] for row in probe_rows if row["probe"] == name)}
        for name in settings["formatting_probes"]
    }
    report = {
        "status": "exploratory_only", "n_unique_documents": len(rows), "n_topic_pairs": len(groups),
        "former_test_reclassified_as_development": True,
        "input_hashes": {path: sha256(path) for path in ["pilot/artifacts/corpus.csv", "detector.py", "research_cycle.py", "cycle2/exploratory_plan.json"]},
        "reproduction": reproduction, "stability": summaries, "formatting": probe_summary,
        "permutation_control": {"repetitions": len(null_rows), "auroc": quantiles(null_aucs),
                                "monte_carlo_tail_fraction": float((1 + sum(value >= 1.0 for value in null_aucs)) / (1 + len(null_aucs)))},
        "source_disjoint": {"status": "not_identifiable", "reason": "One human publication/issue and one generating assistant; source/era is perfectly confounded with origin."},
        "quality_validation": {"status": "not_available", "reason": "No independent quality labels."},
        "proficiency_translation_controls": {"status": "not_available", "reason": "No independently documented subgroup data."},
        "causal_conclusion": "The cutoff mechanism reproduces exactly; these data cannot distinguish causal effects of era, source, generator, proficiency, or training diversity.",
        "uncertainty": settings["interval_interpretation"],
    }
    write_json(RESULTS / "diagnostics.json", report)
    write_json(RESULTS / "development_splits.json", splits)
    write_csv(RESULTS / "fold_results.csv", fold_rows)
    write_csv(RESULTS / "feature_shifts.csv", feature_shifts)
    write_csv(RESULTS / "formatting_probes.csv", probe_rows)
    write_csv(RESULTS / "permutation_control.csv", null_rows)
    render_diagnostic_tables(report)
    print(json.dumps({"status": report["status"], "fits": 600, "assessment_rows": len(fold_rows), "formatting": probe_summary}, indent=2))


def render_diagnostic_tables(report):
    lines = [r"\begin{table}[htbp]", r"\centering\small", r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Features & Cutoff rule & Mean FPR & Mean FNR & Median AUROC & Recall interval \\", r"\midrule"]
    names = {"original": "All ten", "pilot_selected": "Pilot selected", "length_only": "Length only",
             "repetition_only": "Repetition only", "without_repetition": "No repetition"}
    for item in report["stability"]:
        policy = {"legacy": "Legacy", "negative_max": "Negative max", "separating_midpoint": "Midpoint"}[item["policy"]]
        interval = f"[{item['recall']['q025']:.2f}, {item['recall']['q975']:.2f}]"
        lines.append(f"{names[item['configuration']]} & {policy} & {item['fpr']['mean']:.3f} & {item['fnr']['mean']:.3f} & {item['auroc']['median']:.3f} & {interval} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}",
                  r"\caption{Exploratory cycle-two diagnostics over 100 repeated topic-pair splits of the same 24 pilot documents. Intervals are the 2.5th and 97.5th percentiles across splits, not confidence intervals for modern-web performance.}", r"\end{table}"])
    (RESULTS / "diagnostic_tables.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def derive_quality(annotation):
    values = [annotation.get(name, "uncertain") for name in (*DIMENSIONS, "critical_error")]
    if any(value not in {"0", "1", "2", "uncertain"} for value in values[:3]) or values[3] not in {"0", "1", "uncertain"}:
        raise ValueError("Invalid rubric score.")
    if "uncertain" in values:
        return None
    return int(sum(value == "2" for value in values[:3]) >= 2 or values[3] == "1")


def kappa(first, second, categories=2, weighted=False):
    if len(first) != len(second) or not len(first):
        return None
    table = np.zeros((categories, categories))
    for left, right in zip(first, second):
        table[int(left), int(right)] += 1
    table /= table.sum()
    expected = np.outer(table.sum(axis=1), table.sum(axis=0))
    disagreement = np.abs(np.arange(categories)[:, None] - np.arange(categories)[None, :])
    if not weighted:
        disagreement = (disagreement > 0).astype(float)
    denominator = float(np.sum(disagreement * expected))
    return float(1 - np.sum(disagreement * table) / denominator) if denominator > 0 else None


def annotation_audit(documents, annotations, replicates=2000):
    if not annotations:
        return {"status": "not_available", "n_independent_annotations": 0, "n_paired_documents": 0,
                "raw_agreement": None, "binary_kappa": None, "ordinal_kappa": None,
                "reason": "No independent human ratings supplied.", "errors": [], "resolved_quality": {}}
    ids = {row["id"]: row for row in documents}
    grouped = defaultdict(list)
    errors = []
    rubric_hash = sha256(ROOT / "rubric.json")
    for row in annotations:
        if row.get("id") not in ids:
            errors.append("Annotation references an unknown document.")
            continue
        if any(row.get(flag) != "true" for flag in ("blinded", "independent", "human")):
            errors.append("Nonhuman, nonindependent or unblinded rating is ineligible.")
            continue
        if row.get("rubric_sha256") != rubric_hash or not row.get("rater_id"):
            errors.append("Rating lacks a valid rubric hash or pseudonymous rater ID.")
            continue
        if not row.get("rationale") or not row.get("evidence"):
            errors.append("Rating lacks rationale or evidence reference (use not_applicable when justified).")
            continue
        try:
            derive_quality(row)
        except ValueError as error:
            errors.append(str(error))
            continue
        grouped[row["id"]].append(row)
    pairs = defaultdict(list)
    resolved = {}
    uncertain = 0
    adjudication_required = 0
    for identifier, ratings in grouped.items():
        primary = sorted([row for row in ratings if row.get("role") == "primary"], key=lambda row: row["rater_id"])
        adjudicators = [row for row in ratings if row.get("role") == "adjudicator"]
        if len(primary) != 2 or primary[0]["rater_id"] == primary[1]["rater_id"]:
            errors.append(f"{identifier}: requires exactly two distinct primary human raters.")
            continue
        first, second = primary
        binary = (derive_quality(first), derive_quality(second))
        pair_id = "|".join(row["rater_id"] for row in primary)
        pairs[pair_id].append((identifier, first, second))
        if None in binary:
            uncertain += 1
        disagree = any(first[field] != second[field] for field in (*DIMENSIONS, "critical_error"))
        if disagree or None in binary:
            adjudication_required += 1
            if len(adjudicators) != 1 or adjudicators[0]["rater_id"] in {row["rater_id"] for row in primary}:
                errors.append(f"{identifier}: requires a distinct human adjudicator.")
                continue
            resolved[identifier] = derive_quality(adjudicators[0])
        else:
            resolved[identifier] = binary[0]
    pair_reports = []
    rng = np.random.default_rng(20260910)
    for pair_id, records in pairs.items():
        complete = [record for record in records if derive_quality(record[1]) is not None and derive_quality(record[2]) is not None]
        first = [derive_quality(record[1]) for record in complete]
        second = [derive_quality(record[2]) for record in complete]
        clusters = sorted({ids[record[0]]["cluster_id"] for record in complete})
        boot = []
        if len(clusters) >= 2:
            cluster_records = {cluster: [record for record in complete if ids[record[0]]["cluster_id"] == cluster] for cluster in clusters}
            for _ in range(replicates):
                sampled = [record for cluster in rng.choice(clusters, size=len(clusters), replace=True) for record in cluster_records[cluster]]
                value = kappa([derive_quality(record[1]) for record in sampled], [derive_quality(record[2]) for record in sampled])
                if value is not None:
                    boot.append(value)
        ordinal = {}
        for dimension in DIMENSIONS:
            rated = [record for record in records if record[1][dimension] != "uncertain" and record[2][dimension] != "uncertain"]
            ordinal[dimension] = {"n": len(rated), "linear_weighted_kappa": kappa(
                [int(record[1][dimension]) for record in rated], [int(record[2][dimension]) for record in rated], 3, True)}
        pair_reports.append({"rater_pair": pair_id, "n": len(records), "n_binary_complete": len(complete),
                             "raw_agreement": float(np.mean(np.array(first) == np.array(second))) if complete else None,
                             "binary_kappa": kappa(first, second), "ordinal": ordinal,
                             "binary_kappa_cluster_ci95": np.quantile(boot, [0.025, 0.975]).tolist() if boot else None,
                             "bootstrap_valid": len(boot), "bootstrap_undefined_or_unavailable": replicates - len(boot)})
    missing = sorted(set(ids) - set(grouped))
    if missing:
        errors.append(f"{len(missing)} documents have no eligible independent ratings.")
    return {"status": "audited" if not errors else "incomplete", "n_independent_annotations": sum(len(value) for value in grouped.values()),
            "n_paired_documents": sum(len(value) for value in pairs.values()), "pair_reports": pair_reports,
            "uncertain_primary_documents": uncertain, "adjudication_required": adjudication_required,
            "resolved_quality": resolved, "errors": errors,
            "note": "Pair-specific Cohen kappas; no fabricated pooled Cohen kappa across changing raters. Attestations require external human verification."}


def audit():
    sampling = read_json(ROOT / "sampling.json")
    protocol = read_json(ROOT / "protocol.json")["confirmatory"]
    documents = read_csv(ROOT / "data/corpus.csv")
    annotations = read_csv(ROOT / "data/annotations.csv")
    custody = read_json(ROOT / "data/custody.json")
    issues = []
    if not documents:
        issues.append("No contemporary corpus has been acquired.")
    historical_hashes = {normalized_hash(row["text"]) for row in read_csv("pilot/artifacts/corpus.csv")}
    clusters, documents_seen, identifiers = {}, set(), set()
    linkage = {name: {} for name in ("source_id", "author_id", "prompt_family_id")}
    token_sets = []
    for row in documents:
        identifier = row.get("id", "")
        if not identifier or identifier in identifiers:
            issues.append("Empty or duplicate document ID.")
        identifiers.add(identifier)
        required = ("text", "cluster_id", "source_id", "author_id", "topic", "genre", "created_date",
                    "origin_evidence", "permission", "permission_evidence", "collection_method")
        if any(row.get(field, "") in MISSING for field in required if field != "author_id") or not row.get("author_id"):
            issues.append(f"{identifier}: missing provenance or permission fields.")
        if row.get("origin") == "human" and row.get("author_id", "") in MISSING:
            issues.append(f"{identifier}: human provenance needs an audited pseudonymous author ID.")
        if row.get("pii_reviewed") != "true":
            issues.append(f"{identifier}: PII review is not attested.")
        if row.get("role") not in {"train", "tune", "calibration"}:
            issues.append(f"{identifier}: final/challenge texts must remain with the custodian.")
        if row.get("origin") not in {"human", "generated"}:
            issues.append(f"{identifier}: uncertain/mixed origins are challenge-only, not binary development labels.")
        if row.get("origin") == "generated" and any(row.get(field, "") in MISSING for field in ("generator_id", "prompt_family_id")):
            issues.append(f"{identifier}: generation provenance is incomplete.")
        try:
            created = date.fromisoformat(row.get("created_date", ""))
            if not date.fromisoformat(sampling["date_start"]) <= created <= date.fromisoformat(sampling["date_end"]):
                issues.append(f"{identifier}: outside the frozen contemporary window.")
        except ValueError:
            issues.append(f"{identifier}: invalid creation date.")
        text = row.get("text", "")
        normalized = normalized_hash(text)
        if normalized in historical_hashes or normalized in documents_seen:
            issues.append(f"{identifier}: old pilot or duplicate text is ineligible for this corpus.")
        documents_seen.add(normalized)
        tokens = detector.words(text)
        if len(tokens) < detector.MIN_WORDS:
            issues.append(f"{identifier}: short text belongs to the abstention challenge.")
        trigrams = set(zip(tokens, tokens[1:], tokens[2:]))
        for other_id, other_role, other_cluster, other_tokens in token_sets:
            union = trigrams | other_tokens
            if union and len(trigrams & other_tokens) / len(union) >= 0.8 and row.get("cluster_id") != other_cluster:
                issues.append(f"{identifier}/{other_id}: near duplicates assigned to different clusters.")
        token_sets.append((identifier, row.get("role"), row.get("cluster_id"), trigrams))
        for mapping, value in [(clusters, row.get("cluster_id")), *[(linkage[field], row.get(field)) for field in linkage]]:
            if value not in MISSING and value is not None:
                previous = mapping.setdefault(value, row.get("role"))
                if previous != row.get("role"):
                    issues.append(f"{identifier}: related source/author/prompt/cluster crosses roles.")
    agreement = annotation_audit(documents, annotations)
    issues.extend(agreement["errors"])
    complete_pairs = sum(pair["n_binary_complete"] for pair in agreement.get("pair_reports", []))
    if complete_pairs < protocol["minimum_agreement_audit_documents"]:
        issues.append("Independent agreement audit has fewer than 200 binary-complete paired documents.")
    for pair in agreement.get("pair_reports", []):
        if pair["binary_kappa"] is None or pair["binary_kappa"] < protocol["minimum_quality_kappa"] or pair["raw_agreement"] is None or pair["raw_agreement"] < protocol["minimum_raw_quality_agreement"]:
            issues.append("At least one rater pair fails the frozen agreement criteria.")
    for row in documents:
        derived = agreement["resolved_quality"].get(row["id"])
        expected = str(derived) if derived is not None else "uncertain"
        if row.get("quality") != expected:
            issues.append(f"{row['id']}: quality label differs from independent resolution.")
    counts = Counter(row.get("role") for row in documents)
    eligible_counts = Counter(row.get("role") for row in documents if row.get("quality") in {"0", "1"} and row.get("origin") in {"human", "generated"})
    for role in ("train", "tune", "calibration"):
        if eligible_counts[role] < sampling["target_documents"][role]:
            issues.append(f"{role}: {eligible_counts[role]} binary-labeled documents, target {sampling['target_documents'][role]}.")
        cells = {(row.get("origin"), row.get("quality")) for row in documents if row.get("role") == role}
        if not {(origin, quality) for origin in ("human", "generated") for quality in ("0", "1")}.issubset(cells):
            issues.append(f"{role}: the four origin-by-quality cells are incomplete.")
    for target in ("ai", "low_quality"):
        negative_clusters = {row["cluster_id"] for row in documents if row.get("role") == "calibration" and
                             (row.get("origin") == "human" if target == "ai" else row.get("quality") == "0")}
        if len(negative_clusters) < protocol["minimum_independent_calibration_negatives_per_head"]:
            issues.append(f"{target}: fewer than 72 independent negative calibration clusters.")
    generators = {row["generator_id"] for row in documents if row.get("origin") == "generated"}
    if len(generators) < 3:
        issues.append("Fewer than three documented development generator IDs; family independence needs external verification.")
    if not custody.get("independent_human_annotation_attestation") or not custody.get("custodian_attestation"):
        issues.append("Independent human annotation and custodian attestations are missing.")
    if not custody.get("source_and_author_disjointness_attestation"):
        issues.append("External final-set source/author disjointness audit is missing.")
    if custody.get("final_unseen_by_developer") is not True or any(not custody.get(field) for field in
            ("final_texts_sha256", "final_labels_sha256", "challenge_texts_sha256", "challenge_labels_sha256")):
        issues.append("A separately held, hash-committed untouched final and challenge set is unavailable.")
    if not (ROOT / "model_freeze.json").exists():
        issues.append("No confirmatory model/threshold freeze exists; the old pilot model is not eligible.")
    write_json(RESULTS / "annotation_agreement.json", agreement)
    report = {"status": "not_ready" if issues else "requires_external_release_review", "n_contemporary_documents": len(documents),
              "role_counts": dict(counts), "binary_eligible_role_counts": dict(eligible_counts),
              "quality_uncertain_or_missing": sum(row.get("quality") not in {"0", "1"} for row in documents),
              "blockers": list(dict.fromkeys(issues)),
              "final_evaluation": None, "challenge_evaluation": None, "error_adjusted_prevalence": None,
              "annual_trends": None, "automatic_release_permitted": False,
              "note": "Metadata checks cannot establish genuine consent, human independence, representativeness, or unseen custody. No final scoring is implemented or authorized by a Boolean attestation."}
    write_json(RESULTS / "readiness.json", report)
    print(json.dumps(report, indent=2))
    return report


def freeze():
    files = [ROOT / name for name in ("protocol.json", "rubric.json", "sampling.json")]
    files += [Path("detector.py"), Path("research_cycle.py")]
    record = {"status": "locally_frozen_methodology_not_external_preregistration", "date": "2026-09-10",
              "files": {str(path): sha256(path) for path in files},
              "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
              "model_frozen": False, "corpus_acquired": False, "final_evaluation_authorized": False}
    destination = ROOT / "method_freeze.json"
    if destination.exists() and read_json(destination) != record:
        raise ValueError("A different method freeze exists; create a versioned amendment rather than overwrite it.")
    write_json(destination, record)
    print(json.dumps(record, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("diagnose", "snapshot"))
    args = parser.parse_args()
    try:
        if args.stage == "diagnose":
            diagnose()
        elif args.stage == "snapshot":
            from cycle2_workflow import method_hashes
            record = {"date": "2026-09-10", "status": "local_method_snapshot_not_external_preregistration",
                      "canonical_files": method_hashes(),
                      "diagnostic_files": {path: sha256(path) for path in ("research_cycle.py", "cycle2/exploratory_plan.json")},
                      "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
                      "confirmatory_model_frozen": False, "qualified_corpus_acquired": False,
                      "independent_annotation_audit_completed": False, "final_evaluation_completed": False}
            destination = ROOT / "method_snapshot.json"
            if destination.exists() and read_json(destination) != record:
                raise ValueError("Method snapshot differs; preserve it and record a versioned amendment.")
            write_json(destination, record)
            print(json.dumps(record, indent=2))
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()