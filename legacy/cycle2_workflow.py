"""Admission, calibration, and sealed evaluation; metadata assertions need human verification."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np

import detector
from cycle2_stats import agreement, classification, cluster_intervals, np_threshold


ROOT = Path("cycle2")
DIMENSIONS = ("factual", "information", "coherence")
METADATA = ("id", "text", "cluster", "source", "topic", "author_group", "genre", "proficiency",
            "formatting", "generator_family", "authorship_pattern", "origin_evidence", "rights_basis",
            "rights_evidence", "privacy_review_evidence", "publication_date", "collection_date", "condition")
METHOD_FILES = ("cycle2/protocol.json", "cycle2/rubric.md", "cycle2/sampling.md",
                "cycle2_workflow.py", "cycle2_stats.py", "detector.py")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def method_hashes():
    return {name: sha(name) for name in METHOD_FILES}


def protocol():
    return read_json(ROOT / "protocol.json")


def text_hash(text):
    return hashlib.sha256(" ".join(detector.words(text)).encode()).hexdigest()


def quality(rating):
    if type(rating.get("uncertain")) is not bool:
        raise ValueError("Every rating needs an explicit uncertainty decision.")
    if rating.get("uncertain") is True:
        return None
    if any(type(rating.get(name)) is not int or rating[name] not in (0, 1, 2) for name in DIMENSIONS):
        raise ValueError("Resolved annotations require three integer dimension scores in [0, 2].")
    if type(rating.get("central_error")) is not bool:
        raise ValueError("central_error must be a Boolean.")
    return int(rating["central_error"] or sum(rating[name] == 2 for name in DIMENSIONS) >= 2)


def audit_bundle(path, expected_role=None):
    config, bundle = protocol(), read_json(path)
    errors, resolved, paired = [], [], []
    if bundle.get("role") not in config["roles"] or (expected_role and bundle.get("role") != expected_role):
        errors.append("Incorrect or missing bundle role.")
    if not bundle.get("sampling_frame") or not bundle.get("selection_method"):
        errors.append("Sampling frame and selection method are required.")
    review = bundle.get("review", {})
    for field in ("independent_human_review", "rights_review_complete", "privacy_review_complete",
                  "provenance_review_complete", "annotation_process_review_complete"):
        if review.get(field) is not True:
            errors.append(f"Human review is not complete: {field}.")
    if not review.get("reviewer_id") or not review.get("evidence_reference"):
        errors.append("Human reviewer and review evidence must be documented.")
    panel = bundle.get("panel", [])
    registry = {item["id"]: item for item in bundle.get("annotators", []) if item.get("id")}
    if len(panel) != 2 or len(set(panel)) != 2:
        errors.append("A fixed pair of distinct independent human raters is required.")
    for identifier in panel:
        rater = registry.get(identifier, {})
        if not all(rater.get(field) is True for field in ("human", "independent_of_authors_and_generator")) or not rater.get("evidence_reference"):
            errors.append(f"Rater independence evidence missing: {identifier}.")
    ratings = defaultdict(list)
    for rating in bundle.get("annotations", []):
        ratings[rating.get("document_id")].append(rating)
    documents = bundle.get("documents", [])
    identifiers, normalized, cluster_owners = set(), {}, {}
    for document in documents:
        if any(not isinstance(document.get(field), str) or not document[field].strip() for field in METADATA):
            errors.append("A document is missing required text/provenance/stratum metadata.")
            continue
        identifier = document["id"]
        if identifier in identifiers:
            errors.append("Duplicate document ID.")
        identifiers.add(identifier)
        if "ai" not in document or document.get("ai") not in (0, 1, None) or isinstance(document.get("ai"), bool):
            errors.append("ai must be 0, 1, or null.")
        for field in ("source", "author_group"):
            if document[field] != "not_applicable":
                entity = (field, document[field])
                owner = cluster_owners.setdefault(entity, document["cluster"])
                if owner != document["cluster"]:
                    errors.append("A source or author crosses declared leakage clusters.")
        expected_pattern = {0: "human", 1: "fully_generated"}.get(document.get("ai"))
        if expected_pattern and document["authorship_pattern"] != expected_pattern:
            errors.append("Origin label and authorship pattern disagree.")
        if document.get("ai") is None and document["authorship_pattern"] not in ("mixed", "unknown"):
            errors.append("Unknown binary origin needs mixed/unknown authorship metadata.")
        try:
            published, collected = date.fromisoformat(document["publication_date"]), date.fromisoformat(document["collection_date"])
            if not date.fromisoformat(config["contemporary_start"]) <= published <= collected <= date.fromisoformat(config["collection_cutoff"]):
                errors.append("Document dates are outside the frozen collection frame.")
        except ValueError:
            errors.append("Invalid ISO date in document metadata.")
        digest = text_hash(document["text"])
        if digest in normalized:
            prior = normalized[digest]
            if not document["condition"].startswith("challenge_") or document.get("parent_id") != prior["id"] or document["cluster"] != prior["cluster"]:
                errors.append("Unapproved normalized duplicate; challenge variants must reference an earlier same-cluster parent.")
        else:
            normalized[digest] = document
        entries = ratings.get(identifier, [])
        independent = [entry for entry in entries if entry.get("role") == "independent"]
        if len(independent) != 2 or {entry.get("annotator_id") for entry in independent} != set(panel):
            errors.append("Each document needs both fixed-panel independent ratings.")
            continue
        if any(entry.get("blinded") is not True or not entry.get("evidence_reference") for entry in entries):
            errors.append("Annotation blinding and evidence references are required.")
        ordered = sorted(independent, key=lambda entry: panel.index(entry["annotator_id"]))
        try:
            first, second = map(quality, ordered)
            if first is not None and second is not None:
                paired.append({"first": first, "second": second, "cluster": document["cluster"], "ratings": ordered})
            if first is not None and first == second:
                final_quality = first
            else:
                adjudications = [entry for entry in entries if entry.get("role") == "adjudication"]
                if len(adjudications) != 1:
                    final_quality = None
                else:
                    decision = adjudications[0]
                    rater = registry.get(decision.get("annotator_id"), {})
                    if decision.get("annotator_id") in panel or rater.get("human") is not True or rater.get("independent_of_authors_and_generator") is not True or not rater.get("evidence_reference"):
                        errors.append("Adjudicator must be a distinct documented independent human.")
                    final_quality = quality(decision)
            resolved.append({**document, "low_quality": final_quality, "normalized_sha256": digest})
        except ValueError as error:
            errors.append(str(error))
    if set(ratings) - identifiers:
        errors.append("Annotations reference unknown documents.")
    binary = agreement([item["first"] for item in paired], [item["second"] for item in paired])
    dimension_results = {}
    for dimension in DIMENSIONS:
        dimension_results[dimension] = agreement([item["ratings"][0][dimension] for item in paired],
                                                [item["ratings"][1][dimension] for item in paired], (0, 1, 2), quadratic=True)
    bounds, valid_replicates = None, 0
    groups = sorted({item["cluster"] for item in paired})
    if len(groups) >= 2:
        rng = np.random.default_rng(config["seed"])
        grouped = [[item for item in paired if item["cluster"] == group] for group in groups]
        kappas = []
        for _ in range(config["bootstrap_replicates"]):
            sample = [item for group_index in rng.integers(0, len(groups), len(groups)) for item in grouped[group_index]]
            value = agreement([item["first"] for item in sample], [item["second"] for item in sample])["kappa"]
            if value is not None:
                kappas.append(value)
        if kappas:
            bounds, valid_replicates = np.quantile(kappas, [0.025, 0.975]).tolist(), len(kappas)
    coverage = sum(item["low_quality"] is not None for item in resolved) / len(documents) if documents else None
    if binary["n"] < config["minimum_audit_documents"]:
        errors.append("Insufficient independent annotation pairs for the admission audit.")
    if binary["raw_agreement"] is None or binary["raw_agreement"] < config["minimum_raw_agreement"]:
        errors.append("Raw agreement does not meet the frozen admission criterion.")
    if bounds is None or bounds[0] < config["minimum_kappa_lower_bound"]:
        errors.append("Kappa uncertainty does not meet the frozen admission criterion.")
    if coverage is None or coverage < config["minimum_quality_resolution_fraction"]:
        errors.append("Too many unresolved quality labels.")
    report = {"status": "admissible_on_supplied_records" if not errors else "blocked",
              "errors": sorted(set(errors)), "documents": len(documents), "resolved_fraction": coverage,
              "binary_agreement": binary, "binary_kappa_cluster_95": bounds,
              "valid_kappa_bootstrap_replicates": valid_replicates, "dimension_agreement": dimension_results,
              "adjudication_records": sum(entry.get("role") == "adjudication" for entry in bundle.get("annotations", [])),
              "verification_limit": "The code validates supplied records, not real consent, human identity, independence, or genuine final-data blinding."}
    return bundle, resolved, report


def require_admitted(path, role):
    bundle, documents, audit = audit_bundle(path, role)
    if audit["errors"]:
        raise ValueError("Corpus admission blocked: " + "; ".join(audit["errors"]))
    return bundle, documents, audit


def disjoint(first, second, topics=False):
    for field in ("cluster", "source", "author_group") + (("topic",) if topics else ()):
        left = {item[field] for item in first} - {"not_applicable"}
        right = {item[field] for item in second} - {"not_applicable"}
        if left & right:
            raise ValueError(f"Leakage across partitions: shared {field}.")
    if {item["normalized_sha256"] for item in first} & {item["normalized_sha256"] for item in second}:
        raise ValueError("Normalized text crosses partitions.")


def eligible(documents):
    return [item for item in documents if item["ai"] in (0, 1) and item["low_quality"] in (0, 1)
            and len(detector.words(item["text"])) >= detector.MIN_WORDS]


def fit(development_path, calibration_path, output):
    if (output / "model.json").exists():
        raise ValueError("A fitted artifact already exists here; do not silently overwrite it.")
    config = protocol()
    if config["primary_features"] != detector.FEATURES:
        raise ValueError("Feature implementation and frozen protocol disagree.")
    _, development, dev_audit = require_admitted(development_path, "development")
    calibration_bundle, calibration, cal_audit = require_admitted(calibration_path, "calibration")
    if calibration_bundle["review"].get("calibration_independence_reviewed") is not True:
        raise ValueError("Independent calibration assumptions need documented human review.")
    disjoint(development, calibration)
    training, calibration_rows = eligible(development), eligible(calibration)
    if not training or not calibration_rows:
        raise ValueError("No eligible training/calibration documents.")
    if {(item["ai"], item["low_quality"]) for item in training} != {(0, 0), (0, 1), (1, 0), (1, 1)}:
        raise ValueError("Development data must include all four origin-by-quality cells.")
    matrix = np.vstack([detector.features(item["text"]) for item in training])
    mean, scale = matrix.mean(axis=0), matrix.std(axis=0)
    scale[scale < 1e-12] = 1
    standardized = (matrix - mean) / scale
    calibration_matrix = (np.vstack([detector.features(item["text"]) for item in calibration_rows]) - mean) / scale
    artifact = {"version": 1, "features": detector.FEATURES, "min_words": detector.MIN_WORDS,
                "mean": mean.tolist(), "scale": scale.tolist(), "heads": {},
                "regularization": config["primary_regularization"],
                "warning": "Research manual-review aid only; uncalibrated scores, not proof of origin or poor quality."}
    threshold_records = {}
    for target in detector.TARGETS:
        labels = np.array([item[target] for item in training])
        if set(labels) != {0, 1}:
            raise ValueError(f"Training is missing a {target} class.")
        coefficients = detector.fit_head(standardized, labels, config["primary_regularization"])
        scores = detector.scores(calibration_matrix, coefficients)
        representatives = {}
        for index in sorted(range(len(calibration_rows)), key=lambda index: calibration_rows[index]["id"]):
            document = calibration_rows[index]
            if document[target] == 0:
                representatives.setdefault(document["cluster"], index)
        if len(representatives) < config["minimum_calibration_negative_clusters_per_head"]:
            raise ValueError(f"Insufficient independent negative clusters for {target} calibration.")
        threshold_record = np_threshold(scores[list(representatives.values())], config["alpha_per_head"], config["delta_per_head"])
        artifact["heads"][target] = {"coefficients": coefficients.tolist(), "threshold": threshold_record["threshold"]}
        threshold_records[target] = threshold_record
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "model.json", artifact)
    record = {"methods": method_hashes(), "inputs": {str(development_path): sha(development_path), str(calibration_path): sha(calibration_path)},
              "model_sha256": sha(output / "model.json"), "development_audit": dev_audit, "calibration_audit": cal_audit,
              "thresholds": threshold_records, "final_data_accessed": False,
              "training_eligible": len(training), "calibration_eligible": len(calibration_rows)}
    write_json(output / "fit_record.json", record)
    print("Fitted and independently calibrated candidate; final validity remains untested.")


def verify_fit(output):
    record = read_json(output / "fit_record.json")
    if record["methods"] != method_hashes() or record["model_sha256"] != sha(output / "model.json"):
        raise ValueError("Methods or model changed after fitting.")
    if any(sha(path) != expected for path, expected in record["inputs"].items()):
        raise ValueError("Development/calibration files changed after fitting.")
    return record


def freeze(output, seal_path):
    verify_fit(output)
    if (output / "freeze.json").exists():
        raise ValueError("Freeze already exists; create a new protocol/version for changes.")
    seal = read_json(seal_path)
    if not isinstance(seal.get("evaluation_sha256"), str) or len(seal["evaluation_sha256"]) != 64 or any(character not in "0123456789abcdef" for character in seal["evaluation_sha256"]):
        raise ValueError("An actual custodian-supplied evaluation hash is required.")
    if any(seal.get(field) is not True for field in ("independent_human_custodian", "developers_have_not_accessed_final_text_or_labels")) or not seal.get("custodian_id") or not seal.get("sealing_evidence_reference"):
        raise ValueError("Independent custody and no-access attestations are required.")
    write_json(output / "freeze.json", {"fit_record_sha256": sha(output / "fit_record.json"),
                                        "seal": seal, "method_hashes": method_hashes()})
    print("Local final-evaluation freeze recorded; human custody assertions still require external verification.")


def evaluate(output, evaluation_path):
    record = verify_fit(output)
    frozen = read_json(output / "freeze.json")
    if frozen["fit_record_sha256"] != sha(output / "fit_record.json") or frozen["method_hashes"] != method_hashes():
        raise ValueError("Frozen methods changed.")
    if sha(evaluation_path) != frozen["seal"]["evaluation_sha256"]:
        raise ValueError("Released evaluation data do not match the custodian's pre-evaluation hash.")
    marker = output / "evaluation_started.json"
    with marker.open("x", encoding="utf-8") as handle:
        json.dump({"evaluation_sha256": sha(evaluation_path), "notice": "This final set is now accessed; do not retune against it."}, handle)
    config = protocol()
    _, documents, audit = require_admitted(evaluation_path, "evaluation")
    prior = []
    for path in record["inputs"]:
        _, rows, _ = audit_bundle(path)
        prior.extend(rows)
    primary = [item for item in documents if item["condition"] == config["primary_condition"]]
    disjoint(prior, documents)
    disjoint(prior, primary, topics=True)
    disjoint(prior, [item for item in documents if item["condition"] == "challenge_topic"], topics=True)
    pilot_hashes = {text_hash(item["text"]) for item in detector.load_data(Path("pilot/artifacts/corpus.csv"), targets=("ai",))}
    if any(item["normalized_sha256"] in pilot_hashes for item in documents):
        raise ValueError("The old pilot cannot be reused as new final data.")
    cells = Counter((item["ai"], item["low_quality"]) for item in eligible(primary))
    if any(cells[(origin, quality_label)] < config["minimum_final_per_origin_quality_cell"] for origin in (0, 1) for quality_label in (0, 1)):
        raise ValueError("Insufficient primary final origin-by-quality cells.")
    for field, minimum in (("cluster", config["minimum_final_clusters"]), ("genre", config["minimum_final_genres"]), ("topic", config["minimum_final_topics"])):
        if len({item[field] for item in primary}) < minimum:
            raise ValueError(f"Insufficient final {field} diversity.")
    if len({item["generator_family"] for item in primary if item["ai"] == 1}) < config["minimum_final_generator_families"]:
        raise ValueError("Insufficient independent generator families.")
    for condition in config["required_challenges"]:
        if sum(item["condition"] == condition for item in documents) < config["minimum_challenge_documents_per_condition"]:
            raise ValueError(f"Required challenge missing or undersized: {condition}.")
    document_index = {item["id"]: item for item in documents}
    for item in documents:
        if item["condition"] == "challenge_short" and len(detector.words(item["text"])) >= detector.MIN_WORDS:
            raise ValueError("Short-text challenge includes a non-short document.")
        if item["condition"] == "challenge_mixed" and (item["ai"] is not None or item["authorship_pattern"] != "mixed"):
            raise ValueError("Mixed-authorship challenge must not invent binary origin labels.")
        if item["condition"] == "challenge_translation" and not item.get("translation_provenance"):
            raise ValueError("Translation challenge requires documented translation provenance.")
        if item["condition"] == "challenge_format":
            parent = document_index.get(item.get("parent_id"), {})
            if parent.get("cluster") != item["cluster"] or parent.get("normalized_sha256") != item["normalized_sha256"] or parent.get("ai") != item["ai"]:
                raise ValueError("Formatting controls must preserve a same-cluster parent's words and origin.")
    training_generators = {item["generator_family"] for item in prior if item["ai"] == 1}
    if not any(item["condition"] == "challenge_generator" and item["ai"] == 1 for item in documents):
        raise ValueError("Unseen-generator challenge has no documented generated examples.")
    if any(item["generator_family"] in training_generators for item in documents if item["condition"] == "challenge_generator" and item["ai"] == 1):
        raise ValueError("Unseen-generator challenge reuses a development/calibration generator.")
    artifact = read_json(output / "model.json")
    report = {"status": "evaluated_on_supplied_admitted_records", "audit": audit,
              "freeze_sha256": sha(output / "freeze.json"), "conditions": {},
              "prevalence": {"status": "not_identified", "reason": "No independently audited probability sample or transportability study."},
              "warning": "Metadata and aggregate accuracy cannot establish representativeness or human-review truth; manual-review research only."}
    for condition in sorted({item["condition"] for item in documents}):
        subset = [item for item in documents if item["condition"] == condition]
        scored = [(item, detector.predict(item["text"], artifact)) for item in subset]
        condition_report = {"total": len(subset), "scored": sum(prediction["status"] == "scored" for _, prediction in scored), "targets": {}, "subgroups": {}}
        for target in (*detector.TARGETS, "joint"):
            records = []
            for item, prediction in scored:
                if prediction["status"] != "scored":
                    continue
                label = item["ai"] * item["low_quality"] if target == "joint" and item["ai"] is not None and item["low_quality"] is not None else (item.get(target) if target != "joint" else None)
                if label is None:
                    continue
                score = float(prediction["review_candidate"]) if target == "joint" else prediction["heads"][target]["score"]
                threshold = 0.5 if target == "joint" else artifact["heads"][target]["threshold"]
                records.append((item, label, score, threshold))
            labels, scores, thresholds = ([item[index] for item in records] for index in (1, 2, 3))
            threshold = thresholds[0] if thresholds else 0.5
            result = classification(labels, scores, threshold)
            result["label_and_length_coverage"] = len(records) / len(subset)
            if target == "joint":
                result.pop("auroc", None)
            if records:
                result["cluster_bootstrap"] = cluster_intervals(labels, scores, thresholds, [item[0]["cluster"] for item in records], config["seed"], config["bootstrap_replicates"])
                result["always_negative_control"] = classification(labels, np.zeros(len(records)), 0.5)
            condition_report["targets"][target] = result
            for field in config["subgroups"]:
                condition_report["subgroups"].setdefault(field, {}).setdefault(target, {})
                for value in sorted({item[0][field] for item in records}):
                    members = [item for item in records if item[0][field] == value]
                    summary = classification([item[1] for item in members], [item[2] for item in members], threshold)
                    summary["small_cell_warning"] = len(members) < 20
                    if target == "joint":
                        summary.pop("auroc", None)
                    condition_report["subgroups"][field][target][value] = summary
        report["conditions"][condition] = condition_report
    write_json(output / "final_results.json", report)
    print("Saved final and challenge results without changing model or thresholds.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("bundle", type=Path)
    audit.add_argument("report", type=Path)
    training = commands.add_parser("fit")
    training.add_argument("development", type=Path)
    training.add_argument("calibration", type=Path)
    training.add_argument("output", type=Path)
    freezing = commands.add_parser("freeze")
    freezing.add_argument("output", type=Path)
    freezing.add_argument("seal", type=Path)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("output", type=Path)
    evaluation.add_argument("bundle", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "audit":
            if args.bundle.resolve() == args.report.resolve():
                raise ValueError("Audit output must not overwrite its input bundle.")
            _, _, report = audit_bundle(args.bundle)
            write_json(args.report, report)
            print(json.dumps(report, indent=2))
            if report["errors"]:
                raise SystemExit(2)
        elif args.command == "fit":
            fit(args.development, args.calibration, args.output)
        elif args.command == "freeze":
            freeze(args.output, args.seal)
        else:
            evaluate(args.output, args.bundle)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()