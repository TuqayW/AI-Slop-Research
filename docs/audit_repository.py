"""Audit the uploaded record without fitting models or changing historical data."""

import ast
import csv
import hashlib
import importlib.util
import json
import platform
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def counts(rows, fields):
    return {" | ".join(key): value for key, value in sorted(
        Counter(tuple(row[field] for field in fields) for row in rows).items())}


def duplicate_groups(rows, normalize):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        key = hashlib.sha256(normalize(row["text"]).encode()).hexdigest()
        groups[key].append({"row": index + 2, "id": row.get("id"),
                            "group": row.get("prompt_id", row.get("group_id")),
                            "split": row.get("split"),
                            "label": row.get("ai", row.get("label"))})
    return [{"text_sha256": key, "members": members,
             "cross_split": len({member["split"] for member in members}) > 1}
            for key, members in groups.items() if len(members) > 1]


def check_metrics(value, location, checks):
    if isinstance(value, dict):
        if {"tn", "fp", "fn", "tp"}.issubset(value):
            true_negative, false_positive, false_negative, true_positive = (
                value[key] for key in ("tn", "fp", "fn", "tp"))
            total = true_negative + false_positive + false_negative + true_positive
            precision_denominator = true_positive + false_positive
            f1_denominator = 2 * true_positive + false_positive + false_negative
            expected = {
                "n": total,
                "accuracy": (true_negative + true_positive) / total if total else None,
                "precision": true_positive / precision_denominator if precision_denominator else None,
                "recall": true_positive / (true_positive + false_negative) if true_positive + false_negative else None,
                "fpr": false_positive / (false_positive + true_negative) if false_positive + true_negative else None,
                "f1": 2 * true_positive / f1_denominator if f1_denominator else None,
            }
            errors = []
            for key, target in expected.items():
                if key not in value:
                    continue
                actual = value[key]
                if target is None:
                    if actual not in (None, 0):
                        errors.append(key)
                elif actual is None or abs(actual - target) > 1e-12:
                    errors.append(key)
            checks.append({"location": location, "errors": errors})
        for key, child in value.items():
            check_metrics(child, location + "/" + key, checks)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            check_metrics(child, location + "/" + str(index), checks)


def main():
    output = Path("docs/audit_results.json")
    report = {"audit_date": "2026-09-10", "scope": "retrospective repository audit; no new detector evaluation"}
    manifest = {}
    for folder in ("training", "analysis", "evaluation", "dataset", "data", "models", "samples", "legacy", "tools", "release"):
        for path in sorted(Path(folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                manifest[str(path)] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    for name in ("README.md", "docs/research_cycle.py", "detector_v11.py"):
        path = Path(name)
        manifest[name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    report["input_manifest"] = manifest
    report["environment"] = {"python": platform.python_version(),
        "pandas": pd.__version__, "sklearn_available": importlib.util.find_spec("sklearn") is not None,
        "joblib_available": importlib.util.find_spec("joblib") is not None}
    datasets = {}
    for path in sorted(Path("dataset").glob("*.csv")):
        rows = load_csv(path)
        label = "ai" if "ai" in rows[0] else "label"
        record = {"n": len(rows), "columns": list(rows[0]), "labels": counts(rows, [label])}
        if "split" in rows[0]:
            record["split_labels"] = counts(rows, ["split", label])
            for field in ("group", "prompt_id", "group_id"):
                if field in rows[0]:
                    groups = defaultdict(set)
                    for row in rows:
                        groups[row[field]].add(row["split"])
                    record[field] = {"unique": len(groups), "cross_split": sum(len(splits) > 1 for splits in groups.values())}
            record["raw_duplicates"] = duplicate_groups(rows, str)
            record["normalized_duplicates"] = duplicate_groups(rows, lambda text: re.sub(r"\s+", " ", text).strip().lower())
        if "low_quality" in rows[0]:
            record["nonempty_quality_labels"] = sum(bool(row["low_quality"].strip()) for row in rows)
        if path.name in {"combined_v11.csv", "hc3_paired_v10.csv"}:
            record["composition"] = counts(rows, ["split", "source" if "source" in rows[0] else "domain", label])
        datasets[str(path)] = record
    report["datasets"] = datasets
    benchmark = load_csv(Path("dataset/human_detectors_grouped.csv"))
    article_groups = defaultdict(set)
    for row in benchmark:
        article_groups[(row["source"], row["title"], row["issue"])].add(row["split"])
    report["benchmark"] = {
        "article_pairs": len(article_groups),
        "article_pairs_crossing_splits": sum(len(value) > 1 for value in article_groups.values()),
        "generation_conditions": counts(benchmark, ["generation_model", "ai"]),
        "mean_whitespace_words": {label: sum(len(row["text"].split()) for row in benchmark if row["ai"] == label) / sum(row["ai"] == label for row in benchmark) for label in ("0", "1")},
        "formatting_documents": {pattern: {label: sum(bool(re.search(pattern, row["text"])) for row in benchmark if row["ai"] == label) for label in ("0", "1")} for pattern in (r"<[^>]+>", r"\*\*", "“|”", '"', r"<br\s*/?>")},
    }
    raw_benchmark = json.loads(Path("data/benchmark/human_detectors.json").read_text())
    raw_rows = list(raw_benchmark.values())
    raw_pairs = Counter((str(row["prompt_id"]), str(int(row["ground_truth"] == "AI-generated")), row["article"].strip()) for row in raw_rows)
    processed_pairs = Counter((row["prompt_id"], row["ai"], row["text"]) for row in benchmark)
    report["benchmark"]["raw_to_processed_text_and_label_match"] = raw_pairs == processed_pairs
    report["benchmark"]["raw_metadata_fields"] = list(raw_rows[0])
    pairs = []
    raw_counts = {}
    for domain in ("finance", "medicine", "open_qa"):
        path = Path("data/hc3") / f"hc3_{domain}.jsonl"
        lines = path.read_text().splitlines()
        raw_counts[domain] = sum(bool(line.strip()) for line in lines)
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            human = [str(text).strip() for text in item.get("human_answers", []) if len(str(text).split()) >= 80]
            generated = [str(text).strip() for text in item.get("chatgpt_answers", []) if len(str(text).split()) >= 80]
            if human and generated:
                pairs.append({"domain": domain, "group_id": f"{domain}:{line_number}", "human": human[0], "generated": generated[0]})
    paired = pd.DataFrame(pairs).sample(frac=1, random_state=42).reset_index(drop=True)
    paired["split"] = "train"
    paired.loc[int(len(paired) * .8):, "split"] = "valid"
    expected = Counter((row.group_id, str(label), row.split, text) for row in paired.itertuples() for label, text in ((0, row.human), (1, row.generated)))
    saved_pairs = load_csv(Path("dataset/hc3_paired_v10.csv"))
    actual = Counter((row["group_id"], row["label"], row["split"], row["text"]) for row in saved_pairs)
    report["hc3"] = {"raw_questions": raw_counts, "qualified_pairs": len(paired), "raw_pair_reconstruction_matches": expected == actual}
    paired_frame = pd.read_csv("dataset/hc3_paired_v10.csv")
    expected_combined = Counter()
    for split, cap in (("train", 600), ("valid", 150)):
        for domain in sorted(paired_frame.domain.unique()):
            part = paired_frame[(paired_frame.split == split) & (paired_frame.domain == domain)]
            selected = part.group_id.drop_duplicates().sample(n=min(cap, part.group_id.nunique()), random_state=42)
            for row in part[part.group_id.isin(selected)].itertuples():
                expected_combined[("hc3_" + row.group_id, str(row.label), split, row.text)] += 1
        for row in benchmark:
            if row["split"] == split:
                expected_combined[(row["prompt_id"], row["ai"], split, row["text"])] += 1
    combined = load_csv(Path("dataset/combined_v11.csv"))
    actual_combined = Counter((row["prompt_id"], row["ai"], row["split"], row["text"]) for row in combined)
    report["combined_reconstruction_matches"] = expected_combined == actual_combined
    checks = []
    for folder in ("models", "legacy/pilot/artifacts", "legacy/cycle2/artifacts", "legacy/cycle2/results"):
        for path in sorted(Path(folder).glob("**/*.json")):
            check_metrics(json.loads(path.read_text()), str(path), checks)
    report["confusion_metric_arithmetic"] = checks
    report["arithmetic_failures"] = [check for check in checks if check["errors"]]
    settings = {}
    for path in sorted(Path("training").glob("*.py")):
        source = path.read_text()
        tree = ast.parse(source)
        settings[str(path)] = [ast.get_source_segment(source, node) for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in {"TfidfVectorizer", "LinearSVC", "LogisticRegression", "GroupKFold"}]
    report["training_constructor_calls"] = settings
    report["external_samples"] = {name: {"present": (Path("samples") / name).exists()} for name in ("article_human.txt", "article_ai.txt", "article_mixed.txt", "article.txt")}
    report["external_samples"]["article.txt"]["whitespace_words"] = len(Path("samples/article.txt").read_text().split())
    report["portable_copy_matches"] = digest(Path("models/current/hc3_v11_portable.joblib")) == digest(Path("release/V11_RELEASE/hc3_v11_portable.joblib"))
    report["limits"] = ["No fitted-model scoring or AUROC reproduction without sklearn/joblib.",
        "No near-duplicate discovery, provenance certification, permission adjudication, or new human quality annotation.",
        "A content hash establishes identity, not authorship or reuse permission."]
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "files_hashed": len(manifest),
        "metric_blocks_checked": len(checks), "arithmetic_failures": report["arithmetic_failures"],
        "hc3_reconstructed": report["hc3"]["raw_pair_reconstruction_matches"],
        "combined_reconstructed": report["combined_reconstruction_matches"]}, indent=2))


if __name__ == "__main__":
    main()