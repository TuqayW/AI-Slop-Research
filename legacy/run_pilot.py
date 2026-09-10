"""Reproduce a small authorship pilot without assigning content-quality labels."""

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

import detector


DATA = Path("pilot")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprints(output):
    paths = [DATA / name for name in ("protocol.json", "human_articles.json", "generated_articles.json")]
    paths.extend([output / "corpus.csv", Path("detector.py"), Path("run_pilot.py")])
    return {str(path): digest(path) for path in paths}


def prepare(output):
    protocol = read_json(DATA / "protocol.json")
    human = read_json(DATA / "human_articles.json")
    generated = {item["topic"]: item for item in read_json(DATA / "generated_articles.json")}
    topics = [topic for split in detector.SPLITS for topic in protocol[f"{split}_topics"]]
    if len(set(topics)) != len(topics) or {item["topic"] for item in human} != set(topics):
        raise ValueError("Protocol and source topics must match exactly without split overlap.")
    if set(generated) != set(topics) or len(human) != len(topics):
        raise ValueError("Every source article requires one generated counterpart.")
    style_rng = random.Random(protocol["seed"])
    styles = {topic: style_rng.choice(protocol["generation_styles"]) for topic in topics}
    rows, manifest = [], []
    for source in human:
        topic = source["topic"]
        split = next(split for split in detector.SPLITS if topic in protocol[f"{split}_topics"])
        if generated[topic]["style"] != styles[topic]:
            raise ValueError(f"Style differs from the fixed random assignment: {topic}")
        for origin, item in ((0, source), (1, generated[topic])):
            identifier = f"{'generated' if origin else 'historical'}-{topic}"
            rows.append({"id": identifier, "text": item["text"], "group": topic,
                         "split": split, "ai": origin, "low_quality": ""})
            manifest.append({
                "id": identifier, "split": split, "group": topic, "ai": origin,
                "word_count": len(detector.words(item["text"])),
                "text_sha256": hashlib.sha256(item["text"].encode()).hexdigest(),
                "source_title": source["title"],
                "source_url": protocol["text_url"] if not origin else None,
                "source_passage": source.get("source_passage") if not origin else None,
                "prompt": f"{protocol['generation_instruction']} Topic: {topic}. Style: {styles[topic]}." if origin else None,
                "quality_label": None,
            })
    output.mkdir(parents=True, exist_ok=True)
    with (output / "corpus.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "text", "group", "split", "ai", "low_quality"])
        writer.writeheader()
        writer.writerows(rows)
    detector.load_data(output / "corpus.csv", targets=("ai",))
    write_json(output / "manifest.json", manifest)
    print(f"Prepared {len(rows)} documents in {len(topics)} article-topic pairs.")


def fit_candidate(training, validation, feature_set, regularization, protocol, dataset_hash):
    train_matrix = np.vstack([detector.features(row["text"]) for row in training])
    valid_matrix = np.vstack([detector.features(row["text"]) for row in validation])
    mean, scale = train_matrix.mean(axis=0), train_matrix.std(axis=0)
    scale[scale < 1e-12] = 1
    active = np.array([feature_set == "all" or name not in protocol["removed_length_features"]
                       for name in detector.FEATURES])
    train_matrix, valid_matrix = (train_matrix - mean) / scale, (valid_matrix - mean) / scale
    coefficients = detector.fit_head(train_matrix[:, active],
                                   np.array([int(row["ai"]) for row in training]), regularization)
    padded = np.zeros(len(detector.FEATURES) + 1)
    padded[0] = coefficients[0]
    padded[np.flatnonzero(active) + 1] = coefficients[1:]
    probabilities = detector.scores(valid_matrix, padded)
    labels = np.array([int(row["ai"]) for row in validation])
    threshold = detector.choose_threshold(labels, probabilities, "ai", protocol["validation_max_fpr"])
    artifact = {
        "version": 1, "features": detector.FEATURES, "min_words": detector.MIN_WORDS,
        "mean": mean.tolist(), "scale": scale.tolist(),
        "heads": {"ai": {"coefficients": padded.tolist(), "threshold": threshold}},
        "feature_set": feature_set, "active_feature_count": int(active.sum()),
        "regularization": regularization, "validation_max_fpr": protocol["validation_max_fpr"],
        "dataset_sha256": dataset_hash,
        "warning": "Historical single-generator authorship pilot only. Uncalibrated; quality not trained; not proof of AI use.",
    }
    return artifact, detector.metrics(labels, probabilities, threshold)


def selection_key(candidate):
    result = candidate["validation"]
    return (result["recall"], result["auroc"], -result["fpr"],
            -candidate["active_feature_count"], candidate["regularization"])


def select(output):
    if (output / "test_results.json").exists():
        raise ValueError("Test results already exist here. Do not retune against them; use a fresh study and test set.")
    protocol = read_json(DATA / "protocol.json")
    rows = detector.load_data(output / "corpus.csv", targets=("ai",))
    training = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "valid"]
    ledger, models = [], {}
    for feature_set in protocol["candidate_feature_sets"]:
        for regularization in protocol["candidate_regularization"]:
            identifier = f"{feature_set}-lambda-{regularization:g}"
            artifact, result = fit_candidate(training, validation, feature_set, regularization,
                                             protocol, digest(output / "corpus.csv"))
            models[identifier] = artifact
            ledger.append({"id": identifier, "feature_set": feature_set,
                           "regularization": regularization,
                           "active_feature_count": artifact["active_feature_count"],
                           "threshold": artifact["heads"]["ai"]["threshold"],
                           "validation": result})
    selected = max(ledger, key=selection_key)
    baseline = protocol["baseline"]
    baseline_id = f"{baseline['feature_set']}-lambda-{baseline['regularization']:g}"
    write_json(output / "baseline_model.json", models[baseline_id])
    write_json(output / "selected_model.json", models[selected["id"]])
    write_json(output / "selection.json", {
        "input_fingerprints": fingerprints(output), "candidates": ledger,
        "baseline_id": baseline_id, "selected_id": selected["id"],
        "model_fingerprints": {name: digest(output / name) for name in ("baseline_model.json", "selected_model.json")},
        "test_scores_examined": False,
    })
    print(json.dumps({"selected": selected, "test_scores_examined": False}, indent=2))


def evaluate(output):
    if (output / "test_results.json").exists():
        raise ValueError("Final evaluation already exists; inspect the saved report rather than retuning.")
    selection = read_json(output / "selection.json")
    if selection["input_fingerprints"] != fingerprints(output):
        raise ValueError("Inputs or implementation changed after selection.")
    for name, expected in selection["model_fingerprints"].items():
        if digest(output / name) != expected:
            raise ValueError("A frozen model changed after selection.")
    test = [row for row in detector.load_data(output / "corpus.csv", targets=("ai",)) if row["split"] == "test"]
    labels = np.array([int(row["ai"]) for row in test])
    report = {"selection_sha256": digest(output / "selection.json"), "models": {},
              "warning": "Six test documents from three topic pairs, one historical issue and one assistant. No modern-web or slop accuracy claim."}
    predictions = []
    for name in ("baseline", "selected"):
        artifact = read_json(output / f"{name}_model.json")
        outcomes = [detector.predict(row["text"], artifact) for row in test]
        result = detector.metrics(labels, [item["heads"]["ai"]["score"] for item in outcomes],
                                  artifact["heads"]["ai"]["threshold"])
        result["fpr_one_sided_95_upper_if_independent"] = (
            float(beta.ppf(0.95, result["fp"] + 1, result["tn"])) if result["tn"] else 1.0
        )
        report["models"][name] = result
        predictions.extend({"model": name, "id": row["id"], "true_ai": int(row["ai"]), **outcome}
                           for row, outcome in zip(test, outcomes))
    report["models"]["always_negative"] = detector.metrics(labels, np.zeros(len(test)), 0.5)
    write_json(output / "test_results.json", report)
    write_json(output / "test_predictions.json", predictions)
    render_results(output, selection, report)
    print(json.dumps(report, indent=2))


def render_results(output, selection, report):
    candidates = selection["candidates"]
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.2), layout="constrained")
    for feature_set, color in (("all", "#24547a"), ("without_length", "#b65b37")):
        points = sorted((item for item in candidates if item["feature_set"] == feature_set), key=lambda item: item["regularization"])
        axes[0].plot([item["regularization"] for item in points],
                     [item["validation"]["recall"] for item in points], "o-", color=color,
                     label=feature_set.replace("_", " "))
    axes[0].set(xscale="log", ylim=(-0.04, 1.04), xlabel="L2 penalty", ylabel="Validation recall",
                title="Selection data only (3 AI / 3 historical)")
    axes[0].legend(frameon=False, fontsize=8)
    selected = report["models"]["selected"]
    confusion = np.array([[selected["tn"], selected["fp"]], [selected["fn"], selected["tp"]]])
    axes[1].imshow(confusion, cmap="Blues", vmin=0, vmax=3)
    axes[1].set(xticks=[0, 1], xticklabels=["Historical", "AI"], yticks=[0, 1],
                yticklabels=["Historical", "AI"], xlabel="Predicted origin", ylabel="Recorded origin",
                title="Selected model: six held-out documents")
    for row_index in range(2):
        for column_index in range(2):
            value = confusion[row_index, column_index]
            axes[1].text(column_index, row_index, str(value), ha="center", va="center",
                         color="white" if value > 1.5 else "black", fontsize=16)
    figure.savefig(output / "pilot_results.pdf")
    plt.close(figure)
    lines = [r"\begin{table}[htbp]", r"\centering\small",
             r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Features & $\lambda$ & Validation AUROC & Recall & FPR \\", r"\midrule"]
    for item in candidates:
        result = item["validation"]
        label = "All ten" if item["feature_set"] == "all" else "Without two length features"
        lines.append(f"{label} & {item['regularization']:g} & {result['auroc']:.3f} & {result['recall']:.3f} & {result['fpr']:.3f} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}",
                  r"\caption{Complete validation ledger. These six documents select settings; they do not estimate generalization.}", r"\end{table}",
                  r"\begin{table}[htbp]", r"\centering\small", r"\begin{tabular}{lrrrrrr}", r"\toprule",
                  r"Screen & TP & FP & TN & FN & Accuracy & AUROC \\", r"\midrule"])
    for name, result in report["models"].items():
        lines.append(f"{name.replace('_', ' ').capitalize()} & {result['tp']} & {result['fp']} & {result['tn']} & {result['fn']} & {result['accuracy']:.3f} & {result['auroc']:.3f} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}",
                  r"\caption{Final authorship pilot: three historical and three generated articles. The selected and baseline configurations were fixed before test scoring.}", r"\end{table}"])
    (output / "results_tables.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "select", "evaluate"))
    parser.add_argument("--output", type=Path, default=DATA / "artifacts")
    args = parser.parse_args()
    try:
        {"prepare": prepare, "select": select, "evaluate": evaluate}[args.stage](args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()