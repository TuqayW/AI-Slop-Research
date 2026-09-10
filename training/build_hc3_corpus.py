import csv
import json
import re
from pathlib import Path
from collections import Counter

INPUT = "human_detectors.json"
OUTPUT = "dataset/hc3_corpus.csv"


def words(text):
    return re.findall(r"\b\w+(?:['’]\w+)?\b", text)


with open(INPUT, encoding="utf-8") as f:
    data = json.load(f)

items = list(data.values()) if isinstance(data, dict) else data

rows = []

for x in items:
    text = x.get("article", "").strip()

    if len(words(text)) < 80:
        continue

    # Keep the human/AI pair together.
    group = (
        str(x.get("source", "")),
        str(x.get("title", "")),
        str(x.get("issue", "")),
        str(x.get("generation_model", "")),
    )

    rows.append({
        "id": str(x.get("id", "")),
        "text": text,
        "group": "|".join(group),
        "source": str(x.get("source", "")),
        "generation_model": str(x.get("generation_model", "")),
        "ai": "1" if x["ground_truth"] == "AI-generated" else "0",
        "low_quality": "",
    })


# Deterministic group assignment.
groups = sorted(set(r["group"] for r in rows))

train_groups = set(groups[:90])
valid_groups = set(groups[90:120])
test_groups = set(groups[120:])

for r in rows:
    if r["group"] in train_groups:
        r["split"] = "train"
    elif r["group"] in valid_groups:
        r["split"] = "valid"
    else:
        r["split"] = "test"


Path("dataset").mkdir(exist_ok=True)

with open(OUTPUT, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "id",
            "text",
            "group",
            "source",
            "generation_model",
            "split",
            "ai",
            "low_quality",
        ],
    )

    writer.writeheader()
    writer.writerows(rows)


print("Created:", OUTPUT)
print("Rows:", len(rows))
print("Groups:", len(groups))
print("Splits:", Counter(r["split"] for r in rows))
print("Labels:", Counter(r["ai"] for r in rows))
print("Models:", Counter(r["generation_model"] for r in rows))
