import csv
import json
from pathlib import Path
from collections import Counter

with open("human_detectors.json", encoding="utf-8") as f:
    data = json.load(f)

items = list(data.values()) if isinstance(data, dict) else data

groups = {}

for x in items:
    text = x.get("article", "").strip()

    if not text:
        continue

    key = (
        x.get("source", ""),
        x.get("title", ""),
        x.get("issue", ""),
    )

    groups.setdefault(key, []).append(x)

print("Underlying article groups:", len(groups))
print("Rows per group:", Counter(len(v) for v in groups.values()))

bad = []

for key, rows in groups.items():
    labels = Counter(x["ground_truth"] for x in rows)

    if len(rows) != 10 or labels["Human-written"] != 5 or labels["AI-generated"] != 5:
        bad.append((key, len(rows), labels))

print("Bad groups:", len(bad))

if bad:
    for x in bad[:10]:
        print(x)
    raise SystemExit("Dataset structure check failed.")

keys = sorted(groups.keys())

# 30 underlying articles:
# 18 train, 6 validation, 6 untouched test.
train_keys = set(keys[:18])
valid_keys = set(keys[18:24])
test_keys = set(keys[24:30])

rows = []

for key, members in groups.items():
    if key in train_keys:
        split = "train"
    elif key in valid_keys:
        split = "valid"
    else:
        split = "test"

    for x in members:
        rows.append({
            "id": str(x["id"]),
            "text": x["article"].strip(),
            "group": "|".join(key),
            "source": str(x.get("source", "")),
            "generation_model": str(x.get("generation_model", "")),
            "split": split,
            "ai": "1" if x["ground_truth"] == "AI-generated" else "0",
            "low_quality": "",
        })

out = Path("dataset/hc3_grouped.csv")

with out.open("w", encoding="utf-8", newline="") as f:
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

print("Created:", out)
print("Rows:", len(rows))

for split in ("train", "valid", "test"):
    part = [r for r in rows if r["split"] == split]

    print(
        split,
        "rows =", len(part),
        "human =", sum(r["ai"] == "0" for r in part),
        "ai =", sum(r["ai"] == "1" for r in part),
        "groups =", len({r["group"] for r in part}),
    )
