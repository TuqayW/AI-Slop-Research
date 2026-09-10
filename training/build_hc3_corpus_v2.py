import csv
import json
from pathlib import Path
from collections import Counter

INPUT = "human_detectors.json"
OUTPUT = "dataset/human_detectors_grouped.csv"

with open(INPUT, encoding="utf-8") as f:
    data = json.load(f)

items = list(data.values()) if isinstance(data, dict) else data

groups = {}

for x in items:
    pid = str(x["prompt_id"])
    groups.setdefault(pid, []).append(x)

print("Prompt groups:", len(groups))
print("Rows per prompt:", Counter(len(v) for v in groups.values()))

bad = []

for pid, rows in groups.items():
    labels = Counter(x["ground_truth"] for x in rows)

    if len(rows) != 10:
        bad.append((pid, "wrong row count", len(rows)))

    if labels["Human-written"] != 5:
        bad.append((pid, "wrong human count", labels))

    if labels["AI-generated"] != 5:
        bad.append((pid, "wrong AI count", labels))

if bad:
    print("Bad groups:", len(bad))

    for item in bad[:20]:
        print(item)

    raise SystemExit("Dataset structure check failed.")

prompt_ids = sorted(groups)

train_ids = set(prompt_ids[:18])
valid_ids = set(prompt_ids[18:24])
test_ids = set(prompt_ids[24:30])

rows = []

for pid, members in groups.items():
    if pid in train_ids:
        split = "train"
    elif pid in valid_ids:
        split = "valid"
    else:
        split = "test"

    for x in members:
        rows.append({
            "id": str(x["id"]),
            "text": x["article"].strip(),
            "group": "prompt_" + pid,
            "prompt_id": pid,
            "source": str(x.get("source", "")),
            "title": str(x.get("title", "")),
            "issue": str(x.get("issue", "")),
            "generation_model": str(x.get("generation_model", "")),
            "split": split,
            "ai": "1" if x["ground_truth"] == "AI-generated" else "0",
            "low_quality": "",
        })

out = Path(OUTPUT)

with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "id",
            "text",
            "group",
            "prompt_id",
            "source",
            "title",
            "issue",
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

for split in ("train", "valid", "test"):
    part = [r for r in rows if r["split"] == split]

    print(
        split,
        "rows =", len(part),
        "human =", sum(r["ai"] == "0" for r in part),
        "AI =", sum(r["ai"] == "1" for r in part),
        "prompt_groups =", len(
            {r["prompt_id"] for r in part}
        ),
    )
