import csv
from collections import Counter

with open(
    "dataset/human_detectors_grouped.csv",
    encoding="utf-8",
    newline=""
) as f:
    rows = list(csv.DictReader(f))

groups = {}

for r in rows:
    groups.setdefault(r["prompt_id"], set()).add(r["split"])

print("Prompt groups:", len(groups))

leaking = {
    k: v for k, v in groups.items()
    if len(v) != 1
}

print("Leaking prompt groups:", len(leaking))

print(
    "Rows:",
    Counter(r["split"] for r in rows)
)

print(
    "Labels:",
    Counter(
        (r["split"], r["ai"])
        for r in rows
    )
)

for split in ("train", "valid", "test"):
    ids = sorted({
        r["prompt_id"]
        for r in rows
        if r["split"] == split
    })

    print(split, ids)
