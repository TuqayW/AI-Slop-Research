from pathlib import Path
import csv
import re

splits = {
    "collier.txt": "train",
    "davis.txt": "train",
    "haffkine.txt": "train",
    "shaler.txt": "train",
    "wynkoop.txt": "valid",
    "macdougal.txt": "valid",
    "jacoby.txt": "test",
    "darwin.txt": "test",
}

def tok(text):
    return re.findall(r"\b\w+(?:['’]\w+)?\b", text)

rows = []

for name, split in splits.items():
    text = (Path("dataset/human") / name).read_text(
        encoding="utf-8",
        errors="replace"
    )

    t = tok(text)
    start = 0

    while start + 180 <= len(t):
        chunk = t[start:start + 220]

        if len(chunk) >= 180:
            rows.append({
                "id": name[:-4] + "_" + str(start),
                "text": " ".join(chunk),
                "group": name,
                "split": split,
                "ai": "0",
                "low_quality": "",
            })

        start += 220

out = Path("dataset/local_corpus.csv")

with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "id",
            "text",
            "group",
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
    print(
        split,
        sum(r["split"] == split for r in rows)
    )