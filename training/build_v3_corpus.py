from pathlib import Path
import csv
import re

HUMAN_SPLITS = {
    "collier.txt": "train",
    "davis.txt": "train",
    "haffkine.txt": "train",
    "shaler.txt": "train",
    "wynkoop.txt": "valid",
    "macdougal.txt": "valid",
    "jacoby.txt": "test",
    "darwin.txt": "test",
}

def tokens(text):
    return re.findall(r"\b\w+(?:['’]\w+)?\b", text)

def get_chunks(text):
    t = tokens(text)
    out = []

    for start in range(0, len(t) - 180 + 1, 220):
        chunk = t[start:start + 220]

        if len(chunk) >= 180:
            out.append(" ".join(chunk))

    return out

ai_files = sorted(Path("dataset/ai").glob("*.txt"))

if len(ai_files) != 30:
    raise ValueError(f"Expected 30 AI files, found {len(ai_files)}")

ai_rows = []

for path in ai_files:
    name = path.name

    if "_train_" in name:
        split = "train"
    elif "_valid_" in name:
        split = "valid"
    elif "_test_" in name:
        split = "test"
    else:
        raise ValueError(f"Unknown split: {name}")

    text = path.read_text(encoding="utf-8")

    if len(tokens(text)) < 80:
        raise ValueError(f"Too short: {name}")

    ai_rows.append({
        "id": "ai_" + path.stem,
        "text": text.strip(),
        "group": path.stem,
        "split": split,
        "ai": "1",
        "low_quality": "",
    })

counts = {
    "train": sum(r["split"] == "train" for r in ai_rows),
    "valid": sum(r["split"] == "valid" for r in ai_rows),
    "test": sum(r["split"] == "test" for r in ai_rows),
}

human_rows = []

for split in ("train", "valid", "test"):
    files = [
        name for name, value in HUMAN_SPLITS.items()
        if value == split
    ]

    target = counts[split]
    base = target // len(files)
    extra = target % len(files)

    for index, name in enumerate(files):
        needed = base + (1 if index < extra else 0)

        path = Path("dataset/human") / name
        chunks = get_chunks(
            path.read_text(
                encoding="utf-8",
                errors="replace"
            )
        )

        if len(chunks) < needed:
            raise ValueError(
                f"{name} has only {len(chunks)} usable chunks."
            )

        for j in range(needed):
            human_rows.append({
                "id": f"human_{path.stem}_{j}",
                "text": chunks[j],
                "group": name,
                "split": split,
                "ai": "0",
                "low_quality": "",
            })

rows = human_rows + ai_rows

out = Path("dataset/v3_corpus.csv")

with out.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:
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
print("Total:", len(rows))

for split in ("train", "valid", "test"):
    part = [r for r in rows if r["split"] == split]
    human = sum(r["ai"] == "0" for r in part)
    ai = sum(r["ai"] == "1" for r in part)

    print(
        split,
        "rows =", len(part),
        "human =", human,
        "ai =", ai
    )
