import pandas as pd
import numpy as np


INPUT = "dataset/human_detectors_grouped.csv"
OUTPUT = "dataset/hc3_short_v6.csv"

df = pd.read_csv(INPUT)


def make_chunks(text, min_words=180, max_words=260):
    words = str(text).split()

    chunks = []

    # Use non-overlapping sections.
    # Keep only reasonably sized segments.
    start = 0

    while start < len(words):
        remaining = len(words) - start

        if remaining < min_words:
            break

        end = min(start + max_words, len(words))
        chunk = words[start:end]

        if len(chunk) >= min_words:
            chunks.append(" ".join(chunk))

        start += max_words

    return chunks


rows = []

# Only build the new short corpus from the existing TRAIN split.
train = df[df["split"] == "train"].copy()

for _, row in train.iterrows():
    chunks = make_chunks(row["text"])

    for i, chunk in enumerate(chunks):
        rows.append({
            "original_id": row["id"],
            "prompt_id": row["prompt_id"],
            "source": row["source"],
            "generation_model": row["generation_model"],
            "ai": int(row["ai"]),
            "chunk_id": i,
            "text": chunk,
            "words": len(chunk.split()),
        })


out = pd.DataFrame(rows)

out.to_csv(
    OUTPUT,
    index=False,
)

print("Original training documents:", len(train))
print("Short training chunks:", len(out))

print("\nAI distribution:")
print(out["ai"].value_counts())

print("\nWord counts:")
print(
    out["words"].agg(
        ["count", "mean", "min", "max"]
    )
)

print("\nChunks by original document:")
print(
    out.groupby("original_id").size().describe()
)

print("\nChunks by generation model:")
print(
    out.groupby(
        ["generation_model", "ai"]
    ).size()
)

print("\nSaved:", OUTPUT)
