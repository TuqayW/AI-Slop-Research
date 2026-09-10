import pandas as pd
import numpy as np


INPUT = "dataset/human_detectors_grouped.csv"
OUTPUT = "dataset/dev_v9.csv"

df = pd.read_csv(INPUT)

# Use only the current training + validation material.
# The comparison test is excluded.
df = df[df["split"].isin(["train", "valid"])].copy()

df["words"] = (
    df["text"]
    .astype(str)
    .str.split()
    .str.len()
)

# Keep documents that naturally fall near the target
# length rather than creating overlapping chunks.
dev = df[
    (df["words"] >= 350)
    & (df["words"] <= 1000)
].copy()

print("Candidate documents:", len(dev))

print("\nLabels:")
print(dev["ai"].value_counts())

print("\nSources:")
print(dev["source"].value_counts())

print("\nGeneration models:")
print(dev["generation_model"].value_counts())

print("\nWord counts:")
print(
    dev["words"].agg(
        ["count", "mean", "median", "min", "max"]
    )
)

print("\nDocuments per prompt:")
print(
    dev.groupby("prompt_id")
       .size()
       .describe()
)

# Keep the complete documents and metadata.
dev[
    [
        "id",
        "prompt_id",
        "source",
        "generation_model",
        "ai",
        "words",
        "text",
    ]
].to_csv(
    OUTPUT,
    index=False,
)

print("\nSaved:", OUTPUT)
