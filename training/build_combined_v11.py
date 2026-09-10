import pandas as pd


HC3_PATH = "dataset/hc3_paired_v10.csv"
BENCHMARK_PATH = "dataset/human_detectors_grouped.csv"
OUTPUT = "dataset/combined_v11.csv"

SEED = 42

# Maximum number of paired HC3 questions per domain.
# Each pair contributes exactly one human + one AI text.
TRAIN_CAP = 600
VALID_CAP = 150


hc3 = pd.read_csv(HC3_PATH)
benchmark = pd.read_csv(BENCHMARK_PATH)

print("HC3 rows:", len(hc3))
print("Benchmark rows:", len(benchmark))


def cap_hc3(data, cap, split_name):
    parts = []

    for domain in sorted(data["domain"].unique()):
        part = data[data["domain"] == domain].copy()

        # Sample complete question-pairs, not individual labels.
        group_ids = (
            part["group_id"]
            .drop_duplicates()
            .sample(
                n=min(
                    cap,
                    part["group_id"].nunique(),
                ),
                random_state=SEED,
            )
        )

        part = part[
            part["group_id"].isin(group_ids)
        ].copy()

        parts.append(part)

        print(
            split_name,
            "|",
            domain,
            "| pairs=",
            part["group_id"].nunique(),
            "| rows=",
            len(part),
        )

    return pd.concat(
        parts,
        ignore_index=True,
    )


hc3_train = cap_hc3(
    hc3[hc3["split"] == "train"],
    TRAIN_CAP,
    "HC3 train",
)

hc3_valid = cap_hc3(
    hc3[hc3["split"] == "valid"],
    VALID_CAP,
    "HC3 valid",
)


# Convert original benchmark data to the same schema.
bench_train = benchmark[
    benchmark["split"] == "train"
].copy()

bench_valid = benchmark[
    benchmark["split"] == "valid"
].copy()


bench_train = bench_train[
    [
        "id",
        "prompt_id",
        "source",
        "generation_model",
        "ai",
        "text",
    ]
].copy()

bench_valid = bench_valid[
    [
        "id",
        "prompt_id",
        "source",
        "generation_model",
        "ai",
        "text",
    ]
].copy()


bench_train["origin"] = "benchmark"
bench_valid["origin"] = "benchmark"

hc3_train["origin"] = "hc3"
hc3_valid["origin"] = "hc3"

hc3_train["id"] = (
    "hc3_" + hc3_train["group_id"].astype(str)
)

hc3_valid["id"] = (
    "hc3_" + hc3_valid["group_id"].astype(str)
)

hc3_train["prompt_id"] = (
    "hc3_" + hc3_train["group_id"].astype(str)
)

hc3_valid["prompt_id"] = (
    "hc3_" + hc3_valid["group_id"].astype(str)
)

hc3_train["source"] = (
    "HC3_" + hc3_train["domain"].astype(str)
)

hc3_valid["source"] = (
    "HC3_" + hc3_valid["domain"].astype(str)
)

hc3_train["generation_model"] = (
    hc3_train["label"]
    .map({
        0: "human",
        1: "chatgpt",
    })
)

hc3_valid["generation_model"] = (
    hc3_valid["label"]
    .map({
        0: "human",
        1: "chatgpt",
    })
)

hc3_train["ai"] = hc3_train["label"]
hc3_valid["ai"] = hc3_valid["label"]


columns = [
    "id",
    "prompt_id",
    "source",
    "generation_model",
    "ai",
    "text",
    "origin",
]


train = pd.concat(
    [
        bench_train[columns],
        hc3_train[columns],
    ],
    ignore_index=True,
)

valid = pd.concat(
    [
        bench_valid[columns],
        hc3_valid[columns],
    ],
    ignore_index=True,
)


train = train.sample(
    frac=1.0,
    random_state=SEED,
).reset_index(drop=True)

valid = valid.sample(
    frac=1.0,
    random_state=SEED,
).reset_index(drop=True)


train["split"] = "train"
valid["split"] = "valid"

out = pd.concat(
    [
        train,
        valid,
    ],
    ignore_index=True,
)


# Verify that no prompt/group ID is shared
# between train and validation.
train_groups = set(
    train["prompt_id"].astype(str)
)

valid_groups = set(
    valid["prompt_id"].astype(str)
)

leaking = train_groups.intersection(
    valid_groups
)

print("\nLEAKING GROUPS:", len(leaking))

print("\nFINAL ROWS:")
print(out["split"].value_counts())

print("\nLABEL BALANCE:")
print(
    out.groupby(
        ["split", "ai"]
    ).size()
)

print("\nSOURCE BALANCE:")
print(
    out.groupby(
        ["split", "origin"]
    ).size()
)

print("\nHC3 DOMAIN BALANCE:")
print(
    out[
        out["origin"] == "hc3"
    ]
    .groupby(
        ["split", "source", "ai"]
    )
    .size()
)

print("\nWORD COUNTS:")
out["words"] = (
    out["text"]
    .astype(str)
    .str.split()
    .str.len()
)

print(
    out.groupby(
        ["split", "ai"]
    )["words"]
    .agg(
        [
            "count",
            "mean",
            "median",
            "min",
            "max",
        ]
    )
)

out[
    columns + ["split"]
].to_csv(
    OUTPUT,
    index=False,
)

print("\nSAVED:", OUTPUT)
