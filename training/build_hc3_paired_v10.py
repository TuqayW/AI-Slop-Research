import json
import os
import random
import pandas as pd


FILES = [
    ("finance", "hc3_finance.jsonl"),
    ("medicine", "hc3_medicine.jsonl"),
    ("open_qa", "hc3_open_qa.jsonl"),
]

OUTPUT = "dataset/hc3_paired_v10.csv"

SEED = 42
random.seed(SEED)

records = []

for domain, path in FILES:
    print("Reading:", path)

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue

            obj = json.loads(line)

            humans = [
                str(x).strip()
                for x in obj.get("human_answers", [])
                if len(str(x).split()) >= 80
            ]

            ais = [
                str(x).strip()
                for x in obj.get("chatgpt_answers", [])
                if len(str(x).split()) >= 80
            ]

            if not humans or not ais:
                continue

            # One human + one AI answer per question.
            human = humans[0]
            ai = ais[0]

            group_id = f"{domain}:{line_no}"

            records.append({
                "domain": domain,
                "group_id": group_id,
                "human_text": human,
                "ai_text": ai,
                "human_words": len(human.split()),
                "ai_words": len(ai.split()),
            })


df = pd.DataFrame(records)

# Shuffle the paired questions, not the individual texts.
df = df.sample(
    frac=1.0,
    random_state=SEED,
).reset_index(drop=True)

# 80/20 split by paired question.
cut = int(len(df) * 0.80)

df["split"] = "train"
df.loc[cut:, "split"] = "valid"

rows = []

for _, row in df.iterrows():

    rows.append({
        "domain": row["domain"],
        "group_id": row["group_id"],
        "label": 0,
        "text": row["human_text"],
        "words": row["human_words"],
    })

    rows.append({
        "domain": row["domain"],
        "group_id": row["group_id"],
        "label": 1,
        "text": row["ai_text"],
        "words": row["ai_words"],
    })


out = pd.DataFrame(rows)

# Carry the split from the pair.
split_map = dict(
    zip(
        df["group_id"],
        df["split"],
    )
)

out["split"] = out["group_id"].map(split_map)

out.to_csv(
    OUTPUT,
    index=False,
)


print("\nPAIRED QUESTIONS:", len(df))
print("TOTAL TEXTS:", len(out))

print("\nPAIR SPLIT:")
print(df["split"].value_counts())

print("\nTEXT LABELS:")
print(out["label"].value_counts())

print("\nDOMAIN / LABEL:")
print(
    out.groupby(
        ["domain", "split", "label"]
    ).size()
)

print("\nWORD COUNTS:")
print(
    out.groupby(
        ["split", "label"]
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

print("\nPAIRED LENGTHS BY DOMAIN:")
print(
    df.groupby("domain")[
        ["human_words", "ai_words"]
    ].mean()
)

print("\nSAVED:", OUTPUT)
