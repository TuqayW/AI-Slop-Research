import json
import os
import glob
import pandas as pd


FILES = [
    "hc3_finance.jsonl",
    "hc3_medicine.jsonl",
    "hc3_open_qa.jsonl",
]

OUTPUT = "dataset/hc3_dev_corpus.csv"


rows = []


for path in FILES:
    domain = os.path.basename(path).replace(
        "hc3_", ""
    ).replace(
        ".jsonl", ""
    )

    print("Reading:", path)

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        for line_no, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            question = str(
                obj.get("question", "")
            ).strip()

            for answer in obj.get(
                "human_answers",
                [],
            ):
                text = str(answer).strip()

                if len(text.split()) >= 80:
                    rows.append({
                        "source_file": path,
                        "domain": domain,
                        "line": line_no,
                        "label": 0,
                        "text": text,
                        "words": len(text.split()),
                    })

            for answer in obj.get(
                "chatgpt_answers",
                [],
            ):
                text = str(answer).strip()

                if len(text.split()) >= 80:
                    rows.append({
                        "source_file": path,
                        "domain": domain,
                        "line": line_no,
                        "label": 1,
                        "text": text,
                        "words": len(text.split()),
                    })


df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT,
    index=False,
)


print("\nTOTAL ROWS:", len(df))

print("\nLABELS:")
print(df["label"].value_counts())

print("\nDOMAINS:")
print(df["domain"].value_counts())

print("\nLABEL BY DOMAIN:")
print(
    df.groupby(
        ["domain", "label"]
    ).size()
)

print("\nWORD COUNTS:")
print(
    df["words"].agg(
        [
            "count",
            "mean",
            "median",
            "min",
            "max",
        ]
    )
)

print("\nWORD COUNTS BY LABEL:")
print(
    df.groupby("label")["words"]
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

print("\n80-300 WORDS BY LABEL:")
print(
    df[
        (df["words"] >= 80)
        & (df["words"] <= 300)
    ]
    .groupby("label")
    .size()
)

print("\nSAVED:", OUTPUT)
