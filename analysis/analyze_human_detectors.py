import json
from collections import Counter, defaultdict

with open("human_detectors.json", encoding="utf-8") as f:
    data = json.load(f)

items = list(data.values()) if isinstance(data, dict) else data

groups = defaultdict(list)

for x in items:
    key = (
        x["source"],
        x["title"],
        x["issue"],
    )
    groups[key].append(x)

print("records:", len(items))
print("underlying article groups:", len(groups))
print("rows per group:", Counter(len(v) for v in groups.values()))

bad = []

for key, rows in groups.items():
    labels = Counter(x["ground_truth"] for x in rows)
    models = Counter(x["generation_model"] for x in rows)

    if len(rows) != 10 or labels != Counter({
        "Human-written": 5,
        "AI-generated": 5,
    }):
        bad.append((key, labels, models, len(rows)))

print("groups with unexpected structure:", len(bad))

for key, labels, models, n in bad[:10]:
    print("\nBAD GROUP")
    print("key:", key)
    print("rows:", n)
    print("labels:", labels)
    print("models:", models)
