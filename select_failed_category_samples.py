
from pathlib import Path
import pandas as pd

INPUT = Path("results/final_test/final_test_per_image.csv")
OUTPUT = Path("results/final_test/category_analysis/selected_failed_samples.csv")

df = pd.read_csv(INPUT)

parts = df["filename"].str.split("_", expand=True)

df["Manipulation_Type"] = parts[1].str.upper()
df["Manipulation_Method"] = parts[2].str.upper()

# Select zero-IoU failures from the three largest problem groups
categories = [
    ("S", "NNN"),
    ("S", "NND"),
    ("S", "NRD")
]

selected = []

for manipulation_type, method in categories:

    group = df[
        (df["Manipulation_Type"] == manipulation_type) &
        (df["Manipulation_Method"] == method) &
        (df["IoU"] == 0)
    ]

    sample = group.head(5)
    selected.append(sample)

    print(f"\n{manipulation_type} + {method}")
    print("Available failures:", len(group))
    print("Selected:", len(sample))

    for filename in sample["filename"]:
        print(filename)

result = pd.concat(selected, ignore_index=True)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT, index=False)

print("\nSelected images:", len(result))
print("Saved:", OUTPUT)
