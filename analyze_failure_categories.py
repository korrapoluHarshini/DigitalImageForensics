
from pathlib import Path
import pandas as pd

INPUT = Path("results/final_test/final_test_per_image.csv")
OUTPUT_DIR = Path("results/final_test/category_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

# Extract filename components
parts = df["filename"].str.split("_", expand=True)

df["Manipulation_Type"] = parts[1]
df["Manipulation_Method"] = parts[2]

df["Zero_IoU"] = df["IoU"] == 0

summary = df.groupby(
    ["Manipulation_Type", "Manipulation_Method"]
).agg(
    Total_Images=("filename", "count"),
    Zero_IoU_Images=("Zero_IoU", "sum"),
    Mean_IoU=("IoU", "mean"),
    Mean_Precision=("Precision", "mean"),
    Mean_Recall=("Recall", "mean"),
    Mean_F1=("F1", "mean")
).reset_index()

summary["Failure_Percentage"] = (
    summary["Zero_IoU_Images"]
    / summary["Total_Images"] * 100
)

summary = summary.sort_values(
    "Failure_Percentage",
    ascending=False
)

summary.to_csv(
    OUTPUT_DIR / "corrected_category_summary.csv",
    index=False
)

print("\n========== CORRECTED CATEGORY ANALYSIS ==========")
print(summary.to_string(index=False))

print("\nTotal images:", len(df))
print("Zero-IoU images:", int(df["Zero_IoU"].sum()))
print("Failure percentage:", round(df["Zero_IoU"].mean() * 100, 2))

print("\nSaved:", OUTPUT_DIR / "corrected_category_summary.csv")
