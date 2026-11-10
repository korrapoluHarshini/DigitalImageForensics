import pandas as pd
import numpy as np

INPUT = "results/final_test/failure_analysis/detailed_failure_analysis.csv"

df = pd.read_csv(INPUT)

# Divide images according to ground-truth manipulated area
bins = [0, 1, 5, 10, 100]
labels = ["0-1%", "1-5%", "5-10%", "Above 10%"]

df["Area_Category"] = pd.cut(
    df["GT_Area_Percentage"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

# Identify successful and failed images
df["Result"] = np.where(
    df["IoU"] == 0,
    "Failed",
    "Successful"
)

summary = df.groupby(
    "Area_Category",
    observed=False
).agg(
    Total_Images=("filename", "count"),
    Failed_Images=("Result", lambda x: (x == "Failed").sum()),
    Mean_IoU=("IoU", "mean"),
    Mean_F1=("F1", "mean"),
    Mean_Max_Probability=("Max_Probability", "mean")
)

summary["Failure_Percentage"] = (
    100 * summary["Failed_Images"] / summary["Total_Images"]
)

print("\nFAILURE ANALYSIS BY MANIPULATED AREA")
print(summary.round(3).to_string())

summary.to_csv(
    "results/final_test/failure_analysis/area_failure_summary.csv"
)

print("\nSaved area_failure_summary.csv")