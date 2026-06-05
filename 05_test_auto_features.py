
import pandas as pd
from importlib.machinery import SourceFileLoader

m = SourceFileLoader("feat", "04_auto_extract_features.py").load_module()

df = pd.read_csv("data/processed/manifest_4class_all.csv")

cols = [
    "Intensity", "Davg", "Dcirc", "Dmax", "Dmin", "Dperp",
    "Aspect", "Area", "Perim", "Roundness", "Elongation"
]

sample = df.sample(50, random_state=42)

errors = []
bad = 0

for idx, row in sample.iterrows():
    try:
        auto, _ = m.extract_features(row["image_path"])
        errors.append({
            c: abs(float(auto[c]) - float(row[c]))
            for c in cols
        })
    except Exception as e:
        bad += 1
        print("Failed:", row["image_path"], e)

err = pd.DataFrame(errors)

print("tested:", len(err), "failed:", bad)

print("\nMean absolute error:")
print(err.mean().sort_values())

print("\nMedian absolute error:")
print(err.median().sort_values())
