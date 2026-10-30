"""Sanity check: run one shipped model's pipeline on the dish table.

Run:  "<venv python>" scripts/check_prediction.py
"""

from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

saved = joblib.load(DATA / "Sodium_mg_reg.joblib")
pipeline = saved["pipeline"]
features = saved["features"]
print("model features:", features)

for filename in ["dishes_engineered.csv", "Cleaned_Indian_Food_Dataset.csv"]:
    path = DATA / filename
    print("\nChecking:", path.name)
    try:
        df = pd.read_csv(path)
        missing = [col for col in features if col not in df.columns]
        print("Rows:", len(df))
        print("Missing model features:", missing)
        if not missing and not df.empty:
            row = df.iloc[[0]][features]
            print("SUCCESS! Test sodium prediction:", pipeline.predict(row)[0])
            break
    except Exception as error:
        print("Could not test this file:", error)
