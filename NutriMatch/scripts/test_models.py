"""Quick check: can every .joblib in data/ be loaded by this Python environment?

Run:  "<venv python>" scripts/test_models.py
"""

from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
files = sorted((ROOT / "data").glob("*.joblib"))

print("Models found:", len(files))

for file in files:
    print("\nTesting:", file.name)
    try:
        model = joblib.load(file)
        if isinstance(model, dict):
            feats = model.get("features") or []
            print("SUCCESS: dict  |  features:", len(feats),
                  "| MAE:", model.get("mae"), "| R2:", model.get("r2"))
        else:
            print("SUCCESS:", type(model).__name__)
    except Exception as error:
        print("FAILED:", type(error).__name__, error)
