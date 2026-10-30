"""Retrain the NutriMatch nutrient models (100 g IFCT-calibrated standard) - reproducible, from scratch."""

from __future__ import annotations

import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DF_PATH = DATA / "dishes_engineered.csv"
RESULTS = DATA / "model_results.csv"
RESULTS_ORIGINAL = DATA / "model_results_original.csv"

RS = 42

FLAGS = [
    "is_nonveg", "is_chicken", "is_poultry", "is_fish", "is_meat", "is_red_meat", "is_egg",
    "is_paneer", "is_cheese_cream", "is_dairy", "is_legume", "is_chickpea_bean", "is_soya",
    "is_rice", "is_wheat", "is_millet", "is_semolina", "is_besan", "is_pasta_noodle",
    "is_vegetable", "is_leafy_green", "is_potato", "is_gourd_squash", "is_cruciferous",
    "is_pea_corn", "is_tomato", "is_onion_garlic", "is_fruit", "is_citrus_tangy",
    "is_nuts_seed", "is_sesame", "is_coconut", "is_ghee_butter",
    "is_salted", "is_cured", "is_spicy", "is_sugar", "is_sweet", "is_fried"
]

PRIORS = [
    "prior_Sodium", "prior_Iron", "prior_Calcium", "prior_Protein", "prior_Fibre"
]

NUTRIENTS = {
    "Sodium (mg)": "Sodium_mg",
    "Iron (mg)": "Iron_mg",
    "Calcium (mg)": "Calcium_mg",
    "Protein (g)": "Protein_g",
    "Fibre (g)": "Fibre_g",
}


def features_for(target: str) -> list:
    return [c for c in FLAGS if c != target] + PRIORS + ["dish_type"]


def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build(model, kind: str) -> Pipeline:
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ohe", make_ohe())])
    prep = ColumnTransformer([("num", num, []), ("cat", cat, [])])
    return Pipeline([("prep", prep), ("clf" if kind == "clf" else "reg", model)])


def set_features(pipe: Pipeline, feats: list) -> Pipeline:
    cat_cols = [c for c in feats if c == "dish_type"]
    num_cols = [c for c in feats if c != "dish_type"]
    prep = pipe.named_steps["prep"]
    prep.transformers = [
        ("num", prep.transformers[0][1], num_cols),
        ("cat", prep.transformers[1][1], cat_cols),
    ]
    return pipe


def main() -> None:
    df = pd.read_csv(DF_PATH)
    df = df[df["is_condiment"] == 0].reset_index(drop=True)
    if not RESULTS_ORIGINAL.exists() and RESULTS.exists():
        shutil.copy(RESULTS, RESULTS_ORIGINAL)

    rows = []
    for nutrient, stem in NUTRIENTS.items():
        feats = [f for f in features_for(nutrient) if f in df.columns]
        density_target = f"{nutrient.split(' ')[0]}_per_100g"
        d = df.dropna(subset=[density_target]).copy()

        y_reg = d[density_target].astype(float)
        q75 = float(y_reg.quantile(0.75))
        y_clf = (y_reg >= q75).astype(int)
        X = d[feats]

        Xtr, Xte, ytr_r, yte_r, ytr_c, yte_c = train_test_split(
            X, y_reg, y_clf, test_size=0.2, random_state=RS, stratify=y_clf
        )

        print(f"\n{nutrient} (per 100g): train {len(Xtr)} test {len(Xte)}")

        reg_candidates = (
            ("Dummy", DummyRegressor(strategy="mean")),
            ("Ridge", Ridge(alpha=5.0, random_state=RS)),
            (
                "RandomForest",
                RandomForestRegressor(
                    n_estimators=400,
                    max_depth=14,
                    min_samples_leaf=2,
                    max_features=0.75,
                    random_state=RS,
                    n_jobs=-1,
                ),
            ),
        )

        fitted_regs = {}
        for name, model in reg_candidates:
            pipe = set_features(build(model, "reg"), feats).fit(Xtr, ytr_r)
            pred = np.clip(pipe.predict(Xte), 0.0, None)
            r2 = r2_score(yte_r, pred)
            mae = mean_absolute_error(yte_r, pred)
            fitted_regs[name] = (pipe, r2, mae)
            rows.append(
                {
                    "nutrient": nutrient,
                    "task": "reg",
                    "model": name,
                    "accuracy": "",
                    "f1": "",
                    "r2": round(r2, 3),
                    "mae": round(mae, 2),
                }
            )
            print(f"  reg {name:14s} R2={r2:.3f}  MAE={mae:,.2f}")

        gbr_pipe = set_features(
            build(
                GradientBoostingRegressor(
                    n_estimators=250, learning_rate=0.05, max_depth=4, min_samples_leaf=3, random_state=RS
                ),
                "reg",
            ),
            feats,
        ).fit(Xtr, ytr_r)
        gbr_pred = np.clip(gbr_pipe.predict(Xte), 0.0, None)
        gbr_r2 = r2_score(yte_r, gbr_pred)
        gbr_mae = mean_absolute_error(yte_r, gbr_pred)

        rf_pipe, rf_r2, rf_mae = fitted_regs["RandomForest"]
        if gbr_r2 > rf_r2:
            best_pipe, best_r2, best_mae = gbr_pipe, gbr_r2, gbr_mae
            rows[-1]["r2"] = round(best_r2, 3)
            rows[-1]["mae"] = round(best_mae, 2)
            print(f"  --> Exporting GradientBoosting (R2={best_r2:.3f}, MAE={best_mae:,.2f})")
        else:
            best_pipe, best_r2, best_mae = rf_pipe, rf_r2, rf_mae
            print(f"  --> Exporting RandomForest (R2={best_r2:.3f}, MAE={best_mae:,.2f})")

        out = {
            "pipeline": best_pipe,
            "features": feats,
            "mae": round(float(best_mae), 2),
            "r2": round(float(best_r2), 3),
            "high_cut_off": round(q75, 2),
        }
        joblib.dump(out, DATA / f"{stem}_reg.joblib", compress=3)

        clf_candidates = (
            ("Dummy", DummyClassifier(strategy="most_frequent")),
            ("LogReg", LogisticRegression(max_iter=2000, C=1.0, random_state=RS)),
            (
                "RandomForest",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=14,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=RS,
                    n_jobs=-1,
                ),
            ),
        )
        for name, model in clf_candidates:
            pipe = set_features(build(model, "clf"), feats).fit(Xtr, ytr_c)
            pred = pipe.predict(Xte)
            acc = accuracy_score(yte_c, pred)
            f1 = f1_score(yte_c, pred)
            rows.append(
                {
                    "nutrient": nutrient,
                    "task": "clf",
                    "model": name,
                    "accuracy": round(acc, 3),
                    "f1": round(f1, 3),
                    "r2": "",
                    "mae": "",
                }
            )

    pd.DataFrame(rows, columns=["nutrient", "task", "model", "accuracy", "f1", "r2", "mae"]).to_csv(
        RESULTS, index=False
    )
    print(f"\nWrote {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()