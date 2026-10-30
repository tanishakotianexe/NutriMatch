# NutriMatch

A nutrient-first food recommendation website for Indian dishes. Pick a nutrient the
doctor asked you to increase (iron, calcium, protein, fibre, folate, vitamin C, ...)
or upload a blood report and let the app find it, then get a ranked list of dishes
that fit the patient's food preferences. A second module estimates the nutrients in
a dish that is **not** in the table, and a third keeps a daily food diary.

Built with Streamlit + pandas + plotly and 5 small Random Forest models. No deep
learning and no API key required — an optional free Gemini (or OpenAI) key only
adds AI-written recipes in the Recipe view.

---

## Pages

| Page | What it does |
| --- | --- |
| **Home** | Watercolour hero, a one-click nutrient picker, the richest dish per nutrient, dataset facts. |
| **Report Reader** | Reads a PDF or photo of a blood report (OCR), extracts lab values (haemoglobin, ferritin, serum iron, calcium, folate, vitamin C, sodium, total protein, albumin), flags low/high/unknown, and turns them into a nutrient priority list you can send straight to the Predictor. |
| **Graphs and Statistics** | Nutrient distributions, correlation heatmap, and the honest model scorecard (Dummy baseline vs LogReg/Ridge vs Random Forest). |
| **Predictor** | Ranks dishes for one or more nutrients. Filters: vegetarian only, avoid sweet, avoid fried, meals only, preferred dish types. Three views: cards, recipe steps, table. |
| **My Daily Nutrition** | Estimates nutrients for a custom dish from its dish type and ingredients (no food label needed), plus a food diary that persists in `data/diary.csv`. |

Navigation is in the top bar; a dark-mode toggle sits next to the brand. The whole
UI (theme, components, pages) lives in one file, `app/app.py`, so there is no
`theme.py` / `components.py` to keep in sync.

---

## Run it

Windows, from the project folder:

```
run_app.cmd
```

The launcher picks the first Python it finds that has streamlit + scikit-learn
(`.venv\Scripts\python`, then the system Python 3.11) and starts the app. Then
open <http://localhost:8501>.

Manual start (use the same interpreter you want to keep running the app):

```
python -m streamlit run app\app.py
```

### Environment

Tested on **Python 3.11** with:

| package | version |
| --- | --- |
| streamlit | 1.65.0 |
| scikit-learn | 1.6.1 |
| numpy | 2.4.6 |
| pandas | 3.0.6 |
| plotly | 7.1.0 |
| joblib | 1.6.0 |

The `.joblib` files in `data/` are pickles, so they belong to the environment
that exported them. If you change Python or upgrade scikit-learn/numpy and the
estimator starts showing "Could not estimate ...", re-export the models with the
interpreter that runs the app:

```
python scripts/train_models.py
```

The app also patches the common old-pickle case (trees saved before
scikit-learn 1.4 lack `monotonic_cst`) so older files keep working.

The **Report Reader** needs three extra OCR packages (the rest of the app works
without them):

```
pip install pymupdf rapidocr-onnxruntime pillow
```

Optional images in `assets/`: `logo.png` (top bar), `hero_blur_bg.jpg` (faint
texture behind the hero), `hero_photo.jpg` (photo that replaces the illustration).

### Recipe view: optional AI key

The **Recipe** tab looks for the dish in the local Indian recipe dataset first.
When there is no good match, it asks an LLM to write the recipe; without a key it
falls back to TheMealDB (free, no key).

| Setting | Where | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | environment or `.streamlit/secrets.toml` | Google Gemini — default, free tier is enough |
| `OPENAI_API_KEY` | environment or `.streamlit/secrets.toml` | OpenAI (used when no Gemini key is present) |
| `LLM_PROVIDER` | environment or `secrets.toml` | force `gemini` or `openai` when both keys exist |
| `GEMINI_MODEL` / `OPENAI_MODEL` | environment or `secrets.toml` | override the model (defaults: `gemini-2.5-flash`, `gpt-4o-mini`) |

Example `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-key-here"
```

AI-written recipes are labelled in the app: they are a cooking guide, not a
verified source.

---

## Data

| File | Rows | Role |
| --- | --- | --- |
| `data/Indian_Food_Nutrition_Processed.csv` | 1,014 | Source table of Indian dishes with 11 per-serving nutrients. |
| `data/dishes_engineered.csv` | 1,014 | Working copy: the 11 nutrients plus derived flags (`is_nonveg`, `is_rice`, `is_fried`, `is_sweet`, ...) and a `dish_type`. Used by every page and by training. |
| `data/Cleaned_Indian_Food_Dataset.csv` | 50,682 | Indian recipe instructions, matched to dishes for the "Recipe" view. |
| `data/diary.csv` | grows | The daily food diary (date, meal, dish, servings). Created on first use; nutrients are recomputed from the dish table, so the file stays tiny. |
| `data/model_results.csv` | 30 | Metrics from the current training run, displayed in the Graphs tab. |
| `data/model_results_original.csv` | 30 | Backup of the previous metrics file (pre-retrain). |

Nutrient values are per serving. Some cells are missing in the source table
(different dishes are missing different nutrients); the app drops a dish from a
ranking when its target nutrient is missing.

---

## The models

`scripts/train_models.py` trains everything from scratch and writes both the
`.joblib` files and `data/model_results.csv`. Run it when the dish table changes:

```
.venv\Scripts\python scripts\train_models.py
```

**Task.** For each of 5 nutrients (sodium, iron, calcium, protein, fibre):

* **regression** — predict the nutrient amount,
* **classification** — predict "high in this nutrient" = value in the top quarter of the table (that is why the Dummy baseline is always ~0.75 accuracy).

**Features** — exactly what the app's custom-dish estimator can collect from a
user: the 14 ingredient flags (rice, wheat, vegetables, fruit, legumes, egg,
dairy, paneer, chicken, meat, fish, fried, sweet, non-veg) and the dish type. The
target nutrient is never a feature. Condiment rows are excluded so training
matches inference. Macros (calories, carbs, protein, fats) are **not** inputs:
the estimator form no longer asks for a food label, so a dish is judged on what it
is and what is in it.

**Protocol** — 80/20 holdout split, `random_state=42`, stratified for the
classification task, metrics reported on the held-out 20%. Models: Dummy baseline,
Ridge / Logistic Regression, Random Forest (300 trees). No tuning, no leakage
tricks — the point is an honest comparison against the baseline.

### Results (holdout, `data/model_results.csv`)

| Nutrient | Regressor | R² | MAE | Classifier | F1 | Dummy accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| Protein (g) | Random Forest | 0.27 | 2.0 g | Logistic Regression | 0.48 | 0.75 |
| Calcium (mg) | Random Forest | 0.16 | 33.8 mg | Random Forest | 0.45 | 0.75 |
| Fibre (g) | Random Forest | 0.21 | 1.0 g | Logistic Regression | 0.46 | 0.75 |
| Sodium (mg) | Random Forest | 0.03 | 637.3 mg | Random Forest | 0.41 | 0.75 |
| Iron (mg) | Random Forest | 0.01 | 0.5 mg | Random Forest | 0.37 | 0.75 |

Read these with three caveats the app also states:

* **Ingredient + dish type alone explains little.** Without the macros, R² drops a
  lot (it used to be 0.5-0.7 with calories/carbs/protein/fats as inputs). Sodium
  and iron are nearly unpredictable this way — their R² is close to the Dummy
  baseline and the app says so next to the estimate. Ridge is slightly better than
  the Random Forest on every nutrient in this run (`data/model_results.csv` lists
  both); the shipped estimator is still the Random Forest.
* **Accuracy is not the metric to quote here.** About 75% of dishes are "not
  high", so a model that always answers "not high" already scores 75% accuracy
  while finding no high dishes at all (F1 = 0). That is why the Graphs tab plots
  F1 and the Predictor shows F1 next to the baseline.
* **Sodium's values are extreme.** Sodium ranges from 0.2 mg to ~14,000 mg per
  serving, and a handful of very salty dishes dominate the variance, which is why
  its MAE is so much larger than the others. The Graphs table lists every model.

The shipped models are the Random Forest regressors only, compressed
(`joblib.dump(..., compress=3)`, ~4-5 MB each). The classification metrics in the
CSV come from the same training run but no classifier files are shipped — the app
does not need them.

---

## Project layout

```
NutriMatch/
├─ app/app.py                 # the entire app (theme + components + 5 pages)
├─ app/_obsolete/             # old theme.py + components.py, merged into app.py (unused, safe to delete)
├─ assets/                    # logo.png, hero_blur_bg.jpg (optional extras)
├─ data/                      # dish table, recipe book, models, model_results.csv, diary.csv
├─ scripts/
│  ├─ train_models.py         # trains + exports the 5 regression models and the metrics CSV
│  ├─ test_models.py          # loads every .joblib and reports success/failure
│  └─ check_prediction.py     # runs one model on the dish table (sanity check)
├─ requirements.txt
└─ run_app.cmd                # Windows launcher
```

---

## Assumptions and limitations

* Nutrient values are per serving from a published Indian food composition table;
  they are **estimates of typical preparations**, not measured values for a
  specific restaurant dish.
* The custom-dish estimator's inputs (ingredient tick-boxes + dish type) are a
  coarse description of a dish, and macros are no longer asked for, so its
  estimates are weaker than a label-based model would be. Its MAE is printed with
  every estimate for that reason, and sodium/iron come out close to the baseline.
* The Report Reader reads the values printed on the report; it does not diagnose
  anything and only maps an analyte to a nutrient when the link is direct (e.g.
  haemoglobin/ferritin → iron). Vitamin D and B12 are shown but marked as not
  covered by the dish table.
* "Better" always means *more of the chosen nutrient per serving*. The app does
  not model allergies, medical conditions, medication interactions or
  preparation-level differences, and it is not medical advice.
* The training set is 1,014 dishes (919 after excluding condiments), so models are
  deliberately kept small and always compared against a Dummy baseline.
