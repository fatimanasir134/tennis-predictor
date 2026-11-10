# Tennis Match Outcome Predictor — Implementation Plan

## Overview

Build a machine learning web application that predicts the win probability of a match
between two ATP players. The app uses historical match data (`atp_tennis.csv`), a
Random Forest classifier trained on ranking-based and match-context features (no betting
odds, no player names), and a Streamlit dashboard where users select two players and
instantly see predicted win probabilities.

**Guiding constraints:**
- Betting odds (`Odd_1`, `Odd_2`) are excluded — they are post-market signals and
  constitute leakage / unrealistic performance inflation.
- Player names (`Player_1`, `Player_2`, `Winner`) are excluded from the feature set —
  they are used only to drive the UI player selector and to derive the target variable.
- Streamlit is the required web UI framework.
- Train/test split is **time-based** (not random) to prevent temporal leakage.

---

## Dataset Summary

**File:** `atp_tennis.csv`
**Columns (17 total):**

| Column | Type | Description | Notes |
|---|---|---|---|
| `Tournament` | string | Tournament name | High-cardinality (~200+ unique); use `Series` instead as a feature |
| `Date` | date | Match date (YYYY-MM-DD) | Used for time-based splitting only |
| `Series` | string | Tournament tier (International, Grand Slam, etc.) | Low-cardinality; encode as feature |
| `Court` | string | Indoor / Outdoor | Low-cardinality; encode as feature |
| `Surface` | string | Hard / Clay / Grass / Carpet | Low-cardinality; encode as feature |
| `Round` | string | Match round (1st Round, QF, SF, F, etc.) | Ordinal; encode as feature |
| `Best of` | int | Max sets (3 or 5) | Numeric feature |
| `Player_1` | string | Name of player 1 | **UI only — not a model feature** |
| `Player_2` | string | Name of player 2 | **UI only — not a model feature** |
| `Winner` | string | Name of winner | **Used only to derive target — not a feature** |
| `Rank_1` | int | ATP ranking of player 1 | Core feature |
| `Rank_2` | int | ATP ranking of player 2 | Core feature |
| `Pts_1` | int | ATP ranking points of player 1 | `-1` = missing sentinel; impute or drop |
| `Pts_2` | int | ATP ranking points of player 2 | `-1` = missing sentinel; impute or drop |
| `Odd_1` | float | Betting odds for player 1 | **EXCLUDED — leakage / constraint** |
| `Odd_2` | float | Betting odds for player 2 | **EXCLUDED — leakage / constraint** |
| `Score` | string | Final score string | **EXCLUDED — post-match leakage** |

**Target variable:** `Player_1_wins` — binary (1 if `Winner == Player_1`, else 0).

**Data quality issues:**
- `-1` is used as a missing-value sentinel in `Pts_1`, `Pts_2`, `Odd_1`, `Odd_2`.
- Early rows (year 2000) have `-1` for nearly all points and odds columns.
- `Score` may contain retirement/walkover strings — exclude entirely.
- `Tournament` is too high-cardinality to one-hot encode directly; use `Series`.

**Train / Validation / Hold-out split (time-based):**
- **Training set:** matches before 2019-01-01
- **Validation set:** matches in 2019 (used during model tuning)
- **Hold-out set:** matches from 2020 onwards (used for live prediction challenge and
  final evaluation — never seen during training or tuning)

---

## Recommended Tech Stack

| Component | Choice | Rationale |
|---|---|---|
| Language | Python 3.10+ | Widely supported; all target libraries require it |
| Data handling | `pandas`, `numpy` | Industry standard for tabular data |
| ML | `scikit-learn` | Random Forest, pipelines, metrics, permutation importance |
| Model interpretation | `scikit-learn` built-in + permutation, `shap` optional | Built-in RF importance is free; permutation importance corrects for correlated features |
| Visualisation | `matplotlib`, `seaborn` | Lightweight; built into most Python installs |
| Web UI | **Streamlit** | Required per constraint; single-file apps, minimal boilerplate, beginner-friendly |
| Dependency management | `pip` + `requirements.txt` | Simplest beginner-friendly approach |
| Testing | `pytest` | Lightweight smoke tests per module |

**Streamlit vs. Dash:** Streamlit requires far less code for interactive dashboards
(no callbacks, no layout classes) and is the correct choice for a workshop setting.
Dash is more powerful for production apps but has a steeper learning curve.

---

## Project Structure

```
tennis-predictor/
├── atp_tennis.csv                  # Raw dataset (do not modify)
├── requirements.txt                # All Python dependencies
├── IMPLEMENTATION_PLAN.md          # This document
│
├── data/
│   └── processed/                  # Saved cleaned DataFrames (train/val/holdout)
│
├── models/
│   ├── rf_model.pkl                # Saved trained Random Forest model
│   ├── encoder.pkl                 # Saved fitted OneHotEncoder
│   └── feature_columns.pkl         # Saved feature column list (for UI alignment)
│
├── src/
│   ├── 01_data_preparation.py      # Module 1 — cleaning, splitting, saving
│   ├── 02_feature_engineering.py   # Module 2 — feature creation and encoding
│   ├── 03_model_building.py        # Module 3 — train, tune, evaluate RF
│   ├── 04_model_interpretation.py  # Module 4 — feature importance charts
│   ├── 05_live_prediction.py       # Module 5 — held-out match prediction demo
│   └── utils.py                    # Shared helpers (load data, encode, predict)
│
├── app/
│   └── streamlit_app.py            # Module 6 — Streamlit web UI
│
└── tests/
    ├── test_data_preparation.py
    ├── test_feature_engineering.py
    ├── test_model_building.py
    └── test_live_prediction.py
```

---

## Phase 1 — Project Setup

**Intent:** Establish the development environment, confirm the dataset loads correctly,
and lay down the folder structure so every subsequent phase has a stable foundation.

**Expected Outcomes:**
- Python virtual environment created and all dependencies installable via `requirements.txt`.
- `atp_tennis.csv` loads without errors; shape and column names confirmed.
- All folders (`data/processed/`, `models/`, `src/`, `app/`, `tests/`) exist.
- `requirements.txt` lists every library needed by all 8 phases.

**Todo List:**
- [ ] Create the folder structure (`data/processed/`, `models/`, `src/`, `app/`, `tests/`).
- [ ] Write `requirements.txt` with pinned versions: `pandas`, `numpy`, `scikit-learn`,
      `matplotlib`, `seaborn`, `shap`, `streamlit`, `pytest`, `joblib`.
- [ ] Write a minimal `src/utils.py` with a `load_raw_data()` helper that reads
      `atp_tennis.csv` and returns a DataFrame.
- [ ] Smoke-test: confirm `load_raw_data()` returns the correct shape and column list.

**Status:** `[ ] pending`

---

## Phase 2 — Module 1: Data Preparation

**Intent:** Clean the raw dataset, create the binary target variable, remove all leakage
columns, handle missing-value sentinels, and produce time-based train/validation/hold-out
splits that are saved to disk for use by later phases.

**Expected Outcomes:**
- Binary target column `Player_1_wins` created correctly.
- Columns `Score`, `Odd_1`, `Odd_2`, `Player_1`, `Player_2`, `Winner`, `Tournament`
  dropped from the feature set.
- `-1` sentinels in `Pts_1`, `Pts_2` replaced with `NaN` then median-imputed.
- Three DataFrames saved: `data/processed/train.csv`, `val.csv`, `holdout.csv`.
- Split boundary: train < 2019, val = 2019, holdout >= 2020.

**Todo List:**
- [ ] Load raw data via `utils.load_raw_data()`.
- [ ] Parse `Date` column as `datetime`.
- [ ] Create target: `Player_1_wins = (Winner == Player_1).astype(int)`.
- [ ] Drop leakage / name columns: `Score`, `Odd_1`, `Odd_2`, `Player_1`, `Player_2`,
      `Winner`, `Tournament`.
- [ ] Replace `-1` with `NaN` in `Pts_1`, `Pts_2`.
- [ ] Impute missing `Pts_1`, `Pts_2` with column medians (computed on training set only).
- [ ] Apply time-based split using `Date`.
- [ ] Save train/val/holdout CSVs to `data/processed/`.
- [ ] Print split sizes as a sanity check.

**Relevant Context:** `atp_tennis.csv` columns `Date`, `Rank_1`, `Rank_2`, `Pts_1`,
`Pts_2`, `Series`, `Court`, `Surface`, `Round`, `Best of`.

**Status:** `[ ] pending`

---

## Phase 3 — Module 2: Feature Engineering

**Intent:** Transform the cleaned splits into a numeric feature matrix ready for
scikit-learn. Derive interaction features (rank/points differences) and encode
categorical columns. Ensure that all encoding fit parameters come from the training
set only and are then applied to validation and hold-out sets.

**Expected Outcomes:**
- New derived features: `Rank_diff` (`Rank_1 - Rank_2`), `Pts_diff` (`Pts_1 - Pts_2`).
- Categorical columns (`Surface`, `Court`, `Series`, `Round`) one-hot encoded using a
  `OneHotEncoder` fit on training data only.
- `Best of` kept as a numeric feature.
- Final feature matrices `X_train`, `X_val`, `X_holdout` and target vectors `y_train`,
  `y_val`, `y_holdout` saved (or returned by a reusable function in `utils.py`).
- Feature column list saved to `models/feature_columns.pkl` for UI alignment.

**Todo List:**
- [ ] Load `data/processed/train.csv`, `val.csv`, `holdout.csv`.
- [ ] Compute `Rank_diff = Rank_1 - Rank_2` and `Pts_diff = Pts_1 - Pts_2` for each split.
- [ ] Define categorical columns: `['Surface', 'Court', 'Series', 'Round']`.
- [ ] Fit `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` on training set only.
- [ ] Transform all three splits with the fitted encoder.
- [ ] Concatenate numeric and encoded columns into final feature matrices.
- [ ] Save encoder to `models/encoder.pkl` and feature column list to
      `models/feature_columns.pkl`.
- [ ] Print feature matrix shape as a sanity check.

**Relevant Context:** Output of Phase 2 (`data/processed/`).

**Status:** `[ ] pending`

---

## Phase 4 — Module 3: Model Building

**Intent:** Train a Random Forest classifier, tune key hyperparameters with
cross-validation on the training set, evaluate on the validation set, and save the
final model to disk. Report accuracy and AUC-ROC to set realistic expectations
(target ~65-72% accuracy given no betting odds).

**Expected Outcomes:**
- `RandomForestClassifier` trained on `X_train` / `y_train`.
- Hyperparameter search (`n_estimators`, `max_depth`, `min_samples_leaf`) via
  `RandomizedSearchCV` with 5-fold cross-validation.
- Validation accuracy, AUC-ROC, and classification report printed.
- Best model saved to `models/rf_model.pkl`.

**Todo List:**
- [ ] Load feature matrices from Phase 3 output (or call engineering functions).
- [ ] Define hyperparameter grid: `n_estimators` [100, 200], `max_depth` [None, 10, 20],
      `min_samples_leaf` [1, 5, 10].
- [ ] Run `RandomizedSearchCV` (5-fold, `scoring='roc_auc'`, `n_iter=10`).
- [ ] Fit best estimator on full `X_train`.
- [ ] Evaluate on `X_val`: print accuracy, AUC-ROC, classification report.
- [ ] Save model with `joblib.dump` to `models/rf_model.pkl`.

**Relevant Context:** Output of Phase 3; `scikit-learn` `RandomForestClassifier`.

**Status:** `[ ] pending`

---

## Phase 5 — Module 4: Model Interpretation

**Intent:** Explain what the model has learned using two complementary methods — the
Random Forest's built-in (impurity-based) feature importance and permutation importance.
Produce charts that can be embedded in the Streamlit UI or saved as images.

**Expected Outcomes:**
- Bar chart of top-15 built-in feature importances saved to `data/processed/`.
- Bar chart of top-15 permutation importances (computed on validation set) saved.
- Brief printed commentary comparing the two (highlights if they agree or diverge).
- Optional: SHAP summary plot if `shap` library is available.

**Todo List:**
- [ ] Load `models/rf_model.pkl` and validation feature matrix.
- [ ] Extract `.feature_importances_` from the trained RF; map to feature names.
- [ ] Plot and save top-15 built-in importances as `data/processed/importance_builtin.png`.
- [ ] Run `permutation_importance` (from `sklearn.inspection`) on validation set
      (n_repeats=10); plot and save top-15 as `data/processed/importance_permutation.png`.
- [ ] Print a comparison summary to stdout.
- [ ] (Optional) Generate SHAP summary plot if `shap` is installed.

**Relevant Context:** `models/rf_model.pkl`, `models/feature_columns.pkl`, Phase 3 output.

**Status:** `[ ] pending`

---

## Phase 6 — Module 5: Live Prediction Challenge

**Intent:** Simulate a "live" prediction scenario using the hold-out set (matches from
2020 onwards that the model has never seen). Select a small number of representative
matches, run the model, and compare predicted win probabilities to actual outcomes.
This demonstrates that the model generalises beyond its training window.

**Expected Outcomes:**
- 3-5 hold-out matches displayed with: player names, surface, round, predicted
  `P(Player_1 wins)`, actual winner, and whether the prediction was correct.
- Overall hold-out accuracy and AUC-ROC printed.
- Results saved as `data/processed/live_prediction_results.csv`.

**Note on live simulation:** The dataset is entirely historical. "Live" here means
the hold-out split acts as the unseen future — matches from 2020+ were withheld from
all training and tuning decisions and serve as a realistic proxy for real-time prediction.

**Todo List:**
- [ ] Load `data/processed/holdout.csv` and `models/rf_model.pkl`.
- [ ] Reconstruct hold-out feature matrix using the saved encoder and feature columns.
- [ ] Run `predict_proba` on the full hold-out set.
- [ ] Compute hold-out accuracy and AUC-ROC; print results.
- [ ] Sample 3-5 matches; build a results table with player names, surface, round,
      predicted probability, actual winner, correct/incorrect flag.
- [ ] Save results table to `data/processed/live_prediction_results.csv`.
- [ ] Print the results table to stdout.

**Relevant Context:** `data/processed/holdout.csv`, `models/rf_model.pkl`,
`models/encoder.pkl`, `models/feature_columns.pkl`.

**Status:** `[ ] pending`

---

## Phase 7 — Module 6: Streamlit Web UI Dashboard

**Intent:** Build an interactive single-page Streamlit app where a user selects two
ATP players from dropdown menus, and the app looks up their most recent rankings,
constructs a feature vector, runs the saved model, and displays predicted win
probabilities. Also surfaces the feature importance charts from Phase 5.

**Expected Outcomes:**
- Running `streamlit run app/streamlit_app.py` launches the dashboard in a browser.
- Two player dropdowns populated from unique player names in the dataset.
- "Predict" button triggers model inference and displays a win probability bar/gauge.
- Feature importance section shows the two charts from Phase 5.
- No player name is ever passed to the model — only derived numeric/encoded features.
- Context controls (Surface, Round, Series, Court, Best of) allow setting match context.

**Todo List:**
- [ ] Load `models/rf_model.pkl`, `models/encoder.pkl`, `models/feature_columns.pkl`,
      and the raw dataset (for player name list and last-known rankings).
- [ ] Build a player-to-latest-rank lookup dictionary from the raw data
      (most recent match date per player gives `Rank_1` or `Rank_2`).
- [ ] Add two `st.selectbox` widgets for Player 1 and Player 2.
- [ ] Add context selectors: `Surface`, `Court`, `Series`, `Round`, `Best of`.
- [ ] On button click: construct feature vector, run `predict_proba`, display results.
- [ ] Display results as a horizontal bar chart (`st.bar_chart` or `matplotlib`).
- [ ] Add a collapsible section showing the two feature importance charts.
- [ ] Add a "Live Prediction Challenge" section showing
      `data/processed/live_prediction_results.csv` as a table.

**Relevant Context:** `models/`, `data/processed/`, `src/utils.py`.

**Status:** `[ ] pending`

---

## Phase 8 — Testing and Validation

**Intent:** Verify that each module produces the correct outputs, that the data pipeline
has no leakage, that the model meets a minimum performance bar, and that the Streamlit
app runs without errors.

**Expected Outcomes:**
- All `pytest` tests pass with no errors.
- Model hold-out AUC-ROC >= 0.60 (realistic threshold without odds features).
- Streamlit app launches and renders the prediction UI without runtime errors.
- No player name, betting odds, or post-match score appears in the model feature set.

**Todo List:**
- [ ] `tests/test_data_preparation.py`: assert target values are 0/1 only; assert leakage
      columns absent from output DataFrames; assert no `-1` sentinels remain; assert
      split sizes are non-zero.
- [ ] `tests/test_feature_engineering.py`: assert `Rank_diff` and `Pts_diff` computed
      correctly; assert feature matrix has no NaN values; assert feature columns match
      `models/feature_columns.pkl`.
- [ ] `tests/test_model_building.py`: assert model file exists after training; assert
      validation AUC-ROC >= 0.60; assert `predict_proba` output sums to 1.0 per row.
- [ ] `tests/test_live_prediction.py`: assert results CSV exists; assert predicted
      probabilities are in [0, 1]; assert actual winner column is present.
- [ ] Manual smoke test: run `streamlit run app/streamlit_app.py` and verify the UI
      renders, dropdowns populate, and a prediction is returned without error.

**Status:** `[ ] pending`

---

## Assumptions

1. Each row in `atp_tennis.csv` represents one completed match (no duplicates per match).
2. `Rank_1` and `Rank_2` reflect the players' rankings at the time of the match (no future leakage from ranking columns themselves).
3. The most recent ranking for a player in the dataset is a reasonable proxy for their current ranking in the Streamlit UI.
4. `-1` in `Pts_1` / `Pts_2` means the points data was not available, not that the player had negative points.
5. `Tournament` is dropped in favour of `Series` to avoid high-cardinality encoding problems; this trades some granularity for generalisation.
6. The hold-out set (2020+) is large enough to produce meaningful evaluation metrics.

---

## Open Questions

1. **Split boundary year:** 2019 for validation, 2020+ for hold-out — confirm this gives adequate hold-out size given the dataset date range.
2. **Rank imputation:** Should missing `Rank_1` / `Rank_2` rows be dropped or imputed? Current plan: drop rows with missing ranks, as rank is the core feature.
3. **Round encoding:** Should `Round` be ordinal (1st Round=1 through Final=7) or one-hot? Ordinal preserves natural ordering; one-hot is safer with RF.
4. **SHAP in UI:** Include SHAP plots in the Streamlit UI or keep them as a script extra only?

---

## Risks and Trade-offs

| # | Risk | Recommended Approach |
|---|---|---|
| 1 | **Temporal data leakage** — a random train/test split lets the model see future matches during training, inflating accuracy and producing false confidence. | Use a strict time-based split: all training data precedes all test data. Never shuffle before splitting. |
| 2 | **Exclusion of betting odds** — odds encode much of the predictive signal. Without them, the model will achieve lower accuracy (~65-70%), which may appear underwhelming. | Set expectations explicitly in the UI and documentation. Frame the goal as a ranking-based predictor, not a market-beating oracle. |
| 3 | **High-cardinality Tournament column** — one-hot encoding 200+ tournament names creates a sparse, noisy feature space that degrades RF performance. | Drop `Tournament` entirely; use `Series` (tournament tier) as a proxy. |
