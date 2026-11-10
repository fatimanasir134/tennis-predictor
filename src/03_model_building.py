"""Module 3 — Model Building.

Trains a Random Forest classifier on the engineered feature matrices,
tunes key hyperparameters via ``RandomizedSearchCV`` (5-fold CV), evaluates
on the validation set, and saves the best model to disk.

Steps
-----
1. Load processed splits and re-run feature engineering to get matrices.
2. Define hyperparameter search space.
3. Run ``RandomizedSearchCV`` (5-fold, scoring='roc_auc', n_iter=10).
4. Evaluate best estimator on validation set: accuracy, AUC-ROC,
   classification report.
5. Save model to ``models/rf_model.pkl``.

Outputs
-------
- ``models/rf_model.pkl`` — best trained ``RandomForestClassifier``
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"


# ---------------------------------------------------------------------------
# Hyperparameter grid
# ---------------------------------------------------------------------------
PARAM_DIST = {
    "n_estimators":    [100, 200],
    "max_depth":       [None, 10, 20],
    "min_samples_leaf": [1, 5, 10],
}


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    random_state: int = 42,
    n_iter: int = 10,
    cv: int = 5,
) -> RandomForestClassifier:
    """Tune and train a ``RandomForestClassifier`` on *X_train* / *y_train*.

    Parameters
    ----------
    X_train : np.ndarray
        Numeric feature matrix for the training split.
    y_train : np.ndarray
        Binary target vector for the training split.
    random_state : int
        Seed for reproducibility.
    n_iter : int
        Number of parameter settings sampled in ``RandomizedSearchCV``.
    cv : int
        Number of cross-validation folds.

    Returns
    -------
    RandomForestClassifier
        Best estimator fitted on the full *X_train*.
    """
    base_rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)

    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=PARAM_DIST,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    print(f"\n  Best CV AUC-ROC : {search.best_score_:.4f}")
    print(f"  Best params     : {search.best_params_}")

    # Refit best estimator on full training set (RandomizedSearchCV already does
    # this by default via refit=True, but we make it explicit for clarity)
    best_model = search.best_estimator_
    return best_model


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_on_val(
    model: RandomForestClassifier,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> None:
    """Print accuracy, AUC-ROC, and a classification report for the val set."""
    y_pred  = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_proba)

    print("\n-- Validation Results ------------------------------------------")
    print(f"  Accuracy : {acc:.4f}  ({acc * 100:.1f}%)")
    print(f"  AUC-ROC  : {auc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_val, y_pred, target_names=["P2 wins", "P1 wins"]))


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run model building end-to-end and save the trained model."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from utils import load_processed_splits
    from _feature_helpers import add_derived_features, build_feature_matrix, CATEGORICAL_COLS

    print("Loading processed splits …")
    train, val, holdout = load_processed_splits()
    print(f"  Train   : {len(train):>6,} rows")
    print(f"  Val     : {len(val):>6,} rows")
    print(f"  Hold-out: {len(holdout):>6,} rows")

    print("\nLoading encoder and feature columns …")
    encoder         = joblib.load(MODELS_DIR / "encoder.pkl")
    feature_columns = joblib.load(MODELS_DIR / "feature_columns.pkl")

    print("Building feature matrices …")
    train_fe   = add_derived_features(train)
    val_fe     = add_derived_features(val)

    X_train = build_feature_matrix(train_fe, encoder, feature_columns)
    X_val   = build_feature_matrix(val_fe,   encoder, feature_columns)
    y_train = train["Player_1_wins"].to_numpy()
    y_val   = val["Player_1_wins"].to_numpy()

    print(f"  X_train: {X_train.shape}  |  X_val: {X_val.shape}")

    print("\nRunning RandomizedSearchCV (this may take a minute) …")
    model = train_model(X_train, y_train)

    evaluate_on_val(model, X_val, y_val)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "rf_model.pkl")
    print("\nModel saved -> models/rf_model.pkl")
    print("\nModel building complete  OK")


if __name__ == "__main__":
    main()
