"""Tests for Module 3 — Model Building (src/03_model_building.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path / import setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "rf_model.pkl"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    """Load the trained RandomForestClassifier from disk."""
    import joblib
    if not MODEL_PATH.exists():
        pytest.skip("models/rf_model.pkl not found — run src/03_model_building.py first.")
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def val_data():
    """Return (X_val, y_val) from the engineered validation split."""
    import importlib
    import pandas as pd
    import joblib

    processed = PROJECT_ROOT / "data" / "processed"
    train   = pd.read_csv(processed / "train.csv")
    val     = pd.read_csv(processed / "val.csv")
    holdout = pd.read_csv(processed / "holdout.csv")

    fe_mod = importlib.import_module("02_feature_engineering")
    result = fe_mod.engineer_features(train, val, holdout)
    # (X_train, X_val, X_holdout, y_train, y_val, y_holdout, encoder, feature_columns)
    X_val = result[1]
    y_val = result[4]
    return X_val, y_val


# ---------------------------------------------------------------------------
# Tests: model artefact
# ---------------------------------------------------------------------------

class TestModelArtefact:
    def test_model_file_exists(self):
        """rf_model.pkl must be present after running Module 3."""
        assert MODEL_PATH.exists(), (
            "models/rf_model.pkl not found. Run src/03_model_building.py first."
        )

    def test_model_is_random_forest(self, model):
        from sklearn.ensemble import RandomForestClassifier
        assert isinstance(model, RandomForestClassifier), (
            f"Expected RandomForestClassifier, got {type(model).__name__}"
        )

    def test_model_is_fitted(self, model):
        """Model must have been fitted (has estimators_)."""
        assert hasattr(model, "estimators_"), "Model does not appear to be fitted."
        assert len(model.estimators_) > 0, "Model has no estimators — was it trained?"


# ---------------------------------------------------------------------------
# Tests: validation performance
# ---------------------------------------------------------------------------

class TestModelPerformance:
    def test_val_auc_roc_above_threshold(self, model, val_data):
        """Validation AUC-ROC must be >= 0.60."""
        from sklearn.metrics import roc_auc_score
        X_val, y_val = val_data
        proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, proba)
        assert auc >= 0.60, (
            f"Validation AUC-ROC {auc:.4f} is below the minimum threshold of 0.60."
        )

    def test_val_accuracy_above_chance(self, model, val_data):
        """Validation accuracy must exceed random chance (> 0.50)."""
        from sklearn.metrics import accuracy_score
        X_val, y_val = val_data
        acc = accuracy_score(y_val, model.predict(X_val))
        assert acc > 0.50, (
            f"Validation accuracy {acc:.4f} is at or below random chance."
        )


# ---------------------------------------------------------------------------
# Tests: predict_proba contract
# ---------------------------------------------------------------------------

class TestPredictProba:
    def test_proba_rows_sum_to_one(self, model, val_data):
        """Each row of predict_proba must sum to 1.0."""
        X_val, _ = val_data
        row_sums = model.predict_proba(X_val).sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), (
            "predict_proba rows do not sum to 1.0."
        )

    def test_proba_values_in_range(self, model, val_data):
        """All predicted probabilities must be in [0, 1]."""
        X_val, _ = val_data
        proba = model.predict_proba(X_val)
        assert (proba >= 0).all() and (proba <= 1).all(), (
            "predict_proba returned values outside [0, 1]."
        )

    def test_proba_output_shape(self, model, val_data):
        """predict_proba output must have shape (n_samples, 2)."""
        X_val, _ = val_data
        proba = model.predict_proba(X_val)
        assert proba.shape == (len(X_val), 2), (
            f"Expected shape ({len(X_val)}, 2), got {proba.shape}."
        )
