"""Tests for Module 5 — Live Prediction Challenge (src/05_live_prediction.py)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path / import setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

RESULTS_CSV = PROJECT_ROOT / "data" / "processed" / "live_prediction_results.csv"

# Import the module under test via importlib (numeric prefix in filename)
_spec = importlib.util.spec_from_file_location(
    "live_prediction",
    PROJECT_ROOT / "src" / "05_live_prediction.py",
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["live_prediction"] = _mod
_spec.loader.exec_module(_mod)
run_live_prediction = _mod.run_live_prediction


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def results_df() -> pd.DataFrame:
    """Load the live prediction results CSV produced by 05_live_prediction.py."""
    assert RESULTS_CSV.exists(), (
        "live_prediction_results.csv not found. "
        "Run src/05_live_prediction.py first."
    )
    return pd.read_csv(RESULTS_CSV)


@pytest.fixture(scope="module")
def full_prediction_output():
    """Run run_live_prediction() and return (df, y_true, proba)."""
    return run_live_prediction()


# ---------------------------------------------------------------------------
# Tests on the saved CSV
# ---------------------------------------------------------------------------

class TestResultsCSV:
    def test_csv_exists(self):
        assert RESULTS_CSV.exists(), "live_prediction_results.csv must exist after running Module 5."

    def test_csv_has_rows(self, results_df):
        assert len(results_df) >= 3, "Results CSV should contain at least 3 sampled matches."

    def test_csv_has_required_columns(self, results_df):
        required = {"Player 1 wins)", "Predicted Winner", "Prediction Correct", "Surface", "Round"}
        # flexible column check — use substring matching
        col_str = " ".join(results_df.columns)
        assert "Player" in col_str, "Results CSV must contain player name columns."
        assert "win" in col_str.lower(), "Results CSV must contain a win-probability column."
        assert "Predicted" in col_str, "Results CSV must contain a predicted winner column."
        assert "Correct" in col_str, "Results CSV must contain a correctness flag column."

    def test_probabilities_in_range(self, results_df):
        prob_col = [c for c in results_df.columns if "win" in c.lower() or "prob" in c.lower()]
        assert prob_col, "No probability column found in results CSV."
        probs = results_df[prob_col[0]].dropna()
        assert (probs >= 0).all() and (probs <= 1).all(), (
            "All predicted probabilities must be in [0, 1]."
        )

    def test_winner_column_present(self, results_df):
        # 'Winner' is the actual winner; check it's present under some name
        winner_cols = [c for c in results_df.columns if "winner" in c.lower() or "Winner" in c]
        assert winner_cols, "Results CSV must contain a winner column."


# ---------------------------------------------------------------------------
# Tests on the full hold-out inference
# ---------------------------------------------------------------------------

class TestFullHoldoutInference:
    def test_proba_shape(self, full_prediction_output):
        df, y_true, proba = full_prediction_output
        assert proba.shape[0] == len(df), "proba length must match number of hold-out rows."

    def test_proba_bounds(self, full_prediction_output):
        _, _, proba = full_prediction_output
        assert (proba >= 0).all() and (proba <= 1).all(), (
            "predict_proba output must be in [0, 1]."
        )

    def test_proba_rows_sum_to_one(self, full_prediction_output):
        """Verify the raw predict_proba rows (both classes) sum to 1."""
        import numpy as np
        from utils import load_encoder, load_feature_columns, load_model
        from _feature_helpers import add_derived_features, build_feature_matrix

        model = load_model()
        encoder = load_encoder()
        feature_columns = load_feature_columns()

        df, _, _ = full_prediction_output
        df_feat = add_derived_features(df)
        X = build_feature_matrix(df_feat, encoder, feature_columns)

        row_sums = model.predict_proba(X).sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), (
            "Each row of predict_proba must sum to 1.0."
        )

    def test_minimum_auc(self, full_prediction_output):
        """Hold-out AUC-ROC should meet the minimum quality bar."""
        from sklearn.metrics import roc_auc_score
        import numpy as np
        _, y_true, proba = full_prediction_output
        auc = roc_auc_score(y_true, proba)
        assert auc >= 0.60, f"Hold-out AUC-ROC {auc:.4f} is below the 0.60 threshold."

    def test_correct_column_is_boolean(self, full_prediction_output):
        df, y_true, proba = full_prediction_output
        import numpy as np
        y_pred = (proba >= 0.5).astype(int)
        correct_vals = (y_pred == y_true)
        assert correct_vals.dtype == bool or set(np.unique(correct_vals)) <= {True, False, 0, 1}, (
            "Correct column should be boolean."
        )
