"""Tests for Module 2 — Feature Engineering."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure src/ is on the path so _feature_helpers and 02_feature_engineering can be imported
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import importlib

fe_mod = importlib.import_module("02_feature_engineering")
engineer_features = fe_mod.engineer_features

from _feature_helpers import add_derived_features, build_feature_matrix, CATEGORICAL_COLS  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def processed_splits():
    import pandas as pd
    processed = Path(__file__).resolve().parent.parent / "data" / "processed"
    train   = pd.read_csv(processed / "train.csv")
    val     = pd.read_csv(processed / "val.csv")
    holdout = pd.read_csv(processed / "holdout.csv")
    return train, val, holdout


@pytest.fixture(scope="module")
def engineered(processed_splits):
    train, val, holdout = processed_splits
    return engineer_features(train, val, holdout)


# ---------------------------------------------------------------------------
# Tests: derived features
# ---------------------------------------------------------------------------

def test_rank_diff_computed(processed_splits):
    train, _, _ = processed_splits
    df = add_derived_features(train)
    expected = (train["Rank_1"] - train["Rank_2"]).values
    np.testing.assert_array_equal(df["Rank_diff"].values, expected)


def test_pts_diff_computed(processed_splits):
    train, _, _ = processed_splits
    df = add_derived_features(train)
    expected = (train["Pts_1"] - train["Pts_2"]).values
    np.testing.assert_array_almost_equal(df["Pts_diff"].values, expected)


# ---------------------------------------------------------------------------
# Tests: feature matrices
# ---------------------------------------------------------------------------

def test_feature_matrix_no_nan(engineered):
    X_train, X_val, X_holdout, *_ = engineered
    assert not np.isnan(X_train).any(),   "NaN found in X_train"
    assert not np.isnan(X_val).any(),     "NaN found in X_val"
    assert not np.isnan(X_holdout).any(), "NaN found in X_holdout"


def test_feature_matrix_shapes_consistent(engineered):
    X_train, X_val, X_holdout, y_train, y_val, y_holdout, _, feature_columns = engineered
    n_cols = len(feature_columns)
    assert X_train.shape[1]   == n_cols, f"X_train cols {X_train.shape[1]} != {n_cols}"
    assert X_val.shape[1]     == n_cols, f"X_val cols {X_val.shape[1]} != {n_cols}"
    assert X_holdout.shape[1] == n_cols, f"X_holdout cols {X_holdout.shape[1]} != {n_cols}"


def test_row_counts_match_targets(engineered):
    X_train, X_val, X_holdout, y_train, y_val, y_holdout, *_ = engineered
    assert X_train.shape[0]   == len(y_train)
    assert X_val.shape[0]     == len(y_val)
    assert X_holdout.shape[0] == len(y_holdout)


def test_feature_columns_saved(tmp_path, engineered):
    """feature_columns saved by main() should match what engineer_features returns."""
    import joblib
    models_dir = Path(__file__).resolve().parent.parent / "models"
    if not (models_dir / "feature_columns.pkl").exists():
        pytest.skip("models/feature_columns.pkl not yet generated — run 02_feature_engineering.py first")
    saved = joblib.load(models_dir / "feature_columns.pkl")
    _, _, _, _, _, _, _, feature_columns = engineered
    assert saved == feature_columns


def test_encoder_saved():
    import joblib
    models_dir = Path(__file__).resolve().parent.parent / "models"
    if not (models_dir / "encoder.pkl").exists():
        pytest.skip("models/encoder.pkl not yet generated — run 02_feature_engineering.py first")
    encoder = joblib.load(models_dir / "encoder.pkl")
    # Verify it was fit on all categorical columns
    assert list(encoder.feature_names_in_) == CATEGORICAL_COLS


def test_targets_are_binary(engineered):
    _, _, _, y_train, y_val, y_holdout, *_ = engineered
    for name, y in [("y_train", y_train), ("y_val", y_val), ("y_holdout", y_holdout)]:
        unique = set(y)
        assert unique <= {0, 1}, f"{name} contains values outside {{0, 1}}: {unique}"
