"""Tests for Module 1 — Data Preparation (src/01_data_preparation.py)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make project src importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import load_raw_data

# Import the module under test via importlib (numeric prefix in filename)
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "data_preparation",
    PROJECT_ROOT / "src" / "01_data_preparation.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
prepare_data = _mod.prepare_data
LEAKAGE_COLUMNS = _mod.LEAKAGE_COLUMNS


@pytest.fixture(scope="module")
def splits():
    """Load raw data once and return (train, val, holdout)."""
    raw = load_raw_data()
    return prepare_data(raw)


@pytest.fixture(scope="module")
def train(splits):
    return splits[0]


@pytest.fixture(scope="module")
def val(splits):
    return splits[1]


@pytest.fixture(scope="module")
def holdout(splits):
    return splits[2]


# ---------------------------------------------------------------------------
# Target variable
# ---------------------------------------------------------------------------

def test_target_is_binary(train, val, holdout):
    """Player_1_wins must contain only 0 and 1."""
    for split in (train, val, holdout):
        unique = set(split["Player_1_wins"].unique())
        assert unique <= {0, 1}, f"Non-binary values found: {unique}"


def test_target_has_both_classes(train):
    """Training set must have both wins (1) and losses (0)."""
    assert set(train["Player_1_wins"].unique()) == {0, 1}


# ---------------------------------------------------------------------------
# Leakage columns absent
# ---------------------------------------------------------------------------

def test_leakage_columns_absent(train, val, holdout):
    """No leakage or player-name column should appear in any split."""
    for split in (train, val, holdout):
        for col in LEAKAGE_COLUMNS:
            assert col not in split.columns, f"Leakage column '{col}' found in split"


# ---------------------------------------------------------------------------
# No -1 sentinels remain in Pts columns
# ---------------------------------------------------------------------------

def test_no_minus_one_sentinels(train, val, holdout):
    """After cleaning, Pts_1 and Pts_2 must not contain -1."""
    for split in (train, val, holdout):
        for col in ("Pts_1", "Pts_2"):
            assert -1 not in split[col].values, f"-1 sentinel found in {col}"


# ---------------------------------------------------------------------------
# No NaN values in Pts columns after imputation
# ---------------------------------------------------------------------------

def test_no_nan_after_imputation(train, val, holdout):
    """Pts_1 and Pts_2 must be fully imputed — no NaN."""
    for split in (train, val, holdout):
        for col in ("Pts_1", "Pts_2"):
            assert split[col].isna().sum() == 0, f"NaN found in {col}"


# ---------------------------------------------------------------------------
# Split sizes are non-zero and non-overlapping
# ---------------------------------------------------------------------------

def test_split_sizes_nonzero(train, val, holdout):
    assert len(train) > 0, "Training set is empty"
    assert len(val) > 0, "Validation set is empty"
    assert len(holdout) > 0, "Hold-out set is empty"


def test_split_date_boundaries(train, val, holdout):
    """Verify strict time-based boundaries."""
    assert train["Date"].max() < pd.Timestamp("2019-01-01")
    assert val["Date"].min() >= pd.Timestamp("2019-01-01")
    assert val["Date"].max() < pd.Timestamp("2020-01-01")
    assert holdout["Date"].min() >= pd.Timestamp("2020-01-01")


def test_no_overlap_between_splits(train, val, holdout):
    """Train / val / holdout date ranges must not overlap."""
    train_dates = set(train["Date"])
    val_dates = set(val["Date"])
    holdout_dates = set(holdout["Date"])
    # Note: date sets can share the same calendar date if multiple matches
    # occurred, but all dates in train must be < all dates in val, etc.
    assert train["Date"].max() < val["Date"].min()
    assert val["Date"].max() < holdout["Date"].min()


# ---------------------------------------------------------------------------
# Expected feature columns are present
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = {"Date", "Series", "Court", "Surface", "Round", "Best of",
                    "Rank_1", "Rank_2", "Pts_1", "Pts_2", "Player_1_wins"}

def test_expected_columns_present(train):
    missing = EXPECTED_COLUMNS - set(train.columns)
    assert not missing, f"Expected columns missing from train: {missing}"
