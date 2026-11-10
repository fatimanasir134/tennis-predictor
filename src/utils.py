"""Shared utility helpers for the tennis match outcome predictor."""

from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "atp_tennis.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_raw_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw ATP tennis CSV and return it as a DataFrame.

    Parameters
    ----------
    path : str or Path, optional
        Path to the CSV file.  Defaults to ``atp_tennis.csv`` at the
        project root.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with all original columns intact.
    """
    df = pd.read_csv(path)
    return df


def load_processed_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all three processed splits from ``data/processed/``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train, val, holdout)``
    """
    train   = pd.read_csv(PROCESSED_DIR / "train.csv")
    val     = pd.read_csv(PROCESSED_DIR / "val.csv")
    holdout = pd.read_csv(PROCESSED_DIR / "holdout.csv")
    return train, val, holdout


# ---------------------------------------------------------------------------
# Model / artefact loading helpers
# ---------------------------------------------------------------------------

def load_encoder():
    """Load and return the fitted ``OneHotEncoder`` from ``models/encoder.pkl``."""
    return joblib.load(MODELS_DIR / "encoder.pkl")


def load_feature_columns() -> list[str]:
    """Load and return the ordered feature column list from ``models/feature_columns.pkl``."""
    return joblib.load(MODELS_DIR / "feature_columns.pkl")


def load_model():
    """Load and return the trained ``RandomForestClassifier`` from ``models/rf_model.pkl``."""
    return joblib.load(MODELS_DIR / "rf_model.pkl")


# ---------------------------------------------------------------------------
# Feature engineering helpers (thin wrappers — logic lives in 02_feature_engineering)
# ---------------------------------------------------------------------------

def get_feature_matrix(
    df: pd.DataFrame,
    encoder,
    feature_columns: list[str],
) -> np.ndarray:
    """Return a numeric feature matrix for *df* using a pre-fitted *encoder*.

    Thin wrapper around
    ``02_feature_engineering.build_feature_matrix`` so the Streamlit app
    and live-prediction script can re-use the same logic without importing
    the full module.

    Parameters
    ----------
    df : pd.DataFrame
        A split or single-row DataFrame **without** derived features (they
        are computed here).
    encoder : OneHotEncoder
        A fitted encoder (from ``load_encoder()``).
    feature_columns : list[str]
        Ordered column list from training (from ``load_feature_columns()``).

    Returns
    -------
    np.ndarray
        2-D float array of shape ``(n_rows, len(feature_columns))``.
    """
    from _feature_helpers import add_derived_features, build_feature_matrix  # noqa: PLC0415
    df = add_derived_features(df)
    return build_feature_matrix(df, encoder, feature_columns)
