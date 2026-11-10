"""Internal feature-engineering helpers shared by multiple modules.

This module is intentionally kept free of project-level imports so it can
be safely imported from both ``src/02_feature_engineering.py`` and
``src/utils.py`` without triggering circular imports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL_COLS = ["Surface", "Court", "Series", "Round"]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with ``Rank_diff`` and ``Pts_diff`` appended."""
    df = df.copy()
    df["Rank_diff"] = df["Rank_1"] - df["Rank_2"]
    df["Pts_diff"]  = df["Pts_1"]  - df["Pts_2"]
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    encoder: OneHotEncoder,
    feature_columns: list[str],
) -> np.ndarray:
    """Encode *df* and return a float matrix aligned to *feature_columns*.

    Parameters
    ----------
    df : pd.DataFrame
        Split DataFrame with derived features already added (i.e. after
        calling ``add_derived_features``).
    encoder : OneHotEncoder
        A fitted encoder.
    feature_columns : list[str]
        Ordered column names from the training set.

    Returns
    -------
    np.ndarray
        Shape ``(n_rows, len(feature_columns))``.
    """
    numeric_part = ["Rank_1", "Rank_2", "Pts_1", "Pts_2", "Best of", "Rank_diff", "Pts_diff"]
    ohe_part = encoder.transform(df[CATEGORICAL_COLS])
    ohe_cols = encoder.get_feature_names_out(CATEGORICAL_COLS).tolist()

    numeric_df = df[numeric_part].reset_index(drop=True)
    ohe_df = pd.DataFrame(ohe_part, columns=ohe_cols)

    combined = pd.concat([numeric_df, ohe_df], axis=1)
    combined = combined.reindex(columns=feature_columns, fill_value=0)
    return combined.to_numpy(dtype=float)
