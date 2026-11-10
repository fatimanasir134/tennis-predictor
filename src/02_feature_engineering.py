"""Module 2 — Feature Engineering.

Transforms the cleaned train / val / holdout splits into numeric feature
matrices ready for scikit-learn.

Steps
-----
1. Derive interaction features: ``Rank_diff``, ``Pts_diff``.
2. One-hot encode categorical columns (``Surface``, ``Court``, ``Series``,
   ``Round``) — encoder is **fit on training data only**.
3. Concatenate numeric and encoded columns into final feature matrices.
4. Save the fitted encoder to ``models/encoder.pkl`` and the ordered feature
   column list to ``models/feature_columns.pkl``.

Outputs
-------
- ``models/encoder.pkl``          — fitted ``OneHotEncoder``
- ``models/feature_columns.pkl``  — ordered list of feature column names
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Shared helpers (no project-level imports — safe to import from anywhere)
from _feature_helpers import (  # noqa: E402
    CATEGORICAL_COLS,
    add_derived_features,
    build_feature_matrix,
)

# Re-export so callers can do: from 02_feature_engineering import add_derived_features
__all__ = ["add_derived_features", "build_feature_matrix", "engineer_features"]

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"

NUMERIC_PART = ["Rank_1", "Rank_2", "Pts_1", "Pts_2", "Best of", "Rank_diff", "Pts_diff"]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def engineer_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    holdout: pd.DataFrame,
) -> tuple:
    """Run the full feature-engineering pipeline on all three splits.

    Returns
    -------
    tuple
        ``(X_train, X_val, X_holdout, y_train, y_val, y_holdout, encoder, feature_columns)``
    """
    # 1. Derived features
    train   = add_derived_features(train)
    val     = add_derived_features(val)
    holdout = add_derived_features(holdout)

    # 2. Targets
    y_train   = train["Player_1_wins"].to_numpy()
    y_val     = val["Player_1_wins"].to_numpy()
    y_holdout = holdout["Player_1_wins"].to_numpy()

    # 3. Fit encoder on training data only
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(train[CATEGORICAL_COLS])

    # 4. Build ordered feature column list from training set
    ohe_cols = encoder.get_feature_names_out(CATEGORICAL_COLS).tolist()
    feature_columns = NUMERIC_PART + ohe_cols

    # 5. Build matrices
    X_train   = build_feature_matrix(train,   encoder, feature_columns)
    X_val     = build_feature_matrix(val,     encoder, feature_columns)
    X_holdout = build_feature_matrix(holdout, encoder, feature_columns)

    return X_train, X_val, X_holdout, y_train, y_val, y_holdout, encoder, feature_columns


def main() -> None:
    """Run feature engineering end-to-end and save artefacts to disk."""
    print("Loading processed splits …")
    train   = pd.read_csv(PROCESSED_DIR / "train.csv")
    val     = pd.read_csv(PROCESSED_DIR / "val.csv")
    holdout = pd.read_csv(PROCESSED_DIR / "holdout.csv")

    print(f"  Train   : {len(train):>6,} rows")
    print(f"  Val     : {len(val):>6,} rows")
    print(f"  Hold-out: {len(holdout):>6,} rows")

    X_train, X_val, X_holdout, y_train, y_val, y_holdout, encoder, feature_columns = \
        engineer_features(train, val, holdout)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder,         MODELS_DIR / "encoder.pkl")
    joblib.dump(feature_columns, MODELS_DIR / "feature_columns.pkl")

    print(f"\nFeature matrix shapes:")
    print(f"  X_train   : {X_train.shape}")
    print(f"  X_val     : {X_val.shape}")
    print(f"  X_holdout : {X_holdout.shape}")
    print(f"\nFeature columns ({len(feature_columns)} total):")
    print(f"  {feature_columns}")
    print("\nFiles written:")
    print("  models/encoder.pkl")
    print("  models/feature_columns.pkl")
    print("\nFeature engineering complete  OK")


if __name__ == "__main__":
    main()
