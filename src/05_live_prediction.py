"""Module 5 — Live Prediction Challenge.

Simulates a "live" prediction scenario using the hold-out set (matches from
2020 onwards).  The model has never seen these matches during training or
hyperparameter tuning, so they act as a realistic proxy for real-time
prediction.

Outputs
-------
- Hold-out accuracy and AUC-ROC printed to stdout.
- 5 sampled matches displayed in a results table (player names, surface,
  round, predicted probability, actual winner, correct/incorrect flag).
- Results table saved to ``data/processed/live_prediction_results.csv``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

# ---------------------------------------------------------------------------
# Path helpers — allow running as a script from any working directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import (  # noqa: E402
    PROCESSED_DIR,
    load_encoder,
    load_feature_columns,
    load_model,
    load_raw_data,
)


# ---------------------------------------------------------------------------
# Core prediction logic
# ---------------------------------------------------------------------------

def build_holdout_display_df() -> pd.DataFrame:
    """Return the holdout split enriched with Player_1 and Player_2 names.

    The processed holdout CSV has the player name columns removed, so we
    re-join with the raw dataset on the natural key (Date, Rank_1, Rank_2).
    Duplicate-free matches are kept; the remainder retain NaN names.
    """
    raw = load_raw_data()
    raw["Date"] = pd.to_datetime(raw["Date"])
    # Keep only 2020+ rows from raw to limit the join size
    raw_holdout = raw[raw["Date"] >= "2020-01-01"][
        ["Date", "Rank_1", "Rank_2", "Player_1", "Player_2", "Winner"]
    ].copy()

    holdout = pd.read_csv(PROCESSED_DIR / "holdout.csv")
    holdout["Date"] = pd.to_datetime(holdout["Date"])

    merged = holdout.merge(
        raw_holdout,
        on=["Date", "Rank_1", "Rank_2"],
        how="left",
    )
    return merged


def run_live_prediction() -> pd.DataFrame:
    """Run model inference on the full hold-out set.

    Returns
    -------
    pd.DataFrame
        Full hold-out DataFrame with extra columns:
        ``P1_win_prob``, ``Predicted_winner``, ``Correct``.
    """
    model = load_model()
    encoder = load_encoder()
    feature_columns = load_feature_columns()

    df = build_holdout_display_df()

    # Reconstruct feature matrix (same logic as _feature_helpers)
    from _feature_helpers import add_derived_features, build_feature_matrix  # noqa: PLC0415

    df_feat = add_derived_features(df)
    X = build_feature_matrix(df_feat, encoder, feature_columns)
    y_true = df["Player_1_wins"].to_numpy()

    proba = model.predict_proba(X)[:, 1]  # P(Player_1 wins)
    y_pred = (proba >= 0.5).astype(int)

    df = df.copy()
    df["P1_win_prob"] = np.round(proba, 4)
    df["Predicted_winner"] = np.where(y_pred == 1, df["Player_1"], df["Player_2"])
    df["Correct"] = (y_pred == y_true)

    return df, y_true, proba


def sample_results(df: pd.DataFrame, n: int = 5, seed: int = 42) -> pd.DataFrame:
    """Return *n* representative sampled rows as a display table."""
    cols = ["Date", "Surface", "Round", "Player_1", "Player_2",
            "P1_win_prob", "Predicted_winner", "Winner", "Correct"]
    sample = df.sample(n=n, random_state=seed)[cols].reset_index(drop=True)
    # Rename for clarity
    sample = sample.rename(columns={
        "P1_win_prob": "P(Player 1 wins)",
        "Predicted_winner": "Predicted Winner",
        "Correct": "Prediction Correct",
    })
    return sample


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Live Prediction Challenge — Hold-out Set (2020+)")
    print("=" * 60)

    df, y_true, proba = run_live_prediction()

    acc = accuracy_score(y_true, (proba >= 0.5).astype(int))
    auc = roc_auc_score(y_true, proba)

    print(f"\nHold-out set size : {len(df):,} matches")
    print(f"Accuracy          : {acc:.4f}  ({acc * 100:.1f}%)")
    print(f"AUC-ROC           : {auc:.4f}")

    # -----------------------------------------------------------------------
    # Sample 5 matches for the display table
    # -----------------------------------------------------------------------
    results_table = sample_results(df, n=5, seed=42)

    print("\n--- Sample Predictions ---")
    print(results_table.to_string(index=False))

    # -----------------------------------------------------------------------
    # Save results to disk
    # -----------------------------------------------------------------------
    out_path = PROCESSED_DIR / "live_prediction_results.csv"
    results_table.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
