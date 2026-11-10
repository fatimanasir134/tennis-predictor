"""Module 1 — Data Preparation.

Cleans the raw ATP tennis dataset, creates the binary target variable,
removes leakage columns, handles missing-value sentinels, and produces
time-based train / validation / hold-out splits saved to data/processed/.

Split boundaries
----------------
- Train    : Date < 2019-01-01
- Validation: 2019-01-01 <= Date < 2020-01-01
- Hold-out : Date >= 2020-01-01
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Columns that must never appear in the feature set
LEAKAGE_COLUMNS = ["Score", "Odd_1", "Odd_2", "Player_1", "Player_2", "Winner", "Tournament"]


def prepare_data(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean *raw_df* and return (train, val, holdout) DataFrames.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Output of ``utils.load_raw_data()``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train, val, holdout) — each includes the ``Player_1_wins`` target
        column and ``Date`` (for traceability) but no leakage columns.
    """
    df = raw_df.copy()

    # ------------------------------------------------------------------
    # 1. Parse Date
    # ------------------------------------------------------------------
    df["Date"] = pd.to_datetime(df["Date"])

    # ------------------------------------------------------------------
    # 2. Create binary target *before* dropping name columns
    # ------------------------------------------------------------------
    df["Player_1_wins"] = (df["Winner"] == df["Player_1"]).astype(int)

    # ------------------------------------------------------------------
    # 3. Drop leakage / name columns
    # ------------------------------------------------------------------
    df = df.drop(columns=LEAKAGE_COLUMNS)

    # ------------------------------------------------------------------
    # 4. Replace -1 sentinel with NaN in Pts_1, Pts_2
    #    Also drop rows where Rank_1 or Rank_2 are -1 (only 14 rows)
    # ------------------------------------------------------------------
    df["Pts_1"] = df["Pts_1"].replace(-1, np.nan)
    df["Pts_2"] = df["Pts_2"].replace(-1, np.nan)
    df = df[df["Rank_1"] != -1].copy()
    df = df[df["Rank_2"] != -1].copy()

    # ------------------------------------------------------------------
    # 5. Time-based split
    # ------------------------------------------------------------------
    train = df[df["Date"] < "2019-01-01"].copy()
    val   = df[(df["Date"] >= "2019-01-01") & (df["Date"] < "2020-01-01")].copy()
    holdout = df[df["Date"] >= "2020-01-01"].copy()

    # ------------------------------------------------------------------
    # 6. Impute Pts_1, Pts_2 using training-set medians only
    # ------------------------------------------------------------------
    pts1_median = train["Pts_1"].median()
    pts2_median = train["Pts_2"].median()

    for split in (train, val, holdout):
        split["Pts_1"] = split["Pts_1"].fillna(pts1_median)
        split["Pts_2"] = split["Pts_2"].fillna(pts2_median)

    return train, val, holdout


def main() -> None:
    """Run data preparation end-to-end and save outputs to disk."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from utils import load_raw_data

    print("Loading raw data …")
    raw_df = load_raw_data()
    print(f"  Raw shape: {raw_df.shape}")

    train, val, holdout = prepare_data(raw_df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val.to_csv(PROCESSED_DIR / "val.csv", index=False)
    holdout.to_csv(PROCESSED_DIR / "holdout.csv", index=False)

    print("\nSplit sizes:")
    print(f"  Train   : {len(train):>6,} rows  ({train['Date'].min().date()} – {train['Date'].max().date()})")
    print(f"  Val     : {len(val):>6,} rows  ({val['Date'].min().date()} – {val['Date'].max().date()})")
    print(f"  Hold-out: {len(holdout):>6,} rows  ({holdout['Date'].min().date()} – {holdout['Date'].max().date()})")
    print(f"\nColumns saved: {[c for c in train.columns]}")
    print("\nFiles written to data/processed/  OK")


if __name__ == "__main__":
    main()
