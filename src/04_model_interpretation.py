"""Module 4 — Model Interpretation.

Explains what the trained Random Forest has learned using two complementary
methods:

1. **Built-in (impurity) feature importance** — ``model.feature_importances_``
   averaged over all trees.
2. **Permutation importance** — shuffles each feature on the *validation* set
   and measures the drop in AUC-ROC (``sklearn.inspection.permutation_importance``).

Outputs
-------
- ``data/processed/importance_builtin.png``     — top-15 built-in importances
- ``data/processed/importance_permutation.png`` — top-15 permutation importances
- ``data/processed/importance_shap.png``        — SHAP summary (if shap installed)
"""

from pathlib import Path
import sys

import joblib
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving to file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"

# Make src importable when running as a script
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _horizontal_bar(
    names: list[str],
    values: np.ndarray,
    title: str,
    xlabel: str,
    save_path: Path,
    *,
    errors: np.ndarray | None = None,
) -> None:
    """Save a horizontal bar chart of feature importances.

    Parameters
    ----------
    names : list[str]
        Feature names (already sorted, most important last for bottom-up display).
    values : np.ndarray
        Importance scores aligned to *names*.
    title : str
        Chart title.
    xlabel : str
        X-axis label.
    save_path : Path
        Destination file path (PNG).
    errors : np.ndarray, optional
        Standard deviations for error bars (permutation importance only).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(names))

    if errors is not None:
        ax.barh(y_pos, values, xerr=errors, align="center", color="#3b82d4",
                ecolor="#57606a", capsize=3)
    else:
        ax.barh(y_pos, values, align="center", color="#3b82d4")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=11, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {save_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Built-in feature importance
# ---------------------------------------------------------------------------

def plot_builtin_importance(
    model,
    feature_columns: list[str],
    top_n: int = 15,
) -> pd.DataFrame:
    """Extract and plot the RF's impurity-based feature importances.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``feature`` and ``importance``, sorted
        descending.
    """
    importances = model.feature_importances_
    df_imp = pd.DataFrame({"feature": feature_columns, "importance": importances})
    df_imp = df_imp.sort_values("importance", ascending=False).head(top_n)

    # Sort ascending so most important appears at the top of a horizontal bar chart
    df_imp = df_imp.sort_values("importance", ascending=True)

    _horizontal_bar(
        names=df_imp["feature"].tolist(),
        values=df_imp["importance"].to_numpy(),
        title=f"Top {top_n} Built-in Feature Importances\n(mean impurity decrease)",
        xlabel="Mean decrease in impurity",
        save_path=PROCESSED_DIR / "importance_builtin.png",
    )
    return df_imp.sort_values("importance", ascending=False)


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------

def plot_permutation_importance(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_columns: list[str],
    top_n: int = 15,
    *,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute permutation importance on the validation set and plot top-N.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``feature``, ``importance_mean``,
        ``importance_std``, sorted descending by mean.
    """
    result = permutation_importance(
        model, X_val, y_val,
        n_repeats=n_repeats,
        scoring="roc_auc",
        random_state=random_state,
        n_jobs=-1,
    )

    df_perm = pd.DataFrame({
        "feature":          feature_columns,
        "importance_mean":  result.importances_mean,
        "importance_std":   result.importances_std,
    })
    df_perm = df_perm.sort_values("importance_mean", ascending=False).head(top_n)
    df_perm = df_perm.sort_values("importance_mean", ascending=True)  # for chart

    _horizontal_bar(
        names=df_perm["feature"].tolist(),
        values=df_perm["importance_mean"].to_numpy(),
        errors=df_perm["importance_std"].to_numpy(),
        title=f"Top {top_n} Permutation Importances (validation set)\n"
              f"(n_repeats={n_repeats}, scoring=roc_auc)",
        xlabel="Mean AUC-ROC drop when feature is shuffled",
        save_path=PROCESSED_DIR / "importance_permutation.png",
    )
    return df_perm.sort_values("importance_mean", ascending=False)


# ---------------------------------------------------------------------------
# SHAP summary (optional)
# ---------------------------------------------------------------------------

def plot_shap_summary(
    model,
    X_val: np.ndarray,
    feature_columns: list[str],
    max_display: int = 15,
) -> bool:
    """Generate a SHAP summary plot if the ``shap`` library is available.

    Returns
    -------
    bool
        ``True`` if the plot was generated, ``False`` if shap is not installed.
    """
    try:
        import shap  # noqa: PLC0415
    except ImportError:
        return False

    explainer = shap.TreeExplainer(model)
    # Use at most 500 rows for speed
    sample = X_val[:500] if len(X_val) > 500 else X_val
    shap_values = explainer.shap_values(sample)

    # For binary classification shap returns a list [class0, class1]
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    fig, ax = plt.subplots(figsize=(8, 7))
    shap.summary_plot(
        sv, sample,
        feature_names=feature_columns,
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    plt.title("SHAP Summary Plot (top features)", fontsize=11)
    plt.tight_layout()
    save_path = PROCESSED_DIR / "importance_shap.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {save_path.relative_to(PROJECT_ROOT)}")
    return True


# ---------------------------------------------------------------------------
# Comparison summary
# ---------------------------------------------------------------------------

def print_comparison(df_builtin: pd.DataFrame, df_perm: pd.DataFrame) -> None:
    """Print a side-by-side comparison of the two importance rankings."""
    top5_builtin = df_builtin.head(5)["feature"].tolist()
    top5_perm    = df_perm.head(5)["feature"].tolist()
    common       = set(top5_builtin) & set(top5_perm)

    print("\n-- Feature Importance Comparison --------------------------------")
    print(f"  Top-5 built-in   : {top5_builtin}")
    print(f"  Top-5 permutation: {top5_perm}")
    if len(common) >= 3:
        print(f"  Agreement  : GOOD — {len(common)}/5 features appear in both top-5 lists.")
        print(f"    Shared: {sorted(common)}")
    else:
        print(f"  Agreement  : DIVERGENT — only {len(common)}/5 features overlap.")
        print("  Note: built-in importance can overstate high-cardinality or correlated")
        print("  features; permutation importance on the val set is generally more reliable.")
    print("-----------------------------------------------------------------")


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run model interpretation end-to-end."""
    from _feature_helpers import add_derived_features, build_feature_matrix

    print("Loading model and artefacts …")
    model           = joblib.load(MODELS_DIR / "rf_model.pkl")
    encoder         = joblib.load(MODELS_DIR / "encoder.pkl")
    feature_columns = joblib.load(MODELS_DIR / "feature_columns.pkl")

    print("Loading validation split …")
    val   = pd.read_csv(PROCESSED_DIR / "val.csv")
    val   = add_derived_features(val)
    X_val = build_feature_matrix(val, encoder, feature_columns)
    y_val = val["Player_1_wins"].to_numpy()
    print(f"  Validation rows: {len(val):,}  |  Features: {X_val.shape[1]}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Built-in feature importances …")
    df_builtin = plot_builtin_importance(model, feature_columns, top_n=15)

    print("\n[2/3] Permutation importances (n_repeats=10) …")
    df_perm = plot_permutation_importance(
        model, X_val, y_val, feature_columns, top_n=15, n_repeats=10
    )

    print_comparison(df_builtin, df_perm)

    print("\n[3/3] SHAP summary plot (optional) …")
    shap_generated = plot_shap_summary(model, X_val, feature_columns, max_display=15)
    if not shap_generated:
        print("  shap not installed — skipping SHAP plot.")

    print("\nModel interpretation complete  OK")


if __name__ == "__main__":
    main()
