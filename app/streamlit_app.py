"""Module 6 — Streamlit Web UI Dashboard.

Launches an interactive dashboard where a user selects two ATP players,
configures match context (Surface, Court, Series, Round, Best of), clicks
"Predict", and instantly sees predicted win probabilities.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — required for Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — make src importable regardless of working directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import streamlit as st

from utils import (  # noqa: E402
    PROCESSED_DIR,
    load_encoder,
    load_feature_columns,
    load_model,
    load_raw_data,
)
from _feature_helpers import add_derived_features, build_feature_matrix  # noqa: E402

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ATP Tennis Match Predictor",
    page_icon="🎾",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Context-selector option lists (derived from the training dataset)
# ---------------------------------------------------------------------------
SURFACE_OPTIONS = ["Hard", "Clay", "Grass", "Carpet"]
COURT_OPTIONS   = ["Outdoor", "Indoor"]
SERIES_OPTIONS  = [
    "International", "International Gold", "ATP250", "ATP500",
    "Masters", "Masters 1000", "Masters Cup", "Grand Slam",
]
ROUND_OPTIONS   = [
    "1st Round", "2nd Round", "3rd Round", "4th Round",
    "Round Robin", "Quarterfinals", "Semifinals", "The Final",
]
BEST_OF_OPTIONS = [3, 5]

SURFACE_COLORS = {
    "Hard":   "#3b82d4",
    "Clay":   "#c0522b",
    "Grass":  "#2e7d32",
    "Carpet": "#6a1b9a",
}


# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model artefacts …")
def _load_artefacts():
    """Load model, encoder, and feature columns once and cache them."""
    model           = load_model()
    encoder         = load_encoder()
    feature_columns = load_feature_columns()
    return model, encoder, feature_columns


@st.cache_data(show_spinner="Loading player data …")
def _load_player_lookup() -> dict[str, dict]:
    """Return a dict mapping player name → {Rank, Pts} from their most recent match."""
    raw = load_raw_data()
    raw["Date"] = pd.to_datetime(raw["Date"])

    p1 = raw[["Date", "Player_1", "Rank_1", "Pts_1"]].rename(
        columns={"Player_1": "player", "Rank_1": "rank", "Pts_1": "pts"}
    )
    p2 = raw[["Date", "Player_2", "Rank_2", "Pts_2"]].rename(
        columns={"Player_2": "player", "Rank_2": "rank", "Pts_2": "pts"}
    )
    combined = pd.concat([p1, p2], ignore_index=True)
    combined.replace(-1, np.nan, inplace=True)

    latest = (
        combined.sort_values("Date")
        .dropna(subset=["rank"])
        .groupby("player")
        .last()
        .reset_index()
    )

    lookup: dict[str, dict] = {}
    for row in latest.itertuples(index=False):
        lookup[row.player] = {
            "rank": int(row.rank),
            "pts":  float(row.pts) if not np.isnan(row.pts) else 0.0,
        }
    return lookup


@st.cache_data(show_spinner="Loading match history …")
def _load_full_history() -> pd.DataFrame:
    """Return the full raw DataFrame with parsed dates."""
    raw = load_raw_data()
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw["Pts_1"] = raw["Pts_1"].replace(-1, np.nan)
    raw["Pts_2"] = raw["Pts_2"].replace(-1, np.nan)
    raw["Rank_1"] = raw["Rank_1"].replace(-1, np.nan)
    raw["Rank_2"] = raw["Rank_2"].replace(-1, np.nan)
    return raw


@st.cache_data(show_spinner="Computing model metrics …")
def _load_model_metrics() -> dict | None:
    """Load hold-out metrics if the live_prediction_results CSV exists."""
    results_path = PROCESSED_DIR / "live_prediction_results.csv"
    if not results_path.exists():
        return None
    df = pd.read_csv(results_path)
    metrics = {}
    if "Correct" in df.columns:
        metrics["accuracy"] = df["Correct"].mean() * 100
        metrics["n_matches"] = len(df)
    if "P1_Win_Prob" in df.columns:
        metrics["avg_confidence"] = df["P1_Win_Prob"].apply(lambda p: max(p, 1 - p)).mean() * 100
    return metrics


def _sorted_player_list(lookup: dict) -> list[str]:
    """Return player names sorted by ascending rank (best first)."""
    return sorted(lookup.keys(), key=lambda p: lookup[p]["rank"])


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------

def _predict(
    p1_rank: int, p1_pts: float,
    p2_rank: int, p2_pts: float,
    surface: str, court: str, series: str,
    round_: str, best_of: int,
    model, encoder, feature_columns: list[str],
) -> tuple[float, float]:
    """Return (p1_prob, p2_prob) for a single synthetic match row."""
    row = pd.DataFrame([{
        "Rank_1":   p1_rank,
        "Rank_2":   p2_rank,
        "Pts_1":    p1_pts,
        "Pts_2":    p2_pts,
        "Best of":  best_of,
        "Surface":  surface,
        "Court":    court,
        "Series":   series,
        "Round":    round_,
    }])
    row = add_derived_features(row)
    X   = build_feature_matrix(row, encoder, feature_columns)
    proba = model.predict_proba(X)[0]   # shape (2,) → [P(0), P(1)]
    p1_prob = float(proba[1])
    p2_prob = float(proba[0])
    return p1_prob, p2_prob


# ---------------------------------------------------------------------------
# UI helpers — charts
# ---------------------------------------------------------------------------

def _probability_chart(
    p1_name: str, p1_prob: float,
    p2_name: str, p2_prob: float,
) -> plt.Figure:
    """Return a horizontal bar chart showing the two win probabilities."""
    fig, ax = plt.subplots(figsize=(7, 2.0))

    colors = ["#3b82d4", "#e8705a"]
    bars = ax.barh(
        [p2_name, p1_name],
        [p2_prob * 100, p1_prob * 100],
        color=colors[::-1],
        height=0.5,
    )
    for bar, pct in zip(bars, [p2_prob * 100, p1_prob * 100]):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlim(0, 115)
    ax.set_xlabel("Win probability (%)", fontsize=9)
    ax.axvline(50, color="#aaa", linewidth=0.8, linestyle="--")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    return fig


def _confidence_gauge(prob: float) -> plt.Figure:
    """Return a simple arc-gauge chart showing prediction confidence."""
    confidence = max(prob, 1 - prob)
    fig, ax = plt.subplots(figsize=(3, 1.8), subplot_kw={"projection": "polar"})

    # Draw background arc (grey)
    theta = np.linspace(np.pi, 0, 200)
    ax.plot(theta, [1] * 200, color="#e5e7eb", linewidth=12, solid_capstyle="round")

    # Draw filled arc (coloured by confidence)
    fill_end = np.pi * (1 - confidence)
    theta_fill = np.linspace(np.pi, fill_end, 200)
    if confidence >= 0.75:
        color = "#22c55e"
    elif confidence >= 0.60:
        color = "#f59e0b"
    else:
        color = "#ef4444"
    ax.plot(theta_fill, [1] * len(theta_fill), color=color, linewidth=12, solid_capstyle="round")

    ax.set_ylim(0, 1.5)
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(-1)
    ax.axis("off")
    ax.text(0, -0.3, f"{confidence * 100:.0f}%", ha="center", va="center",
            fontsize=16, fontweight="bold", transform=ax.transData)
    ax.text(0, -0.65, "Confidence", ha="center", va="center",
            fontsize=9, color="#57606a", transform=ax.transData)
    fig.tight_layout()
    return fig


def _surface_win_rate_chart(history: pd.DataFrame, player: str) -> plt.Figure:
    """Return a bar chart of a player's win rate per surface."""
    p1_rows = history[history["Player_1"] == player].copy()
    p1_rows["won"] = (p1_rows["Winner"] == player).astype(int)

    p2_rows = history[history["Player_2"] == player].copy()
    p2_rows["won"] = (p2_rows["Winner"] == player).astype(int)

    p1_rows = p1_rows[["Surface", "won"]]
    p2_rows = p2_rows[["Surface", "won"]]
    combined = pd.concat([p1_rows, p2_rows], ignore_index=True)

    stats = combined.groupby("Surface")["won"].agg(["mean", "count"]).reset_index()
    stats.columns = ["Surface", "WinRate", "Matches"]
    stats = stats[stats["Matches"] >= 3]  # only surfaces with ≥3 matches

    if stats.empty:
        return None

    fig, ax = plt.subplots(figsize=(4, 2.2))
    colors = [SURFACE_COLORS.get(s, "#3b82d4") for s in stats["Surface"]]
    bars = ax.bar(stats["Surface"], stats["WinRate"] * 100, color=colors, width=0.5)

    for bar, row in zip(bars, stats.itertuples()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{row.WinRate * 100:.0f}%\n({row.Matches})",
            ha="center", va="bottom", fontsize=8
        )

    ax.set_ylim(0, 110)
    ax.set_ylabel("Win Rate (%)", fontsize=8)
    ax.axhline(50, color="#aaa", linewidth=0.7, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)
    fig.tight_layout()
    return fig


def _h2h_timeline_chart(h2h_df: pd.DataFrame, p1_name: str, p2_name: str) -> plt.Figure:
    """Return a scatter timeline of H2H results."""
    if h2h_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 1.8))
    p1_wins  = h2h_df[h2h_df["Winner"] == p1_name]
    p2_wins  = h2h_df[h2h_df["Winner"] == p2_name]

    ax.scatter(p1_wins["Date"], [1] * len(p1_wins), color="#3b82d4",
               s=70, zorder=3, label=f"{p1_name} won")
    ax.scatter(p2_wins["Date"], [1] * len(p2_wins), color="#e8705a",
               s=70, zorder=3, label=f"{p2_name} won", marker="D")

    ax.set_ylim(0.5, 1.5)
    ax.set_yticks([])
    ax.set_xlabel("Date", fontsize=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=7)
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _get_h2h(history: pd.DataFrame, p1_name: str, p2_name: str) -> pd.DataFrame:
    """Return all matches between p1 and p2 (in either slot)."""
    mask = (
        ((history["Player_1"] == p1_name) & (history["Player_2"] == p2_name)) |
        ((history["Player_1"] == p2_name) & (history["Player_2"] == p1_name))
    )
    return history[mask].sort_values("Date").reset_index(drop=True)


def _player_surface_record(history: pd.DataFrame, player: str) -> pd.DataFrame:
    """Return a summary DataFrame of a player's record per surface."""
    p1 = history[history["Player_1"] == player][["Surface", "Winner"]].copy()
    p2 = history[history["Player_2"] == player][["Surface", "Winner"]].copy()
    combined = pd.concat([p1, p2], ignore_index=True)
    combined["Won"] = (combined["Winner"] == player).astype(int)
    stats = combined.groupby("Surface")["Won"].agg(["sum", "count"]).reset_index()
    stats.columns = ["Surface", "Wins", "Matches"]
    stats["Losses"] = stats["Matches"] - stats["Wins"]
    stats["Win%"] = (stats["Wins"] / stats["Matches"] * 100).round(1)
    return stats[stats["Matches"] >= 1].sort_values("Matches", ascending=False)


# ---------------------------------------------------------------------------
# Main app layout
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- Header ------------------------------------------------------------
    st.title("🎾 ATP Tennis Match Predictor")
    st.markdown(
        "Select two players and configure the match context. "
        "The model predicts win probability based on ATP rankings, "
        "points, and match conditions — **no betting odds used**."
    )
    st.divider()

    # ---- Load artefacts (cached) ------------------------------------------
    model, encoder, feature_columns = _load_artefacts()
    lookup  = _load_player_lookup()
    history = _load_full_history()
    players = _sorted_player_list(lookup)

    # ---- Player selectors --------------------------------------------------
    st.subheader("Players")
    col1, col2 = st.columns(2)
    with col1:
        p1_name = st.selectbox("Player 1", options=players, index=0, key="p1")
    with col2:
        default_p2_idx = 1 if len(players) > 1 else 0
        p2_name = st.selectbox("Player 2", options=players, index=default_p2_idx, key="p2")

    p1_info = lookup[p1_name]
    p2_info = lookup[p2_name]
    col1.caption(f"Last known rank: **#{p1_info['rank']}**  |  Points: {p1_info['pts']:,.0f}")
    col2.caption(f"Last known rank: **#{p2_info['rank']}**  |  Points: {p2_info['pts']:,.0f}")

    # ---- Match context selectors -------------------------------------------
    st.subheader("Match Context")
    ctx_col1, ctx_col2, ctx_col3 = st.columns(3)
    with ctx_col1:
        surface = st.selectbox("Surface", SURFACE_OPTIONS, index=0)
        court   = st.selectbox("Court",   COURT_OPTIONS,   index=0)
    with ctx_col2:
        series  = st.selectbox("Series",  SERIES_OPTIONS,  index=0)
        round_  = st.selectbox("Round",   ROUND_OPTIONS,   index=0)
    with ctx_col3:
        best_of = st.selectbox("Best of", BEST_OF_OPTIONS, index=0)

    st.divider()

    # ---- Predict button ----------------------------------------------------
    same_player = (p1_name == p2_name)
    if same_player:
        st.warning("Please select two different players.")

    predict_clicked = st.button(
        "⚡ Predict",
        type="primary",
        disabled=same_player,
        use_container_width=True,
    )

    if predict_clicked and not same_player:
        p1_prob, p2_prob = _predict(
            p1_rank=p1_info["rank"],
            p1_pts=p1_info["pts"],
            p2_rank=p2_info["rank"],
            p2_pts=p2_info["pts"],
            surface=surface,
            court=court,
            series=series,
            round_=round_,
            best_of=best_of,
            model=model,
            encoder=encoder,
            feature_columns=feature_columns,
        )

        st.subheader("Prediction Result")

        favourite = p1_name if p1_prob >= p2_prob else p2_name
        fav_prob  = max(p1_prob, p2_prob)

        st.success(f"**Predicted winner: {favourite}** ({fav_prob * 100:.1f}% win probability)")

        # ---- Probability bar + confidence gauge (side by side) ------------
        res_col1, res_col2 = st.columns([3, 1])
        with res_col1:
            fig = _probability_chart(p1_name, p1_prob, p2_name, p2_prob)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with res_col2:
            gauge = _confidence_gauge(p1_prob)
            st.pyplot(gauge, use_container_width=True)
            plt.close(gauge)

        # ---- Metric row ---------------------------------------------------
        m1, m2 = st.columns(2)
        m1.metric(label=f"🎾 {p1_name}", value=f"{p1_prob * 100:.1f}%", help="P(Player 1 wins)")
        m2.metric(label=f"🎾 {p2_name}", value=f"{p2_prob * 100:.1f}%", help="P(Player 2 wins)")

        # ---- Confidence label ---------------------------------------------
        confidence = fav_prob
        if confidence >= 0.75:
            conf_label, conf_color = "High confidence", "🟢"
        elif confidence >= 0.60:
            conf_label, conf_color = "Moderate confidence", "🟡"
        else:
            conf_label, conf_color = "Low confidence (near coin-flip)", "🔴"
        st.info(f"{conf_color} **{conf_label}** — The model favours {favourite} by "
                f"{abs(p1_prob - p2_prob) * 100:.1f} percentage points.")

        st.caption(
            "⚠️ Prediction is based solely on ATP rankings, points, and match context. "
            "Betting odds and player names are never used by the model."
        )

    st.divider()

    # ==========================================================================
    # NEW FEATURE 1: Head-to-Head Stats
    # ==========================================================================
    with st.expander("🤜🤛 Head-to-Head History", expanded=not same_player):
        if same_player:
            st.info("Select two different players to view head-to-head stats.")
        else:
            h2h = _get_h2h(history, p1_name, p2_name)

            if h2h.empty:
                st.info(f"No recorded matches found between **{p1_name}** and **{p2_name}** in the dataset.")
            else:
                p1_h2h_wins = (h2h["Winner"] == p1_name).sum()
                p2_h2h_wins = (h2h["Winner"] == p2_name).sum()
                total_h2h   = len(h2h)

                h2h_c1, h2h_c2, h2h_c3 = st.columns(3)
                h2h_c1.metric(f"🏆 {p1_name}", f"{p1_h2h_wins}W",
                              delta=f"{p1_h2h_wins - p2_h2h_wins:+d} vs opponent")
                h2h_c2.metric("Total Matches", str(total_h2h))
                h2h_c3.metric(f"🏆 {p2_name}", f"{p2_h2h_wins}W",
                              delta=f"{p2_h2h_wins - p1_h2h_wins:+d} vs opponent")

                # H2H timeline
                timeline_fig = _h2h_timeline_chart(h2h, p1_name, p2_name)
                if timeline_fig:
                    st.pyplot(timeline_fig, use_container_width=True)
                    plt.close(timeline_fig)

                # H2H surface breakdown
                h2h_surface = h2h.groupby("Surface").apply(
                    lambda g: pd.Series({
                        f"{p1_name} W": (g["Winner"] == p1_name).sum(),
                        f"{p2_name} W": (g["Winner"] == p2_name).sum(),
                        "Matches":       len(g),
                    })
                ).reset_index()

                st.markdown("**H2H record by surface:**")
                st.dataframe(h2h_surface, use_container_width=True, hide_index=True)

                # Recent 5 matches table
                st.markdown("**Last 5 meetings:**")
                display_cols = ["Date", "Tournament", "Surface", "Round", "Winner", "Score"]
                available_cols = [c for c in display_cols if c in h2h.columns]
                recent = h2h.tail(5)[available_cols].copy()
                recent["Date"] = recent["Date"].dt.strftime("%Y-%m-%d")
                st.dataframe(recent, use_container_width=True, hide_index=True)

    # ==========================================================================
    # NEW FEATURE 2: Player Surface Win Rates
    # ==========================================================================
    with st.expander("🏟️ Surface Win Rates", expanded=False):
        if same_player:
            st.info("Select two different players to compare surface stats.")
        else:
            surf_c1, surf_c2 = st.columns(2)

            for col, player in [(surf_c1, p1_name), (surf_c2, p2_name)]:
                with col:
                    st.markdown(f"**{player}**")
                    fig = _surface_win_rate_chart(history, player)
                    if fig:
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    else:
                        st.info("Not enough surface data.")

                    # Tabular breakdown
                    record = _player_surface_record(history, player)
                    st.dataframe(
                        record[["Surface", "Wins", "Losses", "Matches", "Win%"]],
                        use_container_width=True,
                        hide_index=True,
                    )

    # ==========================================================================
    # NEW FEATURE 3: Match History Explorer
    # ==========================================================================
    # Compute year bounds outside the expander so the slider always sees real bounds
    _year_min = int(history["Date"].dt.year.min())
    _year_max = int(history["Date"].dt.year.max())

    with st.expander("📂 Match History Explorer", expanded=False):
        st.markdown(
            f"Browse and filter all historical ATP matches in the dataset "
            f"({_year_min}–{_year_max}). Use the filters below to narrow down."
        )

        filter_col, table_col = st.columns([1, 3])

        with filter_col:
            st.markdown("**Filters**")
            filter_player = st.text_input("Player name contains", value="", key="mh_player")
            filter_surface = st.multiselect(
                "Surface", SURFACE_OPTIONS, default=SURFACE_OPTIONS, key="mh_surface"
            )
            filter_series = st.multiselect(
                "Series", SERIES_OPTIONS, default=SERIES_OPTIONS, key="mh_series"
            )
            year_range = st.slider(
                f"Year range ({_year_min}–{_year_max})",
                _year_min, _year_max,
                (_year_min, _year_max),
                key="mh_year",
            )

        with table_col:
            mh = history.copy()
            mh = mh[mh["Surface"].isin(filter_surface)]
            mh = mh[mh["Series"].isin(filter_series)]
            mh = mh[mh["Date"].dt.year.between(year_range[0], year_range[1])]
            if filter_player.strip():
                mask = (
                    mh["Player_1"].str.contains(filter_player.strip(), case=False, na=False) |
                    mh["Player_2"].str.contains(filter_player.strip(), case=False, na=False)
                )
                mh = mh[mask]

            st.caption(f"Showing **{len(mh):,}** matches")
            display_mh = mh[["Date", "Tournament", "Surface", "Series", "Round",
                              "Player_1", "Player_2", "Winner"]].copy()
            display_mh["Date"] = display_mh["Date"].dt.strftime("%Y-%m-%d")
            st.dataframe(display_mh.tail(200), use_container_width=True, hide_index=True)
            st.caption("Showing latest 200 matches of the filtered set.")

    # ==========================================================================
    # NEW FEATURE 4: Model Performance Dashboard
    # ==========================================================================
    with st.expander("📈 Model Performance", expanded=False):
        st.markdown(
            "Summary of the trained Random Forest model's performance on the **hold-out set** "
            "(matches from 2020 onwards — never seen during training)."
        )
        metrics = _load_model_metrics()
        if metrics:
            perf_c1, perf_c2, perf_c3 = st.columns(3)
            if "accuracy" in metrics:
                perf_c1.metric("Hold-out Accuracy", f"{metrics['accuracy']:.1f}%")
            if "n_matches" in metrics:
                perf_c2.metric("Hold-out Matches", f"{metrics['n_matches']:,}")
            if "avg_confidence" in metrics:
                perf_c3.metric("Avg Prediction Confidence", f"{metrics['avg_confidence']:.1f}%")
        else:
            st.info("Run `src/05_live_prediction.py` to generate hold-out performance metrics.")

        # Model metadata
        st.markdown("**Model configuration**")
        st.table(pd.DataFrame([
            {"Property": "Algorithm",           "Value": "Random Forest Classifier"},
            {"Property": "Features",            "Value": "Rank, Points, Surface, Court, Series, Round, Best of"},
            {"Property": "Derived features",    "Value": "Rank_diff, Pts_diff"},
            {"Property": "Train set boundary",  "Value": "Matches before 2019-01-01"},
            {"Property": "Validation set",      "Value": "Matches in 2019"},
            {"Property": "Hold-out set",        "Value": "Matches from 2020 onwards"},
            {"Property": "Excluded features",   "Value": "Betting odds, Player names, Score"},
        ]))

    # ==========================================================================
    # Existing features (preserved)
    # ==========================================================================

    # ---- Feature importance charts ----------------------------------------
    with st.expander("📊 Feature Importance Charts", expanded=False):
        st.markdown(
            "These charts show which features most influence the model's predictions."
        )
        builtin_path     = PROCESSED_DIR / "importance_builtin.png"
        permutation_path = PROCESSED_DIR / "importance_permutation.png"
        shap_path        = PROCESSED_DIR / "importance_shap.png"

        imp_col1, imp_col2 = st.columns(2)
        with imp_col1:
            if builtin_path.exists():
                st.image(str(builtin_path), caption="Built-in (impurity) importance", use_container_width=True)
            else:
                st.info("Run `src/04_model_interpretation.py` to generate this chart.")
        with imp_col2:
            if permutation_path.exists():
                st.image(str(permutation_path), caption="Permutation importance (validation set)", use_container_width=True)
            else:
                st.info("Run `src/04_model_interpretation.py` to generate this chart.")

        if shap_path.exists():
            st.image(str(shap_path), caption="SHAP summary plot", use_container_width=True)

    # ---- Live Prediction Challenge ----------------------------------------
    with st.expander("🔮 Live Prediction Challenge (Hold-out 2020+)", expanded=False):
        st.markdown(
            "The table below shows predictions made on the **hold-out set** — "
            "matches from 2020 onwards that were never seen during training or tuning."
        )
        results_path = PROCESSED_DIR / "live_prediction_results.csv"
        if results_path.exists():
            results_df = pd.read_csv(results_path)
            st.dataframe(results_df, use_container_width=True, hide_index=True)
        else:
            st.info("Run `src/05_live_prediction.py` to generate this table.")

    # ---- Footer -----------------------------------------------------------
    st.divider()
    st.caption(
        "Model: Random Forest Classifier · Features: ATP Rank, Points, Surface, "
        "Court, Series, Round, Best of · Training data: ATP matches up to 2018 · "
        "No betting odds used."
    )


if __name__ == "__main__":
    main()
