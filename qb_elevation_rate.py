"""
QB Elevation Rate — v6 (Optimized Threshold)
=============================================
Measures whether a QB produces more receiver career highs AND fewer career
lows than expected, using era-adjusted stats and seasons-weighted expected share.

  Net Elevation = Elevation O/E - Depression O/E

  > 0 = QB produces more highs than lows relative to expectation (true elevator)
  = 0 = QB is neutral — highs and lows cancel out
  < 0 = QB produces more lows than highs (suppressor)

Components:
  - Elevation O/E: actual career highs / expected (seasons-weighted)
  - Depression O/E: actual career lows / expected (seasons-weighted)
  - Era adjustment: stats normalized by league-season average
  - Seasons-weighted expected: proportional to seasons together, not equal 1/N
  - 50/50 blend of yards/game and TDs/game

Data: nfl_data_py (Python wrapper for nflfastR), 2006–2024 regular season.

Inclusion Criteria:
  - QB: 50+ career games as primary passer, 10+ qualifying receivers
  - Receiver (WR/TE): 6+ games in a season with a given QB
  - Multi-QB filter: receiver must have 6+ game seasons with at least 2 different QBs

Changes from v5:
  - Lowered receiver game threshold from 8 to 6 (optimal bias-variance tradeoff)
  - Raised QB starts minimum from 32 to 50 (removes short-career noise)
  - Added minimum 10 qualifying receivers per QB (removes small-sample outliers)
  - Final pool: 62 QBs, 1,357 receiver pairs
"""

import nfl_data_py as nfl
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
YEARS = list(range(2006, 2025))
MIN_QB_CAREER_STARTS = 50
MIN_REC_GAMES_SEASON = 6
MIN_QUALIFYING_RECEIVERS = 10
POSITIONS = ["WR", "TE"]

# ── 1. Load weekly data ───────────────────────────────────────────────────────
print("Loading weekly data...")
weekly = nfl.import_weekly_data(YEARS)
weekly = weekly[weekly["season_type"] == "REG"].copy()

# ── 2. League averages per season ────────────────────────────────────────────
rec_all = weekly[weekly["position"].isin(POSITIONS)].copy()
rec_all["receiving_yards"] = rec_all["receiving_yards"].fillna(0)
rec_all["receiving_tds"] = rec_all["receiving_tds"].fillna(0)

league_avgs = (
    rec_all.groupby("season")
    .agg(
        lg_total_rec_yards=("receiving_yards", "sum"),
        lg_total_rec_tds=("receiving_tds", "sum"),
        lg_player_weeks=("player_id", "count"),
    )
    .reset_index()
)
league_avgs["lg_ypg"] = league_avgs["lg_total_rec_yards"] / league_avgs["lg_player_weeks"]
league_avgs["lg_tdpg"] = league_avgs["lg_total_rec_tds"] / league_avgs["lg_player_weeks"]

# ── 3. Identify primary QB per team-week ──────────────────────────────────────
qb_data = weekly[weekly["position"] == "QB"].copy()
qb_data["attempts"] = qb_data["attempts"].fillna(0)

primary_qb = (
    qb_data.sort_values("attempts", ascending=False)
    .groupby(["season", "week", "recent_team"])
    .first()
    .reset_index()[["season", "week", "recent_team", "player_id", "player_display_name"]]
    .rename(columns={"player_id": "qb_id", "player_display_name": "qb_name"})
)

# ── 4. Build receiver-QB-season pairs ─────────────────────────────────────────
rec_data = weekly[weekly["position"].isin(POSITIONS)].copy()
rec_data["receiving_yards"] = rec_data["receiving_yards"].fillna(0)
rec_data["receiving_tds"] = rec_data["receiving_tds"].fillna(0)

rec_qb = rec_data.merge(primary_qb, on=["season", "week", "recent_team"], how="inner")

rec_qb_season = (
    rec_qb.groupby(
        ["season", "qb_id", "qb_name", "player_id", "player_display_name", "position"]
    )
    .agg(
        games_together=("week", "count"),
        total_rec_yards=("receiving_yards", "sum"),
        total_rec_tds=("receiving_tds", "sum"),
        total_targets=("targets", "sum"),
        total_receptions=("receptions", "sum"),
    )
    .reset_index()
)

rec_qb_season["raw_ypg"] = rec_qb_season["total_rec_yards"] / rec_qb_season["games_together"]
rec_qb_season["raw_tdpg"] = rec_qb_season["total_rec_tds"] / rec_qb_season["games_together"]

rec_qb_season = rec_qb_season.merge(
    league_avgs[["season", "lg_ypg", "lg_tdpg"]], on="season", how="left"
)
rec_qb_season["era_adj_ypg"] = rec_qb_season["raw_ypg"] / rec_qb_season["lg_ypg"]
rec_qb_season["era_adj_tdpg"] = rec_qb_season["raw_tdpg"] / rec_qb_season["lg_tdpg"]

# ── 5. Apply inclusion criteria ───────────────────────────────────────────────
rqs = rec_qb_season[rec_qb_season["games_together"] >= MIN_REC_GAMES_SEASON].copy()

rec_qb_counts = (
    rqs.groupby("player_id")["qb_id"]
    .nunique()
    .reset_index()
    .rename(columns={"qb_id": "num_qbs"})
)
multi_qb_receivers = rec_qb_counts[rec_qb_counts["num_qbs"] >= 2]["player_id"].tolist()
rqs_multi = rqs[rqs["player_id"].isin(multi_qb_receivers)].copy()

# ── 6. Flag career HIGHS and LOWS (era-adjusted) ────────────────────────────
# Highs
ypg_best = rqs_multi.loc[rqs_multi.groupby("player_id")["era_adj_ypg"].idxmax()][
    ["player_id", "season", "qb_id"]
].rename(columns={"season": "best_ypg_season", "qb_id": "best_ypg_qb"})
tdpg_best = rqs_multi.loc[rqs_multi.groupby("player_id")["era_adj_tdpg"].idxmax()][
    ["player_id", "season", "qb_id"]
].rename(columns={"season": "best_tdpg_season", "qb_id": "best_tdpg_qb"})

# Lows
ypg_worst = rqs_multi.loc[rqs_multi.groupby("player_id")["era_adj_ypg"].idxmin()][
    ["player_id", "season", "qb_id"]
].rename(columns={"season": "worst_ypg_season", "qb_id": "worst_ypg_qb"})
tdpg_worst = rqs_multi.loc[rqs_multi.groupby("player_id")["era_adj_tdpg"].idxmin()][
    ["player_id", "season", "qb_id"]
].rename(columns={"season": "worst_tdpg_season", "qb_id": "worst_tdpg_qb"})

rqs_multi = rqs_multi.merge(ypg_best, on="player_id", how="left")
rqs_multi = rqs_multi.merge(tdpg_best, on="player_id", how="left")
rqs_multi = rqs_multi.merge(ypg_worst, on="player_id", how="left")
rqs_multi = rqs_multi.merge(tdpg_worst, on="player_id", how="left")

rqs_multi["is_ypg_high"] = (
    (rqs_multi["season"] == rqs_multi["best_ypg_season"])
    & (rqs_multi["qb_id"] == rqs_multi["best_ypg_qb"])
).astype(int)
rqs_multi["is_tdpg_high"] = (
    (rqs_multi["season"] == rqs_multi["best_tdpg_season"])
    & (rqs_multi["qb_id"] == rqs_multi["best_tdpg_qb"])
).astype(int)
rqs_multi["is_ypg_low"] = (
    (rqs_multi["season"] == rqs_multi["worst_ypg_season"])
    & (rqs_multi["qb_id"] == rqs_multi["worst_ypg_qb"])
).astype(int)
rqs_multi["is_tdpg_low"] = (
    (rqs_multi["season"] == rqs_multi["worst_tdpg_season"])
    & (rqs_multi["qb_id"] == rqs_multi["worst_tdpg_qb"])
).astype(int)

# ── 7. QB career starts filter ───────────────────────────────────────────────
qb_career_starts = (
    primary_qb.groupby("qb_id").size().reset_index(name="career_starts")
)
qualifying_qbs = qb_career_starts[
    qb_career_starts["career_starts"] >= MIN_QB_CAREER_STARTS
]["qb_id"].tolist()

# ── 8. Receiver metadata ─────────────────────────────────────────────────────
rec_num_qbs_all = (
    rqs_multi.groupby("player_id")["qb_id"]
    .nunique()
    .reset_index()
    .rename(columns={"qb_id": "num_total_qbs"})
)
rec_total_seasons = (
    rqs_multi.groupby("player_id")
    .agg(total_qualifying_seasons=("season", "count"))
    .reset_index()
)

# ── 9. Build QB-receiver pairs ───────────────────────────────────────────────
qb_rec_pairs = (
    rqs_multi[rqs_multi["qb_id"].isin(qualifying_qbs)]
    .groupby(["qb_id", "qb_name", "player_id", "player_display_name", "position"])
    .agg(
        ypg_career_high=("is_ypg_high", "max"),
        tdpg_career_high=("is_tdpg_high", "max"),
        ypg_career_low=("is_ypg_low", "max"),
        tdpg_career_low=("is_tdpg_low", "max"),
        seasons_together=("season", "count"),
        best_raw_ypg=("raw_ypg", "max"),
        best_raw_tdpg=("raw_tdpg", "max"),
        worst_raw_ypg=("raw_ypg", "min"),
        worst_raw_tdpg=("raw_tdpg", "min"),
        best_era_adj_ypg=("era_adj_ypg", "max"),
        best_era_adj_tdpg=("era_adj_tdpg", "max"),
        worst_era_adj_ypg=("era_adj_ypg", "min"),
        worst_era_adj_tdpg=("era_adj_tdpg", "min"),
        total_games=("games_together", "sum"),
    )
    .reset_index()
)

qb_rec_pairs = qb_rec_pairs.merge(rec_num_qbs_all, on="player_id", how="left")
qb_rec_pairs = qb_rec_pairs.merge(rec_total_seasons, on="player_id", how="left")
qb_rec_pairs["expected_share_equal"] = 1.0 / qb_rec_pairs["num_total_qbs"]
qb_rec_pairs["expected_share_seasons"] = (
    qb_rec_pairs["seasons_together"] / qb_rec_pairs["total_qualifying_seasons"]
)

# ── 10. Compute Elevation, Depression, and Net at QB level ───────────────────
qb_elev = (
    qb_rec_pairs.groupby(["qb_id", "qb_name"])
    .agg(
        qualifying_receivers=("player_id", "count"),
        ypg_highs=("ypg_career_high", "sum"),
        tdpg_highs=("tdpg_career_high", "sum"),
        ypg_lows=("ypg_career_low", "sum"),
        tdpg_lows=("tdpg_career_low", "sum"),
        expected_equal=("expected_share_equal", "sum"),
        expected_seasons=("expected_share_seasons", "sum"),
    )
    .reset_index()
)

# Elevation O/E
qb_elev["elev_ypg_oe"] = qb_elev["ypg_highs"] / qb_elev["expected_seasons"]
qb_elev["elev_tdpg_oe"] = qb_elev["tdpg_highs"] / qb_elev["expected_seasons"]
qb_elev["elev_blended"] = (qb_elev["elev_ypg_oe"] + qb_elev["elev_tdpg_oe"]) / 2

# Depression O/E
qb_elev["depr_ypg_oe"] = qb_elev["ypg_lows"] / qb_elev["expected_seasons"]
qb_elev["depr_tdpg_oe"] = qb_elev["tdpg_lows"] / qb_elev["expected_seasons"]
qb_elev["depr_blended"] = (qb_elev["depr_ypg_oe"] + qb_elev["depr_tdpg_oe"]) / 2

# Net Elevation
qb_elev["net_ypg"] = qb_elev["elev_ypg_oe"] - qb_elev["depr_ypg_oe"]
qb_elev["net_tdpg"] = qb_elev["elev_tdpg_oe"] - qb_elev["depr_tdpg_oe"]
qb_elev["net_blended"] = qb_elev["elev_blended"] - qb_elev["depr_blended"]

# Raw rates
qb_elev["yards_rate"] = qb_elev["ypg_highs"] / qb_elev["qualifying_receivers"]
qb_elev["td_rate"] = qb_elev["tdpg_highs"] / qb_elev["qualifying_receivers"]

qb_elev = qb_elev.merge(qb_career_starts, on="qb_id", how="left")

# ── 11. Apply minimum qualifying receivers filter ────────────────────────────
qb_elev = qb_elev[qb_elev["qualifying_receivers"] >= MIN_QUALIFYING_RECEIVERS].copy()
qb_elev = qb_elev.sort_values("net_blended", ascending=False)

# ── 12. Output ────────────────────────────────────────────────────────────────
print(
    f"\n{'Rank':<5} {'QB':<25} {'Net':>6} {'Elev':>6} {'Depr':>6} "
    f"{'Rcvrs':>6} {'Starts':>7}"
)
print("-" * 70)
for i, (_, row) in enumerate(qb_elev.iterrows(), 1):
    print(
        f"{i:<5} {row['qb_name']:<25} {row['net_blended']:>+6.2f} "
        f"{row['elev_blended']:>6.2f} {row['depr_blended']:>6.2f} "
        f"{int(row['qualifying_receivers']):>6} {int(row['career_starts']):>7}"
    )

# Save
qb_elev.to_csv("qb_elevation_rate_v6.csv", index=False)
qb_rec_pairs.to_csv("qb_receiver_pairs_v6.csv", index=False)
league_avgs.to_csv("league_averages_by_season.csv", index=False)
print(f"\nSaved qb_elevation_rate_v6.csv, qb_receiver_pairs_v6.csv, league_averages_by_season.csv")
