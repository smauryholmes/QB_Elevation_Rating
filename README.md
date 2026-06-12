# QB Elevation Rate

Which QBs make their receivers better? A diff-in-diff approach using nflfastR data, 2006-2024.

## What this is

QB Elevation Rate measures whether a QB's pass-catchers (WR/TE) systematically have their career-best or career-worst seasons while playing with that QB. Receivers who played for multiple QBs form a natural experiment, borrowing from the difference-in-differences (DiD) framework in econometrics.

**Elevation Score** = (career highs vs expected) - (career lows vs expected)

- Positive: receivers peak with this QB more than they bottom out
- Zero: career highs and lows cancel out
- Negative: receivers hit career lows more than career highs

## Key design choices

- **Era-adjusted stats**: each season's per-game stats divided by the league average that year (league-wide receiving YPG dropped from ~39 in 2007 to ~34 in 2024)
- **Stints-weighted expected share**: a QB who had a receiver for 5 of his 13 qualifying stints faces a higher bar (5/13 = 0.385 expected) than one who had him for 1 stint (1/13 = 0.077). A stint = one receiver-QB-season with 6+ games; a mid-year trade can create two stints in the same season.
- **50/50 blend**: yards/game and TDs/game weighted equally (tested alternative splits up to 75/25; rank correlation >0.95 against baseline)
- **Inclusion criteria**: QB 50+ starts, 10+ qualifying receivers; receiver 6+ games in a season with a QB, must have played for 2+ QBs
- **Scope**: WR/TE receiving stats only. RB receiving and all rushing excluded.

## Files

| File | Description |
|------|-------------|
| `qb_elevation_rate.py` | Reproduces everything from raw nflfastR data. Requires `nfl_data_py` and `pandas`. |
| `qb_elevation_rate_v6.csv` | QB-level results: 62 QBs with elevation scores, component rates, and metadata. |
| `qb_receiver_pairs_v6.csv` | All 1,357 QB-receiver pair breakdowns with career high/low flags and expected shares. |
| `league_averages_by_season.csv` | League-wide receiving averages by season (used for era adjustment). |

## How to run

```bash
pip install nfl_data_py pandas
python qb_elevation_rate.py
```

The script pulls data directly from nflfastR via the `nfl_data_py` wrapper. No external data files needed.

## Results (top 10 / bottom 5)

| Rank | QB | Score | Highs vs Exp | Lows vs Exp | Receivers |
|-----:|:---|------:|-------------:|------------:|----------:|
| 1 | Josh McCown | +1.04 | 1.63x | 0.59x | 19 |
| 2 | Peyton Manning | +0.94 | 1.57x | 0.63x | 17 |
| 3 | Eli Manning | +0.91 | 1.47x | 0.57x | 22 |
| 4 | Chad Henne | +0.88 | 1.32x | 0.44x | 12 |
| 5 | Drew Brees | +0.71 | 1.32x | 0.61x | 25 |
| 6 | Josh Allen | +0.69 | 1.26x | 0.57x | 15 |
| 7 | Justin Herbert | +0.63 | 1.06x | 0.42x | 10 |
| 8 | Jason Campbell | +0.61 | 1.21x | 0.61x | 15 |
| 9 | Jameis Winston | +0.58 | 1.41x | 0.83x | 21 |
| 10 | Ben Roethlisberger | +0.56 | 1.29x | 0.73x | 21 |
| ... | | | | | |
| 58 | Mark Sanchez | -0.76 | 0.67x | 1.43x | 17 |
| 59 | Jimmy Garoppolo | -0.79 | 0.53x | 1.31x | 14 |
| 60 | Jared Goff | -0.88 | 0.53x | 1.40x | 20 |
| 61 | Alex Smith | -0.89 | 0.59x | 1.48x | 33 |
| 62 | Colin Kaepernick | -0.99 | 0.99x | 1.99x | 16 |

Full rankings, methodology, and discussion: see the Reddit post.

## Data source

All data pulled from [nflfastR](https://github.com/nflverse/nflfastR) via [nfl_data_py](https://github.com/nflverse/nfl_data_py).


