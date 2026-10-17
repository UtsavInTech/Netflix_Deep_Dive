"""Loading, cleaning and feature engineering for the Netflix catalogue.

The two source files are joined here:

* ``netflix_titles.csv`` - 8,807 catalogue rows (title, type, cast, genres...)
* ``netflix_imdb.csv``   - 5,283 IMDb rows (score, votes, runtime)

Every transformation in this module is deliberately pure: it takes
DataFrames in and returns DataFrames out, so the logic can be unit tested
without launching Streamlit.
"""

from __future__ import annotations

import pandas as pd

from src.config import (
    DEFAULT_EPISODE_MINUTES,
    EPISODES_PER_SEASON,
    IMDB_CSV,
    NETFLIX_CSV,
)

TEXT_COLUMNS = ["description", "cast", "director", "country"]


# --------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------- #
def read_raw(netflix_path=NETFLIX_CSV, imdb_path=IMDB_CSV):
    """Read both CSVs from disk with a helpful error if they are missing."""
    missing = [str(p) for p in (netflix_path, imdb_path) if not pd.io.common.file_exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing required data file(s): " + ", ".join(missing) +
            ". Download them into the project root - see the README."
        )
    return pd.read_csv(netflix_path), pd.read_csv(imdb_path)


# --------------------------------------------------------------------- #
# Cleaning helpers
# --------------------------------------------------------------------- #
def parse_duration(duration: pd.Series) -> pd.DataFrame:
    """Split the free-text ``duration`` column into two real numbers.

    Netflix stores '90 min' for films and '2 Seasons' for series in the
    same column. The original version of this project pulled the first
    integer out of both, which silently gave every 90-minute film a
    'season count' of 90. Two separate patterns keep the units honest.
    """
    duration = duration.fillna("").astype(str)
    runtime_minutes = duration.str.extract(r"(\d+)\s*min", expand=False).astype(float)
    seasons = duration.str.extract(r"(\d+)\s*Season", expand=False).astype(float)
    return pd.DataFrame({"runtime_minutes": runtime_minutes, "seasons": seasons})


def normalise_key(titles: pd.Series) -> pd.Series:
    """Lower-cased, whitespace-stripped title used as the join key."""
    return titles.astype(str).str.strip().str.lower()


def prepare_imdb(imdb_df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate the IMDb table so the join cannot multiply rows.

    46 titles appear more than once (remakes, re-releases). Joining
    naively added 43 phantom rows to the catalogue. We keep the entry with
    the most votes, which is reliably the well-known one.
    """
    imdb = imdb_df.rename(columns={"imdb_score": "imdb_rating"}).copy()
    imdb["type"] = imdb["type"].str.upper().map({"MOVIE": "Movie", "SHOW": "TV Show"})
    imdb["_key"] = normalise_key(imdb["title"])
    imdb = (
        imdb.sort_values("imdb_votes", ascending=False)
        .drop_duplicates(subset=["_key", "type"])
        .rename(columns={"runtime": "imdb_runtime"})
    )
    return imdb[["_key", "type", "imdb_rating", "imdb_votes", "imdb_runtime"]]


def estimate_watch_hours(row) -> float:
    """Total hours needed to finish a title.

    Films use their real runtime. Series are an estimate: seasons x
    ``EPISODES_PER_SEASON`` x episode length, preferring the IMDb
    per-episode runtime when we have it.
    """
    if row["type"] == "Movie":
        minutes = row["runtime_minutes"]
        return float(minutes) / 60 if pd.notna(minutes) else float("nan")

    seasons = row["seasons"]
    if pd.isna(seasons):
        return float("nan")
    episode_minutes = row["imdb_runtime"]
    if pd.isna(episode_minutes) or episode_minutes <= 0:
        episode_minutes = DEFAULT_EPISODE_MINUTES
    return float(seasons) * EPISODES_PER_SEASON * float(episode_minutes) / 60


# --------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------- #
def build_dataset(netflix_df: pd.DataFrame, imdb_df: pd.DataFrame) -> pd.DataFrame:
    """Clean, join and enrich the catalogue. Returns one row per title."""
    df = netflix_df.drop_duplicates(subset=["show_id"]).copy()

    df["type"] = df["type"].fillna("Unknown").str.strip()
    df[["runtime_minutes", "seasons"]] = parse_duration(df["duration"])

    # Join on title AND type so 'Sherlock' the film never inherits the
    # score of 'Sherlock' the series.
    df["_key"] = normalise_key(df["title"])
    df = df.merge(prepare_imdb(imdb_df), on=["_key", "type"], how="left")

    # imdb_rating stays NaN when unknown. The original code filled it with
    # 0.0, which is not a neutral value - it is the worst possible score,
    # and it dragged 57% of the catalogue to the bottom of every ranking.
    df["has_rating"] = df["imdb_rating"].notna()

    df["estimated_watch_hours"] = df.apply(estimate_watch_hours, axis=1)

    df["year_added"] = pd.to_datetime(
        df["date_added"], format="mixed", errors="coerce"
    ).dt.year

    for col in TEXT_COLUMNS + ["listed_in"]:
        df[col] = df[col].fillna("")

    # One text blob per title, reused by the recommender and the classifier.
    df["content_text"] = (
        df["listed_in"] + " " + df["description"] + " "
        + df["cast"] + " " + df["director"] + " " + df["country"]
    ).str.strip()

    # The classifier gets a strictly narrower blob: no genre labels, which
    # would leak the answer (96% of series carry a genre containing "TV").
    df["model_text"] = (
        df["description"] + " " + df["cast"] + " "
        + df["director"] + " " + df["country"]
    ).str.strip()

    return df.drop(columns=["_key"]).reset_index(drop=True)


def load_dataset() -> pd.DataFrame:
    """Read the CSVs from disk and return the analysis-ready DataFrame."""
    netflix_df, imdb_df = read_raw()
    return build_dataset(netflix_df, imdb_df)


def genre_list(df: pd.DataFrame):
    """All distinct genres, exploded out of the comma-separated column."""
    genres = df["listed_in"].str.split(",").explode().str.strip()
    return sorted(g for g in genres.unique() if g)


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing-value summary, shown on the Data Quality tab."""
    report = pd.DataFrame(
        {
            "missing": df.isna().sum(),
            "missing_pct": (df.isna().mean() * 100).round(1),
            "unique": df.nunique(),
        }
    )
    # Text columns were filled with "" rather than NaN, so count blanks too.
    for col in TEXT_COLUMNS + ["listed_in"]:
        if col in df.columns:
            blank = (df[col].astype(str).str.strip() == "").sum()
            report.loc[col, "missing"] = blank
            report.loc[col, "missing_pct"] = round(blank / len(df) * 100, 1)
    return report.sort_values("missing_pct", ascending=False)
