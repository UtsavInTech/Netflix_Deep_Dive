"""Tests for the cleaning and joining logic - these encode the bugs that
were fixed, so a regression fails the suite instead of shipping."""

import pandas as pd

from src.data import data_quality_report, genre_list, parse_duration


def test_parse_duration_separates_minutes_from_seasons():
    parsed = parse_duration(pd.Series(["90 min", "2 Seasons", "1 Season", "", None]))
    assert parsed["runtime_minutes"].tolist()[0] == 90
    # The original bug: '90 min' was read as 90 seasons.
    assert pd.isna(parsed.loc[0, "seasons"])
    assert parsed["seasons"].tolist()[1:3] == [2.0, 1.0]
    assert pd.isna(parsed.loc[1, "runtime_minutes"])


def test_join_does_not_duplicate_rows(tiny_catalogue):
    """'Alpha' appears twice in the IMDb table; the join must not fan out."""
    assert len(tiny_catalogue) == 4
    assert tiny_catalogue["show_id"].is_unique


def test_join_keeps_the_highest_vote_duplicate(tiny_catalogue):
    alpha_movie = tiny_catalogue[
        (tiny_catalogue["title"] == "Alpha") & (tiny_catalogue["type"] == "Movie")
    ].iloc[0]
    assert alpha_movie["imdb_rating"] == 7.5  # not the 12-vote 4.0 entry


def test_join_respects_content_type(tiny_catalogue):
    """'Alpha' the series must not inherit 'Alpha' the film's score."""
    alpha_show = tiny_catalogue[
        (tiny_catalogue["title"] == "Alpha") & (tiny_catalogue["type"] == "TV Show")
    ].iloc[0]
    assert pd.isna(alpha_show["imdb_rating"])


def test_unmatched_ratings_stay_missing_not_zero(tiny_catalogue):
    """Filling with 0.0 silently ranked unrated titles as the worst on Netflix."""
    unmatched = tiny_catalogue[~tiny_catalogue["has_rating"]]
    assert len(unmatched) > 0
    assert unmatched["imdb_rating"].isna().all()
    assert (tiny_catalogue["imdb_rating"].fillna(1) != 0).all()


def test_watch_hours_use_the_right_units(tiny_catalogue):
    gamma = tiny_catalogue.set_index("title").loc["Gamma"]  # 120 min film
    assert gamma["estimated_watch_hours"] == 2.0
    beta = tiny_catalogue.set_index("title").loc["Beta"]  # 2 seasons, 50 min eps
    assert beta["estimated_watch_hours"] == 2 * 8 * 50 / 60


def test_genre_list_is_deduplicated_and_stripped(tiny_catalogue):
    genres = genre_list(tiny_catalogue)
    assert "International TV Shows" in genres
    assert genres == sorted(set(genres))
    assert all(g == g.strip() for g in genres)


def test_model_text_excludes_genre_labels(tiny_catalogue):
    """Genres contain the literal word 'TV' for 96% of series - a second leak."""
    row = tiny_catalogue.iloc[1]
    assert "TV Dramas" not in row["model_text"]
    assert "TV Dramas" in row["content_text"]


def test_data_quality_report_covers_every_column(tiny_catalogue):
    report = data_quality_report(tiny_catalogue)
    assert set(report.columns) == {"missing", "missing_pct", "unique"}
    assert len(report) == tiny_catalogue.shape[1]


def test_real_catalogue_has_expected_shape(catalogue):
    assert len(catalogue) == 8807
    assert catalogue["show_id"].is_unique
    assert catalogue["has_rating"].mean() > 0.40
