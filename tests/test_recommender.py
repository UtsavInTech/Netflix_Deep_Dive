"""Tests for filtering, searching and content-based recommendation."""

import pandas as pd

from src.recommender import (
    ContentRecommender,
    apply_filters,
    fits_in_time,
    rank_by_rating,
    search_titles,
    watch_plan,
)


def test_search_handles_regex_metacharacters(catalogue):
    """A '(' in the search box used to raise an unbalanced-parenthesis error."""
    for query in ["(", "[", "*", "a+b"]:
        result = search_titles(catalogue, query)
        assert isinstance(result, pd.DataFrame)


def test_search_is_case_insensitive(catalogue):
    assert len(search_titles(catalogue, "breaking bad")) > 0


def test_empty_search_returns_nothing(catalogue):
    assert search_titles(catalogue, "   ").empty


def test_filters_compose(catalogue):
    result = apply_filters(catalogue, "Movie", "Comedies", (2015, 2020), 6.0)
    assert (result["type"] == "Movie").all()
    assert result["listed_in"].str.contains("Comedies").all()
    assert result["release_year"].between(2015, 2020).all()
    assert (result["imdb_rating"] >= 6.0).all()


def test_genre_filter_handles_ampersands(catalogue):
    """'Children & Family Movies' breaks a regex-mode str.contains."""
    result = apply_filters(catalogue, genre="Children & Family Movies")
    assert len(result) > 0


def test_rank_by_rating_excludes_unrated(catalogue):
    top = rank_by_rating(catalogue, n=10)
    assert top["has_rating"].all()
    assert top["imdb_rating"].is_monotonic_decreasing


def test_vote_floor_filters_low_confidence_scores(catalogue):
    top = rank_by_rating(catalogue, n=10, min_votes=50_000)
    assert (top["imdb_votes"] >= 50_000).all()


def test_fits_in_time_respects_the_budget(catalogue):
    within = fits_in_time(catalogue, 2.0)
    assert (within["estimated_watch_hours"] <= 2.0).all()


def test_watch_plan_returns_both_formats(catalogue):
    """The original app showed only films under 4 hours and only series above."""
    plan = watch_plan(catalogue, 6.0, "Both", n=50, min_rating=0)
    assert plan["type"].nunique() == 2


def test_watch_plan_honours_format_choice(catalogue):
    plan = watch_plan(catalogue, 10.0, "TV Show", n=10)
    assert (plan["type"] == "TV Show").all()


def test_recommender_finds_a_known_relative(catalogue):
    rec = ContentRecommender(catalogue)
    results = rec.similar_to_title("Breaking Bad", n=5)
    assert "Better Call Saul" in results["title"].tolist()


def test_recommender_never_returns_the_query_itself(catalogue):
    rec = ContentRecommender(catalogue)
    results = rec.similar_to_title("Breaking Bad", n=10)
    assert "Breaking Bad" not in results["title"].tolist()


def test_recommender_same_type_filter(catalogue):
    rec = ContentRecommender(catalogue)
    results = rec.similar_to_title("Breaking Bad", n=5, same_type_only=True)
    assert (results["type"] == "TV Show").all()


def test_recommender_handles_unknown_title(catalogue):
    rec = ContentRecommender(catalogue)
    assert rec.similar_to_title("This Title Does Not Exist At All").empty


def test_free_text_query_returns_relevant_results(catalogue):
    rec = ContentRecommender(catalogue)
    results = rec.similar_to_text("spanish heist thriller mastermind", n=10)
    assert len(results) > 0
    assert results["similarity"].is_monotonic_decreasing
