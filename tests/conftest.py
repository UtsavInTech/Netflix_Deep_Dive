"""Shared fixtures. The catalogue is loaded once per test session."""

import pandas as pd
import pytest

from src.data import build_dataset, load_dataset


@pytest.fixture(scope="session")
def catalogue():
    return load_dataset()


@pytest.fixture
def tiny_catalogue():
    """A hand-built 4-row catalogue - fast, and every value is known."""
    netflix = pd.DataFrame(
        {
            "show_id": ["s1", "s2", "s3", "s4"],
            "type": ["Movie", "TV Show", "Movie", "TV Show"],
            "title": ["Alpha", "Beta", "Gamma", "Alpha"],
            "director": ["Dir A", "", "Dir C", ""],
            "cast": ["Actor A", "Actor B", "", "Actor D"],
            "country": ["India", "United States", "", "Japan"],
            "date_added": ["September 25, 2021", "July 1, 2020", None, "March 3, 2019"],
            "release_year": [2020, 2018, 1999, 2015],
            "rating": ["PG-13", "TV-MA", "R", "TV-14"],
            "duration": ["90 min", "2 Seasons", "120 min", "1 Season"],
            "listed_in": ["Dramas", "TV Dramas, International TV Shows", "Comedies", "TV Comedies"],
            "description": ["A quiet drama.", "A tense series.", "A funny film.", "A short series."],
        }
    )
    imdb = pd.DataFrame(
        {
            "title": ["Alpha", "Alpha", "Beta", "Zeta"],
            "type": ["MOVIE", "MOVIE", "SHOW", "MOVIE"],
            "imdb_score": [7.5, 4.0, 8.2, 6.0],
            "imdb_votes": [10_000.0, 12.0, 50_000.0, 900.0],
            "runtime": [90, 90, 50, 100],
        }
    )
    return build_dataset(netflix, imdb)
